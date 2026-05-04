import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from scipy.stats import f as f_distribution
from scipy.stats import pearsonr
from scipy.stats import t as t_distribution
from tqdm import tqdm

# Setup paths
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import (
    CAUSAL_GRAPHS_DIR,
    CAUSALITY_METHOD,
    DATA_FINAL,
    DEVICE,
    GRANGER_MAX_LAG,
    GRANGER_MAX_LAG_SECONDS,
    GRAPH_DENSITY_TARGET,
    LAGGED_PEARSON_CONFIDENCE_ALPHA,
    LAGGED_PEARSON_LAGS,
    LAGGED_PEARSON_P_PRUNE_THRESHOLD,
    LAGGED_PEARSON_P_SELECT_THRESHOLD,
    LOBE_MAPPING,
    LOBE_NAMES,
    MASTER_MANIFEST,
    MIN_EDGES_PER_GRAPH,
    MULTIVIEW_GENERATION_ENFORCE_QUALITY_GATE,
    MULTIVIEW_GENERATION_MAX_ZERO_EDGE_RATE,
    MULTIVIEW_GENERATION_POLICY,
    NUM_LOBES,
    PARTIAL_CORR_FDR_ALPHA,
    PARTIAL_CORR_FDR_ENABLED,
    PARTIAL_CORR_GLASSO_ALPHA,
    PARTIAL_CORR_GLASSO_MAX_ITER,
    PARTIAL_CORR_GLASSO_TOL,
    PARTIAL_CORR_MIN_ABS_EDGE,
    PARTIAL_CORR_MIN_SAMPLES,
    RIDGE_GRANGER_CONFIDENCE_ALPHA,
    RIDGE_GRANGER_HIGH_CONF_P_THRESHOLD,
    RIDGE_GRANGER_HYBRID_BETA,
    RIDGE_GRANGER_LAGS,
    RIDGE_GRANGER_LAMBDA,
    RIDGE_GRANGER_P_PRUNE_THRESHOLD,
    SPARSITY_METHOD,
    SPARSITY_QUANTILE,
    SPARSITY_TOPK_PER_NODE,
)
from src.core.hyperparams import _MULTIVIEW_VIEW_ORDER, CONFIDENCE_LOG_EPS, FISHER_Z_EPS

# For backwards compatibility with existing code
_FISHER_EPS = FISHER_Z_EPS
_CONFIDENCE_EPS = CONFIDENCE_LOG_EPS
from src.features.causal_inference import (
    compute_granger_causality,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _zero_lagged_payload(device: torch.device) -> dict[str, Any]:
    """Return zero-matrix payload for failed lagged correlation computation."""
    z = torch.zeros(NUM_LOBES, NUM_LOBES, device=device)
    o = torch.ones(NUM_LOBES, NUM_LOBES, device=device)
    return {
        "weighted_z_matrix": z,
        "z_matrix": z,
        "p_matrix": o,
        "confidence_matrix": z,
        "selected_lag_matrix": torch.zeros(NUM_LOBES, NUM_LOBES, dtype=torch.long, device=device),
        "low_confidence_mask": o.to(torch.bool),
        "selected_r_matrix": z,
    }


def _zero_ridge_payload(device: torch.device) -> dict[str, Any]:
    """Return zero-matrix payload for failed ridge Granger computation."""
    z = torch.zeros(NUM_LOBES, NUM_LOBES, device=device)
    o = torch.ones(NUM_LOBES, NUM_LOBES, device=device)
    return {
        "weighted_effect_matrix": z,
        "effect_matrix": z,
        "p_matrix": o,
        "confidence_matrix": z,
        "low_confidence_mask": o.to(torch.bool),
    }


class _LobeWarningTracker:
    """Rate-limited warning tracker for lobe coverage gaps.

    Provides per-instance warning instead of global singleton to avoid
    breaking determinism in parallel runs.
    """

    def __init__(self, max_warnings_per_lobe: int = 3):
        self._warned: dict = {}
        self._max_warnings = max_warnings_per_lobe

    def should_warn(self, lobe_id: int) -> bool:
        if lobe_id not in self._warned:
            self._warned[lobe_id] = 0
        if self._warned[lobe_id] < self._max_warnings:
            self._warned[lobe_id] += 1
            return True
        return False

    def reset(self) -> None:
        self._warned.clear()


_zero_lobe_warned = _LobeWarningTracker()


def _empty_sparsification_info() -> dict[str, object]:
    """Return a default sparsification metadata payload."""
    return {
        "triggered": False,
        "min_edge_fallback": False,
        "significance_pruning_applied": False,
        "pruned_edge_candidates": 0,
        "retained_candidates_after_pruning": 0,
        "dead_lobe_repair": False,
        "dead_lobe_repair_added_edges": 0,
        "unresolved_dead_lobes": 0,
        "topk_per_node_k": 0,
        "primary_edge_count": 0,
        "primary_dead_lobes": 0,
        "final_edge_count": 0,
    }


def _stabilize_sign(dominant_signal: torch.Tensor, roi_data: torch.Tensor) -> torch.Tensor:
    """Stabilize PCA eigenvariate sign against a robust anchor ROI signal."""
    roi_means = roi_data.mean(dim=0).abs()
    if roi_means.numel() == 0:
        return dominant_signal

    anchor_roi = roi_data[:, int(torch.argmax(roi_means).item())]
    dot = torch.dot(
        dominant_signal / (dominant_signal.norm() + 1e-8),
        anchor_roi / (anchor_roi.norm() + 1e-8),
    )
    return dominant_signal if dot >= 0 else -dominant_signal


def aggregate_to_lobes(ts_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aggregate 170-ROI time series to 12-lobe representations using smart aggregation.

    Two complementary signals are extracted per lobe:

    1. **PCA Eigenvariate (dominant signal)**
       Computes the first principal component via SVD of the mean-centred ROI matrix.
       Captures the direction of maximum variance within the lobe, avoiding the
       signal cancellation that occurs with simple averaging when ROIs are
       anti-correlated (common in motor and cingulate areas in ASD).

    2. **Regional Homogeneity features (intra-lobe connectivity)**
       * ``coherence`` – Mean pairwise Pearson correlation of ROIs inside the lobe.
         Clamped to ``[-1, 1]``.  Higher values indicate tighter local synchrony.
       * ``spatial_variance`` – Mean standard deviation of ROI activations across
         time, averaged over all ROIs in the lobe.  Reflects the spread of activity.

    Both features are set to zero when NaN/Inf is detected so that downstream graph
    construction is never blocked by a single bad ROI.

    Args:
        ts_raw (Tensor): Raw ROI time series, shape ``(T, 170)`` where ``T`` is the
                         number of fMRI time points and 170 is the AAL3 ROI count.
                         Values should be z-scored (mean=0, std≈1).

    Returns:
        Tuple[Tensor, Tensor]:
            * ``ts_lobes`` – Lobe-level time series, shape ``(T, NUM_LOBES)``.
              Used as input to causal graph construction.
            * ``features_internal`` – Internal feature matrix, shape ``(NUM_LOBES, 2)``.
              Column 0: coherence; column 1: spatial_variance.
              Concatenated into node features by ``graph_factory.py``.

    Raises:
        No exceptions raised; failures fall back to zero vectors with a warning log.
    """
    num_rois = ts_raw.shape[1]
    lobe_signals = []
    lobe_internal_features = []
    zero_lobes: list = []  # True when a lobe used the zero-signal fallback

    for lobe_id in range(NUM_LOBES):
        # Get ROIs belonging to this lobe (already 0-based indices)
        indices = [i for i in LOBE_MAPPING[lobe_id] if i < num_rois]

        if not indices:
            if _zero_lobe_warned.should_warn(lobe_id):
                logger.warning(
                    f"Lobe {lobe_id} ({LOBE_NAMES[lobe_id]}): No matching ROIs in atlas. "
                    "Using zero-signal. (Subsequent warnings suppressed.)"
                )
            lobe_signals.append(torch.zeros(ts_raw.shape[0], device=ts_raw.device))
            lobe_internal_features.append(torch.tensor([0.0, 0.0], device=ts_raw.device))
            zero_lobes.append(True)
            continue

        # Extract raw ROIs for this lobe: Shape (Timepoints, Num_ROIs_in_Lobe)
        roi_data = ts_raw[:, indices]

        # Filter out ROIs whose time series contains any NaN (atlas coverage gaps,
        # brainstem/subcortical ROIs beyond atlas bounds).  This must happen before
        # the PCA block so that NaN values don't propagate into the lobe signal.
        valid_roi_mask = ~torch.isnan(roi_data).any(dim=0)  # (N_rois_in_lobe,)
        if not valid_roi_mask.any():
            if _zero_lobe_warned.should_warn(lobe_id):
                logger.warning(
                    f"Lobe {lobe_id} ({LOBE_NAMES[lobe_id]}): all {len(indices)} ROIs "
                    "have NaN time series (atlas coverage gap). Using zero-signal fallback. "
                    "(Subsequent warnings suppressed.)"
                )
            lobe_signals.append(torch.zeros(ts_raw.shape[0], device=ts_raw.device))
            lobe_internal_features.append(torch.tensor([0.0, 0.0], device=ts_raw.device))
            zero_lobes.append(True)
            continue
        if valid_roi_mask.sum().item() < len(indices):
            n_dropped = len(indices) - valid_roi_mask.sum().item()
            logger.debug(
                f"Lobe {lobe_id} ({LOBE_NAMES[lobe_id]}): dropped {n_dropped} NaN ROI(s), "
                f"using {valid_roi_mask.sum().item()}/{len(indices)} valid ROIs."
            )
            roi_data = roi_data[:, valid_roi_mask]

        # --- 1. DOMINANT SIGNAL EXTRACTION (PCA/EIGENVARIATE) ---
        # Instead of mean(), use first principal component to avoid signal cancellation
        try:
            # Center the data
            centered = roi_data - roi_data.mean(dim=0)
            # Perform SVD (Singular Value Decomposition) for PCA
            u, s, vh = torch.linalg.svd(centered, full_matrices=False)
            # First Principal Component captures max variance
            # This preserves the magnitude of activity even when signals are out-of-sync
            dominant_signal = u[:, 0] * s[0]
            dominant_signal = _stabilize_sign(dominant_signal, roi_data)
        except Exception as e:
            logger.debug(f"Lobe {lobe_id}: SVD failed ({str(e)}), falling back to mean")
            dominant_signal = roi_data.mean(dim=1)

        lobe_signals.append(dominant_signal)

        # --- 2. INTRA-LOBE SYNCHRONY (Regional Homogeneity - ReHo) ---
        # Measure how synchronized ROIs within this lobe are
        # ASD hypothesis: Local over-connectivity means HIGH coherence within lobes

        # roi_data is already NaN-free after the filtering block above
        valid_rois = roi_data

        if valid_rois.shape[1] > 1:
            try:
                # Compute correlation matrix of valid ROIs within this lobe
                intra_corr = torch.corrcoef(valid_rois.T)
                # Average off-diagonal correlation (all pairs)
                mask = ~torch.eye(intra_corr.shape[0], dtype=torch.bool, device=ts_raw.device)
                coherence = intra_corr[mask].mean()
                coherence = torch.clamp(coherence, -1.0, 1.0)  # Ensure valid range

                # Spatial heterogeneity (variance across valid ROIs over time)
                spatial_variance = valid_rois.std(dim=1).mean()
            except Exception as e:
                logger.debug(f"Lobe {lobe_id}: ReHo computation failed ({str(e)})")
                coherence = torch.tensor(0.0, device=ts_raw.device)
                spatial_variance = torch.tensor(0.0, device=ts_raw.device)
        else:
            # Single ROI in lobe: trivial values
            coherence = torch.tensor(1.0, device=ts_raw.device)  # Perfect self-correlation
            spatial_variance = torch.tensor(0.0, device=ts_raw.device)

        # SAFETY: Replace NaN/Inf with 0 to prevent downstream crashes
        if torch.isnan(coherence) or torch.isinf(coherence):
            coherence = torch.tensor(0.0, device=ts_raw.device)
        if torch.isnan(spatial_variance) or torch.isinf(spatial_variance):
            spatial_variance = torch.tensor(0.0, device=ts_raw.device)

        lobe_internal_features.append(torch.stack([coherence, spatial_variance]))
        zero_lobes.append(False)

    # Stack results
    ts_lobes = torch.stack(lobe_signals, dim=1)           # (Timepoints, 12)
    features_internal = torch.stack(lobe_internal_features, dim=0)  # (12, 2)
    zero_lobe_mask = torch.tensor(zero_lobes, dtype=torch.bool)      # (12,)

    return ts_lobes, features_internal, zero_lobe_mask


def _assert_fisher_z_transformed(z_matrix: torch.Tensor, r_matrix: torch.Tensor) -> None:
    """Guardrail: fail fast if correlations appear to bypass Fisher-Z transform."""
    offdiag_mask = ~torch.eye(NUM_LOBES, dtype=torch.bool, device=z_matrix.device)
    # Use edges where Fisher-Z should differ noticeably from raw r.
    # Lowered threshold to 0.10 to catch more edges (sparse graph safety).
    probe_mask = offdiag_mask & (r_matrix.abs() >= 0.10) & (r_matrix.abs() < 0.999)
    if not bool(probe_mask.any()):
        # Sparse graph: warn but don't skip check entirely
        logger.warning(
            "Fisher-Z check: no edges with |r| >= 0.10; "
            "consider verifying transform was applied for very sparse graphs."
        )
        # Fall back to any non-zero off-diagonal edge
        probe_mask = offdiag_mask & (r_matrix.abs() > 1e-6)
        if not bool(probe_mask.any()):
            return  # Truly empty graph, nothing to check

    # Fisher-Z transform: z = arctanh(r). For |r| >= 0.10, z should differ from r.
    # Check if z is TOO close to r (indicating missing transform).
    z_probe = z_matrix[probe_mask]
    r_probe = r_matrix[probe_mask]
    # For proper Fisher-Z, z and r should differ by > 1e-3 for |r| >= 0.10
    if torch.allclose(z_probe, r_probe, atol=1e-3, rtol=1e-3):
        raise AssertionError(
            "Lagged-Pearson output appears untransformed (raw correlations leaked)."
        )


def _compute_lagged_pearson_multilag(
    ts_lobe: torch.Tensor,
    lags: tuple[int, ...],
    p_select_threshold: float,
    confidence_alpha: float,
) -> dict[str, torch.Tensor]:
    """Compute multi-lag lagged-Pearson edges with Fisher-Z and confidence scaling.

    For each directed edge i->j:
      1) Evaluate Pearson correlation over multiple lags.
      2) Convert each lag correlation to Fisher-Z.
      3) Select lag with max |z| under p < p_select_threshold.
         If none pass, choose lag with max |z| and mark as low-confidence.
      4) Scale selected z by sigmoid(alpha * confidence),
         where confidence = -log(p + eps).
    """
    if ts_lobe.shape[0] <= max(lags):
        logger.warning("Insufficient timepoints for multi-lag correlation")
        return _zero_lagged_payload(ts_lobe.device)

    if torch.isnan(ts_lobe).any() or torch.isinf(ts_lobe).any():
        logger.warning("Input contains NaN/Inf values - returning zero matrix")
        return _zero_lagged_payload(ts_lobe.device)

    # Standardize per lobe before lagged correlation.
    ts_mean = ts_lobe.mean(dim=0, keepdim=True)
    ts_std = ts_lobe.std(dim=0, keepdim=True).clamp_min(1e-6)
    ts_norm = (ts_lobe - ts_mean) / ts_std
    ts_np = ts_norm.detach().cpu().numpy()

    z_sel = np.zeros((NUM_LOBES, NUM_LOBES), dtype=np.float32)
    p_sel = np.ones((NUM_LOBES, NUM_LOBES), dtype=np.float32)
    r_sel = np.zeros((NUM_LOBES, NUM_LOBES), dtype=np.float32)
    lag_sel = np.zeros((NUM_LOBES, NUM_LOBES), dtype=np.int64)
    low_conf = np.zeros((NUM_LOBES, NUM_LOBES), dtype=bool)

    for src in range(NUM_LOBES):
        for dst in range(NUM_LOBES):
            if src == dst:
                continue

            best_any = (0.0, 1.0, 0.0, 0)      # z, p, r, lag
            best_sig = (0.0, 1.0, 0.0, 0)      # z, p, r, lag
            best_any_abs = -np.inf
            best_sig_abs = -np.inf

            for lag in lags:
                if ts_np.shape[0] <= lag + 2:
                    continue
                x = ts_np[:-lag, src]
                y = ts_np[lag:, dst]

                try:
                    r_val, p_val = pearsonr(x, y)
                except Exception:
                    r_val, p_val = 0.0, 1.0

                if not np.isfinite(r_val):
                    r_val = 0.0
                if not np.isfinite(p_val):
                    p_val = 1.0

                r_clipped = float(np.clip(r_val, -1.0 + _FISHER_EPS, 1.0 - _FISHER_EPS))
                z_val = float(np.arctanh(r_clipped))
                abs_z = abs(z_val)

                if abs_z > best_any_abs:
                    best_any_abs = abs_z
                    best_any = (z_val, float(p_val), r_clipped, int(lag))

                if p_val < p_select_threshold and abs_z > best_sig_abs:
                    best_sig_abs = abs_z
                    best_sig = (z_val, float(p_val), r_clipped, int(lag))

            if best_sig_abs > -np.inf:
                z_star, p_star, r_star, lag_star = best_sig
                low_conf[src, dst] = False
            else:
                z_star, p_star, r_star, lag_star = best_any
                low_conf[src, dst] = True

            z_sel[src, dst] = z_star
            p_sel[src, dst] = p_star
            r_sel[src, dst] = r_star
            lag_sel[src, dst] = lag_star

    z_t = torch.from_numpy(z_sel).to(ts_lobe.device)
    p_t = torch.from_numpy(p_sel).to(ts_lobe.device)
    r_t = torch.from_numpy(r_sel).to(ts_lobe.device)
    lag_t = torch.from_numpy(lag_sel).to(ts_lobe.device)
    low_t = torch.from_numpy(low_conf).to(ts_lobe.device)

    confidence_t = -torch.log(p_t + _CONFIDENCE_EPS)
    weight_scale = torch.sigmoid(confidence_alpha * confidence_t)
    weighted_z = z_t * weight_scale

    weighted_z.fill_diagonal_(0.0)
    z_t.fill_diagonal_(0.0)
    p_t.fill_diagonal_(1.0)
    confidence_t.fill_diagonal_(0.0)

    # Mandatory consistency guard: detect accidental raw-r leakage.
    _assert_fisher_z_transformed(z_t, r_t)

    if torch.isnan(weighted_z).any() or torch.isinf(weighted_z).any():
        logger.warning("Multi-lag weighted Fisher-Z produced NaN/Inf - returning zeros")
        weighted_z = torch.zeros_like(weighted_z)

    return {
        "weighted_z_matrix": weighted_z,
        "z_matrix": z_t,
        "p_matrix": p_t,
        "confidence_matrix": confidence_t,
        "selected_lag_matrix": lag_t,
        "low_confidence_mask": low_t,
        "selected_r_matrix": r_t,
    }


def _partial_corr_zero_payload(device: torch.device) -> dict[str, torch.Tensor]:
    """Return an all-zero partial-correlation payload for safe fallbacks."""
    zeros = torch.zeros(NUM_LOBES, NUM_LOBES, device=device)
    ones = torch.ones(NUM_LOBES, NUM_LOBES, device=device)
    return {
        "weighted_partial_matrix": zeros,
        "partial_corr_matrix": zeros,
        "precision_matrix": zeros,
        "confidence_matrix": zeros,
        "pvalue_matrix": ones,
        "fdr_significant_mask": torch.zeros(NUM_LOBES, NUM_LOBES, dtype=torch.bool, device=device),
        "low_confidence_mask": torch.ones(NUM_LOBES, NUM_LOBES, dtype=torch.bool, device=device),
    }


def _benjamini_hochberg_reject(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """Return BH rejection mask for a flat vector of p-values."""
    p = np.asarray(p_values, dtype=np.float64)
    finite_mask = np.isfinite(p)
    reject = np.zeros_like(p, dtype=bool)
    if not finite_mask.any():
        return reject

    p_valid = np.clip(p[finite_mask], 0.0, 1.0)
    m = p_valid.size
    if m == 0:
        return reject

    order = np.argsort(p_valid)
    ranked = p_valid[order]
    thresholds = float(alpha) * (np.arange(1, m + 1) / float(m))
    passing = ranked <= thresholds
    if not np.any(passing):
        return reject

    max_rank = int(np.where(passing)[0].max())
    cutoff = ranked[max_rank]
    reject_valid = p_valid <= cutoff
    reject[np.where(finite_mask)[0]] = reject_valid
    return reject


def _compute_partial_corr_glasso_matrix(
    ts_lobe: torch.Tensor,
    alpha: float,
    max_iter: int,
    tol: float,
    min_abs_edge: float,
    min_samples: int,
    fdr_enabled: bool,
    fdr_alpha: float,
) -> dict[str, torch.Tensor]:
    """Estimate sparse partial-correlation edges with GraphicalLasso.

    This method fits a sparse precision matrix (inverse covariance) and converts it
    to partial correlations:

        rho_ij = -Theta_ij / sqrt(Theta_ii * Theta_jj)

    where Theta is the precision matrix. The resulting adjacency is symmetric and
    captures conditional dependence rather than temporal precedence.
    """
    min_required = max(int(min_samples), NUM_LOBES + 2)
    if ts_lobe.shape[0] < min_required:
        logger.warning(
            "Insufficient timepoints for partial_corr_glasso (%d < %d)",
            int(ts_lobe.shape[0]),
            min_required,
        )
        return _partial_corr_zero_payload(ts_lobe.device)

    if torch.isnan(ts_lobe).any() or torch.isinf(ts_lobe).any():
        logger.warning("Input contains NaN/Inf values - returning zero partial-correlation matrix")
        return _partial_corr_zero_payload(ts_lobe.device)

    ts_mean = ts_lobe.mean(dim=0, keepdim=True)
    ts_std = ts_lobe.std(dim=0, keepdim=True).clamp_min(1e-6)
    ts_norm = (ts_lobe - ts_mean) / ts_std
    ts_np = ts_norm.detach().cpu().numpy().astype(np.float64)

    try:
        from sklearn.covariance import GraphicalLasso

        glasso = GraphicalLasso(
            alpha=float(alpha),
            max_iter=int(max_iter),
            tol=float(tol),
            assume_centered=False,
        )
        glasso.fit(ts_np)
        precision_np = np.asarray(glasso.precision_, dtype=np.float64)
    except Exception as e:
        logger.warning(f"GraphicalLasso failed ({e}) - returning zero partial-correlation matrix")
        return _partial_corr_zero_payload(ts_lobe.device)

    if precision_np.shape != (NUM_LOBES, NUM_LOBES):
        logger.warning(
            "GraphicalLasso precision shape mismatch (%s) - returning zero matrix",
            precision_np.shape,
        )
        return _partial_corr_zero_payload(ts_lobe.device)

    diag = np.clip(np.diag(precision_np), _CONFIDENCE_EPS, None)
    denom = np.sqrt(np.outer(diag, diag))
    partial_np = -precision_np / (denom + _CONFIDENCE_EPS)
    np.fill_diagonal(partial_np, 0.0)
    partial_np = np.nan_to_num(partial_np, nan=0.0, posinf=0.0, neginf=0.0)

    abs_partial = np.abs(partial_np)
    min_abs_edge = float(max(min_abs_edge, 0.0))

    pvalue_np = np.ones((NUM_LOBES, NUM_LOBES), dtype=np.float64)
    fdr_sig_np = np.zeros((NUM_LOBES, NUM_LOBES), dtype=bool)

    n_samples_eff = int(ts_np.shape[0])
    if n_samples_eff > NUM_LOBES + 2:
        dof = max(n_samples_eff - NUM_LOBES, 1)
        abs_r = np.clip(abs_partial, 0.0, 1.0 - _FISHER_EPS)
        denom = np.maximum(1.0 - abs_r ** 2, _CONFIDENCE_EPS)
        t_stat = abs_r * np.sqrt(dof / denom)
        pvalue_np = 2.0 * (1.0 - t_distribution.cdf(t_stat, df=dof))
        pvalue_np = np.nan_to_num(pvalue_np, nan=1.0, posinf=1.0, neginf=1.0)
        np.fill_diagonal(pvalue_np, 1.0)

        if bool(fdr_enabled):
            offdiag_mask = ~np.eye(NUM_LOBES, dtype=bool)
            reject_flat = _benjamini_hochberg_reject(
                pvalue_np[offdiag_mask],
                alpha=float(max(min(fdr_alpha, 1.0), 1e-8)),
            )
            fdr_sig_np[offdiag_mask] = reject_flat
            partial_np = np.where(fdr_sig_np, partial_np, 0.0)
            abs_partial = np.abs(partial_np)

    if min_abs_edge > 0.0:
        partial_np[abs_partial < min_abs_edge] = 0.0
        abs_partial = np.abs(partial_np)

    confidence_np = np.zeros_like(abs_partial, dtype=np.float64)
    offdiag_mask = ~np.eye(NUM_LOBES, dtype=bool)
    nonzero_offdiag = abs_partial[offdiag_mask]
    nonzero_offdiag = nonzero_offdiag[nonzero_offdiag > 0]
    if nonzero_offdiag.size > 0:
        scale = float(np.quantile(nonzero_offdiag, 0.75))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = float(np.mean(nonzero_offdiag))
        confidence_np = np.clip(abs_partial / (scale + _CONFIDENCE_EPS), 0.0, 1.0)

    low_conf_np = abs_partial < max(min_abs_edge, 1e-8)
    np.fill_diagonal(confidence_np, 0.0)
    np.fill_diagonal(low_conf_np, True)

    partial_t = torch.from_numpy(partial_np.astype(np.float32)).to(ts_lobe.device)
    precision_t = torch.from_numpy(
        np.nan_to_num(precision_np, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    ).to(ts_lobe.device)
    confidence_t = torch.from_numpy(confidence_np.astype(np.float32)).to(ts_lobe.device)
    pvalue_t = torch.from_numpy(pvalue_np.astype(np.float32)).to(ts_lobe.device)
    fdr_sig_t = torch.from_numpy(fdr_sig_np.astype(bool)).to(ts_lobe.device)
    low_conf_t = torch.from_numpy(low_conf_np.astype(bool)).to(ts_lobe.device)

    if torch.isnan(partial_t).any() or torch.isinf(partial_t).any():
        logger.warning("Partial-correlation output contained NaN/Inf - returning zeros")
        return _partial_corr_zero_payload(ts_lobe.device)

    return {
        "weighted_partial_matrix": partial_t,
        "partial_corr_matrix": partial_t.clone(),
        "precision_matrix": precision_t,
        "confidence_matrix": confidence_t,
        "pvalue_matrix": pvalue_t,
        "fdr_significant_mask": fdr_sig_t,
        "low_confidence_mask": low_conf_t,
    }


def _ridge_solve(X: np.ndarray, y: np.ndarray, ridge_lambda: float) -> np.ndarray:
    """Solve ridge regression in closed form with numerical fallback."""
    xtx = X.T @ X
    reg = ridge_lambda * np.eye(X.shape[1], dtype=np.float64)
    rhs = X.T @ y
    try:
        beta = np.linalg.solve(xtx + reg, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(xtx + reg) @ rhs
    return beta


def _compute_ridge_granger_matrix(
    ts_lobe: torch.Tensor,
    lags: tuple[int, ...],
    ridge_lambda: float,
    confidence_alpha: float,
    high_conf_p_threshold: float,
) -> dict[str, torch.Tensor]:
    """Compute ridge-regularized pairwise VAR Granger effects and significance.

    For each directed edge src->dst:
      1) Fit restricted ridge model: dst_t ~ past(dst).
      2) Fit full ridge model: dst_t ~ past(dst) + past(src).
      3) Compute approximate F-test p-value from RSS improvement.
      4) Define signed effect size from source-lag coefficient vector.
      5) Scale edge by confidence gate: w = effect * sigmoid(alpha * -log(p)).
    """
    if len(lags) == 0:
        raise ValueError("RIDGE_GRANGER_LAGS cannot be empty")

    max_lag = max(lags)
    if ts_lobe.shape[0] <= max_lag + 2:
        logger.warning("Insufficient timepoints for ridge Granger")
        return _zero_ridge_payload(ts_lobe.device)

    if torch.isnan(ts_lobe).any() or torch.isinf(ts_lobe).any():
        logger.warning("Input contains NaN/Inf values - returning zero ridge Granger matrix")
        return _zero_ridge_payload(ts_lobe.device)

    # Standardize per lobe before regression.
    ts_mean = ts_lobe.mean(dim=0, keepdim=True)
    ts_std = ts_lobe.std(dim=0, keepdim=True).clamp_min(1e-6)
    ts_norm = (ts_lobe - ts_mean) / ts_std
    ts_np = ts_norm.detach().cpu().numpy().astype(np.float64)

    n_time = ts_np.shape[0]
    n_obs = n_time - max_lag
    lag_count = len(lags)

    effect = np.zeros((NUM_LOBES, NUM_LOBES), dtype=np.float32)
    pvals = np.ones((NUM_LOBES, NUM_LOBES), dtype=np.float32)

    for dst in range(NUM_LOBES):
        y = ts_np[max_lag:, dst]

        y_lags = np.column_stack([ts_np[max_lag - lag:n_time - lag, dst] for lag in lags])

        beta_restricted = _ridge_solve(y_lags, y, ridge_lambda)
        y_hat_restricted = y_lags @ beta_restricted
        rss_restricted = float(np.sum((y - y_hat_restricted) ** 2))

        for src in range(NUM_LOBES):
            if src == dst:
                continue

            x_lags = np.column_stack([ts_np[max_lag - lag:n_time - lag, src] for lag in lags])

            full_design = np.concatenate([y_lags, x_lags], axis=1)
            beta_full = _ridge_solve(full_design, y, ridge_lambda)
            y_hat_full = full_design @ beta_full
            rss_full = float(np.sum((y - y_hat_full) ** 2))

            src_coef = beta_full[lag_count:]
            src_sum = float(np.sum(src_coef))
            src_norm = float(np.linalg.norm(src_coef))
            signed_effect = float(np.sign(src_sum) * src_norm) if src_norm > 0 else 0.0
            if not np.isfinite(signed_effect):
                signed_effect = 0.0
            effect[src, dst] = signed_effect

            df1 = lag_count
            df2 = n_obs - full_design.shape[1]
            if df2 <= 1 or rss_full <= 1e-12:
                pvals[src, dst] = 1.0
                continue

            f_num = (rss_restricted - rss_full) / max(df1, 1)
            f_den = rss_full / max(df2, 1)
            f_stat = float(f_num / (f_den + 1e-12))
            if not np.isfinite(f_stat) or f_stat <= 0:
                pvals[src, dst] = 1.0
                continue

            p_val = float(1.0 - f_distribution.cdf(f_stat, df1, df2))
            if not np.isfinite(p_val):
                p_val = 1.0
            pvals[src, dst] = float(np.clip(p_val, 0.0, 1.0))

    effect_t = torch.from_numpy(effect).to(ts_lobe.device)
    p_t = torch.from_numpy(pvals).to(ts_lobe.device)

    confidence_t = -torch.log(p_t + _CONFIDENCE_EPS)
    scale_t = torch.sigmoid(confidence_alpha * confidence_t)
    weighted_t = effect_t * scale_t

    weighted_t.fill_diagonal_(0.0)
    effect_t.fill_diagonal_(0.0)
    p_t.fill_diagonal_(1.0)
    confidence_t.fill_diagonal_(0.0)
    low_conf = p_t >= float(high_conf_p_threshold)

    if torch.isnan(weighted_t).any() or torch.isinf(weighted_t).any():
        logger.warning("Ridge Granger produced NaN/Inf - returning zeros")
        weighted_t = torch.zeros_like(weighted_t)

    return {
        "weighted_effect_matrix": weighted_t,
        "effect_matrix": effect_t,
        "p_matrix": p_t,
        "confidence_matrix": confidence_t,
        "low_confidence_mask": low_conf,
    }


def _compute_ridge_granger_hybrid_matrix(
    ts_lobe: torch.Tensor,
    ridge_lags: tuple[int, ...],
    ridge_lambda: float,
    ridge_conf_alpha: float,
    ridge_high_conf_p: float,
    pearson_lags: tuple[int, ...],
    pearson_p_select: float,
    pearson_conf_alpha: float,
    beta: float,
) -> dict[str, torch.Tensor]:
    """Optional extension: blend ridge Granger with lagged-Pearson edges."""
    ridge_payload = _compute_ridge_granger_matrix(
        ts_lobe=ts_lobe,
        lags=ridge_lags,
        ridge_lambda=ridge_lambda,
        confidence_alpha=ridge_conf_alpha,
        high_conf_p_threshold=ridge_high_conf_p,
    )
    pearson_payload = _compute_lagged_pearson_multilag(
        ts_lobe=ts_lobe,
        lags=pearson_lags,
        p_select_threshold=pearson_p_select,
        confidence_alpha=pearson_conf_alpha,
    )

    b = float(np.clip(beta, 0.0, 1.0))
    weighted_hybrid = b * ridge_payload["weighted_effect_matrix"] + (1.0 - b) * pearson_payload["weighted_z_matrix"]
    p_hybrid = torch.minimum(ridge_payload["p_matrix"], pearson_payload["p_matrix"])
    conf_hybrid = -torch.log(p_hybrid + _CONFIDENCE_EPS)
    low_conf = p_hybrid >= float(ridge_high_conf_p)

    weighted_hybrid.fill_diagonal_(0.0)
    p_hybrid.fill_diagonal_(1.0)
    conf_hybrid.fill_diagonal_(0.0)

    return {
        "weighted_effect_matrix": weighted_hybrid,
        "effect_matrix": weighted_hybrid,
        "p_matrix": p_hybrid,
        "confidence_matrix": conf_hybrid,
        "low_confidence_mask": low_conf,
    }


def compute_causality_matrix(
    ts_lobe: torch.Tensor,
    method: str = None,
    max_lag: int = None,
    return_metadata: bool = False,
) -> Any:
    """
    Compute causal adjacency matrix using configured method.

    Args:
        ts_lobe: Time series for lobes (shape: [timepoints, n_lobes])
        method: Causality method
            ('ridge_granger', 'ridge_granger_hybrid',
             'lagged_pearson', 'partial_corr_glasso')
                If None, uses CAUSALITY_METHOD from config
        max_lag: Max lag in timepoints for Granger causality. If None, uses GRANGER_MAX_LAG from config.
                 For multi-site studies, this allows per-subject adaptation based on TR.

    Returns:
        If return_metadata=False:
            Causal adjacency matrix (shape: [n_lobes, n_lobes])
        If return_metadata=True:
            Tuple[Tensor, Dict[str, Tensor]] with auxiliary edge metadata.
    """
    if method is None:
        method = CAUSALITY_METHOD

    if max_lag is None:
        max_lag = GRANGER_MAX_LAG

    metadata: dict[str, torch.Tensor] = {}

    # Convert to numpy for causal inference methods
    ts_lobe.cpu().numpy()

    try:
        if method == 'ridge_granger':
            logger.debug(
                "Computing ridge Granger causality (lags=%s, lambda=%.4f)",
                RIDGE_GRANGER_LAGS,
                float(RIDGE_GRANGER_LAMBDA),
            )
            ridge_payload = _compute_ridge_granger_matrix(
                ts_lobe=ts_lobe,
                lags=tuple(RIDGE_GRANGER_LAGS),
                ridge_lambda=float(RIDGE_GRANGER_LAMBDA),
                confidence_alpha=float(RIDGE_GRANGER_CONFIDENCE_ALPHA),
                high_conf_p_threshold=float(RIDGE_GRANGER_HIGH_CONF_P_THRESHOLD),
            )
            causal_matrix = ridge_payload["weighted_effect_matrix"]
            metadata = {
                "pvalue_matrix": ridge_payload["p_matrix"],
                "confidence_matrix": ridge_payload["confidence_matrix"],
                "low_confidence_mask": ridge_payload["low_confidence_mask"],
                "effect_matrix": ridge_payload["effect_matrix"],
            }
            return (causal_matrix, metadata) if return_metadata else causal_matrix

        elif method == 'ridge_granger_hybrid':
            logger.debug(
                "Computing hybrid ridge-granger graph (beta=%.2f)",
                float(RIDGE_GRANGER_HYBRID_BETA),
            )
            hybrid_payload = _compute_ridge_granger_hybrid_matrix(
                ts_lobe=ts_lobe,
                ridge_lags=tuple(RIDGE_GRANGER_LAGS),
                ridge_lambda=float(RIDGE_GRANGER_LAMBDA),
                ridge_conf_alpha=float(RIDGE_GRANGER_CONFIDENCE_ALPHA),
                ridge_high_conf_p=float(RIDGE_GRANGER_HIGH_CONF_P_THRESHOLD),
                pearson_lags=tuple(LAGGED_PEARSON_LAGS),
                pearson_p_select=float(LAGGED_PEARSON_P_SELECT_THRESHOLD),
                pearson_conf_alpha=float(LAGGED_PEARSON_CONFIDENCE_ALPHA),
                beta=float(RIDGE_GRANGER_HYBRID_BETA),
            )
            causal_matrix = hybrid_payload["weighted_effect_matrix"]
            metadata = {
                "pvalue_matrix": hybrid_payload["p_matrix"],
                "confidence_matrix": hybrid_payload["confidence_matrix"],
                "low_confidence_mask": hybrid_payload["low_confidence_mask"],
                "effect_matrix": hybrid_payload["effect_matrix"],
            }
            return (causal_matrix, metadata) if return_metadata else causal_matrix

        elif method == 'lagged_pearson':
            logger.debug(f"Computing multi-lag Pearson correlation (lags={LAGGED_PEARSON_LAGS})")
            lagged_payload = _compute_lagged_pearson_multilag(
                ts_lobe=ts_lobe,
                lags=tuple(LAGGED_PEARSON_LAGS),
                p_select_threshold=float(LAGGED_PEARSON_P_SELECT_THRESHOLD),
                confidence_alpha=float(LAGGED_PEARSON_CONFIDENCE_ALPHA),
            )
            causal_matrix = lagged_payload["weighted_z_matrix"]
            metadata = {
                "pvalue_matrix": lagged_payload["p_matrix"],
                "confidence_matrix": lagged_payload["confidence_matrix"],
                "selected_lag_matrix": lagged_payload["selected_lag_matrix"],
                "low_confidence_mask": lagged_payload["low_confidence_mask"],
                "fisher_z_matrix": lagged_payload["z_matrix"],
            }
            return (causal_matrix, metadata) if return_metadata else causal_matrix

        elif method == 'partial_corr_glasso':
            logger.debug(
                "Computing partial-correlation GraphicalLasso (alpha=%.4f)",
                float(PARTIAL_CORR_GLASSO_ALPHA),
            )
            partial_payload = _compute_partial_corr_glasso_matrix(
                ts_lobe=ts_lobe,
                alpha=float(PARTIAL_CORR_GLASSO_ALPHA),
                max_iter=int(PARTIAL_CORR_GLASSO_MAX_ITER),
                tol=float(PARTIAL_CORR_GLASSO_TOL),
                min_abs_edge=float(PARTIAL_CORR_MIN_ABS_EDGE),
                min_samples=int(PARTIAL_CORR_MIN_SAMPLES),
                fdr_enabled=bool(PARTIAL_CORR_FDR_ENABLED),
                fdr_alpha=float(PARTIAL_CORR_FDR_ALPHA),
            )
            causal_matrix = partial_payload["weighted_partial_matrix"]
            metadata = {
                "confidence_matrix": partial_payload["confidence_matrix"],
                "pvalue_matrix": partial_payload["pvalue_matrix"],
                "fdr_significant_mask": partial_payload["fdr_significant_mask"],
                "low_confidence_mask": partial_payload["low_confidence_mask"],
                "partial_corr_matrix": partial_payload["partial_corr_matrix"],
                "precision_matrix": partial_payload["precision_matrix"],
            }
            return (causal_matrix, metadata) if return_metadata else causal_matrix

        else:
            logger.warning(f"Unknown causality method '{method}', falling back to ridge_granger")
            ridge_payload = _compute_ridge_granger_matrix(
                ts_lobe=ts_lobe,
                lags=tuple(RIDGE_GRANGER_LAGS),
                ridge_lambda=float(RIDGE_GRANGER_LAMBDA),
                confidence_alpha=float(RIDGE_GRANGER_CONFIDENCE_ALPHA),
                high_conf_p_threshold=float(RIDGE_GRANGER_HIGH_CONF_P_THRESHOLD),
            )
            causal_matrix = ridge_payload["weighted_effect_matrix"]
            metadata = {
                "pvalue_matrix": ridge_payload["p_matrix"],
                "confidence_matrix": ridge_payload["confidence_matrix"],
                "low_confidence_mask": ridge_payload["low_confidence_mask"],
                "effect_matrix": ridge_payload["effect_matrix"],
            }
            return (causal_matrix, metadata) if return_metadata else causal_matrix

    except Exception as e:
        logger.warning(f"Causality computation failed ({method}): {e}, falling back to ridge_granger")
        try:
            ridge_payload = _compute_ridge_granger_matrix(
                ts_lobe=ts_lobe,
                lags=tuple(RIDGE_GRANGER_LAGS),
                ridge_lambda=float(RIDGE_GRANGER_LAMBDA),
                confidence_alpha=float(RIDGE_GRANGER_CONFIDENCE_ALPHA),
                high_conf_p_threshold=float(RIDGE_GRANGER_HIGH_CONF_P_THRESHOLD),
            )
            causal_matrix = ridge_payload["weighted_effect_matrix"]
            metadata = {
                "pvalue_matrix": ridge_payload["p_matrix"],
                "confidence_matrix": ridge_payload["confidence_matrix"],
                "low_confidence_mask": ridge_payload["low_confidence_mask"],
                "effect_matrix": ridge_payload["effect_matrix"],
            }
            return (causal_matrix, metadata) if return_metadata else causal_matrix
        except Exception as ridge_error:
            logger.error(f"Ridge fallback failed: {ridge_error}")
            zero_matrix = torch.zeros(NUM_LOBES, NUM_LOBES, device=ts_lobe.device)
            zero_meta = {
                "pvalue_matrix": torch.ones(NUM_LOBES, NUM_LOBES, device=ts_lobe.device),
                "confidence_matrix": torch.zeros(NUM_LOBES, NUM_LOBES, device=ts_lobe.device),
                "low_confidence_mask": torch.ones(NUM_LOBES, NUM_LOBES, dtype=torch.bool, device=ts_lobe.device),
            }
            return (zero_matrix, zero_meta) if return_metadata else zero_matrix


def _repair_dead_lobes(
    adj_matrix: torch.Tensor,
    causal_matrix: torch.Tensor,
) -> tuple[torch.Tensor, int, int]:
    """Inject strongest incident edges for isolated lobes.

    The training gate marks a graph as degenerate when any lobe has both
    zero in-degree and zero out-degree. Min-edge fallback alone cannot prevent
    this because top-k edges can concentrate on a subset of lobes. This helper
    repairs isolated lobes by adding each lobe's strongest incident edge from
    the pre-sparsified causal matrix.

    Args:
        adj_matrix: Post-sparsification adjacency matrix.
        causal_matrix: Pre-sparsification causal matrix used as repair source.

    Returns:
        Tuple[Tensor, int, int]:
            repaired adjacency matrix,
            number of edges injected,
            unresolved dead-lobe count.
    """
    repaired = adj_matrix.clone()
    source = causal_matrix.clone()
    repaired.fill_diagonal_(0.0)
    source.fill_diagonal_(0.0)

    added_edges = 0

    # Iterate at most NUM_LOBES times; each pass should reduce dead-lobe count.
    for _ in range(NUM_LOBES):
        edge_mask = repaired != 0
        in_deg = edge_mask.sum(dim=0)
        out_deg = edge_mask.sum(dim=1)
        dead_nodes = torch.where((in_deg == 0) & (out_deg == 0))[0]
        if dead_nodes.numel() == 0:
            break

        progressed = False
        for node in dead_nodes.tolist():
            row_abs = source[node].abs().clone()
            col_abs = source[:, node].abs().clone()
            row_abs[node] = 0.0
            col_abs[node] = 0.0

            best_out_val, best_out_idx = torch.max(row_abs, dim=0)
            best_in_val, best_in_idx = torch.max(col_abs, dim=0)

            if float(best_out_val.item()) <= 0.0 and float(best_in_val.item()) <= 0.0:
                continue

            if float(best_out_val.item()) >= float(best_in_val.item()):
                dst = int(best_out_idx.item())
                weight = source[node, dst]
                if float(weight.abs().item()) <= 0.0:
                    continue
                if float(repaired[node, dst].abs().item()) == 0.0:
                    added_edges += 1
                repaired[node, dst] = weight
            else:
                src = int(best_in_idx.item())
                weight = source[src, node]
                if float(weight.abs().item()) <= 0.0:
                    continue
                if float(repaired[src, node].abs().item()) == 0.0:
                    added_edges += 1
                repaired[src, node] = weight

            progressed = True

        if not progressed:
            break

    repaired.fill_diagonal_(0.0)
    edge_mask = repaired != 0
    unresolved_dead_lobes = int(
        ((edge_mask.sum(dim=0) == 0) & (edge_mask.sum(dim=1) == 0)).sum().item()
    )
    return repaired, added_edges, unresolved_dead_lobes


def adaptive_sparsification(
    causal_matrix: torch.Tensor,
    method: str = None,
    min_edges: int = None,
    pvalue_matrix: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
    """
    Apply adaptive sparsification to a causal adjacency matrix.

    Four strategies are supported (select via ``method`` or ``SPARSITY_METHOD`` config):

        * **``'topk_per_node'``** – Keeps the strongest outgoing and incoming edges
            per node (``SPARSITY_TOPK_PER_NODE``), then unions those sets. This avoids
            lobe isolation caused by a single global threshold.

    * **``'adaptive_proportional'``** – Keeps a number of edges proportional to the
      total network strength ``sqrt(sum(|adj|)) × 10``, capped at ``NUM_LOBES²``.
      Preserves more edges in strongly connected graphs and fewer in weak ones.

    * **``'adaptive_statistical'``** – For Granger causality, retains edges where
      ``-log10(p) > 1.3`` (i.e.  ``p < 0.05``).  For other methods, keeps edges
      exceeding ``median + std`` of non-zero values.  Falls back to keeping the
      top ``min_edges`` if the threshold would leave too few edges.

    * **``'fixed'``** – Quantile-based threshold: retains the top
      ``(1 - SPARSITY_QUANTILE)`` fraction of edges by absolute weight
      (default: top 30 %, ``SPARSITY_QUANTILE=0.70``).

    All methods guarantee a minimum of ``min_edges`` edges remain in the graph,
    falling back to a top-k selection if the primary threshold is too aggressive.

    Args:
        causal_matrix (Tensor): Signed causal adjacency matrix, shape
                                ``(n_lobes, n_lobes)``.  Zero diagonal assumed.
        method (str, optional): Sparsification strategy.  One of
                                ``'topk_per_node'``, ``'adaptive_proportional'``,
                                ``'adaptive_statistical'``, ``'fixed'``. Defaults to
                                ``SPARSITY_METHOD`` from
                                config.
        min_edges (int, optional): Minimum number of edges to retain.  Defaults to
                                   ``MIN_EDGES_PER_GRAPH`` from config (12).

    Returns:
        Tensor: Sparsified adjacency matrix, shape ``(n_lobes, n_lobes)``.
                Zero entries indicate absent edges; non-zero values preserve the
                original signed causal weights.

    Note:
        The returned matrix is on the same device as ``causal_matrix``.
    """
    if method is None:
        method = SPARSITY_METHOD

    if min_edges is None:
        min_edges = MIN_EDGES_PER_GRAPH

    # Self-loops are not valid causal edges in this pipeline
    causal_matrix = causal_matrix.clone()
    causal_matrix.fill_diagonal_(0.0)
    offdiag_mask = ~torch.eye(NUM_LOBES, dtype=torch.bool, device=causal_matrix.device)
    fallback_info = _empty_sparsification_info()

    # Light significance pruning: remove very weak candidate edges before top-k.
    if pvalue_matrix is not None:
        p_matrix = pvalue_matrix.to(causal_matrix.device).clone()
        p_matrix.fill_diagonal_(1.0)
        if str(CAUSALITY_METHOD) == 'partial_corr_glasso':
            prune_threshold = float(max(min(PARTIAL_CORR_FDR_ALPHA, 1.0), 0.0))
        elif str(CAUSALITY_METHOD).startswith('ridge_granger'):
            prune_threshold = float(RIDGE_GRANGER_P_PRUNE_THRESHOLD)
        else:
            prune_threshold = float(LAGGED_PEARSON_P_PRUNE_THRESHOLD)
        weak_mask = (p_matrix > prune_threshold) & offdiag_mask
        retained_mask = (~weak_mask) & offdiag_mask
        fallback_info["significance_pruning_applied"] = True
        fallback_info["pruned_edge_candidates"] = int(weak_mask.sum().item())
        fallback_info["retained_candidates_after_pruning"] = int(retained_mask.sum().item())
        causal_matrix = torch.where(retained_mask, causal_matrix, torch.tensor(0.0, device=causal_matrix.device))

    abs_matrix = torch.abs(causal_matrix)
    offdiag_values = abs_matrix[offdiag_mask]

    if method == 'topk_per_node':
        # Structural safeguard: guarantee each lobe contributes strong edges before
        # any fallback repair is considered.
        k = int(max(1, min(SPARSITY_TOPK_PER_NODE, NUM_LOBES - 1)))
        keep_mask = torch.zeros_like(causal_matrix, dtype=torch.bool)

        # Keep strongest outgoing edges per source lobe.
        for src in range(NUM_LOBES):
            row_abs = abs_matrix[src].clone()
            row_abs[src] = 0.0
            non_zero_count = int((row_abs > 0).sum().item())
            if non_zero_count == 0:
                continue
            k_src = min(k, non_zero_count)
            top_idx = torch.topk(row_abs, k_src).indices
            keep_mask[src, top_idx] = True

        # Keep strongest incoming edges per target lobe.
        for dst in range(NUM_LOBES):
            col_abs = abs_matrix[:, dst].clone()
            col_abs[dst] = 0.0
            non_zero_count = int((col_abs > 0).sum().item())
            if non_zero_count == 0:
                continue
            k_dst = min(k, non_zero_count)
            top_idx = torch.topk(col_abs, k_dst).indices
            keep_mask[top_idx, dst] = True

        keep_mask &= offdiag_mask
        adj_matrix = torch.where(
            keep_mask,
            causal_matrix,
            torch.tensor(0.0, device=causal_matrix.device)
        )
        fallback_info["topk_per_node_k"] = k

        # Keep the min-edge fallback for pathological all-zero/near-zero inputs.
        num_edges = int((adj_matrix != 0).sum().item())
        if num_edges < min_edges:
            fallback_info["triggered"] = True
            fallback_info["min_edge_fallback"] = True
            flat_values = offdiag_values
            k_global = min(min_edges, flat_values.numel())
            threshold_value = torch.topk(flat_values, k_global).values[-1]
            adj_matrix = torch.where(
                (abs_matrix >= threshold_value) & offdiag_mask,
                causal_matrix,
                torch.tensor(0.0, device=causal_matrix.device)
            )

    elif method == 'adaptive_proportional':
        # Keep edges proportional to network strength
        total_strength = abs_matrix.sum().item()
        target_edges = max(min_edges, int(np.sqrt(total_strength) * 10))
        target_edges = min(target_edges, NUM_LOBES * (NUM_LOBES - 1))  # Exclude diagonal

        # Keep top target_edges by absolute weight
        flat_values = offdiag_values
        if target_edges >= len(flat_values):
            # Keep all edges
            causal_matrix.fill_diagonal_(0.0)
            fallback_info["primary_edge_count"] = int((causal_matrix != 0).sum().item())
            fallback_info["final_edge_count"] = int((causal_matrix != 0).sum().item())
            return causal_matrix, fallback_info

        threshold_value = torch.topk(flat_values, target_edges).values[-1]
        adj_matrix = torch.where(
            (abs_matrix >= threshold_value) & offdiag_mask,
            causal_matrix,
            torch.tensor(0.0, device=causal_matrix.device)
        )

    elif method == 'adaptive_statistical':
        # Keep edges above statistical significance threshold.
        # For Granger causality use subject-adaptive thresholding.
        # For other methods: use median + 1 std as threshold.
        if CAUSALITY_METHOD == 'granger':
            non_zero_vals = offdiag_values[offdiag_values > 0]
            if non_zero_vals.numel() > min_edges:
                threshold_value = torch.quantile(non_zero_vals, 0.70)
            else:
                threshold_value = torch.tensor(0.0, device=causal_matrix.device)
        else:
            non_zero = offdiag_values[offdiag_values > 0]
            if len(non_zero) > 0:
                threshold_value = non_zero.median() + non_zero.std()
            else:
                threshold_value = 0.0

        adj_matrix = torch.where(
            (abs_matrix >= threshold_value) & offdiag_mask,
            causal_matrix,
            torch.tensor(0.0, device=causal_matrix.device)
        )

        # Ensure minimum edges
        num_edges = (adj_matrix != 0).sum().item()
        if num_edges < min_edges:
            fallback_info["triggered"] = True
            fallback_info["min_edge_fallback"] = True
            # Fall back to keeping top min_edges
            flat_values = offdiag_values
            k = min(min_edges, flat_values.numel())
            threshold_value = torch.topk(flat_values, k).values[-1]
            adj_matrix = torch.where(
                (abs_matrix >= threshold_value) & offdiag_mask,
                causal_matrix,
                torch.tensor(0.0, device=causal_matrix.device)
            )

    elif method == 'fixed':
        # Quantile over off-diagonal values only — including the zero-padded diagonal
        # inflates the quantile and causes over-dense graphs.
        # Target density: GRAPH_DENSITY_TARGET (default 20%) of directed edges.
        target_q = 1.0 - GRAPH_DENSITY_TARGET
        thresh = torch.quantile(offdiag_values, target_q)
        adj_matrix = torch.where(
            (abs_matrix >= thresh) & offdiag_mask,
            causal_matrix,
            torch.tensor(0.0, device=causal_matrix.device)
        )

        # Ensure minimum edges
        num_edges = (adj_matrix != 0).sum().item()
        if num_edges < min_edges:
            fallback_info["triggered"] = True
            fallback_info["min_edge_fallback"] = True
            flat_values = offdiag_values
            k = min(min_edges, flat_values.numel())
            threshold_value = torch.topk(flat_values, k).values[-1]
            adj_matrix = torch.where(
                (abs_matrix >= threshold_value) & offdiag_mask,
                causal_matrix,
                torch.tensor(0.0, device=causal_matrix.device)
            )

    else:
        logger.warning(f"Unknown sparsity method '{method}', using fixed")
        target_q = 1.0 - GRAPH_DENSITY_TARGET
        thresh = torch.quantile(offdiag_values, target_q)
        adj_matrix = torch.where(
            (abs_matrix >= thresh) & offdiag_mask,
            causal_matrix,
            torch.tensor(0.0, device=causal_matrix.device)
        )
    adj_matrix.fill_diagonal_(0.0)
    primary_edge_mask = adj_matrix != 0
    fallback_info["primary_edge_count"] = int(primary_edge_mask.sum().item())
    fallback_info["primary_dead_lobes"] = int(
        ((primary_edge_mask.sum(dim=0) == 0) & (primary_edge_mask.sum(dim=1) == 0)).sum().item()
    )

    # Ensure no lobe is completely isolated (zero in-degree and zero out-degree).
    adj_matrix, repaired_edges, unresolved_dead = _repair_dead_lobes(
        adj_matrix=adj_matrix,
        causal_matrix=causal_matrix,
    )
    if repaired_edges > 0:
        fallback_info["triggered"] = True
        fallback_info["dead_lobe_repair"] = True
        fallback_info["dead_lobe_repair_added_edges"] = int(repaired_edges)
        logger.debug(
            "Dead-lobe repair injected %d edge(s) after sparsification",
            repaired_edges,
        )
    if unresolved_dead > 0:
        fallback_info["unresolved_dead_lobes"] = int(unresolved_dead)
        logger.debug(
            "Dead-lobe repair left %d unresolved isolated lobe(s)",
            unresolved_dead,
        )
    fallback_info["final_edge_count"] = int((adj_matrix != 0).sum().item())

    return adj_matrix, fallback_info


def construct_graph(subject_id: str, split: str, tr: float = 2.0, method: str = None, output_dir: Path = None) -> tuple[bool, dict[str, object]]:
    method = method or CAUSALITY_METHOD
    output_dir = output_dir or CAUSAL_GRAPHS_DIR
    """
    Construct causal graph for a single subject.

    Args:
        subject_id: Subject identifier
        split: Data split (train/val/test)
        tr: Repetition time in seconds (used to calculate per-subject max_lag in timepoints)

    Returns:
        Tuple[bool, Dict[str, object]]: (success, sparsification metadata)
    """

    ts_path = DATA_FINAL / split / "time_series" / f"{subject_id}_ts.npy"
    output_path = output_dir / f"{subject_id}_graph.pt"

    if not ts_path.exists():
        logger.debug(f"Time series not found for {subject_id}")
        return False, _empty_sparsification_info()

    try:
        # Load and move to GPU for fast matrix math
        ts_data = torch.from_numpy(np.load(ts_path)).float().to(DEVICE)

        # Single z-score normalisation: NiftiLabelsMasker uses standardize=False
        # (abide_download.py) so we apply exactly one z-score here before PCA.
        # Use nanmean/nan-safe std so partially-NaN ROI columns are still z-scored
        # on their valid timepoints.  All-NaN columns remain NaN and are filtered
        # by valid_roi_mask inside aggregate_to_lobes.
        ts_mean = torch.nanmean(ts_data, dim=0, keepdim=True)
        ts_var  = torch.nanmean((ts_data - ts_mean).pow(2), dim=0, keepdim=True)
        ts_std  = ts_var.sqrt()
        # Floor non-NaN std at 1e-8 (prevents division by zero for constant ROIs).
        # NaN std (all-NaN column) is intentionally preserved so valid_roi_mask
        # can detect and drop those columns in aggregate_to_lobes.
        ts_std  = torch.where(torch.isnan(ts_std), ts_std, ts_std.clamp(min=1e-8))
        ts_data = (ts_data - ts_mean) / ts_std

        # Fill NaN values with 0 before Granger computation
        # These correspond to edge ROIs that are systematically missing across all subjects
        # (ROIs 34, 35, 80, 81, 133, 166, 169 - Brainstem/Cerebellum edge regions)
        ts_data = torch.nan_to_num(ts_data, nan=0.0)

        # Validate input data
        if ts_data.shape[0] < 10:
            logger.warning(f"{subject_id}: Insufficient timepoints ({ts_data.shape[0]})")
            return False, _empty_sparsification_info()

        # 1. Smart Aggregation (PCA + Regional Homogeneity)
        ts_lobes, internal_features, zero_lobe_mask = aggregate_to_lobes(ts_data)

        # 2. Compute 12x12 Causal Matrix (Phase 1: Granger causality with cleaned signals)
        # Calculate max_lag in timepoints based on subject-specific TR
        max_lag_timepoints = max(1, int(GRANGER_MAX_LAG_SECONDS / tr))
        causal_metadata: dict[str, torch.Tensor] = {}
        causal_result = compute_causality_matrix(
            ts_lobes,
            max_lag=max_lag_timepoints,
            return_metadata=True,
        )
        if isinstance(causal_result, tuple):
            causal_matrix, causal_metadata = causal_result
        else:
            causal_matrix = causal_result
            causal_metadata = {}
        causal_matrix.fill_diagonal_(0.0)

        #  CRITICAL FIX: VALIDATE BEFORE SPARSIFICATION
        # Check if matrix is all zeros (this would cause issues)
        if (causal_matrix == 0).all():
            logger.warning(f"{subject_id}: Causal matrix is all zeros - skipping")
            return False, _empty_sparsification_info()

        # Log pre-sparsification statistics
        pre_sparse_stats = {
            'max': float(causal_matrix.abs().max()),
            'mean': float(causal_matrix.abs().mean()),
            'non_zero': int((causal_matrix != 0).sum())
        }

        # 3. Adaptive Sparsification (Phase 1: subject-specific thresholding)
        pvalue_matrix = causal_metadata.get('pvalue_matrix')
        adj_matrix, sparsification_info = adaptive_sparsification(
            causal_matrix,
            pvalue_matrix=pvalue_matrix,
        )

        #  CRITICAL FIX: VALIDATE AFTER SPARSIFICATION
        num_edges = (adj_matrix != 0).sum().item()

        if num_edges == 0:
            # This subject would have ZERO edges - not usable for GNN
            logger.warning(
                f"{subject_id}: Zero edges after sparsification | "
                f"Pre-sparse: max={pre_sparse_stats['max']:.4f}, "
                f"mean={pre_sparse_stats['mean']:.4f}, "
                f"non_zero={pre_sparse_stats['non_zero']} | "
                f"Method: {CAUSALITY_METHOD}, Sparsity: {SPARSITY_METHOD}"
            )
            return False, sparsification_info

        # Log success statistics
        if str(CAUSALITY_METHOD).startswith('ridge_granger'):
            high_conf_threshold = float(RIDGE_GRANGER_HIGH_CONF_P_THRESHOLD)
        elif str(CAUSALITY_METHOD) == 'lagged_pearson':
            high_conf_threshold = float(LAGGED_PEARSON_P_SELECT_THRESHOLD)
        else:
            high_conf_threshold = float('nan')

        pvalue_matrix_for_stats = causal_metadata.get('pvalue_matrix')
        if pvalue_matrix_for_stats is not None and np.isfinite(high_conf_threshold):
            high_confidence_edges_pre_topk = int((pvalue_matrix_for_stats < high_conf_threshold).sum().item())
        else:
            high_confidence_edges_pre_topk = 0

        post_sparse_stats = {
            'edges': num_edges,
            'density': num_edges / (NUM_LOBES * (NUM_LOBES - 1)),
            'max_weight': float(adj_matrix.abs().max()),
            'mean_weight': float(adj_matrix[adj_matrix != 0].abs().mean()),
            'high_confidence_edges_pre_topk': high_confidence_edges_pre_topk,
            'partial_corr_fdr_enabled': bool(PARTIAL_CORR_FDR_ENABLED) if str(CAUSALITY_METHOD) == 'partial_corr_glasso' else False,
            'partial_corr_fdr_alpha': float(PARTIAL_CORR_FDR_ALPHA) if str(CAUSALITY_METHOD) == 'partial_corr_glasso' else None,
            'partial_corr_fdr_significant_edges_pre_topk': int(
                causal_metadata.get('fdr_significant_mask', torch.zeros_like(causal_matrix, dtype=torch.bool)).sum().item()
            ) if str(CAUSALITY_METHOD) == 'partial_corr_glasso' else 0,
            'topk_per_node_k': int(sparsification_info.get('topk_per_node_k', 0)),
            'sparsification_fallback_triggered': bool(sparsification_info.get('triggered', False)),
            'significance_pruning_applied': bool(sparsification_info.get('significance_pruning_applied', False)),
            'pruned_edge_candidates': int(sparsification_info.get('pruned_edge_candidates', 0)),
            'retained_candidates_after_pruning': int(sparsification_info.get('retained_candidates_after_pruning', 0)),
            'min_edge_fallback': bool(sparsification_info.get('min_edge_fallback', False)),
            'dead_lobe_repair': bool(sparsification_info.get('dead_lobe_repair', False)),
            'dead_lobe_repair_added_edges': int(sparsification_info.get('dead_lobe_repair_added_edges', 0)),
            'unresolved_dead_lobes': int(sparsification_info.get('unresolved_dead_lobes', 0)),
            'primary_edges_before_repair': int(sparsification_info.get('primary_edge_count', 0)),
            'primary_dead_lobes_before_repair': int(sparsification_info.get('primary_dead_lobes', 0)),
        }

        logger.debug(
            f"{subject_id}: Constructed graph | "
            f"Edges: {post_sparse_stats['edges']}, "
            f"Density: {post_sparse_stats['density']:.2%}, "
            f"Max weight: {post_sparse_stats['max_weight']:.3f}"
        )

        # 4. Save structured data for Graph Factory
        graph_package = {
            'adj': adj_matrix.cpu(),
            'internal_features': internal_features.cpu(),  # (12, 2) ReHo features
            'zero_lobe_mask': zero_lobe_mask.cpu(),        # (12,) bool — True = atlas gap / zero-signal
            'edge_confidence': causal_metadata.get('confidence_matrix', torch.zeros_like(causal_matrix)).cpu(),
            'edge_pvalues': causal_metadata.get('pvalue_matrix', torch.ones_like(causal_matrix)).cpu(),
            'selected_lag_matrix': causal_metadata.get('selected_lag_matrix', torch.zeros_like(causal_matrix, dtype=torch.long)).cpu(),
            'low_confidence_mask': causal_metadata.get('low_confidence_mask', torch.zeros_like(causal_matrix, dtype=torch.bool)).cpu(),
            'subject_id': subject_id,
            'lobe_order': [LOBE_NAMES[i] for i in range(NUM_LOBES)],
            'sparsification_info': sparsification_info,
            'stats': post_sparse_stats  # Useful for debugging
        }

        torch.save(graph_package, output_path)
        return True, sparsification_info

    except Exception as e:
        logger.error(f"Causal error for {subject_id}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False, _empty_sparsification_info()


def _construct_single_graph(args: tuple[str, str, float]) -> tuple[str, bool, dict[str, object], str | None]:
    """
    Process a single subject to construct causal graph.
    Designed to be run in parallel via joblib.

    Args:
        args: Tuple of (subject_id, split, tr)

    Returns:
        Tuple of (subject_id, success, fallback, failure_reason)
    """
    subject_id, split, tr = args
    result, fallback_info = construct_graph(subject_id, split, tr=tr)

    if result:
        return subject_id, True, fallback_info, None
    else:
        ts_path = DATA_FINAL / split / "time_series" / f"{subject_id}_ts.npy"
        reason = "missing_ts" if not ts_path.exists() else "zero_edges"
        return subject_id, False, fallback_info, reason


def main(n_jobs: int = -1):
    """Construct causal graphs for all subjects.

    Args:
        n_jobs: Number of parallel workers (-1 = all cores, default: -1)
    """
    logger.info("="*60)
    logger.info(f"CONSTRUCTING 12×12 CAUSAL GRAPHS (Method={CAUSALITY_METHOD}, MaxLag={GRANGER_MAX_LAG_SECONDS}s)")
    if SPARSITY_METHOD == "topk_per_node":
        logger.info(f"Sparsity: top-k per node (k={SPARSITY_TOPK_PER_NODE})")
    else:
        logger.info(f"Sparsity: Keep top {(1-SPARSITY_QUANTILE)*100:.0f}% of edges")
    logger.info(f"Parallel workers: {n_jobs}")
    logger.info("="*60)

    CAUSAL_GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MASTER_MANIFEST)

    tasks = [
        (row['subject_id'], row['split'], row.get('TR', 2.0))
        for _, row in manifest.iterrows()
    ]

    logger.info(f"Processing {len(tasks)} subjects...")

    results = Parallel(n_jobs=n_jobs, prefer="processes", verbose=0)(
        delayed(_construct_single_graph)(task) for task in tqdm(tasks, desc="Building Graphs", mininterval=10.0)
    )

    stats = {
        'total': len(tasks),
        'success': 0,
        'failed': 0,
        'zero_edges': 0,
        'missing_ts': 0
    }
    fallback_by_group: dict[int, int] = {}
    min_edge_fallback_by_group: dict[int, int] = {}
    dead_repair_by_group: dict[int, int] = {}
    total_fallback_triggered = 0
    total_min_edge_fallback = 0
    total_dead_repair = 0

    for subject_id, success, fallback_info, reason in results:
        if success:
            stats['success'] += 1
            dx_group_row = manifest[manifest['subject_id'] == subject_id]
            if not dx_group_row.empty:
                dx_group = int(dx_group_row.iloc[0].get('DX_GROUP', -1))
                if bool(fallback_info.get('triggered', False)):
                    fallback_by_group[dx_group] = fallback_by_group.get(dx_group, 0) + 1
                    total_fallback_triggered += 1
                if bool(fallback_info.get('min_edge_fallback', False)):
                    min_edge_fallback_by_group[dx_group] = min_edge_fallback_by_group.get(dx_group, 0) + 1
                    total_min_edge_fallback += 1
                if bool(fallback_info.get('dead_lobe_repair', False)):
                    dead_repair_by_group[dx_group] = dead_repair_by_group.get(dx_group, 0) + 1
                    total_dead_repair += 1
        else:
            stats['failed'] += 1
            if reason == "missing_ts":
                stats['missing_ts'] += 1
            else:
                stats['zero_edges'] += 1

    # Calculate zero_edges count
    stats['zero_edges'] = stats['failed'] - stats['missing_ts']

    # Print comprehensive report
    logger.info("\n" + "="*60)
    logger.info("GRAPH CONSTRUCTION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total subjects: {stats['total']}")
    logger.info(f"✓ Successfully constructed: {stats['success']}")
    logger.info(f"✗ Failed: {stats['failed']}")
    logger.info(f"  ↳ Zero edges: {stats['zero_edges']}")
    logger.info(f"  ↳ Missing time series: {stats['missing_ts']}")
    logger.info(f"\nSuccess rate: {stats['success']/stats['total']*100:.1f}%")
    logger.info(f"Output directory: {CAUSAL_GRAPHS_DIR}")
    logger.info("="*60)

    # Report sparsification interventions by diagnostic group.
    if stats['success'] > 0:
        logger.info(
            "Sparsification interventions: triggered=%d (%.1f%%), min-edge=%d (%.1f%%), dead-lobe-repair=%d (%.1f%%)",
            total_fallback_triggered,
            100.0 * total_fallback_triggered / stats['success'],
            total_min_edge_fallback,
            100.0 * total_min_edge_fallback / stats['success'],
            total_dead_repair,
            100.0 * total_dead_repair / stats['success'],
        )

    if fallback_by_group:
        logger.warning(
            "Sparsification intervention triggered by DX_GROUP:\n"
            + "\n".join(
                f"  DX_GROUP={gid}: {cnt} subjects"
                for gid, cnt in sorted(fallback_by_group.items())
            )
        )

    if min_edge_fallback_by_group:
        logger.warning(
            "Min-edge fallback by DX_GROUP:\n"
            + "\n".join(
                f"  DX_GROUP={gid}: {cnt} subjects"
                for gid, cnt in sorted(min_edge_fallback_by_group.items())
            )
        )

    if dead_repair_by_group:
        logger.warning(
            "Dead-lobe repair by DX_GROUP:\n"
            + "\n".join(
                f"  DX_GROUP={gid}: {cnt} subjects"
                for gid, cnt in sorted(dead_repair_by_group.items())
            )
        )

        # Flag potential class-imbalanced sparsification artifacts.
        # Supports both encodings: ASD/Control as (1/0) or (1/2).
        asd_count = fallback_by_group.get(1, 0)
        ctrl_count = fallback_by_group.get(0, fallback_by_group.get(2, 0))
        if asd_count > 2 * max(ctrl_count, 1):
            logger.warning(
                "ASD subjects trigger sparsification intervention >2x more than Controls "
                "(ASD=%d, Control=%d). Investigate graph sparsification thresholds.",
                asd_count,
                ctrl_count,
            )

    # Warning if too many zero-edge graphs
    if stats['zero_edges'] > stats['total'] * 0.1:  # More than 10%
        logger.warning(
            f"\n⚠️  HIGH ZERO-EDGE RATE: {stats['zero_edges']} subjects ({stats['zero_edges']/stats['total']*100:.1f}%)"
        )
        logger.warning(
            f"Consider lowering SPARSITY_QUANTILE from {SPARSITY_QUANTILE} to 0.70 or 0.60"
        )
        logger.warning(
            "This will keep more edges (top 30% or 40% instead of top 20%)"
        )




# ─── TASK 2: Multi-View Causal Graph Construction (DD-010) ───────────────────────

def construct_multiview_graphs(
    subject_id: str,
    time_series: torch.Tensor,
    lobe_to_roi: dict,
    tr: float,
    output_dir: Path,
    rng: np.random.Generator = None,
) -> bool:
    """
    Generate 6 causal graph views per subject for CausalInvarianceLoss training.

    Views:
        base:            Existing saved graph (reused from causal_graphs/); just
                         reads from disk rather than recomputing.
        extended_lag:    Granger with max_lag = round(GRANGER_MAX_LAG_SECONDS / tr * 1.5).
        bootstrap_0/1/2: Granger fitted on 80% random timepoint subsample (seeds 0/1/2).
        high_confidence: Top-15% edges only from base, with remainder zeroed.

    All 6 views are saved as a single dict to:
        output_dir / subject_id / "multiview_graphs.pt"

    Args:
        subject_id: ABIDE subject identifier string.
        time_series: Raw time series tensor (T, num_rois).
        lobe_to_roi: Dict from lobe index to list of ROI indices.
        tr: Repetition time in seconds.
        output_dir: Root directory for multiview outputs (CAUSAL_GRAPHS_MULTIVIEW_DIR).
        rng: Optional numpy Generator for reproducible bootstrap sampling.

    Returns:
        True if all 6 views were successfully generated and saved, False otherwise.
    """

    if rng is None:
        rng = np.random.default_rng(seed=42)

    base_path = CAUSAL_GRAPHS_DIR / f"{subject_id}_graph.pt"
    if not base_path.exists():
        logger.warning("Base graph not found for %s; skipping multiview construction.", subject_id)
        return False

    try:
        base_graph = torch.load(base_path, weights_only=False)
        adj_base = base_graph['adj'].float()
    except Exception as e:
        logger.warning("Failed to load base graph for %s: %s", subject_id, e)
        return False

    ts_np = time_series.numpy() if isinstance(time_series, torch.Tensor) else time_series
    T, num_rois = ts_np.shape

    # Very short runs cannot support stable Granger estimation; build safe
    # fallback views from the already-validated base graph.
    if T < 12:
        logger.warning(
            "%s: only %d timepoints available for multiview Granger; "
            "using base-derived fallback views",
            subject_id,
            T,
        )
        adj_extended = adj_base.clone()
        adj_bootstraps = [adj_base.clone(), adj_base.clone(), adj_base.clone()]
    else:
        # Use the same NaN-safe z-score + lobe aggregation path as base graph
        # construction to prevent NaN propagation into multiview branches.
        ts_tensor = torch.as_tensor(ts_np, dtype=torch.float32, device=DEVICE)
        ts_mean = torch.nanmean(ts_tensor, dim=0, keepdim=True)
        ts_var = torch.nanmean((ts_tensor - ts_mean).pow(2), dim=0, keepdim=True)
        ts_std = ts_var.sqrt()
        ts_std = torch.where(torch.isnan(ts_std), ts_std, ts_std.clamp(min=1e-8))
        ts_z = (ts_tensor - ts_mean) / ts_std

        ts_lobes, _, _ = aggregate_to_lobes(ts_z)
        lobe_ts = ts_lobes.detach().cpu().numpy()
        lobe_ts = np.nan_to_num(lobe_ts, nan=0.0, posinf=0.0, neginf=0.0)

        # Clamp lags so Granger has enough samples (requires n_timepoints >= lag + 10).
        max_supported_lag = max(1, T - 10)
        base_lag = max(1, round(GRANGER_MAX_LAG_SECONDS / max(tr, 0.1)))
        base_lag = min(base_lag, max_supported_lag)
        ext_lag = min(max(1, round(base_lag * 1.5)), max_supported_lag)

        # 1. Extended-lag view
        try:
            adj_ext_np = compute_granger_causality(lobe_ts, max_lag=ext_lag)
            if np.count_nonzero(adj_ext_np) == 0:
                raise ValueError("extended-lag causality returned all-zero matrix")
            adj_extended = torch.tensor(adj_ext_np, dtype=torch.float32)
        except Exception as e:
            logger.warning("Extended-lag Granger failed for %s: %s", subject_id, e)
            adj_extended = adj_base.clone()

        # 2-4. Bootstrap views (80% timepoint subsample, 3 seeds)
        adj_bootstraps = []
        for seed in range(3):
            rng_seed = np.random.default_rng(seed=seed)
            n_keep = max(int(T * 0.80), base_lag + 10)
            n_keep = min(n_keep, T)

            # If the subsample cannot support Granger for the requested lag,
            # avoid generating a degenerate all-zero adjacency.
            if n_keep < base_lag + 10:
                logger.warning(
                    "Bootstrap %d skipped for %s: n_keep=%d insufficient for lag=%d; "
                    "using base adjacency",
                    seed,
                    subject_id,
                    n_keep,
                    base_lag,
                )
                adj_bootstraps.append(adj_base.clone())
                continue

            idx = rng_seed.choice(T, size=n_keep, replace=False)
            idx = np.sort(idx)
            lobe_ts_sub = lobe_ts[idx]
            try:
                adj_np = compute_granger_causality(lobe_ts_sub, max_lag=base_lag)
                if np.count_nonzero(adj_np) == 0:
                    raise ValueError("bootstrap causality returned all-zero matrix")
                adj_bootstraps.append(torch.tensor(adj_np, dtype=torch.float32))
            except Exception as e:
                logger.warning("Bootstrap %d Granger failed for %s: %s", seed, subject_id, e)
                adj_bootstraps.append(adj_base.clone())

    # 5. High-confidence view: keep top-15% edges from base
    adj_flat = adj_base.flatten()
    nonzero_vals = adj_flat[adj_flat > 0]
    if nonzero_vals.numel() > 0:
        threshold_val = float(torch.quantile(nonzero_vals, 0.85))
        adj_high_conf = (adj_base >= threshold_val).float() * adj_base
    else:
        adj_high_conf = adj_base.clone()

    views = {
        "base":             adj_base,
        "extended_lag":     adj_extended,
        "bootstrap_0":      adj_bootstraps[0],
        "bootstrap_1":      adj_bootstraps[1],
        "bootstrap_2":      adj_bootstraps[2],
        "high_confidence":  adj_high_conf,
    }

    out_path = output_dir / subject_id / "multiview_graphs.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    package = {
        "views": {k: v.cpu() for k, v in views.items()},
        "subject_id": subject_id,
        "lobe_order": [LOBE_NAMES[i] for i in range(NUM_LOBES)],
        "fallback_views": {
            "extended_lag": bool(torch.equal(views["extended_lag"], views["base"])),
            "bootstrap_0": bool(torch.equal(views["bootstrap_0"], views["base"])),
            "bootstrap_1": bool(torch.equal(views["bootstrap_1"], views["base"])),
            "bootstrap_2": bool(torch.equal(views["bootstrap_2"], views["base"])),
            "high_confidence": bool(torch.equal(views["high_confidence"], views["base"])),
        },
    }
    torch.save(package, out_path)
    return True


def _assess_multiview_generation_quality(multiview_dir: Path) -> dict:
    """Compute zero-edge rates per multiview type from generated artifacts."""
    files = sorted(multiview_dir.glob("*/multiview_graphs.pt"))
    zero_counts = dict.fromkeys(_MULTIVIEW_VIEW_ORDER, 0)
    checked = 0

    for fp in files:
        try:
            payload = torch.load(fp, map_location="cpu", weights_only=False)
        except Exception:
            continue

        if not isinstance(payload, dict):
            continue

        views = payload.get("views", payload)
        if not isinstance(views, dict):
            continue

        checked += 1
        for view in _MULTIVIEW_VIEW_ORDER:
            adj = views.get(view)
            if adj is None:
                zero_counts[view] += 1
                continue

            if torch.is_tensor(adj):
                adj_t = adj.detach().cpu().float()
            else:
                adj_t = torch.as_tensor(adj, dtype=torch.float32)

            if adj_t.ndim != 2 or adj_t.shape[0] != adj_t.shape[1]:
                zero_counts[view] += 1
                continue

            if int((adj_t != 0).sum().item()) == 0:
                zero_counts[view] += 1

    rates = {
        view: (zero_counts[view] / max(checked, 1))
        for view in _MULTIVIEW_VIEW_ORDER
    }
    failing = [
        view
        for view in _MULTIVIEW_VIEW_ORDER
        if view != "base" and rates[view] > MULTIVIEW_GENERATION_MAX_ZERO_EDGE_RATE
    ]

    return {
        "checked_packages": checked,
        "zero_edge_rates": rates,
        "failing_views": failing,
    }


def _package_has_degenerate_non_base_views(package_path: Path) -> bool:
    """Return True when an existing multiview package is malformed/degenerate."""
    try:
        payload = torch.load(package_path, map_location="cpu", weights_only=False)
    except Exception:
        return True

    if not isinstance(payload, dict):
        return True

    views = payload.get("views", payload)
    if not isinstance(views, dict):
        return True

    for view in _MULTIVIEW_VIEW_ORDER:
        if view == "base":
            continue

        adj = views.get(view)
        if adj is None:
            return True

        if torch.is_tensor(adj):
            adj_t = adj.detach().cpu().float()
        else:
            adj_t = torch.as_tensor(adj, dtype=torch.float32)

        if adj_t.ndim != 2 or adj_t.shape[0] != adj_t.shape[1]:
            return True

        if int((adj_t != 0).sum().item()) == 0:
            return True

    return False


def main_multiview():
    """
    Task 2 entry point: generate multi-view causal graphs for all subjects.

    Reads subjects from MASTER_MANIFEST.  For each subject that already has
    a base graph in CAUSAL_GRAPHS_DIR, constructs 5 additional views and
    saves to CAUSAL_GRAPHS_MULTIVIEW_DIR.

    Usage (via pipeline registry with --multiview flag, or directly):
        python -m src.features.construct_causal --multiview
    """
    from src.core.config import CAUSAL_GRAPHS_MULTIVIEW_DIR, MASTER_MANIFEST

    logger.info("=" * 70)
    logger.info("MULTI-VIEW CAUSAL GRAPH CONSTRUCTION (Task 2 — DD-010)")
    logger.info("=" * 70)

    manifest = pd.read_csv(MASTER_MANIFEST)
    all_subjects = manifest['subject_id'].astype(str).tolist()

    # Build lobe_to_roi from config mapping (lobe index -> list[0-based ROI indices]).
    lobe_to_roi: dict[int, list] = {
        int(lobe_idx): [int(roi_idx) for roi_idx in roi_indices]
        for lobe_idx, roi_indices in LOBE_MAPPING.items()
    }

    CAUSAL_GRAPHS_MULTIVIEW_DIR.mkdir(parents=True, exist_ok=True)

    success, regenerated, skipped, failed = 0, 0, 0, 0
    for sub_id in tqdm(all_subjects, desc="Multi-view graphs"):
        out_file = CAUSAL_GRAPHS_MULTIVIEW_DIR / sub_id / "multiview_graphs.pt"
        base_path = CAUSAL_GRAPHS_DIR / f"{sub_id}_graph.pt"

        if out_file.exists():
            if not base_path.exists():
                logger.warning(
                    "Removing stale multiview package for %s because base graph is missing.",
                    sub_id,
                )
                try:
                    out_file.unlink()
                except Exception as exc:
                    logger.warning("Failed to remove stale package for %s: %s", sub_id, exc)
                failed += 1
                continue

            needs_regen = False
            if MULTIVIEW_GENERATION_ENFORCE_QUALITY_GATE:
                needs_regen = _package_has_degenerate_non_base_views(out_file)

            if not needs_regen:
                skipped += 1
                continue

            logger.info("Regenerating degenerate multiview package for %s", sub_id)
            try:
                out_file.unlink()
                regenerated += 1
            except Exception as exc:
                logger.warning("Failed to remove existing package for %s: %s", sub_id, exc)
                failed += 1
                continue

        if not base_path.exists():
            failed += 1
            continue

        # Load time series — needed for bootstrap/extended-lag views
        # Try standard final split layout
        ts_path = None
        for split in ("train", "val", "test"):
            candidate = DATA_FINAL / split / "time_series" / f"{sub_id}_ts.npy"
            if candidate.exists():
                ts_path = candidate
                break

        if ts_path is None:
            # Fall back to processed dir
            from src.core.paths import DATA_TIME_SERIES as _DTS
            candidate2 = _DTS / f"{sub_id}_ts.npy"
            if candidate2.exists():
                ts_path = candidate2

        if ts_path is None:
            logger.debug("No time series for %s; copying base graph only.", sub_id)
            # Still create multiview with base-derived views only (no bootstrap)
            base_graph = torch.load(base_path, weights_only=False)
            adj_base = base_graph['adj'].float()
            adj_flat = adj_base.flatten()
            nz = adj_flat[adj_flat > 0]
            adj_hc = (adj_base >= float(torch.quantile(nz, 0.85))).float() * adj_base if nz.numel() > 0 else adj_base.clone()
            views = {
                "base": adj_base, "extended_lag": adj_base.clone(),
                "bootstrap_0": adj_base.clone(), "bootstrap_1": adj_base.clone(),
                "bootstrap_2": adj_base.clone(), "high_confidence": adj_hc,
            }
            out_file.parent.mkdir(parents=True, exist_ok=True)
            package = {
                "views": {k: v.cpu() for k, v in views.items()},
                "subject_id": sub_id,
                "lobe_order": [LOBE_NAMES[i] for i in range(NUM_LOBES)],
            }
            torch.save(package, out_file)
            success += 1
            continue

        try:
            ts_np = np.load(ts_path)  # (T, num_rois)
            row = manifest[manifest['subject_id'].astype(str) == sub_id]
            tr = float(row.get('TR', pd.Series([2.0])).values[0]) if len(row) > 0 else 2.0
            ts_tensor = torch.tensor(ts_np, dtype=torch.float32)
            ok = construct_multiview_graphs(
                subject_id=sub_id,
                time_series=ts_tensor,
                lobe_to_roi=lobe_to_roi,
                tr=tr,
                output_dir=CAUSAL_GRAPHS_MULTIVIEW_DIR,
            )
            if ok:
                success += 1
            else:
                failed += 1
        except Exception as exc:
            logger.warning("Multi-view construction failed for %s: %s", sub_id, exc)
            failed += 1

    logger.info(
        "Multi-view construction complete: %d success | %d regenerated | %d skipped | %d failed",
        success, regenerated, skipped, failed,
    )
    logger.info("Output directory: %s", CAUSAL_GRAPHS_MULTIVIEW_DIR)

    if MULTIVIEW_GENERATION_ENFORCE_QUALITY_GATE:
        quality = _assess_multiview_generation_quality(CAUSAL_GRAPHS_MULTIVIEW_DIR)
        checked = quality["checked_packages"]
        rates = quality["zero_edge_rates"]
        logger.info(
            "Multiview generation quality: checked=%d | base=%.1f%% | ext=%.1f%% | b0=%.1f%% | b1=%.1f%% | b2=%.1f%% | hc=%.1f%%",
            checked,
            100.0 * rates.get("base", 1.0),
            100.0 * rates.get("extended_lag", 1.0),
            100.0 * rates.get("bootstrap_0", 1.0),
            100.0 * rates.get("bootstrap_1", 1.0),
            100.0 * rates.get("bootstrap_2", 1.0),
            100.0 * rates.get("high_confidence", 1.0),
        )

        policy = str(MULTIVIEW_GENERATION_POLICY).strip().lower()
        if policy not in {"fail", "warn"}:
            logger.warning(
                "Unknown MULTIVIEW_GENERATION_POLICY=%r; falling back to 'warn'",
                MULTIVIEW_GENERATION_POLICY,
            )
            policy = "warn"

        if checked == 0:
            msg = "Multiview quality gate found zero readable multiview packages"
            if policy == "fail":
                raise RuntimeError(msg)
            logger.warning(msg)
        elif quality["failing_views"]:
            msg = (
                "Multiview generation quality gate failed: non-base views exceed "
                f"max zero-edge rate {MULTIVIEW_GENERATION_MAX_ZERO_EDGE_RATE:.2f}; "
                f"failing views={quality['failing_views']}"
            )
            if policy == "fail":
                raise RuntimeError(msg)
            logger.warning(msg)


if __name__ == "__main__":
    import argparse
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--multiview", action="store_true", help="Run multi-view graph construction (Task 2)")
    _parser.add_argument("--n-jobs", type=int, default=-1, help="Number of parallel workers (-1=all cores)")
    _args = _parser.parse_args()
    if _args.multiview:
        main_multiview()
    else:
        main(n_jobs=_args.n_jobs)
