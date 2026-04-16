import logging
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Tuple
import sys

# Setup paths
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import (
    LOBE_MAPPING, NUM_LOBES, LOBE_NAMES, SPARSITY_QUANTILE,
    DATA_FINAL, MASTER_MANIFEST, CAUSAL_GRAPHS_DIR, DEVICE,
    CAUSALITY_METHOD, GRANGER_MAX_LAG, GRANGER_MAX_LAG_SECONDS, SPARSITY_METHOD,
    MIN_EDGES_PER_GRAPH, GRAPH_DENSITY_TARGET,
)

# Fixed lag for lagged-Pearson fallback path.
_LAGGED_PEARSON_LAG = 1
from src.features.causal_inference import (
    compute_granger_causality,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Tracks which lobe IDs have already emitted a zero-signal warning so that atlas
# coverage-gap messages appear once per process run rather than once per subject.
_zero_lobe_warned: set = set()


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


def aggregate_to_lobes(ts_raw: torch.Tensor) -> tuple:
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
            if lobe_id not in _zero_lobe_warned:
                logger.warning(
                    f"Lobe {lobe_id} ({LOBE_NAMES[lobe_id]}): No matching ROIs in atlas. "
                    "Using zero-signal. (Subsequent subjects with the same gap suppressed.)"
                )
                _zero_lobe_warned.add(lobe_id)
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
            if lobe_id not in _zero_lobe_warned:
                logger.warning(
                    f"Lobe {lobe_id} ({LOBE_NAMES[lobe_id]}): all {len(indices)} ROIs "
                    "have NaN time series (atlas coverage gap). Using zero-signal fallback. "
                    "(Subsequent subjects with the same gap suppressed.)"
                )
                _zero_lobe_warned.add(lobe_id)
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


def _compute_lagged_pearson_matrix(ts_lobe: torch.Tensor) -> torch.Tensor:
    """Compute lagged Pearson correlation matrix for fallback causality."""
    # Validate input
    if ts_lobe.shape[0] <= _LAGGED_PEARSON_LAG:
        logger.warning("Insufficient timepoints for lagged correlation")
        return torch.zeros(NUM_LOBES, NUM_LOBES, device=ts_lobe.device)
    
    # Check for NaN/Inf in input
    if torch.isnan(ts_lobe).any() or torch.isinf(ts_lobe).any():
        logger.warning("Input contains NaN/Inf values - returning zero matrix")
        return torch.zeros(NUM_LOBES, NUM_LOBES, device=ts_lobe.device)
    
    # 1. Standardize (Z-Score) signals for valid correlation
    ts_mean = ts_lobe.mean(dim=0)
    ts_std = ts_lobe.std(dim=0) + 1e-6  # Prevent division by zero
    ts_std = (ts_lobe - ts_mean) / ts_std
    
    # 2. Slice for Lag (t-1 -> t)
    ts_prev = ts_std[:-_LAGGED_PEARSON_LAG]
    ts_curr = ts_std[_LAGGED_PEARSON_LAG:]
    
    # 3. Compute Adjacency Matrix (12x12 for 12 regions)
    directed_adj = (ts_prev.T @ ts_curr) / (ts_std.shape[0] - _LAGGED_PEARSON_LAG)
    
    # 4. Validate output
    if torch.isnan(directed_adj).any() or torch.isinf(directed_adj).any():
        logger.warning("Lagged correlation produced NaN/Inf - returning zero matrix")
        return torch.zeros(NUM_LOBES, NUM_LOBES, device=ts_lobe.device)
    
    return directed_adj


def compute_causality_matrix(ts_lobe: torch.Tensor, method: str = None, max_lag: int = None) -> torch.Tensor:
    """
    Compute causal adjacency matrix using configured method.
    
    Args:
        ts_lobe: Time series for lobes (shape: [timepoints, n_lobes])
        method: Causality method ('granger', 'transfer_entropy', 'lagged_pearson')
                If None, uses CAUSALITY_METHOD from config
        max_lag: Max lag in timepoints for Granger causality. If None, uses GRANGER_MAX_LAG from config.
                 For multi-site studies, this allows per-subject adaptation based on TR.
    
    Returns:
        Causal adjacency matrix (shape: [n_lobes, n_lobes])
    """
    if method is None:
        method = CAUSALITY_METHOD
    
    if max_lag is None:
        max_lag = GRANGER_MAX_LAG
    
    # Convert to numpy for causal inference methods
    ts_numpy = ts_lobe.cpu().numpy()
    
    try:
        if method == 'granger':
            # CPU-only Granger (GPU path removed in Task 6 — DD-014)
            logger.debug(f"Computing Granger causality on CPU (max_lag={max_lag} timepoints)")
            causal_matrix_np = compute_granger_causality(
                ts_numpy,
                max_lag=max_lag
            )

        elif method == 'lagged_pearson':
            logger.debug(f"Computing lagged Pearson correlation (lag={_LAGGED_PEARSON_LAG})")
            pearson_adj = _compute_lagged_pearson_matrix(ts_lobe)
            # Fisher-Z transform: z = arctanh(r) — stabilises variance of correlations
            # Clips to (-0.999, 0.999) to avoid ±inf on perfect correlations
            pearson_adj = pearson_adj.clamp(-0.999, 0.999)
            return torch.arctanh(pearson_adj)
        
        else:
            logger.warning(f"Unknown causality method '{method}', falling back to lagged_pearson")
            fallback_adj = _compute_lagged_pearson_matrix(ts_lobe)
            fallback_adj = fallback_adj.clamp(-0.999, 0.999)
            return torch.arctanh(fallback_adj)
        
        # Convert back to torch tensor
        causal_matrix = torch.from_numpy(causal_matrix_np).float().to(DEVICE)
        
        return causal_matrix
    
    except Exception as e:
        logger.warning(f"Causality computation failed ({method}): {e}, falling back to lagged_pearson")
        return _compute_lagged_pearson_matrix(ts_lobe)


def adaptive_sparsification(
    causal_matrix: torch.Tensor,
    method: str = None,
    min_edges: int = None
) -> torch.Tensor:
    """
    Apply adaptive sparsification to a causal adjacency matrix.

    Three strategies are supported (select via ``method`` or ``SPARSITY_METHOD`` config):

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
                                ``'adaptive_proportional'``, ``'adaptive_statistical'``,
                                ``'fixed'``.  Defaults to ``SPARSITY_METHOD`` from
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
    abs_matrix = torch.abs(causal_matrix)
    offdiag_mask = ~torch.eye(NUM_LOBES, dtype=torch.bool, device=causal_matrix.device)
    offdiag_values = abs_matrix[offdiag_mask]
    
    fallback_triggered = False

    if method == 'adaptive_proportional':
        # Keep edges proportional to network strength
        total_strength = abs_matrix.sum().item()
        target_edges = max(min_edges, int(np.sqrt(total_strength) * 10))
        target_edges = min(target_edges, NUM_LOBES * (NUM_LOBES - 1))  # Exclude diagonal
        
        # Keep top target_edges by absolute weight
        flat_values = offdiag_values
        if target_edges >= len(flat_values):
            # Keep all edges
            causal_matrix.fill_diagonal_(0.0)
            return causal_matrix, fallback_triggered
        
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
            fallback_triggered = True
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
            fallback_triggered = True
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
    return adj_matrix, fallback_triggered


def construct_graph(subject_id: str, split: str, tr: float = 2.0) -> Tuple[bool, bool]:
    """
    Construct causal graph for a single subject.
    
    Args:
        subject_id: Subject identifier
        split: Data split (train/val/test)
        tr: Repetition time in seconds (used to calculate per-subject max_lag in timepoints)
    
    Returns:
        Tuple[bool, bool]: (success, used_fallback_sparsification)
    """

    ts_path = DATA_FINAL / split / "time_series" / f"{subject_id}_ts.npy"
    output_path = CAUSAL_GRAPHS_DIR / f"{subject_id}_graph.pt"
    
    if not ts_path.exists():
        logger.debug(f"Time series not found for {subject_id}")
        return False, False
    
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

        # Validate input data
        if ts_data.shape[0] < 10:
            logger.warning(f"{subject_id}: Insufficient timepoints ({ts_data.shape[0]})")
            return False, False
        
        # 1. Smart Aggregation (PCA + Regional Homogeneity)
        ts_lobes, internal_features, zero_lobe_mask = aggregate_to_lobes(ts_data)
        
        # 2. Compute 12x12 Causal Matrix (Phase 1: Granger causality with cleaned signals)
        # Calculate max_lag in timepoints based on subject-specific TR
        max_lag_timepoints = max(1, int(GRANGER_MAX_LAG_SECONDS / tr))
        causal_matrix = compute_causality_matrix(ts_lobes, max_lag=max_lag_timepoints)
        causal_matrix.fill_diagonal_(0.0)
        
        #  CRITICAL FIX: VALIDATE BEFORE SPARSIFICATION 
        # Check if matrix is all zeros (this would cause issues)
        if (causal_matrix == 0).all():
            logger.warning(f"{subject_id}: Causal matrix is all zeros - skipping")
            return False, False
        
        # Log pre-sparsification statistics
        pre_sparse_stats = {
            'max': float(causal_matrix.abs().max()),
            'mean': float(causal_matrix.abs().mean()),
            'non_zero': int((causal_matrix != 0).sum())
        }
        
        # 3. Adaptive Sparsification (Phase 1: subject-specific thresholding)
        adj_matrix, sparsification_fallback = adaptive_sparsification(causal_matrix)
        
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
            return False, False
        
        # Log success statistics
        post_sparse_stats = {
            'edges': num_edges,
            'density': num_edges / (NUM_LOBES * (NUM_LOBES - 1)),
            'max_weight': float(adj_matrix.abs().max()),
            'mean_weight': float(adj_matrix[adj_matrix != 0].abs().mean())
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
            'subject_id': subject_id,
            'lobe_order': [LOBE_NAMES[i] for i in range(NUM_LOBES)],
            'stats': post_sparse_stats  # Useful for debugging
        }
        
        torch.save(graph_package, output_path)
        return True, sparsification_fallback
        
    except Exception as e:
        logger.error(f"Causal error for {subject_id}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False, False


def main():
    logger.info("="*60)
    logger.info(f"CONSTRUCTING 12×12 CAUSAL GRAPHS (Method={CAUSALITY_METHOD}, MaxLag={GRANGER_MAX_LAG_SECONDS}s)")
    logger.info(f"Sparsity: Keep top {(1-SPARSITY_QUANTILE)*100:.0f}% of edges")
    logger.info("="*60)
    
    CAUSAL_GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    
    manifest = pd.read_csv(MASTER_MANIFEST)
    
    # Statistics tracking
    stats = {
        'total': len(manifest),
        'success': 0,
        'failed': 0,
        'zero_edges': 0,
        'missing_ts': 0
    }
    # Track MIN_EDGES fallback by DX_GROUP to detect class-imbalanced graph quality
    fallback_by_group: Dict[int, int] = {}
    
    for _, row in tqdm(
        manifest.iterrows(), total=len(manifest), desc="Building Graphs",
        miniters=max(1, len(manifest) // 20), mininterval=10.0
    ):
        tr = row.get('TR', 2.0)  # Get per-subject TR, default to 2.0 if missing
        result, fallback = construct_graph(row['subject_id'], row['split'], tr=tr)
        dx_group = int(row.get('DX_GROUP', -1))

        if result:
            stats['success'] += 1
            if fallback:
                fallback_by_group[dx_group] = fallback_by_group.get(dx_group, 0) + 1
        else:
            stats['failed'] += 1
            # Check reason for failure
            ts_path = DATA_FINAL / row['split'] / "time_series" / f"{row['subject_id']}_ts.npy"
            if not ts_path.exists():
                stats['missing_ts'] += 1
    
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
    
    # Report MIN_EDGES fallback by diagnostic group
    if fallback_by_group:
        logger.warning(
            "MIN_EDGES fallback triggered (graph connectivity too sparse for primary threshold):\n"
            + "\n".join(
                f"  DX_GROUP={gid}: {cnt} subjects"
                for gid, cnt in sorted(fallback_by_group.items())
            )
        )

        # Flag potential class-imbalanced sparsification artifacts.
        # Supports both encodings: ASD/Control as (1/0) or (1/2).
        asd_count = fallback_by_group.get(1, 0)
        ctrl_count = fallback_by_group.get(0, fallback_by_group.get(2, 0))
        if asd_count > 2 * max(ctrl_count, 1):
            logger.warning(
                "ASD subjects trigger MIN_EDGES fallback >2x more than Controls "
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
    from src.features.causal_inference import compute_granger_causality

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

    # Helper: aggregate ROI signals to lobe-level
    def _aggregate_lobes(ts_full: np.ndarray) -> np.ndarray:
        """Average ROI time series within each lobe → (T, NUM_LOBES)."""
        lobe_ts = np.zeros((ts_full.shape[0], NUM_LOBES), dtype=np.float32)
        for lobe_idx, roi_list in lobe_to_roi.items():
            if roi_list:
                lobe_ts[:, lobe_idx] = ts_full[:, roi_list].mean(axis=1)
        return lobe_ts

    lobe_ts = _aggregate_lobes(ts_np)

    # 1. Extended-lag view
    base_lag = max(1, round(GRANGER_MAX_LAG_SECONDS / max(tr, 0.1)))
    ext_lag = round(base_lag * 1.5)
    try:
        adj_ext_np = compute_granger_causality(lobe_ts, max_lag=ext_lag)
        adj_extended = torch.tensor(adj_ext_np, dtype=torch.float32)
    except Exception as e:
        logger.warning("Extended-lag Granger failed for %s: %s", subject_id, e)
        adj_extended = adj_base.clone()

    # 2-4. Bootstrap views (80% timepoint subsample, 3 seeds)
    adj_bootstraps = []
    for seed in range(3):
        rng_seed = np.random.default_rng(seed=seed)
        n_keep = max(int(T * 0.80), base_lag + 10)
        idx = rng_seed.choice(T, size=n_keep, replace=False)
        idx = np.sort(idx)
        lobe_ts_sub = lobe_ts[idx]
        try:
            adj_np = compute_granger_causality(lobe_ts_sub, max_lag=base_lag)
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
    }
    torch.save(package, out_path)
    return True


def main_multiview():
    """
    Task 2 entry point: generate multi-view causal graphs for all subjects.

    Reads subjects from MASTER_MANIFEST.  For each subject that already has
    a base graph in CAUSAL_GRAPHS_DIR, constructs 5 additional views and
    saves to CAUSAL_GRAPHS_MULTIVIEW_DIR.

    Usage (via pipeline registry with --multiview flag, or directly):
        python -m src.features.construct_causal --multiview
    """
    from src.core.config import CAUSAL_GRAPHS_MULTIVIEW_DIR, MASTER_MANIFEST, GRANGER_MAX_LAG_SECONDS

    logger.info("=" * 70)
    logger.info("MULTI-VIEW CAUSAL GRAPH CONSTRUCTION (Task 2 — DD-010)")
    logger.info("=" * 70)

    manifest = pd.read_csv(MASTER_MANIFEST)
    all_subjects = manifest['subject_id'].astype(str).tolist()

    # Build lobe_to_roi from config mapping (lobe index -> list[0-based ROI indices]).
    lobe_to_roi: Dict[int, list] = {
        int(lobe_idx): [int(roi_idx) for roi_idx in roi_indices]
        for lobe_idx, roi_indices in LOBE_MAPPING.items()
    }

    CAUSAL_GRAPHS_MULTIVIEW_DIR.mkdir(parents=True, exist_ok=True)

    success, skipped, failed = 0, 0, 0
    for sub_id in tqdm(all_subjects, desc="Multi-view graphs"):
        out_file = CAUSAL_GRAPHS_MULTIVIEW_DIR / sub_id / "multiview_graphs.pt"
        if out_file.exists():
            skipped += 1
            continue

        base_path = CAUSAL_GRAPHS_DIR / f"{sub_id}_graph.pt"
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
        "Multi-view construction complete: %d success | %d skipped | %d failed",
        success, skipped, failed,
    )
    logger.info("Output directory: %s", CAUSAL_GRAPHS_MULTIVIEW_DIR)


if __name__ == "__main__":
    import argparse
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--multiview", action="store_true", help="Run multi-view graph construction (Task 2)")
    _args = _parser.parse_args()
    if _args.multiview:
        main_multiview()
    else:
        main()
