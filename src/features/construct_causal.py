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
    CAUSALITY_METHOD, GRANGER_MAX_LAG, GRANGER_MAX_LAG_SECONDS, SPARSITY_METHOD, MIN_EDGES_PER_GRAPH
)

# Legacy lag value for compute_lagged_causality() — kept local to prevent re-introduction in config.
_LEGACY_CAUSAL_LAG = 1
from src.features.causal_inference import (
    compute_granger_causality,
    compute_granger_causality_gpu,
    compute_transfer_entropy,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
    
    for lobe_id in range(NUM_LOBES):
        # Get ROIs belonging to this lobe (already 0-based indices)
        indices = [i for i in LOBE_MAPPING[lobe_id] if i < num_rois]
        
        if not indices:
            logger.warning(f"Lobe {lobe_id} ({LOBE_NAMES[lobe_id]}): No matching ROIs in atlas. Using zero-signal.")
            lobe_signals.append(torch.zeros(ts_raw.shape[0], device=ts_raw.device))
            lobe_internal_features.append(torch.tensor([0.0, 0.0], device=ts_raw.device))
            continue
        
        # Extract raw ROIs for this lobe: Shape (Timepoints, Num_ROIs_in_Lobe)
        roi_data = ts_raw[:, indices]
        
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
            # Sign stabilisation: orient the eigenvariate so it is positively
            # correlated with the simple ROI mean.  This is more robust than
            # using the sign of the largest loading (vh[0]), which can flip
            # when two ROIs have equal magnitude but opposite directions.
            raw_mean = roi_data.mean(dim=1)  # (T,)
            correlation = torch.dot(
                dominant_signal / (dominant_signal.norm() + 1e-8),
                raw_mean / (raw_mean.norm() + 1e-8),
            )
            if correlation < 0:
                dominant_signal = -dominant_signal
        except Exception as e:
            logger.debug(f"Lobe {lobe_id}: SVD failed ({str(e)}), falling back to mean")
            dominant_signal = roi_data.mean(dim=1)
        
        lobe_signals.append(dominant_signal)
        
        # --- 2. INTRA-LOBE SYNCHRONY (Regional Homogeneity - ReHo) ---
        # Measure how synchronized ROIs within this lobe are
        # ASD hypothesis: Local over-connectivity means HIGH coherence within lobes
        
        # Filter out completely NaN ROIs
        valid_roi_mask = ~torch.isnan(roi_data).all(dim=0)
        valid_rois = roi_data[:, valid_roi_mask]
        
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
    
    # Stack results
    ts_lobes = torch.stack(lobe_signals, dim=1)           # (Timepoints, 12)
    features_internal = torch.stack(lobe_internal_features, dim=0)  # (12, 2)
    
    return ts_lobes, features_internal


def compute_lagged_causality(ts_lobe: torch.Tensor) -> torch.Tensor:
    """
    Legacy function: Compute lagged Pearson correlation (kept for backward compatibility).
    
    For new graphs, use compute_causality_matrix() which supports Granger causality.
    """
    # Validate input
    if ts_lobe.shape[0] <= _LEGACY_CAUSAL_LAG:
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
    ts_prev = ts_std[:-_LEGACY_CAUSAL_LAG]
    ts_curr = ts_std[_LEGACY_CAUSAL_LAG:]
    
    # 3. Compute Adjacency Matrix (12x12 for 12 regions)
    directed_adj = (ts_prev.T @ ts_curr) / (ts_std.shape[0] - _LEGACY_CAUSAL_LAG)
    
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
            # Check for GPU
            use_gpu = torch.cuda.is_available() and DEVICE.type == 'cuda'
            if use_gpu:
                logger.debug(f"Computing Granger causality on GPU (max_lag={max_lag} timepoints)")
                try:
                    causal_matrix_np = compute_granger_causality_gpu(
                        ts_numpy,
                        max_lag=max_lag,
                        device=DEVICE
                    )
                except Exception as e:
                    logger.warning(f"GPU Granger failed: {e}. Falling back to CPU.")
                    causal_matrix_np = compute_granger_causality(
                        ts_numpy,
                        max_lag=max_lag
                    )
            else:
                logger.debug(f"Computing Granger causality on CPU (max_lag={max_lag} timepoints)")
                causal_matrix_np = compute_granger_causality(
                    ts_numpy,
                    max_lag=max_lag
                )
        
        elif method == 'transfer_entropy':
            logger.debug("Computing transfer entropy")
            causal_matrix_np = compute_transfer_entropy(
                ts_numpy,
                k=1,
                bins=10
            )
        
        elif method == 'lagged_pearson':
            logger.debug(f"Computing lagged Pearson correlation (lag={_LEGACY_CAUSAL_LAG})")
            # Use legacy function
            return compute_lagged_causality(ts_lobe)
        
        else:
            logger.warning(f"Unknown causality method '{method}', falling back to lagged_pearson")
            return compute_lagged_causality(ts_lobe)
        
        # Convert back to torch tensor
        causal_matrix = torch.from_numpy(causal_matrix_np).float().to(DEVICE)
        
        return causal_matrix
    
    except Exception as e:
        logger.warning(f"Causality computation failed ({method}): {e}, falling back to lagged_pearson")
        return compute_lagged_causality(ts_lobe)


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
            torch.tensor(0.0, device=DEVICE)
        )
    
    elif method == 'adaptive_statistical':
        # Keep edges above statistical significance threshold
        # For Granger causality: -log10(p) > -log10(0.05) ≈ 1.3
        # For other methods: use median + 1 std as threshold
        if CAUSALITY_METHOD == 'granger':
            threshold_value = 1.3  # p < 0.05
        else:
            non_zero = abs_matrix[abs_matrix > 0]
            if len(non_zero) > 0:
                threshold_value = non_zero.median() + non_zero.std()
            else:
                threshold_value = 0.0
        
        adj_matrix = torch.where(
            (abs_matrix >= threshold_value) & offdiag_mask,
            causal_matrix,
            torch.tensor(0.0, device=DEVICE)
        )
        
        # Ensure minimum edges
        num_edges = (adj_matrix != 0).sum().item()
        if num_edges < min_edges:
            fallback_triggered = True
            # Fall back to keeping top min_edges
            flat_values = offdiag_values
            threshold_value = torch.topk(flat_values, min_edges).values[-1]
            adj_matrix = torch.where(
                (abs_matrix >= threshold_value) & offdiag_mask,
                causal_matrix,
                torch.tensor(0.0, device=DEVICE)
            )
    
    elif method == 'fixed':
        # Use fixed quantile threshold (original method)
        thresh = torch.quantile(offdiag_values, SPARSITY_QUANTILE)
        adj_matrix = torch.where(
            (abs_matrix >= thresh) & offdiag_mask,
            causal_matrix,
            torch.tensor(0.0, device=DEVICE)
        )
        
        # Ensure minimum edges
        num_edges = (adj_matrix != 0).sum().item()
        if num_edges < min_edges:
            fallback_triggered = True
            flat_values = offdiag_values
            threshold_value = torch.topk(flat_values, min_edges).values[-1]
            adj_matrix = torch.where(
                (abs_matrix >= threshold_value) & offdiag_mask,
                causal_matrix,
                torch.tensor(0.0, device=DEVICE)
            )
    
    else:
        logger.warning(f"Unknown sparsity method '{method}', using fixed")
        thresh = torch.quantile(offdiag_values, SPARSITY_QUANTILE)
        adj_matrix = torch.where(
            (abs_matrix >= thresh) & offdiag_mask,
            causal_matrix,
            torch.tensor(0.0, device=DEVICE)
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
        ts_mean = ts_data.mean(dim=0, keepdim=True)
        ts_std  = ts_data.std(dim=0, keepdim=True).clamp(min=1e-8)
        ts_data = (ts_data - ts_mean) / ts_std

        # Validate input data
        if ts_data.shape[0] < 10:
            logger.warning(f"{subject_id}: Insufficient timepoints ({ts_data.shape[0]})")
            return False, False
        
        # 1. Smart Aggregation (PCA + Regional Homogeneity)
        ts_lobes, internal_features = aggregate_to_lobes(ts_data)
        
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
    
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Building Graphs"):
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


if __name__ == "__main__":
    main()