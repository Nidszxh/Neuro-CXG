import logging
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

# Setup paths
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import (
    LOBE_MAPPING, NUM_LOBES, LOBE_NAMES, CAUSAL_LAG, SPARSITY_QUANTILE,
    DATA_FINAL, MASTER_MANIFEST, CAUSAL_GRAPHS_DIR, DEVICE,
    CAUSALITY_METHOD, GRANGER_MAX_LAG, SPARSITY_METHOD, MIN_EDGES_PER_GRAPH
)
from src.features.causal_inference import (
    compute_granger_causality,
    compute_granger_causality_gpu,
    compute_transfer_entropy,
    compute_multilag_causality
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def aggregate_to_lobes(ts_raw: torch.Tensor) -> tuple:
    """
    Smart Aggregation with Regional Homogeneity (ReHo) and Eigenvariate Extraction.
    
    Instead of simple averaging (which cancels anti-correlated signals), we extract:
    1. Dominant Signal (PCA/Eigenvariate): Captures main driver without cancellation
    2. Intra-Lobe Coherence: Local connectivity within lobe (ASD biomarker)
    
    Returns:
        lobe_signals: (Timepoints, 12) - The cleaned time series for graph edges
        lobe_features: (12, 2) - Internal features (Coherence, Variance)
    """
    num_rois = ts_raw.shape[1]
    lobe_signals = []
    lobe_internal_features = []
    
    for lobe_id in range(NUM_LOBES):
        # Get ROIs belonging to this lobe (convert 1-based AAL indices to 0-based)
        indices = [i-1 for i in LOBE_MAPPING[lobe_id] if i <= num_rois]
        
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
        except Exception as e:
            logger.debug(f"Lobe {lobe_id}: SVD failed ({str(e)}), falling back to mean")
            dominant_signal = roi_data.mean(dim=1)
        
        lobe_signals.append(dominant_signal)
        
        # --- 2. INTRA-LOBE SYNCHRONY (Regional Homogeneity - ReHo) ---
        # Measure how synchronized ROIs within this lobe are
        # ASD hypothesis: Local over-connectivity means HIGH coherence within lobes
        if len(indices) > 1:
            try:
                # Compute correlation matrix of ROIs within this lobe
                intra_corr = torch.corrcoef(roi_data.T)
                # Average off-diagonal correlation (all pairs)
                mask = ~torch.eye(intra_corr.shape[0], dtype=torch.bool, device=ts_raw.device)
                coherence = intra_corr[mask].mean()
                coherence = torch.clamp(coherence, -1.0, 1.0)  # Ensure valid range
                
                # Spatial heterogeneity (variance across ROIs over time)
                # Measures how much activation spreads within the lobe
                spatial_variance = roi_data.std(dim=1).mean()
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
    if ts_lobe.shape[0] <= CAUSAL_LAG:
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
    ts_prev = ts_std[:-CAUSAL_LAG]
    ts_curr = ts_std[CAUSAL_LAG:]
    
    # 3. Compute Adjacency Matrix (12x12 for 12 regions)
    directed_adj = (ts_prev.T @ ts_curr) / (ts_std.shape[0] - CAUSAL_LAG)
    
    # 4. Validate output
    if torch.isnan(directed_adj).any() or torch.isinf(directed_adj).any():
        logger.warning("Lagged correlation produced NaN/Inf - returning zero matrix")
        return torch.zeros(NUM_LOBES, NUM_LOBES, device=ts_lobe.device)
    
    return directed_adj


def compute_causality_matrix(ts_lobe: torch.Tensor, method: str = None) -> torch.Tensor:
    """
    Compute causal adjacency matrix using configured method.
    
    Args:
        ts_lobe: Time series for lobes (shape: [timepoints, n_lobes])
        method: Causality method ('granger', 'transfer_entropy', 'lagged_pearson')
                If None, uses CAUSALITY_METHOD from config
    
    Returns:
        Causal adjacency matrix (shape: [n_lobes, n_lobes])
    """
    if method is None:
        method = CAUSALITY_METHOD
    
    # Convert to numpy for causal inference methods
    ts_numpy = ts_lobe.cpu().numpy()
    
    try:
        if method == 'granger':
            # Check for GPU
            use_gpu = torch.cuda.is_available() and DEVICE.type == 'cuda'
            if use_gpu:
                logger.debug(f"Computing Granger causality on GPU (max_lag={GRANGER_MAX_LAG})")
                try:
                    causal_matrix_np = compute_granger_causality_gpu(
                        ts_numpy,
                        max_lag=GRANGER_MAX_LAG,
                        device=DEVICE
                    )
                except Exception as e:
                    logger.warning(f"GPU Granger failed: {e}. Falling back to CPU.")
                    causal_matrix_np = compute_granger_causality(
                        ts_numpy,
                        max_lag=GRANGER_MAX_LAG
                    )
            else:
                logger.debug(f"Computing Granger causality on CPU (max_lag={GRANGER_MAX_LAG})")
                causal_matrix_np = compute_granger_causality(
                    ts_numpy,
                    max_lag=GRANGER_MAX_LAG
                )
        
        elif method == 'transfer_entropy':
            logger.debug("Computing transfer entropy")
            causal_matrix_np = compute_transfer_entropy(
                ts_numpy,
                k=1,
                bins=10
            )
        
        elif method == 'lagged_pearson':
            logger.debug(f"Computing lagged Pearson correlation (lag={CAUSAL_LAG})")
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
    Apply adaptive sparsification to causality matrix.
    
    Args:
        causal_matrix: Causal adjacency matrix
        method: Sparsification method ('adaptive_proportional', 'adaptive_statistical', 'fixed')
        min_edges: Minimum number of edges to keep
    
    Returns:
        Sparsified adjacency matrix
    """
    if method is None:
        method = SPARSITY_METHOD
    
    if min_edges is None:
        min_edges = MIN_EDGES_PER_GRAPH
    
    abs_matrix = torch.abs(causal_matrix)
    
    if method == 'adaptive_proportional':
        # Keep edges proportional to network strength
        total_strength = abs_matrix.sum().item()
        target_edges = max(min_edges, int(np.sqrt(total_strength) * 10))
        target_edges = min(target_edges, NUM_LOBES * NUM_LOBES)  # Cap at max possible
        
        # Keep top target_edges by absolute weight
        flat_values = abs_matrix.flatten()
        if target_edges >= len(flat_values):
            # Keep all edges
            return causal_matrix
        
        threshold_value = torch.topk(flat_values, target_edges).values[-1]
        adj_matrix = torch.where(
            abs_matrix >= threshold_value,
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
            abs_matrix >= threshold_value,
            causal_matrix,
            torch.tensor(0.0, device=DEVICE)
        )
        
        # Ensure minimum edges
        num_edges = (adj_matrix != 0).sum().item()
        if num_edges < min_edges:
            # Fall back to keeping top min_edges
            flat_values = abs_matrix.flatten()
            threshold_value = torch.topk(flat_values, min_edges).values[-1]
            adj_matrix = torch.where(
                abs_matrix >= threshold_value,
                causal_matrix,
                torch.tensor(0.0, device=DEVICE)
            )
    
    elif method == 'fixed':
        # Use fixed quantile threshold (original method)
        thresh = torch.quantile(abs_matrix, SPARSITY_QUANTILE)
        adj_matrix = torch.where(
            abs_matrix >= thresh,
            causal_matrix,
            torch.tensor(0.0, device=DEVICE)
        )
        
        # Ensure minimum edges
        num_edges = (adj_matrix != 0).sum().item()
        if num_edges < min_edges:
            flat_values = abs_matrix.flatten()
            threshold_value = torch.topk(flat_values, min_edges).values[-1]
            adj_matrix = torch.where(
                abs_matrix >= threshold_value,
                causal_matrix,
                torch.tensor(0.0, device=DEVICE)
            )
    
    else:
        logger.warning(f"Unknown sparsity method '{method}', using fixed")
        thresh = torch.quantile(abs_matrix, SPARSITY_QUANTILE)
        adj_matrix = torch.where(
            abs_matrix >= thresh,
            causal_matrix,
            torch.tensor(0.0, device=DEVICE)
        )
    
    return adj_matrix


def construct_graph(subject_id: str, split: str) -> bool:

    ts_path = DATA_FINAL / split / "time_series" / f"{subject_id}_ts.npy"
    output_path = CAUSAL_GRAPHS_DIR / f"{subject_id}_graph.pt"
    
    if not ts_path.exists():
        logger.debug(f"Time series not found for {subject_id}")
        return False
    
    try:
        # Load and move to GPU for fast matrix math
        ts_data = torch.from_numpy(np.load(ts_path)).float().to(DEVICE)
        
        # Validate input data
        if ts_data.shape[0] < 10:
            logger.warning(f"{subject_id}: Insufficient timepoints ({ts_data.shape[0]})")
            return False
        
        # 1. Smart Aggregation (PCA + Regional Homogeneity)
        ts_lobes, internal_features = aggregate_to_lobes(ts_data)
        
        # 2. Compute 12x12 Causal Matrix (Phase 1: Granger causality with cleaned signals)
        causal_matrix = compute_causality_matrix(ts_lobes)
        
        #  CRITICAL FIX: VALIDATE BEFORE SPARSIFICATION 
        # Check if matrix is all zeros (this would cause issues)
        if (causal_matrix == 0).all():
            logger.warning(f"{subject_id}: Causal matrix is all zeros - skipping")
            return False
        
        # Log pre-sparsification statistics
        pre_sparse_stats = {
            'max': float(causal_matrix.abs().max()),
            'mean': float(causal_matrix.abs().mean()),
            'non_zero': int((causal_matrix != 0).sum())
        }
        
        # 3. Adaptive Sparsification (Phase 1: subject-specific thresholding)
        adj_matrix = adaptive_sparsification(causal_matrix)
        
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
            return False
        
        # Log success statistics
        post_sparse_stats = {
            'edges': num_edges,
            'density': num_edges / (NUM_LOBES * NUM_LOBES),
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
        return True
        
    except Exception as e:
        logger.error(f"Causal error for {subject_id}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


def main():
    logger.info("="*60)
    logger.info(f"CONSTRUCTING 12×12 CAUSAL GRAPHS (Lag={CAUSAL_LAG})")
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
    
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Building Graphs"):
        result = construct_graph(row['subject_id'], row['split'])
        
        if result:
            stats['success'] += 1
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