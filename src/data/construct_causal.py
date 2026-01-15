"""
Causal Graph Construction Module

Constructs directed causal graphs from fMRI time series using lagged partial 
correlation to enforce temporal precedence.

Pipeline:
1. Aggregate 170 AAL ROIs → 5 anatomical lobes
2. Compute directed edges using lagged correlation (t-1 → t)
3. Sparsify to top 20% of connections
4. Save as PyTorch graph objects
"""

import logging
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

# Setup paths
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    LOBE_MAPPING, NUM_LOBES, CAUSAL_LAG, SPARSITY_QUANTILE,
    DATA_FINAL, MASTER_MANIFEST, CAUSAL_GRAPHS_DIR, DEVICE
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def aggregate_to_lobes(ts_170: torch.Tensor) -> torch.Tensor:
    """
    Aggregate 170 AAL ROI signals into 5 high-level lobe signals.
    
    Uses anatomical mapping defined in config.LOBE_MAPPING to average
    time series across ROIs within each lobe.
    
    Args:
        ts_170: Time series tensor of shape (timepoints, n_rois)
                where n_rois should be 170 (or compatible AAL version)
    
    Returns:
        Tensor of shape (timepoints, 5) with aggregated lobe signals
        
    Raises:
        ValueError: If input shape is invalid
    """
    if ts_170.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got shape {ts_170.shape}")
    
    if ts_170.shape[1] not in [116, 117, 170]:
        logger.warning(f"Unusual ROI count: {ts_170.shape[1]} (expected 116/117/170)")
    
    lobe_signals = []
    num_rois = ts_170.shape[1]
    
    for lobe_id in range(NUM_LOBES):
        # AAL indices are 1-based, adjust for 0-based numpy indexing
        indices = [i-1 for i in LOBE_MAPPING[lobe_id] if i <= num_rois]
        
        if not indices:
            logger.warning(f"No valid ROIs for lobe {lobe_id}")
            # Use zeros as fallback
            lobe_ts = torch.zeros(ts_170.shape[0], device=ts_170.device)
        else:
            lobe_ts = ts_170[:, indices].mean(dim=1)
        
        lobe_signals.append(lobe_ts)
    
    return torch.stack(lobe_signals, dim=1)  # Shape: (Time, 5)


def compute_causal_edges(ts_lobe: torch.Tensor) -> torch.Tensor:
    """
    Compute directed edges using lagged partial correlation.
    
    Implements temporal precedence by computing correlation between
    signals at time t-1 and t. Entry [i, j] in output represents
    influence from lobe i at t-1 to lobe j at t.
    
    Args:
        ts_lobe: Lobe time series of shape (timepoints, 5)
    
    Returns:
        Directed adjacency matrix of shape (5, 5)
        
    Raises:
        ValueError: If time series is too short for lagged analysis
    """
    if ts_lobe.shape[0] < 10:
        raise ValueError(f"Time series too short: {ts_lobe.shape[0]} timepoints")
    
    # 1. Standardize signals
    ts_standardized = (ts_lobe - ts_lobe.mean(dim=0)) / (ts_lobe.std(dim=0) + 1e-6)
    
    # 2. Create lagged matrices
    ts_curr = ts_standardized[CAUSAL_LAG:]
    ts_prev = ts_standardized[:-CAUSAL_LAG]
    
    # 3. Compute directed correlation
    # (prev.T @ curr) creates 5x5 matrix where entry [i, j] is 
    # correlation of lobe i at t-1 with lobe j at t
    directed_adj = (ts_prev.T @ ts_curr) / (ts_standardized.shape[0] - CAUSAL_LAG)
    
    # 4. Validate output
    if torch.isnan(directed_adj).any() or torch.isinf(directed_adj).any():
        raise ValueError("Invalid values in causal adjacency matrix")
    
    return directed_adj


def construct_graph(subject_id: str, split: str) -> bool:
    """
    Construct and save causal graph for a single subject.
    
    Args:
        subject_id: Subject identifier
        split: Data split ('train', 'val', or 'test')
        
    Returns:
        True if successful, False otherwise
    """
    ts_path = DATA_FINAL / split / "time_series" / f"{subject_id}_ts.npy"
    output_path = CAUSAL_GRAPHS_DIR / f"{subject_id}_graph.pt"
    
    # Check if already processed
    if output_path.exists():
        logger.debug(f"Graph already exists for {subject_id}")
        return True
    
    # Load time series
    if not ts_path.exists():
        logger.warning(f"Missing time series for {subject_id}")
        return False
    
    try:
        ts_data = torch.from_numpy(np.load(ts_path)).float().to(DEVICE)
    except Exception as e:
        logger.error(f"Error loading time series for {subject_id}: {e}")
        return False
    
    # Validate data quality
    if torch.isnan(ts_data).any() or torch.isinf(ts_data).any():
        logger.warning(f"Invalid values in time series for {subject_id}")
        return False
    
    if ts_data.shape[0] < 50:
        logger.warning(
            f"Insufficient timepoints ({ts_data.shape[0]}) for {subject_id}"
        )
        return False
    
    try:
        # 1. Aggregate to lobes
        ts_lobes = aggregate_to_lobes(ts_data)
        
        # 2. Compute causal edges
        causal_matrix = compute_causal_edges(ts_lobes)
        
        # 3. Sparsify: Keep top 20% of connections
        thresh = torch.quantile(torch.abs(causal_matrix), SPARSITY_QUANTILE)
        adj_matrix = torch.where(
            torch.abs(causal_matrix) > thresh, 
            causal_matrix, 
            torch.tensor(0.0, device=causal_matrix.device)
        )
        
        # 4. Prepare graph data
        graph_data = {
            'adj': adj_matrix.cpu(),
            'node_features': ts_lobes.mean(dim=0).cpu(),  # Mean signal as feature
            'metadata': {
                'subject_id': subject_id,
                'n_timepoints': ts_data.shape[0],
                'n_edges': (adj_matrix != 0).sum().item()
            }
        }
        
        # 5. Save
        torch.save(graph_data, output_path)
        
        logger.debug(
            f"Created graph for {subject_id}: "
            f"{graph_data['metadata']['n_edges']} edges"
        )
        return True
        
    except Exception as e:
        logger.error(f"Failed to process {subject_id}: {e}", exc_info=True)
        return False


def main():
    """Main execution function."""
    # Validate configuration
    logger.info("Starting causal graph construction")
    logger.info(f"Using device: {DEVICE}")
    logger.info(f"Causal lag: {CAUSAL_LAG}")
    logger.info(f"Sparsity quantile: {SPARSITY_QUANTILE}")
    
    # Create output directory
    CAUSAL_GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load manifest
    if not MASTER_MANIFEST.exists():
        logger.error(f"Master manifest not found: {MASTER_MANIFEST}")
        logger.error("Run manifest.py first!")
        return
    
    manifest = pd.read_csv(MASTER_MANIFEST)
    logger.info(f"Loaded manifest with {len(manifest)} subjects")
    
    # Process all subjects
    success_count = 0
    fail_count = 0
    
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Processing"):
        sub_id = row['subject_id']
        split = row['split']
        
        success = construct_graph(sub_id, split)
        
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    # Summary
    logger.info("="*60)
    logger.info("CAUSAL GRAPH CONSTRUCTION COMPLETE")
    logger.info(f"Successfully processed: {success_count}/{len(manifest)}")
    logger.info(f"Failed: {fail_count}/{len(manifest)}")
    logger.info(f"Output directory: {CAUSAL_GRAPHS_DIR}")
    logger.info("="*60)


if __name__ == "__main__":
    main()