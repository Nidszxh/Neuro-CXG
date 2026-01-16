import logging
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

# Setup paths
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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

def aggregate_to_lobes(ts_raw: torch.Tensor) -> torch.Tensor:
    """
    Strictly aggregates AAL time series (116, 164, or 170) into 5 lobe signals.
    Ensures Node 0 = Frontal, Node 1 = Temporal, etc.
    """
    num_rois = ts_raw.shape[1]
    lobe_signals = []
    
    for lobe_id in range(NUM_LOBES):
        # Convert AAL 1-based indices to 0-based Python indices
        # Filter indices to ensure they exist in the current time-series file
        indices = [i-1 for i in LOBE_MAPPING[lobe_id] if i <= num_rois]
        
        if not indices:
            logger.warning(f"Lobe {lobe_id} has no matching ROIs in this atlas. Using zero-signal.")
            lobe_ts = torch.zeros(ts_raw.shape[0], device=ts_raw.device)
        else:
            # Average the BOLD signals of all ROIs in this lobe
            lobe_ts = ts_raw[:, indices].mean(dim=1)
        
        lobe_signals.append(lobe_ts)
    
    return torch.stack(lobe_signals, dim=1)  # Output Shape: (Timepoints, 5)

def compute_lagged_causality(ts_lobe: torch.Tensor) -> torch.Tensor:
    """
    Computes Directed Causal Influence using Lagged Pearson Correlation.
    Matrix[i, j] = Correlation between Lobe i (t-1) and Lobe j (t).
    """
    # 1. Standardize (Z-Score) signals for valid correlation
    ts_std = (ts_lobe - ts_lobe.mean(dim=0)) / (ts_lobe.std(dim=0) + 1e-6)
    
    # 2. Slice for Lag (t-1 -> t)
    # ts_prev: signals from 0 to T-1
    # ts_curr: signals from 1 to T
    ts_prev = ts_std[:-CAUSAL_LAG]
    ts_curr = ts_std[CAUSAL_LAG:]
    
    # 3. Compute Adjacency Matrix (5x5)
    # The dot product of standardized shifted signals is the correlation coefficient
    directed_adj = (ts_prev.T @ ts_curr) / (ts_std.shape[0] - CAUSAL_LAG)
    
    return directed_adj

def construct_graph(subject_id: str, split: str) -> bool:
    """Processes a single subject and saves the 5x5 causal matrix."""
    ts_path = DATA_FINAL / split / "time_series" / f"{subject_id}_ts.npy"
    output_path = CAUSAL_GRAPHS_DIR / f"{subject_id}_graph.pt"
    
    if not ts_path.exists():
        return False
    
    try:
        # Load and move to GPU for fast matrix math
        ts_data = torch.from_numpy(np.load(ts_path)).float().to(DEVICE)
        
        # 1. Aggregate to 5 Lobes
        ts_lobes = aggregate_to_lobes(ts_data)
        
        # 2. Compute 5x5 Causal Matrix
        causal_matrix = compute_lagged_causality(ts_lobes)
        
        # 3. Sparsification (Keep strongest 20% of directed edges)
        # Note: In 5x5 (25 edges), this keeps exactly 5 edges.
        abs_matrix = torch.abs(causal_matrix)
        thresh = torch.quantile(abs_matrix, SPARSITY_QUANTILE)
        
        # Zero out weak connections
        adj_matrix = torch.where(
            abs_matrix >= thresh, 
            causal_matrix, 
            torch.tensor(0.0, device=DEVICE)
        )
        
        # 4. Save structured data for Graph Factory
        # We save as a dict to keep it flexible for the Factory
        graph_package = {
            'adj': adj_matrix.cpu(),
            'subject_id': subject_id,
            'lobe_order': ['Frontal', 'Temporal', 'Parietal', 'Occipital', 'Limbic']
        }
        
        torch.save(graph_package, output_path)
        return True
        
    except Exception as e:
        logger.error(f"Causal error for {subject_id}: {e}")
        return False

def main():
    logger.info(f"🚀 Constructing 5x5 Causal Graphs (Lag={CAUSAL_LAG})")
    CAUSAL_GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    
    manifest = pd.read_csv(MASTER_MANIFEST)
    success = 0
    
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Building Graphs"):
        if construct_graph(row['subject_id'], row['split']):
            success += 1
            
    logger.info(f"✓ Successfully generated {success} causal graphs in {CAUSAL_GRAPHS_DIR}")

if __name__ == "__main__":
    main()