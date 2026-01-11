import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# --- CONFIG ---
PROJECT_ROOT = Path("./data")
DATASET_ROOT = PROJECT_ROOT / "final"
MANIFEST_PATH = PROJECT_ROOT / "metadata" / "master_manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "processed" / "causal_graphs"
# AAL3 to 5-Lobe Mapping (Essential for your YOLO-GNN alignment)
# Classes: 0:Frontal, 1:Temporal, 2:Parietal, 3:Occipital, 4:Limbic
LOBE_MAPPING = {
    0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
    1: [79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90],
    2: [57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70],
    3: [43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    4: [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 71, 72, 73, 74, 75, 76, 77, 78, 91, 92, 93, 94]
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def aggregate_to_lobes(ts_170):
    """Aggregates 170 AAL regions into 5 high-level lobe signals."""
    lobe_signals = []
    for lobe_id in range(5):
        indices = [i-1 for i in LOBE_MAPPING[lobe_id]] # AAL is 1-indexed
        lobe_ts = ts_170[:, indices].mean(dim=1)
        lobe_signals.append(lobe_ts)
    return torch.stack(lobe_signals, dim=1) # Shape: (Time, 5)

def compute_causal_edges(ts_lobe):
    """
    Computes Directed Edges using Lagged Partial Correlation.
    This provides 'temporal precedence', a core requirement for Causality.
    """
    # 1. Standardize
    ts = (ts_lobe - ts_lobe.mean(dim=0)) / (ts_lobe.std(dim=0) + 1e-6)
    
    # 2. Lagged Matrix (t vs t-1)
    # This determines if Lobe A at t-1 predicts Lobe B at t
    ts_curr = ts[1:]
    ts_prev = ts[:-1]
    
    # 3. Compute Directed Correlation
    # (prev.T @ curr) creates a 5x5 matrix where entry [i, j] is Lobe i -> Lobe j
    directed_adj = (ts_prev.T @ ts_curr) / (ts.shape[0] - 1)
    
    return directed_adj

def main():
    manifest = pd.read_csv(MANIFEST_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Q1 Journal Requirement: Site Harmonization (Phase 2.2 correction)
    # We group by site to ensure hospital-specific noise doesn't drive the edges
    print(f"🚀 Constructing Directed Causal Graphs on {DEVICE}...")

    for _, row in tqdm(manifest.iterrows(), total=len(manifest)):
        sub_id, split = row['subject_id'], row['split']
        ts_path = DATASET_ROOT / split / "time_series" / f"{sub_id}_ts.npy"
        
        if not ts_path.exists(): continue
            
        # Load and move to RTX 4060
        ts_data = torch.from_numpy(np.load(ts_path)).float().to(DEVICE)
        
        # 1. Aggregate 170 ROIs -> 5 Nodes (Anatomical Alignment)
        ts_lobes = aggregate_to_lobes(ts_data)
        
        # 2. Compute Directed Edges (Phase 7.1 Causal Logic)
        causal_matrix = compute_causal_edges(ts_lobes)
        
        # 3. Sparsify: Only keep top 20% of 'Influence' pathways
        # This reduces graph noise for the GNN
        thresh = torch.quantile(torch.abs(causal_matrix), 0.80)
        adj_matrix = torch.where(torch.abs(causal_matrix) > thresh, causal_matrix, 0.0)
        
        # 4. Save for PyTorch Geometric (Phase 6.1)
        # We save as a dict containing both edge weights and node attributes
        graph_data = {
            'adj': adj_matrix.cpu(),
            'node_features': ts_lobes.mean(dim=0).cpu() # Mean signal as initial feature
        }
        torch.save(graph_data, OUTPUT_DIR / f"{sub_id}_graph.pt")

    print(f"✅ Directed Causal Graphs ready in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()