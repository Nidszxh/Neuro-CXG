"""
Circular connectome visualization for Neuro-CXG publication.

Generates circular brain network plots with nodes colored by network membership
(DMN, Salience, Visual/Cerebellar, Limbic) and edges colored by mean causal weight.
ASD and Control groups shown side-by-side.

Usage:
    python src/analysis/circular_connectome.py --output results/paper_figures/
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

# Suppress matplotlib warnings
warnings.filterwarnings("ignore", ".*Colorbar layout.*")

# Add project to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.config import (
    PROJECT_ROOT, NUM_LOBES, LOBE_NAMES,
    CAUSAL_GRAPHS_DIR, MASTER_MANIFEST
)
import pandas as pd
import torch

# Network membership for each lobe (index-based)
# DMN: Default Mode Network regions
# Salience: Salience Network regions  
# Visual/Cerebellar: Visual and Cerebellar regions
# Limbic: Limbic system regions
NETWORK_MEMBERSHIP = {
    0: "DMN",           # Frontal_Sup_L (example - adjust based on your atlas)
    1: "DMN",           # Frontal_Sup_R
    2: "Salience",      # Frontal_Inf_Oper_L
    3: "Salience",      # Frontal_Inf_Oper_R
    4: "Visual",        # Temporal_Sup_L
    5: "Visual",        # Temporal_Sup_R
    6: "DMN",           # Parietal_Sup_L
    7: "DMN",           # Parietal_Sup_R
    8: "Limbic",        # Occipital_Mid_L
    9: "Limbic",        # Occipital_Mid_R
    10: "Visual",       # Cerebellum_Crus1_L
    11: "Visual",       # Cerebellum_Crus1_R
}

NETWORK_COLORS = {
    "DMN": "#1f77b4",          # Blue
    "Salience": "#ff7f0e",     # Orange
    "Visual": "#2ca02c",       # Green
    "Limbic": "#d62728",       # Red
    "Cerebellar": "#9467bd",   # Purple
}


def get_network_color(lobe_idx):
    """Get network color for a lobe index."""
    network = NETWORK_MEMBERSHIP.get(lobe_idx, "Other")
    return NETWORK_COLORS.get(network, "#7f7f7f")


def compute_group_average_causal(subject_ids, graphs_dir):
    """Compute average causal adjacency matrix for a group."""
    matrices = []
    for sid in subject_ids:
        graph_path = graphs_dir / f"{sid}_graph.pt"
        if not graph_path.exists():
            continue
        try:
            data = torch.load(graph_path, weights_only=False)
            if "adj" in data:
                matrices.append(data["adj"].detach().cpu().numpy())
        except Exception:
            continue
    if not matrices:
        return None
    return np.mean(np.stack(matrices), axis=0)


def create_circular_connectome(adj_matrix, output_path, title="Causal Connectivity"):
    """Create circular connectome plot."""
    import networkx as nx
    
    # Create graph
    G = nx.DiGraph()
    
    # Add nodes
    for i in range(NUM_LOBES):
        G.add_node(i, label=LOBE_NAMES[i], color=get_network_color(i))
    
    # Add edges with weights
    for i in range(NUM_LOBES):
        for j in range(NUM_LOBES):
            weight = adj_matrix[i, j]
            if abs(weight) > 0.1:  # Threshold for visibility
                G.add_edge(i, j, weight=weight)
    
    # Create circular layout
    pos = nx.circular_layout(G)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw edges with colormap
    edges = G.edges()
    if edges:
        edge_weights = [G[u][v]['weight'] for u, v in edges]
        norm = Normalize(vmin=min(edge_weights), vmax=max(edge_weights))
        edge_colors = [cm.RdBu_r(norm(w)) for w in edge_weights]
        
        nx.draw_networkx_edges(
            G, pos,
            edgelist=edges,
            edge_color=edge_colors,
            width=2,
            alpha=0.6,
            arrows=True,
            arrowsize=10,
            connectionstyle="arc3,rad=0.1",
            ax=ax
        )
    
    # Draw nodes colored by network
    for network in set(NETWORK_MEMBERSHIP.values()):
        nodes = [i for i in range(NUM_LOBES) if NETWORK_MEMBERSHIP.get(i) == network]
        if nodes:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=nodes,
                node_color=NETWORK_COLORS[network],
                node_size=800,
                alpha=0.9,
                ax=ax
            )
    
    # Draw labels
    labels = {i: LOBE_NAMES[i][:4] for i in range(NUM_LOBES)}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                    markersize=10, label=network)
        for network, color in NETWORK_COLORS.items()
    ]
    ax.legend(handles=legend_elements, loc='upper right', title='Network')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='results/paper_figures/causal_graphs/')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load manifest
    if not MASTER_MANIFEST.exists():
        print(f"Error: {MASTER_MANIFEST} not found")
        print("Run: python src/run_pipeline.py --auto --skip-download --skip-split")
        return
    
    manifest = pd.read_csv(MASTER_MANIFEST)
    
    # Get ASD and Control subjects - DX_GROUP is numeric (1=ASD, 2=Control or 0=Control)
    asd_subjects = manifest[manifest['DX_GROUP'] == 1]['subject_id'].astype(str).tolist()
    control_subjects = manifest[manifest['DX_GROUP'] == 2]['subject_id'].astype(str).tolist()
    if not control_subjects:
        control_subjects = manifest[manifest['DX_GROUP'] == 0]['subject_id'].astype(str).tolist()
    
    print(f"Found {len(asd_subjects)} ASD subjects, {len(control_subjects)} Control subjects")
    
    # Compute average causal matrices
    print("Computing ASD average causal matrix...")
    asd_avg = compute_group_average_causal(asd_subjects, CAUSAL_GRAPHS_DIR)
    
    print("Computing Control average causal matrix...")
    control_avg = compute_group_average_causal(control_subjects, CAUSAL_GRAPHS_DIR)
    
    if asd_avg is None or control_avg is None:
        print("Error: Could not compute average causal matrices")
        print(f"Check that graphs exist in: {CAUSAL_GRAPHS_DIR}")
        return
    
    # Generate circular connectome plots
    print("\nGenerating circular connectome plots...")
    create_circular_connectome(
        asd_avg,
        output_dir / 'circular_connectome_ASD.png',
        title='ASD: Causal Connectivity'
    )
    create_circular_connectome(
        control_avg,
        output_dir / 'circular_connectome_Control.png',
        title='Control: Causal Connectivity'
    )
    
    # Also generate difference plot
    diff_matrix = asd_avg - control_avg
    create_circular_connectome(
        diff_matrix,
        output_dir / 'circular_connectome_Difference.png',
        title='ASD - Control: Causal Connectivity Difference'
    )
    
    print(f"\nAll figures saved to: {output_dir}")


if __name__ == '__main__':
    main()
