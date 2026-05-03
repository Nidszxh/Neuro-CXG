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
    CAUSAL_GRAPHS_DIR, MASTER_MANIFEST, LOBE_TO_NETWORK, NETWORK_NAMES
)
import pandas as pd
import torch

# Network membership for each lobe - sync with LOBE_TO_NETWORK from config
NETWORK_MEMBERSHIP = {
    i: NETWORK_NAMES.get(net, "Other")
    for i, net in LOBE_TO_NETWORK.items()
}

NETWORK_COLORS = {
    "DMN": "#1f77b4",          # Blue
    "Salience": "#ff7f0e",     # Orange
    "Visual_Cerebellar": "#2ca02c",  # Green
    "Limbic": "#d62728",       # Red
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


def create_circular_connectome(adj_matrix, output_path, title="Causal Connectivity", 
                               top_edges=20):
    """Create circular connectome plot showing strongest edges.
    
    Uses two-tier rendering: strong edges thick/opaque, weaker edges thin/transparent.
    
    Args:
        adj_matrix: NxN causal adjacency matrix (averaged across subjects)
        output_path: Where to save the PNG
        title: Plot title
        top_edges: Number of strongest edges to display (default: 20)
    """
    import networkx as nx
    
    G = nx.DiGraph()
    
    for i in range(NUM_LOBES):
        G.add_node(i, label=LOBE_NAMES[i])
    
    # Collect and sort edges
    edges_with_weights = []
    for i in range(NUM_LOBES):
        for j in range(NUM_LOBES):
            if i == j:
                continue
            weight = float(adj_matrix[i, j])
            if weight != 0:
                edges_with_weights.append((i, j, weight))
    
    edges_with_weights.sort(key=lambda x: abs(x[2]), reverse=True)
    top_edge_list = edges_with_weights[:top_edges]
    
    for src, dst, weight in top_edge_list:
        G.add_edge(src, dst, weight=weight)
    
    pos = nx.circular_layout(G)
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    edges = list(G.edges())
    if edges:
        all_weights = [G[u][v]['weight'] for u, v in edges]
        abs_weights = np.array([abs(w) for w in all_weights])
        max_abs_w = max(abs_weights)
        min_abs_w = min(abs_weights)
        median_w = np.median(abs_weights)
        
        # Color by signed weight
        norm = Normalize(vmin=-max_abs_w, vmax=max_abs_w)
        cmap = cm.RdBu_r
        edge_colors = [cmap(norm(w)) for w in all_weights]
        
        # Two-tier width: above median = thick (3.0-5.0), below = thin (1.0-2.5)
        widths = []
        alphas = []
        for aw in abs_weights:
            if aw >= median_w:
                # Strong edges
                t = (aw - median_w) / max(max_abs_w - median_w, 1e-8)
                widths.append(3.0 + 2.0 * t)
                alphas.append(0.9)
            else:
                # Weaker edges
                t = (aw - min_abs_w) / max(median_w - min_abs_w, 1e-8)
                widths.append(1.0 + 1.5 * t)
                alphas.append(0.5 + 0.3 * t)
        
        # Draw in two passes: weaker edges first, then stronger on top
        weak_mask = [aw < median_w for aw in abs_weights]
        strong_mask = [not m for m in weak_mask]
        
        weak_edges = [e for e, m in zip(edges, weak_mask) if m]
        strong_edges = [e for e, m in zip(edges, strong_mask) if m]
        
        if weak_edges:
            weak_idx = [i for i, m in enumerate(weak_mask) if m]
            weak_colors = [cmap(norm(G[u][v]['weight'])) for u, v in weak_edges]
            weak_widths = [widths[i] for i in weak_idx]
            weak_alphas = [alphas[i] for i in weak_idx]
            nx.draw_networkx_edges(
                G, pos, edgelist=weak_edges,
                edge_color=weak_colors, width=weak_widths, 
                alpha=min(weak_alphas),  # Use minimum alpha for consistency
                arrows=True, arrowsize=10, arrowstyle='-|>',
                connectionstyle='arc3,rad=0.15', node_size=700, ax=ax
            )
        
        if strong_edges:
            strong_idx = [i for i, m in enumerate(strong_mask) if m]
            strong_colors = [cmap(norm(G[u][v]['weight'])) for u, v in strong_edges]
            strong_widths = [widths[i] for i in strong_idx]
            nx.draw_networkx_edges(
                G, pos, edgelist=strong_edges,
                edge_color=strong_colors, width=strong_widths, alpha=0.9,
                arrows=True, arrowsize=14, arrowstyle='-|>',
                connectionstyle='arc3,rad=0.15', node_size=700, ax=ax
            )
    
    # Draw nodes colored by network (smaller size to not obscure edges)
    for network in sorted(set(NETWORK_MEMBERSHIP.values())):
        nodes = [i for i in range(NUM_LOBES) if NETWORK_MEMBERSHIP.get(i) == network]
        if nodes:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=nodes,
                node_color=NETWORK_COLORS.get(network, "#7f7f7f"),
                node_size=700,
                alpha=0.95,
                edgecolors='#333333',
                linewidths=1.5,
                ax=ax
            )
    
    # Draw labels
    labels = {i: LOBE_NAMES[i].replace('_', '\n') for i in range(NUM_LOBES)}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold', ax=ax)
    
    ax.set_title(f"{title}\n(Top {len(edges)} strongest edges shown)", 
                 fontsize=13, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                    markersize=10, label=network, markeredgecolor='#333333', markeredgewidth=1)
        for network, color in sorted(NETWORK_COLORS.items())
    ]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.0, 0.5), 
              title='Network', fontsize=10, title_fontsize=11)
    
    # Add colorbar for edge weights
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label('Causal weight', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path} ({len(edges)} edges)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='results/paper_figures/causal_graphs/')
    parser.add_argument('--top-edges', type=int, default=30,
                       help='Number of strongest edges to show per plot (default: 30)')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load manifest
    if not MASTER_MANIFEST.exists():
        print(f"Error: {MASTER_MANIFEST} not found")
        print("Run: python src/run_pipeline.py --auto --skip-download --skip-split")
        return
    
    manifest = pd.read_csv(MASTER_MANIFEST)
    
    # Get ASD and Control subjects
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
    print(f"\nGenerating circular connectome plots (top {args.top_edges} edges)...")
    create_circular_connectome(
        asd_avg,
        output_dir / 'circular_connectome_ASD.png',
        title='ASD: Causal Connectivity',
        top_edges=args.top_edges
    )
    create_circular_connectome(
        control_avg,
        output_dir / 'circular_connectome_Control.png',
        title='Control: Causal Connectivity',
        top_edges=args.top_edges
    )
    
    # Also generate difference plot
    diff_matrix = asd_avg - control_avg
    create_circular_connectome(
        diff_matrix,
        output_dir / 'circular_connectome_Difference.png',
        title='ASD - Control: Causal Connectivity Difference',
        top_edges=args.top_edges
    )
    
    print(f"\nAll figures saved to: {output_dir}")


if __name__ == '__main__':
    main()
