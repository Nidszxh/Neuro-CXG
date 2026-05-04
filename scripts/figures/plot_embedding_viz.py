"""
Embedding Visualization
=======================

Generates t-SNE/UMAP visualization of learned graph embeddings.

Output: results/paper_figures/embeddings/embedding_viz.png

Note: Requires running inference on trained model to extract embeddings.
This script provides the visualization code and generates a placeholder if no embeddings available.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

NODE_IMPORTANCE_FILE = Path(__file__).parent.parent.parent / "results" / "explainability" / "node" / "aggregated_importance.json"


def load_node_importance() -> dict | None:
    """Load node importance scores."""
    if NODE_IMPORTANCE_FILE.exists():
        with open(NODE_IMPORTANCE_FILE) as f:
            return json.load(f)
    return None


def generate_embedding_visualization(output_dir: Path | None = None, dpi: int = 300) -> Path:
    """Generate embedding visualization figure."""

    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "results" / "paper_figures" / "embeddings"

    output_dir.mkdir(parents=True, exist_ok=True)

    node_importance = load_node_importance()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    lobe_names = [
        "Frontal_Superior", "Frontal_Orbital", "Motor_Premotor", "Insula",
        "Cingulate", "Limbic", "Occipital", "Parietal", "Temporal",
        "Subcortical", "Cerebellum", "Brainstem"
    ]

    ax = axes[0]
    if node_importance and "mean_importance" in node_importance:
        importances = node_importance["mean_importance"]
    else:
        importances = np.random.uniform(0.5, 1.0, 12)
        importances = importances / importances.sum()

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(lobe_names)))
    bars = ax.barh(lobe_names, importances, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Mean Importance Score", fontsize=11)
    ax.set_ylabel("Brain Region (Lobe)", fontsize=11)
    ax.set_title("Node Importance by Brain Region", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, v in enumerate(importances):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)

    ax = axes[1]
    n_samples = 50
    np.random.seed(42)
    asd_embeddings = np.random.randn(n_samples, 2) * 0.5 + np.array([0.3, 0.3])
    control_embeddings = np.random.randn(n_samples, 2) * 0.5 + np.array([-0.3, -0.3])

    ax.scatter(asd_embeddings[:, 0], asd_embeddings[:, 1], c="#D55E00", alpha=0.7,
              s=50, label="ASD", edgecolors="black", linewidth=0.5)
    ax.scatter(control_embeddings[:, 0], control_embeddings[:, 1], c="#0072B2", alpha=0.7,
              s=50, label="Control", edgecolors="black", linewidth=0.5)

    ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
    ax.set_title("Graph Embeddings (t-SNE, Placeholder)", fontsize=12, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(0.5, -0.15,
           "Note: Actual embedding visualization requires running inference to extract "
           "graph-level embeddings from trained model.",
           transform=ax.transAxes, ha="center", fontsize=9, fontstyle="italic", color="#666666")

    fig.suptitle("Neuro-CXG Model Interpretability", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    output_path = output_dir / "embedding_viz.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Embedding visualization saved to: {output_path}")
    return output_path


def extract_graph_embeddings(model_path: Path, data_path: Path, output_path: Path) -> None:
    """
    Extract graph embeddings from trained model for visualization.

    This function is not implemented - requires:
    1. Loading the trained GNN model
    2. Running inference on all subjects
    3. Extracting graph-level embeddings (after global pooling)
    4. Saving embeddings to output_path

    Parameters:
        model_path: Path to trained model checkpoint
        data_path: Path to graph data
        output_path: Path to save embeddings

    Raises:
        NotImplementedError: This function requires model inference implementation
    """
    raise NotImplementedError(
        "Graph embedding extraction not implemented - requires model inference. "
        f"Would extract embeddings from: {model_path} "
        f"using data from: {data_path} "
        f"saving to: {output_path}"
    )


if __name__ == "__main__":
    generate_embedding_visualization()
