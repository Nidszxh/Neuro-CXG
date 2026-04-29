"""
Generate publication-ready figures for Neuro-CXG paper.

Produces camera-ready 300 DPI PNG + PDF figures:
1. ROC curves (all models on one plot)
2. Ablation bar chart
3. Training curves (loss + AUC by fold)
4. Attention heatmap (brain regions)
5. Causal graph visualization (ASD vs Control)
6. Confidence calibration plot

Usage:
    python src/analysis/generate_paper_figures.py --output results/paper_figures/
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.rcsetup as rcsetup
from matplotlib import cm
import seaborn as sns

# Add project to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.config import RESULTS_DIR, NUM_LOBES, LOBE_NAMES
from src.models.evaluation import evaluate_loader, compute_metrics
import torch

# Publication-quality matplotlib config
plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
    }
)


def setup_output_dir(output_dir: Path) -> Path:
    """Create output directory structure."""
    output_dir = Path(output_dir)
    (output_dir / "roc_curves").mkdir(parents=True, exist_ok=True)
    (output_dir / "ablations").mkdir(parents=True, exist_ok=True)
    (output_dir / "training_curves").mkdir(parents=True, exist_ok=True)
    (output_dir / "attention").mkdir(parents=True, exist_ok=True)
    (output_dir / "causal_graphs").mkdir(parents=True, exist_ok=True)
    (output_dir / "calibration").mkdir(parents=True, exist_ok=True)
    return output_dir


def generate_roc_curves(output_dir: Path):
    """Generate combined ROC curve figure for all models."""
    print("Generating ROC curves...")

    # Load evaluation results
    eval_path = RESULTS_DIR / "evaluation" / "comprehensive_results.json"
    if not eval_path.exists():
        print(f"  Warning: {eval_path} not found, skipping ROC curves")
        return

    with open(eval_path) as f:
        results = json.load(f)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Plot diagonal reference
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random (AUC=0.5)")

    # Extract ensemble metrics
    if "ensemble_metrics" in results:
        metrics = results["ensemble_metrics"]
        # We need probabilities to plot ROC - load from saved data
        probs = np.array(results.get("ensemble_probs", []))
        labels = np.array(results.get("labels", []))

        if len(probs) > 0 and len(labels) > 0:
            from sklearn.metrics import roc_curve

            fpr, tpr, _ = roc_curve(labels, probs)
            auc = metrics.get("auc", 0.5)
            ax.plot(fpr, tpr, lw=2, label=f"Neuro-CXG (AUC={auc:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - Neuro-CXG Ensemble")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # Save
    fig.savefig(output_dir / "roc_curves" / "roc_curve.png", bbox_inches="tight")
    fig.savefig(output_dir / "roc_curves" / "roc_curve.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved ROC curve to roc_curves/")


def generate_ablation_figure(output_dir: Path):
    """Generate ablation study bar chart."""
    print("Generating ablation figure...")

    # Load ablation results
    abl_path = RESULTS_DIR / "experiments" / "ablations" / "ablation_summary.json"
    if not abl_path.exists():
        print(f"  Warning: {abl_path} not found, skipping ablation figure")
        return

    with open(abl_path) as f:
        ablations = json.load(f)

    # Extract data for plotting
    names = []
    aucs = []
    colors = []

    for key, vals in ablations.items():
        if "auc" in str(key).lower() or "test_auc" in vals:
            name = key.replace("_", " ").title()
            names.append(name)
            aucs.append(vals.get("test_auc", vals.get("auc", 0.5)))
            colors.append("steelblue" if "baseline" in key else "lightcoral")

    if not names:
        print("  No ablation data found")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    bars = ax.barh(names, aucs, color=colors)
    ax.set_xlabel("Test AUC")
    ax.set_title("Ablation Study Results")
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)

    # Add value labels
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"{auc:.3f}", va="center")

    fig.savefig(output_dir / "ablations" / "ablation_bar_chart.png", bbox_inches="tight")
    fig.savefig(output_dir / "ablations" / "ablation_bar_chart.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved ablation figure to ablations/")


def generate_training_curves(output_dir: Path):
    """Generate training curves (loss + AUC by fold)."""
    print("Generating training curves...")

    # Look for training monitor data
    train_dir = RESULTS_DIR / "training"
    if not train_dir.exists():
        print(f"  Warning: {train_dir} not found, skipping training curves")
        return

    # Find fold monitor files
    monitor_files = list(train_dir.glob("**/fold_*_metrics.json"))
    if not monitor_files:
        print("  No training monitor data found")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    ax_loss, ax_auc = axes

    for i, mf in enumerate(sorted(monitor_files)[:5]):  # Max 5 folds
        with open(mf) as f:
            data = json.load(f)

        epochs = [d["epoch"] for d in data]
        train_loss = [d.get("train_loss", 0) for d in data]
        val_auc = [d.get("val_auc", 0.5) for d in data]

        ax_loss.plot(epochs, train_loss, label=f"Fold {i+1} Train Loss")
        ax_auc.plot(epochs, val_auc, label=f"Fold {i+1} Val AUC")

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training Loss by Fold")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_auc.set_xlabel("Epoch")
    ax_auc.set_ylabel("AUC")
    ax_auc.set_title("Validation AUC by Fold")
    ax_auc.legend()
    ax_auc.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "training_curves" / "training_curves.png", bbox_inches="tight")
    fig.savefig(output_dir / "training_curves" / "training_curves.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved training curves to training_curves/")


def generate_attention_heatmap(output_dir: Path):
    """Generate brain region attention heatmap."""
    print("Generating attention heatmap...")

    # Load node importance data
    importance_path = RESULTS_DIR / "explainability" / "node_importance.json"
    if not importance_path.exists():
        print(f"  Warning: {importance_path} not found, skipping attention heatmap")
        return

    with open(importance_path) as f:
        importance = json.load(f)

    # Extract lobe importance (mean across subjects)
    lobe_scores = np.zeros(NUM_LOBES)
    for lobe_idx, lobe_name in enumerate(LOBE_NAMES):
        key = f"lobe_{lobe_idx}"
        if key in importance:
            lobe_scores[lobe_idx] = np.mean(importance[key])

    # Create heatmap
    fig, ax = plt.subplots(1, 1, figsize=(6, 8))
    im = ax.imshow(lobe_scores.reshape(-1, 1), cmap="RdBu_r", aspect="auto")

    ax.set_yticks(range(NUM_LOBES))
    ax.set_yticklabels(LOBE_NAMES)
    ax.set_xticks([])
    ax.set_title("Brain Region Importance (Attention)")
    plt.colorbar(im, ax=ax, label="Importance Score")

    fig.savefig(output_dir / "attention" / "attention_heatmap.png", bbox_inches="tight")
    fig.savefig(output_dir / "attention" / "attention_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved attention heatmap to attention/")


def generate_causal_graphs(output_dir: Path):
    """Generate causal graph visualizations (ASD vs Control)."""
    print("Generating causal graph visualizations...")

    try:
        from src.analysis.visualize_causal_graph import plot_comparison

        output_path = output_dir / "causal_graphs" / "causal_graph_comparison.png"
        result = plot_comparison(
            str(output_path), threshold=0.3, dpi=300
        )
        if result is not None:
            print("  Saved causal graph comparison")
        else:
            print("  Causal graph generation skipped (no data)")
    except Exception as e:
        print(f"  Warning: causal graph visualization failed: {e}")


def generate_calibration_plot(output_dir: Path):
    """Generate confidence calibration plot."""
    print("Generating calibration plot...")

    eval_path = RESULTS_DIR / "evaluation" / "comprehensive_results.json"
    if not eval_path.exists():
        print(f"  Warning: {eval_path} not found, skipping calibration plot")
        return

    with open(eval_path) as f:
        results = json.load(f)

    probs = np.array(results.get("ensemble_probs", []))
    labels = np.array(results.get("labels", []))

    if len(probs) == 0 or len(labels) == 0:
        print("  No probability data found")
        return

    from sklearn.calibration import calibration_curve

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    fraction_of_positives, mean_predicted_value = calibration_curve(labels, probs, n_bins=10)

    ax.plot(mean_predicted_value, fraction_of_positives, "o-", label="Neuro-CXG")
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Confidence Calibration Plot")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(output_dir / "calibration" / "calibration_plot.png", bbox_inches="tight")
    fig.savefig(output_dir / "calibration" / "calibration_plot.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved calibration plot to calibration/")


def generate_architecture_diagram(output_dir: Path):
    """Generate architecture diagram as SVG using matplotlib."""
    print("Generating architecture diagram...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

    # Panel A: Data Pipeline Flowchart
    ax1.axis("off")
    ax1.set_title("A. Data Pipeline Flowchart", fontsize=14, fontweight="bold")

    pipeline_steps = [
        "ABIDE\nDownload",
        "Train/Val/Test\nSplit",
        "Temporal\nFeatures",
        "Spatial\nFeatures",
        "Fold-Safe\nHarmonization",
        "Causal\nGraphs",
        "GNN Training\n(5-Fold CV)",
        "Evaluation &\nExplainability",
    ]

    y_positions = np.linspace(0.9, 0.1, len(pipeline_steps))
    for i, (step, y) in enumerate(zip(pipeline_steps, y_positions)):
        color = "lightblue" if i % 2 == 0 else "lightgreen"
        ax1.text(
            0.5,
            y,
            step,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=color, edgecolor="black"),
            fontsize=10,
        )
        if i < len(pipeline_steps) - 1:
            ax1.annotate(
                "",
                xy=(0.5, y_positions[i + 1] + 0.05),
                xytext=(0.5, y - 0.05),
                arrowprops=dict(arrowstyle="->", lw=2, color="gray"),
            )

    # Panel B: GNN Architecture
    ax2.axis("off")
    ax2.set_title("B. GNN Architecture", fontsize=14, fontweight="bold")

    gnn_layers = [
        "Input:\n12 Lobe Nodes\n24 Features",
        "GATv2\n+ Gradient\nReversal",
        "Anatomical\nHierarchy\nPooling",
        "MLP\nClassifier",
        "Output:\nASD/Control",
    ]

    y_positions = np.linspace(0.9, 0.1, len(gnn_layers))
    for i, (layer, y) in enumerate(zip(gnn_layers, y_positions)):
        color = cm.tab10(i / len(gnn_layers))
        ax2.text(
            0.5,
            y,
            layer,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=color, edgecolor="black", alpha=0.7),
            fontsize=10,
        )
        if i < len(gnn_layers) - 1:
            ax2.annotate(
                "",
                xy=(0.5, y_positions[i + 1] + 0.05),
                xytext=(0.5, y - 0.05),
                arrowprops=dict(arrowstyle="->", lw=2, color="darkblue"),
            )

    fig.tight_layout()
    fig.savefig(output_dir / "architecture_diagram.png", bbox_inches="tight")
    fig.savefig(output_dir / "architecture_diagram.svg", bbox_inches="tight")
    plt.close(fig)
    print("  Saved architecture diagram to architecture_diagram.{png,svg}")


def main():
    parser = argparse.ArgumentParser(description="Generate Neuro-CXG paper figures")
    parser.add_argument(
        "--output", type=str, default="results/paper_figures", help="Output directory for figures"
    )
    args = parser.parse_args()

    output_dir = setup_output_dir(Path(args.output))

    print(f"Generating paper figures in {output_dir}...")
    print("=" * 60)

    generate_roc_curves(output_dir)
    generate_ablation_figure(output_dir)
    generate_training_curves(output_dir)
    generate_attention_heatmap(output_dir)
    generate_causal_graphs(output_dir)
    generate_calibration_plot(output_dir)
    generate_architecture_diagram(output_dir)

    print("=" * 60)
    print(f"All figures saved to {output_dir}")
    print("\nFigure checklist:")
    print("  [1] ROC curves (all models) - roc_curves/")
    print("  [2] Ablation bar chart - ablations/")
    print("  [3] Training curves - training_curves/")
    print("  [4] Attention heatmap - attention/")
    print("  [5] Causal graphs - causal_graphs/")
    print("  [6] Calibration plot - calibration/")
    print("  [7] Architecture diagram - architecture_diagram.{png,svg}")


if __name__ == "__main__":
    main()
