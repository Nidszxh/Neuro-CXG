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
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

# Add project to path
from src.core.config import LOBE_NAMES, NUM_LOBES, PROJECT_ROOT, RESULTS_DIR

# Load publication-quality matplotlib style from configs/
matplotlib_rc_path = PROJECT_ROOT / "configs" / "matplotlib.rc"
if matplotlib_rc_path.exists():
    plt.style.use(str(matplotlib_rc_path))
    print(f"  Loaded matplotlib style from {matplotlib_rc_path}")
else:
    print(f"  Warning: {matplotlib_rc_path} not found, using defaults")

# Import ColorPalette for consistent styling
from src.core.plotting import ColorPalette, FigureSize, apply_publication_style

# Set color cycle consistently
palette = ColorPalette()
from cycler import cycler

plt.rcParams["axes.prop_cycle"] = cycler(color=palette.cycle())

# Override DPI for publication
plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300})


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
        print(
            f"  Skipping ROC curves: {eval_path} not found. "
            f"Run: python src/run_evaluation.py --full"
        )
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
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    apply_publication_style(ax)

    # Save with both formats
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
        raise FileNotFoundError(
            f"Ablation summary not found: {abl_path}\n"
            f"Fix: Run: python -m src.experiments.run_ablations"
        )

    with open(abl_path) as f:
        ablations = json.load(f)

    # Extract data for plotting
    names = []
    aucs = []
    colors = []

    for key, vals in ablations.items():
        test_auc = vals.get("test_auc", vals.get("auc", None))
        if test_auc is not None:
            name = key.replace("_", " ").title()
            names.append(name)
            aucs.append(test_auc)
            colors.append(palette.CONTROL if "baseline" in key.lower() else palette.ASD)

    if not names:
        print("  No ablation data found")
        return

    fig, ax = plt.subplots(1, 1, figsize=FigureSize.BAR_CHART)
    bars = ax.barh(
        names,
        aucs,
        color=[
            palette.ASD if "baseline" not in k.lower() else palette.CONTROL
            for k in names
        ],
    )
    ax.set_xlabel("Test AUC", fontsize=12)
    ax.set_title("Ablation Study Results", fontsize=14, fontweight="bold")
    ax.axvline(x=0.5, color=palette.NEUTRAL, linestyle="--", alpha=0.5, lw=1.5)

    # Add value labels
    for bar, auc in zip(bars, aucs, strict=False):
        ax.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{auc:.3f}",
            va="center",
            fontsize=10,
        )

    ax.set_xlim(0.4, 1.05)
    apply_publication_style(ax)

    # Save figure
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        output_dir / "ablation_study.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.savefig(output_dir / "ablation_study.pdf", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved ablation figure to {output_dir}")


def generate_training_curves(output_dir: Path):
    """Generate training curves (loss + AUC by fold)."""
    print("Generating training curves...")

    # Look for training monitor data
    train_dir = RESULTS_DIR / "experiments" / "training"
    if not train_dir.exists():
        raise FileNotFoundError(
            f"Training directory not found: {train_dir}\n"
            f"Fix: Run: python src/run_pipeline.py --auto"
        )

    # Find fold monitor files
    monitor_files = list(train_dir.glob("training_history_fold*.json"))
    if not monitor_files:
        raise FileNotFoundError(
            f"No training history files found in {train_dir}\n"
            f"Fix: Run training with --auto flag"
        )

    fig, axes = plt.subplots(2, 1, figsize=FigureSize.QUAD_PANEL)
    ax_loss, ax_auc = axes

    for i, mf in enumerate(sorted(monitor_files)[:5]):  # Max 5 folds
        with open(mf) as f:
            data = json.load(f)

        epochs = list(range(len(data.get("train_loss", []))))
        train_loss = data.get("train_loss", [])
        val_auc = data.get("val_auc", [])

        color = palette.cycle()[i]
        ax_loss.plot(
            epochs,
            train_loss,
            label=f"Fold {i+1} Train",
            color=color,
            lw=2.5,
            alpha=0.8,
        )
        ax_auc.plot(
            epochs, val_auc, label=f"Fold {i+1} Val", color=color, lw=2.5, alpha=0.8
        )

    ax_loss.set(xlabel="Epoch", ylabel="Loss", title="Training Loss by Fold")
    ax_loss.legend(fontsize=10)
    apply_publication_style(ax_loss)

    ax_auc.set(xlabel="Epoch", ylabel="AUC", title="Validation AUC by Fold")
    ax_auc.legend(loc="lower right", fontsize=10)
    ax_auc.set_ylim([0.4, 1.0])
    apply_publication_style(ax_auc)
    apply_publication_style(ax_auc)

    fig.savefig(
        output_dir / "training_curves" / "training_curves.png", bbox_inches="tight"
    )
    fig.savefig(
        output_dir / "training_curves" / "training_curves.pdf", bbox_inches="tight"
    )
    plt.close(fig)
    print("  Saved training curves to training_curves/")


def generate_attention_heatmap(output_dir: Path):
    """Generate brain region attention heatmap."""
    print("Generating attention heatmap...")

    # Try to load node importance data from explainability output
    summary_path = RESULTS_DIR / "explainability" / "summary.json"
    if not summary_path.exists():
        print(
            f"  Skipping attention heatmap: {summary_path} not found. "
            f"Run: python src/run_explainability.py"
        )
        return

    with open(summary_path) as f:
        summary = json.load(f)

    # Extract lobe importance from gradcam_top5_differential
    lobe_scores = np.zeros(NUM_LOBES)
    if "gradcam_top5_differential" in summary:
        for item in summary["gradcam_top5_differential"]:
            region = item.get("region", "")
            delta = item.get("delta", 0)
            if region in LOBE_NAMES:
                lobe_idx = LOBE_NAMES.index(region)
                lobe_scores[lobe_idx] = delta

    # Create heatmap
    fig, ax = plt.subplots(figsize=FigureSize.HEATMAP)
    im = ax.imshow(lobe_scores.reshape(-1, 1), cmap="RdBu_r", aspect="auto")

    ax.set_yticks(range(NUM_LOBES))
    ax.set_yticklabels(LOBE_NAMES)
    ax.set_xticks([])
    ax.set_title(
        "Brain Region Importance (Attention)", fontsize=14, fontweight="bold", pad=20
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Importance Score", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    fig.savefig(output_dir / "attention" / "attention_heatmap.png", bbox_inches="tight")
    fig.savefig(output_dir / "attention" / "attention_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved attention heatmap to attention/")


def generate_causal_graphs(output_dir: Path):
    """Generate causal graph visualizations (ASD vs Control)."""
    print("Generating causal graph visualizations...")

    # Suppress matplotlib colorbar warning globally for this function
    import warnings

    warnings.filterwarnings("ignore", ".*Colorbar layout.*")

    try:
        import pandas as pd

        from src.analysis.visualize_causal_graph import plot_comparison
        from src.core.config import MASTER_MANIFEST

        # Load manifest to get example subjects
        df = pd.read_csv(MASTER_MANIFEST)

        # Filter to valid subjects (not in excluded list)
        from src.core.hyperparams import EXCLUDED_SUBJECTS

        df = df[~df["subject_id"].isin(EXCLUDED_SUBJECTS)]

        asd_subjects = df[df["DX_GROUP"] == 1]["subject_id"].tolist()
        control_subjects = df[df["DX_GROUP"] == 2]["subject_id"].tolist()

        if not asd_subjects:
            asd_subjects = df[df["DX_GROUP"] == 1]["subject_id"].tolist()
        if not control_subjects:
            # DX_GROUP might be 1=ASD, 2=Control in some formats
            control_subjects = df[df["DX_GROUP"] == 0]["subject_id"].tolist()

        if not asd_subjects or not control_subjects:
            # Debug: show column values
            print(f"  Debug: DX_GROUP unique values: {df['DX_GROUP'].unique()}")
            print("  Skipping causal graph: no ASD/Control subjects found")
            return

        asd_subject = asd_subjects[0]
        control_subject = control_subjects[0]

        output_path = output_dir / "causal_graphs" / "causal_graph_comparison.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        plot_comparison(
            asd_subject,
            control_subject,
            output_path,
            threshold=0.0,
            dpi=300,
        )
        if output_path.exists():
            print("  Saved causal graph comparison")
        else:
            print("  Causal graph generation failed (no output)")
    except Exception as e:
        if "Colorbar layout" not in str(e):
            print(f"  Warning: causal graph visualization failed: {e}")


def generate_calibration_plot(output_dir: Path):
    """Generate confidence calibration plot."""
    print("Generating calibration plot...")

    eval_path = RESULTS_DIR / "evaluation" / "comprehensive_results.json"
    if not eval_path.exists():
        raise FileNotFoundError(
            f"Evaluation results not found: {eval_path}\n"
            f"Fix: Run: python src/run_evaluation.py"
        )

    with open(eval_path) as f:
        results = json.load(f)

    probs = np.array(results.get("ensemble_probs", []))
    labels = np.array(results.get("labels", []))

    if len(probs) == 0 or len(labels) == 0:
        raise ValueError(
            f"No probability data found in {eval_path}.\n"
            f"Fix: Ensure run_evaluation.py saves 'ensemble_probs' and 'labels'."
        )

    from sklearn.calibration import calibration_curve

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    fraction_of_positives, mean_predicted_value = calibration_curve(
        labels, probs, n_bins=10
    )

    ax.plot(mean_predicted_value, fraction_of_positives, "o-", label="Neuro-CXG")
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Confidence Calibration Plot")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(
        output_dir / "calibration" / "calibration_plot.png", bbox_inches="tight"
    )
    fig.savefig(
        output_dir / "calibration" / "calibration_plot.pdf", bbox_inches="tight"
    )
    plt.close(fig)
    print("  Saved calibration plot to calibration/")


def generate_dataflow_diagram(output_dir: Path):
    """Generate LaTeX-style data pipeline diagram."""
    print("Generating dataflow diagram...")

    # Use LaTeX-style font
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis("off")

    # Clean horizontal layout
    stages = [
        ("ABIDE I", "15 sites, 154 subjects"),
        ("Split", "60/20/20\nstratified"),
        ("Features", "24 features\n12 lobes"),
        ("Harmonize", "ComBat\nfold-safe"),
        ("Causal", "Ridge-Granger\n12×12"),
        ("GNN", "GATv2+GRL\n5-fold CV"),
        ("Eval", "AUC=0.877"),
    ]

    n = len(stages)
    xs = np.linspace(0.06, 0.94, n)

    for i, ((name, desc), x) in enumerate(zip(stages, xs, strict=False)):
        # Clean rectangle
        rect = plt.Rectangle(
            (x - 0.055, 0.2),
            0.11,
            0.6,
            facecolor="#f5f5f5",
            edgecolor="#333",
            linewidth=1.5,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)

        # Stage name
        ax.text(
            x,
            0.55,
            name,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            transform=ax.transAxes,
        )

        # Description
        ax.text(
            x,
            0.38,
            desc,
            ha="center",
            va="center",
            fontsize=8,
            transform=ax.transAxes,
            color="#555",
        )

        # Arrow
        if i < n - 1:
            ax.annotate(
                "",
                xy=(xs[i + 1] - 0.058, 0.5),
                xytext=(x + 0.058, 0.5),
                arrowprops={"arrowstyle": "->", "color": "#333", "lw": 1.5},
                transform=ax.transAxes,
            )

    # Top label
    ax.text(
        0.5,
        0.95,
        "Data Processing Pipeline",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
        transform=ax.transAxes,
    )

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / "dataflow_diagram.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print("  Saved dataflow_diagram.png")


def generate_feature_extraction_diagram(output_dir: Path):
    """Generate clean, minimal feature extraction diagram."""
    print("Generating feature extraction diagram...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # === Panel A: Feature Groups (clean grid) ===
    ax1 = axes[0]
    ax1.axis("off")
    ax1.set_title("A. 24 Features", fontsize=13, fontweight="bold", pad=10)

    # Feature groups as clean boxes
    groups = [
        ("Temporal\n8", "#3498db"),
        ("Frequency\n10", "#e74c3c"),
        ("Internal\n2", "#27ae60"),
        ("Spatial\n4", "#9b59b6"),
    ]

    for i, (label, color) in enumerate(groups):
        x = 0.2 + i * 0.2
        rect = plt.Rectangle(
            (x - 0.08, 0.4),
            0.16,
            0.3,
            facecolor=color,
            alpha=0.85,
            edgecolor="black",
            linewidth=2,
            transform=ax1.transAxes,
        )
        ax1.add_patch(rect)
        ax1.text(
            x,
            0.55,
            label,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="white",
            transform=ax1.transAxes,
        )

    # Feature list
    features_text = (
        "Temporal: mean, std, skew, kurt, PSD, MSSD, range, autocorr\n"
        "Frequency: delta/theta/alpha/beta (power + peak) + entropy + phase_std\n"
        "Internal: coherence, spatial_variance\n"
        "Spatial: x, y, z_depth, size"
    )
    ax1.text(
        0.5,
        0.15,
        features_text,
        ha="center",
        va="center",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "#f8f9fa",
            "edgecolor": "#bdc3c7",
        },
        transform=ax1.transAxes,
    )

    # Exclusions note
    ax1.text(
        0.5,
        0.02,
        "Excluded: gamma (Nyquist), conf_std/detection_count (site leakage)",
        ha="center",
        fontsize=8,
        style="italic",
        color="#7f8c8d",
        transform=ax1.transAxes,
    )

    # === Panel B: Extraction pipeline (clean flow) ===
    ax2 = axes[1]
    ax2.axis("off")
    ax2.set_title("B. Pipeline", fontsize=13, fontweight="bold", pad=10)

    steps = [
        ("fMRI", "#3498db"),
        ("12 Lobes", "#9b59b6"),
        ("Bandpass", "#e67e22"),
        ("Features", "#e74c3c"),
        ("ComBat", "#f39c12"),
        ("Graph", "#2c3e50"),
    ]

    n_steps = len(steps)
    x_pos = np.linspace(0.1, 0.9, n_steps)

    for i, ((step, color), x) in enumerate(zip(steps, x_pos, strict=False)):
        circle = plt.Circle((x, 0.5), 0.055, color=color, alpha=0.85)
        ax2.add_patch(circle)
        ax2.text(
            x,
            0.5,
            str(i + 1),
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white",
        )
        ax2.text(
            x,
            0.28,
            step,
            ha="center",
            fontsize=9,
            fontweight="bold",
            transform=ax2.transAxes,
        )

        if i < n_steps - 1:
            ax2.annotate(
                "",
                xy=(x_pos[i + 1] - 0.07, 0.5),
                xytext=(x + 0.07, 0.5),
                arrowprops={"arrowstyle": "->", "lw": 2, "color": "#2c3e50"},
            )

    # Key stats
    ax2.text(
        0.5,
        0.12,
        "TR=2.0s • 0.01-0.15 Hz • Ridge-Granger (70% + 30% Pearson)",
        ha="center",
        fontsize=9,
        transform=ax2.transAxes,
    )

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / "feature_extraction.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print("  Saved feature_extraction.png")


def generate_architecture_diagram(output_dir: Path):
    """Generate clean, minimal GNN architecture diagram."""
    print("Generating GNN architecture diagram...")

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")
    ax.set_title("GNN Architecture", fontsize=16, fontweight="bold", pad=15)

    # Clean layer boxes with numbers
    layers = [
        ("Input\n12×24", "#3498db"),
        ("Linear\nLayerNorm", "#9b59b6"),
        ("GATv2×3\n48ch, 4head", "#e74c3c"),
        ("Edge\nGate", "#f39c12"),
        ("GRL\nα=0.10", "#e67e22"),
        ("Pool\nmean+max+sum", "#27ae60"),
        ("MLP\nClassifier", "#16a085"),
        ("Output\nASD/Control", "#17a2b8"),
    ]

    n = len(layers)
    x_positions = np.linspace(0.07, 0.93, n)

    for i, ((layer, color), x) in enumerate(zip(layers, x_positions, strict=False)):
        # Clean box
        rect = plt.Rectangle(
            (x - 0.05, 0.35),
            0.1,
            0.3,
            facecolor=color,
            alpha=0.85,
            edgecolor="black",
            linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(
            x,
            0.5,
            layer,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
        )

        # Arrow
        if i < n - 1:
            ax.annotate(
                "",
                xy=(x_positions[i + 1] - 0.055, 0.5),
                xytext=(x + 0.055, 0.5),
                arrowprops={"arrowstyle": "->", "lw": 2.5, "color": "#2c3e50"},
            )

    # Key hyperparameters (clean sidebar)
    hyperparams = (
        "HYPERPARAMETERS\n"
        "─────────────────\n"
        "Hidden: 48\n"
        "Heads: 4\n"
        "Layers: 3\n"
        "Dropout: 0.33\n"
        "GRL: α=0.10\n"
        "Node emb: 16D\n"
        "Site emb: 16D"
    )
    ax.text(
        0.97,
        0.5,
        hyperparams,
        ha="right",
        va="center",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "#ecf0f1",
            "edgecolor": "#34495e",
            "linewidth": 2,
        },
        transform=ax.transAxes,
        family="monospace",
    )

    # Input/output annotations
    ax.text(
        0.02,
        0.5,
        "fMRI\nFeatures",
        ha="left",
        va="center",
        fontsize=9,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.98,
        0.5,
        "ASD /\nControl",
        ha="right",
        va="center",
        fontsize=9,
        fontweight="bold",
        transform=ax.transAxes,
    )

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / "gnn_architecture.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print("  Saved gnn_architecture.png")


def generate_per_site_chart(output_dir: Path):
    """Generate per-site AUC bar chart."""
    print("Generating per-site AUC chart...")

    eval_path = RESULTS_DIR / "evaluation" / "comprehensive_results.json"
    if not eval_path.exists():
        raise FileNotFoundError(
            f"Evaluation results not found: {eval_path}\n"
            f"Fix: Run: python src/run_evaluation.py"
        )

    with open(eval_path) as f:
        results = json.load(f)

    subgroup = results.get("subgroup_analysis", {})
    site_data = {k: v for k, v in subgroup.items() if k.startswith("site_")}

    if not site_data:
        raise ValueError(
            f"No site data found in {eval_path}\n"
            f"Fix: Ensure evaluation includes subgroup analysis with sites."
        )

    names = list(site_data.keys())
    aucs = [site_data[k].get("auc", 0.5) for k in names]
    ns = [site_data[k].get("n", 0) for k in names]

    fig, ax = plt.subplots(1, 1, figsize=FigureSize.BAR_CHART)
    colors = [
        palette.ASD if site_data[k].get("n_asd", 0) > 0 else palette.NEUTRAL
        for k in names
    ]
    [
        "ASD + Control" if site_data[k].get("n_asd", 0) > 0 else "Control only"
        for k in names
    ]
    ax.bar(range(len(names)), aucs, color=colors, edgecolor="black", linewidth=0.5)

    # Create legend handles manually since bars have same color but different categories
    asd_handle = plt.Rectangle(
        (0, 0), 1, 1, facecolor=palette.ASD, edgecolor="black", linewidth=0.5
    )
    ctrl_handle = plt.Rectangle(
        (0, 0), 1, 1, facecolor=palette.NEUTRAL, edgecolor="black", linewidth=0.5
    )
    ax.legend(
        [asd_handle, ctrl_handle],
        ["ASD + Control", "Control only"],
        fontsize=10,
        loc="lower right",
    )

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(
        [f"{n}\n(n={ns[i]})" for i, n in enumerate(names)],
        rotation=45,
        ha="right",
        fontsize=10,
    )
    ax.set_ylabel("Test AUC", fontsize=12)
    ax.set_title("Per-Site AUC (n per site shown)", fontsize=14, fontweight="bold")
    ax.axhline(y=0.5, color=palette.NEUTRAL, linestyle="--", alpha=0.5, lw=1.5)
    ax.set_ylim(0, 1)
    apply_publication_style(ax)

    fig.savefig(output_dir / "per_site_auc.png", bbox_inches="tight")
    fig.savefig(output_dir / "per_site_auc.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved per-site AUC chart to per_site_auc.{png,pdf}")


def generate_bootstrap_ci_figure(output_dir: Path):
    """Generate Bootstrap CI visualization."""
    print("Generating Bootstrap CI figure...")

    eval_path = RESULTS_DIR / "evaluation" / "comprehensive_results.json"
    if not eval_path.exists():
        raise FileNotFoundError(
            f"Evaluation results not found: {eval_path}\n"
            f"Fix: Run: python src/run_evaluation.py"
        )

    with open(eval_path) as f:
        results = json.load(f)

    ci = results.get("ensemble_ci_95", {})
    metrics = results.get("ensemble_metrics", {})

    if not ci or not metrics:
        raise ValueError(
            f"No Bootstrap CI data found in {eval_path}\n"
            f"Fix: Ensure evaluation includes bootstrap CI computation."
        )

    metric_names = ["auc", "f1", "accuracy", "sensitivity", "specificity"]
    labels = ["AUC", "F1", "Accuracy", "Sensitivity", "Specificity"]

    fig, ax = plt.subplots(1, 1, figsize=FigureSize.SINGLE)

    y_pos = np.arange(len(metric_names))
    values = [metrics.get(m, 0.5) for m in metric_names]
    errors = [
        [metrics.get(m, 0.5) - ci.get(m, [0.5, 0.5])[0] for m in metric_names],
        [ci.get(m, [0.5, 0.5])[1] - metrics.get(m, 0.5) for m in metric_names],
    ]

    ax.barh(
        y_pos,
        values,
        xerr=errors,
        capsize=5,
        color=palette.CONTROL,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Value (with 95% CI)", fontsize=12)
    ax.set_title("Bootstrap 95% Confidence Intervals", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1)
    apply_publication_style(ax)

    fig.savefig(output_dir / "bootstrap_ci.png", bbox_inches="tight")
    fig.savefig(output_dir / "bootstrap_ci.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved Bootstrap CI figure to bootstrap_ci.{png,pdf}")


def main():
    parser = argparse.ArgumentParser(description="Generate Neuro-CXG paper figures")
    parser.add_argument(
        "--output",
        type=str,
        default=str(RESULTS_DIR / "paper_figures"),
        help="Output directory for figures",
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
    generate_per_site_chart(output_dir)
    generate_bootstrap_ci_figure(output_dir)
    generate_dataflow_diagram(output_dir)
    generate_feature_extraction_diagram(output_dir)
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
    print("  [7] Per-site AUC chart - per_site_auc.{png,pdf}")
    print("  [8] Bootstrap CI figure - bootstrap_ci.{png,pdf}")
    print("  [9] Dataflow diagram - dataflow_diagram.png")
    print("  [10] Feature extraction - feature_extraction.png")
    print("  [11] GNN architecture - gnn_architecture.png")
    print("  [9] Architecture diagram - architecture_diagram.{png,svg}")


if __name__ == "__main__":
    main()
