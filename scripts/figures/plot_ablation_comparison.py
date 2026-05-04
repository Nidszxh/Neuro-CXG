"""
Ablation Study Comparison Figure
=================================

Generates publication-quality bar chart comparing all ablation experiments.

Output: results/paper_figures/ablations/ablation_comparison.png
"""

from pathlib import Path

import matplotlib.pyplot as plt

ABLATION_DATA = {
    "Full Model": {"auc": 0.8587, "std": 0.0240, "f1": 0.8121},
    "A: FlatMLP": {"auc": 0.7267, "std": 0.0075, "f1": 0.6729},
    "B: Spatial Only": {"auc": 0.5377, "std": 0.0231, "f1": 0.4981},
    "C: No Freq": {"auc": 0.7463, "std": 0.0256, "f1": 0.6993},
    "D: Pearson Edges": {"auc": 0.8574, "std": 0.0245, "f1": 0.8092},
    "D2: Ridge Granger": {"auc": 0.8466, "std": 0.0326, "f1": 0.7977},
    "E: No Conditioning": {"auc": 0.7441, "std": 0.0250, "f1": 0.6797},
    "Baseline LR": {"auc": 0.6171, "std": 0.0425, "f1": 0.5725},
}

BASELINE_AUC = 0.8587


def generate_ablation_figure(output_dir: Path | None = None, dpi: int = 300) -> Path:
    """Generate ablation comparison figure."""

    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "results" / "paper_figures" / "ablations"

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    experiments = list(ABLATION_DATA.keys())
    aucs = [ABLATION_DATA[e]["auc"] for e in experiments]
    stds = [ABLATION_DATA[e]["std"] for e in experiments]

    colors = []
    for e in experiments:
        if e == "Full Model":
            colors.append("#0072B2")
        elif "LR" in e:
            colors.append("#56B4E9")
        else:
            colors.append("#999999")

    bars = ax.bar(range(len(experiments)), aucs, yerr=stds, capsize=4,
                  color=colors, edgecolor="black", linewidth=0.8, alpha=0.85)

    ax.axhline(y=BASELINE_AUC, color="#0072B2", linestyle="--", linewidth=1.5,
               label=f"Full Model ({BASELINE_AUC:.3f})", alpha=0.7)

    for i, (exp, auc) in enumerate(zip(experiments, aucs)):
        delta = (auc - BASELINE_AUC) * 100
        label = f"{delta:+.1f}%"
        va = "bottom" if delta >= 0 else "top"
        offset = 0.02 if delta >= 0 else -0.02
        ax.annotate(label, (i, auc + stds[i] + offset),
                    ha="center", va=va, fontsize=8, fontweight="bold")

    ax.set_xticks(range(len(experiments)))
    ax.set_xticklabels(experiments, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Cross-Validation AUC", fontsize=11, fontweight="bold")
    ax.set_xlabel("Ablation Experiment", fontsize=11, fontweight="bold")
    ax.set_title("Ablation Study: Component Contribution Analysis", fontsize=13, fontweight="bold", pad=15)

    ax.set_ylim(0.45, 1.0)
    ax.set_xlim(-0.6, len(experiments) - 0.4)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    legend_elements = [
        plt.Line2D([0], [0], color="#0072B2", linewidth=1.5, linestyle="--", label="Full Model Baseline"),
        plt.Rectangle((0, 0), 1, 1, fc="#0072B2", ec="black", label="Full Model"),
        plt.Rectangle((0, 0), 1, 1, fc="#56B4E9", ec="black", label="Baseline Comparison"),
        plt.Rectangle((0, 0), 1, 1, fc="#999999", ec="black", label="Ablation"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()

    output_path = output_dir / "ablation_comparison.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Ablation figure saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_ablation_figure()
