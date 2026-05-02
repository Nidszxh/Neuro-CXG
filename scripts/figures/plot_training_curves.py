"""
Training Curves Figure
=======================

Generates loss and AUC curves over epochs for each fold.

Output: results/paper_figures/training_curves/training_curves.png
"""

import json
from pathlib import Path
from typing import Optional, List, Dict

import matplotlib.pyplot as plt
import numpy as np


TRAINING_DIR = Path(__file__).parent.parent.parent / "results" / "experiments" / "training"


def load_training_history() -> List[Dict]:
    """Load training history from all folds."""
    histories = []
    for fold in range(5):
        history_file = TRAINING_DIR / f"training_history_fold{fold}.json"
        if history_file.exists():
            with open(history_file) as f:
                histories.append(json.load(f))
    return histories


def generate_training_curves(output_dir: Optional[Path] = None, dpi: int = 300) -> Path:
    """Generate training curves figure."""

    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "results" / "paper_figures" / "training_curves"

    output_dir.mkdir(parents=True, exist_ok=True)

    histories = load_training_history()

    if not histories:
        print("No training history found")
        return output_dir / "training_curves.png"  # Return expected path

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]

    for fold_idx, history in enumerate(histories):
        epochs = range(1, len(history["train_loss"]) + 1)
        color = colors[fold_idx]

        axes[0, 0].plot(epochs, history["train_loss"], color=color, alpha=0.7,
                       label=f"Fold {fold_idx}", linewidth=1.5)

        axes[0, 1].plot(epochs, history["val_loss"], color=color, alpha=0.7,
                       label=f"Fold {fold_idx}", linewidth=1.5)

        axes[1, 0].plot(epochs, history["val_auc"], color=color, alpha=0.7,
                       label=f"Fold {fold_idx}", linewidth=1.5)

        axes[1, 1].plot(epochs, history["val_f1"], color=color, alpha=0.7,
                       label=f"Fold {fold_idx}", linewidth=1.5)

    for ax in axes[0]:
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Loss", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.3, linestyle="--")

    for ax in axes[1]:
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Metric", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.3, linestyle="--")
        ax.set_ylim(0.5, 1.0)

    axes[0, 0].set_title("Training Loss", fontsize=12, fontweight="bold")
    axes[0, 1].set_title("Validation Loss", fontsize=12, fontweight="bold")
    axes[1, 0].set_title("Validation AUC", fontsize=12, fontweight="bold")
    axes[1, 1].set_title("Validation F1", fontsize=12, fontweight="bold")

    axes[1, 0].legend(loc="lower right", fontsize=8, ncol=5)

    fig.suptitle("Neuro-CXG Training Curves (5-Fold Cross-Validation)", fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()

    output_path = output_dir / "training_curves.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Training curves saved to: {output_path}")
    return output_path


def generate_mean_training_curves(output_dir: Optional[Path] = None, dpi: int = 300) -> Path:
    """Generate mean training curves across folds."""

    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "results" / "paper_figures" / "training_curves"

    output_dir.mkdir(parents=True, exist_ok=True)

    histories = load_training_history()

    if not histories:
        print("No training history found")
        return output_dir / "training_curves_mean.png"  # Return expected path

    max_epochs = max(len(h["train_loss"]) for h in histories)

    def pad_to_length(arr, length):
        """Pad array to max length for aligned averaging."""
        result = np.full(length, np.nan)
        result[:len(arr)] = arr
        return result

    mean_train_loss = np.nanmean([pad_to_length(h["train_loss"], max_epochs) for h in histories], axis=0)
    std_train_loss = np.nanstd([pad_to_length(h["train_loss"], max_epochs) for h in histories], axis=0)

    mean_val_loss = np.nanmean([pad_to_length(h["val_loss"], max_epochs) for h in histories], axis=0)
    std_val_loss = np.nanstd([pad_to_length(h["val_loss"], max_epochs) for h in histories], axis=0)

    mean_val_auc = np.nanmean([pad_to_length(h["val_auc"], max_epochs) for h in histories], axis=0)
    std_val_auc = np.nanstd([pad_to_length(h["val_auc"], max_epochs) for h in histories], axis=0)

    mean_val_f1 = np.nanmean([pad_to_length(h["val_f1"], max_epochs) for h in histories], axis=0)
    std_val_f1 = np.nanstd([pad_to_length(h["val_f1"], max_epochs) for h in histories], axis=0)

    epochs = range(1, max_epochs + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, mean_train_loss, color="#0072B2", linewidth=2, label="Train")
    axes[0].fill_between(epochs, mean_train_loss - std_train_loss,
                         mean_train_loss + std_train_loss, alpha=0.2, color="#0072B2")
    axes[0].plot(epochs, mean_val_loss, color="#D55E00", linewidth=2, label="Validation")
    axes[0].fill_between(epochs, mean_val_loss - std_val_loss,
                         mean_val_loss + std_val_loss, alpha=0.2, color="#D55E00")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves", fontweight="bold")
    axes[0].legend()
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].plot(epochs, mean_val_auc, color="#009E73", linewidth=2)
    axes[1].fill_between(epochs, mean_val_auc - std_val_auc,
                         mean_val_auc + std_val_auc, alpha=0.2, color="#009E73")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC")
    axes[1].set_title("Validation AUC", fontweight="bold")
    axes[1].set_ylim(0.6, 0.9)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    axes[2].plot(epochs, mean_val_f1, color="#CC79A7", linewidth=2)
    axes[2].fill_between(epochs, mean_val_f1 - std_val_f1,
                         mean_val_f1 + std_val_f1, alpha=0.2, color="#CC79A7")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1 Score")
    axes[2].set_title("Validation F1", fontweight="bold")
    axes[2].set_ylim(0.5, 0.85)
    axes[2].spines["top"].set_visible(False)
    axes[2].spines["right"].set_visible(False)

    fig.suptitle("Mean Training Curves ± Std (5-Fold CV)", fontsize=12, fontweight="bold")
    plt.tight_layout()

    output_path = output_dir / "training_curves_mean.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"Mean training curves saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_training_curves()
    generate_mean_training_curves()