import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _to_json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_safe(v) for v in obj]
    return obj


class TrainingMonitor:
    """
    Real-time and post-hoc training visualization.

    Tracks metrics across epochs and generates diagnostic plots:
    - Loss curves (detect overfitting/underfitting)
    - AUC progression (track learning quality)
    - Learning rate schedule (verify warmup/annealing)
    - Gradient norms (detect training instabilities)
    - Confusion matrix evolution (detect class collapse)
    """

    def __init__(self, output_dir: Path, num_folds: int = 5):
        """
        Initialize training monitor.

        Args:
            output_dir: Directory to save plots
            num_folds: Number of cross-validation folds
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_folds = num_folds

        self.fold_histories = {
            fold_id: {
                "train_loss": [],
                "val_loss": [],
                "val_auc": [],
                "val_f1": [],
                "val_acc": [],
                "learning_rate": [],
                "grad_norm": [],
                "confusion_matrices": [],
            }
            for fold_id in range(num_folds)
        }

        logger.info("TrainingMonitor initialized")
        logger.info(f"  Output directory: {output_dir}")
        logger.info(f"  Tracking {num_folds} folds")

    def log_epoch(
        self,
        fold_id: int,
        epoch: int,
        metrics: Dict[str, float],
        grad_norm: Optional[float] = None,
        confusion_matrix: Optional[np.ndarray] = None,
    ):
        """
        Log metrics for one epoch.

        Args:
            fold_id: Current fold (0-4)
            epoch: Current epoch number
            metrics: Dictionary with keys: 'train_loss', 'val_loss', 'val_auc', 'val_f1', 'lr'
            grad_norm: Optional gradient norm (for stability tracking)
            confusion_matrix: Optional 2x2 confusion matrix
        """
        if fold_id not in self.fold_histories:
            raise ValueError(f"Invalid fold_id: {fold_id}")

        history = self.fold_histories[fold_id]

        history["train_loss"].append(metrics.get("train_loss", 0.0))
        history["val_loss"].append(metrics.get("val_loss", 0.0))
        history["val_auc"].append(metrics.get("val_auc", 0.0))
        history["val_f1"].append(metrics.get("val_f1", 0.0))
        history["val_acc"].append(metrics.get("val_acc", 0.0))
        history["learning_rate"].append(metrics.get("lr", 0.0))

        if grad_norm is not None:
            history["grad_norm"].append(grad_norm)

        if confusion_matrix is not None:
            history["confusion_matrices"].append(confusion_matrix.copy())

    def plot_training_curves(self, fold_id: int, figsize: Tuple[int, int] = (18, 12)) -> Optional[Path]:
        """
        Generate comprehensive training diagnostic plot.

        Creates a 4-panel figure:
        1. Loss curves (train/val)
        2. AUC progression
        3. Learning rate schedule
        4. Gradient norm (if tracked)
        """
        logger.info(f"Generating training curves for fold {fold_id}...")

        history = self.fold_histories[fold_id]

        if not history["train_loss"]:
            logger.warning(f"No training history for fold {fold_id}")
            return None

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        epochs = range(1, len(history["train_loss"]) + 1)

        ax = axes[0, 0]
        ax.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2.5, color="#3498db", alpha=0.8)
        ax.plot(epochs, history["val_loss"], label="Val Loss", linewidth=2.5, color="#e74c3c", alpha=0.8)

        best_val_idx = np.argmin(history["val_loss"])
        best_val_epoch = best_val_idx + 1
        best_val_loss = history["val_loss"][best_val_idx]
        ax.scatter([best_val_epoch], [best_val_loss], color="#e74c3c", s=200, zorder=5, marker="*", label=f"Best Val (Epoch {best_val_epoch})")

        ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
        ax.set_ylabel("Loss", fontsize=12, fontweight="bold")
        ax.set_title("Loss Curves", fontsize=14, fontweight="bold", pad=15)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(alpha=0.3, linestyle="--")
        ax.set_ylim(bottom=0)

        ax = axes[0, 1]
        ax.plot(epochs, history["val_auc"], color="#2ecc71", linewidth=2.5, alpha=0.8, label="Validation AUC")
        ax.axhline(0.5, color="#95a5a6", linestyle="--", label="Random (0.5)", alpha=0.7, linewidth=2)

        best_auc = max(history["val_auc"])
        best_auc_idx = history["val_auc"].index(best_auc)
        best_auc_epoch = best_auc_idx + 1

        ax.axhline(best_auc, color="#27ae60", linestyle="--", label=f"Best: {best_auc:.4f} (Epoch {best_auc_epoch})", alpha=0.7, linewidth=2)
        ax.scatter([best_auc_epoch], [best_auc], color="#27ae60", s=200, zorder=5, marker="*")

        ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
        ax.set_ylabel("Validation AUC", fontsize=12, fontweight="bold")
        ax.set_title("AUC Progression", fontsize=14, fontweight="bold", pad=15)
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(alpha=0.3, linestyle="--")
        ax.set_ylim([0.4, 1.0])

        ax = axes[1, 0]
        ax.plot(epochs, history["learning_rate"], color="#f39c12", linewidth=2.5, alpha=0.8)
        ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
        ax.set_ylabel("Learning Rate", fontsize=12, fontweight="bold")
        ax.set_title("LR Schedule (Warmup + Cosine)", fontsize=14, fontweight="bold", pad=15)
        ax.set_yscale("log")
        ax.grid(alpha=0.3, linestyle="--")

        ax = axes[1, 1]
        if history["grad_norm"]:
            ax.plot(epochs, history["grad_norm"], color="#9b59b6", linewidth=2.5, alpha=0.8, label="Gradient Norm")
            ax.axhline(1.0, color="#e74c3c", linestyle="--", label="Clip Threshold (1.0)", linewidth=2, alpha=0.7)
            ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
            ax.set_ylabel("Gradient Norm", fontsize=12, fontweight="bold")
            ax.set_title("Gradient Stability", fontsize=14, fontweight="bold", pad=15)
            ax.legend(loc="upper right", fontsize=10)
            ax.grid(alpha=0.3, linestyle="--")
            ax.set_ylim(bottom=0)
        else:
            ax.text(0.5, 0.5, "Gradient norm not tracked", ha="center", va="center", transform=ax.transAxes, fontsize=14, color="#7f8c8d")
            ax.axis("off")

        plt.suptitle(f"Training Diagnostics - Fold {fold_id}", fontsize=18, fontweight="bold", y=0.995)
        plt.tight_layout()

        fold_plots_dir = self.output_dir / "fold_plots"
        fold_plots_dir.mkdir(parents=True, exist_ok=True)
        output_path = fold_plots_dir / f"training_curves_fold_{fold_id}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

        logger.info(f"Training curves saved to {output_path}")
        return output_path

    def plot_confusion_evolution(
        self,
        fold_id: int,
        key_epochs: Optional[List[int]] = None,
        figsize: Tuple[int, int] = (18, 12),
    ) -> Optional[Path]:
        """
        Visualize confusion matrix changes across epochs.
        """
        logger.info(f"Generating confusion matrix evolution for fold {fold_id}...")

        history = self.fold_histories[fold_id]
        cm_history = history["confusion_matrices"]

        if not cm_history:
            logger.warning(f"No confusion matrices tracked for fold {fold_id}")
            return None

        if key_epochs is None:
            total_epochs = len(cm_history)
            key_epochs = [0, 9, 24, 49, 74, -1]
            key_epochs = [e if e < total_epochs else -1 for e in key_epochs]

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()

        class_names = ["Control", "ASD"]

        for idx, epoch_idx in enumerate(key_epochs):
            if idx >= len(axes):
                break
            ax = axes[idx]
            cm = cm_history[epoch_idx]
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=class_names, yticklabels=class_names)
            ax.set_title(f"Epoch {epoch_idx + 1}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")

        plt.tight_layout()
        output_path = self.output_dir / f"confusion_evolution_fold_{fold_id}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Confusion evolution saved to {output_path}")
        return output_path

    def plot_fold_comparison(self, figsize: Tuple[int, int] = (14, 8)) -> Optional[Path]:
        """Compare training curves across folds."""
        if not any(self.fold_histories[fold_id]["val_auc"] for fold_id in self.fold_histories):
            logger.warning("No validation AUC data to compare")
            return None

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        for fold_id, history in self.fold_histories.items():
            if not history["val_auc"]:
                continue
            epochs = range(1, len(history["val_auc"]) + 1)
            axes[0].plot(epochs, history["val_auc"], label=f"Fold {fold_id}")

        axes[0].set_title("Validation AUC by Fold")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("AUC")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        final_aucs = [history["val_auc"][-1] for history in self.fold_histories.values() if history["val_auc"]]
        if final_aucs:
            axes[1].bar(range(len(final_aucs)), final_aucs, color="#3498db")
            axes[1].set_title("Final Validation AUC per Fold")
            axes[1].set_xlabel("Fold")
            axes[1].set_ylabel("AUC")
            axes[1].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / "fold_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Fold comparison saved to {output_path}")
        return output_path

    def save_histories(self, output_dir: Optional[Path] = None) -> None:
        """Save training histories to JSON files."""
        output_dir = Path(output_dir) if output_dir else self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        for fold_id, history in self.fold_histories.items():
            output_path = output_dir / f"training_history_fold{fold_id}.json"
            with open(output_path, "w") as f:
                json.dump(_to_json_safe(history), f)

        logger.info(f"Training histories saved to {output_dir}")

    def save_history(self, fold_id: int, output_dir: Optional[Path] = None) -> Optional[Path]:
        """Save a single fold history to JSON and return the path."""
        if fold_id not in self.fold_histories:
            logger.warning(f"Invalid fold_id for save_history: {fold_id}")
            return None

        output_dir = Path(output_dir) if output_dir else self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"training_history_fold{fold_id}.json"
        with open(output_path, "w") as f:
            json.dump(_to_json_safe(self.fold_histories[fold_id]), f)

        logger.info(f"Training history saved to {output_path}")
        return output_path
