"""
Consolidated diagnostics for Neuro-CXG training and graph analysis.

Replaces the former separate modules:
  - training_diagnostics.py  (TrainingMonitor)
  - graph_topology.py        (CausalGraphAnalyzer)

External interface is unchanged — all previously exported names are re-exported:
    from src.analysis.diagnostics import TrainingMonitor, CausalGraphAnalyzer
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import mannwhitneyu
from tqdm import tqdm

from src.core.config import LOBE_NAMES, NUM_LOBES
from src.core.plotting import (
    ColorPalette,
    apply_professional_style,
)
from src.models.evaluation import _json_safe as _to_json_safe

palette = ColorPalette()

logger = logging.getLogger(__name__)

# ── TrainingMonitor ────────────────────────────────────────────────────────────


class TrainingMonitor:
    """
    Real-time and post-hoc training visualisation.

    Tracks per-epoch metrics across all CV folds and generates diagnostic plots:
      - Loss curves (detect overfitting/underfitting)
      - AUC progression (track learning quality)
      - Learning-rate schedule (verify warmup/cosine annealing)
      - Gradient norms (detect training instabilities)
      - Confusion-matrix evolution (detect class collapse)
    """

    def __init__(self, output_dir: Path, num_folds: int = 5):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_folds = num_folds

        self.fold_histories: dict[int, dict] = {
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
        logger.info(
            "TrainingMonitor initialised — output: %s, folds: %d", output_dir, num_folds
        )

    # ── logging ───────────────────────────────────────────────────────────────

    def log_epoch(
        self,
        fold_id: int,
        epoch: int,
        metrics: dict[str, float],
        grad_norm: float | None = None,
        confusion_matrix: np.ndarray | None = None,
    ) -> None:
        """Append one epoch's metrics to the fold history."""
        if fold_id not in self.fold_histories:
            raise ValueError(f"Invalid fold_id: {fold_id}")
        h = self.fold_histories[fold_id]
        h["train_loss"].append(metrics.get("train_loss", 0.0))
        h["val_loss"].append(metrics.get("val_loss", 0.0))
        h["val_auc"].append(metrics.get("val_auc", 0.0))
        h["val_f1"].append(metrics.get("val_f1", 0.0))
        h["val_acc"].append(metrics.get("val_acc", 0.0))
        h["learning_rate"].append(metrics.get("lr", 0.0))
        if grad_norm is not None:
            h["grad_norm"].append(grad_norm)
        if confusion_matrix is not None:
            h["confusion_matrices"].append(confusion_matrix.copy())

    # ── plotting ──────────────────────────────────────────────────────────────

    def plot_training_curves(
        self, fold_id: int, figsize: tuple[int, int] = (16, 12)
    ) -> Path | None:
        """4-panel training diagnostic: loss · AUC · LR schedule · gradient norm."""
        h = self.fold_histories[fold_id]
        if not h["train_loss"]:
            logger.warning("No training history for fold %d", fold_id)
            return None

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        plt.subplots_adjust(
            hspace=0.35, wspace=0.3, left=0.08, right=0.95, top=0.92, bottom=0.10
        )
        epochs = list(range(1, len(h["train_loss"]) + 1))

        # Loss
        ax = axes[0, 0]
        ax.plot(
            epochs,
            h["train_loss"],
            label="Train Loss",
            lw=2.5,
            color=palette.CONTROL,
            alpha=0.9,
        )
        ax.plot(
            epochs,
            h["val_loss"],
            label="Val Loss",
            lw=2.5,
            color=palette.ASD,
            alpha=0.9,
        )

        if len(epochs) > 4:
            train_smooth = np.convolve(h["train_loss"], np.ones(5) / 5, mode="valid")
            val_smooth = np.convolve(h["val_loss"], np.ones(5) / 5, mode="valid")
            ax.fill_between(
                epochs[: len(train_smooth)],
                train_smooth,
                alpha=0.15,
                color=palette.CONTROL,
            )
            ax.fill_between(
                epochs[: len(val_smooth)], val_smooth, alpha=0.15, color=palette.ASD
            )

        best_idx = int(np.argmin(h["val_loss"]))
        ax.scatter(
            [best_idx + 1],
            [h["val_loss"][best_idx]],
            color=palette.AMBER,
            s=250,
            zorder=5,
            marker="*",
            edgecolors="#333333",
            linewidths=0.5,
            label=f"Best Val (Epoch {best_idx + 1})",
        )
        ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
        ax.set_ylabel("Loss", fontsize=11, fontweight="bold")
        ax.set_title("Loss Curves", fontsize=13, fontweight="bold", pad=10)
        ax.legend(fontsize=9, framealpha=0.95, fancybox=True, loc="upper right")
        apply_professional_style(ax)
        ax.tick_params(axis="both", which="major", labelsize=10)

        # AUC
        ax = axes[0, 1]
        ax.plot(
            epochs,
            h["val_auc"],
            color=palette.GREEN,
            lw=2.5,
            alpha=0.9,
            label="Validation AUC",
        )

        if len(epochs) > 4:
            auc_smooth = np.convolve(h["val_auc"], np.ones(5) / 5, mode="valid")
            ax.fill_between(
                epochs[: len(auc_smooth)], auc_smooth, alpha=0.2, color=palette.GREEN
            )

        ax.axhline(0.5, color="#95a5a6", ls="--", alpha=0.7, lw=2, label="Random (0.5)")
        best_auc = max(h["val_auc"])
        best_auc_ep = h["val_auc"].index(best_auc) + 1
        ax.axhline(
            best_auc,
            color=palette.GREEN,
            ls="--",
            alpha=0.7,
            lw=2,
            label=f"Best: {best_auc:.3f} (Ep {best_auc_ep})",
        )
        ax.scatter(
            [best_auc_ep],
            [best_auc],
            color=palette.GREEN,
            s=250,
            zorder=5,
            marker="*",
            edgecolors="#333333",
            linewidths=0.5,
        )
        ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
        ax.set_ylabel("Validation AUC", fontsize=11, fontweight="bold")
        ax.set_title("AUC Progression", fontsize=13, fontweight="bold", pad=10)
        ax.legend(loc="lower right", fontsize=9, framealpha=0.95, fancybox=True)
        ax.grid(alpha=0.25, linestyle="-", linewidth=0.5)
        ax.set_ylim([0.4, 1.0])
        ax.tick_params(axis="both", which="major", labelsize=10)

        # Gradient Norm
        ax = axes[1, 0]
        if h["grad_norm"]:
            ax.plot(
                epochs,
                h["grad_norm"],
                color=palette.PINK,
                lw=2.5,
                alpha=0.9,
                label="Gradient Norm",
            )
            ax.axhline(
                1.0,
                color=palette.NEGATIVE,
                ls="--",
                lw=2,
                alpha=0.7,
                label="Clip Threshold (1.0)",
            )

            if len(epochs) > 4:
                grad_smooth = np.convolve(h["grad_norm"], np.ones(5) / 5, mode="valid")
                ax.fill_between(
                    epochs[: len(grad_smooth)],
                    grad_smooth,
                    alpha=0.2,
                    color=palette.PINK,
                )

            ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
            ax.set_ylabel("Gradient Norm", fontsize=11, fontweight="bold")
            ax.set_title("Gradient Stability", fontsize=13, fontweight="bold", pad=10)
            ax.legend(fontsize=9, framealpha=0.95, fancybox=True, loc="upper right")
            ax.grid(alpha=0.25, linestyle="-", linewidth=0.5)
            ax.set_ylim(bottom=0)
            ax.tick_params(axis="both", which="major", labelsize=10)
        else:
            ax.text(
                0.5,
                0.5,
                "Gradient norm not tracked",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
                color="#7f8c8d",
            )
            ax.axis("off")

        # Summary info panel
        ax = axes[1, 1]
        ax.axis("off")
        summary_text = (
            f"Training Summary - Fold {fold_id}\n"
            f"{'─' * 30}\n"
            f"Total Epochs: {len(epochs)}\n"
            f"Best Val Loss: {min(h['val_loss']):.4f} (Epoch {best_idx + 1})\n"
            f"Best Val AUC: {best_auc:.4f} (Epoch {best_auc_ep})\n"
            f"Final Train Loss: {h['train_loss'][-1]:.4f}\n"
            f"Final Val Loss: {h['val_loss'][-1]:.4f}\n"
            f"Final Val AUC: {h['val_auc'][-1]:.4f}"
        )
        ax.text(
            0.5,
            0.5,
            summary_text,
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
            fontfamily="monospace",
            bbox={
                "boxstyle": "round,pad=0.5",
                "facecolor": "#f8f9fa",
                "edgecolor": "#dee2e6",
                "alpha": 0.95,
            },
        )
        axes[1, 1].text(
            0.5,
            0.5,
            f"Epochs: {len(epochs)}\n"
            f"Final Train Loss: {h['train_loss'][-1]:.4f}\n"
            f"Final Val Loss: {h['val_loss'][-1]:.4f}\n"
            f"Best Val AUC: {best_auc:.4f}",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
            fontsize=12,
            bbox={
                "boxstyle": "round",
                "facecolor": "#f8f9fa",
                "edgecolor": "#dee2e6",
                "alpha": 0.9,
            },
        )

        plt.suptitle(
            f"Training Diagnostics — Fold {fold_id}",
            fontsize=18,
            fontweight="bold",
            y=0.995,
        )
        plt.tight_layout()

        out = self.output_dir / "fold_plots" / f"training_curves_fold_{fold_id}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        logger.info("Training curves saved → %s", out)
        return out

    def plot_confusion_evolution(
        self,
        fold_id: int,
        key_epochs: list[int] | None = None,
        figsize: tuple[int, int] = (16, 10),
    ) -> Path | None:
        """Visualise confusion matrix changes across epochs."""
        cm_history = self.fold_histories[fold_id]["confusion_matrices"]
        if not cm_history:
            logger.warning("No confusion matrices tracked for fold %d", fold_id)
            return None

        total = len(cm_history)
        if key_epochs is None:
            key_epochs = [min(e, total - 1) for e in [0, 9, 24, 49, 74, total - 1]]

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        plt.subplots_adjust(
            hspace=0.4, wspace=0.25, left=0.08, right=0.95, top=0.92, bottom=0.08
        )
        axes = axes.flatten()

        for idx, ep_idx in enumerate(key_epochs[:6]):
            ax = axes[idx]
            sns.heatmap(
                cm_history[ep_idx],
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax,
                xticklabels=["Control", "ASD"],
                yticklabels=["Control", "ASD"],
                annot_kws={"size": 11, "fontweight": "bold"},
                cbar_kws={"shrink": 0.6},
            )
            ax.set_xlabel("Predicted", fontsize=10, fontweight="bold")
            ax.set_ylabel("True Label", fontsize=10, fontweight="bold")
            ax.set_title(f"Epoch {ep_idx + 1}", fontsize=12, fontweight="bold", pad=8)
            ax.tick_params(axis="both", which="major", labelsize=9)

        fig.suptitle(
            f"Confusion Matrix Evolution - Fold {fold_id}",
            fontsize=15,
            fontweight="bold",
            y=0.98,
        )
        out = self.output_dir / f"confusion_evolution_fold_{fold_id}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        logger.info("Confusion evolution saved → %s", out)
        return out

    def plot_fold_comparison(self, figsize: tuple[int, int] = (14, 6)) -> Path | None:
        """Compare validation AUC curves across all folds."""
        if not any(h["val_auc"] for h in self.fold_histories.values()):
            logger.warning("No validation AUC data to compare")
            return None

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        plt.subplots_adjust(wspace=0.3, left=0.08, right=0.95, top=0.90, bottom=0.12)

        for fold_id, h in self.fold_histories.items():
            if h["val_auc"]:
                epochs = list(range(1, len(h["val_auc"]) + 1))
                axes[0].plot(
                    epochs,
                    h["val_auc"],
                    label=f"Fold {fold_id}",
                    color=palette.cycle()[fold_id % 8],
                    lw=2.5,
                    alpha=0.8,
                )

                if len(epochs) > 2:
                    auc_smooth = np.convolve(h["val_auc"], np.ones(3) / 3, mode="valid")
                    axes[0].fill_between(
                        epochs[: len(auc_smooth)],
                        auc_smooth,
                        alpha=0.1,
                        color=palette.cycle()[fold_id % 8],
                    )

        axes[0].set_xlabel("Epoch", fontsize=11, fontweight="bold")
        axes[0].set_ylabel("AUC", fontsize=11, fontweight="bold")
        axes[0].set_title(
            "Validation AUC by Fold", fontsize=13, fontweight="bold", pad=10
        )
        axes[0].legend(
            framealpha=0.95, fancybox=True, fontsize=9, ncol=5, loc="lower right"
        )
        axes[0].grid(alpha=0.25, linestyle="-", linewidth=0.5)
        axes[0].tick_params(axis="both", which="major", labelsize=10)
        apply_professional_style(axes[0])

        final = [h["val_auc"][-1] for h in self.fold_histories.values() if h["val_auc"]]
        if final:
            fold_ids = [i for i, h in self.fold_histories.items() if h["val_auc"]]
            bar_colors = [palette.cycle()[fid % 8] for fid in fold_ids]

            bars = axes[1].bar(
                range(len(final)),
                final,
                color=bar_colors,
                edgecolor="#333333",
                linewidth=1.2,
                alpha=0.85,
            )

            for _i, (bar, val) in enumerate(zip(bars, final, strict=False)):
                axes[1].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

            axes[1].axhline(
                y=np.mean(final),
                color=palette.GREEN,
                linestyle="--",
                lw=2,
                label=f"Mean: {np.mean(final):.3f}",
            )
            axes[1].set_xlabel("Fold", fontsize=11, fontweight="bold")
            axes[1].set_ylabel("AUC", fontsize=11, fontweight="bold")
            axes[1].set_title(
                "Final Validation AUC per Fold", fontsize=13, fontweight="bold", pad=10
            )
            axes[1].set_xticks(range(len(final)))
            axes[1].set_xticklabels([f"Fold {i}" for i in fold_ids], fontsize=10)
            axes[1].legend(framealpha=0.95, fancybox=True, fontsize=9)
            axes[1].grid(axis="y", alpha=0.25, linestyle="-", linewidth=0.5)
            axes[1].tick_params(axis="both", which="major", labelsize=10)
            apply_professional_style(axes[1])

            axes[1].set_ylim([min(0.4, min(final) - 0.05), max(1.0, max(final) + 0.05)])

        fig.suptitle(
            "Cross-Validation Fold Comparison", fontsize=15, fontweight="bold", y=0.98
        )

        plt.tight_layout()
        out = self.output_dir / "fold_comparison.png"
        plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        logger.info("Fold comparison saved → %s", out)
        return out

    # ── persistence ───────────────────────────────────────────────────────────

    def save_history(self, fold_id: int, output_dir: Path | None = None) -> Path | None:
        """Save a single fold history and return the file path."""
        if fold_id not in self.fold_histories:
            logger.warning("Invalid fold_id for save_history: %d", fold_id)
            return None
        out = Path(output_dir) if output_dir else self.output_dir
        out.mkdir(parents=True, exist_ok=True)
        p = out / f"training_history_fold{fold_id}.json"
        with open(p, "w") as f:
            json.dump(_to_json_safe(self.fold_histories[fold_id]), f)
        logger.info("Training history saved → %s", p)
        return p


# ── CausalGraphAnalyzer ────────────────────────────────────────────────────────


class CausalGraphAnalyzer:
    """
    Analyse structural properties of causal brain graphs.

    Computes network metrics and compares graph topology between ASD and Control.
    """

    def __init__(self, graphs_dir: Path, manifest: pd.DataFrame):
        self.graphs_dir = Path(graphs_dir)
        self.manifest = manifest.copy()
        self.manifest["subject_id"] = self.manifest["subject_id"].astype(str)
        self.lobe_names = [LOBE_NAMES[i] for i in range(NUM_LOBES)]
        logger.info(
            "CausalGraphAnalyzer initialised — dir: %s, subjects: %d",
            graphs_dir,
            len(manifest),
        )

    # ── graph metrics ─────────────────────────────────────────────────────────

    def compute_graph_properties(self, max_graphs: int | None = None) -> pd.DataFrame:
        """Compute standard graph metrics (degree, density, clustering) for each subject."""
        graph_files = list(self.graphs_dir.glob("*_graph.pt"))
        if max_graphs is not None and len(graph_files) > max_graphs:
            indices = np.random.choice(len(graph_files), max_graphs, replace=False)
            graph_files = [graph_files[i] for i in indices]
        logger.info("Computing properties for %d graphs…", len(graph_files))

        results = []
        for gf in tqdm(graph_files, desc="Graph properties"):
            try:
                data = torch.load(gf, weights_only=True)
                subject_id = gf.stem.replace("_graph", "")
                sub = self.manifest[self.manifest["subject_id"] == subject_id]
                if sub.empty:
                    continue
                dx_group = sub.iloc[0]["DX_GROUP"]
                site_id = sub.iloc[0]["SITE_ID"]
                adj = data["adj"].numpy()
                G = nx.DiGraph(adj)

                try:
                    avg_clust = nx.average_clustering(G.to_undirected())
                except Exception:
                    avg_clust = 0.0
                try:
                    betw = nx.betweenness_centrality(G)
                except Exception:
                    betw = dict.fromkeys(range(NUM_LOBES), 0.0)

                row: dict = {
                    "subject_id": subject_id,
                    "dx_group": dx_group,
                    "site_id": site_id,
                    "num_nodes": G.number_of_nodes(),
                    "num_edges": G.number_of_edges(),
                    "density": nx.density(G),
                    "avg_clustering": avg_clust,
                }
                in_deg = dict(G.in_degree())
                out_deg = dict(G.out_degree())
                for lobe_id, name in enumerate(self.lobe_names):
                    n = name.lower()
                    row[f"{n}_in_degree"] = in_deg.get(lobe_id, 0)
                    row[f"{n}_out_degree"] = out_deg.get(lobe_id, 0)
                    row[f"{n}_betweenness"] = betw.get(lobe_id, 0.0)
                results.append(row)
            except Exception as exc:
                logger.warning("Error processing %s: %s", gf.name, exc)

        df = pd.DataFrame(results)
        logger.info("Computed properties for %d graphs", len(df))
        return df

    # ── ASD vs Control comparison ─────────────────────────────────────────────

    def compare_asd_vs_control(
        self, graph_metrics: pd.DataFrame, output_dir: Path | None = None
    ) -> dict[str, dict]:
        """Mann-Whitney U comparison of graph metrics between ASD and Control."""
        logger.info("Comparing ASD vs Control graph topology…")
        asd = graph_metrics[graph_metrics["dx_group"] == 2]
        ctrl = graph_metrics[graph_metrics["dx_group"] == 1]
        logger.info("  ASD: %d, Control: %d", len(asd), len(ctrl))

        metrics = [
            "num_edges",
            "density",
            "avg_clustering",
            *[
                f"{n.lower()}_{s}"
                for n in self.lobe_names
                for s in ("in_degree", "out_degree")
            ],
        ]
        results: dict[str, dict] = {}

        print("\n" + "=" * 70)
        print("GRAPH TOPOLOGY COMPARISON (ASD vs CONTROL)")
        print("=" * 70)

        for metric in metrics:
            if metric not in graph_metrics.columns:
                continue
            a = asd[metric].dropna().values
            c = ctrl[metric].dropna().values
            if not len(a) or not len(c):
                continue
            u, p = mannwhitneyu(a, c, alternative="two-sided")
            pooled = np.sqrt((a.std() ** 2 + c.std() ** 2) / 2)
            d = (a.mean() - c.mean()) / pooled if pooled else 0.0
            results[metric] = {
                "asd_mean": a.mean(),
                "asd_std": a.std(),
                "control_mean": c.mean(),
                "control_std": c.std(),
                "u_statistic": u,
                "p_value": p,
                "cohens_d": d,
                "significant": p < 0.05,
            }
            print(f"\n{metric.replace('_', ' ').title()}:")
            print(f"  ASD:     {a.mean():.4f} ± {a.std():.4f}")
            print(f"  Control: {c.mean():.4f} ± {c.std():.4f}")
            print(f"  Mann-Whitney U={u:.2f}, p={p:.4f}, d={d:.3f}")
            if p < 0.05:
                direction = "higher" if a.mean() > c.mean() else "lower"
                effect = (
                    "large" if abs(d) > 0.8 else ("medium" if abs(d) > 0.5 else "small")
                )
                print(f"  ASD has significantly {direction} {metric} ({effect} effect)")
        print("=" * 70 + "\n")

        if output_dir is not None:
            self._plot_topology_comparison(graph_metrics, results, Path(output_dir))
        return results

    def _plot_topology_comparison(
        self, gm: pd.DataFrame, comparison: dict, output_dir: Path
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        gm = gm.copy()
        gm["dx_group"] = (
            gm["dx_group"].astype(int).map({2: "ASD", 1: "Control"}).fillna("Unknown")
        )
        sig_metrics = [k for k, v in comparison.items() if v["significant"]][:6]
        if not sig_metrics:
            sig_metrics = list(comparison.keys())[:6]
        palette = {"ASD": "#e74c3c", "Control": "#3498db"}
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        for ax, metric in zip(axes.flatten(), sig_metrics, strict=False):
            sns.boxplot(
                data=gm,
                x="dx_group",
                y=metric,
                ax=ax,
                palette=palette,
                hue="dx_group",
                legend=False,
            )
            ax.set(title=metric.replace("_", " ").title(), xlabel="Diagnosis")
        plt.tight_layout()
        out = output_dir / "topology_comparison.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Topology comparison saved → %s", out)

    # ── visualisation ─────────────────────────────────────────────────────────

    def visualize_average_causal_graph(
        self, output_path: Path, max_graphs: int | None = None, group: str | None = None
    ) -> Path | None:
        """Heatmap of the mean causal adjacency matrix across subjects.

        Args:
            output_path: Save location for the figure
            max_graphs: Maximum number of graphs to sample (None = all)
            group: If provided ('ASD' or 'Control'), filter to that diagnosis group
        """
        graph_files = list(self.graphs_dir.glob("*_graph.pt"))
        if not graph_files:
            logger.warning("No causal graphs found for average visualisation")
            return None

        # Filter by group if requested
        if group is not None and self.manifest is not None:
            # Convert group name to numeric DX_GROUP value
            if group == "ASD":
                dx_value = 1
            elif group == "Control":
                # Try 2 first, then 0
                if 2 in self.manifest["DX_GROUP"].values:
                    dx_value = 2
                else:
                    dx_value = 0
            else:
                # group is already a numeric DX_GROUP value
                dx_value = int(group)  # type: ignore[arg-type]

            group_subjects = set(
                self.manifest[self.manifest["DX_GROUP"] == dx_value][
                    "subject_id"
                ].astype(str)
            )
            graph_files = [
                gf
                for gf in graph_files
                if gf.stem.replace("_graph", "") in group_subjects
            ]
            if not graph_files:
                logger.warning(f"No causal graphs found for group={group}")
                return None

        if max_graphs is not None and len(graph_files) > max_graphs:
            indices = np.random.choice(len(graph_files), max_graphs, replace=False)
            graph_files = [graph_files[i] for i in indices]
        matrices = []
        for gf in graph_files:
            try:
                data = torch.load(gf, weights_only=True)
                if "adj" not in data:
                    continue
                matrices.append(data["adj"].detach().cpu().numpy())
            except Exception as exc:
                logger.warning("Failed to load %s: %s", gf.name, exc)
        if not matrices:
            logger.warning("No adjacency matrices loaded")
            return None

        avg = np.mean(np.stack(matrices), axis=0)
        labels = [LOBE_NAMES[i].replace("_", " ") for i in range(NUM_LOBES)]
        title_suffix = f" ({group})" if group else ""
        fig, ax = plt.subplots(figsize=(12, 10))

        vmax = max(abs(avg.min()), abs(avg.max()))
        sns.heatmap(
            avg,
            xticklabels=labels,
            yticklabels=labels,
            cmap="RdYlBu_r",
            center=0,
            linewidths=0.5,
            vmin=-vmax,
            vmax=vmax,
            ax=ax,
            cbar_kws={"label": "Causal Strength", "shrink": 0.8},
        )

        ax.set_xlabel("Target Region (To)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Source Region (From)", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Average Causal Adjacency Matrix{title_suffix}",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)

        fig.subplots_adjust(bottom=0.18, left=0.15, right=0.92, top=0.92)
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        logger.info("Average causal graph saved → %s", output_path)
        return output_path
