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
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import mannwhitneyu
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import LOBE_NAMES, NUM_LOBES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_json_safe(obj):
    """Recursively convert NumPy scalars/arrays to plain Python for JSON serialisation."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_safe(v) for v in obj]
    return obj


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

        self.fold_histories: Dict[int, Dict] = {
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
        logger.info("TrainingMonitor initialised — output: %s, folds: %d", output_dir, num_folds)

    # ── logging ───────────────────────────────────────────────────────────────

    def log_epoch(
        self,
        fold_id: int,
        epoch: int,
        metrics: Dict[str, float],
        grad_norm: Optional[float] = None,
        confusion_matrix: Optional[np.ndarray] = None,
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
        self, fold_id: int, figsize: Tuple[int, int] = (18, 12)
    ) -> Optional[Path]:
        """4-panel training diagnostic: loss · AUC · LR schedule · gradient norm."""
        h = self.fold_histories[fold_id]
        if not h["train_loss"]:
            logger.warning("No training history for fold %d", fold_id)
            return None

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        epochs = range(1, len(h["train_loss"]) + 1)

        # Loss
        ax = axes[0, 0]
        ax.plot(epochs, h["train_loss"], label="Train Loss", lw=2.5, color="#3498db", alpha=0.8)
        ax.plot(epochs, h["val_loss"], label="Val Loss", lw=2.5, color="#e74c3c", alpha=0.8)
        best_idx = int(np.argmin(h["val_loss"]))
        ax.scatter([best_idx + 1], [h["val_loss"][best_idx]], color="#e74c3c", s=200, zorder=5,
                   marker="*", label=f"Best Val (Epoch {best_idx + 1})")
        ax.set(xlabel="Epoch", ylabel="Loss", title="Loss Curves")
        ax.legend(fontsize=10); ax.grid(alpha=0.3, ls="--"); ax.set_ylim(bottom=0)

        # AUC
        ax = axes[0, 1]
        ax.plot(epochs, h["val_auc"], color="#2ecc71", lw=2.5, alpha=0.8, label="Validation AUC")
        ax.axhline(0.5, color="#95a5a6", ls="--", alpha=0.7, lw=2, label="Random (0.5)")
        best_auc = max(h["val_auc"]); best_auc_ep = h["val_auc"].index(best_auc) + 1
        ax.axhline(best_auc, color="#27ae60", ls="--", alpha=0.7, lw=2,
                   label=f"Best: {best_auc:.4f} (Epoch {best_auc_ep})")
        ax.scatter([best_auc_ep], [best_auc], color="#27ae60", s=200, zorder=5, marker="*")
        ax.set(xlabel="Epoch", ylabel="Validation AUC", title="AUC Progression")
        ax.legend(loc="lower right", fontsize=10); ax.grid(alpha=0.3, ls="--"); ax.set_ylim([0.4, 1.0])

        # LR schedule
        ax = axes[1, 0]
        ax.plot(epochs, h["learning_rate"], color="#f39c12", lw=2.5, alpha=0.8)
        ax.set(xlabel="Epoch", ylabel="Learning Rate", title="LR Schedule (Warmup + Cosine)")
        ax.set_yscale("log"); ax.grid(alpha=0.3, ls="--")

        # Gradient norms
        ax = axes[1, 1]
        if h["grad_norm"]:
            ax.plot(epochs, h["grad_norm"], color="#9b59b6", lw=2.5, alpha=0.8, label="Gradient Norm")
            ax.axhline(1.0, color="#e74c3c", ls="--", lw=2, alpha=0.7, label="Clip Threshold (1.0)")
            ax.set(xlabel="Epoch", ylabel="Gradient Norm", title="Gradient Stability")
            ax.legend(fontsize=10); ax.grid(alpha=0.3, ls="--"); ax.set_ylim(bottom=0)
        else:
            ax.text(0.5, 0.5, "Gradient norm not tracked", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14, color="#7f8c8d")
            ax.axis("off")

        plt.suptitle(f"Training Diagnostics — Fold {fold_id}", fontsize=18, fontweight="bold", y=0.995)
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
        key_epochs: Optional[List[int]] = None,
        figsize: Tuple[int, int] = (18, 12),
    ) -> Optional[Path]:
        """Visualise confusion matrix changes across epochs."""
        cm_history = self.fold_histories[fold_id]["confusion_matrices"]
        if not cm_history:
            logger.warning("No confusion matrices tracked for fold %d", fold_id)
            return None

        total = len(cm_history)
        if key_epochs is None:
            key_epochs = [min(e, total - 1) for e in [0, 9, 24, 49, 74, total - 1]]

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        for idx, ep_idx in enumerate(key_epochs[:6]):
            ax = axes[idx]
            sns.heatmap(cm_history[ep_idx], annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["Control", "ASD"], yticklabels=["Control", "ASD"])
            ax.set(title=f"Epoch {ep_idx + 1}", xlabel="Predicted", ylabel="True")

        plt.tight_layout()
        out = self.output_dir / f"confusion_evolution_fold_{fold_id}.png"
        plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
        logger.info("Confusion evolution saved → %s", out)
        return out

    def plot_fold_comparison(self, figsize: Tuple[int, int] = (14, 8)) -> Optional[Path]:
        """Compare validation AUC curves across all folds."""
        if not any(h["val_auc"] for h in self.fold_histories.values()):
            logger.warning("No validation AUC data to compare")
            return None

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        for fold_id, h in self.fold_histories.items():
            if h["val_auc"]:
                axes[0].plot(range(1, len(h["val_auc"]) + 1), h["val_auc"], label=f"Fold {fold_id}")
        axes[0].set(title="Validation AUC by Fold", xlabel="Epoch", ylabel="AUC")
        axes[0].legend(); axes[0].grid(alpha=0.3)

        final = [h["val_auc"][-1] for h in self.fold_histories.values() if h["val_auc"]]
        if final:
            axes[1].bar(range(len(final)), final, color="#3498db")
            axes[1].set(title="Final Validation AUC per Fold", xlabel="Fold", ylabel="AUC")
            axes[1].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        out = self.output_dir / "fold_comparison.png"
        plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
        logger.info("Fold comparison saved → %s", out)
        return out

    # ── persistence ───────────────────────────────────────────────────────────

    def save_histories(self, output_dir: Optional[Path] = None) -> None:
        """Save all fold histories to JSON files."""
        out = Path(output_dir) if output_dir else self.output_dir
        out.mkdir(parents=True, exist_ok=True)
        for fold_id, h in self.fold_histories.items():
            p = out / f"training_history_fold{fold_id}.json"
            with open(p, "w") as f:
                json.dump(_to_json_safe(h), f)
        logger.info("Training histories saved → %s", out)

    def save_history(self, fold_id: int, output_dir: Optional[Path] = None) -> Optional[Path]:
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
        logger.info("CausalGraphAnalyzer initialised — dir: %s, subjects: %d",
                    graphs_dir, len(manifest))

    # ── graph metrics ─────────────────────────────────────────────────────────

    def compute_graph_properties(self, max_graphs: Optional[int] = None) -> pd.DataFrame:
        """Compute standard graph metrics (degree, density, clustering) for each subject."""
        graph_files = list(self.graphs_dir.glob("*_graph.pt"))
        if max_graphs is not None:
            graph_files = list(
                np.random.choice(graph_files, min(max_graphs, len(graph_files)), replace=False)
            )
        logger.info("Computing properties for %d graphs…", len(graph_files))

        results = []
        for gf in tqdm(graph_files, desc="Graph properties"):
            try:
                data = torch.load(gf, weights_only=False)
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
                    betw = {i: 0.0 for i in range(NUM_LOBES)}

                row: Dict = {
                    "subject_id": subject_id,
                    "dx_group": dx_group,
                    "site_id": site_id,
                    "num_nodes": G.number_of_nodes(),
                    "num_edges": G.number_of_edges(),
                    "density": nx.density(G),
                    "avg_clustering": avg_clust,
                }
                in_deg = dict(G.in_degree()); out_deg = dict(G.out_degree())
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
        self, graph_metrics: pd.DataFrame, output_dir: Optional[Path] = None
    ) -> Dict[str, Dict]:
        """Mann-Whitney U comparison of graph metrics between ASD and Control."""
        logger.info("Comparing ASD vs Control graph topology…")
        asd = graph_metrics[graph_metrics["dx_group"] == 1]
        ctrl = graph_metrics[graph_metrics["dx_group"] == 2]
        logger.info("  ASD: %d, Control: %d", len(asd), len(ctrl))

        metrics = [
            "num_edges", "density", "avg_clustering",
            *[f"{n.lower()}_{s}" for n in self.lobe_names for s in ("in_degree", "out_degree")],
        ]
        results: Dict[str, Dict] = {}

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
            results[metric] = dict(asd_mean=a.mean(), asd_std=a.std(),
                                   control_mean=c.mean(), control_std=c.std(),
                                   u_statistic=u, p_value=p, cohens_d=d,
                                   significant=p < 0.05)
            print(f"\n{metric.replace('_', ' ').title()}:")
            print(f"  ASD:     {a.mean():.4f} ± {a.std():.4f}")
            print(f"  Control: {c.mean():.4f} ± {c.std():.4f}")
            print(f"  Mann-Whitney U={u:.2f}, p={p:.4f}, d={d:.3f}")
            if p < 0.05:
                direction = "higher" if a.mean() > c.mean() else "lower"
                effect = "large" if abs(d) > 0.8 else ("medium" if abs(d) > 0.5 else "small")
                print(f"  ASD has significantly {direction} {metric} ({effect} effect)")
        print("=" * 70 + "\n")

        if output_dir is not None:
            self._plot_topology_comparison(graph_metrics, results, Path(output_dir))
        return results

    def _plot_topology_comparison(
        self, gm: pd.DataFrame, comparison: Dict, output_dir: Path
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        gm = gm.copy()
        gm["dx_group"] = (
            gm["dx_group"].astype(int).map({1: "ASD", 2: "Control"}).fillna("Unknown")
        )
        sig_metrics = [k for k, v in comparison.items() if v["significant"]][:6]
        if not sig_metrics:
            sig_metrics = list(comparison.keys())[:6]
        palette = {"ASD": "#e74c3c", "Control": "#3498db"}
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        for ax, metric in zip(axes.flatten(), sig_metrics):
            sns.boxplot(data=gm, x="dx_group", y=metric, ax=ax, palette=palette)
            ax.set(title=metric.replace("_", " ").title(), xlabel="Diagnosis")
        plt.tight_layout()
        out = output_dir / "topology_comparison.png"
        plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
        logger.info("Topology comparison saved → %s", out)

    # ── visualisation ─────────────────────────────────────────────────────────

    def visualize_average_causal_graph(
        self, output_path: Path, max_graphs: Optional[int] = None
    ) -> Optional[Path]:
        """Heatmap of the mean causal adjacency matrix across all subjects."""
        graph_files = list(self.graphs_dir.glob("*_graph.pt"))
        if not graph_files:
            logger.warning("No causal graphs found for average visualisation")
            return None
        if max_graphs is not None:
            graph_files = list(
                np.random.choice(graph_files, min(max_graphs, len(graph_files)), replace=False)
            )
        matrices = []
        for gf in graph_files:
            try:
                data = torch.load(gf, weights_only=False)
                if "adj" not in data:
                    continue
                matrices.append(data["adj"].detach().cpu().numpy())
            except Exception as exc:
                logger.warning("Failed to load %s: %s", gf.name, exc)
        if not matrices:
            logger.warning("No adjacency matrices loaded")
            return None

        avg = np.mean(np.stack(matrices), axis=0)
        labels = [LOBE_NAMES[i] for i in range(NUM_LOBES)]
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(avg, xticklabels=labels, yticklabels=labels, cmap="RdYlBu_r",
                    center=0, linewidths=0.5, ax=ax)
        ax.set(title="Average Causal Adjacency Matrix",
               xlabel="Target Region", ylabel="Source Region")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close()
        logger.info("Average causal graph saved → %s", out)
        return out
