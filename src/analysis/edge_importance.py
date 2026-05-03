"""
Phase 8.2 — Edge Importance Analysis
======================================
Two complementary methods for scoring the importance of each causal connection:

1. **Gradient-based attribution** — gradient of the ASD logit w.r.t. ``edge_attr``
   (fast, differentiable, works in a single forward+backward pass).

2. **Edge-masking analysis** — systematically zero out one edge at a time and
   measure the drop in ASD probability (ΔP).  Computationally heavier but
   model-agnostic.

Both methods produce a signed score for each directed edge (i → j) and are
aggregated separately for ASD and Control subjects to identify connectivity
patterns that are differentially important for the two classes.

Classes
-------
GradientEdgeAttributor
    Gradient of the target class logit w.r.t. ``edge_attr``.

EdgeMaskingAnalyzer
    Importance via held-out edge masking (ΔP method).

EdgeImportanceAnalyzer
    Orchestrator: runs both methods, aggregates group-level 12×12 matrices,
    and saves heatmaps + chord diagram.

Usage
-----
    from src.analysis.edge_importance import EdgeImportanceAnalyzer
    analyzer = EdgeImportanceAnalyzer(model, test_loader, device)
    results  = analyzer.run(output_dir=Path("results/explainability"))
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import LOBE_NAMES, NUM_LOBES
from src.core.plotting import ColorPalette, apply_publication_style

palette = ColorPalette()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REGION_LABELS: List[str] = [LOBE_NAMES[i] for i in range(NUM_LOBES)]


# ── GradientEdgeAttributor ─────────────────────────────────────────────────────

class GradientEdgeAttributor:
    """
    Gradient of the target class logit w.r.t. ``edge_attr``.

    Because ``edge_attr`` flows through the edge-gate Linear layer and is
    consumed by each GATv2Conv message-passing step, its gradient measures
    how sensitively the classification outcome depends on each causal weight.

    Parameters
    ----------
    model : CausalBrainGNN
    target_class : int
        Class index to explain (1 = ASD).
    """

    def __init__(self, model: torch.nn.Module, target_class: int = 1):
        self.model = model
        self.target_class = target_class

    def compute(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute per-edge gradient scores for a single mini-batch.

        Returns
        -------
        grad_scores : Tensor, shape (num_edges,)
            Absolute gradient of ASD logit w.r.t. each edge weight.
            Higher = more influential causal connection.
        """
        self.model.eval()
        e_attr = edge_attr.clone().detach().requires_grad_(True)

        out = self.model(x, edge_index, e_attr, batch, **kwargs)
        target_logits = out[:, self.target_class].sum()
        target_logits.backward()

        grads = e_attr.grad                          # (E, 1) or (E,)
        if grads is None:
            logger.warning("GradientEdgeAttributor: edge_attr.grad is None")
            return torch.zeros(edge_attr.size(0))

        grads = grads.squeeze(-1).detach().cpu()     # (E,)
        return grads.abs()


# ── EdgeMaskingAnalyzer ───────────────────────────────────────────────────────

class EdgeMaskingAnalyzer:
    """
    Importance via leave-one-edge-out masking (ΔP method).

    For each edge i → j, zeroes out its weight and computes the change in the
    model's ASD probability relative to the unmasked prediction.

    Parameters
    ----------
    model : CausalBrainGNN
    target_class : int
        Class for ΔP computation (1 = ASD).
    max_graphs : int
        Number of graphs to process.  Masking is O(E) forward passes per graph
        so this is capped to avoid excessive runtime.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_class: int = 1,
        max_graphs: int = 50,
    ):
        self.model = model
        self.target_class = target_class
        self.max_graphs   = max_graphs

    def compute_for_graph(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        **kwargs,
    ) -> np.ndarray:
        """
        Run edge masking for a *single* graph (batch size = 1).

        Returns
        -------
        delta_p : np.ndarray, shape (num_edges,)
            Drop in ASD probability when each edge is masked out.
        """
        self.model.eval()
        device = x.device
        batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
        num_edges = edge_attr.size(0)

        with torch.no_grad():
            baseline_out = self.model(x, edge_index, edge_attr, batch, **kwargs)
            baseline_prob = F.softmax(baseline_out, dim=-1)[0, self.target_class].item()

        delta_p = np.zeros(num_edges)
        for e_idx in range(num_edges):
            masked_attr = edge_attr.clone()
            masked_attr[e_idx] = 0.0
            with torch.no_grad():
                out = self.model(x, edge_index, masked_attr, batch, **kwargs)
                prob = F.softmax(out, dim=-1)[0, self.target_class].item()
            delta_p[e_idx] = baseline_prob - prob   # positive = important

        return delta_p


# ── _build_matrix ─────────────────────────────────────────────────────────────

def _edge_scores_to_matrix(
    edge_index: np.ndarray,
    edge_scores: np.ndarray,
    num_nodes: int = NUM_LOBES,
) -> np.ndarray:
    """
    Accumulate per-edge scores into a (num_nodes, num_nodes) matrix.

    Parameters
    ----------
    edge_index : np.ndarray, shape (2, E)
    edge_scores : np.ndarray, shape (E,)

    Returns
    -------
    mat : np.ndarray, shape (num_nodes, num_nodes)
        Each cell [i, j] holds the mean score for the directed edge i→j
        (or 0 if that edge was absent).
    """
    mat    = np.zeros((num_nodes, num_nodes))
    counts = np.zeros((num_nodes, num_nodes))
    src, dst = edge_index
    for s, d, sc in zip(src, dst, edge_scores):
        si, di = int(s) % num_nodes, int(d) % num_nodes
        mat[si, di]    += sc
        counts[si, di] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        mat = np.where(counts > 0, mat / counts, 0.0)
    return mat


# ── EdgeImportanceAnalyzer ─────────────────────────────────────────────────────

class EdgeImportanceAnalyzer:
    """
    Orchestrates edge importance analysis over the full test set.

    Parameters
    ----------
    model : CausalBrainGNN
    test_loader : DataLoader
    device : torch.device
    masking_max_graphs : int
        Max number of graphs to analyse with the (slow) masking method.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        test_loader,
        device: torch.device,
        masking_max_graphs: int = 40,
    ):
        self.model  = model
        self.test_loader = test_loader
        self.device = device
        self.masking_max_graphs = masking_max_graphs

    # ── public API ─────────────────────────────────────────────────────────────

    def run(self, output_dir: Path) -> Dict:
        """
        Execute both edge analysis methods, save figures.

        Returns
        -------
        dict with keys: ``gradient``, ``masking``
            Each contains ``asd_matrix`` and ``control_matrix`` of shape
            ``(NUM_LOBES, NUM_LOBES)``.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("EdgeImportanceAnalyzer: starting analysis → %s", output_dir)

        gradient_results = self._run_gradient_attribution()
        masking_results  = self._run_edge_masking()

        self._plot_edge_matrix(
            gradient_results["asd_matrix"],
            gradient_results["control_matrix"],
            title="Edge Importance – Gradient Attribution",
            save_path=output_dir / "edge_importance_gradient.png",
        )
        self._plot_edge_matrix(
            masking_results["asd_matrix"],
            masking_results["control_matrix"],
            title="Edge Importance – Masking (ΔP)",
            save_path=output_dir / "edge_importance_masking.png",
        )
        self._plot_differential_connectivity(
            gradient_results, output_dir / "edge_differential_connectivity.png"
        )

        return {"gradient": gradient_results, "masking": masking_results}

    # ── gradient attribution ───────────────────────────────────────────────────

    def _run_gradient_attribution(self) -> Dict[str, np.ndarray]:
        attributor = GradientEdgeAttributor(self.model, target_class=1)
        asd_mats:  List[np.ndarray] = []
        ctrl_mats: List[np.ndarray] = []

        for batch in self.test_loader:
            if batch is None:
                continue
            batch = batch.to(self.device)
            labels      = batch.y.cpu().numpy()
            batch_np    = batch.batch.cpu().numpy()
            edge_idx_np = batch.edge_index.cpu().numpy()

            with torch.enable_grad():
                scores = attributor.compute(
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.batch,
                    site_id=None,
                    age=batch.age if hasattr(batch, "age") else None,
                    sex=batch.sex if hasattr(batch, "sex") else None,
                    fiq=batch.fiq if hasattr(batch, "fiq") else None,
                ).numpy()

            for g_idx, label in enumerate(labels):
                edge_mask = batch_np[edge_idx_np[1]] == g_idx
                g_scores  = scores[edge_mask]
                g_edges   = edge_idx_np[:, edge_mask]
                mat = _edge_scores_to_matrix(g_edges, g_scores)
                (asd_mats if label == 1 else ctrl_mats).append(mat)

        results = {}
        if asd_mats:
            results["asd_matrix"]  = np.mean(np.stack(asd_mats),  axis=0)
        else:
            results["asd_matrix"]  = np.zeros((NUM_LOBES, NUM_LOBES))
        if ctrl_mats:
            results["control_matrix"] = np.mean(np.stack(ctrl_mats), axis=0)
        else:
            results["control_matrix"] = np.zeros((NUM_LOBES, NUM_LOBES))

        logger.info(
            "Gradient attribution: %d ASD, %d Control graphs",
            len(asd_mats), len(ctrl_mats),
        )
        return results

    # ── edge masking ───────────────────────────────────────────────────────────

    def _run_edge_masking(self) -> Dict[str, np.ndarray]:
        masker = EdgeMaskingAnalyzer(
            self.model,
            target_class=1,
            max_graphs=self.masking_max_graphs,
        )
        asd_mats:  List[np.ndarray] = []
        ctrl_mats: List[np.ndarray] = []
        processed = 0

        for batch in self.test_loader:
            if batch is None or processed >= self.masking_max_graphs:
                break
            batch = batch.to(self.device)
            labels      = batch.y.cpu().numpy()
            batch_np    = batch.batch.cpu().numpy()
            edge_idx_np = batch.edge_index.cpu().numpy()

            for g_idx, label in enumerate(labels):
                if processed >= self.masking_max_graphs:
                    break
                node_mask = batch_np == g_idx
                edge_mask = batch_np[edge_idx_np[1]] == g_idx

                g_x         = batch.x[node_mask]
                g_edge_attr = batch.edge_attr[edge_mask]
                g_edge_idx  = edge_idx_np[:, edge_mask] - edge_idx_np[:, edge_mask].min()
                g_edge_idx_t = torch.tensor(g_edge_idx, dtype=torch.long, device=self.device)

                if g_edge_attr.size(0) == 0:
                    continue

                try:
                    delta_p = masker.compute_for_graph(
                        g_x,
                        g_edge_idx_t,
                        g_edge_attr,
                        site_id=None,
                        age=batch.age[g_idx:g_idx+1] if hasattr(batch, "age") else None,
                        sex=batch.sex[g_idx:g_idx+1] if hasattr(batch, "sex") else None,
                        fiq=batch.fiq[g_idx:g_idx+1] if hasattr(batch, "fiq") else None,
                    )
                    mat = _edge_scores_to_matrix(g_edge_idx, delta_p)
                    (asd_mats if label == 1 else ctrl_mats).append(mat)
                    processed += 1
                except Exception as exc:
                    logger.warning("Masking failed for graph %d: %s", g_idx, exc)

        results = {}
        results["asd_matrix"]     = np.mean(np.stack(asd_mats),  axis=0) if asd_mats  else np.zeros((NUM_LOBES, NUM_LOBES))
        results["control_matrix"] = np.mean(np.stack(ctrl_mats), axis=0) if ctrl_mats else np.zeros((NUM_LOBES, NUM_LOBES))

        logger.info(
            "Edge masking: %d ASD, %d Control graphs (capped at %d)",
            len(asd_mats), len(ctrl_mats), self.masking_max_graphs,
        )
        return results

    # ── Plotting ───────────────────────────────────────────────────────────────

    def _plot_edge_matrix(
        self,
        asd_mat: np.ndarray,
        ctrl_mat: np.ndarray,
        title: str,
        save_path: Path,
    ) -> None:
        import seaborn as sns

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        for ax, mat, group in zip(axes, [ctrl_mat, asd_mat], ["Control", "ASD"]):
            sns.heatmap(
                mat,
                xticklabels=REGION_LABELS,
                yticklabels=REGION_LABELS,
                cmap="YlOrRd",
                ax=ax,
                linewidths=0.3,
                linecolor="white",
                vmin=0,
            )
            ax.set_title(f"{group}", fontsize=13, fontweight="bold")
            ax.set_xlabel("Target Region", fontsize=10)
            ax.set_ylabel("Source Region", fontsize=10)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
            plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)

        fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Edge matrix plot saved → %s", save_path)

    def _plot_differential_connectivity(
        self, gradient_results: Dict, save_path: Path
    ) -> None:
        """Plot ASD - Control difference matrix (signed) for gradient attribution."""
        import seaborn as sns

        diff = gradient_results["asd_matrix"] - gradient_results["control_matrix"]
        max_abs = np.abs(diff).max() or 1.0

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            diff,
            xticklabels=REGION_LABELS,
            yticklabels=REGION_LABELS,
            cmap="RdBu_r",
            center=0,
            vmin=-max_abs,
            vmax=max_abs,
            ax=ax,
            linewidths=0.3,
            linecolor="white",
        )
        ax.set_title(
            "Differential Edge Importance: ASD − Control\n(positive = more important in ASD)",
            fontsize=13, fontweight="bold",
        )
        ax.set_xlabel("Target Region", fontsize=10)
        ax.set_ylabel("Source Region", fontsize=10)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)

        # Annotate top-5 differentially important edges
        flat_idx = np.argsort(np.abs(diff), axis=None)[::-1][:5]
        for rank, flat_i in enumerate(flat_idx):
            row, col = np.unravel_index(flat_i, diff.shape)
            d_val = diff[row, col]
            logger.info(
                "Top edge %d: %s → %s  Δ=%.4f",
                rank + 1, REGION_LABELS[row], REGION_LABELS[col], d_val,
            )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Differential connectivity plot saved → %s", save_path)
