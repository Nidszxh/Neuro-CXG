"""
Phase 8.1 — Node Importance Analysis
=====================================
Implements GradCAM-style node attribution and GAT attention-weight extraction
for the CausalBrainGNN model.

Classes
-------
AttentionWeightExtractor
    Extracts per-edge attention coefficients from every GATv2Conv layer via
    forward hooks.

GradCAMGraphExplainer
    Computes GradCAM node-importance scores by weighting each layer's output
    activations with the mean gradient of the target class logit.

NodeImportanceAnalyzer
    Runs both methods on an entire data-loader, aggregates results by
    diagnosis class, and saves publication-ready figures.

Usage
-----
    from src.analysis.node_importance import NodeImportanceAnalyzer
    analyzer = NodeImportanceAnalyzer(model, test_loader, device)
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
from src.core.config import LOBE_NAMES, NUM_LOBES, GNN_IN_CHANNELS
from src.core.atlas_config import LOBE_TO_NETWORK, NETWORK_TO_LOBES, NUM_NETWORKS, NETWORK_NAMES
from src.core.plotting import ColorPalette, FigureSize, apply_publication_style

palette = ColorPalette()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REGION_LABELS: List[str] = [LOBE_NAMES[i] for i in range(NUM_LOBES)]
NETWORK_LABELS: List[str] = [NETWORK_NAMES[i] for i in range(NUM_NETWORKS)]


def _aggregate_to_networks(lobe_scores: np.ndarray) -> np.ndarray:
    """
    Aggregate per-lobe importance scores to the network level.

    Uses the LOBE_TO_NETWORK mapping from atlas_config to average lobe
    scores within each of the 4 functional networks (Task 3 — DD-011).

    Args:
        lobe_scores: (NUM_LOBES,) array of per-lobe importance scores.

    Returns:
        (NUM_NETWORKS,) array of per-network importance scores.
    """
    network_scores = np.zeros(NUM_NETWORKS)
    for net_idx, lobe_list in NETWORK_TO_LOBES.items():
        valid_lobes = [idx for idx in lobe_list if idx < NUM_LOBES]
        if valid_lobes:
            network_scores[net_idx] = np.mean(lobe_scores[valid_lobes])
    return network_scores


# ── AttentionWeightExtractor ───────────────────────────────────────────────────

class AttentionWeightExtractor:
    """
    Extracts per-edge attention coefficients produced by every GATv2Conv layer.

    GATv2Conv stores attention weights in the ``_alpha`` attribute after each
    forward call.  This extractor reads those values after a forward pass so
    they can be inspected without modifying the model code.

    Parameters
    ----------
    model : CausalBrainGNN
        The trained GNN model.

    Example
    -------
    >>> extractor = AttentionWeightExtractor(model)
    >>> with torch.no_grad():
    ...     _ = model(x, edge_index, edge_attr, batch)
    >>> attn = extractor.get_attention_weights()   # {layer_idx: Tensor(E, heads)}
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._hooks: List = []
        self._attention: Dict[int, torch.Tensor] = {}
        self._register_hooks()

    def _register_hooks(self) -> None:
        layer_idx = 0
        for name in ("conv1", "conv2", "conv3"):
            if hasattr(self.model, name):
                conv = getattr(self.model, name)
                idx = layer_idx

                def make_hook(i):
                    def hook(_module, _inputs, _outputs):
                        # GATv2Conv stores alpha in ._alpha after forward
                        alpha = getattr(_module, "_alpha", None)
                        if alpha is not None:
                            self._attention[i] = alpha.detach().cpu()
                    return hook

                self._hooks.append(conv.register_forward_hook(make_hook(idx)))
                layer_idx += 1
        logger.debug("AttentionWeightExtractor registered %d hooks", layer_idx)

    def get_attention_weights(self) -> Dict[int, torch.Tensor]:
        """Return the latest captured attention weights keyed by layer index."""
        return dict(self._attention)

    def clear(self) -> None:
        """Clear stored attention tensors (free memory)."""
        self._attention.clear()

    def remove_hooks(self) -> None:
        """Deregister all hooks from the model."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ── GradCAMGraphExplainer ──────────────────────────────────────────────────────

class GradCAMGraphExplainer:
    """
    GradCAM-style node importance for graph neural networks.

    For each forward pass the explainer:
      1. Captures the post-GAT node embeddings at every layer via forward hooks.
      2. Computes the gradient of the target class logit w.r.t. those embeddings.
      3. Computes per-channel importance weights as the mean-over-nodes gradient.
      4. Aggregates channel-wise weighted activations → one importance score per node.

    Parameters
    ----------
    model : CausalBrainGNN
        Trained model.  Must be in ``eval`` mode and have ``requires_grad=True``
        on all parameters.
    target_class : int
        Class index for gradient computation (1 = ASD).
    """

    def __init__(self, model: torch.nn.Module, target_class: int = 1):
        self.model = model
        self.target_class = target_class
        self._activations: Dict[str, torch.Tensor] = {}
        self._gradients: Dict[str, torch.Tensor] = {}
        self._hooks: List = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        for name in ("conv1", "conv2", "conv3"):
            if hasattr(self.model, name):
                conv = getattr(self.model, name)
                key = name

                def make_fwd(k):
                    def fwd_hook(_module, _input, output):
                        # output is the node embedding tensor (N, H*heads or N, H)
                        if isinstance(output, tuple):
                            output = output[0]
                        self._activations[k] = output

                    return fwd_hook

                def make_bwd(k):
                    def bwd_hook(_module, _grad_input, grad_output):
                        if isinstance(grad_output, tuple):
                            grad_output = grad_output[0]
                        if grad_output is not None:
                            self._gradients[k] = grad_output.detach()

                    return bwd_hook

                self._hooks.append(conv.register_forward_hook(make_fwd(key)))
                self._hooks.append(conv.register_full_backward_hook(make_bwd(key)))

    def compute(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Run one forward+backward pass and return per-node importance scores.

        Parameters
        ----------
        x, edge_index, edge_attr, batch : Tensors
            Standard PyTorch Geometric inputs.

        Returns
        -------
        node_scores : Tensor, shape (num_nodes,)
            GradCAM importance score for every node in the batch.
        """
        self._activations.clear()
        self._gradients.clear()
        self.model.zero_grad()

        # Forward
        out = self.model(x, edge_index, edge_attr, batch, **kwargs)
        #  Use logit of target class as scalar for backprop
        target_logits = out[:, self.target_class].sum()
        target_logits.backward()

        node_scores = torch.zeros(x.size(0), device="cpu")
        layer_count = 0
        for key in ("conv1", "conv2", "conv3"):
            if key not in self._activations or key not in self._gradients:
                continue
            act = self._activations[key].detach().cpu()   # (N, C)
            grad = self._gradients[key].detach().cpu()    # (N, C)
            # Per-channel importance weights (global average pooling over nodes)
            weights = grad.mean(dim=0, keepdim=True)      # (1, C)
            cam = (weights * act).sum(dim=1)              # (N,)
            cam = F.relu(cam)                             # Retain positive influence
            node_scores = node_scores + cam
            layer_count += 1

        if layer_count > 0:
            node_scores = node_scores / layer_count       # average across layers

        return node_scores

    def remove_hooks(self) -> None:
        """Deregister all hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ── NodeImportanceAnalyzer ─────────────────────────────────────────────────────

class NodeImportanceAnalyzer:
    """
    Orchestrates node importance analysis over an entire test set.

    Runs both GradCAM and attention-weight extraction, then aggregates
    results separately for ASD and Control subjects.

    Parameters
    ----------
    model : CausalBrainGNN
        Trained model in eval mode.
    test_loader : DataLoader
        Loader for the held-out test split.
    device : torch.device
        CPU or CUDA device.
    """

    def __init__(self, model: torch.nn.Module, test_loader, device: torch.device):
        self.model = model
        self.test_loader = test_loader
        self.device = device

    # ── public API ─────────────────────────────────────────────────────────────

    def run(self, output_dir: Path) -> Dict:
        """
        Execute full node importance analysis and save figures.

        Returns a dict with aggregated results::

            {
                "gradcam": {
                    "asd_mean":     np.ndarray (NUM_LOBES,),
                    "control_mean": np.ndarray (NUM_LOBES,),
                    "diff":         np.ndarray (NUM_LOBES,),   # ASD - Control
                },
                "attention": {
                    "layer_{i}": {"asd_mean": ..., "control_mean": ...}
                },
            }
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("NodeImportanceAnalyzer: starting analysis → %s", output_dir)

        gradcam_results = self._run_gradcam()
        attn_results    = self._run_attention_extraction()

        # Plot GradCAM
        self._plot_gradcam(gradcam_results, output_dir / "node_importance_gradcam.png")

        # Plot attention weights per layer
        self._plot_attention(attn_results, output_dir / "attention_weights_by_layer.png")

        # Compute and plot ASD-Control difference
        asd_scores  = gradcam_results.get("asd_mean",     np.zeros(NUM_LOBES))
        ctrl_scores = gradcam_results.get("control_mean", np.zeros(NUM_LOBES))
        diff = asd_scores - ctrl_scores
        top_regions = np.argsort(np.abs(diff))[::-1]

        max_abs_diff = float(np.abs(diff).max())
        if max_abs_diff < 1e-4:
            logger.warning(
                "All GradCAM differential scores are near-zero (max |Δ|=%.2e). "
                "Model is likely predicting one class for most/all inputs. "
                "Interpretability output is not reliable.",
                max_abs_diff,
            )

        logger.info("Top differentially-important regions (ASD - Control):")
        for rank, idx in enumerate(top_regions[:5]):
            logger.info(
                "  %d. %s  Δ=%.4f", rank + 1, REGION_LABELS[idx], diff[idx]
            )

        self._plot_diff_bar(diff, output_dir / "node_importance_asd_vs_control.png")

        combined = {
            "gradcam":   {**gradcam_results, "diff": diff},
            "attention": attn_results,
        }

        # Task 3: Network-level attribution (DD-011)
        asd_net  = _aggregate_to_networks(asd_scores)
        ctrl_net = _aggregate_to_networks(ctrl_scores)
        net_diff = asd_net - ctrl_net
        combined["gradcam"]["asd_network_mean"]     = asd_net
        combined["gradcam"]["control_network_mean"] = ctrl_net
        combined["gradcam"]["network_diff"]          = net_diff

        logger.info("Network-level GradCAM (ASD - Control):")
        for ni in range(NUM_NETWORKS):
            logger.info("  %s: Δ=%.4f (ASD=%.4f, Ctrl=%.4f)",
                        NETWORK_LABELS[ni], net_diff[ni], asd_net[ni], ctrl_net[ni])

        self._plot_network_diff(net_diff, asd_net, ctrl_net,
                                output_dir / "node_importance_network_level.png")
        return combined

    # ── GradCAM pass ───────────────────────────────────────────────────────────

    def _run_gradcam(self) -> Dict[str, np.ndarray]:
        """Collect GradCAM node scores across the test set."""
        explainer = GradCAMGraphExplainer(self.model, target_class=1)
        self.model.eval()

        asd_scores:     List[np.ndarray] = []
        control_scores: List[np.ndarray] = []

        for batch in self.test_loader:
            if batch is None:
                continue
            batch = batch.to(self.device)
            labels = batch.y.cpu().numpy()

            with torch.enable_grad():
                node_scores = explainer.compute(
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.batch,
                    site_id=None,          # attribution mode: no site embedding
                    age=batch.age  if hasattr(batch, "age")  else None,
                    sex=batch.sex  if hasattr(batch, "sex")  else None,
                    fiq=batch.fiq  if hasattr(batch, "fiq")  else None,
                )

            node_scores = node_scores.numpy()
            # Split scores back into individual graphs
            batch_numpy = batch.batch.cpu().numpy()
            for graph_idx, label in enumerate(labels):
                mask = (batch_numpy == graph_idx)
                g_scores = node_scores[mask]               # (num_lobes,)
                if len(g_scores) != NUM_LOBES:
                    continue
                # Zero out attributions for atlas-gap / zero-signal nodes so that
                # uninformative zero-padded lobes don't distort importance rankings.
                if hasattr(batch, 'zero_lobe_mask'):
                    zlm = batch.zero_lobe_mask[mask].cpu().numpy().astype(bool)
                    if zlm.any():
                        g_scores = g_scores.copy()
                        g_scores[zlm] = 0.0
                if label == 1:
                    asd_scores.append(g_scores)
                else:
                    control_scores.append(g_scores)

        explainer.remove_hooks()

        results: Dict[str, np.ndarray] = {}
        if asd_scores:
            results["asd_mean"] = np.mean(np.stack(asd_scores), axis=0)
            results["asd_std"]  = np.std(np.stack(asd_scores),  axis=0)
        if control_scores:
            results["control_mean"] = np.mean(np.stack(control_scores), axis=0)
            results["control_std"]  = np.std(np.stack(control_scores),  axis=0)

        logger.info(
            "GradCAM: collected %d ASD, %d Control graphs",
            len(asd_scores), len(control_scores),
        )

        # Warn when all scores are near-zero — a typical sign of a saturated /
        # degenerate model where softmax output barely varies across inputs and
        # back-propagated gradients collapse to zero.
        if results:
            all_means = [v for k, v in results.items() if k.endswith("_mean")]
            global_max = float(max(arr.max() for arr in all_means)) if all_means else 0.0
            if global_max < 1e-5:
                logger.warning(
                    "GradCAM scores are effectively zero (max=%.2e). "
                    "This typically means the model predicts one class for all "
                    "inputs (degenerate/biased model) so softmax gradients vanish. "
                    "Attribution results will not be meaningful until model quality improves.",
                    global_max,
                )

        return results

    # ── Attention extraction pass ──────────────────────────────────────────────

    def _run_attention_extraction(self) -> Dict:
        """Aggregate mean attention weight per brain region for each class."""
        extractor = AttentionWeightExtractor(self.model)
        self.model.eval()

        # We accumulate *edge* attention weights keyed by (layer, edge_idx)
        # Summarise per destination node (average attention flowing into each region)
        asd_node_attn:     Dict[int, List[np.ndarray]] = {i: [] for i in range(NUM_LOBES)}
        ctrl_node_attn:    Dict[int, List[np.ndarray]] = {i: [] for i in range(NUM_LOBES)}

        with torch.no_grad():
            for batch in self.test_loader:
                if batch is None:
                    continue
                batch = batch.to(self.device)
                labels = batch.y.cpu().numpy()

                _ = self.model(
                    batch.x,
                    batch.edge_index,
                    batch.edge_attr,
                    batch.batch,
                    site_id=None,
                    age=batch.age if hasattr(batch, "age") else None,
                    sex=batch.sex if hasattr(batch, "sex") else None,
                    fiq=batch.fiq if hasattr(batch, "fiq") else None,
                )
                attn_by_layer = extractor.get_attention_weights()

                # Use the last captured layer
                if not attn_by_layer:
                    extractor.clear()
                    continue
                last_layer_key = max(attn_by_layer.keys())
                alpha = attn_by_layer[last_layer_key]      # (E, heads) or (E,)
                if alpha.dim() > 1:
                    alpha = alpha.mean(dim=-1)             # (E,)
                alpha = alpha.cpu().numpy()

                edge_index_np = batch.edge_index.cpu().numpy()
                batch_np      = batch.batch.cpu().numpy()

                for graph_idx, label in enumerate(labels):
                    edge_mask = (batch_np[edge_index_np[1]] == graph_idx)
                    dst_nodes = edge_index_np[1, edge_mask] % NUM_LOBES
                    e_alpha   = alpha[edge_mask]

                    node_attn = np.zeros(NUM_LOBES)
                    counts    = np.zeros(NUM_LOBES)
                    for dst, a in zip(dst_nodes, e_alpha):
                        node_attn[dst] += a
                        counts[dst]    += 1
                    with np.errstate(invalid="ignore", divide="ignore"):
                        node_attn = np.where(counts > 0, node_attn / counts, 0.0)

                    storage = asd_node_attn if label == 1 else ctrl_node_attn
                    for ri in range(NUM_LOBES):
                        storage[ri].append(node_attn[ri])

                extractor.clear()

        extractor.remove_hooks()

        def _agg(storage: Dict[int, List]) -> np.ndarray:
            return np.array([np.mean(v) if v else 0.0 for v in storage.values()])

        return {
            "asd_mean":     _agg(asd_node_attn),
            "control_mean": _agg(ctrl_node_attn),
        }

    # ── Plotting ───────────────────────────────────────────────────────────────

    def _plot_gradcam(self, results: Dict, save_path: Path) -> None:
        """Side-by-side bar plot of GradCAM scores for ASD vs Control."""
        asd     = results.get("asd_mean",     np.zeros(NUM_LOBES))
        ctrl    = results.get("control_mean", np.zeros(NUM_LOBES))
        asd_sd  = results.get("asd_std",      np.zeros(NUM_LOBES))
        ctrl_sd = results.get("control_std",  np.zeros(NUM_LOBES))

        x = np.arange(NUM_LOBES)
        w = 0.38
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(x - w / 2, ctrl, w, label="Control", color="#3498db",
               yerr=ctrl_sd, capsize=3, alpha=0.85, ecolor="#1a5276")
        ax.bar(x + w / 2, asd, w, label="ASD",     color="#e74c3c",
               yerr=asd_sd,  capsize=3, alpha=0.85, ecolor="#7b241c")

        ax.set_xticks(x)
        ax.set_xticklabels(REGION_LABELS, rotation=45, ha="right", fontsize=10)
        ax.set_ylabel("GradCAM Importance Score", fontsize=12, fontweight="bold")
        ax.set_title("Node Importance by Brain Region (GradCAM)", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("GradCAM plot saved → %s", save_path)

    def _plot_attention(self, results: Dict, save_path: Path) -> None:
        """Heatmap of node-level attention weights for ASD vs Control."""
        asd  = results.get("asd_mean",     np.zeros(NUM_LOBES))
        ctrl = results.get("control_mean", np.zeros(NUM_LOBES))
        mat  = np.stack([ctrl, asd])        # (2, NUM_LOBES)

        fig, ax = plt.subplots(figsize=(14, 3.5))
        import seaborn as sns
        sns.heatmap(
            mat,
            xticklabels=REGION_LABELS,
            yticklabels=["Control", "ASD"],
            cmap="YlOrRd",
            ax=ax,
            linewidths=0.5,
            fmt=".3f",
            annot=True,
            annot_kws={"size": 8},
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
        ax.set_title("Mean GAT Attention Weight per Brain Region", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Attention heatmap saved → %s", save_path)

    def _plot_diff_bar(self, diff: np.ndarray, save_path: Path) -> None:
        """Horizontal bar chart of ASD - Control GradCAM score difference."""
        order = np.argsort(diff)
        sorted_diff   = diff[order]
        sorted_labels = [REGION_LABELS[i] for i in order]
        colors = [palette.ASD if v > 0 else palette.CONTROL for v in sorted_diff]

        fig, ax = plt.subplots(figsize=(9, 7))
        y = np.arange(NUM_LOBES)
        ax.barh(y, sorted_diff, color=colors, edgecolor="white", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_yticks(y)
        ax.set_yticklabels(sorted_labels, fontsize=10)
        ax.set_xlabel("ΔImportance (ASD − Control)", fontsize=12, fontweight="bold")
        ax.set_title(
            "Differential Node Importance: ASD vs Control (GradCAM)",
            fontsize=13, fontweight="bold",
        )

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=palette.ASD, label="Higher in ASD"),
            Patch(facecolor=palette.CONTROL, label="Higher in Control"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=10)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Differential importance plot saved → %s", save_path)

    def _plot_network_diff(
        self,
        net_diff: np.ndarray,
        asd_net: np.ndarray,
        ctrl_net: np.ndarray,
        save_path: Path,
    ) -> None:
        """Bar chart of ASD vs Control GradCAM importance at the network level (Task 3 — DD-011)."""
        x = np.arange(NUM_NETWORKS)
        w = 0.30
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x - w / 2, ctrl_net, w, label="Control", color="#3498db", alpha=0.85)
        ax.bar(x + w / 2, asd_net,  w, label="ASD",     color="#e74c3c", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(NETWORK_LABELS, fontsize=12, fontweight="bold")
        ax.set_ylabel("Mean GradCAM Importance", fontsize=12)
        ax.set_title(
            "Network-Level Node Importance: ASD vs Control\n"
            "(two-level anatomical hierarchical pooling, DD-011)",
            fontsize=13, fontweight="bold",
        )
        ax.legend(fontsize=11)
        ax.grid(axis="y", alpha=0.3)

        # Annotate difference
        for ni in range(NUM_NETWORKS):
            delta = net_diff[ni]
            ypos = max(asd_net[ni], ctrl_net[ni]) + 0.002
            ax.text(x[ni], ypos, f"Δ{delta:+.3f}", ha="center", fontsize=9,
                    color="#e74c3c" if delta > 0 else "#3498db")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Network-level importance plot saved → %s", save_path)
