#!/usr/bin/env python
"""Visualize directed causal brain graphs for one or two subjects.

Recommended usage:
    python -m src.analysis.visualize_causal_graph --subject CMU_a_0050656
    python -m src.analysis.visualize_causal_graph --auto-pair --site-id CMU

By default, output is saved under results/visualizations.
"""

from __future__ import annotations

import argparse
import logging
import warnings
from collections.abc import Sequence

import matplotlib

matplotlib.use('Agg')
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Suppress tight_layout warning with colorbars
warnings.filterwarnings('ignore', message='.*tight_layout.*not compatible.*')

import pandas as pd
import torch

from src.core.config import (
    CAUSAL_GRAPHS_DIR,
    LOBE_NAMES,
    MASTER_MANIFEST,
    NUM_LOBES,
    RESULTS_DIR,
)

logger = logging.getLogger(__name__)

# Preferred anatomical display order.
DISPLAY_ORDER = [
    "Frontal_Superior",
    "Frontal_Orbital",
    "Motor_Premotor",
    "Parietal",
    "Occipital",
    "Cerebellum",
    "Brainstem",
    "Subcortical",
    "Temporal",
    "Limbic",
    "Cingulate",
    "Insula",
]

LOBE_COLORS = {
    "Frontal_Superior": "#4e79a7",
    "Frontal_Orbital": "#4e79a7",
    "Motor_Premotor": "#f28e2b",
    "Parietal": "#e15759",
    "Occipital": "#76b7b2",
    "Cerebellum": "#59a14f",
    "Brainstem": "#edc948",
    "Subcortical": "#b07aa1",
    "Temporal": "#ff9da7",
    "Limbic": "#9c755f",
    "Cingulate": "#bab0ac",
    "Insula": "#f1ce63",
}

def _resolve_lobe_order(raw_order: Sequence[str]) -> list[str]:
    """Resolve lobe order with fallback to config names when missing."""
    if raw_order and len(raw_order) == NUM_LOBES:
        return list(raw_order)
    return [LOBE_NAMES[i] for i in range(NUM_LOBES)]

def _compute_stats_from_adj(adj: np.ndarray) -> dict[str, float]:
    """Compute graph stats robustly from adjacency matrix."""
    mask = ~np.eye(adj.shape[0], dtype=bool)
    off_diag = adj[mask]
    nonzero = off_diag[off_diag != 0]
    possible_edges = adj.shape[0] * (adj.shape[0] - 1)
    edges = int(nonzero.size)
    density = edges / max(possible_edges, 1)
    if nonzero.size == 0:
        return {
            "edges": 0,
            "density": 0.0,
            "mean_weight": 0.0,
            "max_weight": 0.0,
        }
    return {
        "edges": edges,
        "density": float(density),
        "mean_weight": float(np.mean(np.abs(nonzero))),
        "max_weight": float(np.max(np.abs(nonzero))),
    }

def load_graph(subject_id: str) -> tuple[np.ndarray, list[str], dict[str, float]]:
    """Load graph file and return adjacency, lobe order, and stats."""
    graph_path = CAUSAL_GRAPHS_DIR / f"{subject_id}_graph.pt"
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph not found: {graph_path}")

    data = torch.load(graph_path, map_location="cpu", weights_only=True)
    if "adj" not in data:
        raise KeyError(f"Missing 'adj' key in graph file: {graph_path}")

    adj_tensor = data["adj"].detach().cpu().to(torch.float32)
    if adj_tensor.ndim != 2 or adj_tensor.shape[0] != adj_tensor.shape[1]:
        raise ValueError(f"Invalid adjacency shape for {subject_id}: {tuple(adj_tensor.shape)}")

    lobe_order = _resolve_lobe_order(data.get("lobe_order", []))
    stats = data.get("stats") or _compute_stats_from_adj(adj_tensor.numpy())
    return adj_tensor.numpy(), lobe_order, stats

def build_graph(adj: np.ndarray, lobe_order: Sequence[str], threshold: float) -> nx.DiGraph:
    """Build directed graph from adjacency matrix with thresholding."""
    graph = nx.DiGraph()
    graph.add_nodes_from(lobe_order)

    for src_idx, src in enumerate(lobe_order):
        for dst_idx, dst in enumerate(lobe_order):
            if src_idx == dst_idx:
                continue
            weight = float(adj[src_idx, dst_idx])
            if abs(weight) > threshold:
                graph.add_edge(src, dst, weight=weight)

    return graph

def _position_map(lobe_order: Sequence[str]) -> dict[str, tuple[float, float]]:
    """Create stable anatomical circular layout for provided lobe order."""
    display_index = {name: i for i, name in enumerate(DISPLAY_ORDER)}
    used = []
    for i, name in enumerate(lobe_order):
        used.append(display_index.get(name, len(DISPLAY_ORDER) + i))

    denom = max(len(used), 1)
    pos: dict[str, tuple[float, float]] = {}
    for name, idx in zip(lobe_order, used, strict=False):
        angle = 2 * np.pi * idx / denom - np.pi / 2
        pos[name] = (float(np.cos(angle)), float(np.sin(angle)))
    return pos

def draw_graph(
    ax,
    adj: np.ndarray,
    lobe_order: Sequence[str],
    title: str,
    stats: dict[str, float],
    threshold: float,
) -> None:
    """Draw one causal graph panel."""
    graph = build_graph(adj, lobe_order, threshold=threshold)
    pos = _position_map(lobe_order)

    node_colors = [LOBE_COLORS.get(name, "#999999") for name in lobe_order]
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=920,
        alpha=0.92,
    )

    labels = {name: name.replace("_", "\n") for name in lobe_order}
    nx.draw_networkx_labels(graph, pos, labels=labels, ax=ax, font_size=7, font_weight="bold")

    if graph.number_of_edges() > 0:
        weights = np.array([graph[u][v]["weight"] for u, v in graph.edges()], dtype=float)
        abs_weights = np.abs(weights)
        w_min = float(abs_weights.min())
        w_max = float(abs_weights.max())
        scale = max(w_max - w_min, 1e-8)
        norm = (abs_weights - w_min) / scale
        edge_widths = 1.5 + 4.5 * norm
        edge_colors = plt.cm.RdYlBu_r(norm)

        # Draw edges with z-order: stronger edges on top
        abs_weights_sorted = sorted(zip(edge_widths, edge_colors, graph.edges(), strict=False),
                                       key=lambda x: x[0])

        for width, color, (u, v) in abs_weights_sorted:
            nx.draw_networkx_edges(
                graph,
                pos,
                edgelist=[(u, v)],
                edge_color=[color],
                width=width,
                arrows=True,
                arrowstyle="-|>",
                arrowsize=12,
                connectionstyle="arc3,rad=0.14",
                min_source_margin=20,
                min_target_margin=20,
                ax=ax
            )

    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.text(
        0.01,
        0.01,
        (
            f"Edges: {int(stats.get('edges', 0))} | "
            f"Density: {float(stats.get('density', 0.0)):.1%}\n"
            f"Mean |w|: {float(stats.get('mean_weight', 0.0)):.2f} | "
            f"Max |w|: {float(stats.get('max_weight', 0.0)):.2f}"
        ),
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "alpha": 0.72},
    )
    ax.axis("off")

def _load_manifest() -> pd.DataFrame:
    """Load manifest with safe defaults."""
    manifest = pd.read_csv(MASTER_MANIFEST)
    if "subject_id" not in manifest.columns:
        raise KeyError(f"Manifest missing subject_id column: {MASTER_MANIFEST}")
    return manifest

def resolve_dx_label(subject_id: str, manifest: pd.DataFrame) -> str:
    """Map DX_GROUP to readable label for one subject."""
    rows = manifest[manifest["subject_id"] == subject_id]
    if rows.empty or "DX_GROUP" not in rows.columns:
        return "Unknown"
    dx = int(rows.iloc[0]["DX_GROUP"])
    if dx == 2:
        return "ASD"
    if dx == 1:
        return "Control"
    return f"DX_{dx}"

def pick_asd_control_pair(manifest: pd.DataFrame, site_id: str | None) -> tuple[str, str]:
    """Pick one ASD and one Control subject, optionally constrained to site."""
    base = manifest
    if site_id:
        base = base[base["SITE_ID"].astype(str) == str(site_id)]
        if base.empty:
            raise ValueError(f"No manifest rows found for site: {site_id}")

    asd_rows = base[base["DX_GROUP"] == 2]
    ctrl_rows = base[base["DX_GROUP"] == 1]
    if asd_rows.empty or ctrl_rows.empty:
        raise ValueError("Could not find both ASD and Control subjects for requested selection")

    return str(asd_rows.iloc[0]["subject_id"]), str(ctrl_rows.iloc[0]["subject_id"])

def plot_single(subject_id: str, output_path: Path, threshold: float, dpi: int) -> None:
    """Render a single-subject causal graph."""
    manifest = _load_manifest()
    dx_label = resolve_dx_label(subject_id, manifest)

    adj, lobe_order, stats = load_graph(subject_id)
    fig, ax = plt.subplots(figsize=(8, 8))
    draw_graph(
        ax=ax,
        adj=adj,
        lobe_order=lobe_order,
        title=f"{subject_id} ({dx_label})",
        stats=stats,
        threshold=threshold,
    )

    vmax = max(float(stats.get("max_weight", 0.0)), 1e-8)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r, norm=plt.Normalize(vmin=0.0, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("Causal weight magnitude", fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot to %s", output_path)

def plot_comparison(
    asd_subject: str,
    control_subject: str,
    output_path: Path,
    threshold: float,
    dpi: int,
) -> None:
    """Render side-by-side ASD and Control graph comparison."""
    adj_asd, lo_asd, stats_asd = load_graph(asd_subject)
    adj_ctrl, lo_ctrl, stats_ctrl = load_graph(control_subject)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(
        "Directed causal brain graph comparison\n"
        "Arrow direction: source region -> influenced region",
        fontsize=12,
        y=1.01,
    )

    draw_graph(
        axes[0],
        adj_asd,
        lo_asd,
        f"ASD: {asd_subject}",
        stats_asd,
        threshold=threshold,
    )
    draw_graph(
        axes[1],
        adj_ctrl,
        lo_ctrl,
        f"Control: {control_subject}",
        stats_ctrl,
        threshold=threshold,
    )

    vmax = max(float(stats_asd.get("max_weight", 0.0)), float(stats_ctrl.get("max_weight", 0.0)), 1e-8)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r, norm=plt.Normalize(vmin=0.0, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.028, pad=0.02)
    cbar.set_label("Causal weight magnitude", fontsize=10)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved comparison plot to %s", output_path)

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize directed causal graph(s)")
    parser.add_argument("--subject", type=str, default=None, help="Plot one subject")
    parser.add_argument("--asd-subject", type=str, default=None, help="ASD subject for comparison")
    parser.add_argument("--control-subject", type=str, default=None, help="Control subject for comparison")
    parser.add_argument("--auto-pair", action="store_true", help="Auto-pick one ASD and one Control from manifest")
    parser.add_argument("--site-id", type=str, default=None, help="Optional site filter for auto-pair")
    parser.add_argument("--threshold", type=float, default=0.0, help="Absolute edge threshold")
    parser.add_argument("--dpi", type=int, default=160, help="Output image DPI")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path")
    return parser.parse_args()

def main() -> None:
    args = _parse_args()

    if args.subject:
        output = args.output or (RESULTS_DIR / "visualizations" / f"causal_graph_{args.subject}.png")
        plot_single(args.subject, output_path=output, threshold=args.threshold, dpi=args.dpi)
        return

    if args.asd_subject and args.control_subject:
        asd_id = args.asd_subject
        ctrl_id = args.control_subject
    else:
        manifest = _load_manifest()
        asd_id, ctrl_id = pick_asd_control_pair(manifest=manifest, site_id=args.site_id)

    output = args.output or (RESULTS_DIR / "visualizations" / "causal_graph_comparison.png")
    plot_comparison(
        asd_subject=asd_id,
        control_subject=ctrl_id,
        output_path=output,
        threshold=args.threshold,
        dpi=args.dpi,
    )

if __name__ == "__main__":
    main()
