#!/usr/bin/env python
"""Comprehensive per-subject diagnostics for Neuro-CXG pipeline artifacts.

Recommended usage:
    python -m src.analysis.subject_analysis

Outputs:
    results/subject_analysis/subject_analysis_<timestamp>.csv
    results/subject_analysis/subject_analysis_<timestamp>.txt
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import (
    CAUSAL_GRAPHS_DIR,
    DATA_FINAL,
    LOBE_NAMES,
    MASTER_MANIFEST,
    MIN_EDGES_PER_GRAPH,
    NODE_ATTRIBUTES_HARMONIZED,
    NUM_LOBES,
    RESULTS_DIR,
)
from src.core.validators import summarize_graph_degeneracy_from_adj

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

LOBE_NAME_LIST = [LOBE_NAMES[i] for i in range(NUM_LOBES)]
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "subject_analysis"


def _build_ts_index() -> Dict[str, Path]:
    """Map subject_id to its time-series file path across train/val/test splits."""
    ts_index: Dict[str, Path] = {}
    for split in ("train", "val", "test"):
        ts_dir = DATA_FINAL / split / "time_series"
        if not ts_dir.exists():
            continue
        for ts_file in ts_dir.glob("*_ts.npy"):
            subject_id = ts_file.stem.replace("_ts", "")
            ts_index[subject_id] = ts_file
    return ts_index


def _analyze_time_series(ts_path: Path) -> Dict[str, object]:
    """Compute per-subject time-series quality metrics."""
    out: Dict[str, object] = {}
    arr = np.load(ts_path)

    if arr.ndim != 2:
        raise ValueError(f"Unexpected time-series shape for {ts_path}: {arr.shape}")

    timepoints, n_rois = arr.shape
    nan_mask = np.isnan(arr)
    nan_col_mask = nan_mask.any(axis=0)

    out["ts_timepoints"] = int(timepoints)
    out["ts_n_rois"] = int(n_rois)
    out["ts_nan_rois"] = int(nan_col_mask.sum())
    out["ts_nan_fraction"] = float(nan_mask.mean())

    valid_col_mask = ~nan_mask.all(axis=0)
    if valid_col_mask.any():
        valid_arr = arr[:, valid_col_mask]
        out["ts_all_zero_rois"] = int((valid_arr == 0).all(axis=0).sum())
        out["ts_constant_rois"] = int((np.nanstd(valid_arr, axis=0) < 1e-6).sum())
    else:
        out["ts_all_zero_rois"] = 0
        out["ts_constant_rois"] = 0

    return out


def _analyze_graph(graph_path: Path) -> Dict[str, object]:
    """Compute per-subject graph quality metrics from graph file."""
    out: Dict[str, object] = {}
    data = torch.load(graph_path, map_location="cpu", weights_only=False)

    if "adj" not in data:
        raise KeyError(f"Missing adj in graph file: {graph_path}")

    adj = data["adj"].detach().cpu().to(torch.float32)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Invalid adjacency shape in {graph_path}: {tuple(adj.shape)}")

    stats = data.get("stats") or {}
    possible_edges = NUM_LOBES * (NUM_LOBES - 1)

    nonzero_mask = adj != 0
    off_diag_mask = ~torch.eye(adj.shape[0], dtype=torch.bool)
    edge_count = int((nonzero_mask & off_diag_mask).sum().item())

    deg_summary = summarize_graph_degeneracy_from_adj(adj, min_edges=MIN_EDGES_PER_GRAPH)

    nonzero_vals = adj[nonzero_mask & off_diag_mask]
    mean_weight = float(nonzero_vals.abs().mean().item()) if nonzero_vals.numel() > 0 else 0.0
    max_weight = float(nonzero_vals.abs().max().item()) if nonzero_vals.numel() > 0 else 0.0

    out["graph_edges"] = int(stats.get("edges", edge_count))
    out["graph_density"] = float(stats.get("density", edge_count / max(possible_edges, 1)))
    out["graph_mean_weight"] = float(stats.get("mean_weight", mean_weight))
    out["graph_max_weight"] = float(stats.get("max_weight", max_weight))
    out["graph_nan_in_adj"] = bool(torch.isnan(adj).any().item())

    in_deg = adj.abs().sum(dim=0)
    out_deg = adj.abs().sum(dim=1)
    dead_mask = (in_deg == 0) & (out_deg == 0)
    dead_ids = [i for i, is_dead in enumerate(dead_mask.tolist()) if is_dead]
    dead_names = [LOBE_NAME_LIST[i] for i in dead_ids]

    out["graph_dead_lobes"] = len(dead_ids)
    out["graph_dead_lobe_names"] = "|".join(dead_names)
    out["graph_is_degenerate"] = bool(deg_summary["is_degenerate"])

    zero_lobe_mask = data.get("zero_lobe_mask")
    if isinstance(zero_lobe_mask, torch.Tensor) and zero_lobe_mask.numel() == NUM_LOBES:
        fallback_ids = [i for i, v in enumerate(zero_lobe_mask.bool().tolist()) if v]
        out["graph_fallback_lobes"] = "|".join(LOBE_NAME_LIST[i] for i in fallback_ids)
    else:
        out["graph_fallback_lobes"] = ""

    for i, name in enumerate(LOBE_NAME_LIST):
        out[f"lobe_{i}_name"] = name
        out[f"lobe_{i}_in_deg"] = float(in_deg[i].item())
        out[f"lobe_{i}_out_deg"] = float(out_deg[i].item())

    return out


def _analyze_harmonized_row(row: pd.Series) -> Dict[str, object]:
    """Compute harmonized feature quality metrics for one subject row."""
    out: Dict[str, object] = {}
    out["harm_nan_features"] = int(row.isna().sum())

    zero_lobes = 0
    for lobe_name in LOBE_NAME_LIST:
        lobe_cols = [c for c in row.index if c.startswith(f"{lobe_name}_")]
        if not lobe_cols:
            continue
        vals = row[lobe_cols].to_numpy(dtype=float)
        if np.all(np.isfinite(vals)) and np.allclose(vals, 0.0):
            zero_lobes += 1
    out["harm_zero_lobes"] = int(zero_lobes)
    return out


def _safe_manifest_value(manifest_row: pd.Series, col: str, default=None):
    if col not in manifest_row.index:
        return default
    value = manifest_row[col]
    if pd.isna(value):
        return default
    return value


def run_analysis(limit: Optional[int] = None) -> pd.DataFrame:
    """Run full subject-level analysis across manifest, ts, graph, and harmonized files."""
    logger.info("Loading manifest from %s", MASTER_MANIFEST)
    manifest_df = pd.read_csv(MASTER_MANIFEST)
    if "subject_id" not in manifest_df.columns:
        raise KeyError(f"subject_id missing in manifest: {MASTER_MANIFEST}")
    manifest_df["subject_id"] = manifest_df["subject_id"].astype(str)
    manifest_df = manifest_df.set_index("subject_id")

    logger.info("Building time-series index")
    ts_index = _build_ts_index()
    logger.info("Found %d time-series files", len(ts_index))

    logger.info("Loading harmonized features from %s", NODE_ATTRIBUTES_HARMONIZED)
    harm_df = pd.read_csv(NODE_ATTRIBUTES_HARMONIZED)
    if "subject_id" not in harm_df.columns:
        raise KeyError(f"subject_id missing in harmonized file: {NODE_ATTRIBUTES_HARMONIZED}")
    harm_df["subject_id"] = harm_df["subject_id"].astype(str)
    harm_df = harm_df.set_index("subject_id")
    logger.info("Found %d harmonized feature rows", len(harm_df))

    logger.info("Indexing graph files from %s", CAUSAL_GRAPHS_DIR)
    graph_files = {
        graph_file.name.replace("_graph.pt", ""): graph_file
        for graph_file in sorted(CAUSAL_GRAPHS_DIR.glob("*_graph.pt"))
    }
    logger.info("Found %d graph files", len(graph_files))

    subject_ids = sorted(set(manifest_df.index) | set(ts_index.keys()) | set(harm_df.index) | set(graph_files.keys()))
    if limit is not None and limit > 0:
        subject_ids = subject_ids[:limit]

    logger.info("Running diagnostics for %d subjects", len(subject_ids))

    rows: List[Dict[str, object]] = []
    log_step = max(1, len(subject_ids) // 10)

    for idx, subject_id in enumerate(subject_ids):
        if idx % log_step == 0:
            logger.info("Progress: %d/%d", idx + 1, len(subject_ids))

        row: Dict[str, object] = {"subject_id": subject_id}

        if subject_id in manifest_df.index:
            row["in_manifest"] = True
            m = manifest_df.loc[subject_id]
            dx_group = _safe_manifest_value(m, "DX_GROUP", default=np.nan)
            row["dx_group"] = int(dx_group) if pd.notna(dx_group) else None
            if row["dx_group"] == 2:
                row["dx_label"] = "ASD"
            elif row["dx_group"] == 1:
                row["dx_label"] = "Control"
            else:
                row["dx_label"] = "Unknown"
            row["site_id"] = str(_safe_manifest_value(m, "SITE_ID", default=""))
            row["split"] = str(_safe_manifest_value(m, "split", default=""))
            cv_fold = _safe_manifest_value(m, "cv_fold", default=np.nan)
            row["cv_fold"] = int(cv_fold) if pd.notna(cv_fold) else None
            row["age_at_scan"] = _safe_manifest_value(m, "AGE_AT_SCAN", default=None)
            row["sex"] = _safe_manifest_value(m, "SEX", default=None)
            row["fiq"] = _safe_manifest_value(m, "FIQ", default=None)
        else:
            row["in_manifest"] = False
            row["dx_group"] = None
            row["dx_label"] = "Unknown"
            row["site_id"] = None
            row["split"] = None
            row["cv_fold"] = None
            row["age_at_scan"] = None
            row["sex"] = None
            row["fiq"] = None

        if subject_id in ts_index:
            row["ts_exists"] = True
            try:
                row.update(_analyze_time_series(ts_index[subject_id]))
            except Exception as exc:
                logger.warning("Time-series analysis failed for %s: %s", subject_id, exc)
                row["ts_exists"] = "error"
        else:
            row["ts_exists"] = False
            row["ts_timepoints"] = None
            row["ts_n_rois"] = None
            row["ts_nan_rois"] = None
            row["ts_nan_fraction"] = None
            row["ts_all_zero_rois"] = None
            row["ts_constant_rois"] = None

        if subject_id in graph_files:
            row["graph_exists"] = True
            try:
                row.update(_analyze_graph(graph_files[subject_id]))
            except Exception as exc:
                logger.warning("Graph analysis failed for %s: %s", subject_id, exc)
                row["graph_exists"] = "error"
        else:
            row["graph_exists"] = False
            row["graph_edges"] = None
            row["graph_density"] = None
            row["graph_mean_weight"] = None
            row["graph_max_weight"] = None
            row["graph_nan_in_adj"] = None
            row["graph_dead_lobes"] = None
            row["graph_dead_lobe_names"] = None
            row["graph_is_degenerate"] = None
            row["graph_fallback_lobes"] = None
            for i in range(NUM_LOBES):
                row[f"lobe_{i}_name"] = LOBE_NAME_LIST[i]
                row[f"lobe_{i}_in_deg"] = None
                row[f"lobe_{i}_out_deg"] = None

        if subject_id in harm_df.index:
            row["harm_exists"] = True
            try:
                row.update(_analyze_harmonized_row(harm_df.loc[subject_id]))
            except Exception as exc:
                logger.warning("Harmonized feature analysis failed for %s: %s", subject_id, exc)
                row["harm_exists"] = "error"
        else:
            row["harm_exists"] = False
            row["harm_nan_features"] = None
            row["harm_zero_lobes"] = None

        rows.append(row)

    return pd.DataFrame(rows)


def build_report(df: pd.DataFrame) -> str:
    """Build text summary report from per-subject diagnostics DataFrame."""
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("NEURO-CXG SUBJECT ANALYSIS REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 78)

    total = len(df)
    lines.append(f"Total subjects scanned: {total}")

    in_manifest = int((df["in_manifest"] == True).sum())
    lines.append(f"In manifest: {in_manifest} ({100 * in_manifest / max(total, 1):.1f}%)")

    if "split" in df.columns:
        split_counts = df[df["split"].notna()]["split"].value_counts().to_dict()
        if split_counts:
            lines.append("Split counts:")
            for split_name, count in split_counts.items():
                lines.append(f"  {split_name}: {int(count)}")

    if "dx_label" in df.columns:
        dx_counts = df[df["dx_label"].notna()]["dx_label"].value_counts().to_dict()
        if dx_counts:
            lines.append("Diagnosis counts:")
            for dx_name, count in dx_counts.items():
                lines.append(f"  {dx_name}: {int(count)}")

    ts_exists = int((df["ts_exists"] == True).sum())
    graph_exists = int((df["graph_exists"] == True).sum())
    harm_exists = int((df["harm_exists"] == True).sum())
    lines.append("")
    lines.append("Artifact availability:")
    lines.append(f"  Time-series present: {ts_exists} ({100 * ts_exists / max(total, 1):.1f}%)")
    lines.append(f"  Graph present:       {graph_exists} ({100 * graph_exists / max(total, 1):.1f}%)")
    lines.append(f"  Harmonized present:  {harm_exists} ({100 * harm_exists / max(total, 1):.1f}%)")

    if "graph_is_degenerate" in df.columns:
        degenerate = int((df["graph_is_degenerate"] == True).sum())
    lines.append(
        "  Degenerate graphs:   "
        f"{degenerate} ({100 * degenerate / max(graph_exists, 1):.1f}% of existing) "
        "[criterion: edge_count < MIN_EDGES_PER_GRAPH OR dead lobe present]"
    )

    if "graph_dead_lobe_names" in df.columns:
        exploded = (
            df[df["graph_dead_lobe_names"].notna() & (df["graph_dead_lobe_names"] != "")]
            ["graph_dead_lobe_names"]
            .str.split("|")
            .explode()
            .value_counts()
        )
        if not exploded.empty:
            lines.append("")
            lines.append("Most frequent dead lobes:")
            for lobe_name, count in exploded.items():
                lines.append(f"  {lobe_name}: {int(count)}")

    lines.append("")
    lines.append("Top flagged subjects:")
    flagged = df[
        (df["graph_is_degenerate"] == True)
        | (df["graph_nan_in_adj"] == True)
        | (df["ts_nan_rois"].fillna(0) > 20)
    ]
    if flagged.empty:
        lines.append("  None")
    else:
        for subject_id in flagged["subject_id"].head(20).tolist():
            lines.append(f"  {subject_id}")

    lines.append("=" * 78)
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comprehensive per-subject pipeline analysis")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for number of subjects")
    parser.add_argument("--prefix", type=str, default="subject_analysis", help="Output filename prefix")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    df = run_analysis(limit=args.limit)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = args.output_dir / f"{args.prefix}_{timestamp}.csv"
    txt_path = args.output_dir / f"{args.prefix}_{timestamp}.txt"

    df.to_csv(csv_path, index=False)
    logger.info("Saved per-subject CSV to %s", csv_path)

    report = build_report(df)
    txt_path.write_text(report)
    logger.info("Saved summary report to %s", txt_path)

    print(report)


if __name__ == "__main__":
    main()
