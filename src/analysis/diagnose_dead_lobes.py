#!/usr/bin/env python
"""Diagnose dead or near-dead lobes in causal graph construction.

This script inspects one subject time-series file and reports:
- raw ROI quality for selected lobes
- aggregate_to_lobes outputs and fallback mask
- causality matrix in/out strengths and dead-lobe detection

Recommended usage:
    python -m src.analysis.diagnose_dead_lobes --subject CMU_a_0050656
    python -m src.analysis.diagnose_dead_lobes --split train --lobes 4,8,9,11
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import DATA_FINAL, LOBE_MAPPING, LOBE_NAMES, NUM_LOBES
from src.features.construct_causal import aggregate_to_lobes, compute_causality_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_lobe_ids(raw: str) -> List[int]:
    """Parse comma-separated lobe ids."""
    ids = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 0 or value >= NUM_LOBES:
            raise ValueError(f"Invalid lobe id {value}; expected 0..{NUM_LOBES - 1}")
        ids.append(value)
    if not ids:
        raise ValueError("No valid lobe ids parsed")
    return ids


def _find_subject_ts_path(subject_id: str) -> Path:
    """Locate subject time-series file across train/val/test."""
    for split in ("train", "val", "test"):
        path = DATA_FINAL / split / "time_series" / f"{subject_id}_ts.npy"
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find time-series file for subject {subject_id}")


def _pick_first_subject(split: str) -> Path:
    """Pick first available time-series file for a split."""
    ts_dir = DATA_FINAL / split / "time_series"
    if not ts_dir.exists():
        raise FileNotFoundError(f"Time-series directory does not exist: {ts_dir}")
    ts_files = sorted(ts_dir.glob("*_ts.npy"))
    if not ts_files:
        raise FileNotFoundError(f"No time-series files found in {ts_dir}")
    return ts_files[0]


def _print_lobe_raw_stats(ts_tensor: torch.Tensor, lobe_id: int) -> None:
    """Print raw ROI diagnostics for one lobe."""
    roi_indices = LOBE_MAPPING[lobe_id]
    valid_indices = [idx for idx in roi_indices if idx < ts_tensor.shape[1]]

    print(
        f"Lobe {lobe_id} ({LOBE_NAMES[lobe_id]}): "
        f"mapped_rois={len(roi_indices)}, valid_rois={len(valid_indices)}"
    )

    if not valid_indices:
        print("  No valid ROI indices for this subject")
        return

    roi_data = ts_tensor[:, valid_indices]
    roi_nan_any = torch.isnan(roi_data).any(dim=0)
    roi_nan_all = torch.isnan(roi_data).all(dim=0)

    print(f"  ROI columns with any NaN: {int(roi_nan_any.sum().item())}")
    print(f"  ROI columns with all NaN: {int(roi_nan_all.sum().item())}")

    finite_mask = ~torch.isnan(roi_data)
    if finite_mask.any():
        finite_vals = roi_data[finite_mask]
        print(f"  Finite value range: [{float(finite_vals.min()):.4f}, {float(finite_vals.max()):.4f}]")

    valid_roi_data = roi_data[:, ~roi_nan_any]
    if valid_roi_data.shape[1] == 0:
        print("  No NaN-free ROI columns left after filtering")
        return

    mean_per_roi = valid_roi_data.mean(dim=0)
    std_per_roi = valid_roi_data.std(dim=0)

    print(f"  Mean range: [{float(mean_per_roi.min()):.4f}, {float(mean_per_roi.max()):.4f}]")
    print(f"  Std range:  [{float(std_per_roi.min()):.6f}, {float(std_per_roi.max()):.6f}]")
    print(f"  Zero values in valid ROI block: {int((valid_roi_data == 0).sum().item())}")


def _report_lobe_signal_stats(ts_lobes: torch.Tensor, zero_lobe_mask: torch.Tensor) -> List[int]:
    """Print lobe-wise stats after aggregation and return zero-mask ids."""
    print("\n--- aggregate_to_lobes output ---")
    print(f"ts_lobes shape: {tuple(ts_lobes.shape)}")

    fallback_ids = [i for i, v in enumerate(zero_lobe_mask.bool().tolist()) if v]

    for lobe_id in range(NUM_LOBES):
        lobe_ts = ts_lobes[:, lobe_id]
        has_nan = bool(torch.isnan(lobe_ts).any().item())
        is_zero = bool(torch.allclose(lobe_ts, torch.zeros_like(lobe_ts)))
        std_val = float(torch.nan_to_num(lobe_ts).std().item())
        fallback_tag = " fallback" if lobe_id in fallback_ids else ""
        print(
            f"  Lobe {lobe_id:2d} ({LOBE_NAMES[lobe_id]:<16}) "
            f"std={std_val:.4f} nan={has_nan} zero={is_zero}{fallback_tag}"
        )

    return fallback_ids


def _report_causality_stats(causal_adj: torch.Tensor) -> List[int]:
    """Print causal in/out strengths and return dead lobe ids."""
    print("\n--- Causality matrix diagnostics ---")
    print(f"causal_adj shape: {tuple(causal_adj.shape)}")

    in_strength = causal_adj.abs().sum(dim=0)
    out_strength = causal_adj.abs().sum(dim=1)

    dead_ids = []
    for lobe_id in range(NUM_LOBES):
        in_v = float(in_strength[lobe_id].item())
        out_v = float(out_strength[lobe_id].item())
        is_dead = (in_v == 0.0) and (out_v == 0.0)
        if is_dead:
            dead_ids.append(lobe_id)
        print(
            f"  Lobe {lobe_id:2d} ({LOBE_NAMES[lobe_id]:<16}) "
            f"in={in_v:.4f} out={out_v:.4f} dead={is_dead}"
        )

    return dead_ids


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose dead lobe behavior for one subject")
    parser.add_argument("--subject", type=str, default=None, help="Subject id without _ts suffix")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Used only when --subject is not provided",
    )
    parser.add_argument(
        "--lobes",
        type=str,
        default="4,8,9,11",
        help="Comma-separated lobe ids to inspect in raw ROI space",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    lobe_ids = _parse_lobe_ids(args.lobes)

    if args.subject:
        ts_path = _find_subject_ts_path(args.subject)
    else:
        ts_path = _pick_first_subject(args.split)

    subject_id = ts_path.stem.replace("_ts", "")
    print(f"Subject: {subject_id}")
    print(f"Time-series path: {ts_path}")

    ts_raw = np.load(ts_path)
    if ts_raw.ndim != 2:
        raise ValueError(f"Unexpected time-series shape for {subject_id}: {ts_raw.shape}")

    print(f"Raw TS shape: {ts_raw.shape}, dtype: {ts_raw.dtype}")
    ts_tensor = torch.from_numpy(ts_raw).to(torch.float32)

    print("\n--- Raw ROI diagnostics for requested lobes ---")
    for lobe_id in lobe_ids:
        _print_lobe_raw_stats(ts_tensor, lobe_id)

    ts_lobes, internal_features, zero_lobe_mask = aggregate_to_lobes(ts_tensor)
    print(f"\nInternal feature shape: {tuple(internal_features.shape)}")

    fallback_ids = _report_lobe_signal_stats(ts_lobes, zero_lobe_mask)

    causal_adj = compute_causality_matrix(ts_lobes)
    dead_ids = _report_causality_stats(causal_adj)

    print("\n--- Summary ---")
    print(f"Requested lobe ids: {lobe_ids}")
    print(f"Fallback lobes from aggregate_to_lobes: {fallback_ids}")
    print(f"Dead lobes from causal adjacency: {dead_ids}")

    if dead_ids:
        print("Dead lobe names:")
        for lobe_id in dead_ids:
            print(f"  - {lobe_id}: {LOBE_NAMES[lobe_id]}")


if __name__ == "__main__":
    main()
