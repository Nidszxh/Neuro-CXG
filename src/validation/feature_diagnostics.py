"""
Feature Pipeline End-to-End Diagnostic Tool
============================================
Covers all feature-pipeline validation tasks:
  1. 28-feature tensor shape/range audit per group (temporal/frequency/internal/spatial)
  2. Granger causality edge validation — causal matrix before/after sparsification
  3. Graph edge-density distribution across all .pt files
  4. Frequency feature fMRI validity audit (TR → fs → Nyquist)

Usage:
    python -m src.validation.feature_diagnostics              # full audit
    python -m src.validation.feature_diagnostics --quick      # skip edge histogram
    python -m src.validation.feature_diagnostics --sample 5   # audit N graph files
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import (
    ALL_FEATURE_NAMES,
    CAUSAL_GRAPHS_DIR,
    FEATURE_GROUPS,
    GNN_IN_CHANNELS,
    GRANGER_MAX_LAG,
    GRANGER_SIGNIFICANCE_LEVEL,
    MASTER_MANIFEST,
    MIN_EDGES_PER_GRAPH,
    NUM_LOBES,
    NODE_ATTRIBUTES_HARMONIZED,
    SPARSITY_QUANTILE,
    DATA_FINAL,
    DEFAULT_TR,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── feature group index ranges ────────────────────────────────────────────────
_GROUP_SLICES: Dict[str, slice] = {}
_offset = 0
for _grp, _feats in FEATURE_GROUPS.items():
    _GROUP_SLICES[_grp] = slice(_offset, _offset + len(_feats))
    _offset += len(_feats)


# ─────────────────────────────────────────────────────────────────────────────
# 1. 28-FEATURE TENSOR AUDIT
# ─────────────────────────────────────────────────────────────────────────────

def audit_feature_tensor(graph_path: Path) -> bool:
    """
    Load a single graph .pt file and print shape/value ranges per feature group.

    Returns True if all checks pass.
    """
    logger.info("=" * 70)
    logger.info(f"FEATURE TENSOR AUDIT: {graph_path.name}")
    logger.info("=" * 70)

    if not graph_path.exists():
        logger.error(f"Graph not found: {graph_path}")
        return False

    data = torch.load(graph_path, weights_only=False)

    # Handle both raw dict format and PyTorch Geometric Data objects
    if hasattr(data, "x"):
        x = data.x  # PyG Data object
    elif isinstance(data, dict) and "x" in data:
        x = data["x"]
    elif isinstance(data, dict) and "adj" in data:
        logger.warning("Graph stored in dict format with 'adj' key — no 'x' tensor found.")
        logger.warning("This graph was built BEFORE graph_factory assembles features; "
                       "use ABIDECausalDataset to get assembled tensors.")
        _audit_adj_dict(data)
        return True
    else:
        logger.error(f"Unrecognised graph format: {type(data)}, keys={list(data.keys()) if isinstance(data, dict) else 'N/A'}")
        return False

    if x is None:
        logger.error("x tensor is None")
        return False

    expected_shape = (NUM_LOBES, GNN_IN_CHANNELS)
    ok = True

    logger.info(f"\nNode feature tensor shape: {tuple(x.shape)}")
    if tuple(x.shape) != expected_shape:
        logger.error(f"  ✗ Expected {expected_shape}, got {tuple(x.shape)}")
        ok = False
    else:
        logger.info(f"  ✓ Shape matches expected {expected_shape}")

    # Per-group stats
    logger.info(f"\n{'Group':<12} {'Indices':<14} {'Min':>10} {'Max':>10} {'Mean':>10} {'NaN':>6} {'~Zero':>6}")
    logger.info("-" * 70)
    for grp, sl in _GROUP_SLICES.items():
        cols = x[:, sl]
        nan_count = torch.isnan(cols).sum().item()
        inf_count = torch.isinf(cols).sum().item()
        near_zero = (cols.abs() < 1e-8).sum().item()
        total_vals = cols.numel()

        if nan_count > 0 or inf_count > 0:
            logger.warning(f"  ✗ {grp}: {nan_count} NaN, {inf_count} Inf values!")
            ok = False

        all_zero = near_zero == total_vals
        if all_zero:
            logger.warning(f"  ✗ {grp}: ALL values are zero (silent zero-padding)")
            ok = False

        idx_str = f"[{sl.start}:{sl.stop}]"
        v_min = cols.min().item() if not all_zero else 0.0
        v_max = cols.max().item() if not all_zero else 0.0
        v_mean = cols.mean().item() if not all_zero else 0.0
        logger.info(
            f"  {grp:<10} {idx_str:<14} {v_min:>10.4f} {v_max:>10.4f} "
            f"{v_mean:>10.4f} {nan_count:>6} {near_zero:>6}/{total_vals}"
        )

    # Edge check
    if hasattr(data, "edge_index"):
        n_edges = data.edge_index.shape[1]
        logger.info(f"\nEdges: {n_edges}")
        if n_edges == 0:
            logger.error("  ✗ Zero edges — graph is disconnected")
            ok = False
        elif n_edges < MIN_EDGES_PER_GRAPH:
            logger.warning(f"  ⚠ Below minimum floor ({MIN_EDGES_PER_GRAPH})")
        else:
            logger.info(f"  ✓ Edge count above minimum floor ({MIN_EDGES_PER_GRAPH})")

    logger.info(f"\nResult: {'✓ PASS' if ok else '✗ FAIL'}")
    return ok


def _audit_adj_dict(data: dict) -> None:
    """Print adj-dict graph stats (dict format from construct_causal.py)."""
    adj = data["adj"]
    n_edges = (adj != 0).sum().item()
    logger.info(f"  adj shape   : {tuple(adj.shape)}")
    logger.info(f"  non-zero    : {n_edges}")
    logger.info(f"  weight range: [{adj.min().item():.4f}, {adj.max().item():.4f}]")

    if "internal_features" in data:
        intf = data["internal_features"]
        logger.info(f"  internal_features shape: {tuple(intf.shape)}")
        nan_count = torch.isnan(intf).sum().item()
        logger.info(f"  internal_features NaN  : {nan_count}")


def audit_feature_tensor_via_dataset(n_samples: int = 3) -> bool:
    """
    Audit assembled feature tensors through ABIDECausalDataset (authoritative path).
    """
    logger.info("\n" + "=" * 70)
    logger.info("FEATURE TENSOR AUDIT via ABIDECausalDataset")
    logger.info("=" * 70)

    try:
        from src.features.graph_factory import ABIDECausalDataset
    except ImportError as e:
        logger.error(f"Cannot import ABIDECausalDataset: {e}")
        return False

    try:
        ds = ABIDECausalDataset(split="train")
    except Exception as e:
        logger.error(f"Dataset construction failed: {e}")
        return False

    if len(ds) == 0:
        logger.error("Train dataset is empty — cannot audit features")
        return False

    all_pass = True
    checked = 0
    for i in range(min(n_samples, len(ds))):
        sample = ds[i]
        if sample is None:
            continue
        checked += 1
        sub = getattr(sample, "sub_id", f"idx_{i}")
        logger.info(f"\n  Subject: {sub}")
        logger.info(f"  x shape : {tuple(sample.x.shape)}")

        for grp, sl in _GROUP_SLICES.items():
            cols = sample.x[:, sl]
            nan_c = torch.isnan(cols).sum().item()
            zero_c = (cols.abs() < 1e-8).sum().item()
            flag = "⚠ all-zero" if zero_c == cols.numel() else ("✗ has NaN" if nan_c else "✓")
            if "✗" in flag or "⚠" in flag:
                all_pass = False
            logger.info(
                f"    {grp:<12} [{sl.start}:{sl.stop}]"
                f"  min={cols.min().item():.4f}  max={cols.max().item():.4f}  {flag}"
            )

    logger.info(f"\n  Checked {checked} subjects — {'✓ ALL PASS' if all_pass else '✗ ISSUES FOUND'}")
    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
# 2. GRANGER CAUSALITY EDGE VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_granger_edges(subject_id: Optional[str] = None, n_subjects: int = 5) -> None:
    """
    For sample subjects, print causal matrix before/after sparsification.
    Confirms Granger weights are non-trivial (not all zeros / fallback-to-pearson).
    """
    logger.info("\n" + "=" * 70)
    logger.info("GRANGER CAUSALITY EDGE VALIDATION")
    logger.info("=" * 70)

    if not MASTER_MANIFEST.exists():
        logger.error(f"Manifest not found: {MASTER_MANIFEST}")
        return

    manifest = pd.read_csv(MASTER_MANIFEST)

    if subject_id:
        subjects = [subject_id]
    else:
        sample = manifest.sample(min(n_subjects, len(manifest)), random_state=42)
        subjects = sample["subject_id"].tolist()

    for sub_id in subjects:
        graph_path = CAUSAL_GRAPHS_DIR / f"{sub_id}_graph.pt"
        if not graph_path.exists():
            logger.warning(f"  {sub_id}: graph not found")
            continue

        data = torch.load(graph_path, weights_only=False)
        adj = data["adj"] if isinstance(data, dict) else getattr(data, "edge_attr", None)

        if isinstance(data, dict) and "adj" in data:
            adj = data["adj"]
            n_total = adj.numel()
            n_nonzero = (adj.abs() > 0).sum().item()
            pct_nonzero = 100.0 * n_nonzero / max(n_total, 1)
            adj_vals = adj[adj != 0]
            is_all_same = (adj_vals.std().item() < 1e-6) if len(adj_vals) > 1 else True
            max_val = adj.abs().max().item()

            logger.info(f"\n  {sub_id}:")
            logger.info(f"    adj shape   : {tuple(adj.shape)}")
            logger.info(f"    non-zero    : {n_nonzero}/{n_total} ({pct_nonzero:.1f}%)")
            logger.info(f"    weight range: [{adj.min().item():.4f}, {adj.max().item():.4f}]")
            logger.info(f"    max -log10p : {max_val:.4f}")

            if max_val < 1e-6:
                logger.error("    ✗ All edge weights are zero — Granger test silent failure!")
            elif is_all_same and len(adj_vals) > 1:
                logger.warning("    ⚠ All edge weights are identical — Granger may have fallen back to lagged Pearson")
            else:
                expected_min = -np.log10(GRANGER_SIGNIFICANCE_LEVEL)  # 1.301 for p=0.05
                strong_edges = (adj_vals > expected_min).sum().item()
                logger.info(f"    significant edges (>{expected_min:.2f}): {strong_edges}/{n_nonzero}")
                logger.info(f"    ✓ Non-trivial edge weights detected")
        else:
            logger.warning(f"  {sub_id}: unexpected graph format — cannot directly inspect adj matrix")


# ─────────────────────────────────────────────────────────────────────────────
# 3. GRAPH EDGE DENSITY DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def audit_edge_density(max_graphs: int = 0) -> Dict[str, object]:
    """
    Histogram of edge counts across all .pt graph files.
    Flags if most graphs are using the MIN_EDGES_PER_GRAPH floor.

    Args:
        max_graphs: If > 0, cap the number of graphs scanned (for speed).

    Returns:
        Dict with summary statistics.
    """
    logger.info("\n" + "=" * 70)
    logger.info("GRAPH EDGE DENSITY DISTRIBUTION")
    logger.info("=" * 70)

    graph_files = sorted(CAUSAL_GRAPHS_DIR.glob("*.pt"))
    if not graph_files:
        logger.error(f"No graphs found in {CAUSAL_GRAPHS_DIR}")
        return {}

    if max_graphs > 0:
        graph_files = graph_files[:max_graphs]

    logger.info(f"Scanning {len(graph_files)} graphs...")
    edge_counts: List[int] = []
    zero_edge_count = 0
    floor_count = 0  # hitting MIN_EDGES_PER_GRAPH exactly

    for gf in graph_files:
        try:
            data = torch.load(gf, weights_only=False)
            if isinstance(data, dict) and "adj" in data:
                adj = data["adj"]
                n_edges = int((adj.abs() > 0).sum().item())
            elif hasattr(data, "edge_index"):
                n_edges = int(data.edge_index.shape[1])
            else:
                continue
            edge_counts.append(n_edges)
            if n_edges == 0:
                zero_edge_count += 1
            elif n_edges == MIN_EDGES_PER_GRAPH:
                floor_count += 1
        except Exception as e:
            logger.warning(f"  Could not load {gf.name}: {e}")

    if not edge_counts:
        logger.error("Could not read any graph files")
        return {}

    arr = np.array(edge_counts)
    pct_floor = 100.0 * floor_count / len(arr)
    pct_zero = 100.0 * zero_edge_count / len(arr)
    max_possible = NUM_LOBES * (NUM_LOBES - 1)  # directed: 132

    logger.info(f"\n  Graphs scanned    : {len(arr)}")
    logger.info(f"  Min edges         : {arr.min()}")
    logger.info(f"  Max edges         : {arr.max()} (max possible: {max_possible})")
    logger.info(f"  Mean edges        : {arr.mean():.1f}")
    logger.info(f"  Median edges      : {np.median(arr):.0f}")
    logger.info(f"  Std edges         : {arr.std():.1f}")
    logger.info(f"  Zero-edge graphs  : {zero_edge_count} ({pct_zero:.1f}%)")
    logger.info(f"  At floor ({MIN_EDGES_PER_GRAPH:2d} edges): {floor_count} ({pct_floor:.1f}%)")

    # ASCII histogram
    bins = [0, 12, 24, 36, 48, 64, 80, 96, 112, 132]
    logger.info("\n  Edge count distribution:")
    for lo, hi in zip(bins[:-1], bins[1:]):
        count = ((arr >= lo) & (arr < hi)).sum()
        bar = "█" * (count * 40 // max(len(arr), 1))
        logger.info(f"    [{lo:3d}-{hi:3d}): {bar} {count}")

    # Diagnosis
    if pct_floor > 50:
        logger.error(
            f"\n  ✗ {pct_floor:.0f}% of graphs are at the minimum edge floor "
            f"({MIN_EDGES_PER_GRAPH}). Granger sparsification is eliminating nearly "
            f"all edges — causality is not being used effectively.\n"
            f"  → Consider lowering SPARSITY_QUANTILE (currently {SPARSITY_QUANTILE}) "
            f"or GRANGER_SIGNIFICANCE_LEVEL (currently {GRANGER_SIGNIFICANCE_LEVEL})."
        )
    elif pct_zero > 5:
        logger.warning(
            f"\n  ⚠ {pct_zero:.1f}% zero-edge graphs. These subjects are excluded "
            f"from training — check construct_causal.py sparsification."
        )
    else:
        logger.info(f"\n  ✓ Edge density looks healthy.")

    return {
        "n_graphs": len(arr),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "pct_at_floor": pct_floor,
        "pct_zero": pct_zero,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. FREQUENCY FEATURE fMRI VALIDITY AUDIT
# ─────────────────────────────────────────────────────────────────────────────

def audit_frequency_features() -> None:
    """
    Check whether fMRI-adapted frequency bands are valid given the actual TRs
    in the ABIDE phenotype file.

    Verifies:
    - Sampling frequency fs = 1/TR (ABIDE: mostly 2.0–2.5 s → 0.4–0.5 Hz)
    - Nyquist limit = fs/2
    - Gamma band (0.20–0.25 Hz) proximity to Nyquist
    - Recommendation on dropping/merging gamma band
    """
    logger.info("\n" + "=" * 70)
    logger.info("FREQUENCY FEATURE fMRI VALIDITY AUDIT")
    logger.info("=" * 70)

    from src.features.frequency_features import extract_band_power

    # ── Load actual TRs from manifest ────────────────────────────────────────
    if MASTER_MANIFEST.exists():
        manifest = pd.read_csv(MASTER_MANIFEST)
        if "TR" in manifest.columns:
            tr_vals = manifest["TR"].dropna()
            logger.info(f"\n  TR (seconds) from master manifest ({len(tr_vals)} subjects):")
            logger.info(f"    Min  : {tr_vals.min():.3f} s")
            logger.info(f"    Max  : {tr_vals.max():.3f} s")
            logger.info(f"    Mean : {tr_vals.mean():.3f} s")
            logger.info(f"    Mode : {tr_vals.mode().iloc[0]:.3f} s")
            med_tr = float(tr_vals.median())
        else:
            logger.warning("  'TR' column not found in master_manifest.csv — using DEFAULT_TR")
            med_tr = DEFAULT_TR
    else:
        logger.warning(f"  Manifest not found: {MASTER_MANIFEST} — using DEFAULT_TR={DEFAULT_TR}")
        med_tr = DEFAULT_TR

    # ── Band validity check ───────────────────────────────────────────────────
    bands_to_check = {
        "delta" : (0.01,  0.027),
        "theta" : (0.027, 0.073),
        "alpha" : (0.073, 0.15),
        "beta"  : (0.15,  0.20),
        "gamma" : (0.20,  0.25),
    }

    logger.info(f"\n  Checking bands against TR={med_tr:.2f}s (median):")
    logger.info(f"  {'Band':<8} {'Low Hz':>8} {'High Hz':>9} {'< Nyquist?':>12} {'Status'}")
    logger.info("  " + "-" * 60)

    issues = []
    for name, (lo, hi) in bands_to_check.items():
        fs = 1.0 / med_tr
        nyquist = fs / 2.0
        valid = hi <= nyquist
        status = "✓ valid" if valid else "✗ EXCEEDS NYQUIST"
        margin = nyquist - hi
        if not valid:
            issues.append(name)
        elif margin < 0.02:
            status = f"⚠ marginal (margin={margin:.4f} Hz)"
            issues.append(name)
        logger.info(f"  {name:<8} {lo:>8.3f} {hi:>9.3f} {str(valid):>12}   {status}")

    logger.info(f"\n  fs (median TR) = {1/med_tr:.4f} Hz, Nyquist = {1/(2*med_tr):.4f} Hz")

    if issues:
        logger.warning(f"\n  ⚠ Marginal/invalid bands: {issues}")
        logger.warning(
            "  Recommendation: For ABIDE with TR≈2s (fs=0.5 Hz, Nyquist=0.25 Hz),\n"
            "  the 'gamma' band (0.20–0.25 Hz) is right at the Nyquist limit.\n"
            "  Power estimates will be unreliable. Options:\n"
            "    1. Merge gamma into beta (0.15–0.25 Hz)\n"
            "    2. Drop gamma_power and gamma_peak features entirely\n"
            "    3. Apply anti-aliasing filter before Welch PSD"
        )
    else:
        logger.info("  ✓ All bands are valid for the median TR in this dataset")

    # ── Functional test with synthetic signal ─────────────────────────────────
    logger.info("\n  Functional test (synthetic 0.10 Hz sine, TR=2s, 200 TPs):")
    fs_test = 1.0 / 2.0
    t = np.arange(200) * 2.0
    test_signal = np.sin(2 * np.pi * 0.10 * t)  # 0.10 Hz → should land in alpha band
    feats = extract_band_power(test_signal, fs=fs_test)
    expected_max_band = "alpha"
    actual_max = max(
        ((b + "_power", feats.get(b + "_power", 0.0)) for b in ["delta", "theta", "alpha", "beta", "gamma"]),
        key=lambda kv: kv[1],
    )
    logger.info(f"    Band powers: " + ", ".join(
        f"{b}={feats.get(b+'_power', 0):.4f}" for b in ["delta", "theta", "alpha", "beta", "gamma"]
    ))
    if expected_max_band in actual_max[0]:
        logger.info(f"    ✓ Peak band correctly identified as '{expected_max_band}'")
    else:
        logger.warning(f"    ⚠ Expected peak in '{expected_max_band}', got '{actual_max[0]}'")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Feature pipeline diagnostic tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sample", type=int, default=3,
                        help="Number of graph files to audit (default: 3)")
    parser.add_argument("--quick", action="store_true",
                        help="Skip edge density scan (faster for large datasets)")
    parser.add_argument("--subject", type=str, default=None,
                        help="Specific subject ID to audit (Granger section)")
    parser.add_argument("--max-graphs", type=int, default=0,
                        help="Cap graphs scanned for edge density (0 = all)")
    args = parser.parse_args()

    # ── 1 + 2. Feature tensor audit (via dataset, authoritative) ──────────────
    all_ok = audit_feature_tensor_via_dataset(n_samples=args.sample)

    # Also audit a raw .pt file to show adj-level details
    graph_files = sorted(CAUSAL_GRAPHS_DIR.glob("*.pt"))
    if graph_files:
        audit_feature_tensor(graph_files[0])

    # ── 2. Granger validation ─────────────────────────────────────────────────
    validate_granger_edges(
        subject_id=args.subject,
        n_subjects=args.sample,
    )

    # ── 3. Edge density ───────────────────────────────────────────────────────
    if not args.quick:
        audit_edge_density(max_graphs=args.max_graphs)

    # ── 4. Frequency bands ───────────────────────────────────────────────────
    audit_frequency_features()

    logger.info("\n" + "=" * 70)
    logger.info("DIAGNOSTIC COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
