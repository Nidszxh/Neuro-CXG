"""
Developer audit tool for Neuro-CXG.

Consolidates the former separate dev-only CLI tools:
  - code_audit.py        (CodeAuditor — static checks for hardcoded constants / bad imports)
  - feature_diagnostics.py (runtime checks — feature tensors, Granger edges, edge density, freq bands)

Usage
-----
    python -m src.validation.dev_audit                    # run code-audit only (fast, no data needed)
    python -m src.validation.dev_audit --features         # run feature-pipeline diagnostics
    python -m src.validation.dev_audit --all              # run both
    python -m src.validation.dev_audit --features --quick # skip edge-density histogram
    python -m src.validation.dev_audit --features --sample 5 --subject <sub_id>
"""

import argparse
import ast
import logging
import re
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
    DATA_FINAL,
    DEFAULT_TR,
    FEATURE_GROUPS,
    GNN_IN_CHANNELS,
    GRANGER_MAX_LAG,
    GRANGER_SIGNIFICANCE_LEVEL,
    MASTER_MANIFEST,
    MIN_EDGES_PER_GRAPH,
    NODE_ATTRIBUTES_HARMONIZED,
    NUM_LOBES,
    SPARSITY_QUANTILE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

# Feature-group index slices for the 28-dim node feature vector
_GROUP_SLICES: Dict[str, slice] = {}
_offset = 0
for _grp, _feats in FEATURE_GROUPS.items():
    _GROUP_SLICES[_grp] = slice(_offset, _offset + len(_feats))
    _offset += len(_feats)

# Constants that should NOT be hardcoded in source files
_CONFIG_CONSTANTS = {
    "NUM_LOBES": 12,
    "NUM_TEMPORAL_FEATURES": 20,
    "NUM_SPATIAL_FEATURES": 6,
    "NUM_INTERNAL_FEATURES": 2,
    "GNN_IN_CHANNELS": 28,
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: CODE AUDIT  (static analysis — no data files required)
# ══════════════════════════════════════════════════════════════════════════════

class CodeAuditor:
    """Static-analysis checks: hardcoded dimensions, missing config imports, legacy shapes."""

    def __init__(self) -> None:
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []
        self.total_files: int = 0

    def check_file(self, filepath: Path) -> None:
        try:
            content = filepath.read_text()
        except Exception as exc:
            self.warnings.append(f"{filepath.relative_to(SRC_DIR)}: cannot read — {exc}")
            return

        self._check_hardcoded_dimensions(filepath, content)
        self._check_hardcoded_lobe_names(filepath, content)
        self._check_config_imports(filepath, content)
        self._check_shape_comments(filepath, content)
        try:
            ast.parse(content)
        except SyntaxError as exc:
            self.errors.append(
                f"{filepath.relative_to(SRC_DIR)}: Syntax error at line {exc.lineno}: {exc.msg}"
            )

    def _check_hardcoded_dimensions(self, filepath: Path, content: str) -> None:
        rel = filepath.relative_to(SRC_DIR)
        patterns = [
            (r"\(5\s*,\s*8\)", "Shape (5, 8) is legacy 5-lobe; use (NUM_LOBES, NUM_TEMPORAL_FEATURES)"),
            (r"\(5\s*,\s*6\)", "Shape (5, 6) is legacy 5-lobe; use (NUM_LOBES, NUM_SPATIAL_FEATURES)"),
            (r"\(5\s*,\s*14\)", "Shape (5, 14) is legacy; use (NUM_LOBES, GNN_IN_CHANNELS)"),
            (r"\(12\s*,\s*14\)", "Shape (12, 14) is outdated — use (NUM_LOBES, GNN_IN_CHANNELS)=(12, 28)"),
            (r"\.reshape\s*\(\s*-1\s*,\s*5\s*,\s*14\)", "reshape(-1, 5, 14) should use NUM_LOBES + GNN_IN_CHANNELS"),
            (r"\.reshape\s*\(\s*-1\s*,\s*5\s*,\s*8\)", "reshape(-1, 5, 8) should use NUM_LOBES + NUM_TEMPORAL_FEATURES"),
        ]
        for pattern, msg in patterns:
            for m in re.finditer(pattern, content):
                line_no = content[: m.start()].count("\n") + 1
                line = content.split("\n")[line_no - 1]
                if not line.strip().startswith("#"):
                    self.warnings.append(f"{rel}:{line_no}: {msg}")

    def _check_hardcoded_lobe_names(self, filepath: Path, content: str) -> None:
        rel = filepath.relative_to(SRC_DIR)
        for m in re.finditer(r"lobe_names\s*=\s*\[\s*'Frontal'\s*,.*?\]", content, re.DOTALL):
            if "LOBE_NAMES" not in content or "#" in m.group():
                continue
            line_no = content[: m.start()].count("\n") + 1
            self.warnings.append(f"{rel}:{line_no}: Use LOBE_NAMES from config rather than hardcoded list")

    def _check_config_imports(self, filepath: Path, content: str) -> None:
        rel = filepath.relative_to(SRC_DIR)
        needs_config = [
            "construct_causal.py", "extract_spatial.py", "gnn_model.py",
            "graph_factory.py", "fold_safe_harmonization.py", "causal_gnn.py",
        ]
        if any(name in str(filepath) for name in needs_config):
            uses_config_names = "NUM_LOBES" in content or "LOBE_NAMES" in content
            imports_config = "from src.core.config import" in content or "from .core.config import" in content
            if uses_config_names and not imports_config:
                self.warnings.append(
                    f"{rel}: References NUM_LOBES/LOBE_NAMES but doesn't import from config"
                )

    def _check_shape_comments(self, filepath: Path, content: str) -> None:
        rel = filepath.relative_to(SRC_DIR)
        patterns = [
            (r"#.*\(5.*8\).*", "5×8 shape in comment"),
            (r"#.*5 lobe", "5-lobe reference in comment"),
            (r"#.*5 node", "5-node reference in comment"),
        ]
        for pattern, msg in patterns:
            for m in re.finditer(pattern, content, re.IGNORECASE):
                line_no = content[: m.start()].count("\n") + 1
                self.info.append(f"{rel}:{line_no}: May reference old architecture — {msg}")

    def print_report(self) -> bool:
        logger.info("=" * 70)
        logger.info("CODE AUDIT REPORT  (%d files scanned)", self.total_files)
        logger.info("=" * 70)
        if self.errors:
            for e in self.errors:
                logger.error("  ERROR: %s", e)
        else:
            logger.info("  ERRORS: none")
        if self.warnings:
            for w in self.warnings:
                logger.warning("  WARN:  %s", w)
        else:
            logger.info("  WARNINGS: none")
        if self.info:
            for msg in self.info[:10]:
                logger.info("  INFO:  %s", msg)
            if len(self.info) > 10:
                logger.info("  INFO:  … and %d more", len(self.info) - 10)
        logger.info("=" * 70)
        return len(self.errors) == 0


def run_code_audit() -> int:
    """Scan all Python files under src/ and return 0 if no errors found."""
    auditor = CodeAuditor()
    files = [p for p in SRC_DIR.rglob("*.py") if "__pycache__" not in str(p)]
    auditor.total_files = len(files)
    logger.info("Code-auditing %d Python files…", len(files))
    for f in sorted(files):
        auditor.check_file(f)
    return 0 if auditor.print_report() else 1


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: FEATURE-PIPELINE DIAGNOSTICS  (runtime — needs data files)
# ══════════════════════════════════════════════════════════════════════════════

def audit_feature_tensor(graph_path: Path) -> bool:
    """
    Load a single graph .pt file and print shape/value ranges per feature group.
    Returns True if all checks pass.
    """
    logger.info("=" * 70)
    logger.info("FEATURE TENSOR AUDIT: %s", graph_path.name)
    logger.info("=" * 70)
    if not graph_path.exists():
        logger.error("Graph not found: %s", graph_path)
        return False

    data = torch.load(graph_path, weights_only=False)
    if hasattr(data, "x"):
        x = data.x
    elif isinstance(data, dict) and "x" in data:
        x = data["x"]
    elif isinstance(data, dict) and "adj" in data:
        logger.warning("Graph is in adj-dict format — no 'x' tensor (assembled by graph_factory).")
        _audit_adj_dict(data)
        return True
    else:
        logger.error("Unrecognised graph format: %s", type(data))
        return False

    if x is None:
        logger.error("x tensor is None")
        return False

    expected = (NUM_LOBES, GNN_IN_CHANNELS)
    ok = True
    logger.info("Node feature tensor shape: %s", tuple(x.shape))
    if tuple(x.shape) != expected:
        logger.error("  ✗ Expected %s, got %s", expected, tuple(x.shape))
        ok = False
    else:
        logger.info("  ✓ Shape matches %s", expected)

    logger.info("%-12s %-14s %10s %10s %10s %6s %6s", "Group", "Indices", "Min", "Max", "Mean", "NaN", "~Zero")
    logger.info("-" * 70)
    for grp, sl in _GROUP_SLICES.items():
        cols = x[:, sl]
        nan_c = torch.isnan(cols).sum().item()
        inf_c = torch.isinf(cols).sum().item()
        zero_c = (cols.abs() < 1e-8).sum().item()
        total = cols.numel()
        if nan_c or inf_c:
            logger.warning("  ✗ %s: %d NaN, %d Inf values!", grp, nan_c, inf_c)
            ok = False
        all_zero = zero_c == total
        if all_zero:
            logger.warning("  ✗ %s: ALL values are zero (silent zero-padding)", grp)
            ok = False
        idx_str = f"[{sl.start}:{sl.stop}]"
        mi = cols.min().item() if not all_zero else 0.0
        ma = cols.max().item() if not all_zero else 0.0
        me = cols.mean().item() if not all_zero else 0.0
        logger.info("  %-10s %-14s %10.4f %10.4f %10.4f %6d %6d/%d",
                    grp, idx_str, mi, ma, me, nan_c, zero_c, total)

    if hasattr(data, "edge_index"):
        n_edges = data.edge_index.shape[1]
        logger.info("Edges: %d", n_edges)
        if n_edges == 0:
            logger.error("  ✗ Zero edges — disconnected graph"); ok = False
        elif n_edges < MIN_EDGES_PER_GRAPH:
            logger.warning("  ⚠ Below minimum floor (%d)", MIN_EDGES_PER_GRAPH)
        else:
            logger.info("  ✓ Edge count above minimum floor (%d)", MIN_EDGES_PER_GRAPH)

    logger.info("Result: %s", "✓ PASS" if ok else "✗ FAIL")
    return ok


def _audit_adj_dict(data: dict) -> None:
    adj = data["adj"]
    logger.info("  adj shape   : %s", tuple(adj.shape))
    logger.info("  non-zero    : %d", (adj != 0).sum().item())
    logger.info("  weight range: [%.4f, %.4f]", adj.min().item(), adj.max().item())
    if "internal_features" in data:
        intf = data["internal_features"]
        logger.info("  internal_features shape: %s, NaN: %d",
                    tuple(intf.shape), torch.isnan(intf).sum().item())


def audit_feature_tensor_via_dataset(n_samples: int = 3) -> bool:
    """Audit assembled feature tensors through ABIDECausalDataset (authoritative path)."""
    logger.info("=" * 70)
    logger.info("FEATURE TENSOR AUDIT via ABIDECausalDataset")
    logger.info("=" * 70)
    try:
        from src.features.graph_factory import ABIDECausalDataset
    except ImportError as exc:
        logger.error("Cannot import ABIDECausalDataset: %s", exc); return False
    try:
        ds = ABIDECausalDataset(split="train")
    except Exception as exc:
        logger.error("Dataset construction failed: %s", exc); return False
    if not ds:
        logger.error("Train dataset is empty"); return False

    all_ok = True
    for i in range(min(n_samples, len(ds))):
        sample = ds[i]
        if sample is None:
            continue
        sub = getattr(sample, "sub_id", f"idx_{i}")
        logger.info("  Subject: %s  x shape: %s", sub, tuple(sample.x.shape))
        for grp, sl in _GROUP_SLICES.items():
            cols = sample.x[:, sl]
            nan_c = torch.isnan(cols).sum().item()
            zero_c = (cols.abs() < 1e-8).sum().item()
            flag = "⚠ all-zero" if zero_c == cols.numel() else ("✗ has NaN" if nan_c else "✓")
            if "✗" in flag or "⚠" in flag:
                all_ok = False
            logger.info("    %-12s [%d:%d]  min=%.4f  max=%.4f  %s",
                        grp, sl.start, sl.stop, cols.min().item(), cols.max().item(), flag)

    logger.info("Checked %d subjects — %s", min(n_samples, len(ds)),
                "✓ ALL PASS" if all_ok else "✗ ISSUES FOUND")
    return all_ok


def validate_granger_edges(subject_id: Optional[str] = None, n_subjects: int = 5) -> None:
    """Print causal matrix stats for sample subjects to verify non-trivial edge weights."""
    logger.info("=" * 70)
    logger.info("GRANGER CAUSALITY EDGE VALIDATION")
    logger.info("=" * 70)
    if not MASTER_MANIFEST.exists():
        logger.error("Manifest not found: %s", MASTER_MANIFEST); return

    manifest = pd.read_csv(MASTER_MANIFEST)
    subjects = [subject_id] if subject_id else (
        manifest.sample(min(n_subjects, len(manifest)), random_state=42)["subject_id"].tolist()
    )
    expected_min = -np.log10(GRANGER_SIGNIFICANCE_LEVEL)   # 1.301 for p=0.05

    for sub_id in subjects:
        gp = CAUSAL_GRAPHS_DIR / f"{sub_id}_graph.pt"
        if not gp.exists():
            logger.warning("  %s: graph not found", sub_id); continue
        data = torch.load(gp, weights_only=False)
        if not (isinstance(data, dict) and "adj" in data):
            logger.warning("  %s: unexpected graph format", sub_id); continue
        adj = data["adj"]
        n_total = adj.numel()
        n_nz = (adj.abs() > 0).sum().item()
        adj_vals = adj[adj != 0]
        max_val = adj.abs().max().item()
        is_all_same = (adj_vals.std().item() < 1e-6) if len(adj_vals) > 1 else True

        logger.info("  %s:", sub_id)
        logger.info("    adj shape    : %s", tuple(adj.shape))
        logger.info("    non-zero     : %d/%d (%.1f%%)", n_nz, n_total, 100 * n_nz / max(n_total, 1))
        logger.info("    weight range : [%.4f, %.4f]", adj.min().item(), adj.max().item())
        if max_val < 1e-6:
            logger.error("    ✗ All edge weights are zero — Granger test silent failure!")
        elif is_all_same and len(adj_vals) > 1:
            logger.warning("    ⚠ All weights identical — Granger may have fallen back to lagged Pearson")
        else:
            strong = (adj_vals > expected_min).sum().item()
            logger.info("    significant edges (>%.2f): %d/%d  ✓", expected_min, strong, n_nz)


def audit_edge_density(max_graphs: int = 0) -> Dict[str, object]:
    """Histogram of edge counts across all .pt graph files."""
    logger.info("=" * 70)
    logger.info("GRAPH EDGE DENSITY DISTRIBUTION")
    logger.info("=" * 70)
    graph_files = sorted(CAUSAL_GRAPHS_DIR.glob("*.pt"))
    if not graph_files:
        logger.error("No graphs found in %s", CAUSAL_GRAPHS_DIR); return {}
    if max_graphs > 0:
        graph_files = graph_files[:max_graphs]

    logger.info("Scanning %d graphs…", len(graph_files))
    counts: List[int] = []
    zero_cnt = floor_cnt = 0
    for gf in graph_files:
        try:
            data = torch.load(gf, weights_only=False)
            if isinstance(data, dict) and "adj" in data:
                n = int((data["adj"].abs() > 0).sum().item())
            elif hasattr(data, "edge_index"):
                n = int(data.edge_index.shape[1])
            else:
                continue
            counts.append(n)
            zero_cnt += n == 0
            floor_cnt += n == MIN_EDGES_PER_GRAPH
        except Exception as exc:
            logger.warning("  Could not load %s: %s", gf.name, exc)

    if not counts:
        logger.error("Could not read any graph files"); return {}

    arr = np.array(counts)
    max_possible = NUM_LOBES * (NUM_LOBES - 1)
    pct_floor = 100.0 * floor_cnt / len(arr)
    pct_zero = 100.0 * zero_cnt / len(arr)

    logger.info("  Graphs : %d  |  min=%d  max=%d/%d  mean=%.1f  median=%.0f  std=%.1f",
                len(arr), arr.min(), arr.max(), max_possible, arr.mean(),
                np.median(arr), arr.std())
    logger.info("  Zero-edge: %d (%.1f%%)  At floor (%d): %d (%.1f%%)",
                zero_cnt, pct_zero, MIN_EDGES_PER_GRAPH, floor_cnt, pct_floor)

    bins = [0, 12, 24, 36, 48, 64, 80, 96, 112, 132]
    logger.info("  Distribution:")
    for lo, hi in zip(bins[:-1], bins[1:]):
        cnt = int(((arr >= lo) & (arr < hi)).sum())
        bar = "█" * (cnt * 40 // max(len(arr), 1))
        logger.info("    [%3d-%3d): %s %d", lo, hi, bar, cnt)

    if pct_floor > 50:
        logger.error("%.0f%% at minimum edge floor — Granger sparsification too aggressive. "
                     "Lower SPARSITY_QUANTILE (%.2f) or GRANGER_SIGNIFICANCE_LEVEL (%.3f).",
                     pct_floor, SPARSITY_QUANTILE, GRANGER_SIGNIFICANCE_LEVEL)
    elif pct_zero > 5:
        logger.warning("%.1f%% zero-edge graphs — check construct_causal.py sparsification.", pct_zero)
    else:
        logger.info("  ✓ Edge density looks healthy.")

    return dict(n_graphs=len(arr), mean=float(arr.mean()), median=float(np.median(arr)),
                std=float(arr.std()), min=int(arr.min()), max=int(arr.max()),
                pct_at_floor=pct_floor, pct_zero=pct_zero)


def audit_frequency_features() -> None:
    """Check fMRI frequency band validity against actual TR values in the manifest."""
    logger.info("=" * 70)
    logger.info("FREQUENCY FEATURE fMRI VALIDITY AUDIT")
    logger.info("=" * 70)
    from src.features.extract_temporal import extract_band_power

    if MASTER_MANIFEST.exists():
        manifest = pd.read_csv(MASTER_MANIFEST)
        if "TR" in manifest.columns:
            tr = manifest["TR"].dropna()
            logger.info("TR (s): min=%.3f  max=%.3f  mean=%.3f  mode=%.3f",
                        tr.min(), tr.max(), tr.mean(), tr.mode().iloc[0])
            med_tr = float(tr.median())
        else:
            logger.warning("'TR' column missing from manifest — using DEFAULT_TR=%.2f", DEFAULT_TR)
            med_tr = DEFAULT_TR
    else:
        logger.warning("Manifest not found — using DEFAULT_TR=%.2f", DEFAULT_TR)
        med_tr = DEFAULT_TR

    bands = dict(delta=(0.01, 0.027), theta=(0.027, 0.073),
                 alpha=(0.073, 0.15), beta=(0.15, 0.20), gamma=(0.20, 0.25))
    fs = 1.0 / med_tr
    nyquist = fs / 2.0
    logger.info("fs=%.4f Hz  Nyquist=%.4f Hz  (median TR=%.2f s)", fs, nyquist, med_tr)
    logger.info("%-8s %9s %10s %13s  %s", "Band", "Low Hz", "High Hz", "< Nyquist?", "Status")
    logger.info("-" * 60)
    issues = []
    for name, (lo, hi) in bands.items():
        valid = hi <= nyquist
        margin = nyquist - hi
        if not valid:
            status = "✗ EXCEEDS NYQUIST"
            issues.append(name)
        elif margin < 0.02:
            status = f"⚠ marginal (margin={margin:.4f} Hz)"
            issues.append(name)
        else:
            status = "✓ valid"
        logger.info("%-8s %9.3f %10.3f %13s  %s", name, lo, hi, str(valid), status)

    if issues:
        logger.warning("Marginal/invalid bands: %s. Consider merging or dropping 'gamma' for ABIDE "
                       "(TR≈2s → Nyquist=0.25 Hz is right at the gamma upper bound).", issues)
    else:
        logger.info("✓ All bands valid for median TR in this dataset.")

    # Functional test: 0.10 Hz sine should peak in alpha band
    t = np.arange(200) * 2.0
    test_sig = np.sin(2 * np.pi * 0.10 * t)
    feats = extract_band_power(test_sig, fs=0.5)
    actual_max = max(((b, feats.get(f"{b}_power", 0.0)) for b in bands), key=lambda kv: kv[1])
    logger.info("Functional test (0.10 Hz sine, TR=2s): peak band = '%s'  %s",
                actual_max[0], "✓" if "alpha" in actual_max[0] else "⚠ unexpected")


def run_feature_diagnostics(
    n_samples: int = 3,
    quick: bool = False,
    subject_id: Optional[str] = None,
    max_graphs: int = 0,
) -> None:
    """Run the full feature-pipeline diagnostic suite."""
    audit_feature_tensor_via_dataset(n_samples=n_samples)
    graph_files = sorted(CAUSAL_GRAPHS_DIR.glob("*.pt"))
    if graph_files:
        audit_feature_tensor(graph_files[0])
    validate_granger_edges(subject_id=subject_id, n_subjects=n_samples)
    if not quick:
        audit_edge_density(max_graphs=max_graphs)
    audit_frequency_features()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: UNIFIED ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--all", action="store_true", help="Run both code-audit and feature diagnostics")
    mode.add_argument("--features", action="store_true", help="Run feature-pipeline diagnostics only")
    # If neither flag is given, code-audit runs by default.
    parser.add_argument("--sample", type=int, default=3,
                        help="Graph files to audit per feature test (default: 3)")
    parser.add_argument("--quick", action="store_true",
                        help="Skip edge-density histogram scan")
    parser.add_argument("--subject", type=str, default=None,
                        help="Specific subject ID for Granger edge audit")
    parser.add_argument("--max-graphs", type=int, default=0,
                        help="Cap graphs scanned for edge-density (0 = all)")
    args = parser.parse_args()

    exit_code = 0

    if not args.features:   # code-audit unless user explicitly asked for features-only
        exit_code |= run_code_audit()

    if args.features or args.all:
        run_feature_diagnostics(
            n_samples=args.sample,
            quick=args.quick,
            subject_id=args.subject,
            max_graphs=args.max_graphs,
        )

    logger.info("=" * 70)
    logger.info("AUDIT COMPLETE")
    logger.info("=" * 70)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
