import argparse
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ── Project root on path ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import (
    CAUSAL_GRAPHS_DIR,
    MASTER_MANIFEST,
    NUM_LOBES,
    GNN_IN_CHANNELS,
    FEATURE_GROUPS,
    ALL_FEATURE_NAMES,
    MIN_EDGES_PER_GRAPH,
    DEFAULT_TR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("diagnose_features")

# ── Feature-group slice indices (into the 28-dim node feature vector) ─────────
_GROUPS = {
    "temporal":  FEATURE_GROUPS["temporal"],
    "frequency": FEATURE_GROUPS["frequency"],
    "internal":  FEATURE_GROUPS["internal"],
    "spatial":   FEATURE_GROUPS["spatial"],
}

_GROUP_SLICES: dict[str, tuple[int, int]] = {}
_offset = 0
for _gname in ("temporal", "frequency", "internal", "spatial"):
    _size = len(_GROUPS[_gname])
    _GROUP_SLICES[_gname] = (_offset, _offset + _size)
    _offset += _size


# ─────────────────────────────────────────────────────────────────────────────
#  1.  Single-graph feature audit
# ─────────────────────────────────────────────────────────────────────────────

def audit_single_graph(graph_path: Path) -> bool:
    """
    Load one .pt graph file and print per-feature-group statistics.

    Returns True if all checks pass, False otherwise.
    """
    logger.info("=" * 70)
    logger.info("SINGLE-GRAPH FEATURE AUDIT: %s", graph_path.name)
    logger.info("=" * 70)

    try:
        g = torch.load(graph_path, weights_only=False)
    except Exception as exc:
        logger.error("Failed to load %s: %s", graph_path, exc)
        return False

    # ── Basic shape validation ────────────────────────────────────────────────
    if "adj" not in g:
        logger.error("Key 'adj' missing from graph dict — graph file may be corrupt.")
        return False

    adj: torch.Tensor = g["adj"]
    logger.info("\n[Graph structure]")
    logger.info("  adj shape          : %s", list(adj.shape))
    logger.info("  adj dtype          : %s", adj.dtype)
    logger.info("  non-zero edges     : %d", (adj != 0).sum().item())
    logger.info("  adj value range    : [%.4f, %.4f]", adj.min().item(), adj.max().item())

    # Edge weight distribution (Granger: should be -log10(p) ≥ 0 mostly positive)
    edge_vals = adj[adj != 0]
    if edge_vals.numel() > 0:
        logger.info("  edge weight  mean  : %.4f", edge_vals.mean().item())
        logger.info("  edge weight  std   : %.4f", edge_vals.std().item())
        pct_positive = (edge_vals > 0).float().mean().item() * 100
        pct_negative = (edge_vals < 0).float().mean().item() * 100
        logger.info("  %% positive weights: %.1f%%  (expected ~100%% for Granger -log10(p))",
                    pct_positive)
        logger.info("  %% negative weights: %.1f%%  (expected ~0%% for Granger)", pct_negative)

        all_zeros = edge_vals.abs().max().item() < 1e-9
        if all_zeros:
            logger.error("  *** ALL EDGE WEIGHTS ARE ZERO — Granger test may have silently failed! ***")
            return False
        else:
            logger.info("  ✓ Edge weights are non-trivial (Granger appears active)")

    # ── Internal features ─────────────────────────────────────────────────────
    if "internal_features" in g:
        ifeats: torch.Tensor = g["internal_features"]
        logger.info("\n[Internal features (PCA eigenvariate + ReHo)]")
        logger.info("  shape              : %s", list(ifeats.shape))
        logger.info("  value range        : [%.4f, %.4f]", ifeats.min().item(), ifeats.max().item())
        nan_count = torch.isnan(ifeats).sum().item()
        inf_count = torch.isinf(ifeats).sum().item()
        logger.info("  NaN count          : %d", nan_count)
        logger.info("  Inf count          : %d", inf_count)
        if nan_count > 0 or inf_count > 0:
            logger.warning("  *** Internal features contain NaN/Inf — check construct_causal.py ***")
    else:
        logger.warning("[Internal features] key 'internal_features' missing from graph — will be zeros in loader.")

    # ── Node feature tensor (if present; reconstructed by graph_factory.py) ──
    # The stored .pt file has 'adj' + 'internal_features'; node features are
    # assembled on-the-fly in graph_factory.py. We audit those below via the
    # dataset loader instead. Still check for unexpected extra keys:
    known_keys = {"adj", "internal_features", "subject_id", "label",
                  "dx_group", "site_id", "x", "edge_index", "edge_attr", "y", "pos"}
    extra_keys = set(g.keys()) - known_keys
    if extra_keys:
        logger.info("\n[Additional keys in graph dict]: %s", extra_keys)

    # If the graph already has node features (x), audit them directly
    if "x" in g:
        x: torch.Tensor = g["x"]
        _audit_node_features(x, source="stored in .pt")
    else:
        logger.info("\n[Node features] Not stored in graph (assembled by graph_factory.py — run --dataset-loader audit)")

    logger.info("\n✓  Single-graph audit complete for %s", graph_path.name)
    return True


def _audit_node_features(x: torch.Tensor, source: str = "") -> None:
    """Print per-group statistics for a (NUM_LOBES, GNN_IN_CHANNELS) node tensor."""
    logger.info("\n[Node features%s]", f" — {source}" if source else "")
    logger.info("  shape           : %s  (expected [%d, %d])", list(x.shape), NUM_LOBES, GNN_IN_CHANNELS)

    shape_ok = x.shape == (NUM_LOBES, GNN_IN_CHANNELS)
    if not shape_ok:
        logger.error("  *** SHAPE MISMATCH! Expected (%d, %d), got %s ***",
                     NUM_LOBES, GNN_IN_CHANNELS, list(x.shape))
    else:
        logger.info("  ✓ Shape correct")

    nan_total = torch.isnan(x).sum().item()
    inf_total = torch.isinf(x).sum().item()
    all_zero_rows = (x.abs().sum(dim=1) == 0).sum().item()
    logger.info("  NaN total       : %d", nan_total)
    logger.info("  Inf total       : %d", inf_total)
    logger.info("  all-zero rows   : %d / %d  (could indicate silent zero-padding)", all_zero_rows, NUM_LOBES)

    if nan_total > 0 or inf_total > 0:
        logger.warning("  *** NaN/Inf detected — check upstream feature extraction! ***")

    # Per-group breakdown
    for group_name, (start, end) in _GROUP_SLICES.items():
        if end > x.shape[1]:
            logger.warning("  Group '%s' slice [%d:%d] exceeds feature dim %d — skipping",
                           group_name, start, end, x.shape[1])
            continue
        group_x = x[:, start:end]
        g_nan = torch.isnan(group_x).sum().item()
        g_inf = torch.isinf(group_x).sum().item()
        g_zero_pct = (group_x == 0).float().mean().item() * 100
        logger.info(
            "  [%-10s] cols %2d–%2d  |  min=%7.4f  max=%7.4f  mean=%7.4f  std=%6.4f  "
            "NaN=%d  Inf=%d  zero%%=%.1f%%",
            group_name, start, end - 1,
            group_x.min().item(), group_x.max().item(),
            group_x.mean().item(), group_x.std().item(),
            g_nan, g_inf, g_zero_pct,
        )
        if g_zero_pct > 90:
            logger.warning("    *** '%s' group is >90%% zeros — possible missing features! ***",
                           group_name)


# ─────────────────────────────────────────────────────────────────────────────
#  2.  Dataset-loader feature audit (assembles features via graph_factory)
# ─────────────────────────────────────────────────────────────────────────────

def audit_via_dataset_loader(subject_id: str = None, n_samples: int = 5) -> bool:
    """
    Load graphs through ABIDECausalDataset and audit assembled node features.
    This exercises the full pipeline including harmonized CSV + spatial coords.
    """
    logger.info("\n" + "=" * 70)
    logger.info("DATASET-LOADER AUDIT (full pipeline, %d samples)", n_samples)
    logger.info("=" * 70)

    try:
        from src.features.graph_factory import ABIDECausalDataset
    except Exception as exc:
        logger.error("Cannot import ABIDECausalDataset: %s", exc)
        return False

    try:
        ds = ABIDECausalDataset(split="train")
    except Exception as exc:
        logger.error("Dataset init failed: %s", exc)
        return False

    if len(ds) == 0:
        logger.warning("Dataset is empty — no valid graphs found.")
        return False

    logger.info("Dataset has %d subjects in 'train' split.", len(ds))

    # Find subject index if requested
    indices = []
    if subject_id is not None:
        for i in range(len(ds)):
            if str(ds.manifest.iloc[i]["subject_id"]) == subject_id:
                indices = [i]
                break
        if not indices:
            logger.warning("Subject '%s' not found in train split — using first %d samples.",
                           subject_id, n_samples)
            indices = list(range(min(n_samples, len(ds))))
    else:
        indices = list(range(min(n_samples, len(ds))))

    all_ok = True
    for idx in indices:
        sample = ds[idx]
        if sample is None:
            logger.warning("  sample[%d] returned None — subject excluded from graph building.", idx)
            all_ok = False
            continue

        sub = sample.sub_id
        label_str = "ASD" if sample.y.item() == 1 else "Control"
        logger.info("\n  ── Subject: %s  |  label: %s  |  edges: %d ──",
                    sub, label_str, sample.edge_index.shape[1])
        _audit_node_features(sample.x, source=f"subject {sub}")

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
#  3.  Edge-count histogram across ALL graphs
# ─────────────────────────────────────────────────────────────────────────────

def audit_edge_distribution() -> dict:
    """
    Load every .pt graph file and compute edge count statistics.

    Reports:
      - Min / Max / Mean / Median edge counts
      - Fraction of graphs hitting the MIN_EDGES_PER_GRAPH floor
      - Simple ASCII histogram
    """
    logger.info("\n" + "=" * 70)
    logger.info("EDGE-COUNT DISTRIBUTION ACROSS ALL GRAPHS")
    logger.info("=" * 70)

    graph_files = list(CAUSAL_GRAPHS_DIR.glob("*_graph.pt"))
    if not graph_files:
        logger.error("No graph files found in %s", CAUSAL_GRAPHS_DIR)
        return {}

    logger.info("Scanning %d graph files in %s …", len(graph_files), CAUSAL_GRAPHS_DIR)

    edge_counts = []
    granger_zero_count = 0
    error_count = 0

    for gf in graph_files:
        try:
            g = torch.load(gf, weights_only=False)
            if "adj" not in g:
                error_count += 1
                continue
            adj: torch.Tensor = g["adj"]
            n_edges = (adj != 0).sum().item()
            edge_counts.append(n_edges)

            # Check if this graph has all-zero edge weights (Granger fallback indicator)
            edge_vals = adj[adj != 0]
            if edge_vals.numel() > 0 and edge_vals.abs().max().item() < 1e-9:
                granger_zero_count += 1
        except Exception:
            error_count += 1

    if not edge_counts:
        logger.error("Could not read any valid graphs.")
        return {}

    counts = np.array(edge_counts)
    at_floor = (counts <= MIN_EDGES_PER_GRAPH).sum()
    floor_pct = at_floor / len(counts) * 100

    logger.info("\n[Edge count statistics]")
    logger.info("  Graphs loaded     : %d  (errors: %d)", len(counts), error_count)
    logger.info("  Min edges         : %d", counts.min())
    logger.info("  Max edges         : %d", counts.max())
    logger.info("  Mean edges        : %.1f", counts.mean())
    logger.info("  Median edges      : %.1f", np.median(counts))
    logger.info("  Std dev           : %.1f", counts.std())
    logger.info("  MIN_EDGES_PER_GRAPH floor : %d", MIN_EDGES_PER_GRAPH)
    logger.info("  Graphs at/below floor  : %d / %d  (%.1f%%)",
                at_floor, len(counts), floor_pct)

    if floor_pct > 50:
        logger.warning(
            "  *** >50%% of graphs hit the minimum-edge floor — Granger statistical\n"
            "      threshold is eliminating most edges and the fallback is dominating.\n"
            "      Consider relaxing SPARSITY_QUANTILE or GRANGER_MAX_LAG."
        )
    else:
        logger.info("  ✓ Most graphs have edges above the floor (Granger is contributing)")

    if granger_zero_count > 0:
        logger.warning(
            "  *** %d graphs have non-zero edge positions but all-zero weights — "
            "possible numerical issue in causality computation. ***",
            granger_zero_count
        )

    # ASCII histogram (10 bins)
    logger.info("\n[Edge count histogram]")
    bins = np.linspace(counts.min(), counts.max() + 1, 11, dtype=int)
    hist, bin_edges = np.histogram(counts, bins=bins)
    bar_max = max(hist)
    scale = 40 / max(bar_max, 1)
    for i, (h, lo, hi) in enumerate(zip(hist, bin_edges[:-1], bin_edges[1:])):
        bar = "█" * int(h * scale)
        logger.info("  [%3d–%3d]  %s (%d)", lo, hi - 1, bar, h)

    return {
        "n_graphs": len(counts),
        "min": int(counts.min()),
        "max": int(counts.max()),
        "mean": float(counts.mean()),
        "median": float(np.median(counts)),
        "std": float(counts.std()),
        "at_floor_pct": float(floor_pct),
        "granger_zero_graphs": granger_zero_count,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  4.  TR / Nyquist / gamma-band audit
# ─────────────────────────────────────────────────────────────────────────────

def audit_tr_and_nyquist() -> None:
    """
    Check TR values from the master manifest and evaluate the validity of the
    gamma frequency band relative to the Nyquist limit.

    For fMRI:
      fs = 1 / TR (sampling frequency)
      Nyquist = fs / 2 = 1 / (2 * TR)

    Standard gamma band defined in frequency_features.py: 0.20–0.25 Hz
    At TR=2s → Nyquist = 0.25 Hz → gamma band sits exactly at the Nyquist limit.
    At TR>2s → gamma band exceeds Nyquist → INVALID frequencies.
    """
    logger.info("\n" + "=" * 70)
    logger.info("TR / NYQUIST / GAMMA-BAND AUDIT")
    logger.info("=" * 70)

    GAMMA_LOW  = 0.20   # Hz — from frequency_features.py
    GAMMA_HIGH = 0.25   # Hz — from frequency_features.py
    BETA_HIGH  = 0.20   # Hz

    if not MASTER_MANIFEST.exists():
        logger.warning("Master manifest not found at %s — using DEFAULT_TR=%.1f s",
                       MASTER_MANIFEST, DEFAULT_TR)
        trs = np.array([DEFAULT_TR])
    else:
        try:
            df = pd.read_csv(MASTER_MANIFEST)
            if "TR" not in df.columns:
                logger.warning("'TR' column missing from manifest — using DEFAULT_TR=%.1f s", DEFAULT_TR)
                trs = np.array([DEFAULT_TR])
            else:
                trs = df["TR"].dropna().values
        except Exception as exc:
            logger.error("Failed to load manifest: %s", exc)
            trs = np.array([DEFAULT_TR])

    unique_trs, counts = np.unique(trs, return_counts=True)

    logger.info("\n[TR distribution in manifest]")
    logger.info("  Total subjects with TR : %d", len(trs))
    for tr_val, cnt in zip(unique_trs, counts):
        fs      = 1.0 / tr_val
        nyquist = fs / 2.0
        gamma_ok = nyquist >= GAMMA_HIGH
        status = "✓ OK" if gamma_ok else "✗ GAMMA EXCEEDS NYQUIST"
        logger.info(
            "  TR=%.3f s  →  fs=%.4f Hz  Nyquist=%.4f Hz  gamma=[%.2f–%.2f Hz]  %s  (%d subjects)",
            tr_val, fs, nyquist, GAMMA_LOW, GAMMA_HIGH, status, cnt,
        )

    # Overall risk assessment
    n_risky = sum(1.0 / tr / 2.0 < GAMMA_HIGH for tr in unique_trs
                  if 1.0 / tr / 2.0 < GAMMA_HIGH)
    risky_subjects = sum(cnt for tr_val, cnt in zip(unique_trs, counts)
                         if 1.0 / tr_val / 2.0 < GAMMA_HIGH)
    risky_pct = risky_subjects / len(trs) * 100 if len(trs) > 0 else 0.0

    logger.info("\n[Gamma-band risk assessment]")
    logger.info("  Gamma band definition : [%.2f – %.2f] Hz", GAMMA_LOW, GAMMA_HIGH)
    logger.info("  Subjects with TR where gamma > Nyquist : %d / %d  (%.1f%%)",
                risky_subjects, len(trs), risky_pct)

    if risky_pct > 0:
        logger.warning(
            "\n  *** RECOMMENDATION: %.1f%% of subjects have TR > 2s, making the\n"
            "      gamma band (%.2f–%.2f Hz) marginal or invalid.\n"
            "      Options:\n"
            "        (a) Drop gamma features for those subjects (set to 0)\n"
            "        (b) Merge gamma into beta: use beta = [%.2f–%.2f Hz]\n"
            "        (c) Filter by TR ≤ 2s before feature extraction\n"
            "      This affects %d subjects in the dataset.",
            risky_pct, GAMMA_LOW, GAMMA_HIGH, BETA_HIGH - 0.05, GAMMA_HIGH,
            risky_subjects,
        )
    else:
        logger.info("  ✓ All subjects have TR ≤ 2s — gamma band is valid (at Nyquist limit)")

    # Also check for unreasonably long TRs (>3s) that compress all bands
    very_long = sum(cnt for tr_val, cnt in zip(unique_trs, counts) if tr_val > 3.0)
    if very_long > 0:
        logger.warning(
            "  *** %d subjects have TR > 3s (fs < 0.33 Hz) — even alpha/beta bands\n"
            "      become marginal. Consider excluding these subjects.",
            very_long,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  5.  Granger causality sample audit
# ─────────────────────────────────────────────────────────────────────────────

def audit_granger_sample(graph_path: Path) -> None:
    """
    Print pre- and post-sparsification edge weight statistics for a single graph.
    Confirms Granger -log10(p) values are non-trivial.
    """
    logger.info("\n" + "=" * 70)
    logger.info("GRANGER CAUSALITY SANITY CHECK: %s", graph_path.name)
    logger.info("=" * 70)

    try:
        g = torch.load(graph_path, weights_only=False)
    except Exception as exc:
        logger.error("Cannot load %s: %s", graph_path, exc)
        return

    if "adj" not in g:
        logger.error("No 'adj' key in graph.")
        return

    adj: torch.Tensor = g["adj"]
    flat = adj.flatten()
    nonzero_mask = flat != 0
    nonzero_vals = flat[nonzero_mask]

    logger.info("\n[Post-sparsification adjacency matrix]")
    logger.info("  shape           : %s", list(adj.shape))
    logger.info("  total cells     : %d  (diagonal excluded from edges)", adj.numel())
    logger.info("  non-zero edges  : %d", nonzero_vals.numel())
    logger.info("  zero entries    : %d", (flat == 0).sum().item())

    if nonzero_vals.numel() == 0:
        logger.error("  *** ALL values are ZERO — graph has no edges! ***")
        return

    logger.info("\n  Edge weight distribution (post-sparsification):")
    logger.info("    min    : % .4f", nonzero_vals.min().item())
    logger.info("    max    : % .4f", nonzero_vals.max().item())
    logger.info("    mean   : % .4f", nonzero_vals.mean().item())
    logger.info("    median : % .4f", nonzero_vals.median().item())
    logger.info("    std    : % .4f", nonzero_vals.std().item())

    # Granger interpretation: -log10(p-value) → values should be ≥ 1.3 for p < 0.05
    # Higher = more significant causality
    pct_below_1_3 = (nonzero_vals < 1.3).float().mean().item() * 100
    pct_above_2_0 = (nonzero_vals > 2.0).float().mean().item() * 100

    logger.info("\n  Granger significance interpretation (-log10 scale):")
    logger.info("    -log10(0.05) = 1.30  →  edges below 1.3: %.1f%%", pct_below_1_3)
    logger.info("    -log10(0.01) = 2.00  →  edges above 2.0: %.1f%%", pct_above_2_0)

    if nonzero_vals.max().item() < 0.1:
        logger.error(
            "  *** Max edge weight < 0.1 — these are NOT -log10(p) Granger values.\n"
            "      The graph likely fell back to lagged Pearson correlation,\n"
            "      OR Granger tests are producing p≈1 for all pairs.\n"
            "      Check causal_inference.py compute_granger_causality()."
        )
    elif nonzero_vals.max().item() < 1.3:
        logger.warning(
            "  *** Max edge weight %.4f < 1.3 — no edges are statistically significant\n"
            "      at p < 0.05 on Granger scale. Time series may be too short or\n"
            "      GRANGER_MAX_LAG may be too high.",
            nonzero_vals.max().item()
        )
    else:
        logger.info("  ✓ Edge weights look consistent with -log10(p) Granger values")

    # Print the full 12×12 matrix for visual inspection
    logger.info("\n  12×12 adjacency matrix (post-sparsification, rounded to 2 dp):")
    adj_np = adj.cpu().numpy()
    header = "      " + "  ".join(f"{i:5d}" for i in range(adj_np.shape[1]))
    logger.info(header)
    for row_i, row in enumerate(adj_np):
        row_str = "  ".join(f"{v:5.2f}" for v in row)
        logger.info("  r%02d:  %s", row_i, row_str)


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Feature Pipeline Diagnostics")
    parser.add_argument("--subject", default=None,
                        help="Specific subject ID to focus on (e.g. Pitt_0050003)")
    parser.add_argument("--no-histogram", action="store_true",
                        help="Skip edge-count histogram (faster for large datasets)")
    parser.add_argument("--no-gamma-audit", action="store_true",
                        help="Skip TR/Nyquist gamma-band audit")
    parser.add_argument("--no-loader", action="store_true",
                        help="Skip dataset-loader audit (faster; avoids loading all CSVs)")
    parser.add_argument("--graph-path", default=None,
                        help="Path to a specific .pt graph file (overrides --subject lookup)")
    args = parser.parse_args()

    logger.info("Neuro-CXG Feature Pipeline Diagnostics")
    logger.info("Project root : %s", PROJECT_ROOT)
    logger.info("Graphs dir   : %s", CAUSAL_GRAPHS_DIR)
    logger.info("GNN_IN_CHANNELS : %d  |  NUM_LOBES : %d", GNN_IN_CHANNELS, NUM_LOBES)
    logger.info("ALL_FEATURE_NAMES (%d): %s", len(ALL_FEATURE_NAMES), ALL_FEATURE_NAMES)

    # ── Pick a representative graph file ─────────────────────────────────────
    if args.graph_path:
        chosen_path = Path(args.graph_path)
    elif args.subject:
        chosen_path = CAUSAL_GRAPHS_DIR / f"{args.subject}_graph.pt"
    else:
        graph_files = sorted(CAUSAL_GRAPHS_DIR.glob("*_graph.pt"))
        chosen_path = graph_files[0] if graph_files else None

    if chosen_path is None or not chosen_path.exists():
        logger.error(
            "No graph file found. Run src/features/construct_causal.py first, "
            "or pass --graph-path <path>. Looked for: %s", chosen_path
        )
    else:
        # 1. Single-graph feature audit
        audit_single_graph(chosen_path)

        # 2. Granger / edge-weight sanity check on same graph
        audit_granger_sample(chosen_path)

    # 3. Dataset-loader audit (full pipeline: harmonized CSV → node features)
    if not args.no_loader:
        audit_via_dataset_loader(
            subject_id=args.subject,
            n_samples=5,
        )
    else:
        logger.info("\n[Dataset-loader audit SKIPPED (--no-loader)]")

    # 4. Edge-count histogram
    if not args.no_histogram and CAUSAL_GRAPHS_DIR.exists():
        audit_edge_distribution()
    else:
        logger.info("\n[Edge histogram SKIPPED (--no-histogram or graphs dir missing)]")

    # 5. TR / Nyquist audit
    if not args.no_gamma_audit:
        audit_tr_and_nyquist()
    else:
        logger.info("\n[TR/Nyquist audit SKIPPED (--no-gamma-audit)]")

    logger.info("\n" + "=" * 70)
    logger.info("DIAGNOSTICS COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
