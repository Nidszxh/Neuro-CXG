#!/usr/bin/env python
"""
src/run_explainability.py
Phase 8 Unified Explainability Pipeline
=========================================
Orchestrates all four explainability sub-phases for Neuro-CXG:

    Phase 8.1  Node Importance Analysis  (GradCAM + GAT attention weights)
    Phase 8.2  Edge Importance Analysis  (gradient attribution + edge masking)
    Phase 8.3  Feature Attribution       (Integrated Gradients / saliency maps)
    Phase 8.4  Literature Validation     (cross-reference with ASD networks)

All figures are saved to ``results/explainability/`` by default.
A summary JSON is written at the end with key findings.

Usage
-----
    # Full pipeline (all phases)
    python src/run_explainability.py

    # Use a specific checkpoint fold
    python src/run_explainability.py --fold 3

    # Run only specific phases
    python src/run_explainability.py --phases node edge

    # Custom output directory
    python src/run_explainability.py --output-dir results/explain_v2

    # Disable the slow edge-masking (keeps only gradient attribution)
    python src/run_explainability.py --no-masking

Outputs
-------
results/explainability/
    node/
        node_importance_gradcam.png
        attention_weights_by_layer.png
        node_importance_asd_vs_control.png
    edge/
        edge_importance_gradient.png
        edge_importance_masking.png
        edge_differential_connectivity.png
    features/
        feature_importance_ig.png
        feature_importance_per_class.png
        feature_importance_temporal_vs_spatial.png
    literature/
        literature_validation.json
        literature_validation.txt
        literature_validation_heatmap.png
    summary.json
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

# ── project imports ────────────────────────────────────────────────────────────
from src.core.config import (
    ALL_FEATURE_NAMES,
    LOBE_NAMES,
    NUM_LOBES,
    RESULTS_DIR,
    get_active_checkpoint_dir,
)
from src.features.graph_factory import ABIDECausalDataset
from src.models.factory import load_model
from src.models.training_utils import make_loader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

REGION_LABELS: list[str] = [LOBE_NAMES[i] for i in range(NUM_LOBES)]

# ── helpers ────────────────────────────────────────────────────────────────────

def _build_test_loader(batch_size: int = 16) -> DataLoader:
    """Build a DataLoader for the held-out test split."""
    dataset = ABIDECausalDataset(split="test")
    graphs  = [g for g in dataset if g is not None]
    loader  = make_loader(graphs, batch_size=batch_size, shuffle=False)
    logger.info("Test loader: %d graphs, batch_size=%d", len(graphs), batch_size)
    return loader

def _best_fold(num_folds: int = 5) -> int:
    """Select the fold with the highest saved validation AUC from training JSONs."""
    best_auc, best_fold_id = -1.0, 0
    training_dir = RESULTS_DIR / "experiments" / "training"
    for fold_id in range(num_folds):
        history_path = training_dir / f"training_history_fold{fold_id}.json"
        if not history_path.exists():
            continue
        try:
            with open(history_path) as f:
                h = json.load(f)
            auc = max(h.get("val_auc", [0.0]))
            if auc > best_auc:
                best_auc, best_fold_id = auc, fold_id
        except Exception:
            pass
    logger.info("Auto-selected fold %d (val AUC=%.4f)", best_fold_id, best_auc)
    return best_fold_id

# ── phase runners ──────────────────────────────────────────────────────────────

def run_phase_node(model, test_loader, device, output_dir: Path) -> dict:
    """Phase 8.1 — Node Importance Analysis."""
    from src.analysis.node_importance import NodeImportanceAnalyzer
    logger.info("=" * 55)
    logger.info("PHASE 8.1  NODE IMPORTANCE ANALYSIS")
    logger.info("=" * 55)
    analyzer = NodeImportanceAnalyzer(model, test_loader, device)
    results  = analyzer.run(output_dir / "node")

    # Task 3: persist anatomical network embeddings used by hierarchical pooling.
    network_embeddings = None
    if hasattr(model, "get_last_network_embeddings"):
        network_embeddings = model.get_last_network_embeddings()

    if network_embeddings is not None and torch.is_tensor(network_embeddings):
        node_dir = output_dir / "node"
        node_dir.mkdir(parents=True, exist_ok=True)
        net_np = network_embeddings.detach().cpu().numpy()
        np.save(node_dir / "network_embeddings_last_batch.npy", net_np)
        results["network_embeddings"] = {
            "shape": list(net_np.shape),
            "mean_abs_per_network": np.abs(net_np).mean(axis=(0, 2)).tolist(),
            "saved_path": str(node_dir / "network_embeddings_last_batch.npy"),
        }

    logger.info("Phase 8.1 complete — figures saved to %s/node/", output_dir)
    return results

def run_phase_edge(
    model,
    test_loader,
    device,
    output_dir: Path,
    run_masking: bool = True,
) -> dict:
    """Phase 8.2 — Edge Importance Analysis."""
    from src.analysis.edge_importance import EdgeImportanceAnalyzer
    logger.info("=" * 55)
    logger.info("PHASE 8.2  EDGE IMPORTANCE ANALYSIS")
    logger.info("=" * 55)
    analyzer = EdgeImportanceAnalyzer(
        model, test_loader, device,
        masking_max_graphs=40 if run_masking else 0,
    )
    results = analyzer.run(output_dir / "edge")
    logger.info("Phase 8.2 complete — figures saved to %s/edge/", output_dir)
    return results

def run_phase_features(model, test_loader, device, output_dir: Path) -> dict | None:
    """Phase 8.3 — Feature Attribution (saliency maps)."""
    logger.info("=" * 55)
    logger.info("PHASE 8.3  FEATURE ATTRIBUTION (SALIENCY MAPS)")
    logger.info("=" * 55)
    try:
        from src.analysis.feature_attribution import FeatureAttributionAnalyzer
        feat_dir = output_dir / "features"
        feat_dir.mkdir(parents=True, exist_ok=True)
        feature_names = ALL_FEATURE_NAMES
        analyzer = FeatureAttributionAnalyzer(
            model=model,
            test_loader=test_loader,
            feature_names=list(feature_names),
            device=str(device),
        )
        attributions = analyzer.compute_attributions()
        analyzer.visualize_feature_importance(
            attributions, feat_dir / "feature_importance_ig.png"
        )
        analyzer.visualize_per_class(feat_dir / "feature_importance_per_class.png")
        analyzer.compare_temporal_vs_spatial(
            attributions, output_path=feat_dir / "feature_importance_temporal_vs_spatial.png"
        )
        logger.info("Phase 8.3 complete — figures saved to %s/features/", output_dir)

        # Return per-region importance scores for literature validation
        if isinstance(attributions, torch.Tensor):
            attr_np = attributions.detach().cpu().numpy()   # (N, 12, 28)
        else:
            attr_np = np.array(attributions)
        region_scores = np.abs(attr_np).mean(axis=(0, 2))   # (12,) mean over subjects+features
        return {"region_scores": region_scores}
    except Exception as exc:
        logger.error("Phase 8.3 failed: %s", exc, exc_info=True)
        return None

def run_phase_literature(
    gradcam_scores: np.ndarray | None,
    feature_region_scores: np.ndarray | None,
    output_dir: Path,
    top_n: int = 6,
) -> dict:
    """Phase 8.4 — Literature / Clinical Validation."""
    from src.analysis.literature_validation import run_literature_validation
    logger.info("=" * 55)
    logger.info("PHASE 8.4  CLINICAL / LITERATURE VALIDATION")
    logger.info("=" * 55)
    results = run_literature_validation(
        gradcam_asd_scores=gradcam_scores,
        attention_asd_scores=feature_region_scores,
        output_dir=output_dir / "literature",
        top_n=top_n,
    )
    logger.info("Phase 8.4 complete — report saved to %s/literature/", output_dir)
    return results

# ── main pipeline ──────────────────────────────────────────────────────────────

def run_explainability_pipeline(
    fold_id: int,
    output_dir: Path,
    phases: list[str],
    run_masking: bool,
    batch_size: int,
) -> None:
    """Full orchestration for Phase 8 explainability."""
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    checkpoint_path = get_active_checkpoint_dir() / f"best_model_fold{fold_id}.pt"
    model = load_model(checkpoint_path=checkpoint_path, device=device)
    test_loader = _build_test_loader(batch_size=batch_size)

    summary: dict = {
        "fold_used": fold_id,
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "num_test_graphs": len(test_loader.dataset),
        "phases_run": phases,
    }

    node_results    = {}
    edge_results    = {}
    feature_results = None
    lit_results     = {}

    # ── 8.1 Node importance ────────────────────────────────────────────────────
    if "node" in phases:
        try:
            node_results = run_phase_node(model, test_loader, device, output_dir)
        except Exception as exc:
            logger.error("Phase 8.1 error: %s", exc, exc_info=True)

    # ── 8.2 Edge importance ────────────────────────────────────────────────────
    if "edge" in phases:
        try:
            edge_results = run_phase_edge(model, test_loader, device, output_dir, run_masking)
        except Exception as exc:
            logger.error("Phase 8.2 error: %s", exc, exc_info=True)

    # ── 8.3 Feature attribution ────────────────────────────────────────────────
    if "features" in phases:
        try:
            feature_results = run_phase_features(model, test_loader, device, output_dir)
        except Exception as exc:
            logger.error("Phase 8.3 error: %s", exc, exc_info=True)

    # ── 8.4 Literature validation  ─────────────────────────────────────────────
    if "literature" in phases:
        try:
            # Pull GradCAM ASD scores from Phase 8.1
            gradcam_scores = None
            if node_results and "gradcam" in node_results:
                gradcam_scores = node_results["gradcam"].get("asd_mean")

            feat_region_scores = None
            if feature_results and "region_scores" in feature_results:
                feat_region_scores = feature_results["region_scores"]

            lit_results = run_phase_literature(
                gradcam_scores, feat_region_scores, output_dir
            )
            summary["top_regions"]     = [r["name"]     for r in lit_results.get("top_regions", [])]
            summary["top_networks"]    = [k for k, v in lit_results.get("network_coverage", {}).items() if v["hit"]]
            summary["overlap_scores"]  = lit_results.get("overlap_scores", {})
        except Exception as exc:
            logger.error("Phase 8.4 error: %s", exc, exc_info=True)

    # ── GradCAM top-5 log ──────────────────────────────────────────────────────
    if node_results and "gradcam" in node_results:
        g = node_results["gradcam"]
        asd   = g.get("asd_mean",     np.zeros(NUM_LOBES))
        ctrl  = g.get("control_mean", np.zeros(NUM_LOBES))
        diff  = g.get("diff",         asd - ctrl)
        top5  = np.argsort(np.abs(diff))[::-1][:5]
        summary["gradcam_top5_differential"] = [
            {"region": REGION_LABELS[i], "delta": float(diff[i])} for i in top5
        ]

    if node_results and "network_embeddings" in node_results:
        summary["network_embeddings"] = node_results["network_embeddings"]

    # ── Edge top-5 log ─────────────────────────────────────────────────────────
    if edge_results and "gradient" in edge_results:
        diff_mat = (
            edge_results["gradient"]["asd_matrix"] - edge_results["gradient"]["control_matrix"]
        )
        flat_top = np.argsort(np.abs(diff_mat), axis=None)[::-1][:5]
        top_edges = []
        for flat_i in flat_top:
            row, col = np.unravel_index(flat_i, diff_mat.shape)
            top_edges.append({
                "source": REGION_LABELS[row],
                "target": REGION_LABELS[col],
                "delta":  float(diff_mat[row, col]),
            })
        summary["top5_differential_edges"] = top_edges

    # ── Save summary ───────────────────────────────────────────────────────────
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Summary JSON saved → %s", summary_path)

    # ── Final report ───────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("PHASE 8 EXPLAINABILITY PIPELINE COMPLETE")
    logger.info("=" * 65)
    logger.info("Output directory : %s", output_dir)
    logger.info("Phases run       : %s", ", ".join(phases))
    if "gradcam_top5_differential" in summary:
        logger.info("\nTop-5 Differentially-Important Regions (GradCAM, ASD−Control):")
        for rank, r in enumerate(summary["gradcam_top5_differential"], start=1):
            logger.info("  %d. %-25s Δ=%.4f", rank, r["region"], r["delta"])
    if "top5_differential_edges" in summary:
        logger.info("\nTop-5 Differentially-Important Edges (gradient, ASD−Control):")
        for rank, e in enumerate(summary["top5_differential_edges"], start=1):
            logger.info("  %d. %-22s → %-22s Δ=%.4f", rank, e["source"], e["target"], e["delta"])
    if "top_regions" in summary:
        logger.info("\nTop regions (literature validation): %s", ", ".join(summary["top_regions"]))
    if "top_networks" in summary:
        logger.info("Matching ASD networks  : %s", ", ".join(summary["top_networks"]))
    logger.info("=" * 65)

# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 8 Explainability Pipeline for Neuro-CXG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help=(
            "Checkpoint fold index to use (0–4).  "
            "If omitted, the fold with the highest recorded val AUC is selected automatically."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "explainability",
        help="Directory to save all explainability outputs.",
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=["node", "edge", "features", "literature"],
        default=["node", "edge", "features", "literature"],
        help="Subset of phases to run.",
    )
    parser.add_argument(
        "--no-masking",
        action="store_true",
        default=False,
        help="Skip the slow edge-masking (ΔP) analysis in Phase 8.2.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for the test DataLoader.",
    )
    args = parser.parse_args()

    fold_id = args.fold if args.fold is not None else _best_fold()

    run_explainability_pipeline(
        fold_id=fold_id,
        output_dir=args.output_dir,
        phases=list(args.phases),
        run_masking=not args.no_masking,
        batch_size=args.batch_size,
    )

if __name__ == "__main__":
    main()
