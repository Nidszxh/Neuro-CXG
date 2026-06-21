#!/usr/bin/env python
"""
src/run_pipeline.py
Neuro-CXG Unified Pipeline Runner
==================================
Last Updated: August 17, 2026

Orchestrates the Neuro-CXG pipeline from data download to evaluation.
Stage order and numbering below mirror `src/pipeline/registry.py` (STAGES),
the source of truth (see `--dry-run` output).

Data Preparation (Stages 1-3):
  1. ABIDE Download - fMRI data + 7-slice ALFF export
  2. Train/Val/Test Split - 2D stratification (DX_GROUP + SITE_ID)
  3. Master Manifest - Subject-phenotype mapping

Optional Data Prep:
  4. Site-Stratified CV Fold Assignment - opt-in GroupKFold by site cluster

Atlas & Validation (Stages 5-6):
  5. Atlas Validation - Verify AAL3 atlas files
  6. Post-Download Integrity - PNG/NPY validation

Feature Extraction (Stages 7-12):
  7. Atlas-Based Annotation - Generate YOLO labels
  8. YOLO Training - 12-region ROI detection (YOLO26n)
  9. Spatial Features - 3D coordinate aggregation (YOLO or atlas centroids)
  10. Temporal Features - 20 features/ROI (8 time + 12 frequency)
  11. Harmonization - Fold-safe neuroHarmonize (protects DX_GROUP)
  12. Pre-GNN Integrity - Validate dataset completeness

Graph Construction & Validation (Stages 13-16):
  13. Causal Graphs - Granger causality/lagged correlation (12x12)
  14. Multiview Graphs - Optional multi-view causal graph construction
  15. Diagnostics - Comprehensive health report
  16. Quality Validation - YOLO quality, graph sparsity checks

Main Training (Stage 17):
  17. GNN Training - 5-fold CV with GAT+GRL

Post-Training Analysis (Stages 18-25):
  18. Visualizations - Comprehensive plots and figures
  19. Causal Graph Visualization - Render directed ASD-vs-Control graphs
  20. Circular Connectome - Connectome ring visualization
  21. 3D Brain Visualization - Nilearn brain rendering
  22. Evaluation - Bootstrap CI, permutation test, subgroups
  23. Explainability - Node/edge importance, feature attribution
  24. Result Analysis - Per-subject predictions, misclassification
  25. Subject Analysis - Per-subject artifact integrity diagnostics

Usage:
  # Interactive mode (default)
  python src/run_pipeline.py --interactive

  # Automatic mode (recommended for non-interactive shells)
  python src/run_pipeline.py --auto

  # Skip data prep
  python src/run_pipeline.py --auto --skip-download --skip-split

  # Run only post-training analysis
  python src/run_pipeline.py --auto --analysis-only

For detailed documentation, see:
  - AGENTS.md (agent instructions, commands, gotchas)
  - docs/architecture.md (architecture, stage registry map)
  - docs/setup.md (environment setup)
  - docs/paper/results.md (canonical results)
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

# Setup Pathing
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.core.config import (
    BASELINE_CHECKPOINT_DIR,
    CAUSAL_GRAPHS_DIR,
    CAUSAL_GRAPHS_MULTIVIEW_DIR,
    CHECKPOINT_DIR,
    DATA_METADATA,
    DATA_PROCESSED,
    DATA_ROOT,
    DATA_TIME_SERIES,
    FINAL_TEST,
    FINAL_TRAIN,
    FINAL_VAL,
    MASTER_MANIFEST,
    NODE_ATTRIBUTES_HARMONIZED,
    NODE_ATTRIBUTES_TEMPORAL,
    NODE_FEATURES_3D,
    PROJECT_ROOT,
    RESULTS_DIR,
    YOLO_WEIGHTS_PATH,
    validate_environment,
)
from src.core.validators import validate_gnn_training_inputs
from src.pipeline.registry import STAGES as STAGE_REGISTRY
from src.pipeline.registry import completion_snapshot, stage_map

# Standard logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [PIPELINE] - %(levelname)s - %(message)s",
)
logger = logging.getLogger("pipeline")

# Path constants for new stages
DOWNLOAD_LOG = DATA_METADATA / "download_log.csv"

# Runnable modules that intentionally remain standalone debug/self-test tools and
# are not part of the default orchestration chain.
EXEMPT_ENTRYPOINT_MODULES: set[str] = {
    "src.run_pipeline",
    "src.core.config",
    "src.features.graph_factory",
    "src.features.causal_inference",
    "src.analysis.feature_attribution",
    "src.analysis.generate_paper_figures",
    "src.experiments.data_quality",
    "src.experiments.run_ablations",
    "src.experiments.run_learning_curve",
    "src.experiments.test_random_edges_on_test",
}


def _checkpoints_available() -> bool:
    """Return True when fold checkpoint files exist in CHECKPOINT_DIR or BASELINE_CHECKPOINT_DIR.

    Checks CHECKPOINT_DIR first (new training output), then falls back to the
    packaged baseline models so that evaluation stages run out-of-the-box even
    before a fresh training run completes.
    """
    for ckpt_dir in (CHECKPOINT_DIR, BASELINE_CHECKPOINT_DIR):
        if ckpt_dir.exists() and any(ckpt_dir.glob("best_model_fold*.pt")):
            return True
    return False


def _graph_files_available() -> bool:
    """Return True when at least one causal graph file exists."""
    return CAUSAL_GRAPHS_DIR.exists() and any(CAUSAL_GRAPHS_DIR.glob("*_graph.pt"))


def _source_timeseries_dir() -> Path | None:
    """Return the best source time-series directory before split.

    Supports canonical and legacy layouts:
    - data/processed/time_series (preferred)
    - data/timeseries (legacy)
    - data/processed (legacy flat fallback)
    """
    candidates = [
        DATA_TIME_SERIES,
        DATA_ROOT / "timeseries",
        DATA_PROCESSED,
    ]

    best_dir = None
    best_count = 0
    for candidate in candidates:
        if not candidate.exists():
            continue
        count = len(list(candidate.glob("*_ts.npy")))
        if count > best_count:
            best_dir = candidate
            best_count = count
    return best_dir


def _file_has_rows(path: Path, pattern: str) -> bool:
    """Return True when path exists and has at least one matching file."""
    return path.exists() and any(path.glob(pattern))


def _discover_runnable_src_modules() -> list[str]:
    """Return src modules that expose a __main__ entrypoint."""
    modules: list[str] = []
    src_root = PROJECT_ROOT / "src"
    for py_file in sorted(src_root.rglob("*.py")):
        if "__pycache__" in str(py_file):
            continue
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if "__main__" not in content:
            continue
        rel = py_file.relative_to(PROJECT_ROOT).with_suffix("")
        modules.append(".".join(rel.parts))
    return modules


def _check_stage_coverage(staged_modules: set[str], strict: bool = False) -> None:
    """Log uncovered runnable modules; fail in strict mode."""
    discovered = _discover_runnable_src_modules()
    uncovered = sorted(
        m
        for m in discovered
        if m not in staged_modules and m not in EXEMPT_ENTRYPOINT_MODULES
    )
    if not uncovered:
        logger.info("✓ Stage coverage check: all runnable src modules are orchestrated")
        return

    logger.warning(
        "Found %d runnable src module(s) not mapped to pipeline stages:", len(uncovered)
    )
    for mod in uncovered:
        logger.warning("  - %s", mod)

    if strict:
        logger.error(
            "Strict stage coverage failed. Add stages or exempt modules before continuing."
        )
        sys.exit(1)


def _log_stage_registry_status(runtime_stage_keys: set[str]) -> None:
    """Compare runtime stage keys against the declarative stage registry."""
    registry_keys = {stage.key for stage in STAGE_REGISTRY}

    missing_in_registry = sorted(runtime_stage_keys - registry_keys)
    if missing_in_registry:
        logger.warning(
            "Stage registry missing %d runtime key(s): %s",
            len(missing_in_registry),
            ", ".join(missing_in_registry),
        )

    extra_in_registry = sorted(registry_keys - runtime_stage_keys)
    if extra_in_registry:
        logger.info(
            "Stage registry has %d additional key(s) not active in this run context: %s",
            len(extra_in_registry),
            ", ".join(extra_in_registry),
        )

    snapshot = completion_snapshot()
    tracked = [key for key in runtime_stage_keys if key in snapshot]
    completed = sum(1 for key in tracked if snapshot[key])
    logger.info(
        "Stage registry snapshot: %d/%d stage sentinels already present",
        completed,
        len(tracked),
    )


def prompt_user(message, default=True):

    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        response = input(f"\n{message} {suffix}: ").strip().lower()

        if response == "":
            return default
        elif response in ["y", "yes"]:
            return True
        elif response in ["n", "no"]:
            return False
        else:
            logger.info("Please enter 'y' or 'n'")


def clear_old_state():
    """
    Prevents 'Shape Mismatches' by clearing old 164/170 ROI data.
    This ensures the new 12-region architecture has a clean environment.
    """
    logger.info("Cleaning legacy pipeline state for 12-region alignment...")

    # Files to remove to force regeneration
    to_delete = [NODE_FEATURES_3D, NODE_ATTRIBUTES_TEMPORAL, NODE_ATTRIBUTES_HARMONIZED]
    for f in to_delete:
        if f.exists():
            f.unlink()
            logger.debug(f"Removed stale metadata: {f.name}")

    # Remove old causal graphs (regenerated at current architecture)
    if CAUSAL_GRAPHS_DIR.exists():
        shutil.rmtree(CAUSAL_GRAPHS_DIR)
        CAUSAL_GRAPHS_DIR.mkdir(parents=True)
        logger.info("Reset Causal Graph directory (cleared previous causal graphs)")

    # Remove old multiview graphs so Stage 15 doesn't reuse stale degenerate files.
    if CAUSAL_GRAPHS_MULTIVIEW_DIR.exists():
        shutil.rmtree(CAUSAL_GRAPHS_MULTIVIEW_DIR)
        CAUSAL_GRAPHS_MULTIVIEW_DIR.mkdir(parents=True)
        logger.info(
            "Reset Multi-view Causal Graph directory (cleared stale view artifacts)"
        )


def run_module(module_path, args_list=None, description="", function_name=None):

    # Use the same Python executable that's running this script
    python_exe = sys.executable

    if function_name:
        # Call specific function within module; propagate False return as exit code 1
        cmd = [
            python_exe,
            "-c",
            f"import sys; from {module_path} import {function_name}; "
            f"result = {function_name}(); sys.exit(0 if result is not False else 1)",
        ]
    else:
        # Run module as script
        cmd = [python_exe, "-m", module_path]
        if args_list:
            cmd.extend(args_list)

    log_msg = description if description else f"Module: {module_path}"
    logger.info(f"Running: {log_msg}")
    logger.debug(f"Command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        logger.error(f"Module {module_path} failed with exit code {exc.returncode}")
        sys.exit(1)

    logger.info(f"Completed: {log_msg}")


def check_download_status():
    """Check if ABIDE data has been downloaded.

    Returns True when:
    - A download_log.csv exists with successful/skipped entries, OR
    - Split directories already contain time-series .npy files (data was
      downloaded and split in a prior run that did not produce a log).
    """
    # Fast-path: if split data is present, download already happened.
    train_ts = FINAL_TRAIN / "time_series"
    if train_ts.exists() and any(train_ts.glob("*_ts.npy")):
        logger.info(
            "Download status: split data present — treating download as complete."
        )
        return True

    # Second fast-path: pre-split source pools exist (legacy/canonical layouts).
    source_images = DATA_ROOT / "images"
    source_ts_dir = _source_timeseries_dir()
    if (
        source_ts_dir is not None
        and source_images.exists()
        and any(source_images.glob("*.png"))
    ):
        logger.info(
            "Download status: source pools present (images + time series in %s) — treating download as complete.",
            source_ts_dir,
        )
        return True

    if not DOWNLOAD_LOG.exists():
        logger.warning(
            "Download log not found and no split data detected. Data may not be downloaded."
        )
        return False

    # Count downloaded subjects from log file
    try:
        log_df = pd.read_csv(DOWNLOAD_LOG)
        total = len(log_df)
        # Success includes both 'success' and 'skipped' (already downloaded)
        successful = len(
            log_df[log_df["status"].str.lower().isin(["success", "skipped"])]
        )
        logger.info(
            f"Download status: {successful}/{total} subjects ready (downloaded/skipped)"
        )
        return successful > 0
    except Exception as e:
        logger.warning(f"Could not parse download log: {e}")
        return False


def check_split_status():
    """Check if train/val/test splits are complete and non-empty."""
    splits_exist = all([FINAL_TRAIN.exists(), FINAL_VAL.exists(), FINAL_TEST.exists()])

    if not splits_exist:
        logger.warning("Train/val/test splits not found.")
        return False

    split_paths = {
        "train": FINAL_TRAIN,
        "val": FINAL_VAL,
        "test": FINAL_TEST,
    }

    is_complete = True
    for split_name, split_root in split_paths.items():
        images_dir = split_root / "images"
        ts_dir = split_root / "time_series"

        image_count = len(list(images_dir.glob("*.png"))) if images_dir.exists() else 0
        ts_count = len(list(ts_dir.glob("*_ts.npy"))) if ts_dir.exists() else 0

        logger.info(
            f"Split status [{split_name}]: images={image_count}, time_series={ts_count}"
        )

        if image_count == 0 or ts_count == 0:
            is_complete = False

    if not is_complete:
        logger.warning(
            "Split folders exist but are empty/incomplete. Stage 2 will run to rebuild splits."
        )
        return False

    source_images_dir = DATA_ROOT / "images"
    remaining_source_images = (
        len(list(source_images_dir.glob("*.png"))) if source_images_dir.exists() else 0
    )
    source_ts_dir = _source_timeseries_dir()
    remaining_source_ts = (
        len(list(source_ts_dir.glob("*_ts.npy"))) if source_ts_dir else 0
    )

    if remaining_source_images > 0 or remaining_source_ts > 0:
        logger.warning(
            "Detected unsplit source data (images=%d, ts=%d from %s). Stage 2 will run.",
            remaining_source_images,
            remaining_source_ts,
            source_ts_dir if source_ts_dir else "none",
        )
        return False

    return True


def show_execution_plan(stages):
    """Display what will be executed."""
    logger.info("\n" + "=" * 70)
    logger.info("EXECUTION PLAN")
    logger.info("=" * 70)
    for i, (stage_name, will_run, reason) in enumerate(stages, 1):
        status = "✓ WILL RUN" if will_run else "○ SKIP"
        logger.info("%d. %-40s %-15s %s", i, stage_name, status, reason)
    logger.info("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Neuro-CXG: 12-Region Causal GNN Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (prompts for each stage)
  python run_pipeline.py --interactive

  # Automatic mode (run all missing stages)
  python run_pipeline.py --auto

  # Skip data preparation stages
  python run_pipeline.py --skip-download --skip-split

  # Force complete rebuild
  python run_pipeline.py --force-reset

  # Show execution plan without running
  python run_pipeline.py --dry-run

  # Run only post-training analysis
  python run_pipeline.py --analysis-only

  # Run only visualizations
  python run_pipeline.py --visualizations-only

  # Full pipeline with analysis
  python run_pipeline.py --auto

  # Skip slow analysis stages
  python run_pipeline.py --skip-evaluation --skip-explainability

  # Regenerate features without full reset
  python run_pipeline.py --regenerate-features
        """,
    )

    # Execution modes
    parser.add_argument(
        "--interactive",
        action="store_true",
        default=True,
        help="Prompt user before each stage (default)",
    )
    parser.add_argument(
        "--auto", action="store_true", help="Run all missing stages without prompts"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show execution plan without running"
    )

    # Stage control - Skip flags
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip ABIDE download (use existing data)",
    )
    parser.add_argument(
        "--skip-split",
        action="store_true",
        help="Skip train/val/test split (use existing splits)",
    )
    parser.add_argument(
        "--skip-manifest", action="store_true", help="Skip manifest generation"
    )
    parser.add_argument(
        "--skip-annotate",
        action="store_true",
        help="Skip atlas-based annotation (use existing labels)",
    )
    parser.add_argument(
        "--skip-yolo",
        action="store_true",
        help="Skip YOLO training (use existing weights)",
    )
    parser.add_argument(
        "--use-yolo-spatial",
        action="store_true",
        help="Use YOLO-derived spatial features (default).",
    )
    parser.add_argument(
        "--use-atlas-spatial", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--skip-spatial",
        action="store_true",
        help="Skip spatial feature extraction (use existing features)",
    )
    parser.add_argument(
        "--multiview",
        action="store_true",
        help="Run optional multi-view causal graph construction stage after causal_graphs",
    )
    parser.add_argument(
        "--site-stratified-cv",
        action="store_true",
        help="Run optional site-stratified CV fold assignment stage after split",
    )
    parser.add_argument("--skip-gnn", action="store_true", help="Skip GNN training")
    parser.add_argument(
        "--skip-integrity", action="store_true", help="Skip all integrity checks"
    )
    parser.add_argument(
        "--skip-atlas-validation", action="store_true", help="Skip atlas validation"
    )
    parser.add_argument(
        "--skip-validation", action="store_true", help="Skip validation checks"
    )
    parser.add_argument(
        "--11-lobes",
        "--11-lobe",
        dest="lobes_11",
        action="store_true",
        help="Use 11 lobes (exclude Brainstem)",
    )

    # Post-training analysis flags
    parser.add_argument(
        "--skip-visualizations",
        action="store_true",
        help="Skip generating visualizations after training",
    )
    parser.add_argument(
        "--skip-graph-visualization",
        action="store_true",
        help="Skip causal graph visualization stage",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip comprehensive evaluation (bootstrap CI, permutation test, subgroups)",
    )
    parser.add_argument(
        "--skip-explainability",
        action="store_true",
        help="Skip explainability analysis (node/edge importance, feature attribution)",
    )
    parser.add_argument(
        "--skip-result-analysis",
        action="store_true",
        help="Skip result analysis (per-subject predictions, misclassification analysis)",
    )
    parser.add_argument(
        "--skip-subject-analysis",
        action="store_true",
        help="Skip per-subject artifact integrity analysis",
    )
    parser.add_argument(
        "--visualizations-only",
        action="store_true",
        help="Only run visualizations (skip all other stages)",
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Only run post-training analysis stages (vis, eval, explain, results)",
    )

    parser.add_argument(
        "--config-hash",
        type=str,
        default=None,
        help="Expected 8-character config hash to enforce reproducibility",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Global random seed for reproducibility"
    )
    parser.add_argument(
        "--skip-diagnostics",
        action="store_true",
        help="Skip comprehensive health check",
    )
    parser.add_argument(
        "--skip-comprehensive-validation",
        action="store_true",
        help="Skip comprehensive validation & tuning suite (YOLO quality, sparsity, stratification)",
    )

    # Feature regeneration
    parser.add_argument(
        "--regenerate-features",
        action="store_true",
        help="Regenerate spatial/temporal features, harmonization, and graphs (keeps other data)",
    )

    # Force/Diagnostics
    parser.add_argument(
        "--force-reset",
        action="store_true",
        help="Wipe all intermediate CSVs and Graphs",
    )

    args = parser.parse_args()

    # 11-lobe mode: set env var BEFORE any config imports
    if args.lobes_11:
        os.environ["NEURO_CXG_11_LOBES"] = "1"

    # Seed propagation for reproducibility
    os.environ["NEURO_CXG_SEED"] = str(args.seed)

    # Backward-compatible alias: --use-atlas-spatial forces atlas-centroid features.
    if args.use_yolo_spatial and args.use_atlas_spatial:
        parser.error("Use only one of --use-yolo-spatial or --use-atlas-spatial")
    if args.use_atlas_spatial:
        args.use_yolo_spatial = False
    elif not args.use_yolo_spatial:
        # Default behavior: use YOLO-derived spatial features unless atlas is requested.
        args.use_yolo_spatial = True

    # 11-lobe mode is handled via env var set BEFORE argparse above
    # Just log the confirmation here
    if args.lobes_11:
        logger.info("=" * 50)
        logger.info("11-LOBE MODE ENABLED (via --11-lobes flag)")
        logger.info("  Brainstem will be excluded from all computations")
        logger.info("=" * 50)

    # Override interactive mode if --auto is set
    interactive = args.interactive and not args.auto

    # Signal to subprocesses (pipeline_checks, etc.) that they should skip any
    # interactive prompts and apply safe defaults automatically.
    if args.auto:
        os.environ["NEURO_CXG_NONINTERACTIVE"] = "1"

    logger.info("\n" + "=" * 70)
    logger.info("NEURO-CXG PIPELINE RUNNER")
    logger.info("12-Region Causal GNN for fMRI Analysis")
    logger.info("=" * 70)

    # STAGE 0: PRE-FLIGHT VALIDATION
    logger.info("\nStage 0: Pre-Flight Validation")

    if not validate_environment():
        logger.error("Environment validation failed. Check config.py and data paths.")
        sys.exit(1)

    from src.validation.config_snapshot import get_config_hash

    current_hash = get_config_hash()
    logger.info(f"Current Config Hash: {current_hash}")
    if args.config_hash:
        if args.config_hash != current_hash:
            logger.error(
                f"Config hash mismatch! Expected {args.config_hash}, got {current_hash}"
            )
            sys.exit(1)
        else:
            logger.info("Config hash match confirmed.")

    logger.info("Environment validation passed")

    # Ensure the training checkpoint directory exists so gnn_model.py can save there.
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Checkpoint directory ready: {CHECKPOINT_DIR}")

    # DETERMINE STAGE EXECUTION BASED ON FLAGS

    registry_by_key = stage_map()
    execution_order = [stage.key for stage in STAGE_REGISTRY]

    yolo_weights = YOLO_WEIGHTS_PATH
    if (
        "yolo" in registry_by_key
        and registry_by_key["yolo"].output_sentinel is not None
    ):
        yolo_weights = registry_by_key["yolo"].output_sentinel

    data_downloaded = check_download_status() if not args.skip_download else True
    data_split = check_split_status() if not args.skip_split else True
    existing_labels = (
        registry_by_key["annotate"].is_complete()
        if "annotate" in registry_by_key
        else _file_has_rows(FINAL_TRAIN / "labels", "*.txt")
    )
    existing_spatial = (
        registry_by_key["spatial_features"].is_complete()
        if "spatial_features" in registry_by_key
        else NODE_FEATURES_3D.exists()
    )
    existing_temporal = (
        registry_by_key["temporal_features"].is_complete()
        if "temporal_features" in registry_by_key
        else NODE_ATTRIBUTES_TEMPORAL.exists()
    )
    existing_harmonized = (
        registry_by_key["harmonization"].is_complete()
        if "harmonization" in registry_by_key
        else NODE_ATTRIBUTES_HARMONIZED.exists()
    )
    existing_graphs = (
        registry_by_key["causal_graphs"].is_complete()
        if "causal_graphs" in registry_by_key
        else _graph_files_available()
    )
    existing_checkpoints = _checkpoints_available()
    # Compute readiness and planned outputs once so downstream stages can run in the    # same invocation even when their artifacts are produced earlier in this run.
    download_will_run = not args.skip_download and not data_downloaded
    split_will_run = not args.skip_split and (not data_split or args.force_reset)
    labels_will_run = not args.skip_annotate and (
        args.force_reset or not existing_labels
    )
    yolo_will_run = (
        (not yolo_weights.exists() or args.force_reset)
        and not args.skip_yolo
        and args.use_yolo_spatial
    )
    spatial_will_run = not args.skip_spatial and (
        not existing_spatial or args.force_reset or args.regenerate_features
    )
    temporal_will_run = (
        not existing_temporal or args.force_reset or args.regenerate_features
    )
    harmonization_will_run = (
        not existing_harmonized or args.force_reset or args.regenerate_features
    )
    graphs_will_run = (
        not existing_graphs or args.force_reset or args.regenerate_features
    )
    gnn_training_will_run = (
        not args.skip_gnn and not args.visualizations_only and not args.analysis_only
    )

    download_ready_or_planned = data_downloaded or download_will_run
    split_ready_or_planned = data_split or split_will_run
    labels_ready_or_planned = existing_labels or labels_will_run
    yolo_ready_or_planned = (
        (not args.use_yolo_spatial) or yolo_weights.exists() or yolo_will_run
    )
    spatial_ready_or_planned = existing_spatial or spatial_will_run
    temporal_ready_or_planned = existing_temporal or temporal_will_run
    graphs_ready_or_planned = existing_graphs or graphs_will_run
    harmonized_ready_or_planned = existing_harmonized or harmonization_will_run
    checkpoints_ready_or_planned = existing_checkpoints or gnn_training_will_run

    stage_should_run = {
        "download": download_will_run,
        "split": split_will_run,
        "manifest": not args.skip_manifest
        and (
            not MASTER_MANIFEST.exists()
            or args.force_reset
            or (not args.skip_split and not data_split)
        ),
        "atlas_validation": not args.skip_atlas_validation,
        "post_download_integrity": not args.skip_integrity
        and download_ready_or_planned,
        "annotate": labels_will_run and split_ready_or_planned,
        "site_stratified_cv": args.site_stratified_cv and split_ready_or_planned,
        "yolo": yolo_will_run and labels_ready_or_planned,
        "spatial_features": spatial_will_run
        and split_ready_or_planned
        and yolo_ready_or_planned,
        "temporal_features": temporal_will_run and split_ready_or_planned,
        "harmonization": harmonization_will_run
        and spatial_ready_or_planned
        and temporal_ready_or_planned,
        "pre_gnn_integrity": not args.skip_integrity and harmonized_ready_or_planned,
        "causal_graphs": graphs_will_run and harmonized_ready_or_planned,
        "multiview_graphs": args.multiview and graphs_ready_or_planned,
        "diagnostics": not args.skip_diagnostics and graphs_ready_or_planned,
        "quality_validation": not args.skip_comprehensive_validation
        and graphs_ready_or_planned,
        "gnn_training": gnn_training_will_run
        and graphs_ready_or_planned
        and harmonized_ready_or_planned,
        "visualizations": not args.skip_visualizations and checkpoints_ready_or_planned,
        "graph_visualization": not args.skip_graph_visualization
        and graphs_ready_or_planned,
        "evaluation": not args.skip_evaluation and checkpoints_ready_or_planned,
        "explainability": not args.skip_explainability and checkpoints_ready_or_planned,
        "result_analysis": not args.skip_result_analysis
        and checkpoints_ready_or_planned,
        "subject_analysis": not args.skip_subject_analysis
        and graphs_ready_or_planned
        and harmonized_ready_or_planned,
    }

    stage_reasons = {
        "download": "Download ABIDE fMRI data + 7-slice ALFF export (Stage 1)",
        "split": "2D stratification by DX_GROUP + SITE_ID (Stage 2)",
        "manifest": "Maps subjects to phenotypes (Stage 3)",
        "atlas_validation": "Verify AAL3 atlas files exist and are valid (Stage 5)",
        "post_download_integrity": "Validate PNG/NPY files after download (Stage 6)",
        "annotate": "Generate YOLO training labels from AAL3 atlas (Stage 7)",
        "site_stratified_cv": "Regenerate cv_fold with site-stratified GroupKFold by site-cluster",
        "yolo": (
            "Train YOLO26n for 12-region detection (Stage 8)"
            if not yolo_weights.exists()
            else "Force retrain"
        ),
        "spatial_features": "YOLO inference -> 3D spatial coords aggregation (Stage 9)",
        "temporal_features": "20 features per ROI: 8 time-domain + 12 frequency (Stage 10)",
        "harmonization": "Fold-safe neuroHarmonize, protects DX_GROUP (Stage 11)",
        "pre_gnn_integrity": "Validate dataset completeness per split (Stage 12)",
        "causal_graphs": "Granger causality/lagged correlation (Stage 13)",
        "multiview_graphs": "Optional multi-view causal graph construction (Stage 14)",
        "diagnostics": "Comprehensive health report after graphs built (Stage 15)",
        "quality_validation": "YOLO quality, graph sparsity, stratification (Stage 16)",
        "gnn_training": "Main training phase — 5-fold CV with GAT+GRL (Stage 17)",
        "visualizations": "Generate comprehensive visualizations (Phase 9 Reporting)",
        "graph_visualization": "Render directed ASD-vs-Control causal graph comparison",
        "evaluation": "Ensemble evaluation, bootstrap CI, permutation test, subgroups (Phase 9.2)",
        "explainability": "Node/edge importance, feature attribution, literature validation (Phase 8)",
        "result_analysis": "Per-subject predictions, misclassification analysis, site effects (Phase 9.3)",
        "subject_analysis": "Per-subject artifact diagnostics and summary report",
    }

    # Build executable stage dictionary from registry metadata.
    stages = {}
    for stage_meta in STAGE_REGISTRY:
        module_name = stage_meta.module
        reason = stage_reasons.get(stage_meta.key, stage_meta.name)
        name = stage_meta.name

        if stage_meta.key == "spatial_features":
            name = "Spatial Feature Extraction (12-region)"
            if args.use_yolo_spatial:
                module_name = "src.features.extract_spatial"
                reason = "YOLO inference -> 3D spatial coords aggregation (Stage 9)"
            else:
                module_name = "src.features.extract_spatial_atlas"
                reason = "Atlas centroids -> 3D spatial coords aggregation (Stage 9)"
        elif stage_meta.key == "split":
            name = "Train/Val/Test Split (2D Stratified)"
        elif stage_meta.key == "post_download_integrity":
            name = "Post-Download Integrity Check"
        elif stage_meta.key == "yolo":
            name = "YOLO Training (ROI Detection)"
        elif stage_meta.key == "causal_graphs":
            name = "Causal Graph Construction (12x12)"
        elif stage_meta.key == "quality_validation":
            name = "Quality Validation (YOLO & Graph Sparsity)"
        elif stage_meta.key == "gnn_training":
            name = "GNN Training (5-Fold CV)"
        elif stage_meta.key == "result_analysis":
            name = "Result Interpretation & Analysis"
        elif stage_meta.key == "subject_analysis":
            name = "Subject-Level Analysis"

        stage_args = list(stage_meta.args) if stage_meta.args else []
        if stage_meta.key == "gnn_training":
            stage_args.extend(["--seed", str(args.seed)])
        stages[stage_meta.key] = {
            "name": name,
            "should_run": stage_should_run.get(stage_meta.key, False),
            "reason": reason,
            "module": module_name,
            "function": stage_meta.function,
            "args": stage_args,
        }

    _log_stage_registry_status(set(stages.keys()))

    # Special handling for analysis-only mode
    if args.analysis_only:
        logger.info("📊 Analysis-only mode: Running post-training analysis stages only")
        analysis_stages = {
            "visualizations",
            "graph_visualization",
            "evaluation",
            "explainability",
            "result_analysis",
            "subject_analysis",
        }
        for key in stages.keys():
            if key not in analysis_stages:
                stages[key]["should_run"] = False

    # Validate that all runnable src entrypoints are either staged or explicitly exempt.
    # `spatial_features` can route to either YOLO-based or atlas-based extractor,
    # so mark both modules as covered for strict stage-audit mode.
    staged_modules = {stage_info["module"] for stage_info in stages.values()}
    staged_modules.update(
        {"src.features.extract_spatial", "src.features.extract_spatial_atlas"}
    )
    _check_stage_coverage(staged_modules, strict=False)

    # Special handling for visualizations-only mode
    if args.visualizations_only:
        logger.info("🎨 Visualizations-only mode: Skipping all training stages")
        for key in stages.keys():
            if key != "visualizations":
                stages[key]["should_run"] = False

    # Show execution plan
    stage_list = [
        (stage_info["name"], stage_info["should_run"], stage_info["reason"])
        for stage_info in stages.values()
    ]
    show_execution_plan(stage_list)

    if args.dry_run:
        logger.info("🔍 Dry-run mode: Exiting without execution")
        return

    # Reset state if requested
    if args.force_reset:
        if interactive and not prompt_user(
            "⚠️  This will delete intermediate files. Continue?", default=False
        ):
            logger.info("Aborted by user")
            sys.exit(0)
        clear_old_state()

    # STAGE EXECUTION
    # Execute stages in declarative registry order.
    for stage_key in execution_order:
        if stage_key not in stages:
            continue

        stage = stages[stage_key]

        if not stage["should_run"]:
            logger.info(f"⏭️  Skipping: {stage['name']}")
            continue

        # Special handling for long-running stages
        if stage_key == "yolo":
            msg = f"Run {stage['name']}? (This may take 1-2 hours)"
        elif stage_key == "download":
            msg = f"Run {stage['name']}? (This may take 2-4 hours)"
        elif stage_key == "gnn_training":
            msg = f"🚀 Start {stage['name']}? (Main training phase, ~20-30 min)"
        elif stage_key == "visualizations":
            msg = f"🎨 Generate {stage['name']}? (Creates plots and analysis)"
        elif stage_key == "graph_visualization":
            msg = f"🧠 Render {stage['name']}? (ASD-vs-Control directed graph plot)"
        elif stage_key == "evaluation":
            msg = f"📊 Run {stage['name']}? (Bootstrap CI, permutation test, subgroups)"
        elif stage_key == "explainability":
            msg = f"🔬 Run {stage['name']}? (Node/edge importance, feature attribution)"
        elif stage_key == "result_analysis":
            msg = (
                f"📈 Run {stage['name']}? (Per-subject predictions, misclassification)"
            )
        elif stage_key == "subject_analysis":
            msg = f"🧾 Run {stage['name']}? (Generates per-subject diagnostics CSV/TXT)"
        else:
            msg = f"Run {stage['name']}?"

        if (
            interactive and not args.visualizations_only and not args.analysis_only
        ):  # No prompt in special modes
            if not prompt_user(msg, default=True):
                logger.info(f"⏭️  User skipped: {stage['name']}")
                continue

        if stage_key == "gnn_training":
            try:
                validate_gnn_training_inputs()
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    "Pre-training validation failed for gnn_training stage. " f"{exc}"
                ) from exc

        # Execute with function name if specified
        function_name = stage.get("function", None)
        args_list = stage.get("args", None)
        run_module(
            stage["module"],
            args_list=args_list,
            description=stage["name"],
            function_name=function_name,
        )

    # COMPLETION

    logger.info("\n" + "=" * 70)
    logger.info("NEURO-CXG PIPELINE EXECUTION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"📁 Checkpoints: {CHECKPOINT_DIR}")
    logger.info(f"📁 Causal graphs: {CAUSAL_GRAPHS_DIR}")
    logger.info(f"📁 Features: {DATA_METADATA}")

    # Check if analysis outputs exist
    if (RESULTS_DIR / "visualizations").exists():
        logger.info(f"📁 Visualizations: {RESULTS_DIR / 'visualizations'}")
    if (RESULTS_DIR / "evaluation").exists():
        logger.info(f"📁 Evaluation results: {RESULTS_DIR / 'evaluation'}")
    if (RESULTS_DIR / "explainability").exists():
        logger.info(f"📁 Explainability: {RESULTS_DIR / 'explainability'}")
    if (RESULTS_DIR / "analysis").exists():
        logger.info(f"📁 Result analysis: {RESULTS_DIR / 'analysis'}")
    if (RESULTS_DIR / "subject_analysis").exists():
        logger.info(f"📁 Subject analysis: {RESULTS_DIR / 'subject_analysis'}")
    if (RESULTS_DIR / "experiments" / "ablations").exists():
        logger.info(f"📁 Ablation studies: {RESULTS_DIR / 'experiments' / 'ablations'}")

    logger.info("=" * 70)
    logger.info("\n✨ Post-Training Analysis Commands:")
    logger.info("   python src/run_pipeline.py --auto --visualizations-only")
    logger.info("   python src/run_pipeline.py --auto --analysis-only")
    logger.info("   python -m src.analysis.visualize_causal_graph --auto-pair")
    logger.info("   python -m src.analysis.subject_analysis")
    logger.info("   python -m src.validation.pipeline_checks")
    logger.info("   python src/run_evaluation.py")
    logger.info("   python src/run_explainability.py")
    logger.info("   python src/run_result_analysis.py")
    logger.info("=" * 70 + "\n")


if __name__ == "__main__":
    main()
