"""
Unified entry point to validate and execute the full Neuro-CXG pipeline end to end.

Stages (in order):
1) Environment validation (paths, lobe mapping, hardware)
2) Optional YOLO training (skipped if weights already present unless forced)
3) Stratified split (DX_GROUP + SITE_ID)
4) ROI feature extraction (YOLO inference → spatial coords, 5-lobe filter)
5) Temporal feature harmonization (neuroCombat with protected DX_GROUP)
6) Causal graph construction (lagged partial correlation, sparsity)
7) GNN training (5-fold stratified CV with checkpointing)

The script performs lightweight pre-checks before each stage and stops on failure.
Run from project root: `python -m src.run_pipeline` or `python src/run_pipeline.py`.
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

# Ensure local imports work regardless of launch location
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from config import (
    PROJECT_ROOT as CFG_PROJECT_ROOT,
    DATA_FINAL,
    DATA_IMAGES,
    DATA_PROCESSED,
    MASTER_MANIFEST,
    NODE_ATTRIBUTES_TEMPORAL,
    NODE_FEATURES_3D,
    NODE_ATTRIBUTES_HARMONIZED,
    CAUSAL_GRAPHS_DIR,
    CHECKPOINT_DIR,
    RESULTS_DIR,
    validate_environment,
    validate_lobe_mapping,
    validate_paths,
)

# Stage modules
from data.extract_features import extract_features, MODEL_PATH as YOLO_BEST_PATH
from data.harmonize import run_harmonization
from data.construct_causal import main as run_construct_causal
from data.split import run_stratified_split
from models.gnn_model import run_kfold_training
from pipelines.roi_detection import main as run_yolo_training
from data.check_progress import check_health
from utils.manifest import create_manifest
from utils.integrity_check import check_dataset_integrity
from utils.compute_roi import main as run_compute_roi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("pipeline")


def run_step(name: str, func, check=None):
    """Helper to run a stage with logging and optional post-check."""
    logger.info("==> %s", name)
    func()
    if check:
        check()
    logger.info("✓ %s complete", name)


def ensure_workdir():
    """Switch to project root so relative paths inside legacy modules stay correct."""
    if Path.cwd() != CFG_PROJECT_ROOT:
        os.chdir(CFG_PROJECT_ROOT)
        logger.info("Working directory set to %s", CFG_PROJECT_ROOT)


def check_manifest():
    if not MASTER_MANIFEST.exists():
        raise FileNotFoundError(
            f"Master manifest missing at {MASTER_MANIFEST}. Run manifest generation before pipeline."
        )


def check_split_outputs():
    expected = [DATA_FINAL / split / sub for split in ["train", "val", "test"] for sub in ["images", "labels", "time_series"]]
    missing = [p for p in expected if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Split step missing outputs: {missing}")


def check_split_inputs():
    pngs = list(DATA_IMAGES.glob("*.png"))
    if not pngs:
        raise FileNotFoundError(
            "No source PNGs found in data/images. If you already split the data, rerun with --skip-split, or restore images before splitting."
        )


def check_extract_outputs():
    if NODE_FEATURES_3D.exists():
        return

    # Fallback: legacy path used by extract_features before alignment
    legacy_path = DATA_PROCESSED / "metadata" / "node_features_3d.csv"
    if legacy_path.exists():
        logger.info("Relocating node_features_3d.csv from legacy path %s to %s", legacy_path, NODE_FEATURES_3D)
        NODE_FEATURES_3D.parent.mkdir(parents=True, exist_ok=True)
        legacy_path.replace(NODE_FEATURES_3D)
        return

    raise FileNotFoundError(f"Expected node features at {NODE_FEATURES_3D}")


def check_temporal_outputs():
    if NODE_ATTRIBUTES_TEMPORAL.exists():
        return
    logger.info("Temporal attributes not found at %s; generating via compute_roi", NODE_ATTRIBUTES_TEMPORAL)
    run_compute_roi()
    if not NODE_ATTRIBUTES_TEMPORAL.exists():
        raise FileNotFoundError(f"Failed to generate temporal attributes at {NODE_ATTRIBUTES_TEMPORAL}")


def check_harmonize_outputs():
    if not NODE_ATTRIBUTES_HARMONIZED.exists():
        raise FileNotFoundError(f"Expected harmonized features at {NODE_ATTRIBUTES_HARMONIZED}")


def check_graph_outputs():
    if not CAUSAL_GRAPHS_DIR.exists() or not any(CAUSAL_GRAPHS_DIR.glob("*_graph.pt")):
        raise FileNotFoundError(f"No causal graphs found in {CAUSAL_GRAPHS_DIR}")


def check_gnn_outputs():
    missing = [p for p in [CHECKPOINT_DIR / f"best_model_fold{i}.pt" for i in range(5)] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing GNN checkpoints: {missing}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run full Neuro-CXG pipeline end to end.")
    parser.add_argument("--force-yolo-train", action="store_true", help="Train YOLO even if weights already exist.")
    parser.add_argument("--skip-yolo-train", action="store_true", help="Skip YOLO training stage.")
    parser.add_argument("--skip-split", action="store_true", help="Skip stratified split stage.")
    parser.add_argument("--skip-gnn", action="store_true", help="Skip GNN training stage.")
    parser.add_argument("--run-download", action="store_true", help="Run ABIDE download + extraction (src/data/abide_download.py).")
    parser.add_argument("--run-health", action="store_true", help="Run dataset health report (src/data/check_progress.py).")
    parser.add_argument("--run-manifest", action="store_true", help="Regenerate master manifest (src/utils/manifest.py).")
    parser.add_argument("--run-integrity", action="store_true", help="Run integrity check (src/utils/integrity_check.py) in non-interactive mode (auto exit).")
    parser.add_argument("--log-file", type=str, help="Path to write pipeline logs (in addition to console).")
    return parser.parse_args()


def add_file_logging(path: Path):
    """Attach a file handler to root logger to capture all module logs."""
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler = logging.FileHandler(path)
    handler.setFormatter(fmt)
    handler.setLevel(logging.INFO)
    root_logger = logging.getLogger()
    # Avoid duplicate handlers for the same file
    if all(not (isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == handler.baseFilename) for h in root_logger.handlers):
        root_logger.addHandler(handler)
    return handler


def main():
    args = parse_args()
    ensure_workdir()

    # Optional file logging
    if args.log_file:
        log_path = Path(args.log_file)
        if not log_path.is_absolute():
            log_path = CFG_PROJECT_ROOT / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        add_file_logging(log_path)
        logger.info("File logging enabled at %s", log_path)

    # Stage 0: Environment validation
    logger.info("Validating configuration and environment")
    validate_lobe_mapping()
    validate_paths()
    validate_environment()

    # Optional Stage: ABIDE download and preprocessing
    if args.run_download:
        logger.info("Running ABIDE download and slice/TS extraction (non-interactive)")
        subprocess.run([sys.executable, "-m", "src.data.abide_download"], check=True)

    # Optional Stage: Dataset health report (class/site balance, slice counts)
    if args.run_health:
        logger.info("Running dataset health report (check_progress)")
        check_health()

    # Stage: Stratified split (DX_GROUP + SITE_ID)
    if not args.skip_split:
        check_split_inputs()
        run_step("Stratified split", run_stratified_split, check_split_outputs)
    else:
        logger.info("Stratified split explicitly skipped")
        check_split_outputs()

    # Stage: Manifest generation (required for downstream steps)
    if args.run_manifest or not MASTER_MANIFEST.exists():
        run_step("Manifest generation", create_manifest, check_manifest)
    else:
        check_manifest()

    # Optional Stage: Integrity check (auto-select exit to avoid interactive prompt)
    if args.run_integrity:
        logger.info("Running integrity check (non-interactive exit after report)")
        # integrity_check prompts; send "3" to exit after report
        subprocess.run([sys.executable, "-m", "src.utils.integrity_check"], input="3\n", text=True, check=True)

    # Stage 1: YOLO training (optional)
    if not args.skip_yolo_train:
        # Prefer newer ROI_Detection_v20_Final4 weights if present
        alt_yolo = RESULTS_DIR / "ROI_Detection_v20_Final4" / "weights" / "best.pt"
        if not YOLO_BEST_PATH.exists() and alt_yolo.exists():
            logger.info("Updating YOLO weight path to %s", alt_yolo)
            # Monkey-patch extract_features MODEL_PATH for this session
            import data.extract_features as ef
            ef.MODEL_PATH = alt_yolo
        if args.force_yolo_train or not YOLO_BEST_PATH.exists():
            run_step("YOLO training", run_yolo_training)
        else:
            logger.info("Skipping YOLO training: weights already present at %s", YOLO_BEST_PATH)
    else:
        logger.info("YOLO training explicitly skipped")

    # Stage 3: ROI feature extraction (YOLO inference)
    if not YOLO_BEST_PATH.exists():
        raise FileNotFoundError(f"YOLO weights not found at {YOLO_BEST_PATH}. Provide weights or enable --force-yolo-train.")
    run_step("ROI feature extraction", extract_features, check_extract_outputs)

    # Stage 4: Temporal harmonization (neuroCombat)
    check_temporal_outputs()
    run_step("Feature harmonization", run_harmonization, check_harmonize_outputs)

    # Stage 5: Causal graph construction
    run_step("Causal graph construction", run_construct_causal, check_graph_outputs)

    # Stage 6: GNN training (5-fold CV)
    if not args.skip_gnn:
        run_step("GNN training", run_kfold_training, check_gnn_outputs)
    else:
        logger.info("GNN training explicitly skipped")

    logger.info("Pipeline execution complete. All outputs written to configured directories.")


if __name__ == "__main__":
    main()
