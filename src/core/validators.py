"""Validation and runtime helper functions for Neuro-CXG config."""

from __future__ import annotations

import logging
from pathlib import Path

from src.core.atlas_config import LOBE_MAPPING, NUM_LOBES
from src.core.hyperparams import (
    AUC_GOOD_THRESHOLD,
    AUC_RANDOM_THRESHOLD,
    AUC_WEAK_THRESHOLD,
    DEVICE,
    F1_BROKEN_THRESHOLD,
    F1_GOOD_THRESHOLD,
    F1_WEAK_THRESHOLD,
    LOSS_RANDOM_THRESHOLD,
    YOLO_DEGREES,
    YOLO_FLIPLR,
)
from src.core.paths import (
    BASELINE_CHECKPOINT_DIR,
    CAUSAL_GRAPHS_DIR,
    CHECKPOINT_DIR,
    DATA_FINAL,
    DATA_METADATA,
    DATA_ROOT,
    MASTER_MANIFEST,
    NODE_ATTRIBUTES_HARMONIZED,
    NODE_FEATURES_3D,
)
from src.core.feature_registry import GNN_IN_CHANNELS


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_training_health(metrics: dict) -> str:
    """Return a short health-status string for training metrics."""
    auc = metrics.get("auc", 0.5)
    f1 = metrics.get("f1", 0.0)
    loss = metrics.get("loss", 0.693)

    # Critical failures
    if auc < AUC_RANDOM_THRESHOLD:
        return "CRITICAL: Random guessing (AUC < 0.52)"

    if f1 < F1_BROKEN_THRESHOLD and loss > LOSS_RANDOM_THRESHOLD:
        return "CRITICAL: Class collapse (F1 ~= 0, Loss ~= 0.693)"

    # Weak signal
    if auc < AUC_WEAK_THRESHOLD:
        if f1 < F1_WEAK_THRESHOLD:
            return "WARNING: Weak signal, class imbalance likely"
        return "OK: Learning but needs improvement"

    # Good performance
    if auc >= AUC_GOOD_THRESHOLD and f1 >= F1_GOOD_THRESHOLD:
        return "EXCELLENT: Clinical utility achieved"

    return "OK: Model learning"


def log_training_diagnostics(fold: int, epoch: int, metrics: dict) -> None:
    """Log detailed diagnostics for training monitoring."""
    health = validate_training_health(metrics)

    logger.info("\nFold %d, Epoch %d Diagnostics:", fold, epoch)
    logger.info("  Health: %s", health)
    logger.info("  AUC: %.4f (random=0.50, good>=0.70)", metrics["auc"])
    logger.info("  F1: %.4f (broken<0.01, good>=0.50)", metrics["f1"])
    logger.info("  Loss: %.4f (random=0.693, good<0.50)", metrics["loss"])

    if "cm" in metrics:
        tn, fp, fn, tp = metrics["cm"].ravel()
        logger.info("  Confusion Matrix: TN=%s, FP=%s, FN=%s, TP=%s", tn, fp, fn, tp)

        if tp == 0:
            logger.warning("  No true positives! Model predicting all Control.")

        if fp + tn == 0:
            logger.warning("  No negative predictions! Model predicting all ASD.")

    logger.info("")


def validate_lobe_mapping() -> bool:
    """Validate LOBE_MAPPING completeness, uniqueness, and ROI range."""
    # Resolve from src.core.config when available so runtime/monkeypatched
    # config state is honored (backward-compatible with prior monolithic config).
    try:
        import src.core.config as cfg  # Local import avoids hard circular dependency.

        lobe_mapping = getattr(cfg, "LOBE_MAPPING", LOBE_MAPPING)
        num_lobes = getattr(cfg, "NUM_LOBES", NUM_LOBES)
    except Exception:
        lobe_mapping = LOBE_MAPPING
        num_lobes = NUM_LOBES

    if len(lobe_mapping) != num_lobes:
        raise ValueError(
            f"LOBE_MAPPING has {len(lobe_mapping)} regions, expected NUM_LOBES={num_lobes}"
        )

    all_rois: list[int] = []
    for lobe_id, indices in lobe_mapping.items():
        for idx in indices:
            # Range check (0-indexed, so valid range is 0-169)
            if not (0 <= idx <= 169):
                raise ValueError(
                    f"LOBE_MAPPING[{lobe_id}] contains out-of-range index {idx} "
                    f"(1-indexed: {idx + 1}). Valid 1-indexed range is [1, 170]."
                )
            all_rois.append(idx)

    # Duplicate check
    seen: set[int] = set()
    duplicates: list[int] = []
    for idx in all_rois:
        if idx in seen:
            duplicates.append(idx + 1)
        seen.add(idx)
    if duplicates:
        raise ValueError(
            f"LOBE_MAPPING contains duplicate ROI indices (1-indexed): {sorted(set(duplicates))}"
        )

    # Full coverage of 170 AAL ROIs
    expected = set(range(170))
    covered = set(all_rois)
    missing = expected - covered
    if missing:
        missing_1idx = sorted(i + 1 for i in missing)
        raise ValueError(
            f"LOBE_MAPPING does not cover {len(missing)} AAL ROI(s) "
            f"(1-indexed): {missing_1idx}"
        )

    logger.info(
        "validate_lobe_mapping ok: %d regions, %d ROIs, no duplicates, full coverage",
        num_lobes,
        len(all_rois),
    )
    return True


def get_active_checkpoint_dir() -> Path:
    """Return the checkpoint directory containing current fold models."""
    for candidate in (CHECKPOINT_DIR, BASELINE_CHECKPOINT_DIR):
        if candidate.exists() and any(candidate.glob("best_model_fold*.pt")):
            if candidate != CHECKPOINT_DIR:
                logger.debug(
                    "CHECKPOINT_DIR is empty; using baseline checkpoints from %s",
                    candidate,
                )
            return candidate
    return CHECKPOINT_DIR


def validate_environment() -> bool:
    """Check if the 12-region architecture is ready for execution."""
    logger.info("VALIDATING NEURO-CXG 12-REGION ARCHITECTURE")

    for path in [DATA_ROOT, DATA_METADATA, CAUSAL_GRAPHS_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    validate_lobe_mapping()

    # Prevent accidental anatomy-destroying augmentation settings.
    if YOLO_FLIPLR > 0 or YOLO_DEGREES > 0:
        logger.warning("DANGER: YOLO augmentations (fliplr/degrees) are enabled.")
        logger.warning("This can invert hemispheres and collapse classification quality.")

    logger.info("Target: %d nodes | Features: %d", NUM_LOBES, GNN_IN_CHANNELS)
    logger.info("Device: %s", DEVICE)
    return True


def validate_graph_construction_inputs() -> bool:
    """Pre-check before graph construction to ensure all required inputs exist."""
    logger.info("Validating graph construction inputs...")

    errors: list[str] = []

    if not NODE_ATTRIBUTES_HARMONIZED.exists():
        errors.append(f"Missing: {NODE_ATTRIBUTES_HARMONIZED}")

    if not NODE_FEATURES_3D.exists():
        errors.append(f"Missing: {NODE_FEATURES_3D}")

    if not MASTER_MANIFEST.exists():
        errors.append(f"Missing: {MASTER_MANIFEST}")

    if DATA_FINAL.exists():
        ts_count = sum(
            1
            for path in (DATA_FINAL / "train" / "time_series").glob("*.npy")
            if path.is_file()
        )
        if ts_count == 0:
            errors.append(f"No time series files found in {DATA_FINAL / 'train' / 'time_series'}")
    else:
        errors.append(f"Data directory not found: {DATA_FINAL}")

    if errors:
        logger.error("Graph construction validation FAILED:")
        for err in errors:
            logger.error("  %s", err)
        raise FileNotFoundError("\n".join(errors))

    logger.info("Graph construction inputs validated")
    return True


def validate_gnn_training_inputs() -> bool:
    """Pre-check before GNN training to ensure dataset can be loaded."""
    logger.info("Validating GNN training inputs...")

    errors: list[str] = []

    if not CAUSAL_GRAPHS_DIR.exists():
        errors.append(f"Missing causal graphs directory: {CAUSAL_GRAPHS_DIR}")
    else:
        graph_count = sum(1 for path in CAUSAL_GRAPHS_DIR.glob("*.pt") if path.is_file())
        if graph_count == 0:
            errors.append(f"No graph files found in {CAUSAL_GRAPHS_DIR}")
        else:
            logger.info("  Found %d graph files", graph_count)

    if not MASTER_MANIFEST.exists():
        errors.append(f"Missing manifest: {MASTER_MANIFEST}")

    if not NODE_ATTRIBUTES_HARMONIZED.exists():
        errors.append(f"Missing features: {NODE_ATTRIBUTES_HARMONIZED}")

    if errors:
        logger.error("GNN training validation FAILED:")
        for err in errors:
            logger.error("  %s", err)
        raise FileNotFoundError("\n".join(errors))

    logger.info("GNN training inputs validated")
    return True
