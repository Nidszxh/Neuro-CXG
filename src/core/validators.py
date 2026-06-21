"""Validation and runtime helper functions for Neuro-CXG config."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from src.core.atlas_config import LOBE_MAPPING, NUM_LOBES
from src.core.feature_registry import GNN_IN_CHANNELS
from src.core.hyperparams import (
    DEVICE,
    K_FOLDS,
    YOLO_DEGREES,
    YOLO_FLIPLR,
)
from src.core.paths import (
    BASELINE_CHECKPOINT_DIR,
    CAUSAL_GRAPHS_DIR,
    CHECKPOINT_DIR,
    DATA_METADATA,
    DATA_ROOT,
    HARMONIZED_FOLDS_DIR,
    MASTER_MANIFEST,
    NODE_ATTRIBUTES_HARMONIZED,
)

logger = logging.getLogger(__name__)


def summarize_graph_degeneracy_from_adj(
    adj: torch.Tensor,
    min_edges: int,
) -> dict[str, int | bool]:
    """Summarize edge/dead-lobe degeneracy from an adjacency matrix."""
    adj_t = torch.as_tensor(adj, dtype=torch.float32)
    if adj_t.ndim != 2 or adj_t.shape[0] != adj_t.shape[1]:
        raise ValueError(
            f"Expected square adjacency matrix, got shape={tuple(adj_t.shape)}"
        )

    n = int(adj_t.shape[0])
    offdiag = ~torch.eye(n, dtype=torch.bool, device=adj_t.device)
    edge_mask = (adj_t != 0) & offdiag
    edge_count = int(edge_mask.sum().item())

    in_deg = edge_mask.sum(dim=0)
    out_deg = edge_mask.sum(dim=1)
    dead_lobes = int(((in_deg == 0) & (out_deg == 0)).sum().item())

    return {
        "edge_count": edge_count,
        "dead_lobes": dead_lobes,
        "is_degenerate": bool(edge_count < int(min_edges) or dead_lobes > 0),
    }


def summarize_graph_degeneracy_from_edge_index(
    edge_index: torch.Tensor | None,
    num_nodes: int,
    min_edges: int,
) -> dict[str, int | bool]:
    """Summarize edge/dead-lobe degeneracy from COO edge indices."""
    n = int(num_nodes)
    if edge_index is None or not torch.is_tensor(edge_index) or edge_index.numel() == 0:
        return {
            "edge_count": 0,
            "dead_lobes": n,
            "is_degenerate": True,
        }

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            f"Expected edge_index shape (2, E), got {tuple(edge_index.shape)}"
        )

    e = int(edge_index.shape[1])
    device = edge_index.device
    out_deg = torch.zeros(n, dtype=torch.long, device=device)
    in_deg = torch.zeros(n, dtype=torch.long, device=device)
    ones = torch.ones(e, dtype=torch.long, device=device)
    out_deg.scatter_add_(0, edge_index[0].long(), ones)
    in_deg.scatter_add_(0, edge_index[1].long(), ones)
    dead_lobes = int(((in_deg == 0) & (out_deg == 0)).sum().item())

    return {
        "edge_count": e,
        "dead_lobes": dead_lobes,
        "is_degenerate": bool(e < int(min_edges) or dead_lobes > 0),
    }


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

    # Full coverage of AAL ROIs
    # For 12-lobe mode: require all 170 ROIs
    # For 11-lobe mode: require first 166 ROIs (exclude Brainstem 167-170)
    from src.core.atlas_config import USE_11_LOBES

    if USE_11_LOBES:
        expected = set(range(166))  # Standard 166 AAL ROIs (excluding Brainstem)
    else:
        expected = set(range(170))  # All 170 AAL ROIs

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
    """Check if the multi-lobe architecture is ready for execution."""
    logger.info(f"VALIDATING NEURO-CXG {NUM_LOBES}-REGION ARCHITECTURE")

    for path in [DATA_ROOT, DATA_METADATA, CAUSAL_GRAPHS_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    validate_lobe_mapping()

    # Prevent accidental anatomy-destroying augmentation settings.
    if YOLO_FLIPLR > 0 or YOLO_DEGREES > 0:
        logger.warning("DANGER: YOLO augmentations (fliplr/degrees) are enabled.")
        logger.warning(
            "This can invert hemispheres and collapse classification quality."
        )

    logger.info("Target: %d nodes | Features: %d", NUM_LOBES, GNN_IN_CHANNELS)
    logger.info("Device: %s", DEVICE)
    return True


def validate_gnn_training_inputs() -> bool:
    """Pre-check before GNN training to ensure dataset can be loaded."""
    logger.info("Validating GNN training inputs...")

    errors: list[str] = []

    if not CAUSAL_GRAPHS_DIR.exists():
        errors.append(f"Missing causal graphs directory: {CAUSAL_GRAPHS_DIR}")
    else:
        graph_count = sum(
            1 for path in CAUSAL_GRAPHS_DIR.glob("*.pt") if path.is_file()
        )
        if graph_count == 0:
            errors.append(f"No graph files found in {CAUSAL_GRAPHS_DIR}")
        else:
            logger.info("  Found %d graph files", graph_count)

    if not MASTER_MANIFEST.exists():
        errors.append(f"Missing manifest: {MASTER_MANIFEST}")

    if not NODE_ATTRIBUTES_HARMONIZED.exists():
        errors.append(f"Missing features: {NODE_ATTRIBUTES_HARMONIZED}")

    missing_harmonized_folds = [
        fold
        for fold in range(K_FOLDS)
        if not (HARMONIZED_FOLDS_DIR / f"harmonized_fold_{fold}.csv").exists()
    ]
    if missing_harmonized_folds:
        missing_details = ", ".join(
            f"fold {fold} (harmonized_fold_{fold}.csv)"
            for fold in missing_harmonized_folds
        )
        errors.append(
            "Missing harmonized fold files: "
            f"{missing_details}. Directory: {HARMONIZED_FOLDS_DIR}"
        )

    if errors:
        logger.error("GNN training validation FAILED:")
        for err in errors:
            logger.error("  %s", err)
        raise FileNotFoundError("\n".join(errors))

    logger.info("GNN training inputs validated")
    return True
