"""Model factory utilities for consistent CausalBrainGNN construction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from src.core.config import (
    GNN_DROPOUT,
    GNN_EDGE_GATE,
    GNN_GRL_ALPHA,
    GNN_HIDDEN_CHANNELS,
    GNN_IN_CHANNELS,
    GNN_NODE_EMB_DIM,
    GNN_NUM_HEADS,
    GNN_NUM_LAYERS,
    GNN_POOLING,
    GNN_SITE_EMBEDDING_DIM,
    GNN_USE_DEMOGRAPHICS,
    GNN_USE_GRL,
    GNN_USE_SITE_EMBEDDING,
    NUM_LOBES,
    get_active_checkpoint_dir,
)
from src.models.causal_gnn import CausalBrainGNN

if TYPE_CHECKING:
    from src.models.training_utils import attach_feature_scaler_from_checkpoint

logger = logging.getLogger(__name__)


def build_model(
    device: torch.device | None = None,
    *,
    use_grl: bool = GNN_USE_GRL,
    grl_alpha: float = GNN_GRL_ALPHA,
    num_sites: int = 20,
    **overrides,
) -> CausalBrainGNN:
    """Build a CausalBrainGNN with project defaults and optional overrides."""
    params = {
        "num_node_features": GNN_IN_CHANNELS,
        "hidden_channels": GNN_HIDDEN_CHANNELS,
        "num_classes": 2,
        "num_heads": GNN_NUM_HEADS,
        "num_layers": GNN_NUM_LAYERS,
        "pooling": GNN_POOLING,
        "dropout": GNN_DROPOUT,
        "use_site_embedding": GNN_USE_SITE_EMBEDDING,
        "use_demographics": GNN_USE_DEMOGRAPHICS,
        "use_grl": use_grl,
        "grl_alpha": grl_alpha,
        "edge_gate": GNN_EDGE_GATE,
        "num_nodes": NUM_LOBES,
        "node_emb_dim": GNN_NODE_EMB_DIM,
        "num_sites": num_sites,
    }
    params.update(overrides)

    model = CausalBrainGNN(**params)
    if device is not None:
        model = model.to(device)
    return model


def load_model(
    fold_id: int | None = None,
    *,
    checkpoint_path: Path | None = None,
    device: torch.device | None = None,
    attach_scaler: bool = True,
) -> CausalBrainGNN:
    """
    Load a trained CausalBrainGNN checkpoint.

    Supports two calling conventions::

        load_model(fold_id=3)                    # → active_dir / best_model_fold3.pt
        load_model(checkpoint_path=Path("..."))   # explicit path

    Architecture knobs (site_embedding, use_demographics, node_emb_dim) are
    inferred from the checkpoint tensor shapes so that evaluation remains
    compatible when config defaults drift between training and evaluation runs.

    Args:
        fold_id: Fold index (0–K-1). Mutually exclusive with *checkpoint_path*.
        checkpoint_path: Explicit path to a ``best_model_fold*.pt`` file.
            Mutually exclusive with *fold_id*.
        device: Target device. Defaults to CUDA if available.
        attach_scaler: Whether to attach the feature scaler from the
            checkpoint (recommended for evaluation).
    """
    if (fold_id is None) == (checkpoint_path is None):
        raise ValueError("Pass exactly one of fold_id or checkpoint_path")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if fold_id is not None:
        checkpoint_path = get_active_checkpoint_dir() / f"best_model_fold{fold_id}.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)

    has_site_embedding = any(k.startswith("site_embedding.") for k in state)
    site_dim = GNN_SITE_EMBEDDING_DIM if has_site_embedding else 0

    saved_lin_in = state.get("lin_in.weight")
    if saved_lin_in is None:
        raise KeyError(f"Checkpoint missing required key: lin_in.weight ({checkpoint_path})")

    saved_in_features = int(saved_lin_in.shape[1])
    node_emb_dim = max(saved_in_features - GNN_IN_CHANNELS - site_dim, 0)

    model = build_model(
        device=device,
        use_grl=GNN_USE_GRL,
        grl_alpha=GNN_GRL_ALPHA,
        hidden_channels=int(saved_lin_in.shape[0]),
        use_site_embedding=has_site_embedding,
        node_emb_dim=node_emb_dim,
    )

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning("Missing keys in checkpoint: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys in checkpoint: %s", unexpected)

    if attach_scaler:
        from src.models.training_utils import attach_feature_scaler_from_checkpoint
        attach_feature_scaler_from_checkpoint(model, ckpt, expected_dim=GNN_IN_CHANNELS)

    model.eval()
    return model
