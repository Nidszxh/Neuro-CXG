"""Model factory utilities for consistent CausalBrainGNN construction."""

from __future__ import annotations

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
    GNN_USE_DEMOGRAPHICS,
    GNN_USE_GRL,
    GNN_USE_SITE_EMBEDDING,
    NUM_LOBES,
)
from src.models.causal_gnn import CausalBrainGNN


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
