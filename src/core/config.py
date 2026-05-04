"""Backward-compatible config re-export module for Neuro-CXG."""

import logging

from .atlas_config import *
from .feature_registry import *
from .hyperparams import *
from .paths import *  # includes CAUSAL_GRAPHS_MULTIVIEW_DIR (Task 2)
from .validators import (
    validate_environment,
    validate_gnn_training_inputs,
    get_active_checkpoint_dir,
    validate_graph_construction_inputs,
)

# Keep legacy behavior for modules that rely on global logging defaults from config import.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    validate_environment()
