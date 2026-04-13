"""Backward-compatible config re-export module for Neuro-CXG."""

import logging

from .atlas_config import *
from .feature_registry import *
from .hyperparams import *
from .paths import *
from .validators import (
    get_active_checkpoint_dir,
    log_training_diagnostics,
    validate_environment,
    validate_gnn_training_inputs,
    validate_graph_construction_inputs,
    validate_lobe_mapping,
    validate_training_health,
)

# Keep legacy behavior for modules that rely on global logging defaults from config import.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    validate_environment()
