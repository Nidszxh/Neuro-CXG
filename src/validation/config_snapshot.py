"""
Configuration Snapshot Utility

Captures active hyperparameter values for run artifacts.
Enables reproducibility by logging which config values were active during a specific run.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def get_config_snapshot(module: Any = None) -> dict[str, str]:
    """Capture all uppercase config values from a module.

    Args:
        module: Config module (default: src.core.hyperparams)

    Returns:
        Dict of config key-value pairs
    """
    if module is None:
        try:
            from src.core import hyperparams as module
        except ImportError:
            return {}

    snapshot = {}
    for k in dir(module):
        if k.isupper() and not k.startswith("_"):
            try:
                val = getattr(module, k)
                if isinstance(val, (set, frozenset)):
                    snapshot[k] = str(sorted(val))
                else:
                    snapshot[k] = str(val)
            except Exception:
                snapshot[k] = "<error>"

    return snapshot


def get_config_hash(module: Any = None) -> str:
    """Get hash of config snapshot for artifact versioning.

    Args:
        module: Config module (default: src.core.hyperparams)

    Returns:
        8-character config hash
    """
    snapshot = get_config_snapshot(module)
    config_str = json.dumps(snapshot, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:8]


def save_config_snapshot(
    output_dir: Path,
    name: str = "config_snapshot.json",
    module: Any = None,
) -> Path:
    """Save config snapshot to run artifact directory.

    Args:
        output_dir: Directory to save snapshot
        name: Filename for snapshot
        module: Config module to snapshot

    Returns:
        Path to saved snapshot
    """
    snapshot = get_config_snapshot(module)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_path = output_dir / name
    with open(save_path, "w") as f:
        json.dump(snapshot, f, indent=2)

    logger.info(f"Config snapshot saved to {save_path}")
    return save_path
