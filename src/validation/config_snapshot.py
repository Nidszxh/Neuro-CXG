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


def load_config_snapshot(snapshot_path: Path) -> dict[str, str]:
    """Load config snapshot from file."""
    with open(snapshot_path) as f:
        return json.load(f)


def compare_snapshots(snapshot_a: dict[str, str], snapshot_b: dict[str, str]) -> dict[str, tuple]:
    """Compare two config snapshots and return differences.

    Args:
        snapshot_a: First config snapshot (from load_config_snapshot)
        snapshot_b: Second config snapshot

    Returns:
        Dict of key -> (value_a, value_b, delta) for changed keys
    """
    differences = {}

    all_keys = set(snapshot_a.keys()) | set(snapshot_b.keys())

    for key in sorted(all_keys):
        val_a = snapshot_a.get(key, "<missing>")
        val_b = snapshot_b.get(key, "<missing>")

        if val_a != val_b:
            differences[key] = (val_a, val_b, _compute_delta(val_a, val_b))

    return differences


def _compute_delta(val_a: str, val_b: str) -> str:
    """Compute human-readable delta between two values."""
    try:
        num_a = float(val_a)
        num_b = float(val_b)
        delta = num_b - num_a
        if abs(delta) < 0.001:
            return "~0"
        return f"{delta:+.4f}"
    except (ValueError, TypeError):
        return f"→ {val_b}"


def print_snapshot_diff(differences: dict[str, tuple], file=None) -> None:
    """Pretty-print snapshot differences.

    Args:
        differences: Output from compare_snapshots
        file: Optional file handle for output
    """
    if not differences:
        print("No differences found", file=file)
        return

    print("=" * 80, file=file)
    print("CONFIG SNAPSHOT DIFFERENCES", file=file)
    print("=" * 80, file=file)

    for key, (val_a, val_b, delta) in sorted(differences.items()):
        print(f"\n{key}:", file=file)
        print(f"  A: {val_a}", file=file)
        print(f"  B: {val_b}", file=file)
        print(f"  Δ: {delta}", file=file)

    print("\n" + "=" * 80, file=file)
    print(f"Total: {len(differences)} changed parameters", file=file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare config snapshots")
    parser.add_argument("snapshot_a", type=Path, help="First snapshot JSON file")
    parser.add_argument("snapshot_b", type=Path, help="Second snapshot JSON file")
    parser.add_argument("--output", type=Path, default=None, help="Output file (optional)")

    args = parser.parse_args()

    snap_a = load_config_snapshot(args.snapshot_a)
    snap_b = load_config_snapshot(args.snapshot_b)

    diffs = compare_snapshots(snap_a, snap_b)

    output_file = open(args.output, "w") if args.output else None
    print_snapshot_diff(diffs, file=output_file)
    if output_file:
        output_file.close()
        print(f"\nDiff saved to {args.output}")
