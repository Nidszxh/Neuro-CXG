"""Lightweight JSON experiment tracker for Neuro-CXG training runs."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.core import config
from src.core.config import RESULTS_DIR


class ExperimentTracker:
    """Persist structured run metadata and fold metrics for comparison."""

    def __init__(
        self,
        experiment_name: str,
        output_root: Optional[Path] = None,
        run_id: Optional[str] = None,
    ) -> None:
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_root = output_root or (RESULTS_DIR / "experiments" / "runs")
        self.output_dir = self.output_root / self.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.record: Dict[str, Any] = {
            "run_id": self.run_id,
            "experiment": experiment_name,
            "config_hash": self._hash_config(),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "hyperparams": self._capture_hyperparams(),
            "fold_metrics": [],
            "notes": {},
        }
        self._flush()

    def _hash_config(self) -> str:
        """Create a compact hash over relevant training configuration values."""
        relevant: Dict[str, Any] = {
            key: value
            for key, value in vars(config).items()
            if key.startswith(("GNN_", "K_FOLDS", "FOCAL_", "CAUSALITY_"))
        }
        serialized = json.dumps(
            {k: str(v) for k, v in sorted(relevant.items())},
            sort_keys=True,
        )
        return hashlib.md5(serialized.encode("utf-8")).hexdigest()[:8]

    def _capture_hyperparams(self) -> Dict[str, Any]:
        """Capture key model/training hyperparameters for run comparisons."""
        return {
            "hidden_channels": config.GNN_HIDDEN_CHANNELS,
            "num_heads": config.GNN_NUM_HEADS,
            "dropout": config.GNN_DROPOUT,
            "max_lr": config.GNN_ONECYCLE_MAX_LR,
            "focal_alpha": config.FOCAL_LOSS_ALPHA,
            "focal_gamma": config.FOCAL_LOSS_GAMMA,
            "batch_size": config.GNN_BATCH_SIZE,
            "epochs": config.GNN_EPOCHS,
            "k_folds": config.K_FOLDS,
        }

    def add_note(self, key: str, value: Any) -> None:
        """Attach run-level metadata that is not naturally fold-scoped."""
        self.record["notes"][key] = value
        self._flush()

    def log_fold(self, fold: int, metrics: Dict[str, Any]) -> None:
        """Append fold metrics and flush to disk immediately."""
        self.record["fold_metrics"].append({"fold": int(fold), **metrics})
        self._flush()

    def finalize(self, summary: Dict[str, Any]) -> None:
        """Store final summary and completion timestamp."""
        self.record["summary"] = summary
        self.record["completed_at"] = datetime.now().isoformat(timespec="seconds")
        self._flush()

    def _flush(self) -> None:
        """Write current run record to disk."""
        with (self.output_dir / "run.json").open("w", encoding="utf-8") as handle:
            json.dump(self.record, handle, indent=2, default=str)
