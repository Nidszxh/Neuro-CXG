"""
Training utilities for GNN model.

Provides reusable components for training loops:
- EarlyStopping: Monitor validation metrics with patience
- WarmupScheduler: Linear learning rate warmup
- TrainingTracker: Track metrics across epochs/folds
- CheckpointManager: Save/load model checkpoints
- _apply_structural_dropout: Zero node features for structural learning (Task 1, DD-009)
- EdgeStructureContrastiveLoss: NT-Xent loss for edge-structure enforcement (Task 1, DD-009)

These utilities reduce code duplication and improve maintainability
while keeping PyTorch raw (no pytorch-lightning dependency).
"""

import hashlib
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from src.core.config import (
    EVAL_FIXED_THRESHOLD,
    EVAL_THRESHOLD_POLICY,
    GNN_GRL_ALPHA_MAX,
    GNN_ONECYCLE_PCT_START,
)
from src.core.hyperparams import _MULTIVIEW_VIEW_ORDER
from src.models.evaluation import evaluate_loader, resolve_threshold

logger = logging.getLogger(__name__)


def attach_feature_scaler_from_checkpoint(
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    expected_dim: int | None = None,
) -> bool:
    """Attach fold-wise preprocessing metadata from checkpoint to a model.

    Training stores fold-internal preprocessing artifacts in checkpoints:
    - ``feature_mean`` / ``feature_std`` (global train-fold stats)
    - ``feature_mask`` (MI-selected channel mask)
    - ``site_feature_means`` / ``site_feature_stds`` (per-site train-fold stats)

    Attaching them to the model enables inference-time parity via
    ``CausalBrainGNN._encode`` while remaining backward-compatible with older
    checkpoints that only contain mean/std.

    Returns True when at least one preprocessing artifact was attached.
    """
    if not isinstance(checkpoint, dict):
        return False

    attached = False

    # Global scaler stats (legacy + current).
    feature_mean = checkpoint.get("feature_mean")
    feature_std = checkpoint.get("feature_std")
    if feature_mean is not None and feature_std is not None:
        try:
            mean_t = torch.as_tensor(feature_mean, dtype=torch.float32).view(-1)
            std_t = torch.as_tensor(feature_std, dtype=torch.float32).view(-1)
            if mean_t.numel() == 0 or std_t.numel() == 0 or mean_t.numel() != std_t.numel():
                raise ValueError("invalid scaler shape")
            if expected_dim is not None and mean_t.numel() != int(expected_dim):
                raise ValueError("scaler dim mismatch")
            model._feature_mean = mean_t
            model._feature_std = std_t.clamp_min(1e-06)
            attached = True
        except Exception:
            logger.warning("Ignoring malformed checkpoint feature scaler metadata")

    target_dim = int(expected_dim) if expected_dim is not None else None

    # Fold-internal MI feature mask.
    feature_mask = checkpoint.get("feature_mask")
    if feature_mask is not None:
        try:
            mask_t = torch.as_tensor(feature_mask, dtype=torch.float32).view(-1)
            if target_dim is not None and mask_t.numel() != target_dim:
                raise ValueError("feature mask dim mismatch")
            model._feature_mask = mask_t
            attached = True
        except Exception:
            logger.warning("Ignoring malformed checkpoint feature_mask metadata")

    selected_idx = checkpoint.get("selected_feature_idx")
    if isinstance(selected_idx, (list, tuple)):
        try:
            model._selected_feature_idx = [int(i) for i in selected_idx]
        except Exception:
            logger.warning("Ignoring malformed checkpoint selected_feature_idx metadata")

    preprocessing_mode = checkpoint.get("preprocessing_mode")
    if isinstance(preprocessing_mode, str) and preprocessing_mode.strip():
        model._preprocessing_mode = preprocessing_mode.strip().lower()

    site_norm_mode = checkpoint.get("site_normalization_mode")
    if isinstance(site_norm_mode, str) and site_norm_mode.strip():
        model._site_normalization_mode = site_norm_mode.strip().lower()

    # Per-site stats for within-site normalization fallback at inference time.
    site_means_raw = checkpoint.get("site_feature_means")
    site_stds_raw = checkpoint.get("site_feature_stds")
    if isinstance(site_means_raw, dict) and isinstance(site_stds_raw, dict):
        site_means: dict[int, torch.Tensor] = {}
        site_stds: dict[int, torch.Tensor] = {}
        bad_sites = 0
        for sid_key, mean_vals in site_means_raw.items():
            std_vals = site_stds_raw.get(sid_key)
            if std_vals is None:
                bad_sites += 1
                continue
            try:
                sid = int(sid_key)
                mean_t = torch.as_tensor(mean_vals, dtype=torch.float32).view(-1)
                std_t = torch.as_tensor(std_vals, dtype=torch.float32).view(-1).clamp_min(1e-6)
                if mean_t.numel() == 0 or mean_t.numel() != std_t.numel():
                    raise ValueError("invalid site scaler shape")
                if target_dim is not None and mean_t.numel() != target_dim:
                    raise ValueError("site scaler dim mismatch")
                site_means[sid] = mean_t
                site_stds[sid] = std_t
            except Exception:
                bad_sites += 1

        if site_means and site_stds:
            model._site_feature_means = site_means
            model._site_feature_stds = site_stds
            attached = True
        if bad_sites > 0:
            logger.warning("Ignored malformed site-normalization metadata for %d site(s)", bad_sites)

    return attached


class _MultiviewCache:
    """Lightweight in-memory LRU cache for multiview graph packages."""

    def __init__(self, maxsize: int = 512):
        self._maxsize = max(1, int(maxsize))
        self._cache: OrderedDict[str, dict[str, torch.Tensor]] = OrderedDict()

    def get(self, key: str) -> dict[str, torch.Tensor] | None:
        value = self._cache.get(key)
        if value is not None:
            self._cache.move_to_end(key)
        return value

    def set(self, key: str, value: dict[str, torch.Tensor]) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        """Clear cache for new training run."""
        self._cache.clear()


_multiview_cache: _MultiviewCache | None = None


def get_multiview_cache(maxsize: int = 512) -> _MultiviewCache:
    """Get or create the multiview cache (per-run scoped)."""
    global _multiview_cache
    if _multiview_cache is None:
        _multiview_cache = _MultiviewCache(maxsize=maxsize)
    return _multiview_cache


def reset_multiview_cache(maxsize: int = 512) -> None:
    """Reset multiview cache for new training run."""
    global _multiview_cache
    _multiview_cache = _MultiviewCache(maxsize=maxsize)


def make_loader(
    dataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 4,
) -> DataLoader:
    """Create a tuned torch_geometric DataLoader for small-graph workloads."""
    cpu_count = os.cpu_count() or 4
    effective_workers = min(num_workers, cpu_count)
    if len(dataset) < 800 and not torch.cuda.is_available():
        effective_workers = min(effective_workers, 2)

    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": effective_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": effective_workers > 0,
        "drop_last": bool(shuffle),
    }
    if effective_workers > 0:
        kwargs["prefetch_factor"] = 4
    return DataLoader(dataset, **kwargs)


class EarlyStopping:
    """
    Early stopping with patience and minimum delta.

    Monitors a validation metric and stops training if it doesn't improve
    for a specified number of epochs.

    Args:
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as improvement
        mode: 'min' for loss (lower is better), 'max' for AUC/F1 (higher is better)

    Example:
        early_stop = EarlyStopping(patience=10, mode='max')
        for epoch in range(100):
            val_auc = validate(model, val_loader)
            if early_stop(val_auc):
                break
    """
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: float | None = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """Check if training should stop. Returns True if stopping criterion met."""
        if self.best_score is None:
            self.best_score = score
            return False

        improved = False
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
                return True

        return False

    def reset(self):
        """Reset state for new training run."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


class WarmupScheduler:
    """
    Linear learning rate warmup.

    Gradually increases learning rate from 0 to base_lr over warmup_epochs.

    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of epochs for warmup
        base_lr: Target learning rate after warmup
    """
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_epochs: int, base_lr: float):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.current_epoch = 0

    def step(self):
        """Increment epoch and update learning rate."""
        if self.current_epoch < self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.current_epoch += 1

    def reset(self):
        """Reset for new training run."""
        self.current_epoch = 0


@dataclass
class FoldMetrics:
    """Stores metrics for a single fold."""
    fold: int
    auc: float
    f1: float
    acc: float
    threshold: float
    best_epoch: int
    train_time: float = 0.0
    val_probs: np.ndarray | None = None
    val_labels: np.ndarray | None = None


class TrainingTracker:
    """
    Track training metrics across epochs and folds.

    Aggregates results from k-fold cross-validation and provides summary statistics.

    Example:
        tracker = TrainingTracker(k_folds=5)
        for fold in range(5):
            tracker.add_fold_result(fold=fold, auc=0.85, f1=0.80, acc=0.78,
                                    threshold=0.55, best_epoch=45)
        summary = tracker.get_summary()
    """
    def __init__(self, k_folds: int):
        self.k_folds = k_folds
        self.fold_results: list[FoldMetrics] = []

    def add_fold_result(self, fold: int, auc: float, f1: float, acc: float,
                        threshold: float, best_epoch: int, train_time: float = 0.0,
                        val_probs: np.ndarray | None = None,
                        val_labels: np.ndarray | None = None):
        """Add results from a completed fold."""
        self.fold_results.append(FoldMetrics(
            fold=fold, auc=auc, f1=f1, acc=acc, threshold=threshold,
            best_epoch=best_epoch, train_time=train_time,
            val_probs=val_probs, val_labels=val_labels,
        ))

    def get_summary(self) -> dict[str, Any]:
        """Compute summary statistics across all folds."""
        if not self.fold_results:
            return {}

        aucs = [r.auc for r in self.fold_results]
        f1s = [r.f1 for r in self.fold_results]
        accs = [r.acc for r in self.fold_results]
        thresholds = [r.threshold for r in self.fold_results]
        epochs = [r.best_epoch for r in self.fold_results]

        return {
            'mean_auc': np.mean(aucs),
            'std_auc': np.std(aucs),
            'mean_f1': np.mean(f1s),
            'std_f1': np.std(f1s),
            'mean_acc': np.mean(accs),
            'std_acc': np.std(accs),
            'mean_threshold': np.mean(thresholds),
            'mean_best_epoch': np.mean(epochs),
            'per_fold_aucs': aucs,
            'per_fold_f1s': f1s,
            'per_fold_epochs': epochs,
        }

    def log_summary(self):
        """Log summary statistics."""
        summary = self.get_summary()
        if not summary:
            logger.warning("No fold results to summarize")
            return

        logger.info(f"\n{'='*70}")
        logger.info("FINAL CROSS-VALIDATION RESULTS")
        logger.info(f"{'='*70}")
        logger.info(f"Mean AUC: {summary['mean_auc']:.4f} ± {summary['std_auc']:.4f}")
        logger.info(f"Mean F1: {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}")
        logger.info(f"Mean Accuracy: {summary['mean_acc']:.4f} ± {summary['std_acc']:.4f}")
        logger.info(f"Mean Threshold: {summary['mean_threshold']:.3f}")
        logger.info(f"Mean Best Epoch: {summary['mean_best_epoch']:.1f}")
        logger.info(f"\nPer-fold AUCs: {[f'{x:.4f}' for x in summary['per_fold_aucs']]}")
        logger.info(f"Per-fold F1s: {[f'{x:.4f}' for x in summary['per_fold_f1s']]}")
        logger.info(f"Per-fold Best Epochs: {summary['per_fold_epochs']}")
        logger.info(f"{'='*70}\n")

    def get_ensemble_predictions(self) -> tuple | None:
        """
        Get ensemble predictions by concatenating validation predictions from all folds.

        Returns:
            (probs, labels) tuple or None if no predictions stored
        """
        probs_list = [r.val_probs for r in self.fold_results if r.val_probs is not None]
        labels_list = [r.val_labels for r in self.fold_results if r.val_labels is not None]

        if not probs_list:
            return None

        return np.concatenate(probs_list), np.concatenate(labels_list)


class CheckpointManager:
    """
    Manage model checkpoints during training.

    Args:
        checkpoint_dir: Directory to save checkpoints
        monitor: Metric to monitor ('auc', 'f1', 'loss')
        mode: 'max' (higher is better) or 'min' (lower is better)
    """
    def __init__(self, checkpoint_dir: Path, monitor: str = 'auc', mode: str = 'max'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.best_score = float('-inf') if mode == 'max' else float('inf')
        self.run_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]

    def should_save(self, score: float) -> bool:
        if self.mode == 'max':
            return score > self.best_score
        else:
            return score < self.best_score

    def save(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None,
             epoch: int, metrics: dict[str, Any], fold: int | None = None):
        """Save model checkpoint with metadata."""
        score = metrics.get(self.monitor)
        if score is None:
            logger.warning(f"Metric '{self.monitor}' not found in metrics")
            return

        if self.should_save(score):
            self.best_score = score
            filename = f"best_model_fold{fold}.pt" if fold is not None else "best_model.pt"
            filepath = self.checkpoint_dir / filename
            checkpoint = {
                'model_state': model.state_dict(),
                'epoch': epoch,
                'run_id': self.run_id,
                'timestamp': time.strftime("%Y%m%d_%H%M%S"),
                **metrics,
            }
            if optimizer is not None:
                checkpoint['optimizer_state'] = optimizer.state_dict()
            torch.save(checkpoint, filepath)
            logger.info(f"✓ Saved checkpoint: {filename} (epoch {epoch}, {self.monitor}={score:.4f})")

    def load(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None,
             fold: int | None = None, allow_partial: bool = False) -> dict[str, Any]:
        """Load model checkpoint."""
        filename = f"best_model_fold{fold}.pt" if fold is not None else "best_model.pt"
        filepath = self.checkpoint_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        checkpoint = torch.load(filepath, weights_only=True)

        saved_auc = checkpoint.get('auc')
        if saved_auc is None:
            logger.warning("Checkpoint %s has no 'auc' metric.", filename)
        elif saved_auc < 0.60:
            logger.warning(
                "Loaded checkpoint %s has low AUC=%.4f (<0.60). "
                "This may be a collapsed fold; canonical run target is ~0.74 CV AUC.",
                filename, saved_auc,
            )

        if allow_partial:
            model_state = model.state_dict()
            checkpoint_state = checkpoint['model_state']
            compatible_state = {}
            skipped_keys = []
            for key, value in checkpoint_state.items():
                if key in model_state and model_state[key].shape == value.shape:
                    compatible_state[key] = value
                else:
                    skipped_keys.append(key)
            model.load_state_dict(compatible_state, strict=False)
            if skipped_keys:
                logger.warning(
                    f"Loaded checkpoint with {len(skipped_keys)} incompatible keys skipped. "
                    "This is expected when comparing models with different input dimensions."
                )
        else:
            model.load_state_dict(checkpoint['model_state'])

        if optimizer is not None and 'optimizer_state' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state'])

        scaler_attached = attach_feature_scaler_from_checkpoint(model, checkpoint)
        if scaler_attached:
            logger.info("Loaded fold preprocessing metadata from %s", filename)

        logger.info(f"Loaded checkpoint: {filename} (epoch {checkpoint.get('epoch', 'unknown')})")
        return checkpoint

    def reset(self):
        """Reset best score for new training run."""
        self.best_score = float('-inf') if self.mode == 'max' else float('inf')


# ── TASK 1: Structural Learning Enforcement (DD-009) ────────────────────────────

def _apply_structural_dropout(
    batch,
    dropout_prob: float = 0.30,
    training: bool = True,
):
    """Zero all node features for ~30% of graphs per batch during training.

    Forces those graphs to be classified from edge structure alone, enforcing
    that the model learns to use causal graph topology (not just node features).

    Rationale (DD-009 / Root Cause 1): The model uses node features as the
    primary signal; edge structure is secondary. By zeroing node features for
    a random subset of graphs in each batch we deprive the model of its main
    signal and force gradient updates through the edge-gating and GAT paths.

    At evaluation time (training=False) the batch is returned unchanged.

    Args:
        batch: PyG Batch object with .x (N_total, F) and .batch (N_total,) tensors.
        dropout_prob: Fraction of graphs to zero. Default 0.30 (30%).
        training: Only applies when True; returns batch unchanged at eval time.

    Returns:
        Modified Batch clone — original loader data is NOT mutated.
    """
    if not training or dropout_prob <= 0.0:
        return batch

    batch = batch.clone()
    num_graphs = int(batch.batch.max().item()) + 1

    # Draw per-graph zero mask
    zero_graph_mask = torch.rand(num_graphs, device=batch.x.device) < dropout_prob  # (G,)

    # Expand to node level via assignment vector
    node_zero_mask = zero_graph_mask[batch.batch]  # (N_total,)

    batch.x = batch.x.clone()
    batch.x[node_zero_mask] = 0.0
    return batch


class EdgeStructureContrastiveLoss(nn.Module):
    """NT-Xent style contrastive loss that enforces structural learning.

    Encourages the model to produce similar graph-level representations for
    the same graph viewed under two conditions:
      - Full features  (all node features available)
      - Edge-only view (node features zeroed by structural dropout)

    If the model truly extracts information from causal edges, these two
    views should be close in embedding space.  High loss indicates that the
    model's class signal lives almost entirely in node features (the failure mode).

    Rationale (DD-009): Complementary to structural dropout — the dropout
    forces edge-based gradients during training; this loss explicitly
    pushes full-feature and edge-only representations toward alignment.

    Args:
        temperature: Softmax temperature τ. Default 0.5.
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_full: torch.Tensor,       # (B, D) embeddings from full-feature pass
        z_edge_only: torch.Tensor,  # (B, D) embeddings from edge-only pass
    ) -> torch.Tensor:
        B = z_full.size(0)
        if B < 2:
            return torch.tensor(0.0, device=z_full.device, requires_grad=True)

        z_f = F.normalize(z_full, dim=1)
        z_e = F.normalize(z_edge_only, dim=1)

        # Concatenate both views → (2B, D)
        z = torch.cat([z_f, z_e], dim=0)

        # Pairwise cosine similarity (2B, 2B) / τ
        sim = torch.mm(z, z.t()) / self.temperature

        # Mask out self-similarity from denominator
        eye = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(eye, float('-inf'))

        # Positive pairs: index i (full) ↔ i+B (edge-only)
        labels = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(0,  B, device=z.device),
        ])

        return F.cross_entropy(sim, labels)


def _expand_site_targets_to_nodes(batch) -> torch.Tensor | None:
    """Expand per-graph site labels to per-node labels using ``batch.batch``."""
    site_targets = getattr(batch, "site_id", None)
    if site_targets is None or not torch.is_tensor(site_targets):
        return None

    site_targets = site_targets.view(-1)
    if site_targets.numel() == 0:
        return None

    num_graphs = int(batch.batch.max().item()) + 1
    if site_targets.numel() != num_graphs:
        return None

    return site_targets[batch.batch]


def _extract_batch_subject_ids(batch, num_graphs: int) -> list[str] | None:
    """Extract per-graph subject IDs from a PyG batch.

    Handles common collate representations:
    - list/tuple of subject IDs (preferred)
    - scalar string for single-graph batches
    - node-level repeated subject IDs (compressed by first node per graph)
    """
    raw_ids = getattr(batch, "sub_id", None)
    if raw_ids is None:
        return None

    if isinstance(raw_ids, str):
        ids = [raw_ids]
    elif isinstance(raw_ids, (list, tuple)):
        ids = [str(x) for x in raw_ids]
    elif isinstance(raw_ids, np.ndarray):
        ids = [str(x) for x in raw_ids.tolist()]
    elif torch.is_tensor(raw_ids):
        ids = [str(x) for x in raw_ids.view(-1).tolist()]
    else:
        ids = [str(raw_ids)]

    if len(ids) == num_graphs:
        return ids

    # Some collations may carry node-level repeated IDs; compress to graph-level.
    if len(ids) == batch.x.shape[0]:
        compressed = []
        for g_idx in range(num_graphs):
            node_ids = (batch.batch == g_idx).nonzero(as_tuple=False).view(-1)
            if node_ids.numel() == 0:
                return None
            compressed.append(ids[int(node_ids[0].item())])
        return compressed

    return None


def _load_multiview_package(file_path: Path, cache: _MultiviewCache | None = None) -> dict[str, torch.Tensor] | None:
    """Load and normalize one subject's multiview adjacency package."""
    if cache is None:
        cache = get_multiview_cache()

    cache_key = str(file_path)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        payload = torch.load(file_path, map_location="cpu", weights_only=True)
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    views = payload.get("views", payload)
    if not isinstance(views, dict):
        return None

    normalized: dict[str, torch.Tensor] = {}
    for name, value in views.items():
        if torch.is_tensor(value):
            tensor = value.detach().cpu().float()
        else:
            tensor = torch.as_tensor(value, dtype=torch.float32)
        normalized[str(name)] = tensor

    if "base" not in normalized:
        return None

    cache.set(cache_key, normalized)
    return normalized


def _build_multiview_batches(batch, multiview_dir: Path) -> list | None:
    """Construct per-view batch clones with replaced edges from multiview files.

    Returns:
        List of batches in fixed order (base first) or None when not available.
    """
    num_graphs = int(batch.batch.max().item()) + 1
    subject_ids = _extract_batch_subject_ids(batch, num_graphs)
    if subject_ids is None:
        return None

    graph_node_ids: list[torch.Tensor] = []
    for g_idx in range(num_graphs):
        node_ids = (batch.batch == g_idx).nonzero(as_tuple=False).view(-1)
        if node_ids.numel() == 0:
            return None
        graph_node_ids.append(node_ids)

    views_by_graph: list[dict[str, torch.Tensor]] = []
    for sub_id in subject_ids:
        package_path = multiview_dir / str(sub_id) / "multiview_graphs.pt"
        if not package_path.exists():
            return None

        package = _load_multiview_package(package_path)
        if package is None:
            return None
        views_by_graph.append(package)

    view_batches = []
    device = batch.x.device

    for view_name in _MULTIVIEW_VIEW_ORDER:
        edge_index_parts: list[torch.Tensor] = []
        edge_attr_parts: list[torch.Tensor] = []

        for g_idx in range(num_graphs):
            n_nodes = int(graph_node_ids[g_idx].numel())
            graph_views = views_by_graph[g_idx]

            adj = graph_views.get(view_name)
            if adj is None:
                adj = graph_views["base"]

            adj = adj.to(device=device, dtype=torch.float32)
            if tuple(adj.shape) != (n_nodes, n_nodes):
                base_adj = graph_views["base"].to(device=device, dtype=torch.float32)
                if tuple(base_adj.shape) != (n_nodes, n_nodes):
                    return None
                adj = base_adj

            adj = adj.clone()
            adj.fill_diagonal_(0.0)
            local_edges = (adj != 0).nonzero(as_tuple=False).t().contiguous()

            if local_edges.numel() == 0:
                base_adj = graph_views["base"].to(device=device, dtype=torch.float32).clone()
                base_adj.fill_diagonal_(0.0)
                local_edges = (base_adj != 0).nonzero(as_tuple=False).t().contiguous()
                adj = base_adj

            if local_edges.numel() == 0:
                return None

            node_ids = graph_node_ids[g_idx]
            global_edges = node_ids[local_edges]
            edge_weights = adj[local_edges[0], local_edges[1]].unsqueeze(1)

            edge_index_parts.append(global_edges)
            edge_attr_parts.append(edge_weights)

        view_batch = batch.clone()
        view_batch.edge_index = torch.cat(edge_index_parts, dim=1)
        view_batch.edge_attr = torch.cat(edge_attr_parts, dim=0)
        view_batches.append(view_batch)

    return view_batches


# ── EPOCH-LEVEL TRAINING ─────────────────────────────────────────────────────────

def train_one_epoch_with_accumulation(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0,
    site_loss_weight: float = 0.0,
    use_grl: bool = False,
    # Task 1 additions (DD-009) — default 0.0 = backward compatible
    structural_dropout_prob: float = 0.0,
    edge_contrastive_weight: float = 0.0,
    # Task 2 additions (DD-010)
    invariance_loss_fn: nn.Module | None = None,
    invariance_weight: float = 0.0,
    multiview_dir: Path | None = None,
    # Task 4 additions (DD-012)
    spatial_invariance_loss_fn: nn.Module | None = None,
    spatial_invariance_weight: float = 0.0,
) -> tuple[float, float]:
    """
    Train for one epoch with gradient accumulation.

    When structural_dropout_prob > 0 and edge_contrastive_weight > 0,
    runs a second forward pass on the edge-only version of each batch and
    adds EdgeStructureContrastiveLoss to the focal loss:
        total = focal + edge_contrastive_weight * contrastive

    Args:
        model: Model to train
        loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function (FocalLoss)
        device: Device to run on
        gradient_accumulation_steps: Accumulate gradients over N batches
        max_grad_norm: Gradient clipping threshold
        site_loss_weight: Weight for adversarial site loss (GRL mode)
        use_grl: Whether to use gradient reversal layer
        structural_dropout_prob: Fraction of graphs to zero node features for
        edge_contrastive_weight: Weight for EdgeStructureContrastiveLoss
        invariance_loss_fn: Optional CausalInvarianceLoss module
        invariance_weight: Weight for multi-view invariance loss
        multiview_dir: Directory containing per-subject multi-view graphs
        spatial_invariance_loss_fn: Optional spatial-channel adversarial loss module
        spatial_invariance_weight: Weight for spatial invariance loss

    Returns:
        Average total loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()
    epoch_grad_norm = 0.0

    # Instantiate contrastive loss once per epoch (reuse temperature)
    contrastive_fn = EdgeStructureContrastiveLoss(temperature=0.5) if edge_contrastive_weight > 0 else None

    for i, data in enumerate(loader):
        if data is None:
            continue

        x_raw = getattr(data, 'x', None)
        edge_index_raw = getattr(data, 'edge_index', None)
        batch_raw = getattr(data, 'batch', None)
        if x_raw is None or edge_index_raw is None or batch_raw is None:
            logger.warning("Batch %d missing x/edge_index/batch; skipping batch", i)
            continue

        num_nodes = int(x_raw.size(0)) if x_raw.dim() >= 1 else 0
        if num_nodes <= 0:
            logger.warning("Batch %d has zero nodes; skipping batch", i)
            continue

        b_cpu = batch_raw.view(-1).detach().to(device='cpu').long()
        if b_cpu.numel() != num_nodes:
            logger.error(
                "Skipping batch %d due to batch vector mismatch: len(batch)=%d vs num_nodes=%d",
                i,
                int(b_cpu.numel()),
                num_nodes,
            )
            continue
        if bool((b_cpu < 0).any()):
            logger.error("Skipping batch %d due to negative graph indices in batch vector", i)
            continue

        ei_cpu = edge_index_raw.detach().to(device='cpu').long()
        if ei_cpu.dim() != 2 or int(ei_cpu.size(0)) != 2:
            logger.error(
                "Skipping batch %d due to malformed edge_index shape %s",
                i,
                tuple(ei_cpu.shape),
            )
            continue
        if ei_cpu.numel() > 0:
            ei_min = int(ei_cpu.min().item())
            ei_max = int(ei_cpu.max().item())
            if ei_min < 0 or ei_max >= num_nodes:
                logger.error(
                    "Skipping batch %d due to out-of-range edge_index values [min=%d, max=%d] for num_nodes=%d",
                    i,
                    ei_min,
                    ei_max,
                    num_nodes,
                )
                continue

        # Validate/sanitize targets on CPU before any CUDA kernels run.
        y_raw = getattr(data, 'y', None)
        if y_raw is None:
            logger.warning("Batch %d missing labels; skipping batch", i)
            continue

        y_cpu = y_raw.view(-1).detach().to(device='cpu').long()
        num_classes = 2
        try:
            if hasattr(model, 'classifier') and len(model.classifier) > 0:
                num_classes = int(model.classifier[-1].out_features)
        except Exception:
            num_classes = 2

        # Common legacy encoding fix: DX_GROUP style {1,2} -> {0,1}.
        uniq_y = torch.unique(y_cpu)
        if num_classes == 2 and bool(torch.all((uniq_y == 1) | (uniq_y == 2))):
            y_cpu = y_cpu - 1
            logger.warning("Batch %d labels remapped from {1,2} to {0,1}", i)

        invalid_y = (y_cpu < 0) | (y_cpu >= num_classes)
        if bool(invalid_y.any()):
            bad_vals = sorted({int(v) for v in y_cpu[invalid_y].tolist()})
            logger.error(
                "Skipping batch %d due to invalid class labels %s for num_classes=%d",
                i,
                bad_vals,
                num_classes,
            )
            continue
        data.y = y_cpu

        # Site IDs can also trigger CUDA asserts (embedding/cross-entropy indices).
        site_raw = getattr(data, 'site_id', None)
        if site_raw is not None and hasattr(model, 'site_embedding'):
            site_cpu = site_raw.view(-1).detach().to(device='cpu').long()
            num_sites = int(model.site_embedding.num_embeddings)
            invalid_site = (site_cpu < 0) | (site_cpu >= num_sites)
            if bool(invalid_site.any()):
                bad_sites = sorted({int(v) for v in site_cpu[invalid_site].tolist()})
                logger.warning(
                    "Batch %d had invalid site_id values %s; remapping to 0 (valid range [0, %d])",
                    i,
                    bad_sites,
                    num_sites - 1,
                )
                site_cpu[invalid_site] = 0
            data.site_id = site_cpu

        data = data.to(device)

        multiview_batches = None
        if (
            invariance_loss_fn is not None
            and invariance_weight > 0.0
            and multiview_dir is not None
            and hasattr(model, "forward_multiview")
        ):
            multiview_batches = _build_multiview_batches(data, Path(multiview_dir))

        emb_full = None

        # ── Forward pass ────────────────────────────────────────────────────
        if use_grl:
            out, site_logits = model(
                data.x, data.edge_index, data.edge_attr, data.batch,
                getattr(data, 'site_id', None),
                getattr(data, 'age', None),
                getattr(data, 'sex', None),
                getattr(data, 'fiq', None),
                return_site_logits=True,
            )
            class_loss = criterion(out, data.y)
            site_targets = getattr(data, 'site_id', None)
            if site_targets is None:
                loss = class_loss
            else:
                site_targets = site_targets.view(-1).long()
                num_site_classes = int(site_logits.size(1))
                valid_site_mask = (site_targets >= 0) & (site_targets < num_site_classes)

                if not bool(valid_site_mask.all()):
                    invalid_vals = torch.unique(site_targets[~valid_site_mask]).detach().cpu().tolist()
                    logger.warning(
                        "Skipping invalid site_id values for GRL site loss: %s (valid range [0, %d])",
                        [int(v) for v in invalid_vals],
                        num_site_classes - 1,
                    )

                import torch.nn.functional as _F
                if bool(valid_site_mask.any()):
                    site_loss = _F.cross_entropy(site_logits[valid_site_mask], site_targets[valid_site_mask])
                    loss = class_loss + site_loss_weight * site_loss
                else:
                    loss = class_loss

        else:
            if multiview_batches is not None:
                assert invariance_loss_fn is not None, "invariance_loss_fn required when multiview_batches is not None"
                logits_full, multiview_embeddings = model.forward_multiview(multiview_batches)
                emb_full = multiview_embeddings[0]
                focal = criterion(logits_full, data.y)
                invariance = invariance_loss_fn(multiview_embeddings)
                loss = focal + invariance_weight * invariance
            elif hasattr(model, "_forward_with_embedding"):
                logits_full, emb_full = model._forward_with_embedding(
                    data.x, data.edge_index, data.edge_attr, data.batch,
                    site_id=getattr(data, 'site_id', None),
                    age=getattr(data, 'age', None),
                    sex=getattr(data, 'sex', None),
                    fiq=getattr(data, 'fiq', None),
                )
                loss = criterion(logits_full, data.y)
            else:
                out = model.forward_batch(data) if hasattr(model, "forward_batch") else model(
                    data.x, data.edge_index, data.edge_attr, data.batch,
                    getattr(data, 'site_id', None),
                    getattr(data, 'age', None),
                    getattr(data, 'sex', None),
                    getattr(data, 'fiq', None),
                )
                loss = criterion(out, data.y)

            # Task 1: structural dual-view contrastive regularization.
            if (
                emb_full is not None
                and structural_dropout_prob > 0.0
                and edge_contrastive_weight > 0.0
                and contrastive_fn is not None
                and hasattr(model, '_forward_with_embedding')
            ):
                data_edge = _apply_structural_dropout(data, structural_dropout_prob, training=True)
                _, emb_edge = model._forward_with_embedding(
                    data_edge.x, data_edge.edge_index, data_edge.edge_attr, data_edge.batch,
                    site_id=getattr(data_edge, 'site_id', None),
                    age=getattr(data_edge, 'age', None),
                    sex=getattr(data_edge, 'sex', None),
                    fiq=getattr(data_edge, 'fiq', None),
                )
                contrastive = contrastive_fn(emb_full, emb_edge)
                loss = loss + edge_contrastive_weight * contrastive

        # Task 4: spatial-channel adversarial regularization.
        if spatial_invariance_loss_fn is not None and spatial_invariance_weight > 0.0:
            node_site_targets = _expand_site_targets_to_nodes(data)
            if node_site_targets is not None:
                spatial_loss = spatial_invariance_loss_fn(data.x, node_site_targets)
                loss = loss + spatial_invariance_weight * spatial_loss

        # ── Gradient accumulation ────────────────────────────────────────────
        loss = loss / gradient_accumulation_steps
        loss.backward()

        if (i + 1) % gradient_accumulation_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            epoch_grad_norm = max(epoch_grad_norm, (grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)))

        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1

    # Final update if not divisible
    if num_batches % gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / max(num_batches, 1), epoch_grad_norm




@torch.no_grad()
def _evaluate_model(model: torch.nn.Module, loader: torch.utils.data.DataLoader,
                    device: torch.device, threshold: float = 0.5) -> dict[str, Any]:
    return evaluate_loader(model, loader, device, threshold=threshold)


def train_fold_with_onecycle(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    epochs: int,
    max_lr: float,
    patience: int,
    use_grl: bool,
    grl_weight: float,
    fold: int,
    min_epochs_before_stopping: int = 0,
    weight_decay: float = 0.0,
    gradient_accumulation_steps: int = 2,
    pct_start: float = GNN_ONECYCLE_PCT_START,
    grl_alpha_max: float = GNN_GRL_ALPHA_MAX,
    # Task 1 additions (DD-009)
    structural_dropout_prob: float = 0.0,
    edge_contrastive_weight: float = 0.0,
    # Task 2 additions (DD-010)
    invariance_loss_fn: nn.Module | None = None,
    invariance_weight: float = 0.0,
    multiview_dir: Path | None = None,
    # Task 4 additions (DD-012)
    spatial_invariance_loss_fn: nn.Module | None = None,
    spatial_invariance_weight: float = 0.0,
) -> tuple:
    assert pct_start * epochs < patience, (
        f"Warmup ({pct_start * epochs:.0f} epochs) >= patience ({patience}): "
        "adjust GNN_ONECYCLE_PCT_START / GNN_ONECYCLE_WARMUP_FRACTION"
    )

    model.to(device)

    optim_params = list(model.parameters())
    if spatial_invariance_loss_fn is not None:
        spatial_invariance_loss_fn = spatial_invariance_loss_fn.to(device)
        optim_params.extend(list(spatial_invariance_loss_fn.parameters()))

    optimizer = torch.optim.AdamW(
        optim_params,
        lr=max_lr / 25.0,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=epochs,
        pct_start=pct_start,
        anneal_strategy='cos',
    )

    early_stopping = EarlyStopping(patience=patience, min_delta=0.0001, mode='max')
    history = []
    best_state = None
    best_metrics = {
        'auc': 0.0, 'auprc': 0.0, 'f1': 0.0, 'acc': 0.0,
        'threshold': 0.5, 'best_epoch': 0,
    }

    for epoch in range(1, epochs + 1):
        # Anneal GRL alpha (Ganin et al. 2016 schedule)
        if use_grl and hasattr(model, 'set_grl_alpha'):
            progress = (epoch - 1) / max(epochs - 1, 1)
            model.set_grl_alpha(progress, alpha_max=grl_alpha_max)

        loss, grad_norm = train_one_epoch_with_accumulation(
            model, train_loader, optimizer, criterion, device,
            gradient_accumulation_steps=gradient_accumulation_steps,
            site_loss_weight=grl_weight,
            use_grl=use_grl,
            structural_dropout_prob=structural_dropout_prob,
            edge_contrastive_weight=edge_contrastive_weight,
            invariance_loss_fn=invariance_loss_fn,
            invariance_weight=invariance_weight,
            multiview_dir=multiview_dir,
            spatial_invariance_loss_fn=spatial_invariance_loss_fn,
            spatial_invariance_weight=spatial_invariance_weight,
        )
        scheduler.step()

        metrics = _evaluate_model(model, val_loader, device, threshold=0.5)
        val_loss = metrics.get("loss", None)
        opt_threshold, _ = resolve_threshold(metrics["probs"], metrics["labels"], EVAL_THRESHOLD_POLICY, EVAL_FIXED_THRESHOLD)
        metrics_opt = _evaluate_model(model, val_loader, device, threshold=opt_threshold)

        epoch_metrics = {
            'epoch': epoch,
            'train_loss': loss,
            'val_loss': val_loss if val_loss is not None else loss,
            'grad_norm': grad_norm,
            'auc': metrics['auc'],
            'auprc': metrics['auprc'],
            'f1': metrics_opt['f1'],
            'acc': metrics_opt['acc'],
            'threshold': opt_threshold,
            'cm': metrics_opt['cm'],
            'lr': optimizer.param_groups[0]['lr'],
        }
        history.append(epoch_metrics)

        if metrics['auc'] > best_metrics['auc']:
            best_metrics = {
                'auc': metrics['auc'],
                'auprc': metrics['auprc'],
                'f1': metrics_opt['f1'],
                'acc': metrics_opt['acc'],
                'threshold': opt_threshold,
                'best_epoch': epoch,
            }
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        should_stop = early_stopping(metrics['auc'])
        if epoch >= int(max(min_epochs_before_stopping, 0)) and should_stop:
            logger.info(
                "Fold %s: early stopping at epoch %s (min_epochs=%s, patience=%s)",
                fold,
                epoch,
                int(max(min_epochs_before_stopping, 0)),
                int(patience),
            )
            break

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    return best_state, best_metrics, history
