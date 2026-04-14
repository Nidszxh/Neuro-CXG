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
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from torch_geometric.loader import DataLoader
from src.core.config import GNN_ONECYCLE_PCT_START, GNN_GRL_ALPHA_MAX
from src.models.evaluation import evaluate_loader, optimal_threshold

logger = logging.getLogger(__name__)


def make_loader(
    dataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 4,
) -> DataLoader:
    """Create a tuned torch_geometric DataLoader for small-graph workloads."""
    effective_workers = num_workers
    if len(dataset) < 800:
        effective_workers = min(num_workers, 2)

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
        self.best_score = None
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
    val_probs: Optional[np.ndarray] = None
    val_labels: Optional[np.ndarray] = None


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
        self.fold_results: List[FoldMetrics] = []

    def add_fold_result(self, fold: int, auc: float, f1: float, acc: float,
                        threshold: float, best_epoch: int, train_time: float = 0.0,
                        val_probs: Optional[np.ndarray] = None,
                        val_labels: Optional[np.ndarray] = None):
        """Add results from a completed fold."""
        self.fold_results.append(FoldMetrics(
            fold=fold, auc=auc, f1=f1, acc=acc, threshold=threshold,
            best_epoch=best_epoch, train_time=train_time,
            val_probs=val_probs, val_labels=val_labels,
        ))

    def get_summary(self) -> Dict[str, Any]:
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

    def get_ensemble_predictions(self) -> Optional[tuple]:
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

    def save(self, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer],
             epoch: int, metrics: Dict[str, Any], fold: Optional[int] = None):
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

    def load(self, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None,
             fold: Optional[int] = None, allow_partial: bool = False) -> Dict[str, Any]:
        """Load model checkpoint."""
        filename = f"best_model_fold{fold}.pt" if fold is not None else "best_model.pt"
        filepath = self.checkpoint_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        checkpoint = torch.load(filepath, weights_only=False)

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
) -> float:
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

    Returns:
        Average total loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    # Instantiate contrastive loss once per epoch (reuse temperature)
    contrastive_fn = EdgeStructureContrastiveLoss(temperature=0.5) if edge_contrastive_weight > 0 else None

    for i, data in enumerate(loader):
        if data is None:
            continue

        data = data.to(device)

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
                site_targets = site_targets.view(-1)
                import torch.nn.functional as _F
                site_loss = _F.cross_entropy(site_logits, site_targets)
                loss = class_loss + site_loss_weight * site_loss

        elif structural_dropout_prob > 0.0 and edge_contrastive_weight > 0.0 and hasattr(model, '_forward_with_embedding'):
            # Task 1: dual-view forward for structural learning
            # View 1: full features
            logits_full, emb_full = model._forward_with_embedding(
                data.x, data.edge_index, data.edge_attr, data.batch,
                site_id=getattr(data, 'site_id', None),
                age=getattr(data, 'age', None),
                sex=getattr(data, 'sex', None),
                fiq=getattr(data, 'fiq', None),
            )
            focal = criterion(logits_full, data.y)

            # View 2: edge-only (node features zeroed for dropout_prob fraction)
            data_edge = _apply_structural_dropout(data, structural_dropout_prob, training=True)
            _, emb_edge = model._forward_with_embedding(
                data_edge.x, data_edge.edge_index, data_edge.edge_attr, data_edge.batch,
                site_id=getattr(data_edge, 'site_id', None),
                age=getattr(data_edge, 'age', None),
                sex=getattr(data_edge, 'sex', None),
                fiq=getattr(data_edge, 'fiq', None),
            )
            contrastive = contrastive_fn(emb_full, emb_edge)
            loss = focal + edge_contrastive_weight * contrastive

        else:
            out = model.forward_batch(data) if hasattr(model, "forward_batch") else model(
                data.x, data.edge_index, data.edge_attr, data.batch,
                getattr(data, 'site_id', None),
                getattr(data, 'age', None),
                getattr(data, 'sex', None),
                getattr(data, 'fiq', None),
            )
            loss = criterion(out, data.y)

        # ── Gradient accumulation ────────────────────────────────────────────
        loss = loss / gradient_accumulation_steps
        loss.backward()

        if (i + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * gradient_accumulation_steps
        num_batches += 1

    # Final update if not divisible
    if num_batches % gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / max(num_batches, 1)


def _find_optimal_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> tuple:
    return optimal_threshold(y_probs, y_true)


@torch.no_grad()
def _evaluate_model(model: torch.nn.Module, loader: torch.utils.data.DataLoader,
                    device: torch.device, threshold: float = 0.5) -> Dict[str, Any]:
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
    weight_decay: float = 0.0,
    gradient_accumulation_steps: int = 2,
    pct_start: float = GNN_ONECYCLE_PCT_START,
    grl_alpha_max: float = GNN_GRL_ALPHA_MAX,
    # Task 1 additions (DD-009)
    structural_dropout_prob: float = 0.0,
    edge_contrastive_weight: float = 0.0,
) -> tuple:
    assert pct_start < (patience / max(epochs, 1)), (
        f"Warmup ({pct_start * epochs:.0f} epochs) >= patience ({patience}): "
        "adjust GNN_ONECYCLE_PCT_START / GNN_ONECYCLE_WARMUP_FRACTION"
    )

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
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

        loss = train_one_epoch_with_accumulation(
            model, train_loader, optimizer, criterion, device,
            gradient_accumulation_steps=gradient_accumulation_steps,
            site_loss_weight=grl_weight,
            use_grl=use_grl,
            structural_dropout_prob=structural_dropout_prob,
            edge_contrastive_weight=edge_contrastive_weight,
        )
        scheduler.step()

        metrics = _evaluate_model(model, val_loader, device, threshold=0.5)
        opt_threshold, _ = _find_optimal_threshold(metrics['labels'], metrics['probs'])
        metrics_opt = _evaluate_model(model, val_loader, device, threshold=opt_threshold)

        epoch_metrics = {
            'epoch': epoch,
            'loss': loss,
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

        if early_stopping(metrics['auc']):
            logger.info(f"Fold {fold}: early stopping at epoch {epoch}")
            break

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    return best_state, best_metrics, history
