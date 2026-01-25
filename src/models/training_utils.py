"""
Training utilities for GNN model.

Provides reusable components for training loops:
- EarlyStopping: Monitor validation metrics with patience
- WarmupScheduler: Linear learning rate warmup
- TrainingTracker: Track metrics across epochs/folds
- CheckpointManager: Save/load model checkpoints

These utilities reduce code duplication and improve maintainability
while keeping PyTorch raw (no pytorch-lightning dependency).
"""

import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


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
    Useful for stabilizing training at the start.
    
    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of epochs for warmup
        base_lr: Target learning rate after warmup
    
    Example:
        warmup = WarmupScheduler(optimizer, warmup_epochs=5, base_lr=0.001)
        for epoch in range(epochs):
            if epoch < 5:
                warmup.step()
            train(model, loader)
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
            # ... train model ...
            tracker.add_fold_result(
                fold=fold, auc=0.85, f1=0.80, acc=0.78, 
                threshold=0.55, best_epoch=45
            )
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
            fold=fold,
            auc=auc,
            f1=f1,
            acc=acc,
            threshold=threshold,
            best_epoch=best_epoch,
            train_time=train_time,
            val_probs=val_probs,
            val_labels=val_labels
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
            'per_fold_epochs': epochs
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
    
    Simplifies saving/loading best models with metadata.
    
    Args:
        checkpoint_dir: Directory to save checkpoints
        monitor: Metric to monitor ('auc', 'f1', 'loss')
        mode: 'max' (higher is better) or 'min' (lower is better)
    
    Example:
        ckpt = CheckpointManager(checkpoint_dir, monitor='auc', mode='max')
        for epoch in range(epochs):
            val_auc = validate(model, val_loader)
            if ckpt.should_save(val_auc):
                ckpt.save(model, optimizer, epoch, {'auc': val_auc, 'threshold': 0.5})
    """
    def __init__(self, checkpoint_dir: Path, monitor: str = 'auc', mode: str = 'max'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.best_score = float('-inf') if mode == 'max' else float('inf')
    
    def should_save(self, score: float) -> bool:
        """Check if current score is better than best."""
        if self.mode == 'max':
            return score > self.best_score
        else:
            return score < self.best_score
    
    def save(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
             epoch: int, metrics: Dict[str, Any], fold: Optional[int] = None):
        """
        Save model checkpoint with metadata.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch
            metrics: Dictionary of metrics (e.g., {'auc': 0.85, 'f1': 0.80})
            fold: Optional fold number for k-fold CV
        """
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
                'optimizer_state': optimizer.state_dict(),
                'epoch': epoch,
                **metrics
            }
            
            torch.save(checkpoint, filepath)
            logger.info(f"✓ Saved checkpoint: {filename} (epoch {epoch}, {self.monitor}={score:.4f})")
    
    def load(self, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None,
             fold: Optional[int] = None) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            model: Model to load weights into
            optimizer: Optional optimizer to load state
            fold: Optional fold number for k-fold CV
        
        Returns:
            Dictionary with checkpoint metadata
        """
        filename = f"best_model_fold{fold}.pt" if fold is not None else "best_model.pt"
        filepath = self.checkpoint_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        checkpoint = torch.load(filepath, weights_only=False)
        model.load_state_dict(checkpoint['model_state'])
        
        if optimizer is not None and 'optimizer_state' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        logger.info(f"Loaded checkpoint: {filename} (epoch {checkpoint.get('epoch', 'unknown')})")
        return checkpoint
    
    def reset(self):
        """Reset best score for new training run."""
        self.best_score = float('-inf') if self.mode == 'max' else float('inf')


def train_one_epoch_with_accumulation(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float = 1.0
) -> float:
    """
    Train for one epoch with gradient accumulation.
    
    Args:
        model: Model to train
        loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to run on
        gradient_accumulation_steps: Accumulate gradients over N batches
        max_grad_norm: Gradient clipping threshold
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0
    optimizer.zero_grad()
    
    for i, data in enumerate(loader):
        if data is None:
            continue
        
        data = data.to(device)
        
        # Forward pass
        out = model(
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
            getattr(data, 'site_id', None),
            getattr(data, 'age', None),
            getattr(data, 'sex', None),
            getattr(data, 'fiq', None)
        )
        loss = criterion(out, data.y)
        
        # Normalize loss for gradient accumulation
        loss = loss / gradient_accumulation_steps
        loss.backward()
        
        # Update weights every N steps
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
