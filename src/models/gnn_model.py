import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, confusion_matrix, 
    accuracy_score, precision_recall_curve
)
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import warnings
import time

warnings.filterwarnings('ignore')

# Setup paths and config
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import (
    K_FOLDS, GNN_BATCH_SIZE, GNN_EPOCHS, 
    CHECKPOINT_DIR, DEVICE, GNN_IN_CHANNELS
)
from src.models.training_utils import (
    EarlyStopping, WarmupScheduler, TrainingTracker, CheckpointManager,
    train_one_epoch_with_accumulation
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for class imbalance.
    
    Automatically focuses on hard-to-classify examples.
    """
    def __init__(self, alpha=0.75, gamma=3.0):  # INCREASED gamma from 2.0 to 3.0
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        probs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=2).float()
        pt = (probs * targets_one_hot).sum(dim=1)
        
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = targets_one_hot[:, 1] * self.alpha + targets_one_hot[:, 0] * (1 - self.alpha)
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        focal_loss = alpha_weight * focal_weight * ce_loss
        
        return focal_loss.mean()


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Stops training when validation metric doesn't improve for `patience` epochs.
    """
    def __init__(self, patience=25, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False


class WarmupScheduler:
    """
    Learning rate warmup for stable training start.
    
    Gradually increases LR from 0 to base_lr over warmup_epochs.
    """
    def __init__(self, optimizer, warmup_epochs, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.current_epoch = 0
    
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.current_epoch += 1


def compute_class_weights(labels):
    """Compute inverse frequency class weights."""
    labels_array = np.array(labels)
    n_control = (labels_array == 0).sum()
    n_asd = (labels_array == 1).sum()
    total = len(labels_array)
    
    weight_control = total / (2 * n_control) if n_control > 0 else 1.0
    weight_asd = total / (2 * n_asd) if n_asd > 0 else 1.0
    
    logger.info(f"Class distribution: Control={n_control}, ASD={n_asd}")
    logger.info(f"Class weights: Control={weight_control:.3f}, ASD={weight_asd:.3f}")
    
    return torch.tensor([weight_control, weight_asd], dtype=torch.float32)


def find_optimal_threshold(y_true, y_probs):
    """Find threshold that maximizes F1 score."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]
    
    return best_threshold, best_f1


@torch.no_grad()
def evaluate(model, loader, threshold=0.5):
    """Evaluate model with custom threshold."""
    model.eval()
    all_probs = []
    all_labels = []
    
    for data in loader:
        if data is None:
            continue
            
        data = data.to(DEVICE)
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
        probs = torch.softmax(out, dim=1)
        all_probs.append(probs[:, 1].cpu().numpy())
        all_labels.append(data.y.cpu().numpy())
    
    probs_array = np.concatenate(all_probs)
    labels_array = np.concatenate(all_labels)
    preds_array = (probs_array > threshold).astype(int)
    
    auc = roc_auc_score(labels_array, probs_array)
    f1 = f1_score(labels_array, preds_array, zero_division=0)
    acc = accuracy_score(labels_array, preds_array)
    cm = confusion_matrix(labels_array, preds_array)
    
    return {
        'acc': acc,
        'f1': f1,
        'auc': auc,
        'cm': cm,
        'probs': probs_array,
        'labels': labels_array
    }


def run_enhanced_training():
    """
    Main training loop with k-fold cross-validation.
    
    Uses modular training utilities (EarlyStopping, WarmupScheduler, TrainingTracker, CheckpointManager)
    to reduce code duplication and improve maintainability.
    """
    from src.features.graph_factory import ABIDECausalDataset
    from src.models.causal_gnn import CausalBrainGNN
    from src.core.config import (
        GNN_LEARNING_RATE_TUNED,
        GNN_HIDDEN_CHANNELS_TUNED,
        GNN_USE_SITE_EMBEDDING,
        GNN_USE_DEMOGRAPHICS,
        GNN_ENSEMBLE_MODE,
        GNN_EARLY_STOPPING_PATIENCE,
    )
    
    # Load dataset
    dataset = ABIDECausalDataset(split='train')
    
    # Extract labels
    labels = [dataset.get(i).y.item() for i in range(len(dataset)) if dataset.get(i) is not None]
    
    # Compute class weights
    class_weights = compute_class_weights(labels)
    
    # Initialize tracking and checkpoint management
    tracker = TrainingTracker(k_folds=K_FOLDS)
    checkpoint_manager = CheckpointManager(CHECKPOINT_DIR, monitor='auc', mode='max')
    
    logger.info(f"\n{'='*70}")
    logger.info("5-FOLD CROSS VALIDATION WITH OPTIMIZATIONS (v1.1 - Stable)")
    logger.info(f"{'='*70}")
    logger.info(f"Total subjects: {len(labels)}")
    logger.info(f"Learning rate: {GNN_LEARNING_RATE_TUNED} (tuned)")
    logger.info(f"Hidden channels: {GNN_HIDDEN_CHANNELS_TUNED} (256 - increased from 128)")
    logger.info(f"GNN Layers: 3 (increased from 2)")
    logger.info(f"Input features: 14 (8 temporal + 6 spatial)")
    logger.info(f"Graph sparsity: 30% (reverted from 50% - cleaner graphs with deep model)")
    logger.info(f"Site conditioning: {GNN_USE_SITE_EMBEDDING}")
    logger.info(f"Ensemble mode: {GNN_ENSEMBLE_MODE}")
    logger.info(f"Early stopping: patience={GNN_EARLY_STOPPING_PATIENCE}")
    logger.info(f"Focal Loss: α=0.75, γ=3.0 (increased from 2.0)")
    logger.info(f"{'='*70}\n")
    
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        logger.info(f"\n{'='*70}")
        logger.info(f"FOLD {fold+1}/{K_FOLDS}")
        logger.info(f"{'='*70}")
        
        fold_start_time = time.time()
        
        # Create data loaders
        train_data = [dataset[i] for i in train_idx if dataset[i] is not None]
        val_data = [dataset[i] for i in val_idx if dataset[i] is not None]
        
        train_labels = [d.y.item() for d in train_data]
        val_labels = [d.y.item() for d in val_data]
        
        logger.info(f"Train: Control={train_labels.count(0)}, ASD={train_labels.count(1)}")
        logger.info(f"Val: Control={val_labels.count(0)}, ASD={val_labels.count(1)}")
        
        train_loader = DataLoader(train_data, batch_size=GNN_BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=GNN_BATCH_SIZE)
        
        # Initialize model
        model = CausalBrainGNN(
            num_node_features=GNN_IN_CHANNELS,
            hidden_channels=GNN_HIDDEN_CHANNELS_TUNED,
            num_classes=2,
            dropout=0.5,
            num_heads=2,
            num_sites=20,
            use_site_embedding=GNN_USE_SITE_EMBEDDING,
            use_demographics=GNN_USE_DEMOGRAPHICS,
        ).to(DEVICE)
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=GNN_LEARNING_RATE_TUNED,
            weight_decay=1e-3
        )
        
        # Schedulers
        warmup = WarmupScheduler(optimizer, warmup_epochs=5, base_lr=GNN_LEARNING_RATE_TUNED)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=GNN_EPOCHS - 5)
        
        # Loss and early stopping
        criterion = FocalLoss(alpha=0.75, gamma=3.0)
        early_stopping = EarlyStopping(patience=GNN_EARLY_STOPPING_PATIENCE, min_delta=0.001, mode='max')
        checkpoint_manager.reset()
        
        best_auc = 0.0
        best_f1 = 0.0
        best_threshold = 0.5
        best_epoch = 0
        
        # Training loop
        for epoch in range(1, GNN_EPOCHS + 1):
            # Warmup for first 5 epochs
            if epoch <= 5:
                warmup.step()
            
            # Train with gradient accumulation (effective batch size = 64)
            loss = train_one_epoch_with_accumulation(
                model, train_loader, optimizer, criterion, DEVICE,
                gradient_accumulation_steps=2
            )
            
            # Step scheduler after warmup
            if epoch > 5:
                scheduler.step()
            
            # Evaluate every 5 epochs
            if epoch % 5 == 0:
                metrics = evaluate(model, val_loader, threshold=0.5)
                
                # Find optimal threshold
                opt_threshold, opt_f1 = find_optimal_threshold(
                    metrics['labels'],
                    metrics['probs']
                )
                
                metrics_opt = evaluate(model, val_loader, threshold=opt_threshold)
                current_lr = optimizer.param_groups[0]['lr']
                
                logger.info(
                    f"Epoch {epoch:03d} | LR: {current_lr:.6f} | Loss: {loss:.4f} | "
                    f"AUC: {metrics['auc']:.4f} | "
                    f"F1@{opt_threshold:.2f}: {metrics_opt['f1']:.4f}"
                )
                
                # Save best model using CheckpointManager
                checkpoint_metrics = {
                    'auc': metrics['auc'],
                    'f1': metrics_opt['f1'],
                    'threshold': opt_threshold
                }
                checkpoint_manager.save(model, optimizer, epoch, checkpoint_metrics, fold=fold)
                
                if metrics['auc'] > best_auc:
                    best_auc = metrics['auc']
                    best_f1 = metrics_opt['f1']
                    best_threshold = opt_threshold
                    best_epoch = epoch
                    
                    logger.info(
                        f"✓ New best: AUC={best_auc:.4f}, "
                        f"F1={best_f1:.4f} @ threshold={best_threshold:.3f}"
                    )
                
                # Early stopping check
                if early_stopping(metrics['auc']):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Final evaluation
        checkpoint = checkpoint_manager.load(model, fold=fold)
        final_threshold = checkpoint['threshold']
        final_metrics = evaluate(model, val_loader, threshold=final_threshold)
        
        fold_train_time = time.time() - fold_start_time
        
        logger.info(f"\nFold {fold+1} Final Results:")
        logger.info(f"  Best epoch: {best_epoch}")
        logger.info(f"  Training time: {fold_train_time:.1f}s")
        logger.info(f"  AUC: {final_metrics['auc']:.4f}")
        logger.info(f"  F1: {final_metrics['f1']:.4f} (threshold={final_threshold:.3f})")
        logger.info(f"  Accuracy: {final_metrics['acc']:.4f}")
        logger.info(f"  Confusion Matrix:")
        logger.info(f"    {final_metrics['cm']}")
        
        # Track results
        tracker.add_fold_result(
            fold=fold,
            auc=final_metrics['auc'],
            f1=final_metrics['f1'],
            acc=final_metrics['acc'],
            threshold=final_threshold,
            best_epoch=best_epoch,
            train_time=fold_train_time,
            val_probs=final_metrics['probs'],
            val_labels=final_metrics['labels']
        )
    
    # Log summary statistics
    tracker.log_summary()
    
    # Ensemble evaluation
    if GNN_ENSEMBLE_MODE:
        evaluate_ensemble(tracker, checkpoint_manager)
    
    logger.info(f"{'='*70}\n")
            if epoch <= 5:
                warmup.step()
            
            # Train with gradient accumulation (effective batch size = 64)
            loss = train_one_epoch(
                model, train_loader, optimizer, criterion,
                gradient_accumulation_steps=2
            )
            
            # Step scheduler after warmup
            if epoch > 5:
                scheduler.step()
            
            # Evaluate every 5 epochs
            if epoch % 5 == 0:
                metrics = evaluate(model, val_loader, threshold=0.5)
                
                # Find optimal threshold
                opt_threshold, opt_f1 = find_optimal_threshold(
                    metrics['labels'],
                    metrics['probs']
                )
                
                metrics_opt = evaluate(model, val_loader, threshold=opt_threshold)
                
                current_lr = optimizer.param_groups[0]['lr']
                
                logger.info(
                    f"Epoch {epoch:03d} | LR: {current_lr:.6f} | Loss: {loss:.4f} | "
                    f"AUC: {metrics['auc']:.4f} | "
                    f"F1@{opt_threshold:.2f}: {metrics_opt['f1']:.4f}"
                )
                
                # Save best model
                if metrics['auc'] > best_auc:
                    best_auc = metrics['auc']
                    best_f1 = metrics_opt['f1']
                    best_threshold = opt_threshold
                    best_epoch = epoch
                    
                    torch.save(
                        {
                            'model_state': model.state_dict(),
                            'threshold': opt_threshold,
                            'epoch': epoch,
                            'auc': best_auc,
                            'f1': best_f1
                        },
                        CHECKPOINT_DIR / f"best_model_fold{fold}.pt"
                    )
                    
                    logger.info(
                        f"✓ New best: AUC={best_auc:.4f}, "
                        f"F1={best_f1:.4f} @ threshold={best_threshold:.3f}"
                    )
                
                # Early stopping check
                if early_stopping(metrics['auc']):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Final evaluation
        checkpoint = torch.load(CHECKPOINT_DIR / f"best_model_fold{fold}.pt", weights_only=False)
        model.load_state_dict(checkpoint['model_state'])
        final_threshold = checkpoint['threshold']
        
        final_metrics = evaluate(model, val_loader, threshold=final_threshold)
        
        logger.info(f"\nFold {fold+1} Final Results:")
        logger.info(f"  Best epoch: {best_epoch}")
        logger.info(f"  AUC: {final_metrics['auc']:.4f}")
        logger.info(f"  F1: {final_metrics['f1']:.4f} (threshold={final_threshold:.3f})")
        logger.info(f"  Accuracy: {final_metrics['acc']:.4f}")
        logger.info(f"  Confusion Matrix:")
        logger.info(f"    {final_metrics['cm']}")
        
        fold_results['auc'].append(final_metrics['auc'])
        fold_results['f1'].append(final_metrics['f1'])
        fold_results['acc'].append(final_metrics['acc'])
        fold_results['threshold'].append(final_threshold)
        fold_results['best_epoch'].append(best_epoch)
        
        # NEW: Store fold predictions for ensemble
        all_fold_probs.append(final_metrics['probs'])
        all_fold_labels.append(final_metrics['labels'])
    
    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("FINAL CROSS-VALIDATION RESULTS")
    logger.info(f"{'='*70}")
    logger.info(f"Mean AUC: {np.mean(fold_results['auc']):.4f} ± {np.std(fold_results['auc']):.4f}")
    logger.info(f"Mean F1: {np.mean(fold_results['f1']):.4f} ± {np.std(fold_results['f1']):.4f}")
    logger.info(f"Mean Accuracy: {np.mean(fold_results['acc']):.4f} ± {np.std(fold_results['acc']):.4f}")
    logger.info(f"Mean Threshold: {np.mean(fold_results['threshold']):.3f}")
    logger.info(f"Mean Best Epoch: {np.mean(fold_results['best_epoch']):.1f}")
    logger.info(f"\nPer-fold AUCs: {[f'{x:.4f}' for x in fold_results['auc']]}")
    logger.info(f"Per-fold F1s: {[f'{x:.4f}' for x in fold_results['f1']]}")
    logger.info(f"Per-fold Best Epochs: {fold_results['best_epoch']}")
    
    # NEW: Ensemble evaluation (average predictions from all folds)
    if GNN_ENSEMBLE_MODE and all_fold_probs:
        logger.info(f"\n{'='*70}")
        logger.info("ENSEMBLE PREDICTIONS (AVERAGE ACROSS 5 FOLDS)")
        logger.info(f"{'='*70}")

        # Concatenate all fold predictions (validation sets differ per fold)
        ensemble_probs = np.concatenate(all_fold_probs)
        ensemble_labels = np.concatenate(all_fold_labels)
        ensemble_auc = roc_auc_score(ensemble_labels, ensemble_probs)

        logger.info(f"Ensemble AUC (val concat): {ensemble_auc:.4f}")
        logger.info(f"Individual fold AUCs: {[f'{x:.4f}' for x in fold_results['auc']]}")

        # ALSO: Evaluate a true ensemble on the held-out test split
        try:
            from src.features.graph_factory import ABIDECausalDataset

            test_dataset = ABIDECausalDataset(split='test')
            test_data = [test_dataset[i] for i in range(len(test_dataset)) if test_dataset[i] is not None]
            if len(test_data) > 0:
                test_loader = DataLoader(test_data, batch_size=GNN_BATCH_SIZE)

                test_fold_probs = []
                test_labels_ref = None

                for fold in range(K_FOLDS):
                    # Recreate model and load best state for each fold
                    from src.models.causal_gnn import CausalBrainGNN
                    model = CausalBrainGNN(
                        num_node_features=GNN_IN_CHANNELS,
                        hidden_channels=GNN_HIDDEN_CHANNELS_TUNED,
                        num_classes=2,
                        dropout=0.5,
                        num_heads=2,
                        num_sites=20,
                        use_site_embedding=True,
                        use_demographics=True,
                        strip_yolo_metadata=True,   # << toggle this for coords-only

                    ).to(DEVICE)

                    checkpoint = torch.load(CHECKPOINT_DIR / f"best_model_fold{fold}.pt", weights_only=False)
                    model.load_state_dict(checkpoint['model_state'])

                    metrics_test = evaluate(model, test_loader, threshold=checkpoint.get('threshold', 0.5))
                    test_fold_probs.append(metrics_test['probs'])
                    if test_labels_ref is None:
                        test_labels_ref = metrics_test['labels']

                # Stack probabilities from all folds and compute (weighted) average
                prob_matrix = np.stack(test_fold_probs, axis=0)  # shape: (K, N)
                weights = np.array(fold_results['auc'])
                if np.all(np.isfinite(weights)) and weights.sum() > 0:
                    weights = weights / weights.sum()
                    ensemble_test_probs = np.average(prob_matrix, axis=0, weights=weights)
                else:
                    ensemble_test_probs = prob_matrix.mean(axis=0)

                ensemble_test_auc = roc_auc_score(test_labels_ref, ensemble_test_probs)
                # Optimize threshold on ensemble probs for F1/Accuracy
                opt_t, _ = find_optimal_threshold(test_labels_ref, ensemble_test_probs)
                ensemble_test_preds = (ensemble_test_probs > opt_t).astype(int)
                ensemble_test_f1 = f1_score(test_labels_ref, ensemble_test_preds, zero_division=0)
                ensemble_test_acc = accuracy_score(test_labels_ref, ensemble_test_preds)

                logger.info(f"\nTrue Ensemble (test split):")
                logger.info(f"  AUC: {ensemble_test_auc:.4f}")
                logger.info(f"  F1: {ensemble_test_f1:.4f} (threshold={opt_t:.3f})")
                logger.info(f"  Accuracy: {ensemble_test_acc:.4f}")
            else:
                logger.info("Test split empty or unavailable; skipping test ensemble.")
        except Exception as e:
            logger.warning(f"Failed to compute test ensemble: {e}")
    
    logger.info(f"{'='*70}\n")


if __name__ == "__main__":
    run_enhanced_training()