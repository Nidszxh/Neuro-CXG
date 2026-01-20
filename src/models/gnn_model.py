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

warnings.filterwarnings('ignore')

# Setup paths and config
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import (
    K_FOLDS, GNN_BATCH_SIZE, GNN_EPOCHS, 
    CHECKPOINT_DIR, DEVICE, GNN_IN_CHANNELS
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for class imbalance.
    
    Automatically focuses on hard-to-classify examples.
    """
    def __init__(self, alpha=0.75, gamma=2.0):
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


def train_one_epoch(model, loader, optimizer, criterion, gradient_accumulation_steps=1):
    """
    Train for one epoch with gradient accumulation.
    
    Args:
        gradient_accumulation_steps: Accumulate gradients over N batches
                                      (effective batch size = batch_size * N)
    """
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    for i, data in enumerate(loader):
        if data is None:
            continue
            
        data = data.to(DEVICE)
        
        # Forward pass
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = criterion(out, data.y)
        
        # Normalize loss for gradient accumulation
        loss = loss / gradient_accumulation_steps
        loss.backward()
        
        # Update weights every N steps
        if (i + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * gradient_accumulation_steps
    
    # Final update if not divisible
    if (i + 1) % gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
    
    return total_loss / len(loader)


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
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
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
    """Main training loop with all PART 3 optimizations."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    from src.features.graph_factory import ABIDECausalDataset
    from src.models.enhanced_gnn import EnhancedCausalBrainGNN
    
    # Load dataset
    dataset = ABIDECausalDataset(split='train')
    
    # Extract labels
    labels = []
    for i in range(len(dataset)):
        data = dataset.get(i)
        if data is not None:
            labels.append(data.y.item())
    
    # Compute class weights
    class_weights = compute_class_weights(labels)
    
    logger.info(f"\n{'='*70}")
    logger.info("ENHANCED 5-FOLD CV WITH OPTIMIZATIONS")
    logger.info(f"{'='*70}")
    logger.info(f"Total subjects: {len(labels)}")
    logger.info(f"Learning rate: 0.0005 (reduced from 0.001)")
    logger.info(f"Early stopping: patience=25")
    logger.info(f"Focal Loss: α=0.75, γ=2.0")
    logger.info(f"Gradient accumulation: 2 steps (effective batch=64)")
    logger.info(f"{'='*70}\n")
    
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_results = {
        'auc': [],
        'f1': [],
        'acc': [],
        'threshold': [],
        'best_epoch': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        logger.info(f"\n{'='*70}")
        logger.info(f"FOLD {fold+1}/{K_FOLDS}")
        logger.info(f"{'='*70}")
        
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
        model = EnhancedCausalBrainGNN(
            num_node_features=GNN_IN_CHANNELS,
            hidden_channels=64,
            num_classes=2,
            dropout=0.5,
            num_heads=2
        ).to(DEVICE)
        
        # Optimizer with REDUCED learning rate
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.0005,  # Reduced from 0.001
            weight_decay=1e-3
        )
        
        # Learning rate warmup
        warmup = WarmupScheduler(optimizer, warmup_epochs=5, base_lr=0.0005)
        
        # Cosine annealing after warmup
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=GNN_EPOCHS - 5
        )
        
        # Focal Loss
        criterion = FocalLoss(alpha=0.75, gamma=2.0)
        
        # Early stopping
        early_stopping = EarlyStopping(patience=25, min_delta=0.001, mode='max')
        
        best_auc = 0.0
        best_f1 = 0.0
        best_threshold = 0.5
        best_epoch = 0
        
        for epoch in range(1, GNN_EPOCHS + 1):
            # Warmup for first 5 epochs
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
        checkpoint = torch.load(CHECKPOINT_DIR / f"best_model_fold{fold}.pt")
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
    
    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("FINAL ENHANCED CROSS-VALIDATION RESULTS")
    logger.info(f"{'='*70}")
    logger.info(f"Mean AUC: {np.mean(fold_results['auc']):.4f} ± {np.std(fold_results['auc']):.4f}")
    logger.info(f"Mean F1: {np.mean(fold_results['f1']):.4f} ± {np.std(fold_results['f1']):.4f}")
    logger.info(f"Mean Accuracy: {np.mean(fold_results['acc']):.4f} ± {np.std(fold_results['acc']):.4f}")
    logger.info(f"Mean Threshold: {np.mean(fold_results['threshold']):.3f}")
    logger.info(f"Mean Best Epoch: {np.mean(fold_results['best_epoch']):.1f}")
    logger.info(f"\nPer-fold AUCs: {[f'{x:.4f}' for x in fold_results['auc']]}")
    logger.info(f"Per-fold F1s: {[f'{x:.4f}' for x in fold_results['f1']]}")
    logger.info(f"Per-fold Best Epochs: {fold_results['best_epoch']}")
    logger.info(f"{'='*70}\n")


if __name__ == "__main__":
    run_enhanced_training()