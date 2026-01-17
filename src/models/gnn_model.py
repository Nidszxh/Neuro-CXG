import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, accuracy_score, precision_recall_curve
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import warnings

warnings.filterwarnings('ignore', message='.*torch-scatter.*')

# Setup paths and config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.core.config import (
    K_FOLDS, GNN_BATCH_SIZE, GNN_LEARNING_RATE,
    GNN_EPOCHS, CHECKPOINT_DIR, DEVICE, GNN_IN_CHANNELS
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Focuses training on hard-to-classify examples by down-weighting
    easy examples. Prevents the model from being "lazy" and always
    predicting the majority class.
    
    Formula: FL(p_t) = -α(1-p_t)^γ * log(p_t)
    
    Args:
        alpha: Weighting factor for minority class (default: 0.75 for ~25% minority)
        gamma: Focusing parameter (default: 2.0, standard value)
    """
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model logits (batch_size, 2)
            targets: Ground truth labels (batch_size,)
        """
        # Get probabilities
        probs = F.softmax(inputs, dim=1)
        
        # Get probability of correct class
        targets_one_hot = F.one_hot(targets, num_classes=2).float()
        pt = (probs * targets_one_hot).sum(dim=1)
        
        # Focal term: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # Alpha weighting (higher for minority class)
        alpha_weight = targets_one_hot[:, 1] * self.alpha + targets_one_hot[:, 0] * (1 - self.alpha)
        
        # Cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Focal loss
        focal_loss = alpha_weight * focal_weight * ce_loss
        
        return focal_loss.mean()


def compute_class_weights(labels):
    """
    Compute class weights for imbalanced dataset.
    
    Args:
        labels: List of binary labels (0 or 1)
    
    Returns:
        Tensor of class weights [weight_control, weight_asd]
    """
    labels_array = np.array(labels)
    
    # Count each class
    n_control = (labels_array == 0).sum()
    n_asd = (labels_array == 1).sum()
    total = len(labels_array)
    
    # Inverse frequency weighting
    weight_control = total / (2 * n_control) if n_control > 0 else 1.0
    weight_asd = total / (2 * n_asd) if n_asd > 0 else 1.0
    
    logger.info(f"Class distribution: Control={n_control}, ASD={n_asd}")
    logger.info(f"Class weights: Control={weight_control:.3f}, ASD={weight_asd:.3f}")
    
    return torch.tensor([weight_control, weight_asd], dtype=torch.float32)


def find_optimal_threshold(y_true, y_probs):
    """
    Find optimal classification threshold by maximizing F1 score.
    
    Instead of using default 0.5 threshold, find the threshold that
    maximizes F1 score on validation data.
    
    Args:
        y_true: Ground truth labels
        y_probs: Predicted probabilities for positive class
    
    Returns:
        Tuple of (optimal_threshold, best_f1)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    
    # F1 = 2 * (precision * recall) / (precision + recall)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    # Find threshold that maximizes F1
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]
    
    return best_threshold, best_f1


def balanced_batch_sampler(dataset, batch_size):
    """
    Create balanced batches with equal Control and ASD samples.
    
    Args:
        dataset: PyTorch Geometric dataset
        batch_size: Target batch size
    
    Returns:
        List of balanced batch indices
    """
    # Separate indices by class
    control_indices = []
    asd_indices = []
    
    for i in range(len(dataset)):
        data = dataset[i]
        if data is not None:
            label = data.y.item()
            if label == 0:
                control_indices.append(i)
            else:
                asd_indices.append(i)
    
    # Create balanced batches
    balanced_batches = []
    samples_per_class = batch_size // 2
    
    # Shuffle
    np.random.shuffle(control_indices)
    np.random.shuffle(asd_indices)
    
    # Oversample minority class if needed
    if len(asd_indices) < len(control_indices):
        # Repeat ASD indices to match Control
        asd_indices = asd_indices * (len(control_indices) // len(asd_indices) + 1)
        asd_indices = asd_indices[:len(control_indices)]
    
    # Create batches
    for i in range(0, len(control_indices), samples_per_class):
        batch = []
        batch.extend(control_indices[i:i + samples_per_class])
        batch.extend(asd_indices[i:i + samples_per_class])
        
        if len(batch) >= 4:  # Minimum batch size
            balanced_batches.append(batch)
    
    return balanced_batches


def train_one_epoch(model, loader, optimizer, criterion, use_class_weights=False, class_weights=None):
    """Train for one epoch with optional class weighting."""
    model.train()
    total_loss = 0
    
    for data in loader:
        # CRITICAL FIX: Skip null graphs that have no edges
        if data is None:
            continue
            
        data = data.to(DEVICE)
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        
        # Use class-weighted loss if specified
        if use_class_weights and class_weights is not None:
            weights = class_weights.to(DEVICE)
            loss = F.cross_entropy(out, data.y, weight=weights)
        else:
            loss = criterion(out, data.y)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, threshold=0.5):
    """
    Evaluate model with custom threshold.
    
    Args:
        model: GNN model
        loader: Data loader
        threshold: Classification threshold (default 0.5)
    """
    model.eval()
    all_probs = []
    all_labels = []
    
    for data in loader:
        # CRITICAL FIX: Skip null graphs that have no edges
        if data is None:
            continue
            
        data = data.to(DEVICE)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        probs = torch.softmax(out, dim=1)
        all_probs.append(probs[:, 1].cpu().numpy())
        all_labels.append(data.y.cpu().numpy())
    
    probs_array = np.concatenate(all_probs)
    labels_array = np.concatenate(all_labels)
    
    # Apply custom threshold
    preds_array = (probs_array > threshold).astype(int)
    
    # Calculate metrics
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


def run_kfold_training_balanced():
    """Main training loop with class imbalance fixes."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    from src.features.graph_factory import ABIDECausalDataset
    from src.models.causal_gnn import CausalBrainGNN
    
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
    logger.info("BALANCED 5-FOLD CV WITH CLASS IMBALANCE FIXES")
    logger.info(f"{'='*70}")
    logger.info(f"Total subjects: {len(labels)}")
    logger.info(f"Using Focal Loss (α={0.75}, γ={2.0})")
    logger.info(f"Class weights: {class_weights.numpy()}")
    logger.info(f"{'='*70}\n")
    
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_results = {
        'auc': [],
        'f1': [],
        'acc': [],
        'threshold': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        logger.info(f"\n{'='*70}")
        logger.info(f"FOLD {fold+1}/{K_FOLDS}")
        logger.info(f"{'='*70}")
        
        # Create balanced training batches
        train_data = [dataset[i] for i in train_idx if dataset[i] is not None]
        val_data = [dataset[i] for i in val_idx if dataset[i] is not None]
        
        # Check class distribution in fold
        train_labels = [d.y.item() for d in train_data]
        val_labels = [d.y.item() for d in val_data]
        
        logger.info(f"Train: Control={train_labels.count(0)}, ASD={train_labels.count(1)}")
        logger.info(f"Val: Control={val_labels.count(0)}, ASD={val_labels.count(1)}")
        
        # Standard loaders (will use Focal Loss to handle imbalance)
        train_loader = DataLoader(train_data, batch_size=GNN_BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=GNN_BATCH_SIZE)
        
        # Initialize model
        model = CausalBrainGNN(
            num_node_features=GNN_IN_CHANNELS,
            hidden_channels=64
        ).to(DEVICE)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=GNN_LEARNING_RATE,
            weight_decay=1e-3
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=GNN_EPOCHS
        )
        
        # Use Focal Loss instead of CrossEntropy
        criterion = FocalLoss(alpha=0.75, gamma=2.0)
        
        best_auc = 0.0
        best_f1 = 0.0
        best_threshold = 0.5
        patience = 25
        no_improve = 0
        
        for epoch in range(1, GNN_EPOCHS + 1):
            # Train
            loss = train_one_epoch(
                model, train_loader, optimizer, criterion,
                use_class_weights=False  # Focal Loss handles this
            )
            scheduler.step()
            
            # Evaluate with default threshold
            metrics = evaluate(model, val_loader, threshold=0.5)
            
            # Find optimal threshold every 10 epochs
            if epoch % 10 == 0:
                opt_threshold, opt_f1 = find_optimal_threshold(
                    metrics['labels'],
                    metrics['probs']
                )
                
                # Re-evaluate with optimal threshold
                metrics_opt = evaluate(model, val_loader, threshold=opt_threshold)
                
                logger.info(
                    f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
                    f"AUC: {metrics['auc']:.4f} | "
                    f"F1@0.5: {metrics['f1']:.4f} | "
                    f"F1@{opt_threshold:.2f}: {metrics_opt['f1']:.4f}"
                )
                
                # Save if AUC improves (primary metric)
                if metrics['auc'] > best_auc:
                    best_auc = metrics['auc']
                    best_f1 = metrics_opt['f1']
                    best_threshold = opt_threshold
                    no_improve = 0
                    
                    # Save model
                    torch.save(
                        {
                            'model_state': model.state_dict(),
                            'threshold': opt_threshold,
                            'epoch': epoch
                        },
                        CHECKPOINT_DIR / f"best_model_fold{fold}.pt"
                    )
                    
                    logger.info(
                        f"✓ New best: AUC={best_auc:.4f}, "
                        f"F1={best_f1:.4f} @ threshold={best_threshold:.3f}"
                    )
                else:
                    no_improve += 1
            
            if no_improve >= patience // 10:  # Adjusted for 10-epoch eval
                logger.info(f"Early stop at epoch {epoch}")
                break
        
        # Final evaluation with best threshold
        checkpoint = torch.load(CHECKPOINT_DIR / f"best_model_fold{fold}.pt", weights_only=False)
        model.load_state_dict(checkpoint['model_state'])
        final_threshold = checkpoint['threshold']
        
        final_metrics = evaluate(model, val_loader, threshold=final_threshold)
        
        logger.info(f"\nFold {fold+1} Final Results:")
        logger.info(f"  AUC: {final_metrics['auc']:.4f}")
        logger.info(f"  F1: {final_metrics['f1']:.4f} (threshold={final_threshold:.3f})")
        logger.info(f"  Accuracy: {final_metrics['acc']:.4f}")
        logger.info(f"  Confusion Matrix:")
        logger.info(f"    {final_metrics['cm']}")
        
        fold_results['auc'].append(final_metrics['auc'])
        fold_results['f1'].append(final_metrics['f1'])
        fold_results['acc'].append(final_metrics['acc'])
        fold_results['threshold'].append(final_threshold)
    
    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("FINAL CROSS-VALIDATION RESULTS")
    logger.info(f"{'='*70}")
    logger.info(f"Mean AUC: {np.mean(fold_results['auc']):.4f} ± {np.std(fold_results['auc']):.4f}")
    logger.info(f"Mean F1: {np.mean(fold_results['f1']):.4f} ± {np.std(fold_results['f1']):.4f}")
    logger.info(f"Mean Accuracy: {np.mean(fold_results['acc']):.4f} ± {np.std(fold_results['acc']):.4f}")
    logger.info(f"Mean Threshold: {np.mean(fold_results['threshold']):.3f}")
    logger.info(f"\nPer-fold AUCs: {[f'{x:.4f}' for x in fold_results['auc']]}")
    logger.info(f"Per-fold F1s: {[f'{x:.4f}' for x in fold_results['f1']]}")
    logger.info(f"{'='*70}\n")


if __name__ == "__main__":
    run_kfold_training_balanced()