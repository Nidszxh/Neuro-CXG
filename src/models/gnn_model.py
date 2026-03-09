import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, confusion_matrix,
    accuracy_score, precision_recall_curve, average_precision_score
)
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import sys
import warnings
import time

warnings.filterwarnings('ignore')

# Setup paths and config
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import (
    K_FOLDS, GNN_BATCH_SIZE, GNN_EPOCHS, 
    CHECKPOINT_DIR, DEVICE, GNN_IN_CHANNELS,
    ALL_FEATURE_NAMES,
    GNN_HIDDEN_CHANNELS,
    GNN_DROPOUT,
    GNN_WEIGHT_DECAY,
    GNN_NUM_HEADS,
    GNN_NUM_LAYERS,
    GNN_POOLING,
    GNN_USE_GRL,
    GNN_GRL_ALPHA,
    GNN_SITE_LOSS_WEIGHT,
    GNN_EDGE_GATE,
    GNN_ONECYCLE_MAX_LR,
    GNN_EARLY_STOPPING_PATIENCE,
    FOCAL_LOSS_ALPHA,
    FOCAL_LOSS_GAMMA,
    GNN_USE_SITE_EMBEDDING,
    GNN_USE_DEMOGRAPHICS,
    GNN_NODE_EMB_DIM,
    NUM_LOBES,
    CAUSAL_GRAPHS_DIR,
    DATA_METADATA,
    RESULTS_TRAINING_DIR,
)
from src.models.training_utils import (
    TrainingTracker, 
    CheckpointManager,
    train_fold_with_onecycle
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Analysis modules
from src.analysis.diagnostics import CausalGraphAnalyzer, TrainingMonitor
try:
    from src.analysis.feature_attribution import FeatureAttributionAnalyzer
    FEATURE_ANALYSIS_AVAILABLE = True
except ImportError:
    FEATURE_ANALYSIS_AVAILABLE = False
    logger.warning("FeatureAttributionAnalyzer unavailable (requires Captum)")


# FOCAL LOSS (Keep - not in training_utils)

class FocalLoss(nn.Module):
    """
    Focal Loss for class imbalance.
    
    Automatically focuses on hard-to-classify examples.
    Args:
        alpha: Weight for positive class (ASD)
        gamma: Focusing parameter (higher = more focus on hard examples)
        pos_weight: Additional weight for positive class (multiplicative with alpha)
    """
    def __init__(self, alpha=0.75, gamma=3.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Raw logits (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)
        """
        probs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=2).float()
        pt = (probs * targets_one_hot).sum(dim=1)
        
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = targets_one_hot[:, 1] * self.alpha + targets_one_hot[:, 0] * (1 - self.alpha)
        
        if self.pos_weight is not None:
            weight = targets_one_hot[:, 1] * self.pos_weight + targets_one_hot[:, 0]
            ce_loss = F.cross_entropy(inputs, targets, reduction='none') * weight
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        focal_loss = alpha_weight * focal_weight * ce_loss
        
        return focal_loss.mean()


# UTILITY FUNCTIONS

def compute_class_weights(labels):
    """
    Compute inverse frequency class weights.
    
    Args:
        labels: List of labels (0=Control, 1=ASD)
    
    Returns:
        Tensor of class weights [weight_control, weight_asd]
    """
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
    """
    Find classification threshold that maximizes F1 score.
    
    Args:
        y_true: Ground truth labels
        y_probs: Predicted probabilities
    
    Returns:
        (best_threshold, best_f1): Optimal threshold and corresponding F1 score
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]
    
    return best_threshold, best_f1


@torch.no_grad()
def evaluate(model, loader, threshold=0.5):
    """
    Evaluate model on a dataset.
    
    Args:
        model: Trained GNN model
        loader: DataLoader with evaluation data
        threshold: Classification threshold
    
    Returns:
        Dictionary with metrics: acc, f1, auc, auprc, cm, probs, labels
    """
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
    
    if not all_probs:
        logger.warning("No predictions collected during evaluation")
        return {
            'acc': 0.0, 'f1': 0.0, 'auc': 0.5, 'auprc': 0.0,
            'cm': np.zeros((2, 2)),
            'probs': np.array([]), 'labels': np.array([])
        }
    
    probs_array = np.concatenate(all_probs)
    labels_array = np.concatenate(all_labels)
    
    # SAFETY: Check for NaN in probs before computing metrics
    if np.isnan(probs_array).any():
        logger.error(f"Predictions contain NaN values! Skipping AUC computation.")
        logger.error(f"  NaN count: {np.isnan(probs_array).sum()} / {len(probs_array)}")
        return {
            'acc': 0.0, 'f1': 0.0, 'auc': 0.5, 'auprc': 0.0,
            'cm': np.zeros((2, 2)),
            'probs': probs_array, 'labels': labels_array
        }
    
    preds_array = (probs_array > threshold).astype(int)
    
    auc = roc_auc_score(labels_array, probs_array)
    auprc = average_precision_score(labels_array, probs_array)
    f1 = f1_score(labels_array, preds_array, zero_division=0)
    acc = accuracy_score(labels_array, preds_array)
    cm = confusion_matrix(labels_array, preds_array)
    
    return {
        'acc': acc,
        'f1': f1,
        'auc': auc,
        'auprc': auprc,
        'cm': cm,
        'probs': probs_array,
        'labels': labels_array
    }


def evaluate_ensemble(tracker: TrainingTracker, checkpoint_manager: CheckpointManager):
    """
    Evaluate ensemble of all fold models on test set.
    
    Args:
        tracker: TrainingTracker with fold results
        checkpoint_manager: CheckpointManager for loading models
    """
    from src.features.graph_factory import ABIDECausalDataset
    from src.models.causal_gnn import CausalBrainGNN
    
    logger.info(f"\n{'='*70}")
    logger.info("ENSEMBLE EVALUATION (TEST SET)")
    logger.info(f"{'='*70}")
    
    try:
        # Load test dataset
        test_dataset = ABIDECausalDataset(split='test')
        test_data = [test_dataset[i] for i in range(len(test_dataset)) if test_dataset[i] is not None]
        
        if len(test_data) == 0:
            logger.warning("Test set is empty, skipping ensemble evaluation")
            return
        
        test_loader = DataLoader(test_data, batch_size=GNN_BATCH_SIZE)
        
        # Collect predictions from all folds
        test_fold_probs = []
        test_labels_ref = None
        fold_aucs = []
        
        for fold in range(K_FOLDS):
            # Initialize model
            model = CausalBrainGNN(
                num_node_features=GNN_IN_CHANNELS,
                hidden_channels=GNN_HIDDEN_CHANNELS,
                num_classes=2,
                dropout=GNN_DROPOUT,
                num_heads=GNN_NUM_HEADS,
                num_layers=GNN_NUM_LAYERS,
                pooling=GNN_POOLING,
                num_sites=20,
                use_site_embedding=GNN_USE_SITE_EMBEDDING,
                use_demographics=GNN_USE_DEMOGRAPHICS,
                use_grl=GNN_USE_GRL,
                grl_alpha=GNN_GRL_ALPHA,
                edge_gate=GNN_EDGE_GATE,
                num_nodes=NUM_LOBES,
                node_emb_dim=GNN_NODE_EMB_DIM,
            ).to(DEVICE)
            try:
                checkpoint = checkpoint_manager.load(model, fold=fold)
                threshold = checkpoint.get('threshold', 0.5)
                fold_auc = checkpoint.get('auc', 0.0)
                fold_aucs.append(fold_auc)
            except FileNotFoundError:
                logger.warning(f"Checkpoint for fold {fold} not found, skipping")
                continue
            
            # Evaluate on test set
            metrics = evaluate(model, test_loader, threshold=threshold)
            test_fold_probs.append(metrics['probs'])
            
            if test_labels_ref is None:
                test_labels_ref = metrics['labels']
        
        if not test_fold_probs:
            logger.warning("No fold predictions collected")
            return

        if test_labels_ref is None or len(test_labels_ref) == 0:
            logger.warning("No reference labels collected from test set")
            return

        # Compute weighted ensemble
        prob_matrix = np.stack(test_fold_probs, axis=0)  # (K_folds, N_samples)
        
        # Weight by validation AUC
        weights = np.array(fold_aucs)
        if np.all(np.isfinite(weights)) and weights.sum() > 0:
            weights = weights / weights.sum()
            ensemble_probs = np.average(prob_matrix, axis=0, weights=weights)
            logger.info(f"Using AUC-weighted ensemble: {weights}")
        else:
            ensemble_probs = prob_matrix.mean(axis=0)
            logger.info("Using uniform ensemble averaging")
        
        # Compute ensemble metrics
        ensemble_auc = roc_auc_score(test_labels_ref, ensemble_probs)
        
        # Find optimal threshold for ensemble
        opt_threshold, _ = find_optimal_threshold(test_labels_ref, ensemble_probs)
        ensemble_preds = (ensemble_probs > opt_threshold).astype(int)
        ensemble_f1 = f1_score(test_labels_ref, ensemble_preds, zero_division=0)
        ensemble_acc = accuracy_score(test_labels_ref, ensemble_preds)
        ensemble_cm = confusion_matrix(test_labels_ref, ensemble_preds)
        
        # Report results
        logger.info(f"\nEnsemble Results (Test Set):")
        logger.info(f"  AUC: {ensemble_auc:.4f}")
        logger.info(f"  F1: {ensemble_f1:.4f} (threshold={opt_threshold:.3f})")
        logger.info(f"  Accuracy: {ensemble_acc:.4f}")
        logger.info(f"  Confusion Matrix:")
        logger.info(f"    {ensemble_cm}")
        logger.info(f"{'='*70}\n")
        
    except Exception as e:
        logger.error(f"Ensemble evaluation failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())


# MAIN TRAINING FUNCTION


def _set_global_seed(seed: int = 42) -> None:
    """Set all random seeds for full reproducibility.

    Sets Python, NumPy, PyTorch, and CUDA seeds.  Also forces cuDNN into
    deterministic mode (deterministic=True, benchmark=False) so that
    convolution algorithms are selected deterministically across runs.
    This trades a small amount of runtime performance for exact reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # CUDA determinism: required for reproducible training on GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Global seed set to %d (cuDNN deterministic mode enabled)", seed)


def run_training():
    """
    Main training loop with k-fold cross-validation.
    
    Uses modular training utilities for maintainability:
    - EarlyStopping: Prevents overfitting
    - OneCycleLR: Faster convergence with warmup
    - TrainingTracker: Aggregate fold results
    - CheckpointManager: Save/load best models
    """
    from src.features.graph_factory import ABIDECausalDataset
    from src.models.causal_gnn import CausalBrainGNN

    _set_global_seed(42)

    # Load dataset
    dataset = ABIDECausalDataset(split='train')
    
    # Extract labels for stratification
    labels = []
    site_labels = []
    for i in range(len(dataset)):
        data = dataset.get(i)
        if data is not None:
            labels.append(data.y.item())
            if hasattr(data, 'site_id') and data.site_id is not None and data.site_id.numel() > 0:
                site_labels.append(int(data.site_id.view(-1)[0].item()))
            else:
                site_labels.append(-1)
    
    if not labels:
        logger.error("No valid training data found!")
        return
    
    # Compute class weights (informational)
    class_weights = compute_class_weights(labels)
    
    # Initialize tracking
    tracker = TrainingTracker(k_folds=K_FOLDS)
    checkpoint_manager = CheckpointManager(CHECKPOINT_DIR, monitor='auc', mode='max')
    
    # Initialize training monitor for analysis
    analysis_dir = RESULTS_TRAINING_DIR
    monitor = TrainingMonitor(analysis_dir, num_folds=K_FOLDS)
    
    # Print configuration
    logger.info(f"\n{'='*70}")
    logger.info("GNN TRAINING - 5-FOLD CROSS-VALIDATION")
    logger.info(f"{'='*70}")
    logger.info(f"Total subjects: {len(labels)}")
    logger.info(f"OneCycle max LR: {GNN_ONECYCLE_MAX_LR}")
    logger.info(f"Hidden channels: {GNN_HIDDEN_CHANNELS}")
    logger.info(f"Input features: {GNN_IN_CHANNELS} (20 temporal + 6 spatial)")
    logger.info(f"Site conditioning: {GNN_USE_SITE_EMBEDDING}")
    logger.info(f"Demographics: {GNN_USE_DEMOGRAPHICS}")
    logger.info(f"Early stopping patience: {GNN_EARLY_STOPPING_PATIENCE}")
    logger.info(f"Focal Loss: α={FOCAL_LOSS_ALPHA}, γ={FOCAL_LOSS_GAMMA}")
    logger.info(f"{'='*70}\n")
    
    # K-fold cross-validation (strict manifest-only enforcement)
    if 'cv_fold' not in dataset.manifest.columns:
        raise ValueError(
            "cv_fold column not found in manifest. "
            "Run split.py first to generate predefined CV folds."
        )
    
    cv_folds = dataset.manifest['cv_fold'].values
    if cv_folds.min() < 0 or cv_folds.max() >= K_FOLDS:
        raise ValueError(
            f"Invalid cv_fold values: found [{cv_folds.min()}, {cv_folds.max()}], "
            f"expected [0, {K_FOLDS-1}]. Run split.py to regenerate folds."
        )
    
    # Build fold splits from manifest (aligned with harmonization)
    cv_splits = []
    for f in range(K_FOLDS):
        t_idx = np.where(cv_folds != f)[0]
        v_idx = np.where(cv_folds == f)[0]
        cv_splits.append((t_idx, v_idx))
        logger.debug(f"Fold {f}: train={len(t_idx)}, val={len(v_idx)}")
            
    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        logger.info(f"\n{'='*70}")
        logger.info(f"FOLD {fold+1}/{K_FOLDS}")
        logger.info(f"{'='*70}")

        _set_global_seed(42 + fold)  # deterministic per-fold model initialisation
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
            hidden_channels=GNN_HIDDEN_CHANNELS,
            num_classes=2,
            dropout=GNN_DROPOUT,
            num_heads=GNN_NUM_HEADS,
            num_layers=GNN_NUM_LAYERS,
            pooling=GNN_POOLING,
            num_sites=20,
            use_site_embedding=GNN_USE_SITE_EMBEDDING,
            use_demographics=GNN_USE_DEMOGRAPHICS,
            use_grl=GNN_USE_GRL,
            grl_alpha=GNN_GRL_ALPHA,
            edge_gate=GNN_EDGE_GATE,
            num_nodes=NUM_LOBES,
            node_emb_dim=GNN_NODE_EMB_DIM,
        ).to(DEVICE)
        
        # Loss function
        n_control = max((np.array(train_labels) == 0).sum(), 1)
        n_asd = max((np.array(train_labels) == 1).sum(), 1)
        pos_weight = float(n_control / n_asd)
        criterion = FocalLoss(alpha=FOCAL_LOSS_ALPHA, gamma=FOCAL_LOSS_GAMMA, pos_weight=pos_weight)
        checkpoint_manager.reset()

        best_state, best_metrics, history = train_fold_with_onecycle(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            device=DEVICE,
            epochs=GNN_EPOCHS,
            max_lr=GNN_ONECYCLE_MAX_LR,
            patience=GNN_EARLY_STOPPING_PATIENCE,
            use_grl=GNN_USE_GRL,
            grl_weight=GNN_SITE_LOSS_WEIGHT,
            fold=fold,
            weight_decay=GNN_WEIGHT_DECAY,
        )

        for entry in history:
            if entry['epoch'] % 10 == 0:
                monitor.log_epoch(
                    fold_id=fold,
                    epoch=entry['epoch'],
                    metrics={
                        'train_loss': entry['loss'],
                        'val_loss': 1.0 - entry['auc'],
                        'val_auc': entry['auc'],
                        'val_auprc': entry['auprc'],
                        'val_f1': entry['f1'],
                        'val_acc': entry['acc'],
                        'lr': entry['lr']
                    },
                    grad_norm=0.0,
                    confusion_matrix=entry['cm']
                )
                logger.info(
                    f"Epoch {entry['epoch']:03d} | LR: {entry['lr']:.6f} | Loss: {entry['loss']:.4f} | "
                    f"AUC: {entry['auc']:.4f} | AUPRC: {entry['auprc']:.4f} | "
                    f"F1@{entry['threshold']:.2f}: {entry['f1']:.4f}"
                )

        model.load_state_dict(best_state)
        checkpoint_metrics = {
            'auc': best_metrics['auc'],
            'auprc': best_metrics['auprc'],
            'f1': best_metrics['f1'],
            'threshold': best_metrics['threshold']
        }
        checkpoint_manager.save(model, None, best_metrics['best_epoch'], checkpoint_metrics, fold=fold)

        logger.info(
            f"✓ Best fold {fold}: AUC={best_metrics['auc']:.4f}, "
            f"AUPRC={best_metrics['auprc']:.4f}, F1={best_metrics['f1']:.4f}"
        )
        
        # Final evaluation with best checkpoint
        final_threshold = best_metrics['threshold']
        final_metrics = evaluate(model, val_loader, threshold=final_threshold)
        best_epoch = best_metrics['best_epoch']
        
        fold_train_time = time.time() - fold_start_time
        
        # Log fold results
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
        
        # Generate training visualizations for this fold
        logger.info("\nGenerating fold visualizations...")
        plot_path = monitor.plot_training_curves(fold)
        logger.info(f"  Training curves saved to: {plot_path}")
        
        history_path = monitor.save_history(fold)
        logger.info(f"  Training history saved to: {history_path}")
    
    # Log cross-validation summary
    tracker.log_summary()
    
    # Ensemble evaluation on test set (combine all folds)
    evaluate_ensemble(tracker, checkpoint_manager)
    
    # POST-TRAINING ANALYSIS
    logger.info(f"\n{'='*70}")
    logger.info("POST-TRAINING ANALYSIS")
    logger.info(f"{'='*70}\n")
    
    # 1. Feature Attribution Analysis (if Captum available)
    if FEATURE_ANALYSIS_AVAILABLE:
        try:
            logger.info("Running feature attribution analysis...")
            from src.features.graph_factory import ABIDECausalDataset
            
            # Load test set
            test_dataset = ABIDECausalDataset(split='test')
            test_loader = DataLoader(
                [d for d in test_dataset if d is not None],
                batch_size=GNN_BATCH_SIZE
            )
            
            # Define feature names (8 temporal + 6 spatial)
            feature_names = ALL_FEATURE_NAMES.copy()
            if len(feature_names) != GNN_IN_CHANNELS:
                logger.warning(
                    f"Feature name count ({len(feature_names)}) does not match "
                    f"GNN_IN_CHANNELS ({GNN_IN_CHANNELS}). Adjusting list for attribution."
                )
                if len(feature_names) > GNN_IN_CHANNELS:
                    feature_names = feature_names[:GNN_IN_CHANNELS]
                else:
                    missing = GNN_IN_CHANNELS - len(feature_names)
                    feature_names.extend([f"feature_{i+1}" for i in range(missing)])
            
            # Load best model (fold 0 as representative)
            best_model = CausalBrainGNN(
                num_node_features=GNN_IN_CHANNELS,
                hidden_channels=GNN_HIDDEN_CHANNELS,
                num_classes=2,
                dropout=GNN_DROPOUT,
                num_heads=GNN_NUM_HEADS,
                num_layers=GNN_NUM_LAYERS,
                pooling=GNN_POOLING,
                num_sites=20,
                use_site_embedding=GNN_USE_SITE_EMBEDDING,
                use_demographics=GNN_USE_DEMOGRAPHICS,
                use_grl=GNN_USE_GRL,
                grl_alpha=GNN_GRL_ALPHA,
                edge_gate=GNN_EDGE_GATE,
                num_nodes=NUM_LOBES,
                node_emb_dim=GNN_NODE_EMB_DIM,
            ).to(DEVICE)
            checkpoint_manager.load(best_model, fold=0, allow_partial=True)
            
            # Compute feature attributions
            feature_analyzer = FeatureAttributionAnalyzer(
                best_model, test_loader, feature_names, device=DEVICE
            )
            attributions = feature_analyzer.compute_attributions()
            
            # Visualize and save
            feature_output = analysis_dir / 'features'
            feature_output.mkdir(parents=True, exist_ok=True)
            feature_analyzer.visualize_feature_importance(
                attributions,
                str(feature_output / 'feature_importance.png')
            )
            logger.info(f"  Feature importance plot saved to: {feature_output / 'feature_importance.png'}")
            
        except Exception as e:
            logger.warning(f"Feature attribution analysis failed: {e}")
    
    # 2. Causal Graph Analysis
    try:
        logger.info("\nRunning causal graph analysis...")
        import pandas as pd
        
        # Load manifest
        manifest_path = DATA_METADATA / 'master_manifest.csv'
        manifest = pd.read_csv(manifest_path)
        
        # Compute graph properties
        graph_analyzer = CausalGraphAnalyzer(CAUSAL_GRAPHS_DIR, manifest)
        graph_metrics = graph_analyzer.compute_graph_properties()
        
        # Compare ASD vs Control
        graph_output = analysis_dir / 'graphs'
        graph_output.mkdir(parents=True, exist_ok=True)
        graph_analyzer.compare_asd_vs_control(
            graph_metrics,
            str(graph_output)
        )
        logger.info(f"  Graph analysis plots saved to: {graph_output}")
        
    except Exception as e:
        logger.warning(f"Causal graph analysis failed: {e}")
    
    logger.info(f"\n{'='*70}")
    logger.info("TRAINING AND ANALYSIS COMPLETE")
    logger.info(f"{'='*70}\n")


# CLI

if __name__ == "__main__":
    run_training()