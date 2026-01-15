import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from tqdm import tqdm
import sys

# Setup paths and config
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    K_FOLDS, GNN_BATCH_SIZE, GNN_LEARNING_RATE, GNN_WEIGHT_DECAY,
    GNN_EPOCHS, CHECKPOINT_DIR, DEVICE
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_one_epoch(model, loader, optimizer, criterion, scheduler):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(DEVICE)
        optimizer.zero_grad()
        # Ensure edge_attr is passed for causal influence weights
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        # Gradient clipping to prevent exploding gradients in GNNs
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []
    
    for data in loader:
        data = data.to(DEVICE)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        probs = torch.softmax(out, dim=1)
        preds = out.argmax(dim=1)
        
        all_probs.extend(probs[:, 1].cpu().numpy()) # Probability of ASD
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())
    
    return {
        'acc': (np.array(all_preds) == np.array(all_labels)).mean(),
        'f1': f1_score(all_labels, all_preds),
        'auc': roc_auc_score(all_labels, all_probs), # Proper probability-based AUC
        'cm': confusion_matrix(all_labels, all_preds)
    }

def run_kfold_training():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    from data.graph_factory import ABIDECausalDataset
    from models.causal_gnn import CausalBrainGNN
    
    full_dataset = ABIDECausalDataset(split='train')
    
    # Stratification targets
    labels = [int(data.y) for data in full_dataset]
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    
    fold_results = []

    logger.info(f"Starting {K_FOLDS}-fold cross-validation on {DEVICE}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        logger.info(f"{'='*60}")
        logger.info(f"FOLD {fold+1}/{K_FOLDS}")
        logger.info(f"{'='*60}")
        
        # Subset indices
        train_loader = DataLoader(
            [full_dataset[i] for i in train_idx], 
            batch_size=GNN_BATCH_SIZE, 
            shuffle=True
        )
        val_loader = DataLoader(
            [full_dataset[i] for i in val_idx], 
            batch_size=GNN_BATCH_SIZE
        )
        
        model = CausalBrainGNN(
            num_node_features=full_dataset[0].x.shape[1], 
            hidden_channels=64
        ).to(DEVICE)
        
        # Optimizer with Cosine Annealing (Standard for high-impact deep learning)
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=GNN_LEARNING_RATE, 
            weight_decay=GNN_WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=GNN_EPOCHS
        )
        
        # Label Smoothing helps with the 'noisy' labels in neuroimaging
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        best_fold_auc = 0
        patience_counter = 0
        patience = 15
        
        for epoch in range(1, GNN_EPOCHS + 1):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, scheduler)
            
            if epoch % 10 == 0 or epoch == 1:
                metrics = evaluate(model, val_loader)
                logger.info(
                    f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
                    f"Val Acc: {metrics['acc']:.4f} | AUC: {metrics['auc']:.4f} | F1: {metrics['f1']:.4f}"
                )
                
                # Save based on AUC (Better indicator of diagnostic potential than Accuracy)
                if metrics['auc'] > best_fold_auc:
                    best_fold_auc = metrics['auc']
                    patience_counter = 0
                    torch.save(model.state_dict(), CHECKPOINT_DIR / f"best_model_fold{fold}.pt")
                    logger.info(f"New best model saved (AUC: {best_fold_auc:.4f})")
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        fold_results.append(best_fold_auc)
        logger.info(f"Fold {fold+1} best AUC: {best_fold_auc:.4f}")

    logger.info(f"{'='*60}")
    logger.info(f"CROSS-VALIDATION RESULTS")
    logger.info(f"Mean AUC: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")
    logger.info(f"Per-fold AUCs: {[f'{x:.4f}' for x in fold_results]}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    run_kfold_training()