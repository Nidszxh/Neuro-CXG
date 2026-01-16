import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, accuracy_score
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import warnings

# Suppress torch-scatter optional dependency warning (code runs fine without it)
warnings.filterwarnings('ignore', message='.*torch-scatter.*')

# Setup paths and config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    K_FOLDS, GNN_BATCH_SIZE, GNN_LEARNING_RATE,
    GNN_EPOCHS, CHECKPOINT_DIR, DEVICE, GNN_IN_CHANNELS
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(DEVICE)
        optimizer.zero_grad()
        # forward(x, edge_index, edge_attr, batch)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_probs = []
    all_labels = []
    
    for data in loader:
        data = data.to(DEVICE)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        probs = torch.softmax(out, dim=1)
        all_probs.append(probs[:, 1].cpu().numpy())
        all_labels.append(data.y.cpu().numpy())
    
    probs_array = np.concatenate(all_probs)
    labels_array = np.concatenate(all_labels)
    preds_array = (probs_array > 0.5).astype(int)
    
    return {
        'acc': accuracy_score(labels_array, preds_array),
        'f1': f1_score(labels_array, preds_array, zero_division=0),
        'auc': roc_auc_score(labels_array, probs_array),
        'cm': confusion_matrix(labels_array, preds_array)
    }

def run_kfold_training():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    from data.graph_factory import ABIDECausalDataset
    from models.causal_gnn import CausalBrainGNN
    
    # Load dataset (now strictly 5-node)
    dataset = ABIDECausalDataset(split='train')
    
    # Extract labels for stratification
    labels = []
    for i in range(len(dataset)):
        data = dataset.get(i)
        labels.append(data.y.item())
    
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_aucs = []

    logger.info(f"🚀 Starting 5-Fold CV | Input Channels: {GNN_IN_CHANNELS} | Nodes: 5")

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        logger.info(f"\n--- FOLD {fold+1} ---")
        
        # Efficient Subsetting
        train_data = [dataset[i] for i in train_idx if dataset[i] is not None]
        val_data = [dataset[i] for i in val_idx if dataset[i] is not None]
        
        train_loader = DataLoader(train_data, batch_size=GNN_BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=GNN_BATCH_SIZE)
        
        # Initialize Architecture (He initialization happens internally now)
        model = CausalBrainGNN(num_node_features=GNN_IN_CHANNELS, hidden_channels=64).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=GNN_LEARNING_RATE, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=GNN_EPOCHS)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        best_auc = 0.0
        patience = 20
        no_improve = 0
        
        for epoch in range(1, GNN_EPOCHS + 1):
            loss = train_one_epoch(model, train_loader, optimizer, criterion)
            scheduler.step()
            
            # Evaluate every epoch for high-resolution tracking
            metrics = evaluate(model, val_loader)
            
            if metrics['auc'] > best_auc:
                best_auc = metrics['auc']
                no_improve = 0
                torch.save(model.state_dict(), CHECKPOINT_DIR / f"best_model_fold{fold}.pt")
            else:
                no_improve += 1
            
            if epoch % 10 == 0:
                logger.info(f"Ep {epoch:03d} | Loss: {loss:.4f} | Val AUC: {metrics['auc']:.4f} | F1: {metrics['f1']:.4f}")
            
            if no_improve >= patience:
                logger.info(f"Early stop at epoch {epoch}")
                break
        
        fold_aucs.append(best_auc)
        logger.info(f"Fold {fold+1} Finished. Best AUC: {best_auc:.4f}")

    logger.info(f"\n{'='*30}\nFINAL CV RESULTS\n{'='*30}")
    logger.info(f"Mean AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    logger.info(f"Fold AUCs: {fold_aucs}")

if __name__ == "__main__":
    run_kfold_training()