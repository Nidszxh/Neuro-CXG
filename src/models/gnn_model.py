import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# --- CONFIG ---
PROJECT_ROOT = Path("./data")
CHECKPOINT_DIR = Path("./models/checkpoints")
K_FOLDS = 5
BATCH_SIZE = 32
LR = 0.001
WEIGHT_DECAY = 1e-3  # Increased for Q1-level regularization
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
    from graph_factory import ABIDECausalDataset
    from causal_gnn import CausalBrainGNN
    
    full_dataset = ABIDECausalDataset(split='train')
    
    # Stratification targets
    labels = [int(data.y) for data in full_dataset]
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    
    overall_metrics = []

    print(f"🚀 Training Causal-GNN on {DEVICE}...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n--- 📂 FOLD {fold+1}/{K_FOLDS} ---")
        
        # Subset indices
        train_loader = DataLoader([full_dataset[i] for i in train_idx], batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader([full_dataset[i] for i in val_idx], batch_size=BATCH_SIZE)
        
        model = CausalBrainGNN(num_node_features=full_dataset[0].x.shape[1], hidden_channels=64).to(DEVICE)
        
        # Optimizer with Cosine Annealing (Standard for high-impact deep learning)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        
        # Label Smoothing helps with the 'noisy' labels in neuroimaging
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        best_fold_auc = 0
        
        for epoch in range(1, EPOCHS + 1):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, scheduler)
            
            if epoch % 10 == 0 or epoch == 1:
                metrics = evaluate(model, val_loader)
                print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val AUC: {metrics['auc']:.4f} | F1: {metrics['f1']:.4f}")
                
                # Save based on AUC (Better indicator of diagnostic potential than Accuracy)
                if metrics['auc'] > best_fold_auc:
                    best_fold_auc = metrics['auc']
                    torch.save(model.state_dict(), CHECKPOINT_DIR / f"best_model_fold{fold}.pt")
        
        overall_metrics.append(best_fold_auc)

    print(f"\n" + "="*30)
    print(f"MEAN CV AUC: {np.mean(overall_metrics):.4f} ± {np.std(overall_metrics):.4f}")
    print("="*30)

if __name__ == "__main__":
    run_kfold_training()