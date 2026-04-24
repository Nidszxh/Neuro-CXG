"""
Brainstem Removal Ablation
=========================

Tests whether Brainstem features are noisy/harmful.

If removing Brainstem improves performance → features are noisy
If no change → features are neutral

Quick run using existing features (no YOLO needed).
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import (
    DEVICE, GNN_BATCH_SIZE, GNN_EPOCHS, GNN_ONECYCLE_MAX_LR,
    GNN_EARLY_STOPPING_PATIENCE, GNN_MIN_EPOCHS_BEFORE_STOPPING,
    GNN_WEIGHT_DECAY, K_FOLDS, GNN_IN_CHANNELS, NUM_LOBES,
    LOBE_NAMES,
)
from src.features.graph_factory import ABIDECausalDataset
from src.models.factory import build_model
from src.models.training_utils import make_loader, train_fold_with_onecycle

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/experiments/brainstem_ablation")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Brainstem is lobe index 11
BRAINSTEM_IDX = 11
BRAINSTEM_NAME = LOBE_NAMES[BRAINSTEM_IDX]


class NoBrainstemDataset(Dataset):
    """Wrapper that zeros out Brainstem features."""
    
    def __init__(self, base_dataset: ABIDECausalDataset):
        super().__init__(None, None, None)
        self.ds = base_dataset
        self.subject_ids = base_dataset.subject_ids
        self.manifest = base_dataset.manifest
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        sample = self.ds[idx]
        if sample is None:
            return None
        
        # Clone to avoid modifying original
        sample = sample.clone()
        
        # Zero out Brainstem features (features 11*24 to 12*24 in flat, or rows 11 in node features)
        # Each lobe has 24 features (8 temporal + 12 freq + 2 internal + 4 spatial)
        # Brainstem is the 12th lobe (index 11)
        n_feats_per_lobe = GNN_IN_CHANNELS // NUM_LOBES  # 24/12 = 2? No wait...
        
        # Actually simpler: each node has GNN_IN_CHANNELS features
        # Brainstem is node index 11
        # Zero the entire Brainstem node features
        if sample.x.shape[0] > BRAINSTEM_IDX:
            sample.x[BRAINSTEM_IDX, :] = 0
            
        return sample
    
    def get(self, idx):
        return self[idx]


def run_kfold_cv(dataset, model_factory, experiment_name):
    """Run 5-fold CV."""
    logger.info(f"\n{'='*70}")
    logger.info(f"{experiment_name}")
    logger.info(f"{'='*70}")
    
    labels = []
    for i in range(len(dataset)):
        d = dataset.get(i) if hasattr(dataset, 'get') else dataset[i]
        if d is not None:
            labels.append(int(d.y.item()))
    
    logger.info(f"Total subjects: {len(labels)} (Control={labels.count(0)}, ASD={labels.count(1)})")
    
    cv_folds = dataset.manifest['cv_fold'].to_numpy()
    cv_splits = [(np.where(cv_folds != f)[0], np.where(cv_folds == f)[0]) for f in range(K_FOLDS)]
    
    fold_aucs, fold_f1s = [], []
    
    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        t0 = time.time()
        train_data = [dataset[i] for i in train_idx if dataset[i] is not None]
        val_data = [dataset[i] for i in val_idx if dataset[i] is not None]
        
        if not train_data or not val_data:
            continue
            
        train_loader = make_loader(train_data, batch_size=GNN_BATCH_SIZE, shuffle=True)
        val_loader = make_loader(val_data, batch_size=GNN_BATCH_SIZE)
        
        model = model_factory().to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        
        best_state, best_metrics, _ = train_fold_with_onecycle(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            device=DEVICE,
            epochs=GNN_EPOCHS,
            max_lr=GNN_ONECYCLE_MAX_LR,
            patience=GNN_EARLY_STOPPING_PATIENCE,
            min_epochs_before_stopping=GNN_MIN_EPOCHS_BEFORE_STOPPING,
            use_grl=False,
            grl_weight=0.0,
            fold=fold,
            weight_decay=GNN_WEIGHT_DECAY,
        )
        
        fold_aucs.append(best_metrics["auc"])
        fold_f1s.append(best_metrics["f1"])
        
        logger.info(f"Fold {fold+1}/{K_FOLDS}: AUC={best_metrics['auc']:.4f} F1={best_metrics['f1']:.4f} ({time.time()-t0:.0f}s)")
    
    mean_auc = float(np.mean(fold_aucs))
    std_auc = float(np.std(fold_aucs))
    mean_f1 = float(np.mean(fold_f1s))
    
    logger.info(f"\n{'='*70}")
    logger.info(f"RESULTS: AUC = {mean_auc:.4f} ± {std_auc:.4f} | F1 = {mean_f1:.4f}")
    logger.info(f"{'='*70}")
    
    return {"mean_auc": mean_auc, "std_auc": std_auc, "fold_aucs": fold_aucs, "mean_f1": mean_f1}


def main():
    parser = argparse.ArgumentParser(description="Brainstem removal ablation")
    parser.add_argument("--baseline", action="store_true", 
                       help="Also run baseline with Brainstem for comparison")
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("BRAINSTEM REMOVAL ABLATION")
    logger.info("="*70)
    logger.info(f"Removing lobe: {BRAINSTEM_NAME} (index {BRAINSTEM_IDX})")
    
    # Load base dataset
    logger.info("\nLoading base dataset...")
    base_ds = ABIDECausalDataset(split="train")
    logger.info(f"Loaded {len(base_ds)} subjects")
    
    # Create no-brainstem dataset
    no_brainstem_ds = NoBrainstemDataset(base_ds)
    
    def model_factory():
        return build_model(
            device=DEVICE,
            use_site_embedding=True,
            use_demographics=True,
            use_grl=False,
            edge_gate=True,
        )
    
    # Run without Brainstem
    logger.info("\n--- Running without Brainstem ---")
    no_brainstem_results = run_kfold_cv(
        no_brainstem_ds, 
        model_factory,
        "WITHOUT BRAINSTEM",
    )
    
    baseline_auc = None
    if args.baseline:
        logger.info("\n--- Running baseline (with Brainstem) ---")
        baseline_results = run_kfold_cv(
            base_ds,
            model_factory,
            "BASELINE (WITH BRAINSTEM)",
        )
        baseline_auc = baseline_results["mean_auc"]
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("COMPARISON SUMMARY")
    logger.info("="*70)
    
    no_brain_auc = no_brainstem_results["mean_auc"]
    
    if baseline_auc is not None:
        delta = no_brain_auc - baseline_auc
        sign = "+" if delta >= 0 else ""
        logger.info(f"With Brainstem AUC:     {baseline_auc:.4f}")
        logger.info(f"Without Brainstem AUC: {no_brain_auc:.4f}")
        logger.info(f"Delta:                 {sign}{delta:.4f}")
    else:
        # Reference: full model AUC ~0.86
        ref = 0.8587
        delta = no_brain_auc - ref
        sign = "+" if delta >= 0 else ""
        logger.info(f"Reference AUC (full): {ref:.4f}")
        logger.info(f"Without Brainstem:    {no_brain_auc:.4f}")
        logger.info(f"Delta:                {sign}{delta:.4f}")
    
    # Interpretation
    logger.info("\n" + "-"*70)
    logger.info("INTERPRETATION:")
    if delta > 0.02:
        logger.info("Brainstem is NOISY - removing improves performance")
    elif delta < -0.02:
        logger.info("Brainstem is USEFUL - removing hurts performance")
    else:
        logger.info("Brainstem is NEUTRAL - no significant impact")
    logger.info("-"*70)
    
    # Save results
    import pandas as pd
    results_df = pd.DataFrame([{
        "experiment": "no_brainstem",
        "auc": no_brain_auc,
        "std": no_brainstem_results["std_auc"],
        "f1": no_brainstem_results["mean_f1"],
        "fold_aucs": str(no_brainstem_results["fold_aucs"]),
    }])
    results_df.to_csv(RESULTS_DIR / "brainstem_ablation_results.csv", index=False)
    logger.info(f"\nResults saved → {RESULTS_DIR / 'brainstem_ablation_results.csv'}")


if __name__ == "__main__":
    main()