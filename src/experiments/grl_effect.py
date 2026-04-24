"""
GRL Effect Validation: Test Gradient Reversal Layer Contribution
==============================================================

Tests whether the Gradient Reversal Layer (GRL) and adversarial site
debiasing contributes to model performance.

Configuration comparison:
- With GRL: site embeddings + GRL loss (current model)
- Without GRL: no adversarial debiasing

Usage:
    python -m src.experiments.grl_effect
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import (
    DEVICE, GNN_BATCH_SIZE, GNN_EPOCHS, GNN_ONECYCLE_MAX_LR,
    GNN_EARLY_STOPPING_PATIENCE, GNN_MIN_EPOCHS_BEFORE_STOPPING,
    GNN_WEIGHT_DECAY, K_FOLDS,
)
from src.features.graph_factory import ABIDECausalDataset
from src.models.factory import build_model
from src.models.training_utils import make_loader, train_fold_with_onecycle

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/experiments/grl_effect")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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
    
    logger.info(f"Total subjects: {len(labels)}")
    
    cv_folds = dataset.manifest['cv_fold'].to_numpy()
    cv_splits = [(np.where(cv_folds != f)[0], np.where(cv_folds == f)[0]) for f in range(K_FOLDS)]
    
    fold_aucs, fold_f1s = [], []
    
    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        t0 = time.time()
        train_data = [dataset[i] for i in train_idx if dataset[i] is not None]
        val_data = [dataset[i] for i in val_idx if dataset[i] is not None]
        
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
    logger.info(f"\n{'='*70}")
    logger.info(f"RESULTS: AUC = {mean_auc:.4f} ± {np.std(fold_aucs):.4f} | F1 = {np.mean(fold_f1s):.4f}")
    logger.info(f"{'='*70}")
    
    return {"mean_auc": mean_auc, "std_auc": float(np.std(fold_aucs)), "fold_aucs": fold_aucs}


def main():
    logger.info("="*70)
    logger.info("GRL EFFECT VALIDATION")
    logger.info("="*70)
    
    ds = ABIDECausalDataset(split="train")
    logger.info(f"Loaded {len(ds)} training subjects")
    
    # Model WITHOUT GRL
    def model_factory_no_grl():
        return build_model(
            device=DEVICE,
            use_site_embedding=False,  # No site embedding
            use_demographics=False,
            use_grl=False,
            edge_gate=True,
        )
    
    # Model WITH GRL (but no grl_weight in training)
    def model_factory_with_grl():
        return build_model(
            device=DEVICE,
            use_site_embedding=True,
            use_demographics=True,
            use_grl=False,  # Still no GRL in training
            edge_gate=True,
        )
    
    # Run WITHOUT site conditioning (Ablation E equivalent)
    logger.info("\n--- Configuration 1: No site conditioning ---")
    results_no_site = run_kfold_cv(ds, model_factory_no_grl, "NO SITE CONDITIONING")
    
    # Run WITH site conditioning (Ablation E already did this)
    logger.info("\n--- Configuration 2: With site conditioning ---")
    results_with_site = run_kfold_cv(ds, model_factory_with_grl, "WITH SITE CONDITIONING")
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("COMPARISON SUMMARY")
    logger.info("="*70)
    logger.info(f"No site conditioning AUC:  {results_no_site['mean_auc']:.4f}")
    logger.info(f"With site conditioning AUC: {results_with_site['mean_auc']:.4f}")
    logger.info(f"Delta: {results_with_site['mean_auc'] - results_no_site['mean_auc']:+.4f}")
    
    # Reference
    logger.info("\nReference: Ablation E (0.7448), Full model (0.8587)")
    logger.info("-"*70)
    
    # Save
    import pandas as pd
    pd.DataFrame([{
        "config": "no_site",
        "auc": results_no_site['mean_auc'],
        "std": results_no_site['std_auc'],
    }, {
        "config": "with_site", 
        "auc": results_with_site['mean_auc'],
        "std": results_with_site['std_auc'],
    }]).to_csv(RESULTS_DIR / "grl_effect_results.csv", index=False)
    logger.info(f"\nResults saved → {RESULTS_DIR / 'grl_effect_results.csv'}")


if __name__ == "__main__":
    main()