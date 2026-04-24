"""
Shuffled Edges Validation: Test Graph Structure Value
================================================

Tests whether the graph edge structure contributes to classification
or if the model relies purely on node features.

If shuffled edges achieve similar AUC to real edges → graph structure not discriminative
If shuffled edges significantly worse → graph edges carry predictive signal

Usage:
    python -m src.experiments.shuffled_edges
    python -m src.experiments.shuffled_edges --baseline  # Compare with real edges
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score
from torch_geometric.data import Data, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import (
    DEVICE, GNN_BATCH_SIZE, GNN_EPOCHS, GNN_ONECYCLE_MAX_LR,
    GNN_EARLY_STOPPING_PATIENCE, GNN_MIN_EPOCHS_BEFORE_STOPPING,
    GNN_WEIGHT_DECAY, K_FOLDS,
    NUM_LOBES, GNN_IN_CHANNELS,
)
from src.features.graph_factory import ABIDECausalDataset
from src.models.factory import build_model
from src.models.training_utils import make_loader, train_fold_with_onecycle

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/experiments/shuffled_edges")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class ShuffledEdgeDataset(Dataset):
    """
    Wrapper that randomizes edge indices while preserving node features.
    
    Tests whether the model exploits:
    - A) The actual edge topology (discriminative graph structure)
    - B) Node features only (graph is just a scaffold)
    
    If shuffled ≈ real → edges don't matter (node features drive prediction)
    If shuffled < real → edges carry discriminative signal
    """
    
    def __init__(self, base_dataset: ABIDECausalDataset, seed: int = 42):
        super().__init__(None, None, None)
        self.ds = base_dataset
        self.rng = np.random.default_rng(seed)
        self.subject_ids = base_dataset.subject_ids
        self.manifest = base_dataset.manifest
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        sample = self.ds[idx]
        if sample is None:
            return None
            
        # Clone to avoid modifying original
        new_sample = sample.clone()
        
        # Randomize edge index (shuffle edge order)
        num_edges = new_sample.edge_index.shape[1]
        if num_edges > 0:
            perm = self.rng.permutation(num_edges)
            new_sample.edge_index = new_sample.edge_index[:, perm]
            if new_sample.edge_attr is not None:
                new_sample.edge_attr = new_sample.edge_attr[perm]
                
        return new_sample
    
    def get(self, idx):
        return self[idx]


def run_kfold_cv(
    dataset: Dataset,
    model_factory,
    experiment_name: str,
    folds: int = K_FOLDS,
) -> dict:
    """Run 5-fold CV and return metrics."""
    logger.info(f"\n{'='*70}")
    logger.info(f"{experiment_name}")
    logger.info(f"{'='*70}")
    
    # Collect labels
    labels = []
    for i in range(len(dataset)):
        d = dataset.get(i) if hasattr(dataset, 'get') else dataset[i]
        if d is not None:
            labels.append(int(d.y.item()))
    
    n_ctrl = labels.count(0)
    n_asd = labels.count(1)
    logger.info(f"Total subjects: {len(labels)} (Control={n_ctrl}, ASD={n_asd})")
    
    # Get cv_folds from manifest
    if hasattr(dataset, 'manifest') and hasattr(dataset.manifest, 'columns'):
        if 'cv_fold' not in dataset.manifest.columns:
            raise ValueError("cv_fold column required in manifest")
        cv_folds = dataset.manifest['cv_fold'].to_numpy()
    else:
        raise ValueError("Dataset manifest required")
    
    cv_splits = []
    for fold_id in range(folds):
        train_idx = np.where(cv_folds != fold_id)[0]
        val_idx = np.where(cv_folds == fold_id)[0]
        cv_splits.append((train_idx, val_idx))
    
    fold_aucs = []
    fold_f1s = []
    
    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        t0 = time.time()
        
        train_data = [dataset[i] for i in train_idx if dataset[i] is not None]
        val_data = [dataset[i] for i in val_idx if dataset[i] is not None]
        
        if not train_data or not val_data:
            logger.warning(f"Fold {fold}: insufficient data, skipping")
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
            use_grl=False,  # No GRL for fair comparison
            grl_weight=0.0,
            fold=fold,
            weight_decay=GNN_WEIGHT_DECAY,
        )
        
        auc = best_metrics["auc"]
        f1 = best_metrics["f1"]
        fold_aucs.append(auc)
        fold_f1s.append(f1)
        
        logger.info(
            f"Fold {fold+1}/{folds}: AUC={auc:.4f} F1={f1:.4f} "
            f"(elapsed {time.time()-t0:.0f}s)"
        )
    
    mean_auc = float(np.mean(fold_aucs))
    std_auc = float(np.std(fold_aucs))
    mean_f1 = float(np.mean(fold_f1s))
    
    logger.info(f"\n{'='*70}")
    logger.info(f"RESULTS: AUC = {mean_auc:.4f} ± {std_auc:.4f} | F1 = {mean_f1:.4f}")
    logger.info(f"{'='*70}")
    
    return {
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "mean_f1": mean_f1,
        "fold_aucs": fold_aucs,
    }


def main():
    parser = argparse.ArgumentParser(description="Shuffled edges validation")
    parser.add_argument("--baseline", action="store_true", 
                    help="Also run baseline (real edges) for comparison")
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("SHUFFLED EDGES VALIDATION")
    logger.info("="*70)
    
    # Load base dataset
    logger.info("\nLoading base dataset...")
    base_ds = ABIDECausalDataset(split="train")
    logger.info(f"Loaded {len(base_ds)} training subjects")
    
    # Create shuffled dataset
    shuffled_ds = ShuffledEdgeDataset(base_ds, seed=42)
    
    def model_factory():
        return build_model(
            device=DEVICE,
            use_site_embedding=True,
            use_demographics=True,
            use_grl=False,
            edge_gate=True,
        )
    
    # Run shuffled edges experiment
    shuffled_results = run_kfold_cv(
        shuffled_ds, 
        model_factory,
        "SHUFFLED EDGES",
    )
    
    baseline_auc = None
    if args.baseline:
        # Run baseline (real edges) for comparison
        logger.info("\n" + "="*70)
        logger.info("Running baseline (real edges) for comparison...")
        baseline_results = run_kfold_cv(
            base_ds,
            model_factory,
            "BASELINE (REAL EDGES)",
        )
        baseline_auc = baseline_results["mean_auc"]
    
    # Print comparison
    logger.info("\n" + "="*70)
    logger.info("COMPARISON SUMMARY")
    logger.info("="*70)
    shuffled_auc = shuffled_results["mean_auc"]
    
    if baseline_auc is not None:
        delta = shuffled_auc - baseline_auc
        sign = "+" if delta >= 0 else ""
        logger.info(f"Shuffled edges AUC:  {shuffled_auc:.4f}")
        logger.info(f"Real edges AUC:     {baseline_auc:.4f}")
        logger.info(f"Delta:            {sign}{delta:.4f}")
    else:
        # Reference: full model test AUC
        ref_auc = 0.8587
        delta = shuffled_auc - ref_auc
        sign = "+" if delta >= 0 else ""
        logger.info(f"Shuffled edges AUC:  {shuffled_auc:.4f}")
        logger.info(f"Reference AUC:    {ref_auc:.4f} (Ablation D)")
        logger.info(f"Delta:            {sign}{delta:.4f}")
    
    # Interpretation
    logger.info("\n" + "-"*70)
    logger.info("INTERPRETATION:")
    if delta < -0.05:
        logger.info("Graph edges carry discriminative signal (shuffled < real)")
    elif delta > 0.05:
        logger.info("Surprising: shuffled > real (possibly overfitting to edges)")
    else:
        logger.info("Graph structure has minimal discriminative value")
    logger.info("-"*70)
    
    # Save results
    import pandas as pd
    results_df = pd.DataFrame([{
        "experiment": "shuffled_edges",
        "auc": shuffled_auc,
        "std": shuffled_results["std_auc"],
        "f1": shuffled_results["mean_f1"],
        "fold_aucs": str(shuffled_results["fold_aucs"]),
    }])
    results_df.to_csv(RESULTS_DIR / "shuffled_edges_results.csv", index=False)
    logger.info(f"\nResults saved → {RESULTS_DIR / 'shuffled_edges_results.csv'}")


if __name__ == "__main__":
    main()