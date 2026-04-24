"""
Baseline Logistic Regression: Publication Requirement
=============================================

Publication requires baseline comparison with simple logistic regression.
This validates that the GNN outperforms traditional methods.

Usage:
    python -m src.experiments.baseline_lr
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import (
    NODE_ATTRIBUTES_HARMONIZED, MASTER_MANIFEST, K_FOLDS, NUM_LOBES, 
    GNN_IN_CHANNELS,
)
from src.features.graph_factory import _load_csv_cached

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/experiments/baseline_lr")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    logger.info("="*70)
    logger.info("BASELINE LOGISTIC REGRESSION")
    logger.info("="*70)
    
    # Load data
    logger.info("\nLoading features...")
    features = _load_csv_cached(NODE_ATTRIBUTES_HARMONIZED, index_col='subject_id')
    manifest = _load_csv_cached(MASTER_MANIFEST)
    
    # Filter to train split
    train_subs = set(manifest[manifest['split'] == 'train']['subject_id'].astype(str))
    features = features.loc[features.index.isin(train_subs)]
    
    # Get features (all lobe × all feature types)
    feature_cols = [c for c in features.columns if c != 'subject_id']
    X = features[feature_cols].values
    
    # Get labels
    manifest = manifest.set_index('subject_id')
    manifest = manifest.loc[features.index]
    y = manifest['DX_GROUP'].values
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Add site as feature (one-hot encoding)
    sites = manifest['SITE_ID'].astype(str).values
    unique_sites = np.unique(sites)
    site_to_idx = {s: i for i, s in enumerate(unique_sites)}
    site_features = np.zeros((len(sites), len(unique_sites)))
    for i, s in enumerate(sites):
        site_features[i, site_to_idx[s]] = 1
    
    X_full = np.hstack([X, site_features])
    
    logger.info(f"Feature matrix: {X_full.shape}")
    logger.info(f"Labels: {y.shape} (ASD={y.sum()}, Control={len(y)-y.sum()})")
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_aucs = []
    fold_f1s = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_full, y)):
        X_train, X_val = X_full[train_idx], X_full[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train LR with regularization
        clf = LogisticRegression(
            max_iter=1000, 
            C=0.1,  # L2 regularization
            class_weight='balanced',
            solver='lbfgs',
            random_state=42,
        )
        clf.fit(X_train, y_train)
        
        # Predict
        y_prob = clf.predict_proba(X_val)[:, 1]
        y_pred = clf.predict(X_val)
        
        auc = roc_auc_score(y_val, y_prob)
        f1 = f1_score(y_val, y_pred)
        
        fold_aucs.append(auc)
        fold_f1s.append(f1)
        
        logger.info(f"Fold {fold+1}/{K_FOLDS}: AUC={auc:.4f} F1={f1:.4f}")
    
    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    mean_f1 = np.mean(fold_f1s)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"RESULTS: AUC = {mean_auc:.4f} ± {std_auc:.4f} | F1 = {mean_f1:.4f}")
    logger.info(f"{'='*70}")
    
    # Comparison
    logger.info("\nCOMPARISON:")
    logger.info(f"  LR baseline:      {mean_auc:.4f}")
    logger.info(f"  FlatMLP (Abl A):  0.7267")
    logger.info(f"  GNN (full):       0.8587")
    logger.info(f"  GNN vs LR:        +{0.8587 - mean_auc:.4f}")
    logger.info("-"*70)
    
    # Save
    pd.DataFrame([{
        "model": "LogisticRegression",
        "auc": mean_auc,
        "std": std_auc,
        "f1": mean_f1,
        "fold_aucs": str(fold_aucs),
    }]).to_csv(RESULTS_DIR / "lr_baseline_results.csv", index=False)
    
    logger.info(f"\nResults saved → {RESULTS_DIR / 'lr_baseline_results.csv'}")


if __name__ == "__main__":
    main()