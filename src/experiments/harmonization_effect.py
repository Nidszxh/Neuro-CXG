"""
Harmonization Effect Validation: Test Signal Preservation
=====================================================

Tests whether harmonization preserves or removes useful signal.

Comparison:
- Harmonized features (current)
- Raw temporal features

Usage:
    python -m src.experiments.harmonization_effect
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
    NODE_ATTRIBUTES_TEMPORAL, NODE_ATTRIBUTES_HARMONIZED,
    MASTER_MANIFEST, K_FOLDS,
)
from src.features.graph_factory import _load_csv_cached

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/experiments/harmonization_effect")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_lr_cv(features_df, manifest_df, name):
    """Run LR baseline on features."""
    # Filter to train
    train_subs = set(manifest_df[manifest_df['split'] == 'train']['subject_id'].astype(str))
    features_df = features_df[features_df.index.isin(train_subs)]
    
    feature_cols = [c for c in features_df.columns if c != 'subject_id']
    X = features_df[feature_cols].values
    
    manifest_df = manifest_df.set_index('subject_id').loc[features_df.index]
    y = manifest_df['DX_GROUP'].values
    
    # Handle missing
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # CV
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_aucs, fold_f1s = [], []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        clf = LogisticRegression(max_iter=1000, C=0.1, class_weight='balanced', random_state=42)
        clf.fit(X_train, y_train)
        
        y_prob = clf.predict_proba(X_val)[:, 1]
        y_pred = clf.predict(X_val)
        
        fold_aucs.append(roc_auc_score(y_val, y_prob))
        fold_f1s.append(f1_score(y_val, y_pred))
    
    mean_auc = np.mean(fold_aucs)
    logger.info(f"{name}: AUC = {mean_auc:.4f} ± {np.std(fold_aucs):.4f}")
    
    return mean_auc, np.std(fold_aucs)


def main():
    logger.info("="*70)
    logger.info("HARMONIZATION EFFECT VALIDATION")
    logger.info("="*70)
    
    # Load features
    logger.info("\nLoading features...")
    harmonized = _load_csv_cached(NODE_ATTRIBUTES_HARMONIZED, index_col='subject_id')
    raw = _load_csv_cached(NODE_ATTRIBUTES_TEMPORAL, index_col='subject_id')
    manifest = _load_csv_cached(MASTER_MANIFEST)
    
    logger.info(f"Harmonized: {harmonized.shape}")
    logger.info(f"Raw: {raw.shape}")
    
    # Run comparison
    logger.info("\n--- Raw Features ---")
    raw_auc, raw_std = run_lr_cv(raw, manifest, "Raw")
    
    logger.info("\n--- Harmonized Features ---")
    harm_auc, harm_std = run_lr_cv(harmonized, manifest, "Harmonized")
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("COMPARISON")
    logger.info("="*70)
    logger.info(f"Raw AUC:        {raw_auc:.4f}")
    logger.info(f"Harmonized AUC: {harm_auc:.4f}")
    delta = harm_auc - raw_auc
    sign = "+" if delta >= 0 else ""
    logger.info(f"Delta:          {sign}{delta:.4f}")
    
    # Interpretation
    logger.info("\n" + "-"*70)
    if delta > 0.05:
        logger.info("Harmonization IMPROVES signal (removes site noise)")
    elif delta < -0.05:
        logger.info("Harmonization HURTS signal (removes useful variance)")
    else:
        logger.info("Harmonization has minimal effect")
    logger.info("-"*70)
    
    # Also compare to GNN
    logger.info(f"\nReference: GNN (full) = 0.8587")
    logger.info(f"          GNN vs harm LR = +{0.8587 - harm_auc:.4f}")
    
    # Save
    pd.DataFrame([{
        "features": "raw",
        "auc": raw_auc,
        "std": raw_std,
    }, {
        "features": "harmonized",
        "auc": harm_auc,
        "std": harm_std,
    }]).to_csv(RESULTS_DIR / "harmonization_effect_results.csv", index=False)
    
    logger.info(f"\nResults saved → {RESULTS_DIR / 'harmonization_effect_results.csv'}")


if __name__ == "__main__":
    main()