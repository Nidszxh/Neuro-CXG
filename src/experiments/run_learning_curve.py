"""
Learning Curve Experiment: Performance vs Training Set Size
=============================================================

Runs 5-fold CV at different training set subsamples to assess sample efficiency.
Used for supplementary figure in paper.

Usage:
    python -m src.experiments.run_learning_curve --subsamples 20 40 60 80 100
    python -m src.experiments.run_learning_curve --dry-run
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import (
    CHECKPOINT_DIR,
    DEVICE,
    GNN_BATCH_SIZE,
    GNN_DROPOUT,
    GNN_EPOCHS,
    GNN_GRL_ALPHA,
    GNN_GRL_ALPHA_MAX,
    GNN_HIDDEN_CHANNELS,
    GNN_IN_CHANNELS,
    GNN_NUM_LAYERS,
    GNN_NUM_HEADS,
    GNN_ONECYCLE_MAX_LR,
    GNN_EARLY_STOPPING_PATIENCE,
    GNN_MIN_EPOCHS_BEFORE_STOPPING,
    GNN_POOLING,
    GNN_SITE_LOSS_WEIGHT,
    GNN_WEIGHT_DECAY,
    K_FOLDS,
    NUM_LOBES,
    RESULTS_ABLATIONS_DIR,
    USE_FOCAL_LOSS,
    USE_CLASS_WEIGHTS,
    FOCAL_LOSS_ALPHA,
    FOCAL_LOSS_GAMMA,
)
from src.features.graph_factory import ABIDECausalDataset
from src.models.losses import FocalLoss
from src.models.factory import build_model
from src.models.training_utils import make_loader, train_fold_with_onecycle

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_kfold_subsample(
    dataset: ABIDECausalDataset,
    sample_fraction: float,
    folds: int = K_FOLDS,
) -> Dict:
    """
    Run 5-fold CV on a subsample of the dataset.
    sample_fraction: 0.0 to 1.0 (percentage of training data to use)
    """
    logger.info(f"\n{'━'*70}")
    logger.info(f"LEARNING CURVE: {sample_fraction*100:.0f}% of training data ({sample_fraction:.2f})")
    logger.info(f"{'━'*70}")

    manifest = dataset.manifest
    if "cv_fold" not in manifest.columns:
        raise ValueError("Requires cv_fold splits from split.py")

    cv_folds = manifest["cv_fold"].to_numpy()
    n_total = len(manifest)

    cv_splits = []
    for fold_id in range(folds):
        train_idx = np.where(cv_folds != fold_id)[0]
        val_idx = np.where(cv_folds == fold_id)[0]

        n_train = len(train_idx)
        n_subsample = max(int(n_train * sample_fraction), 20)
        n_subsample = min(n_subsample, n_train)

        np.random.seed(42 + fold_id)
        subsample_idx = np.random.choice(train_idx, n_subsample, replace=False)
        cv_splits.append((subsample_idx, val_idx))

    labels = []
    for i in range(len(dataset)):
        d = dataset.get(i)
        if d is not None:
            labels.append(int(d.y.item()))

    n_ctrl = labels.count(0)
    n_asd = labels.count(1)
    logger.info(f"  Original: {n_total} subjects | Subsample: ~{int(n_total * sample_fraction * 0.71)} train")

    fold_aucs: List[float] = []
    fold_f1s: List[float] = []

    class_weight_tensor = None
    if USE_CLASS_WEIGHTS:
        labels_arr = np.array(labels)
        n_control = max(int((labels_arr == 0).sum()), 1)
        n_asd = max(int((labels_arr == 1).sum()), 1)
        total = max(len(labels_arr), 1)
        class_weight_tensor = torch.tensor(
            [total / (2 * n_control), total / (2 * n_asd)],
            dtype=torch.float32,
            device=DEVICE,
        )

    if USE_FOCAL_LOSS:
        pos_weight = None
        if USE_CLASS_WEIGHTS:
            labels_arr = np.array(labels)
            n_control = max(int((labels_arr == 0).sum()), 1)
            n_asd = max(int((labels_arr == 1).sum()), 1)
            pos_weight = float(n_control / n_asd)
        criterion = FocalLoss(alpha=FOCAL_LOSS_ALPHA, gamma=FOCAL_LOSS_GAMMA, pos_weight=pos_weight)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)

    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        t0 = time.time()

        train_data = [dataset[i] for i in train_idx if dataset[i] is not None]
        val_data = [dataset[i] for i in val_idx if dataset[i] is not None]

        if not train_data or not val_data:
            logger.warning(f"  Fold {fold}: insufficient data, skipping")
            continue

        train_loader = make_loader(train_data, batch_size=GNN_BATCH_SIZE, shuffle=True)
        val_loader = make_loader(val_data, batch_size=GNN_BATCH_SIZE)

        model_factory = lambda: build_model(
            device=DEVICE,
            use_site_embedding=True,
            use_demographics=True,
            use_grl=True,
            grl_alpha=GNN_GRL_ALPHA,
            edge_gate=True,
        )
        model = model_factory().to(DEVICE)

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
            use_grl=True,
            grl_weight=GNN_SITE_LOSS_WEIGHT if True else 0.0,
            fold=fold,
            weight_decay=GNN_WEIGHT_DECAY,
            grl_alpha_max=GNN_GRL_ALPHA_MAX,
        )

        auc = best_metrics["auc"]
        f1 = best_metrics["f1"]
        fold_aucs.append(auc)
        fold_f1s.append(f1)

        logger.info(
            f"  Fold {fold + 1}/{folds}: AUC={auc:.4f}  F1={f1:.4f}  "
            f"(elapsed {time.time()-t0:.0f}s)"
        )

    if not fold_aucs:
        logger.error(f"  All folds failed for {sample_fraction*100:.0f}%")
        return {}

    mean_auc = float(np.mean(fold_aucs))
    std_auc = float(np.std(fold_aucs))
    mean_f1 = float(np.mean(fold_f1s))

    logger.info(f"\n  ╔══ RESULT: {sample_fraction*100:.0f}% → AUC = {mean_auc:.4f} ± {std_auc:.4f} ══╗")

    return {
        "sample_fraction": sample_fraction,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "mean_f1": mean_f1,
        "fold_aucs": fold_aucs,
    }


def main():
    parser = argparse.ArgumentParser(description="Learning curve experiment")
    parser.add_argument(
        "--subsamples", nargs="+", type=float, default=[0.2, 0.4, 0.6, 0.8, 1.0],
        help="Training set subsamples (default: 20 40 60 80 100 percent)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print plan without training")
    args = parser.parse_args()

    logger.info("\n" + "=" * 70)
    logger.info("NEURO-CXG: LEARNING CURVE EXPERIMENT")
    logger.info("=" * 70)
    logger.info(f"  Subsamples: {[int(s*100) for s in args.subsamples]}%")
    logger.info(f"  Device: {DEVICE}")

    if args.dry_run:
        for s in args.subsamples:
            logger.info(f"  [DRY-RUN] Would run {s*100:.0f}% subsample")
        return

    dataset = ABIDECausalDataset(split="train")
    logger.info(f"  Loaded {len(dataset)} training subjects")

    results: Dict[str, Dict] = {}
    for sample_frac in args.subsamples:
        result = run_kfold_subsample(dataset, sample_frac)
        if result:
            results[f"{int(sample_frac*100)}pct"] = result

    logger.info("\n" + "=" * 70)
    logger.info("LEARNING CURVE SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Subsample':<12} {'AUC ± std':>18}")
    logger.info("-" * 32)
    for key, res in results.items():
        auc = res["mean_auc"]
        std = res["std_auc"]
        logger.info(f"  {key:<10} {auc:.4f} ± {std:.4f}")
    logger.info("-" * 32)
    logger.info(f"  {'100% (full)':<10} {results.get('100pct', {}).get('mean_auc', 'N/A'):.4f}")
    logger.info("=" * 70)

    import pandas as pd
    import json
    out_csv = RESULTS_ABLATIONS_DIR / "learning_curve.csv"
    pd.DataFrame([
        {
            "subsample": k,
            "mean_auc": v["mean_auc"],
            "std_auc": v["std_auc"],
            "mean_f1": v["mean_f1"],
        }
        for k, v in results.items()
    ]).to_csv(out_csv, index=False)
    logger.info(f"\n  Results saved → {out_csv}")

    out_json = RESULTS_ABLATIONS_DIR / "learning_curve.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"  JSON saved → {out_json}")


if __name__ == "__main__":
    main()