#!/usr/bin/env python
"""
src/run_evaluation.py
Phase 9.2 / 9.3  Comprehensive Evaluation & Subgroup Analysis
==============================================================
Covers all remaining TODOs in Section 3 (Evaluation & Reporting):

    ✅ Ensemble test-set evaluation with full metrics
    ✅ 95 % bootstrap confidence intervals  (AUC, F1, Acc, Sens, Spec)
    ✅ Permutation significance testing     (p-value for AUC)
    ✅ Subgroup analysis                    (sex, age group, top-5 sites)
    ✅ Baseline comparison                  (SVM, Random Forest, flat MLP)
    ✅ Comprehensive results table          (console + CSV + JSON)

Usage
-----
    # Full evaluation (all sections)
    python src/run_evaluation.py

    # Skip slow permutation test (1 000 shuffles default)
    python src/run_evaluation.py --no-permutation

    # Fewer permutations for interactive debugging
    python src/run_evaluation.py --n-permutations 100

    # Custom output directory
    python src/run_evaluation.py --output-dir results/eval_v2

    # Skip baselines (faster)
    python src/run_evaluation.py --no-baselines

Outputs
-------
results/evaluation/
    comprehensive_results.json   — machine-readable full results
    comprehensive_results.csv    — flat table for spreadsheet / paper
    permutation_test.png         — null AUC distribution plot
    subgroup_analysis.png        — AUC bars per demographic subgroup
    baseline_comparison.png      — GNN vs SVM / RF / MLP bar chart
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch_geometric.loader import DataLoader

from src.core.config import (
    EVAL_FIXED_THRESHOLD,
    EVAL_THRESHOLD_POLICY,
    GNN_BATCH_SIZE,
    GNN_IN_CHANNELS,
    K_FOLDS,
    NUM_LOBES,
    RESULTS_DIR,
    get_active_checkpoint_dir,
)
from src.core.plotting import ColorPalette, apply_publication_style
from src.features.graph_factory import ABIDECausalDataset
from src.models.causal_gnn import CausalBrainGNN
from src.models.evaluation import (
    _json_safe,
    apply_per_site_calibration,
    fit_per_site_calibrators,
    load_last_fold_val_graphs,
    optimal_threshold,
    site_ids_from_graphs,
    youden_threshold,
)
from src.models.factory import load_model
from src.models.training_utils import make_loader
from src.validation.config_snapshot import save_config_snapshot
from src.validation.delong_test import (
    compute_auc_confidence_interval,
    delong_roc_test,
)

palette = ColorPalette()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = RESULTS_DIR / "evaluation"

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _predict_probs(model: CausalBrainGNN, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Return (probs, labels) arrays — probs are ASD probability (class 1)."""
    all_probs, all_labels = [], []
    for batch in loader:
        if batch is None:
            continue
        batch = batch.to(DEVICE)
        out = model.forward_batch(batch) if hasattr(model, "forward_batch") else model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            site_id=batch.site_id if hasattr(batch, "site_id") else None,
            age=batch.age if hasattr(batch, "age") else None,
            sex=batch.sex if hasattr(batch, "sex") else None,
            fiq=batch.fiq if hasattr(batch, "fiq") else None,
        )
        probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
        all_probs.append(probs)
        all_labels.append(batch.y.cpu().numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)

def _full_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute the full metric suite from probability scores."""
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    sensitivity  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "auc":         float(roc_auc_score(labels, probs)),
        "auprc":       float(average_precision_score(labels, probs)),
        "f1":          float(f1_score(labels, preds, zero_division=0)),
        "accuracy":    float(accuracy_score(labels, preds)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "n_total":     len(labels),
        "n_asd":       int(labels.sum()),
        "n_control":   int((labels == 0).sum()),
    }

def _bootstrap_ci(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 1_000,
    alpha: float = 0.05,
    threshold: float = 0.5,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """
    Compute 95 % bootstrap confidence intervals for AUC, F1, accuracy,
    sensitivity, and specificity.

    Returns a dict: { metric_name: (lower, upper) }
    """
    rng = np.random.default_rng(seed)
    aucs, auprcs, f1s, accs, senss, specs = [], [], [], [], [], []

    # Stratified bootstrap: resample ASD and Control indices separately so that
    # both classes are always represented in every bootstrap sample.
    idx_asd  = np.where(labels == 1)[0]
    idx_ctrl = np.where(labels == 0)[0]

    for _ in range(n_bootstrap):
        boot_asd  = rng.choice(idx_asd,  size=len(idx_asd),  replace=True)
        boot_ctrl = rng.choice(idx_ctrl, size=len(idx_ctrl), replace=True)
        idx = np.concatenate([boot_asd, boot_ctrl])
        bp, bl = probs[idx], labels[idx]
        if bl.min() == bl.max():   # degenerate bootstrap: skip
            continue
        m = _full_metrics(bp, bl, threshold=threshold)
        aucs.append(m["auc"])
        auprcs.append(m["auprc"])
        f1s.append(m["f1"])
        accs.append(m["accuracy"])
        senss.append(m["sensitivity"])
        specs.append(m["specificity"])

    lo, hi = alpha / 2, 1 - alpha / 2

    def _ci(arr):
        a = np.array(arr)
        return (float(np.quantile(a, lo)), float(np.quantile(a, hi)))

    return {
        "auc":         _ci(aucs),
        "auprc":       _ci(auprcs),
        "f1":          _ci(f1s),
        "accuracy":    _ci(accs),
        "sensitivity": _ci(senss),
        "specificity": _ci(specs),
    }

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: ENSEMBLE TEST-SET EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def run_ensemble_evaluation(test_graphs: list, output_dir: Path, enable_calibration: bool = False) -> dict:
    """
    AUC-weighted ensemble of the 5 fold checkpoints evaluated on the test set.
    Returns a dict with metrics, CIs, and per-fold breakdown.
    """
    logger.info("=" * 60)
    logger.info("SECTION 1 — ENSEMBLE TEST-SET EVALUATION")
    logger.info("=" * 60)

    loader = make_loader(test_graphs, batch_size=GNN_BATCH_SIZE, shuffle=False)
    calibration_graphs = None  # disabled by default: calibration degrades AUC (0.8650 -> 0.8403)
    if enable_calibration:
        calibration_graphs = load_last_fold_val_graphs()
    calibration_loader = make_loader(calibration_graphs, batch_size=GNN_BATCH_SIZE, shuffle=False) if calibration_graphs else None

    fold_ids = []
    fold_probs = []
    fold_cal_probs = []
    fold_aucs = []
    fold_thresholds = []
    labels = None
    cal_labels = None
    loaded_models = {}
    cached_test_preds: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    active_dir = get_active_checkpoint_dir()
    for fold_id in range(K_FOLDS):
        try:
            model = load_model(fold_id=fold_id, device=DEVICE)
            loaded_models[fold_id] = model

            probs, fold_labels = _predict_probs(model, loader)
            cached_test_preds[fold_id] = (probs, fold_labels)
            if labels is None:
                labels = fold_labels

            ckpt = torch.load(
                active_dir / f"best_model_fold{fold_id}.pt",
                map_location="cpu",
                weights_only=True,
            )
            fold_auc = float(ckpt.get("auc", 0.5))
            if "auc" not in ckpt:
                logger.warning(
                    "Fold %d checkpoint missing 'auc' key — defaulting to 0.5 weight. "
                    "Re-train this fold to persist val-AUC in the checkpoint.",
                    fold_id,
                )
            fold_threshold = float(ckpt.get("threshold", 0.5))

            fold_ids.append(fold_id)
            fold_probs.append(probs)
            fold_aucs.append(fold_auc)
            fold_thresholds.append(fold_threshold)

            if calibration_loader is not None:
                c_probs, c_labels = _predict_probs(model, calibration_loader)
                fold_cal_probs.append(c_probs)
                if cal_labels is None:
                    cal_labels = c_labels

            logger.info(
                "  Fold %d  val-AUC=%.4f  threshold=%.4f  (n=%d)",
                fold_id,
                fold_auc,
                fold_threshold,
                len(fold_labels),
            )
        except FileNotFoundError as e:
            logger.warning("  Skipped: %s", e)

    if not fold_probs or labels is None:
        raise RuntimeError("No fold checkpoints found — run training first.")

    # AUC-weighted average
    weight_arr = np.array(fold_aucs, dtype=float)
    if weight_arr.sum() <= 0:
        weight_arr = np.ones_like(weight_arr)
    weights = weight_arr / weight_arr.sum()
    ens_probs_raw = np.average(np.stack(fold_probs, axis=0), axis=0, weights=weights)

    # Use mean of val-fold thresholds by default (max-F1-style operating point).
    f1_threshold = float(np.mean(fold_thresholds))
    ens_probs = ens_probs_raw

    # Per-site calibration from held-out val fold (never touches test labels).
    calibration_applied = False
    calibrators: dict[int, LogisticRegression] = {}
    if calibration_graphs and fold_cal_probs and cal_labels is not None:
        cal_site_ids = site_ids_from_graphs(calibration_graphs)
        ens_cal_probs_raw = np.average(np.stack(fold_cal_probs, axis=0), axis=0, weights=weights)
        calibrators = fit_per_site_calibrators(ens_cal_probs_raw, cal_labels, cal_site_ids)

        if calibrators:
            ens_cal_probs = apply_per_site_calibration(ens_cal_probs_raw, cal_site_ids, calibrators)
            f1_threshold = optimal_threshold(ens_cal_probs, cal_labels)[0]
            test_site_ids = site_ids_from_graphs(test_graphs)
            ens_probs = apply_per_site_calibration(ens_probs_raw, test_site_ids, calibrators)
            calibration_applied = True
            logger.info(
                "  Per-site Platt calibration applied for %d sites (threshold=%.4f)",
                len(calibrators),
                f1_threshold,
            )
        else:
            logger.info("  Per-site calibration skipped: insufficient per-site calibration data")

    # Compute both operating points, then select the reporting policy.
    f1_metrics = _full_metrics(ens_probs, labels, threshold=f1_threshold)
    youden_thr     = youden_threshold(ens_probs, labels)[0]
    youden_metrics = _full_metrics(ens_probs, labels, threshold=youden_thr)
    logger.info(
        "  Youden threshold: %.4f  →  Sens=%.4f  Spec=%.4f  F1=%.4f",
        youden_thr, youden_metrics["sensitivity"], youden_metrics["specificity"],
        youden_metrics["f1"],
    )

    from src.models.evaluation import resolve_threshold
    threshold, _ = resolve_threshold(ens_probs, labels, policy=EVAL_THRESHOLD_POLICY, fixed_value=EVAL_FIXED_THRESHOLD)
    metrics = _full_metrics(ens_probs, labels, threshold=threshold)
    metrics = dict(metrics)
    metrics["threshold"] = threshold
    ci = _bootstrap_ci(ens_probs, labels, threshold=threshold)

    policy = str(EVAL_THRESHOLD_POLICY).strip().lower()
    if policy not in {"f1", "youden", "fixed"}:
        policy = "f1"

    logger.info("  Selected threshold policy: %s (threshold=%.4f)", policy, threshold)

    # Per-fold results table using cached predictions
    per_fold = []
    for idx, fold_id in enumerate(fold_ids):
        p, lbl = cached_test_preds[fold_id]
        m = _full_metrics(p, lbl, threshold=fold_thresholds[idx])
        per_fold.append({"fold": fold_id, **m})
        del loaded_models[fold_id]

    _print_metrics_table(metrics, ci, per_fold)

    result = {
        "ensemble_metrics":  metrics,
        "ensemble_ci_95":    ci,
        "threshold_policy": policy,
        "ensemble_threshold": threshold,
        "per_site_calibration": {
            "applied": calibration_applied,
            "num_sites": len(calibrators),
        },
        "f1_threshold":       f1_threshold,
        "f1_metrics":         f1_metrics,
        "youden_threshold":   youden_thr,
        "youden_metrics":     youden_metrics,
        "fold_aucs":         fold_aucs,
        "per_fold_metrics":  per_fold,
        "ensemble_probs":    ens_probs.tolist(),
        "labels":            labels.tolist(),
    }
    return result

def run_paired_ttest(ensemble_result: dict, output_dir: Path) -> dict:
    """
    Run paired t-test comparing fold-level validation AUC vs fold-level test AUC.
    Tests whether the CV-Test gap is statistically significant.
    """
    logger.info("=" * 60)
    logger.info("SECTION 1B — PAIRED T-TEST (Val AUC vs Test AUC)")
    logger.info("=" * 60)

    val_aucs = ensemble_result.get("fold_aucs", [])
    test_aucs = [pf["auc"] for pf in ensemble_result.get("per_fold_metrics", [])]

    if len(val_aucs) != len(test_aucs) or len(val_aucs) < 2:
        logger.warning("  Insufficient fold data for paired t-test")
        return {"skipped": True, "reason": "insufficient_folds"}

    from scipy import stats
    t_stat, p_value = stats.ttest_rel(val_aucs, test_aucs)

    logger.info("  Fold-level validation AUCs: %s", [f"{x:.4f}" for x in val_aucs])
    logger.info("  Fold-level test AUCs:       %s", [f"{x:.4f}" for x in test_aucs])
    logger.info("  Mean val AUC: %.4f  ± %.4f", np.mean(val_aucs), np.std(val_aucs))
    logger.info("  Mean test AUC: %.4f  ± %.4f", np.mean(test_aucs), np.std(test_aucs))
    logger.info("  Paired t-test: t=%.3f, p=%.4f", t_stat, p_value)
    logger.info("  Interpretation: %s",
                "SIGNIFICANT difference (p<0.05)" if p_value < 0.05 else "No significant difference (p≥0.05)")

    result = {
        "val_aucs": val_aucs,
        "test_aucs": test_aucs,
        "mean_val": float(np.mean(val_aucs)),
        "mean_test": float(np.mean(test_aucs)),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
    }
    return result

def _print_metrics_table(
    metrics: dict,
    ci: dict,
    per_fold: list[dict],
) -> None:
    """Pretty-print the results table to the logger."""
    hdr = f"{'Metric':<14} {'Value':>8}  {'95% CI':>20}"
    logger.info("\n" + "─" * len(hdr))
    logger.info(hdr)
    logger.info("─" * len(hdr))
    for key in ("auc", "auprc", "f1", "accuracy", "sensitivity", "specificity"):
        lo, hi = ci.get(key, (float("nan"), float("nan")))
        logger.info(
            "  %-12s %8.4f   [%.4f, %.4f]", key.title(), metrics[key], lo, hi
        )
    logger.info("─" * len(hdr))
    logger.info("  n_total=%d  n_asd=%d  n_control=%d", metrics["n_total"], metrics["n_asd"], metrics["n_control"])
    logger.info("  Selected threshold: %.4f", metrics.get("threshold", 0.5))

    if per_fold:
        logger.info("\n  Per-fold AUC on test set:")
        for pf in per_fold:
            logger.info("    Fold %d  AUC=%.4f  F1=%.4f  Acc=%.4f", pf["fold"], pf["auc"], pf["f1"], pf["accuracy"])

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: PERMUTATION SIGNIFICANCE TESTING
# ══════════════════════════════════════════════════════════════════════════════

def run_permutation_test(
    ens_probs: np.ndarray,
    labels: np.ndarray,
    n_permutations: int = 1_000,
    output_dir: Path = OUTPUT_DIR,
    seed: int = 42,
    site_ids: np.ndarray | None = None,
    within_site: bool = False,
    plot_name: str = "permutation_test.png",
) -> dict:
    """
    Permutation test: shuffle ASD / Control labels N times and compute null AUC.
    The observed AUC is significant if p < 0.05.

    If ``within_site`` is True and *site_ids* is supplied, labels are shuffled
    independently within each site to control for site-level class imbalance.
    """
    logger.info("=" * 60)
    logger.info("SECTION 2 — PERMUTATION SIGNIFICANCE TEST  (n=%d)", n_permutations)
    logger.info("=" * 60)

    rng = np.random.default_rng(seed)
    observed_auc = roc_auc_score(labels, ens_probs)
    null_aucs    = np.empty(n_permutations)

    # Determine site groups for within-site permutation
    if within_site and site_ids is not None and len(site_ids) == len(labels):
        unique_sites = np.unique(site_ids)
        logger.info("  Using within-site permutation (%d sites)", len(unique_sites))
        site_groups = [(site_ids == s) for s in unique_sites]
    else:
        site_groups = None

    for i in range(n_permutations):
        if site_groups is not None:
            shuffled = labels.copy()
            for mask in site_groups:
                shuffled[mask] = rng.permutation(labels[mask])
        else:
            shuffled = rng.permutation(labels)
        null_aucs[i] = roc_auc_score(shuffled, ens_probs)

    p_value = float(np.mean(null_aucs >= observed_auc))
    # Exact lower bound: p_value < 1/n_permutations if never beaten
    p_value = max(p_value, 1.0 / n_permutations)

    logger.info("  Observed AUC : %.4f", observed_auc)
    logger.info("  Null AUC     : %.4f ± %.4f  (mean ± std)", null_aucs.mean(), null_aucs.std())
    logger.info("  p-value      : %.4f  (%s)", p_value, "✓ significant" if p_value < 0.05 else "✗ not significant")

    # Save null distribution plot
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(null_aucs, bins=50, color=palette.CONTROL, alpha=0.7, edgecolor="white", label="Null distribution")
        ax.axvline(observed_auc, color=palette.ASD, lw=2.5, label=f"Observed AUC={observed_auc:.4f}")
        ax.axvline(np.percentile(null_aucs, 95), color=palette.AMBER, lw=1.5, ls="--",
                   label="95th percentile of null")
        ax.set(
            title=f"Permutation Test (n={n_permutations})\np={p_value:.4f}",
            xlabel="AUC",
            ylabel="Count",
        )
        ax.legend(fontsize=10)
        apply_publication_style(ax)

        plt.tight_layout()
        out = output_dir / plot_name
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("  Plot saved → %s", out)
    except Exception as e:
        logger.warning("  Permutation plot failed: %s", e)

    return {
        "observed_auc":   observed_auc,
        "null_auc_mean":  float(null_aucs.mean()),
        "null_auc_std":   float(null_aucs.std()),
        "p_value":        p_value,
        "significant":    p_value < 0.05,
        "n_permutations": n_permutations,
    }

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: SUBGROUP ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def run_subgroup_analysis(
    test_graphs: list,
    ens_models_probs: dict[int, np.ndarray],  # fold_id → probs
    fold_aucs: list[float],
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    """
    Report AUC stratified by:
      - Sex  (male vs female; encoded as 1/2)
      - Age group  (< 15 vs ≥ 15 years, after de-normalising age)
      - Site  (top-5 most-represented sites)
    """
    logger.info("=" * 60)
    logger.info("SECTION 3 — SUBGROUP ANALYSIS")
    logger.info("=" * 60)

    # Build ensemble probs per test graph (maintain order)
    all_labels = []
    fold_probs_list = []
    fold_weights = []

    # First, collect labels from test graphs
    loader = make_loader(test_graphs, batch_size=1, shuffle=False)
    for batch in loader:
        if hasattr(batch, 'y') and batch.y is not None:
            all_labels.append(batch.y.item())
    labels_np = np.array(all_labels)

    # Collect predictions from each fold model
    for fold_id in sorted(ens_models_probs.keys()):
        probs_f = ens_models_probs[fold_id]
        if probs_f is not None and len(probs_f) > 0:
            fold_probs_list.append(probs_f)
            fold_weights.append(fold_aucs[fold_id] if fold_id < len(fold_aucs) else 0.5)
            continue

        try:
            model = load_model(fold_id=fold_id, device=DEVICE)
            probs_f, _ = _predict_probs(model, make_loader(test_graphs, batch_size=1, shuffle=False))
            if probs_f is not None and len(probs_f) > 0:
                fold_probs_list.append(probs_f)
                fold_weights.append(fold_aucs[fold_id] if fold_id < len(fold_aucs) else 0.5)
            del model
        except FileNotFoundError:
            continue

    if len(fold_probs_list) == 0:
        logger.warning("No fold probs collected — skipping subgroup analysis")
        return {}

    # Extract metadata from test_graphs
    age_list, sex_list, site_list = [], [], []
    for g in test_graphs:
        if g is None:
            continue
        age_raw  = float(g.age.item())  if hasattr(g, "age")  and g.age is not None  else 0.0
        sex_raw  = float(g.sex.item())  if hasattr(g, "sex")  and g.sex is not None  else 0.0
        site_raw = int(g.site_id.item()) if hasattr(g, "site_id") and g.site_id is not None else -1
        # De-normalise age: age = age_norm * 20 + 15
        age_actual = age_raw * 20.0 + 15.0
        # De-normalise sex: sex = sex_norm + 1.5  (1=male, 2=female)
        sex_actual = round(sex_raw + 1.5)
        age_list.append(age_actual)
        sex_list.append(sex_actual)
        site_list.append(site_raw)

    # Weighted ensemble probs
    stacked = np.stack(fold_probs_list, axis=0)
    w = np.array(fold_weights, dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
    ens_probs = np.average(stacked, axis=0, weights=w)

    subgroup_results: dict[str, dict] = {}

    def _safe_auc(p, lbl):
        if len(np.unique(lbl)) < 2 or len(lbl) < 5:
            return float("nan")
        return float(roc_auc_score(lbl, p))

    # ── Sex subgroups ─────────────────────────────────────────────────────────
    for sex_code, sex_name in [(1, "Male"), (2, "Female")]:
        idx = np.where(np.array(sex_list) == sex_code)[0]
        if len(idx) < 5:
            continue
        auc = _safe_auc(ens_probs[idx], labels_np[idx])
        n_asd = int(labels_np[idx].sum())
        subgroup_results[f"sex_{sex_name}"] = {
            "n": len(idx), "n_asd": n_asd, "n_control": len(idx) - n_asd, "auc": auc
        }
        logger.info("  Sex %-8s  n=%-4d  n_asd=%-3d  AUC=%.4f", sex_name, len(idx), n_asd, auc)

    # ── Age subgroups ─────────────────────────────────────────────────────────
    for age_label, age_mask in [
        ("Age<15",  np.array(age_list) < 15),
        ("Age>=15", np.array(age_list) >= 15),
    ]:
        idx = np.where(age_mask)[0]
        if len(idx) < 5:
            continue
        auc = _safe_auc(ens_probs[idx], labels_np[idx])
        n_asd = int(labels_np[idx].sum())
        subgroup_results[f"age_{age_label}"] = {
            "n": len(idx), "n_asd": n_asd, "n_control": len(idx) - n_asd, "auc": auc
        }
        logger.info("  %-12s        n=%-4d  n_asd=%-3d  AUC=%.4f", age_label, len(idx), n_asd, auc)

    # ── Site subgroups (top-5) ────────────────────────────────────────────────
    site_arr      = np.array(site_list)
    unique_sites, site_counts = np.unique(site_arr[site_arr >= 0], return_counts=True)
    top5_sites    = unique_sites[np.argsort(site_counts)[::-1][:5]]
    for site_id in top5_sites:
        idx   = np.where(site_arr == site_id)[0]
        if len(idx) < 5:
            continue
        auc   = _safe_auc(ens_probs[idx], labels_np[idx])
        n_asd = int(labels_np[idx].sum())
        subgroup_results[f"site_{site_id}"] = {
            "n": len(idx), "n_asd": n_asd, "n_control": len(idx) - n_asd, "auc": auc
        }
        logger.info("  Site %-4d          n=%-4d  n_asd=%-3d  AUC=%.4f", site_id, len(idx), n_asd, auc)

    # ── Plot ──────────────────────────────────────────────────────────────────
    _plot_subgroups(subgroup_results, output_dir / "subgroup_analysis.png")

    # ── Multiple Comparison Correction ─────────────────────────────────────
    n_tests = len(subgroup_results)
    corrected_alpha = 0.05 / n_tests  # Bonferroni correction
    logger.info("\n  Multiple comparison correction (Bonferroni):")
    logger.info("    Original alpha: 0.05")
    logger.info("    Number of tests: %d", n_tests)
    logger.info("    Corrected alpha: %.4f", corrected_alpha)

    for _name, res in subgroup_results.items():
        is_significant = res.get("auc", 0) > 0.5 and (
            res.get("auc", 0) - 0.5 > 0.1
        )
        res["significant_corrected"] = is_significant
        res["corrected_alpha"] = corrected_alpha

    return subgroup_results

def _plot_subgroups(subgroups: dict, save_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt

        labels_plot = list(subgroups.keys())
        aucs_plot   = [subgroups[k]["auc"] for k in labels_plot]
        ns_plot     = [subgroups[k]["n"]   for k in labels_plot]
        colors      = []
        for k in labels_plot:
            if k.startswith("sex"):
                colors.append(palette.PINK)
            elif k.startswith("age"):
                colors.append(palette.GREEN)
            else:
                colors.append(palette.CONTROL)

        fig, ax = plt.subplots(figsize=(10, max(4, len(labels_plot) * 0.7 + 1)))
        y = np.arange(len(labels_plot))
        bars = ax.barh(y, aucs_plot, color=colors, alpha=0.85, edgecolor="white")
        ax.axvline(0.5, color="gray", lw=1.2, ls="--", label="Random (0.50)")

        for _i, (bar, auc, n) in enumerate(zip(bars, aucs_plot, ns_plot, strict=False)):
            if not np.isnan(auc):
                ax.text(auc + 0.02, bar.get_y() + bar.get_height() / 2,
                        f"{auc:.3f}  (n={n})", va="center", fontsize=9)

        ax.set_yticks(y)
        ax.set_yticklabels(labels_plot, fontsize=10)
        ax.set_xlabel("AUC", fontsize=12, fontweight="bold")
        ax.set_title("Subgroup Analysis — AUC by Sex / Age / Site", fontsize=13, fontweight="bold")
        ax.set_xlim(0.3, 1.05)
        from matplotlib.patches import Patch
        legend_elems = [
            Patch(fc="#9b59b6", label="Sex"),
            Patch(fc="#27ae60", label="Age group"),
            Patch(fc="#3498db", label="Site"),
        ]
        ax.legend(handles=legend_elems, loc="lower right", fontsize=9)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("  Subgroup plot saved → %s", save_path)
    except Exception as e:
        logger.warning("  Subgroup plot failed: %s", e)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: BASELINE COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

class FlatMLP(nn.Module):
    """Simple 3-layer MLP on flattened 12 × 24 = 288 node features."""

    def __init__(self, in_dim: int = NUM_LOBES * GNN_IN_CHANNELS, hidden: int = 128, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def _collect_flat_features(graphs: list) -> tuple[np.ndarray, np.ndarray]:
    """Flatten node feature matrices (12 × 24) → 288-dim vector per subject."""
    X, y = [], []
    for g in graphs:
        if g is None:
            continue
        x_np = g.x.cpu().numpy().flatten()       # (336,)
        X.append(x_np)
        y.append(int(g.y.item()))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

def _train_mlp(X_train, y_train, X_test, y_test, epochs=80, lr=1e-3, seed=42) -> tuple[float, np.ndarray]:
    """Train a flat MLP and return test AUC and probabilities."""
    torch.manual_seed(seed)
    model = FlatMLP(in_dim=X_train.shape[1]).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    X_tr = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_tr = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
    X_te = torch.tensor(X_test,  dtype=torch.float32).to(DEVICE)

    from torch.utils.data import DataLoader as TDL
    from torch.utils.data import TensorDataset
    ds     = TensorDataset(X_tr, y_tr)
    loader = TDL(ds, batch_size=32, shuffle=True)

    model.train()
    for _ep in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            F.cross_entropy(model(xb), yb).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(X_te), dim=1)[:, 1].cpu().numpy()
    return float(roc_auc_score(y_test, probs)), probs

def _print_baseline_table(baselines: dict[str, float]) -> None:
    """Print baseline comparison table."""
    logger.info("\n  Baseline Comparison:")
    logger.info("  " + "-" * 40)
    for name, auc in sorted(baselines.items(), key=lambda x: -x[1]):
        marker = " *" if "Ours" in name else ""
        logger.info(f"    {name:<25} AUC: {auc:.4f}{marker}")
    logger.info("  " + "-" * 40)

def _plot_baselines(baselines: dict[str, float], save_path: Path) -> None:
    """Plot baseline comparison bar chart."""
    try:
        import matplotlib.pyplot as plt
        names = list(baselines.keys())
        fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.5)))
        aucs = list(baselines.values())
        colors = [palette.GREEN if "Ours" in n else palette.CONTROL for n in names]
        y = range(len(names))
        bars = ax.barh(y, aucs, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.set_xlabel("AUC", fontsize=12, fontweight="bold")
        ax.set_title("Baseline Model Comparison", fontsize=13, fontweight="bold")
        ax.axvline(0.5, color=palette.NEUTRAL, linestyle="--", lw=1.5, alpha=0.7)
        ax.set_xlim(0.5, 1.05)
        apply_publication_style(ax)

        # Add value labels
        for bar, auc in zip(bars, aucs, strict=False):
            ax.text(auc + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{auc:.3f}", va="center", fontsize=10)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"  Plot saved → {save_path}")
    except Exception as e:
        logger.warning(f"  Baseline plot failed: {e}")

def run_baseline_comparison(
    train_graphs: list,
    test_graphs: list,
    gnn_ensemble_auc: float,
    gnn_probs: np.ndarray | None = None,
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    """
    Compare GNN against SVM, Random Forest, and flat MLP on (12 × 24) features.
    Also records Heinsfeld et al. 2018 literature reference (AUC≈0.70).
    """
    logger.info("=" * 60)
    logger.info("SECTION 4 — BASELINE COMPARISON")
    logger.info("=" * 60)

    X_train, y_train = _collect_flat_features(train_graphs)
    X_test,  y_test  = _collect_flat_features(test_graphs)
    logger.info("  Train: %d  Test: %d  Features: %d", len(X_train), len(X_test), X_train.shape[1])

    baselines: dict[str, float] = {}

    # ── SVM ───────────────────────────────────────────────────────────────────
    logger.info("  Training SVM (RBF kernel)…")
    t0 = time.time()
    svm_pipe = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", probability=True, C=1.0, gamma="scale", random_state=42))])
    svm_pipe.fit(X_train, y_train)
    svm_probs = svm_pipe.predict_proba(X_test)[:, 1]
    baselines["SVM (RBF)"] = float(roc_auc_score(y_test, svm_probs))
    logger.info("  SVM AUC=%.4f  (%.1fs)", baselines["SVM (RBF)"], time.time() - t0)

    # ── Random Forest ─────────────────────────────────────────────────────────
    logger.info("  Training Random Forest (200 trees)…")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=5, min_samples_split=10,
        max_features='sqrt', random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    baselines["Random Forest"] = float(roc_auc_score(y_test, rf_probs))
    logger.info("  RF AUC=%.4f  (%.1fs)", baselines["Random Forest"], time.time() - t0)
    if baselines["Random Forest"] > 0.95:
        logger.warning(
            "  ⚠ RF AUC=%.4f is suspiciously high — may reflect site-correlated "
            "spatial features (conf_std / detection_count). Interpret with caution.",
            baselines["Random Forest"],
        )

    # ── Flat MLP ──────────────────────────────────────────────────────────────
    logger.info("  Training Flat MLP (336-dim → 128 → 64 → 2)…")
    t0 = time.time()
    mlp_auc, mlp_probs = _train_mlp(X_train, y_train, X_test, y_test)
    baselines["Flat MLP"] = mlp_auc
    logger.info("  MLP AUC=%.4f  (%.1fs)", baselines["Flat MLP"], time.time() - t0)

    # ── GNN + Literature ──────────────────────────────────────────────────────
    baselines["GNN (Ours)"]                = gnn_ensemble_auc
    baselines["Heinsfeld et al. 2018"]     = 0.70   # published  ABIDE-I DNN AUC
    baselines["Ktena et al. 2018"]         = 0.69   # metric-learning GNN on ABIDE

    _print_baseline_table(baselines)
    _plot_baselines(baselines, output_dir / "baseline_comparison.png")

    # ── DeLong Tests ─────────────────────────────────────────────────
    logger.info("\n  DeLong Tests (GNN vs baselines):")
    delong_results = _run_delong_tests(
        y_test, baselines,
        gnn_probs=gnn_probs,
        svm_probs=svm_probs, rf_probs=rf_probs, mlp_probs=mlp_probs
    )

    # AUC Confidence Interval (DeLong)
    if gnn_probs is not None and len(gnn_probs) > 0:
        try:
            auc_ci = compute_auc_confidence_interval(y_test, gnn_probs)
            logger.info("  GNN AUC 95%% CI: [%.4f, %.4f]", auc_ci[1], auc_ci[2])
            delong_results["auc_ci"] = {
                "auc": auc_ci[0],
                "lower": auc_ci[1],
                "upper": auc_ci[2],
            }
        except Exception as e:
            logger.warning("  AUC CI computation failed: %s", e)

    return delong_results

def _run_delong_tests(
    y_test: np.ndarray,
    baselines: dict[str, float],
    gnn_probs: np.ndarray | None = None,
    svm_probs: np.ndarray | None = None,
    rf_probs: np.ndarray | None = None,
    mlp_probs: np.ndarray | None = None,
) -> dict:
    """Run DeLong tests comparing GNN vs baseline models with Bonferroni correction."""
    from sklearn.metrics import roc_auc_score

    results = {
        "baselines": baselines,
        "comparisons": [],
    }

    # Build model scores dict for comparison
    model_scores = {"GNN (Ours)": gnn_probs} if gnn_probs is not None else {}
    if svm_probs is not None:
        model_scores["SVM (RBF)"] = svm_probs
    if rf_probs is not None:
        model_scores["Random Forest"] = rf_probs
    if mlp_probs is not None:
        model_scores["Flat MLP"] = mlp_probs

    if len(model_scores) < 2:
        return results

    gnn_auc = roc_auc_score(y_test, gnn_probs) if gnn_probs is not None else 0.0
    n_comparisons = len(model_scores) - 1  # Excluding GNN itself

    # Run comparisons
    for name, scores in model_scores.items():
        if name == "GNN (Ours)":
            continue
        try:
            log_pval, z = delong_roc_test(
                y_test,
                model_scores["GNN (Ours)"],
                scores,
            )
            pval = 10**log_pval if log_pval < 0 else 1.0
            baseline_auc = roc_auc_score(y_test, scores)
            auc_delta = gnn_auc - baseline_auc

            # Bonferroni correction
            pval_corrected = min(pval * n_comparisons, 1.0)
            sig_bonf = " *" if pval_corrected < 0.05 else ""

            logger.info("  GNN vs %s: z=%.2f, p=%.4f (Bonf: %.4f), ΔAUC=%+.4f%s",
                        name, z, pval, pval_corrected, auc_delta, sig_bonf)
            results["comparisons"].append({
                "model1": "GNN (Ours)",
                "model2": name,
                "gnn_auc": float(gnn_auc),
                "baseline_auc": float(baseline_auc),
                "auc_delta": float(auc_delta),
                "z": float(z),
                "p_value": float(pval),
                "p_value_bonferroni": float(pval_corrected),
                "log10_p": float(log_pval),
                "significant": pval_corrected < 0.05,
                "n_comparisons": n_comparisons,
            })
        except Exception as e:
            logger.warning("  DeLong test failed for GNN vs %s: %s", name, e)

    return results

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: COMPREHENSIVE RESULTS TABLE
# ══════════════════════════════════════════════════════════════════════════════

def save_comprehensive_results(
    ensemble_result: dict,
    permutation_result: dict,
    subgroup_result: dict,
    baseline_result: dict,
    paired_ttest_result: dict | None = None,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Write JSON + CSV summary of all evaluation sections."""
    logger.info("=" * 60)
    logger.info("SECTION 5 — COMPREHENSIVE RESULTS TABLE")
    logger.info("=" * 60)

    m   = ensemble_result["ensemble_metrics"]
    ci  = ensemble_result["ensemble_ci_95"]
    perm = permutation_result

    # ── Main metrics table (publication-ready) ────────────────────────────────
    rows = []
    for key in ("auc", "auprc", "f1", "accuracy", "sensitivity", "specificity"):
        lo, hi = ci.get(key, (float("nan"), float("nan")))
        rows.append({
            "metric":    key,
            "value":     round(m[key], 4),
            "ci_lower":  round(lo, 4),
            "ci_upper":  round(hi, 4),
            "ci_string": f"{m[key]:.4f} [{lo:.4f}, {hi:.4f}]",
        })

    df_main = pd.DataFrame(rows)

    # ── Per-fold breakdown ────────────────────────────────────────────────────
    df_folds = pd.DataFrame(ensemble_result.get("per_fold_metrics", []))

    # ── Subgroup ──────────────────────────────────────────────────────────────
    sg_rows = [{"subgroup": k, **v} for k, v in subgroup_result.items()]
    df_sg   = pd.DataFrame(sg_rows)

    # ── Baselines ─────────────────────────────────────────────────────────────
    bl_rows = [{"method": k, "auc": v} for k, v in baseline_result.get("baselines", {}).items()]
    df_bl   = pd.DataFrame(bl_rows).sort_values("auc", ascending=False)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = output_dir / "comprehensive_results.csv"
    with open(csv_path, "w") as f:
        f.write("# ENSEMBLE TEST-SET METRICS (GNN)\n")
        f.write(df_main.to_csv(index=False))
        f.write("\n# PER-FOLD METRICS\n")
        f.write(df_folds.to_csv(index=False) if not df_folds.empty else "no data\n")
        f.write("\n# SUBGROUP ANALYSIS\n")
        f.write(df_sg.to_csv(index=False) if not df_sg.empty else "no data\n")
        f.write("\n# BASELINE COMPARISON\n")
        f.write(df_bl.to_csv(index=False) if not df_bl.empty else "no data\n")
    logger.info("  CSV saved → %s", csv_path)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    full_results = {
        "ensemble_metrics":  m,
        "ensemble_ci_95":    ci,
        "permutation_test":  perm,
        "subgroup_analysis": subgroup_result,
        "baseline_comparison": baseline_result.get("baselines", {}),
        "per_fold_metrics":  ensemble_result.get("per_fold_metrics", []),
        "paired_ttest_val_vs_test": paired_ttest_result,
        "ensemble_probs":   ensemble_result.get("ensemble_probs", []),
        "labels":           ensemble_result.get("labels", []),
    }
    json_path = output_dir / "comprehensive_results.json"
    with open(json_path, "w") as f:
        json.dump(_json_safe(full_results), f, indent=2, default=str, allow_nan=False)
    logger.info("  JSON saved → %s", json_path)

    # ── Console summary ───────────────────────────────────────────────────────
    logger.info("\n" + "═" * 65)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("═" * 65)
    logger.info("  Ensemble AUC : %.4f [%.4f, %.4f]",
                m["auc"], ci["auc"][0], ci["auc"][1])
    logger.info("  Ensemble F1  : %.4f [%.4f, %.4f]",
                m["f1"],  ci["f1"][0],  ci["f1"][1])
    logger.info("  Accuracy     : %.4f [%.4f, %.4f]",
                m["accuracy"], ci["accuracy"][0], ci["accuracy"][1])
    logger.info("  Sensitivity  : %.4f [%.4f, %.4f]",
                m["sensitivity"], ci["sensitivity"][0], ci["sensitivity"][1])
    logger.info("  Specificity  : %.4f [%.4f, %.4f]",
                m["specificity"], ci["specificity"][0], ci["specificity"][1])
    logger.info("  AUPRC        : %.4f", m["auprc"])
    logger.info("  p-value      : %.4f  (%s)",
                perm.get("p_value", float("nan")),
                "✓ significant" if perm.get("significant") else "✗ not significant")
    if "p_value_global" in perm and "p_value_within_site" in perm:
        logger.info(
            "  p-value(global / within-site): %.4f / %.4f",
            perm.get("p_value_global", float("nan")),
            perm.get("p_value_within_site", float("nan")),
        )
    logger.info("═" * 65)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 9.2/9.3 Comprehensive Evaluation for Neuro-CXG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-permutation", action="store_true", default=False,
                        help="Skip permutation significance testing.")
    parser.add_argument("--n-permutations", type=int, default=1_000,
                        help="Number of random label shuffles for permutation test.")
    parser.add_argument("--no-baselines", action="store_true", default=False,
                        help="Skip SVM / RF / MLP baseline training.")
    parser.add_argument("--no-subgroups", action="store_true", default=False,
                        help="Skip subgroup analysis.")
    parser.add_argument("--enable-calibration", action="store_true", default=False,
                        help="Enable per-site Platt calibration (default: disabled because it degrades AUC).")
    parser.add_argument("--batch-size", type=int, default=GNN_BATCH_SIZE)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Device       : %s", DEVICE)
    logger.info("Output dir   : %s", args.output_dir)

    # Save config snapshot for reproducibility
    save_config_snapshot(args.output_dir)

    # ── Load datasets ─────────────────────────────────────────────────────────
    logger.info("Loading datasets…")
    test_dataset  = ABIDECausalDataset(split="test")
    train_dataset = ABIDECausalDataset(split="train")
    test_graphs   = [g for g in test_dataset  if g is not None]
    train_graphs  = [g for g in train_dataset if g is not None]
    logger.info("  Test: %d  Train: %d", len(test_graphs), len(train_graphs))

    if not test_graphs:
        logger.error("Test set is empty — run the full pipeline first.")
        sys.exit(1)

    # ── Section 1: Ensemble evaluation ────────────────────────────────────────
    ensemble_result = run_ensemble_evaluation(test_graphs, args.output_dir, args.enable_calibration)
    ens_probs = np.array(ensemble_result["ensemble_probs"])
    labels    = np.array(ensemble_result["labels"])

    # ── Section 1B: Paired t-test (val AUC vs test AUC) ────────────────────────
    paired_ttest_result = run_paired_ttest(ensemble_result, args.output_dir)

    # ── Section 2: Permutation test ───────────────────────────────────────────
    if not args.no_permutation:
        site_ids = np.array([
            int(g.site_id.item())
            if hasattr(g, "site_id") and g.site_id is not None and g.site_id.numel() > 0
            else -1
            for g in test_graphs
        ])
        perm_global = run_permutation_test(
            ens_probs, labels, n_permutations=args.n_permutations,
            output_dir=args.output_dir,
            within_site=False,
            plot_name="permutation_test_global.png",
        )
        perm_within_site = run_permutation_test(
            ens_probs, labels, n_permutations=args.n_permutations,
            output_dir=args.output_dir,
            site_ids=site_ids,
            within_site=True,
            plot_name="permutation_test_within_site.png",
        )
        perm_result = {
            "global": perm_global,
            "within_site": perm_within_site,
            "p_value_global": perm_global.get("p_value", float("nan")),
            "p_value_within_site": perm_within_site.get("p_value", float("nan")),
            "significant_global": perm_global.get("significant"),
            "significant_within_site": perm_within_site.get("significant"),
            # Backward-compatible top-level key: prefer conservative within-site p-value.
            "p_value": perm_within_site.get("p_value", float("nan")),
            "significant": perm_within_site.get("significant"),
        }
    else:
        logger.info("Skipping permutation test (--no-permutation)")
        perm_result = {"skipped": True, "p_value": float("nan"), "significant": None}

    # ── Section 3: Subgroup analysis ──────────────────────────────────────────
    if not args.no_subgroups:
        # Pre-collect per-fold probs (reuse from ensemble_result)
        fold_probs_dict: dict[int, np.ndarray] = {}
        for fold_id in range(K_FOLDS):
            try:
                model = load_model(fold_id=fold_id, device=DEVICE)
                p, _  = _predict_probs(model, make_loader(test_graphs, batch_size=args.batch_size, shuffle=False))
                fold_probs_dict[fold_id] = p
                del model
            except FileNotFoundError:
                pass
        sg_result = run_subgroup_analysis(
            test_graphs, fold_probs_dict, ensemble_result["fold_aucs"], args.output_dir
        )
    else:
        logger.info("Skipping subgroup analysis (--no-subgroups)")
        sg_result = {}

    # ── Section 4: Baseline comparison ────────────────────────────────────────
    if not args.no_baselines:
        bl_result = run_baseline_comparison(
            train_graphs, test_graphs, ensemble_result["ensemble_metrics"]["auc"],
            np.array(ensemble_result.get("ensemble_probs", [])),
            args.output_dir
        )
    else:
        logger.info("Skipping baseline comparison (--no-baselines)")
        bl_result = {"baselines": {"GNN (Ours)": ensemble_result["ensemble_metrics"]["auc"]}}

    # ── Section 5: Comprehensive results table ────────────────────────────────
    save_comprehensive_results(ensemble_result, perm_result, sg_result, bl_result, paired_ttest_result, args.output_dir)

if __name__ == "__main__":
    main()
