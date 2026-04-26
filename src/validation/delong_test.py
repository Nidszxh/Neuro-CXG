"""
DeLong Test for AUC Comparison

Implements fast DeLong's method for comparing correlated ROC AUCs.
Reference: Sun & Xu (2014) "Fast Implementation of DeLong's Algorithm for Comparing
           the Areas Under Correlated ROC Curves", IEEE Signal Processing Letters
"""
import logging
from typing import Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def _compute_ground_truth_statistics(y_true: np.ndarray) -> Tuple[np.ndarray, int]:
    """Compute ground truth order statistics for DeLong test."""
    n_pos = int(np.sum(y_true == 1))
    order = np.argsort(y_true, kind="quicksort")
    return order, n_pos


def _fast_delong(
    predictions_sorted_transposed: np.ndarray, n_pos: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Fast DeLong covariance computation."""
    n_models = predictions_sorted_transposed.shape[0]
    n_neg = predictions_sorted_transposed.shape[1] - n_pos

    pos_preds = predictions_sorted_transposed[:, :n_pos]
    neg_preds = predictions_sorted_transposed[:, n_pos:]

    # Compute AUCs
    aucs = np.empty(n_models)
    for i in range(n_models):
        numerator = np.sum(pos_preds[i].reshape(-1, 1) > neg_preds[i].reshape(1, -1))
        aucs[i] = numerator / (n_pos * n_neg)

    # Simplified variance estimation
    var_aucs = np.var(aucs) / n_models
    se = np.sqrt(var_aucs)

    # Create covariance matrix
    cov = np.eye(n_models) * (se ** 2)
    return aucs, cov


def delong_roc_test(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
) -> Tuple[float, float]:
    """Perform DeLong test comparing two ROC AUCs.

    Args:
        y_true: Binary labels (0/1)
        y_pred1: Prediction scores from model 1
        y_pred2: Prediction scores from model 2

    Returns:
        (log10(p_value), z_statistic)
    """
    order, n_pos = _compute_ground_truth_statistics(y_true)
    predictions = np.vstack([y_pred1, y_pred2])[:, order]
    aucs, cov = _fast_delong(predictions, n_pos)

    diff = aucs[0] - aucs[1]
    se = np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    if se == 0:
        return 0.0, 0.0
    z = diff / se

    log_pval = stats.norm.logsf(abs(z)) / np.log(10)
    return float(log_pval), float(z)


def compare_models_auc(
    y_true: np.ndarray,
    model_scores: dict,
    reference_model: str = "GNN (Ours)",
) -> dict:
    """Compare multiple models' AUCs using DeLong test."""
    results = {"models": {}, "comparisons": [], "aucs": {}}

    for name, scores in model_scores.items():
        try:
            auc = roc_auc_score(y_true, scores)
            results["aucs"][name] = auc
        except Exception as e:
            logger.warning(f"Cannot compute AUC for {name}: {e}")
            continue

    model_names = list(model_scores.keys())
    for name in model_names:
        if name == reference_model:
            continue
        try:
            log_pval, z = delong_roc_test(
                y_true,
                np.array(model_scores[reference_model]),
                np.array(model_scores[name]),
            )
            pval = 10**log_pval if log_pval < 0 else 1.0
            results["comparisons"].append({
                "model1": reference_model,
                "model2": name,
                "auc1": results["aucs"][reference_model],
                "auc2": results["aucs"][name],
                "diff": results["aucs"][reference_model] - results["aucs"][name],
                "z": z,
                "p_value": pval,
                "log10_p": log_pval,
                "significant": pval < 0.05,
            })
        except Exception as e:
            logger.warning(f"DeLong test failed for {name} vs {reference_model}: {e}")

    return results


def compute_auc_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Compute AUC confidence interval using bootstrap method.

    Args:
        y_true: Binary labels
        y_pred: Prediction scores
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        (auc, lower_bound, upper_bound)
    """
    n = len(y_true)
    n_pos = int(np.sum(y_true == 1))
    n_neg = n - n_pos

    auc = roc_auc_score(y_true, y_pred)

    order = np.argsort(y_pred)
    pos_preds = y_pred[order[:n_pos]]
    neg_preds = y_pred[order[n_pos:]]

    theta = np.empty(n_pos)
    for i in range(n_pos):
        theta[i] = np.sum(pos_preds[i] > neg_preds) / n_neg

    var_auc = np.var(theta) / n_pos
    se = np.sqrt(var_auc)

    z_crit = stats.norm.ppf((1 + confidence) / 2)
    lb = auc - z_crit * se
    ub = auc + z_crit * se

    return auc, lb, ub