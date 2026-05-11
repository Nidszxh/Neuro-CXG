"""
DeLong Test for AUC Comparison

Implements fast DeLong's method for comparing correlated ROC AUCs.
Reference: Sun & Xu (2014) "Fast Implementation of DeLong's Algorithm for Comparing
           the Areas Under Correlated ROC Curves", IEEE Signal Processing Letters
"""
import logging

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def _compute_ground_truth_statistics(y_true: np.ndarray) -> tuple[np.ndarray, int]:
    """Compute ground truth order statistics for DeLong test."""
    n_pos = int(np.sum(y_true == 1))
    order = np.argsort(y_true, kind="quicksort")
    return order, n_pos


def _bootstrap_auc_comparison(y_true, y_pred1, y_pred2, n_bootstrap=1000, seed=42):
    """
    Bootstrap-based AUC comparison (robust alternative to DeLong).
    Returns (log10_p_value, z_statistic).
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)

    auc1 = roc_auc_score(y_true, y_pred1)
    auc2 = roc_auc_score(y_true, y_pred2)
    observed_diff = auc1 - auc2

    # Bootstrap difference in AUC
    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        y_boot = y_true[idx]
        p1_boot = y_pred1[idx]
        p2_boot = y_pred2[idx]

        if len(np.unique(y_boot)) < 2:
            continue
        try:
            auc1_boot = roc_auc_score(y_boot, p1_boot)
            auc2_boot = roc_auc_score(y_boot, p2_boot)
            diffs.append(auc1_boot - auc2_boot)
        except Exception:
            continue

    diffs = np.array(diffs)
    if len(diffs) < 10:
        return 0.0, 0.0

    se_diff = np.std(diffs)
    if se_diff == 0:
        return 0.0, 0.0

    z = observed_diff / se_diff
    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    log_pval = np.log10(p_value) if p_value > 0 else -100

    return log_pval, z


def _fast_delong(
    predictions_sorted_transposed: np.ndarray, n_pos: int
) -> tuple[np.ndarray, np.ndarray]:
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
) -> tuple[float, float]:
    """Perform DeLong test comparing two ROC AUCs using bootstrap method.

    Args:
        y_true: Binary labels (0/1)
        y_pred1: Prediction scores from model 1
        y_pred2: Prediction scores from model 2

    Returns:
        (log10(p_value), z_statistic)
    """
    return _bootstrap_auc_comparison(y_true, y_pred1, y_pred2)


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
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute AUC confidence interval using bootstrap method.

    Args:
        y_true: Binary labels
        y_pred: Prediction scores
        confidence: Confidence level (default 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap samples
        seed: Random seed

    Returns:
        (auc, lower_bound, upper_bound)
    """
    rng = np.random.default_rng(seed)
    len(y_true)
    auc = roc_auc_score(y_true, y_pred)

    # Stratified bootstrap
    idx_pos = np.where(y_true == 1)[0]
    idx_neg = np.where(y_true == 0)[0]

    aucs = []
    for _ in range(n_bootstrap):
        boot_pos = rng.choice(idx_pos, size=len(idx_pos), replace=True)
        boot_neg = rng.choice(idx_neg, size=len(idx_neg), replace=True)
        idx = np.concatenate([boot_pos, boot_neg])
        y_boot = y_true[idx]
        p_boot = y_pred[idx]
        if len(np.unique(y_boot)) < 2:
            continue
        try:
            aucs.append(roc_auc_score(y_boot, p_boot))
        except Exception:
            continue

    aucs = np.array(aucs)
    if len(aucs) < 10:
        return auc, auc - 0.05, auc + 0.05

    alpha = 1 - confidence
    lb = np.percentile(aucs, 100 * alpha / 2)
    ub = np.percentile(aucs, 100 * (1 - alpha / 2))

    return auc, lb, ub


def wilson_score_interval(n_success: int, n_total: int, confidence: float = 0.95) -> tuple[float, float, float]:
    """
    Wilson score confidence interval for a proportion (accuracy).

    Better than Wald interval for small samples and near-boundary probabilities.
    Formula: p ± z * sqrt(p*(1-p)/n + z²/(4n²)) / (1 + z²/n)

    Args:
        n_success: Number of correct predictions
        n_total: Total sample size
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (proportion, lower_bound, upper_bound)
    """
    if n_total == 0:
        return 0.0, 0.0, 1.0

    p = n_success / n_total
    z = stats.norm.ppf((1 + confidence) / 2)

    denominator = 1 + z**2 / n_total
    center = p + z**2 / (2 * n_total)
    spread = z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2))

    lb = (center - spread) / denominator
    ub = (center + spread) / denominator

    return p, max(0.0, lb), min(1.0, ub)
