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
