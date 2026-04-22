"""Shared evaluation utilities for Neuro-CXG models."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def optimal_threshold(probs: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Find the classification threshold that maximizes F1 score.

    Args:
        probs: Predicted positive-class probabilities.
        labels: Ground-truth binary labels.

    Returns:
        A tuple of (best_threshold, best_f1).
    """
    if probs.size == 0 or labels.size == 0 or np.unique(labels).size < 2:
        return 0.5, 0.0

    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f1_scores = 2.0 * precision * recall / (precision + recall + 1e-10)
    best_idx = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    return best_threshold, float(f1_scores[best_idx])


def youden_threshold(probs: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Find threshold that maximizes Youden's J = sensitivity + specificity - 1.

    This balances sensitivity (recall) and specificity, reducing false negatives
    compared to F1-maximizing threshold.

    Args:
        probs: Predicted positive-class probabilities.
        labels: Ground-truth binary labels.

    Returns:
        A tuple of (best_threshold, best_youden_j).
    """
    if probs.size == 0 or labels.size == 0 or np.unique(labels).size < 2:
        return 0.5, 0.0

    fpr, tpr, thresholds = roc_curve(labels, probs)
    youden_j = tpr - fpr
    best_idx = int(np.argmax(youden_j))
    best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    return best_threshold, float(youden_j[best_idx])


def compute_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute binary-classification metrics from probabilities and labels."""
    if probs.size == 0 or labels.size == 0:
        cm = np.zeros((2, 2), dtype=int)
        return {
            "auc": 0.5,
            "auprc": 0.0,
            "f1": 0.0,
            "acc": 0.0,
            "accuracy": 0.0,
            "cm": cm,
            "sensitivity": 0.0,
            "specificity": 0.0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "n_total": 0,
            "threshold": float(threshold),
        }

    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    if np.unique(labels).size < 2:
        auc = 0.5
        auprc = float(labels.mean()) if labels.size else 0.0
    else:
        auc = float(roc_auc_score(labels, probs))
        auprc = float(average_precision_score(labels, probs))

    f1 = float(f1_score(labels, preds, zero_division=0))
    acc = float(accuracy_score(labels, preds))
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "auc": auc,
        "auprc": auprc,
        "f1": f1,
        "acc": acc,
        "accuracy": acc,
        "cm": cm,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "n_total": int(labels.size),
        "threshold": float(threshold),
    }


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Run batched inference and compute metrics for a data loader."""
    model.eval()
    all_probs = []
    all_labels = []

    for batch in loader:
        if batch is None:
            continue

        batch = batch.to(device)
        if hasattr(model, "forward_batch"):
            logits = model.forward_batch(batch)
        else:
            logits = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
                getattr(batch, "site_id", None),
                getattr(batch, "age", None),
                getattr(batch, "sex", None),
                getattr(batch, "fiq", None),
            )

        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        labels = batch.y.detach().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels)

    if not all_probs:
        base = compute_metrics(np.array([]), np.array([]), threshold=threshold)
        return {"probs": np.array([]), "labels": np.array([]), **base}

    probs_array = np.concatenate(all_probs)
    labels_array = np.concatenate(all_labels)

    if np.isnan(probs_array).any():
        base = compute_metrics(np.array([]), np.array([]), threshold=threshold)
        base.update({"probs": probs_array, "labels": labels_array})
        return base

    metrics = compute_metrics(probs_array, labels_array, threshold=threshold)
    return {"probs": probs_array, "labels": labels_array, **metrics}
