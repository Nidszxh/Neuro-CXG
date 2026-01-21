import logging
import numpy as np
import torch
from pathlib import Path

# Reuse evaluation helpers from training module
from src.models.gnn_model import evaluate, find_optimal_threshold

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import (
    K_FOLDS,
    GNN_BATCH_SIZE,
    GNN_IN_CHANNELS,
    GNN_HIDDEN_CHANNELS_TUNED,
    DEVICE,
    CHECKPOINT_DIR,
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def run_test_ensemble():
    """Compute a true ensemble on the held-out test split from saved fold checkpoints."""
    from src.features.graph_factory import ABIDECausalDataset
    from torch_geometric.loader import DataLoader
    from src.models.causal_gnn import CausalBrainGNN

    ds = ABIDECausalDataset(split='test')
    test_data = [ds[i] for i in range(len(ds)) if ds[i] is not None]

    if len(test_data) == 0:
        logger.info("Test split empty or unavailable; nothing to evaluate.")
        return

    test_loader = DataLoader(test_data, batch_size=GNN_BATCH_SIZE)

    fold_probs = []
    labels_ref = None
    weights = []

    for fold in range(K_FOLDS):
        ckpt_path = CHECKPOINT_DIR / f"best_model_fold{fold}.pt"
        if not ckpt_path.exists():
            logger.warning(f"Missing checkpoint: {ckpt_path}")
            continue

        checkpoint = torch.load(ckpt_path, weights_only=False)

        model = CausalBrainGNN(
            num_node_features=GNN_IN_CHANNELS,
            hidden_channels=GNN_HIDDEN_CHANNELS_TUNED,
            num_classes=2,
            dropout=0.5,
            num_heads=2,
            num_sites=20,
            use_site_embedding=True,
            use_demographics=True,
        ).to(DEVICE)

        model.load_state_dict(checkpoint['model_state'])

        metrics = evaluate(model, test_loader, threshold=checkpoint.get('threshold', 0.5))
        fold_probs.append(metrics['probs'])
        weights.append(checkpoint.get('auc', 0.0))
        if labels_ref is None:
            labels_ref = metrics['labels']

    if not fold_probs:
        logger.warning("No fold predictions collected; aborting.")
        return

    prob_matrix = np.stack(fold_probs, axis=0)
    weights = np.array(weights)

    if np.all(np.isfinite(weights)) and weights.sum() > 0:
        weights = weights / weights.sum()
        ensemble_probs = np.average(prob_matrix, axis=0, weights=weights)
    else:
        ensemble_probs = prob_matrix.mean(axis=0)

    # Metrics
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
    auc = roc_auc_score(labels_ref, ensemble_probs)
    thr, _ = find_optimal_threshold(labels_ref, ensemble_probs)
    preds = (ensemble_probs > thr).astype(int)
    f1 = f1_score(labels_ref, preds, zero_division=0)
    acc = accuracy_score(labels_ref, preds)

    logger.info("\n======================================================================")
    logger.info("TRUE ENSEMBLE (TEST SPLIT)")
    logger.info("======================================================================")
    logger.info(f"AUC: {auc:.4f}")
    logger.info(f"F1: {f1:.4f} (threshold={thr:.3f})")
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info("======================================================================\n")


if __name__ == "__main__":
    run_test_ensemble()
