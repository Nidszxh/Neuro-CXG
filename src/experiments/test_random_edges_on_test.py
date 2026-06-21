"""
Quick Test: Use main pipeline models with random edge topology on test set.

Uses same inference approach as run_evaluation.py for fair comparison.
"""

import argparse
import logging

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.core.config import (
    CHECKPOINT_DIR,
    DEVICE,
    GNN_BATCH_SIZE,
)
from src.features.graph_factory import ABIDECausalDataset
from src.models.factory import build_model
from src.models.training_utils import make_loader

logger = logging.getLogger(__name__)


def load_model(fold_id: int):
    """Load model for a specific fold."""
    checkpoint_path = CHECKPOINT_DIR / f"best_model_fold{fold_id}.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)

    model = build_model(device=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    return model


def predict_probs(model, loader):
    """Run inference using same approach as run_evaluation.py."""
    all_probs, all_labels = [], []
    model.eval()

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            batch = batch.to(DEVICE)

            out = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
                site_id=batch.site_id if hasattr(batch, "site_id") else None,
                age=batch.age if hasattr(batch, "age") else None,
                sex=batch.sex if hasattr(batch, "sex") else None,
                fiq=batch.fiq if hasattr(batch, "fiq") else None,
            )

            probs = torch.softmax(out, dim=1)[:, 1].detach().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(batch.y.cpu().numpy())

    return np.concatenate(all_probs), np.concatenate(all_labels)


def randomize_graphs(test_ds, seed: int = 42):
    """Create test data with randomized edge topology (F ablation)."""
    rng = np.random.RandomState(seed)
    randomized_data = []

    for i in range(len(test_ds)):
        data = test_ds[i]
        if data is None:
            continue

        data = data.clone()

        edge_index = data.edge_index
        num_edges = edge_index.shape[1]

        if num_edges < 2:
            randomized_data.append(data)
            continue

        src = edge_index[0].numpy()
        dst = edge_index[1].numpy()
        perm = rng.permutation(num_edges)

        new_edge_index = torch.stack(
            [torch.from_numpy(src[perm]), torch.from_numpy(dst[perm])]
        )
        new_edge_attr = data.edge_attr[perm]

        data.edge_index = new_edge_index
        data.edge_attr = new_edge_attr
        randomized_data.append(data)

    return randomized_data


def identity_graphs(test_ds, num_lobes: int = 12):
    """Create test data with fully connected edges (G ablation)."""
    identity_data = []

    for i in range(len(test_ds)):
        data = test_ds[i]
        if data is None:
            continue

        data = data.clone()

        # Create fully connected graph
        src_nodes = []
        dst_nodes = []
        for s in range(num_lobes):
            for d in range(num_lobes):
                if s != d:
                    src_nodes.append(s)
                    dst_nodes.append(d)

        new_edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
        new_edge_attr = torch.ones(new_edge_index.shape[1], 1, dtype=torch.float32)

        data.edge_index = new_edge_index
        data.edge_attr = new_edge_attr
        identity_data.append(data)

    return identity_data


def main():
    parser = argparse.ArgumentParser(description="Test edge topology on test set")
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Which folds to test (default: 0 1 2 3 4)",
    )
    parser.add_argument(
        "--single-fold", type=int, default=None, help="Run single fold instead of all"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for edge randomization"
    )
    parser.add_argument(
        "--mode",
        choices=["all", "original", "random", "identity"],
        default="all",
        help="Which graph topology to test",
    )
    args = parser.parse_args()

    folds = [args.single_fold] if args.single_fold is not None else args.folds

    logger.info("=" * 70)
    logger.info("TEST: ORIGINAL vs RANDOM vs IDENTITY EDGES ON HELD-OUT TEST SET")
    logger.info("=" * 70)

    results = {"original": [], "random": [], "identity": []}

    test_ds = ABIDECausalDataset(split="test")

    for fold_id in folds:
        model = load_model(fold_id)
        logger.info(f"\n--- FOLD {fold_id} ---")

        # Original
        test_data = [test_ds[i] for i in range(len(test_ds)) if test_ds[i] is not None]
        loader = make_loader(test_data, batch_size=GNN_BATCH_SIZE, shuffle=False)
        probs, labels = predict_probs(model, loader)
        orig_auc = roc_auc_score(labels, probs)
        results["original"].append(orig_auc)
        logger.info(f"  Original edges: AUC = {orig_auc:.4f}")

        # Random (F)
        if args.mode in ["all", "random"]:
            test_data = randomize_graphs(test_ds, seed=args.seed + fold_id)
            loader = make_loader(test_data, batch_size=GNN_BATCH_SIZE, shuffle=False)
            probs, labels = predict_probs(model, loader)
            rand_auc = roc_auc_score(labels, probs)
            results["random"].append(rand_auc)
            logger.info(f"  Random edges (F): AUC = {rand_auc:.4f}")

        # Identity (G) - fully connected
        if args.mode in ["all", "identity"]:
            test_data = identity_graphs(test_ds)
            loader = make_loader(test_data, batch_size=GNN_BATCH_SIZE, shuffle=False)
            probs, labels = predict_probs(model, loader)
            ident_auc = roc_auc_score(labels, probs)
            results["identity"].append(ident_auc)
            logger.info(f"  Identity edges (G): AUC = {ident_auc:.4f}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Fold':<6} {'Original':<12} {'Random (F)':<12} {'Identity (G)':<12}")
    logger.info("-" * 45)
    for i, fold in enumerate(folds):
        orig = results["original"][i]
        rand = results["random"][i] if len(results["random"]) > i else 0
        ident = results["identity"][i] if len(results["identity"]) > i else 0
        logger.info(f"{fold:<6} {orig:<12.4f} {rand:<12.4f} {ident:<12.4f}")
    logger.info("-" * 45)
    logger.info(
        f"{'Mean':<6} {np.mean(results['original']):<12.4f} {np.mean(results['random']):<12.4f} {np.mean(results['identity']):<12.4f}"
    )
    logger.info("=" * 70)

    orig_mean = np.mean(results["original"])
    rand_mean = np.mean(results["random"])
    ident_mean = np.mean(results["identity"])

    logger.info(f"\nOriginal vs Random (F): {rand_mean - orig_mean:+.4f}")
    logger.info(f"Original vs Identity (G): {ident_mean - orig_mean:+.4f}")
    logger.info(f"Random (F) vs Identity (G): {ident_mean - rand_mean:+.4f}")


if __name__ == "__main__":
    main()
