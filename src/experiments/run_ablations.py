"""
Ablation Studies: Identify Signal Sources for AUC Improvement
==============================================================

Runs five controlled ablations to diagnose what drives the GNN's AUC=0.63:

  A — FlatMLP (no graph structure): 5-fold MLP on flattened 12×24 node features.
      If AUC_MLP > AUC_GNN → signal is in features, not topology.

  B — Spatial only (4 features): Zero out temporal+frequency+internal.
      Establishes how much AUC comes from anatomy alone.

  C — Temporal+Spatial (12 features, no frequency): Zero out frequency+internal.
      Tests whether 10 frequency features add signal or noise.

  D — Lagged Pearson edges (vs. Granger): Rebuild graphs with lagged_pearson method.
      If Pearson AUC > Granger AUC → revert causality method.

  E — No site embeddings / demographics: Confirm conditioning is helping.

Usage:
    python -m src.experiments.run_ablations                      # all ablations
    python -m src.experiments.run_ablations --ablations A B E    # specific ablations
    python -m src.experiments.run_ablations --ablations D        # D: also rebuilds graphs
    python -m src.experiments.run_ablations --dry-run            # print plan only
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score
from torch_geometric.nn import global_mean_pool

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import (
    ALL_FEATURE_NAMES,
    CAUSAL_GRAPHS_DIR,
    CHECKPOINT_DIR,
    DATA_PROCESSED,
    DEVICE,
    FEATURE_GROUPS,
    FOCAL_LOSS_ALPHA,
    FOCAL_LOSS_GAMMA,
    USE_FOCAL_LOSS,
    USE_CLASS_WEIGHTS,
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
)
from src.models.losses import FocalLoss
from src.models.factory import build_model
from src.models.training_utils import make_loader, train_fold_with_onecycle

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = RESULTS_ABLATIONS_DIR
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Feature group index ranges ────────────────────────────────────────────────
_GROUP_SLICES: Dict[str, slice] = {}
_offset = 0
for _grp, _feats in FEATURE_GROUPS.items():
    _GROUP_SLICES[_grp] = slice(_offset, _offset + len(_feats))
    _offset += len(_feats)

TEMPORAL_SLICE   = _GROUP_SLICES["temporal"]    # indices 0:8 (8 features)
FREQUENCY_SLICE  = _GROUP_SLICES["frequency"]   # indices 8:18 (10 features)
INTERNAL_SLICE   = _GROUP_SLICES["internal"]    # indices 18:20 (2 features)
SPATIAL_SLICE    = _GROUP_SLICES["spatial"]     # indices 20:24 (4 features)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET WRAPPER FOR FEATURE MASKING
# ─────────────────────────────────────────────────────────────────────────────

class MaskedDataset:
    """
    Wraps ABIDECausalDataset and zeroes out specified feature-group columns.
    Zero-masking preserves graph topology while ablating feature contributions.
    """

    def __init__(self, base_dataset, keep_groups: List[str]):
        self.ds = base_dataset
        # Build a boolean mask: True = keep, False = zero
        self.mask = torch.zeros(GNN_IN_CHANNELS, dtype=torch.bool)
        for grp in keep_groups:
            sl = _GROUP_SLICES[grp]
            self.mask[sl] = True

        dropped = [g for g in FEATURE_GROUPS if g not in keep_groups]
        logger.info(f"  MaskedDataset: keep={keep_groups}, zero={dropped}")
        logger.info(f"  Active features: {self.mask.sum().item()} / {GNN_IN_CHANNELS}")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        if sample is None:
            return None
        sample = sample.clone()
        sample.x = sample.x * self.mask.float()
        return sample

    def get(self, idx):
        return self[idx]


# ─────────────────────────────────────────────────────────────────────────────
# FLAT MLP MODEL (ABLATION A)
# ─────────────────────────────────────────────────────────────────────────────

class FlatMLP(nn.Module):
    """
    MLP baseline: flattens all node features (12 × GNN_IN_CHANNELS) into a
    single vector and classifies without any message-passing.
    Provides an upper-bound on feature-driven AUC.

    Accepts the same forward signature as CausalBrainGNN so it works with
    train_fold_with_onecycle unchanged.
    """

    def __init__(
        self,
        in_channels: int = GNN_IN_CHANNELS,
        hidden: int = 256,
        num_classes: int = 2,
        dropout: float = 0.45,
        num_nodes: int = NUM_LOBES,
    ):
        super().__init__()
        flat_dim = num_nodes * in_channels  # 12 × 24 = 288
        self.net = nn.Sequential(
            nn.Linear(flat_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes),
        )
        self.num_nodes = num_nodes

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        site_id=None,
        age=None,
        sex=None,
        fiq=None,
        return_site_logits: bool = False,
    ) -> torch.Tensor:
        # x: (total_nodes_in_batch, GNN_IN_CHANNELS)
        # Since all graphs have exactly NUM_LOBES nodes, reshape safely:
        batch_size = int(batch.max().item()) + 1
        x_flat = x.view(batch_size, self.num_nodes * x.shape[-1])
        return self.net(x_flat)


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH REBUILD FOR ABLATION D
# ─────────────────────────────────────────────────────────────────────────────

def build_pearson_graphs(output_dir: Path) -> bool:
    """
    Rebuild causal graphs using lagged_pearson instead of Granger.
    Saves to `output_dir` so existing Granger graphs are not overwritten.
    Returns True on success.
    """
    logger.info("\n" + "=" * 70)
    logger.info("ABLATION D: REBUILDING GRAPHS WITH LAGGED PEARSON")
    logger.info("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Monkey-patch the module-level constant in construct_causal
    # NOTE: This is fragile for parallel execution but safe for sequential ablation runs.
    # TODO: Refactor to use dependency injection.
    import src.features.construct_causal as cc_mod


    try:
        logger.info(f"  Method override : {original_method} → lagged_pearson")
        logger.info(f"  Output dir      : {output_dir}")

        import pandas as pd
        from src.core.config import MASTER_MANIFEST
        from tqdm import tqdm

        manifest = pd.read_csv(MASTER_MANIFEST)
        success, failed = 0, 0
        for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Building Pearson graphs"):
            result = cc_mod.construct_graph(row["subject_id"], row["split"], method="lagged_pearson", output_dir=output_dir)
            if result:
                success += 1
            else:
                failed += 1

        logger.info(f"  Built {success}/{success+failed} graphs in {output_dir}")
        return success > 0

    finally:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# GRAPH REBUILD FOR ABLATION D2 (Ridge Granger)
# ─────────────────────────────────────────────────────────────────────────────

def build_ridge_granger_graphs(output_dir: Path) -> bool:
    """
    Rebuild causal graphs using ridge_granger instead of lagged_pearson.
    Saves to `output_dir` so existing graphs are not overwritten.
    Returns True on success.
    """
    logger.info("\n" + "=" * 70)
    logger.info("ABLATION D2: REBUILDING GRAPHS WITH RIDGE GRANGER")
    logger.info("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Monkey-patch the module-level constant in construct_causal
    import src.features.construct_causal as cc_mod


    try:
        logger.info(f"  Method override : {original_method} → ridge_granger")
        logger.info(f"  Output dir      : {output_dir}")

        import pandas as pd
        from src.core.config import MASTER_MANIFEST
        from tqdm import tqdm

        manifest = pd.read_csv(MASTER_MANIFEST)
        success, failed = 0, 0
        for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Building Ridge Granger graphs"):
            result = cc_mod.construct_graph(row["subject_id"], row["split"], method="ridge_granger", output_dir=output_dir)
            if result:
                success += 1
            else:
                failed += 1

        logger.info(f"  Built {success}/{success+failed} graphs in {output_dir}")
        return success > 0

    finally:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# CORE TRAINING RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_kfold(
    dataset,
    model_factory,
    ablation_name: str,
    folds: int = K_FOLDS,
    use_grl: bool = False,
    grl_alpha_max: float = 1.0,
) -> Dict:
    """
    5-fold stratified CV for any dataset/model combination.
    Returns summary dict with per-fold and mean AUC.

    Args:
        use_grl: Whether to use Gradient Reversal Layer for site debiasing.
        grl_alpha_max: Maximum GRL alpha value for annealing schedule.
    """
    logger.info(f"\n{'━'*70}")
    logger.info(f"ABLATION {ablation_name}: 5-Fold CV")
    logger.info(f"{'━'*70}")

    # Collect labels for stratification
    labels = []
    for i in range(len(dataset)):
        d = dataset[i] if hasattr(dataset, "__getitem__") else dataset.get(i)
        if d is not None:
            labels.append(int(d.y.item()))

    if not labels:
        logger.error(f"  No valid data for ablation {ablation_name}")
        return {}

    n_ctrl = labels.count(0)
    n_asd = labels.count(1)
    logger.info(f"  Total subjects: {len(labels)}  (Control={n_ctrl}, ASD={n_asd})")

    manifest = None
    if hasattr(dataset, "manifest"):
        manifest = dataset.manifest
    elif hasattr(dataset, "ds") and hasattr(dataset.ds, "manifest"):
        manifest = dataset.ds.manifest

    if manifest is None or "cv_fold" not in manifest.columns:
        raise ValueError(
            "Ablations require manifest-defined cv_fold splits for site-stratified protocol. "
            "Run split.py first to generate predefined folds."
        )

    cv_folds = manifest["cv_fold"].to_numpy()
    if cv_folds.min() < 0 or cv_folds.max() >= folds:
        raise ValueError(
            f"Invalid cv_fold values: found [{cv_folds.min()}, {cv_folds.max()}], "
            f"expected [0, {folds - 1}]."
        )

    cv_splits = []
    for fold_id in range(folds):
        train_idx = np.where(cv_folds != fold_id)[0]
        val_idx = np.where(cv_folds == fold_id)[0]
        cv_splits.append((train_idx, val_idx))

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
        val_data   = [dataset[i] for i in val_idx   if dataset[i] is not None]

        if not train_data or not val_data:
            logger.warning(f"  Fold {fold}: insufficient data, skipping")
            continue

        train_loader = make_loader(train_data, batch_size=GNN_BATCH_SIZE, shuffle=True)
        val_loader   = make_loader(val_data,   batch_size=GNN_BATCH_SIZE)

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
            use_grl=use_grl,
            grl_weight=GNN_SITE_LOSS_WEIGHT if use_grl else 0.0,
            fold=fold,
            weight_decay=GNN_WEIGHT_DECAY,
            grl_alpha_max=grl_alpha_max,
        )

        auc = best_metrics["auc"]
        f1  = best_metrics["f1"]
        fold_aucs.append(auc)
        fold_f1s.append(f1)

        logger.info(
            f"  Fold {fold + 1}/{folds}: AUC={auc:.4f}  F1={f1:.4f}  "
            f"(elapsed {time.time()-t0:.0f}s, best epoch={best_metrics['best_epoch']})"
        )

    if not fold_aucs:
        logger.error(f"  All folds failed for ablation {ablation_name}")
        return {}

    mean_auc = float(np.mean(fold_aucs))
    std_auc  = float(np.std(fold_aucs))
    mean_f1  = float(np.mean(fold_f1s))

    logger.info(f"\n  ╔══ RESULT: AUC = {mean_auc:.4f} ± {std_auc:.4f}  |  F1 = {mean_f1:.4f} ══╗")
    logger.info(f"  Per-fold AUCs: {[round(a, 4) for a in fold_aucs]}")

    return {
        "ablation": ablation_name,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "mean_f1": mean_f1,
        "fold_aucs": fold_aucs,
        "n_subjects": len(labels),
    }


# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL ABLATION RUNNERS
# ─────────────────────────────────────────────────────────────────────────────

def _gnn_factory_default(**override_kwargs):
    """Return a factory function that creates CausalBrainGNN with optional overrides."""
    def factory():
        return build_model(device=DEVICE, **override_kwargs)

    return factory


def run_ablation_a(base_ds) -> Dict:
    """A — FlatMLP: no graph structure, flattened node features only."""

    def mlp_factory():
        return FlatMLP(
            in_channels=GNN_IN_CHANNELS,
            hidden=256,
            num_classes=2,
            dropout=GNN_DROPOUT,
        )

    return run_kfold(base_ds, mlp_factory, ablation_name="A (FlatMLP, no graph)")


def run_ablation_b(base_ds) -> Dict:
    """B — Spatial only: zero temporal + frequency + internal features (4 spatial features)."""
    masked_ds = MaskedDataset(base_ds, keep_groups=["spatial"])
    factory = _gnn_factory_default(
        use_site_embedding=False,
        use_demographics=False,
        use_grl=False,
        edge_gate=True,
    )
    return run_kfold(masked_ds, factory, ablation_name="B (Spatial only, 4 features)")


def run_ablation_c(base_ds) -> Dict:
    """C — Temporal+Spatial (12 features, no frequency): zero frequency+internal, keep temporal+spatial."""
    masked_ds = MaskedDataset(base_ds, keep_groups=["temporal", "spatial"])
    factory = _gnn_factory_default(
        use_site_embedding=False,
        use_demographics=False,
        use_grl=False,
        edge_gate=True,
    )
    return run_kfold(masked_ds, factory, ablation_name="C (Temporal+Spatial, no frequency)")


def run_ablation_d() -> Dict:
    """D — Lagged Pearson edges: rebuild graphs with 'lagged_pearson' method."""
    pearson_dir = DATA_PROCESSED / "causal_graphs_pearson"

    # Check if already built
    n_existing = sum(1 for _ in pearson_dir.glob("*.pt")) if pearson_dir.exists() else 0
    n_granger  = sum(1 for _ in CAUSAL_GRAPHS_DIR.glob("*.pt")) if CAUSAL_GRAPHS_DIR.exists() else 0

    if n_existing < max(1, n_granger // 2):
        logger.info(f"  Pearson graphs: {n_existing} found, Granger graphs: {n_granger}")
        ok = build_pearson_graphs(pearson_dir)
        if not ok:
            logger.error("  Graph rebuild failed — skipping ablation D")
            return {}
    else:
        logger.info(f"  Reusing {n_existing} existing lagged-Pearson graphs in {pearson_dir}")

    # Create dataset pointing to Pearson graph directory
    from src.features.graph_factory import ABIDECausalDataset

    class PearsonDataset(ABIDECausalDataset):
        def _load_data_sources(self):
            super()._load_data_sources()
            self.adj_dir = pearson_dir  # redirect to Pearson graphs

    try:
        pearl_ds = PearsonDataset(split="train")
    except Exception as e:
        logger.error(f"  Failed to load Pearson dataset: {e}")
        return {}

    factory = _gnn_factory_default(
        use_site_embedding=True,
        use_demographics=True,
        use_grl=True,
        grl_alpha=GNN_GRL_ALPHA,  # Use same alpha as main pipeline (0.10)
        edge_gate=True,
    )
    return run_kfold(
        pearl_ds, factory,
        ablation_name="D (Lagged Pearson edges)",
        use_grl=True,
        grl_alpha_max=GNN_GRL_ALPHA_MAX,
    )


def run_ablation_e(base_ds) -> Dict:
    """E — No site embeddings, no demographics conditioning."""
    factory = _gnn_factory_default(
        use_site_embedding=False,
        use_demographics=False,
        use_grl=False,
        edge_gate=True,
    )
    return run_kfold(base_ds, factory, ablation_name="E (No site/demographics)")


def run_ablation_d2() -> Dict:
    """D2 — Ridge Granger edges: rebuild graphs with 'ridge_granger' method and stronger GRL."""
    ridge_granger_dir = DATA_PROCESSED / "causal_graphs_ridge_granger"

    # Check if already built
    n_existing = sum(1 for _ in ridge_granger_dir.glob("*.pt")) if ridge_granger_dir.exists() else 0

    if n_existing < 100:
        logger.info(f"  Ridge Granger graphs: {n_existing} found, building...")
        ok = build_ridge_granger_graphs(ridge_granger_dir)
        if not ok:
            logger.error("  Graph rebuild failed — skipping ablation D2")
            return {}
    else:
        logger.info(f"  Reusing {n_existing} existing ridge-granger graphs in {ridge_granger_dir}")

    # Create dataset pointing to Ridge Granger graph directory
    from src.features.graph_factory import ABIDECausalDataset

    class RidgeGrangerDataset(ABIDECausalDataset):
        def _load_data_sources(self):
            super()._load_data_sources()
            self.adj_dir = ridge_granger_dir  # redirect to Ridge Granger graphs

    try:
        ridge_ds = RidgeGrangerDataset(split="train")
    except Exception as e:
        logger.error(f"  Failed to load Ridge Granger dataset: {e}")
        return {}

    # Use same config as main pipeline (GRL alpha = 0.10)
    factory = _gnn_factory_default(
        use_site_embedding=True,
        use_demographics=True,
        use_grl=True,
        grl_alpha=GNN_GRL_ALPHA,  # Use same alpha as main pipeline (0.10)
        edge_gate=True,
    )
    return run_kfold(
        ridge_ds, factory,
        ablation_name="D2 (Ridge Granger edges)",
        use_grl=True,
        grl_alpha_max=GNN_GRL_ALPHA_MAX,
    )


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: Dict[str, Dict], baseline_auc: float = 0.63) -> None:
    logger.info("\n" + "=" * 70)
    logger.info("ABLATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Ablation':<45} {'AUC ± std':>14} {'vs baseline':>12}")
    logger.info("-" * 72)

    for key, res in results.items():
        if not res:
            logger.info(f"  {key:<43} {'FAILED':>14}")
            continue
        auc = res["mean_auc"]
        std = res["std_auc"]
        delta = auc - baseline_auc
        sign = "+" if delta >= 0 else ""
        logger.info(
            f"  {res['ablation']:<43} {auc:.4f}±{std:.4f} {sign}{delta:+.4f}"
        )

    logger.info("-" * 72)
    logger.info(f"  {'Baseline GNN (full)':<43} {baseline_auc:.4f}  (reference)")
    logger.info("=" * 70)

    # Save to CSV
    import pandas as pd
    rows = [
        {
            "ablation": r.get("ablation", k),
            "mean_auc": r.get("mean_auc", float("nan")),
            "std_auc": r.get("std_auc", float("nan")),
            "mean_f1": r.get("mean_f1", float("nan")),
            "fold_aucs": str(r.get("fold_aucs", [])),
        }
        for k, r in results.items()
        if r
    ]
    if rows:
        out_csv = RESULTS_DIR / "ablation_results.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        logger.info(f"\n  Results saved → {out_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

ABLATION_MAP = {"A": "FlatMLP (no graph)", "B": "Spatial only", "C": "Temporal (no freq)",
                "D": "Lagged Pearson", "D2": "Ridge Granger", "E": "No site/demographics"}


def main():
    parser = argparse.ArgumentParser(description="Ablation study runner")
    parser.add_argument(
        "--ablations", nargs="+", default=list(ABLATION_MAP.keys()),
        choices=list(ABLATION_MAP.keys()),
        help="Which ablations to run (default: all A-E)"
    )
    parser.add_argument("--baseline-auc", type=float, default=0.63,
                        help="Baseline AUC for comparison (default: 0.63)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without training")
    args = parser.parse_args()

    logger.info("\n" + "=" * 70)
    logger.info("NEURO-CXG: ABLATION STUDIES")
    logger.info("=" * 70)
    logger.info(f"  Running ablations : {args.ablations}")
    logger.info(f"  Baseline AUC      : {args.baseline_auc}")
    logger.info(f"  Device            : {DEVICE}")
    if args.dry_run:
        for ab in args.ablations:
            logger.info(f"  [DRY-RUN] Would run Ablation {ab}: {ABLATION_MAP[ab]}")
        return

    # Load base training dataset once (shared by A/B/C/E)
    from src.features.graph_factory import ABIDECausalDataset
    logger.info("\nLoading base training dataset...")
    base_ds = ABIDECausalDataset(split="train")
    logger.info(f"  Loaded {len(base_ds)} training subjects")

    results: Dict[str, Dict] = {}

    if "A" in args.ablations:
        results["A"] = run_ablation_a(base_ds)

    if "B" in args.ablations:
        results["B"] = run_ablation_b(base_ds)

    if "C" in args.ablations:
        results["C"] = run_ablation_c(base_ds)

    if "D" in args.ablations:
        results["D"] = run_ablation_d()  # loads its own dataset

    if "D2" in args.ablations:
        results["D2"] = run_ablation_d2()  # Ridge Granger with stronger GRL

    if "E" in args.ablations:
        results["E"] = run_ablation_e(base_ds)

    print_summary(results, baseline_auc=args.baseline_auc)


if __name__ == "__main__":
    main()
