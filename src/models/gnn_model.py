import logging
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score

# Targeted warning suppression for known harmless issues
warnings.filterwarnings("ignore", category=DeprecationWarning, module="neuroHarmonize")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch_geometric")
warnings.filterwarnings(
    "ignore", message=".*CUDA initialization.*", category=UserWarning
)
warnings.filterwarnings("ignore", message=".*dataclass_transform.*")

# Setup paths and config
from src.core.config import (
    ALL_FEATURE_NAMES,
    CAUSAL_GRAPHS_DIR,
    CAUSAL_GRAPHS_MULTIVIEW_DIR,
    CHECKPOINT_DIR,
    DATA_METADATA,
    DEVICE,
    EVAL_THRESHOLD_POLICY,
    FOCAL_LOSS_ALPHA,
    FOCAL_LOSS_GAMMA,
    GNN_AUTO_GRL_GRID_SEARCH,
    GNN_BATCH_SIZE,
    GNN_EARLY_STOPPING_PATIENCE,
    GNN_EDGE_CONTRASTIVE_WEIGHT,
    GNN_ENFORCE_GRAPH_QUALITY_GATE,
    GNN_ENFORCE_MULTIVIEW_QUALITY_GATE,
    GNN_EPOCHS,
    GNN_FOLD_PREPROCESSING_MODE,
    GNN_GRL_ALPHA,
    GNN_HIDDEN_CHANNELS,
    GNN_IN_CHANNELS,
    GNN_INVARIANCE_WEIGHT,
    GNN_MAX_DEGENERATE_GRAPH_RATE,
    GNN_MI_FEATURE_SELECTION_ENABLED,
    GNN_MI_MAX_KEEP_RATIO,
    GNN_MI_MIN_KEEP_RATIO,
    GNN_MIN_EDGES_FOR_NONDEGENERATE,
    GNN_MIN_EPOCHS_BEFORE_STOPPING,
    GNN_MULTIVIEW_MAX_ZERO_EDGE_RATE,
    GNN_MULTIVIEW_QUALITY_SAMPLE_SIZE,
    GNN_ONECYCLE_MAX_LR,
    GNN_ONECYCLE_WARMUP_FRACTION,
    GNN_SEED,
    GNN_SITE_LOSS_WEIGHT,
    GNN_SITE_NORMALIZATION_MODE,
    GNN_SPATIAL_INVARIANCE_WEIGHT,
    GNN_STRUCTURAL_DROPOUT_PROB,
    GNN_USE_DEMOGRAPHICS,
    GNN_USE_GRL,
    GNN_USE_SITE_EMBEDDING,
    GNN_WEIGHT_DECAY,
    HARMONIZED_FOLDS_DIR,
    K_FOLDS,
    MASTER_MANIFEST,
    NODE_ATTRIBUTES_HARMONIZED,
    NODE_FEATURES_3D,
    NUM_LOBES,
    NUM_SPATIAL_FEATURES,
    RESULTS_TRAINING_DIR,
    USE_CLASS_WEIGHTS,
    USE_FOCAL_LOSS,
)
from src.core.experiment_tracker import ExperimentTracker
from src.core.validators import summarize_graph_degeneracy_from_edge_index
from src.models.evaluation import evaluate_loader
from src.models.factory import build_model
from src.models.losses import build_criterion
from src.models.training_utils import (
    CheckpointManager,
    TrainingTracker,
    make_loader,
    train_fold_with_onecycle,
)

logger = logging.getLogger(__name__)

# Analysis modules
from src.analysis.diagnostics import CausalGraphAnalyzer, TrainingMonitor

try:
    from src.analysis.feature_attribution import FeatureAttributionAnalyzer

    FEATURE_ANALYSIS_AVAILABLE = True
except ImportError:
    FEATURE_ANALYSIS_AVAILABLE = False
    logger.warning("FeatureAttributionAnalyzer unavailable (requires Captum)")

# ── LOSS IMPORTS ────────────────────────────────────────────────────────────────
# CausalInvarianceLoss and SpatialInvarianceLoss are defined in src.models.losses
from src.models.losses import CausalInvarianceLoss, SpatialInvarianceLoss

# UTILITY FUNCTIONS


@torch.no_grad()
def evaluate(model, loader, threshold=0.5):
    """Compatibility wrapper around shared loader evaluation."""
    return evaluate_loader(model, loader, DEVICE, threshold=threshold)


def _graph_site_id(graph_obj) -> int:
    """Extract integer site id from a graph sample."""
    if (
        hasattr(graph_obj, "site_id")
        and graph_obj.site_id is not None
        and graph_obj.site_id.numel() > 0
    ):
        return int(graph_obj.site_id.view(-1)[0].item())
    return -1


def _fit_mi_feature_selection(train_data):
    """Fit fold-internal MI feature selector on train fold only.

    Uses a conservative score-floor policy instead of a median split so only
    near-zero MI channels are pruned. When MI is uninformative, falls back to
    keeping all channels.
    """
    n_features = int(GNN_IN_CHANNELS)
    min_ratio = float(np.clip(GNN_MI_MIN_KEEP_RATIO, 0.0, 1.0))
    max_ratio = float(np.clip(GNN_MI_MAX_KEEP_RATIO, min_ratio, 1.0))

    min_k = max(1, int(np.ceil(min_ratio * n_features)))
    max_k = max(min_k, int(np.floor(max_ratio * n_features)))

    X = np.stack([d.x.mean(dim=0).detach().cpu().numpy() for d in train_data], axis=0)
    y = np.asarray([int(d.y.item()) for d in train_data], dtype=np.int64)

    if np.unique(y).size < 2:
        logger.warning(
            "MI selection: single class in train fold; keeping all %d features",
            n_features,
        )
        scores = np.ones(n_features, dtype=np.float64)
        score_max = 1.0
        floor_threshold = 0.0
        selected_idx = np.arange(n_features, dtype=np.int64)
        candidate_k = int(n_features)
    else:
        scores = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        score_max = float(scores.max())
        floor_threshold = 0.10 * score_max

        if score_max < 1e-9:
            logger.warning(
                "MI selection: all MI scores near zero; keeping all %d features",
                n_features,
            )
            selected_idx = np.arange(n_features, dtype=np.int64)
            candidate_k = int(n_features)
        else:
            above_floor = scores >= floor_threshold
            candidate_k = int(above_floor.sum())

            selected_k = int(np.clip(candidate_k, min_k, max_k))
            if selected_k <= 0:
                selected_k = min_k

            selected_idx = np.argsort(scores)[::-1][:selected_k]
            selected_idx = np.sort(selected_idx).astype(np.int64)

    selected_k = int(len(selected_idx))
    mask = torch.zeros(n_features, dtype=torch.float32)
    mask[selected_idx] = 1.0

    logger.info(
        "MI selection: kept %d/%d features (%.0f%%), score range [%.4f, %.4f], floor threshold %.4f",
        selected_k,
        n_features,
        100.0 * selected_k / max(n_features, 1),
        float(scores.min()),
        float(scores.max()),
        float(floor_threshold),
    )

    metadata = {
        "enabled": True,
        "original_features": n_features,
        "selected_features": int(selected_k),
        "selected_ratio": float(selected_k / max(n_features, 1)),
        "min_allowed": int(min_k),
        "max_allowed": int(max_k),
        "candidate_k": int(candidate_k),
        "score_max": float(score_max),
        "floor_threshold": float(floor_threshold),
    }
    return selected_idx.tolist(), mask, metadata


def _apply_feature_mask(graphs, feature_mask: torch.Tensor) -> None:
    """Apply feature mask in-place without changing channel dimensionality."""
    if feature_mask is None:
        return
    for d in graphs:
        d.x = d.x * feature_mask.to(device=d.x.device, dtype=d.x.dtype).view(1, -1)


def _fit_site_normalization_stats(train_data):
    """Fit per-site and global normalization stats on train fold only."""
    per_site = {}
    all_nodes = []
    for d in train_data:
        sid = _graph_site_id(d)
        per_site.setdefault(sid, []).append(d.x)
        all_nodes.append(d.x)

    site_stats = {}
    for sid, xs in per_site.items():
        cat = torch.cat(xs, dim=0)
        mean = cat.mean(dim=0, keepdim=True)
        std = cat.std(dim=0, keepdim=True).clamp_min(1e-6)
        site_stats[int(sid)] = (mean, std)

    global_cat = torch.cat(all_nodes, dim=0)
    global_mean = global_cat.mean(dim=0, keepdim=True)
    global_std = global_cat.std(dim=0, keepdim=True).clamp_min(1e-6)
    return site_stats, (global_mean, global_std)


def _apply_site_normalization(graphs, site_stats, global_stats) -> None:
    """Apply per-site normalization with fallback to global stats."""
    global_mean, global_std = global_stats
    for d in graphs:
        sid = _graph_site_id(d)
        mean, std = site_stats.get(int(sid), (global_mean, global_std))
        d.x = (d.x - mean.to(device=d.x.device, dtype=d.x.dtype)) / std.to(
            device=d.x.device, dtype=d.x.dtype
        )


def _site_stats_to_serializable(site_stats):
    """Convert site stats dict to checkpoint-safe lists."""
    means = {}
    stds = {}
    for sid, (mean, std) in site_stats.items():
        means[str(int(sid))] = mean.squeeze(0).detach().cpu().numpy().tolist()
        stds[str(int(sid))] = std.squeeze(0).detach().cpu().numpy().tolist()
    return means, stds


# MAIN TRAINING FUNCTION


def _set_global_seed(seed: int = 42) -> None:
    """Set all random seeds for full reproducibility.

    Sets Python, NumPy, PyTorch, and CUDA seeds.  Also forces cuDNN into
    deterministic mode (deterministic=True, benchmark=False) so that
    convolution algorithms are selected deterministically across runs.
    This trades a small amount of runtime performance for exact reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # CUDA determinism: required for reproducible training on GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Global seed set to %d (cuDNN deterministic mode enabled)", seed)


def _compute_site_auc_values(
    probs: np.ndarray,
    labels: np.ndarray,
    site_ids: np.ndarray,
    min_samples: int = 10,
) -> list:
    """Compute per-site AUC values for sites with enough examples and both classes."""
    site_auc_values = []
    for site in np.unique(site_ids):
        if site < 0:
            continue
        mask = site_ids == site
        if mask.sum() < min_samples:
            continue
        if np.unique(labels[mask]).size < 2:
            continue
        site_auc_values.append(float(roc_auc_score(labels[mask], probs[mask])))
    return site_auc_values


def _assess_graph_degeneracy(dataset) -> dict:
    """Estimate degenerate-graph rate using unified edge/dead-lobe criterion."""
    valid_graphs = 0
    degenerate_graphs = 0
    edge_counts = []
    dead_lobe_counts = []

    for i in range(len(dataset)):
        data = dataset.get(i)
        if data is None:
            continue

        valid_graphs += 1
        stats = summarize_graph_degeneracy_from_edge_index(
            getattr(data, "edge_index", None),
            num_nodes=getattr(data, "num_nodes", NUM_LOBES),
            min_edges=GNN_MIN_EDGES_FOR_NONDEGENERATE,
        )
        edge_counts.append(int(stats["edge_count"]))
        dead_lobe_counts.append(int(stats["dead_lobes"]))

        if bool(stats["is_degenerate"]):
            degenerate_graphs += 1

    degenerate_rate = degenerate_graphs / max(valid_graphs, 1)
    mean_edges = float(np.mean(edge_counts)) if edge_counts else 0.0

    return {
        "valid_graphs": valid_graphs,
        "degenerate_graphs": degenerate_graphs,
        "degenerate_rate": degenerate_rate,
        "mean_edges": mean_edges,
        "mean_dead_lobes": (
            float(np.mean(dead_lobe_counts)) if dead_lobe_counts else 0.0
        ),
    }


def _assess_multiview_quality(multiview_dir: Path, sample_size: int = 0) -> dict:
    """Measure zero-edge rates per multiview branch.

    Returns a dict with checked package count, per-view zero-edge rates, and
    failing views whose zero-edge rate exceeds GNN_MULTIVIEW_MAX_ZERO_EDGE_RATE.
    """
    files = sorted(multiview_dir.glob("*/multiview_graphs.pt"))
    if sample_size > 0 and len(files) > sample_size:
        sample_idx = np.linspace(0, len(files) - 1, sample_size, dtype=int)
        files = [files[i] for i in sample_idx]

    view_order = list(CausalInvarianceLoss._VIEW_ORDER)
    zero_counts = dict.fromkeys(view_order, 0)
    fallback_counts = {view: 0 for view in view_order if view != "base"}
    checked = 0

    for fp in files:
        try:
            payload = torch.load(fp, map_location="cpu", weights_only=True)
        except Exception:
            continue

        if not isinstance(payload, dict):
            continue

        views = payload.get("views", payload)
        fallback_flags = (
            payload.get("fallback_views", {}) if isinstance(payload, dict) else {}
        )
        if not isinstance(views, dict):
            continue

        checked += 1
        for view in view_order:
            adj = views.get(view)
            if adj is None:
                zero_counts[view] += 1
                continue

            if torch.is_tensor(adj):
                adj_t = adj.detach().cpu().float()
            else:
                adj_t = torch.as_tensor(adj, dtype=torch.float32)

            if adj_t.ndim != 2 or adj_t.shape[0] != adj_t.shape[1]:
                zero_counts[view] += 1
                continue

            edge_count = int((adj_t != 0).sum().item())
            if edge_count == 0:
                zero_counts[view] += 1

            if view != "base" and bool(fallback_flags.get(view, False)):
                fallback_counts[view] += 1

    rates = {view: (zero_counts[view] / max(checked, 1)) for view in view_order}
    fallback_rates = {
        view: (fallback_counts[view] / max(checked, 1)) for view in fallback_counts
    }
    failing = [
        view
        for view in view_order
        if view != "base" and rates[view] > GNN_MULTIVIEW_MAX_ZERO_EDGE_RATE
    ]

    return {
        "checked_packages": checked,
        "zero_edge_rates": rates,
        "fallback_rates": fallback_rates,
        "failing_views": failing,
    }


def _run_training_once(
    *,
    use_grl: bool,
    grl_alpha: float,
    checkpoint_dir: Path,
    run_name: str,
    run_post_analysis: bool,
) -> dict:
    """
    Main training loop with k-fold cross-validation.

    Uses modular training utilities for maintainability:
    - EarlyStopping: Prevents overfitting
    - OneCycleLR: Faster convergence with warmup
    - TrainingTracker: Aggregate fold results
    - CheckpointManager: Save/load best models
    """
    from src.features.graph_factory import ABIDECausalDataset, _load_csv_cached

    _set_global_seed(GNN_SEED)

    # Prime CSV/Feather caches before the dataset constructor reads them.
    _load_csv_cached(MASTER_MANIFEST)
    _load_csv_cached(NODE_ATTRIBUTES_HARMONIZED, index_col="subject_id")
    _load_csv_cached(NODE_FEATURES_3D, index_col="subject_id")

    # Load dataset
    dataset = ABIDECausalDataset(split="train")

    # Extract labels for stratification
    labels = []
    site_labels = []
    for i in range(len(dataset)):
        data = dataset.get(i)
        if data is not None:
            labels.append(data.y.item())
            if (
                hasattr(data, "site_id")
                and data.site_id is not None
                and data.site_id.numel() > 0
            ):
                site_labels.append(int(data.site_id.view(-1)[0].item()))
            else:
                site_labels.append(-1)

    if not labels:
        logger.error("No valid training data found!")
        return {
            "run_name": run_name,
            "grl_alpha": grl_alpha,
            "mean_auc": 0.0,
            "site_auc_variance": float("inf"),
            "site_auc_count": 0,
        }

    graph_quality = _assess_graph_degeneracy(dataset)
    logger.info(
        "Graph quality: degenerate=%d/%d (%.1f%%; edge<threshold or dead-lobe), threshold=%d, mean_edges=%.2f, mean_dead_lobes=%.2f",
        graph_quality["degenerate_graphs"],
        graph_quality["valid_graphs"],
        100.0 * graph_quality["degenerate_rate"],
        GNN_MIN_EDGES_FOR_NONDEGENERATE,
        graph_quality["mean_edges"],
        graph_quality.get("mean_dead_lobes", 0.0),
    )
    if (
        GNN_ENFORCE_GRAPH_QUALITY_GATE
        and graph_quality["degenerate_rate"] > GNN_MAX_DEGENERATE_GRAPH_RATE
    ):
        raise RuntimeError(
            "Graph quality gate failed: degenerate graph rate "
            f"{graph_quality['degenerate_rate']:.2%} exceeds "
            f"GNN_MAX_DEGENERATE_GRAPH_RATE={GNN_MAX_DEGENERATE_GRAPH_RATE:.2%}."
        )

    # Initialize tracking
    tracker = TrainingTracker(k_folds=K_FOLDS)
    checkpoint_manager = CheckpointManager(checkpoint_dir, monitor="auc", mode="max")
    experiment_tracker = ExperimentTracker(experiment_name=f"gnn_training_{run_name}")
    experiment_tracker.add_note("use_grl", use_grl)
    experiment_tracker.add_note("grl_alpha", float(grl_alpha))
    experiment_tracker.add_note("checkpoint_dir", str(checkpoint_dir))

    # Initialize training monitor for analysis
    analysis_dir = (
        RESULTS_TRAINING_DIR if run_post_analysis else (RESULTS_TRAINING_DIR / run_name)
    )
    analysis_dir.mkdir(parents=True, exist_ok=True)
    monitor = TrainingMonitor(analysis_dir, num_folds=K_FOLDS)

    # Print configuration
    logger.info(f"\n{'='*70}")
    logger.info("GNN TRAINING - 5-FOLD CROSS-VALIDATION (%s)", run_name)
    logger.info(f"{'='*70}")
    logger.info(f"Total subjects: {len(labels)} (curated ABIDE-I cohort)")
    logger.info(f"OneCycle max LR: {GNN_ONECYCLE_MAX_LR}")
    logger.info(f"Hidden channels: {GNN_HIDDEN_CHANNELS}")
    logger.info(
        f"Input features: {GNN_IN_CHANNELS} (registry count={len(ALL_FEATURE_NAMES)})"
    )
    logger.info(f"Site conditioning: {GNN_USE_SITE_EMBEDDING}")
    logger.info(f"Demographics: {GNN_USE_DEMOGRAPHICS}")
    logger.info(f"GRL enabled: {use_grl} (alpha_max={grl_alpha:.2f})")
    logger.info(f"Early stopping patience: {GNN_EARLY_STOPPING_PATIENCE}")
    if USE_FOCAL_LOSS:
        logger.info(
            f"Loss: FocalLoss (α={FOCAL_LOSS_ALPHA}, γ={FOCAL_LOSS_GAMMA}, class_weights={USE_CLASS_WEIGHTS})"
        )
    else:
        logger.info(f"Loss: CrossEntropy (class_weights={USE_CLASS_WEIGHTS})")
    logger.info(f"Threshold policy: {EVAL_THRESHOLD_POLICY}")
    logger.info("Fold preprocessing mode: %s", GNN_FOLD_PREPROCESSING_MODE)
    if str(GNN_FOLD_PREPROCESSING_MODE).strip().lower() != "legacy_global":
        logger.info(
            "Wave-1 preprocessing: MI feature selection=%s (%.2f-%.2f), normalization=%s",
            GNN_MI_FEATURE_SELECTION_ENABLED,
            GNN_MI_MIN_KEEP_RATIO,
            GNN_MI_MAX_KEEP_RATIO,
            GNN_SITE_NORMALIZATION_MODE,
        )
    logger.info(
        "Aux losses: structural_dropout=%.3f, edge_contrastive=%.3f, "
        "invariance=%.3f, spatial_invariance=%.3f",
        GNN_STRUCTURAL_DROPOUT_PROB,
        GNN_EDGE_CONTRASTIVE_WEIGHT,
        GNN_INVARIANCE_WEIGHT,
        GNN_SPATIAL_INVARIANCE_WEIGHT,
    )
    logger.info(f"{'='*70}\n")

    # K-fold cross-validation (strict manifest-only enforcement)
    if "cv_fold" not in dataset.manifest.columns:
        raise ValueError(
            "cv_fold column not found in manifest. "
            "Run split.py first to generate predefined CV folds."
        )

    cv_folds = dataset.manifest["cv_fold"].values
    if cv_folds.min() < 0 or cv_folds.max() >= K_FOLDS:
        raise ValueError(
            f"Invalid cv_fold values: found [{cv_folds.min()}, {cv_folds.max()}], "
            f"expected [0, {K_FOLDS-1}]. Run split.py to regenerate folds."
        )

    # Build fold splits from manifest (aligned with harmonization)
    cv_splits = []
    for f in range(K_FOLDS):
        t_idx = np.where(cv_folds != f)[0]
        v_idx = np.where(cv_folds == f)[0]
        cv_splits.append((t_idx, v_idx))
        logger.debug(f"Fold {f}: train={len(t_idx)}, val={len(v_idx)}")

    base_subject_ids = [str(s) for s in dataset.subject_ids]
    # Hard assertion: fold-harmonized files must exist if we reach training
    missing_fold_files = [
        f
        for f in range(K_FOLDS)
        if not (HARMONIZED_FOLDS_DIR / f"harmonized_fold_{f}.csv").exists()
    ]
    if missing_fold_files:
        missing_details = ", ".join(
            f"fold {fold} (harmonized_fold_{fold}.csv)" for fold in missing_fold_files
        )
        raise FileNotFoundError(
            "Missing fold-specific harmonized files: "
            f"{missing_details}. Directory: {HARMONIZED_FOLDS_DIR}. "
            "Run fold_safe_harmonization.py before gnn_training."
        )

    fold_audit_path = HARMONIZED_FOLDS_DIR / "fold_unseen_site_audit.csv"
    if fold_audit_path.exists():
        try:
            fold_audit = pd.read_csv(fold_audit_path)
            required_cols = {"unseen_row_count", "val_row_count"}
            if required_cols.issubset(set(fold_audit.columns)) and not fold_audit.empty:
                val_rows = (
                    pd.to_numeric(fold_audit["val_row_count"], errors="coerce")
                    .fillna(0)
                    .astype(int)
                )
                unseen_rows = (
                    pd.to_numeric(fold_audit["unseen_row_count"], errors="coerce")
                    .fillna(0)
                    .astype(int)
                )
                all_unseen = bool(
                    (val_rows > 0).all() and (unseen_rows == val_rows).all()
                )
                if all_unseen:
                    raise RuntimeError(
                        "Detected fold_unseen_site_audit with 100% unseen validation rows in all folds. "
                        "This indicates site-stratified CV with fold-safe harmonization mismatch. "
                        "Recommended (Option A): run standard StratifiedKFold CV and regenerate "
                        "fold harmonization artifacts without --site-stratified-cv."
                    )
        except RuntimeError:
            raise
        except Exception as exc:
            logger.warning(
                "Failed to parse unseen-site audit (%s): %s", fold_audit_path, exc
            )

    multiview_present = CAUSAL_GRAPHS_MULTIVIEW_DIR.exists() and any(
        CAUSAL_GRAPHS_MULTIVIEW_DIR.glob("*/multiview_graphs.pt")
    )

    multiview_available = multiview_present
    if multiview_present and GNN_ENFORCE_MULTIVIEW_QUALITY_GATE:
        quality = _assess_multiview_quality(
            CAUSAL_GRAPHS_MULTIVIEW_DIR,
            sample_size=GNN_MULTIVIEW_QUALITY_SAMPLE_SIZE,
        )
        checked = quality["checked_packages"]
        rates = quality["zero_edge_rates"]
        logger.info(
            "Multiview quality: checked=%d | base=%.1f%% | ext=%.1f%% | b0=%.1f%% | b1=%.1f%% | b2=%.1f%% | hc=%.1f%%",
            checked,
            100.0 * rates.get("base", 1.0),
            100.0 * rates.get("extended_lag", 1.0),
            100.0 * rates.get("bootstrap_0", 1.0),
            100.0 * rates.get("bootstrap_1", 1.0),
            100.0 * rates.get("bootstrap_2", 1.0),
            100.0 * rates.get("high_confidence", 1.0),
        )
        fallback_rates = quality.get("fallback_rates", {})
        if fallback_rates:
            logger.info(
                "Multiview fallback rates: ext=%.1f%% | b0=%.1f%% | b1=%.1f%% | b2=%.1f%% | hc=%.1f%%",
                100.0 * fallback_rates.get("extended_lag", 0.0),
                100.0 * fallback_rates.get("bootstrap_0", 0.0),
                100.0 * fallback_rates.get("bootstrap_1", 0.0),
                100.0 * fallback_rates.get("bootstrap_2", 0.0),
                100.0 * fallback_rates.get("high_confidence", 0.0),
            )
        if checked == 0 or quality["failing_views"]:
            multiview_available = False
            logger.warning(
                "Disabling multiview invariance: degenerate views exceed max zero-edge rate %.2f. "
                "Failing views=%s",
                GNN_MULTIVIEW_MAX_ZERO_EDGE_RATE,
                quality["failing_views"],
            )

    invariance_enabled = multiview_available and GNN_INVARIANCE_WEIGHT > 0.0
    invariance_criterion = (
        CausalInvarianceLoss(temperature=0.07) if invariance_enabled else None
    )
    if invariance_enabled:
        logger.info(
            "Multi-view causal graphs detected — enabling CausalInvarianceLoss (weight=%.3f).",
            GNN_INVARIANCE_WEIGHT,
        )
    elif multiview_available:
        logger.info(
            "Multi-view graphs detected but GNN_INVARIANCE_WEIGHT=%.3f, so invariance loss is disabled.",
            GNN_INVARIANCE_WEIGHT,
        )
    else:
        if multiview_present:
            logger.info(
                "Multi-view graphs detected but disabled by quality gate — training uses single-view objective."
            )
        else:
            logger.info(
                "No multi-view graphs detected — training falls back to standard single-view objective."
            )

    site_auc_values = []
    unique_sites = sorted({s for s in site_labels if s >= 0})
    num_sites_detected = max((max(unique_sites) + 1) if unique_sites else 1, 1)
    logger.info(
        "Detected %d unique site IDs in training split (classifier size=%d)",
        len(unique_sites),
        num_sites_detected,
    )

    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        logger.info(f"\n{'='*70}")
        logger.info(f"FOLD {fold+1}/{K_FOLDS}")
        logger.info(f"{'='*70}")

        _set_global_seed(
            GNN_SEED
        )  # deterministic initialisation (same seed for all folds)
        fold_start_time = time.time()

        # Enforce fold-specific harmonized features (no global fallback).
        fold_temporal_path = HARMONIZED_FOLDS_DIR / f"harmonized_fold_{fold}.csv"
        candidate_dataset = ABIDECausalDataset(
            split="train",
            temporal_features_path=fold_temporal_path,
        )
        candidate_subject_ids = [str(s) for s in candidate_dataset.subject_ids]
        if candidate_subject_ids != base_subject_ids:
            logger.warning(
                f"Fold {fold} harmonized file subject ordering differs from base dataset; "
                f"this may be due to NaN-filtering differences. Proceeding with fold-specific data."
            )
            # Still verify no duplicate or invalid subjects
            if len(set(candidate_subject_ids)) != len(candidate_subject_ids):
                raise ValueError(
                    f"Fold {fold} has duplicate subjects in harmonized file"
                )
        fold_dataset = candidate_dataset
        logger.info("Using fold-specific harmonized features: %s", fold_temporal_path)

        # Create fold-local data copies (avoid in-place mutation leaking across folds)
        train_data = []
        for i in train_idx:
            sample = fold_dataset[i]
            if sample is not None:
                train_data.append(sample.clone())

        val_data = []
        for i in val_idx:
            sample = fold_dataset[i]
            if sample is not None:
                val_data.append(sample.clone())

        train_labels = [d.y.item() for d in train_data]
        val_labels = [d.y.item() for d in val_data]
        val_site_ids = np.array(
            [
                (
                    int(d.site_id.view(-1)[0].item())
                    if hasattr(d, "site_id") and d.site_id is not None
                    else -1
                )
                for d in val_data
            ]
        )

        logger.info(
            f"Train: Control={train_labels.count(0)}, ASD={train_labels.count(1)}"
        )
        logger.info(f"Val: Control={val_labels.count(0)}, ASD={val_labels.count(1)}")

        preprocess_mode = str(GNN_FOLD_PREPROCESSING_MODE).strip().lower()
        site_norm_mode = str(GNN_SITE_NORMALIZATION_MODE).strip().lower()
        feature_mask = torch.ones(GNN_IN_CHANNELS, dtype=torch.float32)
        selected_feature_idx = list(range(GNN_IN_CHANNELS))
        feature_selection_meta = {
            "enabled": False,
            "original_features": int(GNN_IN_CHANNELS),
            "selected_features": int(GNN_IN_CHANNELS),
            "selected_ratio": 1.0,
            "min_allowed": int(GNN_IN_CHANNELS),
            "max_allowed": int(GNN_IN_CHANNELS),
            "candidate_k": int(GNN_IN_CHANNELS),
        }
        site_feature_means = {}
        site_feature_stds = {}

        if preprocess_mode == "legacy_global":
            if train_data:
                train_x = torch.cat([d.x for d in train_data], dim=0)
                feat_mean = train_x.mean(dim=0, keepdim=True)
                feat_std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
                for d in train_data:
                    d.x = (d.x - feat_mean) / feat_std
                for d in val_data:
                    d.x = (d.x - feat_mean) / feat_std
                logger.info(
                    "Applied legacy fold-global feature standardization (train-fit only)"
                )
            else:
                feat_mean = torch.zeros((1, GNN_IN_CHANNELS), dtype=torch.float32)
                feat_std = torch.ones((1, GNN_IN_CHANNELS), dtype=torch.float32)
        else:
            if GNN_MI_FEATURE_SELECTION_ENABLED and train_data:
                selected_feature_idx, feature_mask, feature_selection_meta = (
                    _fit_mi_feature_selection(train_data)
                )
                n_retained = int(feature_mask.sum().item())
                logger.info(
                    "Fold %d: MI selection retained %d/%d features",
                    fold,
                    n_retained,
                    GNN_IN_CHANNELS,
                )
                safety_floor = max(8, int(0.30 * GNN_IN_CHANNELS))
                if n_retained < safety_floor:
                    logger.error(
                        "Fold %d: MI selection retained %d/%d features (safety floor=%d); falling back to all features",
                        fold,
                        n_retained,
                        GNN_IN_CHANNELS,
                        safety_floor,
                    )
                    feature_mask = torch.ones(GNN_IN_CHANNELS, dtype=torch.float32)
                    selected_feature_idx = list(range(GNN_IN_CHANNELS))
                    feature_selection_meta = {
                        **feature_selection_meta,
                        "selected_features": int(GNN_IN_CHANNELS),
                        "selected_ratio": 1.0,
                        "fallback_to_all_features": True,
                    }
                _apply_feature_mask(train_data, feature_mask)
                _apply_feature_mask(val_data, feature_mask)
                logger.info(
                    "Applied fold MI feature mask: kept %d/%d features (%.1f%%)",
                    feature_selection_meta["selected_features"],
                    feature_selection_meta["original_features"],
                    100.0 * feature_selection_meta["selected_ratio"],
                )

            if not train_data:
                feat_mean = torch.zeros((1, GNN_IN_CHANNELS), dtype=torch.float32)
                feat_std = torch.ones((1, GNN_IN_CHANNELS), dtype=torch.float32)
            elif site_norm_mode == "within_site":
                site_stats, (feat_mean, feat_std) = _fit_site_normalization_stats(
                    train_data
                )
                _apply_site_normalization(train_data, site_stats, (feat_mean, feat_std))
                _apply_site_normalization(val_data, site_stats, (feat_mean, feat_std))
                site_feature_means, site_feature_stds = _site_stats_to_serializable(
                    site_stats
                )
                logger.info(
                    "Applied fold within-site normalization (train-fit only, %d site profiles)",
                    len(site_feature_means),
                )
            elif site_norm_mode == "global":
                train_x = torch.cat([d.x for d in train_data], dim=0)
                feat_mean = train_x.mean(dim=0, keepdim=True)
                feat_std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
                for d in train_data:
                    d.x = (d.x - feat_mean) / feat_std
                for d in val_data:
                    d.x = (d.x - feat_mean) / feat_std
                logger.info("Applied fold-global normalization (train-fit only)")
            elif site_norm_mode == "none":
                feat_mean = torch.zeros((1, GNN_IN_CHANNELS), dtype=torch.float32)
                feat_std = torch.ones((1, GNN_IN_CHANNELS), dtype=torch.float32)
                logger.info("Skipped fold feature normalization (mode=none)")
            else:
                raise ValueError(
                    f"Unknown GNN_SITE_NORMALIZATION_MODE={GNN_SITE_NORMALIZATION_MODE!r}. "
                    "Expected one of: within_site, global, none."
                )

        train_loader = make_loader(train_data, batch_size=GNN_BATCH_SIZE, shuffle=True)
        val_loader = make_loader(val_data, batch_size=GNN_BATCH_SIZE)

        # Initialize model
        model = build_model(
            device=DEVICE,
            use_grl=use_grl,
            grl_alpha=grl_alpha,
        )

        # Loss function
        n_control = max((np.array(train_labels) == 0).sum(), 1)
        n_asd = max((np.array(train_labels) == 1).sum(), 1)
        if USE_CLASS_WEIGHTS:
            total = len(train_labels)
            weight_control = total / (2 * n_control)
            weight_asd = total / (2 * n_asd)
            torch.tensor([weight_control, weight_asd], dtype=torch.float32).to(DEVICE)
            logger.info(f"Class distribution: Control={n_control}, ASD={n_asd}")
            logger.info(
                f"Class weights: Control={weight_control:.3f}, ASD={weight_asd:.3f}"
            )

        criterion = build_criterion(
            train_labels,
            device=DEVICE,
            use_focal_loss=USE_FOCAL_LOSS,
            use_class_weights=USE_CLASS_WEIGHTS,
            focal_alpha=FOCAL_LOSS_ALPHA,
            focal_gamma=FOCAL_LOSS_GAMMA,
        )
        checkpoint_manager.reset()

        # Task 4: residual site signal adversarial regularization on spatial channels.
        spatial_invariance_criterion = SpatialInvarianceLoss(
            spatial_start_idx=GNN_IN_CHANNELS - NUM_SPATIAL_FEATURES,
            num_sites=num_sites_detected,
            reversal_weight=1.0,
        )

        best_state, best_metrics, history = train_fold_with_onecycle(
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
            pct_start=GNN_ONECYCLE_WARMUP_FRACTION,
            grl_alpha_max=grl_alpha,
            # Task 1: structural learning enforcement (DD-009)
            structural_dropout_prob=GNN_STRUCTURAL_DROPOUT_PROB,
            edge_contrastive_weight=GNN_EDGE_CONTRASTIVE_WEIGHT,
            # Task 2: multi-view causal invariance (DD-010)
            invariance_loss_fn=invariance_criterion,
            invariance_weight=GNN_INVARIANCE_WEIGHT if invariance_enabled else 0.0,
            multiview_dir=CAUSAL_GRAPHS_MULTIVIEW_DIR if invariance_enabled else None,
            # Task 4: spatial-channel adversarial invariance (DD-012)
            spatial_invariance_loss_fn=spatial_invariance_criterion,
            spatial_invariance_weight=GNN_SPATIAL_INVARIANCE_WEIGHT,
        )

        for entry in history:
            if entry["epoch"] % 10 == 0:
                monitor.log_epoch(
                    fold_id=fold,
                    epoch=entry["epoch"],
                    metrics={
                        "train_loss": entry["train_loss"],
                        "val_loss": entry["val_loss"],
                        "val_inverse_auc": 1.0
                        - entry["auc"],  # inverse AUC for monitoring
                        "val_auc": entry["auc"],
                        "val_auprc": entry["auprc"],
                        "val_f1": entry["f1"],
                        "val_acc": entry["acc"],
                        "lr": entry["lr"],
                    },
                    grad_norm=entry.get("grad_norm", 0.0),
                    confusion_matrix=entry["cm"],
                )
                logger.info(
                    f"Epoch {entry['epoch']:03d} | LR: {entry['lr']:.6f} | Loss: {entry['train_loss']:.4f} | "
                    f"AUC: {entry['auc']:.4f} | AUPRC: {entry['auprc']:.4f} | "
                    f"F1@{entry['threshold']:.2f}: {entry['f1']:.4f}"
                )

        model.load_state_dict(best_state)
        checkpoint_metrics = {
            "auc": best_metrics["auc"],
            "auprc": best_metrics["auprc"],
            "f1": best_metrics["f1"],
            "threshold": best_metrics["threshold"],
            # Persist fold-wise preprocessing metadata for inference-time parity.
            "feature_mean": feat_mean.squeeze(0).cpu().numpy().tolist(),
            "feature_std": feat_std.squeeze(0).cpu().numpy().tolist(),
            "feature_mask": feature_mask.cpu().numpy().tolist(),
            "selected_feature_idx": [int(i) for i in selected_feature_idx],
            "feature_selection_meta": feature_selection_meta,
            "site_feature_means": site_feature_means,
            "site_feature_stds": site_feature_stds,
            "preprocessing_mode": preprocess_mode,
            "site_normalization_mode": site_norm_mode,
        }
        checkpoint_manager.save(
            model, None, best_metrics["best_epoch"], checkpoint_metrics, fold=fold
        )

        logger.info(
            f"✓ Best fold {fold}: AUC={best_metrics['auc']:.4f}, "
            f"AUPRC={best_metrics['auprc']:.4f}, F1={best_metrics['f1']:.4f}"
        )

        # Final evaluation with best checkpoint
        final_threshold = best_metrics["threshold"]
        final_metrics = evaluate(model, val_loader, threshold=final_threshold)
        site_auc_values.extend(
            _compute_site_auc_values(
                final_metrics["probs"],
                final_metrics["labels"],
                val_site_ids,
            )
        )
        best_epoch = best_metrics["best_epoch"]

        fold_train_time = time.time() - fold_start_time

        # Log fold results
        logger.info(f"\nFold {fold+1} Final Results:")
        logger.info(f"  Best epoch: {best_epoch}")
        logger.info(f"  Training time: {fold_train_time:.1f}s")
        logger.info(f"  AUC: {final_metrics['auc']:.4f}")
        logger.info(
            f"  F1: {final_metrics['f1']:.4f} (threshold={final_threshold:.3f})"
        )
        logger.info(f"  Accuracy: {final_metrics['acc']:.4f}")
        logger.info("  Confusion Matrix:")
        logger.info(f"    {final_metrics['cm']}")

        # Track results
        tracker.add_fold_result(
            fold=fold,
            auc=final_metrics["auc"],
            f1=final_metrics["f1"],
            acc=final_metrics["acc"],
            threshold=final_threshold,
            best_epoch=best_epoch,
            train_time=fold_train_time,
            val_probs=final_metrics["probs"],
            val_labels=final_metrics["labels"],
        )
        experiment_tracker.log_fold(
            fold=fold,
            metrics={
                "auc": float(final_metrics["auc"]),
                "f1": float(final_metrics["f1"]),
                "acc": float(final_metrics["acc"]),
                "threshold": float(final_threshold),
                "best_epoch": int(best_epoch),
                "train_time_sec": float(fold_train_time),
            },
        )

        # Generate training visualizations for this fold
        if run_post_analysis:
            logger.info("\nGenerating fold visualizations...")
            plot_path = monitor.plot_training_curves(fold)
            logger.info(f"  Training curves saved to: {plot_path}")

            history_path = monitor.save_history(fold)
            logger.info(f"  Training history saved to: {history_path}")

    # Log cross-validation summary
    tracker.log_summary()
    summary = tracker.get_summary()
    site_auc_variance = (
        float(np.var(site_auc_values)) if site_auc_values else float("inf")
    )
    experiment_tracker.finalize(
        {
            **summary,
            "run_name": run_name,
            "grl_alpha": float(grl_alpha),
            "site_auc_variance": site_auc_variance,
            "site_auc_count": len(site_auc_values),
        }
    )
    logger.info(
        "Per-site validation AUC variance (%s): %.6f from %d site-level AUC values",
        run_name,
        site_auc_variance,
        len(site_auc_values),
    )

    # POST-TRAINING ANALYSIS
    if run_post_analysis:
        logger.info(f"\n{'='*70}")
        logger.info("POST-TRAINING ANALYSIS")
        logger.info(f"{'='*70}\n")

    # 1. Feature Attribution Analysis (if Captum available)
    if run_post_analysis and FEATURE_ANALYSIS_AVAILABLE:
        try:
            logger.info("Running feature attribution analysis...")
            from src.features.graph_factory import ABIDECausalDataset

            # Load test set
            test_dataset = ABIDECausalDataset(split="test")
            test_loader = make_loader(
                [d for d in test_dataset if d is not None], batch_size=GNN_BATCH_SIZE
            )

            # Define feature names (8 temporal + 6 spatial)
            feature_names = ALL_FEATURE_NAMES.copy()
            if len(feature_names) != GNN_IN_CHANNELS:
                logger.warning(
                    f"Feature name count ({len(feature_names)}) does not match "
                    f"GNN_IN_CHANNELS ({GNN_IN_CHANNELS}). Adjusting list for attribution."
                )
                if len(feature_names) > GNN_IN_CHANNELS:
                    feature_names = feature_names[:GNN_IN_CHANNELS]
                else:
                    missing = GNN_IN_CHANNELS - len(feature_names)
                    feature_names.extend([f"feature_{i+1}" for i in range(missing)])

            # Load best model (fold 0 as representative)
            best_model = build_model(
                device=DEVICE,
                use_grl=use_grl,
                grl_alpha=grl_alpha,
            )
            checkpoint_manager.load(best_model, fold=0, allow_partial=True)

            # Compute feature attributions
            feature_analyzer = FeatureAttributionAnalyzer(
                best_model, test_loader, feature_names, device=DEVICE
            )
            attributions = feature_analyzer.compute_attributions()

            # Visualize and save
            feature_output = analysis_dir / "features"
            feature_output.mkdir(parents=True, exist_ok=True)
            feature_analyzer.visualize_feature_importance(
                attributions, feature_output / "feature_importance.png"
            )
            logger.info(
                f"  Feature importance plot saved to: {feature_output / 'feature_importance.png'}"
            )

        except Exception as e:
            logger.warning(f"Feature attribution analysis failed: {e}")

    # 2. Causal Graph Analysis
    if run_post_analysis:
        try:
            logger.info("\nRunning causal graph analysis...")

            # Load manifest
            manifest_path = DATA_METADATA / "master_manifest.csv"
            manifest = pd.read_csv(manifest_path)

            # Compute graph properties
            graph_analyzer = CausalGraphAnalyzer(CAUSAL_GRAPHS_DIR, manifest)
            graph_metrics = graph_analyzer.compute_graph_properties()

            # Compare ASD vs Control
            graph_output = analysis_dir / "graphs"
            graph_output.mkdir(parents=True, exist_ok=True)
            graph_analyzer.compare_asd_vs_control(graph_metrics, graph_output)
            logger.info(f"  Graph analysis plots saved to: {graph_output}")

        except Exception as e:
            logger.warning(f"Causal graph analysis failed: {e}")

    if run_post_analysis:
        logger.info(f"\n{'='*70}")
        logger.info("TRAINING AND ANALYSIS COMPLETE")
        logger.info(f"{'='*70}\n")

    return {
        "run_name": run_name,
        "grl_alpha": grl_alpha,
        "mean_auc": float(summary.get("mean_auc", 0.0)),
        "site_auc_variance": site_auc_variance,
        "site_auc_count": len(site_auc_values),
    }


def run_training():
    """Entry point for model training with optional GRL alpha grid search."""
    if GNN_AUTO_GRL_GRID_SEARCH:
        _GRL_CANDIDATES = [0.10, 0.25, 0.50, 1.0]
        logger.info("Starting GRL alpha grid search: %s", _GRL_CANDIDATES)
        candidate_results = []

        for alpha in _GRL_CANDIDATES:
            run_name = f"grl_alpha_{alpha:.2f}"
            candidate_dir = CHECKPOINT_DIR / run_name
            result = _run_training_once(
                use_grl=True,
                grl_alpha=float(alpha),
                checkpoint_dir=candidate_dir,
                run_name=run_name,
                run_post_analysis=False,
            )
            candidate_results.append(result)
            logger.info(
                "Candidate α=%.2f -> mean AUC=%.4f, site-AUC variance=%.6f",
                alpha,
                result["mean_auc"],
                result["site_auc_variance"],
            )

        if not candidate_results:
            logger.warning(
                "GRL grid search produced no valid results; falling back to config defaults"
            )
            return _run_training_once(
                use_grl=GNN_USE_GRL,
                grl_alpha=GNN_GRL_ALPHA,
                checkpoint_dir=CHECKPOINT_DIR,
                run_name="default",
                run_post_analysis=True,
            )

        best_mean_auc = max(r["mean_auc"] for r in candidate_results)
        viable = [r for r in candidate_results if r["mean_auc"] >= best_mean_auc - 0.01]
        selected = (
            min(viable, key=lambda r: r["site_auc_variance"])
            if viable
            else max(
                candidate_results,
                key=lambda r: r["mean_auc"],
            )
        )
        selected_alpha = float(selected["grl_alpha"])

        logger.info(
            "Selected GRL alpha %.2f (mean AUC=%.4f, site-AUC variance=%.6f)",
            selected_alpha,
            selected["mean_auc"],
            selected["site_auc_variance"],
        )

        return _run_training_once(
            use_grl=True,
            grl_alpha=selected_alpha,
            checkpoint_dir=CHECKPOINT_DIR,
            run_name=f"grl_selected_{selected_alpha:.2f}",
            run_post_analysis=True,
        )

    return _run_training_once(
        use_grl=GNN_USE_GRL,
        grl_alpha=GNN_GRL_ALPHA,
        checkpoint_dir=CHECKPOINT_DIR,
        run_name="default",
        run_post_analysis=True,
    )


# CLI


def parse_args():
    """Parse command-line arguments for GNN training.

    Returns:
        Parsed arguments with optional seed override.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Neuro-CXG GNN Training")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: uses GNN_SEED from config)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Apply seed override if provided
    if args.seed is not None:
        GNN_SEED = args.seed
        logger.info(f"Using CLI-provided seed: {GNN_SEED}")

    run_training()
