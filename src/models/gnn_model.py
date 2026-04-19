import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, confusion_matrix,
    accuracy_score, roc_curve
)
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import sys
import warnings
import time

warnings.filterwarnings('ignore')

# Setup paths and config
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import (
    K_FOLDS, GNN_BATCH_SIZE, GNN_EPOCHS, 
    CHECKPOINT_DIR, DEVICE, GNN_IN_CHANNELS,
    ALL_FEATURE_NAMES,
    GNN_HIDDEN_CHANNELS,
    GNN_DROPOUT,
    GNN_WEIGHT_DECAY,
    GNN_NUM_HEADS,
    GNN_NUM_LAYERS,
    GNN_POOLING,
    GNN_USE_GRL,
    GNN_GRL_ALPHA,
    GNN_AUTO_GRL_GRID_SEARCH,
    GRL_ALPHA_CANDIDATES,
    GNN_SITE_LOSS_WEIGHT,
    GNN_EDGE_GATE,
    GNN_ONECYCLE_MAX_LR,
    GNN_ONECYCLE_WARMUP_FRACTION,
    GNN_EARLY_STOPPING_PATIENCE,
    GNN_STRUCTURAL_DROPOUT_PROB,
    GNN_EDGE_CONTRASTIVE_WEIGHT,
    GNN_INVARIANCE_WEIGHT,
    GNN_SPATIAL_INVARIANCE_WEIGHT,
    GNN_ENFORCE_MULTIVIEW_QUALITY_GATE,
    GNN_MULTIVIEW_MAX_ZERO_EDGE_RATE,
    GNN_MULTIVIEW_QUALITY_SAMPLE_SIZE,
    GNN_ENFORCE_GRAPH_QUALITY_GATE,
    GNN_MAX_DEGENERATE_GRAPH_RATE,
    GNN_MIN_EDGES_FOR_NONDEGENERATE,
    FOCAL_LOSS_ALPHA,
    FOCAL_LOSS_GAMMA,
    USE_FOCAL_LOSS,
    USE_CLASS_WEIGHTS,
    EVAL_THRESHOLD_POLICY,
    GNN_USE_SITE_EMBEDDING,
    GNN_USE_DEMOGRAPHICS,
    GNN_NODE_EMB_DIM,
    NUM_LOBES,
    NUM_SPATIAL_FEATURES,
    CAUSAL_GRAPHS_DIR,
    CAUSAL_GRAPHS_MULTIVIEW_DIR,
    MASTER_MANIFEST,
    NODE_ATTRIBUTES_HARMONIZED,
    NODE_FEATURES_3D,
    DATA_METADATA,
    RESULTS_TRAINING_DIR,
    HARMONIZED_FOLDS_DIR,
)
from src.core.validators import summarize_graph_degeneracy_from_edge_index
from src.models.training_utils import (
    TrainingTracker,
    CheckpointManager,
    make_loader,
    train_fold_with_onecycle,
)
from src.core.experiment_tracker import ExperimentTracker
from src.models.evaluation import evaluate_loader, optimal_threshold
from src.models.factory import build_model
from src.models.losses import FocalLoss

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Analysis modules
from src.analysis.diagnostics import CausalGraphAnalyzer, TrainingMonitor
try:
    from src.analysis.feature_attribution import FeatureAttributionAnalyzer
    FEATURE_ANALYSIS_AVAILABLE = True
except ImportError:
    FEATURE_ANALYSIS_AVAILABLE = False
    logger.warning("FeatureAttributionAnalyzer unavailable (requires Captum)")


# ── TASK 2: Causal Invariance Loss (DD-010) ──────────────────────────────────────

class CausalInvarianceLoss(nn.Module):
    """
    NT-Xent contrastive loss across multiple causal graph views of the same subject.

    Rationale (DD-010 / Root Cause 2): A single noisy Granger estimate per subject
    means one bad estimation fails entirely.  Multi-view construction generates 6
    views per subject (base, extended_lag, 3 bootstraps, high_confidence).  This
    loss enforces that graph embeddings are invariant across views of the same
    subject, making the classifier robust to estimation noise.

    Positive pairs:  different views of the SAME subject.
    Negative pairs:  views from DIFFERENT subjects.
    Temperature τ = 0.07 (tight, following SimCLR convention for small batches).
    """

    _VIEW_ORDER = (
        "base",
        "extended_lag",
        "bootstrap_0",
        "bootstrap_1",
        "bootstrap_2",
        "high_confidence",
    )

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings_list: list) -> torch.Tensor:
        """
        Args:
            embeddings_list: List of V tensors each of shape (B, D), one per view.
                             B graphs must correspond to the same subjects across views.
        Returns:
            Scalar NT-Xent invariance loss.
        """
        V = len(embeddings_list)
        if V < 2:
            return torch.tensor(0.0, device=embeddings_list[0].device, requires_grad=True)

        B = embeddings_list[0].size(0)
        if B < 2:
            return torch.tensor(0.0, device=embeddings_list[0].device, requires_grad=True)

        # Normalize all views
        zs = [F.normalize(e, dim=1) for e in embeddings_list]

        # Symmetric NT-Xent across ALL view pairs
        pair_losses = []
        labels = torch.arange(B, device=zs[0].device)
        for i in range(V):
            for j in range(i + 1, V):
                sim_ij = torch.mm(zs[i], zs[j].t()) / self.temperature  # (B, B)
                pair_losses.append(F.cross_entropy(sim_ij, labels))
                pair_losses.append(F.cross_entropy(sim_ij.t(), labels))

        if not pair_losses:
            return torch.tensor(0.0, device=zs[0].device, requires_grad=True)

        return torch.stack(pair_losses).mean()


# ── TASK 4: Spatial Invariance Loss (DD-012) ─────────────────────────────────────

class SpatialInvarianceLoss(nn.Module):
    """
    Gradient reversal applied to the spatial feature slice of node features.

    Rationale (DD-012 / Root Cause 4): Even after removing conf_std and
    detection_count, the remaining spatial features (x, y, z_depth, size) can
    carry residual site-correlated variance from scanner FOV differences.
    A reversed gradient on the spatial channels penalises the encoder for
    extracting site-predictive information from spatial coordinates.

    Args:
        spatial_start_idx: First column index of the spatial feature block in x.
                           Default: GNN_IN_CHANNELS - NUM_SPATIAL_FEATURES.
        num_sites: Number of acquisition sites for the site classifier head.
        reversal_weight: Gradient reversal strength (λ). Default 0.1.
    """

    def __init__(self, spatial_start_idx: int, num_sites: int = 20, reversal_weight: float = 0.1):
        super().__init__()
        self.spatial_start_idx = spatial_start_idx
        self.reversal_weight = reversal_weight
        self.site_head = nn.Sequential(
            nn.Linear(NUM_SPATIAL_FEATURES, 16),
            nn.GELU(),
            nn.Linear(16, num_sites),
        )

    def forward(self, x: torch.Tensor, site_targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node feature matrix (N, F).
            site_targets: Integer site labels per node (N,).
        Returns:
            Scalar site classification loss on reversed-gradient spatial features.
        """
        from src.models.causal_gnn import GradientReversal
        spatial = x[:, self.spatial_start_idx:]
        spatial_rev = GradientReversal.apply(spatial, self.reversal_weight)
        site_logits = self.site_head(spatial_rev)
        return F.cross_entropy(site_logits, site_targets)


# UTILITY FUNCTIONS

def compute_class_weights(labels):
    """
    Compute inverse frequency class weights.
    
    Args:
        labels: List of labels (0=Control, 1=ASD)
    
    Returns:
        Tensor of class weights [weight_control, weight_asd]
    """
    labels_array = np.array(labels)
    n_control = (labels_array == 0).sum()
    n_asd = (labels_array == 1).sum()
    total = len(labels_array)
    
    weight_control = total / (2 * n_control) if n_control > 0 else 1.0
    weight_asd = total / (2 * n_asd) if n_asd > 0 else 1.0
    
    logger.info(f"Class distribution: Control={n_control}, ASD={n_asd}")
    logger.info(f"Class weights: Control={weight_control:.3f}, ASD={weight_asd:.3f}")
    
    return torch.tensor([weight_control, weight_asd], dtype=torch.float32)


def find_optimal_threshold(y_true, y_probs):
    """Compatibility wrapper around the shared threshold utility.

    Uses EVAL_THRESHOLD_POLICY so training/evaluation report the same operating
    point family ("f1" or "youden").
    """
    policy = str(EVAL_THRESHOLD_POLICY).strip().lower()
    if policy == "youden":
        if len(np.unique(y_true)) < 2:
            return 0.5, 0.0
        fpr, tpr, thresholds = roc_curve(y_true, y_probs)
        if thresholds.size == 0:
            return 0.5, 0.0
        j = tpr - fpr
        best_idx = int(np.argmax(j))
        thr = float(thresholds[best_idx]) if np.isfinite(thresholds[best_idx]) else 0.5
        return thr, float(j[best_idx])
    return optimal_threshold(y_probs, y_true)


@torch.no_grad()
def evaluate(model, loader, threshold=0.5):
    """Compatibility wrapper around shared loader evaluation."""
    return evaluate_loader(model, loader, DEVICE, threshold=threshold)


def evaluate_ensemble(
    tracker: TrainingTracker,
    checkpoint_manager: CheckpointManager,
    use_grl: bool,
    grl_alpha: float,
):
    """
    Evaluate ensemble of all fold models on test set.
    
    Args:
        tracker: TrainingTracker with fold results
        checkpoint_manager: CheckpointManager for loading models
    """
    from src.features.graph_factory import ABIDECausalDataset
    
    logger.info(f"\n{'='*70}")
    logger.info("ENSEMBLE EVALUATION (TEST SET)")
    logger.info(f"{'='*70}")
    
    try:
        # Load test dataset
        test_dataset = ABIDECausalDataset(split='test')
        test_data = [test_dataset[i] for i in range(len(test_dataset)) if test_dataset[i] is not None]
        
        if len(test_data) == 0:
            logger.warning("Test set is empty, skipping ensemble evaluation")
            return
        
        test_loader = make_loader(test_data, batch_size=GNN_BATCH_SIZE)
        
        # Collect predictions from all folds
        test_fold_probs = []
        test_labels_ref = None
        fold_aucs = []
        
        for fold in range(K_FOLDS):
            # Initialize model
            model = build_model(
                device=DEVICE,
                use_grl=use_grl,
                grl_alpha=grl_alpha,
            )
            try:
                checkpoint = checkpoint_manager.load(model, fold=fold)
                threshold = checkpoint.get('threshold', 0.5)
                fold_auc = checkpoint.get('auc', 0.0)
                fold_aucs.append(fold_auc)
            except FileNotFoundError:
                logger.warning(f"Checkpoint for fold {fold} not found, skipping")
                continue
            
            # Evaluate on test set
            metrics = evaluate(model, test_loader, threshold=threshold)
            test_fold_probs.append(metrics['probs'])
            
            if test_labels_ref is None:
                test_labels_ref = metrics['labels']
        
        if not test_fold_probs:
            logger.warning("No fold predictions collected")
            return

        if test_labels_ref is None or len(test_labels_ref) == 0:
            logger.warning("No reference labels collected from test set")
            return

        # Compute weighted ensemble
        prob_matrix = np.stack(test_fold_probs, axis=0)  # (K_folds, N_samples)
        
        # Weight by validation AUC
        weights = np.array(fold_aucs)
        if np.all(np.isfinite(weights)) and weights.sum() > 0:
            weights = weights / weights.sum()
            ensemble_probs = np.average(prob_matrix, axis=0, weights=weights)
            logger.info(f"Using AUC-weighted ensemble: {weights}")
        else:
            ensemble_probs = prob_matrix.mean(axis=0)
            logger.info("Using uniform ensemble averaging")
        
        # Compute ensemble metrics
        ensemble_auc = roc_auc_score(test_labels_ref, ensemble_probs)
        
        # Find optimal threshold for ensemble
        opt_threshold, _ = find_optimal_threshold(test_labels_ref, ensemble_probs)
        ensemble_preds = (ensemble_probs > opt_threshold).astype(int)
        ensemble_f1 = f1_score(test_labels_ref, ensemble_preds, zero_division=0)
        ensemble_acc = accuracy_score(test_labels_ref, ensemble_preds)
        ensemble_cm = confusion_matrix(test_labels_ref, ensemble_preds)
        
        # Report results
        logger.info(f"\nEnsemble Results (Test Set):")
        logger.info(f"  AUC: {ensemble_auc:.4f}")
        logger.info(f"  F1: {ensemble_f1:.4f} (threshold={opt_threshold:.3f})")
        logger.info(f"  Accuracy: {ensemble_acc:.4f}")
        logger.info(f"  Confusion Matrix:")
        logger.info(f"    {ensemble_cm}")
        logger.info(f"{'='*70}\n")
        
    except Exception as e:
        logger.error(f"Ensemble evaluation failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())


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
        "mean_dead_lobes": float(np.mean(dead_lobe_counts)) if dead_lobe_counts else 0.0,
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
    zero_counts = {view: 0 for view in view_order}
    fallback_counts = {view: 0 for view in view_order if view != "base"}
    checked = 0

    for fp in files:
        try:
            payload = torch.load(fp, map_location="cpu", weights_only=False)
        except Exception:
            continue

        if not isinstance(payload, dict):
            continue

        views = payload.get("views", payload)
        fallback_flags = payload.get("fallback_views", {}) if isinstance(payload, dict) else {}
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

    rates = {
        view: (zero_counts[view] / max(checked, 1))
        for view in view_order
    }
    fallback_rates = {
        view: (fallback_counts[view] / max(checked, 1))
        for view in fallback_counts
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
    from src.features.graph_factory import ABIDECausalDataset
    from src.features.graph_factory import _load_csv_cached

    _set_global_seed(42)

    # Prime CSV/Feather caches before the dataset constructor reads them.
    _load_csv_cached(MASTER_MANIFEST)
    _load_csv_cached(NODE_ATTRIBUTES_HARMONIZED, index_col='subject_id')
    _load_csv_cached(NODE_FEATURES_3D, index_col='subject_id')

    # Load dataset
    dataset = ABIDECausalDataset(split='train')
    
    # Extract labels for stratification
    labels = []
    site_labels = []
    for i in range(len(dataset)):
        data = dataset.get(i)
        if data is not None:
            labels.append(data.y.item())
            if hasattr(data, 'site_id') and data.site_id is not None and data.site_id.numel() > 0:
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
    
    # Compute class weights (informational)
    class_weights = compute_class_weights(labels)

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
    checkpoint_manager = CheckpointManager(checkpoint_dir, monitor='auc', mode='max')
    experiment_tracker = ExperimentTracker(experiment_name=f"gnn_training_{run_name}")
    experiment_tracker.add_note("use_grl", use_grl)
    experiment_tracker.add_note("grl_alpha", float(grl_alpha))
    experiment_tracker.add_note("checkpoint_dir", str(checkpoint_dir))
    
    # Initialize training monitor for analysis
    analysis_dir = RESULTS_TRAINING_DIR if run_post_analysis else (RESULTS_TRAINING_DIR / run_name)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    monitor = TrainingMonitor(analysis_dir, num_folds=K_FOLDS)
    
    # Print configuration
    logger.info(f"\n{'='*70}")
    logger.info("GNN TRAINING - 5-FOLD CROSS-VALIDATION (%s)", run_name)
    logger.info(f"{'='*70}")
    logger.info(f"Total subjects: {len(labels)}")
    logger.info(f"OneCycle max LR: {GNN_ONECYCLE_MAX_LR}")
    logger.info(f"Hidden channels: {GNN_HIDDEN_CHANNELS}")
    logger.info(f"Input features: {GNN_IN_CHANNELS} (registry count={len(ALL_FEATURE_NAMES)})")
    logger.info(f"Site conditioning: {GNN_USE_SITE_EMBEDDING}")
    logger.info(f"Demographics: {GNN_USE_DEMOGRAPHICS}")
    logger.info(f"GRL enabled: {use_grl} (alpha_max={grl_alpha:.2f})")
    logger.info(f"Early stopping patience: {GNN_EARLY_STOPPING_PATIENCE}")
    if USE_FOCAL_LOSS:
        logger.info(f"Loss: FocalLoss (α={FOCAL_LOSS_ALPHA}, γ={FOCAL_LOSS_GAMMA}, class_weights={USE_CLASS_WEIGHTS})")
    else:
        logger.info(f"Loss: CrossEntropy (class_weights={USE_CLASS_WEIGHTS})")
    logger.info(f"Threshold policy: {EVAL_THRESHOLD_POLICY}")
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
    if 'cv_fold' not in dataset.manifest.columns:
        raise ValueError(
            "cv_fold column not found in manifest. "
            "Run split.py first to generate predefined CV folds."
        )
    
    cv_folds = dataset.manifest['cv_fold'].values
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
        f for f in range(K_FOLDS)
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

    multiview_present = (
        CAUSAL_GRAPHS_MULTIVIEW_DIR.exists()
        and any(CAUSAL_GRAPHS_MULTIVIEW_DIR.glob("*/multiview_graphs.pt"))
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
    invariance_criterion = CausalInvarianceLoss(temperature=0.07) if invariance_enabled else None
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
            logger.info("No multi-view graphs detected — training falls back to standard single-view objective.")
            
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

        _set_global_seed(42 + fold)  # deterministic per-fold model initialisation
        fold_start_time = time.time()

        # Enforce fold-specific harmonized features (no global fallback).
        fold_temporal_path = HARMONIZED_FOLDS_DIR / f"harmonized_fold_{fold}.csv"
        candidate_dataset = ABIDECausalDataset(
            split='train',
            temporal_features_path=fold_temporal_path,
        )
        candidate_subject_ids = [str(s) for s in candidate_dataset.subject_ids]
        if candidate_subject_ids != base_subject_ids:
            raise ValueError(
                f"Fold {fold} harmonized file subject ordering mismatch in {fold_temporal_path}; "
                "aborting to prevent fold leakage."
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
        val_site_ids = np.array([
            int(d.site_id.view(-1)[0].item()) if hasattr(d, 'site_id') and d.site_id is not None else -1
            for d in val_data
        ])
        
        logger.info(f"Train: Control={train_labels.count(0)}, ASD={train_labels.count(1)}")
        logger.info(f"Val: Control={val_labels.count(0)}, ASD={val_labels.count(1)}")

        # Fold-wise feature standardization (fit on train fold only, apply to train+val).
        if train_data:
            train_x = torch.cat([d.x for d in train_data], dim=0)
            feat_mean = train_x.mean(dim=0, keepdim=True)
            feat_std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
            for d in train_data:
                d.x = (d.x - feat_mean) / feat_std
            for d in val_data:
                d.x = (d.x - feat_mean) / feat_std
            logger.info("Applied fold-wise node feature standardization (train-fit only)")
        else:
            feat_mean = torch.zeros((1, GNN_IN_CHANNELS), dtype=torch.float32)
            feat_std = torch.ones((1, GNN_IN_CHANNELS), dtype=torch.float32)
        
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
        class_weight_tensor = None
        if USE_CLASS_WEIGHTS:
            class_weight_tensor = compute_class_weights(train_labels).to(DEVICE)

        if USE_FOCAL_LOSS:
            pos_weight = float(n_control / n_asd) if USE_CLASS_WEIGHTS else None
            criterion = FocalLoss(
                alpha=FOCAL_LOSS_ALPHA,
                gamma=FOCAL_LOSS_GAMMA,
                pos_weight=pos_weight,
            )
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
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
            if entry['epoch'] % 10 == 0:
                monitor.log_epoch(
                    fold_id=fold,
                    epoch=entry['epoch'],
                    metrics={
                        'train_loss': entry['loss'],
                        'val_loss': 1.0 - entry['auc'],
                        'val_auc': entry['auc'],
                        'val_auprc': entry['auprc'],
                        'val_f1': entry['f1'],
                        'val_acc': entry['acc'],
                        'lr': entry['lr']
                    },
                    grad_norm=0.0,
                    confusion_matrix=entry['cm']
                )
                logger.info(
                    f"Epoch {entry['epoch']:03d} | LR: {entry['lr']:.6f} | Loss: {entry['loss']:.4f} | "
                    f"AUC: {entry['auc']:.4f} | AUPRC: {entry['auprc']:.4f} | "
                    f"F1@{entry['threshold']:.2f}: {entry['f1']:.4f}"
                )

        model.load_state_dict(best_state)
        checkpoint_metrics = {
            'auc': best_metrics['auc'],
            'auprc': best_metrics['auprc'],
            'f1': best_metrics['f1'],
            'threshold': best_metrics['threshold'],
            # Persist fold-wise scaling stats for inference-time parity.
            'feature_mean': feat_mean.squeeze(0).cpu().numpy().tolist(),
            'feature_std': feat_std.squeeze(0).cpu().numpy().tolist(),
        }
        checkpoint_manager.save(model, None, best_metrics['best_epoch'], checkpoint_metrics, fold=fold)

        logger.info(
            f"✓ Best fold {fold}: AUC={best_metrics['auc']:.4f}, "
            f"AUPRC={best_metrics['auprc']:.4f}, F1={best_metrics['f1']:.4f}"
        )
        
        # Final evaluation with best checkpoint
        model._feature_mean = feat_mean.squeeze(0).detach().cpu()
        model._feature_std = feat_std.squeeze(0).detach().cpu()
        final_threshold = best_metrics['threshold']
        final_metrics = evaluate(model, val_loader, threshold=final_threshold)
        site_auc_values.extend(
            _compute_site_auc_values(
                final_metrics['probs'],
                final_metrics['labels'],
                val_site_ids,
            )
        )
        best_epoch = best_metrics['best_epoch']
        
        fold_train_time = time.time() - fold_start_time
        
        # Log fold results
        logger.info(f"\nFold {fold+1} Final Results:")
        logger.info(f"  Best epoch: {best_epoch}")
        logger.info(f"  Training time: {fold_train_time:.1f}s")
        logger.info(f"  AUC: {final_metrics['auc']:.4f}")
        logger.info(f"  F1: {final_metrics['f1']:.4f} (threshold={final_threshold:.3f})")
        logger.info(f"  Accuracy: {final_metrics['acc']:.4f}")
        logger.info(f"  Confusion Matrix:")
        logger.info(f"    {final_metrics['cm']}")
        
        # Track results
        tracker.add_fold_result(
            fold=fold,
            auc=final_metrics['auc'],
            f1=final_metrics['f1'],
            acc=final_metrics['acc'],
            threshold=final_threshold,
            best_epoch=best_epoch,
            train_time=fold_train_time,
            val_probs=final_metrics['probs'],
            val_labels=final_metrics['labels']
        )
        experiment_tracker.log_fold(
            fold=fold,
            metrics={
                'auc': float(final_metrics['auc']),
                'f1': float(final_metrics['f1']),
                'acc': float(final_metrics['acc']),
                'threshold': float(final_threshold),
                'best_epoch': int(best_epoch),
                'train_time_sec': float(fold_train_time),
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
    site_auc_variance = float(np.var(site_auc_values)) if site_auc_values else float('inf')
    experiment_tracker.finalize(
        {
            **summary,
            'run_name': run_name,
            'grl_alpha': float(grl_alpha),
            'site_auc_variance': site_auc_variance,
            'site_auc_count': len(site_auc_values),
        }
    )
    logger.info(
        "Per-site validation AUC variance (%s): %.6f from %d site-level AUC values",
        run_name,
        site_auc_variance,
        len(site_auc_values),
    )
    
    # Ensemble evaluation on test set (combine all folds)
    if run_post_analysis:
        evaluate_ensemble(tracker, checkpoint_manager, use_grl=use_grl, grl_alpha=grl_alpha)
    
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
            test_dataset = ABIDECausalDataset(split='test')
            test_loader = make_loader(
                [d for d in test_dataset if d is not None],
                batch_size=GNN_BATCH_SIZE
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
            feature_output = analysis_dir / 'features'
            feature_output.mkdir(parents=True, exist_ok=True)
            feature_analyzer.visualize_feature_importance(
                attributions,
                str(feature_output / 'feature_importance.png')
            )
            logger.info(f"  Feature importance plot saved to: {feature_output / 'feature_importance.png'}")
            
        except Exception as e:
            logger.warning(f"Feature attribution analysis failed: {e}")
    
    # 2. Causal Graph Analysis
    if run_post_analysis:
        try:
            logger.info("\nRunning causal graph analysis...")
            import pandas as pd

            # Load manifest
            manifest_path = DATA_METADATA / 'master_manifest.csv'
            manifest = pd.read_csv(manifest_path)

            # Compute graph properties
            graph_analyzer = CausalGraphAnalyzer(CAUSAL_GRAPHS_DIR, manifest)
            graph_metrics = graph_analyzer.compute_graph_properties()

            # Compare ASD vs Control
            graph_output = analysis_dir / 'graphs'
            graph_output.mkdir(parents=True, exist_ok=True)
            graph_analyzer.compare_asd_vs_control(
                graph_metrics,
                str(graph_output)
            )
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
        "mean_auc": float(summary.get('mean_auc', 0.0)),
        "site_auc_variance": site_auc_variance,
        "site_auc_count": len(site_auc_values),
    }


def run_training():
    """Entry point for model training with optional GRL alpha grid search."""
    if GNN_AUTO_GRL_GRID_SEARCH and GRL_ALPHA_CANDIDATES:
        logger.info("Starting GRL alpha grid search: %s", GRL_ALPHA_CANDIDATES)
        candidate_results = []

        for alpha in GRL_ALPHA_CANDIDATES:
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
            logger.warning("GRL grid search produced no valid results; falling back to config defaults")
            return _run_training_once(
                use_grl=GNN_USE_GRL,
                grl_alpha=GNN_GRL_ALPHA,
                checkpoint_dir=CHECKPOINT_DIR,
                run_name="default",
                run_post_analysis=True,
            )

        best_mean_auc = max(r["mean_auc"] for r in candidate_results)
        viable = [r for r in candidate_results if r["mean_auc"] >= best_mean_auc - 0.01]
        selected = min(viable, key=lambda r: r["site_auc_variance"]) if viable else max(
            candidate_results,
            key=lambda r: r["mean_auc"],
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

if __name__ == "__main__":
    run_training()
