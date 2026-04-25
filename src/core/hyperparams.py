import torch

from src.core.paths import RESULTS_DIR

# --- YOLO DETECTION PARAMETERS (Fixed for Medical Integrity) ---
YOLO_MODEL_SIZE = "yolo26n.pt"
YOLO_PROJECT_NAME = "ROI_Detection_v31"  # Output directory name from training
YOLO_WEIGHTS_PATH = RESULTS_DIR / "experiments" / "detection" / "ROI_Detection_v31" / "weights" / "best.pt"
YOLO_IMGSZ = 640
YOLO_BATCH_SIZE = 32
YOLO_EPOCHS = 100
YOLO_CONF_THRESHOLD = 0.30

# Medical augmentation settings.
# CRITICAL: fliplr=0.0 and degrees=0.0 preserve Left/Right and 3D Z-alignment.
YOLO_HSV_H = 0.0
YOLO_HSV_S = 0.0
YOLO_HSV_V = 0.1  # Minimal brightness variation for scanner intensity
YOLO_DEGREES = 0.0  # No rotation - maintains exact 3D centroid coordinates
YOLO_FLIPLR = 0.0  # No flipping - prevents Left/Right hemisphere confusion
YOLO_FLIPUD = 0.0
YOLO_MOSAIC = 0.0  # No mosaic - maintains global anatomical context

# Consolidated YOLO training configuration.
# Pass to model.train() with **YOLO_TRAIN_CONFIG to eliminate parameter duplication.
YOLO_TRAIN_CONFIG = {
    "epochs": YOLO_EPOCHS,
    "imgsz": YOLO_IMGSZ,
    "batch": YOLO_BATCH_SIZE,
    "device": 0,
    "seed": 42,
    "deterministic": True,
    "plots": True,
    "save": True,
    "val": True,
    "patience": 25,
    "workers": 8,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "label_smoothing": 0.0,
    "box": 7.5,
    "cls": 2.0,
    # Medical augmentation - anatomical protection
    "hsv_h": YOLO_HSV_H,
    "hsv_s": YOLO_HSV_S,
    "hsv_v": YOLO_HSV_V,
    "degrees": YOLO_DEGREES,
    "fliplr": YOLO_FLIPLR,
    "flipud": YOLO_FLIPUD,
    "mosaic": YOLO_MOSAIC,
    "mixup": 0.0,
}

# --- CAUSAL GRAPH PARAMETERS ---
SPARSITY_QUANTILE = 0.70  # Keep top 30% edges (high selectivity - Phase 3)
# Target graph density: keep only the top GRAPH_DENSITY_TARGET fraction of edges.
# Literature recommends 10-30% for functional connectivity graphs.
# The fixed sparsification method quantiles over off-diagonal values only.
GRAPH_DENSITY_TARGET = 0.30  # Keep top 30% of directional edges (~40/132 for 12-node graphs)

# Phase 1/2 enhancements (Apr 2026)
# Default to ridge-regularized Granger edges for stronger statistical signal.
CAUSALITY_METHOD = "lagged_pearson"  # Best performing method  # Options: 'granger', 'ridge_granger', 'ridge_granger_hybrid', 'lagged_pearson', 'partial_corr_glasso'
# 2026-04-24: GRL_ALPHA changed to 1.0 based on Ablation D results showing +0.05 improvement
GRANGER_MAX_LAG = 5  # Test lags 1-5 TRs (legacy, kept for backward compatibility)
GRANGER_MAX_LAG_SECONDS = 10.0  # Test causality up to 10s of history; adjusted by subject TR
GRANGER_SIGNIFICANCE_LEVEL = 0.05  # Statistical significance threshold
GRANGER_USE_GPU = True  # Use GPU-accelerated Granger causality (auto-detects CUDA availability)

# Ridge-regularized pairwise VAR Granger controls.
RIDGE_GRANGER_LAGS = (1, 2, 3, 4, 5)
RIDGE_GRANGER_LAMBDA = 1.0
RIDGE_GRANGER_CONFIDENCE_ALPHA = 0.75  # w = effect * sigmoid(alpha * confidence)
RIDGE_GRANGER_HIGH_CONF_P_THRESHOLD = 0.10
RIDGE_GRANGER_P_PRUNE_THRESHOLD = 0.20

# Optional hybrid graph: beta * ridge_granger + (1-beta) * lagged_pearson.
RIDGE_GRANGER_HYBRID_BETA = 0.70

# Lagged-Pearson edge construction controls (signal recovery pass)
LAGGED_PEARSON_LAGS = (1, 2, 3, 4)  # Multi-lag candidates evaluated per directed edge
LAGGED_PEARSON_P_SELECT_THRESHOLD = 0.10  # Prefer lag with max |z| among p < threshold
LAGGED_PEARSON_P_PRUNE_THRESHOLD = 0.20  # Zero weak edges before top-k candidate selection
LAGGED_PEARSON_CONFIDENCE_ALPHA = 0.75  # w = z * sigmoid(alpha * confidence)

# GraphicalLasso partial-correlation controls.
# This method yields sparse conditional-dependence edges without p-value pruning,
# and is useful as a robust, low-variance alternative in small-sample folds.
PARTIAL_CORR_GLASSO_ALPHA = 0.02
PARTIAL_CORR_GLASSO_MAX_ITER = 200
PARTIAL_CORR_GLASSO_TOL = 1e-4
PARTIAL_CORR_MIN_ABS_EDGE = 0.02
PARTIAL_CORR_MIN_SAMPLES = 40
PARTIAL_CORR_FDR_ENABLED = True
PARTIAL_CORR_FDR_ALPHA = 0.10

# --- GRAPH CONSTRUCTION PARAMETERS ---
# Default policy: keep strongest edges per node (outgoing + incoming) so each
# lobe remains represented before any fallback repair logic.
SPARSITY_METHOD = "topk_per_node"  # Options: topk_per_node/adaptive_proportional/adaptive_statistical/fixed
SPARSITY_TOPK_PER_NODE = 3  # Strongest outgoing/incoming edges retained per node
MIN_EDGES_PER_GRAPH = 12  # Ensure minimum connectivity for 12-region graphs

# --- DATA QUALITY FILTERS ---
# Curated removal list used to reduce the source ABIDE cohort from 1035 -> 1015.
# Ordering reflects severity ranking from results/analysis/worst_subjects_ranking_1035.csv.
CURATED_WORST_SUBJECTS_1015: frozenset = frozenset(
    {
        "Caltech_0051486",
        "Caltech_0051491",
        "Caltech_0051472",
        "Caltech_0051478",
        "SDSU_0050209",
        "Caltech_0051471",
        "SDSU_0050216",
        "SDSU_0050195",
        "SDSU_0050192",
        "Caltech_0051469",
        "Caltech_0051464",
        "Caltech_0051467",
        "CMU_b_0050645",
        "CMU_b_0050658",
        "Caltech_0051460",
        "SDSU_0050184",
        "CMU_b_0050651",
        "Pitt_0050045",
        "SBL_0051575",
        "SDSU_0050204",
    }
)

# Backward-compatible name used across pipeline modules.
EXCLUDED_SUBJECTS: frozenset = CURATED_WORST_SUBJECTS_1015
# Drop subjects where more than this many temporal feature entries are NaN.
MAX_NAN_ROIS: int = 30

# --- GNN MODEL PARAMETERS (Phase 3: regularized for small graphs) ---
GNN_HIDDEN_CHANNELS = 32
GNN_NUM_HEADS = 2
GNN_NUM_CLASSES = 2  # 0: Control, 1: ASD
GNN_DROPOUT = 0.35
GNN_WEIGHT_DECAY = 5e-4
GNN_LEARNING_RATE = 0.001
GNN_BATCH_SIZE = 32
GNN_EPOCHS = 100
K_FOLDS = 5

GNN_NUM_LAYERS = 2
GNN_SKIP_CONNECTIONS = True
GNN_USE_SITE_EMBEDDING = True
GNN_SITE_EMBEDDING_DIM = 16
GNN_NODE_EMB_DIM = 16
GNN_USE_DEMOGRAPHICS = True
GNN_EARLY_STOPPING_PATIENCE = 30
# Guardrail against premature stopping on noisy/unstable folds.
GNN_MIN_EPOCHS_BEFORE_STOPPING = 30
GNN_POOLING = "anatomical"  # Options: 'anatomical', 'attention', 'mean_max_sum'
GNN_USE_GRL = True
GNN_GRL_ALPHA = 0.10
GNN_GRL_ALPHA_MAX = 1.0
GRL_ALPHA_CANDIDATES = [0.10, 0.25, 0.50, 1.0]
GNN_AUTO_GRL_GRID_SEARCH = False
# Non-zero weight enables actual adversarial site debiasing when GRL is active.
GNN_SITE_LOSS_WEIGHT = 0.15
GNN_EDGE_GATE = True
GNN_ONECYCLE_MAX_LR = 0.001
GNN_ONECYCLE_PCT_START = 0.2
GNN_ONECYCLE_WARMUP_FRACTION = 0.05

# Auxiliary regularization defaults (kept conservative by default).
# These were previously hardcoded in gnn_model.py and can now be tuned safely.
# NOTE: Setting structural_dropout and edge_contrastive to force graph learning
# (previously disabled due to known issue DD-009 - model ignoring graph structure)
GNN_STRUCTURAL_DROPOUT_PROB = 0.0  # Disabled - counterproductive without contrastive complement
GNN_EDGE_CONTRASTIVE_WEIGHT = 0.0  # Keep auxiliary contrastive objective off for baseline stability
GNN_INVARIANCE_WEIGHT = 0.0
GNN_SPATIAL_INVARIANCE_WEIGHT = 0.0

# Wave-1 fold-internal preprocessing controls.
# `legacy_global` restores the previous fold-global z-score behavior.
GNN_FOLD_PREPROCESSING_MODE = "wave1"

# Fold-internal mutual-information feature selection.
GNN_MI_FEATURE_SELECTION_ENABLED = False
GNN_MI_MIN_KEEP_RATIO = 0.30
GNN_MI_MAX_KEEP_RATIO = 0.60

# Fold-internal normalization policy.
# - "within_site": fit per-site stats on train fold only; unseen sites fallback to global train stats.
# - "global": fit one global train-fold scaler.
# - "none": disable fold-internal feature normalization.
GNN_SITE_NORMALIZATION_MODE = "within_site"

# Fold-safe harmonization policy for validation/test rows from sites that are
# absent in fold-train. Options:
# - "passthrough": harmonize seen sites, leave unseen rows unchanged.
# - "fail": abort fold harmonization when unseen sites are detected.
HARMONIZATION_UNSEEN_SITE_POLICY = "fail"

# Multi-view integrity gate: disable invariance training when non-base views are
# largely degenerate (zero-edge), rather than silently training on broken views.
GNN_ENFORCE_MULTIVIEW_QUALITY_GATE = True
GNN_MULTIVIEW_MAX_ZERO_EDGE_RATE = 0.20
GNN_MULTIVIEW_QUALITY_SAMPLE_SIZE = 512

# Multi-view generation integrity gate (Stage 15): fail/warn immediately after
# creating multiview artifacts if non-base view types are mostly zero-edge.
MULTIVIEW_GENERATION_ENFORCE_QUALITY_GATE = True
MULTIVIEW_GENERATION_MAX_ZERO_EDGE_RATE = 0.20
MULTIVIEW_GENERATION_POLICY = "fail"  # Options: "fail", "warn"

# Base graph quality gate: prevent training on heavily degenerate graphs.
GNN_ENFORCE_GRAPH_QUALITY_GATE = True
GNN_MAX_DEGENERATE_GRAPH_RATE = 0.35
GNN_MIN_EDGES_FOR_NONDEGENERATE = 12

# --- HARDWARE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CLASS IMBALANCE HANDLING ---
FOCAL_LOSS_ALPHA = 0.50
FOCAL_LOSS_GAMMA = 1.5

DEFAULT_THRESHOLD = 0.5
OPTIMIZE_THRESHOLD = True

# Evaluation operating-point policy.
# - "f1": max-F1 threshold on held-out calibration fold
# - "youden": max(sensitivity + specificity - 1) - reduces false negatives
# - "fixed": use EVAL_FIXED_THRESHOLD directly (deployment lock)
EVAL_THRESHOLD_POLICY = "youden"
EVAL_FIXED_THRESHOLD = 0.5263  # Kept for backward compatibility
EVAL_PER_SITE_MIN_SAMPLES = 10  # Minimum samples per site for per-site calibration
EVAL_SEED = 42  # Random seed for bootstrap/permutation tests

# Site robustness gate derived from cross-site experiment output
# (`results/experiments/data_quality/cross_site_auc.csv`).
SITE_ROBUSTNESS_GATE_ENABLED = True
SITE_ROBUSTNESS_MIN_SITE_AUC = 0.55
SITE_ROBUSTNESS_MAX_WEAK_SITE_FRACTION = 0.40
SITE_ROBUSTNESS_MIN_EVALUABLE_SITES = 5
SITE_ROBUSTNESS_GATE_POLICY = "warn"  # Options: "warn", "fail"

USE_FOCAL_LOSS = True
USE_CLASS_WEIGHTS = False
USE_BALANCED_SAMPLING = False

# EVAL_FREQUENCY removed in Task 6 (DD-014) — was unused throughout the codebase.

# --- DIAGNOSTIC THRESHOLDS ---
AUC_RANDOM_THRESHOLD = 0.52
AUC_WEAK_THRESHOLD = 0.60
AUC_GOOD_THRESHOLD = 0.70
AUC_EXCELLENT_THRESHOLD = 0.80

F1_BROKEN_THRESHOLD = 0.01
F1_WEAK_THRESHOLD = 0.30
F1_GOOD_THRESHOLD = 0.50
F1_EXCELLENT_THRESHOLD = 0.70

LOSS_RANDOM_THRESHOLD = 0.693
LOSS_LEARNING_THRESHOLD = 0.65
LOSS_CONVERGED_THRESHOLD = 0.50
