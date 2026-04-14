"""Model, training, and experiment hyperparameters for Neuro-CXG."""

import torch

from src.core.paths import RESULTS_DIR

# --- YOLO DETECTION PARAMETERS (Fixed for Medical Integrity) ---
YOLO_MODEL_SIZE = "yolo26n.pt"
YOLO_PROJECT_NAME = "ROI_Detection_v30"  # Output directory name from training
YOLO_WEIGHTS_PATH = RESULTS_DIR / "experiments" / "detection" / "ROI_Detection_v29" / "weights" / "best.pt"
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
# CAUSAL_LAG = 1  # DEPRECATED: only used in lagged-Pearson fallback path.
#                 # Use _LAGGED_PEARSON_LAG = 1 inline in construct_causal.py instead.
#                 # Retained as comment to prevent accidental re-introduction.
SPARSITY_QUANTILE = 0.70  # Keep top 30% edges (high selectivity - Phase 3)
# Target graph density: keep only the top GRAPH_DENSITY_TARGET fraction of edges.
# Literature recommends 10-30% for functional connectivity graphs.
# The fixed sparsification method quantiles over off-diagonal values only.
GRAPH_DENSITY_TARGET = 0.30  # Keep top 30% of directional edges (~40/132 for 12-node graphs)

# Phase 1 enhancements (Feb 2026)
CAUSALITY_METHOD = "granger"  # Options: 'granger', 'transfer_entropy', 'lagged_pearson'
GRANGER_MAX_LAG = 5  # Test lags 1-5 TRs (legacy, kept for backward compatibility)
GRANGER_MAX_LAG_SECONDS = 10.0  # Test causality up to 10s of history; adjusted by subject TR
GRANGER_SIGNIFICANCE_LEVEL = 0.05  # Statistical significance threshold

# --- GRAPH CONSTRUCTION PARAMETERS ---
SPARSITY_METHOD = "adaptive_statistical"  # Options: adaptive_proportional/adaptive_statistical/fixed
MIN_EDGES_PER_GRAPH = 12  # Ensure minimum connectivity for 12-region graphs

# --- DATA QUALITY FILTERS ---
# Subjects confirmed to have near-complete NaN coverage (>50% empty ROIs).
# These are permanently excluded and never reach the GNN dataset.
# Identified by the post-download integrity check.
EXCLUDED_SUBJECTS: frozenset = frozenset(
    {
        # Caltech: near-complete NaN coverage (masker extraction failed)
        "Caltech_0051486",  # 170/170 NaN — all ROIs empty, completely unusable
        "Caltech_0051491",  # 168/170 NaN — masker extraction effectively failed
        "Caltech_0051478",  # 108/170 NaN — >63% coverage loss, beyond recovery
        "Caltech_0051472",  # 148/170 NaN — >87% coverage loss, beyond recovery
        # Degenerate causal graphs: dead lobe(s) with zero in+out degree
        # Identified by subject_analysis.py (2026-03-09).
        "SDSU_0050209",  # Dead lobe in graph — partial FOV at SDSU scanner (train, fold 2)
        "SDSU_0050216",  # Dead lobe in graph — partial FOV at SDSU scanner (train, fold 0)
        "UCLA_1_0051220",  # Dead lobe in graph — partial FOV at UCLA_1 scanner (train, fold 4)
        "UCLA_1_0051277",  # Dead lobe in graph — partial FOV at UCLA_1 scanner (train, fold 3)
        "UCLA_2_0051303",  # Dead lobe in graph — partial FOV at UCLA_2 scanner (val)
    }
)
# Drop subjects where more than this many temporal feature entries are NaN.
MAX_NAN_ROIS: int = 30

# --- GNN MODEL PARAMETERS (Phase 3: regularized for small graphs) ---
GNN_HIDDEN_CHANNELS = 128
GNN_NUM_HEADS = 4
GNN_NUM_CLASSES = 2  # 0: Control, 1: ASD
GNN_DROPOUT = 0.35
GNN_WEIGHT_DECAY = 5e-5
GNN_LEARNING_RATE = 0.001
GNN_BATCH_SIZE = 32
GNN_EPOCHS = 100
K_FOLDS = 5

GNN_NUM_LAYERS = 2
GNN_SKIP_CONNECTIONS = True
GNN_USE_SITE_EMBEDDING = True
GNN_NODE_EMB_DIM = 16
GNN_USE_DEMOGRAPHICS = True
GNN_EARLY_STOPPING_PATIENCE = 30
GNN_POOLING = "attention"  # Options: 'attention', 'mean_max_sum'
GNN_USE_GRL = False
GNN_GRL_ALPHA = 0.0
GNN_GRL_ALPHA_MAX = 0.3
GRL_ALPHA_CANDIDATES = [0.05, 0.10, 0.20]
GNN_AUTO_GRL_GRID_SEARCH = False
GNN_SITE_LOSS_WEIGHT = 0.0
GNN_EDGE_GATE = True
GNN_ONECYCLE_MAX_LR = 0.002
GNN_ONECYCLE_PCT_START = 0.2
GNN_ONECYCLE_WARMUP_FRACTION = 0.15

# --- HARDWARE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CLASS IMBALANCE HANDLING ---
FOCAL_LOSS_ALPHA = 0.62
FOCAL_LOSS_GAMMA = 2.0

DEFAULT_THRESHOLD = 0.5
OPTIMIZE_THRESHOLD = True

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
