from pathlib import Path
import torch
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- PROJECT STRUCTURE ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT       = PROJECT_ROOT / "data"
DATA_PROCESSED  = DATA_ROOT / "processed"
DATA_FINAL      = DATA_ROOT / "final"
DATA_IMAGES     = DATA_ROOT / "images"
DATA_LABELS     = DATA_ROOT / "labels"
DATA_ATLASES    = DATA_ROOT / "raw" / "atlases"
DATA_METADATA   = DATA_ROOT / "metadata"

# Final split directories
FINAL_TRAIN     = DATA_FINAL / "train"
FINAL_VAL       = DATA_FINAL / "val"
FINAL_TEST      = DATA_FINAL / "test"

MODEL_ROOT      = PROJECT_ROOT / "models"
CHECKPOINT_DIR  = MODEL_ROOT   / "checkpoints"
RESULTS_DIR     = PROJECT_ROOT / "results"
# Result subdirectory constants — import these instead of hardcoding paths
RESULTS_TRAINING_DIR     = RESULTS_DIR / "experiments" / "training"
RESULTS_ABLATIONS_DIR    = RESULTS_DIR / "experiments" / "ablations"
RESULTS_DATA_QUALITY_DIR = RESULTS_DIR / "experiments" / "data_quality"
RESULTS_EVALUATION_DIR   = RESULTS_DIR / "evaluation"
RESULTS_FIGURES_DIR      = RESULTS_DIR / "figures"
CONFIG_DIR      = PROJECT_ROOT / "configs"

# --- FILE PATHS ---
CONFIG_BRAIN_YAML = CONFIG_DIR   / "brain.yaml"
ATLAS_PATH      = DATA_ATLASES   / "AAL3v1.nii"
ATLAS_METADATA  = DATA_METADATA  / "atlas_metadata.json"
PHENO_PATH      = DATA_PROCESSED / "Phenotypic_V1_0b_preprocessed1.csv"
MASTER_MANIFEST = DATA_METADATA  / "master_manifest.csv"

# --- fMRI ACQUISITION DEFAULTS ---
# Default TR (seconds) when missing in manifest metadata.
DEFAULT_TR = 2.0

# Output files for the pipeline
NODE_ATTRIBUTES_TEMPORAL     = DATA_METADATA  / "node_attributes_temporal.csv"
NODE_ATTRIBUTES_HARMONIZED   = DATA_METADATA  / "node_attributes_harmonized.csv"
NODE_FEATURES_3D             = DATA_METADATA  / "node_features_3d.csv"
CAUSAL_GRAPHS_DIR            = DATA_PROCESSED / "causal_graphs"

# --- ANATOMICAL MAPPING (12-Region Neuroanatomical Subdivision) ---
# Note: AAL ROI IDs are 1-indexed; convert to 0-indexed for array access.
# Updated January 2026: Expanded from 5 lobes to 12 functionally-distinct brain regions
def _idx(ids):
    return [i - 1 for i in ids]


LOBE_MAPPING = {
    0: _idx([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),  # Frontal_Superior (Left+Right)
    1: _idx([21, 22, 25, 26, 27, 28]),  # Frontal_Orbital (Left+Right)
    2: _idx([17, 18, 19, 20, 23, 24]),  # Motor_Premotor (Central, includes 23-24)
    3: _idx([29, 30, 31, 32]),  # Insula (Left+Right, 29-30 missing previously)
    4: _idx([33, 34, 35, 36, 37, 38, 151, 152, 153, 154, 155, 156]),  # Cingulate + ACC subdivisions
    5: _idx([39, 40, 41, 42, 91, 92, 93, 94]),  # Limbic (Hippocampus, Amygdala)
    6: _idx([43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56]),  # Occipital
    7: _idx([57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]),  # Parietal
    8: _idx([79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90]),  # Temporal
    9: _idx([71, 72, 73, 74, 75, 76, 77, 78, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150]),  # Subcortical (Thalamus, Basal Ganglia, Subthalamic nucleus, SNpc)
    10: _idx(list(range(95, 121)) + list(range(157, 167))),  # Cerebellum (Vermis + Hemispheres)
    11: _idx([167, 168, 169, 170])  # Brainstem (Midbrain, Pons, Medulla)
}

LOBE_NAMES = {
    0: 'Frontal_Superior',
    1: 'Frontal_Orbital', 
    2: 'Motor_Premotor',
    3: 'Insula',
    4: 'Cingulate',
    5: 'Limbic',
    6: 'Occipital',
    7: 'Parietal',
    8: 'Temporal',
    9: 'Subcortical',
    10: 'Cerebellum',
    11: 'Brainstem'
}
NUM_LOBES = 12  # Updated from 5 to 12 regions
NUM_FREQUENCY_FEATURES = 12  # 5 bands x 2 features + 2 global
NUM_TEMPORAL_FEATURES = 20  # 8 basic + 12 frequency
NUM_SPATIAL_FEATURES = 6   # x, y, z_depth, size, conf_std, detection_count per lobe

# --- FEATURE REGISTRY (The Golden Standard) ---
# Explicit feature definitions. GNN_IN_CHANNELS is calculated dynamically from this.
FEATURE_GROUPS = {
    'temporal': ["mean", "std", "skew", "kurtosis", "psd", "mssd", "range", "autocorr"],
    'frequency': [
        "delta_power", "delta_peak", "theta_power", "theta_peak",
        "alpha_power", "alpha_peak", "beta_power", "beta_peak",
        "gamma_power", "gamma_peak", "spectral_entropy", "phase_std"
    ],
    'internal': ["coherence", "spatial_variance"],  # NEW: PCA/ReHo features from Phase 2
    'spatial': ["x", "y", "z_depth", "size", "conf_std", "detection_count"]
}

# ALL_FEATURE_NAMES: Concatenation order used everywhere (temporal + frequency + internal + spatial)
ALL_FEATURE_NAMES = (
    FEATURE_GROUPS['temporal'] + 
    FEATURE_GROUPS['frequency'] + 
    FEATURE_GROUPS['internal'] + 
    FEATURE_GROUPS['spatial']
)

# Dynamic calculation - should be 28
GNN_IN_CHANNELS = len(ALL_FEATURE_NAMES)

# --- YOLO DETECTION PARAMETERS (Fixed for Medical Integrity) ---
YOLO_MODEL_SIZE = "yolo26n.pt"
YOLO_PROJECT_NAME = "ROI_Detection_v29"  # Output directory name from training
YOLO_WEIGHTS_PATH = RESULTS_DIR / "experiments" / "detection" / "ROI_Detection_v29" / "weights" / "best.pt"
YOLO_IMGSZ = 640
YOLO_BATCH_SIZE = 32
YOLO_EPOCHS = 100
YOLO_CONF_THRESHOLD = 0.30

# Medical Augmentation Settings:
# CRITICAL: fliplr=0.0 and degrees=0.0 preserve Left/Right and 3D Z-alignment.
YOLO_HSV_H = 0.0   
YOLO_HSV_S = 0.0   
YOLO_HSV_V = 0.1   # Minimal brightness variation for scanner intensity
YOLO_DEGREES = 0.0 # No rotation - maintains exact 3D centroid coordinates
YOLO_FLIPLR = 0.0  # No flipping - prevents Left/Right hemisphere confusion
YOLO_FLIPUD = 0.0  
YOLO_MOSAIC = 0.0  # No mosaic - maintains global anatomical context

# Consolidated YOLO Training Configuration
# Pass to model.train() with **YOLO_TRAIN_CONFIG to eliminate parameter duplication
YOLO_TRAIN_CONFIG = {
    'epochs': YOLO_EPOCHS,
    'imgsz': YOLO_IMGSZ,
    'batch': YOLO_BATCH_SIZE,
    'device': 0,
    'seed': 42,
    'deterministic': True,
    'plots': True,
    'save': True,
    'val': True,
    'patience': 25,
    'workers': 8,
    'optimizer': 'AdamW',
    'lr0': 0.001,
    'label_smoothing': 0.0,
    'box': 7.5,
    'cls': 2.0,
    # Medical augmentation - anatomical protection
    'hsv_h': YOLO_HSV_H,
    'hsv_s': YOLO_HSV_S,
    'hsv_v': YOLO_HSV_V,
    'degrees': YOLO_DEGREES,
    'fliplr': YOLO_FLIPLR,
    'flipud': YOLO_FLIPUD,
    'mosaic': YOLO_MOSAIC,  
    'mixup': 0.0
}

# --- CAUSAL GRAPH PARAMETERS ---
CAUSAL_LAG = 1           # 1 TR lag for temporal precedence
SPARSITY_QUANTILE = 0.70 # Keep top 30% edges (High Selectivity - Phase 3)

# Phase 1 Enhancements (Feb 2026)
CAUSALITY_METHOD = 'granger'  # Directed causality (options: 'granger', 'transfer_entropy', 'lagged_pearson')
GRANGER_MAX_LAG = 5  # Test lags 1-5 TRs for Granger causality
GRANGER_SIGNIFICANCE_LEVEL = 0.05  # Statistical significance threshold

SPARSITY_METHOD = 'adaptive_statistical'  # Options: 'adaptive_proportional', 'adaptive_statistical', 'fixed'
MIN_EDGES_PER_GRAPH = 12  # Ensure minimum connectivity for 12-region graphs

# --- GNN MODEL PARAMETERS (Phase 3: Regularized for Small Graphs) ---
# Reduced from 256 to 64 channels to prevent overfitting on 12-node graphs
GNN_HIDDEN_CHANNELS = 128       # Increased capacity for 28-feature inputs
GNN_IN_CHANNELS_DYNAMIC = len(ALL_FEATURE_NAMES)  # Should be 28
GNN_NUM_HEADS = 4               # Multi-head attention is crucial
GNN_NUM_CLASSES = 2      # 0: Control, 1: ASD
GNN_DROPOUT = 0.45               # Reduced to prevent underfitting
GNN_WEIGHT_DECAY = 1e-4         # L2 Regularization (NEW)
GNN_LEARNING_RATE = 0.001      # Stable learning rate
GNN_BATCH_SIZE = 32
GNN_EPOCHS = 100  # More epochs with early stopping
K_FOLDS = 5

GNN_NUM_GNN_LAYERS = 3          # Restore depth for full graph coverage
GNN_SKIP_CONNECTIONS = True     # Enable residual connections
GNN_USE_SITE_EMBEDDING = True   # Reduce site bias
GNN_USE_DEMOGRAPHICS = True     # Add age/sex/IQ conditioning
GNN_EARLY_STOPPING_PATIENCE = 20
GNN_POOLING = "attention"        # Options: "attention", "mean_max_sum"
GNN_USE_GRL = True              # Enable gradient reversal site classifier
GNN_GRL_ALPHA = 1.0             # GRL strength (higher = stronger site invariance)
GNN_SITE_LOSS_WEIGHT = 0.2      # Weight for site classification loss
GNN_EDGE_GATE = True            # Soft gate edge_attr before message passing
GNN_ONECYCLE_MAX_LR = 0.003     # Peak LR for OneCycle schedule
GNN_ONECYCLE_PATIENCE = 20      # Early stopping patience for OneCycle training

# --- HARDWARE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# CLASS IMBALANCE HANDLING (ADD TO src/config.py)

# Focal Loss Parameters (Experiment v1.3: Prioritize underrepresented Control class)
FOCAL_LOSS_ALPHA = 0.62  # Weight for ASD (prioritize minority class)
FOCAL_LOSS_GAMMA = 2.0   # Increase focus on hard examples

# Classification Threshold
DEFAULT_THRESHOLD = 0.5  # Default classification threshold
OPTIMIZE_THRESHOLD = True  # Find optimal threshold per fold

# Training Strategy
USE_FOCAL_LOSS = True  # Use Focal Loss instead of CrossEntropy
USE_CLASS_WEIGHTS = False  # Use class weights (alternative to Focal Loss)
USE_BALANCED_SAMPLING = False  # Oversample minority class in batches

# Early Stopping
PATIENCE = 25  # Epochs without improvement before stopping
EVAL_FREQUENCY = 10  # Evaluate and check threshold every N epochs

# DIAGNOSTIC THRESHOLDS

# AUC Thresholds for Health Checks
AUC_RANDOM_THRESHOLD = 0.52  # Below this is essentially random
AUC_WEAK_THRESHOLD = 0.60    # Weak but useful signal
AUC_GOOD_THRESHOLD = 0.70    # Clinical utility threshold
AUC_EXCELLENT_THRESHOLD = 0.80  # Publication-ready

# F1 Thresholds
F1_BROKEN_THRESHOLD = 0.01   # Below this indicates complete class collapse
F1_WEAK_THRESHOLD = 0.30     # Weak but learning
F1_GOOD_THRESHOLD = 0.50     # Balanced performance
F1_EXCELLENT_THRESHOLD = 0.70  # Strong performance

# Loss Thresholds
LOSS_RANDOM_THRESHOLD = 0.693  # log(2) - random guessing
LOSS_LEARNING_THRESHOLD = 0.65  # Model is learning
LOSS_CONVERGED_THRESHOLD = 0.50  # Model has converged


def validate_training_health(metrics: dict) -> str:

    auc = metrics.get('auc', 0.5)
    f1 = metrics.get('f1', 0.0)
    loss = metrics.get('loss', 0.693)
    
    # Critical failures
    if auc < AUC_RANDOM_THRESHOLD:
        return "CRITICAL: Random guessing (AUC < 0.52)"
    
    if f1 < F1_BROKEN_THRESHOLD and loss > LOSS_RANDOM_THRESHOLD:
        return "CRITICAL: Class collapse (F1 ≈ 0, Loss ≈ 0.693)"
    
    # Weak signal
    if auc < AUC_WEAK_THRESHOLD:
        if f1 < F1_WEAK_THRESHOLD:
            return "WARNING: Weak signal, class imbalance likely"
        else:
            return "OK: Learning but needs improvement"
    
    # Good performance
    if auc >= AUC_GOOD_THRESHOLD and f1 >= F1_GOOD_THRESHOLD:
        return "EXCELLENT: Clinical utility achieved"
    
    return "OK: Model learning"


def log_training_diagnostics(fold: int, epoch: int, metrics: dict):
    """
    Log detailed diagnostics for training monitoring.
    
    Args:
        fold: Current fold number
        epoch: Current epoch
        metrics: Dictionary with all metrics
    """
    health = validate_training_health(metrics)
    
    logger.info(f"\nFold {fold}, Epoch {epoch} Diagnostics:")
    logger.info(f"  Health: {health}")
    logger.info(f"  AUC: {metrics['auc']:.4f} (random=0.50, good≥0.70)")
    logger.info(f"  F1: {metrics['f1']:.4f} (broken<0.01, good≥0.50)")
    logger.info(f"  Loss: {metrics['loss']:.4f} (random=0.693, good<0.50)")
    
    if 'cm' in metrics:
        tn, fp, fn, tp = metrics['cm'].ravel()
        logger.info(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        
        if tp == 0:
            logger.warning("  ⚠️  No true positives! Model predicting all Control.")
        
        if fp + tn == 0:
            logger.warning("  ⚠️  No negative predictions! Model predicting all ASD.")
    
    logger.info("")
    
    
# --- VALIDATION LOGIC ---
def validate_lobe_mapping() -> bool:
    """Validate LOBE_MAPPING completeness, uniqueness, and ROI range.

    Checks:
    1. Exactly NUM_LOBES regions are defined.
    2. All ROI 0-indexed values are within [0, 169] (1-indexed: [1, 170]).
    3. No ROI index appears in more than one lobe (no duplicates).
    4. All 170 ROI indices are covered (full coverage).

    Returns:
        True on success.

    Raises:
        ValueError: If any check fails with a descriptive message.
    """
    # 1. Correct number of regions
    if len(LOBE_MAPPING) != NUM_LOBES:
        raise ValueError(
            f"LOBE_MAPPING has {len(LOBE_MAPPING)} regions, expected NUM_LOBES={NUM_LOBES}"
        )

    all_rois: list = []
    for lobe_id, indices in LOBE_MAPPING.items():
        for idx in indices:
            # 2. Range check (0-indexed, so valid range is 0–169)
            if not (0 <= idx <= 169):
                raise ValueError(
                    f"LOBE_MAPPING[{lobe_id}] contains out-of-range index {idx} "
                    f"(1-indexed: {idx + 1}). Valid 1-indexed range is [1, 170]."
                )
            all_rois.append(idx)

    # 3. Duplicate check
    seen: set = set()
    duplicates: list = []
    for idx in all_rois:
        if idx in seen:
            duplicates.append(idx + 1)  # Report as 1-indexed
        seen.add(idx)
    if duplicates:
        raise ValueError(
            f"LOBE_MAPPING contains duplicate ROI indices (1-indexed): {sorted(set(duplicates))}"
        )

    # 4. Full coverage of 170 AAL ROIs
    expected = set(range(170))
    covered = set(all_rois)
    missing = expected - covered
    if missing:
        # Convert to 1-indexed for readability in the error message
        missing_1idx = sorted(i + 1 for i in missing)
        raise ValueError(
            f"LOBE_MAPPING does not cover {len(missing)} AAL ROI(s) "
            f"(1-indexed): {missing_1idx}"
        )

    logger.info(
        "✓ validate_lobe_mapping: %d regions, %d ROIs, no duplicates, full coverage",
        NUM_LOBES,
        len(all_rois),
    )
    return True


def validate_environment():
    """Checks if the 12-region architecture is ready for execution."""
    logger.info("VALIDATING NEURO-CXG 12-REGION ARCHITECTURE")
    
    # Check paths
    for p in [DATA_ROOT, DATA_METADATA, CAUSAL_GRAPHS_DIR]:
        p.mkdir(parents=True, exist_ok=True)
    
    # Check Lobe Mapping (comprehensive)
    validate_lobe_mapping()

    # Check YOLO Augmentations (Prevent 0.5 AUC failure)
    if YOLO_FLIPLR > 0 or YOLO_DEGREES > 0:
        logger.warning("⚠️  DANGER: YOLO augmentations (fliplr/degrees) are enabled.")
        logger.warning("This will likely cause the model to fail (0.5 AUC).")
    
    logger.info(f"✓ Target: {NUM_LOBES} nodes | Features: {GNN_IN_CHANNELS}")
    logger.info(f"✓ Device: {DEVICE}")
    return True


def validate_graph_construction_inputs():
    """Pre-check before graph construction to ensure all required inputs exist."""
    logger.info("Validating graph construction inputs...")
    
    errors = []
    
    # Check harmonized node attributes
    if not NODE_ATTRIBUTES_HARMONIZED.exists():
        errors.append(f"Missing: {NODE_ATTRIBUTES_HARMONIZED}")
    
    # Check 3D node features
    if not NODE_FEATURES_3D.exists():
        errors.append(f"Missing: {NODE_FEATURES_3D}")
    
    # Check master manifest
    if not MASTER_MANIFEST.exists():
        errors.append(f"Missing: {MASTER_MANIFEST}")
    
    # Check if time series directory has any data
    if DATA_FINAL.exists():
        ts_count = sum(1 for p in (DATA_FINAL / "train" / "time_series").glob("*.npy") if p.is_file())
        if ts_count == 0:
            errors.append(f"No time series files found in {DATA_FINAL / 'train' / 'time_series'}")
    else:
        errors.append(f"Data directory not found: {DATA_FINAL}")
    
    if errors:
        logger.error("Graph construction validation FAILED:")
        for err in errors:
            logger.error(f"  ✗ {err}")
        raise FileNotFoundError("\n".join(errors))
    
    logger.info("✓ Graph construction inputs validated")
    return True


def validate_gnn_training_inputs():
    """Pre-check before GNN training to ensure dataset can be loaded."""
    logger.info("Validating GNN training inputs...")
    
    errors = []
    
    # Check causal graphs directory
    if not CAUSAL_GRAPHS_DIR.exists():
        errors.append(f"Missing causal graphs directory: {CAUSAL_GRAPHS_DIR}")
    else:
        graph_count = sum(1 for p in CAUSAL_GRAPHS_DIR.glob("*.pt") if p.is_file())
        if graph_count == 0:
            errors.append(f"No graph files found in {CAUSAL_GRAPHS_DIR}")
        else:
            logger.info(f"  Found {graph_count} graph files")
    
    # Check manifest
    if not MASTER_MANIFEST.exists():
        errors.append(f"Missing manifest: {MASTER_MANIFEST}")
    
    # Check harmonized features
    if not NODE_ATTRIBUTES_HARMONIZED.exists():
        errors.append(f"Missing features: {NODE_ATTRIBUTES_HARMONIZED}")
    
    if errors:
        logger.error("GNN training validation FAILED:")
        for err in errors:
            logger.error(f"  ✗ {err}")
        raise FileNotFoundError("\n".join(errors))
    
    logger.info("✓ GNN training inputs validated")
    return True

if __name__ == "__main__":
    validate_environment()