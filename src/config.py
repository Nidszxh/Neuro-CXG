"""
Central Configuration for Neuro-CXG Project
Single source of truth for all paths, parameters, and constants.
"""

from pathlib import Path
import torch
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- PROJECT STRUCTURE ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT       = PROJECT_ROOT / "data"
DATA_PROCESSED  = DATA_ROOT / "processed"
DATA_FINAL      = DATA_ROOT / "final"
DATA_ATLASES    = DATA_ROOT / "atlases"
DATA_METADATA   = DATA_ROOT / "metadata"

MODEL_ROOT      = PROJECT_ROOT / "models"
CHECKPOINT_DIR  = MODEL_ROOT / "checkpoints"
RESULTS_DIR     = PROJECT_ROOT / "results"
CONFIG_DIR      = PROJECT_ROOT / "configs"

# --- FILE PATHS ---
ATLAS_PATH      = DATA_ATLASES / "AAL3v1.nii"
ATLAS_METADATA  = DATA_METADATA / "atlas_metadata.json"
PHENO_PATH      = DATA_PROCESSED / "Phenotypic_V1_0b_preprocessed1.csv"
MASTER_MANIFEST = DATA_METADATA / "master_manifest.csv"

# Output files for the pipeline
NODE_ATTRIBUTES_TEMPORAL     = DATA_METADATA / "node_attributes_temporal.csv"
NODE_ATTRIBUTES_HARMONIZED   = DATA_METADATA / "node_attributes_harmonized.csv"
NODE_FEATURES_3D             = DATA_METADATA / "node_features_3d.csv"
CAUSAL_GRAPHS_DIR            = DATA_PROCESSED / "causal_graphs"

# --- ANATOMICAL MAPPING (5-Lobe Standard) ---
# Note: ROI IDs are 1-indexed (AAL Standard). Internal code converts to 0-indexed.
LOBE_MAPPING = {
    0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],  # Frontal
    1: [79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90],  # Temporal
    2: [57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70],  # Parietal
    3: [43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],  # Occipital
    4: [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 71, 72, 73, 74, 75, 76, 77, 78, 91, 92, 93, 94]  # Limbic
}

LOBE_NAMES = {0: 'Frontal', 1: 'Temporal', 2: 'Parietal', 3: 'Occipital', 4: 'Limbic'}
NUM_LOBES = 5
NUM_TEMPORAL_FEATURES = 6  # Mean, Std, Skew, Kurtosis, PSD, ALFF per ROI

# --- TEMPORAL FEATURE EXTRACTION PARAMETERS ---
DEFAULT_TR = 2.0  # Default TR (seconds) for fMRI—fallback if not in phenotype CSV

# --- YOLO DETECTION PARAMETERS (Fixed for Medical Integrity) ---
YOLO_MODEL_SIZE = "yolo11s.pt"
YOLO_IMGSZ = 640
YOLO_BATCH_SIZE = 24
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

# --- CAUSAL GRAPH PARAMETERS ---
CAUSAL_LAG = 1           # 1 TR lag for temporal precedence
SPARSITY_QUANTILE = 0.80 # Keep top 20% strongest causal edges

# --- GNN MODEL PARAMETERS ---
GNN_IN_CHANNELS = 9      # 6 Temporal stats + 3 Spatial (x,y,z)
GNN_HIDDEN_CHANNELS = 64
GNN_NUM_HEADS = 4
GNN_NUM_CLASSES = 2      # 0: Control, 1: ASD
GNN_DROPOUT = 0.5
GNN_LEARNING_RATE = 0.001
GNN_BATCH_SIZE = 32
GNN_EPOCHS = 100
K_FOLDS = 5

# --- HARDWARE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


"""
Updated Configuration with Class Imbalance Handling

Add these parameters to src/config.py to enable the class balance fixes.
"""

# ============================================================
# CLASS IMBALANCE HANDLING (ADD TO src/config.py)
# ============================================================

# Focal Loss Parameters
FOCAL_LOSS_ALPHA = 0.75  # Weight for minority class (ASD)
FOCAL_LOSS_GAMMA = 2.0   # Focusing parameter (higher = more focus on hard examples)

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

# ============================================================
# DIAGNOSTIC THRESHOLDS
# ============================================================

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

# ============================================================
# VALIDATION FUNCTION UPDATES
# ============================================================

def validate_training_health(metrics: dict) -> str:
    """
    Diagnose training health from metrics.
    
    Args:
        metrics: Dictionary with 'auc', 'f1', 'loss'
    
    Returns:
        Health status string
    """
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
def validate_environment():
    """Checks if the 5-node architecture is ready for execution."""
    logger.info("="*60)
    logger.info("VALIDATING NEURO-CXG 5-NODE ARCHITECTURE")
    logger.info("="*60)
    
    # Check paths
    for p in [DATA_ROOT, DATA_METADATA, CAUSAL_GRAPHS_DIR]:
        p.mkdir(parents=True, exist_ok=True)
    
    # Check Lobe Mapping
    if len(LOBE_MAPPING) != NUM_LOBES:
        raise ValueError(f"Config Error: NUM_LOBES is {NUM_LOBES} but mapping has {len(LOBE_MAPPING)}")

    # Check YOLO Augmentations (Prevent 0.5 AUC failure)
    if YOLO_FLIPLR > 0 or YOLO_DEGREES > 0:
        logger.warning("⚠️  DANGER: YOLO augmentations (fliplr/degrees) are enabled.")
        logger.warning("This will likely cause the model to fail (0.5 AUC).")
    
    logger.info(f"✓ Target: {NUM_LOBES} nodes | Features: {GNN_IN_CHANNELS}")
    logger.info(f"✓ Device: {DEVICE}")
    logger.info("="*60)
    return True

if __name__ == "__main__":
    validate_environment()