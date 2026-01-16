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