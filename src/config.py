from pathlib import Path
import torch

# PROJECT STRUCTURE
# Root directory of the project
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Goes up to project root from src/config.py

# Data directories
DATA_ROOT       = PROJECT_ROOT / "data"
DATA_RAW        = DATA_ROOT / "raw"
DATA_PROCESSED  = DATA_ROOT / "processed"
DATA_FINAL      = DATA_ROOT / "final"
DATA_IMAGES     = DATA_ROOT / "images"
DATA_LABELS     = DATA_ROOT / "labels"
DATA_ATLASES    = DATA_ROOT / "atlases"
DATA_METADATA   = DATA_ROOT / "metadata"

# Model directories
MODEL_ROOT      = PROJECT_ROOT / "models"
CHECKPOINT_DIR  = MODEL_ROOT / "checkpoints"
RESULTS_DIR     = PROJECT_ROOT / "results"

# Config directory
CONFIG_DIR      = PROJECT_ROOT / "configs"

# FILE PATHS

# Atlas
ATLAS_PATH      = DATA_ATLASES / "AAL3v1.nii"
ATLAS_METADATA  = DATA_METADATA / "roi_centroids.json"

# Phenotypic data
PHENO_PATH      = DATA_PROCESSED / "Phenotypic_V1_0b_preprocessed1.csv"
# Manifests and metadata
MASTER_MANIFEST              = DATA_METADATA / "master_manifest.csv"
NODE_ATTRIBUTES_TEMPORAL     = DATA_METADATA / "node_attributes_temporal.csv"
NODE_ATTRIBUTES_HARMONIZED   = DATA_METADATA / "node_attributes_harmonized.csv"
NODE_FEATURES_3D             = DATA_METADATA / "node_features_3d.csv"

# Causal graphs
CAUSAL_GRAPHS_DIR   = DATA_PROCESSED / "causal_graphs"

# YOLO config
YOLO_CONFIG         = CONFIG_DIR / "brain.yaml"

# AAL3 TO LOBE MAPPING (CRITICAL: Keep consistent across all scripts!)

LOBE_MAPPING = {
    0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],  # Frontal
    1: [79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90],  # Temporal
    2: [57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70],  # Parietal
    3: [43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],  # Occipital
    4: [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 71, 72, 73, 74, 75, 76, 77, 78, 91, 92, 93, 94]  # Limbic
}

# Lobe names for visualization
LOBE_NAMES = {
    0: 'Frontal',
    1: 'Temporal',
    2: 'Parietal',
    3: 'Occipital',
    4: 'Limbic'
}

NUM_LOBES = 5

# DATA PROCESSING PARAMETERS

# ABIDE download
TARGET_SLICES = 5  # Number of z-slices to extract per subject
SLICE_POSITIONS = [0.3, 0.4, 0.5, 0.6, 0.7]  # Relative positions along z-axis
DEFAULT_TR = 2.0  # Default repetition time for fMRI

# Time series extraction
BANDPASS_LOW = 0.01  # Hz
BANDPASS_HIGH = 0.08  # Hz

# Feature extraction
NUM_TEMPORAL_FEATURES = 6  # mean, std, skew, kurt, psd, mssd

# Causal graph construction
CAUSAL_LAG = 1  # Time lag for causal inference
SPARSITY_QUANTILE = 0.80  # Keep top 20% of connections

# TRAIN/VAL/TEST SPLIT

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# YOLO TRAINING PARAMETERS

YOLO_MODEL_SIZE = "yolo11s"  # Small model for subtle boundaries
YOLO_IMGSZ = 640
YOLO_BATCH_SIZE = 24  # Optimized for RTX 4060 8GB
YOLO_EPOCHS = 100
YOLO_PATIENCE = 25
YOLO_CONF_THRESHOLD = 0.35
YOLO_WORKERS = 8

# Augmentation parameters (medical-specific)
YOLO_HSV_H = 0.0  # No hue variation for medical images
YOLO_HSV_S = 0.0  # No saturation variation
YOLO_HSV_V = 0.2  # Slight brightness variation
YOLO_DEGREES = 10  # Subtle rotation for head tilt
YOLO_FLIPLR = 0.5  # Left-right symmetry
YOLO_FLIPUD = 0.0  # No up-down flip (anatomically incorrect)

# GNN MODEL PARAMETERS

GNN_HIDDEN_CHANNELS = 64
GNN_NUM_HEADS = 4  # For GAT attention
GNN_NUM_CLASSES = 2  # Binary classification (ASD vs Control)
GNN_DROPOUT = 0.5
GNN_LABEL_SMOOTHING = 0.1

# Training parameters
GNN_BATCH_SIZE = 32
GNN_LEARNING_RATE = 0.001
GNN_WEIGHT_DECAY = 1e-3
GNN_EPOCHS = 100
GNN_GRAD_CLIP = 1.0  # Gradient clipping norm
K_FOLDS = 5

# HARDWARE & PERFORMANCE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 8  # For data loading
PIN_MEMORY = True if torch.cuda.is_available() else False

# LOGGING & VISUALIZATION

LOG_INTERVAL = 10  # Log every N epochs
SAVE_PLOTS = True
PLOT_DPI = 300

# HELPER FUNCTIONS

def ensure_directories():
    directories = [
        DATA_RAW, DATA_PROCESSED, DATA_FINAL, DATA_IMAGES, DATA_LABELS,
        DATA_ATLASES, DATA_METADATA, MODEL_ROOT, CHECKPOINT_DIR, RESULTS_DIR,
        CAUSAL_GRAPHS_DIR
    ]
    
    for split in ['train', 'val', 'test']:
        directories.extend([
            DATA_FINAL / split / 'images',
            DATA_FINAL / split / 'labels',
            DATA_FINAL / split / 'time_series'
        ])
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    print("✓ All required directories created/verified")


def validate_paths():
    """Validate that critical files exist."""
    critical_files = [
        (ATLAS_PATH, "AAL3 Atlas"),
        (PHENO_PATH, "Phenotypic Data"),
        (YOLO_CONFIG, "YOLO Configuration")
    ]
    
    missing = []
    for path, name in critical_files:
        if not path.exists():
            missing.append(f"{name} at {path}")
    
    if missing:
        print("⚠️  Missing critical files:")
        for item in missing:
            print(f"   - {item}")
        return False
    
    print("✓ All critical files present")
    return True


if __name__ == "__main__":
    print("="*60)
    print("Neuro-CXG Configuration")
    print("="*60)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Device: {DEVICE}")
    print(f"Number of Workers: {NUM_WORKERS}")
    print("="*60)
    
    ensure_directories()
    validate_paths()