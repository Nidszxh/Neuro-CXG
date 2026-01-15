# Neuro-CXG: AI Coding Agent Instructions

## Project Overview
**Neuro-CXG** is a Graph Neural Network framework for brain disorder classification using causal inference and explainable AI. It combines fMRI neuroimaging with GNNs to identify autism spectrum disorder (ASD) from the ABIDE dataset.

**Core Pipeline**: YOLO ROI Detection → Feature Extraction → Causal Graph Construction → GNN Classification → Explainability Analysis

---

## Architecture & Data Flow

### 1. **ROI Detection (YOLO Stage)**
- **File**: [src/pipelines/roi_detection.py](src/pipelines/roi_detection.py)
- **Config**: [configs/brain.yaml](configs/brain.yaml) - defines 5 brain lobe classes (Frontal, Temporal, Parietal, Occipital, Limbic)
- **Key Detail**: Uses YOLOv11-small (not nano) for subtle lobe boundary detection. Batch size=24 (RTX 4060 8GB limit). Medical-specific augmentation: disable HSV color variance (invalid for brain scans), subtle rotation (±10°), left-right flip only.

### 2. **Time Series & Feature Extraction**
- **File**: [src/data/abide_download.py](src/data/abide_download.py)
- **Process**: Downloads preprocessed fMRI from ABIDE/S3, extracts 170 AAL3 ROI time series using NiftiLabelsMasker with filtering (0.01-0.08 Hz), outputs 6 temporal features per ROI (mean, std, skew, kurtosis, PSD, MSSD)
- **Critical**: Atlas resampling to functional space before masking; handles multiple AAL atlas versions dynamically (170 ROIs or 116 depending on atlas)

### 3. **Causal Graph Construction**
- **File**: [src/data/construct_causal.py](src/data/construct_causal.py)
- **Aggregation**: 170 AAL ROIs → 5 lobes using `LOBE_MAPPING` dict (defined in both this file and graph_factory.py - keep synchronized!)
- **Causal Logic**: Directed lagged partial correlation (t-1 → t) to enforce temporal precedence; sparsify to top 20% edges
- **Output**: PyTorch graph files (subject_id_graph.pt) with adjacency matrices and edge weights

### 4. **GNN Training (K-Fold Cross-Validation)**
- **Models**: [src/models/causal_gnn.py](src/models/causal_gnn.py) (architecture), [src/models/gnn_model.py](src/models/gnn_model.py) (training loop)
- **Architecture**: CausalBrainGNN uses GATv2Conv (attention with edge attributes), LayerNorm (not BatchNorm for small graphs), skip connections, hierarchical readout (mean+max pooling)
- **Training**: 5-fold stratified K-fold, 100 epochs, seed=42 for reproducibility, gradient clipping (norm=1.0), weight decay=1e-3
- **Key Constraint**: `edge_attr` must be passed to model.forward() - causal weights are essential features, not optional

### 5. **Dataset Loading**
- **File**: [src/data/graph_factory.py](src/data/graph_factory.py) - ABIDECausalDataset
- **Critical Validation**: Checks intersection of manifest CSV + node_attributes + 3D coords + physical .pt files on disk; reports dropped subjects
- **Dynamic ROI Detection**: Handles variable ROI counts (117 vs 170) by dividing feature count by 6; trims to nearest multiple if needed
- **Label Mapping**: DX_GROUP=1 → ASD (label 1), else control (label 0)

---

## Important Conventions & Patterns

### Neuroimaging-Specific Patterns
- **AAL Indices are 1-based**: When accessing numpy arrays, subtract 1: `array[i-1]` for AAL ROI i
- **LOBE_MAPPING Must Be Consistent**: Definition in construct_causal.py and graph_factory.py must match. If updating one, update both.
- **No Color Augmentation**: Medical images lack color variation; disable HSV in YOLO config to avoid invalid transformations
- **Site Harmonization**: Group results by site in analysis; different hospitals introduce systematic bias

### PyTorch & GNN Specifics
- **Deterministic Reproducibility**: Always set `seed=42, deterministic=True` in YOLO training; use `torch.manual_seed(42)` in model __init__
- **Edge Attributes Required**: Causal GNN uses directed edge weights as feature attributes (edge_attr), not optional
- **LayerNorm Over BatchNorm**: For small brain graphs (5-100 nodes), LayerNorm is more stable than BatchNorm
- **Skip Connections**: Prevent over-smoothing in deep GNNs on small graphs; use residual paths
- **Explainability Hook**: `get_node_importance()` method computes gradient-based saliency maps for each brain region

### Data Splits & Manifest
- **master_manifest.csv**: Source of truth; includes subject_id, split (train/val/test), DX_GROUP, site, age
- **Split Strategy**: Configured in dataset loader; stratified K-fold ensures class balance
- **Validation**: ABIDECausalDataset prints ✅/⚠️ warnings on initialization showing dropped subjects and counts

### Training Metrics
- **Classification Metrics**: Accuracy, F1-score, ROC-AUC (from probability outputs, not logits)
- **Confusion Matrix**: Track sensitivity/specificity for medical validation
- **Gradient Clipping**: max_norm=1.0 prevents exploding gradients in graph convolutions
- **Learning Rate Schedule**: Cosine annealing with warmup (not specified in code, but standard for medical ML)

---

## Essential Commands & Workflows

### Setup & Data Preparation
```bash
# Install dependencies
pip install -r requirements.txt

# Download ABIDE data and extract time series (saves as .npy files)
python -m src.data.abide_download

# Construct causal graphs (creates .pt files in data/processed/causal_graphs/)
python -m src.data.construct_causal

# Verify data integrity
python -m src.utils.integrity
```

### Training & Evaluation
```bash
# Train YOLO ROI detection model
python -m src.pipelines.roi_detection

# Train GNN with 5-fold cross-validation
python -m src.models.gnn_model

# Test on held-out test set
# (Script not shown, but use test split in ABIDECausalDataset)
```

### Files to Watch
- [src/data/graph_factory.py](src/data/graph_factory.py) - Data loading, intersection validation, LOBE_MAPPING
- [configs/brain.yaml](configs/brain.yaml) - YOLO class definitions and data paths
- [src/models/causal_gnn.py](src/models/causal_gnn.py) - Model architecture, edge_attr handling
- [results/](results/) - Logs, checkpoints, hyperparameters (args.yaml auto-generated by YOLO)

---

## Common Pitfalls & Solutions

| Issue | Cause | Fix |
|-------|-------|-----|
| **ValueError: wrong feature count** | ROI aggregation mismatch (e.g., 170 ROIs → num_feats != 6) | graph_factory.py handles this; check if atlas version changed |
| **Missing .pt files** | Causal graph construction incomplete | Run construct_causal.py first; check S3 connectivity for ABIDE download |
| **OOM with batch=32** | RTX 4060 8GB overloaded | Reduce batch to 24 (tested working); or increase workers for async loading |
| **NaN loss in GNN** | Exploding gradients or invalid edge_attr | Verify edge weights are finite; check gradient clipping is enabled |
| **Class imbalance** | Unequal ASD/control in folds | Use StratifiedKFold (already implemented); monitor confusion matrix per fold |
| **Causal graphs all zeros** | Sparsification threshold too high | Adjust quantile threshold in construct_causal.py (currently 0.80 keeps top 20%) |

---

## File Organization & Key Artifacts

- **Data Pipeline**: [src/data/](src/data/) (download, feature extraction, causal graphs)
- **Model Definitions**: [src/models/](src/models/) (CausalBrainGNN, training loop)
- **Preprocessing**: [src/pipelines/](src/pipelines/) (YOLO ROI detection), [src/transform/](src/transform/) (harmonization)
- **Utilities**: [src/utils/](src/utils/) (integrity checks, annotations, manifests)
- **Checkpoints**: [models/checkpoints/](models/checkpoints/) (best_model_fold*.pt for each fold)
- **Results**: [results/](results/) (YOLO weights, MLflow logs, ROI detection outputs)
- **Configs**: [configs/brain.yaml](configs/brain.yaml) (YOLO training hyperparameters)

---

## When Adding Features
- **New ROI aggregation method?** Update LOBE_MAPPING in both construct_causal.py and graph_factory.py
- **New GNN architecture?** Ensure edge_attr is properly threaded through; add to explainability hook
- **New YOLO augmentation?** Remember medical context (no color variance, head-tilt only, left-right symmetry)
- **New dataset split?** Ensure manifest.csv includes new split name; update ABIDECausalDataset.split parameter
