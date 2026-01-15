# Neuro-CXG Codebase Guide for AI Agents

## Project Overview
**Neuro-CXG** is a Graph Neural Network framework for brain disorder classification (ASD vs Control) using causal inference on fMRI data. It combines:
- **YOLO11** for anatomical ROI detection in brain slices
- **Causal graph construction** (5-lobe aggregation with lagged correlation)
- **Graph Neural Networks** (GAT/GCN) on causal adjacency matrices

**Data Flow**: Raw fMRI → 2D brain slices → YOLO detections (5 lobes) → temporal features → causal graphs → GNN classifier

## Architecture & Complete Data Pipeline

### End-to-End Pipeline (Critical Sequencing)

```
Raw fMRI (ABIDE) 
  ↓ [abide_download.py]
Extract 5 z-slices per subject (640×640 PNGs)
  ↓ [roi_detection.py - YOLO training]
Detect bounding boxes for 5 brain lobes across slices
  ↓ [extract_features.py]
Aggregate detections into 3D spatial coords per lobe (5 nodes)
  ↓ [extract_features.py → harmonize.py]
Extract temporal features (6 per node) + harmonize with neuroCombat
  ↓ [construct_causal.py]
Build directed causal graphs (170 AAL→5 lobes, lagged correlation)
  ↓ [gnn_model.py]
Train GNN with 5-fold stratified cross-validation
```

### Critical: Data Format Transformations

1. **Feature Extraction** → [src/data/extract_features.py]
   - Inference: `model.predict()` with `stream=True` (RAM management for i7-13650HX)
   - Input: Directory of subject slices named `{subject_id}_z{depth}.png`
   - Output: 3D spatial coords aggregated per lobe (5 detections/subject)
   - **Critical Filter**: Only subjects with ALL 5 lobes detected proceed to next stage
   - Merges with phenotype manifest to create `node_features_3d.csv`

2. **Batch Effect Harmonization** → [src/data/harmonize.py]
   - Removes site-specific scanner bias using neuroCombat
   - **CRITICAL**: `DX_GROUP` (diagnosis) is protected covariate—NOT harmonized away
   - Fills missing `AGE_AT_SCAN`/`SEX` with median/mode before ComBat
   - Output: `node_attributes_harmonized.csv` (features ready for GNN)

3. **Stratified Data Splitting** → [src/data/split.py]
   - Splits on `DX_GROUP` AND `SITE_ID` (2D stratification—journal requirement)
   - 70% train / 15% val / 15% test
   - Preserves subject-level grouping: all slices of one subject go to same split
   - Moves files to `data/final/{train,val,test}/{images,labels,time_series}`

4. **Graph Construction** → [src/data/construct_causal.py]
   - Aggregates 170 AAL ROIs → 5 lobes using `LOBE_MAPPING` from config
   - Computes **lagged partial correlation** (t-1 → t with lag=1 TR)
   - Sparsifies to top 20% correlations (`SPARSITY_QUANTILE=0.8`)
   - Output: PyTorch Geometric `Data` objects saved as `.pt` files

5. **GNN Training** → [src/models/gnn_model.py] + [src/models/causal_gnn.py]
   - Loads graphs via `ABIDECausalDataset` (handles file paths, label lookups)
   - 5-fold stratified CV on full train set
   - GATv2 with 4 attention heads processes causal edges with weights
   - Label smoothing (0.1) and gradient clipping (1.0) for stable training
   - Saves best-AUC model per fold to `models/checkpoints/best_model_fold{0-4}.pt`

## Critical Patterns & Conventions

### 1. Configuration as Single Source of Truth
- **ALL** hardcoded constants must live in [src/config.py]
- Import from config everywhere else (e.g., `from config import LOBE_MAPPING, NUM_LOBES`)
- TODO.md notes this is currently half-complete (duplicates exist in construct_causal.py, graph_factory.py)
- AAL3 → Lobe mapping: 5 lobes, 1-indexed ROIs (AAL standard), convert to 0-indexed internally
- Config provides validation functions: `validate_lobe_mapping()`, `validate_paths()`, `validate_environment()`

### 2. Path Handling
- Always import paths from config: `from config import DATA_ROOT, CHECKPOINT_DIR, CAUSAL_GRAPHS_DIR`
- Pattern: `PROJECT_ROOT = Path(__file__).resolve().parents[1]` (from src/ to project root)
- Never hardcode relative paths like `./data` or `../..`
- Use Path().exists() for validation before loading
- **Anti-pattern found**: `extract_features.py` and `gnn_model.py` use hardcoded `Path("./data")` instead of importing from config

### 3. Tensor/Data Shapes (Critical for Graph Construction)
- **Input time series**: `(timepoints, num_rois)` where num_rois ∈ {116, 117, 170} depending on atlas
- **After lobe aggregation**: `(timepoints, 5)` (always 5 lobes)
- **Graph node features (x)**: `(5, num_features)` where num_features = 6 (temporal) + 3 (spatial coords)
- **Edge index**: `(2, num_edges)` — 2D tensor for PyTorch Geometric format
- **Edge attributes**: `(num_edges,)` — causal correlation weights (floats in [-1, 1])
- **Batch label (y)**: scalar 0 (Control) or 1 (ASD)

### 4. AAL3 Neuroanatomy
- **170 ROIs** (1-indexed, AAL standard)
- **5 Lobes** in config.LOBE_MAPPING:
  - Frontal (0): ROIs 1-26
  - Temporal (1): ROIs 79-90
  - Parietal (2): ROIs 57-70
  - Occipital (3): ROIs 43-54
  - Limbic (4): ROIs 31-42, 71-78, 91-94
- **Validation**: `config.validate_lobe_mapping()` checks completeness, no duplicates, range [1-170]
- When indexing: convert AAL 1-indexed to Python 0-indexed: `aal_roi - 1`

### 5. YOLO-Specific (Medical Image Tuning)
- Model size: `yolo11s` (not nano—needs parameter capacity for subtle brain anatomy)
- Input size: 640×640
- Batch: 24 (RTX 4060 8GB limit; batch 32 causes OOM)
- Augmentation **disabled for medical images**: `HSV_H=0.0, HSV_S=0.0` (no color variation in grayscale medical images)
- Flips: Only left-right `(fliplr=0.5, flipud=0.0)` — up-down flip breaks anatomical validity
- Label smoothing: 0.1 (handles fuzzy brain boundaries)
- Box loss weight: 10.0 (prioritizes spatial accuracy for graph node coordinates)
- Config file: `configs/brain.yaml` (defines 5 ROI classes for YOLO)

### 6. Causal Graph Construction Details
- **Lag**: t-1 → t (1 TR lag enforces temporal precedence)
- **Method**: Partial correlation (controls for confounding by other ROIs)
- **Sparsity**: Keep top 20% of correlations by setting `SPARSITY_QUANTILE = 0.80`
- **Output format**: PyTorch `.pt` files containing `torch_geometric.Data` objects:
  ```python
  Data(
    x=node_features,           # shape: (5, num_features)
    edge_index=edge_indices,   # shape: (2, num_edges)
    edge_attr=weights,         # shape: (num_edges,)
    y=diagnosis_label,         # 0 or 1
    subject_id=string          # for tracking
  )
  ```
- **Spatial coordinates**: Aggregated from YOLO detections (mean x, y, z_depth per lobe)

### 7. GNN Architecture & Training
- **Model**: `CausalBrainGNN` (class in [src/models/causal_gnn.py])
  - Input embedding: LayerNorm (not BatchNorm—graphs are small)
  - Layer 1: GATv2Conv with 4 heads, edge_dim=1 (causal weights)
  - Layer 2: GATv2Conv with 4 heads, edge_dim=1
  - Skip connections: Residual links prevent over-smoothing in 5-node graphs
  - Readout: Concat mean-pooling + max-pooling (captures global + peak local activity)
  - Output: 2-class softmax (Control vs ASD)

- **Training details**:
  - Optimizer: AdamW with `lr=0.001, weight_decay=1e-3`
  - Scheduler: CosineAnnealingLR over EPOCHS
  - Loss: CrossEntropyLoss with `label_smoothing=0.1`
  - Gradient clipping: `max_norm=1.0` (prevents explosion in small graphs)
  - K-fold: 5-fold stratified by DX_GROUP
  - Metrics: Accuracy, F1, ROC-AUC (from probs[:,1]), confusion matrix per fold
  - Checkpointing: Save best model per fold (top validation AUC)

## Development Workflows

### Running the Full Pipeline
```bash
# 0. Validate environment first (catches missing files, CUDA issues)
python src/config.py

# 1. Train YOLO (one-time, outputs best.pt to results/)
python src/pipelines/roi_detection.py

# 2. Extract spatial features from detections (produces node_features_3d.csv)
python src/data/extract_features.py

# 3. Harmonize temporal features with neuroCombat (removes batch effects)
python src/data/harmonize.py

# 4. Stratified split into train/val/test (2D stratification by DX_GROUP + SITE_ID)
python src/data/split.py

# 5. Build causal graphs (produces .pt files in causal_graphs/)
python src/data/construct_causal.py

# 6. Train GNN with 5-fold CV (saves checkpoints per fold)
python src/models/gnn_model.py
```

### Testing & Validation
```bash
# Run comprehensive test suite
pytest src/test_suite.py -v

# Run config validation tests only
pytest src/test_suite.py::TestConfiguration -v

# Run data integrity tests
pytest src/test_suite.py::TestDataIntegrity -v

# Run with coverage report
pytest src/test_suite.py --cov=src --cov-report=html

# Check dataset loading (verify labels, shapes, sample counts)
python -c "from src.data.graph_factory import ABIDECausalDataset; \
ds = ABIDECausalDataset('train'); print(f'Loaded {len(ds)} subjects, first graph has {ds[0].x.shape[0]} nodes')"

# Validate lobe mapping before graph construction
python -c "from src.config import validate_lobe_mapping; validate_lobe_mapping(); print('✓ LOBE_MAPPING valid')"

# Validate entire environment
python -c "from src.config import validate_environment; validate_environment()"
```

### Debugging Data Issues

**If graphs aren't loading:**
1. Check manifest exists: `ls -la data/processed/Phenotypic_V1_0b_preprocessed1.csv`
2. Verify graphs exist: `ls data/processed/causal_graphs/ | wc -l`
3. Load single graph: `python -c "import torch; g=torch.load('data/processed/causal_graphs/Caltech_0051456_graph.pt'); print(f'Nodes: {g.x.shape[0]}, Edges: {g.edge_index.shape[1]}')"` 

**If features are missing:**
- Check `node_features_3d.csv` exists (output of `extract_features.py`)
- Verify all-5-lobes filter: `python -c "import pandas as pd; df=pd.read_csv('data/processed/metadata/node_features_3d.csv'); print(f'Complete subjects: {len(df)}')"` 
- If count dropped significantly, lobes weren't detected → check YOLO model path in `extract_features.py`

**If training crashes with CUDA OOM:**
- Reduce `GNN_BATCH_SIZE` from 32 to 16 in `config.py`
- Check GPU memory: `nvidia-smi` (need ~6GB for batch 32)
- Reduce YOLO batch from 24 to 16 if retraining ROI detector

**If stratification fails:**
- Ensure `SITE_ID` and `DX_GROUP` columns exist in phenotype CSV
- Check no subjects missing from manifest: `python -c "import pandas as pd; df=pd.read_csv('data/processed/Phenotypic_V1_0b_preprocessed1.csv'); print(f'Groups: {df.DX_GROUP.value_counts().to_dict()}')"` 

## Key Files Reference

| File | Purpose |
|------|---------|
| [src/config.py] | ALL constants, paths, hyperparameters; validation functions |
| [src/data/extract_features.py] | YOLO inference → 3D spatial aggregation; all-5-lobes filter |
| [src/data/harmonize.py] | neuroCombat batch effect removal; protects DX_GROUP |
| [src/data/split.py] | 2D stratified split (by DX_GROUP + SITE_ID) |
| [src/data/construct_causal.py] | AAL→Lobe aggregation; lagged correlation; graph creation |
| [src/data/graph_factory.py] | PyTorch Geometric dataset loader |
| [src/models/causal_gnn.py] | GATv2 architecture with skip connections |
| [src/models/gnn_model.py] | k-fold training loop; metrics computation |
| [src/pipelines/roi_detection.py] | YOLO training entry point |
| [src/test.py] | Configuration & data integrity tests |

## Integration Points & Critical Dependencies

### Data Flow Checkpoints
- **extract_features.py** → requires: `best.pt` (YOLO weights in results/), PNG images in `data/final/{train,val,test}/images/`
- **harmonize.py** → requires: temporal features CSV output from extract_features
- **construct_causal.py** → requires: node_attributes_harmonized.csv, time series .npy files in `data/final/{split}/time_series/`
- **gnn_model.py** → requires: all `.pt` graphs in `data/processed/causal_graphs/`, master_manifest.csv, harmonized features

### Protected Covariates in Harmonization
In [src/data/harmonize.py], `DX_GROUP` (diagnosis) is NEVER passed to neuroCombat—it's protected so batch harmonization doesn't remove disease signal. This is a journal Q1 requirement. Missing values in `AGE_AT_SCAN`/`SEX` are imputed (median/mode) BEFORE ComBat.

### Dataset Filtering: All-5-Lobes Requirement
[src/data/extract_features.py] enforces that only subjects with ALL 5 brain lobes detected proceed to GNN training. This is critical: 5-node graphs assume complete detection. Check the `node_count == 5` filter before downstream processing.

### Stratified k-fold Details
[src/data/split.py] performs 2D stratification on both `DX_GROUP` (diagnosis) AND `SITE_ID` (scanner site). This ensures:
- Balanced ASD/Control across folds (addresses class imbalance)
- Balanced sites across folds (addresses batch effects from different scanners)

## Known Issues & Refactoring Targets (From TODO.md)

- **Duplicate LOBE_MAPPING**: Remove from [src/data/construct_causal.py] lines 11-16 and [src/data/graph_factory.py] lines 108-113; import from config only
- **Inconsistent path handling**: [src/data/extract_features.py] and [src/models/gnn_model.py] use hardcoded `Path("./data")` instead of importing from config
- **GNN model parameters**: [src/models/gnn_model.py] lines 11-19 hardcode `K_FOLDS=5, BATCH_SIZE=32, LR=0.001`—should import from config
- **Missing validation on startup**: Only `validate_environment()` call in [src/config.py] is commented out; consider uncommenting for CI/CD

## Medical/Scientific Context

- **Diagnosis label**: 0=Control (healthy), 1=ASD (autism spectrum disorder)
- **Data source**: ABIDE initiative (public fMRI dataset, multi-site)
- **Causal inference goal**: Identify ROI→ROI influence patterns distinctive to ASD using lagged correlations
- **Explainability**: Edge weights in causal graph provide subject-specific feature importance (gradient-based saliency in `causal_gnn.py::get_node_importance()`)
- **Medical tuning**: YOLO augmentation disabled for grayscale medical images; left-right flips only (preserve anatomy)
