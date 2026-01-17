# Neuro-CXG Codebase Guide for AI Agents

## Project Overview
**Neuro-CXG** is a Graph Neural Network framework for brain disorder classification (ASD vs Control) using causal inference on fMRI data. It combines:
- **YOLO11** for anatomical ROI detection in brain slices
- **Causal graph construction** (5-lobe aggregation with lagged correlation)
- **Graph Neural Networks** (GAT/GCN) on causal adjacency matrices

**Data Flow**: Raw fMRI → 2D brain slices → YOLO detections (5 lobes) → temporal features → causal graphs → GNN classifier

## Quick Start for Agents
**Most common tasks:**
- **Run full pipeline**: `python src/run_pipeline.py --run-diagnostics --run-safe-harmonize` (orchestrates all stages)
- **Validate environment**: `python -c "from src.core.config import validate_environment; validate_environment()"` (pre-flight check)
- **Debug data issues**: Use [src/pipeline_diagnostics.py] — comprehensive health check for all pipeline stages
- **Find config values**: ALL constants live in [src/config.py]—never hardcode paths or parameters
- **Add new code**: Follow imports from config pattern (see extract_features.py, gnn_model.py for correct examples)

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
   - Computes **lagged Pearson correlation** (t-1 → t with lag=1 TR for temporal precedence)
   - Creates 5×5 directed adjacency matrix (matrix[i,j] = correlation between lobe i at t-1 and lobe j at t)
   - Sparsifies to top 20% correlations (`SPARSITY_QUANTILE=0.8`; keeps ~5 edges per graph)
   - Output: Dictionary with 'adj' (5×5 tensor), 'subject_id', 'lobe_order' saved as `.pt` files

5. **GNN Training** → [src/models/gnn_model.py] + [src/models/causal_gnn.py]
   - Loads graphs via `ABIDECausalDataset` (handles file paths, label lookups)
   - 5-fold stratified CV on full train set
   - GATv2 with 4 attention heads processes causal edges with weights
   - Label smoothing (0.1) and gradient clipping (1.0) for stable training
   - Saves best-AUC model per fold to `models/checkpoints/best_model_fold{0-4}.pt`

## Critical Patterns & Conventions

### 1. Configuration as Single Source of Truth ✅ (Refactored January 2026)
- **ALL** hardcoded constants must live in [src/config.py]
- Import from config everywhere else (e.g., `from src.core.config import LOBE_MAPPING, NUM_LOBES`)
- AAL3 → Lobe mapping: 5 lobes, 1-indexed ROIs (AAL standard), convert to 0-indexed internally
- Config provides comprehensive validation functions:
  - `validate_environment()`: Pre-flight checks (paths, CUDA, lobe mapping)
  - `validate_graph_construction_inputs()`: Pre-checks before causal graph building
  - `validate_gnn_training_inputs()`: Pre-checks before GNN training
  - `validate_lobe_mapping()`: Checks completeness, no duplicates, range [1-170]
- **Status**: ✅ All path references centralized (100% imported from config)

### 2. Path Handling ✅ (Refactored January 2026)
- **Always import paths from config**: `from src.core.config import DATA_ROOT, CHECKPOINT_DIR, CAUSAL_GRAPHS_DIR`
- Pattern in config: `PROJECT_ROOT = Path(__file__).resolve().parents[1]` (from src/config.py to project root)
- Pattern in submodules: 
  ```python
  sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
  from src.core.config import DATA_ROOT, DATA_FINAL, DATA_PROCESSED
  ```
- **Status**: ✅ ALL modules centralized (split.py, manifest.py, annotate.py, check_progress.py, integrity_check.py)
- Never hardcode relative paths like `./data` or `../..`
- Use Path().exists() for validation before loading

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

### 5. YOLO-Specific (Medical Image Tuning - CRITICAL)
- Model size: `yolo11s` (defined in config.YOLO_MODEL_SIZE)
- Input size: 640×640 (config.YOLO_IMGSZ)
- Batch: 24 (config.YOLO_BATCH_SIZE; batch 32+ causes OOM on RTX 4060 8GB)
- **Medical augmentation disabled** (config.py enforces):
  - `YOLO_HSV_H=0.0, YOLO_HSV_S=0.0` (no color/saturation—grayscale medical images don't need this)
  - `YOLO_DEGREES=0.0` (no rotation—preserves exact 3D centroid coordinates for 5-lobe aggregation)
  - `YOLO_FLIPLR=0.0` (no left-right flip—prevents Left/Right hemisphere confusion; critical for causal graph directionality)
  - `YOLO_FLIPUD=0.0, YOLO_MOSAIC=0.0` (no flipping/mosaic—maintains global anatomical context)
- Confidence threshold: 0.30 (config.YOLO_CONF_THRESHOLD; 0.35 used in extract_features inference)
- Epochs: 100 (config.YOLO_EPOCHS)
- Config file: `configs/brain.yaml` (defines 5 ROI classes: Frontal, Temporal, Parietal, Occipital, Limbic)
- **Key insight**: Medical image preprocessing is opposite of natural image YOLO—disable all augmentation that breaks anatomical alignment

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
  - Input embedding: LayerNorm (not BatchNorm—graphs are small; stabilizes 9 features with varying scales)
  - Layer 1: GATv2Conv with **2 heads**, edge_dim=1 (causal weights; sufficient for 5-node graphs)
  - Layer 2: GATv2Conv with **2 heads**, edge_dim=1 (concat=True; output is hidden_channels * 2)
  - Skip connections: Residual links prevent over-smoothing in 5-node graphs
  - Readout: Concat mean-pooling (global brain state) + max-pooling (pathological lobe hub)
  - Output: 2-class softmax (Control vs ASD)
  - Weight initialization: Kaiming normal for Linear layers, zeros for biases

- **Training details**:
  - Optimizer: AdamW with `lr=0.001, weight_decay=1e-3`
  - Scheduler: CosineAnnealingLR over EPOCHS
  - Loss: CrossEntropyLoss with `label_smoothing=0.1`
  - Gradient clipping: `max_norm=1.0` (prevents explosion in small graphs)
  - K-fold: 5-fold stratified by DX_GROUP
  - Metrics: Accuracy, F1, ROC-AUC (from probs[:,1]), confusion matrix per fold
  - Checkpointing: Save best model per fold (top validation AUC)
  - Dropout: 0.5 (high dropout to prevent memorizing site-specific noise)

- **Current Results** (5-fold CV on full training set):
  - Mean AUC: **0.5354 ± 0.0562** (range: 0.4584-0.6056)
  - Mean F1: **0.6586 ± 0.0164** (range: 0.6479-0.6911)
  - Mean Accuracy: **0.5193 ± 0.0454**
  - Mean Optimal Threshold: **0.588**
  - Note: AUC near random (0.5) suggests need for architecture tuning, class rebalancing, or feature engineering

### 8. Error Handling & Logging ✅ (Refactored January 2026)
- **Logging**: All modules use Python `logging` instead of `print()` statements
  - Setup in each module: 
    ```python
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    ```
  - Usage: `logger.info()`, `logger.warning()`, `logger.error()`, `logger.debug()`
  - **Status**: ✅ All core modules updated (split.py, manifest.py, annotate.py, check_progress.py, etc.)

- **Try-Catch Error Handling**: All I/O operations wrapped with specific error types
  - **CSV Loading**: `FileNotFoundError`, `pd.errors.ParserError` caught separately
  - **File Operations**: `FileNotFoundError`, `ValueError` for invalid arrays
  - **Graph Construction**: `torch.isnan()`, `torch.isinf()` checks before use
  - **DataLoader**: Null-safety for graphs with zero edges (returns `None`, skipped in training loop)
  - **Status**: ✅ All data utilities updated (extract_features.py, harmonize.py, compute_roi.py)

- **Graph Edge Cases**: 
  - Empty edge_index handled gracefully (validated in `graph_factory.py` line ~145)
  - Zero-edge graphs detected after sparsification (validated in `construct_causal.py` line ~110)
  - Subjects with insufficient edges skipped with warning logs
  - Training loop skips null graphs: `if data is None: continue` in `train_one_epoch()` and `evaluate()`

## Development Workflows

### Complete End-to-End Pipeline (Recommended)

**For existing data (skip download and split):**
```bash
# Clean run with all validations
python src/run_pipeline.py --run-diagnostics --run-manifest --skip-split --run-safe-harmonize --log-file logs/pipeline.log

# Skip diagnostics for faster execution
python src/run_pipeline.py --run-manifest --skip-split --run-safe-harmonize
```

**From scratch (with ABIDE download):**
```bash
# Full pipeline with data download (takes 2-4 hours)
python src/run_pipeline.py --run-diagnostics --run-download --run-manifest --run-safe-harmonize --log-file logs/pipeline.log

# Note: ABIDE download requires phenotype CSV with 'TR' column
# If missing, skip --run-download and use pre-processed data
```

**Pipeline execution order:**
1. Optional: Pipeline diagnostics (health check)
2. Environment validation (paths, CUDA, lobe mapping)
3. Optional: ABIDE download + preprocessing
4. Stratified split (or skip if already split)
5. Manifest generation
6. YOLO ROI detection (or skip if weights exist)
7. ROI feature extraction (1033 subjects with 5 lobes)
8. Temporal feature extraction (164 ROIs, ~1min for 1035 subjects)
9. Safe harmonization (neuroCombat with NaN/Inf handling)
10. Causal graph construction (lagged correlation)
11. GNN training (5-fold stratified CV)

### Running the Full Pipeline
```bash
# 0. Optional: Validate entire pipeline health (diagnostics on all stages)
python src/pipeline_diagnostics.py

# OR run the full pipeline with built-in diagnostics
python src/run_pipeline.py --run-diagnostics

# Full pipeline with all stages
python src/run_pipeline.py

# Full pipeline using safe harmonization (robust NaN/Inf handling)
python src/run_pipeline.py --run-safe-harmonize

# Just diagnostics before starting pipeline
python src/pipeline_diagnostics.py

# 1. Train YOLO (one-time, outputs best.pt to results/)
python src/pipelines/roi_detection.py

# 2. Extract spatial features from detections (produces node_features_3d.csv)
python src/data/extract_features.py

# 3. Extract temporal features (6 per ROI, produces node_attributes_temporal.csv)
python src/utils/compute_roi.py

# 4. Harmonize temporal features with neuroCombat (removes batch effects)
# Option A: Standard harmonization
python src/data/harmonize.py
# Option B: Safe harmonization with robust NaN handling
python src/safe_harmonization.py

# 5. Stratified split into train/val/test (2D stratification by DX_GROUP + SITE_ID)
python src/data/split.py

# 6. Build causal graphs (produces .pt files in causal_graphs/)
python src/data/construct_causal.py

# 7. Train GNN with 5-fold CV (saves checkpoints per fold)
python src/models/gnn_model.py
```

### Pipeline Command Examples
```bash
# Run diagnostics first to catch issues
python src/run_pipeline.py --run-diagnostics

# Force YOLO retraining and run full pipeline
python src/run_pipeline.py --force-yolo-train

# Skip YOLO/GNN, just run data pipeline
python src/run_pipeline.py --skip-yolo-train --skip-gnn

# Run with safe harmonization (robust NaN/Inf handling)
python src/run_pipeline.py --run-safe-harmonize

# Full pipeline from scratch (download, split, extract, harmonize, construct, train)
python src/run_pipeline.py --run-download --run-manifest --run-safe-harmonize --log-file logs/pipeline.log
```

### Testing & Validation
```bash
# Run comprehensive pipeline diagnostics (health check)
python src/pipeline_diagnostics.py

# Check dataset loading (verify labels, shapes, sample counts)
python -c "from src.features.graph_factory import ABIDECausalDataset; \
ds = ABIDECausalDataset('train'); print(f'Loaded {len(ds)} subjects, first graph has {ds[0].x.shape[0]} nodes')"

# Validate lobe mapping before graph construction
python -c "from src.core.config import validate_lobe_mapping; validate_lobe_mapping(); print('✓ LOBE_MAPPING valid')"

# Validate entire environment
python -c "from src.config import validate_environment; validate_environment()"

# Run safe harmonization (handles NaN/Inf robustly)
python src/safe_harmonization.py
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

**If temporal features CSV is corrupted:**
- **Symptom**: `pandas.errors.ParserError: Error tokenizing data. C error: Expected 1 fields in line 5, saw 986`
- **Cause**: CSV has header comments or malformed structure
- **Fix**: Delete and regenerate: `rm data/metadata/node_attributes_temporal.csv && python src/utils/compute_roi.py`
- **Note**: As of Jan 2026, `compute_roi.py` writes clean CSV without header comments

**If harmonization fails with NaN warnings:**
- Use safe harmonization: `python src/safe_harmonization.py` instead of `python src/data/harmonize.py`
- Or via pipeline: `python src/run_pipeline.py --run-safe-harmonize`
- Safe harmonization includes:
  - Pre-harmonization NaN/Inf detection
  - Feature-wise median imputation
  - Outlier capping (5σ threshold)
  - Post-harmonization validation

**If training crashes with CUDA OOM:**
- Reduce `GNN_BATCH_SIZE` from 32 to 16 in `config.py`
- Check GPU memory: `nvidia-smi` (need ~6GB for batch 32)
- Reduce YOLO batch from 24 to 16 if retraining ROI detector

**If stratification fails:**
- Ensure `SITE_ID` and `DX_GROUP` columns exist in phenotype CSV
- Check no subjects missing from manifest: `python -c "import pandas as pd; df=pd.read_csv('data/processed/Phenotypic_V1_0b_preprocessed1.csv'); print(f'Groups: {df.DX_GROUP.value_counts().to_dict()}')"` 

**If atlas validation fails:**
- **166 ROI atlas error** (fixed Jan 2026): Pipeline now accepts 164-166 ROIs for AAL3v1
- Run atlas validator: `python src/atlas_validator.py`
- Check atlas exists: `ls -la data/atlases/AAL3v1.nii`

**If ABIDE download fails:**
- **Missing 'TR' column error**: Phenotype CSV missing required column for download
- **Solution**: Skip download with existing data: `python src/run_pipeline.py --skip-split --run-manifest`
- **Path error** (fixed Jan 2026): Ensure `abide_download.py` uses `parents[2]` not `parents[0]`

**If argparse conflicts occur:**
- **Symptom**: `error: unrecognized arguments` when pipeline calls submodules
- **Fix** (applied Jan 2026): Pipeline now calls `compute_roi` via subprocess, not direct import
- Affected file: `run_pipeline.py` uses `subprocess.run()` for isolated argument parsing

## Common Pitfalls & Code Anti-Patterns

**Path Handling:**
- ❌ **Bad**: `Path("./data")` or `Path("../../../data")` (relative paths break across modules)
- ✅ **Good**: Import from config: `from src.core.config import DATA_ROOT, DATA_METADATA`
- **Status**: split.py, manifest.py, annotate.py still use hardcoded paths—refactor when touching these files

**Graph Construction:**
- ❌ **Bad**: Assuming all subjects have 5 lobes detected (some have missing detections)
- ✅ **Good**: extract_features.py enforces `node_count == 5` filter; only 1033/1035 subjects proceed
- **Impact**: Graph shape assumptions depend on this filter; skip it and downstream training crashes with shape mismatches

**Config Duplication:**
- ❌ **Bad**: Hardcoding `LOBE_MAPPING` or `NUM_LOBES` in multiple files
- ✅ **Good**: Import once from config in every module that needs it
- **Status**: Code is clean; keep it that way

**Temporal Features:**
- ❌ **Bad**: Computing ROI features from raw 170 AAL time series without aggregating to 5 lobes
- ✅ **Good**: Aggregate 170 AAL → 5 lobes FIRST (in config.LOBE_MAPPING), then extract 6 stats per lobe
- **Output shape**: Should be `(num_subjects, 5 lobes * 6 features) = (N, 30)`, not `(N, 170*6)`

**YOLO Augmentation:**
- ❌ **Bad**: Enabling `fliplr=True` or `degrees=15` for medical imaging (breaks anatomical consistency)
- ✅ **Good**: All augmentation disabled in config.py; medical images require anatomical alignment
- **Reason**: Left-right flips reverse hemisphere signals; rotations misalign Z-depth slices

**Graph Edge Attributes:**
- ❌ **Bad**: Missing `edge_attr` in PyTorch Geometric Data objects (GAT expects it)
- ✅ **Good**: Always include causal correlation weights: `Data(x=..., edge_index=..., edge_attr=weights)`
- **Shape**: `edge_attr` must be `(num_edges,)` float tensor with values in [-1, 1]

**Protected Covariates:**
- ❌ **Bad**: Passing `DX_GROUP` (diagnosis) to neuroCombat harmonization
- ✅ **Good**: Keep diagnosis out of ComBat; it's a protected covariate (journal requirement)
- **Location**: Both [src/data/harmonize.py] and [src/safe_harmonization.py] enforce this

## Key Files Reference

| File | Purpose |
|------|---------|
| [src/config.py] | ALL constants, paths, hyperparameters; validation functions |
| [src/pipeline_diagnostics.py] | Comprehensive health check for all pipeline stages |
| [src/safe_harmonization.py] | Robust feature harmonization with NaN/Inf handling |
| [src/data/extract_features.py] | YOLO inference → 3D spatial aggregation; all-5-lobes filter |
| [src/data/harmonize.py] | neuroCombat batch effect removal; protects DX_GROUP |
| [src/data/split.py] | 2D stratified split (by DX_GROUP + SITE_ID) |
| [src/data/construct_causal.py] | AAL→Lobe aggregation; lagged correlation; graph creation |
| [src/data/graph_factory.py] | PyTorch Geometric dataset loader |
| [src/models/causal_gnn.py] | GATv2 architecture with skip connections |
| [src/models/gnn_model.py] | k-fold training loop; metrics computation |
| [src/pipelines/roi_detection.py] | YOLO training entry point |
| [src/test.py] | Configuration & data integrity tests |
| [src/utils/compute_roi.py] | Temporal feature extraction from time series |
| [src/utils/manifest.py] | Master manifest generation |
| [src/run_pipeline.py] | Unified entry point (orchestrates all stages) |

## Integration Points & Critical Dependencies

### Data Flow Checkpoints
- **extract_features.py** → requires: `best.pt` (YOLO weights in results/), PNG images in `data/final/{train,val,test}/images/`
- **harmonize.py** → requires: temporal features CSV output from extract_features
- **construct_causal.py** → requires: node_attributes_harmonized.csv, time series .npy files in `data/final/{split}/time_series/`
- **gnn_model.py** → requires: all `.pt` graphs in `data/processed/causal_graphs/`, master_manifest.csv, harmonized features

### Protected Covariates in Harmonization
In [src/data/harmonize.py] and [src/safe_harmonization.py], `DX_GROUP` (diagnosis) is NEVER passed to neuroCombat—it's protected so batch harmonization doesn't remove disease signal. This is a journal Q1 requirement. Missing values in `AGE_AT_SCAN`/`SEX` are imputed (median/mode) BEFORE ComBat.

### Robust Harmonization with Safe NaN Handling
[src/safe_harmonization.py] provides production-grade harmonization with:
- Pre-harmonization NaN/Inf detection and repair
- Feature-wise median imputation for missing values
- Outlier capping (values beyond 5 standard deviations)
- Post-harmonization validation (ensures zero NaNs)
- Comprehensive logging for debugging

### Dataset Filtering: All-5-Lobes Requirement
[src/data/extract_features.py] enforces that only subjects with ALL 5 brain lobes detected proceed to GNN training. This is critical: 5-node graphs assume complete detection. Check the `node_count == 5` filter before downstream processing.

### Stratified k-fold Details
[src/data/split.py] performs 2D stratification on both `DX_GROUP` (diagnosis) AND `SITE_ID` (scanner site). This ensures:
- Balanced ASD/Control across folds (addresses class imbalance)
- Balanced sites across folds (addresses batch effects from different scanners)

## Recent Fixes & Important Changes (January 2026)

### Module Import Path Fix (January 16, 2026)
- **[src/pipeline_diagnostics.py]** - Fixed subprocess import issue: changed `sys.path.append(parents[1])` to `sys.path.insert(0, str(Path(__file__).resolve().parent))` and removed non-existent `validate_lobe_mapping` function import, replacing with inline validation. Now runs correctly via `python -m src.pipeline_diagnostics`.
- **Why it matters**: When `run_pipeline.py` calls modules as subprocesses (to avoid argparse conflicts), they execute in isolated environments. The correct path setup is: modules in `src/` should add their own directory to sys.path FIRST (parent dir), not the project root.

### Path Resolution Fixes
- **[src/data/abide_download.py]** - Fixed `PROJECT_ROOT = Path(__file__).resolve().parents[2]` (was parents[0], incorrectly pointed to src/data/ instead of project root)

### CSV/Data Format Fixes
- **[src/utils/compute_roi.py]** - Removed CSV header comments (`# atlas_name:` etc.) that broke pandas CSV parsing. Now writes clean CSV directly with `df.to_csv()`. Metadata moved to accompanying `.roi_coverage.json` file.
- **[src/safe_harmonization.py]** - Fixed pandas FutureWarning by replacing `df[col].fillna(..., inplace=True)` with proper assignment `df[col] = df[col].fillna(...)`

### Pipeline Diagnostics & Validation
- **[src/pipeline_diagnostics.py]** - Updated to accept **166 ROIs** (AAL3v1 variant) in atlas validation. Previously only accepted 116/117/170, causing false positives for valid AAL3v1 atlases.
- **[src/run_pipeline.py]** - Major refactor:
  - Added `--run-diagnostics` flag for comprehensive pre-flight checks
  - Added `--run-safe-harmonize` flag to use robust NaN/Inf handling
  - Changed `compute_roi` invocation from direct import to subprocess call (avoids argparse conflicts with sys.argv)
  - Integrated all new diagnostic and harmonization tools

### Atlas Support
- **AAL3v1 (166 ROIs)** is now fully supported alongside AAL116/117/170 variants
- Temporal feature extraction correctly detects 164 ROIs from AAL3v1 (2 ROIs may be empty/unused in specific templates)

## Medical/Scientific Context

- **Diagnosis label**: 0=Control (healthy), 1=ASD (autism spectrum disorder)
- **Data source**: ABIDE initiative (public fMRI dataset, multi-site, ~1000 subjects)
- **Causal inference goal**: Identify lobe→lobe influence patterns distinctive to ASD using lagged correlations
- **Explainability**: Edge weights in causal graph provide subject-specific feature importance (gradient-based saliency in `causal_gnn.py::get_node_importance()`)
- **Medical tuning**: YOLO augmentation disabled for grayscale medical images; preserves anatomical alignment
- **Graph construction**: Lagged Pearson correlation (not partial correlation as initially documented) between lobe i at t-1 and lobe j at t
- **Key metric**: AUC is primary metric for imbalanced classification; current results (~0.535) indicate baseline performance

### Why These Design Choices?

**5-Lobe Aggregation:**
- Reduces 170×170=28,900 edges to 5×5=25 edges (computational efficiency)
- Anatomically interpretable for clinical validation
- Reduces scanner-specific noise through within-lobe averaging
- Better statistical power with limited samples (n~1000)

**2-Head GAT (not 4-head):**
- Sufficient attention capacity for 5-node graphs
- 4+ heads cause redundancy on small graphs
- Maintains edge_attr (causal weights) integration
- Skip connections prevent over-smoothing

**Lagged Correlation (not Granger):**
- Scalable to 170 ROIs before aggregation
- Temporal precedence via lag=1 TR enforces directionality
- Robust and interpretable compared to VAR models
- Sparsification (top 20%) keeps strongest connections only
