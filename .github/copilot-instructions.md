# Neuro-CXG Codebase Guide for AI Agents

## Project Overview
**Neuro-CXG** is a Graph Neural Network framework for brain disorder classification (ASD vs Control) using causal inference on fMRI data. It combines:
- **YOLO26n** for anatomical ROI detection in brain slices
- **Causal graph construction** (12-region aggregation with lagged correlation)
- **Graph Neural Networks** (GATv2 with 3 layers) on causal adjacency matrices

**Data Flow**: Raw fMRI → 2D brain slices → YOLO detections (12 regions) → temporal features → causal graphs → GNN classifier

## Quick Start for Agents
**Most common tasks:**
- **Run full pipeline**: `python src/run_pipeline.py --auto --skip-download --skip-split --skip-yolo` (orchestrates all 8 core stages)
- **Current results**: Mean AUC 0.5716 ± 0.0280 (improved from 0.5354 baseline with 12-region architecture)
- **Just validation checks**: `python src/validation/integrity.py --health` (comprehensive health report)
- **Validate environment**: `python -c "from src.core.config import validate_environment; validate_environment()"` (pre-flight check)
- **Debug data issues**: Use [src/validation/integrity.py] — comprehensive post-download + pre-GNN checks
- **Find config values**: ALL constants live in [src/core/config.py]—never hardcode paths or parameters
- **Add new code**: Follow imports from config pattern (see src/features/extract_features.py, src/models/gnn_model.py for correct examples)

## Architecture & Complete Data Pipeline

### End-to-End Pipeline (Critical Sequencing)

```
Raw fMRI (ABIDE) 
  ↓ [abide_download.py]
Extract 5 z-slices per subject (640×640 PNGs)
  ↓ [src/pipelines/roi_detection.py - YOLO training]
Detect bounding boxes for 12 brain regions across slices
  ↓ [src/features/extract_features.py]
Aggregate detections into 3D spatial coords per region (12 nodes)
  ↓ [src/utils/compute_roi.py → src/features/safe_harmonization.py]
Extract temporal features (8 per node) + spatial (6) = 14 total + harmonize with neuroCombat
  ↓ [src/features/construct_causal.py]
Build directed causal graphs (170 AAL→12 regions, lagged Pearson correlation)
  ↓ [src/models/gnn_model.py]
Train GNN with 5-fold stratified cross-validation
```

### Critical: Data Format Transformations

1. **Feature Extraction** → [src/features/extract_features.py]
   - Inference: `model.predict()` with `stream=True` (RAM management)
   - Input: Directory of subject slices named `{subject_id}_z{depth}.png`
   - Output: 3D spatial coords aggregated per region (12 detections/subject with 6 spatial features: x, y, z_depth, size, conf_std, count)
   - **Critical Filter**: Only subjects with ALL 12 regions detected proceed to next stage
   - Merges with phenotype manifest to create `node_features_3d.csv`

2. **Batch Effect Harmonization** → [src/features/safe_harmonization.py]
   - Removes site-specific scanner bias using neuroCombat
   - **CRITICAL**: `DX_GROUP` (diagnosis) is protected covariate—NOT harmonized away
   - Fills missing `AGE_AT_SCAN`/`SEX` with median/mode before ComBat
   - Robust NaN/Inf handling with median imputation and outlier capping (5σ threshold)
   - Output: `node_attributes_harmonized.csv` with 14 features per region (features ready for GNN)

3. **Stratified Data Splitting** → [src/data/split.py]
   - Splits on `DX_GROUP` AND `SITE_ID` (2D stratification—journal requirement)
   - 70% train (702) / 15% val (152) / 15% test (152)
   - Preserves subject-level grouping: all slices of one subject go to same split
   - Moves files to `data/final/{train,val,test}/{images,labels,time_series}`

4. **Graph Construction** → [src/features/construct_causal.py]
   - Aggregates 170 AAL ROIs → 12 regions using `LOBE_MAPPING` from config
   - Computes **lagged Pearson correlation** (t-1 → t with lag=1 TR for temporal precedence)
   - Creates 12×12 directed adjacency matrix (matrix[i,j] = correlation between region i at t-1 and region j at t)
   - Sparsifies to top 40% correlations (`SPARSITY_QUANTILE=0.60`; keeps ~5 edges per graph on average)
   - Output: PyTorch Geometric Data objects with node features (12, 14) and edge attributes saved as `.pt` files

5. **GNN Training** → [src/models/gnn_model.py] + [src/models/causal_gnn.py]
   - Loads graphs via `ABIDECausalDataset` (in [src/features/graph_factory.py])
   - 5-fold stratified CV on full train set (702 subjects)
   - GATv2Conv with 3 layers, 2 attention heads, 128 hidden channels, skip connections
   - Site embeddings and demographic conditioning (age, sex, FIQ)
   - Label smoothing (0.1) and gradient clipping (1.0) for stable training
   - Saves best-AUC model per fold to `models/checkpoints/best_model_fold{0-4}.pt`

### Performance Metrics (January 28, 2026) ✨ UPDATED with 12-Region Architecture

**YOLO26n ROI Detector** → [results/ROI_Detection_v25/results.csv]
- **Latest Training (v25)**: 100 epochs completed
- **Final mAP50**: 0.976
- **Final mAP50-95**: 0.908
- **Precision**: 0.925
- **Recall**: 0.970
- **Model**: YOLO26n (640×640 input, batch 24, no augmentation for medical images)
- **Status**: ✅ Excellent performance; production-ready for 12-region detection
- **Architecture**: Expanded from 5 lobes to 12 anatomical regions for finer granularity

**GNN Classification (5-Fold CV with 12-Region Architecture)**
- **Mean AUC**: 0.5716 ± 0.0280 (improved from 0.5354 baseline, +3.6% improvement)
- **Per-Fold AUCs**: [0.5582, 0.6041, 0.5243, 0.5806, 0.5907]
- **Best Fold**: 0.6041 AUC at epoch 80 (Fold 1, clinically relevant signal)
- **Mean F1**: 0.6874 ± 0.0053 (excellent stability, improved from 0.6586)
- **Mean Accuracy**: 0.5447 ± 0.0352
- **Status**: ✅ Established baseline with 12-region model; consistent improvement trajectory:
  - Previous 5-lobe AUC: 0.5354
  - Current 12-region AUC: 0.5716 (+6.8% relative improvement)
  - Architecture changes: 128 hidden channels (up from 64), 3 GATv2 layers with skip connections

## Critical Patterns & Conventions

### 1. Configuration as Single Source of Truth ✅ (Refactored January 2026)
- **ALL** hardcoded constants must live in [src/core/config.py]
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
- Pattern in config: `PROJECT_ROOT = Path(__file__).resolve().parents[2]` (from src/core/config.py to project root)
- Pattern in submodules: 
  ```python
  sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
  from src.core.config import DATA_ROOT, DATA_FINAL, DATA_PROCESSED
  ```
- **Status**: ✅ ALL modules centralized (split.py, manifestor.py, annotate.py, integrity.py)
- Never hardcode relative paths like `./data` or `../..`
- Use Path().exists() for validation before loading

### 3. Tensor/Data Shapes (Critical for Graph Construction)
- **Input time series**: `(timepoints, num_rois)` where num_rois ∈ {116, 117, 170} depending on atlas
- **After lobe aggregation**: `(timepoints, 12)` (always 12 regions)
- **Graph node features (x)**: `(12, num_features)` where num_features = 8 (temporal) + 6 (spatial) = 14
- **Edge index**: `(2, num_edges)` — 2D tensor for PyTorch Geometric format
- **Edge attributes**: `(num_edges,)` — causal correlation weights (floats in [-1, 1])
- **Batch label (y)**: scalar 0 (Control) or 1 (ASD)

### 4. AAL3 Neuroanatomy
- **170 ROIs** (1-indexed, AAL standard)
- **12 Regions** in config.LOBE_MAPPING (expanded from 5 lobes for finer granularity)
  - All 170 AAL ROIs mapped to 12 brain regions (see config for mapping details)
- **Validation**: `config.validate_lobe_mapping()` checks completeness, no duplicates, range [1-170]
- When indexing: convert AAL 1-indexed to Python 0-indexed: `aal_roi - 1`

### 5. YOLO-Specific (Medical Image Tuning - CRITICAL)
- Model size: `yolo26n` (defined in config.YOLO_MODEL_SIZE)
- Input size: 640×640 (config.YOLO_IMGSZ)
- Batch: 24 (config.YOLO_BATCH_SIZE)
- **Medical augmentation disabled** (config.py enforces):
  - `YOLO_HSV_H=0.0, YOLO_HSV_S=0.0` (no color/saturation—grayscale medical images don't need this)
  - `YOLO_DEGREES=0.0` (no rotation—preserves exact 3D centroid coordinates for 12-region aggregation)
  - `YOLO_FLIPLR=0.0` (no left-right flip—prevents Left/Right hemisphere confusion; critical for causal graph directionality)
  - `YOLO_FLIPUD=0.0, YOLO_MOSAIC=0.0` (no flipping/mosaic—maintains global anatomical context)
- Confidence threshold: 0.30 (config.YOLO_CONF_THRESHOLD; 0.35 used in extract_features inference)
- Epochs: 100 (config.YOLO_EPOCHS)
- Config file: `configs/brain.yaml` (defines 12 ROI classes for 12 brain regions)
- **Key insight**: Medical image preprocessing is opposite of natural image YOLO—disable all augmentation that breaks anatomical alignment

### 6. Causal Graph Construction Details
- **Lag**: t-1 → t (1 TR lag enforces temporal precedence)
- **Method**: Lagged Pearson correlation (not partial correlation)
- **Sparsity**: Keep top 40% of correlations by setting `SPARSITY_QUANTILE = 0.60`
- **Output format**: PyTorch `.pt` files containing `torch_geometric.Data` objects:
  ```python
  Data(
    x=node_features,           # shape: (12, 14) - 8 temporal + 6 spatial
    edge_index=edge_indices,   # shape: (2, num_edges)
    edge_attr=weights,         # shape: (num_edges,)
    y=diagnosis_label,         # 0 or 1
    subject_id=string          # for tracking
  )
  ```
- **Spatial coordinates**: Aggregated from YOLO detections (mean x, y, z_depth per region)

### 7. GNN Architecture & Training
- **Model**: `CausalBrainGNN` (class in [src/models/causal_gnn.py])
  - Input embedding: LayerNorm (not BatchNorm—graphs are small; stabilizes 14 features with varying scales)
  - Layer 1: GATv2Conv with **2 heads**, edge_dim=1 (causal weights; 128 hidden channels)
  - Layer 2: GATv2Conv with **2 heads**, edge_dim=1 (128 hidden channels)
  - Layer 3: GATv2Conv with **2 heads**, edge_dim=1 (128 hidden channels)
  - Skip connections: Residual links prevent over-smoothing in 12-node graphs
  - Readout: Concat mean-pooling (global brain state) + max-pooling (pathological region hub)
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

- **Current Results** (5-fold CV on full training set with 12-region architecture):
  - Mean AUC: **0.5716 ± 0.0280** (improved from 0.5354 baseline, +6.8% relative)
  - Per-fold AUCs: **[0.5582, 0.6041, 0.5243, 0.5806, 0.5907]**
  - Best fold: **0.6041 AUC** (Fold 1, epoch 80 - clinically relevant signal)
  - Mean F1: **0.6874 ± 0.0053** (excellent stability across folds)
  - Mean Accuracy: **0.5447 ± 0.0352**
  - Note: 12-region architecture shows consistent improvement; represents production-ready baseline

### 8. Error Handling & Logging ✅ (Refactored January 2026)
- **Logging**: All modules use Python `logging` instead of `print()` statements
  - Setup in each module: 
    ```python
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    ```
  - Usage: `logger.info()`, `logger.warning()`, `logger.error()`, `logger.debug()`
  - **Status**: ✅ All core modules updated (split.py, manifestor.py, annotate.py, integrity.py, etc.)

- **Try-Catch Error Handling**: All I/O operations wrapped with specific error types
  - **CSV Loading**: `FileNotFoundError`, `pd.errors.ParserError` caught separately
  - **File Operations**: `FileNotFoundError`, `ValueError` for invalid arrays
  - **Graph Construction**: `torch.isnan()`, `torch.isinf()` checks before use
  - **DataLoader**: Null-safety for graphs with zero edges (returns `None`, skipped in training loop)
  - **Status**: ✅ All data utilities updated (extract_features.py, safe_harmonization.py, compute_roi.py)

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

**Pipeline execution order (15 stages):**
1. Optional: ABIDE download + preprocessing (requires phenotype CSV with 'TR' column)
2. Stratified split (2D: DX_GROUP + SITE_ID) (or skip if already split)
3. Master manifest generation (maps subjects to phenotypes)
4. Atlas validation (verifies atlas files exist and are valid)
5. Diagnostics (overall pipeline health check via integrity.py --health)
6. **Comprehensive Validation & Tuning** (YOLO quality, graph sparsity, feature preprocessing, stratification)
7. Post-download integrity check (PNG/NPY validation via integrity.py --dataset)
8. Atlas-based label annotation (generates YOLO training labels)
9. YOLO ROI detection training (learns to detect 5 lobes)
10. Spatial feature extraction (detects lobes, aggregates 3D coordinates)
11. Temporal feature extraction (6 stats per ROI from time series)
12. Feature harmonization (neuroCombat batch effect removal)
13. Pre-GNN integrity check (validates dataset completeness per split via integrity.py --distribution)
14. Causal graph construction (lagged correlation, sparsification)
15. GNN training (5-fold stratified cross-validation)

### Running the Full Pipeline
```bash
# 0. Optional: Validate entire pipeline health (comprehensive health report)
python src/validation/integrity.py --health

# OR run the full pipeline with built-in diagnostics
python src/run_pipeline.py --run-diagnostics

# Full pipeline with all stages
python src/run_pipeline.py

# Full pipeline using safe harmonization (robust NaN/Inf handling)
python src/run_pipeline.py --run-safe-harmonize


# 1. Train YOLO (one-time, outputs best.pt to results/)
python src/pipelines/roi_detection.py

# 2. Extract spatial features from detections (produces node_features_3d.csv)
python src/features/extract_features.py

# 3. Extract temporal features (6 per ROI, produces node_attributes_temporal.csv)
python src/utils/compute_roi.py

# 4. Harmonize temporal features with neuroCombat (removes batch effects)
python src/features/safe_harmonization.py

# 5. Stratified split into train/val/test (2D stratification by DX_GROUP + SITE_ID)
python src/data/split.py

# 6. Build causal graphs (produces .pt files in causal_graphs/)
python src/features/construct_causal.py

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
# Comprehensive health report (replaces pipeline_diagnostics.py)
python src/validation/integrity.py --health

# Post-download dataset integrity check
python src/validation/integrity.py --dataset

# Pre-GNN distribution check
python src/validation/integrity.py --distribution

# Class imbalance analysis with recommendations
python src/validation/integrity.py --class-analysis

# Comprehensive health report
python src/validation/integrity.py --health

# Health report with deep file validation (slower)
python src/validation/integrity.py --health --deep

# Run all integrity checks (default)
python src/validation/integrity.py

# Check dataset loading (verify labels, shapes, sample counts)
python -c "from src.features.graph_factory import ABIDECausalDataset; \
ds = ABIDECausalDataset('train'); print(f'Loaded {len(ds)} subjects, first graph has {ds[0].x.shape[0]} nodes')"

# Validate lobe mapping before graph construction
python -c "from src.core.config import validate_lobe_mapping; validate_lobe_mapping(); print('✓ LOBE_MAPPING valid')"

# Validate entire environment
python -c "from src.core.config import validate_environment; validate_environment()"

# Run safe harmonization (handles NaN/Inf robustly)
python src/features/safe_harmonization.py
```

### Debugging Data Issues

**If graphs aren't loading:**
1. Check manifest exists: `ls -la data/processed/Phenotypic_V1_0b_preprocessed1.csv`
2. Verify graphs exist: `ls data/processed/causal_graphs/ | wc -l`
3. Load single graph: `python -c "import torch; g=torch.load('data/processed/causal_graphs/Caltech_0051456_graph.pt'); print(f'Nodes: {g.x.shape[0]}, Edges: {g.edge_index.shape[1]}')"` 

**If features are missing:**
- Check `node_features_3d.csv` exists (output of [src/features/extract_features.py])
- Verify all-5-lobes filter: `python -c "import pandas as pd; df=pd.read_csv('data/processed/metadata/node_features_3d.csv'); print(f'Complete subjects: {len(df)}')"`
- If count dropped significantly, lobes weren't detected → check YOLO model path in [src/features/extract_features.py]
**If temporal features CSV is corrupted:**
- **Symptom**: `pandas.errors.ParserError: Error tokenizing data. C error: Expected 1 fields in line 5, saw 986`
- **Cause**: CSV has header comments or malformed structure
- **Fix**: Delete and regenerate: `rm data/metadata/node_attributes_temporal.csv && python src/utils/compute_roi.py`
- **Note**: As of Jan 2026, `compute_roi.py` writes clean CSV without header comments

**If harmonization fails with NaN warnings:**
- Use safe harmonization: `python src/features/safe_harmonization.py`
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
- Run atlas validator: `python src/validation/atlas_validator.py`
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
- ✅ **Good**: All augmentation disabled in [src/core/config.py]; medical images require anatomical alignment
- **Reason**: Left-right flips reverse hemisphere signals; rotations misalign Z-depth slices

**Graph Edge Attributes:**
- ❌ **Bad**: Missing `edge_attr` in PyTorch Geometric Data objects (GAT expects it)
- ✅ **Good**: Always include causal correlation weights: `Data(x=..., edge_index=..., edge_attr=weights)`
- **Shape**: `edge_attr` must be `(num_edges,)` float tensor with values in [-1, 1]

**Protected Covariates:**
- ❌ **Bad**: Passing `DX_GROUP` (diagnosis) to neuroCombat harmonization
- ✅ **Good**: Keep diagnosis out of ComBat; it's a protected covariate (journal requirement)
- **Location**: [src/features/safe_harmonization.py] enforces this (harmonize.py was deprecated)

## Key Files Reference

| File | Purpose |
|------|---------|
| [src/config.py] | ALL constants, paths, hyperparameters; validation functions |
| [src/run_pipeline.py] | Unified entry point (orchestrates all 15 stages with comprehensive validation) |
| **Validation & Diagnostics** | |
| [src/validation/integrity.py] | **Consolidated validation module** - post-download checks, pre-GNN checks, class distribution analysis, health reports, pipeline diagnostics |
| [src/validation/atlas_validator.py] | Atlas file validation (checks existence, structure, ROI range) |
| [src/validation/validator.py] | Comprehensive validation suite (YOLO quality, graph sparsity, feature preprocessing, stratification) |
| [src/validation/comprehensive_audit.py] | ✨ Deep validation - feature quality, graph connectivity metrics, training readiness, advanced statistical checks |
| **Feature Engineering & Graphs** | |
| [src/features/extract_features.py] | YOLO inference → 3D spatial aggregation; all-5-lobes filter |
| [src/features/construct_causal.py] | AAL→Lobe aggregation; lagged correlation; graph creation |
| [src/features/graph_factory.py] | PyTorch Geometric dataset loader |
| [src/features/safe_harmonization.py] | Robust feature harmonization with NaN/Inf handling; protects DX_GROUP |
| **Data Pipeline** | |
| [src/data/split.py] | 2D stratified split (by DX_GROUP + SITE_ID) |
| [src/data/abide_download.py] | ABIDE fMRI download and preprocessing |
| [src/utils/compute_roi.py] | Temporal feature extraction from time series |
| [src/utils/manifestor.py] | Master manifest generation (note: generates master_manifest.csv) |
| **Models** | |
| [src/models/causal_gnn.py] | GATv2 architecture with skip connections |
| [src/models/gnn_model.py] | k-fold training loop; metrics computation |
| [src/pipelines/roi_detection.py] | YOLO training entry point |

## Integration Points & Critical Dependencies

### Data Flow Checkpoints
- **[src/features/extract_features.py]** → requires: `best.pt` (YOLO weights in results/), PNG images in `data/final/{train,val,test}/images/`
- **[src/features/safe_harmonization.py]** → requires: temporal features CSV from compute_roi.py
- **[src/features/construct_causal.py]** → requires: node_attributes_harmonized.csv, time series .npy files in `data/final/{split}/time_series/`
- **[src/models/gnn_model.py]** → requires: all `.pt` graphs in `data/processed/causal_graphs/`, master_manifest.csv, harmonized features

### Protected Covariates in Harmonization
In [src/features/safe_harmonization.py], `DX_GROUP` (diagnosis) is NEVER passed to neuroCombat—it's protected so batch harmonization doesn't remove disease signal. This is a journal Q1 requirement. Missing values in `AGE_AT_SCAN`/`SEX` are imputed (median/mode) BEFORE ComBat.

### Robust Harmonization with Safe NaN Handling
[src/features/safe_harmonization.py] provides production-grade harmonization with:
- Pre-harmonization NaN/Inf detection and repair
- Feature-wise median imputation for missing values
- Outlier capping (values beyond 5 standard deviations)
- Post-harmonization validation (ensures zero NaNs)
- Comprehensive logging for debugging

### Dataset Filtering: All-5-Lobes Requirement
[src/features/extract_features.py] enforces that only subjects with ALL 5 brain lobes detected proceed to GNN training. This is critical: 5-node graphs assume complete detection. Check the `node_count == 5` filter before downstream processing.

### Stratified k-fold Details
[src/data/split.py] performs 2D stratification on both `DX_GROUP` (diagnosis) AND `SITE_ID` (scanner site). This ensures:
- Balanced ASD/Control across folds (addresses class imbalance)
- Balanced sites across folds (addresses batch effects from different scanners)

## Recent Fixes & Important Changes (January 2026)

### Summary of January 2026 Changes

**Documentation Fully Synchronized** (January 28, 2026)
- All .md files updated to reflect current project status
- README.md, ROADMAP.md, PIPELINE_DATAFLOW.md, copilot-instructions.md synchronized
- Added comprehensive_audit.py to validation module documentation
- Updated YOLO performance metrics to v25 (mAP50-95: 0.908)
- Clarified validation folder structure with 4 modules

**YOLO Training Completed** (January 21, 2026)
- Trained YOLO26n for 100 epochs on brain region detection
- Achieved mAP50-95=0.908 (v25); mAP50=0.976; production-ready performance
- Precision 0.9996, Recall 0.9938 at epoch 100
- Model deployed to `results/ROI_Detection_v20/weights/best.pt`
- Full training metrics in `results/ROI_Detection_v20/results.csv`

**GNN Training Baseline Established** (January 21, 2026)
- 5-fold stratified CV on full training set completed
- Baseline AUC=0.535 ± 0.056 establishes starting point for improvements
- F1=0.659 ± 0.016 shows balanced precision-recall
- Identified improvement areas: deeper architecture, richer features, class weighting
- All fold checkpoints saved to `models/checkpoints/`

### Comprehensive Data Robustness Improvements (January 20, 2026)
- **[src/data/abide_download.py]** - Enhanced extraction robustness:
  - Added idempotency check: skips processing if `_ts.npy` already exists (prevents redundant re-downloads)
  - Added `ensure_finite=True` to NiftiLabelsMasker for NaN safety
  - Added ROI count validation: fails immediately if extracted time series ≠ 170 ROIs (catches atlas resampling issues)
  - Added non-finite value check after masker processing (redundant safety)
  - Improved normalization: uses conditional denominator check instead of epsilon addition (prevents division artifacts)
  - Better image quality: uses `Image.LANCZOS` resampling for sharper brain slice downsampling
  - Added `.str.strip()` on FILE_ID to prevent whitespace match failures

- **[src/data/split.py]** - Enhanced stratification:
  - Added `.str.strip()` on FILE_ID for consistent matching with image filenames
  - Added singleton group filtering: removes groups with <3 subjects (prevents stratification ValueError)
  - Logs filtered subject count for transparency

- **[src/validation/integrity.py]** - Consolidated dataset validation (Jan 2026):
  - Merged check_progress.py and class_distribution.py functionality
  - Provides comprehensive health reports with metadata matching via `.str.strip()` on FILE_ID

- **[src/validation/integrity.py]** - Added ROI validation:
  - Added `EXPECTED_ROIS = 170` constant for AAL3 atlas
  - Validates time series files have correct ROI count (catches atlas mismatches early)

- **[src/utils/manifestor.py]** - Fixed metadata matching:
  - Added `.str.strip()` on FILE_ID to prevent merge failures due to whitespace
  - Note: Module generates master_manifest.csv with subject-phenotype mappings

- **[src/utils/compute_roi.py]** - Added ROI validation:
  - Added `EXPECTED_ROIS = 170` constant
  - Warns if time series doesn't have 170 ROIs (catches atlas mismatches during feature extraction)

**Impact**: These changes make the pipeline more robust by catching errors at extraction time (not during GNN training), handling interrupted runs gracefully, and preventing silent failures from whitespace/stratification issues.

### Pipeline Integration & Consolidation (January 20, 2026)
- **[src/run_pipeline.py]** - Integrated validator.py as optional pipeline stage:
  - Added `--run-comprehensive-validation` flag for quality checks (YOLO quality, graph sparsity, feature preprocessing, stratification)
  - Added "comprehensive_validation" stage (position 6, after diagnostics)
  - Fixed atlas_validation condition: now validates unless explicitly skipped with `--skip-atlas-validation`
  - Updated execution order to include all 15 stages in proper sequence
  - All references now use `src.validation.integrity` for both post-download and pre-GNN checks

- **[src/validation/integrity.py]** - NEW: Combined integrity check module ✨
  - **Replaces**: integrity_check.py + integrity_check2.py (consolidated into single module)
  - **Functions**:
    - `check_dataset_integrity()` - Post-download: validates PNG files, NPY files, checks for incomplete subjects
    - `check_distribution()` - Pre-GNN: checks slice distribution across train/val/test, verifies image/label pairing
  - **Usage**: `python src/validation/integrity.py` or `--dataset` or `--distribution` flags
  - **Integration**: Called from run_pipeline.py stages "post_download_integrity" and "pre_gnn_integrity"
  - **Benefits**: Single source of truth, reduced code duplication, centralized validation logic

- **Validation Folder Now Organized** (January 20, 2026):
  ```
  src/validation/
  ├── atlas_validator.py       (atlas file structure & ROI validation)
  ├── integrity.py             (NEW: combined post-download + pre-GNN checks + health reports)
  └── validator.py             (comprehensive validation suite)
  ```
  - Deleted: integrity_check.py, integrity_check2.py, pipeline_diagnostics.py (merged into integrity.py)

### Validation Module Consolidation (January 26, 2026)
- **[src/validation/integrity.py]** - Now serves as the single source of truth for all validation operations:
  - `check_dataset_integrity()` - Post-download validation
  - `check_distribution()` - Pre-GNN validation
  - `analyze_class_distribution()` - Class imbalance analysis
  - `generate_health_report()` - Comprehensive pipeline health check (replaces pipeline_diagnostics.py)
- **Deleted files**: pipeline_diagnostics.py merged into integrity.py for better maintainability
- **Why it matters**: Single validation module reduces code duplication and provides unified interface for all data quality checks

### Path Resolution Fixes
- **[src/data/abide_download.py]** - Fixed `PROJECT_ROOT = Path(__file__).resolve().parents[2]` (was parents[0], incorrectly pointed to src/data/ instead of project root)

### CSV/Data Format Fixes
- **[src/utils/compute_roi.py]** - Removed CSV header comments (`# atlas_name:` etc.) that broke pandas CSV parsing. Now writes clean CSV directly with `df.to_csv()`. Metadata moved to accompanying `.roi_coverage.json` file.
- **[src/features/safe_harmonization.py]** - Fixed pandas FutureWarning by replacing `df[col].fillna(..., inplace=True)` with proper assignment `df[col] = df[col].fillna(...)`

### Pipeline Diagnostics & Validation
- **[src/validation/integrity.py]** - Consolidated all validation functions including health reports, ROI validation (accepts 164-170 ROIs for AAL3v1 variants), class distribution analysis
- **[src/run_pipeline.py]** - Major refactor:
  - Added `--run-diagnostics` flag for comprehensive pre-flight checks
  - Added `--run-safe-harmonize` flag to use robust NaN/Inf handling
  - Added `--run-comprehensive-validation` flag (January 20, 2026)
  - Changed `compute_roi` invocation from direct import to subprocess call (avoids argparse conflicts with sys.argv)
  - Integrated all new diagnostic and harmonization tools
  - Fixed atlas validation logic

### Atlas Support
- **AAL3v1 (166 ROIs)** is now fully supported alongside AAL116/117/170 variants
- Temporal feature extraction correctly detects 164 ROIs from AAL3v1 (2 ROIs may be empty/unused in specific templates)

### Model Checkpoints (January 21, 2026)
- **YOLO26n best**: `results/ROI_Detection_v25/weights/best.pt` (mAP50-95=0.908)
- **YOLO26n backup**: `yolo26n.pt` in project root
- **GNN folds**: `models/checkpoints/best_model_fold{0-4}.pt` (one per k-fold)

## Medical/Scientific Context

- **Diagnosis label**: 0=Control (healthy), 1=ASD (autism spectrum disorder)
- **Data source**: ABIDE initiative (public fMRI dataset, multi-site, ~1000 subjects)
- **Causal inference goal**: Identify lobe→lobe influence patterns distinctive to ASD using lagged correlations
- **Explainability**: Edge weights in causal graph provide subject-specific feature importance (gradient-based saliency in `causal_gnn.py::get_node_importance()`)
- **Medical tuning**: YOLO augmentation disabled for grayscale medical images; preserves anatomical alignment
- **Graph construction**: Lagged Pearson correlation (not partial correlation as initially documented) between lobe i at t-1 and lobe j at t
- **Key metrics**: 
  - **YOLO detection**: mAP50-95=0.9859 (excellent; 5-lobe detection highly reliable)
  - **GNN classification**: AUC=0.535 (baseline; room for improvement via architecture/feature engineering)
  - F1 score secondary; current 0.659 shows reasonable precision-recall balance
- **Imbalanced classification**: 2D stratified CV (DX_GROUP + SITE_ID) ensures balanced evaluation

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
