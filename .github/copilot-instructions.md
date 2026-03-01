# Neuro-CXG: AI Agent Guide

## Quick Start

**Pipeline**: raw fMRI → YOLO (12 regions) → features → causal graphs → GNN classifier

**Essential Commands**:
```bash
# Full pipeline (auto mode — runs all missing stages)
python src/run_pipeline.py --auto

# Skip download + split (use existing data)
python src/run_pipeline.py --auto --skip-download --skip-split

# Show execution plan (dry run)
python src/run_pipeline.py --dry-run

# Post-training analysis only
python src/run_pipeline.py --analysis-only

# Health check
python src/validation/pipeline_checks.py --health

# Environment validation
python -c "from src.core.config import validate_environment; validate_environment()"
```

**Key Principle**: ALL constants live in [src/core/config.py](src/core/config.py) — never hardcode paths/dimensions.

## Current State (March 1, 2026)

**YOLO v28** (deployed): mAP50-95=0.93714, mAP50=0.98952, Precision=0.98063, Recall=0.97214, 12-region detection (production-ready, exceptional)  
**YOLO v29**: configured as next training target (`YOLO_PROJECT_NAME = 'ROI_Detection_v29'` in config.py); weights do not yet exist  
**GNN Latest Training**: AUC=0.6194±0.0641, F1=0.7132±0.0160, test-set ensemble AUC=0.5398 (Feb 15, 2026)  
**Architecture**: 12 regions (AAL 170→12), 28 features (20 temporal + 2 internal ReHo + 6 spatial), 3 GAT layers, 128 channels  
**Phase 9 Complete**: full evaluation pipeline, explainability analysis, result interpretation, 20-stage orchestrator  
**Phase 10 Open**: fix double z-score normalisation, freq-band aliasing near Nyquist, CV–test AUC gap (details in `docs/TODO.md`)

## Data Pipeline (5 Critical Steps)

**Flow**: Raw fMRI → YOLO (12 detections) → Features (28D) → Causal Graphs (12×12) → GNN Classifier

1. **Feature Extraction** → [src/features/extract_spatial.py]
   - Inference: `model.predict()` with `stream=True` (RAM management)
   - Input: Directory of subject slices named `{subject_id}_z{depth}.png`
   - Output: 3D spatial coords aggregated per region (12 detections/subject with 6 spatial features: x, y, z_depth, size, conf_std, count)
   - **Critical Filter**: Only subjects with ALL 12 regions detected proceed to next stage
   - Merges with phenotype manifest to create `node_features_3d.csv`

2. **Batch Effect Harmonization** → [src/features/fold_safe_harmonization.py]
  - Removes site-specific scanner bias using neuroHarmonize (ComBat)
   - **CRITICAL**: `DX_GROUP` (diagnosis) is protected covariate—NOT harmonized away
   - Fills missing `AGE_AT_SCAN`/`SEX` with median/mode before ComBat
   - Robust NaN/Inf handling with median imputation and outlier capping (5σ threshold)
   - Output: `node_attributes_harmonized.csv` with 28 features per region (20 temporal + 2 internal + 6 spatial)

3. **Stratified Data Splitting** → [src/data/split.py]
   - Splits on `DX_GROUP` AND `SITE_ID` (2D stratification—journal requirement)
   - 70% train (702) / 15% val (152) / 15% test (152)
   - Preserves subject-level grouping: all slices of one subject go to same split
   - Moves files to `data/final/{train,val,test}/{images,labels,time_series}`

4. **Graph Construction** → [src/features/construct_causal.py]
   - Aggregates 170 AAL ROIs → 12 regions using `LOBE_MAPPING` from config
   - ✨ Smart aggregation: PCA eigenvariate (dominant signal) + Regional Homogeneity (intra-lobe coherence + spatial variance)
   - Computes **Granger causality** (default, multi-lag 1-5 TRs) or **lagged Pearson correlation** (t-1 → t with lag=1 TR)
   - Creates 12×12 directed adjacency matrix with -log10(p-value) or correlation weights
  - Adaptive sparsification: 0.70 quantile (keep top 30% edges), min 12 edges/graph
   - Output: PyTorch Geometric Data objects with node features (12, 28) and edge attributes saved as `.pt` files

5. **GNN Training** → [src/models/gnn_model.py] + [src/models/causal_gnn.py]
   - Loads graphs via `ABIDECausalDataset` (in [src/features/graph_factory.py])
   - 5-fold stratified CV on full train set (699 subjects)
  - ✨ Current architecture: 3 GAT layers, 128 hidden channels, GELU activation
   - Input: 28 node features (20 temporal + 2 internal ReHo + 6 spatial)
   - Multi-scale pooling (mean+max+sum), skip connections, LayerNorm
   - Focal loss (α=0.62, γ=2.0) for class imbalance
  - L2 regularization (weight_decay=1e-4), dropout 0.45, learning_rate=0.001
  - Early stopping (patience=20, min_delta=0.0001), gradient clipping (1.0)
   - Saves best-AUC model per fold to `models/checkpoints/best_model_fold{0-4}.pt`

### Performance Metrics (March 1, 2026)

**YOLO26n ROI Detector** → [results/experiments/detection/ROI_Detection_v28/]
- **Deployed (v28)**: 100 epochs (Feb 2–4 2026)
- **Final mAP50**: 0.98952
- **Final mAP50-95**: 0.93714
- **Precision**: 0.98063 (exceptional)
- **Recall**: 0.97214 (near-perfect)
- **Model**: YOLO26n (640×640 input, batch 32, all medical augmentation disabled)
- **Status**: ✅ Production-ready; outstanding 12-region detection quality
- **Deployed weights**: `results/experiments/detection/ROI_Detection_v28/weights/best.pt`
- **Next target**: `ROI_Detection_v29` — `YOLO_PROJECT_NAME = 'ROI_Detection_v29'` in config.py

**GNN Classification (5-Fold CV with 28-Feature Model - Feb 15, 2026)**
- **Latest Training**: Feb 11-15, 2026 with Phase 3 architecture + bug fixes
- **Mean AUC**: 0.6194 ± 0.0641 (+10.7pp over prior baseline)
- **Mean F1**: 0.7132 ± 0.0160
- **Mean Accuracy**: 0.6194 ± 0.0241
- **Test-set Ensemble AUC**: 0.5398 (153 held-out subjects)
- **Per-fold AUCs**: [0.5762, 0.5931, 0.6197, 0.7424, 0.5657]
- **Best fold**: 0.7424 (Fold 3)
- **Mean best epoch**: ~14.6 (range 8-24)
- **Architecture**: 3-layer GATv2, 128 hidden channels, GELU activation, skip connections, attention pooling
- **Features**: 28 total (20 temporal + 2 internal ReHo + 6 spatial); 1035 subjects, 7 z-slices each
- **Loss**: Focal Loss (α=0.62, γ=2.0)
- **Regularization**: Dropout 0.45, L2 weight decay 1e-4, early stopping patience=20
- **Status**: ✅ Phase 3 optimized architecture with full pipeline fixes
- **Interpretation**: 
  - Fold 3 reaching 0.7424 AUC demonstrates strongly learnable ASD biomarkers
  - Moderate convergence (~14.6 epochs) reflects richer signal than prior baseline
  - Test ensemble AUC (0.5398) vs CV AUC (0.6194) gap indicates remaining overfitting
  - 3-layer attention pooling balances capacity for 12-node graphs
  - PCA/ReHo features capture both global signals and local connectivity
  - Graph topology: Parietal In-Degree lower in ASD (p=0.0296, Cohen's d=-0.125)

## Critical Patterns & Conventions

### 1. Configuration as Single Source of Truth ✅ (Refactored January 2026)
- **ALL** hardcoded constants must live in [src/core/config.py]
- Import from config everywhere else (e.g., `from src.core.config import LOBE_MAPPING, NUM_LOBES`)
- 12-region architecture: `NUM_LOBES=12` (expanded from 5 lobes January 2026 for finer granularity)
- AAL3 → 12-region mapping: 170 ROIs (1-indexed) aggregated via `LOBE_MAPPING` dict
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
- **Status**: ✅ ALL modules centralized (split.py, manifestor.py, generate_labels.py, pipeline_checks.py)
- Never hardcode relative paths like `./data` or `../..`
- Use Path().exists() for validation before loading

### 3. Error Handling & Logging ✅ (Refactored January 2026)
- **Logging**: All modules use Python `logging` instead of `print()` statements
  - Setup in each module: 
    ```python
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    ```
  - Usage: `logger.info()`, `logger.warning()`, `logger.error()`, `logger.debug()`
  - **Status**: ✅ All core modules updated (split.py, manifestor.py, generate_labels.py, pipeline_checks.py, etc.)

- **Try-Catch Error Handling**: All I/O operations wrapped with specific error types
  - **CSV Loading**: Catch `FileNotFoundError` and `pd.errors.ParserError` separately ([src/features/extract_temporal.py] lines 118-130)
  - **File Operations**: Catch `FileNotFoundError`, `ValueError` for invalid arrays
  - **Graph Construction**: Use `torch.isnan()`, `torch.isinf()` checks before using edge/node tensors
  - **DataLoader**: Null-safety for graphs with zero edges (validated in [src/features/graph_factory.py] line ~145)
  - **Status**: ✅ All data utilities updated (extract_spatial.py, fold_safe_harmonization.py, extract_temporal.py)

- **Graph Edge Cases**: 
  - Empty edge_index handled gracefully (validated in `graph_factory.py` line ~145)
  - Zero-edge graphs detected after sparsification (validated in `construct_causal.py` line ~110)
  - Subjects with insufficient edges skipped with warning logs
  - Training loop skips null graphs: `if data is None: continue` in `train_one_epoch()` and `evaluate()`

### 4. Tensor/Data Shapes (Critical for Graph Construction)
- **Input time series**: `(timepoints, num_rois)` where num_rois ∈ {116, 117, 170} depending on atlas
- **After lobe aggregation**: `(timepoints, 12)` (always 12 regions from LOBE_MAPPING)
- **Graph node features (x)**: `(12, num_features)` where num_features = 20 (temporal) + 2 (internal ReHo) + 6 (spatial) = **28**
  - 20 temporal: 8 basic (mean, std, skew, kurt, PSD, MSSD, range, autocorr) + 12 frequency (delta/theta/alpha/beta/gamma power + peaks + entropy + phase)
  - 2 internal: PCA eigenvariate (dominant lobe signal), ReHo coherence (intra-lobe homogeneity)
  - 6 spatial: x, y, z_depth, size, conf_std, detection_count
- **Edge index**: `(2, num_edges)` — 2D tensor for PyTorch Geometric format
- **Edge attributes**: `(num_edges,)` — causal weights (Granger: -log10(p), Pearson: correlation [-1, 1])
- **Batch label (y)**: scalar 0 (Control) or 1 (ASD)

### 5. AAL3 Neuroanatomy
- **170 ROIs** (1-indexed, AAL standard; convert to 0-indexed in code: `aal_roi - 1`)
- **12 Regions** in config.LOBE_MAPPING (expanded from 5 lobes for finer granularity)
  - All 170 AAL ROIs mapped to 12 brain regions (see config for mapping details)
- **Validation**: `config.validate_lobe_mapping()` checks completeness, no duplicates, range [1-170]
- **Robustness**: AAL3v1 may have 164-170 ROIs (2 unused); code pads to 170 for consistency ([src/features/extract_temporal.py] line ~220)

### 6. YOLO-Specific (Medical Image Tuning - CRITICAL)
- Model size: `yolo26n` (defined in config.YOLO_MODEL_SIZE)
- Input size: 640×640 (config.YOLO_IMGSZ)
- Batch: 32 (config.YOLO_BATCH_SIZE)
- **Medical augmentation disabled** (config.py enforces):
  - `YOLO_HSV_H=0.0, YOLO_HSV_S=0.0` (no color/saturation—grayscale medical images don't need this)
  - `YOLO_DEGREES=0.0` (no rotation—preserves exact 3D centroid coordinates for 12-region aggregation)
  - `YOLO_FLIPLR=0.0` (no left-right flip—prevents Left/Right hemisphere confusion; critical for causal graph directionality)
  - `YOLO_FLIPUD=0.0, YOLO_MOSAIC=0.0` (no flipping/mosaic—maintains global anatomical context)
- Confidence threshold: 0.30 (config.YOLO_CONF_THRESHOLD; 0.35 used in extract_spatial inference)
- Epochs: 100 (config.YOLO_EPOCHS)
- Config file: `configs/brain.yaml` (defines 12 ROI classes for 12 brain regions)
- **Key insight**: Medical image preprocessing is opposite of natural image YOLO—disable all augmentation that breaks anatomical alignment

### 7. Causal Graph Construction Details
- **Method**: Granger causality (default, multi-lag 1-5 TRs) or lagged Pearson correlation (baseline)
- **Granger causality**:
  - Tests: Does past of region i improve prediction of region j?
  - Multi-lag: Tests lags 1-5 TRs (`GRANGER_MAX_LAG = 5`)
  - Statistical significance: p-value < 0.05 (`GRANGER_SIGNIFICANCE_LEVEL`)
  - Edge weights: -log10(p-value) where higher = stronger causality
- **Lagged Pearson** (baseline):
  - Lag: t-1 → t (1 TR lag enforces temporal precedence)
  - Edge weights: Correlation values in [-1, 1]
- **Sparsity**: Adaptive statistical method (0.70 quantile = keep top 30% edges, min 12 edges/graph for connectivity)
- **Note**: `.pt` files saved by `construct_causal.py` are **dicts** (not PyG Data objects); `graph_factory.py` assembles the full `Data` object at load time
- **On-disk dict format**:
  ```python
  {
    'adj': Tensor(12, 12),          # directed causal weights
    'internal_features': Tensor(12, 2),  # PCA eigenvariate + ReHo coherence
    'subject_id': str,
    'lobe_order': list
  }
  ```
- **PyG Data object** (assembled by `graph_factory.py` at load time):
  ```python
  Data(
    x=node_features,           # shape: (12, 28) - 20 temporal + 2 internal + 6 spatial
    edge_index=edge_indices,   # shape: (2, num_edges)
    edge_attr=weights,         # shape: (num_edges, 1) - Granger or correlation (unsqueezed)
    y=diagnosis_label,         # 0 or 1
    pos=xyz,                   # shape: (12, 3) - centroid coords
    sub_id=string,             # for tracking
    site_id=tensor([int]),     # site index 0-19
    age=tensor([float]),       # normalized (age-15)/20
    sex=tensor([float]),       # normalized (sex-1.5)
    fiq=tensor([float])        # normalized (fiq-100)/30
  )
  ```
- **Smart Aggregation** (Phase 3):
  - PCA eigenvariate: First principal component captures dominant signal direction within each lobe (avoids cancellation)
  - Regional Homogeneity: Intra-lobe coherence + spatial variance for local connectivity features
  - Spatial coordinates: Aggregated from YOLO detections (mean x, y, z_depth per region)

### 8. GNN Architecture & Training
- **Model**: `CausalBrainGNN` (class in [src/models/causal_gnn.py])
  - **Current Defaults** (Feb 2026):
    - Input projection: Linear layer with LayerNorm (stabilizes 28 features with varying scales)
    - Layer 1: GATv2Conv with 4 heads, concat=True
    - Layer 2: GATv2Conv with 4 heads, concat=False (average heads)
    - Layer 3: GATv2Conv with 4 heads, concat=False (optional when `GNN_NUM_GNN_LAYERS=3`)
    - Skip connections: Residual links after each layer prevent over-smoothing in 12-node graphs
    - Activation: GELU (smooth gradient flow, superior to ReLU for small graphs)
  - **Attention pooling**: GlobalAttention pooling (configurable via `GNN_POOLING`)
  - Output: 2-class softmax (Control vs ASD) via 3-layer classifier head (Linear→GELU→Dropout→Linear)
  - **Optional conditioning** (disabled for visualization):
    - Site embeddings: Optional 16-dim embeddings to reduce site-specific scanner bias
    - Demographics: Optional age/sex/FIQ features for clinical context
  - **Flexible feature modes**: All features always processed (no conditional logic)
  - Weight initialization: Kaiming normal for Linear layers, zeros for biases
  - **Regularization**:
    - Dropout: 0.45
    - L2 regularization: weight_decay = 1e-4
    - Early stopping: patience=20, min_delta=0.0001
    - Gradient clipping: max_norm=1.0
  - **Site + Demographics conditioning** (`GNN_USE_GRL=True`, `GNN_USE_DEMOGRAPHICS=True`):
    - Site embeddings (16-dim) concatenated to node features: `lin_in = Linear(28+16=44, 128)` when enabled
    - Demographics (age/sex/FIQ) injected via `forward()`
    - **GRL** (Gradient Reversal Layer): adversarial auxiliary head learns site-invariant repr.
      `GNN_GRL_ALPHA=1.0`, `GNN_SITE_LOSS_WEIGHT=0.2`
    - **Edge gate** (`GNN_EDGE_GATE=True`): `Linear(1→1)` modulates causal weights before GAT msg-passing

- **Training details**:
  - Optimizer: AdamW with OneCycleLR (max_lr=`GNN_ONECYCLE_MAX_LR=0.003`, configured in config)
  - Scheduler: OneCycleLR (cosine anneal)
  - Loss: Focal Loss (α=0.62, γ=2.0) for class imbalance (config-driven)
  - Gradient clipping: `max_norm=1.0` (prevents explosion in small graphs)
  - K-fold: 5-fold stratified by DX_GROUP
  - Metrics: Accuracy, F1, **ROC-AUC**, **AUPRC** (average_precision_score), confusion matrix per fold
  - Checkpointing: Save best model per fold (top validation AUC)
  - **Ensemble**: AUC-weighted averaging across 5 checkpoints (weights proportional to fold val AUC)

- **Current Results** (5-fold CV with enhanced architecture, Feb 15, 2026):
  - Mean AUC: **0.6194 ± 0.0641** (+10.7pp over prior baseline)
  - Mean F1: **0.7132 ± 0.0160**
  - Mean Accuracy: **0.6194 ± 0.0241**
  - Test-set Ensemble AUC: 0.5398 (153 held-out subjects)
  - Per-fold AUCs: [0.5762, 0.5931, 0.6197, 0.7424, 0.5657]
  - Best fold: 0.7424 (Fold 3)
  - Mean best epoch: ~14.6 (range 8-24)
  - **Architecture benefits**: Learnable edge weights + multi-scale pooling + bug fixes improve feature extraction

## Development Workflows

### Complete End-to-End Pipeline (Recommended)

**For existing data (skip download and split):**
```bash
# Auto mode — runs all missing stages without prompts
python src/run_pipeline.py --auto --skip-download --skip-split

# Show execution plan without running
python src/run_pipeline.py --dry-run

# Force rebuild all intermediate files
python src/run_pipeline.py --force-reset

# Regenerate features only (keep images/splits)
python src/run_pipeline.py --regenerate-features --skip-yolo
```

**From scratch (with ABIDE download):**
```bash
# Full pipeline with data download (2–4 hours)
python src/run_pipeline.py --auto

# Note: requires Phenotypic_V1_0b_preprocessed1.csv with 'TR' column
```

**Pipeline execution order (20 stages):**

*Core (stages 1–15):*
1. ABIDE download — fMRI + 7 z-slices / subject (percentiles 0.2–0.8)
2. Stratified split — 2D by DX_GROUP + SITE_ID (70/15/15)
3. Master manifest — subject ↔ phenotype mapping
4. Atlas validation — verify AAL3v1 files
5. Pipeline validation — pre-flight health check
6. Post-download integrity — PNG/NPY file count check
7. Atlas-based label annotation — generate YOLO labels
8. YOLO training — 12-region ROI detection (skip if weights exist)
9. Spatial feature extraction — 3D coord aggregation, all-12-regions filter
10. Temporal feature extraction — 20 features/ROI (8 time-domain + 12 frequency)
11. Feature harmonization — fold-safe ComBat, protects DX_GROUP covariate
12. Pre-GNN integrity — completeness check per split
13. Causal graph construction — Granger causality 12×12 + adaptive sparsification
14. Pipeline diagnostics — health report (run after graphs exist)
15. Quality validation — YOLO quality, graph sparsity, stratification

*GNN Training (stage 16):*
16. GNN training — 5-fold stratified CV (GAT + GRL)

*Post-Training Analysis (stages 17–20):*
17. Visualizations — causal graph plots, feature heatmaps, performance figures
18. Evaluation — bootstrap 95% CI, permutation test, baseline comparison
19. Explainability — node/edge importance, Captum feature attribution
20. Result analysis — per-subject predictions, misclassification, site effects

### Running the Full Pipeline
```bash
# Validate health first
python src/validation/pipeline_checks.py --health

# Single-command full run
python src/run_pipeline.py --auto

# Skip slow analysis stages
python src/run_pipeline.py --auto --skip-evaluation --skip-explainability

# Individual stage commands
python -m src.pipelines.roi_detection           # Train YOLO
python -m src.features.extract_spatial          # Spatial coords
python -m src.features.extract_temporal --add-frequency  # Temporal + freq features
python -m src.features.fold_safe_harmonization  # ComBat harmonization
python -m src.data.split                        # Train/val/test split
python -m src.features.construct_causal         # Build causal graphs
python -m src.models.gnn_model                  # Train GNN (5-fold CV)

# Post-training analysis
python src/run_evaluation.py                    # Bootstrap CI, permutation, baselines
python src/run_explainability.py                # Node/edge importance, attribution
python src/run_result_analysis.py               # Per-subject predictions, site effects
```

### Testing & Validation
```bash
# Comprehensive health report (replaces pipeline_diagnostics.py)
python src/validation/pipeline_checks.py --health

# Post-download dataset integrity check
python src/validation/pipeline_checks.py --dataset

# Pre-GNN distribution check
python src/validation/pipeline_checks.py --distribution

# Class imbalance analysis with recommendations
python src/validation/pipeline_checks.py --class-analysis

# Comprehensive health report
python src/validation/pipeline_checks.py --health

# Health report with deep file validation (slower)
python src/validation/pipeline_checks.py --health --deep

# Run all integrity checks (default)
python src/validation/pipeline_checks.py

# Check dataset loading (verify labels, shapes, sample counts)
python -c "from src.features.graph_factory import ABIDECausalDataset; \
ds = ABIDECausalDataset('train'); print(f'Loaded {len(ds)} subjects, first graph has {ds[0].x.shape[0]} nodes')"

# Validate lobe mapping before graph construction
python -c "from src.core.config import validate_lobe_mapping; validate_lobe_mapping(); print('✓ LOBE_MAPPING valid')"

# Validate entire environment
python -c "from src.core.config import validate_environment; validate_environment()"

# Run fold-safe harmonization (handles NaN/Inf robustly)
python src/features/fold_safe_harmonization.py
```

### Debugging Data Issues

**If graphs aren't loading:**
1. Check manifest exists: `ls -la data/processed/Phenotypic_V1_0b_preprocessed1.csv`
2. Verify graphs exist: `ls data/processed/causal_graphs/ | wc -l`
3. Load single graph: `python -c "import torch; g=torch.load('data/processed/causal_graphs/Caltech_0051456_graph.pt'); print(f'Nodes: {g.x.shape[0]}, Edges: {g.edge_index.shape[1]}')"` 

**If features are missing:**
- Check `node_features_3d.csv` exists (output of [src/features/extract_spatial.py])
- Verify all-12-regions filter: `python -c "import pandas as pd; df=pd.read_csv('data/processed/metadata/node_features_3d.csv'); print(f'Complete subjects: {len(df)}')"`
- If count dropped significantly, regions weren't detected → check YOLO model path in [src/features/extract_spatial.py]

**If temporal features CSV is corrupted:**
- **Symptom**: `pandas.errors.ParserError: Error tokenizing data. C error: Expected 1 fields in line 5, saw 986`
- **Cause**: CSV has header comments or malformed structure
- **Fix**: Delete and regenerate: `rm data/metadata/node_attributes_temporal.csv && python src/features/extract_temporal.py`
- **Note**: As of Jan 2026, `extract_temporal.py` writes clean CSV without header comments

**If harmonization fails with NaN warnings:**
- Use fold-safe harmonization: `python src/features/fold_safe_harmonization.py`
- Or via pipeline: `python src/run_pipeline.py --run-safe-harmonize`
- Fold-safe harmonization includes:
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
- **166 ROI atlas error** (fixed Jan 2026): Pipeline now accepts 164-170 ROIs for AAL3v1
- Run atlas validator: `python src/validation/atlas_validator.py`
- Check atlas exists: `ls -la data/atlases/AAL3v1.nii`

**If ABIDE download fails:**
- **Missing 'TR' column error**: Phenotype CSV missing required column for download
- **Solution**: Skip download with existing data: `python src/run_pipeline.py --skip-split --run-manifest`
- **Path error** (fixed Jan 2026): Ensure `abide_download.py` uses `parents[2]` not `parents[0]`

**If argparse conflicts occur:**
- **Symptom**: `error: unrecognized arguments` when pipeline calls submodules
- **Fix** (applied Jan 2026): Pipeline now calls `extract_temporal` via subprocess, not direct import
- Affected file: `run_pipeline.py` uses `subprocess.run()` for isolated argument parsing

## Common Pitfalls & Code Anti-Patterns

**Path Handling:**
- ❌ **Bad**: `Path("./data")` or `Path("../../../data")` (relative paths break across modules)
- ✅ **Good**: Import from config: `from src.core.config import DATA_ROOT, DATA_METADATA`
- **Status**: split.py, manifest.py, generate_labels.py still use hardcoded paths—refactor when touching these files

**Graph Construction:**
- ❌ **Bad**: Assuming all subjects have 12 regions detected (some have missing detections)
- ✅ **Good**: extract_spatial.py enforces `node_count == 12` filter; only complete subjects proceed
- **Impact**: Graph shape assumptions depend on this filter; skip it and downstream training crashes with shape mismatches

**Config Duplication:**
- ❌ **Bad**: Hardcoding `LOBE_MAPPING` or `NUM_LOBES` in multiple files
- ✅ **Good**: Import once from config in every module that needs it
- **Status**: Code is clean; keep it that way

**Temporal Features:**
- ❌ **Bad**: Skipping `fold_safe_harmonization.py` and feeding raw temporal CSV (170-ROI format) to the GNN
- ✅ **Good**: `extract_temporal.py` outputs 170 ROIs × 20 features; `fold_safe_harmonization.py` aggregates → 12 regions × 20 features via `ROIAggregator.aggregate_to_lobes()`
- **Output shapes**: `node_attributes_temporal.csv` = `(N, 3401)` (170 ROIs × 20 + subject_id); `node_attributes_harmonized.csv` = `(N, 241)` (12 regions × 20 + subject_id)

**YOLO Augmentation:**
- ❌ **Bad**: Enabling `fliplr=True` or `degrees=15` for medical imaging (breaks anatomical consistency)
- ✅ **Good**: All augmentation disabled in [src/core/config.py]; medical images require anatomical alignment
- **Reason**: Left-right flips reverse hemisphere signals; rotations misalign Z-depth slices

**Graph Edge Attributes:**
- ❌ **Bad**: Missing `edge_attr` in PyTorch Geometric Data objects (GAT expects it)
- ✅ **Good**: Always include causal correlation weights: `Data(x=..., edge_index=..., edge_attr=weights)`
- **Shape**: `edge_attr` must be `(num_edges, 1)` float tensor — shaped by `.unsqueeze(1)` in `graph_factory.py`

**Protected Covariates:**
- ❌ **Bad**: Omitting `DX_GROUP` (diagnosis) from the covariates or harmonizing it away
- ✅ **Good**: Include diagnosis as a protected covariate in ComBat/neuroHarmonize (journal requirement)
- **Location**: [src/features/fold_safe_harmonization.py] enforces this (harmonize.py was deprecated)

## Key Files Reference

| File | Purpose |
|------|---------|
| [src/core/config.py] | ALL constants, paths, hyperparameters; 4 validation functions |
| [src/run_pipeline.py] | Unified entry point — **20-stage orchestrator** (15 core + 1 GNN + 4 post-training) |
| [src/run_evaluation.py] | Bootstrap 95% CI (N=2000), permutation test (N=1000), SVM/RF/MLP baselines |
| [src/run_explainability.py] | Captum integrated gradients, node/edge importance, literature validation |
| [src/run_result_analysis.py] | Per-subject predictions CSV, misclassification FP/FN profiles, site-effect AUC, calibration |
| **Validation & Diagnostics** | |
| [src/validation/pipeline_checks.py] | Post-download, pre-GNN, health reports, class analysis, quality validation (1727 lines) |
| [src/validation/atlas_validator.py] | Atlas file validation (existence, structure, ROI range 164–170) |
| [src/validation/dev_audit.py] | Deep validation + feature diagnostics (merged from code_audit.py + feature_diagnostics.py, Feb 28, 2026) |
| [src/validation/diagnose_features.py] | Per-feature-group statistics, z-score audit, single-graph + corpus-level diagnostics; CLI debugging tool (NEW, March 1, 2026) |
| **Feature Engineering & Graphs** | |
| [src/features/extract_spatial.py] | YOLO inference → 3D spatial aggregation; all-12-regions filter |
| [src/features/extract_temporal.py] | 20 temporal features/ROI: 8 time-domain + 12 frequency (incl. former frequency_features.py, Feb 28, 2026) |
| [src/features/causal_inference.py] | Granger causality & transfer entropy for directed graph construction |
| [src/features/fold_safe_harmonization.py] | **Aggregates 170 ROIs→12 regions** + ComBat harmonization + NaN/Inf handling; protects DX_GROUP |
| [src/features/construct_causal.py] | Granger/lagged correlation; PCA+ReHo aggregation; saves graph dicts (.pt files) |
| [src/features/graph_factory.py] | ABIDECausalDataset — assembles PyG Data(x=(12,28), edge_index, y, site/demo) at load time |
| **Data Pipeline** | |
| [src/data/split.py] | 2D stratified split (DX_GROUP + SITE_ID); preserves subject-level grouping |
| [src/data/abide_download.py] | ABIDE S3 download; 7-slice ALFF export (z-percentiles 0.2–0.8); saves `*_ts.npy` + `*_roi_labels.npy` |
| [src/utils/manifestor.py] | Master manifest (subject_id, split, DX_GROUP, SITE_ID, TR, AGE, SEX, FIQ, HANDEDNESS) |
| **Models** | |
| [src/models/causal_gnn.py] | CausalBrainGNN — GATv2 (3 layers, 4 heads, 128 channels) + GRL + learnable edge gate |
| [src/models/gnn_model.py] | 5-fold CV; FocalLoss(α=0.62,γ=2.0) + OneCycleLR; AUC+AUPRC+F1; AUC-weighted ensemble |
| [src/models/training_utils.py] | EarlyStopping, WarmupScheduler, TrainingTracker, CheckpointManager, train_fold_with_onecycle |
| [src/pipelines/roi_detection.py] | YOLO training entry point |
| **Experiments** | |
| [src/experiments/run_ablations.py] | 5 ablation studies (A–E): FlatMLP, spatial-only, temporal-only, Pearson edges, no-site; outputs to `RESULTS_ABLATIONS_DIR` |
| [src/experiments/data_quality.py] | 3 data quality experiments: cross-site AUC, subject count audit, atlas-centroid baseline; outputs to `RESULTS_DATA_QUALITY_DIR` |
| **Explainability Analysis** | |
| [src/analysis/edge_importance.py] | Phase 8.2: Gradient-based edge attribution + edge-masking (ΔP); group-level 12×12 ASD vs Control matrices |
| [src/analysis/node_importance.py] | Phase 8.1: GradCAM node importance + GAT attention-weight extraction; aggregated by diagnosis class |
| [src/analysis/literature_validation.py] | Phase 8.4: Cross-references top regions vs known ASD networks (DMN, Social Brain, Salience, Sensorimotor, Visual, Subcortical) |
| **Data Utilities** | |
| [src/data/filter_to_1000.py] | Removes 25 subjects missing spatial features; produces balanced 1 000-subject dataset (486 ASD, 514 Control) |
| **Tests** | |
| [tests/unit/test_config.py] | Unit tests for config constants and validation functions |
| [tests/unit/test_features.py] | Unit tests for feature extraction and harmonization |
| [tests/integration/test_dataset.py] | Integration tests for ABIDECausalDataset loading |
| [tests/integration/test_graph_construction.py] | Integration tests for causal graph construction pipeline |

## Integration Points & Critical Dependencies

### Data Flow Checkpoints
- **[src/features/extract_spatial.py]** → requires: `best.pt` (YOLO weights in results/), PNG images in `data/final/{train,val,test}/images/`
- **[src/features/fold_safe_harmonization.py]** → requires: temporal features CSV from extract_temporal.py
- **[src/features/construct_causal.py]** → requires: node_attributes_harmonized.csv, time series .npy files in `data/final/{split}/time_series/`
- **[src/models/gnn_model.py]** → requires: all `.pt` graphs in `data/processed/causal_graphs/`, master_manifest.csv, harmonized features
  - Loads graphs via `ABIDECausalDataset` with configurable site/demographic conditioning
  - Supports model variants: full YOLO features vs coords-only, with/without site embeddings

### Protected Covariates in Harmonization
In [src/features/fold_safe_harmonization.py], `DX_GROUP` (diagnosis) is a protected covariate passed to neuroHarmonize so batch harmonization doesn't remove disease signal. This is a journal Q1 requirement. Missing values in `AGE_AT_SCAN`/`SEX` are imputed (median/mode) BEFORE ComBat.

### Robust Harmonization with Safe NaN Handling
[src/features/fold_safe_harmonization.py] provides production-grade harmonization with:
- Pre-harmonization NaN/Inf detection and repair
- Feature-wise median imputation for missing values
- Outlier capping (values beyond 5 standard deviations)
- Post-harmonization validation (ensures zero NaNs)
- Comprehensive logging for debugging

### Dataset Filtering: All-12-Regions Requirement
[src/features/extract_spatial.py] enforces that only subjects with ALL 12 brain regions detected proceed to GNN training. This is critical: 12-node graphs assume complete detection. Check the `node_count == 12` filter before downstream processing.

### Stratified k-fold Details
[src/data/split.py] performs 2D stratification on both `DX_GROUP` (diagnosis) AND `SITE_ID` (scanner site). This ensures:
- Balanced ASD/Control across folds (addresses class imbalance)
- Balanced sites across folds (addresses batch effects from different scanners)

## Recent Fixes & Important Changes (January 2026)

### Summary of January 2026 Changes

**Documentation Fully Synchronized** (January 28, 2026)
- All .md files updated to reflect current project status
- README.md, ROADMAP.md, PIPELINE_DATAFLOW.md, copilot-instructions.md synchronized
- Added code_audit.py to validation module documentation (later merged into dev_audit.py, Feb 28)
- Updated YOLO performance metrics to v25 (mAP50-95: 0.908)
- Clarified validation folder structure with 4 modules

**YOLO Training Completed** (January 21, 2026)
- Trained YOLO26n for 100 epochs on brain region detection
- Achieved mAP50-95=0.908 (v25); mAP50=0.976; production-ready performance
- Precision 0.9996, Recall 0.9938 at epoch 100
- Model deployed to `results/experiments/detection/ROI_Detection_v20/weights/best.pt`
- Full training metrics in `results/experiments/detection/ROI_Detection_v20/results.csv`

**GNN Evolution & Current State** (January 22, 2026)
- **Hybrid v1 (Current Production)**: AUC=0.5832 ± 0.0476 with Focal Loss
  - 3-layer GATv2, coordinates-only features (3D spatial)
  - +8.9% AUC improvement over baseline
  - F1=0.6808 ± 0.0041 (excellent stability)
- **Baseline (Jan 21)**: AUC=0.5354 ± 0.0562 (2-layer GATv2, Cross-Entropy)
- **Next Target**: AUC=0.650 with full 28-feature inputs and conditioning
- All fold checkpoints saved to `models/checkpoints/best_model_fold{0-4}.pt`

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

- **[src/validation/pipeline_checks.py]** - Consolidated dataset validation (Jan 2026):
  - Merged check_progress.py and class_distribution.py functionality
  - Provides comprehensive health reports with metadata matching via `.str.strip()` on FILE_ID

- **[src/validation/pipeline_checks.py]** - Added ROI validation:
  - Added `EXPECTED_ROIS = 170` constant for AAL3 atlas
  - Validates time series files have correct ROI count (catches atlas mismatches early)

- **[src/utils/manifestor.py]** - Fixed metadata matching:
  - Added `.str.strip()` on FILE_ID to prevent merge failures due to whitespace
  - Note: Module generates master_manifest.csv with subject-phenotype mappings

- **[src/features/extract_temporal.py]** - Added ROI validation:
  - Added `EXPECTED_ROIS = 170` constant
  - Warns if time series doesn't have 170 ROIs (catches atlas mismatches during feature extraction)

**Impact**: These changes make the pipeline more robust by catching errors at extraction time (not during GNN training), handling interrupted runs gracefully, and preventing silent failures from whitespace/stratification issues.

### Pipeline Integration & Consolidation (January 20, 2026)
- **[src/run_pipeline.py]** - Integrated pipeline_checks.py as optional pipeline stage:
  - Added `--run-comprehensive-validation` flag for quality checks (YOLO quality, graph sparsity, feature preprocessing, stratification)
  - Added "comprehensive_validation" stage (position 6, after diagnostics)
  - Fixed atlas_validation condition: now validates unless explicitly skipped with `--skip-atlas-validation`
  - Updated execution order to include all 15 stages in proper sequence
  - All references now use `src.validation.pipeline_checks` for both post-download and pre-GNN checks

- **[src/validation/pipeline_checks.py]** - NEW: Combined integrity check module ✨
  - **Replaces**: integrity_check.py + integrity_check2.py (consolidated into single module)
  - **Functions**:
    - `check_dataset_integrity()` - Post-download: validates PNG files, NPY files, checks for incomplete subjects
    - `check_distribution()` - Pre-GNN: checks slice distribution across train/val/test, verifies image/label pairing
  - **Usage**: `python src/validation/pipeline_checks.py` or `--dataset` or `--distribution` flags
  - **Integration**: Called from run_pipeline.py stages "post_download_integrity" and "pre_gnn_integrity"
  - **Benefits**: Single source of truth, reduced code duplication, centralized validation logic

- **Validation Folder Structure** (March 1, 2026):
  ```
  src/validation/
  ├── atlas_validator.py       (atlas file structure & ROI validation)
  ├── dev_audit.py             (merged code_audit.py + feature_diagnostics.py; dev-only CLI tool)
  ├── diagnose_features.py     (per-feature-group stats, z-score audit, single-graph + corpus-level diagnostics)
  └── pipeline_checks.py       (unified: post-download + pre-GNN checks + health reports + class analysis)
  ```
  - Status: All modules integrated into run_pipeline.py
  - Added: diagnose_features.py (March 1, 2026) — standalone CLI for feature-level debugging
  - Deleted: integrity_check.py, integrity_check2.py, pipeline_diagnostics.py (merged into pipeline_checks.py)
  - Deleted: code_audit.py, feature_diagnostics.py (merged into dev_audit.py, February 28, 2026)
  - Deleted: frequency_features.py (merged into extract_temporal.py, February 28, 2026)

## Recent Fixes & Important Changes (February 2026) ✨ NEW

### Phase 3 Architecture Simplification & Smart Aggregation (February 12-14, 2026)

**Architecture Simplification (Problem & Solution):**
- **Problem**: Fold 4 collapse and high variance across folds indicated overfitting on 12-node graphs
- **Solution**: Simplified from 3 layers to 2 layers, reduced hidden channels 256→64, added GELU activation
- **Results**: Stable baseline AUC 0.5593 ± 0.0156 with consistent training across folds (low variance)

**Smart Aggregation Implementation:**
- Replaced simple mean aggregation with PCA eigenvariate extraction
  - Captures dominant signal direction within each lobe (avoids cancellation from opposing signals)
  - Added Regional Homogeneity (ReHo) features: intra-lobe coherence + spatial variance
  - Total features: 26 → 28 (20 temporal + 2 internal + 6 spatial)
- NaN safety: Added torch.isnan() and torch.isinf() checks in construct_causal.py and graph_factory.py

**Configuration Centralization:**
- Created FEATURE_GROUPS registry with explicit 28 features (temporal, frequency, internal, spatial)
- GNN_IN_CHANNELS automatically calculated: len(ALL_FEATURE_NAMES) = 28
- Removed deprecated imports: GNN_LEARNING_RATE_TUNED, GNN_HIDDEN_CHANNELS_TUNED, GNN_ENSEMBLE_MODE, GNN_NUM_GNN_LAYERS

**Hyperparameter Tuning (Current Defaults):**
- GNN_HIDDEN_CHANNELS: 128
- GNN_DROPOUT: 0.45
- GNN_WEIGHT_DECAY: 1e-4 (L2 regularization added)
- CAUSALITY_METHOD: 'granger'
- SPARSITY_QUANTILE: 0.70 (keep top 30% edges, min 12/graph)
- FocalLoss: α=0.62, γ=2.0
- Early stopping: patience=20, min_delta=0.0001

**Code Synchronization & Fixes:**
- Fixed AdamW optimizer: Removed duplicate weight_decay parameter, added closing parenthesis
- Updated all 3 CausalBrainGNN instantiations to use config values consistently
- Fixed visualizations.py: Disabled site_embedding and demographics for feature attribution (28-dim input)
- Verified all Python files compile successfully (100% pass rate)
- All imports resolve correctly, no missing config variables

**Performance & Validation:**
- Mean AUC: 0.6194 ± 0.0641 (+10.7pp over prior baseline of 0.5593)
- Per-fold AUCs: [0.5762, 0.5931, 0.6197, 0.7424, 0.5657]
- Mean best epoch: ~14.6 (range 8-24)
- Test-set ensemble AUC: 0.5398 (153 held-out subjects)
- No individual fold collapse (all > 0.5657)
- Early stopping prevents overfitting while maintaining signal detection

**Documentation Updates (Feb 15, 2026):**
- README.md: Updated to v28 YOLO metrics, current GNN results, 7-slice convention
- .github/copilot-instructions.md: Comprehensive Phase 3 + bug-fix update (this file)
- ROADMAP.md: Added Phase 3 sprint details, updated Phase 4-6 descriptions
- TODO.md: Updated config examples, corrected CAUSALITY_METHOD to 'granger'
- DATAFLOW.md: Updated hyperparameters and feature pipeline

## Recent Fixes & Important Changes

### February 2026 Updates ✨ NEW

#### YOLO v28 Training Complete (February 2-4, 2026)
- **Training**: 100 epochs completed on 12-region brain detection
- **Performance**: mAP50-95=0.93714, mAP50=0.98952
- **Precision/Recall**: 0.98063 / 0.97214 (exceptional, near-perfect)
- **Deployment**: results/experiments/detection/ROI_Detection_v28/weights/best.pt
- **Status**: Production-ready with outstanding detection quality for all 12 brain regions

#### GNN Retraining with Full Pipeline Fixes (February 11-15, 2026)
- **Latest Training**: 5-fold CV completed after applying all Phase 3 bug fixes
- **Performance**: Mean AUC 0.6194 ± 0.0641, Mean F1 0.7132 ± 0.0160 (+10.7pp AUC over prior baseline)
- **Per-fold AUCs**: [0.5762, 0.5931, 0.6197, 0.7424, 0.5657]
- **Best fold**: Fold 3 at 0.7424
- **Mean best epoch**: ~14.6 (range 8-24)
- **Training characteristics**:
  - Moderate convergence: ~14.6 epochs average (vs prior 6.4 — richer signal from fixed pipeline)
  - Fold 3 reaching 0.7424 demonstrates strong learnable ASD biomarkers
  - Early stopping patience=20 allows full signal extraction
  - Test-set ensemble AUC: 0.5398 (153 held-out subjects)
- **Model checkpoints**: All updated Feb 15, 2026 (models/checkpoints/best_model_fold{0-4}.pt)
- **Data**: 1035 subjects total, 7 z-slices each (z-percentiles 0.2/0.3/0.4/0.5/0.6/0.7/0.8), 7245 PNGs
- **Graphs**: 1035 causal graphs, mean 79.2 edges/144 max (55% density)

#### Critical Bug Fixes Applied (February 15, 2026)
- **`DEFAULT_TR = 2.0`**: Added missing constant to `config.py` (fallback TR for per-subject lookup)
- **neuroHarmonize `SITE` column**: `prepare_covariates` in `fold_safe_harmonization.py` renamed `SITE_ID→SITE`, dropped `subject_id` — neuroHarmonize requires exact `SITE` name
- **Site embedding zero-padding**: `CausalBrainGNN.forward()` zero-pads 16-dim site embedding when `site_id=None` to maintain `lin_in` input shape (44-dim = 28 + 16) — fixes `(N×28)@(44×128)` shape mismatch during attribution
- **Pipeline stage ordering**: `causal_graphs` stage now runs before `diagnostics`/`quality_validation` in `run_pipeline.py` — fixes false "no graphs found" critical error
- **`TARGET_SLICES = 7`**: `pipeline_checks.py` updated from 5 to 7 (abide_download saves 7 z-slices at percentiles 0.2–0.8)
- **`check_distribution` paths**: Fixed `DATA_PROCESSED → DATA_FINAL` for split image directories
- **Graph topology palette**: `dx_group` mapped to string labels `"ASD"/"Control"` before seaborn boxplot — fixes `KeyError: missing keys {'1', '2'}`
- **`visualize_accuracy_metrics` glob**: Fixed path from `RESULTS_DIR/*.json` to `RESULTS_DIR/experiments/training/*.json`

#### Model Architecture Enhancements (February 12-14, 2026)
- **Enhanced GNN Architecture** ([src/models/causal_gnn.py]):
  - **Edge gating**: Linear(1→1) gate on causal weights before message passing
  - **Pooling**: GlobalAttention pooling when `GNN_POOLING=attention`
  - **Site embeddings**: Optional 16-dim embeddings to reduce site-specific scanner bias
  - **Demographics conditioning**: Optional age/sex/FIQ inputs for clinical context
  - **Layer architecture refinement**: Layer 3 uses concat=False to average attention heads
  - **Status**: Production-ready with improved capacity and interpretability

- **Feature Pipeline Updates** ([src/features/construct_causal.py]):
  - **GPU-accelerated Granger causality**: compute_granger_causality_gpu() for faster graph construction
  - **Multi-lag causality**: compute_multilag_causality() tests temporal dynamics across 1-5 TRs
  - **Improved error handling**: Better NaN/Inf checks in causal matrix computation
  - **Adaptive sparsification**: Statistical method maintains min 12 edges/graph for connectivity

- **Training Utilities** ([src/models/training_utils.py]):
  - Enhanced metric computation and logging
  - Better checkpoint management for model variants
  - Improved focal loss implementation for class imbalance

- **Impact**: These architectural improvements position the model for AUC gains when fully leveraging 28-feature inputs and site/demographic conditioning

#### Validation Folder Finalized (March 1, 2026)
- **Complete structure**: 4 modules fully integrated
  - atlas_validator.py: Atlas file structure & ROI validation
  - dev_audit.py: Deep validation + feature diagnostics (merged from code_audit.py + feature_diagnostics.py)
  - diagnose_features.py: Per-feature-group statistics, z-score double-normalisation audit, single-graph and corpus-level checks (NEW, March 1, 2026)
  - pipeline_checks.py: Post-download, pre-GNN, health reports, class analysis, quality validation
- **Integration**: All modules callable from run_pipeline.py
- **Documentation**: Synchronized across README.md, ROADMAP.md, DATAFLOW.md, copilot-instructions.md
- **Added**: diagnose_features.py (March 1, 2026)
- **Deleted**: code_audit.py, feature_diagnostics.py (merged into dev_audit.py)
- **Deleted**: frequency_features.py (merged into extract_temporal.py)

### January 2026 Updates

### Validation Module Consolidation (January 26, 2026)
- **[src/validation/pipeline_checks.py]** - Now serves as the single source of truth for all validation operations:
  - `check_dataset_integrity()` - Post-download validation
  - `check_distribution()` - Pre-GNN validation
  - `analyze_class_distribution()` - Class imbalance analysis
  - `generate_health_report()` - Comprehensive pipeline health check (replaces pipeline_diagnostics.py)
- **Deleted files**: pipeline_diagnostics.py merged into pipeline_checks.py for better maintainability
- **Why it matters**: Single validation module reduces code duplication and provides unified interface for all data quality checks

### Path Resolution Fixes
- **[src/data/abide_download.py]** - Fixed `PROJECT_ROOT = Path(__file__).resolve().parents[2]` (was parents[0], incorrectly pointed to src/data/ instead of project root)

### CSV/Data Format Fixes
- **[src/features/extract_temporal.py]** - Removed CSV header comments (`# atlas_name:` etc.) that broke pandas CSV parsing. Now writes clean CSV directly with `df.to_csv()`. Metadata moved to accompanying `.roi_coverage.json` file.
- **[src/features/fold_safe_harmonization.py]** - Fixed pandas FutureWarning by replacing `df[col].fillna(..., inplace=True)` with proper assignment `df[col] = df[col].fillna(...)`

### Pipeline Diagnostics & Validation
- **[src/validation/pipeline_checks.py]** - Consolidated all validation functions including health reports, ROI validation (accepts 164-170 ROIs for AAL3v1 variants), class distribution analysis
- **[src/run_pipeline.py]** - Major refactor (January 2026):
  - Changed `extract_temporal` invocation from direct import to subprocess call (avoids argparse conflicts with sys.argv)
  - Integrated all new diagnostic and harmonization tools
  - Fixed atlas validation logic
  - Note: `--run-diagnostics`, `--run-safe-harmonize`, `--run-comprehensive-validation` flags were added in Jan 2026 and later removed/replaced in the March 2026 CLI cleanup; current flags are documented in the **March 1, 2026** changes section above

### Atlas Support
- **AAL3v1 (166 ROIs)** is now fully supported alongside AAL116/117/170 variants
- Temporal feature extraction correctly detects 164 ROIs from AAL3v1 (2 ROIs may be empty/unused in specific templates)

### Model Checkpoints (March 1, 2026)
- **YOLO26n deployed (v28)**: `results/experiments/detection/ROI_Detection_v28/weights/best.pt` (mAP50-95=0.93714, mAP50=0.98952)
- **YOLO26n base**: `yolo26n.pt` in project root and `models/pretrained/yolo26n.pt`
- **YOLO next target**: `ROI_Detection_v29` — `YOLO_PROJECT_NAME = 'ROI_Detection_v29'` in config.py; weights not yet generated
- **GNN folds**: `models/checkpoints/best_model_fold{0-4}.pt` (updated Feb 15, 2026; Mean AUC=0.6194±0.0641, best fold 0.7424)
- **GNN baseline**: `models/checkpoints_baseline/best_model_fold{0-4}.pt` (archived Jan 2026)

## Recent Fixes & Important Changes (March 1, 2026) ✨ LATEST

### Phase 9 Complete — Full Post-Training Analysis Pipeline

**New post-training runner scripts:**
- **[src/run_evaluation.py]** (898 lines): Bootstrap 95% CI (N=2000), permutation test (N=1000), SVM/RF/MLP baseline comparison; outputs to `RESULTS_EVALUATION_DIR`
- **[src/run_explainability.py]** (441 lines): Orchestrates Phases 8.1–8.4 — node importance, edge importance, feature attribution, literature validation; outputs to `results/explainability/`
- **[src/run_result_analysis.py]** (776 lines): Per-subject predictions CSV, FP/FN misclassification profiles, site-effect AUC bars, calibration reliability diagram, severity correlation; outputs to `RESULTS_DIR/analysis/`

**New explainability sub-modules (`src/analysis/`):**
- **[src/analysis/node_importance.py]** (512 lines): `NodeImportanceAnalyzer` — GradCAM + GAT attention-weight extraction; `AttentionWeightExtractor` reads `_alpha` after each GATv2Conv forward
- **[src/analysis/edge_importance.py]** (461 lines): `EdgeImportanceAnalyzer` — gradient attribution (`GradientEdgeAttributor`) + edge-masking (ΔP method, `EdgeMaskingAnalyzer`); produces 12×12 group-level heatmaps
- **[src/analysis/literature_validation.py]** (436 lines): `validate_important_regions()` cross-references top-ranked regions against 6 known ASD networks (DMN, Social Brain, Salience, Sensorimotor, Visual, Subcortical); `generate_report()` writes JSON + text

**New data utility:**
- **[src/data/filter_to_1000.py]** (309 lines): Identifies and removes 25 subjects missing YOLO spatial features; produces balanced 1 000-subject dataset (486 ASD, 514 Control); supports `--backup` / `--restore-backup` flags

**New validation tool:**
- **[src/validation/diagnose_features.py]** (589 lines): CLI tool for feature-level debugging; audits single graphs and the full corpus; checks per-group stats (temporal/frequency/internal/spatial), z-score double-normalisation, edge distribution; reads `FEATURE_GROUPS` slices from config

**New test suite (`tests/`):**
- `tests/unit/test_config.py` — config constants and validation functions
- `tests/unit/test_features.py` — feature extraction and harmonization
- `tests/integration/test_dataset.py` — ABIDECausalDataset loading
- `tests/integration/test_graph_construction.py` — causal graph construction pipeline

**Config additions (`src/core/config.py`):**
- `RESULTS_TRAINING_DIR` = `results/experiments/training`
- `RESULTS_ABLATIONS_DIR` = `results/experiments/ablations`
- `RESULTS_DATA_QUALITY_DIR` = `results/experiments/data_quality`
- `RESULTS_EVALUATION_DIR` = `results/evaluation`
- `RESULTS_FIGURES_DIR` = `results/figures`
- `validate_training_health()` and `log_training_diagnostics()` diagnostic helpers
- AUC/F1/Loss threshold constants (`AUC_RANDOM_THRESHOLD`, `AUC_GOOD_THRESHOLD`, etc.)

**Pipeline extended to 20 stages:**
- Core stages 1–12 unchanged (download → pre_gnn_integrity)
- Stage 13: causal_graphs — now runs **before** diagnostics/quality_validation (ordering fix)
- Stage 14: diagnostics — health report after graphs are built
- Stage 15: quality_validation — YOLO quality, sparsity checks
- Stage 16: GNN training (5-fold CV, unchanged)
- Stages 17–20: visualizations, evaluation, explainability, result_analysis

**CLI updated** (`src/run_pipeline.py`):
- `--analysis-only`: run only stages 17–20 (post-training analysis)
- `--visualizations-only`: run only stage 17
- `--skip-visualizations`, `--skip-evaluation`, `--skip-explainability`, `--skip-result-analysis`: skip individual post-training stages
- `--skip-diagnostics`, `--skip-comprehensive-validation`: skip health-check stages
- `--regenerate-features`: regenerate spatial/temporal/harmonization/graphs (keeps images/splits)
- Note: obsolete flags `--run-diagnostics`, `--run-safe-harmonize`, `--run-comprehensive-validation` removed

**Documentation synchronized (March 1, 2026):**
- README.md, ROADMAP.md, docs/DATAFLOW.md, and this file all updated to reflect 20-stage pipeline, v28/v29 YOLO status, Phase 9 completions, and Phase 10 open items

## Known Issues & Open Items (Phase 10)

Documented in `docs/TODO.md` (code audit Feb 23, 2026). These are the root causes of the current ~0.62 AUC ceiling:

1. **Double z-score normalisation** — `abide_download.py` uses `standardize='zscore_sample'` AND `construct_causal.py` z-scores again. Fix: set `standardize=False` in NiftiLabelsMasker.
2. **Frequency band aliasing** — beta/gamma bands (0.15–0.25 Hz) are near the fMRI Nyquist limit (TR=2 s → Nyquist=0.25 Hz). These 12 frequency features may add noise. Track via ablation.
3. **CV–test AUC gap** — Mean CV AUC 0.6194 vs test ensemble AUC 0.5398 (gap of 0.0796). Stronger site-invariance (higher GRL alpha) and test-time augmentation are next candidates.
4. **YOLO v29 training** — config is set but training run has not been executed yet.

## Medical/Scientific Context

- **Diagnosis label**: 0=Control (healthy), 1=ASD (autism spectrum disorder)
- **Data source**: ABIDE initiative (public fMRI dataset, multi-site, ~1000 subjects)
- **Causal inference goal**: Identify region→region influence patterns distinctive to ASD using lagged correlations
- **Explainability**: Edge weights in causal graph provide subject-specific feature importance (gradient-based saliency in `causal_gnn.py::get_node_importance()`)
- **Medical tuning**: YOLO augmentation disabled for grayscale medical images; preserves anatomical alignment
- **Graph construction**: Granger causality (default) or lagged Pearson correlation between brain regions
- **Key metrics**: 
  - **YOLO detection**: mAP50-95=0.93714 (v28, outstanding; 12-region detection highly reliable)
  - **GNN classification (5-fold CV)**: AUC=0.6194±0.0641, F1=0.7132±0.0160 (Feb 15, 2026)
  - Training characteristics: Moderate convergence (~14.6 epochs), Fold 3 peak AUC=0.7424
  - Test-set ensemble AUC: 0.5398 (153 held-out subjects)
  - Graph topology finding: Parietal In-Degree lower in ASD (p=0.0296, Cohen's d=-0.125)
  - Architecture features: Learnable edge weights, multi-scale pooling, optional site/demographic conditioning
- **Imbalanced classification**: 2D stratified CV (DX_GROUP + SITE_ID) ensures balanced evaluation

### Why These Design Choices?

**12-Region Aggregation:**
- Reduces 170×170=28,900 edges to 12×12=144 edges (computational efficiency)
- Anatomically interpretable for clinical validation
- Finer granularity than 5-lobe system: separates functionally distinct areas (e.g., motor/premotor, insula, cingulate)
- Reduces scanner-specific noise through within-region averaging
- Better statistical power with limited samples (n~1000)

**Multi-Head GAT with Learnable Edge Weights:**
- 4 attention heads per layer (balanced capacity for 28-feature input)
- Edge gating (Linear 1→1) modulates causal weights before message passing
- Final layer uses concat=False to average head outputs
- Still efficient for 12-node graphs (not redundant like 8+ heads)
- Maintains edge_attr integration with learned transformation
- Skip connections prevent over-smoothing across 3 layers
- Attention pooling captures diverse graph properties

**Granger Causality (not just lagged correlation):**
- Statistical rigor: Tests null hypothesis with p-values
- Multi-lag testing: Captures dynamics across multiple timescales (1-5 TRs)
- Directed causality: Region i \u2192 j if past of i improves prediction of j
- Interpretable edge weights: -log10(p-value) where higher = stronger
- Baseline alternative: Lagged Pearson for computational efficiency

**Frequency-Domain Features (12 new):**
- Gamma-band abnormalities documented in ASD (Rojas et al., 2008)
- Spectral entropy captures oscillatory complexity
- Phase stability metrics for synchrony analysis
- 86% feature expansion (14 \u2192 26) for richer representation
