# Neuro-CXG Project Guidelines
**Causal Graph Neural Networks for Brain Disorder Classification from fMRI**

**Last Generated:** March 9, 2026  
**For:** AI Coding Agents (GitHub Copilot, Claude, etc.)

---

## Executive Architecture

Neuro-CXG is a **20-stage end-to-end medical ML pipeline** for autism spectrum disorder (ASD) classification using fMRI-derived causal brain graphs. The system achieves **CV AUC 0.7434 ± 0.0417** (March 9, 2026) with production-grade engineering: explicit validation at every stage, fold-safe harmonization, anatomically-preserving constraints, and comprehensive explainability.

**Critical Design Principle:** *Configuration-driven everything* — `src/core/config.py` is the single source of truth. Never write paths, hyperparameters, or feature names directly in code.

---

## Code Style

**Language & Formatting**
- Python 3.x with type hints (prefer explicit over implicit)
- Google-style docstrings for functions/classes
- Standard logging format: `logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')`
- Imports order: stdlib → third-party → local (see [src/models/causal_gnn.py](src/models/causal_gnn.py))
- Never hardcode paths: always import from [src/core/config.py](src/core/config.py)

**Medical Domain Constraints**
- **Reproducibility:** seed=42 everywhere (torch, numpy, random) — required for auditability
- **Anatomical Preservation:** YOLO augmentation uses `fliplr=0.0`, `degrees=0.0` — no rotation or flips to preserve Left/Right hemispheres and 3D alignment
- **Fold-Safe Harmonization:** Always protect `DX_GROUP` (diagnosis) as covariate in ComBat to prevent diagnosis leakage into batch-correction
- **Nyquist Awareness:** fMRI with TR=2s has Nyquist limit of 0.25 Hz; gamma band (0.20-0.25 Hz) is unreliable; delta/theta/alpha are safe
- **Error Handling:** Graceful degradation (never crash), fallback to simpler methods (e.g., PCA fails → mean), log warnings, continue

## Architecture

### **20-Stage Pipeline Organization**

Neuro-CXG follows a **deterministic, validated pipeline** with explicit failure points and recovery mechanisms.

#### **Stage Groups**

| Group | Stages | Purpose | Output |
|-------|--------|---------|--------|
| **Phase 0: Data Acquisition** | 1-3 | Download ABIDE, stratified split, manifest | `data/final/{train,val,test}/`, `master_manifest.csv` |
| **Phase 1: Validation & Labeling** | 4-7 | Atlas check, YOLO label generation | YOLO format `.txt` annotations |
| **Phase 2: ROI Detection** | 8 | Train YOLO26n (12 brain regions) | `best.pt` weights (mAP50-95=0.9598 v29, deployed March 9, 2026) |
| **Phase 3: Feature Extraction** | 9-12 | Spatial + temporal extraction, harmonization | `node_attributes_harmonized.csv` (12 regions × 28 features) |
| **Phase 4: Graph Construction** | 13-15 | Causal inference, integrity, diagnostics | `data/processed/causal_graphs/{sub}_graph.pt` |
| **Phase 5: Model Training** | 16 | 5-fold CV with GATv2 | `models/checkpoints/best_model_fold*.pt` |
| **Phase 6: Post-Training Analysis** | 17-20 | Evaluation, explainability, result analysis | Reports, visualizations, interpretability |

#### **Module Responsibilities**

```
src/core/config.py
├── SINGLE SOURCE OF TRUTH for all paths, hyperparameters, constants
├── Feature definitions: temporal (20), internal (2), spatial (6) = 28 total
├── Neural network architecture (GNN_HIDDEN_CHANNELS, GNN_NUM_LAYERS, etc.)
├── Medical constraints (YOLO_FLIPLR=0.0, K_FOLDS=5, GRANGER_MAX_LAG=5, GRANGER_MAX_LAG_SECONDS=10.0)
└── Diagnostic thresholds (AUC_GOOD_THRESHOLD=0.70, F1_WEAK_THRESHOLD=0.30)

NOTE: `CAUSAL_LAG` is deprecated/commented out in config.py. Use GRANGER_MAX_LAG and GRANGER_MAX_LAG_SECONDS.

src/data/
├── abide_download.py → download ABIDE from S3, extract 7 z-slices per subject (ALFF percentiles [0.21, 0.3–0.8]; 0.21 captures brainstem ROIs 167–170)
├── split.py → 2D stratified split (DX_GROUP × SITE_ID, 70/15/15)
└── filter_to_1000.py → remove subjects missing entire lobe in spatial features

src/features/
├── extract_spatial.py → 6 features/lobe (x, y, z, size, conf_std, count) from YOLO; gate SPATIAL_MIN_REQUIRED_REGIONS=9
├── extract_spatial_atlas.py → atlas-based spatial coords (alternative to YOLO, precomputed AAL3v1 centroids from roi_centroids.json)
├── extract_temporal.py → 20 features/lobe (8 basic + 12 frequency domain); uses SITE_TR_MAP for per-subject TR
├── fold_safe_harmonization.py → ComBat batch correction (neuroHarmonize, DX_GROUP protected; 5-fold CV safe)
├── construct_causal.py → Granger causality or lagged correlation → 12×12 digraph; saves as dict not PyG Data
├── causal_inference.py → low-level Granger/transfer-entropy computation
├── graph_factory.py → ABIDECausalDataset (torch_geometric.Dataset subclass; assembles Data at load time)
└── [Internal pipeline: 170 AAL ROIs → 12 lobes via PCA eigenvariate + ReHo (coherence + spatial_variance)]

src/models/
├── causal_gnn.py → CausalBrainGNN (GATv2 layers, site embedding, skip connections)
├── gnn_model.py → 5-fold CV training orchestration + metrics + checkpointing
└── training_utils.py → EarlyStopping, CheckpointManager, TrainingTracker, Focal Loss

src/pipelines/
├── roi_detection.py → YOLO training script (100 epochs, batch 32, medical augmentation)
└── generate_labels.py → YOLO label generation from atlas

src/validation/
├── atlas_validator.py → verify AAL3 file exists and dimensions match
├── audit_check.py → post-fix validation (1,000-subject count, 12-lobe completeness, feature dims, NaN/Inf check)
├── pipeline_checks.py → comprehensive multi-level validation (YOLO quality, sparsity, etc.)
└── dev_audit.py → development diagnostics (--features flag for deep analysis)

src/analysis/
├── diagnostics.py → TrainingMonitor (metrics tracking), CausalGraphAnalyzer (topology)
├── node_importance.py → GradCAM node attribution, attention extraction
├── edge_importance.py → Granger weight analysis
├── feature_attribution.py → Captum-based feature importance (Integrated Gradients)
└── visualizations.py → publication-ready plots

src/
├── run_pipeline.py → Main orchestrator (21 named stages (1–21) + Stage 0 pre-flight = 22 execution points; CLI flags)
├── run_evaluation.py → Test-set evaluation (bootstrap CI, permutation test, baselines)
├── run_explainability.py → Generate interpretability reports
└── run_result_analysis.py → Per-subject predictions, misclassification analysis

src/analysis/
├── diagnostics.py → TrainingMonitor (metrics tracking), CausalGraphAnalyzer (topology)
├── node_importance.py → GradCAM node attribution, attention extraction
├── edge_importance.py → Granger weight analysis
├── feature_attribution.py → Captum-based feature importance (Integrated Gradients)
├── literature_validation.py → Cross-reference findings against published ASD/fMRI literature
└── visualizations.py → publication-ready plots
```

### **Data Flow: 170 AAL ROIs → 12-Region Graphs**

#### **Step 1: Temporal Aggregation**
```
Raw 170-ROI time series (T, 170)
    ↓
PCA Eigenvariate (first principal component per lobe)
    + Regional Homogeneity (coherence + spatial_variance)
    ↓
12 lobe-level time series (T, 12) + (12, 2) internal features
```
**Why PCA instead of mean?** Simple averaging cancels signals when ROIs are anti-correlated (common in motor/cingulate). PCA preserves magnitude and direction.

#### **Step 2: Graph Construction**
```
12 lobe time series (T, 12)
    ↓
Granger causality test (lag 1-5 TRs) OR lagged Pearson correlation
    ↓
12×12 causality matrix (directed edges, weights = -log10(p-value))
    ↓
Adaptive sparsification (keep top 30% edges, min 12 edges/graph)
    ↓
PyTorch Geometric Data object:
  - x: (12, 28) node features [temporal(20) + internal(2) + spatial(6)]
  - edge_index: (2, E) COO format
  - edge_attr: (E, 1) weights
  - y: 0/1 (Control/ASD)
  - batch: fold assignment
```

#### **Step 3: Feature Engineering (28 dimensions)**

| Category | Features | Computation | Clinical Relevance |
|----------|----------|-------------|-------------------|
| **Temporal (8)** | mean, std, skew, kurtosis, PSD, MSSD, range, autocorr | Descriptive statistics of ROI time series | Overall activity level, complexity |
| **Frequency (12)** | delta/theta/alpha/beta/gamma power (×2 each) + spectral_entropy + phase_std | Welch PSD in 5 frequency bands (0.01-0.25 Hz) | Oscillatory activity per band |
| **Internal (2)** | coherence, spatial_variance | Mean pairwise ROI correlation + variance spread | Intra-lobe synchrony (ReHo analog) |
| **Spatial (6)** | x, y, z_depth, size, conf_std, detection_count | YOLO detection coordinates + statistics | 3D anatomical localization |

**Current Status (March 9, 2026 — all P0/P1 bugs fixed):**
- ✅ Single z-score only: `standardize=False` in NiftiLabelsMasker; z-score in construct_causal.py
- ✅ BANDPASS_HIGH = 0.15 Hz (expanded from 0.08; retains beta band)
- ⚠️ Beta (0.15–0.20 Hz) and gamma (0.20–0.25 Hz) sit at/near Nyquist; gamma **zeroed at runtime** for TR=2s subjects via `UNRELIABLE_FREQ_BANDS_AT_NYQUIST=("gamma",)`. Trust delta/theta/alpha.

**Resolved Bugs (March 8–9, 2026 audit — all fixed):**
- ✅ P0: `DX_GROUP` **added** to ComBat covariates in `fold_safe_harmonization.py` L258 — harmonization now preserves diagnostic signal
- ✅ P0: Circular test-AUC ensemble weighting **fixed** in `run_evaluation.py` L274 — val-fold AUC from checkpoint used
- ✅ P0: `site_id=None` hardcoding **fixed** in `run_evaluation.py` L160 — `batch.site_id` used
- ✅ P0: Double z-score **fixed** in `abide_download.py` L176 — `standardize=False` in NiftiLabelsMasker; single z-score remains in construct_causal.py
- ✅ P0: Dead lobe NaN crash **fixed** in `construct_causal.py` — NaN pre-filter before PCA aggregation
- ✅ P1: Multi-lag Granger Bonferroni correction **applied** in `causal_inference.py` L93-95
- ✅ P1: Fold NaN leakage **fixed** in `fold_safe_harmonization.py` — train-fold stats only
- ✅ P1: PCA sign ambiguity **fixed** in `construct_causal.py` — correlation-with-raw-mean sign flip
- ✅ P1: `CAUSAL_LAG` and `GNN_ONECYCLE_PATIENCE` deprecated constants **removed** from config.py

**Remaining Open Issues (Phase 11):**
- ⚠️ Gamma band zeroed at runtime for TR=2s (Nyquist 0.25 Hz); tracked via `UNRELIABLE_FREQ_BANDS_AT_NYQUIST=("gamma",)` in config (Phase 10.2)
- ⚠️ CV–test AUC gap: 0.7434 (CV) vs 0.6487 (test) = 0.0947; GRL alpha grid search {0.05, 0.1} pending (Phase 11.2)
- ⚠️ Per-site AUC variability: site 9 AUC=0.9500 vs site 16 AUC=0.3281 (Phase 11.2)
- ⚠️ Disk checkpoints from Run 2 (pipeline_20260309_195751); fold3 checkpoint is from epoch=2 collapsed run

### **Key Architectural Decisions & Rationales**

1. **12 regions vs 170 ROIs**
   - ✓ Improves GNN learnability on ~700 training samples (small-sample regime)
   - ✓ Anatomically meaningful (matches clinical brain systems)
   - ✗ Loses within-lobe heterogeneity → potential signal loss
   - Mapping: PCA aggregation of 170 AAL ROIs per LOBE_MAPPING in config.py

2. **GATv2 (Graph Attention Network v2) not GCN**
   - ✓ Directed causal graphs require edge-dependent attention weights
   - ✓ GATv2Conv handles `edge_attr` natively (Granger weights guide message passing)
   - ✓ Multi-head attention improves expressiveness on small graphs
   - ✗ Slightly slower than GCN
   - See: [causal_gnn.py](src/models/causal_gnn.py) architecture

3. **Granger Causality (default) vs lagged Pearson**
   - ✓ Granger tests temporal precedence (X[t-lag] predicts Y[t]) — stronger causal claim
   - ✓ Multivariate F-test controls for confounding
   - ✗ Assumes linear relationships; requires stationarity
   - Alternative: `transfer_entropy` (nonlinear, info-theoretic)
   - Config: `CAUSALITY_METHOD = 'granger'` — see [causal_inference.py](src/features/causal_inference.py)

4. **Site Embedding + Gradient Reversal (Optional)**
   - ✓ Reduces site-specific scanner bias
   - ✗ Current `GNN_GRL_ALPHA=1.0` may be too strong (impairs signal) — reverted in Phase 9
   - Clinical detail: Different sites use different scanners (1.5T-3T, different sequences)
   - See: [causal_gnn.py](src/models/causal_gnn.py#L175) GradientReversal class

5. **Focal Loss (α=0.62, γ=2.0)**
   - ✓ Addresses class imbalance (ASD 486 vs Control 514 in ABIDE)
   - ✓ Focuses on hard examples during training
   - Config: `USE_FOCAL_LOSS = True` in config.py
   - See: [gnn_model.py](src/models/gnn_model.py#L63) FocalLoss implementation

## Build and Test

**Setup**
```bash
# Install
pip install -r requirements.txt

# Verify
python -c "from src.core.config import validate_environment; validate_environment()"
```

**Run Full Pipeline** (Recommended approach)
```bash
# Interactive mode (prompts for each stage)
python src/run_pipeline.py

# Automatic mode (run all missing stages)
python src/run_pipeline.py --auto

# Skip data download/split (use existing data)
python src/run_pipeline.py --auto --skip-download --skip-split

# Dry run (show execution plan without running)
python src/run_pipeline.py --dry-run

# Force reset all intermediate files and rebuild
python src/run_pipeline.py --force-reset

# Run only analysis stages (17-20)
python src/run_pipeline.py --analysis-only

# Run only post-training visualization
python src/run_pipeline.py --visualizations-only
```

**Individual Stages** (for development/debugging)
```bash
# Stage 8: YOLO training (ROI detection, 100 epochs)
python -m src.pipelines.roi_detection

# Stage 9: Spatial features extraction (6/lobe)
python -m src.features.extract_spatial

# Stage 11: Temporal features extraction (20/lobe: 8 time + 12 frequency)
python -m src.features.extract_temporal

# Stage 12: Fold-safe harmonization (ComBat batch correction)
python -m src.features.fold_safe_harmonization

# Stage 14: Causal graph construction (Granger → 12×12 digraphs)
python -m src.features.construct_causal

# Stage 16: GNN training (5-fold CV, ~30 min on GPU)
python -m src.models.gnn_model

# Post-training: Test evaluation with bootstrap CI
python src/run_evaluation.py

# Post-training: Explainability report generation
python src/run_explainability.py
```

**Testing**
```bash
pytest tests/unit/          # Unit tests (fast)
pytest tests/integration/   # Integration tests (slow, requires data)
pytest --cov=src tests/     # Coverage report
```

**Expected Runtimes**
| Stage | Runtime (GPU) | Notes |
|-------|---------------|-------|
| ABIDE Download | 30-60 min | S3 download, one-time |
| YOLO Training | 20-30 min | 100 epochs, batch 32 |
| Feature Extraction | 5 min | Temporal + spatial |
| Harmonization | 2 min | 5-fold CV ComBat |
| Graph Construction | 10 min | Granger causality |
| GNN Training | 30-40 min | 5-fold CV, OneCycle scheduler |
| **Total Pipeline** | **~2 hours** | Sequential execution |

## Project Conventions

### **Feature Engineering** (28 features/lobe)
- **Temporal (20)**: 8 basic (mean, std, skew, kurtosis, psd, mssd, range, autocorr) + 12 frequency (delta/theta/alpha/beta/gamma power×2 + spectral_entropy + phase_std)
- **Internal (2)**: Regional Homogeneity coherence + spatial_variance
- **Spatial (6)**: x, y, z_depth, size, conf_std, detection_count
- **Order matters**: `ALL_FEATURE_NAMES = temporal + frequency + internal + spatial` ([config.py](src/core/config.py#L113-L119))

### **Dataset Loading Pattern**
```python
from src.features.graph_factory import ABIDECausalDataset
from torch_geometric.loader import DataLoader

train_dataset = ABIDECausalDataset(split='train')  # Validates on __init__
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

for batch in train_loader:
    logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
```

### **Checkpoint Management** ([training_utils.py](src/models/training_utils.py))
```python
from src.models.training_utils import CheckpointManager

ckpt_mgr = CheckpointManager(CHECKPOINT_DIR, prefix="best_model")
ckpt_mgr.save(model, optimizer, epoch=10, metrics={'auc': 0.65})
model, optimizer, metadata = ckpt_mgr.load("best_model_fold0.pt")
```

### **Error Handling: Graceful Degradation**
- Never propagate errors downstream
- Fallback to simpler methods (e.g., PCA fails → use mean aggregation)
- Log warnings, continue processing
- Early validation in `__init__` or at function entry (fail fast)

**Example Pattern:**
```python
# In construct_causal.py, aggregate_to_lobes()
try:
    # Try PCA
    u, s, vh = torch.linalg.svd(centered, full_matrices=False)
    dominant_signal = u[:, 0] * s[0]
except Exception as e:
    logger.debug(f"SVD failed ({str(e)}), falling back to mean")
    dominant_signal = roi_data.mean(dim=1)  # Graceful fallback
```

### **Validation Patterns** (Multi-level integrity checking)

**1. Early Validation (prevent bad data from propagating)**
```python
# graph_factory.py: ABIDECausalDataset.__init__
def _validate_feature_dimensions(self):
    """Ensure loaded features match config expectations."""
    sample_sub = self.node_attr.index[0]
    temporal = self._get_subject_temporal(sample_sub)
    
    if temporal.shape != (NUM_LOBES, NUM_TEMPORAL_FEATURES):
        raise ValueError(f"Shape mismatch: expected ({NUM_LOBES}, {NUM_TEMPORAL_FEATURES}), got {temporal.shape}")
```

**2. Comprehensive Checks (pipeline_checks.py)**
- Atlas integrity (AAL3 dimensions, ROI count range 164-170)
- Post-download integrity (PNG/NPY file validation)
- Feature distribution (NaN/Inf detection, outlier checking)
- Graph sparsity (ensure minimum edge count)
- Stratification correctness (DX + SITE balancing)

**3. Fold-Safe Validation**
- Harmonization: Train on fold i, apply to val/test i (never leakage)
- Graphs: Built per-subject, never aggregated across folds
- Metrics: Computed per-fold, then aggregated with proper CI

### **Data Shapes** (Expected)
| Type | Shape | Notes |
|------|-------|-------|
| Node features | `(12, 28)` | 12 lobes × 28 features |
| Causal adjacency | `(12, 12)` | Directed graph edges |
| GNN input x | `(N_nodes, 28)` | Batched graphs (N ≈ 12 × batch_size) |
| Edge index | `(2, E)` | COO sparse format |
| Model output | `(B, 2)` | Batch × {Control, ASD} logits |
| Harmonized tensor | `(1035, 12, 20)` | 1035 subjects × 12 lobes × 20 temporal features |

### **Constants** (Import from config, never hardcode)
```python
from src.core.config import (
    NUM_LOBES,               # 12 regions
    GNN_IN_CHANNELS,         # 28 features
    NUM_TEMPORAL_FEATURES,   # 20
    NUM_SPATIAL_FEATURES,    # 6
    CHECKPOINT_DIR,          # models/checkpoints/
    MASTER_MANIFEST,         # data/metadata/master_manifest.csv
    K_FOLDS,                 # 5
    LOBE_MAPPING,            # {lobe_id: [roi_indices]}
    ALL_FEATURE_NAMES,       # Complete feature list (order-critical)
)
```

### **Reproducibility Checklist**
When making changes to feature extraction or hyperparameters:
1. Set `torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)`
2. Test on same data split (use `--skip-download --skip-split` flag)
3. Compare to known baseline AUC (CV: 0.7434 ± 0.0417, per-fold: [0.7317, 0.7576, 0.7606, 0.6709, 0.7964])
4. Report fold-wise metrics, not just mean
5. Include bootstrap 95% CI in all results

## Integration Points

**PyTorch Geometric**
- Inherit from `torch_geometric.data.Dataset` (see [graph_factory.py](src/features/graph_factory.py))
- Return `torch_geometric.data.Data(x=..., edge_index=..., edge_attr=..., y=...)`
- Use `DataLoader` for batching
- GNN forward signature: `model(x, edge_index, edge_attr, batch)`

**YOLO (Ultralytics)**
- Load: `YOLO(YOLO_WEIGHTS_PATH)` ([config.py](src/core/config.py#L126))
- Train: `model.train(**YOLO_TRAIN_CONFIG)` (consolidated config, no parameter duplication)
- Predict: `model.predict(img_path, conf=YOLO_CONF_THRESHOLD)`

**neuroHarmonize (ComBat)**
- Fold-safe: fit on train, transform val/test ([fold_safe_harmonization.py](src/features/fold_safe_harmonization.py))
- Protect covariates: `batch_col='SITE_ID', covars='DX_GROUP'`

**Statsmodels (Granger Causality)**
- Multi-lag test (1-5 TRs): `grangercausalitytests(data, maxlag=5)`
- Extract p-value: `test[lag][0]['ssr_ftest'][1]` ([causal_inference.py](src/features/causal_inference.py))

**Captum (Explainability)**
- GradCAM: `LayerGradCam(model, model.conv1)` ([node_importance.py](src/analysis/node_importance.py))
- Integrated Gradients, Saliency, DeepLift ([feature_attribution.py](src/analysis/feature_attribution.py))

## Hidden Implementation Details (Advanced)

### **Feature Extraction Edge Cases**

1. **Frequency Features Near Nyquist**
   - Current: TR=2.0s → Nyquist = 0.25 Hz
   - Beta (0.15-0.20 Hz) and Gamma (0.20-0.25 Hz) sit at/near Nyquist
   - **Issue**: fMRI aliasing risk in high-frequency bands
   - **Workaround**: Trust delta/theta/alpha (0.01-0.15 Hz), treat gamma skeptically
   - **Future**: Phase 10.2 will audit these and possibly remove unreliable bands

2. **PCA Eigenvariate vs Mean Aggregation**
   - ROIs within a lobe can be **anti-correlated** (motor, cingulate areas)
   - Simple mean averaging cancels opposing signals
   - PCA's first PC preserves signal amplitude and direction
   - **Trade-off**: Eigenvariate captures dominant variance, loses smaller orthogonal patterns
   - **Code**: [construct_causal.py](src/features/construct_causal.py#L40-L60)

3. **Double Z-Score Normalization Issue**
   - `abide_download.py` line 82: `standardize='zscore_sample'` in NiftiLabelsMasker
   - `construct_causal.py` line 40: Manual z-scoring again before correlation
   - **Impact**: Compresses signal variance → weaker biomarkers
   - **Status**: Known issue, Phase 10.1 fix pending

### **Graph Construction Edge Cases**

1. **Minimum Edge Preservation**
   ```python
   # construct_causal.py: adaptive sparsification
   MIN_EDGES_PER_GRAPH = 12  # Ensure connectivity for 12-node graphs
   SPARSITY_QUANTILE = 0.70  # Keep top 30% edges
   
   # If top 30% < 12 edges, increase threshold
   num_edges = (adj != 0).sum()
   if num_edges < MIN_EDGES_PER_GRAPH:
       adjust_threshold()  # Fall back to less sparse graph
   ```

2. **Zero-Edge Graphs**
   - Subject has extremely weak connectivity → all edges pruned
   - **Handling**: Skip graph with warning log (see graph_factory.py _validate_subjects)

3. **NaN/Inf Propagation**
   ```python
   # Graceful handling in aggregate_to_lobes()
   if torch.isnan(signal).any() or torch.isinf(signal).any():
       lobe_signals.append(torch.zeros(...))  # Don't crash, use zero
       logger.warning("NaN/Inf detected, using zero signal")
   ```

### **Training Loop Patterns**

1. **5-Fold Stratified CV**
   ```python
   # gnn_model.py: stratified k-fold ensures DX + SITE balance per fold
   splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
   splits = splitter.split(manifest, manifest['DX_GROUP'])
   
   for fold, (train_idx, val_idx) in enumerate(splits):
       # Train on fold, validate on fold (never test)
       train_loader = DataLoader(train_dataset[train_idx], ...)
   ```

2. **OneCycle Learning Rate Schedule**
   ```python
   # training_utils.py: OneCycle LR from GNN_LEARNING_RATE to GNN_ONECYCLE_MAX_LR
   scheduler = torch.optim.lr_scheduler.OneCycleLR(
       optimizer,
       max_lr=GNN_ONECYCLE_MAX_LR,  # 0.002
       total_steps=total_epochs * steps_per_epoch
   )
   ```

3. **Early Stopping Logic**
   ```python
   # Monitor validation AUC, stop if no improvement for 20 epochs
   early_stop = EarlyStopping(patience=30, mode='max')
   for epoch in range(100):
       val_auc = validate(model, val_loader)
       if early_stop(val_auc):  # Returns True when patience exhausted
           break
   ```

### **Fold-Safe Harmonization Pipeline**

```python
# fold_safe_harmonization.py: Multi-step process
for fold_id in range(K_FOLDS):
    # 1. Fit ComBat on train data only
    train_features = features_df.iloc[train_idx]
    train_manifest = manifest_df.iloc[train_idx]
    
    model, train_harm = harmonizationLearn(
        train_features,
        train_manifest,
        batch_col='SITE_ID',
        covars='DX_GROUP'  # ← CRITICAL: Protect diagnosis
    )
    
    # 2. Apply to val/test using fitted model
    val_features = features_df.iloc[val_idx]
    val_harm, _ = harmonizationApply(
        val_features,
        model,
        batch_col='SITE_ID'
    )
    
    # 3. Write combined output for graph_factory
    combined = pd.concat([train_harm, val_harm, ...])
```

### **Permutation Testing (run_evaluation.py)**

```python
# Bootstrap CI: resample with replacement N=2000 times
bootstrap_aucs = []
for _ in range(2000):
    sample_idx = np.random.choice(len(y_test), size=len(y_test), replace=True)
    auc = roc_auc_score(y_test[sample_idx], y_pred[sample_idx])
    bootstrap_aucs.append(auc)

ci_lower = np.percentile(bootstrap_aucs, 2.5)
ci_upper = np.percentile(bootstrap_aucs, 97.5)

# Permutation test: shuffle labels N=1000 times
null_aucs = []
for _ in range(1000):
    y_shuffled = np.random.permutation(y_test)
    auc = roc_auc_score(y_shuffled, y_pred)
    null_aucs.append(auc)

p_value = (np.array(null_aucs) >= observed_auc).mean()
```

## Security

**Sensitive Data**
- ABIDE data: public dataset, no PHI (Protected Health Information)
- Phenotypic CSV: anonymized subject IDs (`sub-XXXX`)
- No credentials in code: AWS S3 access via boto3 (credentials in `~/.aws/`)

**Authentication**
- ABIDE S3 bucket: public read-only access (s3://fcp-indi/data/Projects/ABIDE)
- No authentication required for data download

**Validation & Integrity**
- Atlas validation: verify AAL3 file exists and dimensions match ([atlas_validator.py](src/validation/atlas_validator.py))
- Post-download integrity: check PNG/NPY files exist for all subjects ([audit_check.py](src/validation/audit_check.py))
- Pre-GNN validation: check feature dimensions and NaN/Inf values ([pipeline_checks.py](src/validation/pipeline_checks.py))

## Performance Characteristics

**Current Baseline (Feb 2026)**
- **CV AUC**: 0.7434 ± 0.0417 (5 folds; per-fold: [0.7317, 0.7576, 0.7606, 0.6709, 0.7964])
- **Test AUC**: 0.6487 [0.5618, 0.7300] (ensemble on 155 test subjects; p=0.0020 global / 0.0010 within-site)
- **Test F1**: 0.6738 | **AUPRC**: 0.6459 | **Sensitivity**: 0.7975 | **Specificity**: 0.4079
- **CV–test gap**: 0.0947 (site-DX correlations; GRL alpha grid search pending)
- **Best fold**: Fold 4 (AUC=0.7964); **Best epochs**: [42, 81, 75, 72, 75]
- **Source**: pipeline_20260309_194459.log (authoritative; second run 195751 overwrote checkpoints with worse results)

**Computational Requirements**
- GPU: NVIDIA V100 or RTX3090+ (11GB+ VRAM recommended)
- CPU: 16+ cores for parallel processing
- Memory: 32GB RAM for harmonization/graph construction steps

**Known Bottlenecks (Phase 10 Roadmap)**
1. Double z-score normalization compresses signal (Phase 10.1)
2. Frequency features near Nyquist have aliasing risk; gamma band flagged via `UNRELIABLE_FREQ_BANDS_AT_NYQUIST` (Phase 10.2)
3. CV-test gap suggests site-specific patterns → stronger harmonization needed (Phase 10.3)
4. P0 critical bugs in harmonization + evaluation scripts (see audit `plan-neuroCxgAudit.md`)

---

**Reference Documents**
- [CODEBASE_ARCHITECTURE_REPORT.md](../CODEBASE_ARCHITECTURE_REPORT.md) — Comprehensive 10-section analysis
- [AI_AGENT_QUICK_REFERENCE.md](../AI_AGENT_QUICK_REFERENCE.md) — Code patterns & debugging checklists
- [docs/DATAFLOW.md](../docs/DATAFLOW.md) — Visual pipeline flow (20 stages)
- [docs/ROADMAP.md](../docs/ROADMAP.md) — Phase-by-phase implementation roadmap
