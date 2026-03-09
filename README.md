# Neuro-CXG

**Causal Graph Neural Networks for Brain Disorder Classification from fMRI**

A Graph Neural Network framework for brain disorder classification (ASD vs Control) using causal inference and explainable AI (XAI) on fMRI data. Combines YOLO-based ROI detection, causal graph construction, and Graph Attention Networks for interpretable neuroimaging analysis.

**Key Features**

- **YOLO-based ROI Detection**: Automated detection of 12 brain anatomical regions in 2D MRI slices using YOLO26n
- **Advanced Feature Engineering**: 28 features per region (20 temporal + 2 internal + 6 spatial)
  - 8 basic temporal: mean, std, skew, kurtosis, PSD, MSSD, range, autocorr
  - 12 frequency-domain: delta/theta/alpha/beta/gamma power + peak frequencies + spectral entropy + phase std
  - 2 internal connectivity: Regional Homogeneity (ReHo) coherence + spatial variance
  - 6 spatial: x, y, z_depth, size, conf_std, detection_count
- **Causal Graph Construction**: 12×12 directed graphs with Granger causality (multi-lag 1-5 TRs)
- **Graph Neural Networks**: GATv2-based architecture (2 layers, 4 heads, 128 hidden channels, GELU activation) with skip connections and attention pooling
- **Batch Effect Harmonization**: fold-safe neuroHarmonize (ComBat) for multi-site data harmonization
- **Stratified k-fold Validation**: 5-fold CV balanced by diagnosis and scanner site
- **Explainability**: Gradient-based node importance and causal edge weight analysis
- **Unified Pipeline**: Single `run_pipeline.py` orchestrates all stages


### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/Neuro-CXG.git
cd Neuro-CXG

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (pinned versions for reproducibility)
pip install -r requirements.txt

# Verify installation
python -c "from src.core.config import validate_environment; validate_environment()"
```

## Quick Start

### 1. Data Preparation

Download ABIDE data and organize into train/val/test splits:

```bash
# Download raw fMRI and extract 7 z-slices per subject (ALFF percentiles 0.21–0.80)
python -m src.data.abide_download

# Create master manifest
python -m src.utils.manifestor

# Generate stratified train/val/test splits (70/15/15)
python -m src.data.split
```

### 2. ROI Detection (YOLO)

Train YOLO26n to detect 12 brain regions in slices:

```bash
python -m src.pipelines.roi_detection
# Outputs: results/experiments/detection/ROI_Detection_v29/weights/best.pt  (YOLO_PROJECT_NAME in config)
```

### 3. Feature Extraction

Extract temporal, frequency-domain, and spatial features from detected ROIs:

```bash
# Extract 3D spatial coordinates from YOLO detections
python -m src.features.extract_spatial
# Outputs: data/metadata/node_features_3d.csv

# Extract temporal features (8 basic stats + 12 frequency-domain features per ROI)
python -m src.features.extract_temporal
# Outputs: data/metadata/node_attributes_temporal.csv
# 20 total features/ROI: 8 time-domain (mean/std/skew/kurtosis/psd/mssd/range/autocorr)
#                      + 12 frequency (delta/theta/alpha/beta/gamma power+peaks+entropy+phase)
```

### 4. Fold-Safe Harmonization

Remove site-specific scanner bias while protecting disease signal (CV-safe):

```bash
python -m src.features.fold_safe_harmonization
# Outputs: data/metadata/node_attributes_harmonized.csv
# Also writes fold-specific files to: data/metadata/harmonized_folds_cv/
# (Note: DX_GROUP diagnosis protected as covariate)
```

### 5. Graph Construction

Build directed causal graphs from time series:

```bash
python -m src.features.construct_causal
# Outputs: data/processed/causal_graphs/{subject_id}_graph.pt
# Graphs: 12 nodes (regions), directed edges, causal weights
# Methods: Granger causality (default) or lagged Pearson correlation
# Sparsification: Adaptive (top 30% edges, min 12 edges/graph)
```

### 6. Model Training

Train GNN with 5-fold stratified cross-validation:

```bash
python -m src.models.gnn_model
# Checkpoints saved to: models/checkpoints/best_model_fold{0-4}.pt
# Logs metrics: Accuracy, F1, AUC, Confusion Matrix per fold
# Latest reported training (March 9, 2026): Mean CV AUC 0.7434 ± 0.0417, Test AUC 0.6487 [0.5618, 0.7300] (GRL disabled)
```

### OR: Run Full Pipeline (Recommended)

Use the unified pipeline runner to execute all stages:

```bash
# Full pipeline (auto-run missing stages)
python src/run_pipeline.py --auto

# Skip download and data splitting (use existing data)
python src/run_pipeline.py --auto --skip-download --skip-split

# Force reset all intermediate files and rebuild
python src/run_pipeline.py --force-reset

# Show execution plan without running
python src/run_pipeline.py --dry-run

# Run only post-training analysis
python src/run_pipeline.py --analysis-only

# Run only visualization generation
python src/run_pipeline.py --visualizations-only
```

The pipeline orchestrates 20 stages:

**Core Pipeline (Stages 1–15):**
1. ABIDE download — fMRI data + 7-slice ALFF export (optional)
2. Stratified train/val/test split — 2D by DX_GROUP + SITE_ID (70/15/15)
3. Master manifest generation — subject ↔ phenotype mapping
4. Atlas validation — verify AAL3v1 files exist and are valid
5. Pipeline validation — comprehensive pre-flight health check
6. Post-download integrity — PNG/NPY file validation
7. Atlas-based label annotation — generate YOLO training labels
8. YOLO training — 12-region ROI detection (skip if weights exist)
9. Spatial feature extraction — 3D coordinate aggregation, all-12-region filter
10. Temporal feature extraction — 20 features/ROI (8 time-domain + 12 frequency)
11. Fold-safe harmonization — neuroHarmonize (ComBat), protects DX_GROUP covariate
12. Pre-GNN integrity check — validate feature completeness per split
13. Causal graph construction — 12×12 directed Granger causality graphs
14. Pipeline diagnostics — comprehensive health report (post-graph)
15. Quality validation — YOLO quality, graph sparsity, stratification checks

**GNN Training (Stage 16):**
16. GNN training — 5-fold stratified CV with 28-feature input, GATv2+attention pooling

**Post-Training Analysis (Stages 17–20):**
17. Visualizations — comprehensive plots, causal graph figures, feature heatmaps
18. Comprehensive evaluation — bootstrap 95% CI, permutation test, baseline comparison
19. Explainability analysis — node/edge importance, feature attribution (Captum/gradient saliency), literature validation
20. Result interpretation — per-subject predictions, misclassification analysis, site effects

## Current Results (March 9, 2026)

### YOLO26n ROI Detection Performance

**Deployed: ROI_Detection_v29** (100 epochs, March 9, 2026)
- **mAP50**: 0.99428
- **mAP50-95**: 0.9598
- **Precision**: 0.98734
- **Recall**: 0.98383
- **Status**: ✅ Outstanding — production-ready for 12-region ROI detection
- **Deployed weights**: `results/experiments/detection/ROI_Detection_v29/weights/best.pt`
- **Previous deployed (v28)**: mAP50-95=0.93714, mAP50=0.98952 (Feb 2–4 2026)

### GNN Classification Performance (Updated March 9, 2026)

**5-Fold Cross-Validation (719 train subjects) — 28-Feature Model, GRL Disabled:**

| Metric | Mean ± Std | Range | Notes |
|--------|------------|-------|-------|
| **CV AUC** | 0.7434 ± 0.0417 | 0.6709 – 0.7964 | Early stopping active |
| **CV F1** | 0.7419 ± 0.0195 | — | At per-fold optimal threshold |
| **CV Accuracy** | 0.6856 ± 0.0345 | — | — |
| **Test AUC** | 0.6487 [0.5618, 0.7300] | — | 95% bootstrap CI; p=0.0020 (global) |
| **Test F1** | 0.6738 | — | At threshold 0.500 |
| **Test AUPRC** | 0.6459 | — | Average Precision Score (PR-AUC) |
| **Test Sensitivity** | 0.7975 | — | At threshold 0.500 |
| **Test Specificity** | 0.4079 | — | At threshold 0.500 |
| **Test Accuracy** | 0.6065 | — | At threshold 0.500 |
| **Best Epochs** | [42, 81, 75, 72, 75] | — | Per-fold best epoch |

**Test-Set Ensemble AUC**: 0.6487 [95% CI: 0.5618, 0.7300] — **p=0.0020 global / p=0.0010 within-site (statistically significant)**  
Held-out 155 subjects (9 excluded from full 1035: 4 Caltech NaN + 5 degenerate graph subjects); AUC-weighted ensemble using val-fold AUCs.

**Per-Fold CV AUCs (March 9, 2026 — pipeline_20260309_194459.log):**
- Fold 1: 0.7317 (epoch 42)
- Fold 2: 0.7576 (epoch 81)
- Fold 3: **0.7606** ⭐ Best fold (epoch 75)
- Fold 4: 0.6709 (epoch 72)
- Fold 5: 0.7964 (epoch 75)

**Key Findings:**
- ✅ YOLO detection: Exceptional reliability (mAP50-95: 0.9598, v29 deployed)
- ✅ GNN classification: CV AUC 0.7434 — significant improvement from Phase 10 fixes
- ✅ Architecture: 2-layer GATv2, 128 hidden channels, 4 heads, GELU, attention pooling, skip connections
- ✅ Smart aggregation: PCA eigenvariate + ReHo coherence capture both global and local connectivity
- ✅ GRL disabled: `GNN_USE_GRL=False` — alpha=1.0 collapsed all predictions to ~0.56 constant
- 📊 Regularization: Dropout 0.35, L2 weight decay 5e-5, focal loss α=0.62 γ=2.0, early stopping patience=30
- 📊 Test p=0.0020 (global), p=0.0010 (within-site) — statistically significant
- 🔍 Subgroup: Male AUC=0.6662 (n=132), Female=0.5923 (n=23); Age<15 AUC=0.6580 (n=88), Age≥15=0.6348 (n=67)
- 🔍 Per-site: site 9 AUC=0.9500 (best, n=9), site 16 AUC=0.3281 (worst, n=16)
- ⚠️ CV–test gap: 0.7434 (CV) vs 0.6487 (test) = 0.0947 — site-DX correlations in embeddings
- ⚠️ Gamma band zeroed at Nyquist (TR=2 s, Nyquist=0.25 Hz); `UNRELIABLE_FREQ_BANDS_AT_NYQUIST=("gamma",)`

**Interpretation:**
- **YOLO performance**: Production-ready at 0.9598 mAP50-95 (v29)
- **Feature engineering**: 28-feature model (20 temporal + 2 internal ReHo + 6 spatial) properly integrated
- **Architecture**: 2-layer GATv2 with attention pooling; GRL disabled; 16-dim node identity embeddings + 16-dim site embeddings
- **Training stability**: Early stopping patience=30; 9 subjects permanently excluded (dead lobes or critical NaN)
- **Signal detection**: Granger causality with 0.70 sparsity, adaptive statistical thresholding

**Phase 10 Fixes Applied (March 8–9, 2026):**
1. ✅ Single z-score only: `standardize=False` in NiftiLabelsMasker (Phase 10.1)
2. ✅ BANDPASS_HIGH expanded: 0.08 → 0.15 Hz to retain beta band (Phase 10.1)
3. ✅ GRL disabled: `GNN_USE_GRL=False`, `GNN_GRL_ALPHA=0.0` (Phase 10.3)
4. ✅ All P0/P1 audit bugs fixed: DX_GROUP covariate, circular ensemble, site_id, Bonferroni, NaN leakage
5. ✅ Granger causality: multi-lag up to 10 s history (GRANGER_MAX_LAG_SECONDS=10.0), adaptive statistical sparsification
6. ✅ FEATURE_GROUPS registry — 28-dimension consistency enforced from config
7. ✅ Phase 9 complete: full evaluation pipeline, explainability analysis, result interpretation
8. ✅ 9 permanently excluded subjects in `EXCLUDED_SUBJECTS` frozenset (config.py)

## System Architecture

This section describes the end-to-end data flow—from raw fMRI acquisition through to GNN-based ASD classification—and the critical contracts between each stage.

### High-Level Pipeline

```
abide_download.py
    ↓  PNG brain slices (7 z-slices/subject, z-percentiles 0.2–0.8)
    ↓  *_ts.npy  (T × 170 ROI time series)

split.py  →  master_manifest.csv  (manifestor.py)
    ↓  data/final/{train,val,test}/{images,time_series}/

roi_detection.py  (YOLO26n)
    ↓  best.pt weights

extract_spatial.py  ←  best.pt
    ↓  node_features_3d.csv  (N × [12 regions × 6 spatial features])
       Only subjects where ALL 12 regions were detected proceed.

extract_temporal.py  ←  *_ts.npy
    ↓  node_attributes_temporal.csv  (N × [12 regions × 20 temporal features])
       Aggregates 170 AAL ROIs → 12 brain regions via LOBE_MAPPING.

fold_safe_harmonization.py  ←  node_attributes_temporal.csv
    ↓  node_attributes_harmonized.csv  +  harmonized_folds_cv/fold_k.csv
       ComBat (neuroHarmonize) fitted on train fold only; applied to val/test.
       DX_GROUP (diagnosis) is a PROTECTED covariate—never harmonized away.

construct_causal.py  ←  *_ts.npy  +  node_attributes_harmonized.csv
    ↓  causal_graphs/{subject_id}_graph.pt
       12×12 directed adjacency (Granger causality, multi-lag 1-5 TRs).
       Adaptive sparsification: top 30% edges, min 12 edges per graph.

graph_factory.py  (ABIDECausalDataset)
    ↓  PyTG Data(x, edge_index, edge_attr, y, internal_features)
       x shape: (12, 28)  — 20 temporal + 2 internal + 6 spatial
       Validates shape against (NUM_LOBES, GNN_IN_CHANNELS).

gnn_model.py  ←  ABIDECausalDataset
    ↓  5-fold stratified CV on train set (699 subjects)
    ↓  models/checkpoints/best_model_fold{0-4}.pt
       GATv2Conv × 2 layers, 4 heads, 128 hidden channels, GELU, skip connections.
       Focal Loss α=0.62 γ=2.0. AdamW + OneCycleLR. Early stopping patience=30.
```

### Data Contracts Between Modules

| Output File | Produced By | Consumed By | Shape / Content |
|---|---|---|---|
| `*_ts.npy` | `abide_download.py` | `extract_temporal.py`, `construct_causal.py` | `(T, 170)` float32 |
| `*_roi_labels.npy` | `abide_download.py` | (debugging / atlas alignment) | `(num_detected_rois,)` int — actual masker ROI IDs |
| `node_features_3d.csv` | `extract_spatial.py` | `graph_factory.py` | `(N, subject_id + 12×6 cols)` |
| `node_attributes_temporal.csv` | `extract_temporal.py` | `fold_safe_harmonization.py` | `(N, subject_id + 170×20 cols)` = 3401 cols |
| `node_attributes_harmonized.csv` | `fold_safe_harmonization.py` | `graph_factory.py` | `(N, subject_id + 12×20 cols)` = 241 cols — 170→12 aggregation done here |
| `{sub}_graph.pt` | `construct_causal.py` | `graph_factory.py` / `gnn_model.py` | `dict(adj: (12,12), internal_features: (12,2))` |
| `master_manifest.csv` | `manifestor.py` | `graph_factory.py`, `fold_safe_harmonization.py` | `subject_id, DX_GROUP, SITE_ID, split, TR, AGE_AT_SCAN, SEX, FIQ, HANDEDNESS_CATEGORY` |

### Fold-Safe Harmonization — Leakage Prevention

ComBat harmonization is a global fit that can propagate validation-set statistics into training data if applied naively. The pipeline prevents this as follows:

1. `StratifiedKFold(n_splits=5)` partitions the **training** subjects only.
2. For each fold, `harmonizationLearn()` is called **only on the train-half** of that fold.
3. The fitted ComBat model is then applied to the val-half via `harmonizationApply()`.
4. The held-out test set is harmonized using the model from the final/best fold.

This guarantees that no validation-set summary statistics (site means, variances) influence the ComBat betas that normalize the training data.

### Key Invariants

- **All 12 regions required**: If YOLO does not detect all 12 brain regions for a subject, that subject is excluded from all downstream stages. Downstream code unconditionally assumes `x.shape == (12, 28)`.
- **Feature ordering is fixed**: `ALL_FEATURE_NAMES` in `config.py` defines the canonical 28-feature order. Never reorder independently in `extract_temporal.py` or `graph_factory.py`—both must agree.
- **DX_GROUP encoding**: `DX_GROUP=1` (Control) maps to label `0`; `DX_GROUP=2` (ASD) maps to label `1`. This inversion is deliberate and consistent everywhere.
- **Single graph format**: Causal graphs are saved as `dict(adj=Tensor(12,12), internal_features=Tensor(12,2))`, **not** as PyTorch Geometric `Data` objects on disk; `graph_factory.py` assembles the full `Data` object at load time.

## Project Structure

```
Neuro-CXG/
├── .github/
│   └── copilot-instructions.md    # AI agent guidelines + architecture reference
├── configs/
│   └── brain.yaml                 # YOLO class config (12 ROI classes)
├── data/                          # Not tracked in git
│   ├── raw/atlases/               # AAL3v1 reference atlas (.nii)
│   ├── processed/
│   │   ├── Phenotypic_V1_0b_preprocessed1.csv
│   │   └── causal_graphs/         # Per-subject PyTorch graph dicts (.pt)
│   ├── metadata/                  # Generated CSVs (features, manifests)
│   │   ├── master_manifest.csv
│   │   ├── node_features_3d.csv
│   │   ├── node_attributes_temporal.csv
│   │   ├── node_attributes_harmonized.csv
│   │   └── harmonized_folds_cv/   # Per-fold ComBat models
│   └── final/{train,val,test}/    # Split images + time_series
├── src/
│   ├── core/
│   │   └── config.py              # ⭐ SINGLE SOURCE OF TRUTH — all constants & paths
│   ├── run_pipeline.py            # Unified orchestrator (20 stages)
│   ├── run_evaluation.py          # Bootstrap CI, permutation test, baseline comparison
│   ├── run_explainability.py      # Node/edge importance, feature attribution (Captum)
│   ├── run_result_analysis.py     # Per-subject predictions, misclassification analysis
│   ├── validation/
│   │   ├── atlas_validator.py     # AAL atlas structure & ROI range validation   │   ├── audit_check.py         # Post-fix validation (subject count, lobe completeness, NaN/Inf)│   │   ├── pipeline_checks.py     # Post-download, pre-GNN, health reports, class analysis
│   │   └── dev_audit.py           # Deep validation: feature quality, graph connectivity
│   ├── experiments/
│   │   ├── run_ablations.py       # 5 ablation types (A–E)
│   │   └── data_quality.py        # 3 data-quality experiments
│   ├── features/
│   │   ├── extract_spatial.py     # YOLO inference → 3D spatial aggregation   │   ├── extract_spatial_atlas.py  # Atlas-based spatial coords (alternative, uses roi_centroids.json)│   │   ├── extract_temporal.py    # 20 features/ROI (8 time-domain + 12 frequency)
│   │   ├── causal_inference.py    # Granger causality & transfer entropy
│   │   ├── fold_safe_harmonization.py  # CV-safe ComBat + NaN/Inf handling
│   │   ├── construct_causal.py    # 12×12 directed causal graph builder
│   │   └── graph_factory.py       # ABIDECausalDataset — PyG Data loader
│   ├── data/
│   │   ├── split.py               # 2D stratified split (DX_GROUP + SITE_ID)
│   │   ├── abide_download.py      # ABIDE S3 download + 7-slice ALFF export
│   │   └── filter_to_1000.py      # Optional subject count filter
│   ├── models/
│   │   ├── causal_gnn.py          # CausalBrainGNN — GATv2 + GRL + edge gate
│   │   ├── gnn_model.py           # 5-fold training loop with FocalLoss + OneCycleLR
│   │   └── training_utils.py      # EarlyStopping, CheckpointManager, TrainingTracker
│   ├── analysis/
│   │   ├── diagnostics.py         # CausalGraphAnalyzer, TrainingMonitor
│   │   ├── feature_attribution.py # Captum-based integrated gradients
│   │   ├── node_importance.py     # Per-node gradient saliency
│   │   ├── edge_importance.py     # Causal edge weight analysis
│   │   ├── literature_validation.py # ASD biomarker comparison
│   │   └── visualizations.py      # All plots and figures
│   ├── pipelines/
│   │   ├── roi_detection.py       # YOLO training entry point
│   │   └── generate_labels.py     # Atlas-based YOLO label annotation
│   └── utils/
│       └── manifestor.py          # Master manifest generation
├── notebooks/
│   └── eda.ipynb                  # Exploratory data analysis
├── models/
│   │   ├── checkpoints/               # best_model_fold{0-4}.pt (canonical: pipeline_20260309_194459; disk: Run 2 195751)
│   └── checkpoints_baseline/      # Baseline fold checkpoints
├── results/
│   ├── experiments/
│   │   ├── detection/             # YOLO training results (v29 deployed)
│   │   ├── training/              # GNN fold metrics JSON
│   │   ├── ablations/             # Ablation study outputs
│   │   └── data_quality/          # Data quality experiment outputs
│   ├── evaluation/                # Bootstrap CI, ensemble AUC, baselines
│   ├── figures/                   # All generated plots
│   └── analysis/                  # Per-subject predictions, misclassification
├── docs/
│   ├── ROADMAP.md                 # All phases & completion status
│   ├── DATAFLOW.md                # Pipeline visualisation
│   └── TODO.md                    # Deep architectural analysis & open issues
├── requirements.txt               # Pinned versions (torch==2.9.0, etc.)
└── README.md                      # This file
```

## Configuration

All project constants defined in [src/core/config.py](src/core/config.py) (single source of truth):

### Critical Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| NUM_LOBES | 12 | Frontal_Superior, Frontal_Orbital, Motor_Premotor, Insula, Cingulate, Limbic, Occipital, Parietal, Temporal, Subcortical, Cerebellum, Brainstem |
| LOBE_MAPPING | 170→12 | AAL3 atlas ROI aggregation (1-indexed to 0-indexed) |
| GNN_IN_CHANNELS | 28 | Dynamically computed: `len(ALL_FEATURE_NAMES)` |
| GNN_HIDDEN_CHANNELS | 128 | Hidden dim for GATv2Conv |
| GNN_NUM_HEADS | 4 | Attention heads per GAT layer |
| GNN_NUM_GNN_LAYERS | 2 | Number of GATv2 layers |
| GNN_DROPOUT | 0.35 | Dropout for regularisation |
| GNN_WEIGHT_DECAY | 5e-5 | L2 regularisation (AdamW) |
| GNN_POOLING | 'attention' | GlobalAttention pooling |
| GNN_USE_GRL | False | GRL disabled — alpha=1.0 collapsed representations (Phase 10.3) |
| GNN_GRL_ALPHA | 0.0 | GRL adversarial strength (0=disabled) |
| GNN_SITE_LOSS_WEIGHT | 0.0 | Weight for auxiliary site classification loss (0=disabled) |
| GNN_EDGE_GATE | True | Learnable sigmoid gate on causal edge weights |
| GNN_USE_SITE_EMBEDDING | True | 16-dim site embeddings to reduce scanner bias |
| GNN_USE_DEMOGRAPHICS | True | Condition on age / sex / FIQ |
| GNN_NODE_EMB_DIM | 16 | Learnable per-lobe identity embedding dimension |
| GNN_ONECYCLE_MAX_LR | 0.002 | Peak LR for OneCycleLR scheduler |
| GNN_ONECYCLE_PCT_START | 0.2 | Warmup fraction (peak at epoch 20 of 100) |
| GNN_EARLY_STOPPING_PATIENCE | 30 | Epochs without improvement before stopping |
| K_FOLDS | 5 | Stratified cross-validation folds |
| YOLO_MODEL_SIZE | 'yolo26n.pt' | Base model architecture |
| YOLO_PROJECT_NAME | 'ROI_Detection_v29' | Deployed YOLO model output directory (mAP50-95=0.9598, trained March 9, 2026) |
| YOLO_BATCH_SIZE | 32 | YOLO training batch size |
| YOLO_EPOCHS | 100 | YOLO training epochs |
| CAUSALITY_METHOD | 'granger' | Directed causality (options: 'granger', 'lagged_pearson') |
| GRANGER_MAX_LAG_SECONDS | 10.0 | Causal history window in real seconds (adjusted per subject TR) |
| SPARSITY_METHOD | 'adaptive_statistical' | Sparsification method (options: 'adaptive_proportional', 'fixed') |
| SPARSITY_QUANTILE | 0.70 | Keep top 30% causal connections |
| MIN_EDGES_PER_GRAPH | 12 | Minimum connectivity guarantee |
| MAX_NAN_ROIS | 30 | Max NaN temporal entries before subject excluded |
| FOCAL_LOSS_ALPHA | 0.62 | Alpha weight for positive (ASD) class |
| FOCAL_LOSS_GAMMA | 2.0 | Focal parameter — focus on hard examples |
| DEFAULT_TR | 2.0 | Fallback TR (seconds) when missing from manifest |

See [src/core/config.py](src/core/config.py) for all 60+ parameters.

## Data Format

### Time Series Input
- **Shape**: `(timepoints, 170)` - fMRI signal from 170 AAL ROIs
- **Processing**: Bandpass filtered (0.01–0.15 Hz, BANDPASS_LOW/HIGH from config), single z-score in construct_causal.py

### Graph Construction Output

**Intermediate format** (saved by construct_causal.py):
```python
{
  'adj': torch.Tensor(12, 12),       # 12×12 directed adjacency matrix
  'subject_id': str,                # Subject identifier
  'lobe_order': list                # ['Frontal', 'Temporal', ...]
}
```

**Final format** (loaded by graph_factory.py into PyTorch Geometric):
```python
Data(
  x=torch.Tensor(12, 28),          # 12 regions × (20 temporal + 2 internal + 6 spatial)
  edge_index=torch.Tensor(2, K),  # K directed edges (min 12 after sparsification)
  edge_attr=torch.Tensor(K, 1),   # Causal weights shaped (K,1) — Granger: -log10(p), Pearson: [-1,1]
  y=torch.Tensor([0 or 1]),        # Label: 0=Control, 1=ASD
  pos=torch.Tensor(12, 3),         # XYZ centroid coords (first 3 spatial features)
  sub_id=str,                      # Subject identifier
  site_id=torch.Tensor([int]),     # Site index 0–19 (maps to 20 ABIDE sites)
  age=torch.Tensor([float]),       # Normalized age: (age-15)/20
  sex=torch.Tensor([float]),       # Normalized sex: (sex-1.5)
  fiq=torch.Tensor([float])        # Normalized FIQ: (fiq-100)/30
)
```

**Key differences:**
- `construct_causal.py` outputs raw adjacency matrices as dictionaries
- `graph_factory.py` (ABIDECausalDataset) converts to PyG Data objects on-the-fly
- Edge sparsification: Keep top 30% of connections (min 12 edges/graph)
```

## Post-Training Analysis

All analysis stages can be re-run independently after training:

```bash
# Re-run all post-training stages
python src/run_pipeline.py --analysis-only

# Generate visualizations only
python src/run_pipeline.py --visualizations-only
python src/analysis/visualizations.py

# Comprehensive evaluation (bootstrap CI, permutation test, baselines)
python src/run_evaluation.py
# Outputs:
#   results/evaluation/comprehensive_results.{csv,json}
#   Baseline comparison: SVM, Random Forest, FlatMLP vs GNN
#   95% bootstrap CI (N=2000) for AUC, AUPRC, F1, sensitivity, specificity

# Explainability analysis
python src/run_explainability.py
# Outputs:
#   Node/edge importance scores
#   Captum integrated gradients (feature attribution)
#   Literature validation against ASD biomarkers

# Result interpretation
python src/run_result_analysis.py
# Outputs:
#   Per-subject predictions CSV (true label, pred, prob_asd, confidence)
#   Misclassification analysis: FP/FN feature profiles
#   Site-effect investigation: per-site AUC, ASD-prevalence heatmap
#   Calibration: reliability diagram + confidence distribution
```

## Known Issues (as of March 9, 2026)

1. **Frequency band aliasing** — gamma (0.20–0.25 Hz) band sits at the fMRI Nyquist limit for TR=2 s. Gamma features are zeroed at runtime for TR=2 s subjects. Flagged via `UNRELIABLE_FREQ_BANDS_AT_NYQUIST=("gamma",)` in config. (Phase 10.2, open)
2. **CV–test AUC gap** — CV AUC 0.7434 vs test AUC 0.6487 (gap 0.0947). Remaining gap is from site-DX correlations in site embeddings; GRL alpha grid search {0.05, 0.1} pending.
3. **Per-site AUC variability** — site 9 AUC=0.9500 vs site 16 AUC=0.3281. Site effects not fully removed by ComBat + site embeddings alone.
4. **4 incomplete subjects** — subjects with images but no time series files remain in the split directories (not auto-purged; requires interactive pipeline re-run with option 2).
5. **Atlas coverage gaps** — Brainstem (ROIs 167–170) and Frontal_Orbital (ROIs 29–30) have NaN time series in some subjects due to partial FOV; zero-signal fallback applied automatically.

**All P0/P1 audit bugs from plan-neuroCxgAudit.md are resolved** — double z-score fix (Phase 10.1), GRL collapse fix (Phase 10.3), DX_GROUP covariate (Phase 10.5), circular ensemble weighting (Phase 10.5), site_id fix (Phase 10.5), Bonferroni correction (Phase 10.5).

## Validation & Testing

```bash
# Comprehensive health report
python src/validation/pipeline_checks.py --health

# Post-download dataset integrity
python src/validation/pipeline_checks.py --dataset

# Pre-GNN distribution check
python src/validation/pipeline_checks.py --distribution

# Class imbalance analysis
python src/validation/pipeline_checks.py --class-analysis

# Validate environment & config
python -c "from src.core.config import validate_environment; validate_environment()"

# Check lobe mapping completeness
python -c "from src.core.config import validate_lobe_mapping; validate_lobe_mapping()"

# Test dataset loading
python -c "from src.features.graph_factory import ABIDECausalDataset; \
ds = ABIDECausalDataset('train'); \
print(f'Loaded {len(ds)} graphs, node features: {ds[0].x.shape}')"
```
```

## Medical Context

- **Disorder**: Autism Spectrum Disorder (ASD)
- **Control**: Neurotypical development (TD)
- **Data Source**: ABIDE initiative (multi-site, n≈1000)
- **Modality**: resting-state fMRI (RS-fMRI)
- **Label Convention**: 0=Control, 1=ASD
- **Statistical Design**: 2D stratified by site + diagnosis (journal Q1 requirement)
- **Anatomical Framework**: AAL3v1 atlas (170 ROIs → 12 brain regions)

## Key Design Decisions

### Why 12 Regions Instead of 170 ROIs?
- **Computational**: Reduces graph from 170×170 (28,900 edges) to 12×12 (144 edges)
- **Interpretability**: Lobes have clear anatomical meaning for clinicians
- **Noise reduction**: Averaging within lobes reduces scanner-specific noise
- **Statistical power**: Fewer parameters = less overfitting on limited data (n~1000)

### Why GATv2 with 4 Heads?
- **Small graphs**: 12-node graphs benefit from moderate multi-head attention without overparameterization
- **Edge weights**: GAT naturally incorporates causal correlation weights via edge_attr
- **Attention**: Learns which lobe-lobe connections matter most for classification
- **Skip connections**: Prevent over-smoothing in small graphs

### Why Granger Causality (with Lagged Pearson as Baseline)?
- **Directed effects**: Tests temporal precedence with statistical significance
- **Multi-lag**: Captures dynamics across 1-5 TRs
- **Baseline option**: Lagged Pearson remains a fast fallback for ablations
- **Sparsification**: Adaptive top 30% quantile keeps only strong connections