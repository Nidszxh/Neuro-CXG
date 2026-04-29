# Data

## Architecture Status: 12-Lobe Approved (April 28, 2026)

The 12-lobe architecture is approved for publication. Key findings:

- **12-Lobe (Primary)**: YOLO v29 never detects Brainstem → synthetic fallback coordinates
- Counterintuitive: constant Brainstem features act as implicit regularization
- Test AUC: 0.8694 [95% CI: 0.7889–0.9037] — +8.74% vs 11-lobe
- Generalization: CV < Test (+0.0697) indicates robust learning

See `docs/decisions.md` (DD-018) and `FINAL_ARCHITECTURE_ANALYSIS.md` for full analysis.

---

## Scope

This document describes the runtime data model used by the current source code:

- where artifacts are written
- how features are encoded
- which quality gates are enforced
- how data moves from download to model-ready graphs

All paths and constants referenced here come from `src/core/paths.py` and `src/core/feature_registry.py` via `src/core/config.py`.

## Source Datasets

- Imaging source: ABIDE I resting-state fMRI from the public FCP-INDI bucket (s3://fcp-indi/data/Projects/ABIDE)
- Phenotype source: `data/processed/Phenotypic_V1_0b_preprocessed1.csv`
- Atlas source: `data/raw/atlases/AAL3v1.nii`

---

## Final Cohort

| Metric | Value |
|--------|-------|
| **Total subjects** | 1,015 |
| **ASD** | 486 (47.9%) |
| **Control (TYP)** | 514 (50.6%) |
| **Sites** | ~20 |

---

## Exclusion Funnel

| Step | Excluded | Cumulative | Reason |
|------|----------|-----------|--------|
| Download | ~50–100 | 50–100 | Data availability, corruption, bad download |
| Atlas validation | 5–10 | 55–110 | Missing/malformed ROI data |
| Dead lobe filter | 15–20 | 70–130 | Entire lobe missing spatial features |
| Harmonization | 0 | 70–130 | NaN/Inf in features (pre-filtered) |
| **Total excluded** | ~85–115 | — | — |
| **Final cohort** | — | **1,015** | — |

---

## Inclusion Criteria

1. ✅ Diagnosis group: ASD or Control (TYP)
2. ✅ Successful download from ABIDE S3
3. ✅ 7 z-slices extracted successfully (ALFF percentiles [0.21, 0.30–0.80])
4. ✅ Atlas overlaps with ROI extraction (at least 150 of 170 AAL3 ROIs detected)
5. ✅ Spatial features complete for ≥9 brain lobes (gates on region-of-interest detection)
6. ✅ No dead lobes (entire lobe missing post-spatial-extraction)
7. ✅ Temporal features complete and finite (no NaN/Inf)

---

## Exclusion Criteria

1. ❌ Diagnosis = "Other" or blank
2. ❌ Failed download or corrupted nifti file
3. ❌ <7 z-slices extracted (insufficient spatial coverage)
4. ❌ <150 of 170 AAL3 ROIs detected after atlas masking
5. ❌ <9 lobes with spatial features (failed YOLO ROI detection on multiple lobes)
6. ❌ Dead lobe: entire lobe missing YOLO detections (0 detections in lobe)
7. ❌ Any temporal feature NaN/Inf after extraction

---

## Data Curation Pipeline

### Stage 1: Download and Validation

**Validation** (`src/validation/audit_check.py`):
- File exists and readable
- Shape matches expected (7, 192, 144)
- No NaN/Inf in raw array

### Stage 2: Atlas Overlap Check

```
src/validation/atlas_validator.py
├── Load AAL3 atlas (170 ROIs, standard MNI 3mm space)
├── For each subject's fMRI:
│   └── Count voxels > threshold per ROI
└── Filter: ≥150 of 170 ROIs must have ≥1 voxel
```

**Purpose**: Ensure fMRI registration was successful.

### Stage 3: Spatial Feature Extraction

**Gating:**
```python
SPATIAL_MIN_REQUIRED_REGIONS = 9  # ≥9 lobes with ≥1 YOLO detection
```

**Dead Lobe Detection:**
```python
for lobe_id in range(NUM_LOBES):
    if lobe_features[lobe_id].all() == 0:  # No detections
        subject_status = "DEAD_LOBE"
        exclude_subject = True
```

### Stage 4: Temporal Feature Extraction

**Quality Check:**
```python
temporal_features = extract_temporal_all(subjects)
bad_idx = temporal_features.isna().any(axis=1) | temporal_features.isinf().any(axis=1)
if bad_idx.any():
    logger.warning(f"Found {bad_idx.sum()} subjects with NaN/Inf")
```

### Stage 5: Fold-Safe Harmonization

```
src/features/fold_safe_harmonization.py
├── For each of 5 CV folds:
│   ├── Fit ComBat on train subjects only
│   ├── Apply to val/test subjects
│   └── Per-fold output: harmonized_fold_k.csv
└── Output: NODE_ATTRIBUTES_HARMONIZED.csv (global, no leakage)
```

### Stage 6: Causal Graph Construction

**Dead Graph Detection:**
```python
num_edges = (adj_matrix != 0).sum()
if num_edges < MIN_EDGES_PER_GRAPH:  # MIN = 12
    logger.warning(f"Subject {sub_id}: weak connectivity ({num_edges} edges)")
```

## Directory-Level Data Flow

### Raw and intermediate pools

- `data/images/` - exported ALFF PNG slices used for detection and labeling
- `data/processed/time_series/` - canonical pre-split time-series pool (legacy-compatible fallback logic exists)
- `data/metadata/` - manifests and feature tables

### Split datasets

- `data/final/train/`
- `data/final/val/`
- `data/final/test/`

Each split contains:

- `images/` (`*_z*.png`)
- `labels/` (YOLO txt labels)
- `time_series/` (`*_ts.npy`, `*_roi_labels.npy`)

### Processed graph artifacts

- `data/processed/causal_graphs/` - base subject graphs (`<subject>_graph.pt`)
- `data/processed/causal_graphs_multiview/` - optional multiview packages (`<subject>/multiview_graphs.pt`)

## Manifest And Split Contract

- Manifest path: `data/metadata/master_manifest.csv`
- Required training columns include `subject_id`, `split`, `DX_GROUP`, `SITE_ID`, and `cv_fold`
- `split.py` supports:
  - standard stratified train/val/test split
  - optional site-stratified CV fold regeneration (`--site-stratified-cv`)

---

## Subject ID Format and Site Roster

### ID Convention
Format: `<SITE>_<COHORT>_<SUBJECT>`

Examples:
- `CMU_a_0050642` → CMU site, cohort a, subject ID 0050642
- `NYU_0050952` → NYU site, no cohort suffix
- `Leuven_1_0050682` → Leuven, site 1, subject ID

### Site Roster (~20 sites)

| Site | Count | Abbr |
|------|-------|------|
| CMU (Carnegie Mellon) | 24 | CMU |
| Caltech | 24 | Caltech |
| KKI (Kennedy Krieger Institute) | 77 | KKI |
| Leuven | 57 | Leuven_1, Leuven_2 |
| Max Planck Munich | 54 | MaxMun_* |
| NYU (New York University) | 196 | NYU |
| OHSU (Oregon Health & Science) | 25 | OHSU |
| Olin | 36 | Olin |
| Pitt (University of Pittsburgh) | 87 | Pitt |
| SBL (Schulich Brain Lab) | 29 | SBL |
| SDSU (San Diego State) | 29 | SDSU |
| Stanford | 47 | Stanford |
| Trinity | 40 | Trinity |
| UCLA | 76 | UCLA_1, UCLA_2 |
| UM (University of Michigan) | 117 | UM_1, UM_2 |
| USM | 53 | USM |
| Yale | 64 | Yale |
  - optional site-stratified CV fold regeneration (`--site-stratified-cv`)

## Feature Artifacts

### Temporal features

- File: `data/metadata/node_attributes_temporal.csv`
- Producer: `src/features/extract_temporal.py`
- Column pattern: `roi{1..170}_{feature_name}`
- Frequency behavior is Nyquist-aware via `ACTIVE_FREQ_BANDS` and unreliable-band zeroing

### Spatial features

- File: `data/metadata/node_features_3d.csv`
- Producer: `src/features/extract_spatial.py` (YOLO path) or `src/features/extract_spatial_atlas.py` (atlas fallback path)
- Lobe-level features per region are `x`, `y`, `z_depth`, `size`
- `conf_std` and `detection_count` are not consumed by the model feature tensor

### Harmonized outputs

- Main harmonized temporal export: `data/metadata/node_attributes_harmonized.csv`
- Per-fold harmonization exports: `data/metadata/harmonized_folds_cv/harmonized_fold_<k>.csv`
- Harmonized spatial export: `data/metadata/node_features_3d_harmonized.csv`
- Producer: `src/features/fold_safe_harmonization.py`

## Feature Schema And Ordering

The model channel layout is generated from `FEATURE_GROUPS` in `src/core/feature_registry.py`:

1. temporal
2. frequency
3. internal
4. spatial

`ALL_FEATURE_NAMES` defines ordering and `GNN_IN_CHANNELS` is computed from that ordering.

Important implications:

- feature count is dynamic when frequency-band configuration changes
- any feature engineering update must preserve ordering consistency across extraction, dataset assembly, and model loading

## Graph Construction Contract

Producer: `src/features/construct_causal.py`

Per-subject graph package includes at least:

- adjacency (`adj`)
- internal lobe features (`internal_features`)
- zero-lobe mask (`zero_lobe_mask`)
- confidence and p-value matrices
- selected lag metadata
- sparsification diagnostics

Graph quality assumptions enforced downstream:

- non-empty graph edge set
- valid finite node and edge features

## Dataset Assembly Contract

`ABIDECausalDataset` (`src/features/graph_factory.py`) joins:

- harmonized temporal features
- graph-internal features
- spatial features (harmonized spatial CSV preferred when present)

to construct `torch_geometric.data.Data` objects with:

- `x`: `(NUM_LOBES, GNN_IN_CHANNELS)`
- `edge_index`: `(2, E)`
- `edge_attr`: `(E, 1)`
- covariates (`site_id`, `age`, `sex`, `fiq`)

## Data Quality Gates

Key gates present in source:

- post-download integrity checks for PNG/NPY validity (`src/validation/pipeline_checks.py`)
- subject drop-rate guard and graph validity checks in `ABIDECausalDataset`
- minimum edge and dead-lobe repair logic in graph construction
- fold-safe harmonization with unseen-site policy controls

These gates are designed to fail early or emit explicit warnings when data quality is insufficient.

## Rebuild Sequence

To rebuild data artifacts stage-by-stage:

```bash
python -m src.data.abide_download
python -m src.data.split
python -m src.pipelines.generate_labels
python -m src.features.extract_spatial
python -m src.features.extract_temporal --n-jobs -1
python -m src.features.fold_safe_harmonization
python -m src.features.construct_causal --n-jobs -1
```

Or run the orchestrator:

```bash
python src/run_pipeline.py --auto
```

## Reproducibility

**Seed Policy:**
- Global seed is set to 42 everywhere (`GNN_SEED`, `torch.manual_seed`, `numpy.random.seed`, `random.seed`)
- Reproducibility is enforced in training, evaluation, and data split generation

**Artifact Versioning:**
- Each pipeline run produces dated artifacts in `results/experiments/`
- Config hash is tracked via `src/validation/config_snapshot.py`
- Run ID is embedded in output filenames

**Steps to Reproduce:**

From scratch:
```bash
python src/run_pipeline.py --auto --force-reset
```

From intermediates (reuse existing data):
```bash
python src/run_pipeline.py --auto --skip-download --skip-split
```

**Known Reproducibility Risks:**
- Non-deterministic GPU operations (solved by `torch.use_deterministic_algorithms` where possible)
- Site heterogeneity can cause different generalization patterns on test set
- Multiview graph generation can produce different edge structures if random seed is not fixed
