# Data

This document covers ABIDE dataset curation, anatomical parcellation, feature extraction, and harmonization protocols.

---

## Architecture Status: 12-Lobe Approved (April 28, 2026)

The 12-lobe architecture is approved for publication. Key findings:

- **12-Lobe (Primary)**: YOLO v29 never detects Brainstem → synthetic fallback coordinates
- Counterintuitive: constant Brainstem features act as implicit regularization
- Test AUC: 0.8819 [95% CI: 0.8277–0.9322] — +5.3% vs 11-lobe
- Generalization: CV < Test (+0.064) indicates robust learning

See `docs/decisions.md` (DD-018) for full analysis.

---

## Source Datasets

| Dataset | Source | Format | License |
|---------|--------|--------|---------|
| ABIDE I fMRI | s3://fcp-indi/data/Projects/ABIDE | NIfTI (.nii), Time series (.npy) | Public (CC BY-NC-SA) |
| Phenotype | `data/processed/Phenotypic_V1_0b_preprocessed1.csv` | CSV | Public |
| Atlas | `data/raw/atlases/AAL3v1.nii` | NIfTI (170 ROIs, MNI 3mm) | Public |

---

## Directory-Level Data Flow

### Raw and intermediate pools

- `data/images/` — exported ALFF PNG slices used for detection and labeling
- `data/processed/time_series/` — canonical pre-split time-series pool
- `data/metadata/` — manifests and feature tables

### Split datasets

```
data/final/
├── train/
├── val/
└── test/
```

Each split contains:

- `images/` (`*_z*.png`)
- `labels/` (YOLO txt labels)
- `time_series/` (`*_ts.npy`, `*_roi_labels.npy`)

### Processed graph artifacts

- `data/processed/causal_graphs/` — base subject graphs (`<subject>_graph.pt`)
- `data/processed/causal_graphs_multiview/` — optional multiview packages

---

## Final Cohort Table`

| Metric | Value | Provenance |
|--------|-------|-----------|
| **Total subjects** | 1,015 | `12lobes.txt:1015` |
| **ASD** | 522 (51.4%) | `12lobes.txt:1384-1385` |
| **Control (TYP)** | 493 (48.6%) | `12lobes.txt:1384-1385` |
| **Sites** | 20 | `SITE_ID` column |
| **Train/Val/Test** | 707 / 154 / 154 | Split (70/15/15) |
| **Age avg** | 16.8 years | Pipeline diagnostics |
| **Sex (M/F)** | 860/155 | Pipeline diagnostics |

---

## Exclusion Funnel Table

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

## Curation Pipeline

### Stage 1: Download Validation

**Producer:** `src/validation/pipeline_checks.py` (check_dataset_integrity)

**Validation checks:**
- File exists and readable
- Shape matches expected (7, 192, 144)
- No NaN/Inf in raw array

**Output:** `data/metadata/download_log.csv` with per-subject success/failure

### Stage 2: Atlas Overlap Check

**Producer:** `src/validation/atlas_validator.py`

```
├── Load AAL3 atlas (170 ROIs, standard MNI 3mm space)
├── For each subject's fMRI:
│   └── Count voxels > threshold per ROI
└── Filter: ≥150 of 170 ROIs must have ≥1 voxel
```

**Purpose:** Ensure fMRI registration was successful; subjects with poor alignment fail here.

**Gating logic:**
```python
# Minimum ROI coverage threshold
ATLAS_MIN_ROIS = 150  # of 170 AAL3 ROIs must be detected
```

### Stage 3: Spatial Feature Extraction (Dead Lobe Detection)

**Producer:** `src/features/extract_spatial.py`

```
├── Load downloaded fMRI + YOLO pretrained weights
├── For each subject:
│   └── Run YOLO inference on 7 z-slices
│   └── Extract: (x, y, z_depth, size) per lobe
└── Filter: ≥9 of 12 lobes with detections
```

**Gating:**
```python
SPATIAL_MIN_REQUIRED_REGIONS = 9  # ≥9 lobes with ≥1 YOLO detection
```

**Dead Lobe Detection:**
```python
for lobe_id in range(NUM_LOBES):
    if lobe_features[lobe_id].all() == 0:  # No detections in lobe
        subject_status = "DEAD_LOBE"
        exclude_subject = True
```

⚠️ **Safety sentinel:** Spatial feature channels (`x`, `y`, `z_depth`, `size`) are model-consumed. If all 4 are zero for any lobe, the zero_lobe_mask flag is set to prevent silent failures.

### Stage 4: Temporal Feature Extraction (NaN/Inf Check)

**Producer:** `src/features/extract_temporal.py`

```
├── Load 170-ROI AAL3 time series (T=150 for TR=2s scans)
├── Aggregate to 12 lobes (PCA eigenvariate)
├── Compute 20 temporal features per lobe:
│   ├── 8 basic (mean, std, skew, kurtosis, psd, mssd, range, autocorr)
│   └── 12 frequency (5 bands × 2 + spectral_entropy + phase_std)
└── Filter: No NaN/Inf allowed
```

**Quality Check:**
```python
temporal_features = extract_temporal_all(subjects)

# Flag any NaN/Inf
bad_idx = temporal_features.isna().any(axis=1) | temporal_features.isinf().any(axis=1)
if bad_idx.any():
    logger.warning(f"Found {bad_idx.sum()} subjects with NaN/Inf in temporal features")
```

### Stage 5: Fold-Safe Harmonization (NaN Insertion Check)

**Producer:** `src/features/fold_safe_harmonization.py`

```
├── For each of 5 CV folds:
│   ├── Fit ComBat on train subjects only
│   ├── Apply to val/test subjects
│   └── Per-fold output: harmonized_fold_k.csv
└── Output: NODE_ATTRIBUTES_HARMONIZED.csv (global, no leakage)
```

**NaN Insertion Check:**
```python
# Harmonization can create new NaN if it encounters
# unseen SITE or categorical covariate value
harmonized_df = harmonize_fold(train_features, val_features)
if harmonized_df.isna().any().any():
    logger.warning(f"ComBat introduced NaN; {n} subjects affected")
    # These subjects are effectively excluded from downstream use
```

### Stage 6: Causal Graph Construction (Dead Graph Detection)

**Producer:** `src/features/construct_causal.py`

```
├── For each subject:
│   ├── Load 12-lobe harmonized time series
│   ├── Compute Granger causality (lag 1-5)
│   ├── Build directed adjacency (12×12)
│   └── Adaptive sparsification (keep top 30% edges)
└── Filter: ≥12 edges per graph (minimum connectivity)
```

**Dead Graph Detection:**
```python
num_edges = (adj_matrix != 0).sum()
if num_edges < MIN_EDGES_PER_GRAPH:  # MIN = 12
    logger.warning(f"Subject {sub_id}: weak connectivity ({num_edges} edges)")
    # Graph still used, but flagged as low-confidence
```

---

## Subject ID Format and Site Roster

### ID Convention

Format: `<SITE>_<COHORT>_<SUBJECT>`

Examples:
- `CMU_a_0050642` → CMU site, cohort a, subject ID 0050642
- `NYU_0050952` → NYU site, no cohort suffix
- `Leuven_1_0050682` → Leuven, site 1, subject ID

### Site Roster (~20 sites)

| Site | Count | Abbreviation |
|------|-------|--------------|
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

---

## Feature Artifacts

### Temporal Features

| Property | Value |
|----------|-------|
| File | `data/metadata/node_attributes_temporal.csv` |
| Producer | `src/features/extract_temporal.py` |
| Column pattern | `roi{1..170}_{feature_name}` |
| Schema | 8 basic + 12 frequency features per ROI |

### Spatial Features

| Property | Value |
|----------|-------|
| File | `data/metadata/node_features_3d.csv` |
| Producer | `src/features/extract_spatial.py` (YOLO path) or `src/features/extract_spatial_atlas.py` (atlas fallback) |
| Lobe-level features | `x`, `y`, `z_depth`, `size` |
| Note | `conf_std` and `detection_count` are not consumed by the model |

### Harmonized Outputs

| Property | Value |
|----------|-------|
| Main harmonized temporal | `data/metadata/node_attributes_harmonized.csv` |
| Per-fold harmonized | `data/metadata/harmonized_folds_cv/harmonized_fold_<k>.csv` |
| Harmonized spatial | `data/metadata/node_features_3d_harmonized.csv` |
| Producer | `src/features/fold_safe_harmonization.py` |

---

## Feature Schema and Ordering Rules

The model channel layout is generated from `FEATURE_GROUPS` in `src/core/feature_registry.py`:

1. temporal
2. frequency
3. internal
4. spatial

`ALL_FEATURE_NAMES` defines ordering and `GNN_IN_CHANNELS` is computed from that ordering.

⚠️ **Safety sentinel for spatial channels:** If any lobe has all four spatial features (`x`, `y`, `z_depth`, `size`) as zero, the `zero_lobe_mask` is set to prevent silent failure. This ensures the model doesn't learn from degenerate spatial inputs.

**Important implications:**

- Feature count is dynamic when frequency-band configuration changes (gamma band is zeroed for TR=2s due to Nyquist)
- Any feature engineering update must preserve ordering consistency across extraction, dataset assembly, and model loading

---

## Graph Construction Contract

**Producer:** `src/features/construct_causal.py`

Per-subject graph package (`<subject>_graph.pt`) includes:

| Field | Type | Description |
|-------|------|-------------|
| `adj` | Tensor(12, 12) | Directed adjacency matrix |
| `internal_features` | Tensor(12, 2) | Coherence, spatial variance |
| `zero_lobe_mask` | Tensor(12,) | Dead lobe sentinel |
| `edge_confidence` | Tensor(12, 12) | Causality confidence |
| `edge_pvalues` | Tensor(12, 12) | Granger causality p-values |
| `selected_lag_matrix` | Tensor(12, 12) | Optimal lag per pair |
| `low_confidence_mask` | Tensor(12, 12) | Unreliable edges mask |
| `subject_id` | str | Subject identifier |
| `lobe_order` | List[str] | Ordered lobe names |
| `sparsification_info` | Dict | Sparsification diagnostics |
| `stats` | Dict | Runtime statistics |

Graph quality assumptions enforced downstream:

- Non-empty graph edge set (minimum 12 edges)
- Valid finite node and edge features

---

## Dataset Assembly Contract

`ABIDECausalDataset` (`src/features/graph_factory.py`) joins:

- Harmonized temporal features
- Graph-internal features (`coherence`, `spatial_variance`)
- Spatial features (harmonized CSV preferred when present)

To construct `torch_geometric.data.Data` objects with:

| Property | Shape | Description |
|----------|-------|-------------|
| `x` | (NUM_LOBES, GNN_IN_CHANNELS) | Node features |
| `edge_index` | (2, E) | Edge connectivity |
| `edge_attr` | (E, 1) | Edge weights |
| `y` | scalar | Label (ASD/Control) |
| `site_id` | — | Site for GRL |
| `age`, `sex`, `fiq` | — | Demographic covariates |

---

## Data Quality Gates

| Gate | What It Enforces | Where Enforced |
|------|------------------|-----------------|
| Post-download integrity | PNG/NPY validity, shape, no NaN/Inf | `src/validation/pipeline_checks.py` |
| Atlas overlap | ≥150 of 170 ROIs detected | `src/validation/atlas_validator.py` |
| Spatial feature completeness | ≥9 lobes with detections | `src/features/extract_spatial.py` |
| Dead lobe detection | No entirely missing lobes | `src/features/extract_spatial.py` |
| Temporal feature validity | No NaN/Inf | `src/features/extract_temporal.py` |
| Harmonization NaN check | No ComBat-induced NaN | `src/features/fold_safe_harmonization.py` |
| Graph minimum edges | ≥12 edges per graph | `src/features/construct_causal.py` |
| Subject drop-rate guard | Max acceptable drop rate | `ABIDECausalDataset` |

---

## Rebuild Sequence

Stage-by-stage commands:
```bash
python -m src.data.abide_download
python -m src.data.split
python -m src.detection.generate_labels
python -m src.features.extract_spatial
python -m src.features.extract_temporal --n-jobs -1
python -m src.features.fold_safe_harmonization
python -m src.features.construct_causal --n-jobs -1
```

Or via orchestrator:
```bash
python src/run_pipeline.py --auto
```

---

## Reproducibility

### Seed Policy

- Global seed is set to 42 for GNN training (`GNN_SEED`, `torch.manual_seed`). YOLO and evaluation use independent fixed seeds.
- Reproducibility is enforced in training, evaluation, and data split generation

### Artifact Versioning

- Each pipeline run produces dated artifacts in `results/experiments/`
- Config hash is tracked via `src/validation/config_snapshot.py`
- Run ID is embedded in output filenames

### Reproduce Commands

**From scratch:**
```bash
python src/run_pipeline.py --auto --force-reset
```

**From intermediates (reuse existing data):**
```bash
python src/run_pipeline.py --auto --skip-download --skip-split
```

---

## ⚠️ Known Reproducibility Risks

1. **GPU non-deterministic operations:** Some GPU kernels are non-deterministic. This is partially solved by `torch.use_deterministic_algorithms` where possible, but some operations (e.g., pooling, scatter) may still introduce variance.

2. **Site heterogeneity:** Multi-site data introduces scanner/protocol variance that can cause different generalization patterns on test set. The GRL branch (`GNN_GRL_ALPHA = 0.10`) mitigates but does not eliminate this.

3. **Multiview graph generation:** Graph construction using different random seeds can produce different edge structures. Always fix seeds when comparing multiview experiments.