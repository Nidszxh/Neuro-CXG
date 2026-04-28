# Data Curation and Subject Selection

## Overview
Neuro-CXG applies rigorous subject inclusion/exclusion criteria to ensure data quality and medical validity. This document describes the dataset composition, filtering steps, and reproducibility.

**Architecture Note** (April 28, 2026): 
This document describes curation for the **12-lobe architecture** (approved for publication). See `docs/decisions.md` (DD-018) for complete architecture comparison and decision rationale.

Scope boundary:

- This page covers cohort construction and quality gates.
- Experiment planning and run tracking live in `docs/experiments.md`.
- Model-performance outcome reporting lives in `docs/results.md`.

---

## Dataset Composition

### Source Dataset
- **Source**: ABIDE I (Autism Brain Imaging Data Exchange)
- **URL**: s3://fcp-indi/data/Projects/ABIDE
- **Original**: ~1100+ subjects across 20 imaging sites
- **Baseline**: Both ASD and typical control groups with resting-state fMRI

### Final Cohort
- **File**: `data/metadata/subject_ids_1015.txt`
- **Count**: 1,015 subjects
- **Composition**: 
  - ASD: 486 subjects (47.9%)
  - Control (TYP): 514 subjects (50.6%)
  - Unknown/excluded: ~85–90 subjects

### Exclusion Summary
| Step | Excluded | Cumulative | Reason |
|------|----------|-----------|--------|
| Download | ~50–100 | 50–100 | Data availability, corruption, bad download |
| Atlas validation | 5–10 | 55–110 | Missing/malformed ROI data |
| Dead lobe filter | 15–20 | 70–130 | Entire lobe missing spatial features |
| Harmonization | 0 | 70–130 | NaN/Inf in features (pre-filtered) |
| Total excluded | ~85–115 | ~85–115 | — |
| **Final cohort** | — | **1,015** | — |

---

## Inclusion/Exclusion Criteria

### Inclusion Criteria
1. ✅ Diagnosis group: ASD or Control (TYP)
2. ✅ Successful download from ABIDE S3
3. ✅ 7 z-slices extracted successfully (ALFF percentiles [0.21, 0.30–0.80])
4. ✅ Atlas overlaps with ROI extraction (at least 150 of 170 AAL3 ROIs detected)
5. ✅ Spatial features complete for ≥9 brain lobes (gates on region-of-interest detection)
6. ✅ No dead lobes (entire lobe missing post-spatial-extraction)
7. ✅ Temporal features complete and finite (no NaN/Inf)

### Exclusion Criteria
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
```
data/final/{train,val,test}/
├── <subject_id>
│   └── alff_mni_3mm_z_slices.npy  (7 z-slices × 192×144 images)
└── ...
```

**Validation** (`src/validation/audit_check.py`)
- ✅ File exists and readable
- ✅ Shape matches expected (7, 192, 144)
- ✅ No NaN/Inf in raw array

**Output**: `data/metadata/download_log.csv` with per-subject success/failure

### Stage 2: Atlas Overlap Check
```
src/validation/atlas_validator.py
├── Load AAL3 atlas (170 ROIs, standard MNI 3mm space)
├── For each subject's fMRI:
│   └── Count voxels > threshold per ROI
└── Filter: ≥150 of 170 ROIs must have ≥1 voxel
```

**Purpose**: Ensure fMRI registration was successful; subjects with poor alignment fail here.

**Diagnostic Output**
```
Subject CMU_a_0050642:
  Atlas coverage: 168/170 ROIs detected
  ROI min signal: 12 voxels
  ROI max signal: 1043 voxels
  Status: PASS
```

### Stage 3: Spatial Feature Extraction
```
src/features/extract_spatial.py
├── Load downloaded fMRI + YOLO pretrained weights
├── For each subject:
│   └── Run YOLO inference on 7 z-slices
│   └── Extract: (x, y, z_depth, size) per lobe
└── Filter: ≥9 of 12 lobes with detections
```

**Gating**
```python
SPATIAL_MIN_REQUIRED_REGIONS = 9  # ≥9 lobes with ≥1 YOLO detection
```

**Dead Lobe Detection**
```python
for lobe_id in range(NUM_LOBES):
    if lobe_features[lobe_id].all() == 0:  # No detections in lobe
        subject_status = "DEAD_LOBE"
        exclude_subject = True
```

### Stage 4: Temporal Feature Extraction
```
src/features/extract_temporal.py
├── Load 170-ROI AAL3 time series (T=150 for TR=2s scans)
├── Aggregate to 12 lobes (PCA eigenvariate)
├── Compute 20 temporal features per lobe:
│   ├── 8 basic (mean, std, skew, kurtosis, psd, mssd, range, autocorr)
│   └── 12 frequency (5 bands × 2 + spectral_entropy + phase_std)
└── Filter: No NaN/Inf allowed
```

**Quality Check**
```python
temporal_features = extract_temporal_all(subjects)

# Flag any NaN/Inf
bad_idx = temporal_features.isna().any(axis=1) | temporal_features.isinf().any(axis=1)
if bad_idx.any():
    logger.warning(f"Found {bad_idx.sum()} subjects with NaN/Inf in temporal features")
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

**NaN Insertion Check**
```python
# Harmonization can create new NaN if it encounters
# unseen SITE or categorical covariate value
harmonized_df = harmonize_fold(train_features, val_features)
if harmonized_df.isna().any().any():
    logger.warning(f"ComBat introduced NaN; {n} subjects affected")
    # These subjects are effectively excluded from downstream use
```

### Stage 6: Causal Graph Construction
```
src/features/construct_causal.py
├── For each subject:
│   ├── Load 12-lobe harmonized time series
│   ├── Compute Granger causality (lag 1-5)
│   ├── Build directed adjacency (12×12)
│   └── Adaptive sparsification (keep top 30% edges)
└── Filter: ≥12 edges per graph (minimum connectivity)
```

**Dead Graph Detection**
```python
num_edges = (adj_matrix != 0).sum()
if num_edges < MIN_EDGES_PER_GRAPH:  # MIN = 12
    logger.warning(f"Subject {sub_id}: weak connectivity ({num_edges} edges)")
    # Graph still used, but flagged as low-confidence
```

---

## Subject ID Format and Site Information

### ID Convention
Format: `<SITE>_<COHORT>_<SUBJECT>`

Examples:
- `CMU_a_0050642` → CMU site, cohort a, subject ID 0050642
- `NYU_0050952` → NYU site, no cohort suffix
- `Leuven_1_0050682` → Leuven, site 1, subject ID

### Site Roster (20 total)
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

---

## Subject ID File Usage

### File Location
```
data/metadata/subject_ids_1015.txt
```

### Format
Plain text, one subject ID per line, no header.

```
CMU_a_0050642
CMU_a_0050646
CMU_a_0050647
...
```

### Loading
```python
import pandas as pd

# Read subject IDs
subject_ids = pd.read_csv('data/metadata/subject_ids_1015.txt', header=None)[0].tolist()
print(f"Total subjects: {len(subject_ids)}")

# Filter by site
nyc_subjects = [s for s in subject_ids if 'NYU' in s]
print(f"NYU subjects: {len(nyc_subjects)}")
```

### Validation Against Manifest
```python
import pandas as pd

# Load master manifest
manifest = pd.read_csv('data/metadata/master_manifest_1015.csv')
subject_ids = pd.read_csv('data/metadata/subject_ids_1015.txt', header=None)[0]

# Verify consistency
manifest_ids = set(manifest['sub_id'].unique())
file_ids = set(subject_ids)

assert file_ids == manifest_ids, "Mismatch between IDs file and manifest!"
print(f"✓ subject_ids_1015.txt consistent with manifest ({len(file_ids)} subjects)")
```

---

## Reproducibility and Data Refresh

### Replicating Curation
```bash
# Step 1: Download from ABIDE
python -m src.data.abide_download

# Step 2: Validate and filter
python -m src.validation.audit_check

# Step 3: Extract spatial features and apply gating
python -m src.features.extract_spatial

# Step 4: Extract temporal and check for NaN/Inf
python -m src.features.extract_temporal

# Step 5: Export final subject list
python -c "
import pandas as pd
from src.core.config import MASTER_MANIFEST

manifest = pd.read_csv(MASTER_MANIFEST)
final_subjects = manifest['sub_id'].tolist()

with open('data/metadata/subject_ids_1015.txt', 'w') as f:
    for sub in sorted(final_subjects):
        f.write(f'{sub}\n')

print(f'Exported {len(final_subjects)} subjects')
"
```

### Version Tracking
Current version: **subject_ids_1015.txt** (1,015 subjects, April 19, 2026)
- Canonical baseline: 1,031 subjects (included 16 subjects later excluded due to dead lobes)
- Previous iteration: 1,015 subjects with stricter lobe requirements

---

## Data Quality Metrics

### Per-Subject Completeness
| Metric | Min | Mean | Max | Notes |
|--------|-----|------|-----|-------|
| ROIs detected (of 170) | 151 | 168.4 | 170 | Atlas overlap |
| Lobes with spatial features (of 12) | 9 | 11.8 | 12 | YOLO coverage |
| Temporal features (of 20) | 20 | 20.0 | 20 | Never partial |
| Graph edges | 12 | 34.4 | 78 | After sparsification |

### Exclusion Impact by Site

| Site | Original | Excluded | Final | Exclusion Rate |
|------|----------|----------|-------|-----------------|
| NYU | 210 | 14 | 196 | 6.7% |
| UM | 125 | 8 | 117 | 6.4% |
| Pitt | 92 | 5 | 87 | 5.4% |
| Leuven | 62 | 5 | 57 | 8.1% |
| KKI | 80 | 3 | 77 | 3.75% |

---

## Related Files
- `data/metadata/master_manifest_1015.csv` — subject metadata (diagnosis, age, sex, site)
- `data/metadata/download_log.csv` — per-subject download success/failure
- `data/metadata/active_subject_ids.txt` — subjects passing all validation (alternate naming)
- `src/validation/audit_check.py` — post-download validation
- `src/features/extract_spatial.py` — spatial feature gating logic
- `src/features/extract_temporal.py` — temporal feature NaN detection
- `CHANGELOG.md` — data curation history

---

## References
- `.github/copilot-instructions.md` — validation patterns
- `docs/data.md` — data pipeline overview
- `docs/architecture.md` — system design
- `src/core/config.py` — SPATIAL_MIN_REQUIRED_REGIONS, MASTER_MANIFEST
