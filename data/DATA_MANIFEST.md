# Neuro-CXG Data Manifest

## ABIDE-I Dataset

### Source
- **URL**: https://fcon_1000.projects.nitrc.org/indi/abide/
- **Access Date**: 2026
- **Version**: ABIDE-I (Preprocessed)

### Subject Counts

| Stage | Count | Description |
|-------|-------|-------------|
| Raw Download | 1112 | Initial ABIDE-I dataset |
| After Phantom Removal | 1088 | Excluded phantoms |
| After Excluded Subjects | 1015 | Post-curation (see `subject_ids_1015.txt`) |
| Training + Validation | 707 | 70% of curated set |
| Training | 566 | 80% of train+val |
| Validation | 141 | 20% of train+val |
| Test Set | 308 | 30% held-out |

### Site Distribution (13 sites)

| Site | N (1015) | % ASD | Notes |
|------|----------|-------|-------|
| NYU | 184 | 58% | Largest site |
| UM_1 | 129 | 64% | |
| UCLA_1 | 102 | 55% | |
| USM | 85 | 71% | |
| YALE | 81 | 65% | |
| PITT | 74 | 57% | |
| MAX_MUN | 73 | 64% | |
| TRINITY | 67 | 62% | |
| KKI | 66 | 66% | |
| OLIN | 60 | 63% | |
| LEUVEN_2 | 53 | 64% | |
| SBL | 53 | 64% | |
| STANFORD | 45 | 64% | |
| CALTECH | 33 | 64% | |

### Excluded Subjects

The curated subject list is maintained in:
- `data/metadata/subject_ids_1015.txt` (1023 lines including header, 1015 subjects)
- `data/metadata/active_subject_ids.txt` (13933 characters)

Exclusion criteria (from `src/core/hyperparams.py`):
- Corrupted/missing NIfTI files
- Failed quality control
- Extreme motion artifacts
- Missing phenotypic data

## Processed Data Files

### Temporal Features
- `node_attributes_temporal.csv` (55.5 MB)
- `node_attributes_temporal.feather` (18.8 MB)
- 1015 subjects × 12 lobes × 20 temporal features

### Harmonized Features
- `node_attributes_harmonized.csv` (4.2 MB)
- `node_attributes_harmonized.feather` (1.8 MB)
- Post-ComBat harmonization with DX_GROUP protected

### Spatial Features
- `node_features_3d.csv` (511 KB)
- `node_features_3d_harmonized.csv` (511 KB)
- 12 lobes × 4 spatial features (x, y, z, mask)

### Master Manifest
- `master_manifest.csv` (49 KB)
- `master_manifest_1015.csv` (51 KB)
- `master_manifest.feather` (38 KB)

### Causal Graphs
- `data/causal_graphs/` (directory)
- 12×12 directed adjacency matrices per subject
- Method: Ridge Granger Causality (hybrid with lagged Pearson)

## Checksums

For reproducibility verification, checksums of key files:

```
# Temporal features
SHA256 node_attributes_temporal.csv: <to be computed>

# Harmonized features
SHA256 node_attributes_harmonized.csv: <to be computed>

# Master manifest
SHA256 master_manifest.csv: <to be computed>
```

## Data Version

- **Manifest Version**: 1015 subjects
- **Features Version**: 24 features (8 temporal + 10 frequency + 2 internal + 4 spatial)
- **Graph Version**: 12-lobe causal graphs (ridge_granger_hybrid method)

## Reproducibility Notes

1. Subject list is version-controlled in `subject_ids_1015.txt`
2. Master manifest includes all phenotypic variables and site IDs
3. Train/val/test splits are stratified by site and diagnosis (2D stratification)
4. Split configuration is reproducible via `--seed 42`

## External Reproducer Instructions

1. Download ABIDE-I from https://fcon_1000.projects.nitrc.org/indi/abide/
2. Run pipeline: `python src/run_pipeline.py --auto --skip-download`
3. Verify subject count matches: `wc -l data/metadata/subject_ids_1015.txt`
4. Verify features: `head -1 data/metadata/node_attributes_temporal.csv`