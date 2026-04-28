# Data

## Important Note: Architecture Under Review (April 28, 2026)

**Current Status**: The 12-lobe architecture documented below is under evaluation. Comparative analysis (April 28, 2026) revealed:

- **12-Lobe (Current)**: YOLO v29 never detects Brainstem → synthetic fallback coordinates → degenerate constant feature
- **11-Lobe (Proposed)**: Excludes Brainstem → 100% region detection → cleaner features → better pre-training metrics (+0.0097 AUC, +0.0126 F1)

**See**: `LOBE_COMPARISON_ANALYSIS.md` (full analysis) and `docs/decisions.md` (DD-018) for architecture rationale and recommendations.

**Impact on This Document**: References to "12 lobes" and feature dimensions (216 channels) assume current 12-lobe architecture. If architecture switches to 11-lobe:
- Feature dimensions change: 216 → 198 (18 temporal × 11 lobes)
- Graph node count changes: 12 → 11 nodes
- Brainstem-related filters and masks are removed

---

## Scope

This document describes the runtime data model used by the current source code:

- where artifacts are written
- how features are encoded
- which quality gates are enforced
- how data moves from download to model-ready graphs

All paths and constants referenced here come from `src/core/paths.py` and `src/core/feature_registry.py` via `src/core/config.py`.

## Source Datasets

- Imaging source: ABIDE I resting-state fMRI from the public FCP-INDI bucket
- Phenotype source: `data/processed/Phenotypic_V1_0b_preprocessed1.csv`
- Atlas source: `data/raw/atlases/AAL3v1.nii`

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
