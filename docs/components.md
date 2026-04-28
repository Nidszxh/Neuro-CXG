# Components

## Purpose

This page documents the major runtime components in Neuro-CXG and the contract each component exposes to downstream stages.

## System Composition

| Layer | Component | Primary Files |
|---|---|---|
| Orchestration | Stage registry + runner | `src/pipeline/registry.py`, `src/run_pipeline.py` |
| Configuration | Modular constants + validators | `src/core/config.py`, `src/core/*.py` |
| Data ingestion | ABIDE download and split | `src/data/abide_download.py`, `src/data/split.py` |
| Feature build | Spatial/temporal extraction + harmonization | `src/features/extract_spatial.py`, `src/features/extract_temporal.py`, `src/features/fold_safe_harmonization.py` |
| Graph build | Causal adjacency construction | `src/features/construct_causal.py`, `src/features/causal_inference.py` |
| Dataset assembly | PyG graph objects | `src/features/graph_factory.py` |
| Model and training | GNN architecture + CV loop | `src/models/causal_gnn.py`, `src/models/gnn_model.py`, `src/models/training_utils.py` |
| Reporting | Evaluation/explainability/result analysis | `src/run_evaluation.py`, `src/run_explainability.py`, `src/run_result_analysis.py` |
| Validation | Integrity and quality checks | `src/validation/*.py`, `src/core/validators.py` |

## 1) Orchestration Component

### Stage Registry (`src/pipeline/registry.py`)

- Defines immutable stage metadata:
  - stage key
  - module path
  - optional function name
  - dependencies
  - output sentinel
- `Stage.is_complete()` uses sentinel existence and non-empty directory checks.

### Runner (`src/run_pipeline.py`)

- Builds execution plan from:
  - registry order
  - CLI flags
  - discovered artifact readiness
- Executes stage modules using `python -m <module>` or module function invocation.
- Supports interactive and non-interactive (`--auto`) operation.
- Enforces preflight checks (`validate_environment`) and pre-training checks (`validate_gnn_training_inputs`).

## 2) Configuration Component

### Facade (`src/core/config.py`)

- Re-exports config modules to keep a stable import surface.

### Core Config Modules

- `src/core/paths.py`: canonical paths for data, models, results.
- `src/core/feature_registry.py`: feature groups, active frequency bands, channel count (`GNN_IN_CHANNELS`).
- `src/core/hyperparams.py`: causality, sparsity, GNN, threshold policy, quality gates.
- `src/core/atlas_config.py`: 170 ROI to 12 lobe mapping and optional network hierarchy.
- `src/core/validators.py`: environment, training-input, and degeneracy checks.

## 3) Data Ingestion Component

### Download (`src/data/abide_download.py`)

- Pulls ABIDE artifacts and produces split-ready image/time-series assets.

### Split (`src/data/split.py`)

- Default path:
  - 70/15/15 train/val/test split with stratification.
  - 5-fold `cv_fold` generation for train rows.
- Optional stricter CV path:
  - `--site-stratified-cv` rewrites `cv_fold` with GroupKFold over site clusters.

Output contract:
- `data/metadata/master_manifest.csv` must contain split and fold assignments used by harmonization and training.

## 4) Feature Component

### Spatial (`src/features/extract_spatial.py` and `src/features/extract_spatial_atlas.py`)

- Produces lobe-level geometric features in metadata tables.
- Model-consumed spatial channels are restricted to:
  - `x`, `y`, `z_depth`, `size`

### Temporal (`src/features/extract_temporal.py`)

- Computes lobe-level temporal/frequency features with site-aware TR handling.

### Harmonization (`src/features/fold_safe_harmonization.py`)

- Train-fold fit, val/test transform behavior for fold safety.
- Uses `DX_GROUP` as protected covariate.
- Writes:
  - combined harmonized file
  - per-fold harmonized files consumed by training

## 5) Graph Construction Component

### Graph Builder (`src/features/construct_causal.py`)

- Aggregates ROI signals to **12 lobe-level time series** (current; 11 under evaluation).
- Builds directed adjacency matrices with configured causal method.
- Applies sparsification and minimum-edge constraints.
- Saves per-subject graph packages under `data/processed/causal_graphs/`.

**Note** (April 28, 2026): The 12-lobe architecture is under evaluation. See `docs/decisions.md` (DD-018) and `LOBE_COMPARISON_ANALYSIS.md` for:
- Current 12-lobe issue: Brainstem never detected by YOLO → synthetic fallback
- 11-lobe alternative: Brainstem excluded → cleaner features → better pre-training metrics

Users can test 11-lobe via `--11-lobes` CLI flag.

### Causal Inference Core (`src/features/causal_inference.py`)

- Exposes `compute_granger_causality` with CPU/GPU internal pathing.
- Includes robust fallback behavior (for example NaN/Inf or short series returns safe zero matrices).

## 6) Dataset Assembly Component

### `ABIDECausalDataset` (`src/features/graph_factory.py`)

Input sources:
- manifest rows
- harmonized temporal feature table
- spatial table (harmonized preferred when present)
- per-subject graph package

Output object:
- `torch_geometric.data.Data` with:
  - `x` (node features)
  - `edge_index`, `edge_attr`
  - `y`
  - `site_id`, demographics, `sub_id`, `zero_lobe_mask`

Guards:
- skips invalid or degenerate graph subjects
- checks shape and NaN/Inf consistency
- enforces subject-drop quality gate before training proceeds

## 7) Model and Training Component

### Model (`src/models/causal_gnn.py`)

- GATv2 backbone with configurable depth/heads.
- Supports pooling modes:
  - `mean_max_sum`
  - `attention`
  - `anatomical` (hierarchical network pooling)
- Optional GRL site-adversarial branch.
- Optional site and demographic conditioning.

### Training (`src/models/gnn_model.py`)

- Strict 5-fold training using manifest `cv_fold`.
- Per-fold harmonized file requirement (`harmonized_fold_<k>.csv`).
- Quality gates:
  - base graph degeneracy rate
  - optional multiview quality gate
- Optional auxiliary objectives controlled by config:
  - structural dropout
  - edge contrastive
  - causal invariance
  - spatial invariance

### Training Utilities (`src/models/training_utils.py`)

- OneCycle training loop and checkpoint handling.
- Feature-scaler attachment and checkpoint metadata management.
- Shared threshold/metric helpers used across train/eval paths.

## 8) Reporting Components

### Evaluation (`src/run_evaluation.py`)

- Fold-ensemble scoring on test graphs.
- Bootstrap CI, permutation testing, subgroup reports, baseline comparisons.
- Writes machine-readable summary JSON/CSV in `results/evaluation/`.

### Explainability (`src/run_explainability.py`)

- Node, edge, feature, and literature phases.
- Optional phase subset and masking skip controls.
- Writes outputs under `results/explainability/` with a final `summary.json`.

### Result Analysis (`src/run_result_analysis.py`)

- Per-subject predictions and error profiling.
- Site effects and calibration diagnostics.
- Case-study artifacts and final `result_analysis_summary.json`.

## 9) Validation Components

- `src/core/validators.py`: reusable validation functions used by orchestrator and training.
- `src/validation/pipeline_checks.py`: pipeline-level integrity and quality checks.
- `src/validation/audit_check.py`: strict artifact validation pass.
- `src/validation/dev_audit.py`: developer-facing consistency checks.

## Component Change Rules

When updating a component, keep these invariants synchronized:

- If stage behavior changes, update both `src/pipeline/registry.py` and runner logic in `src/run_pipeline.py`.
- If feature channels change, update `src/core/feature_registry.py` and validate `ABIDECausalDataset` shape checks.
- If training prerequisites change, update `validate_gnn_training_inputs()` in `src/core/validators.py`.
- If output artifact names change, update sentinels in `src/pipeline/registry.py` and docs under `docs/`.
