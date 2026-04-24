# Configuration

## Purpose

Neuro-CXG is configuration-driven. Runtime behavior should be changed through `src/core/*` config modules, not by hardcoding values inside stage scripts.

## Configuration Topology

`src/core/config.py` is the compatibility facade. It re-exports:

- `src/core/paths.py`
- `src/core/feature_registry.py`
- `src/core/hyperparams.py`
- `src/core/atlas_config.py`
- selected validators from `src/core/validators.py`

Recommended import pattern:

```python
from src.core.config import CHECKPOINT_DIR, GNN_IN_CHANNELS, EVAL_THRESHOLD_POLICY
```

## 1) Path Configuration (`src/core/paths.py`)

Key roots:

- `PROJECT_ROOT`
- `DATA_ROOT`
- `MODEL_ROOT`
- `RESULTS_DIR`

High-impact file paths:

- `MASTER_MANIFEST` -> `data/metadata/master_manifest.csv`
- `NODE_ATTRIBUTES_HARMONIZED` -> `data/metadata/node_attributes_harmonized.csv`
- `HARMONIZED_FOLDS_DIR` -> `data/metadata/harmonized_folds_cv/`
- `NODE_FEATURES_3D` -> `data/metadata/node_features_3d.csv`
- `NODE_FEATURES_3D_HARMONIZED` -> `data/metadata/node_features_3d_harmonized.csv`
- `CAUSAL_GRAPHS_DIR` -> `data/processed/causal_graphs/`
- `CAUSAL_GRAPHS_MULTIVIEW_DIR` -> `data/processed/causal_graphs_multiview/`
- `CHECKPOINT_DIR` -> `models/checkpoints/`
- `RESULTS_EVALUATION_DIR` -> `results/evaluation/`

## 2) Feature Registry (`src/core/feature_registry.py`)

Feature channel layout is derived from registry definitions.

Current groups:

- temporal base: 8 channels
- frequency: dynamic from `ACTIVE_FREQ_BANDS` + `spectral_entropy` + `phase_std`
- internal: 2 channels (`coherence`, `spatial_variance`)
- spatial: 4 channels (`x`, `y`, `z_depth`, `size`)

Important behavior:

- gamma band is excluded by default via:
  - `UNRELIABLE_FREQ_BANDS_AT_NYQUIST = ("gamma",)`
  - `EXCLUDE_NYQUIST_BANDS = True`
- therefore `GNN_IN_CHANNELS` is computed dynamically from `ALL_FEATURE_NAMES`.

Safety sentinel:

- `NUM_SPATIAL_FEATURES == 4` is enforced by assertion to prevent reintroducing site-leaky channels.

## 3) Atlas Mapping (`src/core/atlas_config.py`)

Defines:

- `LOBE_MAPPING`: maps 170 AAL ROIs into 12 lobe nodes.
- `LOBE_NAMES`: lobe labels used across extraction, graphs, and plots.
- Optional hierarchy:
  - `LOBE_TO_NETWORK`
  - `NETWORK_TO_LOBES`
  - `NUM_NETWORKS = 4`

This hierarchy is used when `GNN_POOLING = "anatomical"`.

## 4) Hyperparameters (`src/core/hyperparams.py`)

### Causal Graph Defaults

- `CAUSALITY_METHOD = "lagged_pearson"`  # Changed from ridge_granger
- `GRANGER_MAX_LAG = 5`
- `GRANGER_MAX_LAG_SECONDS = 10.0`  # Max lag in seconds
- `GRANGER_USE_GPU = True`
- `SPARSITY_METHOD = "topk_per_node"`
- `SPARSITY_TOPK_PER_NODE = 3`
- `MIN_EDGES_PER_GRAPH = 12`

### GNN Defaults (Optimized for Publication)

- `GNN_HIDDEN_CHANNELS = 32`  # Reduced from 64
- `GNN_NUM_HEADS = 2`
- `GNN_NUM_LAYERS = 2`
- `GNN_DROPOUT = 0.35`
- `GNN_POOLING = "anatomical"`  # Changed from mean_max_sum
- `GNN_WEIGHT_DECAY = 5e-4`  # Increased from 5e-5
- `GNN_ONECYCLE_MAX_LR = 0.001`  # Reduced from 0.002
- `GNN_ONECYCLE_WARMUP_FRACTION = 0.05`  # Reduced from 0.15
- `GNN_BATCH_SIZE = 32`
- `GNN_EPOCHS = 100`
- `K_FOLDS = 5`

### Site Bias Controls (Optimized)

- `GNN_USE_SITE_EMBEDDING = True`  # Enabled for better performance
- `GNN_USE_DEMOGRAPHICS = True`  # Enabled for more context
- `GNN_USE_GRL = True`
- `GNN_GRL_ALPHA = 0.10`  # CRITICAL: Do NOT increase to 1.0 - test AUC drops from 0.85 to 0.83
- `GNN_GRL_ALPHA_MAX = 1.0`
- `GNN_SITE_LOSS_WEIGHT = 0.15`

### Auxiliary Objective Controls

- `GNN_STRUCTURAL_DROPOUT_PROB = 0.0`  # Disabled - counterproductive without contrastive
- `GNN_EDGE_CONTRASTIVE_WEIGHT = 0.0`
- `GNN_INVARIANCE_WEIGHT = 0.0`
- `GNN_SPATIAL_INVARIANCE_WEIGHT = 0.0`

### Loss Configuration

- `USE_FOCAL_LOSS = True`  # Enabled for hard-example mining
- `USE_CLASS_WEIGHTS = False`  # Disabled - classes are near-balanced

### Threshold Policy

- `EVAL_THRESHOLD_POLICY = "youden"`  # Changed from fixed
- `EVAL_FIXED_THRESHOLD = 0.5263`  # Kept for backward compatibility

## 5) Quality Gates and Policies

Graph and training quality:

- `GNN_ENFORCE_GRAPH_QUALITY_GATE = True`
- `GNN_MAX_DEGENERATE_GRAPH_RATE = 0.35`
- `GNN_MIN_EDGES_FOR_NONDEGENERATE = 12`

Multiview quality:

- `GNN_ENFORCE_MULTIVIEW_QUALITY_GATE = True`
- `GNN_MULTIVIEW_MAX_ZERO_EDGE_RATE = 0.20`
- `GNN_MULTIVIEW_QUALITY_SAMPLE_SIZE = 512`

Generation-time multiview policy:

- `MULTIVIEW_GENERATION_ENFORCE_QUALITY_GATE = True`
- `MULTIVIEW_GENERATION_MAX_ZERO_EDGE_RATE = 0.20`
- `MULTIVIEW_GENERATION_POLICY = "fail"`

Harmonization unseen-site policy:

- `HARMONIZATION_UNSEEN_SITE_POLICY = "passthrough"`

## 6) Medical Integrity Constraints

YOLO augmentation controls are intentionally conservative:

- `YOLO_FLIPLR = 0.0`
- `YOLO_DEGREES = 0.0`
- `YOLO_MOSAIC = 0.0`

These settings preserve anatomical left/right and spatial consistency.

## 7) Validation Helpers

`src/core/validators.py` provides runtime checks used by the runner and training:

- `validate_environment()`
- `validate_graph_construction_inputs()`
- `validate_gnn_training_inputs()`
- `get_active_checkpoint_dir()`
- graph degeneracy summarizers

## 8) Safe Change Workflow

When updating config values:

1. Change constants in the owning module (not ad-hoc in scripts).
2. Run dry plan:

```bash
python src/run_pipeline.py --dry-run
```

3. Re-run affected stages only.
4. Re-check downstream contracts:
   - feature dimensions
   - sentinel outputs
   - threshold metadata alignment across `run_evaluation.py` and `run_result_analysis.py`

## 9) Common Misconfiguration Patterns

- Updating feature dimensions without updating registry-driven channel expectations.
- Changing output paths without updating stage sentinels in `src/pipeline/registry.py`.
- Mixing artifacts from different runs in a shared output directory.
- Switching threshold policy without regenerating evaluation artifacts.
