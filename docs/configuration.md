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

## 1) Path Configuration

Key roots defined in `src/core/paths.py`:

- `PROJECT_ROOT`
- `DATA_ROOT`
- `MODEL_ROOT`
- `RESULTS_DIR`

High-impact file paths:

| Constant | Path |
|----------|------|
| `MASTER_MANIFEST` | `data/metadata/master_manifest.csv` |
| `NODE_ATTRIBUTES_HARMONIZED` | `data/metadata/node_attributes_harmonized.csv` |
| `HARMONIZED_FOLDS_DIR` | `data/metadata/harmonized_folds_cv/` |
| `NODE_FEATURES_3D` | `data/metadata/node_features_3d.csv` |
| `NODE_FEATURES_3D_HARMONIZED` | `data/metadata/node_features_3d_harmonized.csv` |
| `CAUSAL_GRAPHS_DIR` | `data/processed/causal_graphs/` |
| `CAUSAL_GRAPHS_MULTIVIEW_DIR` | `data/processed/causal_graphs_multiview/` |
| `CHECKPOINT_DIR` | `models/checkpoints/` |
| `RESULTS_EVALUATION_DIR` | `results/evaluation/` |

## 2) Feature Registry

Feature channel layout is derived from `src/core/feature_registry.py`.

**Current groups:**

| Group | Channels |
|-------|----------|
| temporal | 8 (base time-domain features) |
| frequency | 10 (from ACTIVE_FREQ_BANDS + spectral_entropy + phase_std) |
| internal | 2 (coherence, spatial_variance) |
| spatial | 4 (x, y, z_depth, size) |

**Important behavior:**

- Gamma band is excluded by default:
  - `UNRELIABLE_FREQ_BANDS_AT_NYQUIST = ("gamma",)`
  - `EXCLUDE_NYQUIST_BANDS = True`
- `GNN_IN_CHANNELS` is computed dynamically from `ALL_FEATURE_NAMES` (24 total)

**Safety sentinel:**

- ⚠️ `NUM_SPATIAL_FEATURES == 4` is enforced by assertion to prevent reintroducing site-leaky channels (`conf_std`, `detection_count`).

## 3) Atlas Mapping

Defined in `src/core/atlas_config.py`:

- `LOBE_MAPPING`: maps 170 AAL ROIs into **12 lobe nodes**
- `LOBE_NAMES`: lobe labels used across extraction, graphs, and plots
- Optional hierarchy:
  - `LOBE_TO_NETWORK`
  - `NETWORK_TO_LOBES`
  - `NUM_NETWORKS = 4`

This hierarchy is used when `GNN_POOLING = "anatomical"`.

**Architecture Note (May 2026) — APPROVED:**

- **12-Lobe is approved for publication**
- YOLO v29 never detects Brainstem → constant synthetic features act as implicit regularization
- Test AUC: 0.8648 (ridge_granger_hybrid method)
- See `docs/decisions.md` (DD-018) for full analysis
- **Method:** ridge_granger_hybrid (70% Ridge Granger + 30% Lagged Pearson) — best Granger-based performance

## 4) Hyperparameters

### Causal Graph Defaults

| Parameter | Value | Notes |
|-----------|-------|-------|
| `CAUSALITY_METHOD` | `"ridge_granger_hybrid"` | 70% Ridge Granger + 30% Lagged Pearson |
| `RIDGE_GRANGER_HYBRID_BETA` | 0.70 | Weight for Granger component in hybrid |
| `GRANGER_MAX_LAG` | 5 | Number of lags |
| `GRANGER_MAX_LAG_SECONDS` | 10.0 | Max lag in seconds |
| `GRANGER_USE_GPU` | True | GPU acceleration |
| `RIDGE_GRANGER_LAGS` | (1,2,3,4,5) | VAR lag order |
| `RIDGE_GRANGER_LAMBDA` | 1.0 | Ridge regularization |
| `SPARSITY_METHOD` | `"topk_per_node"` | Sparsification method |
| `SPARSITY_TOPK_PER_NODE` | 3 | Edges per node |
| `MIN_EDGES_PER_GRAPH` | 12 | Minimum edges |

### GNN Defaults (Optimized)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `GNN_HIDDEN_CHANNELS` | 32 | Reduced from 64 |
| `GNN_NUM_HEADS` | 2 | Attention heads |
| `GNN_NUM_LAYERS` | 2 | GAT layers |
| `GNN_DROPOUT` | 0.35 | Dropout rate |
| `GNN_POOLING` | `"anatomical"` | Hierarchical pooling |
| `GNN_WEIGHT_DECAY` | 5e-4 | L2 regularization |
| `GNN_ONECYCLE_MAX_LR` | 0.001 | Max learning rate |
| `GNN_ONECYCLE_WARMUP_FRACTION` | 0.05 | Warmup fraction |
| `GNN_BATCH_SIZE` | 32 | Batch size |
| `GNN_EPOCHS` | 100 | Max epochs |
| `K_FOLDS` | 5 | CV folds |

### Site Bias Controls (Optimized)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `GNN_USE_SITE_EMBEDDING` | True | Site conditioning |
| `GNN_USE_DEMOGRAPHICS` | True | Demographic conditioning |
| `GNN_USE_GRL` | True | Gradient reversal |
| `GNN_GRL_ALPHA` | **0.10** | ⚠️ CRITICAL: Do NOT increase to 1.0 - test AUC drops from 0.87 to 0.83 |
| `GNN_GRL_ALPHA_MAX` | 1.0 | Alpha max for annealing |
| `GNN_SITE_LOSS_WEIGHT` | 0.15 | Site loss weight |

### Auxiliary Objective Controls

| Parameter | Value | Notes |
|-----------|-------|-------|
| `GNN_STRUCTURAL_DROPOUT_PROB` | 0.0 | Disabled |
| `GNN_EDGE_CONTRASTIVE_WEIGHT` | 0.0 | Disabled |
| `GNN_INVARIANCE_WEIGHT` | 0.0 | Disabled |
| `GNN_SPATIAL_INVARIANCE_WEIGHT` | 0.0 | Disabled |

### Loss Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| `USE_FOCAL_LOSS` | True | Hard-example mining |
| `USE_CLASS_WEIGHTS` | False | Near-balanced |

### Threshold Policy

| Parameter | Value | Notes |
|-----------|-------|-------|
| `EVAL_THRESHOLD_POLICY` | `"youden"` | Balanced sensitivity/specificity |
| `EVAL_FIXED_THRESHOLD` | 0.5263 | Backward compat |

## 5) Quality Gates and Policies

| Gate | Value | Notes |
|------|-------|-------|
| `GNN_ENFORCE_GRAPH_QUALITY_GATE` | True | Block poor graphs |
| `GNN_MAX_DEGENERATE_GRAPH_RATE` | 0.35 | Max 35% degenerate |
| `GNN_MIN_EDGES_FOR_NONDEGENERATE` | 12 | Min edges |
| `GNN_ENFORCE_MULTIVIEW_QUALITY_GATE` | True | Block poor views |
| `GNN_MULTIVIEW_MAX_ZERO_EDGE_RATE` | 0.20 | Max 20% zero-edge |
| `MULTIVIEW_GENERATION_ENFORCE_QUALITY_GATE` | True | Generation-time gate |
| `HARMONIZATION_UNSEEN_SITE_POLICY` | `"passthrough"` | Unseen sites |

## 6) Medical Integrity Constraints

YOLO augmentation controls are intentionally conservative:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `YOLO_FLIPLR` | 0.0 | No horizontal flip |
| `YOLO_DEGREES` | 0.0 | No rotation |
| `YOLO_MOSAIC` | 0.0 | No mosaic |

These settings preserve anatomical left/right and spatial consistency.

## 7) Validation Helpers

`src/core/validators.py` provides runtime checks:

- `validate_environment()` — core paths and lobe mapping
- `validate_graph_construction_inputs()` — feature/graph prerequisites
- `validate_gnn_training_inputs()` — training prerequisites
- `get_active_checkpoint_dir()` — checkpoint resolution

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
   - threshold metadata alignment across evaluation scripts

## 9) Deprecated Parameters

The following parameters have been removed:

| Parameter | Removed | Notes |
|-----------|---------|-------|
| `EVAL_FREQUENCY` | v2026-04 | Use `--eval-frequency` CLI instead |

## 10) Common Misconfiguration Patterns

- Updating feature dimensions without updating registry-driven channel expectations
- Changing output paths without updating stage sentinels in `src/pipeline/registry.py`
- Mixing artifacts from different runs in a shared output directory
- Switching threshold policy without regenerating evaluation artifacts
- Setting `GNN_GRL_ALPHA = 1.0` — causes test AUC drop ⚠️