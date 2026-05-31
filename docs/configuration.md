# Configuration

Neuro-CXG is configuration-driven. Runtime behavior should be changed through `src/core/*` config modules, not by hardcoding values inside stage scripts.

---

## 1) Configuration Topology

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

---

## 2) Path Configuration

Key roots defined in `src/core/paths.py`:

- `PROJECT_ROOT`
- `DATA_ROOT`
- `MODEL_ROOT`
- `RESULTS_DIR`

### High-Impact File Paths

| Constant | Resolved Path |
|----------|---------------|
| `MASTER_MANIFEST` | `data/metadata/master_manifest.csv` |
| `NODE_ATTRIBUTES_HARMONIZED` | `data/metadata/node_attributes_harmonized.csv` |
| `HARMONIZED_FOLDS_DIR` | `data/metadata/harmonized_folds_cv/` |
| `NODE_FEATURES_3D` | `data/metadata/node_features_3d.csv` |
| `NODE_FEATURES_3D_HARMONIZED` | `data/metadata/node_features_3d_harmonized.csv` |
| `CAUSAL_GRAPHS_DIR` | `data/processed/causal_graphs/` |
| `CAUSAL_GRAPHS_MULTIVIEW_DIR` | `data/processed/causal_graphs_multiview/` |
| `CHECKPOINT_DIR` | `models/checkpoints/` |
| `RESULTS_EVALUATION_DIR` | `results/evaluation/` |

---

## 3) Feature Registry

Feature channel layout is derived from `src/core/feature_registry.py`.

### Current Groups

| Group | Channels | Description |
|-------|----------|-------------|
| temporal | 8 | Base time-domain features (mean, std, skew, kurtosis, psd, mssd, range, autocorr) |
| frequency | 10 | From `ACTIVE_FREQ_BANDS` + spectral_entropy + phase_std |
| internal | 2 | coherence, spatial_variance |
| spatial | 4 | x, y, z_depth, size |

### Important Behavior

- **Gamma band exclusion**: Gamma band is excluded by default:
  - `UNRELIABLE_FREQ_BANDS_AT_NYQUIST = ("gamma",)`
  - `EXCLUDE_NYQUIST_BANDS = True`
- **Dynamic channel count**: `GNN_IN_CHANNELS` is computed dynamically from `ALL_FEATURE_NAMES` (24 total when gamma excluded)

⚠️ **Safety sentinel for spatial channels:** `NUM_SPATIAL_FEATURES == 4` is enforced by assertion to prevent reintroducing site-leaky channels (`conf_std`, `detection_count`).

---

## 4) Atlas Mapping

Defined in `src/core/atlas_config.py`:

- `LOBE_MAPPING`: Maps 170 AAL ROIs into **12 lobe nodes**
- `LOBE_NAMES`: Lobe labels used across extraction, graphs, and plots
- Optional hierarchy:
  - `LOBE_TO_NETWORK`
  - `NETWORK_TO_LOBES`
  - `NUM_NETWORKS = 4`

This hierarchy is used when `GNN_POOLING = "anatomical"`.

### Architecture Status (DD-018)

**12-Lobe is approved for publication** (April 28, 2026):

- YOLO v29 never detects Brainstem → constant synthetic features act as implicit regularization
- Test AUC: 0.8648 (ridge_granger_hybrid method)
- See `docs/decisions.md` (DD-018) for full analysis
- **Method**: ridge_granger_hybrid (70% Ridge Granger + 30% Lagged Pearson) — best Granger-based performance

---

## 5) Hyperparameters

### Causal Graph Defaults

| Parameter | Value | Notes |
|-----------|-------|-------|
| `CAUSALITY_METHOD` | `"ridge_granger_hybrid"` | 70% Ridge Granger + 30% Lagged Pearson |
| `RIDGE_GRANGER_HYBRID_BETA` | 0.70 | Weight for Granger component in hybrid |
| `GRANGER_MAX_LAG` | 5 | Number of lags |
| `GRANGER_MAX_LAG_SECONDS` | 10.0 | Max lag in seconds |
| `GRANGER_USE_GPU` | True | GPU acceleration |
| `RIDGE_GRANGER_LAGS` | (1,2,3,4,5) | VAR lag order |
| `RIDGE_GRANGER_LAMBDA` | 0.1 | Ridge regularization |
| `SPARSITY_METHOD` | `"topk_per_node"` | Sparsification method |
| `SPARSITY_TOPK_PER_NODE` | 3 | Edges per node |
| `MIN_EDGES_PER_GRAPH` | 12 | Minimum edges |

### GNN Architecture Defaults (Best: 48ch/4hd/3L/0.33)

| Parameter | Canonical | Best (May 2026) | Notes |
|-----------|-----------|-----------------|-------|
| `GNN_HIDDEN_CHANNELS` | 32 | **48** | Increased capacity, best AUC |
| `GNN_NUM_HEADS` | 2 | **4** | More attention heads |
| `GNN_NUM_LAYERS` | 2 | **3** | Deeper GNN, better generalization |
| `GNN_DROPOUT` | 0.35 | **0.33** | Optimal regularization |
| `GNN_POOLING` | `"anatomical"` | `"anatomical"` | Hierarchical pooling |
| `GNN_WEIGHT_DECAY` | 5e-4 | 5e-4 | L2 regularization |
| `GNN_ONECYCLE_MAX_LR` | 0.001 | 0.001 | Max learning rate |
| `GNN_ONECYCLE_WARMUP_FRACTION` | 0.05 | **0.20** | Longer warmup for GRL stability |
| `GNN_BATCH_SIZE` | 32 | 32 | Batch size |
| `GNN_EPOCHS` | 100 | 100 | Max epochs |
| `K_FOLDS` | 5 | 5 | CV folds |
| `GNN_SEED` | 42 | 42 | Global seed (everywhere) |

**Best Config Provenance**: May 31, 2026 run, AUC=0.8819 [0.8277, 0.9322]

### Site Bias Controls

| Parameter | Value | Notes |
|-----------|-------|-------|
| `GNN_USE_SITE_EMBEDDING` | True | Site conditioning |
| `GNN_USE_DEMOGRAPHICS` | True | Demographic conditioning |
| `GNN_USE_GRL` | True | Gradient reversal layer |
| `GNN_GRL_ALPHA` | **0.10** | Fixed - no grid search |
| `GNN_GRL_ALPHA_MAX` | 0.10 | No annealing (fixed alpha) |
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
| `EVAL_FIXED_THRESHOLD` | 0.5263 | Backward compatibility |

---

## 6) Quality Gates and Policies

| Gate | Value | Notes |
|------|-------|-------|
| `GNN_ENFORCE_GRAPH_QUALITY_GATE` | True | Block poor graphs |
| `GNN_MAX_DEGENERATE_GRAPH_RATE` | 0.35 | Max 35% degenerate |
| `GNN_MIN_EDGES_FOR_NONDEGENERATE` | 12 | Min edges |
| `GNN_ENFORCE_MULTIVIEW_QUALITY_GATE` | True | Block poor views |
| `GNN_MULTIVIEW_MAX_ZERO_EDGE_RATE` | 0.20 | Max 20% zero-edge |
| `MULTIVIEW_GENERATION_ENFORCE_QUALITY_GATE` | True | Generation-time gate |
| `HARMONIZATION_UNSEEN_SITE_POLICY` | `"fail"` | Unseen sites cause the pipeline to fail |

---

## 7) Medical Integrity Constraints

YOLO augmentation controls are intentionally conservative to preserve anatomical left/right and spatial consistency:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `YOLO_FLIPLR` | 0.0 | No horizontal flip |
| `YOLO_DEGREES` | 0.0 | No rotation |
| `YOLO_MOSAIC` | 0.0 | No mosaic |

---

## 8) Validation Helpers

`src/core/validators.py` provides runtime checks used by the runner and training:

| Function | Purpose |
|----------|---------|
| `validate_environment()` | Core paths and lobe mapping invariants |
| `validate_graph_construction_inputs()` | Feature/graph prerequisites |
| `validate_gnn_training_inputs()` | Training prerequisites (harmonized files, etc.) |
| `get_active_checkpoint_dir()` | Checkpoint resolution |

---

## 9) Safe Change Workflow

When updating config values:

1. Change constants in the owning module (not ad-hoc in scripts).
2. Run dry plan:

```bash
python src/run_pipeline.py --dry-run
```

3. Re-run affected stages only.
4. Re-check downstream contracts:
   - Feature dimensions
   - Sentinel outputs
   - Threshold metadata alignment across evaluation scripts

---

## 10) Common Misconfiguration Patterns

- Updating feature dimensions without updating registry-driven channel expectations
- Changing output paths without updating stage sentinels in `src/pipeline/registry.py`
- Mixing artifacts from different runs in a shared output directory
- Switching threshold policy without regenerating evaluation artifacts
- ⚠️ **Setting `GNN_GRL_ALPHA = 1.0`** — causes test AUC drop from 0.87 to 0.83

---

## 11) Deprecated Parameters

| Parameter | Version Removed | Notes |
|-----------|-----------------|-------|
| `EVAL_FREQUENCY` | v2026-04 | Use `--eval-frequency` CLI instead |
| `LAGGED_PEARSON_METHOD` | v2026-04 | Superseded by ridge_granger_hybrid |
| `GNN_AUTO_GRL_GRID_SEARCH` | v2026-04 | Set to False in gap-closure wave |
| `GNN_EDGE_CONTRASTIVE_WEIGHT` | v2026-04 | Reduced to 0.0 (disabled) |
| `compute_granger_causality_gpu` | v2026-04 | Removed (dead code, Task 6 DD-014) |
| `compute_transfer_entropy` | v2026-04 | Removed (dead code, Task 6 DD-014) |
| `compute_multilag_causality` | v2026-04 | Removed (dead code, Task 6 DD-014) |