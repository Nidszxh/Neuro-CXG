# CHANGELOG

All notable changes to Neuro-CXG are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased] — 2026-04-19

### Wave-1 Generalization Stabilization (Core Pipeline Integration)
**Root Cause**: CV-test gap remained elevated; fold-level preprocessing and robust partial-correlation pruning were not fully integrated into core training/evaluation flow.

**Added**
- Fold-internal MI feature selection in `src/models/gnn_model.py`:
  - Fit on train fold only
  - Applied as a feature mask (dimension-preserving, checkpoint-compatible)
  - Config controls in `src/core/hyperparams.py`:
    - `GNN_MI_FEATURE_SELECTION_ENABLED`
    - `GNN_MI_MIN_KEEP_RATIO`
    - `GNN_MI_MAX_KEEP_RATIO`
- Fold-internal within-site normalization in `src/models/gnn_model.py`:
  - Per-site train-fold stats
  - Unseen-site fallback to global train-fold stats
  - Config control: `GNN_SITE_NORMALIZATION_MODE`
- Fold preprocessing mode switch in `src/core/hyperparams.py`:
  - `GNN_FOLD_PREPROCESSING_MODE = "wave1"`
  - `legacy_global` mode retained for rollback/backward compatibility.
- Partial-correlation FDR pruning support in `src/features/construct_causal.py`:
  - BH/FDR significance mask for `partial_corr_glasso`
  - Approximate two-sided p-values from partial correlations
  - Config controls:
    - `PARTIAL_CORR_FDR_ENABLED`
    - `PARTIAL_CORR_FDR_ALPHA`
- Checkpoint metadata support for Wave-1 preprocessing in `src/models/training_utils.py`:
  - `feature_mask`, `selected_feature_idx`, `feature_selection_meta`
  - `site_feature_means`, `site_feature_stds`
  - `preprocessing_mode`, `site_normalization_mode`

**Changed**
- `src/models/causal_gnn.py` now applies checkpoint-loaded preprocessing metadata at inference:
  - Feature mask
  - Within-site normalization when available
  - Safe fallback behavior for legacy checkpoints.
- `src/experiments/run_ablations.py` now uses manifest `cv_fold` splits (site-stratified protocol parity) instead of ad-hoc `StratifiedKFold`.

**Tests**
- Extended `tests/unit/test_construct_causal_partial_corr.py`:
  - Validates `pvalue_matrix` + `fdr_significant_mask` metadata
  - Validates FDR pruning behavior.
- Added `tests/unit/test_training_utils_checkpoint_preprocessing.py`:
  - Verifies legacy + Wave-1 checkpoint metadata attachment and backward compatibility.
- Added `tests/unit/test_gnn_wave1_preprocessing.py`:
  - MI feature selection bounds/mask behavior
  - Within-site normalization and unseen-site global fallback.

### Deployment Operating Point Lock (Condition C)

**Changed**
- Locked Wave-1 deployment defaults in `src/core/hyperparams.py` to validated Condition C profile:
  - `GNN_MI_MIN_KEEP_RATIO = 0.30`
  - `GNN_MI_MAX_KEEP_RATIO = 0.60`
  - `GNN_SITE_NORMALIZATION_MODE = "within_site"`
  - `PARTIAL_CORR_FDR_ALPHA = 0.10`
- Added fixed-threshold deployment mode:
  - `EVAL_THRESHOLD_POLICY = "fixed"`
  - `EVAL_FIXED_THRESHOLD = 0.5263`
- Extended threshold-policy handling in:
  - `src/run_evaluation.py`
  - `src/run_result_analysis.py`
  - `src/models/gnn_model.py`
  - `src/models/training_utils.py`

**Added**
- Coverage check for fixed threshold support in `tests/unit/test_feature_ordering.py`.

**Artifacts**
- Seed-stability and threshold calibration summaries:
  - `results/analysis/seed_stability_C/summary.json`
  - `results/analysis/seed_stability_C/specificity_calibration_snapshots/summary.json`

### Gap-Closure Wave 1 (Config + Graph Method)
**Root Cause**: CV-test gap remains elevated; current defaults still favor higher-capacity training and single-method causal graphs.

**Added**
- `partial_corr_glasso` causal method in `construct_causal.py` via GraphicalLasso-based sparse partial correlations.
- New config controls in `hyperparams.py`:
  - `PARTIAL_CORR_GLASSO_ALPHA`
  - `PARTIAL_CORR_GLASSO_MAX_ITER`
  - `PARTIAL_CORR_GLASSO_TOL`
  - `PARTIAL_CORR_MIN_ABS_EDGE`
  - `PARTIAL_CORR_MIN_SAMPLES`
- Unit tests for new method behavior and fallbacks: `tests/unit/test_construct_causal_partial_corr.py`.

**Changed**
- Gap-first conservative model defaults in `hyperparams.py`:
  - `GNN_HIDDEN_CHANNELS`: `128` -> `64`
  - `GNN_NUM_HEADS`: `4` -> `2`
  - `GNN_AUTO_GRL_GRID_SEARCH`: `True` -> `False`
  - `GNN_EDGE_CONTRASTIVE_WEIGHT`: `0.1` -> `0.0`
- `construct_causal.py` post-sparsification stats now handle methods without p-values (e.g., partial-correlation) without misreporting high-confidence edges.

### GPU-Accelerated Granger Causality
**Root Cause**: Sequential CPU Granger causality is slow (~42 min for multiview on ~1000 subjects).

**Added**
- `GRANGER_USE_GPU` config flag in `hyperparams.py` (default: `True`)
- `_compute_granger_causality_gpu_impl()` in `causal_inference.py`: GPU-accelerated
  Granger using batched linear regression + vectorized F-test
- Auto-detection: uses GPU when `GRANGER_USE_GPU=True` and CUDA available
- Fallback: auto-falls back to CPU on any error
- Added `use_gpu` param to `compute_granger_causality()` and `compute_causality_matrix()`
- Updated `construct_multiview_graphs()` to use GPU
- Test script: `tests/unit/test_granger_gpu.py`

---

## [Unreleased] — 2026-04-14

### Task 1 — Structural Learning Enforcement (DD-009)
**Root Cause**: Model used node features as primary signal; edge structure largely ignored
(GradientEdgeAttributor returned near-zero scores for most edges).

**Added**
- `_apply_structural_dropout()` in `training_utils.py`: zeros node features for ~30%
  of graphs per batch during training, forcing edge-structure-only classification paths.
- `EdgeStructureContrastiveLoss` in `training_utils.py`: NT-Xent loss (τ=0.5) between
  full-feature and edge-only graph embeddings; weight 0.05 in total loss.
- `_forward_with_embedding()` in `CausalBrainGNN`: returns (logits, embedding) for
  dual-view forward training.
- `structural_dropout_prob` and `edge_contrastive_weight` args to
  `train_one_epoch_with_accumulation` and `train_fold_with_onecycle` (default 0.0 —
  backward compatible). Canonical training now uses 0.30 / 0.05.
- Unit tests: `tests/unit/test_structural_learning.py`

---

### Task 2 — Multi-View Causal Graph Construction (DD-010)
**Root Cause**: Single Granger estimate is noisy; one bad fit propagates directly
to graph embedding without any self-correction.

**Added**
- `construct_multiview_graphs()` in `construct_causal.py`: generates 6 causal graph
  views per subject (base, extended_lag, 3 bootstraps, high_confidence).
- `main_multiview()` entry point in `construct_causal.py` with `--multiview` CLI flag.
- `CAUSAL_GRAPHS_MULTIVIEW_DIR` in `paths.py`: `data/processed/causal_graphs_multiview/`.
- `CausalInvarianceLoss` in `gnn_model.py`: NT-Xent loss (τ=0.07) across views;
  weight 0.15. Activates automatically when multiview dir is populated.
- `forward_multiview()` in `CausalBrainGNN`: forwards list of Batch objects.
- `multiview_graphs` Stage in `registry.py` (opt-in, after `causal_graphs`).
- Unit tests: `tests/unit/test_causal_invariance.py`

---

### Task 3 — Anatomical Hierarchical Pooling (DD-011)
**Root Cause**: Global pooling collapses the brain's two-level functional hierarchy.

**Added**
- `LOBE_TO_NETWORK`, `NETWORK_TO_LOBES`, `NUM_NETWORKS`, `NETWORK_NAMES` in `atlas_config.py`.
- `AnatomicalHierarchyPool` in `causal_gnn.py`: 2-level attention pooling
  (lobes→networks→graph). Stores `last_network_embeddings` for explainability.
- Default `pooling` changed to `"anatomical"` in `CausalBrainGNN`. Old modes retained.
- `_aggregate_to_networks()` and network-level GradCAM plot in `node_importance.py`.
- Unit tests: `tests/unit/test_anatomical_pool.py`

---

### Task 4 — Spatial Feature Cleanup (DD-012)
**Root Cause**: `conf_std` and `detection_count` perfectly predict acquisition site
(RF AUC=1.000 in run 3) — pure site leakage.

**Changed**
- `feature_registry.py`: sentinel `assert NUM_SPATIAL_FEATURES == 4` added.
- `feature_registry.py`: stale "currently 26" comment corrected to "currently 24".
- `graph_factory.py`: fixed 3 stale docstrings (6→4 spatial features).
- `SpatialInvarianceLoss` added to `gnn_model.py` for residual site variance guard.
- Unit tests: `tests/unit/test_spatial_cleanup.py`

---

### Task 5 — Site-Stratified Cross-Validation (DD-013)
**Root Cause**: StratifiedKFold inflates CV AUC by allowing same-scanner subjects
in both training and validation splits.

**Added**
- `SCANNER_MANUFACTURER` map, `_assign_site_clusters()`, `generate_site_stratified_folds()`,
  `run_site_stratified_split()` in `split.py`.
- `--site-stratified-cv` CLI flag in `split.py`.
- `site_stratified_cv` Stage in `registry.py` (opt-in).
- Hard `FileNotFoundError` assertion in `gnn_model._run_training_once()`.

---

### Task 6 — Dead Code Removal (DD-014)
**Root Cause**: Unmaintained functions inflate maintenance burden.

**Removed**
- `compute_granger_causality_gpu`, `compute_transfer_entropy`, `_compute_te_pair`,
  `_conditional_entropy`, `compute_multilag_causality` from `causal_inference.py`.
- GPU branch and `transfer_entropy` branch from `construct_causal.compute_causality_matrix()`.
- `EVAL_FREQUENCY = 10` from `hyperparams.py` (never read).

---

## Previous Notable Changes

### 2026-03-09 — P0/P1 Fixes (CV AUC 0.6194→0.7434, Test AUC→0.6487)
- Disabled high-alpha GRL (alpha=1.0 → GRL off by default)
- Added DX_GROUP as protected ComBat covariate
- Fixed dead-lobe NaN handling before PCA
- Applied fold-safe CV harmonization

### 2026-02-15 — Baseline
- Initial GATv2 architecture, global harmonization, CV AUC ~0.62
