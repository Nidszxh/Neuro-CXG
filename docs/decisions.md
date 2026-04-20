# Design Decision Log

This log records active architectural and modeling decisions reflected in source code.

## DD-001: Aggregate 170 AAL ROIs into 12 lobe-level nodes

- Decision: construct 12-node graphs from lobe-level aggregation instead of 170-node ROI graphs.
- Rationale:
  - better sample efficiency for ABIDE-scale training
  - cleaner clinical interpretation at system level
- Source of truth:
  - `src/core/atlas_config.py` (lobe mapping)
  - `src/features/construct_causal.py` (aggregation + graph packaging)

## DD-002: Keep configuration modular, re-export through a stable facade

- Decision: maintain focused config modules (`paths`, `feature_registry`, `hyperparams`, `atlas_config`, `validators`) and expose them through `src/core/config.py`.
- Rationale:
  - minimizes drift from scattered constants
  - preserves backward compatibility for existing imports
- Source of truth:
  - `src/core/config.py`
  - `src/core/paths.py`
  - `src/core/feature_registry.py`
  - `src/core/hyperparams.py`

## DD-003: Use a declarative stage registry for orchestration

- Decision: define stage metadata and dependencies in `src/pipeline/registry.py`, and execute via `src/run_pipeline.py`.
- Rationale:
  - one place to update stage contracts
  - consistent skip/auto/dry-run behavior
- Source of truth:
  - `src/pipeline/registry.py`
  - `src/run_pipeline.py`

## DD-004: Enforce fold-safe harmonization with diagnosis protection

- Decision: fit ComBat on fold-train only and apply to val/test; include `DX_GROUP` as protected covariate.
- Rationale:
  - prevents fold leakage
  - avoids stripping disease-related variance during site correction
- Source of truth:
  - `src/features/fold_safe_harmonization.py`

## DD-005: Retain only anatomical spatial channels in model input

- Decision: model input uses 4 spatial channels (`x`, `y`, `z_depth`, `size`).
- Clarification:
  - `conf_std` and `detection_count` are still processed for harmonization diagnostics,
    but excluded from graph node feature tensors used by GNN training.
- Rationale:
  - reduces scanner/site proxy leakage in learning signal
- Source of truth:
  - `src/core/feature_registry.py`
  - `src/features/graph_factory.py`
  - `src/features/fold_safe_harmonization.py`

## DD-006: Use directed causal connectivity with ridge-regularized Granger default

- Decision: causal graph construction defaults to `ridge_granger` with fallback methods available by config.
- Rationale:
  - directed edges preserve temporal precedence information
  - regularization improves stability for small-sample fold subsets
- Source of truth:
  - `src/core/hyperparams.py` (`CAUSALITY_METHOD`)
  - `src/features/construct_causal.py`
  - `src/features/causal_inference.py`

## DD-007: Keep multiview graph generation and invariance training optional, with quality gates

- Decision: multiview graph construction is opt-in (`--multiview`), and invariance loss is enabled only when multiview artifacts are present and pass quality checks.
- Rationale:
  - allows robustness experiments without forcing all runs into higher-complexity training
  - guards against degenerate non-base views
- Source of truth:
  - `src/pipeline/registry.py` (`multiview_graphs`)
  - `src/run_pipeline.py` (`--multiview`)
  - `src/models/gnn_model.py` (multiview availability + quality gate)

## DD-008: Keep GRL site-adversarial path enabled but controlled

- Decision: GRL is enabled in config with conservative alpha defaults (`GNN_GRL_ALPHA`, `GNN_GRL_ALPHA_MAX`), with optional alpha grid-search support.
- Rationale:
  - mitigate site leakage while avoiding aggressive adversarial settings
- Source of truth:
  - `src/core/hyperparams.py`
  - `src/models/gnn_model.py`
  - `src/models/training_utils.py`

## DD-009: Support auxiliary structural/invariance losses as tunable controls

- Decision: structural dropout, edge contrastive, causal invariance, and spatial invariance terms are implemented and controlled by config weights.
- Current default posture:
  - conservative baseline (most auxiliary weights set to `0.0`)
  - mechanism remains available for ablations/experiments
- Source of truth:
  - `src/core/hyperparams.py`
  - `src/models/gnn_model.py`
  - `src/models/training_utils.py`

## DD-010: Make site-stratified CV an explicit optional protocol

- Decision: default split path uses stratified folds; optional `--site-stratified-cv` rewrites `cv_fold` with GroupKFold over site clusters.
- Rationale:
  - preserve baseline comparability while enabling stricter cross-site robustness studies
- Source of truth:
  - `src/data/split.py`
  - `src/run_pipeline.py`
  - `src/pipeline/registry.py`

## DD-011: Lock deployment operating point through threshold policy

- Decision: evaluation and downstream analysis use centralized threshold policy (`EVAL_THRESHOLD_POLICY`), currently fixed at `EVAL_FIXED_THRESHOLD = 0.5263`.
- Rationale:
  - deterministic deployment threshold across runs
  - avoids per-run threshold drift in reports
- Source of truth:
  - `src/core/hyperparams.py`
  - `src/run_evaluation.py`
  - `src/run_result_analysis.py`
  - `src/models/training_utils.py`

## DD-012: Preserve explainability as a first-class post-training stage

- Decision: retain scripted node, edge, feature, and literature explainability workflows in the default analysis stack.
- Rationale:
  - supports scientific interpretability requirements beyond single scalar metrics
- Source of truth:
  - `src/run_explainability.py`
  - `src/analysis/node_importance.py`
  - `src/analysis/edge_importance.py`
  - `src/analysis/feature_attribution.py`
  - `src/analysis/literature_validation.py`
