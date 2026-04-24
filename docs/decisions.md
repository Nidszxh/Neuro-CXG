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

## DD-006: Use directed causal connectivity with lagged-pearson default

- Decision: use `lagged_pearson` as the default causal connectivity method (changed from `ridge_granger`).
- Rationale:
  - lagged Pearson correlation with multi-lag selection captures slow hemodynamic coupling (0.01–0.15 Hz) better than VAR-based Granger at the 12-lobe aggregation level
  - produces group-discriminative edges (ASD vs Control p-value < 0.05 for several lobes)
  - lower fold variance than Granger-based methods
- Available alternatives:
  - `ridge_granger` - VAR-based Granger with ridge regularization
  - `lagged_pearson` - multi-lag Pearson correlation (current default)
  - `ridge_granger_hybrid` - beta-weighted combination
  - `partial_corr_glasso` - sparse conditional dependence
## DD-007: Optimized GNN hyperparameters for publication performance (April 2026)

- Decision: use site conditioning, demographic conditioning, and anatomical pooling for publication-quality results.
- Key changes:
  - `GNN_USE_SITE_EMBEDDING = True` - adds site-aware conditioning
  - `GNN_USE_DEMOGRAPHICS = True` - adds demographic context
  - `GNN_GRL_ALPHA_MAX = 1.0` - strong adversarial debiasing (increased from 0.15)
  - `GNN_POOLING = "anatomical"` - 2-level hierarchy pooler (changed from mean_max_sum)
  - `GNN_HIDDEN_CHANNELS = 32` - reduced from 64 to reduce overfitting
  - `GNN_WEIGHT_DECAY = 5e-4` - increased from 5e-5
- Results:
  - CV AUC: 0.8004 ± 0.0293 (was 0.7586 ± 0.0519)
  - Test AUC: 0.8753 (was 0.7325)
  - Test F1: 0.8121 (was 0.6338)
- Source of truth:
  - `src/core/hyperparams.py`

## DD-008: Youden threshold policy for balanced sensitivity/specificity

- Decision: use Youden threshold policy for evaluation reporting.
- Rationale:
  - fixed threshold (0.5263) produced low sensitivity (0.57) for an ASD screening tool
  - Youden threshold provides balanced operating point with sensitivity ~0.73
- Source of truth:
  - `src/core/hyperparams.py` (`EVAL_THRESHOLD_POLICY = "youden"`)

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
