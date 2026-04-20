# Failure Modes

## Scope

This page lists common operational failures in Neuro-CXG, how they present, and how to recover safely.

## Quick Triage Order

1. Run environment and input validation.
2. Confirm required artifacts for the stage you are running.
3. Check quality-gate failures before changing model hyperparameters.
4. Regenerate only missing or stale upstream artifacts.

## 1) Preflight and Input Failures

### Symptom

- Pipeline exits early at preflight or before training.

### Typical Causes

- missing manifest or features
- missing graph files
- missing fold-specific harmonized CSVs
- invalid atlas or lobe mapping setup

### Where It Is Enforced

- `validate_environment()` in `src/core/validators.py`
- `validate_gnn_training_inputs()` in `src/core/validators.py`
- pre-training checks in `src/run_pipeline.py`

### Recovery

```bash
python -c "from src.core.config import validate_environment; validate_environment()"
python src/run_pipeline.py --auto --skip-download --skip-split
```

If fold files are missing, regenerate harmonization explicitly:

```bash
python -m src.features.fold_safe_harmonization
```

## 2) Feature Shape or Schema Mismatch

### Symptom

- dataset initialization fails with temporal/spatial shape mismatch
- runtime errors while assembling `Data` objects

### Typical Causes

- feature extraction changed but registry values not aligned
- stale metadata files from a previous schema

### Where It Is Enforced

- `_validate_feature_dimensions()` in `src/features/graph_factory.py`
- channel definitions in `src/core/feature_registry.py`

### Recovery

```bash
python src/run_pipeline.py --auto --regenerate-features
```

If stale artifacts remain, do a controlled reset:

```bash
python src/run_pipeline.py --auto --force-reset
```

## 3) Graph Degeneracy Gate Failure

### Symptom

- training aborts with graph quality gate failure
- too many subjects dropped due invalid/zero-edge graphs

### Typical Causes

- causality estimation produced sparse or zero-edge graphs at scale
- NaN/Inf or short time-series input cascades into weak graphs

### Where It Is Enforced

- `_assess_graph_degeneracy()` in `src/models/gnn_model.py`
- subject drop-rate gate in `src/features/graph_factory.py`
- threshold constants in `src/core/hyperparams.py`

### Recovery

1. Rebuild graphs:

```bash
python -m src.features.construct_causal --n-jobs -1
```

2. Run diagnostics:

```bash
python -m src.validation.pipeline_checks
python -m src.analysis.diagnose_dead_lobes --split train
```

3. Confirm upstream temporal/spatial artifacts are valid before retraining.

## 4) Multiview Quality Gate Disables Invariance

### Symptom

- logs indicate multiview invariance disabled due failing views
- non-base views show high zero-edge rates

### Typical Causes

- degenerate multiview branches
- stale multiview artifacts from older graph settings

### Where It Is Enforced

- multiview checks in `src/models/gnn_model.py`
- generation-time gate settings in `src/core/hyperparams.py`

### Recovery

```bash
python src/run_pipeline.py --auto --multiview --regenerate-features
```

If still degenerate, run without invariance objective until graph quality is fixed.

## 5) Site-Stratified CV Mismatch

### Symptom

- training/evaluation behavior changes unexpectedly after `--site-stratified-cv`
- fold harmonization appears inconsistent with current `cv_fold`

### Typical Causes

- `cv_fold` was regenerated but harmonized fold files were not refreshed

### Where It Is Enforced

- `run_site_stratified_split()` in `src/data/split.py` (explicit warning)
- fold file checks in `src/models/gnn_model.py` and `src/core/validators.py`

### Recovery

```bash
python -m src.data.split --site-stratified-cv
python -m src.features.fold_safe_harmonization
python -m src.models.gnn_model
```

## 6) Checkpoint Availability Failures

### Symptom

- evaluation/explainability/result-analysis cannot find fold checkpoints
- fold checkpoints partially missing

### Typical Causes

- training not run or interrupted
- checkpoint directory mismatch

### Where It Is Enforced

- model-loading logic in `src/run_evaluation.py`, `src/run_explainability.py`, `src/run_result_analysis.py`
- active checkpoint resolution in `get_active_checkpoint_dir()` (`src/core/validators.py`)

### Recovery

```bash
python -m src.models.gnn_model
python src/run_evaluation.py
```

## 7) Threshold or Metadata Drift Between Reports

### Symptom

- `run_result_analysis.py` outputs differ from evaluation metrics
- warnings about missing evaluation metadata

### Typical Causes

- `results/evaluation/comprehensive_results.json` missing or from a different run
- policy/threshold changed without regenerating evaluation artifacts

### Where It Is Enforced

- metadata load and threshold resolution in `src/run_result_analysis.py`
- threshold policy constants in `src/core/hyperparams.py`

### Recovery

```bash
python src/run_evaluation.py
python src/run_result_analysis.py
```

Keep both outputs in the same run-specific directory context.

## 8) Slow or Stalled Analysis Stages

### Symptom

- evaluation or explainability takes unexpectedly long

### Typical Causes

- high permutation count
- edge masking enabled in explainability

### Recovery

```bash
python src/run_evaluation.py --no-permutation
python src/run_evaluation.py --n-permutations 200
python src/run_explainability.py --no-masking
```

## 9) Safety Misconfiguration in YOLO Augmentation

### Symptom

- warnings about anatomy-destroying augmentation

### Typical Causes

- non-zero `YOLO_FLIPLR` or `YOLO_DEGREES`

### Where It Is Enforced

- warning path in `validate_environment()` (`src/core/validators.py`)

### Recovery

Reset medical-safe defaults in `src/core/hyperparams.py`:

- `YOLO_FLIPLR = 0.0`
- `YOLO_DEGREES = 0.0`

Then rerun:

```bash
python -c "from src.core.config import validate_environment; validate_environment()"
```

## 10) Recovery Strategy: Minimal Rebuild

Use the smallest rebuild that restores consistency:

1. Missing metadata only: regenerate feature/harmonization stages.
2. Broken graphs: regenerate causal graphs (and multiview if used).
3. Fold policy change: regenerate split policy and harmonized folds.
4. Model/report mismatch: retrain then rerun evaluation and analysis.

Prefer targeted reruns over full reset unless schema changes are broad.
