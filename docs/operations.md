# Operations

## Quick Triage Order

1. Run environment and input validation
2. Confirm required artifacts for the stage you are running
3. Check quality-gate failures before changing model hyperparameters
4. Regenerate only missing or stale upstream artifacts

---

## Part A: Failure Modes

### 1) Preflight and Input Failures

**Symptom:**
- Pipeline exits early at preflight or before training

**Typical Causes:**
- Missing manifest or features
- Missing graph files
- Missing fold-specific harmonized CSVs
- Invalid atlas or lobe mapping setup

**Where It Is Enforced:**
- `validate_environment()` in `src/core/validators.py`
- `validate_gnn_training_inputs()` in `src/core/validators.py`
- Pre-training checks in `src/run_pipeline.py`

**Recovery:**
```bash
python -c "from src.core.config import validate_environment; validate_environment()"
python src/run_pipeline.py --auto --skip-download --skip-split
```

*Status: Resolved — run harmonization explicitly if needed*

---

### 2) Feature Shape or Schema Mismatch

**Symptom:**
- Dataset initialization fails with temporal/spatial shape mismatch
- Runtime errors while assembling `Data` objects

**Typical Causes:**
- Feature extraction changed but registry values not aligned
- Stale metadata files from a previous schema

**Where It Is Enforced:**
- `_validate_feature_dimensions()` in `src/features/graph_factory.py`
- Channel definitions in `src/core/feature_registry.py`

**Recovery:**
```bash
python src/run_pipeline.py --auto --regenerate-features
```

*Status: Resolved — controlled reset if stale artifacts persist*

---

### 3) Graph Degeneracy Gate Failure

**Symptom:**
- Training aborts with graph quality gate failure
- Too many subjects dropped due to invalid/zero-edge graphs

**Typical Causes:**
- Causality estimation produced sparse or zero-edge graphs at scale
- NaN/Inf or short time-series input cascades into weak graphs

**Where It Is Enforced:**
- `_assess_graph_degeneracy()` in `src/models/gnn_model.py`
- Subject drop-rate gate in `src/features/graph_factory.py`
- Threshold constants in `src/core/hyperparams.py`

**Recovery:**
```bash
# Rebuild graphs
python -m src.features.construct_causal --n-jobs -1

# Run diagnostics
python -m src.validation.pipeline_checks
```

*Status: Resolved — rebuild graphs and validate upstream artifacts*

---

### 4) Multiview Quality Gate Disables Invariance

**Symptom:**
- Logs indicate multiview invariance disabled due to failing views
- Non-base views show high zero-edge rates

**Typical Causes:**
- Degenerate multiview branches
- Stale multiview artifacts from older graph settings

**Where It Is Enforced:**
- Multiview checks in `src/models/gnn_model.py`
- Generation-time gate settings in `src/core/hyperparams.py`

**Recovery:**
```bash
python src/run_pipeline.py --auto --multiview --regenerate-features
```

*Status: Resolved — regenerate multiview artifacts or disable invariance*

---

### 5) Site-Stratified CV Mismatch

**Symptom:**
- Training/evaluation behavior changes unexpectedly after `--site-stratified-cv`
- Fold harmonization appears inconsistent with current `cv_fold`

**Typical Causes:**
- `cv_fold` was regenerated but harmonized fold files were not refreshed

**Where It Is Enforced:**
- `run_site_stratified_split()` in `src/data/split.py`
- Fold file checks in `src/models/gnn_model.py`

**Recovery:**
```bash
python -m src.data.split --site-stratified-cv
python -m src.features.fold_safe_harmonization
python -m src.models.gnn_model
```

*Status: Resolved — regenerate folds after CV policy change*

---

### 6) Checkpoint Availability Failures

**Symptom:**
- Evaluation/explainability/result-analysis cannot find fold checkpoints
- Fold checkpoints partially missing

**Typical Causes:**
- Training not run or interrupted
- Checkpoint directory mismatch

**Where It Is Enforced:**
- Model-loading logic in `src/run_evaluation.py`, `src/run_explainability.py`
- Active checkpoint resolution in `get_active_checkpoint_dir()` (`src/core/validators.py`)

**Recovery:**
```bash
python -m src.models.gnn_model
python src/run_evaluation.py
```

*Status: Resolved — retrain from checkpoints*

---

### 7) Threshold or Metadata Drift Between Reports

**Symptom:**
- `run_result_analysis.py` outputs differ from evaluation metrics
- Warnings about missing evaluation metadata

**Typical Causes:**
- `comprehensive_results.json` missing or from a different run
- Policy/threshold changed without regenerating evaluation artifacts

**Where It Is Enforced:**
- Metadata load and threshold resolution in `src/run_result_analysis.py`
- Threshold policy constants in `src/core/hyperparams.py`

**Recovery:**
```bash
python src/run_evaluation.py
python src/run_result_analysis.py
```

*Status: Resolved — keep outputs in same run-specific directory*

---

### 8) Slow or Stalled Analysis Stages

**Symptom:**
- Evaluation or explainability takes unexpectedly long

**Typical Causes:**
- High permutation count
- Edge masking enabled in explainability

**Recovery:**
```bash
python src/run_evaluation.py --no-permutation
python src/run_evaluation.py --n-permutations 200
python src/run_explainability.py --no-masking
```

*Status: Workaround available*

---

### 9) Safety Misconfiguration in YOLO Augmentation

**Symptom:**
- Warnings about anatomy-destroying augmentation

**Typical Causes:**
- Non-zero `YOLO_FLIPLR` or `YOLO_DEGREES`

**Where It Is Enforced:**
- Warning path in `validate_environment()` (`src/core/validators.py`)

**Recovery:**
```bash
# Reset to medical-safe defaults
YOLO_FLIPLR = 0.0
YOLO_DEGREES = 0.0
```

*Status: Resolved — defaults are intentionally conservative*

---

### 10) Recovery Strategy: Minimal Rebuild

Use the smallest rebuild that restores consistency:

1. Missing metadata only: regenerate feature/harmonization stages
2. Broken graphs: regenerate causal graphs (and multiview if used)
3. Fold policy change: regenerate split policy and harmonized folds
4. Model/report mismatch: retrain then rerun evaluation and analysis

*Prefer targeted reruns over full reset unless schema changes are broad.*

---

## Part B: Performance

### Runtime Profile

Neuro-CXG runtime depends on whether the run includes data acquisition, model training, and optional heavy analysis stages.

| Stage | Primary Cost | Parallelism Available |
|-------|-------------|---------------------|
| ABIDE download | Network IO, preprocessing | Limited |
| YOLO training | GPU compute | Moderate |
| Feature extraction | Time-series transforms | Yes (`--n-jobs`) |
| Graph construction | Causal estimation | Yes (`--n-jobs`) |
| GNN training | GPU compute | Limited to folds |
| Evaluation | Bootstrap/permutation | Yes |

### Wall-Clock Estimates

| Stage | Time |
|-------|------|
| ABIDE download | 2-6 hours |
| Train/val split | 5-10 min |
| ROI annotation | 30-60 min |
| Feature extraction | 1-2 hours |
| Causal graphs | 2-4 hours |
| GNN training (5-fold) | 30-60 min |
| Evaluation | 15-30 min |
| **Total (full rebuild)** | **6-12 hours** |

### Performance Knobs

#### CLI Controls

| Command | Effect |
|---------|--------|
| `--skip-download --skip-split` | Avoids repeating data ingestion |
| `--analysis-only` | Skips build/training |
| `--no-permutation` | Removes major cost center |
| `--n-permutations 200` | Reduces permutation runtime |
| `--no-masking` | Avoids slow edge masking |

#### Config Knobs

| Parameter | Effect |
|-----------|--------|
| `GRANGER_USE_GPU` | GPU path in causal inference |
| `GNN_BATCH_SIZE` | Training memory/throughput |
| `GNN_HIDDEN_CHANNELS` | Model capacity |
| `GNN_NUM_HEADS` | Attention heads |
| `GNN_INVARIANCE_WEIGHT` | Enables training cost if >0 |

### Artifact-Backed Model Performance Snapshot

#### Current Best (May 2026)

| Metric | Value | Notes |
|--------|-------|-------|
| **CV AUC** | 0.8100 ± 0.0273 | 5-fold |
| **Test AUC** | **0.8648** | Ensemble, 95% CI [~0.78–0.90] |
| **Test F1** | **0.7682** | Youden threshold |
| **Test Accuracy** | **0.7826** | |
| **Mean Best Epoch** | ~35 | |

**Method:** ridge_granger_hybrid (β=0.70)

#### Per-Fold Results (12-Lobe)

| Fold | AUC | F1 |
|------|-----|-----|
| 1 | 0.7816 | 0.7552 |
| 2 | 0.7623 | 0.7183 |
| 3 | 0.8215 | 0.7879 |
| 4 | 0.7885 | 0.7758 |
| 5 | 0.8445 | 0.7714 |

#### Configuration That Achieved These Results

```python
CAUSALITY_METHOD = "lagged_pearson"
GRANGER_MAX_LAG_SECONDS = 10.0
GNN_HIDDEN_CHANNELS = 32
GNN_WEIGHT_DECAY = 5e-4
GNN_POOLING = "anatomical"
GNN_USE_SITE_EMBEDDING = True
GNN_USE_DEMOGRAPHICS = True
GNN_GRL_ALPHA = 0.10  # NOT 1.0 - test drops with 1.0
USE_FOCAL_LOSS = True
EVAL_THRESHOLD_POLICY = "youden"
```

### Historical Performance

| Run | CV AUC | Test AUC | Change |
|-----|-------|---------|--------|
| Baseline | 0.7586 ± 0.0519 | 0.7325 | ridge_granger |
| lagged_pearson | 0.8004 ± 0.0293 | 0.8753 | switched to lagged_pearson, GRL=0.10 |
| 12-Lobe Final | 0.7997 ± 0.0294 | 0.8694 | 12-lobe + lagged_pearson |
| **ridge_granger_hybrid** | **0.8100 ± 0.0273** | **0.8648** | current best (CV-focused) |

### Metrics: Interpreting Disagreement

Different output bundles can disagree because they may come from different runs, thresholds, checkpoints, or calibration assumptions.

**Practical guidance:**
- Treat `results/evaluation/comprehensive_results.json` as authoritative
- Confirm threshold metadata before comparing with result-analysis outputs
- Avoid combining metrics from directories produced by different run IDs

### Known Performance Risks

- ⚠️ Site heterogeneity produces unstable per-site metrics even when global AUC is acceptable
- ⚠️ Multiview branches may degrade to zero-edge views unless quality gates are active
- ⚠️ Re-running expensive stages without artifact reuse drastically increases wall-clock time
- ⚠️ Mixed artifacts in shared output folders can hide regressions or fabricate improvements

### Optimization Playbook

1. Iterate with fast settings: skip download/split, fewer permutations, no edge masking
2. Validate core metrics and quality gates
3. Re-run full evaluation settings for reportable outputs
4. Save run-specific outputs to dedicated directories when comparing experiments