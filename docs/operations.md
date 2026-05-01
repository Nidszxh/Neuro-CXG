# Operations

## Quick Triage Order

1. Run environment and input validation
2. Confirm required artifacts for the stage you are running
3. Check quality-gate failures before changing model hyperparameters
4. Regenerate only missing or stale upstream artifacts

---

## Part A: Failure Modes

### 1) Preflight and Input Failures

| Attribute | Details |
|-----------|---------|
| **Trigger** | Pipeline exits early at preflight or before training |
| **Causes** | Missing manifest/features, missing graph files, missing fold-specific harmonized CSVs, invalid atlas/lobe mapping |
| **Where enforced** | `validate_environment()` in `src/core/validators.py`, `validate_gnn_training_inputs()` in `src/core/validators.py` |
| **Recovery** | ```bash\npython -c \"from src.core.config import validate_environment; validate_environment()\"\npython src/run_pipeline.py --auto --skip-download --skip-split\n``` |
| **Status** | Resolved — run harmonization explicitly if needed |

---

### 2) Feature Shape or Schema Mismatch

| Attribute | Details |
|-----------|---------|
| **Trigger** | Dataset initialization fails with temporal/spatial shape mismatch; runtime errors assembling `Data` objects |
| **Causes** | Feature extraction changed but registry values not aligned; stale metadata files |
| **Where enforced** | `_validate_feature_dimensions()` in `src/features/graph_factory.py`, channel definitions in `src/core/feature_registry.py` |
| **Recovery** | ```bash\npython src/run_pipeline.py --auto --regenerate-features\n``` |
| **Status** | Resolved — controlled reset if stale artifacts persist |

---

### 3) Graph Degeneracy Gate Failure

| Attribute | Details |
|-----------|---------|
| **Trigger** | Training aborts with graph quality gate failure; too many subjects dropped due to invalid/zero-edge graphs |
| **Causes** | Causality estimation produced sparse/zero-edge graphs; NaN/Inf or short time-series cascades into weak graphs |
| **Where enforced** | `_assess_graph_degeneracy()` in `src/models/gnn_model.py`, subject drop-rate gate in `src/features/graph_factory.py` |
| **Recovery** | ```bash\npython -m src.features.construct_causal --n-jobs -1\npython -m src.validation.pipeline_checks\n``` |
| **Status** | Resolved — rebuild graphs and validate upstream artifacts |

---

### 4) Multiview Quality Gate Disables Invariance

| Attribute | Details |
|-----------|---------|
| **Trigger** | Logs indicate multiview invariance disabled due to failing views; non-base views show high zero-edge rates |
| **Causes** | Degenerate multiview branches; stale multiview artifacts from older graph settings |
| **Where enforced** | Multiview checks in `src/models/gnn_model.py`, generation-time gate settings in `src/core/hyperparams.py` |
| **Recovery** | ```bash\npython src/run_pipeline.py --auto --multiview --regenerate-features\n``` |
| **Status** | Resolved — regenerate multiview artifacts or disable invariance |

---

### 5) Site-Stratified CV Mismatch

| Attribute | Details |
|-----------|---------|
| **Trigger** | Training/evaluation behavior changes unexpectedly after `--site-stratified-cv`; fold harmonization inconsistent with current `cv_fold` |
| **Causes** | `cv_fold` regenerated but harmonized fold files not refreshed |
| **Where enforced** | `run_site_stratified_split()` in `src/data/split.py`, fold file checks in `src/models/gnn_model.py` |
| **Recovery** | ```bash\npython -m src.data.split --site-stratified-cv\npython -m src.features.fold_safe_harmonization\npython -m src.models.gnn_model\n``` |
| **Status** | Resolved — regenerate folds after CV policy change |

---

### 6) Checkpoint Availability Failures

| Attribute | Details |
|-----------|---------|
| **Trigger** | Evaluation/explainability/result-analysis cannot find fold checkpoints; fold checkpoints partially missing |
| **Causes** | Training not run or interrupted; checkpoint directory mismatch |
| **Where enforced** | Model-loading logic in `src/run_evaluation.py`, `src/run_explainability.py`, active checkpoint resolution in `get_active_checkpoint_dir()` |
| **Recovery** | ```bash\npython -m src.models.gnn_model\npython src/run_evaluation.py\n``` |
| **Status** | Resolved — retrain from checkpoints |

---

### 7) Threshold or Metadata Drift Between Reports

| Attribute | Details |
|-----------|---------|
| **Trigger** | `run_result_analysis.py` outputs differ from evaluation metrics; warnings about missing evaluation metadata |
| **Causes** | `comprehensive_results.json` missing or from different run; policy/threshold changed without regenerating artifacts |
| **Where enforced** | Metadata load in `src/run_result_analysis.py`, threshold policy constants in `src/core/hyperparams.py` |
| **Recovery** | ```bash\npython src/run_evaluation.py\npython src/run_result_analysis.py\n``` |
| **Status** | Resolved — keep outputs in same run-specific directory |

---

### 8) Slow or Stalled Analysis Stages

| Attribute | Details |
|-----------|---------|
| **Trigger** | Evaluation or explainability takes unexpectedly long |
| **Causes** | High permutation count; edge masking enabled in explainability |
| **Recovery** | ```bash\npython src/run_evaluation.py --no-permutation\npython src/run_evaluation.py --n-permutations 200\npython src/run_explainability.py --no-masking\n``` |
| **Status** | Workaround available |

---

### 9) Safety Misconfiguration in YOLO Augmentation

| Attribute | Details |
|-----------|---------|
| **Trigger** | Warnings about anatomy-destroying augmentation |
| **Causes** | Non-zero `YOLO_FLIPLR` or `YOLO_DEGREES` |
| **Where enforced** | Warning path in `validate_environment()` (`src/core/validators.py`) |
| **Recovery** | ```bash\n# Reset to medical-safe defaults\nYOLO_FLIPLR = 0.0\nYOLO_DEGREES = 0.0\n``` |
| **Status** | Resolved — defaults are intentionally conservative |

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

### Stage-Level Cost Driver Table

| Stage | Primary Cost | Parallelism Available |
|-------|-------------|----------------------|
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

**CLI Controls:**

| Command | Effect |
|---------|--------|
| `--skip-download --skip-split` | Avoids repeating data ingestion |
| `--analysis-only` | Skips build/training |
| `--no-permutation` | Removes major cost center |
| `--n-permutations 200` | Reduces permutation runtime |
| `--no-masking` | Avoids slow edge masking |

**Config Knobs:**

| Parameter | Effect |
|-----------|--------|
| `GRANGER_USE_GPU` | GPU path in causal inference |
| `GNN_BATCH_SIZE` | Training memory/throughput |
| `GNN_HIDDEN_CHANNELS` | Model capacity |
| `GNN_NUM_HEADS` | Attention heads |
| `GNN_INVARIANCE_WEIGHT` | Enables training cost if >0 |

### Artifact-Backed Model Performance Snapshot

**Current Best (May 2026):**

| Metric | Value | Notes |
|--------|-------|-------|
| **CV AUC** | 0.8101 ± 0.0274 | 5-fold |
| **Test AUC** | **0.8651** | Ensemble, 95% CI [~0.78–0.90] |
| **Test F1** | **0.7651** | Youden threshold |
| **Test Accuracy** | **0.7727** | |

**Method:** ridge_granger_hybrid (β=0.70)

**Per-Fold Results (12-Lobe):**

| Fold | AUC | F1 |
|------|-----|-----|
| 1 | 0.8039 | 0.7671 |
| 2 | 0.7833 | 0.7077 |
| 3 | 0.8058 | 0.7682 |
| 4 | 0.7951 | 0.6829 |
| 5 | 0.8626 | 0.7692 |

**Configuration that achieved these results:**

```python
CAUSALITY_METHOD = "ridge_granger_hybrid"
GNN_HIDDEN_CHANNELS = 32
GNN_WEIGHT_DECAY = 5e-4
GNN_POOLING = "anatomical"
GNN_USE_SITE_EMBEDDING = True
GNN_USE_DEMOGRAPHICS = True
GNN_GRL_ALPHA = 0.10  # NOT 1.0 - test drops with 1.0
USE_FOCAL_LOSS = True
EVAL_THRESHOLD_POLICY = "youden"
RIDGE_GRANGER_HYBRID_BETA = 0.70
```

### Historical Performance Table

| Run | CV AUC | Test AUC | Test F1 | Change |
|-----|-------|---------|---------|--------|
| Baseline (ridge_granger) | 0.7586 ± 0.0519 | 0.7325 | 0.6338 | Initial |
| lagged_pearson + GRL=0.10 | 0.8004 ± 0.0293 | 0.8753 | 0.8121 | Method switch |
| 12-Lobe Final | 0.7997 ± 0.0294 | 0.8694 | 0.8000 | 12-lobe approved |
| **ridge_granger_hybrid** | **0.8101 ± 0.0274** | **0.8651** | **0.7651** | Current best |

### Metric Disagreement Guidance

Different output bundles can disagree because they may come from different runs, thresholds, checkpoints, or calibration assumptions.

**Practical guidance:**
- Treat `results/evaluation/comprehensive_results.json` as authoritative
- Confirm threshold metadata before comparing with result-analysis outputs
- Avoid combining metrics from directories produced by different run IDs

### ⚠️ Known Performance Risks

- ⚠️ Site heterogeneity produces unstable per-site metrics even when global AUC is acceptable
- ⚠️ Multiview branches may degrade to zero-edge views unless quality gates are active
- ⚠️ Re-running expensive stages without artifact reuse drastically increases wall-clock time
- ⚠️ Mixed artifacts in shared output folders can hide regressions or fabricate improvements

### Practical Optimization Playbook

1. Iterate with fast settings: skip download/split, fewer permutations, no edge masking
2. Validate core metrics and quality gates
3. Re-run full evaluation settings for reportable outputs
4. Save run-specific outputs to dedicated directories when comparing experiments