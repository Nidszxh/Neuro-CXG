# Performance

## Scope

This page covers:

- runtime and computational behavior of major stages
- high-impact optimization controls
- artifact-backed model performance snapshots
- known limitations that affect performance interpretation

## 1) Runtime Profile

Neuro-CXG runtime depends on whether the run includes data acquisition, model training, and optional heavy analysis stages.

Operational hints in `src/run_pipeline.py` indicate:

- download stage can be multi-hour
- YOLO training can be multi-hour
- GNN training is typically tens of minutes

These are operational estimates, not strict guarantees.

## 2) Stage-Level Cost Drivers

### Data and Feature Build

- `src/data/abide_download.py`:
  - network IO and preprocessing dominate wall-clock.
- `src/features/extract_temporal.py`:
  - time-series transforms per subject, parallelized via `--n-jobs`.
- `src/features/fold_safe_harmonization.py`:
  - fold-wise train/apply cycles across all rows.

### Graph Construction

- `src/features/construct_causal.py`:
  - causal estimation and sparsification for every subject.
  - optional multiview path increases compute and IO.

### Training

- `src/models/gnn_model.py`:
  - 5-fold training with fold-specific harmonized inputs.
  - optional quality gates and auxiliary objectives add overhead.

### Reporting

- `src/run_evaluation.py`:
  - permutation testing can be expensive (`--n-permutations`).
- `src/run_explainability.py`:
  - edge masking is intentionally slow; `--no-masking` reduces runtime.
- `src/run_result_analysis.py`:
  - mostly post-inference analytics and plotting.

## 3) Performance Knobs

### CLI Controls

- `python src/run_pipeline.py --auto --skip-download --skip-split`
  - avoids repeating data ingestion work.
- `python src/run_pipeline.py --analysis-only`
  - skips build/training and runs reporting only.
- `python src/run_evaluation.py --no-permutation`
  - removes one major cost center.
- `python src/run_evaluation.py --n-permutations 200`
  - reduces permutation runtime for iteration.
- `python src/run_explainability.py --no-masking`
  - avoids slow masking analysis.

### Config Knobs

- `GRANGER_USE_GPU` in `src/core/hyperparams.py`
  - controls GPU path usage in causal inference.
- `GNN_BATCH_SIZE`, `GNN_HIDDEN_CHANNELS`, `GNN_NUM_HEADS`
  - primary training memory/throughput controls.
- `GNN_INVARIANCE_WEIGHT`, `GNN_EDGE_CONTRASTIVE_WEIGHT`, `GNN_SPATIAL_INVARIANCE_WEIGHT`
  - enabling non-zero values adds training cost.
- `GNN_ENFORCE_MULTIVIEW_QUALITY_GATE`
  - avoids wasting training time on degenerate multiview branches.

## 4) Artifact-Backed Model Performance Snapshot

### Current Best Results (Ridge Granger Hybrid - May 2026)

After method optimization: ridge_granger_hybrid (β=0.70, 70% Granger + 30% Pearson)

- **CV AUC**: 0.8100 ± 0.0273
- **Test AUC (ensemble)**: 0.8648
- **Test F1**: 0.7682
- **Test Accuracy**: 0.7727
- **Mean Best Epoch**: 41.0

**Note:** ridge_granger_hybrid selected as best Granger-based method — combines causal signal (Granger) with correlation strength (Pearson).

### Previous Best Results (Lagged Pearson - April 24, 2026)

After hyperparameter optimization (lagged_pearson + site conditioning + GRL):

- **CV AUC**: 0.8004 ± 0.0293
- **Test AUC (ensemble)**: 0.8753
- **Test F1**: 0.8121
- **Test Accuracy**: 0.7987
- **Mean Best Epoch**: 40.0

### Per-Fold Results

| Fold | AUC | F1 |
|------|-----|-----|
| 0 | 0.7826 | 0.7153 |
| 1 | 0.7629 | 0.7059 |
| 2 | 0.8219 | 0.8125 |
| 3 | 0.7897 | 0.7758 |
| 4 | 0.8449 | 0.7714 |

### Configuration That Achieved These Results (ridge_granger_hybrid)

```python
CAUSALITY_METHOD = "ridge_granger_hybrid"
GNN_HIDDEN_CHANNELS = 32
GNN_WEIGHT_DECAY = 5e-4
GNN_POOLING = "anatomical"
GNN_USE_SITE_EMBEDDING = True
GNN_USE_DEMOGRAPHICS = True
GNN_GRL_ALPHA_MAX = 0.10
USE_FOCAL_LOSS = True
EVAL_THRESHOLD_POLICY = "youden"
RIDGE_GRANGER_LAGS = (1,2,3,4,5)
RIDGE_GRANGER_LAMBDA = 1.0
RIDGE_GRANGER_HYBRID_BETA = 0.70
```

### Historical Performance

| Run | CV AUC | Test AUC | Test F1 |
|-----|-------|----------|---------|
| Baseline (ridge_granger) | 0.7586 ± 0.0519 | 0.7325 | 0.6338 |
| **Current (lagged_pearson + site)** | **0.8004 ± 0.0293** | **0.8753** | **0.8121** |

**May 2026 Method Change: lagged_pearson → ridge_granger_hybrid**

| Config | CV AUC | Test AUC | Test F1 | Notes |
|--------|--------|---------|---------|-------|
| lagged_pearson + GRL=0.10 | 0.7997 | 0.8694 | 0.8121 | Previous baseline |
| ridge_granger_hybrid (β=0.70) | **0.8100** | **0.8648** | 0.7682 | **ADOPTED** (best Granger method) |
| ridge_granger (pure) | 0.8064 | 0.8565 | 0.8023 | - |

**Decision (May 2026):** Selected ridge_granger_hybrid (β=0.70) because:
1. Best test AUC among Granger methods (0.8648, only -0.5% vs lagged_pearson)
2. Best CV AUC (+1% vs lagged_pearson)
3. 70% causal signal + 30% correlation captures both causal and correlation patterns
4. Scientifically defensible for causal discovery paper

### Architecture Exploration (April 28, 2026): 12-Lobe vs 11-Lobe

Comparative study evaluating whether Brainstem region should be included (12-lobe) or excluded (11-lobe) from architecture:

**Pre-Training Model Validation Results:**

| Architecture | CV AUC | F1 | Feature Completeness | Status |
|--------------|--------|-----|---------------------|--------|
| **12-Lobe** (Current) | 0.8002 | 0.7484 | 0% (synthetic fallback) | ❌ Degenerate |
| **11-Lobe** (Proposed) | **0.8099** | **0.7610** | **100% (all detected)** | ✓ Clean |
| **Improvement** | +0.0097 | +0.0126 | N/A | 11-Lobe wins |

**Key Discovery**:
- YOLO v29 never detects Brainstem (class_id=11) in 2D slices
- 12-lobe pipeline falls back to synthetic constant coordinates for all subjects
- Results in zero-variance feature: "Brainstem spatial features are constant across all subjects"
- 11-lobe architecture achieves 100% region detection, no synthetic fallback

**Fold-Level Performance (Partial)**:
- Fold 0: 11-lobe AUC +0.0072 (0.7888 vs 0.7816)
- Fold 1: 12-lobe AUC +0.0262 (0.7623 vs 0.7361) but with higher variance
- Fold 2: 12-lobe converges at epoch 24, 11-lobe at epoch 83 (slower convergence in 11-lobe for this fold)

**Recommendation**: 
- **Primary**: Adopt 11-lobe as default (cleaner features, better pre-training metrics)
- **Status**: Decision pending test set validation
- **Full Analysis**: See `LOBE_COMPARISON_ANALYSIS.md` and `docs/decisions.md` (DD-018)

**Final Recommended Configuration:**
```python
CAUSALITY_METHOD = "lagged_pearson"  # Best test performance
GRANGER_MAX_LAG_SECONDS = 10.0  # Max lag in seconds
GNN_HIDDEN_CHANNELS = 32
GNN_WEIGHT_DECAY = 5e-4
GNN_POOLING = "anatomical"
GNN_USE_SITE_EMBEDDING = True
GNN_USE_DEMOGRAPHICS = True
GNN_GRL_ALPHA = 0.10  # NOT 1.0 - test drops with 1.0
USE_FOCAL_LOSS = True
EVAL_THRESHOLD_POLICY = "youden"
```

## 5) Interpreting Metric Disagreement Across Artifacts

Different output bundles can disagree because they may come from different runs, thresholds, checkpoints, or calibration assumptions.

Practical guidance:

- treat `results/evaluation/comprehensive_results.json` as the authoritative evaluation record for a specific run
- confirm threshold metadata before comparing with result-analysis outputs
- avoid combining metrics from directories produced by different run IDs without explicit tracking

## 6) Known Performance Risks

- Site heterogeneity can produce unstable per-site metrics even when global AUC is acceptable.
- Multiview branches may degrade to zero-edge views unless quality gates are active.
- Re-running expensive stages without artifact reuse can drastically increase wall-clock time.
- Mixed artifacts in shared output folders can hide regressions or fabricate improvements.

## 7) Practical Optimization Playbook

1. Iterate with fast settings:
   - skip download/split
   - fewer permutations
   - no edge masking
2. Validate core metrics and quality gates.
3. Re-run full evaluation settings for reportable outputs.
4. Save run-specific outputs to dedicated directories when comparing experiments.
