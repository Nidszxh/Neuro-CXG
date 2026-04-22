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

### Current Best Results (April 22, 2026)

After hyperparameter optimization (lagged_pearson + site conditioning + GRL):

- **CV AUC**: 0.8001 ± 0.0293
- **Test AUC (ensemble)**: 0.8748
- **Test F1**: 0.8121
- **Test Accuracy**: 0.7987
- **Mean Best Epoch**: 40.0

### Per-Fold Results

| Fold | AUC | F1 |
|------|-----|-----|
| 0 | 0.7828 | 0.7206 |
| 1 | 0.7621 | 0.7183 |
| 2 | 0.8215 | 0.8075 |
| 3 | 0.7895 | 0.7758 |
| 4 | 0.8445 | 0.7714 |

### Configuration That Achieved These Results

```python
CAUSALITY_METHOD = "lagged_pearson"
GNN_HIDDEN_CHANNELS = 32
GNN_WEIGHT_DECAY = 5e-4
GNN_POOLING = "anatomical"
GNN_USE_SITE_EMBEDDING = True
GNN_USE_DEMOGRAPHICS = True
GNN_GRL_ALPHA_MAX = 1.0
USE_FOCAL_LOSS = True
EVAL_THRESHOLD_POLICY = "youden"
```

### Historical Performance

| Run | CV AUC | Test AUC | Test F1 |
|-----|-------|----------|---------|
| Baseline (ridge_granger) | 0.7586 ± 0.0519 | 0.7325 | 0.6338 |
| **Current (lagged_pearson + site)** | **0.8001 ± 0.0293** | **0.8748** | **0.8121** |

Key improvements:
- +0.04 CV AUC from lagged_pearson edges
- +0.14 Test AUC from site conditioning + strong GRL
- +0.18 F1 from Youden threshold policy
- Reduced variance (±0.052 → ±0.029)

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
