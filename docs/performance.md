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

### Latest Evaluation Bundle

From `results/evaluation/comprehensive_results.json`:

- ensemble AUC: 0.7268
- AUPRC: 0.7194
- F1: 0.6711
- accuracy: 0.6818
- sensitivity: 0.6329
- specificity: 0.7333

### Seed Stability Condition C

From `results/analysis/seed_stability_C/summary.json` aggregates:

- test AUC mean: 0.7592 (std 0.0019)
- CV mean AUC: 0.5728 (std 0.0027)
- signed gap (CV - test): -0.1864

### Latest Result Analysis Summary

From `results/analysis/result_analysis_summary.json`:

- overall AUC: 0.5826
- overall accuracy: 0.5097

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
