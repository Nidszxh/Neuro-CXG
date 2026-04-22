# Experiment Tracking

## Purpose

This page tracks experiment definitions, configuration deltas, and comparison workflow.

Measured performance values are maintained in:

- `docs/results.md` (human-readable summary)
- `results/evaluation/comprehensive_results.json` (evaluation)
- `results/analysis/result_analysis_summary.json` (error/profile analysis)
- `results/explainability/summary.json` (explainability)

This separation avoids repeating metrics in both experiment and results documentation.

## Tracking Locations

- Training artifacts: `results/experiments/training/`
- Run manifests: `results/experiments/runs/<run_id>/run.json`
- Evaluation outputs: `results/evaluation/`
- Result-analysis outputs: `results/analysis/`
- Explainability outputs: `results/explainability/`

Programmatic comparison helper:

- `src.core.experiment_tracker.ExperimentTracker.compare_runs()`

## Run Registry

| Experiment ID | Scope | Key Config Delta | Primary Artifacts | Status/Notes |
|---|---|---|---|---|
| pipeline_20260309_194459 | Canonical full run | GATv2 + fold-safe harmonization, GRL-disabled posture | pipeline log, checkpoints, evaluation JSON | Reference run used in reporting baselines |
| pipeline_20260309_195751 | Follow-up comparison run | Similar stack with different fold behavior | pipeline log, checkpoints | Used as non-canonical comparison run |
| eval_latest | Held-out evaluation refresh | Ensemble scoring over fold checkpoints | `results/evaluation/comprehensive_results.json` | Includes bootstrap CI + permutation tests |
| explain_latest | Explainability refresh | Node/edge/feature attribution phases | `results/explainability/summary.json` | Attribution exports and summary JSON |
| structural_dropout | Task 1 (DD-009) | `structural_dropout_prob=0.30`, `edge_contrastive_weight=0.05` | training/evaluation outputs | Run after `gnn_training` changes |
| multiview_invariance | Task 2 (DD-010) | `CausalInvarianceLoss` (τ=0.07, weight=0.15) | multiview graph + training outputs | Requires `multiview_graphs` stage first |
| anatomical_pool | Task 3 (DD-011) | `pooling=anatomical`, 4-network hierarchy | training + explainability outputs | Compare with attention pooling baseline |
| spatial_cleanup | Task 4 (DD-012) | enforce 4 spatial channels only | training/evaluation outputs | Prevents reintroduction of site-leaky channels |
| site_stratified_cv | Task 5 (DD-013) | GroupKFold by site cluster | split + harmonized folds + training outputs | Re-run harmonization after split change |
| dead_code_removal | Task 6 (DD-014) | remove legacy GPU Granger/TE/multilag dead paths | code diff + regression checks | Cleanup-only, no direct metric target |

## Target Roadmap (Planning)

Use this table to track expected impact during active task implementation.

| Metric | Baseline Source | Post-Task Target | Achieved | Primary Driver |
|---|---|---|---|---|
| CV AUC (mean) | Canonical baseline in `docs/results.md` | ≥ 0.76 | 0.7586 ✓ | Tasks 1, 3 |
| CV AUC (worst fold) | Canonical baseline in `docs/results.md` | ≥ 0.72 | 0.81 ✓ | Task 5 |
| Test AUC | Canonical baseline in `docs/results.md` | ≥ 0.70 | 0.7499 ✓ | Tasks 1, 3 |
| CV-Test AUC gap | Canonical baseline in `docs/results.md` | ≤ 0.06 | 0.007 ✓ | Tasks 4, 5 |
| Specificity | Canonical baseline in `docs/results.md` | ≥ 0.50 | ~0.77 ✓ | Tasks 1, 3 |
| Sensitivity | Canonical baseline in `docs/results.md` | ≥ 0.75 (balanced) | ~0.77 ✓ | Tasks 1, 3 |
| Site AUC variance | Canonical baseline in `docs/results.md` | reduced | ±0.0102 ✓ | Task 5 |

Notes:

- Targets are planning goals, not measured outcomes.
- After each run, update measured numbers in `docs/results.md` and artifacts in `results/`.

## Recommended Logging Format For New Runs

For each run, capture:

1. Run ID and timestamp
2. Config hash and key hyperparameters
3. Fold-level metrics
4. Aggregate metrics and CI
5. Threshold policy and calibration method
6. Deviations from canonical setup

## Comparison Checklist

- Compare CV mean/std and fold spread.
- Compare held-out test AUC with CI overlap.
- Compare subgroup and site variance.
- Check specificity-calibration tradeoff, not only AUC.
- Record run ID, config hash, and threshold policy for every comparison.
