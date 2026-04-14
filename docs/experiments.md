# Experiment Tracking

## Purpose
Track experiments, configurations, and outcomes in a consistent format.

## Tracking Locations
- Training artifacts: results/experiments/training/
- Evaluation summaries: results/evaluation/comprehensive_results.json
- Result analysis summaries: results/analysis/result_analysis_summary.json
- Explainability summaries: results/explainability/summary.json
- Optional run tracker outputs (new): results/experiments/runs/<run_id>/run.json

## Experiment Log

| Experiment ID | Scope | Key Config | Main Outcome | Notes |
|---|---|---|---|---|
| pipeline_20260309_194459 | Canonical full run | GATv2, fold-safe harmonization, GRL off | CV AUC ~0.74, significant test AUC | Reference run in project docs |
| pipeline_20260309_195751 | Follow-up run | Similar pipeline, different fold behavior | Lower/stable compared to canonical | Used for comparison and diagnostics |
| eval_latest | Held-out evaluation | Ensemble over fold checkpoints | See comprehensive_results.json | Includes bootstrap CI + permutation tests |
| explain_latest | Explainability suite | Node/edge/feature attribution | Top regions and edges exported | See explainability summary JSON |
| structural_dropout | Task 1 (DD-009) | structural_dropout_prob=0.30, edge_contrastive_weight=0.05 | Target: test AUC ≥ 0.70, specificity ≥ 0.50 | Run after gnn_training with new training_utils |
| multiview_invariance | Task 2 (DD-010) | CausalInvarianceLoss τ=0.07, weight=0.15 | Target: reduced CV variance across folds | Requires multiview_graphs stage first |
| anatomical_pool | Task 3 (DD-011) | pooling=anatomical, 4 networks | Target: improved DMN/Salience attribution | Old pooling modes available for ablation |
| spatial_cleanup | Task 4 (DD-012) | NUM_SPATIAL_FEATURES=4 sentinel enforced | Target: CV-test gap ≤ 0.06 | conf_std/detection_count permanently excluded |
| site_stratified_cv | Task 5 (DD-013) | GroupKFold by scanner manufacturer × TR | Target: worst-fold CV AUC ≥ 0.72 | Requires re-running fold_safe_harmonization |
| dead_code_removal | Task 6 (DD-014) | GPU Granger / TE / multilag removed | Codebase cleanup; no metric impact expected | Reduces import surface by 3 dead functions |

## Recommended Logging Format for New Runs
For each new run, capture:
1. Run ID and timestamp
2. Config hash / key hyperparameters
3. Fold-level metrics
4. Aggregate summary and notes
5. Deviations from canonical setup

## How to Compare Runs
- Compare CV mean/std and fold spread.
- Compare held-out test AUC with CI overlap.
- Compare subgroup and site variance.
- Check if specificity and calibration improved.
