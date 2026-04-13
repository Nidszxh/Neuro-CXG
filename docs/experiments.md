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
