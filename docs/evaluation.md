# Evaluation Protocol

## Objective
Evaluate ASD vs Control classification quality, robustness, and statistical significance on held-out test data.

## Inputs
- Trained fold checkpoints from models/checkpoints/
- Test split graphs from graph_factory dataset loader
- Manifest metadata for subgroup and site analysis

## Main Metrics
- AUC (ROC area)
- AUPRC (average precision)
- F1 score
- Accuracy
- Sensitivity and Specificity

The evaluation runner now reports both F1-optimised and Youden J-optimised operating points, with `EVAL_THRESHOLD_POLICY` selecting which one is treated as the headline result.

## Metric Definitions
- Sensitivity = TP / (TP + FN)
- Specificity = TN / (TN + FP)
- F1 = 2 * (precision * recall) / (precision + recall)

## Protocol Steps
1. Load fold checkpoints.
2. Generate fold probabilities on test set.
3. Build ensemble probability (validation-informed weighting/thresholding).
4. Compute point metrics at both the F1 threshold and the Youden threshold.
5. Compute bootstrap confidence intervals.
6. Run permutation significance tests:
   - Global label permutation
   - Within-site permutation
7. Run subgroup analysis (sex, age, selected sites).
8. Compare against baseline models.

## Current Reference Artifact
- results/evaluation/comprehensive_results.json

## Edge Cases Handled
- Empty prediction batches
- NaN probabilities
- Single-class edge cases for AUC calculations
- Small subgroup/site cohorts where AUC can be undefined

## Reporting Guidelines
- Always report confidence intervals with test metrics.
- Include permutation p-values for significance claims.
- Separate CV metrics from held-out test metrics to avoid inflation.
- When comparing runs, keep the threshold policy explicit because F1 and specificity can move in opposite directions.
