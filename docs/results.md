# Results Summary

## Reporting Policy
- Canonical benchmark values are sourced from historical pipeline logs.
- Current snapshot values are sourced from the latest on-disk JSON outputs.
- Keep both views side by side to avoid accidental metric drift.

## Canonical Reference Run
- Run ID: pipeline_20260309_194459
- Status: canonical benchmark run used in project reporting

### Detector Performance (YOLO v29)
- mAP50-95: 0.9598
- mAP50: 0.9943
- Precision: 0.9873
- Recall: 0.9838

### Cross-Validation (5-fold)
- CV AUC: 0.7434 +- 0.0417
- Fold AUCs: [0.7317, 0.7576, 0.7606, 0.6709, 0.7964]

### Per-Fold CV Details (Canonical)
| Fold | CV AUC | Best Epoch |
|---|---|---|
| 0 | 0.7317 | 42 |
| 1 | 0.7576 | 81 |
| 2 | 0.7606 | 75 |
| 3 | 0.6709 | 72 |
| 4 | 0.7964 | 75 |

### Held-Out Test (Canonical Log)
| Metric | Value | Notes |
|---|---|---|
| AUC | 0.6487 | 95% CI [0.5618, 0.7300] |
| F1 | 0.6738 | Thresholded classification |
| AUPRC | 0.6459 | Average precision |
| Sensitivity | 0.7975 | Recall for ASD |
| Specificity | 0.4079 | Recall for Control |
| Permutation p-value | 0.0020 | Global significance |
| Within-site p-value | 0.0010 | Site-aware significance |

### Performance Timeline (Historical)
| Date | CV AUC | Test AUC | Major Change |
|---|---|---|---|
| 2026-02-15 | 0.6194 +- 0.0641 | 0.5398 | Baseline with pre-fix issues |
| 2026-03-08 | 0.6309 | N/A | Dead-lobe NaN fix |
| 2026-03-09 | 0.7434 +- 0.0417 | 0.6487 | P0/P1 fixes plus GRL disabled |

### Subgroup Snapshot (Canonical)
| Subgroup | N | AUC |
|---|---|---|
| Male | 132 | 0.6662 |
| Female | 23 | 0.5923 |
| Age < 15 | 88 | 0.6580 |
| Age >= 15 | 67 | 0.6348 |
| OHSU site (best) | 9 | 0.9500 |
| Site 16 (worst) | 16 | 0.3281 |

## Current On-Disk Evaluation Snapshot
Source: results/evaluation/comprehensive_results.json

- AUC: 0.6516 with 95% CI [0.5603, 0.7325]
- AUPRC: 0.6689 with 95% CI [0.5938, 0.7579]
- F1: 0.6849 with 95% CI [0.6516, 0.7123]
- Accuracy: 0.5548
- Sensitivity: 0.9494
- Specificity: 0.1447
- Threshold: 0.4644
- Significance: p=0.001 (global and within-site permutation)

## Why Canonical and On-Disk Numbers May Differ
A later run may overwrite checkpoint files while historical logs preserve prior best runs. Use pipeline logs and explicit run IDs for authoritative comparisons.

## Explainability Highlights
Source: results/explainability/summary.json

- Top regions: Temporal, Limbic, Occipital, Brainstem, Subcortical, Frontal_Orbital
- Top networks: DMN, Social, Salience, Visual, Subcortical
- Differential edge pattern is dominated by connections into Brainstem

## Error and Site Analysis Highlights
Source: results/analysis/result_analysis_summary.json

- Overall accuracy: 0.5742
- Overall AUC: 0.6969
- Site-level variability remains high across small cohorts
- Misclassification profile is skewed toward false positives in current thresholding regime

## Improvement Drivers (Canonical Progress)
| Change | Estimated Impact |
|---|---|
| Disabled unstable high-alpha GRL setting | +5.3 pp |
| Added DX_GROUP as protected ComBat covariate | about +4 pp |
| Fixed dead-lobe NaN handling before PCA | +1.2 pp |
| Added Granger multiple-testing correction | Stability gain |
| Applied fold-safe NaN handling | Anti-leakage gain |
| Stabilized PCA sign handling | Consistency gain |

## Literature Comparison
| System | Method | Performance |
|---|---|---|
| Heinsfeld et al. 2018 | Deep autoencoder + SVM | about 70% accuracy |
| Ktena et al. 2018 | Spectral GCN on correlation matrices | about 70-73% AUC |
| Neuro-CXG | GATv2 on directed causal graphs | CV 0.7434, Test 0.6487 |

## Active Risks and Open Issues
- CV-test gap remains material and likely tied to residual site effects.
- Per-site AUC spread remains wide across small cohorts.
- Specificity remains low in thresholded classification.

## Recommended Reporting Template
When presenting results, include:
1. Run ID and timestamp.
2. CV mean +- std and per-fold values.
3. Held-out test metrics with confidence intervals.
4. Permutation p-values.
5. Subgroup and site analysis caveats.
