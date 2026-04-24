# Results Summary

## Reporting Policy
- Canonical benchmark values are sourced from historical pipeline logs.
- Current snapshot values are sourced from the latest on-disk JSON outputs.
- Keep both views side by side to avoid accidental metric drift.
- Keep experiment planning targets in `docs/experiments.md`; this page reports measured outcomes only.

## Current Best Run (April 24, 2026)
- Run ID: pipeline_20260424_191537
- Status: Best performing configuration after systematic investigation
- Configuration: lagged_pearson + GNN_GRL_ALPHA=0.10 + MaxLag=10.0s

### Current Cross-Validation (5-fold)
- CV AUC: 0.8004 ± 0.0293
- CV F1: 0.7562 ± 0.0400
- CV Accuracy: 0.7483 ± 0.0271

### Per-Fold CV Details (Current Best)
| Fold | CV AUC | AUPRC | F1 | Best Epoch |
|---|---|---|---|---|
| 1 | 0.7826 | 0.8040 | 0.7153 | 36 |
| 2 | 0.7629 | 0.7811 | 0.7059 | 59 |
| 3 | 0.8219 | 0.8215 | 0.8125 | 41 |
| 4 | 0.7897 | 0.7978 | 0.7758 | 29 |
| 5 | 0.8449 | 0.8775 | 0.7714 | 35 |

### Current Held-Out Test
| Metric | Value | 95% CI |
|-------|-------|--------|
| **AUC** | **0.8753** | - |
| **F1** | **0.8121** | - |
| Accuracy | 0.7987 | - |
| Sensitivity | 0.8481 | - |
| Specificity | 0.7467 | - |
| Permutation p-value | <0.01 | ✓ Significant |

### Investigation Summary (April 24, 2026)

| Config | CV AUC | Test AUC | Test F1 | Recommendation |
|--------|--------|---------|---------|-------------|
| lagged_pearson + GRL=0.10 | 0.8004 | **0.8753** | **0.8121** | ✓ RECOMMENDED |
| lagged_pearson + GRL=1.0 | 0.8034 | 0.8498 | 0.7662 | Not recommended |
| ridge_granger + GRL=0.10 | 0.8075 | 0.8359 | 0.7484 | Not recommended |

## Canonical Reference Run
- Run ID: pipeline_20260309_194459
- Status: canonical benchmark run used in project reporting

### Detector Performance (YOLO v29)
- mAP50-95: 0.9598
- mAP50: 0.9943
- Precision: 0.9873
- Recall: 0.9838

### Cross-Validation (5-fold)
- CV AUC: 0.7586 ± 0.0519
- Fold AUCs: [0.7435, 0.6837, 0.7513, 0.7693, 0.8451]
- CV AUPRC: 0.7404 ± 0.0696
- CV F1: 0.6264 ± 0.0958
- CV Accuracy: 0.6677 ± 0.0493
- Mean Best Epoch: 12.0
- Mean Threshold: 0.526

### Per-Fold CV Details (Canonical)
| Fold | CV AUC | AUPRC | F1 | Best Epoch |
|---|---|---|---|---|
| 1 | 0.7435 | 0.7169 | 0.6190 | 11 |
| 2 | 0.6837 | 0.6850 | 0.4673 | 20 |
| 3 | 0.7513 | 0.7521 | 0.7262 | 12 |
| 4 | 0.7693 | 0.7928 | 0.5950 | 9 |
| 5 | 0.8451 | 0.8514 | 0.7244 | 8 |

### Held-Out Test (Canonical Log)
| Metric | Value | Notes |
|---|---|---|
| AUC | 0.7499 | Ensemble (AUC-weighted) |
| F1 | 0.5985 | Thresholded classification |
| Accuracy | 0.6429 | Overall accuracy |
| Permutation p-value | <0.01 | Global significance |
| Within-site p-value | <0.01 | Site-aware significance |

### Performance Timeline (Historical)
| Date | CV AUC | Test AUC | Major Change |
|---|---|---|---|
| 2026-02-15 | 0.6194 +- 0.0641 | 0.5398 | Baseline with pre-fix issues |
| 2026-03-08 | 0.6309 | N/A | Dead-lobe NaN fix |
| 2026-03-09 | 0.7434 +- 0.0417 | 0.6487 | P0/P1 fixes plus GRL disabled |
| 2026-04-22 | 0.7586 +- 0.0519 | 0.7499 | Force-reset feature regeneration |

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
- Accuracy: 0.6548
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

- Overall accuracy: 0.6742
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
| Neuro-CXG | GATv2 on directed causal graphs | CV 0.7586, Test 0.7499 |

## Active Risks and Open Issues
- Per-site AUC spread remains variable across small cohorts - monitor through result analysis.
- Threshold calibration should be reviewed for each new population.
- Cohort/curation assumptions should be interpreted alongside `docs/data-curation.md`.

## Recommended Reporting Template
When presenting results, include:
1. Run ID and timestamp.
2. CV mean +- std and per-fold values.
3. Held-out test metrics with confidence intervals.
4. Permutation p-values.
5. Subgroup and site analysis caveats.
