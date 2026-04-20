# Evaluation

## Scope

This document covers post-training reporting scripts:

- `src/run_evaluation.py`
- `src/run_explainability.py`
- `src/run_result_analysis.py`

These scripts are also wired as stages in `src/pipeline/registry.py` and can be run through `src/run_pipeline.py`.

## 1) Comprehensive Evaluation

Entry point:

```bash
python src/run_evaluation.py
```

### What It Computes

- AUC-weighted fold ensemble on test graphs
- full metric suite (AUC, AUPRC, F1, accuracy, sensitivity, specificity)
- bootstrap confidence intervals
- permutation significance tests:
   - global label shuffling
   - within-site label shuffling
- subgroup analysis (sex, age bins, top-represented sites)
- baseline comparisons (SVM, Random Forest, flat MLP)

### Threshold Policy

Evaluation uses `EVAL_THRESHOLD_POLICY` from `src/core/hyperparams.py`:

- `f1` - F1-optimized threshold
- `youden` - Youden J threshold
- `fixed` - locked threshold from `EVAL_FIXED_THRESHOLD`

### Useful Options

```bash
python src/run_evaluation.py --no-permutation
python src/run_evaluation.py --n-permutations 200
python src/run_evaluation.py --no-baselines --no-subgroups
python src/run_evaluation.py --output-dir results/evaluation_custom
```

### Outputs

Default output directory: `results/evaluation/`

- `comprehensive_results.json`
- `comprehensive_results.csv`
- `permutation_test_global.png`
- `permutation_test_within_site.png`
- `subgroup_analysis.png`
- `baseline_comparison.png`

## 2) Explainability Pipeline

Entry point:

```bash
python src/run_explainability.py
```

### Explainability Phases

- node importance
- edge importance
- feature attribution
- literature validation

The script can auto-select fold by highest recorded validation AUC if `--fold` is not provided.

### Useful Options

```bash
python src/run_explainability.py --fold 3
python src/run_explainability.py --phases node edge
python src/run_explainability.py --no-masking
python src/run_explainability.py --output-dir results/explainability_custom
```

### Outputs

Default output directory: `results/explainability/`

- `node/` (node and attention artifacts)
- `edge/` (edge attribution artifacts)
- `features/` (feature attribution plots)
- `literature/` (literature cross-reference outputs)
- `summary.json`

## 3) Result Interpretation

Entry point:

```bash
python src/run_result_analysis.py
```

### What It Produces

- per-subject predictions and confidence table
- misclassification profiling
- site-effect summary plots
- confidence/calibration plots
- optional severity correlation plots
- case studies (text + csv)

### Threshold And Calibration Alignment

`run_result_analysis.py` attempts to align with evaluation metadata from:

- `results/evaluation/comprehensive_results.json`

If metadata is missing, it recomputes thresholds and uses fallbacks from checkpoints/config.

### Useful Options

```bash
python src/run_result_analysis.py --n-cases 3
python src/run_result_analysis.py --no-heatmap
python src/run_result_analysis.py --no-severity
python src/run_result_analysis.py --output-dir results/analysis_custom
```

### Outputs

Default output directory: `results/analysis/`

- `per_subject_predictions.csv`
- `misclassification_analysis.png`
- `site_effects.png`
- `site_bias_heatmap.png` (unless disabled)
- `calibration.png`
- `severity_correlation.png` (unless disabled)
- `case_studies.csv`
- `case_studies.txt`
- `result_analysis_summary.json`

## 4) Running Through Pipeline Stages

To run only these post-training reports via orchestrator:

```bash
python src/run_pipeline.py --analysis-only
```

To run all stages including reporting:

```bash
python src/run_pipeline.py --auto
```

## 5) Evaluation Reproducibility Checklist

- confirm fold checkpoints exist in the active checkpoint directory
- keep threshold policy explicit in config
- avoid mixing outputs from different runs in the same output directory
- store and review `comprehensive_results.json` as the machine-readable record
