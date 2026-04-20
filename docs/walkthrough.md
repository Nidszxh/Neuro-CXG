# Walkthrough

## Purpose

This walkthrough provides a reproducible end-to-end path for running Neuro-CXG and validating outputs at each checkpoint.

## Scenario

Use this when:

- environment is already set up
- repository artifacts exist locally
- you want a clean, source-aligned run sequence

## 0) Preflight

```bash
pip install -r requirements.txt
python -c "from src.core.config import validate_environment; validate_environment()"
python src/run_pipeline.py --dry-run
```

Expected:

- environment validation passes
- dry-run prints stage plan without execution

## 1) Build Data, Features, Graphs, and Train

### Standard Non-Interactive Run

```bash
python src/run_pipeline.py --auto
```

### Faster Iteration Variant (reuse existing download/split)

```bash
python src/run_pipeline.py --auto --skip-download --skip-split
```

## 2) Checkpoint Verification After Training

Verify core artifacts exist:

```bash
ls data/metadata/master_manifest.csv
ls data/metadata/node_attributes_harmonized.csv
ls data/metadata/harmonized_folds_cv/harmonized_fold_0.csv
ls data/processed/causal_graphs | head
ls models/checkpoints/best_model_fold0.pt
```

Expected:

- manifest and harmonized metadata present
- per-subject graph files present
- fold checkpoints present

## 3) Run Post-Training Reports

```bash
python src/run_evaluation.py
python src/run_explainability.py
python src/run_result_analysis.py
```

Or run via orchestrator analysis mode:

```bash
python src/run_pipeline.py --analysis-only
```

## 4) Verify Reporting Artifacts

```bash
ls results/evaluation/comprehensive_results.json
ls results/explainability/summary.json
ls results/analysis/result_analysis_summary.json
```

Expected:

- all three summary JSON artifacts exist
- each contains machine-readable run outputs for its report family

## 5) Optional Branches

### Site-Stratified CV Protocol

```bash
python -m src.data.split --site-stratified-cv
python -m src.features.fold_safe_harmonization
python -m src.models.gnn_model
```

### Multiview Graph Path

```bash
python src/run_pipeline.py --auto --multiview
```

## 6) Fast Debug Commands

```bash
# Rebuild feature/graph stack without full reset
python src/run_pipeline.py --auto --regenerate-features

# Skip expensive evaluation permutations for quick checks
python src/run_evaluation.py --no-permutation

# Skip slow edge masking in explainability
python src/run_explainability.py --no-masking
```

## 7) Common Run Integrity Checks

If training fails, run:

```bash
python -c "from src.core.config import validate_gnn_training_inputs; validate_gnn_training_inputs()"
```

If graph quality is suspicious, run:

```bash
python -m src.analysis.diagnose_dead_lobes --split train
python -m src.validation.pipeline_checks
```

## 8) Minimal Reproducible Command Set

For a compact reproducible run with existing data:

```bash
python src/run_pipeline.py --auto --skip-download --skip-split
python src/run_evaluation.py
python src/run_explainability.py --no-masking
python src/run_result_analysis.py
```

This sequence regenerates train/eval/report outputs while avoiding the largest repeated ingestion cost.
