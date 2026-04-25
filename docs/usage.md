# Usage

## Primary Entry Point

Use the orchestrator for most workflows:

```bash
python src/run_pipeline.py
```

By default, `run_pipeline.py` is interactive. Use `--auto` for non-interactive execution.

## Core Pipeline Commands

```bash
# Show stage plan only
python src/run_pipeline.py --dry-run

# Full non-interactive run
python src/run_pipeline.py --auto

# Reuse existing downloaded and split data
python src/run_pipeline.py --auto --skip-download --skip-split

# Run only post-training analysis stages
python src/run_pipeline.py --analysis-only

# Run only visualization stage
python src/run_pipeline.py --visualizations-only
```

## High-Impact Flags

- `--multiview` - run optional multiview graph generation stage
- `--site-stratified-cv` - regenerate `cv_fold` with site-stratified grouping
- `--force-reset` - clear intermediate feature/graph artifacts before rebuild
- `--regenerate-features` - rebuild feature and graph stages without full reset
- `--full-src` - include optional extended audit/diagnostic/experiment stages

Skip controls are available for most stages (for example `--skip-evaluation`, `--skip-explainability`, `--skip-gnn`).

## Stage-Level Script Commands

Use these for targeted debugging or partial reruns.

```bash
# Data
python -m src.data.abide_download
python -m src.data.split
python -m src.data.split --site-stratified-cv

# Labels and detection
python -m src.pipelines.generate_labels
python -m src.pipelines.roi_detection

# Features and harmonization
python -m src.features.extract_spatial
python -m src.features.extract_spatial_atlas
python -m src.features.extract_temporal --n-jobs -1
python -m src.features.fold_safe_harmonization

# Graph construction
python -m src.features.construct_causal --n-jobs -1
python -m src.features.construct_causal --multiview

# Training
python -m src.models.gnn_model
```

## Post-Training Commands

```bash
python src/run_evaluation.py
python src/run_explainability.py
python src/run_result_analysis.py
```

Additional analysis scripts:

```bash
python -m src.analysis.visualizations
python -m src.analysis.visualize_causal_graph --auto-pair
python -m src.analysis.subject_analysis
python -m src.validation.pipeline_checks
```

## Script-Specific Options

### Evaluation

```bash
python src/run_evaluation.py --no-permutation
python src/run_evaluation.py --n-permutations 200
python src/run_evaluation.py --no-baselines --no-subgroups
python src/run_evaluation.py --output-dir results/evaluation_custom
```

### Explainability

```bash
python src/run_explainability.py --fold 2
python src/run_explainability.py --phases node edge
python src/run_explainability.py --no-masking
python src/run_explainability.py --output-dir results/explainability_custom
```

### Result Analysis

```bash
python src/run_result_analysis.py --n-cases 3
python src/run_result_analysis.py --no-heatmap
python src/run_result_analysis.py --no-severity
python src/run_result_analysis.py --output-dir results/analysis_custom
```

## Typical Workflows

### Full rebuild from current workspace state

```bash
python src/run_pipeline.py --auto --force-reset
```

### Recompute only analysis from existing checkpoints

```bash
python src/run_pipeline.py --analysis-only
```

### Enable multiview artifacts for invariance experiments

```bash
python src/run_pipeline.py --auto --multiview
```

## Notes

- The runner resolves stage completion through registry sentinels and local artifact presence.
- Training enforces presence of fold-specific harmonized files before starting.
- `src/features/extract_spatial.py` uses YOLO weights from config; use atlas extraction only when intentionally bypassing detector outputs.
