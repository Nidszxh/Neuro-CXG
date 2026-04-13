# Neuro-CXG

Causal Graph Neural Networks for ASD vs Control classification from resting-state fMRI.

Neuro-CXG is a configuration-driven, end-to-end pipeline that goes from ABIDE data ingestion to explainable graph-based predictions. It combines atlas-aware feature extraction, fold-safe harmonization, directed causal graph construction, and GATv2-based classification.

## Quick Start

```bash
pip install -r requirements.txt
python -c "from src.core.config import validate_environment; validate_environment()"
python src/run_pipeline.py --auto --skip-download --skip-split
```

Run evaluation and explainability:

```bash
python src/run_evaluation.py
python src/run_explainability.py
python src/run_result_analysis.py
```

## Current Status

Canonical benchmark run (pipeline_20260309_194459):
- CV AUC: 0.7434 +- 0.0417
- Test AUC: 0.6487, 95% CI [0.5618, 0.7300]

Latest on-disk evaluation snapshot (results/evaluation/comprehensive_results.json):
- AUC: 0.6516, 95% CI [0.5603, 0.7325]
- AUPRC: 0.6689
- F1: 0.6849

For details and caveats, see docs/results.md.

## Documentation Map

- docs/problem.md: problem statement, goals, constraints
- docs/architecture.md: system architecture and stage flow
- docs/decisions.md: engineering decision log and rationale
- docs/setup.md: environment setup and validation
- docs/usage.md: pipeline and stage-level commands
- docs/data.md: dataset flow, artifacts, and quality gates
- docs/evaluation.md: metric definitions and evaluation protocol
- docs/experiments.md: run tracking and comparison workflow
- docs/results.md: benchmark and latest metric summaries
- CHANGELOG.md: notable repository changes

## Core Pipeline

- Data ingest and split
- ROI detection / spatial extraction
- Temporal feature extraction
- Fold-safe ComBat harmonization
- Causal graph construction
- 5-fold GNN training
- Evaluation, explainability, and result analysis

Main orchestrator:

```bash
python src/run_pipeline.py --auto
```

## Project Structure

```text
src/
  core/          # config, paths, hyperparameters, registries
  data/          # data download/split/prep
  features/      # spatial/temporal extraction, harmonization, graph build
  models/        # GNN architecture and training
  pipelines/     # detector training and labeling
  analysis/      # diagnostics and explainability
  validation/    # integrity and consistency checks
```

## Reproducibility Notes

- Use seed=42 consistently.
- Keep configuration as single source of truth via src/core/config.py exports.
- Preserve fold-safe behavior for harmonization and evaluation.

## Testing

```bash
pytest tests/unit/
pytest tests/integration/
```

## License

Apache-2.0. See LICENSE.
