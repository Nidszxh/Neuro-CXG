# Neuro-CXG

Neuro-CXG is a configuration-driven, end-to-end pipeline for ASD vs Control classification from resting-state fMRI. The system ingests ABIDE data, builds subject-level directed causal brain graphs, trains a 5-fold GNN ensemble, and produces evaluation, explainability, and per-subject analysis artifacts.

## What The Code Runs Today

- Pipeline orchestration is declarative: `src/pipeline/registry.py` defines stage metadata and `src/run_pipeline.py` executes it.
- Feature dimensionality is dynamic and config-derived from `src/core/feature_registry.py`.
- Causal graph construction defaults to `lagged_pearson` (`CAUSALITY_METHOD` in `src/core/hyperparams.py`).
- Fold-safe harmonization writes per-fold artifacts to `data/metadata/harmonized_folds_cv/harmonized_fold_<k>.csv` and a combined output to `data/metadata/node_attributes_harmonized.csv`.
- Training enforces fold-specific harmonized inputs and graph quality gates (`src/models/gnn_model.py`).
- Evaluation threshold policy is controlled centrally by `EVAL_THRESHOLD_POLICY` / `EVAL_FIXED_THRESHOLD` in `src/core/hyperparams.py`.

## Code Layout

| Path | Responsibility |
|---|---|
| `src/pipeline/` | Declarative stage registry and stage metadata contracts. |
| `src/run_pipeline.py` | Main orchestrator entrypoint that resolves and executes stage plans. |
| `src/core/` | Configuration modules, constants, and runtime validators. |
| `src/data/` | ABIDE ingestion, split generation, and dataset preparation helpers. |
| `src/pipelines/` | Stage scripts for labeling and ROI detection workflows. |
| `src/features/` | Feature extraction, harmonization, causal graph construction, and dataset assembly. |
| `src/models/` | GNN architecture, training loop, model factory, and training utilities. |
| `src/analysis/` | Explainability, diagnostics, and reporting visualizations. |
| `src/validation/` | Pipeline integrity checks and audit utilities. |
| `src/experiments/` | Targeted ablations and experiment runners. |
| `tests/` | Unit and integration tests. |

Config layering is intentionally modular, with `src/core/config.py` as the stable import facade over:

- `src/core/paths.py`
- `src/core/feature_registry.py`
- `src/core/hyperparams.py`
- `src/core/atlas_config.py`
- `src/core/validators.py`

## Quick Start

```bash
pip install -r requirements.txt
python -c "from src.core.config import validate_environment; validate_environment()"
python src/run_pipeline.py --dry-run
```

Run the full pipeline in non-interactive mode:

```bash
python src/run_pipeline.py --auto
```

Run post-training scripts directly:

```bash
python src/run_evaluation.py
python src/run_explainability.py
python src/run_result_analysis.py
```

## Current Model Performance (April 2026)

| Metric | Baseline | **Optimized** |
|-------|---------|-------------|
| **CV AUC (5-fold)** | 0.7586 ± 0.0519 | **0.8001 ± 0.0293** |
| **Test AUC (ensemble)** | 0.7325 | **0.8748** |
| **Test F1** | 0.6338 | **0.8121** |
| **Test Accuracy** | 0.6429 | **0.7987** |
| **Mean Best Epoch** | 12.0 | 40.0 |

**Key Configuration Changes:**
- `CAUSALITY_METHOD = "lagged_pearson"` (was ridge_granger)
- `GNN_USE_SITE_EMBEDDING = True` (was False)
- `GNN_USE_DEMOGRAPHICS = True` (was False)
- `GNN_GRL_ALPHA_MAX = 1.0` (was 0.15)


*Pipeline status: RUN RESET - force-regenerated features and graphs, higher fold variance observed.*

## Common Execution Modes

```bash
# Full run (auto)
python src/run_pipeline.py --auto

# Reuse existing download and split
python src/run_pipeline.py --auto --skip-download --skip-split

# Optional site-stratified CV fold reassignment stage
python src/run_pipeline.py --auto --site-stratified-cv

# Optional multiview graph generation stage
python src/run_pipeline.py --auto --multiview

# Only post-training analysis stages
python src/run_pipeline.py --analysis-only
```

## Core Outputs

- Harmonized temporal features: `data/metadata/node_attributes_harmonized.csv`
- Fold harmonization outputs: `data/metadata/harmonized_folds_cv/`
- Spatial features: `data/metadata/node_features_3d.csv`
- Subject causal graphs: `data/processed/causal_graphs/`
- Optional multiview graphs: `data/processed/causal_graphs_multiview/`
- Trained checkpoints: `models/checkpoints/best_model_fold*.pt`
- Evaluation bundle: `results/evaluation/`
- Explainability bundle: `results/explainability/`
- Result interpretation bundle: `results/analysis/`

## Documentation Map

- `docs/README.md` - documentation index
- `docs/architecture.md` - architecture, stage orchestration, data contracts
- `docs/components.md` - source module responsibilities
- `docs/configuration.md` - config modules and high-impact constants
- `docs/setup.md` - environment setup and validation
- `docs/usage.md` - command-line usage and run modes
- `docs/data.md` - data model, artifacts, feature schema, quality gates
- `docs/evaluation.md` - evaluation, explainability, and result-analysis contracts
- `docs/decisions.md` - architecture and modeling decisions
- `docs/performance.md` - runtime characteristics and optimization knobs
- `docs/failure-modes.md` - operational failure patterns and fixes
- `docs/extending.md` - how to add stages, features, and models safely
- `docs/walkthrough.md` - end-to-end reproducible workflow

## Development Notes

- Use config exports from `src/core/config.py`; avoid hardcoded paths and hyperparameters.
- Keep reproducibility stable (`seed=42` is used throughout training/evaluation code paths).
- If you modify feature channels, keep `FEATURE_GROUPS`, `ALL_FEATURE_NAMES`, and `GNN_IN_CHANNELS` aligned.
- If you modify stage behavior, update `src/pipeline/registry.py` and corresponding docs together.

## Tests

```bash
pytest tests/unit/
pytest tests/integration/
pytest --cov=src tests/
```

## License

Apache-2.0. See `LICENSE`.
