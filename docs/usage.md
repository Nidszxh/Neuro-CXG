# Usage Guide

## Full Pipeline (Recommended)
Run all missing stages:
```bash
python src/run_pipeline.py --auto
```

Useful variants:
```bash
# Reuse existing downloaded/split data
python src/run_pipeline.py --auto --skip-download --skip-split

# Plan only (no execution)
python src/run_pipeline.py --dry-run

# Force rebuild intermediate artifacts
python src/run_pipeline.py --force-reset

# Post-training analysis only
python src/run_pipeline.py --analysis-only
```

## Stage-Level Commands
```bash
# ROI detection training
python -m src.pipelines.roi_detection

# Spatial features
python -m src.features.extract_spatial

# Temporal features
python -m src.features.extract_temporal

# Fold-safe harmonization
python -m src.features.fold_safe_harmonization

# Causal graph construction
python -m src.features.construct_causal

# GNN training
python -m src.models.gnn_model
```

## Evaluation and Explainability
```bash
python src/run_evaluation.py
python src/run_explainability.py
python src/run_result_analysis.py
```

## Additional Utility Commands
```bash
# Validate environment
python -c "from src.core.config import validate_environment; validate_environment()"

# Run pipeline with execution plan only
python src/run_pipeline.py --dry-run

# Run only visual/reporting stages
python src/run_pipeline.py --analysis-only

# Unit and integration tests
pytest tests/unit/
pytest tests/integration/
```

## Expected Outputs
- Graphs: data/processed/causal_graphs/
- Checkpoints: models/checkpoints/
- Evaluation: results/evaluation/
- Explainability: results/explainability/
- Analysis: results/analysis/

## Configuration Usage
- Core defaults are imported from src/core/config.py.
- Prefer changing constants in src/core/hyperparams.py and src/core/feature_registry.py through config exports.

## Troubleshooting
1. Missing checkpoints
- Run training stage first, or use baseline checkpoint directory if available.

2. Shape mismatch errors
- Rebuild features and graphs:
```bash
python src/run_pipeline.py --force-reset --auto --skip-download --skip-split
```

3. Harmonization artifacts missing
- Run:
```bash
python -m src.features.fold_safe_harmonization
```

## Common Failure Patterns
1. Metric collapse near random
- Check class balance, thresholds, and site confound settings.

2. NaN/Inf issues in training
- Rebuild features and confirm integrity checks pass before retraining.

3. Shape mismatch in model input
- Ensure feature ordering and channel counts match config exports.

4. Unexpected checkpoint behavior
- Confirm active checkpoint directory and run ID provenance before comparison.
