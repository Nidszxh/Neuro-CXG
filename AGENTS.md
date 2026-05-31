# Neuro-CXG Agent Instructions

## Quick Commands

```bash
# Activate virtual environment (ichigo = Python 3.12.13)
source ~/.ichigo/bin/activate
export PYTHONPATH="/home/nidszxh/Projects/Neuro-CXG:$PYTHONPATH"

# Full pipeline (non-interactive)
python src/run_pipeline.py --auto
```

## Testing

```bash
# Unit tests only (no data required)
pytest tests/unit/ -v

# Run specific test file
pytest tests/unit/test_config.py -v

# Lint check
ruff check src/ --statistics
```

## Key Paths (use config imports, not hardcoded)

```python
from src.core.config import (
    CHECKPOINT_DIR,  # models/checkpoints/
    MASTER_MANIFEST,  # data/metadata/master_manifest.csv
    NODE_ATTRIBUTES_HARMONIZED,  # data/metadata/node_attributes_harmonized.csv
    CAUSAL_GRAPHS_DIR,  # data/processed/causal_graphs/
    GNN_IN_CHANNELS,  # Dynamic feature channel count (24 without gamma)
)
```

## Important Config Constraints

- **GNN_GRL_ALPHA = 0.10** — Changing this (especially to 1.0) drops AUC from ~0.88 to ~0.83
- **NUM_SPATIAL_FEATURES = 4** — Enforced by assertion; prevents site-leaky channels
- **CAUSALITY_METHOD = "ridge_granger_hybrid"** — 70% Granger + 30% Pearson blend
- **GNN_IN_CHANNELS** is computed dynamically from `ALL_FEATURE_NAMES` (24 when gamma excluded)

## Pipeline Stages

See `docs/setup.md` for full table. Key stages:
- Stage 14: Causal graph construction
- Stage 18: GNN training (5-fold)
- Stage 21-23: Evaluation/explainability/analysis

## Environment Validation

```bash
python -c "from src.core.config import validate_environment; validate_environment()"
```

## Feature Groups

| Group | Channels | Notes |
|-------|----------|-------|
| temporal | 8 | mean, std, skew, kurtosis, psd, mssd, range, autocorr |
| frequency | 10 | Excludes gamma by default (unreliable at Nyquist) |
| internal | 2 | coherence, spatial_variance |
| spatial | 4 | x, y, z_depth, size |

## Architecture

- 170 AAL ROIs → 12 lobes (AAL3-derived)
- Directed functional connectivity via Ridge Granger Causality
- GNN with domain adversarial debiasing (GRL) + fold-safe ComBat harmonization

## Central Config Hub

`src/core/config.py` re-exports all public names from `paths.py`, `hyperparams.py`, `feature_registry.py`, `atlas_config.py`, and `validators.py`.
Always import from `src.core.config`, not from individual submodules.

## Canonical Results

All metrics in `docs/paper/results.md` — not in README.

## Known Gotchas

- Gamma band excluded by default (`UNRELIABLE_FREQ_BANDS_AT_NYQUIST = ("gamma",)`)
- Integration tests disabled in CI (require data)
- YOLO augmentation conservative (no flip, no rotation) to preserve anatomy
- Site-stratified CV requires: `python -m src.data.split --site-stratified-cv && python -m src.features.fold_safe_harmonization && python -m src.models.gnn_model`
