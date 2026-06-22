# Neuro-CXG Agent Instructions

## Quick Commands

```bash
# Activate the project venv (zangestu = Python 3.12.13), then export PYTHONPATH —
# REQUIRED: the package is not pip-installed, so `import src.*` only works from
# the repo root or with this export on sys.path.
source ~/.zangestu/bin/activate
export PYTHONPATH="/home/nidszxh/Projects/Neuro-CXG:$PYTHONPATH"

# Full pipeline (non-interactive)
python src/run_pipeline.py --auto
```

## Testing

```bash
# Unit tests only — "no data required" means no fMRI data, NOT no deps:
# test_feature_ordering.py and test_harmonization.py import neuroHarmonize at
# collection time, so install requirements.txt first (without it, the former
# fails to collect and the latter has 8 failures).
pytest tests/unit/ -v

# Run specific test file
pytest tests/unit/test_config.py -v

# Lint check (ruff)
ruff check src/ --statistics

# Environment validation
python -c "from src.core.config import validate_environment; validate_environment()"
```

## Key Paths (use config imports, not hardcoded)

```python
from src.core.config import (
    CHECKPOINT_DIR,           # models/checkpoints/
    BASELINE_CHECKPOINT_DIR,  # models/checkpoints_baseline/ (fallback dir; not tracked in git)
    MASTER_MANIFEST,          # data/metadata/master_manifest.csv
    NODE_ATTRIBUTES_HARMONIZED,  # data/metadata/node_attributes_harmonized.csv
    CAUSAL_GRAPHS_DIR,        # data/processed/causal_graphs/
    GNN_IN_CHANNELS,          # Dynamic feature channel count (24 without gamma)
)
```

## Important Config Constraints

- **GNN_GRL_ALPHA = 0.10** — Changing this (especially to 1.0) drops AUC from ~0.88 to ~0.83
- **NUM_SPATIAL_FEATURES = 4** — Enforced by assertion; prevents site-leaky channels
- **CAUSALITY_METHOD = "ridge_granger_hybrid"** — 70% Granger + 30% Pearson blend
- **GNN_IN_CHANNELS** is computed dynamically from `ALL_FEATURE_NAMES` (24 when gamma excluded)

## Pipeline Stages

Source of truth: `src/pipeline/registry.py` (`STAGES` list, 25 stages).
`run_pipeline.py` executes stages in registry order; stage numbers = 1-based position
in that order (same as `--dry-run` output). Key stages:
- Stage 13: Causal graph construction
- Stage 17: GNN training (5-fold)
- Stage 22-24: Evaluation / explainability / result analysis

Adding a new runnable `src/` module (contains `__main__`) triggers a stage-coverage
audit (`_check_stage_coverage` in `run_pipeline.py`): the module must be added to
`STAGES` in the registry or to `EXEMPT_ENTRYPOINT_MODULES`:
```python
EXEMPT_ENTRYPOINT_MODULES = {
    "src.run_pipeline", "src.core.config",
    "src.features.graph_factory", "src.features.causal_inference",
    "src.analysis.feature_attribution", "src.analysis.generate_paper_figures",
    "src.experiments.data_quality", "src.experiments.run_ablations",
    "src.experiments.run_learning_curve", "src.experiments.test_random_edges_on_test",
}
```

## Architecture

- 170 AAL ROIs → 12 lobes (AAL3-derived)
- Directed functional connectivity via Ridge Granger Causality
- GNN with domain adversarial debiasing (GRL) + fold-safe ComBat harmonization

## Central Config Hub

`src/core/config.py` re-exports all public names from `paths.py`, `hyperparams.py`, `feature_registry.py`, `atlas_config.py`, and `validators.py`.
Always import from `src.core.config`, not from individual submodules. (Exception: `src/core/` submodules importing from each other is necessary to avoid circular imports.)

## Canonical Results

All metrics in `docs/paper/results.md` — not in README.

## Known Gotchas

- Gamma band excluded by default (`UNRELIABLE_FREQ_BANDS_AT_NYQUIST = ("gamma",)`)
- Integration tests disabled in CI (require data)
- YOLO augmentation conservative (no flip, no rotation) to preserve anatomy
- Site-stratified CV requires: `python -m src.data.split --site-stratified-cv && python -m src.features.fold_safe_harmonization && python -m src.models.gnn_model`
- `run_pipeline.py` is **interactive by default** — omitting `--auto` prompts at every
  stage and hangs in non-interactive shells. Always pass `--auto`.
- Evaluation/analysis stages fall back to `models/checkpoints_baseline/` (shipped with
  repo, not git-tracked) when `models/checkpoints/` is empty (`_checkpoints_available()`).
- **Validation is centralized in `src/validation/pipeline_checks.py`** — the single
  orchestrator for all pipeline checks (`PipelineValidator` + `AuditCheck`).
  Other `src/validation/` modules: `atlas_validator.py` (atlas download),
  `config_snapshot.py` (config-hash enforcement), `delong_test.py` (AUC stats).

## Reproducibility

```bash
# Reproduce full pipeline (requires data in data/raw/)
./scripts/reproduce.sh

# Reproduce without re-downloading
./scripts/reproduce.sh --skip-download

# Verify seed stability (3 seeds, ~2h each)
./scripts/verify_seeds.sh
```

## 11-Lobe Mode

Set via CLI flag or env var (must be set before config imports):
```bash
python src/run_pipeline.py --auto --11-lobes
# or: NEURO_CXG_11_LOBES=1 python src/run_pipeline.py --auto
```

## Docker

```bash
docker build -t neuro-cxg .
docker run --gpus all -v /path/to/data:/workspace/data -v /path/to/results:/workspace/results neuro-cxg
```

## Feature Groups

| Group | Channels | Notes |
|-------|----------|-------|
| temporal | 8 | mean, std, skew, kurtosis, psd, mssd, range, autocorr |
| frequency | 10 | Excludes gamma by default (unreliable at Nyquist) |
| internal | 2 | coherence, spatial_variance |
| spatial | 4 | x, y, z_depth, size |

## Ruff Linting

`E402` (import not at top) is intentionally ignored — project groups imports after third-party. `E501` (line length) deferred to black.

## Seed Propagation

`--seed` (default 42) only reaches **GNN training**: `run_pipeline.py` sets
`NEURO_CXG_SEED` (read by `GNN_SEED`) and passes `--seed` to `src.models.gnn_model`.
YOLO is pinned to seed 42 via `YOLO_TRAIN_CONFIG`; evaluation/permutation functions
default to seed 42 and do NOT read the env var.

## CI Behavior (`.github/workflows/tests.yml`)

`ruff check src/`, `black --check src/`, and `mypy src/core/ src/models/` run as
real gates — **they fail CI**. Run them locally before pushing:
`ruff check src/ && black --check src/ && mypy src/core/ src/models/ --ignore-missing-imports`

- Integration tests are hard-disabled (`if: false`); the docker-build job is `continue-on-error`.
- pytest config lives in `pytest.ini` (the `[tool.pytest.ini_options]` block was
  removed from `pyproject.toml`; `pytest.ini` is the single source of truth for
  pytest settings).
- All three gates (ruff/black/mypy) currently pass on the working tree and the full
  unit suite is green (97 tests). The unit tests require `neuroHarmonize` (see
  Testing section) and a numpy version that provides `np.trapezoid`/`np.trapz`
  (the repo pins `numpy==1.26.4`; `extract_temporal.py` has a compat shim).

## Pipeline Flags

| Flag | Effect |
|------|--------|
| `--force-reset` | Wipe ALL intermediate CSVs and graphs |
| `--regenerate-features` | Rebuild only spatial/temporal features, harmonization, and graphs |
| `--skip-download` / `--skip-split` | Skip data prep stages (use existing data) |
| `--skip-yolo` | Skip YOLO training — use existing weights |
| `--multiview` | Enable multi-view causal graph construction |
| `--site-stratified-cv` | Use site-stratified GroupKFold instead of StratifiedKFold |
| `--auto` | Non-interactive mode (sets `NEURO_CXG_NONINTERACTIVE=1`) |
| `--dry-run` | Show execution plan without running anything |
| `--analysis-only` | Run only post-training analysis stages |
| `--visualizations-only` | Run only visualization stages |
| `--seed N` | Set global random seed (default: 42) |
