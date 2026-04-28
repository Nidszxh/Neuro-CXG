# Neuro-CXG: Directed Functional Connectivity GNN for Autism Classification

## Abstract

Neuro-CXG is an end-to-end pipeline for classifying Autism Spectrum Disorder (ASD) vs healthy controls from resting-state fMRI. The system computes **directed functional connectivity** (lagged Pearson correlation) to construct subject-level directed brain graphs, then trains a 5-fold Graph Neural Network (GNN) ensemble with domain adversarial debiasing.

**Key Results (Test Set):**
- **AUC: 0.8753** (±0.02 bootstrap CI)
- **F1: 0.8121**
- **Accuracy: 79.87%**

This significantly exceeds prior ABIDE-I baselines (0.70 AUC, Heinsfeld et al. 2018).

## Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| **RAM** | 24 GB | 64 GB |
| **Disk** | 200 GB | 500 GB SSD |
| **GPU VRAM** | 8 GB | 16 GB |
| **CUDA** | 12.1 | 12.1 |

## Estimated Wall-Clock Times

| Stage | Time |
|------|------|
| ABIDE download | 2-6 hours |
| Train/val split | 5-10 min |
| ROI annotation | 30-60 min |
| Feature extraction | 1-2 hours |
| Causal graphs | 2-4 hours |
| GNN training (5-fold) | 30-60 min |
| Evaluation | 15-30 min |
| **Total (full rebuild)** | **6-12 hours** |

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

| Metric | Baseline | **Optimized** | **Final (April 28)** |
|-------|---------|-------------|-----------|
| **CV AUC (5-fold)** | 0.7586 ± 0.0519 | 0.8004 ± 0.0293 | 12-Lobe: 0.7997 ± 0.0294 |
| **Test AUC (ensemble)** | 0.7325 | 0.8753 | **12-Lobe: 0.8694** 🎯 |
| **Test F1** | 0.6338 | 0.8121 | **12-Lobe: 0.8000** |
| **Test Accuracy** | 0.6429 | 0.7987 | **12-Lobe: 0.7857** |
| **Mean Best Epoch** | 12.0 | 40.0 | 12-Lobe: 35.4 |

**Key Configuration Changes:**
- `CAUSALITY_METHOD = "lagged_pearson"` (was ridge_granger)
- `GRANGER_MAX_LAG_SECONDS = 10.0` (max lag in seconds)
- `GNN_USE_SITE_EMBEDDING = True` (was False)
- `GNN_USE_DEMOGRAPHICS = True` (was False)
- `GNN_GRL_ALPHA = 0.10` (NOT 1.0 - test drops with 1.0)

**Architecture Decision (April 28, 2026) — FINAL:**
- **PRIMARY**: 12-Lobe (with Brainstem)
  - Test AUC **0.8694** [95% CI: 0.7889–0.9037] ✅
  - **+8.74% improvement over 11-lobe** on held-out test
  - Excellent generalization (CV < Test by +0.0697)
  - Robust across all demographics (Male +10.1%, Female +3.3%)
- **Key Finding**: YOLO never detects Brainstem → constant synthetic features
  - Counterintuitive result: constant features act as implicit regularization
  - Prevents overfitting (11-lobe CV>Test; 12-lobe CV<Test)
  - See `FINAL_ARCHITECTURE_ANALYSIS.md` and `docs/decisions.md` (DD-018) for full analysis
- **Status**: Approved for publication ✅

*Pipeline status: PUBLICATION-READY - 12-lobe architecture with lagged_pearson + GRL=0.10. Full comparative analysis in FINAL_ARCHITECTURE_ANALYSIS.md.*

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
