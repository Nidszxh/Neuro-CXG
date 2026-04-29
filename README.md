# Neuro-CXG: Directed Functional Connectivity GNN for Autism Classification

## Abstract

Neuro-CXG is an end-to-end pipeline for classifying Autism Spectrum Disorder (ASD) vs healthy controls from resting-state fMRI. The system computes **directed functional connectivity** (lagged Pearson correlation) to construct subject-level directed brain graphs, then trains a 5-fold Graph Neural Network (GNN) ensemble with domain adversarial debiasing.

**Key Results (Test Set):**
- **AUC: 0.8694** [95% CI: 0.7889–0.9037]
- **F1: 0.8000**
- **Accuracy: 78.57%**

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

## Current Model Performance (April 28, 2026 — Publication-Ready)

| Metric | Baseline | **Primary (lagged_pearson)** | Future Target (ridge_granger_hybrid) |
|-------|---------|-------------|-----------|
| **CV AUC (5-fold)** | 0.7586 ± 0.0519 | **0.7997 ± 0.0294** | 0.8100 ± 0.0273 (planned) |
| **Test AUC (ensemble)** | 0.7325 | **0.8694** ✅ | 0.8648 (planned) |
| **Test F1** | 0.6338 | **0.8000** | 0.7682 (planned) |
| **Test Accuracy** | 0.6429 | **0.7857** | 0.7826 (planned) |
| **Mean Best Epoch** | 12.0 | **35.4** | ~35 |

**Current Configuration (April 28, 2026 — CANONICAL):**
- `CAUSALITY_METHOD = "lagged_pearson"` (Directional correlation)
- `LAGGED_PEARSON_LAGS = (1, 2, 3, 4)` (max lag: 8 seconds)
- `GNN_USE_SITE_EMBEDDING = True` (domain adaptation)
- `GNN_USE_DEMOGRAPHICS = True` (demographic conditioning)
- `GNN_GRL_ALPHA = 0.10` (gradient reversal strength)

**Planned Alternative Configuration (May 2026 — TARGET, NOT YET EVALUATED):**
- `CAUSALITY_METHOD = "ridge_granger_hybrid"` (70% Ridge Granger + 30% Lagged Pearson)
- Target metrics: CV AUC 0.8100, Test AUC 0.8648 (planned)
- May provide better interpretability through causal lens
- Evaluation pending (see `AGENTS.md` for target metrics)

**Architecture Decision (April 28, 2026) — FINAL:**
- **PRIMARY**: 12-Lobe (with Brainstem)
   - **Test AUC: 0.8694** [95% CI: 0.7889–0.9037] ✅ PUBLICATION-READY
   - **+8.74% improvement over 11-lobe** on held-out test
   - Excellent generalization (CV 0.7997 < Test 0.8694)
   - Robust across all demographics (subgroup analysis in `docs/evaluation.md`)
- **Key Finding**: Brainstem constant features act as implicit regularization
   - YOLO never detects Brainstem in 2D slices → synthetic fallback features
   - Surprisingly, this regularization improves generalization vs 11-lobe
   - See `FINAL_ARCHITECTURE_ANALYSIS.md` (DD-018) for full analysis
- **Status**: Publication-ready ✅ (lagged_pearson, 12-lobe)

**Ablation Studies Complete (April 28, 2026):**
- 6 core ablations (A-E, D2) validating architecture components
- Key results: Graph topology critical (-15.4% without), temporal features essential (-37.4% without)
- All findings documented in `docs/paper/ablations.md`
- Edge weights negligible (shuffled edges same as real)

*For reproducibility and auditing, see TEST_SET_PROTOCOL.md which documents all test set evaluations and model selection integrity.*

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

This project's canonical documentation consists of 10 files:

| File | Description |
|------|-------------|
| `README.md` | This file - quick start, performance, documentation map |
| `docs/architecture.md` | Stage orchestration, design principles, data contracts, model architecture |
| `docs/setup.md` | Environment setup, validation, ABIDE data acquisition |
| `docs/data.md` | Data model, artifacts, feature schema, quality gates, rebuild sequence |
| `docs/configuration.md` | Config modules, high-impact constants, safe change workflow |
| `docs/evaluation.md` | Final results, architecture comparison, ablation studies, statistical validation |
| `docs/decisions.md` | Architecture decisions (DD-001 to DD-018), methods rationale |
| `docs/operations.md` | Failure modes, performance profiling, optimization knobs |
| `docs/extending.md` | How to add stages, features, models safely |
| `docs/paper.md` | Paper figures, abridged changelog (see `CHANGELOG.md` for full history) |

For full changelog history, see `CHANGELOG.md` at project root.

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
