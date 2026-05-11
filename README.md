# Neuro-CXG: Directed Functional Connectivity GNN for Autism Classification

## Abstract

Neuro-CXG is an end-to-end pipeline for classifying Autism Spectrum Disorder (ASD) vs healthy controls from resting-state fMRI. The system computes **directed functional connectivity** using **Ridge Granger Causality** to construct subject-level directed brain graphs, then trains a 5-fold Graph Neural Network (GNN) ensemble with domain adversarial debiasing.

> **Canonical Metrics**: All performance metrics, ablation studies, and statistical analyses are maintained in `docs/paper/results.md`. The README contains no numerical results — see the results document for complete tables with confidence intervals.

## Key Features

- Directed brain graphs via Ridge Granger Causality (hybrid blend)
- 12-lobe anatomical parcellation (AAL3-derived)
- Domain adversarial debiasing (GRL) for multi-site harmonization
- Fold-safe ComBat harmonization to preserve diagnosis signal
- Comprehensive ablation studies (8 variants)
- Bootstrap confidence intervals + permutation tests

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

## Model Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| CAUSALITY_METHOD | ridge_granger_hybrid | 70% Granger + 30% Pearson |
| RIDGE_GRANGER_HYBRID_BETA | 0.70 | Blend coefficient |
| GNN_GRL_ALPHA | 0.10 | Domain adversarial strength |

## Performance Summary

**See `docs/paper/results.md` for all canonical metrics with confidence intervals.**

## Documentation

| Document | Purpose |
|----------|---------|
| `docs/paper/results.md` | **Canonical metrics** — all performance tables, CIs, ablations |
| `docs/paper/methods.md` | Methodology for paper writing |
| `docs/paper/figures.md` | Figure generation guide |
| `docs/architecture.md` | System design & stage registry |
| `docs/decisions.md` | Design decision log |
| `docs/test_set_protocol.md` | Test set evaluation history |

---

*Neuro-CXG v1.1 — May 11, 2026*