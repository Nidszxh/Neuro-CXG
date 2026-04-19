# Documentation Index

## Start Here
1. [problem.md](problem.md) — Problem statement and clinical motivation
2. [architecture.md](architecture.md) — System design and data flow
3. [setup.md](setup.md) — Installation and environment setup
4. [usage.md](usage.md) — Running the pipeline

## Core Technical Docs
- [data.md](data.md) — Dataset structure and data pipeline
- [data-curation.md](data-curation.md) — Subject selection, filtering criteria, and curation logic
- [evaluation.md](evaluation.md) — Evaluation protocol and metrics
- [experiments.md](experiments.md) — Experimental design and ablations
- [training.md](training.md) — Training infrastructure, loss functions, and optimization
- [gpu-granger-testing.md](gpu-granger-testing.md) — GPU acceleration and test suite for Granger causality
- [results.md](results.md) — Results, metrics, and performance analysis
- [decisions.md](decisions.md) — Architectural decisions and rationales

## Supplementary
- [configs/README.md](../configs/README.md) — Configuration files and hyperparameter guidance

## Maintenance
- Update [CHANGELOG.md](../CHANGELOG.md) for notable updates.
- Keep [README.md](../README.md) and [docs/results.md](results.md) aligned with latest canonical and on-disk metrics.

---

## New Additions (April 2026)

### Training Infrastructure Documentation
**File**: training.md  
**Covers**: FocalLoss implementation, learning rate scheduling, multi-view training objectives, GNN architecture, hyperparameter tuning, training loop patterns, and known issues.

### GPU Granger Causality Testing
**File**: gpu-granger-testing.md  
**Covers**: Test suite overview, running tests, 5 test cases (synthetic causal signal, random data, short time series, NaN handling, speed benchmark), test metrics, implementation details, and performance baseline.

### Data Curation Documentation
**File**: data-curation.md  
**Covers**: Dataset composition (1,015 final subjects), inclusion/exclusion criteria, 6-stage curation pipeline, subject ID format and site roster, reproducibility, data quality metrics, and related files.

---

## Modified Files (Not Documented Until Now)

| File | Status | Reference |
|------|--------|-----------|
| `src/models/losses.py` | NEW | training.md |
| `tests/unit/test_granger_gpu.py` | NEW | gpu-granger-testing.md |
| `data/metadata/subject_ids_1015.txt` | NEW | data-curation.md |

## Key Documentation Sync Points

**CHANGELOG.md**: Contains historical record of all features. Cross-reference for implementation dates and version history.

**Decisions.md**: High-level architectural choices. See data-curation.md for subject selection rationale and training.md for loss function choices.

**Results.md**: Performance metrics. Cross-reference to training.md for baseline hyperparameters used in canonical runs.
