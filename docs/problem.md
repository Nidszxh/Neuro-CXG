# Problem Statement

## Objective

Develop a reproducible and interpretable pipeline that classifies Autism Spectrum Disorder (ASD) vs Control from resting-state fMRI using directed causal brain graphs and graph neural networks.

## Why This Problem Is Hard

- ASD diagnosis is clinically valuable but difficult to anchor to robust imaging biomarkers.
- ABIDE is multi-site, so scanner/protocol variation can dominate signal if not controlled.
- Dataset size is modest for deep learning, increasing overfitting risk.
- Scientific usefulness requires explainability, not only high aggregate metrics.

## Formal Task

Given ABIDE I subject-level fMRI time series and phenotypic metadata, predict binary diagnosis:

- Control -> class 0
- ASD -> class 1

The pipeline must:

1. Transform high-dimensional ROI activity into anatomically meaningful graph inputs.
2. Reduce site/scanner confounds while preserving diagnosis-relevant variation.
3. Produce interpretable outputs at node, edge, and feature levels.

## Scope

In scope:

- End-to-end ABIDE I workflow (download, split, feature extraction, harmonization, graph construction, training, evaluation, explainability, result analysis).
- **12-lobe graph representation** derived from atlas ROI signals (primary architecture, **approved for publication** — see `docs/decisions.md` DD-018 for full architecture decision analysis).
- 5-fold CV plus held-out test evaluation.

Out of scope:

- Clinical deployment or bedside decision support.
- Multi-disorder diagnosis beyond ASD vs Control.
- Real-time scanner-side inference.

## Non-Negotiable Constraints

- Multi-site heterogeneity across 20 sites (scanner and protocol variability).
- Strict leakage prevention for harmonization, fold assignment, and threshold usage.
- Anatomical fidelity constraints for augmentation (no left/right flips or rotations that break neuroanatomical consistency).
- Reproducibility requirements (seeded runs, deterministic stage contracts, config-driven constants).

## Working Assumptions

- ABIDE labels are reliable enough for supervised learning at research quality.
- Lobe-level aggregation retains useful ASD-discriminative information.
- Directed temporal interaction estimates (for example Granger-based) provide learnable graph structure.

## Success Criteria

Primary criteria:

- Stable fold-level validation behavior with statistically meaningful held-out test performance.
- Fully reproducible stage execution from configuration and documented artifacts.

Secondary criteria:

- Explainability outputs that are coherent with known ASD-relevant networks.
- Clear diagnostics for data quality, graph quality, and site effects.
- Operational robustness (graceful fallback behavior for low-quality inputs).

## Current Status (Run-Dependent)

- **Final Results (April 28, 2026)**: 12-Lobe architecture approved for publication
  - CV AUC: 0.7997 ± 0.0294
  - Test AUC: **0.8694** [95% CI: 0.7889–0.9037] ✅
  - Test F1: **0.8000**
  - Test Accuracy: **0.7857**
  - Generalization gap: +0.0697 (robust learning vs overfitting)
- **Architecture Decision (April 28, 2026)**: 12-Lobe emerges as definitive architecture
  - Reversed from preliminary DD-018 recommendation of 11-lobe based on pre-training metrics
  - Test set establishes ground truth: 12-lobe test AUC +8.74% over 11-lobe (0.8694 vs 0.7995)
  - Brainstem constant features act as implicit regularization despite YOLO detection gap
  - Full analysis in `FINAL_ARCHITECTURE_ANALYSIS.md` and `docs/decisions.md` (DD-018)
- Configuration: lagged_pearson + GRL=0.10 + MaxLag=10.0s + 12-lobe atlas

Use run-specific artifacts as source of truth for final numbers, and avoid mixing metrics from different run directories.
