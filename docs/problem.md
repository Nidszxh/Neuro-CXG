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
- 12-lobe graph representation derived from atlas ROI signals.
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

- Canonical project references report CV around 0.74 AUC and test around 0.65 AUC.
- Recent runs show variability by configuration and artifact set, especially around site generalization.
- The dominant open risk remains CV-test gap driven by residual site effects and threshold/reporting alignment.

Use run-specific artifacts as source of truth for final numbers, and avoid mixing metrics from different run directories.
