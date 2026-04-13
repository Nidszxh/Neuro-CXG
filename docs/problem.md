# Problem Statement

## Goal
Build a reproducible, interpretable pipeline for classifying Autism Spectrum Disorder (ASD) vs Control from resting-state fMRI using causal graph representations and graph neural networks.

## Why It Matters
- Clinical ASD diagnosis is still mostly behavioral and expert-driven.
- Neuroimaging can add objective biomarkers, but multi-site noise and small sample sizes make robust modeling difficult.
- Interpretable models are needed for scientific trust, not only classification accuracy.

## Problem Definition
Given subject-level fMRI time series and phenotypic metadata from ABIDE I, predict binary diagnosis:
- Control -> class 0
- ASD -> class 1

The system must:
1. Convert high-dimensional ROI signals into anatomically meaningful graph inputs.
2. Control site/scanner confounds without removing diagnosis signal.
3. Provide explainable outputs (important regions/edges/features).

## Scope
In scope:
- ABIDE I data pipeline (download, split, feature extraction, graph construction, training, evaluation, explainability).
- 12-region lobe-level causal graphs.
- 5-fold cross-validation and held-out test evaluation.

Out of scope:
- Clinical deployment and bedside decision support.
- Multi-disorder classification beyond ASD vs Control.
- Real-time inference from raw MRI scanners.

## Constraints
- Multi-site heterogeneity (20 sites, varying TR and scanner protocols).
- Limited subject count for deep learning.
- Strict leakage prevention requirements in preprocessing/harmonization.
- Neuroanatomical consistency requirements (no left-right flips in data augmentation).

## Assumptions
- ABIDE labels are sufficiently reliable for supervised learning.
- Lobe-level aggregation retains enough signal for ASD discrimination.
- Granger/lagged directed connectivity captures useful temporal interaction structure.

## Success Criteria
Primary:
- Stable cross-validation AUC and statistically significant held-out test AUC.
- Reproducible pipeline execution from config-driven stages.

Secondary:
- Explainability outputs align with known ASD-related networks.
- Clear diagnostics for data quality, graph quality, and site effects.

## Current Status Snapshot (March 2026 artifacts)
- CV AUC: ~0.74 (5-fold)
- Test AUC: ~0.65 with permutation significance
- Known remaining gap: CV-test generalization difference likely tied to residual site effects
