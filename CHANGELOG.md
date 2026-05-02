# CHANGELOG

All notable changes to Neuro-CXG are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### 2026-05-01 - ridge_granger_hybrid Finalization

- Adopted `ridge_granger_hybrid` (β=0.70) as canonical causality method
- Results: CV AUC 0.8101 ± 0.0274, Test AUC 0.8651
- See `docs/paper/results.md` for full metrics

---

## 2026-04-28

### Comprehensive Ablation Study

- Executed 10 experiments: 6 core ablations + 4 paper experiments
- All on 12-lobe regenerated features
- Full documentation: `docs/decisions.md` (DD-018 to DD-028)

---

## 2026-04-24

### 12-Lobe Architecture Adoption

- Added Brainstem as 12th lobe
- Rationale: implicit regularization, stable generalization gap

---

## 2026-04-22

### Force-Reset Pipeline

- Full pipeline rebuild for reproducibility
- Test AUC: 0.7499 (pre-harmonization)

---

## 2026-03-09

### P0/P1 Fixes

- Dead-lobe NaN handling fixed
- Test AUC improved to 0.6487

---

## 2026-03-08

### Dead-Lobe NaN Fix

- Resolved NaN propagation in feature extraction
- Enabled training on previously excluded subjects

---

## 2026-02-15

### Baseline Establishment

- Initial GNN baseline: CV AUC 0.6194, Test AUC 0.5398

---

## Legacy (Pre-DD)

For pre-Decision Document history, see `docs/archive/ANALYSIS_AND_VALIDATION.md`.

---

*Full rationale and design decisions: `docs/decisions.md`*