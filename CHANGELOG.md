# CHANGELOG

All notable changes to Neuro-CXG are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.1.0] — May 11, 2026

### Added
- Optimized GNN hyperparameters: 48ch/4hd/3L/0.33 (best config)

### Changed
- GNN hidden channels: 32 → 48
- GNN num heads: 2 → 4
- GNN num layers: 2 → 3
- GNN dropout: 0.35 → 0.33
- GRL warmup fraction: 0.10 → 0.20 (stability improvement)
- Auto GRL grid search: disabled (fixed alpha=0.10)

### Results
- Test AUC: **0.8810** [0.8277, 0.9322] (3-run stable)
- Test F1: **0.8375** (+9.5% vs canonical 0.7651)
- Sensitivity: **84.8%** (+15.5% vs canonical 73.4%)
- See [`docs/paper/results.md`](docs/paper/results.md) for full metrics

---

## [1.0.0] — May 2, 2026

### Added
- Directed causal graphs using Ridge Granger Causality (hybrid blend β=0.70)
- 12-lobe anatomical parcellation with Brainstem
- Domain adversarial debiasing (GRL, α=0.10)
- Fold-safe ComBat harmonization (DX_GROUP protected)
- Comprehensive ablation studies (A-E, D2)
- Bootstrap confidence intervals + permutation tests
- DeLong statistical comparison with Bonferroni correction

### Changed
- Causality method: lagged_pearson → ridge_granger_hybrid (β=0.70)
- Lambda: 1.0 → 0.1 (reduced regularization)
- Architecture: 12-lobe approved as primary
- CV AUC: 0.8102 ± 0.0273 (from 0.8101)
- Test AUC: 0.8657 (from 0.8651, May 2 fresh run)

### Results
- Test AUC: **0.8657** [95% CI: 0.8017–0.9185]
- CV AUC: **0.8102 ± 0.0273**
- Test F1: 0.7733
- See [`docs/paper/results.md`](docs/paper/results.md) for full metrics

---

## [0.9.0] — May 1, 2026

### Changed
- Causality method: lagged_pearson → ridge_granger_hybrid (β=0.70)
- Test AUC: 0.8651 (canonical candidate)

---

## [0.8.0] — April 28, 2026

### Added
- Comprehensive ablation study (10 experiments)
- 12-lobe regenerated features

---

## [0.7.0] — April 24, 2026

### Added
- Brainstem as 12th lobe
- Implicit regularization effect discovered (variance reduction 46.5%)

---

## [0.6.0] — April 22, 2026

### Added
- Force-reset pipeline for reproducibility

---

## [0.5.0] — March 9, 2026

### Fixed
- Dead-lobe NaN handling
- Test AUC improved to 0.6487

---

## [0.4.0] — March 8, 2026

### Fixed
- NaN propagation in feature extraction
- Previously excluded subjects now trainable

---

## [0.1.0] — February 15, 2026

### Added
- Initial GNN baseline: CV AUC 0.6194, Test AUC 0.5398

---

## Historical References

- **Full decision rationale**: [`docs/decisions.md`](docs/decisions.md) (DD-018 to DD-028)
- **Test set protocol**: [`docs/test_set_protocol.md`](docs/test_set_protocol.md)
- **Canonical metrics**: [`docs/paper/results.md`](docs/paper/results.md)
- **Pre-v1.0 archive**: [`archive_docs/`](archive_docs/) (deprecated, for reference only)

---

*Format: Keep a Changelog | Last updated: May 2, 2026*