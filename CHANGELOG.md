# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added
- New documentation set for project onboarding and maintenance:
  - docs/problem.md
  - docs/architecture.md
  - docs/decisions.md
  - docs/setup.md
  - docs/usage.md
  - docs/data.md
  - docs/evaluation.md
  - docs/experiments.md
  - docs/results.md
  - configs/README.md

### Changed
- .gitignore updated to allow tracking docs/ content.
- Consolidated and removed legacy overview/architecture docs after migrating their content into active docs.

### Notes
- Documentation values now distinguish canonical historical run metrics from current on-disk evaluation artifacts.

## [2026-04-12]

### Changed
- Refactored configuration surface: src/core/config.py now re-exports from dedicated modules (paths, hyperparams, feature registry, atlas config, validators).
- Added stage registry wiring so run_pipeline derives stage metadata/order from src/pipeline/registry.py.
- Improved fold-safe harmonization integration and fold-specific training data loading.
- Added lightweight experiment tracker integration for run metadata and summary persistence.
- Hardened causal graph construction (robust lagged correlation helper, sign stabilization, adaptive sparsification safeguards).

### Fixed
- Resolved CUDA graph/tensor reuse instability by disabling torch.compile path in training until upstream lifecycle behavior is safe for this workload.
- Aligned audit checks with current feature/config dimensions and reduced false-fail behavior for spatial completeness edge cases.
