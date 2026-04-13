# Design Decision Log

This file records major architectural and modeling decisions.

## DD-001: 12-region lobe graph instead of 170-ROI graph
- Decision: aggregate AAL ROIs into 12 anatomical regions before graph learning.
- Alternatives considered:
  - Full 170-node ROI graph.
  - Coarser 5-lobe graph.
- Why chosen:
  - Better sample-efficiency for ABIDE-scale training.
  - Preserves anatomical interpretability while reducing graph complexity.

## DD-002: GATv2 as primary graph backbone
- Decision: use GATv2-based message passing.
- Alternatives considered:
  - GCN, GraphSAGE.
- Why chosen:
  - Attention is useful for weighted directed connectivity.
  - Better expressivity on sparse, heterogeneous edge patterns.

## DD-003: Fold-safe ComBat harmonization
- Decision: harmonization must be fitted on fold-train only and applied to fold-val/test.
- Alternatives considered:
  - Global harmonization before CV.
- Why chosen:
  - Prevents leakage from validation distribution into training normalization.
  - Preserves evaluation integrity in multi-site settings.

## DD-004: Keep diagnosis as protected covariate in harmonization
- Decision: include DX_GROUP in covariates during ComBat.
- Alternatives considered:
  - Harmonize only by site.
- Why chosen:
  - Prevents removal of disease-relevant signal as nuisance variance.

## DD-005: Stage registry for orchestration metadata
- Decision: move stage metadata into src/pipeline/registry.py and have run_pipeline use it.
- Alternatives considered:
  - Ad-hoc stage dictionaries in runner only.
- Why chosen:
  - Cleaner dependency mapping.
  - Less drift between stage declarations and execution logic.

## DD-006: Config split into focused modules
- Decision: split monolithic config into paths, feature_registry, atlas_config, hyperparams, validators.
- Alternatives considered:
  - Keep single large config file.
- Why chosen:
  - Better maintainability and discoverability.
  - Backward compatibility preserved via thin src/core/config.py re-export.

## DD-007: Disable strong GRL by default
- Decision: keep GRL disabled in canonical training configuration.
- Alternatives considered:
  - Always-on high-alpha adversarial site loss.
- Why chosen:
  - High-alpha setting previously collapsed class discrimination.
  - Reintroduced only through controlled experiments.

## DD-008: Keep detailed explainability outputs in default workflow
- Decision: include node, edge, and feature attribution phases after training.
- Alternatives considered:
  - Accuracy-only reporting.
- Why chosen:
  - Clinical/scientific credibility requires interpretable findings.
  - Supports literature-grounded validation.
