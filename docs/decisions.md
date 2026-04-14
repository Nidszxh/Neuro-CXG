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

## DD-009: Structural dropout to enforce edge-structural learning (Task 1)
- Decision: randomly zero all node features for 30% of graphs per batch during training;
  add EdgeStructureContrastiveLoss (weight 0.05) between full-feature and edge-only views.
- Root Cause: Model treats node features as primary signal; edge structure is secondary.
  GradientEdgeAttributor scores showed near-zero importance for most edges.
- Why chosen:
  - Simple mechanism with clear gradient path through conv1/conv2/conv3.
  - NT-Xent loss explicitly measures edge-feature alignment.
  - Backward compatible: structural_dropout_prob and edge_contrastive_weight default to 0.0.

## DD-010: Multi-view causal graphs for robustness to estimation noise (Task 2)
- Decision: Generate 6 causal graph views per subject (base, extended_lag, 3 bootstraps,
  high_confidence) and apply CausalInvarianceLoss (NT-Xent, τ=0.07, weight=0.15).
- Root Cause: Single Granger estimate is noisy — one bad fit fails completely.
- Why chosen:
  - Bootstrap views test robustness to subsample variance.
  - Extended-lag view captures longer-range causal dynamics.
  - High-confidence view enforces that the model relies on strong edges.
  - CausalInvarianceLoss forces invariance across views without requiring labels.
  - Opt-in: invariance loss only activates when CAUSAL_GRAPHS_MULTIVIEW_DIR exists.

## DD-011: Anatomical hierarchical pooling (Task 3)
- Decision: Replace global mean/max/sum pooling with two-level attention pooling:
  Level 1: lobes → 4 functional networks (DMN, Salience, Visual/Cerebellar, Limbic).
  Level 2: networks → graph vector.
- Root Cause: Global pooling collapses the functional hierarchy; DMN and Salience
  are the primary ASD-affected networks but receive equal weight.
- Why chosen:
  - Matches known resting-state functional organization (Power 2011, Yeo 2011).
  - Stores last_network_embeddings for network-level explainability.
  - Old pooling modes (attention, mean_max_sum) preserved for ablations.

## DD-012: Remove conf_std and detection_count from spatial features (Task 4)
- Decision: Keep only 4 anatomical coordinates (x, y, z_depth, size) as spatial features.
  Remove conf_std and detection_count entirely from ALL_FEATURE_NAMES.
- Root Cause: RF trained on only conf_std/detection_count achieved AUC=1.000 predicting
  acquisition site — confirming these features are scanner-identity markers, not anatomy.
- Why chosen:
  - Eliminating site-leaking features reduces CV-test AUC gap.
  - SpatialInvarianceLoss provides an additional gradient-reversal guard on residual site variance.

## DD-013: Site-stratified cross-validation (Task 5)
- Decision: Replace StratifiedKFold with GroupKFold where groups are site clusters
  defined by scanner manufacturer × TR range (5 clusters from 20 ABIDE sites).
- Root Cause: Random KFold includes the same scanner's subjects in both train and
  validation splits; the CV AUC is inflated relative to out-of-site performance.
- Why chosen:
  - Each fold's validation set contains scanner profiles absent from its training fold.
  - Manufacturer × TR clustering respects the dominant technical confounds in ABIDE.
  - Provides more honest estimate of generalisation to unseen scanners.

## DD-014: Dead code removal (Task 6)
- Decision: Remove compute_granger_causality_gpu, compute_transfer_entropy,
  _compute_te_pair, _conditional_entropy, compute_multilag_causality, EVAL_FREQUENCY.
- Why chosen:
  - GPU Granger was unmaintained and had no callers (GPU divergence with CPU path).
  - Transfer entropy was a prototype that was never integrated into the pipeline.
  - Multilag causality is superseded by construct_multiview_graphs() (DD-010).
  - EVAL_FREQUENCY was defined but never read.
  - Dead code inflates maintenance burden and confuses onboarding.
