# Data Documentation

## Dataset Source
- ABIDE I (Autism Brain Imaging Data Exchange)
- Multi-site resting-state fMRI + phenotypic metadata

## Dataset Snapshot
| Property | Value |
|---|---|
| Effective subjects used in training/evaluation | 1031 (with exclusions applied) |
| Typical train / val / test split | 719 / 152 / 155 |
| Multi-site coverage | 20 sites |
| Atlas | AAL3v1, ROI-level signals aggregated to 12 lobes |
| Bandpass defaults | 0.01 to 0.15 Hz |
| TR regime | Site-specific (about 1.5s to 3.0s) |

## Brain Region Set (12 Lobes)
- Frontal_Superior
- Frontal_Orbital
- Motor_Premotor
- Insula
- Cingulate
- Limbic
- Occipital
- Parietal
- Temporal
- Subcortical
- Cerebellum
- Brainstem

## Raw Inputs
- Resting-state fMRI NIfTI volumes
- Phenotype CSV including diagnosis and demographics
- AAL3 atlas for ROI extraction and lobe mapping

## Preprocessing Summary
1. Download and extraction
- Subject-level time series extraction from atlas ROIs.
- 2D slice export for ROI detector training.

2. Split protocol
- Train/validation/test split from manifest.
- Stratified balancing by diagnosis and site.

3. Feature extraction
- Temporal features from ROI/lobe signals.
- Spatial features from ROI detector output or atlas fallback.

4. Harmonization
- ComBat harmonization with fold-safe fitting.
- Diagnosis retained as protected covariate.

5. Graph construction
- Directed causal adjacency per subject.
- Sparsification with minimum edge safeguards.

## Feature Schema
Current feature channels are defined in src/core/feature_registry.py and ordered as:
1. Temporal
2. Frequency
3. Internal
4. Spatial

Notes:
- Spatial proxy features linked to site leakage have been treated carefully in recent refactors.
- Gamma-band handling is Nyquist-aware and controlled through config flags.
- Always use ALL_FEATURE_NAMES and GNN_IN_CHANNELS from config exports.

## Data Splits
- Typical split: 70/15/15 (train/val/test)
- Cross-validation: 5 folds on train set
- Fold assignments are consumed by fold-safe harmonization and 5-fold model training.

## Key Data Artifacts
- data/metadata/master_manifest.csv
- data/metadata/node_attributes_temporal.csv
- data/metadata/node_attributes_harmonized.csv
- data/metadata/harmonized_folds_cv/harmonized_fold_<k>.csv
- data/processed/causal_graphs/<subject>_graph.pt
- data/metadata/node_features_3d.csv
- data/metadata/node_features_3d_harmonized.csv

## Augmentation Notes
- Medical constraints are enforced in detector augmentation settings.
- Left-right preserving configuration is required for anatomical consistency.

## Exclusions and Quality Controls
- Subjects with severe missingness or degenerate graphs are excluded.
- Integrity checks run before training to prevent silent data corruption.
- Dead-lobe and NaN-heavy subjects are tracked and filtered through config-controlled policies.

## Compliance Notes
- ABIDE is publicly available research data.
- No patient-identifying data is stored in this repository.
