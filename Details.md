# Neuro-CXG — Project Details

| Field | Details |
|---|---|
| **Project Name** | Neuro-CXG — Directed Functional Connectivity GNN for Autism Classification |
| **Duration** | ~3 months (Feb 15 – May 11, 2026) |
| **Team Size** | 1 |
| **Contribution** | Full-stack: architecture, data pipeline, GNN design, training, evaluation, explainability |

---

## Problem Being Solved

Classifying Autism Spectrum Disorder (ASD) vs healthy controls from resting-state fMRI using directed functional connectivity graphs with multi-site domain adversarial debiasing.

---

## Dataset

| Property | Value |
|---|---|
| **Source** | ABIDE I (INDI Preprocessed Connectomes Project, CPAC 1.0) |
| **Subjects** | 1,015 (ASD: 522, Control: 493) across 20 sites |
| **Classes** | ASD vs Typically Developing (TYP) |
| **Split** | 70/15/15 stratified (Train: 707, Val: 154, Test: 154) |
| **Input** | Resting-state fMRI, AAL3 parcellation (170 ROIs → 12 lobes) |

---

## Graph Construction

| Aspect | Detail |
|---|---|
| **Nodes** | 12 AAL3-derived brain lobes (Frontal_Superior, Frontal_Orbital, Motor_Premotor, Insula, Cingulate, Limbic, Occipital, Parietal, Temporal, Subcortical, Cerebellum, Brainstem) |
| **Edges** | Directed functional connectivity via `ridge_granger_hybrid` (70% Ridge Granger Causality + 30% Lagged Pearson, beta=0.70) |
| **Sparsification** | Top 30% edges per graph (min 12 edges) |

---

## GNN Architecture

| Component | Configuration |
|---|---|
| **Model** | CausalBrainGNN (custom) |
| **Layers** | 3× GATv2Conv (4 heads, 48 hidden channels) |
| **Activation** | GELU |
| **Skip connections** | Residual add (Linear skip) |
| **Edge gating** | MLP(sigmoid) message modulation |
| **Dropout** | 0.33 |
| **Pooling** | AnatomicalHierarchyPool (lobes → 4 functional networks → graph) |
| **Classifier MLP** | 576 → 128 (GELU+Dropout) → 2 logits |
| **Domain adaptation** | Gradient Reversal Layer (GRL, alpha=0.10) |
| **Loss** | FocalLoss (alpha=0.50, gamma=1.5) |

---

## Frameworks

| Library | Purpose |
|---|---|
| PyTorch 2.9.0 | Deep learning backend |
| PyTorch Geometric 2.7.0 | GNN layers (GATv2Conv), pooling, data loaders |
| Captum 0.8.0 | Feature attribution (Integrated Gradients) |
| scikit-learn 1.8.0 | Metrics, preprocessing |
| nilearn 0.12.1 | Neuroimaging data handling |
| neuroHarmonize 2.4.5 | ComBat harmonization |
| Ultralytics YOLO 8.4.5 | Lobe detection from 2D slices |

---

## Explainability

| Method | Implementation |
|---|---|
| **Node Importance** | GradCAM + GAT attention weights |
| **Edge Importance** | Gradient attribution + edge masking (delta-P) |
| **Feature Attribution** | Integrated Gradients (Captum) |
| **Literature Validation** | Cross-reference against known ASD networks (DMN, Salience, Social Brain, Sensorimotor, Visual, Subcortical) |

---

## Metrics & Results

| Metric | Value | 95% CI |
|---|---|---|
| **Test AUC** | **0.8819** | [0.8277, 0.9322] |
| Test F1 (Youden) | 0.8485 | [0.7953, 0.8982] |
| Accuracy | 83.77% | [77.92%, 88.98%] |
| Sensitivity | 88.61% | [81.01%, 94.94%] |
| Specificity | 78.67% | [69.33%, 88.00%] |
| AUPRC | 0.8752 | [0.8186, 0.9321] |
| CV AUC | 0.8173 ± 0.0493 | — |

**Comparison to prior work:** BrainNetCNN (0.835), Parisot ChebNet (0.810), Heinsfeld (0.700), Random Forest (0.682), Logistic Regression (0.617).

**Statistical significance:** Permutation test p < 0.001 (global and within-site).

---

## Challenges Solved

| Challenge | Solution |
|---|---|
| Multi-site heterogeneity | GRL (alpha=0.10) + fold-safe ComBat harmonization |
| Data leakage in harmonization | ComBat fit on fold-train only; DX_GROUP as covariate |
| Class imbalance | FocalLoss (alpha=0.50, gamma=1.5) |
| Dead brain lobes | YOLO detection on ALFF slices; Brainstem synthetic fallback |
| Nyquist aliasing | Gamma band excluded for TR=2s scans |
| Site-leaky features | Excluded conf_std/detection_count; NUM_SPATIAL_FEATURES=4 assertion |
| GRL instability | Alpha fixed at 0.10, warmup 0.20, annealing disabled |
| Premature early stopping | Patience=50, min_epochs=30 |

---

## Links

- **GitHub:** https://github.com/anomalyco/neuro-cxg
- **Paper:** In preparation (draft at `docs/paper/`)
- **Demo:** N/A
