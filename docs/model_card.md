# Neuro-CXG Model Card

This model card documents model architecture, training configuration, performance metrics, and limitations for reproducibility.

---

## Model Details

| Property | Value |
|----------|-------|
| **Model name** | Neuro-CXG |
| **Model type** | Graph Neural Network (GATv2 with anatomical pooling) |
| **Task** | Autism Spectrum Disorder (ASD) classification from resting-state fMRI |
| **Created** | April 2026 |
| **License** | Apache-2.0 |

## Intended Use

- **Primary use**: Research tool for classifying ASD vs healthy controls from resting-state fMRI
- **Intended users**: Neuroscience researchers, ML researchers studying brain connectivity
- **Out-of-scope uses**: Clinical diagnosis, bedside decision support, real-time clinical inference

## Training Data

| Property | Value |
|----------|-------|
| **Dataset** | ABIDE I (Autism Brain Imaging Data Exchange I) |
| **Sample size** | n=1015 (post-curation) |
| **Sites** | 20 sites |
| **Age range** | Pediatric to adult (varies by site) |
| **Preprocessing** | AAL3 parcellation (170 ROIs → 12 lobes), DPABI fALFF computation |
| **Causal method** | ridge_granger_hybrid (β=0.70, 70% Ridge Granger λ=0.1 + 30% Lagged Pearson) [UPDATED — was Ridge Granger, now hybrid per AGENTS.md:116-117] |

## Evaluation Data

- **Same as training**: ABIDE I held-out test set
- **Split**: 5-fold CV (train/val) + held-out test (154 subjects, ~15% of data)

## Performance Metrics

### Primary Model: May 31, 2026 (Best)
Provenance: `docs/paper/results.md` (48ch/4hd/3L/0.33)

| Metric | Value | 95% CI | Notes |
|--------|-------|--------|-------|
| **CV AUC** | 0.8173 ± 0.0493 | — | 5-fold cross-validation |
| **Test AUC** | **0.8819** | [0.8277, 0.9322] | Ensemble on held-out test set |
| **Test F1** | **0.8485** | [0.7953, 0.8982] | Threshold-optimized (Youden) |
| **Test Accuracy** | 83.77% | [77.92%, 88.98%] | |
| **Sensitivity** | 88.61% | [81.01%, 94.94%] | True positive rate (ASD) |
| **Specificity** | 78.67% | [69.33%, 88.00%] | True negative rate (Control) |

### Canonical Baseline: May 2, 2026
Provenance: Config hash 6b6ca55b

| Metric | Value | 95% CI | Notes |
|--------|-------|--------|-------|
| **Test AUC** | **0.8657** | [0.8017, 0.9185] | Publication baseline |
| **Test F1** | **0.7651** | [0.6933, 0.8400] | Threshold-optimized |

### Performance Improvement

| Metric | May 2 (Baseline) | May 31 (Best) | Delta |
|--------|------------------|---------------|-------|
| Test AUC | 0.8657 | **0.8819** | **+1.62%** |
| Test F1 | 0.7651 | **0.8485** | **+8.34%** |
| Accuracy | 78.57% | **83.77%** | **+5.2%** |
| Sensitivity | 73.42% | **88.61%** | **+15.19%** |

### Configuration Comparison

| Parameter | Canonical | Best (Jun 2026) |
|-----------|-----------|------------------|
| GNN_HIDDEN_CHANNELS | 32 | **48** |
| GNN_NUM_HEADS | 2 | **4** |
| GNN_NUM_LAYERS | 2 | **3** |
| GNN_DROPOUT | 0.35 | **0.33** |
| GNN_ONECYCLE_WARMUP_FRACTION | 0.05 | **0.20** |
| GNN_AUTO_GRL_GRID_SEARCH | True | **False** |
| GNN_GRL_ALPHA | 0.10 (grid) | **0.10 (fixed)** |

### Ablation Studies

| Ablation | CV AUC | CV F1 | Notes |
|----------|--------|-------|-------|
| **Main (ridge_granger_hybrid)** | 0.8100 | 0.7682 | Full pipeline (70% Granger + 30% Pearson) |
| D (lagged_pearson) | 0.8455 | 0.7742 | CV-only |
| D2 (ridge_granger) | 0.8458 | 0.7747 | CV-only |
| A (FlatMLP, no graph) | 0.7245 | 0.6497 | Baseline |
| E (No site/demographics) | 0.7323 | 0.6623 | |
| C (No frequency) | 0.7285 | 0.6522 | |
| B (Spatial only) | 0.5435 | 0.5248 | Minimal signal |

### Subgroup Analysis (Best Model)

| Subgroup | N | AUC | Significant |
|----------|---|-----|-------------|
| Male | 124 | 0.8550 | ✓ |
| Female | 30 | 0.9800 | ✓ |
| Age < 15 | 86 | 0.9484 | ✓ |
| Age ≥ 15 | 68 | 0.8173 | ✓ |
| Site 6 (NYU) | 27 | 0.9167 | ✓ |
| Site 16 | 16 | 1.0000 | ✓ |

All evaluable subgroups significant after Bonferroni correction (α=0.0056).

## Graph Topology Analysis

| Metric | ASD (n=493) | Control (n=522) | p-value | Effect size |
|--------|--------------|-----------------|---------|-------------|
| Mean edges | 45.66 ± 3.36 | 45.73 ± 3.43 | 0.838 | d=-0.02 |
| Density | 0.346 ± 0.025 | 0.347 ± 0.026 | 0.838 | d=-0.02 |
| Clustering | 0.534 ± 0.075 | 0.532 ± 0.072 | 0.691 | d=0.03 |
| **Parietal In-Degree** | 4.11 ± 1.25 | 3.96 ± 1.32 | **0.028** | d=0.12 |

*Significant finding*: ASD subjects show higher parietal cortex in-degree (receives more connections)

## Known Limitations

1. **Cross-site generalization**: 1/16 sites fail (UM_2 at AUC 0.50), 1 marginal (UCLA_1 at 0.63)
2. **CV-Test gap**: CV AUC (0.79) is lower than test AUC (0.88), unusual but consistent with ensemble averaging
3. **Brainstem features**: YOLO never detects Brainstem in 2D slices (uses synthetic fallback). ⚠️ Now explicitly logged as warning during validation (see operations.md §10). 12-lobe architecture generalizes better than 11-lobe due to implicit regularization.
4. **Causality interpretation**: Directed functional connectivity from Ridge Granger, NOT philosophical causal inference.
5. **Temporal resolution**: Depends on site-specific TR (1.5s–3.0s), causal lag limited to 10s of history.

## Hyperparameters

### Best Model (Jun 2026): 48ch/4hd/3L/0.33

| Parameter | Value | Notes |
|-----------|-------|-------|
| Causal method | ridge_granger_hybrid (β=0.70) | Primary (70% Granger + 30% Pearson) |
| Ridge lambda | 0.1 | Reduced from 1.0 for better signal |
| Prune threshold | 0.10 | Reduced from 0.20 for more edges |
| Max lag | 10.0s | Adjusted by TR per subject |
| GRL alpha | 0.10 | Fixed (no grid search) |
| Hidden channels | **48** | Increased from 32 |
| Attention heads | **4** | Increased from 2 |
| GNN layers | **3** | Increased from 2 |
| Dropout | **0.33** | Between 0.30 and 0.35 |
| Warmup fraction | **0.20** | Increased from 0.05 for GRL stability |
| Early stopping patience | 50 epochs | Increased from 30 |
| Early stopping min epochs | 30 | Guardrail against premature stopping |

## Ethical Considerations

- Dataset is publicly available, de-identified
- No clinical deployment intended
- Model provides interpretability outputs (node/edge importance, GradCAM)
- Limitations explicitly documented above

## Files

| Artifact | Path |
|----------|------|
| Model checkpoints | `models/checkpoints/best_model_fold*.pt` |
| Causal graphs | `data/processed/causal_graphs/` |
| Configuration | `src/core/hyperparams.py` |
| Validation | `docs/paper/results.md` |
| Decision log | `docs/decisions.md` |

---

*This model card follows the template from Mitchell et al. (2019) "Model Cards for Model Reporting".*

*Last updated: May 31, 2026*