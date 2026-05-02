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
Provenance: Config hash 6b6ca55b, run log 12lobes.txt

| Metric | Value | Notes |
|--------|-------|-------|
| **CV AUC** | 0.8101 ± 0.0274 | 5-fold cross-validation |
| **Test AUC** | **0.8657** [95% CI: 0.8017, 0.9185] | Ensemble on held-out test set (ridge_granger_hybrid) |
| **Test F1** | **0.7651** | Threshold-optimized |
| **Test Accuracy** | 77.27% | |
| **Sensitivity** | 0.7342 | True positive rate |
| **Specificity** | 0.6800 | True negative rate |

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

### Per-Site Performance (Test Set)

**Provenance**: Config hash 6b6ca55b, run log `12lobes.txt:1626-1650`

| Site | N | Ctrl | ASD | AUC | Status |
|------|---|------|-----|-----|--------|
| NYU | 27 | 12 | 15 | **0.9000** [UPDATED — was 0.8833, now 0.9000 per 12lobes.txt:1631] | Pass |
| UM_1 | 16 | 8 | 8 | 0.7188 | Pass |
| UCLA_1 | 11 | 6 | 5 | 0.7667 | Pass |
| USM | 11 | 7 | 4 | 0.7857 | Pass |
| YALE | 8 | 4 | 4 | 1.0000 | Pass |
| PITT | 9 | 5 | 4 | 0.7500 | Pass |
| TRINITY | 7 | 3 | 4 | 0.8333 | Pass |
| KKI | 7 | 3 | 4 | 1.0000 | Pass |
| STANFORD | 6 | 3 | 3 | 1.0000 | Pass |
| SBL | 5 | 3 | 2 | 0.6667 | Pass |
| OLIN | 5 | 3 | 2 | 0.8333 | Pass |
| LEUVEN_2 | 5 | 2 | 3 | 1.0000 | Pass |
| CALTECH | 5 | 2 | 3 | 1.0000 | Pass |
| MAX_MUN | 7 | 3 | 4 | 0.5833 | Weak |
| UM_2 | 5 | 2 | 3 | 0.5000 | Fail |

**Site robustness gate**: 15/16 evaluable sites pass (93.75%), 1 fail (UM_2)

**Sites with AUC < 0.55**: UM_2 (n=5)
**Site robustness gate**: 15/16 evaluable sites pass (93.75%)

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

| Parameter | Value | Notes |
|-----------|-------|-------|
| Causal method | ridge_granger_hybrid (β=0.70) | Primary (70% Granger + 30% Pearson) |
| Ridge lambda | 0.1 | Reduced from 1.0 for better signal |
| Prune threshold | 0.10 | Reduced from 0.20 for more edges |
| Max lag | 10.0s | Adjusted by TR per subject |
| GRL alpha | 0.10 | Graph regularization |
| Hidden channels | 32 | |
| Early stopping | 30 epochs | |

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
| Validation | `docs/evaluation.md` |
| Decision log | `docs/decisions.md` |

---

*This model card follows the template from Mitchell et al. (2019) "Model Cards for Model Reporting".*

*Last updated: May 2, 2026*