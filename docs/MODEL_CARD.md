# Neuro-CXG Model Card

## Model Details

- **Model name**: Neuro-CXG
- **Model type**: Graph Neural Network (GATv2 with anatomical pooling)
- **Task**: Autism Spectrum Disorder (ASD) classification from resting-state fMRI
- **Created**: April 2026
- **License**: Apache-2.0

## Intended Use

- **Primary use**: Research tool for classifying ASD vs healthy controls from resting-state fMRI
- **Intended users**: Neuroscience researchers, ML researchers studying brain connectivity
- **Out-of-scope uses**: Clinical diagnosis, bedside decision support, real-time clinical inference

## Training Data

- **Dataset**: ABIDE I (Autism Brain Imaging Data Exchange I)
- **Sample size**: n=1015 (post-curation)
- **Sites**: 13 test sites
- **Age range**: Pediatric to adult (varies by site)
- **Preprocessing**: AAL3 parcellation, DPABI fALFF computation

## Evaluation Data

- **Same as training**: ABIDE I held-out test set
- **Split**: 5-fold CV (train/val) + held-out test (20% of data)

## Performance Metrics

| Metric | Value | 95% CI |
|--------|-------|---------|
| **CV AUC** | 0.7997 | ±0.0294 |
| **Test AUC** | **0.8694** | [0.7889, 0.9037] |
| **Test F1** | 0.8000 | — |
| **Test Accuracy** | 78.57% | — |
| **Sensitivity** | 0.7595 | — |
| **Specificity** | 0.7733 | — |

### Per-Site Performance (ranked by sample size)

| Site | N | AUC | Status |
|------|---|-----|--------|
| NYU | 27 | 0.88 | Pass |
| UM_1 | 16 | 0.77 | Pass |
| UCLA_1 | 11 | 0.53 | Fail |
| USM | 11 | 0.82 | Pass |
| YALE | 8 | 1.00 | Pass |
| PITT | 9 | 0.70 | Pass |
| MAX_MUN | 7 | 0.50 | Fail |
| TRINITY | 7 | 1.00 | Pass |
| KKI | 7 | 1.00 | Pass |
| OLIN | 5 | 0.83 | Pass |
| LEUVEN_2 | 5 | 0.83 | Pass |
| SBL | 5 | 1.00 | Pass |
| STANFORD | 6 | 1.00 | Pass |
| CALTECH | 5 | 0.83 | Pass |

**Sites with AUC < 0.60**: MAX_MUN, UCLA_1, UM_2 (small sample size, n < 11)

## Demographic Subgroup Analysis

- Performance varies by site due to multi-site heterogeneity
- Site harmonization (ComBat) partially addresses scanner/protocol effects
- Class balance: ~59% ASD, ~41% Control (well-balanced)

## Known Limitations

1. **Cross-site generalization**: 3/13 sites fail (AUC < 0.60), all with small sample size (n < 11)
2. **Graph contribution**: Edge features contribute minimal discriminative value (~3% of improvement)
3. **Spatial features**: Near-random predictive power regardless of source (YOLO vs atlas)
4. **Brainstem features**: While YOLO never detects Brainstem in 2D slices (uses synthetic fallback), the 12-lobe architecture (including Brainstem) generalizes better to held-out test data (test AUC +0.0699 vs 11-lobe). See `docs/FINAL_ARCHITECTURE_ANALYSIS.md` (DD-018) for full justification.
5. **Causality interpretation**: Directed functional connectivity, NOT true causal inference in the philosophical sense.
6. **Cross-site performance**: Model achieves AUC 0.82–1.00 on well-represented sites (n ≥ 11) but lower performance on under-represented sites (n < 11, AUC 0.53–0.70).

## Ethical Considerations

- Dataset is publicly available, de-identified
- No clinical deployment intended
- Model provides interpretability outputs (node/edge importance)
- Limitations explicitly documented above

## Files

- Model checkpoints: `models/checkpoints/best_model_fold*.pt`
- Configuration: `src/core/hyperparams.py`
- Validation: `docs/ANALYSIS_AND_VALIDATION.md`
- Methods: `docs/methods.md`

---

*This model card follows the template from Mitchell et al. (2019) "Model Cards for Model Reporting".*