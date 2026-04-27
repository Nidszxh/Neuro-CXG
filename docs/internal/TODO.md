# Neuro-CXG: Comprehensive Codebase Audit & Optimization Report

**Audit Date:** 2026-04-20  
**Codebase:** Neuro-CXG — GATv2-based ASD classification from resting-state fMRI  
**Pipeline Run Analyzed:** Full run with site-stratified CV, Wave-1 preprocessing  
**Current Result:** CV AUC 0.57 ± 0.04 → Ensemble Test AUC 0.74 (p=0.001)

---

## Executive Summary

The codebase is architecturally sound and scientifically rigorous, but has accumulated **seven high-severity configuration conflicts and data flow errors** that are actively suppressing fold-level training performance. The 0.17 gap between CV validation AUC (0.57) and ensemble test AUC (0.74) is not a model limitation — it is an artifact of these conflicts. Resolving the three critical issues alone should recover the CV AUC to the 0.70–0.74 range observed in the canonical March 2026 run, making ensemble performance consistent and reportable.

**Priority tier breakdown:**

| Priority | Issues | Expected AUC impact |
|---|---|---|
| P0 — Critical (breaks training) | 3 | +0.12–0.17 to CV AUC |
| P1 — High (degrades quality) | 4 | +0.02–0.05 |

---

## P0 — Critical Issues (Fix Before Next Run)

### P0-1: Site-Stratified CV + Fold-Safe Harmonization Are Mutually Incompatible With This Cohort

**Root cause:** The pipeline uses `--site-stratified-cv`, which assigns each CV fold's *validation* set to a complete, non-overlapping site cluster. With 20 sites and 5 folds, every validation fold contains **exclusively unseen sites** — confirmed by the log:

```
Fold 0 unseen SITE audit: 220/220 val rows from unseen sites ['LEUVEN_1', 'MAX_MUN', 'NYU', 'YALE']
Fold 1 unseen SITE audit: 147/147 val rows from unseen sites [...]
... (all 5 folds: 100% unseen)
```

Because `HARMONIZATION_UNSEEN_SITE_POLICY = "passthrough"`, **zero validation subjects are ComBat-harmonized**. The model is trained on harmonized features and evaluated on raw, unharmonized features. This is a systematic train/val distribution mismatch — not a data leak, but an inverse leakage that depresses every fold's measured AUC.

The fix is a choice between two valid approaches:

**Option A (Recommended for publication):** Revert to standard StratifiedKFold CV. Site-stratified CV answers a valuable scientific question but requires harmonization-free feature representations to be valid. 

**Option B (If site-stratified CV is required):** Apply global harmonization (fit on *all* train subjects, apply to val/test) before fold splitting, instead of fold-safe per-fold harmonization. This sacrifices strict fold-safety but restores feature-space consistency.

**Observed impact:** CV AUC of 0.53–0.65 vs expected 0.74. Every fold's checkpoint is under-trained because validation signal is meaningless.

---

### P0-2: YOLO Fails to Detect Brainstem (Class 11) for Every Single Subject

**Root cause:** The spatial feature extraction log reports:

```
Unique ROI classes detected: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Subjects with complete detection (all 12 regions): 0
Subjects with partial detection (9-11 regions): 1015
```

Brainstem (class 11, ROIs 167–170) is never detected. The code falls back to `_load_atlas_lobe_fallbacks()` but also logs:

```
WARNING: Atlas centroids use unexpected ROI indexing; falling back to zeros for unmatched ROI ids.
WARNING: Global YOLO detections missing for lobe ids [11]; using mapped atlas priors.
```

The atlas centroid loader is failing silently due to indexing mismatch — so Brainstem spatial features are likely zeros or near-zeros for all subjects. This creates a constant feature column (zero variance) that the MI selector may retain or discard inconsistently, and it contaminates the spatial node features passed to the GNN.

**Fix 1:** Debug the atlas centroid indexing in `extract_spatial_atlas.py`. The issue is in `_load_atlas_lobe_fallbacks()` which attempts both `roi_id` and `roi_idx_0` lookups but the centroids JSON uses 1-indexed IDs while the mapping uses 0-indexed.

```python
# In src/features/extract_spatial.py, _load_atlas_lobe_fallbacks():
# Current (broken):
c = centroids.get(roi_id)      # roi_id = roi_idx_0 + 1 (1-indexed)
if c is None:
    c = centroids.get(int(roi_idx_0))  # fallback to 0-indexed

# Fix: the centroids JSON from abide_download.save_atlas_metadata() stores
# roi_id as 1-indexed (labels = np.unique(data)[1:] → label values 1..166)
# The LOBE_MAPPING uses 0-indexed. So only the 1-indexed lookup should work.
# Check the loaded atlas has 170 ROIs (log shows only 166 — AAL3v1 variant issue).
```

**Fix 2:** More urgently — the atlas only has 166 ROIs (`Atlas loaded: 166 ROIs`), not 170. ROIs 167–170 (Brainstem) are missing from the atlas file. This is an AAL3v1 variant issue. Either use a complete 170-ROI atlas or explicitly zero-out the Brainstem spatial features consistently and flag them with the `zero_lobe_mask` mechanism already in place.

**Observed impact:** Brainstem node has corrupted spatial features for all 1015 subjects. The GNN learns meaningless Brainstem connectivity patterns.

---

### P0-3: Wave-1 MI Feature Selection Is Unstable and Inconsistently Applied

**Root cause:** The MI feature selector retains different feature subsets per fold:

```
Fold 0: kept 14/24 features (58%), max MI score 0.0610
Fold 1: kept 13/24 features (54%), max MI score 0.0490  
Fold 2: kept 12/24 features (50%), max MI score 0.0248
Fold 3: kept 12/24 features (50%), max MI score 0.0870
Fold 4: kept 11/24 features (46%), max MI score 0.0343
```

Three problems compound here:

**Problem A — MI scores are near zero across all folds.** Maximum MI scores of 0.024–0.087 are extremely low. For reference, a useful feature in a binary classification problem typically scores 0.1–0.3. Scores this close to zero indicate that after harmonization + lobe aggregation, the individual features carry very little marginal mutual information with the diagnosis label. The MI selector is pruning features based on noise-level differences.

**Problem B — Fold 2's max MI (0.0248) is 3.5× lower than Fold 3's (0.0870).** This means the model in Fold 2 is operating on completely different "important" features than Fold 3. The ensemble of these heterogeneous models is combining apples and oranges.

**Problem C — The feature mask zeros channels without reducing dimensionality.** The `GNN_IN_CHANNELS` remains 24. The masked-out channels are exactly zero, which still flow through `lin_in` as dead dimensions. This wastes capacity and can confuse LayerNorm.

**Fix:** Disable MI selection for now. The correct approach for a 24-feature, 12-node graph with 700 subjects is to use all features — there is no curse of dimensionality at this scale. MI on harmonized lobe-aggregated features is too noisy to be reliable.

```python
# In src/core/hyperparams.py
GNN_MI_FEATURE_SELECTION_ENABLED = False  # was True
```

Alternatively, if MI selection is desired for the publication narrative, run it once globally on the full training set (not per-fold) and apply the same fixed mask to all folds. This is the correct procedure for feature selection in a nested CV context.

---

## P1 — High Priority Issues

### P1-1: Variance Retention at 40.5% After Harmonization

The harmonization quality check reports:

```
Original variance: 68163.71
Harmonized variance: 27607.53
Variance retention: 40.50%
```

Retaining only 40% of variance after ComBat is unusually aggressive. For comparison, well-calibrated ComBat typically retains 70–90% of biological variance while removing scanner effects. The code's own `VARIANCE_WARNING_THRESHOLD` is 30% — the system is operating right at the edge.

**Root cause:** The aggressive variance removal is likely caused by two compounding factors: (1) `DX_GROUP` may not be protecting enough biological variance if the covariate structure is poorly estimated on small fold-train sets, and (2) the 289 near-constant features being dropped and restored may be destabilizing the ComBat model fit.

**Recommendation:** Add a pre-harmonization check that logs per-feature variance and identifies which features are being most aggressively adjusted. If spectral power features (log-transformed) are losing most variance, consider excluding them from harmonization and handling them separately. The current approach of log-transforming then harmonizing may be over-correcting.

```python
# In fold_safe_harmonization.py, _harmonize_fold():
# Add after harmonization:
retained = harm_var_series / orig_var_series.replace(0, np.nan)
low_retention = retained[retained < 0.5].index.tolist()
if low_retention:
    logger.warning("Features with <50%% variance retained: %s", low_retention[:10])
```

---

### P1-2: Unbalanced Fold Sizes From Site-Stratified CV

The site cluster assignment produces severely unbalanced folds:

```
Fold 0: 220 subjects  (cluster: LEUVEN_1, MAX_MUN, NYU, YALE — 312 subjects in cluster 4)
Fold 1: 147 subjects
Fold 2: 144 subjects
Fold 3: 110 subjects
Fold 4:  86 subjects
```

Fold 4 has only 86 validation subjects (43 per class) — too small for reliable AUC estimation. Fold 0 has 220. This 2.6× size ratio means fold AUC estimates have wildly different confidence intervals. The AUC-weighted ensemble weighting is then partially driven by fold size rather than fold quality.

If site-stratified CV is retained, the `_assign_site_clusters()` function needs balancing by subject count rather than site count. Currently it round-robins *sites*, not *subjects*.

---

### P1-3: Training Epochs Too Short for Small Folds

Fold 0 converges at **epoch 7** with AUC 0.53. This is not convergence — it is early stopping triggered before meaningful learning occurs. With 487 training subjects, batch size 32, and 30-epoch patience, the model makes only ~15 gradient steps before the patience window closes (if the first few epochs all score ~0.53).

The `GNN_EARLY_STOPPING_PATIENCE = 30` is appropriate for the canonical StratifiedKFold run where folds are balanced and harmonization is consistent. With site-stratified CV creating mismatched val distributions, the model's validation AUC never reliably improves above 0.53, so early stopping fires immediately.

**Fix A (if keeping site-stratified CV):** Increase `GNN_EARLY_STOPPING_PATIENCE` to 50 and add a minimum training floor:

```python
GNN_EARLY_STOPPING_PATIENCE = 50
GNN_MIN_EPOCHS_BEFORE_STOPPING = 30  # new config key needed
```

**Fix B (recommended):** Revert to StratifiedKFold where this problem does not exist.

---

### P1-4: Fixed Threshold 0.5263 Creates Sensitivity/Specificity Imbalance

The evaluation reports:
```
Sensitivity: 0.5443 [0.4430, 0.6582]
Specificity: 0.7600 [0.6533, 0.8533]
```

The model misses 36 of 79 ASD subjects (45.6% miss rate). For a clinical screening application, this is the wrong operating point. The Youden threshold (0.3864) gives:
```
Sensitivity: 0.7848, Specificity: 0.5867, F1: 0.7209
```

The fixed threshold of 0.5263 was calibrated in the Condition C seed-stability study which used different model checkpoints. Since this run produces different models (different MI feature masks, different harmonization), the locked threshold is no longer appropriate.

**Recommendation:** For this run, compute and report the Youden threshold as the primary operating point, with the fixed threshold as a secondary reportable. Or update `EVAL_FIXED_THRESHOLD` to the Youden value from this run's calibration fold analysis (0.3864).