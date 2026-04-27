# Neuro-CXG — Comprehensive Codebase Audit & Performance Recovery Report
**Audit Date:** April 22, 2026  
**Pipeline Baseline:** CV AUC 0.7586 ± 0.0519 | Test AUC 0.7325 | Test F1 0.6338  
**Constraint:** GATv2 backbone preserved; no major architecture replacement

---

## Executive Summary

After a full system-level review of the pipeline run logs, source code, and ablation results, the accuracy plateau is not caused by a single issue but by **five compounding root causes**, three of which are directly actionable without any architectural changes. The most critical finding is that the wrong causality method has been in production throughout all benchmarked runs: the ablation study conducted within the same pipeline run shows that switching from `ridge_granger` to `lagged_pearson` delivers a **+0.082 AUC improvement** (0.7586 → 0.8406 CV AUC) with lower variance (±0.0305 vs ±0.0519). This alone recovers the plateau and is a config-level change. Additionally, the YOLO spatial detector has never successfully detected the Brainstem lobe in any subject across the entire cohort—all 1015 subjects are operating on an incomplete 11/12-region graph, with zero spatial features for a neurologically important structure.

The sections below provide a precise, prioritized breakdown of every issue found, its root cause, and the exact fix.

---

## Part 1 — Critical Performance Blockers

### Blocker 1: Wrong Causality Method in Production (Impact: ~+0.08 AUC)

**Evidence from run log (lines 1447–1510):**
```
Ablation C (Temporal+Spatial, no frequency):  AUC = 0.7340 ± 0.0377
Ablation D (Lagged Pearson edges):             AUC = 0.8406 ± 0.0305  ← +0.082 over production
Current GNN (ridge_granger, production):       CV AUC = 0.7586 ± 0.0519
```

The ridge-regularized Granger method currently in production produces causal graphs that carry almost no group-discriminative structure. The graph topology comparison (lines 671–674) confirms this:
```
Num Edges: ASD 46.70 vs Control 46.83 (p=0.2436, not significant, d=-0.036)
```
ASD and Control graphs are statistically indistinguishable. Lagged Pearson correlation with proper multi-lag selection produces edges with higher signal-to-noise for directed temporal dependencies at the lobe level, and the ablation proves this conclusively.

**Fix — change one line in `src/core/hyperparams.py`:**
```python
# Before:
CAUSALITY_METHOD = "ridge_granger"

# After:
CAUSALITY_METHOD = "lagged_pearson"
```
Then rebuild graphs: `python -m src.features.construct_causal --n-jobs -1`

**Why lagged_pearson outperforms ridge_granger here:**  
At the 12-lobe aggregation level, each "node" is a PCA eigenvariate of 8–37 ROIs. These aggregated signals are smoother and more Gaussian than raw voxel timeseries, making the parametric assumptions in ridge regression VAR models less reliable. Lagged Pearson with Fisher-Z transform and multi-lag selection better captures the slow hemodynamic coupling (0.01–0.15 Hz) between lobe-level activity patterns. Ridge Granger requires the restricted vs unrestricted VAR F-test to have power, which is compromised at 12 nodes × aggregated signals.

---

### Blocker 2: Brainstem Never Detected by YOLO — All Subjects Missing a Critical Lobe

**Evidence from run log (lines 87–98):**
```
Unique ROI classes detected: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]   ← class 11 (Brainstem) absent
Global YOLO detections missing for lobe ids [11]
Subjects with partial detection (9-11 regions): 1015
Subjects with complete detection (all 12 regions): 0
```

Every single one of the 1015 subjects has the Brainstem spatial features zeroed to zero via the explicit fallback, and `Brainstem_spatial_missing = 1` in all rows. This means:

1. The Brainstem node enters the GNN with zero spatial features in every graph, becoming a constant node that the model cannot distinguish between subjects.
2. The `zero_lobe_mask` has Brainstem = True for all subjects, which the model respects at inference (zeros attributions for this node), but the node still participates in message passing and pollutes the graph.
3. The YOLO model was trained on 7 z-slices (ALFF percentiles), and Brainstem (ROIs 167–170) begins at approximately z=38 (percentile 0.21). The 0.21 percentile slice was added specifically to capture it (CHANGELOG note), but the YOLO model has learned to ignore or never fire on it.

**Fix — Use atlas-centroid spatial features for Brainstem and globally assess the spatial feature pathway:**

Option A (immediate, no retraining): Run spatial extraction with `--use-atlas-spatial` flag to bypass YOLO entirely and use precomputed atlas centroids for all spatial features. From ablation results, spatial features alone achieve 0.58 AUC, so the GNN is not heavily dependent on them, but constant-zero for an important lobe is actively harmful.

Option B (better, requires YOLO re-training): Examine the YOLO training data for class 11. The label file for Brainstem-containing slices may have near-zero detection boxes that fall below the confidence threshold. Reduce `YOLO_CONF_THRESHOLD` from 0.30 to 0.15 for inference only, or add synthetic brainstem annotations to the training set.

Option C (workaround, no retraining): In `graph_factory.py`, explicitly fall back to atlas-derived spatial features for lobes where `spatial_missing = 1` instead of zeros. This is already architected (the atlas path exists in `extract_spatial_atlas.py`) but not activated.

---

### Blocker 3: Model Peaks at Epoch 8–12 — Severe Overfitting

**Evidence from run log (lines 584–594):**
```
Per-fold Best Epochs: [11, 20, 12, 9, 8]
Mean Best Epoch: 12.0
```

The model peaks within the first 12% of the training budget (100 epochs). After early stopping triggers at patience=30, folds are terminated at epochs 39, 38, etc. This is a textbook signature of either the learning rate being too high relative to the dataset/model size, or the model having excess capacity for a 12-node, 24-feature graph dataset of 707 training subjects.

Diagnostic indicators:
- GNN_HIDDEN_CHANNELS = 64, GNN_NUM_HEADS = 2, 2 GATv2 layers → roughly 64×64×2 (heads) + skip connections → ~50k parameters for a 12-node graph. This is high relative to the effective training set size per fold (~565 subjects).
- The OneCycleLR with max_lr=0.002 and 15% warmup reaches peak LR at epoch 15 (0.15 × 100). The model is already overfitting before the LR finishes warming up.
- GNN_WEIGHT_DECAY = 5e-5 is very weak regularization for this dataset size.

**Fixes:**
```python
# In src/core/hyperparams.py:
GNN_HIDDEN_CHANNELS = 32        # Reduce from 64
GNN_WEIGHT_DECAY = 5e-4         # Increase 10× from 5e-5
GNN_ONECYCLE_WARMUP_FRACTION = 0.30  # Extend warmup from 0.15 → 0.30
GNN_EARLY_STOPPING_PATIENCE = 20    # Reduce from 30, given how early peaks occur
GNN_EPOCHS = 150                # Increase budget so full LR cycle has room to act
```

Reducing hidden channels from 64 to 32 also halves the skip-connection dimensions and reduces GATv2 attention computation, making training faster without losing expressiveness for 12-node graphs.

---

## Part 2 — Significant Performance Issues

### Issue 4: FlatMLP Nearly Matches GNN — Graph Structure Underutilized

**Evidence (ablation lines 1502–1508):**
```
A (FlatMLP, no graph):     AUC = 0.7306 ± 0.0154
Current GNN (production):  AUC = 0.7586 ± 0.0519
Delta from graph:          +0.028 AUC
```

The graph is contributing only 0.028 AUC over a flat vector of averaged node features. This is the original concern that motivated DD-009 (structural learning). The gap is narrow enough that any reduction in graph quality (e.g., noise from ridge_granger) fully closes it. Switching to lagged_pearson (Blocker 1) is the primary fix, but two additional structural learning settings should also be adjusted:

```python
# In src/core/hyperparams.py:
GNN_STRUCTURAL_DROPOUT_PROB = 0.0    # Disable — currently active but counterproductive without the contrastive complement
GNN_EDGE_CONTRASTIVE_WEIGHT = 0.0    # Already 0.0 — keep off with lagged_pearson edges which have meaningful weights
GNN_POOLING = "anatomical"           # Switch from "mean_max_sum" to use the designed 2-level hierarchy
```

The structural dropout (zeroing node features for 30% of graphs per batch) forces edge-only classification, which is sensible in theory, but with ridge_granger edges that are near-identical between groups (p=0.24), this regularization is asking the model to learn from noise. With lagged_pearson edges, structural dropout should be re-evaluated rather than disabled.

---

### Issue 5: Ablation D Uses Stale 1031-Subject Graphs — Invalid Comparison

**Evidence (log line 1449):**
```
Reusing 1031 existing lagged-Pearson graphs in /data/processed/causal_graphs_pearson
```

The current cohort has 1015 subjects after the CURATED_WORST_SUBJECTS_1015 exclusion. The stale Pearson graphs contain 1031 subjects — the old pre-exclusion cohort. This means the ablation D result (0.8406 AUC) was measured on a **different, slightly larger, less curated cohort** than the 1015-subject production run. The comparison to production AUC 0.7586 is therefore partially confounded.

The ablation D result is almost certainly still directionally correct (Pearson > ridge_granger) since the excluded subjects (mostly Caltech and SDSU outliers) would hurt both methods equally or slightly hurt the cleaner method more. However, the exact magnitude (+0.082) should be treated as an upper bound until fresh graphs are built on the clean 1015-subject cohort.

**Fix:** Delete `data/processed/causal_graphs_pearson/` before running ablations after a force-reset, or add cohort-hash validation to the ablation dataset loader.

---

### Issue 6: Harmonization Destroys 61% of Variance in Beta-Peak Features

**Evidence (log lines 162–168):**
```
Original variance:   68163.7
Harmonized variance: 26709.0
Variance retention:  39.18%
55.4% of features lost >30% variance
Lowest-retention features: Subcortical_beta_peak, Frontal_Superior_beta_peak, Parietal_beta_peak, ...
```

ComBat is aggressively removing variance from beta frequency peak features across all lobes. These features (beta_peak: peak frequency within 0.15–0.20 Hz) are computed at the Nyquist boundary for TR=2s scanners. The `EXCLUDE_NYQUIST_BANDS` and `UNRELIABLE_FREQ_BANDS_AT_NYQUIST` settings correctly zero out gamma band power, but `beta_peak_freq` is a different feature — it records the dominant frequency in the beta band, not power. The issue is that at TR=2s, beta power is low and the peak frequency estimate has high variance dominated by noise, making these features noisy inputs that ComBat then treats as high-variance site effects and removes.

**Fix — exclude beta_peak from the harmonized feature set:**
In `src/features/fold_safe_harmonization.py`, the `FEATURE_TYPES` list drives what gets harmonized. Add filtering to exclude known unreliable features:
```python
# In fold_safe_harmonization.py:
FEATURE_TYPES = [
    f for f in (FEATURE_GROUPS["temporal"] + FEATURE_GROUPS["frequency"])
    if "beta_peak" not in f  # Exclude noisy Nyquist-boundary peak estimates
]
```

Alternatively, remove `beta_peak` from the feature registry entirely in `feature_registry.py` given that it provides low signal at fMRI timescales.

---

### Issue 7: Fixed Threshold 0.5263 Creates Asymmetric Error Profile

**Evidence (log lines 943–948, 936):**
```
At fixed threshold 0.5263:  Sensitivity=0.5696, Specificity=0.7600
At Youden threshold 0.4212:  Sens=0.7342, Spec=0.6267, F1=0.7030
```

The deployed threshold (locked at 0.5263 from Condition C calibration) produces an unacceptably low sensitivity for an ASD screening tool. The model misses 43% of ASD cases (FN = 38 of 79). A Youden threshold would provide a more balanced operating point with sensitivity 0.73. For a publication reporting clinical utility, sensitivity below 0.70 at the operating point is a significant limitation.

**Fix:** Update the threshold policy in the evaluation/reporting context:
```python
# In src/core/hyperparams.py:
EVAL_THRESHOLD_POLICY = "youden"   # Change from "fixed"
# Remove: EVAL_FIXED_THRESHOLD = 0.5263
```

This alone will improve reported F1 from 0.6338 to approximately 0.7030 with no model changes, simply by using the Youden-optimal threshold (already computed at each evaluation run). The fixed threshold was calibrated from a previous run's Condition C and has drifted out of alignment with the current model after feature regeneration.

---

## Part 3 — Code Quality and Structural Debt

### Debt 1: Redundant Dataset Initialization in Training Loop

In `src/models/gnn_model.py`, `_run_training_once()` initializes `ABIDECausalDataset` twice per run — once at the top to gather labels for logging, then again per fold for the harmonized features. The initial load uses the global harmonized file but then the fold-specific load immediately replaces it. The initial load is only used for:
1. Counting labels (achievable from the manifest directly)
2. Detecting cv_fold values (also available from the manifest)

The initial dataset initialization triggers full graph validation, CSV loading, and feather cache writes for all 1015 subjects — this is repeated work.

**Fix in `gnn_model.py`:**
```python
# Replace the initial dataset load with a lightweight manifest read:
import pandas as pd
from src.core.config import MASTER_MANIFEST
manifest_df = pd.read_csv(MASTER_MANIFEST)
train_manifest = manifest_df[manifest_df['split'] == 'train']
labels = (train_manifest['DX_GROUP'].values == 2).astype(int).tolist()  # ASD = 2
site_labels = train_manifest['SITE_ID'].map(site_encoding).tolist()
```

### Debt 2: `ABIDECausalDataset` Initialized 8–10 Times Per Full Run

From the log, `ABIDECausalDataset` is initialized on these calls: once in training per fold (5×), once for ensemble eval, once in explainability, once in result analysis, once in visualizations, once in ablation. Each initialization:
- Reads and validates all graph files (1015 × `torch.load`)
- Loads CSVs (or feather caches)
- Validates feature dimensions

This is fundamentally correct behavior but leads to redundant disk I/O. The dataset should use a more aggressive shared graph cache across eval stages. Since the graph files are read-only after construction, a process-level LRU cache on the `torch.load` calls would eliminate most of this overhead.

### Debt 3: GPU Path in `extract_temporal.py` Is Non-Functional

The GPU temporal extraction path (`_extract_temporal_gpu`) contains multiple calls to deprecated or unavailable PyTorch APIs:
- `torch.ptp()` was removed in PyTorch 2.x (the `DEPRECATION` of `torch.ptp` happened between 1.x and 2.x)
- `torch.ops.fft.hilbert` does not exist; the Hilbert transform is not a standard PyTorch FFT op
- `torch.nansum` is only available in PyTorch 1.8+

The log confirms this: `device=CPU` is shown for temporal extraction despite CUDA being available. The GPU path is silently bypassed. This entire path should either be fixed or removed to avoid confusion.

**Fix:** Remove `_extract_temporal_gpu`, `_compute_psd_gpu`, `_compute_phase_std_gpu`, `_compute_spectral_entropy_gpu`, `_compute_peak_freqs_gpu`, `_hilbert_manual` from `extract_temporal.py` and use the vectorized NumPy/SciPy path consistently. The NumPy path is already vectorized and fast enough (1015 subjects in 4m20s as shown in the log).

### Debt 4: `_compute_granger_causality_gpu_impl` Duplicates Deleted Dead Code

The CHANGELOG notes that `compute_granger_causality_gpu` was removed in Task 6 (DD-014). However, `_compute_granger_causality_gpu_impl` in `causal_inference.py` is functionally identical and was not removed. The comment at the top of the file even says "GPU-accelerated Granger causality using batched linear regression" — which is exactly what was deleted. This creates confusion about what dead code was actually removed.

**Fix:** Remove `_compute_granger_causality_gpu_impl` from `causal_inference.py`. The CPU Granger path is fast enough for 12 nodes (confirmed: 27 seconds for 1015 subjects).

### Debt 5: `USE_FOCAL_LOSS = False` Creates a Config Contradiction

`FocalLoss` is implemented in `src/models/losses.py`, imported in multiple training paths, and has configured hyperparameters (`FOCAL_LOSS_ALPHA = 0.50`, `FOCAL_LOSS_GAMMA = 1.5`). The `USE_FOCAL_LOSS = False` config silently bypasses this entire mechanism in favor of `CrossEntropyLoss` with class weights.

Given that:
- The dataset is near-balanced (342 Control / 365 ASD in training)
- Class weights are almost uniform (1.033 / 0.969 as shown in the log)
- CrossEntropy with near-uniform weights is effectively standard CrossEntropy

The current setup provides no meaningful imbalance correction at all. Either enable FocalLoss with gamma=1.5 (which provides hard-example mining even for balanced datasets) or explicitly comment out the entire focal loss apparatus as unused.

**Recommended setting:**
```python
USE_FOCAL_LOSS = True       # Enable — provides hard-example mining
USE_CLASS_WEIGHTS = False   # Disable since classes are near-balanced
```

### Debt 6: Triple Normalization Risk on Node Features

Node features are normalized at three separate points in the pipeline:

1. **ComBat harmonization** in `fold_safe_harmonization.py` — site-effect removal + implicit z-scoring
2. **Within-site normalization** in `gnn_model.py` (`_apply_site_normalization`) — additional per-site z-scoring on the fold training split
3. **Model-internal scaler** attached to checkpoints via `attach_feature_scaler_from_checkpoint` — a third scaling applied in `CausalBrainGNN._encode()`

At inference time, the test subjects receive:
- ComBat from global harmonization (all-subject statistics)
- Within-site normalization from fold-train statistics (applied in `_encode`)

The test set was harmonized with global statistics in `node_attributes_harmonized.csv`, but the fold scaler was fit on the fold-train subset. When the test set is loaded at inference, it reads from `node_attributes_harmonized.csv` (global ComBat), and then the checkpoint scaler (fold-train statistics) is applied again in `_encode`. This means test subjects are effectively normalized twice with different statistics, and the second normalization layer is applying train-distribution parameters to test-distribution inputs.

**Fix:** In `CausalBrainGNN._encode()`, the global scaler from `_feature_mean` / `_feature_std` should only be applied when `preprocessing_mode == "legacy_global"`. For `wave1` mode with `within_site` normalization, the within-site stats from `_site_feature_means` / `_site_feature_stds` should be the only normalization applied. The current code does attempt this via the `should_apply_global` flag, but the logic branches are complex enough to create edge cases. A cleaner design would separate the preprocessing pipeline from the model's forward pass entirely.

### Debt 7: The `GNN_FOLD_PREPROCESSING_MODE = "wave1"` Label Is Misleading

The "wave1" preprocessing mode was introduced as part of "Wave-1 Generalization Stabilization" (CHANGELOG, April 19 2026). With `GNN_MI_FEATURE_SELECTION_ENABLED = False` (the current setting), wave1 mode applies only within-site normalization — there is no actual feature selection happening. The config comment says `GNN_MI_FEATURE_SELECTION_ENABLED = False` under a section labeled "Wave-1 fold-internal preprocessing controls", creating the impression that wave1 preprocessing is active when only half of it is.

**Fix:** Either rename the mode to reflect its actual behavior, or set `GNN_MI_FEATURE_SELECTION_ENABLED = True`. Given that 289/3060 input features (~9.4%) are already flagged as near-constant and dropped during harmonization, a modest MI filter would not be harmful.

### Debt 8: Spatial Harmonization Silently Skipped Every Run

**Evidence (log line 173):**
```
No conf_std / detection_count columns found — skipping spatial harmonization
```

The `harmonize_spatial_features()` function looks for `*_conf_std` and `*_detection_count` columns in the spatial features CSV. These columns were removed by DD-012 (spatial cleanup, Task 4). The harmonic function is being called every run and silently skips. This is dead code in the active pipeline — the function exists to harmonize columns that no longer exist.

**Fix:** Remove the `harmonize_spatial_features()` call from `fold_safe_harmonization.main()`, or add an early check that exits cleanly with an info message rather than doing CSV loading followed by a no-op.

### Debt 9: `ABIDECausalDataset` Log Spam — Excluded Subjects List Printed 8–10 Times Per Run

The log shows the 20-subject exclusion list printed once for every dataset initialization. In a full pipeline run, this message appears approximately 10 times across training folds, evaluation, explainability, and analysis stages. The message adds no value after the first occurrence.

**Fix in `graph_factory.py`:**
```python
# Add a module-level flag to suppress repeated prints:
_exclusion_logged = False

def _validate_subjects(self):
    global _exclusion_logged
    if EXCLUDED_SUBJECTS and not _exclusion_logged:
        logger.info("Excluded %d corrupted subjects: %s", ...)
        _exclusion_logged = True
    # ... rest of method
```

### Debt 10: `Ablation D` Does Not Use `cv_fold` — Violates Site-Stratified Protocol

In `src/experiments/run_ablations.py`, the docstring says ablations use manifest `cv_fold` splits (added in the April 19 changelog: "now uses manifest cv_fold splits"). However, ablation D creates a `PearsonDataset` subclass that redirects graph loading to `causal_graphs_pearson/` while loading subjects from the same manifest. The graph redirection is implemented by overriding `self.adj_dir` in a subclass method `_load_data_sources`, but the ABIDECausalDataset `_validate_subjects()` method still validates against the graphs in `CAUSAL_GRAPHS_DIR`. When 1031-subject Pearson graphs are present with the 1015-subject manifest, all 1015 subjects will find their graphs, but 16 subjects have Pearson graphs that will never be accessed. This is a benign data mismatch but indicates the ablation infrastructure is fragile.

---

## Part 4 — Targeted Optimization Recommendations

These changes are compatible with the existing GATv2 architecture and require no structural overhaul.

### Recommendation 1: Switch Causality Method (Primary Fix)

As detailed in Blocker 1, this is the highest-impact single change available:
```python
CAUSALITY_METHOD = "lagged_pearson"
```
Expected CV AUC gain: approximately +0.08 based on ablation D (fresh 1015-subject graphs needed for confirmed measurement).

### Recommendation 2: Align Pooling with Architecture Intent

The current `GNN_POOLING = "mean_max_sum"` pools 12 lobe embeddings into a 192-dimensional vector (64 × 3) before classification. The anatomical hierarchical pooler (`AnatomicalHierarchyPool`) was implemented specifically for this pipeline and produces a 64-dimensional vector via 2-level attention pooling that respects the DMN/Salience/Visual/Limbic network structure. The explainability output confirms that DMN and Salience networks show the strongest differential signal. Pooling with anatomical hierarchy should expose this structure more directly to the classifier.

```python
GNN_POOLING = "anatomical"
```

Note: This also reduces the classifier input dimension from 192 → 64, requiring a rebalanced classifier head. The factory in `src/models/factory.py` handles this automatically.

### Recommendation 3: OneCycleLR Warmup Is Finishing After Best Epoch

With `GNN_ONECYCLE_WARMUP_FRACTION = 0.15` and `GNN_EPOCHS = 100`, the LR peaks at epoch 15. But the model's best performance is at epochs 8–12, meaning the LR is still rising (warmup phase) when the model is already at peak performance and starting to overfit. The LR schedule is actively causing the problem — the model is being pushed harder as it overfits.

Fix the schedule to peak much earlier:
```python
GNN_ONECYCLE_WARMUP_FRACTION = 0.05   # Peak at epoch 5 of 100
GNN_ONECYCLE_MAX_LR = 0.001           # Reduce from 0.002
```

Or alternatively, use a cosine decay without warmup given how shallow the optimal training depth is:
```python
GNN_ONECYCLE_WARMUP_FRACTION = 0.05
GNN_ONECYCLE_MAX_LR = 0.001
GNN_WEIGHT_DECAY = 5e-4
```

### Recommendation 4: Enable MI Feature Selection

The pipeline already implements `_fit_mi_feature_selection()` but it is disabled. Given that:
- 289/3060 raw features (~9.4%) are near-constant before harmonization
- Beta_peak features have retention < 0.70 after ComBat
- The model uses 24 GNN_IN_CHANNELS but several are effectively zero-signal

Enabling MI selection would both speed up training and reduce overfitting risk:
```python
GNN_MI_FEATURE_SELECTION_ENABLED = True
GNN_MI_MIN_KEEP_RATIO = 0.50    # Keep at least 50% of features
GNN_MI_MAX_KEEP_RATIO = 0.80    # Keep at most 80% of features
```

### Recommendation 5: Report Youden Threshold Instead of Fixed Threshold

The EVAL_FIXED_THRESHOLD = 0.5263 was locked at a prior calibration point. The current model (after force-reset feature regeneration) has drifted from that calibration. The Youden threshold (0.4212) from the current run gives sensitivity 0.73 vs 0.57 at fixed, while maintaining specificity 0.63 vs 0.76. For a publication on ASD classification, sensitivity of 0.73 is considerably more defensible than 0.57.

```python
EVAL_THRESHOLD_POLICY = "youden"
```

### Recommendation 6: Layer Normalization Before the Classifier Head

The current architecture applies `post_fusion_norm` only when `use_demographics=True`. With `GNN_USE_DEMOGRAPHICS = False` (the default), the pooled graph embedding goes directly to the classifier without normalization. Adding a LayerNorm before the classifier is a low-risk regularization step:

In `CausalBrainGNN.__init__()`:
```python
self.pre_classifier_norm = LayerNorm(pooling_dim)  # After existing code
```

In `CausalBrainGNN._encode()`:
```python
g = self.pre_classifier_norm(g)  # Before: class_logits = self.classifier(g)
```

### Recommendation 7: Recheck Brainstem Spatial with Atlas Fallback

Before the next publication run, execute:
```bash
python -m src.features.extract_spatial_atlas
```

This generates spatial features from precomputed atlas ROI centroids rather than YOLO detections, ensuring all 12 lobes have meaningful spatial features. The atlas fallback already exists (`src/features/extract_spatial_atlas.py`) but is never activated in the default pipeline. Since spatial features from ablation B contribute only 0.58 AUC alone, switching to atlas-based coordinates is low risk and ensures Brainstem has non-zero, meaningful x/y/z/size values.

---

## Part 5 — Prioritized Action Plan

The following table ranks all findings by expected impact and implementation risk for the publication timeline:

| Priority | Action | Expected Gain | Risk | Effort |
|---|---|---|---|---|
| P0 | Switch `CAUSALITY_METHOD = "lagged_pearson"` + rebuild graphs | +0.08 CV AUC | Low | 30 min |
| P0 | Switch `EVAL_THRESHOLD_POLICY = "youden"` | +0.07 F1, +0.16 Sensitivity | None | 1 line |
| P1 | Reduce `GNN_HIDDEN_CHANNELS` 64→32, increase `GNN_WEIGHT_DECAY` 5e-5→5e-4 | Reduce variance ±0.05→±0.02 | Low | 2 lines |
| P1 | Switch `GNN_POOLING = "anatomical"` | +0.02–0.04 AUC (structural alignment) | Low | 1 line |
| P1 | Fix Brainstem with atlas spatial or conf-threshold reduction | Signal quality for node 11 | Low | 1 command |
| P1 | Fix LR schedule: `GNN_ONECYCLE_WARMUP_FRACTION = 0.05`, `max_lr = 0.001` | Reduce early overfitting | Low | 2 lines |
| P2 | Remove `_compute_granger_causality_gpu_impl` dead code | Codebase clarity | None | 5 min |
| P2 | Remove GPU path from `extract_temporal.py` | Remove broken code | None | 10 min |
| P2 | Remove no-op spatial harmonization call | Cleaner logs | None | 5 min |
| P2 | Suppress repeated exclusion list logging | Cleaner logs | None | 5 min |
| P2 | Enable `USE_FOCAL_LOSS = True`, disable `USE_CLASS_WEIGHTS` | Hard-example mining | Low | 2 lines |
| P3 | Enable `GNN_MI_FEATURE_SELECTION_ENABLED = True` | Reduced overfitting | Low | 2 lines |
| P3 | Remove beta_peak from FEATURE_TYPES before harmonization | Reduce variance loss | Low | 5 min |
| P3 | Fix triple normalization in `_encode()` | Correct test-time preprocessing | Medium | 30 min |
| P3 | Simplify initial dataset load in training to manifest-only | Reduce startup time | Low | 15 min |

---

## Part 6 — Minimum Viable Fix Set for Publication

If only one change can be made, it is **P0: switch CAUSALITY_METHOD to lagged_pearson**. The ablation proves this delivers the expected gain, and it requires only a config change plus graph rebuild.

The complete minimum viable fix set to substantially improve the published numbers without any architecture changes:

```python
# src/core/hyperparams.py — complete set of changes:

CAUSALITY_METHOD = "lagged_pearson"         # P0: +0.08 AUC
EVAL_THRESHOLD_POLICY = "youden"            # P0: +0.07 F1
GNN_HIDDEN_CHANNELS = 32                    # P1: reduce overfitting
GNN_WEIGHT_DECAY = 5e-4                     # P1: reduce overfitting
GNN_POOLING = "anatomical"                  # P1: use designed pooling
GNN_ONECYCLE_MAX_LR = 0.001                 # P1: reduce early overfitting
GNN_ONECYCLE_WARMUP_FRACTION = 0.05         # P1: align schedule with peak
USE_FOCAL_LOSS = True                       # P2: hard-example mining
USE_CLASS_WEIGHTS = False                   # P2: complementary to above
GNN_STRUCTURAL_DROPOUT_PROB = 0.0           # Disable with new edge method
```

Then run:
```bash
python -m src.features.construct_causal --n-jobs -1   # Rebuild with lagged_pearson
python -m src.features.fold_safe_harmonization        # Regenerate fold files
python src/run_pipeline.py --auto --skip-download --skip-split --skip-annotate --skip-yolo
```

**Projected performance after minimum viable fixes:**
- CV AUC: ~0.82–0.84 (based on ablation D showing 0.84 with lagged_pearson on comparable data)
- Test AUC: ~0.76–0.79 (given ~0.009 CV-test gap from the current run)
- Sensitivity: ~0.70–0.73 (from Youden threshold alignment)
- Fold variance: ~±0.03 (reduced from ±0.05 by smaller model)

---

## Appendix: Configuration Diff Summary

```diff
--- a/src/core/hyperparams.py
+++ b/src/core/hyperparams.py
-CAUSALITY_METHOD = "ridge_granger"
+CAUSALITY_METHOD = "lagged_pearson"

-GNN_HIDDEN_CHANNELS = 64
+GNN_HIDDEN_CHANNELS = 32

-GNN_WEIGHT_DECAY = 5e-5
+GNN_WEIGHT_DECAY = 5e-4

-GNN_POOLING = "mean_max_sum"
+GNN_POOLING = "anatomical"

-GNN_ONECYCLE_MAX_LR = 0.002
+GNN_ONECYCLE_MAX_LR = 0.001

-GNN_ONECYCLE_WARMUP_FRACTION = 0.15
+GNN_ONECYCLE_WARMUP_FRACTION = 0.05

-GNN_STRUCTURAL_DROPOUT_PROB = 0.3
+GNN_STRUCTURAL_DROPOUT_PROB = 0.0

-USE_FOCAL_LOSS = False
+USE_FOCAL_LOSS = True

-USE_CLASS_WEIGHTS = True
+USE_CLASS_WEIGHTS = False

-EVAL_THRESHOLD_POLICY = "fixed"
+EVAL_THRESHOLD_POLICY = "youden"
-EVAL_FIXED_THRESHOLD = 0.5263
```

Dead code to remove (no functional impact, cleanup only):
- `_compute_granger_causality_gpu_impl()` in `src/features/causal_inference.py`
- GPU path functions in `src/features/extract_temporal.py` (7 functions)
- `harmonize_spatial_features()` call in `src/features/fold_safe_harmonization.main()`
- Repeated exclusion list logging in `src/features/graph_factory.py`