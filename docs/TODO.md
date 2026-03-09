## Implementation Plan — Neuro-CXG Audit

### Summary

Three root-cause issues drive the CV–Test AUC gap of ~0.08 and currently corrupt all reported metrics: (1) `DX_GROUP` is silently absent from ComBat covariates so harmonization removes the diagnostic signal it is supposed to protect, (2) `site_id=None` is hardcoded in `run_evaluation.py` while the trained model expects real 16-dim site embeddings, and (3) test-set labels are used to weight the ensemble in the same function that then scores the ensemble — circularly inflating reported test AUC. Fixing these three issues first is mandatory before any architectural change can be meaningfully evaluated. The 15 steps that follow address the remaining pipeline, architecture, evaluation, and quality defects in safe dependency order; batching the data-rebuild steps before retraining will limit the total number of full pipeline reruns to three.

---

## P0 — Critical Fixes

---

### Step 1: Add DX_GROUP to ComBat covariates
**Category**: P0 Critical  
**Requires Retraining**: Partial — re-run Stage 12 (harmonization) and Stages 14–16  
**Files to modify**: `src/features/fold_safe_harmonization.py`  
**Depends on**: none

**What**

In `_prepare_covariates()`, `DX_GROUP` is constructed from `manifest_df` but the returned feature matrix only ever contains `[SITE_ID, AGE_AT_SCAN, SEX]`. The call to `harmonizationLearn` therefore removes ASD-correlated variance instead of protecting it.

Locate `_prepare_covariates()` and change the covariate assembly block:

```python
# BEFORE (current — DX_GROUP missing):
batch = manifest_df['SITE_ID'].values
covariates = pd.DataFrame({
    'AGE_AT_SCAN': manifest_df['AGE_AT_SCAN'].values,
    'SEX': manifest_df['SEX'].values
})

# AFTER:
batch = manifest_df['SITE_ID'].values
covariates = pd.DataFrame({
    'AGE_AT_SCAN': manifest_df['AGE_AT_SCAN'].values,
    'SEX': manifest_df['SEX'].values,
    'DX_GROUP': manifest_df['DX_GROUP'].values   # ← CRITICAL: protect diagnosis
})
```

`neuroHarmonize.harmonization_learn()` accepts continuous and binary covariates in this matrix; `DX_GROUP` ∈ {1, 2} is treated as a binary covariate and will be preserved, not regressed out.

**Why**

`fold_safe_harmonization.py` `_prepare_covariates()` builds the covariate matrix passed to ComBat. `DX_GROUP` is loaded from the manifest (`manifest_df['DX_GROUP']`) but is never added to the `covariates` DataFrame. Without it, ComBat's linear model regresses out all variance correlated with both site and diagnosis. `docs/PROGRESS.md` and in-code comments state "DX_GROUP is a protected covariate", but the implementation contradicts the intent. This single omission is the most probable cause of the CV–Test gap because diagnostic signal is stripped during harmonization on folds it should not be touched by.

**Verify by**

```bash
# After re-running Stage 12:
python -c "
import pandas as pd
df = pd.read_csv('data/metadata/node_attributes_harmonized.csv')
manifest = pd.read_csv('data/metadata/master_manifest.csv')
merged = df.merge(manifest[['subject_id','DX_GROUP']], on='subject_id')
asd = merged[merged.DX_GROUP==2].drop(columns=['subject_id','DX_GROUP'])
ctrl = merged[merged.DX_GROUP==1].drop(columns=['subject_id','DX_GROUP'])
# At least one feature should show a statistically significant group effect
from scipy.stats import mannwhitneyu
p_vals = [mannwhitneyu(asd[c], ctrl[c]).pvalue for c in asd.columns]
min_p = min(p_vals)
print(f'Min group-level p-value post-harmonization: {min_p:.4e}')
assert min_p < 0.05, 'No surviving group difference after harmonization — DX_GROUP still unprotected'
print('PASS: Diagnostic signal preserved')
"
```

---

### Step 2: Fix circular ensemble weighting
**Category**: P0 Critical  
**Requires Retraining**: No  
**Files to modify**: `src/run_evaluation.py`  
**Depends on**: none

**What**

In `run_ensemble_evaluation()`, fold weights are computed as `fold_auc = roc_auc_score(labels, probs)` where `labels` and `probs` are the *test set* predictions. Those weights then combine the fold scores into `ens_probs`, which is re-scored against the same `labels`. Replace the weight computation with the fold-level validation AUC already stored in each checkpoint:

```python
# BEFORE (circular — uses test labels as weights):
fold_auc = roc_auc_score(labels, probs)
weights.append(fold_auc)

# AFTER (use val-AUC from checkpoint, identical to how gnn_model.py handles this):
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
val_auc = ckpt.get('auc', 0.5)      # stored by CheckpointManager during training
weights.append(val_auc)
```

The `CheckpointManager.save()` in `training_utils.py` always stores `'auc'` in the checkpoint dict. The parallel path in `gnn_model.py`'s `evaluate_ensemble()` already does this correctly.

**Why**

`run_evaluation.py` `run_ensemble_evaluation()` calls `_predict_probs()` per fold to get `(labels, probs)` on the held-out test set, then scores `roc_auc_score(labels, probs)` to build ensemble weights, then uses those weights to produce `ens_probs`, then calls `roc_auc_score(labels, ens_probs)` again. The labels appear on both sides of the weighting step, causing the reported AUC to be optimistically biased. The bias grows with fold count because higher-performing folds on the test set get more weight in the ensemble that is then evaluated on the — same — test set.

**Verify by**

```bash
python src/run_evaluation.py 2>&1 | grep -E "Ensemble AUC|fold.*AUC"
# Reported test AUC should be ≤ mean of individual fold test AUCs.
# If ensemble AUC > max(fold AUCs) the circular bias is still present.
```

---

### Step 3: Pass real site_id in evaluation
**Category**: P0 Critical  
**Requires Retraining**: No  
**Files to modify**: `src/run_evaluation.py`  
**Depends on**: none

**What**

`_predict_probs()` always calls:

```python
out = model(data.x, data.edge_index, data.edge_attr, data.batch, site_id=None, ...)
```

The model's `CausalBrainGNN.forward()` zero-pads the 16-dim site embedding when `site_id is None`. During training, real non-zero embeddings were used (via `training_utils.py/_evaluate_model()` which calls `getattr(data, 'site_id', None)`), so the model learned to rely on site context. Setting it to `None` at evaluation replaces 16 learned dimensions with zeros, systematically shifting classifier logits in a site-blind direction.

Replace the hardcoded `None` with the actual site tensor:

```python
# BEFORE:
out = model(data.x, data.edge_index, data.edge_attr, data.batch,
            site_id=None, age=..., sex=..., fiq=...)

# AFTER:
out = model(data.x, data.edge_index, data.edge_attr, data.batch,
            site_id=getattr(data, 'site_id', None),
            age=getattr(data, 'age', None),
            sex=getattr(data, 'sex', None),
            fiq=getattr(data, 'fiq', None))
```

**Why**

`training_utils.py _evaluate_model()` explicitly uses `getattr(data, 'site_id', None)` — when the `Data` object has the attribute, it is passed. `_predict_probs()` in `run_evaluation.py` was written without this guard, so the site embedding is permanently zeroed during test evaluation even though `ABIDECausalDataset.__getitem__()` populates `data.site_id`. The distribution mismatch is a silent evaluation error affecting all 152 test subjects.

**Verify by**

```bash
python -c "
from src.features.graph_factory import ABIDECausalDataset
ds = ABIDECausalDataset(split='test')
sample = ds.get(0)
assert hasattr(sample, 'site_id') and sample.site_id is not None, 'site_id not in Data object'
print(f'site_id present: {sample.site_id}')
print('PASS: site_id will be forwarded to model')
"
```

---

## P1 — Pipeline Integrity

---

### Step 4: Apply Bonferroni correction to multi-lag Granger tests
**Category**: P1 Pipeline  
**Requires Retraining**: Partial — re-run Stage 14 (graph construction) and Stage 16  
**Files to modify**: `src/features/causal_inference.py`  
**Depends on**: none (independent of Steps 1–3; batch with Step 6 before Stage 14 rerun)

**What**

`compute_granger_causality()` runs 5 lag tests per directed pair and takes `min(p_values)` as the representative p-value. Testing 5 hypotheses and taking the minimum inflates the Type I error (false edge rate) from α=0.05 to 1 − (1−0.05)^5 ≈ 22.6%.

Apply Bonferroni correction by multiplying the minimum p-value by the number of lags tested:

```python
# BEFORE (in compute_granger_causality, inner loop):
min_p = min(p_values)
weight = -np.log10(min_p + 1e-10)

# AFTER:
min_p = min(p_values)
bonferroni_p = min(min_p * len(p_values), 1.0)   # Bonferroni: p * n_tests, capped at 1.0
weight = -np.log10(bonferroni_p + 1e-10)
```

Also apply the same correction in `compute_granger_causality_gpu()` which has identical logic.

**Why**

`causal_inference.py` `compute_granger_causality()` calls `grangercausalitytests(data, maxlag=5)` per directed edge, collects p-values for lags 1–5, then assigns `min(p_values)` as the edge p-value. This is the classical "pick the best test among many trials" multiple comparison problem. With 5 lags and α=0.05, approximately 22.6% of null edges pass the threshold, producing a causal graph with ~5× too many spurious directed connections. The Bonferroni correction is conservative but appropriate for a fixed `maxlag=5`; it aligns false-positive rate with the per-edge claim.

**Verify by**

```bash
python -c "
from src.features.causal_inference import compute_granger_causality
import numpy as np
np.random.seed(42)
# Pure white noise — should produce near-zero Granger weights
ts = np.random.randn(200, 12)
gc = compute_granger_causality(ts, max_lag=5)
# Off-diagonal weights should be low (no true causality)
off_diag = gc[gc != 0]
print(f'Mean edge weight on white noise: {off_diag.mean():.4f} (expect < 1.5 after Bonferroni)')
assert off_diag.mean() < 1.5, 'Bonferroni not applied — spurious edges remain high'
print('PASS')
"
```

---

### Step 5: Restrict outlier clipping to per-fold train statistics
**Category**: P1 Pipeline  
**Requires Retraining**: Partial — re-run Stage 12 (harmonization) and Stages 14–16  
**Files to modify**: `src/features/fold_safe_harmonization.py`  
**Depends on**: Step 1 (batch with Step 1 into a single Stage 12 rerun)

**What**

`repair_features()` computes `outlier_threshold = df_features.std() * 5` over `df_features`, which is the **full dataset** DataFrame passed in before any fold split. This makes the clipping boundary a function of test and val data, leaking their statistics into feature normalization.

The call site in `harmonize_cv_safe_fold()` must be restructured to compute the threshold only from the training fold:

```python
# BEFORE (in harmonize_cv_safe_fold or equivalent):
df_features = repair_features(df_features, manifest_df)   # full dataset passed

# AFTER: pass the train portion separately so repair_features clips only to train stats
train_mask = manifest_df['split'] == 'train'    # or use fold-level train indices
df_features_train = repair_features(
    df_features[train_mask], manifest_df[train_mask]
)
# Apply the *same train-derived* thresholds to val/test:
# (Requires refactoring repair_features to return thresholds separately)
train_thresholds = df_features_train.std() * 5
df_features = df_features.clip(
    lower=-train_thresholds, upper=train_thresholds, axis=1
)
```

**Why**

`fold_safe_harmonization.py` `repair_features()` computes `df_features.std()` on whatever DataFrame is passed in. In the current call graph this is always the full pre-split feature matrix, so the 5σ threshold is derived partly from val and test subjects. Any outlier in a test subject that would otherwise exceed the threshold does not get clipped, and the clipping boundary itself was shaped by test-set variance. This is a form of leakage that inflates the apparent "cleanliness" of harmonized test features.

**Verify by**

Confirm that after the fix, re-running Stage 12 produces harmonized features where the per-fold val and test rows were clipped to bounds derived exclusively from that fold's train rows (log both bounds at INFO level for inspection).

---

### Step 6: Stabilize SVD sign across subjects
**Category**: P1 Pipeline  
**Requires Retraining**: Partial — re-run Stage 14 (graph construction) and Stage 16  
**Files to modify**: `src/features/construct_causal.py`  
**Depends on**: none (batch with Step 4 before Stage 14 rerun)

**What**

`aggregate_to_lobes()` extracts the dominant PCA signal as `dominant_signal = u[:, 0] * s[0]`. The sign of `u[:, 0]` from SVD is indeterminate — for the same lobe, subject A may produce `+u` and subject B may produce `−u`. Because this signal feeds directly into `compute_causality_matrix()`, Granger causal direction between two lobes can be randomly inverted between subjects, making the resulting graphs inconsistent.

Fix by projecting the first PC onto the lobe's mean signal and flipping if antiparallel:

```python
# After: dominant_signal = u[:, 0] * s[0]
# Add sign stabilization:
lobe_mean = roi_data.mean(dim=1)   # (T,)  — always well-defined for ≥1 ROI
sign = torch.sign((dominant_signal * lobe_mean).sum())
if sign == 0:
    sign = torch.tensor(1.0)
dominant_signal = dominant_signal * sign
```

This anchors the PC to the direction of mean activity, preserving the signal's information while making its sign deterministic across subjects.

**Why**

`construct_causal.py` `aggregate_to_lobes()` performs per-lobe SVD via `torch.linalg.svd()`. PyTorch's SVD does not guarantee sign consistency across calls on different data. The first PC for the same region in different subjects can point in opposite directions. When Granger tests `region_i → region_j` using these inconsistently signed series, the resulting -log10(p) weight is correct on average but the encoded causal direction can be spuriously reversed in a fraction of subjects — systematically degrading graph-level classification signal.

**Verify by**

```bash
python -c "
import torch
from src.features.construct_causal import aggregate_to_lobes
from src.core.config import NUM_LOBES
torch.manual_seed(0)
# Two sign-flipped versions of the same time series
ts = torch.randn(200, 170)
l1, _ = aggregate_to_lobes(ts)
l2, _ = aggregate_to_lobes(-ts)   # pure sign flip of input
# After fix, dominant signals should be identical (sign-stabilized)
corr = (l1 * l2).sum(dim=0) / (l1.norm(dim=0) * l2.norm(dim=0) + 1e-9)
print(f'Mean cross-subject PC correlation: {corr.mean():.4f}  (pre-fix expect ~0; post-fix expect ~1)')
assert corr.mean() > 0.9, 'SVD sign still inconsistent'
print('PASS')
"
```

---

### Step 7: Audit conf_std and detection_count as site proxies
**Category**: P1 Pipeline  
**Requires Retraining**: Partial — re-run Stage 9 (spatial extraction) and Stages 14–16 if confirmed  
**Files to modify**: `src/features/extract_spatial.py`, `src/features/fold_safe_harmonization.py`  
**Depends on**: none

**What**

`conf_std` (std-dev of YOLO confidence across 7 slices) and `detection_count` (number of successful YOLO detections per region) are spatial features that correlate with scanner image quality: higher-quality scanners produce more detections and more consistent confidence scores. Because spatial features are **not** passed through the ComBat harmonizer (only temporal features enter `fold_safe_harmonization.py`), site-correlated variance in `conf_std`/`detection_count` is never corrected and persists into the GNN as an unmitigated site signal — short-circuiting the GRL.

**Verify first** whether site effects are present before modifying:

```bash
python -c "
import pandas as pd
from scipy.stats import kruskal
df = pd.read_csv('data/metadata/node_features_3d.csv')
manifest = pd.read_csv('data/metadata/master_manifest.csv')
merged = df.merge(manifest[['subject_id','SITE_ID']], on='subject_id')
conf_cols = [c for c in df.columns if 'conf_std' in c]
count_cols = [c for c in df.columns if 'detection_count' in c]
for col in conf_cols[:3] + count_cols[:3]:
    groups = [g[col].dropna().values for _, g in merged.groupby('SITE_ID') if len(g) > 5]
    stat, p = kruskal(*groups)
    print(f'{col}: Kruskal-Wallis p={p:.4e}')
"
```

If the majority of `conf_std`/`detection_count` columns show p < 0.05 across sites, add them to the harmonization step by extracting them from the spatial CSV and passing through ComBat alongside temporal features using `DX_GROUP` as protected covariate (same pattern as Step 1). If they do not show site effects, no modification is needed.

**Why**

`extract_spatial.py` populates `{lobe}_conf_std` and `{lobe}_detection_count` from raw YOLO outputs without any site correction. `fold_safe_harmonization.py` only harmonizes `node_attributes_temporal.csv` features. The spatial features flow directly from `NODE_FEATURES_3D` into `graph_factory.py` without any batch-effect correction. Any site signal in these columns bypasses both ComBat and GRL, creating a potential shortcut for site-based classification.

**Verify by**

After modification (if triggered): re-run the Kruskal-Wallis test on the harmonized version of `conf_std`/`detection_count`; p-values should rise above 0.05 for most columns. If the feature is replaced with atlas-derived priors instead, confirm fixed values match expected atlas centroids.

---

## P2 — Architecture & Training

---

### Step 8: Align OneCycleLR warmup with early-stopping budget
**Category**: P2 Architecture  
**Requires Retraining**: Yes — re-run Stage 16  
**Files to modify**: `src/core/config.py`, `src/models/training_utils.py`  
**Depends on**: Steps 1, 5, 6 (complete before retraining)

**What**

`train_fold_with_onecycle()` creates the scheduler with `pct_start=0.3`, meaning 30% of `GNN_EPOCHS=100` epochs (= 30 epochs) are spent warming up. `GNN_ONECYCLE_PATIENCE=20` allows early stopping at epoch 21. For folds that converge quickly (observed range: epochs 8–24 per `docs/PROGRESS.md`), EarlyStopping can fire mid-warmup, forcing the scheduler into a state it was never designed for and wasting gradient steps.

Add a config constant and patch the scheduler construction:

```python
# In config.py, add:
GNN_ONECYCLE_WARMUP_FRACTION = 0.15   # 15 epochs warmup < 20 patience

# In training_utils.py, train_fold_with_onecycle():
# BEFORE:
scheduler = OneCycleLR(optimizer, max_lr=GNN_ONECYCLE_MAX_LR,
                       total_steps=total_steps, pct_start=0.3, ...)
# AFTER:
scheduler = OneCycleLR(optimizer, max_lr=GNN_ONECYCLE_MAX_LR,
                       total_steps=total_steps,
                       pct_start=GNN_ONECYCLE_WARMUP_FRACTION,  # 0.15 < patience/epochs
                       ...)
```

**Why**

`training_utils.py` `train_fold_with_onecycle()` hardcodes `pct_start=0.3`. With `GNN_EPOCHS=100`, this creates a 30-epoch linear warmup phase. `EarlyStopping` that fires at epoch 21 (patience=20) terminates training before the warmup even completes. Ganin et al. 2016 and the OneCycleLR literature both emphasize that the warmup phase is critical for LR-sensitive GNN training; premature termination leaves the model in a transitional learning rate regime that neither the LR head nor the convergence state was designed for. `pct_start=0.15` (15 epochs) creates a 5-epoch buffer below patience=20.

**Verify by**

After retraining, inspect fold training curves using `TrainingMonitor.plot_training_curves()`. All folds should show a complete warmup arc (rising then decaying LR) regardless of early stopping. Verify in `training_history_fold{N}.json` that `learning_rate[0] < learning_rate[5] > learning_rate[-1]` for every fold.

---

### Step 9: Implement GRL alpha annealing
**Category**: P2 Architecture  
**Requires Retraining**: Yes — re-run Stage 16  
**Status**: ⚠️ **CONTEXT UPDATED** — GRL was fully disabled (Phase 10.3) because alpha=1.0 collapsed all representations. Rather than annealing to alpha=1.0, the recommendation is now to grid-search low alpha values {0.05, 0.1, 0.2} and evaluate whether light adversarial training (without collapsing the representation) reduces the CV–test gap.

**Files to modify**: `src/core/config.py` (change `GNN_USE_GRL=True`, `GNN_GRL_ALPHA=0.05`)

**Original rationale retained below for reference:**

`GradientReversal` in `causal_gnn.py` applies `grl_alpha=1.0` from epoch 1. Ganin et al. 2016 (the canonical DANN paper) showed that starting with high alpha destabilizes the classification head because domain confusion gradients overwhelm classification gradients before the classifier has learned a reasonable representation.

`CausalBrainGNN` stores `self.grl_alpha` as a mutable attribute. Add a method for annealing and call it from the training loop:

```python
# In causal_gnn.py, add to CausalBrainGNN:
def set_grl_alpha(self, progress: float) -> None:
    """Anneal GRL alpha using the Ganin et al. 2016 schedule.
    progress: float in [0, 1] = current_epoch / total_epochs
    """
    self.grl_alpha = 2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0

# In training_utils.py, train_fold_with_onecycle(), inside the epoch loop:
# BEFORE: (no alpha update)
# AFTER (at top of each epoch):
progress = epoch / total_epochs
if hasattr(model, 'set_grl_alpha'):
    model.set_grl_alpha(progress)
```

This ramps alpha from ~0 at epoch 0 to ~1.0 at epoch 50, matching the Ganin schedule.

**Why**

`config.py` `GNN_GRL_ALPHA = 1.0` is constant. `causal_gnn.py` stores it as `self.grl_alpha` but nothing ever updates it during training. Site adversarial training with constant alpha=1.0 from epoch 1 typically causes the classification head to collapse early (the adversarial gradient is as large as the classification gradient before any representation is formed). The Ganin annealing schedule is a proven fix: alpha starts near zero (classification-dominated) and grows toward 1.0 as the representation matures.

**Verify by**

Inspect `training_history_fold{N}.json`: domain-adaptation loss should start at a high value and decrease toward 0 as alpha grows. Classification AUC should not crash in epochs 1–5 (which was a failure mode with constant alpha=1.0).

---

### Step 10: Fix attribution target class consistency
**Category**: P2 Architecture  
**Requires Retraining**: No  
**Files to modify**: `src/analysis/feature_attribution.py`  
**Depends on**: none

**What**

`FeatureAttributionAnalyzer.compute_attributions()` sets `target = pred_class` when `target_class=None`, meaning attribution is computed for the predicted class regardless of the true label. For correctly classified ASD subjects, `target=1` is appropriate. For misclassified ASD subjects (predicted as Control, target=0), the attributions explain why the model thinks "Control" rather than what features signal ASD — inverting the clinical interpretation.

Change the default to always target the ASD class (index 1), consistent with `GradCAMGraphExplainer` which hardcodes `target_class=1`:

```python
# In compute_attributions(), change default:
def compute_attributions(
    self,
    n_steps: int = 50,
    target_class: int = 1,          # ← was None; now always target ASD class
    debug: bool = False,
    use_integrated_gradients: bool = False,
) -> np.ndarray:
```

Also update `_get_wrapper_for_batch()` to forward `site_id` (same as Step 3 fix) since it currently passes `site_id=None`.

**Why**

`feature_attribution.py` `compute_attributions()` defaults `target_class=None`, which maps to `target = pred_class` per the code. This creates inconsistent attribution targets across the test set: correctly classified subjects get attributions for the right class; misclassified subjects get attributions for the wrong class. When results are aggregated by diagnosis group, the resulting mean attribution is a mixture of "what signals ASD" and "what signals Control", which is uninterpretable. `node_importance.py`'s `GradCAMGraphExplainer` hardcodes `target_class=1` (ASD) — this step aligns `feature_attribution.py` with that convention.

**Verify by**

```bash
python -c "
import inspect
from src.analysis.feature_attribution import FeatureAttributionAnalyzer
sig = inspect.signature(FeatureAttributionAnalyzer.compute_attributions)
default = sig.parameters['target_class'].default
assert default == 1, f'target_class default is {default}, expected 1'
print('PASS: target_class=1 is the default')
"
```

---

## P3 — Evaluation & Reporting

---

### Step 11: Stratify bootstrap resampling
**Category**: P3 Evaluation  
**Requires Retraining**: No  
**Files to modify**: `src/run_evaluation.py`  
**Depends on**: Steps 2, 3

**What**

`_bootstrap_ci()` uses `np.random.choice(len(y_test), size=len(y_test), replace=True)` — uniform resampling. When ASD (1) subjects total 50% and Control (0) total 50%, random resampling can drop one class entirely in some draws, causing `roc_auc_score` to raise a `ValueError` (only one class present). The current code has no guard for this.

Replace with stratified bootstrap:

```python
def _bootstrap_ci(y_true, y_pred, n_bootstrap=2000, ci=95, seed=42):
    rng = np.random.default_rng(seed)
    asd_idx = np.where(y_true == 1)[0]
    ctrl_idx = np.where(y_true == 0)[0]
    aucs = []
    for _ in range(n_bootstrap):
        # Resample each class separately, then combine
        asd_sample = rng.choice(asd_idx, size=len(asd_idx), replace=True)
        ctrl_sample = rng.choice(ctrl_idx, size=len(ctrl_idx), replace=True)
        idx = np.concatenate([asd_sample, ctrl_sample])
        try:
            aucs.append(roc_auc_score(y_true[idx], y_pred[idx]))
        except ValueError:
            continue     # degenerate sample — skip
    lo = np.percentile(aucs, (100 - ci) / 2)
    hi = np.percentile(aucs, ci + (100 - ci) / 2)
    return float(np.mean(aucs)), float(lo), float(hi)
```

**Why**

The current bootstrap implementation in `run_evaluation.py` `_bootstrap_ci()` does not preserve the class ratio in resampled draws. With 152 test subjects and roughly balanced classes (~76 ASD, ~76 Control), a random draw has a non-trivial probability of containing only one class, causing `roc_auc_score` to fail. The function can silently skip or crash these samples, biasing or truncating the CI. The ABIDE test set is small enough that even a few missing bootstrap samples significantly widens the CI and shifts the center.

**Verify by**

```bash
python -c "
import numpy as np
from src.run_evaluation import _bootstrap_ci   # or inline the function
rng = np.random.default_rng(0)
y = np.array([1]*76 + [0]*76)
p = rng.uniform(0, 1, 152)
mean_auc, lo, hi = _bootstrap_ci(y, p, n_bootstrap=500)
assert lo <= mean_auc <= hi, 'CI bounds invalid'
assert hi - lo < 0.2, 'CI suspiciously wide (degenerate samples not handled)'
print(f'Bootstrap CI: {lo:.4f} – {hi:.4f}  (mean: {mean_auc:.4f})  PASS')
"
```

---

### Step 12: Within-site label permutation test
**Category**: P3 Evaluation  
**Requires Retraining**: No  
**Files to modify**: `src/run_evaluation.py`  
**Depends on**: none

**What**

`run_permutation_test()` shuffles `y_test` globally: `y_shuffled = np.random.permutation(y_test)`. Across ABIDE sites, prevalent ASD:Control ratios differ substantially. Global permutation destroys the site–diagnosis correlation, causing the null distribution to include AUC values that could never occur under the real null (because site itself predicts partial class membership). The result is a null AUC distribution centered near 0.5 even when the model exploits site effects — making the p-value appear favorable when the model is site-fitting, not disease-fitting.

Replace global shuffle with within-site permutation:

```python
def run_permutation_test(y_true, y_pred, site_ids, n_permutations=1000, seed=42):
    rng = np.random.default_rng(seed)
    observed_auc = roc_auc_score(y_true, y_pred)
    null_aucs = []
    for _ in range(n_permutations):
        y_perm = y_true.copy()
        for site in np.unique(site_ids):
            mask = site_ids == site
            y_perm[mask] = rng.permutation(y_true[mask])   # shuffle within site only
        try:
            null_aucs.append(roc_auc_score(y_perm, y_pred))
        except ValueError:
            continue
    p_value = (np.array(null_aucs) >= observed_auc).mean()
    return observed_auc, p_value, null_aucs
```

**Why**

`run_evaluation.py` `run_permutation_test()` constructs the null distribution via `np.random.permutation(y_test)`. If sites S1 and S2 have 80%/20% ASD split respectively, global shuffling can reassign predominantly ASD labels to S2 subjects — a null configuration that the real null (random labeling within-site) never produces. The within-site permutation preserves the marginal class distribution per site while breaking the subject-level correlate of diagnosis, giving a properly calibrated null.

**Verify by**

Compare the null AUC distribution center under both methods. Within-site null center should be ≥ global null center (site-matched null is harder to beat). A p-value decrease indicates the model was being credited for site prediction, not disease classification.

---

### Step 13: Confirm SVM/RF/MLP baselines use same feature representation
**Category**: P3 Evaluation  
**Requires Retraining**: No  
**Files to modify**: `src/run_evaluation.py`  
**Depends on**: none

**What**

The baseline models (SVM, RF, MLP) in `_run_baselines()` should be trained on the same 12×28 flattened node features as the GNN. Verify that `_get_features_and_labels()` returns the assembled `(12, 28)` feature matrix flattened to `(336,)` from `graph_factory` data objects, not from a different feature representation.

```bash
# Verify:
python -c "
from src.features.graph_factory import ABIDECausalDataset
from torch_geometric.loader import DataLoader
ds = ABIDECausalDataset(split='test')
loader = DataLoader([ds[i] for i in range(5)], batch_size=5)
for batch in loader:
    print(f'batch.x shape: {batch.x.shape}')   # should be (5*12, 28)
    flattened = batch.x.view(5, -1)
    print(f'Flattened: {flattened.shape}')      # should be (5, 336)
    break
"
```

If baselines use a different tabular CSV as input (rather than the assembled graph node features), they are testing a different data representation and the comparison is invalid. If confirmed mismatched, align the baseline input to use `graph_factory`-assembled features.

**Why**

**Verify: open `src/run_evaluation.py` `_run_baselines()` and check whether it loads features from `NODE_ATTRIBUTES_HARMONIZED` (harmonized temporal only, without spatial and internal features) or from the assembled `Data.x` (full 28-dim).**
If baselines use only the 240-column harmonized CSV (12 lobes × 20 temporal features) while the GNN uses the full 336-column vector (12 × 28 features), then the comparison is invalid because the GNN has 96 more features.

**Verify by**

Either: (a) features are confirmed to come from the same 12×28 assembled Data.x vector for both GNN and baselines, or (b) baselines are updated to use flat `data.x` from the DataLoader.

---

## P4 — Code Quality & Reproducibility

---

### Step 14: Complete global seeding and per-fold seed reset
**Category**: P4 Quality  
**Requires Retraining**: Yes — re-run Stage 16  
**Files to modify**: `src/models/gnn_model.py`  
**Depends on**: Steps 8, 9 (batch into single Stage 16 rerun)

**What**

`gnn_model.py` calls `torch.manual_seed(42)` but never calls `random.seed(42)`, `np.random.seed(42)`, or sets `torch.backends.cudnn.deterministic = True`. `StratifiedKFold.split()` internally uses Python's `random` module; `np.random` is used in Focal Loss sampling and sklearn's stratification. Without these, fold ordering and training batches are non-deterministic across machines.

Also, the random state is set once at script start but not reset between folds, so fold 2's initialization depends on the exact number of random draws consumed by fold 1 — a hidden ordering dependency.

```python
# In gnn_model.py run_training(), before the fold loop:
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Inside the for-fold loop, at the start of each fold:
fold_seed = 42 + fold_id
random.seed(fold_seed)
np.random.seed(fold_seed)
torch.manual_seed(fold_seed)
```

**Why**

`gnn_model.py` `run_training()` only calls `torch.manual_seed(42)`. `random` (used by `StratifiedKFold`) and `numpy.random` (used by dropout masks, augmentation, and sklearn stratification) are never seeded. Per-fold seed reset is absent: fold N's weight initialization uses a seed that depends on how many random draws fold N-1 consumed. This makes the per-fold AUC variance partly a function of fold order rather than data content. `cudnn.deterministic=False` (default) introduces GPU non-determinism into every backward pass. Together these produce fold-to-fold variance that cannot be attributed to data differences alone.

**Verify by**

Run Stage 16 twice with the same data. The per-fold AUC sequence `[f0, f1, f2, f3, f4]` must be identical across both runs down to 4 decimal places.

---

### Step 15: Add tests for fold_safe_harmonization.py
**Category**: P4 Quality  
**Requires Retraining**: No  
**Files to modify**: `tests/unit/test_harmonization.py` (new file)  
**Depends on**: Step 1 (test must verify DX_GROUP is in covariates)

**What**

`fold_safe_harmonization.py` has zero test coverage despite being the highest leakage-risk module. Create `tests/unit/test_harmonization.py` with at minimum:

```python
class TestFoldSafeHarmonization:

    def test_dx_group_in_covariates(self):
        """_prepare_covariates must include DX_GROUP in the returned DataFrame."""
        from src.features.fold_safe_harmonization import _prepare_covariates
        # Build minimal manifest
        manifest = pd.DataFrame({
            'subject_id': ['s1', 's2'],
            'SITE_ID': ['A', 'B'],
            'AGE_AT_SCAN': [25.0, 30.0],
            'SEX': [1, 2],
            'DX_GROUP': [1, 2]
        })
        batch, covars = _prepare_covariates(manifest)
        assert 'DX_GROUP' in covars.columns, \
            "DX_GROUP must be a protected covariate in ComBat call"

    def test_outlier_clip_uses_train_stats_only(self):
        """repair_features must not use val/test statistics for clip bounds."""
        # Verify clip threshold is function of train rows only (inject outlier in test row
        # and confirm its presence does NOT shift the clip boundary)
        ...

    def test_no_information_leak_across_folds(self):
        """Val features after harmonization must not be identical to train features."""
        # Fit on train, transform val — val output should differ from train output
        ...
```

**Why**

The entire harmonization module — the code most likely to introduce silent leakage — has no automated tests. Step 1 added the DX_GROUP covariate fix; this step ensures a regression test catches the omission if DX_GROUP is accidentally removed in a future refactor. The minimum contract to test: `_prepare_covariates()` returns a DataFrame with `DX_GROUP` column; `repair_features()` operates only on the passed-in subset; fold-separate harmonization does not produce identical outputs for train and val.

**Verify by**

```bash
pytest tests/unit/test_harmonization.py -v
# All 3 new tests must pass; the DX_GROUP test in particular guards Step 1
```

---

### Step 16: Move frequency band boundaries to config.py
**Category**: P4 Quality  
**Requires Retraining**: Partial — re-run Stage 11 if band boundaries change; no retraining needed if boundaries are unchanged (just moved)  
**Files to modify**: `src/core/config.py`, `src/features/extract_temporal.py`  
**Depends on**: none

**What**

`extract_temporal.py` `extract_band_power()` defines:

```python
DEFAULT_BANDS = {
    'delta': (0.01, 0.027),
    'theta': (0.027, 0.073),
    'alpha': (0.073, 0.15),
    'beta':  (0.15, 0.20),
    'gamma': (0.20, 0.25),
}
```

These are hardcoded inside the function. Move them to `config.py`:

```python
# In config.py:
FREQ_BANDS: dict = {
    'delta': (0.01, 0.027),
    'theta': (0.027, 0.073),
    'alpha': (0.073, 0.15),
    'beta':  (0.15, 0.20),
    'gamma': (0.20, 0.25),
}
UNRELIABLE_FREQ_BANDS_AT_NYQUIST: list = ['gamma']  # already in config — align here

# In extract_temporal.py:
from src.core.config import FREQ_BANDS
def extract_band_power(ts, fs, bands=None):
    bands = bands or FREQ_BANDS
    ...
```

**Why**

`extract_temporal.py` is the only place that defines the frequency band boundaries. If a researcher wants to audit the Phase 10.2 gamma-band aliasing issue, they must know to look inside `extract_band_power()` rather than `config.py`. The copilot-instructions.md explicitly states "Config-driven everything — `config.py` is the single source of truth. Never write hyperparameters directly in code." The band boundaries are hyperparameters directly affecting all 12 frequency features in every node's feature vector.

**Verify by**

```bash
python -m src.validation.dev_audit  # CodeAuditor checks for hardcoded constants
# Should report zero warnings about hardcoded frequency values
pytest tests/unit/test_features.py -v  # Ensure test_custom_bands_respected still passes
```

---

### Step 17: Add DX_GROUP label-encoding test to test_dataset.py
**Category**: P4 Quality  
**Requires Retraining**: No  
**Files to modify**: `tests/integration/test_dataset.py`  
**Depends on**: none

**What**

`test_dataset.py` builds a mock manifest with `DX_GROUP=2` (ASD in ABIDE encoding) but no test asserts that `sample.y == 1` (GNN encoding). If `graph_factory.py` label encoding changes (`DX_GROUP == 2 → y = 1` vs some other convention), all existing tests would still pass.

Add to `TestABIDECausalDataset`:

```python
def test_label_encoding_asd(self, dataset):
    """DX_GROUP=2 (ABIDE ASD) must map to y=1 (GNN ASD class)."""
    sample = dataset.get(0)
    assert sample is not None
    # Mock manifest uses DX_GROUP=2 (ASD)
    assert sample.y.item() == 1, (
        f"DX_GROUP=2 should encode as y=1 (ASD), got y={sample.y.item()}"
    )

def test_label_encoding_control(self, dataset_control):
    """DX_GROUP=1 (ABIDE Control) must map to y=0 (GNN Control class)."""
    # Requires a second mock_data_dir fixture with DX_GROUP=1
    ...
```

**Why**

`graph_factory.py` uses `y = torch.tensor(1 if row['DX_GROUP'] == 2 else 0, dtype=torch.long)`. This encodes ABIDE's DX_GROUP=2 (ASD) as y=1 and DX_GROUP=1 (Control) as y=0. The integration test `test_dataset.py` never asserts this mapping — it only checks that shape, edge count, and dtype are valid. If the conditional were accidentally inverted, the GNN would train on inverted labels. The test for shape/type catches careless structural regressions but not semantic inversion.

**Verify by**

```bash
pytest tests/integration/test_dataset.py::TestABIDECausalDataset::test_label_encoding_asd -v
# PASS confirming DX_GROUP=2 → y=1
```

---

### Step 18: Resolve PATIENCE vs GNN_EARLY_STOPPING_PATIENCE duplication
**Category**: P4 Quality  
**Requires Retraining**: No (config rename only; change training_utils.py import if needed)  
**Files to modify**: `src/core/config.py`, `src/models/training_utils.py`  
**Depends on**: none

**What**

`config.py` defines two separate early-stopping patience values (identified in `changes/changes.md` issue #11):

```python
PATIENCE = 25                          # class imbalance / general section
GNN_EARLY_STOPPING_PATIENCE = 20       # GNN model section
```

`training_utils.py` uses `GNN_ONECYCLE_PATIENCE` (which is also 20) but this may reference either constant inconsistently in different callsites.

Audit which constant each file actually imports, remove the duplicate that is not used, and add an inline comment to the surviving constant: `GNN_ONECYCLE_PATIENCE = 20  # must be > GNN_EPOCHS * GNN_ONECYCLE_WARMUP_FRACTION (see Step 8)`.

Also resolve the naming inconsistency between `GNN_NUM_LAYERS` (referenced in docs) and the actual constant `GNN_NUM_GNN_LAYERS` (issue #9 in `changes/changes.md`): standardize to `GNN_NUM_LAYERS` everywhere.

**Why**

`changes/changes.md` reported this duplication in the February 2026 audit and it remains unresolved. Having two patience constants with different values (25 vs 20) means any downstream code that imports the wrong one behaves differently from what training_utils.py implements. The `GNN_NUM_LAYERS` vs `GNN_NUM_GNN_LAYERS` naming inconsistency means any module that imports `GNN_NUM_LAYERS` would get an `ImportError` at import time, which could silently fall back to a default in experiment scripts.

**Verify by**

```bash
pytest tests/unit/test_config.py -v
grep -rn "GNN_NUM_LAYERS\|GNN_NUM_GNN_LAYERS\|GNN_ONECYCLE_PATIENCE\|GNN_EARLY_STOPPING_PATIENCE\|^PATIENCE" src/ tests/
# Only one name should appear for each concept
```

---

## Retrain Schedule

### Phase A — Data Rebuild (Steps 1, 5, 6, 7; then Stage 12; then Steps 4 and 6; then Stage 14)

Implement Steps 1 and 5 together, then trigger a single Stage 12 rerun:

```bash
python src/run_pipeline.py --auto --skip-download --skip-split
# Runs: Stage 12 (fold-safe harmonization with DX_GROUP fix + fold-safe clipping)
```

Implement Steps 4, 6, and resolve Step 7 (if spatial re-run is needed), then trigger Stage 14:

```bash
python src/run_pipeline.py --auto --skip-download --skip-split
# Runs: Stage 14 (graph construction with Bonferroni + SVD sign fix)
```

**Expected impact**: Removal of diagnostic signal stripping (Step 1) is the single highest-impact change. CV AUC should rise meaningfully; a rough expectation given the magnitude of the harmonization bug is CV AUC moving toward 0.65–0.70. Bonferroni and sign stabilization (Steps 4, 6) reduce spurious graph edges, which should make fold-to-fold AUC more consistent (lower ± value).

### Phase B — Training Fixes (Steps 8, 9, 14; single Stage 16 rerun)

After Phase A data is rebuilt, apply Steps 8, 9, and 14 — all training loop modifications — and retrain:

```bash
python -m src.models.gnn_model
# Runs: Stage 16 (GNN 5-fold CV with warmup fix, GRL annealing, full seeding)
```

**Expected impact**: GRL annealing should reduce the CV–Test gap (weaker domain confusion in early epochs → less site overfitting). Warmup alignment prevents premature early stopping. With full seeding, fold variance becomes attributable to data, not randomness. Target: CV AUC ≥ 0.65, CV–Test gap ≤ 0.05.

### Phase C — Evaluation Fixes (Steps 2, 3, 10, 11, 12; no retraining)

Apply evaluation fixes to `run_evaluation.py` and re-run evaluation against the Phase A+B checkpoints:

```bash
python src/run_evaluation.py
```

**Expected impact**: Step 2 removes circular weighting inflation — reported ensemble test AUC will likely decrease from 0.5398 toward the honest value (may appear as a "regression" but reflects reality). Step 3 adds real site embeddings — this typically raises test AUC by 1–3pp. Net effect is a lower but more credible test AUC number. The p-value from Step 12's within-site permutation test will be more stringent (larger) than the current global permutation.

### Phase D — Quality (Steps 13, 15, 16, 17, 18; no retraining)

These are code quality and test coverage improvements. Apply at any point after Phase A. They do not affect model performance but are required before publishing results or submitting Phase 10 PRs.

```bash
pytest tests/ -v --cov=src
# Target: coverage of fold_safe_harmonization.py reaches ≥ 70%
```

### Metric checkpoints

| After | Expected CV AUC | Expected Test AUC | Note |
|---|---|---|---|
| Baseline | 0.6194 ± 0.0641 | 0.5398 | Circular weighting inflates test |
| Phase A (achieved) | 0.6309 ± 0.0316 | — | Dead lobe NaN fix (Phase 10, March 8) |
| Phase B (achieved) | **0.6721 ± 0.0340** | **0.5798 [0.4935, 0.6727]** | GRL disabled (Phase 10.3, March 9); p=0.041 |
| Phase C (next) | stable | 0.58–0.62 | GRL alpha grid {0.05, 0.1} + per-site analysis |
| Phase D | unchanged | unchanged | Test coverage improvement only |

Report fold-wise AUC `[f0, f1, f2, f3, f4]` and 95% bootstrap CI at each checkpoint to detect regressions.
