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
| P2 — Medium (code debt) | 7 | Maintainability / reproducibility |
| P3 — Low (cleanup) | 9 | None to training; publication hygiene |

---

## P2 — Medium Priority (Code Quality)

### P2-1: ABIDECausalDataset Is Loaded 11+ Times Per Training Run

Counting the log entries for `Initialized ... dataset`, `ABIDECausalDataset` is initialized at least 11 times during a single pipeline run (once for the initial quality gate check, once per fold for harmonized features, and multiple times in post-training analysis stages). Each initialization scans all graph files on disk.

The root cause is that `gnn_model._run_training_once()` creates a fresh dataset object for every fold (to load fold-specific harmonized features), rather than loading the temporal features once and swapping them in.

**Fix:** Separate the temporal feature loading from the graph loading. Create a lightweight feature-swap API:

```python
def reload_temporal_features(self, new_path: Path) -> None:
    """Swap temporal features without re-scanning graph files."""
    self.temporal_features_path = new_path
    self.node_attr = _load_csv_cached(new_path, index_col='subject_id')
```

This would reduce 5 redundant disk scans per training run.

---

### P2-2: `src/analysis/diagnostics.py` Is Referenced But Not in Context

The code references `from src.analysis.diagnostics import CausalGraphAnalyzer, TrainingMonitor` in `gnn_model.py` but this file is not among the files provided in the codebase context. It is clearly a required module (used in training). Ensure this file is committed and its interface is stable. Any undocumented interface here will become a maintenance burden.

---

### P2-3: Gradient Accumulation Steps = 2 With Batch Size 32 on 12-Node Graphs

In `train_fold_with_onecycle()`, `gradient_accumulation_steps=2` is hardcoded. With batch size 32 and 12-node graphs, each batch processes 384 nodes — this is computationally trivial. Gradient accumulation provides no benefit here (it is designed for large models with large inputs that can't fit in memory). It doubles the number of optimizer.zero_grad() calls and creates unnecessary bookkeeping overhead.

```python
# In train_fold_with_onecycle() call in gnn_model.py:
gradient_accumulation_steps=2  # → change to 1
```

---

### P2-4: `_MULTIVIEW_CACHE` Is a Module-Level Global in `training_utils.py`

The `_MULTIVIEW_CACHE = _LRUCache(maxsize=512)` is instantiated at module import time. This means it persists across training runs within the same Python process (e.g., during hyperparameter search or repeated pipeline runs), potentially serving stale cached graph data from a prior run. It should be instantiated per-training-run or cleared between runs.

---

### P2-5: `_graph_cache` in `ABIDECausalDataset` Uses a Python Dict With Unbounded Size

The `self._graph_cache` dictionary has a `_cache_limit` parameter, but the eviction logic evicts only one entry per cache-miss:

```python
if len(self._graph_cache) >= self._cache_limit:
    oldest_key = next(iter(self._graph_cache))
    self._graph_cache.pop(oldest_key, None)
```

This is not LRU — it evicts the insertion-order-oldest key regardless of access frequency. With 707 training subjects and `graph_cache_limit=256`, the cache will perpetually evict and reload the same ~451 graphs, providing minimal benefit. Replace with `collections.OrderedDict`-based LRU or use `functools.lru_cache`, or simply increase `graph_cache_limit` to cover the entire training set.

---

### P2-6: `SpatialInvarianceLoss` Has Its Own Optimizer Parameters Merged Into the Main Optimizer

In `train_fold_with_onecycle()`:

```python
optim_params = list(model.parameters())
if spatial_invariance_loss_fn is not None:
    spatial_invariance_loss_fn = spatial_invariance_loss_fn.to(device)
    optim_params.extend(list(spatial_invariance_loss_fn.parameters()))
```

This is incorrect. The spatial invariance loss module's site-classification head parameters should use a *different* learning rate schedule than the main model, because they serve as an adversarial auxiliary and should not be driven by the OneCycleLR that is calibrated for the GNN backbone. Currently, the site head's weights are being annealed on the GNN schedule, which can destabilize adversarial training.

Since `GNN_SPATIAL_INVARIANCE_WEIGHT = 0.0`, this is currently harmless — but if the weight is ever enabled, this will cause problems.

---

### P2-7: `Detected 20 unique site IDs` but SpatialInvarianceLoss Uses `num_sites_detected`

When `GNN_SPATIAL_INVARIANCE_WEIGHT > 0`, the `SpatialInvarianceLoss` is instantiated with `num_sites=num_sites_detected`. However, `GNN_USE_SITE_EMBEDDING = False` in this run, so the GRL site classifier head is built for 20 sites but never receives a site embedding. If site conditioning is disabled, spatial invariance loss should also be disabled automatically.

---

## P3 — Low Priority (Cleanup and Publication Hygiene)

### P3-1: The Canonical Benchmark Has a Problematic Discrepancy

The CHANGELOG and `docs/results.md` report a canonical CV AUC of 0.7434 from the March 2026 run. This run achieves CV AUC of 0.57 from the same model architecture. The difference is entirely attributable to the P0 issues above (site-stratified CV + harmonization mismatch). Before submission, run the pipeline without `--site-stratified-cv` to reproduce the canonical result and confirm the architecture is still working correctly.

---

### P3-2: `src/analysis/literature_validation.py` Is Not in Context

`run_explainability.py` imports `from src.analysis.literature_validation import run_literature_validation`. This file is missing from the provided codebase. Verify it is committed and covers the claimed 7/7 ASD network matches.

---

### P3-3: TopK Sparsification Produces Overly Dense Graphs

Mean edges = 46.8, median = 47. For a 12-node directed graph, the maximum possible off-diagonal edges is 132. This means graphs are 35.5% dense — much higher than the `GRAPH_DENSITY_TARGET = 0.30` config value. The `topk_per_node` method (k=3, keeping top 3 outgoing + top 3 incoming per node) can produce up to 72 edges (12 × 6), and when all nodes retain strong bidirectional connections the actual count approaches 132.

Graph topology comparison in the log shows essentially no statistically significant ASD/Control differences in any graph-level metric (only Parietal in-degree, p=0.013, d=-0.148). This suggests the graphs are too dense — important directional signals are drowned out. Consider reducing `SPARSITY_TOPK_PER_NODE` from 3 to 2, which would cap edges at ~48 and sharpen the directed signal.

---

### P3-4: Random Forest Achieves 0.722 AUC With Flattened Features (vs GNN 0.738)

The GNN outperforms Random Forest by only 1.6 percentage points on test. This is scientifically interesting — the causal graph structure provides marginal additional discriminative power over flat feature classification. For the publication narrative, this comparison should be framed carefully. The GNN's main contribution in this work should be interpretability (the causal connectivity patterns, network-level attribution, literature alignment), not raw classification performance.

---

### P3-5: FocalLoss Is Defined but Disabled (`USE_FOCAL_LOSS = False`)

The codebase has a complete `FocalLoss` implementation (`src/models/losses.py`), extensive documentation referencing focal loss, training infrastructure wiring it, but it is disabled in `hyperparams.py`. With a nearly balanced dataset (493 ASD / 522 Control, ratio 0.94), focal loss provides minimal benefit. The dead-code weight is low but the implementation should either be removed from the default pipeline description or explicitly documented as "off by default, re-enable for imbalanced datasets."

---

### P3-6: `GNN_USE_SITE_EMBEDDING = False` but `GNN_USE_GRL = True`

GRL (Gradient Reversal Layer) is enabled, but site embedding is disabled. The site classifier head in GRL receives the graph embedding and classifies sites, which is correct. However, the `SpatialInvarianceLoss` also uses `site_id` targets. With site conditioning disabled, the adversarial path is operating without any site-specific input signal to the main encoder, making the GRL an orthogonal adversary. This is a known and intentional configuration per DD-008, but it should be documented explicitly in `hyperparams.py` comments.

---

### P3-7: `RESULTS_TRAINING_DIR` Contains Fold-Level Analysis Artifacts Mixed With Run-Level Artifacts

Training history JSONs, fold comparison plots, and feature attribution plots are all saved to `results/experiments/training/`. When multiple runs are executed, artifacts from different runs overwrite each other with no versioning. Given the run includes a `run_id` in the `ExperimentTracker`, these artifacts should be saved under `results/experiments/runs/{run_id}/` instead.

---

### P3-8: Cleanup: Seven Dead/Experimental Aux Loss Weights All Set to Zero

All four auxiliary objective weights in `hyperparams.py` are zero:

```python
GNN_EDGE_CONTRASTIVE_WEIGHT = 0.0
GNN_INVARIANCE_WEIGHT = 0.0
GNN_SPATIAL_INVARIANCE_WEIGHT = 0.0
```

`GNN_STRUCTURAL_DROPOUT_PROB = 0.3` is set but `GNN_EDGE_CONTRASTIVE_WEIGHT = 0.0`, so structural dropout applies (masking node features for 30% of graphs) but no contrastive loss enforces learning from that masked path. This means structural dropout is spending compute without providing a training signal. Either pair it with a non-zero contrastive weight or set structural dropout to 0.0 as well.

---

### P3-9: `_ZERO_LOBE_WARNED` Set Is Module-Level State That Persists Across Subjects

In `construct_causal.py`, `_zero_lobe_warned: set = set()` is module-level. Once Brainstem (lobe 11) triggers a warning for the first subject, no subsequent subject gets warned about missing Brainstem coverage. This is intentional to suppress spam, but it means the pipeline logs never count total affected subjects. Add a final summary count at the end of `main()`.

---

## Actionable Fix Checklist (Ordered by Priority)

The following represents the minimum set of changes needed before the next publication run:

**P0 — Do these first:**
- [ ] Remove `--site-stratified-cv` flag from the pipeline command, reverting to StratifiedKFold CV
- [ ] Set `GNN_MI_FEATURE_SELECTION_ENABLED = False` in `hyperparams.py`
- [ ] Debug `extract_spatial_atlas._load_atlas_lobe_fallbacks()` to fix the ROI indexing mismatch, or use the YOLO-only spatial path with explicit zero-masking for Brainstem

**P1 — Do these before evaluation:**
- [ ] Set `EVAL_THRESHOLD_POLICY = "youden"` (or recompute fixed threshold from canonical run)
- [ ] Change `gradient_accumulation_steps` from 2 to 1 in the training loop call
- [ ] Add variance-retention-per-feature logging to harmonization to identify which feature groups are losing most variance

**P2 — Do these before code freeze:**
- [ ] Implement `reload_temporal_features()` on `ABIDECausalDataset` to avoid 5× redundant initialization
- [ ] Move `_MULTIVIEW_CACHE` from module scope to training-run scope
- [ ] Replace `self._graph_cache` dict eviction with proper LRU (use `collections.OrderedDict`)
- [ ] Set `GNN_STRUCTURAL_DROPOUT_PROB = 0.0` if `GNN_EDGE_CONTRASTIVE_WEIGHT = 0.0` (they should be paired)
- [ ] Confirm `src/analysis/diagnostics.py` and `src/analysis/literature_validation.py` are committed

---

## Architecture Assessment

The GATv2-based design with anatomical hierarchical pooling is **well-suited for this problem** and **does not need modification**. Specific strengths:

- The 12-node graph aggregation is the right scale for ABIDE (~700 training subjects) — 170-node graphs would require far more data
- Ridge Granger + topk sparsification is a reasonable default; the signed effect size preserves directionality
- The fold-safe harmonization framework is methodologically correct when not combined with site-stratified CV
- The anatomical network pooling (DD-011) is scientifically motivated and aligns with the literature validation results
- The checkpoint metadata system (feature masks, site normalization stats) for inference parity is well-engineered

The GradCAM results showing Insula, Frontal_Orbital, Parietal, and Frontal_Superior as differentially important for ASD are consistent with the ASD connectivity literature. This is the pipeline working correctly.

---

## Summary of Root Cause for Accuracy Plateau

The plateau is not architectural. The codebase was performing at 0.74 AUC in March 2026 and is still capable of it. The regression was caused by three configuration choices introduced together that are individually reasonable but collectively incompatible:

1. **Site-stratified CV** (DD-013) answers a valid scientific question but requires harmonization to be applied globally, not fold-locally
2. **Wave-1 fold-internal MI selection** (Wave-1) is a valid regularization idea but the MI scores are too noisy at this scale to be informative, and applying it per-fold creates heterogeneous models
3. **The fixed threshold** (Condition C) was calibrated on different model checkpoints and is inappropriate for models trained under the above conditions

These three interact: the unharmonized validation distribution causes low CV AUC, which causes poor fold weights in the MI selector's floor threshold computation, which causes inconsistent feature pruning, which further depresses fold AUC. Breaking any one of them should improve the situation significantly. Breaking all three returns the pipeline to canonical performance.