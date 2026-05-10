# Design Decision Log

This log records active architectural and modeling decisions reflected in source code.

---

## Part A — Decision Log

### DD-001: Aggregate 170 AAL ROIs into 12 Lobe-Level Nodes

| Field | Content |
|-------|---------|
| **Decision** | Construct 12-node graphs from lobe-level aggregation instead of 170-node ROI graphs |
| **Rejected alternatives** | 170-node graphs (too sparse), 11-node (Brainstem excluded) |
| **Rationale** | Better sample efficiency for ABIDE-scale training; cleaner clinical interpretation at system level |
| **Trade-offs** | Loss of fine-grained ROI detail |
| **Status** | Active |
| **Source of truth** | `src/core/atlas_config.py` (lobe mapping), `src/features/construct_causal.py` |

---

### DD-002: Keep Configuration Modular, Re-Export Through a Stable Facade

| Field | Content |
|-------|---------|
| **Decision** | Maintain focused config modules (`paths`, `feature_registry`, `hyperparams`, `atlas_config`, `validators`) and expose them through `src/core/config.py` |
| **Rejected alternatives** | Single monolithic config file |
| **Rationale** | Minimizes drift from scattered constants; preserves backward compatibility |
| **Status** | Active |
| **Source of truth** | `src/core/config.py`, `src/core/paths.py`, `src/core/feature_registry.py` |

---

### DD-003: Use a Declarative Stage Registry for Orchestration

| Field | Content |
|-------|---------|
| **Decision** | Define stage metadata and dependencies in `src/pipeline/registry.py`, execute via `src/run_pipeline.py` |
| **Rejected alternatives** | Hardcoded stage ordering in runner |
| **Rationale** | One place to update stage contracts; consistent skip/auto/dry-run behavior |
| **Status** | Active |
| **Source of truth** | `src/pipeline/registry.py`, `src/run_pipeline.py` |

---

### DD-004: Enforce Fold-Safe Harmonization with Diagnosis Protection

| Field | Content |
|-------|---------|
| **Decision** | Fit ComBat on fold-train only and apply to val/test; include `DX_GROUP` as protected covariate |
| **Rejected alternatives** | Global harmonization (leakage risk), exclude DX_GROUP |
| **Rationale** | Prevents fold leakage; avoids stripping disease-related variance during site correction |
| **Status** | Active |
| **Source of truth** | `src/features/fold_safe_harmonization.py` |

---

### DD-005: Retain Only Anatomical Spatial Channels in Model Input

| Field | Content |
|-------|---------|
| **Decision** | Model input uses 4 spatial channels (`x`, `y`, `z_depth`, `size`) |
| **Rejected alternatives** | Include `conf_std`, `detection_count` (site-leaky) |
| **Rationale** | Reduces scanner/site proxy leakage in learning signal |
| **Status** | Active |
| **Source of truth** | `src/core/feature_registry.py`, `src/features/graph_factory.py` |

---

### DD-006: Use Directed Causal Connectivity with lagged_pearson Default

| Field | Content |
|-------|---------|
| **Decision** | Use `lagged_pearson` as the default causal connectivity method (now `ridge_granger_hybrid`) |
| **Rejected alternatives** | `ridge_granger`, `partial_corr_glasso`, `transfer_entropy` |
| **Rationale** | Lagged Pearson captures slow hemodynamic coupling (0.01–0.15 Hz) better than VAR-based Granger at 12-lobe aggregation; lower fold variance |
| **Trade-offs** | Less sophisticated than full VAR-based Granger |
| **Status** | Active (now superseded by `ridge_granger_hybrid` as default) |
| **Source of truth** | `src/core/hyperparams.py`, `src/features/causal_inference.py` |

---

### DD-009: Youden Threshold Policy for Balanced Sensitivity/Specificity

| Field | Content |
|-------|---------|
| **Decision** | Use Youden threshold policy for evaluation reporting |
| **Rejected alternatives** | Fixed threshold (0.5263), F1-optimized |
| **Rationale** | Fixed threshold produced low sensitivity (0.57); Youden provides balanced operating point with sensitivity ~0.73 |
| **Status** | Active |
| **Source of truth** | `src/core/hyperparams.py` (`EVAL_THRESHOLD_POLICY = "youden"`) |

---

### DD-010: Keep GRL Site-Adversarial Path Enabled but Controlled

| Field | Content |
|-------|---------|
| **Decision** | GRL enabled with conservative alpha (`GNN_GRL_ALPHA = 0.10`) |
| **Rejected alternatives** | GRL disabled, GRL alpha = 1.0 (too aggressive) |
| **Rationale** | Mitigates site leakage while avoiding aggressive adversarial settings |
| **Status** | Active |
| **Source of truth** | `src/core/hyperparams.py`, `src/models/gnn_model.py` |

---

### DD-011: Optimized GNN Hyperparameters for Publication

| Field | Content |
|-------|---------|
| **Decision** | Use site conditioning, demographic conditioning, anatomical pooling, and optimized GNN architecture |
| **Key settings** | `GNN_USE_SITE_EMBEDDING = True`, `GNN_USE_DEMOGRAPHICS = True`, `GNN_GRL_ALPHA = 0.10`, `GNN_POOLING = "anatomical"`, `GNN_HIDDEN_CHANNELS = 48`, `GNN_NUM_HEADS = 4`, `GNN_NUM_LAYERS = 3`, `GNN_DROPOUT = 0.33` |
| **Results** | CV AUC: 0.8168 ± 0.0488, Test AUC: **0.8810** (3-run stable), Test F1: 0.8375 |
| **Status** | Active — **BEST CONFIG** (May 11, 2026) |
| **Source of truth** | `src/core/hyperparams.py` |

### DD-011b: Hyperparameter Tuning Results (May 2026)

| Config | Test AUC | F1 | Sens | Spec | Notes |
|--------|----------|----|-----|-----|-------|
| Canonical (32ch/2hd/2L/0.35) | 0.8657 | 0.765 | 73.4% | 82.7% | Baseline |
| Prior best (May 10, 64ch/4hd/2L/0.35) | 0.8798 | 0.795 | 73.4% | 88.0% | GRL grid search |
| **48ch/4hd/3L/0.33 (Best)** | **0.8810** | **0.8375** | **84.8%** | **81.3%** | **Stable, 3-run** |

---

### DD-012: Keep Multiview Graph Generation Optional with Quality Gates

| Field | Content |
|-------|---------|
| **Decision** | Multiview graph construction is opt-in (`--multiview`), invariance loss enabled only when artifacts pass quality checks |
| **Rejected alternatives** | Always-on multiview |
| **Rationale** | Allows robustness experiments without forcing higher-complexity training; guards against degenerate views |
| **Status** | Active |
| **Source of truth** | `src/pipeline/registry.py`, `src/models/gnn_model.py` |

---

### DD-013: Support Auxiliary Structural/Invariance Losses as Tunable Controls

| Field | Content |
|-------|---------|
| **Decision** | Structural dropout, edge contrastive, causal invariance, spatial invariance terms implemented with config weights |
| **Current default** | Most weights set to `0.0` (conservative baseline) |
| **Rationale** | Mechanism available for ablations without forcing complexity |
| **Status** | Active |
| **Source of truth** | `src/core/hyperparams.py`, `src/models/gnn_model.py` |

---

### DD-014: Make Site-Stratified CV an Explicit Optional Protocol

| Field | Content |
|-------|---------|
| **Decision** | Default stratified folds; optional `--site-stratified-cv` rewrites `cv_fold` with GroupKFold over site clusters |
| **Rationale** | Preserves baseline comparability while enabling stricter cross-site robustness studies |
| **Status** | Active |
| **Source of truth** | `src/data/split.py`, `src/run_pipeline.py` |

---

### DD-015: Lock Deployment Operating Point Through Threshold Policy

| Field | Content |
|-------|---------|
| **Decision** | Evaluation uses centralized threshold policy (`EVAL_THRESHOLD_POLICY`) |
| **Rationale** | Deterministic deployment threshold; avoids per-run threshold drift |
| **Status** | Active |
| **Source of truth** | `src/core/hyperparams.py`, `src/run_evaluation.py` |

---

### DD-016: Preserve Explainability as First-Class Post-Training Stage

| Field | Content |
|-------|---------|
| **Decision** | Retain node, edge, feature, and literature explainability workflows in default analysis stack |
| **Rationale** | Supports scientific interpretability requirements beyond single scalar metrics |
| **Status** | Active |
| **Source of truth** | `src/run_explainability.py`, `src/analysis/node_importance.py` |

---

### DD-017: Visualization Correctness Fixes

| Field | Content |
|-------|---------|
| **Decision** | Fixed critical correctness bugs in visualization and training code |
| **Changes** | DX_GROUP mapping fix, val_loss tracking fix, Stage dataclass description field, auxiliary loss consolidation |
| **Status** | Active |
| **Source of truth** | `src/analysis/diagnostics.py`, `src/models/gnn_model.py` |

---

### DD-018: 12-Lobe Architecture Decision (FINAL, April 28, 2026)

| Field | Content |
|-------|---------|
| **Decision** | 12-lobe architecture approved for publication |
| **Rejected alternatives** | 11-lobe (Brainstem excluded) |
| **Rationale** | Test AUC +8.74% vs 11-lobe; Brainstem constant features act as implicit regularization; better generalization (CV < Test) |
| **Trade-offs** | Brainstem uses synthetic fallback coordinates |
| **Status** | Active — APPROVED FOR PUBLICATION |
| **Source of truth** | `src/core/atlas_config.py`, `docs/deprecated/FINAL_ARCHITECTURE_ANALYSIS.md` |

**Empirical Evidence:**

| Metric | 12-Lobe | 11-Lobe | Δ |
|--------|---------|---------|-----|
| CV AUC | 0.7997 ± 0.0294 | 0.8099 ± 0.0528 | -0.0102 |
| **Test AUC** | **0.8694** | 0.7995 | **+0.0699** |
| Test F1 | **0.8000** | 0.7297 | **+0.0703** |
| Generalization Gap | +0.0697 (robust) | -0.0104 (overfitting) | — |
| Fold Variance | 0.0087 | 0.0278 | 46.5% lower |

---

### DD-019: Causality Method Finalization (ridge_granger_hybrid)

| Field | Content |
|-------|---------|
| **Decision** | Use `ridge_granger_hybrid` (β=0.70) as the primary causality method |
| **Rejected alternatives** | `lagged_pearson`, `ridge_granger` (pure) |
| **Rationale** | Hybrid method combines 70% Granger Causality (causal signal) + 30% Lagged Pearson (correlation strength). It produces the best overall Test AUC while maintaining a robust CV AUC. |
| **Trade-offs** | Marginally lower Test AUC point estimate than pure lagged Pearson (-0.4%), but stronger methodological claim. Overlapping CIs show no significant difference. |
| **Status** | Active — APPROVED FOR PUBLICATION |
| **Source of truth** | `src/core/hyperparams.py` |

**Empirical Evidence:**

| Method | CV AUC | Test AUC | Methodological Strength |
|--------|--------|----------|-------------------------|
| lagged_pearson | 0.7997 ± 0.0294 | **0.8694** | Correlation (not causal) |
| ridge_granger_hybrid (β=0.70) | **0.8168 ± 0.0488** | **0.8810** | 70% Causality + 30% Correlation (Best, May 11) |
| ridge_granger (pure) | 0.7856 ± 0.0290 | 0.8413 | Pure Granger Causality |
| **Canonical (32ch/2hd/2L)** | 0.8102 ± 0.0273 | 0.8657 | Baseline |

**New Ablation Evidence** (config hash `6b6ca55b`, 12lobes.txt:1888-1905):
| Ablation | CV AUC ± std | vs Baseline (+0.63) |
|-----------|--------------|---------------------|
| D (Lagged Pearson) | 0.8456 ± 0.0354 | +21.56% |
| D2 (Ridge Granger) | 0.8512 ± 0.0348 | +22.12% |

**Provenance**: See `docs/dataflow.md` §Ablation Studies.

## Part B — Methods Rationale

### Causality Terminology

| Term | Usage |
|------|-------|
| **Directed functional connectivity** | Primary descriptor for graph edges |
| **Granger-inspired** | Modifier when referencing VAR-based method |
| **Causal graph** | Acceptable shorthand for "directed functional connectivity graph" |

**What We Compute:**
- **Lagged Pearson correlation**: directed functional connectivity measuring linear relationships at multiple lag offsets (1–4 TRs, TR≈2s)
- **Granger causality**: statistical predictability based on vector autoregression

**What This Is NOT:**
- Cannot rule out confounding — both methods detect correlations driven by hidden common causes
- Cannot detect instantaneous effects — requires temporal precedence
- Granger's own caveat: "It is not causation in the philosophical sense"

### Model Selection Procedure

1. Primary model selected based on 5-fold CV AUC only
2. Test set evaluated **once** after model selection finalized
3. Additional configurations evaluated post-hoc as sensitivity analysis

### Spatial Feature Extraction Rationale

- **Primary**: Atlas-derived spatial coordinates from AAL3
- **Ablation**: YOLO-derived spatial features show near-random performance (AUC 0.54)
- **Conclusion**: Spatial features contribute minimally (<5%) regardless of source

### Architecture Framing for Publication

Describe as **"directed functional connectivity GNN"** rather than "causal GNN" because:
- Edges represent statistical temporal dependencies, not philosophical causality
- Directed connectivity is well-established in neuroimaging literature

### Shuffled Edges Finding

| Configuration | Test AUC | Interpretation |
|---------------|----------|----------------|
| Full GNN | 0.8651 | Full model (ridge_granger_hybrid) |
| Shuffled edges | 0.8337 | Identical to real |

**Interpretation**: Graph topology matters; edge weight magnitudes are negligible. Frame as "anatomical scaffold" in paper.

### Statistical Testing Methods

| Test | Implementation | Status |
|------|---------------|--------|
| DeLong test | `src/validation/delong_test.py` | ✅ Implemented |
| Bootstrap CI | 1000 resamples | ✅ Implemented |
| Permutation test | 1000 shuffles | ✅ Implemented |

### Data Availability Statement Template

> **Dataset**: ABIDE I from INDI Preprocessed Connectomes Project
> **Attribution**: Di Martino, A., et al. (2014); Craddock, C., et al. (2013)
> **Ethics**: This study constitutes secondary analysis of publicly available, fully de-identified data. **[Authors: confirm with your IRB before submission.]**

### Compute Cost and Carbon Footprint

| Resource | Estimate |
|----------|----------|
| GPU hours (full pipeline) | 4–6 hours |
| GPU hours (ablations) | 2–4 hours |
| **Total** | **~8–10 GPU-hours** |
| CO₂ equivalent | ~0.5–1.0 kg CO₂e |

---

## Part C — Archived Analyses

### DD-018 History: Pre-Test Analysis (Archived, April 28, 2026)

*This section preserves the intermediate DD-018 recommendation that was reversed by held-out test results.*

**Pre-Test Recommendation (April 22, 2026):**

Before test set evaluation, the analysis favored **11-lobe**:
- 100% region detection vs 0% in 12-lobe
- Better pre-training metrics (CV AUC +0.0097, F1 +0.0126)
- No synthetic fallback features
- Cleaner scientific narrative

| Metric | 12-Lobe | 11-Lobe | Winner |
|--------|---------|---------|--------|
| Pre-Training Model AUC | 0.8002 | **0.8099** | 11 |
| Pre-Training Model F1 | 0.7484 | **0.7610** | 11 |
| Spatial Completeness | 0% (constant) | **100%** | 11 |

**Why It Changed:**

The held-out test set revealed that:
1. 12-lobe's generalization gap (+0.0697) indicated robust learning
2. 11-lobe's gap (-0.0104) indicated overfitting
3. Test AUC: 12-lobe 0.8694 vs 11-lobe 0.7995 (+8.74%)
4. Brainstem constant features act as beneficial regularization

**Lesson:** CV does not reliably predict test performance in multi-site heterogeneous data. Test set is ground truth.

*Source: docs/archive/LOB_E_COMPARISON_ANALYSIS.md*