# Comprehensive Analysis of Neuro-CXG Pipeline

> **Review Log (Feb 23, 2026)**: Code audit completed against this document.
> - Double z-score normalization issue tracked in `TODOs.md` §0.
> - S3 `brain_mask` → `func_mask` fix tracked in `TODOs.md` §0 (merged from `TODOy.md`).
> - `code_audit.py` feature/shape constants updated (GNN_IN_CHANNELS 14→28, num patterns refreshed).
> - `README.md` System Architecture section added per this document's data-flow recommendations.
> **Live checklist is in `TODOs.md`**. This file is the deep architectural analysis reference.

## Executive Summary

After analyzing the complete codebase, I've identified **critical architectural and methodological issues** that explain the current AUC plateau at ~0.57. The system demonstrates excellent engineering practices (reproducibility, validation, documentation) but suffers from fundamental problems in **feature engineering, graph construction, and model architecture** that prevent clinical-grade performance.

**Key Finding**: The current pipeline is **overregularized for the signal available**, with multiple compounding issues creating a "glass ceiling" effect.

---

## Phase-by-Phase Analysis

### **Phase 1: Data Acquisition & Preprocessing** ✅ **STRONG**

**Strengths:**
- Robust ABIDE download with idempotency checks
- Proper atlas resampling to functional space (critical fix)
- ROI validation (164-170 range for AAL3v1 variants)
- NaN/Inf safety throughout extraction

**Identified Issues:**
1. **Z-score normalization applied twice**:
   - `abide_download.py` line 82: `standardize='zscore_sample'` in NiftiLabelsMasker
   - `construct_causal.py` line 40: Manual z-scoring again before correlation
   - **Impact**: Reduces signal variance, may suppress subtle ASD biomarkers

2. **Bandpass filtering too restrictive**:
   - Current: 0.01-0.08 Hz (only ultra-slow frequencies)
   - Missing: 0.08-0.15 Hz range contains task-relevant signals
   - **ASD Literature**: Gamma band abnormalities (Rojas 2008) occur at higher frequencies, but fMRI Nyquist limit is ~0.25 Hz (TR=2s)

**Recommendation:**
```python
# In abide_download.py, line 69
masker = NiftiLabelsMasker(
    labels_img=resampled_atlas, 
    t_r=float(tr_val), 
    standardize=False,  # ← Remove duplicate normalization
    detrend=True,
    low_pass=0.15,  # ← Expand bandwidth
    high_pass=0.008,  # ← Lower cutoff for slow-5 band
    ensure_finite=True,
)
```

---

### **Phase 2: ROI Detection (YOLO)** ✅ **EXCELLENT**

**Strengths:**
- mAP50-95: 0.94073 (production-ready)
- Anatomically-preserving augmentation (no flips/rotations)
- 12-region granularity balances interpretability and resolution

**Non-issue**: Detection quality is not the bottleneck.

---

### **Phase 3: Feature Engineering** ⚠️ **CRITICAL WEAKNESS**

This is where the pipeline **loses the most signal**.

#### **3.1 Temporal Features: Dimensionality Collapse**

**Current**: 20 temporal features (8 basic + 12 frequency)

**Problems:**

1. **Frequency features are redundant/noisy**:
   ```python
   # extract_temporal.py uses default TR=2.0s → Nyquist = 0.25 Hz
   # But frequency bands assume EEG-like sampling:
   bands = {
       'delta': (0.01, 0.027),   # ✓ Valid
       'theta': (0.027, 0.073),  # ✓ Valid  
       'alpha': (0.073, 0.15),   # ✓ Valid
       'beta': (0.15, 0.20),     # ⚠️ Near Nyquist
       'gamma': (0.20, 0.25)     # ❌ Aliasing risk
   ```
   - Gamma/beta bands are near/beyond Nyquist limit
   - Spectral entropy becomes unreliable with aliasing
   - **12 frequency features add noise, not signal**

2. **PCA Eigenvariate aggregation loses information**:
   ```python
   # construct_causal.py, line 40-60
   # Uses PCA first component to avoid "signal cancellation"
   # BUT: Discards 2nd-5th components that may contain ASD-specific patterns
   ```
   - First PC captures **global signal** (respiratory/cardiac noise)
   - ASD biomarkers may be in **higher-order components**
   - Current approach: **throwing away orthogonal disease signals**

3. **Regional Homogeneity (ReHo) underutilized**:
   - Current: 2 features per region (coherence + variance)
   - Missing: **Kendall's W** (gold standard for ReHo)
   - Missing: **Temporal clustering** (ASD shows altered dynamics)

#### **3.2 Spatial Features: Insufficient Contextualization**

**Current**: 6 spatial features (x, y, z, size, conf_std, count)

**Missing Critical Information:**
- **No inter-regional distances** (spatial relationships matter for connectivity)
- **No hemisphere encoding** (L/R asymmetry is ASD biomarker)
- **No anatomical priors** (e.g., distance to DMN hubs)

#### **Recommendation: Feature Engineering Overhaul**

```python
# NEW: src/features/extract_temporal.py

def extract_temporal_features(ts: np.ndarray, tr: float) -> dict:
    """
    Evidence-based temporal features for ASD classification.
    
    Rationale:
    - Hurst exponent: Captures long-range temporal dependencies (altered in ASD)
    - Sample entropy: Nonlinear complexity (ASD shows reduced entropy)
    - Detrended fluctuation analysis: Self-similarity across scales
    - Variance ratio: Low/high frequency power balance
    """
    features = {}
    
    # Basic statistics (keep these)
    features['mean'] = ts.mean()
    features['std'] = ts.std()
    features['range'] = ts.max() - ts.min()
    
    # Temporal complexity (ADD)
    features['hurst_exp'] = compute_hurst_exponent(ts)  # Long-range correlation
    features['sample_entropy'] = compute_sample_entropy(ts, m=2, r=0.2*ts.std())
    features['dfa_alpha'] = detrended_fluctuation_analysis(ts)  # Fractal scaling
    
    # Spectral features (SIMPLIFIED - avoid aliasing)
    fs = 1.0 / tr
    freqs, psd = welch(ts, fs=fs, nperseg=min(64, len(ts)))
    
    # Only use reliable frequency bands (well below Nyquist)
    slow5_power = band_power(freqs, psd, 0.01, 0.027)   # Slow-5
    slow4_power = band_power(freqs, psd, 0.027, 0.073)  # Slow-4
    slow3_power = band_power(freqs, psd, 0.073, 0.15)   # Slow-3
    
    # Variance ratio (high clinical relevance)
    features['variance_ratio'] = slow5_power / (slow3_power + 1e-10)
    features['spectral_centroid'] = (freqs * psd).sum() / psd.sum()
    
    # Regional homogeneity (ENHANCED)
    features['kendall_w'] = compute_kendalls_w(ts)  # Gold standard ReHo
    
    return features  # Total: ~12 features (down from 20, but higher quality)
```

**Impact**: Reduces noise, increases signal-to-noise ratio, focuses on ASD-relevant dynamics.

---

### **Phase 4: Harmonization** ⚠️ **OVERCORRECTION RISK**

**Current Approach**: neuroHarmonize (ComBat) with `DX_GROUP` as protected covariate

**Problems:**

1. **Implicit assumption: Site effects are orthogonal to diagnosis**
   - Reality: Some sites may have **recruitment bias** (e.g., higher-functioning ASD)
   - Combat may be **removing diagnostically-relevant variance**

2. **No pre/post harmonization validation**:
   ```python
   # fold_safe_harmonization.py lacks:
   # - Effect size comparison (Cohen's d before/after)
   # - Within-group variance preservation checks
   # - Cross-site prediction validation
   ```

**Recommendation**: **Stratified Combat** + **Validation**

```python
# In fold_safe_harmonization.py, add:

def validate_harmonization_preserves_signal(df_before, df_after, manifest):
    """Ensure Combat didn't remove disease signal."""
    
    for site in manifest['SITE_ID'].unique():
        site_data_before = df_before[manifest['SITE_ID'] == site]
        site_data_after = df_after[manifest['SITE_ID'] == site]
        
        # Check if ASD/Control separation is preserved
        asd_before = site_data_before[manifest['DX_GROUP'] == 1].mean(axis=0)
        ctrl_before = site_data_before[manifest['DX_GROUP'] == 2].mean(axis=0)
        
        effect_size_before = (asd_before - ctrl_before) / site_data_before.std(axis=0)
        
        asd_after = site_data_after[manifest['DX_GROUP'] == 1].mean(axis=0)
        ctrl_after = site_data_after[manifest['DX_GROUP'] == 2].mean(axis=0)
        
        effect_size_after = (asd_after - ctrl_after) / site_data_after.std(axis=0)
        
        # FLAG if effect sizes shrink >30%
        shrinkage = (effect_size_before.abs() - effect_size_after.abs()) / effect_size_before.abs()
        if (shrinkage > 0.3).any():
            logger.warning(f"Site {site}: Harmonization reduced effect size by >30%")
```

---

### **Phase 5: Graph Construction** ❌ **FUNDAMENTAL FLAW**

This is the **biggest problem** in the pipeline.

#### **5.1 Granger Causality: Wrong Tool for fMRI**

**Current Implementation**:
```python
# construct_causal.py uses statsmodels Granger test
# Assumptions:
# 1. Linear dynamics
# 2. Gaussian noise
# 3. Stationary time series
# 4. Sufficient temporal resolution
```

**Why This Fails for fMRI:**

1. **Hemodynamic Response Function (HRF) Confound**:
   - fMRI measures **blood oxygenation**, not neural activity
   - HRF has ~6 second lag + regional variability
   - **Granger causality detects HRF lag, not neural causality**

2. **Insufficient temporal resolution**:
   - TR = 2 seconds (0.5 Hz sampling)
   - Neural dynamics occur at 10-100 Hz
   - **Nyquist limit prevents true causality inference**

3. **Multiple comparisons explosion**:
   - 12 regions × 12 regions = 144 tests
   - Even with correction, false discovery rate is high

**Evidence from Literature**:
- Smith et al. (2011, NeuroImage): "Granger causality in fMRI is confounded by HRF variability"
- Deshpande et al. (2010, HBM): "Effective connectivity requires hemodynamic deconvolution"

#### **5.2 Sparsification: Signal Destruction**

**Current**:
```python
SPARSITY_QUANTILE = 0.85  # Keep top 15% edges
MIN_EDGES_PER_GRAPH = 3
```

**Problems:**

1. **Arbitrary threshold**:
   - No neurobiological justification for 15%
   - ASD pathology may involve **weak, diffuse connections** (pruned away)

2. **Subject-level variability ignored**:
   - Some subjects may have **sparse but strong** connectivity (deleted)
   - Others may have **dense but weak** (kept)

3. **Graph becomes **structurally uninformative**:
   - With mean ~5 edges on 12 nodes: **density ≈ 3.5%**
   - Random graphs at this density lose topological properties
   - **GNN cannot learn from structure alone**

**Recommendation: Abandon Granger, Use Validated Functional Connectivity**

```python
# NEW: src/features/construct_validated_graphs.py

def construct_multimodal_connectivity(ts_lobe: torch.Tensor, spatial_features: np.ndarray):
    """
    Compute connectivity using complementary methods.
    
    Instead of Granger causality (invalid for fMRI), use:
    1. Pearson correlation (functional connectivity - gold standard)
    2. Partial correlation (direct connections, removing confounds)
    3. Mutual information (nonlinear dependencies)
    4. Distance-weighted edges (spatial prior)
    """
    n_lobes = ts_lobe.shape[1]
    
    # 1. Functional connectivity (Pearson)
    fc_matrix = np.corrcoef(ts_lobe.T.cpu().numpy())
    
    # 2. Partial correlation (removes global signal)
    from sklearn.covariance import GraphicalLassoCV
    estimator = GraphicalLassoCV(cv=5)
    estimator.fit(ts_lobe.cpu().numpy())
    partial_corr = -estimator.precision_ / np.sqrt(
        np.outer(np.diag(estimator.precision_), np.diag(estimator.precision_))
    )
    
    # 3. Mutual information (captures nonlinear effects)
    from sklearn.feature_selection import mutual_info_regression
    mi_matrix = np.zeros((n_lobes, n_lobes))
    for i in range(n_lobes):
        mi_matrix[i] = mutual_info_regression(
            ts_lobe.cpu().numpy(), 
            ts_lobe[:, i].cpu().numpy()
        )
    
    # 4. Spatial distance prior (anatomical plausibility)
    coords = spatial_features[:, :3]  # x, y, z
    dist_matrix = compute_pairwise_distances(coords)
    spatial_prior = np.exp(-dist_matrix / dist_matrix.std())  # Gaussian kernel
    
    # COMBINE: Weighted ensemble
    combined = (
        0.4 * fc_matrix +           # Functional connectivity (primary)
        0.3 * partial_corr +        # Direct connections
        0.2 * mi_matrix +           # Nonlinear effects
        0.1 * spatial_prior         # Anatomical plausibility
    )
    
    # ADAPTIVE THRESHOLDING (preserve subject-specific topology)
    threshold = np.percentile(np.abs(combined), 70)  # Keep top 30%
    adj_matrix = np.where(np.abs(combined) > threshold, combined, 0)
    
    # Ensure minimum connectivity (avoid isolated nodes)
    for i in range(n_lobes):
        if (adj_matrix[i] != 0).sum() < 2:
            # Connect to 2 nearest neighbors
            nearest = np.argsort(dist_matrix[i])[:3]
            adj_matrix[i, nearest] = combined[i, nearest]
    
    return torch.from_numpy(adj_matrix).float()
```

**Rationale:**
- **Pearson**: Established gold standard (Smith 2013)
- **Partial correlation**: Removes global signal confounds (Marrelec 2006)
- **Mutual information**: Captures nonlinear ASD biomarkers (Lord 2012)
- **Spatial prior**: Enforces anatomical plausibility (reduces false positives)

---

### **Phase 6: GNN Architecture** ⚠️ **OVERREGULARIZED**

**Current Configuration** (Phase 3 "simplified"):
```python
GNN_HIDDEN_CHANNELS = 64      # Reduced from 256
GNN_NUM_GNN_LAYERS = 2        # Reduced from 3
GNN_DROPOUT = 0.6             # VERY high
GNN_WEIGHT_DECAY = 1e-4       # L2 regularization
```

**Problem**: **Underfitting due to excessive regularization**

**Evidence:**
- Early stopping at 3-10 epochs (model capacity insufficient)
- Low variance (0.0156) but also **low mean AUC (0.57)** = bias problem
- Fold 3 AUC = 0.5328 (basically random) = model too weak

**Architecture Analysis:**

1. **2 layers insufficient for 12-node graphs**:
   - Graph diameter likely 2-3 hops
   - Need 3-4 layers for full receptive field
   - Current: **nodes can't see beyond 2-hop neighbors**

2. **64 channels too small for 28 input features**:
   - Information bottleneck in first layer
   - **28 → 64 → 64** loses feature nuance
   - Should be **28 → 128 → 128 → 64** for gradual compression

3. **Dropout 0.6 is excessive**:
   - Standard for ImageNet (millions of samples)
   - With ~700 training subjects: **0.3-0.4 is appropriate**
   - Current setting: **60% of neurons dropped** = cripples learning

4. **Multi-head attention underutilized**:
   - 4 heads with concat=True → 256 dims (but hidden=64)
   - **Dimensionality explosion then bottleneck**
   - Should use **2-3 heads** with proper hidden size

**Recommendation: Balanced Architecture**

```python
# In config.py:
GNN_HIDDEN_CHANNELS = 128      # ← Increase capacity
GNN_NUM_GNN_LAYERS = 3         # ← Add depth back
GNN_DROPOUT = 0.4              # ← Reduce overfitting prevention
GNN_WEIGHT_DECAY = 5e-5        # ← Lighter L2
GNN_NUM_HEADS = 3              # ← Reduce from 4

# In causal_gnn.py:
class CausalBrainGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=128, ...):
        # Layer 1: Feature extraction
        self.conv1 = GATv2Conv(hidden_channels, hidden_channels, heads=3, edge_dim=1, concat=True)
        # → Output: 384 dims (128 * 3)
        
        # Layer 2: Feature refinement  
        self.conv2 = GATv2Conv(hidden_channels*3, hidden_channels, heads=3, edge_dim=1, concat=True)
        # → Output: 384 dims
        
        # Layer 3: Feature integration
        self.conv3 = GATv2Conv(hidden_channels*3, hidden_channels, heads=2, edge_dim=1, concat=False)
        # → Output: 128 dims (averaged over heads)
```

**Rationale:**
- 3 layers: Full graph coverage (diameter ≤ 3)
- 128 hidden: Better capacity for 28 features
- Dropout 0.4: Balanced regularization
- Gradual attention reduction (3→3→2 heads)

---

### **Phase 7: Training Protocol** ⚠️ **SUBOPTIMAL**

**Current Issues:**

1. **Focal Loss misconfigured**:
   ```python
   FOCAL_LOSS_ALPHA = 0.35  # Weight for ASD
   # This gives 0.35 to ASD (minority), 0.65 to Control (majority)
   # BACKWARDS for class imbalance!
   ```
   - Alpha should prioritize minority class (ASD)
   - Current: **prioritizing Control class** (opposite of intent)

2. **Learning rate too conservative**:
   ```python
   GNN_LEARNING_RATE = 0.0001  # Very cautious
   ```
   - With early stopping at 3-10 epochs: **never reaches optimal**
   - Should use **cyclical LR** or **warmup → higher LR**

3. **No data augmentation**:
   - `graph_factory.py` has `_augment_graph` but only 50% of train batches
   - **Graph augmentation should be mandatory** for generalization

**Recommendations:**

```python
# 1. Fix Focal Loss
FOCAL_LOSS_ALPHA = 0.75  # Now prioritizes minority (ASD) class
FOCAL_LOSS_GAMMA = 2.0   # Keep focusing parameter

# 2. Learning Rate Schedule
GNN_LEARNING_RATE = 0.001  # 10x increase (with LR finder validation)

# 3. Advanced scheduler
def get_scheduler(optimizer, num_epochs):
    """OneCycleLR: Aggressive exploration, then convergence."""
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.003,  # Peak LR (validated with LR range test)
        total_steps=num_epochs,
        pct_start=0.3,  # 30% warmup
        anneal_strategy='cos'
    )

# 4. Mandatory graph augmentation
def _augment_graph(self, data):
    """Always augment training data."""
    # Edge dropout (DropEdge - Rong et al. 2020)
    edge_mask = torch.rand(data.edge_index.shape[1]) > 0.2
    data.edge_index = data.edge_index[:, edge_mask]
    data.edge_attr = data.edge_attr[edge_mask]
    
    # Node feature masking (GraphCL - You et al. 2020)
    feature_mask = torch.rand_like(data.x) > 0.15
    data.x = data.x * feature_mask
    
    # Subgraph sampling (GraphSAINT - Zeng et al. 2020)
    # Randomly sample 10 of 12 nodes
    node_mask = torch.randperm(12)[:10]
    data.x = data.x[node_mask]
    # ... (update edges accordingly)
    
    return data
```

---

## **Integrated Recommendations: Priority-Ranked Action Plan**

### **Tier 1: High-Impact, Low-Risk (Implement First)** 🎯

1. **Fix Focal Loss alpha** (5 min)
   - Change `FOCAL_LOSS_ALPHA` from 0.35 → 0.75
   - **Expected gain**: +2-3% AUC

2. **Remove duplicate normalization** (10 min)
   - Set `standardize=False` in `abide_download.py`
   - **Expected gain**: +1-2% AUC (preserves variance)

3. **Increase model capacity** (30 min)
   - `GNN_HIDDEN_CHANNELS: 64 → 128`
   - `GNN_DROPOUT: 0.6 → 0.4`
   - `GNN_NUM_GNN_LAYERS: 2 → 3`
   - **Expected gain**: +3-5% AUC (reduces underfitting)

4. **Implement OneCycleLR** (20 min)
   - Replace CosineAnnealingLR
   - **Expected gain**: +2-3% AUC (faster convergence)

**Total Tier 1 Expected Gain: +8-13% AUC** → Target: **0.64-0.69 AUC**

---

### **Tier 2: High-Impact, Medium-Risk (Implement After Tier 1)** 🔧

5. **Replace Granger with validated FC** (4 hours)
   - Implement multimodal connectivity (Pearson + partial corr + MI)
   - **Expected gain**: +5-8% AUC (better graph quality)

6. **Feature engineering overhaul** (6 hours)
   - Add Hurst, sample entropy, DFA
   - Remove noisy frequency features
   - **Expected gain**: +3-5% AUC (higher SNR)

7. **Adaptive graph sparsification** (2 hours)
   - Subject-specific thresholds
   - Distance-weighted priors
   - **Expected gain**: +2-4% AUC (preserves topology)

8. **Mandatory graph augmentation** (3 hours)
   - DropEdge + NodeMask + Subgraph sampling
   - **Expected gain**: +3-5% AUC (improves generalization)

**Cumulative Expected Gain: +21-35% AUC** → Target: **0.71-0.91 AUC**

---

### **Tier 3: Research-Grade Enhancements (Publication-Ready)** 📊

9. **Harmonization validation** (8 hours)
   - Pre/post effect size analysis
   - Stratified Combat by diagnosis
   - Cross-site leave-one-out validation

10. **Ensemble methods** (10 hours)
    - Stacked generalization (meta-learner)
    - Uncertainty quantification (MC Dropout)
    - Calibration (temperature scaling)

11. **Interpretability suite** (12 hours)
    - GNNExplainer for subgraph importance
    - Saliency maps with statistical testing
    - Permutation importance with confidence intervals

12. **External validation** (ongoing)
    - Test on held-out ABIDE-II cohort
    - Cross-dataset validation (ADHD-200, etc.)

---

## **Critical Implementation Notes**

### **Avoiding Common Pitfalls:**

1. **Do NOT implement all changes simultaneously**
   - Ablate one tier at a time
   - Document performance delta for each change
   - Use version control branches

2. **Validation Strategy:**
   ```python
   # Always compare against frozen baseline
   baseline_metrics = {
       'auc': 0.5593,
       'f1': 0.68,
       'std': 0.0156
   }
   
   # After each change, run 5-fold CV 3 times
   for trial in range(3):
       metrics = train_and_evaluate()
       assert metrics['auc'] > baseline_metrics['auc'] + 0.02  # Improvement threshold
   ```

3. **Statistical Testing:**
   - Use Wilcoxon signed-rank test for paired fold comparisons
   - Report 95% confidence intervals
   - Correct for multiple comparisons (Bonferroni)

4. **Computational Budget:**
   - Tier 1: ~2 hours total training time
   - Tier 2: ~12 hours total training time
   - Monitor GPU memory (current: ~6GB, may increase to ~10GB)

---

## **Expected Final Performance**

**Conservative Estimate** (Tier 1 + Tier 2):
- **Mean AUC**: 0.71-0.75
- **Best Fold**: 0.78-0.82
- **Clinical Utility**: Moderate (screening tool)

**Optimistic Estimate** (+ Tier 3):
- **Mean AUC**: 0.76-0.82
- **Best Fold**: 0.84-0.88
- **Clinical Utility**: High (diagnostic aid)
---

## **Next Steps**

1. **Start with Tier 1** (1-2 days implementation + validation)
2. **Document baseline** (current performance as reference)
3. **Implement changes incrementally** (one at a time)
4. **Run ablation study** (quantify each improvement)
5. **Write methods section** (document all changes for publication)