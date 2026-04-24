# Training Infrastructure

## Overview
This document covers training architectures, loss functions, optimization strategies, and learning objectives used in Neuro-CXG.

---

## Loss Functions

### FocalLoss (`src/models/losses.py`)

**Purpose**: Address class imbalance and hard-example mining in binary classification (ASD vs Control).

**Implementation**
```python
class FocalLoss(nn.Module):
    """Multi-class focal loss with optional positive-class reweighting.
    
    Args:
        alpha: Weight for class 1 (ASD). Class 0 receives (1 - alpha).
        gamma: Focusing parameter for hard-example mining (higher = focus more on hard examples).
        pos_weight: Optional multiplicative weight for class 1 examples.
    """
```

**Algorithm**
1. Compute softmax probabilities from logits
2. One-hot encode targets
3. Extract class probability: $p_t = \text{softmax}(logits)[class]$
4. Focal weight: $(1 - p_t)^{\gamma}$ — upweight hard negatives
5. Alpha weight: per-class balance factor
6. Combined loss: $\text{CrossEntropy} \times \text{alpha\_weight} \times \text{focal\_weight}$
7. Optional class reweighting when `pos_weight` is specified

**Default Configuration** (from `src/core/hyperparams.py`)
- `FOCAL_LOSS_ALPHA = 0.75` — class 1 weight (ASD)
- `FOCAL_LOSS_GAMMA = 3.0` — focusing parameter (typical range 1.0–5.0)
- `pos_weight = class_imbalance_ratio` — computed from train split

**Why Focal Loss?**
- ABIDE dataset: ~486 ASD / ~514 Control → mild imbalance
- Focal loss emphasizes hard samples during training
- Outperforms standard cross-entropy in class-imbalanced settings
- Alternative considered: weighted cross-entropy (less effective on this cohort)

**Usage**
```python
from src.models.losses import FocalLoss

criterion = FocalLoss(
    alpha=0.75, 
    gamma=3.0, 
    pos_weight=1.05  # class imbalance ratio
)
loss = criterion(logits, targets)
```

**Integration Points**
- Primary loss in `gnn_model.py` training loop
- Used in `experiments/run_ablations.py` for comparative runs
- Used in `experiments/data_quality.py` for robustness studies

---

## Training Configuration

### Learning Rate Schedule (OneCycle)
```
max_lr: 0.002 (GNN_ONECYCLE_MAX_LR)
pct_start: 0.2 (20% of epochs increase, 80% decrease)
total_steps: epochs × steps_per_epoch
```

**Rationale**: High initial LR → explore loss landscape; gradual decrease → converge.

### Early Stopping
```
patience: 30 epochs (GNN_EARLY_STOPPING_PATIENCE)
mode: max (maximize validation AUC)
```

### Regularization
- **L2 weight decay**: `5e-5` (GNN_WEIGHT_DECAY)
- **Dropout**: `0.35` (GNN_DROPOUT)
- **Structural dropout** (optional): randomly zero node features in ~30% of graphs to force edge-structure learning
- **Edge-structure contrastive loss** (optional): NT-Xent between full-feature and edge-only embeddings

---

## Multi-View Training Objectives

When multiview graphs are available (`construct_multiview_graphs` stage):

### 1. CausalInvarianceLoss
**Purpose**: Ensure consistent predictions across different causal graph estimates.

- Uses Normalized Temperature-scaled Cross Entropy (NT-Xent) loss
- Temperature τ = 0.07
- Loss weight: 0.15
- Activates automatically when multiview dir populated

### 2. SpatialInvarianceLoss
**Purpose**: Guard against site-specific spatial artifacts leaking into predictions.

- Ensures residual site variance after adversarial training is minimized
- Protects 4-dimensional spatial feature representation

### 3. EdgeStructureContrastiveLoss
**Purpose**: Enforce edge structure (causality) as primary classification signal.

- Contrasts full-feature embedding vs edge-structure-only embedding
- NT-Xent loss with τ = 0.5
- Loss weight: 0.05

---

## GNN Architecture Highlights

### CausalBrainGNN (src/models/causal_gnn.py)

**Input Shape**
- Nodes: 12 brain lobes
- Node features: 28 dimensions (temporal + frequency + internal + spatial)
- Edges: directed causal adjacency (Granger causality)
- Edge attributes: 1 dimension (causality weight)

**Layers**
- GATv2Conv: 2 layers, 4 attention heads, 128 hidden channels
- Activation: GELU
- Skip connections: residual add between layers
- Edge gating: weight incoming messages by edge attributes

**Pooling** (configurable)
- `"attention"`: learnable node attention (default, stable)
- `"anatomical"`: 2-level hierarchy (lobes → networks → graph), with `AnatomicalHierarchyPool`

**Output**
- Logits: shape (batch_size, 2) for binary classification
- Optional: embeddings for explainability

---

## Hyperparameter Tuning Grid

Completed experiments (April 2026):

| Parameter | Tested Values | Notes |
|-----------|---|---|
| `GNN_NUM_LAYERS` | 1, 2, 3 | 2 selected (depth vs overfitting) |
| `GNN_DROPOUT` | 0.2, 0.35, 0.5 | 0.35 selected |
| `FOCAL_LOSS_GAMMA` | 1.0, 2.0, 3.0 | 3.0 selected (hard-example focus) |
| `GNN_LEARNING_RATE` | 1e-4, 1e-3, 1e-2 | 1e-3 selected |
| `GNN_ONECYCLE_MAX_LR` | 0.001, 0.002, 0.005 | 0.002 selected |
| `GNN_EARLY_STOPPING_PATIENCE` | 15, 30, 50 | 30 selected |
| `GRANGER_SPARSITY_QUANTILE` | 0.6, 0.70, 0.80 | 0.70 selected (keep top 30%) |

---

## Canonical Baseline Metrics

**Reference Run**: pipeline_20260309_194459 (March 9, 2026)

| Metric | Value | Notes |
|--------|-------|-------|
| CV AUC | 0.7434 ± 0.0417 | 5-fold stratified |
| Per-fold AUCs | [0.7317, 0.7576, 0.7606, 0.6709, 0.7964] | Fold 4 best |
| Test AUC | 0.6487 [0.5618, 0.7300] | Bootstrap CI, permutation p=0.0020 |
| Test F1 | 0.6738 | At F1-optimal threshold |

**Current Run** (April 24, 2026):
- CV AUC: 0.8004 ± 0.0293
- Test AUC: 0.8753
- Test F1: 0.8121
- Graph health: 48.7 mean edges, 0 dead lobes

---

## Training Loop Patterns

### Standard Single-Fold Training
```python
from src.models.training_utils import EarlyStopping, CheckpointManager, TrainingTracker

early_stop = EarlyStopping(patience=30, mode='max')
ckpt_mgr = CheckpointManager(checkpoint_dir, prefix='best_model')
tracker = TrainingTracker()

for epoch in range(max_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_auc = evaluate(model, val_loader, cfg=config)
    
    tracker.log_epoch(epoch, {'train_loss': train_loss, 'val_auc': val_auc})
    
    if early_stop(val_auc):
        print(f"Early stopping at epoch {epoch}")
        break
    
    if val_auc > best_auc:
        ckpt_mgr.save(model, optimizer, epoch=epoch, metrics={'auc': val_auc})
```

### 5-Fold Cross-Validation
```python
from src.features.graph_factory import ABIDECausalDataset
from sklearn.model_selection import StratifiedKFold

dataset = ABIDECausalDataset(split='train')
manifest = dataset.manifest  # Access fold assignments

for fold_id in range(5):
    train_idx = manifest[manifest['fold'] != fold_id].index
    val_idx = manifest[manifest['fold'] == fold_id].index
    
    # Load fold-specific harmonized features
    fold_features = pd.read_csv(f'data/metadata/harmonized_folds_cv/harmonized_fold_{fold_id}.csv')
    
    train_fold(fold_id, train_idx, val_idx, fold_features)
```

---

## Testing and Validation

### Unit Tests
```bash
# Training utilities
pytest tests/unit/test_training_utils.py -v

# Loss functions
pytest tests/unit/test_losses.py -v

# GNN architecture
pytest tests/unit/test_causal_gnn.py -v
```

### Integration Tests
```bash
# Full training pipeline
pytest tests/integration/test_gnn_training.py -v

# Multi-fold CV
pytest tests/integration/test_cv_pipeline.py -v
```

---

## Known Issues and Workarounds

### Issue 1: CV-Test Gap (0.0947)
- **Symptom**: CV AUC 0.7434 but test AUC 0.6487
- **Root cause**: Site-specific scanner variations not fully captured by ComBat
- **Workaround**: Site-stratified CV (DD-013) partially addresses; GRL alpha tuning pending
- **Status**: Phase 11.2, under investigation

### Issue 2: Gamma Band Aliasing (Nyquist)
- **Symptom**: Gamma band (0.20-0.25 Hz) at Nyquist limit for TR=2s
- **Workaround**: Gamma zeroed at runtime for TR=2s subjects via `UNRELIABLE_FREQ_BANDS_AT_NYQUIST`
- **Status**: Phase 10.2 mitigation in place

### Issue 3: torch.compile CUDA Graph Issue
- **Symptom**: `RuntimeError: tensor reuse in CUDA graphs` during gradient accumulation
- **Fix**: Disabled `torch.compile` in `gnn_model._maybe_compile_model()` (lines 393-423)
- **Status**: FIXED April 12, 2026; re-enable when PyTorch fixes CUDA graph lifecycle

---

## References
- `.github/copilot-instructions.md` — design rationale
- `CHANGELOG.md` — feature history
- `src/core/hyperparams.py` — parameter definitions
- `src/models/gnn_model.py` — full training orchestration
- `src/models/training_utils.py` — utilities (loss, scheduler, checkpointing)
