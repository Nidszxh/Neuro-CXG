# Training Infrastructure

## Overview
This document covers training architectures, loss functions, optimization strategies, and learning objectives used in Neuro-CXG.

---

## Loss Functions

### FocalLoss (`src/models/losses.py`)

**Purpose**: Address class imbalance and hard-example mining in binary classification (ASD vs Control).

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

---

## Training Configuration

### Learning Rate Schedule (OneCycle)
```
max_lr: 0.002 (GNN_ONECYCLE_MAX_LR)
pct_start: 0.2 (20% of epochs increase, 80% decrease)
```

### Early Stopping
```
patience: 30 epochs (GNN_EARLY_STOPPING_PATIENCE)
mode: max (maximize validation AUC)
```

### Regularization
- **L2 weight decay**: `5e-5` (GNN_WEIGHT_DECAY)
- **Dropout**: `0.35` (GNN_DROPOUT)
- **Structural dropout** (optional): randomly zero node features in ~30% of graphs

---

## GNN Architecture Highlights

### CausalBrainGNN (src/models/causal_gnn.py)

**Input Shape**
- Nodes: 12 brain lobes
- Node features: 28 dimensions

**Layers**
- GATv2Conv: 2 layers, 4 attention heads, 128 hidden channels
- Activation: GELU

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

---

## Testing and Validation

### Unit Tests
```bash
pytest tests/unit/
```

### Integration Tests
```bash
pytest tests/integration/
```

---

## Known Issues and Workarounds

- CV-Test Gap: site-specific scanner variations; use site-stratified CV
- Gamma Band: zeroed at runtime for TR=2s subjects via `UNRELIABLE_FREQ_BANDS_AT_NYQUIST`
- CUDA Graph Issue: `torch.compile` disabled in `_maybe_compile_model()`

---

## References
- `src/core/hyperparams.py`
- `src/models/gnn_model.py`
- `src/models/training_utils.py`
