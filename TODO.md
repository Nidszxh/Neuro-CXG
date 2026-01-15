# Neuro-CXG: 2-Day Refactoring Sprint

## Objective
Refine existing implementation for stability, clarity, and performance without adding new features.

---

## Day 1: Critical Fixes & Consistency (8 hours)

### Morning Session (4 hours)

#### 1. Fix LOBE_MAPPING Duplication [CRITICAL] (30 min)
**Files to modify:**
- `src/data/construct_causal.py` - Remove lines 11-16, import from config
- `src/data/graph_factory.py` - Remove lines 108-113, import from config
- `src/config.py` - Add validation function

**Actions:**
```python
# In construct_causal.py and graph_factory.py, replace:
LOBE_MAPPING = {...}  # DELETE THIS

# With:
from config import LOBE_MAPPING, NUM_LOBES
```

**Test:** Run graph construction on 5 test subjects to verify

---

#### 2. Standardize Path Handling [CRITICAL] (1 hour)
**Current Issues:**
- `abide_download.py` uses `Path(__file__).resolve().parents[0]`
- Should use `config.py` paths consistently

**Files to fix:**
- `src/data/abide_download.py` (lines 8-13)
- `src/data/check_progress.py` (lines 3-4)
- `src/data/split.py` (lines 4-8)
- `src/utils/annotate.py` (lines 6-11)

**Standard pattern:**
```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    DATA_ROOT, DATA_IMAGES, DATA_PROCESSED, 
    ATLAS_PATH, PHENO_PATH
)
```

---

#### 3. Replace Print Statements with Basic Logging (1.5 hours)
**Goal:** Add simple logging without full framework overhead

**Quick logging pattern:**
```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Replace:
print(f"✅ Success! Generated 3D Node Features...")
# With:
logger.info("Generated 3D Node Features for %d subjects", len(final_df))
```

**Priority files:**
- `src/data/construct_causal.py`
- `src/data/harmonize.py`
- `src/models/gnn_model.py`
- `src/utils/manifest.py`

---

#### 4. Add Error Handling to Critical I/O Operations (1 hour)
**Pattern to apply:**
```python
# Before:
ts_data = torch.from_numpy(np.load(ts_path)).float()

# After:
try:
    ts_data = torch.from_numpy(np.load(ts_path)).float()
except FileNotFoundError:
    logger.error(f"Time series not found: {ts_path}")
    continue
except Exception as e:
    logger.error(f"Error loading {ts_path}: {e}")
    continue
```

**Critical locations:**
- `src/data/construct_causal.py` - Graph loading loop
- `src/data/graph_factory.py` - get() method
- `src/models/gnn_model.py` - Data loading
- `src/utils/compute_roi.py` - NPY loading

---

### Afternoon Session (4 hours)

#### 5. Fix graph_factory.py Dynamic ROI Issue [HIGH PRIORITY] (1.5 hours)
**Current Problem:** Lines 117-125 have fragile ROI detection

**Refactoring:**
```python
def get(self, idx):
    sub_id = self.manifest.iloc[idx]['subject_id']
    label = 1 if self.manifest.iloc[idx]['DX_GROUP'] == 1 else 0
    
    # Load harmonized features
    try:
        raw_row = self.node_attr.loc[sub_id].values
    except KeyError:
        logger.warning(f"Subject {sub_id} missing from node attributes")
        return None
    
    # EXPLICIT ROI COUNT HANDLING
    num_feats_per_roi = 6  # mean, std, skew, kurt, psd, mssd
    
    # Clean non-feature columns if present
    feature_only = raw_row[:-(raw_row.size % num_feats_per_roi)]
    num_rois = len(feature_only) // num_feats_per_roi
    
    if num_rois not in [116, 117, 170]:
        logger.warning(f"Unexpected ROI count {num_rois} for {sub_id}")
        return None
    
    ts_feats_raw = feature_only.reshape(num_rois, num_feats_per_roi)
    
    # Aggregate to 5 lobes using LOBE_MAPPING from config
    lobe_feats = []
    for lobe_id in range(NUM_LOBES):
        valid_indices = [i-1 for i in LOBE_MAPPING[lobe_id] if i <= num_rois]
        
        if not valid_indices:
            logger.warning(f"No valid ROIs for lobe {lobe_id} in {sub_id}")
            avg_feat = np.zeros(num_feats_per_roi)
        else:
            avg_feat = ts_feats_raw[valid_indices].mean(axis=0)
        
        lobe_feats.append(avg_feat)
    
    # Rest of implementation...
```

**Test:** Load 10 subjects and verify shapes

---

#### 6. Improve construct_causal.py Robustness (1 hour)
**Issues to fix:**
- No validation of time series before aggregation
- Silent failures if graph_data is empty
- No check for valid edge weights

**Improvements:**
```python
def main():
    manifest = pd.read_csv(MANIFEST_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Constructing causal graphs for {len(manifest)} subjects")
    
    success_count = 0
    fail_count = 0
    
    for _, row in tqdm(manifest.iterrows(), total=len(manifest)):
        sub_id, split = row['subject_id'], row['split']
        ts_path = DATASET_ROOT / split / "time_series" / f"{sub_id}_ts.npy"
        
        if not ts_path.exists():
            logger.warning(f"Missing time series for {sub_id}")
            fail_count += 1
            continue
        
        try:
            ts_data = torch.from_numpy(np.load(ts_path)).float().to(DEVICE)
            
            # Validate data quality
            if torch.isnan(ts_data).any() or torch.isinf(ts_data).any():
                logger.warning(f"Invalid values in time series for {sub_id}")
                fail_count += 1
                continue
            
            if ts_data.shape[0] < 50:
                logger.warning(f"Insufficient timepoints ({ts_data.shape[0]}) for {sub_id}")
                fail_count += 1
                continue
            
            # Aggregate and construct graph
            ts_lobes = aggregate_to_lobes(ts_data)
            causal_matrix = compute_causal_edges(ts_lobes)
            
            # Validate output
            if torch.isnan(causal_matrix).any():
                logger.error(f"NaN in causal matrix for {sub_id}")
                fail_count += 1
                continue
            
            # Sparsify and save
            thresh = torch.quantile(torch.abs(causal_matrix), 0.80)
            adj_matrix = torch.where(torch.abs(causal_matrix) > thresh, causal_matrix, 0.0)
            
            graph_data = {
                'adj': adj_matrix.cpu(),
                'node_features': ts_lobes.mean(dim=0).cpu()
            }
            
            torch.save(graph_data, OUTPUT_DIR / f"{sub_id}_graph.pt")
            success_count += 1
            
        except Exception as e:
            logger.error(f"Failed to process {sub_id}: {e}")
            fail_count += 1
            continue
    
    logger.info(f"Completed: {success_count} success, {fail_count} failures")
```

---

#### 7. Clean Up gnn_model.py Training Loop (1 hour)
**Improvements:**
- Add proper early stopping
- Better metric tracking
- Clean up print statements

```python
def run_kfold_training():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting {K_FOLDS}-fold cross-validation on {DEVICE}")
    
    from graph_factory import ABIDECausalDataset
    from causal_gnn import CausalBrainGNN
    
    full_dataset = ABIDECausalDataset(split='train')
    labels = [int(data.y) for data in full_dataset]
    
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        logger.info(f"=" * 60)
        logger.info(f"FOLD {fold+1}/{K_FOLDS}")
        logger.info(f"=" * 60)
        
        train_loader = DataLoader(
            [full_dataset[i] for i in train_idx], 
            batch_size=BATCH_SIZE, 
            shuffle=True
        )
        val_loader = DataLoader(
            [full_dataset[i] for i in val_idx], 
            batch_size=BATCH_SIZE
        )
        
        model = CausalBrainGNN(
            num_node_features=full_dataset[0].x.shape[1], 
            hidden_channels=64
        ).to(DEVICE)
        
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=LR, 
            weight_decay=WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=EPOCHS
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        best_auc = 0
        patience_counter = 0
        patience = 15
        
        for epoch in range(1, EPOCHS + 1):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, scheduler)
            
            if epoch % 10 == 0 or epoch == 1:
                metrics = evaluate(model, val_loader)
                
                logger.info(
                    f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
                    f"Acc: {metrics['acc']:.4f} | AUC: {metrics['auc']:.4f} | "
                    f"F1: {metrics['f1']:.4f}"
                )
                
                if metrics['auc'] > best_auc:
                    best_auc = metrics['auc']
                    patience_counter = 0
                    torch.save(
                        model.state_dict(), 
                        CHECKPOINT_DIR / f"best_model_fold{fold}.pt"
                    )
                    logger.info(f"New best model saved (AUC: {best_auc:.4f})")
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        fold_results.append(best_auc)
        logger.info(f"Fold {fold+1} best AUC: {best_auc:.4f}")
    
    logger.info("=" * 60)
    logger.info(f"CROSS-VALIDATION RESULTS")
    logger.info(f"Mean AUC: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")
    logger.info(f"Per-fold: {[f'{x:.4f}' for x in fold_results]}")
    logger.info("=" * 60)
```

---

#### 8. Code Style Consistency Pass (30 min)
**Quick fixes:**
- Consistent import ordering
- Remove unused imports
- Standardize docstring style
- Remove commented-out code

**Files to clean:**
- All files in `src/data/`
- All files in `src/models/`
- All files in `src/utils/`

---

## Day 2: Testing, Documentation & Final Polish (8 hours)

### Morning Session (4 hours)

#### 9. Create Validation Test Suite (2 hours)
**File:** `tests/test_data_pipeline.py`

```python
import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parents[1] / 'src'))

from config import (
    LOBE_MAPPING, NUM_LOBES, DATA_FINAL, 
    MASTER_MANIFEST, NODE_ATTRIBUTES_HARMONIZED
)
from data.graph_factory import ABIDECausalDataset
from data.construct_causal import aggregate_to_lobes, compute_causal_edges

class TestLobeMappingConsistency:
    """Test LOBE_MAPPING is consistent across codebase."""
    
    def test_lobe_mapping_complete(self):
        """All 5 lobes should have ROI assignments."""
        assert len(LOBE_MAPPING) == 5
        for lobe_id in range(5):
            assert lobe_id in LOBE_MAPPING
            assert len(LOBE_MAPPING[lobe_id]) > 0
    
    def test_no_duplicate_rois(self):
        """No ROI should be assigned to multiple lobes."""
        all_rois = []
        for roi_list in LOBE_MAPPING.values():
            all_rois.extend(roi_list)
        assert len(all_rois) == len(set(all_rois))
    
    def test_roi_range(self):
        """All ROIs should be in range [1, 170]."""
        for roi_list in LOBE_MAPPING.values():
            for roi in roi_list:
                assert 1 <= roi <= 170

class TestGraphConstruction:
    """Test causal graph construction."""
    
    def test_aggregate_to_lobes(self):
        """Test aggregation from 170 ROIs to 5 lobes."""
        # Create dummy time series
        ts_170 = torch.randn(100, 170)
        ts_lobes = aggregate_to_lobes(ts_170)
        
        assert ts_lobes.shape == (100, 5)
        assert not torch.isnan(ts_lobes).any()
        assert not torch.isinf(ts_lobes).any()
    
    def test_compute_causal_edges(self):
        """Test causal edge computation."""
        ts_lobes = torch.randn(100, 5)
        adj_matrix = compute_causal_edges(ts_lobes)
        
        assert adj_matrix.shape == (5, 5)
        assert not torch.isnan(adj_matrix).any()
        assert not torch.isinf(adj_matrix).any()

class TestDatasetLoader:
    """Test dataset loading and validation."""
    
    @pytest.fixture
    def dataset(self):
        """Load train dataset."""
        return ABIDECausalDataset(split='train')
    
    def test_dataset_loads(self, dataset):
        """Dataset should load without errors."""
        assert len(dataset) > 0
    
    def test_sample_structure(self, dataset):
        """Each sample should have correct structure."""
        sample = dataset[0]
        
        assert hasattr(sample, 'x')  # Node features
        assert hasattr(sample, 'edge_index')  # Edges
        assert hasattr(sample, 'edge_attr')  # Edge weights
        assert hasattr(sample, 'y')  # Label
        
        # Check dimensions
        assert sample.x.shape[0] == 5  # 5 lobes
        assert sample.y.shape[0] == 1  # Single label
    
    def test_labels_valid(self, dataset):
        """All labels should be 0 or 1."""
        for i in range(min(10, len(dataset))):
            sample = dataset[i]
            assert sample.y.item() in [0, 1]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

---

#### 10. Update README.md (1 hour)
**Add missing sections:**

```markdown
## Installation

### Requirements
- Python 3.8+
- CUDA-capable GPU (8GB+ VRAM recommended)
- 32GB RAM minimum

### Setup
```bash
# Clone repository
git clone https://github.com/yourusername/Neuro-CXG.git
cd Neuro-CXG

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from src.config import validate_environment; validate_environment()"
```

## Quick Start

### 1. Data Preparation
```bash
# Download ABIDE data and extract time series
python -m src.data.abide_download

# Create train/val/test splits
python -m src.data.split

# Generate master manifest
python -m src.utils.manifest
```

### 2. Feature Extraction
```bash
# Extract temporal features from time series
python -m src.utils.compute_roi

# Harmonize features across sites
python -m src.data.harmonize
```

### 3. Graph Construction
```bash
# Construct causal graphs
python -m src.data.construct_causal
```

### 4. Model Training
```bash
# Train GNN with 5-fold cross-validation
python -m src.models.gnn_model
```

## Project Structure
```
Neuro-CXG/
├── configs/
│   └── brain.yaml          # YOLO configuration
├── data/                   # Data directory (not tracked)
│   ├── raw/               # Original downloads
│   ├── processed/         # Processed time series
│   ├── final/            # Train/val/test splits
│   └── metadata/         # Manifests and features
├── src/
│   ├── config.py         # Central configuration
│   ├── data/             # Data processing modules
│   ├── models/           # GNN architecture and training
│   ├── pipelines/        # ROI detection pipeline
│   └── utils/            # Utilities
├── tests/                # Unit tests
├── notebooks/            # Jupyter notebooks for EDA
└── results/              # Training outputs
```

## Troubleshooting

### Common Issues

**Issue:** `FileNotFoundError: Atlas not found`
- **Solution:** Download AAL3 atlas to `data/atlases/AAL3v1.nii`

**Issue:** `CUDA out of memory`
- **Solution:** Reduce batch size in `config.py` (try BATCH_SIZE=16)

**Issue:** Missing subjects in dataset
- **Solution:** Run `python -m src.data.check_progress` to diagnose

## Citation
```bibtex
@article{neurocxg2025,
  title={Neuro-CXG: Causal Graph Neural Networks for Brain Disorder Classification},
  author={Your Name},
  journal={TBD},
  year={2025}
}
```
```

---

#### 11. Update ROADMAP.md to Match Implementation (1 hour)
**Add section at top:**

```markdown
# Implementation Status (January 2025)

## ✅ Completed Phases

### Phase 1-2: Project Setup & Data Acquisition (100%)
- [x] Environment configuration
- [x] ABIDE data download pipeline
- [x] Time series extraction
- [x] Data organization and splitting

### Phase 3: ROI Detection (100%)
- [x] YOLO training on brain slices
- [x] Automated annotation pipeline
- [x] Model evaluation and optimization

### Phase 4: Feature Extraction (100%)
- [x] Temporal feature computation
- [x] Spatial feature extraction
- [x] Site harmonization with neuroCombat

### Phase 5: Graph Construction (100%)
- [x] Functional connectivity matrices
- [x] Causal graph construction (lagged correlation)
- [x] Graph dataset creation

### Phase 6: GNN Development (100%)
- [x] CausalBrainGNN architecture (GAT-based)
- [x] K-fold cross-validation training
- [x] Model evaluation framework

## 🔧 Current Focus: Refinement (January 15-16, 2025)

### Day 1: Critical Fixes
- [ ] Fix LOBE_MAPPING duplication
- [ ] Standardize path handling
- [ ] Add basic logging
- [ ] Improve error handling
- [ ] Fix graph_factory.py ROI detection

### Day 2: Testing & Documentation
- [ ] Create test suite
- [ ] Update README and ROADMAP
- [ ] Code style consistency
- [ ] Final validation

## 🎯 Next Steps (Post-Refinement)

### Phase 7: Explainability (Planned)
- [ ] Enhanced node importance analysis
- [ ] Edge importance computation
- [ ] Saliency map generation
- [ ] Clinical validation

### Phase 8: Publication Preparation (Planned)
- [ ] Results analysis and visualization
- [ ] Comparison with baselines
- [ ] Manuscript preparation
- [ ] Code release preparation
```

---

### Afternoon Session (4 hours)

#### 12. Comprehensive Code Review & Cleanup (2 hours)

**Checklist for each file:**

✓ Imports organized (stdlib → third-party → local)
✓ No unused imports
✓ Consistent naming conventions
✓ Docstrings for all functions
✓ No hardcoded paths (use config)
✓ Logging instead of print
✓ Error handling for I/O
✓ Type hints for function signatures

**Priority files to review:**
1. `src/config.py`
2. `src/data/graph_factory.py`
3. `src/data/construct_causal.py`
4. `src/models/causal_gnn.py`
5. `src/models/gnn_model.py`

---

#### 13. Update .gitignore (15 min)

Add any missing patterns:
```
# Python
__pycache__/
*.pyc
.pytest_cache/

# Logs
logs/
*.log

# IDE
.vscode/
.idea/
*.swp

# Data (already covered)

# Checkpoints
*.pt
*.pth

# Results
results/*
!results/.gitkeep
```

---

#### 14. Create requirements.txt with Pinned Versions (15 min)

```txt
# Core ML
torch==2.1.0
torchvision==0.16.0
torchaudio==2.1.0

# GNN
torch-geometric==2.4.0
torch-scatter==2.1.2
torch-sparse==0.6.18

# Neuroimaging
nibabel==5.1.0
nilearn==0.10.2
neuroCombat==0.2.12

# AWS S3
boto3==1.29.7
botocore==1.32.7

# YOLO
ultralytics==8.1.0

# ML utilities
scikit-learn==1.3.2
numpy==1.24.3
pandas==2.1.3
scipy==1.11.4
tqdm==4.66.1

# Visualization
matplotlib==3.8.2
seaborn==0.13.0
networkx==3.2.1
pillow==10.1.0

# Experiment tracking & notebooks
jupyter==1.0.0
ipykernel==6.27.1

# Testing
pytest==7.4.3
pytest-cov==4.1.0
```

---

#### 15. Final Integration Test (1 hour)

**Run full pipeline on small subset:**

```bash
# Test script: tests/test_full_pipeline.sh

#!/bin/bash

echo "Testing full Neuro-CXG pipeline..."

# 1. Validate environment
python -c "from src.config import validate_environment; validate_environment()"

# 2. Check data availability
python -m src.data.check_progress

# 3. Test graph construction on 5 subjects
python tests/test_construct_subset.py

# 4. Test dataset loading
python -c "from src.data.graph_factory import ABIDECausalDataset; ds = ABIDECausalDataset('train'); print(f'Loaded {len(ds)} subjects')"

# 5. Test model instantiation
python -c "from src.models.causal_gnn import CausalBrainGNN; model = CausalBrainGNN(9, 64); print(f'Model parameters: {sum(p.numel() for p in model.parameters())}')"

echo "✅ All tests passed!"
```

---

#### 16. Documentation Polish (30 min)

**Add docstrings to key functions:**

Example for `graph_factory.py`:
```python
class ABIDECausalDataset(Dataset):
    """
    PyTorch Geometric dataset for ABIDE causal brain graphs.
    
    Loads causal graphs constructed from fMRI time series, combining:
    - Harmonized temporal features (6 per ROI, aggregated to 5 lobes)
    - Spatial coordinates from YOLO detections
    - Directed causal adjacency matrices
    
    Args:
        split: Data split to load ('train', 'val', or 'test')
        transform: Optional transform to apply to graphs
        pre_transform: Optional pre-transform to apply once
        
    Attributes:
        manifest: DataFrame with subject metadata
        node_attr: Harmonized node features
        coords: 3D spatial coordinates
        adj_dir: Directory containing causal graph files
        
    Example:
        >>> dataset = ABIDECausalDataset(split='train')
        >>> print(f"Loaded {len(dataset)} subjects")
        >>> sample = dataset[0]
        >>> print(f"Graph: {sample.x.shape[0]} nodes, {sample.edge_index.shape[1]} edges")
    """
```

---

## Final Checklist

### Day 1 Deliverables
- [ ] LOBE_MAPPING centralized to config.py only
- [ ] All files use config.py for paths
- [ ] Basic logging added to 5+ key modules
- [ ] Error handling added to critical I/O operations
- [ ] graph_factory.py ROI detection refactored
- [ ] construct_causal.py validation added
- [ ] gnn_model.py training loop cleaned up
- [ ] Code style consistency pass completed

### Day 2 Deliverables
- [ ] Test suite created and passing
- [ ] README.md updated with installation/usage
- [ ] ROADMAP.md updated with current status
- [ ] .gitignore comprehensive
- [ ] requirements.txt pinned versions
- [ ] Full pipeline integration test passing
- [ ] Docstrings added to key functions
- [ ] All code committed and pushed

---

## Success Metrics

**Code Quality:**
- No duplicate LOBE_MAPPING definitions
- Zero hardcoded paths outside config.py
- All print statements replaced with logging
- Error handling on all file I/O

**Stability:**
- Pipeline runs without crashes on 50+ subjects
- Reproducible results with same random seed
- No NaN/Inf values in outputs

**Documentation:**
- README allows new user to run pipeline
- All major functions have docstrings
- Test suite validates core functionality

---

## Time Allocation Summary

**Day 1:** 8 hours
- Critical fixes: 4h
- Refactoring: 3h
- Code cleanup: 1h

**Day 2:** 8 hours
- Testing: 2h
- Documentation: 2h
- Integration: 1.5h
- Final polish: 2.5h

**Total:** 16 hours over 2 days