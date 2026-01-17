# Neuro-CXG

**Causal Graph Neural Networks for Brain Disorder Classification from fMRI**

A Graph Neural Network framework for brain disorder classification (ASD vs Control) using causal inference and explainable AI (XAI) on fMRI data. Combines YOLO-based ROI detection, causal graph construction, and Graph Attention Networks for interpretable neuroimaging analysis.

## Key Features

- **YOLO-based ROI Detection**: Automated detection of 5 brain anatomical lobes in 2D MRI slices using YOLO11s
- **Causal Graph Construction**: 5×5 directed graphs from fMRI time series using lagged Pearson correlation
- **Graph Neural Networks**: GATv2-based architecture (2 heads, 2 layers) for classification with interpretable edge weights
- **Batch Effect Harmonization**: neuroCombat integration for multi-site data harmonization (safe NaN/Inf handling)
- **Stratified k-fold Validation**: 5-fold CV balanced by diagnosis and scanner site (2D stratification)
- **Explainability**: Gradient-based node importance and causal edge weight analysis via `get_node_importance()`
- **Unified Pipeline**: Single `run_pipeline.py` orchestrates all stages with validation and diagnostics


### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/Neuro-CXG.git
cd Neuro-CXG

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (pinned versions for reproducibility)
pip install -r requirements.txt

# Verify installation
python src/config.py
```

## Quick Start

### 1. Data Preparation

Download ABIDE data and organize into train/val/test splits:

```bash
# Download raw fMRI and extract 5 z-slices per subject
python -m src.data.abide_download

# Create master manifest
python -m src.utils.manifest

# Generate stratified train/val/test splits (70/15/15)
python -m src.data.split
```

### 2. ROI Detection (YOLO)

Train YOLO11s to detect 5 brain lobes in slices:

```bash
python -m src.pipelines.roi_detection
# Outputs: results/ROI_Detection_v20_Final4/weights/best.pt
```

### 3. Feature Extraction

Extract temporal and spatial features from detected ROIs:

```bash
# Extract 3D spatial coordinates from YOLO detections
python -m src.data.extract_features
# Outputs: data/processed/metadata/node_features_3d.csv

# Extract temporal features (mean, std, skew, kurtosis, PSD, MSSD)
python -m src.utils.compute_roi
# Outputs: data/processed/metadata/node_attributes_temporal.csv
```

### 4. Batch Effect Harmonization

Remove site-specific scanner bias while protecting disease signal:

```bash
python -m src.data.harmonize
# Outputs: data/processed/metadata/node_attributes_harmonized.csv
# (Note: DX_GROUP diagnosis protected as covariate)
```

### 5. Graph Construction

Build directed causal graphs from time series:

```bash
python -m src.data.construct_causal
# Outputs: data/processed/causal_graphs/{subject_id}_graph.pt
# Graphs: 5 nodes (lobes), directed edges, causal correlation weights
```

### 6. Model Training

Train GNN with 5-fold stratified cross-validation:

```bash
python -m src.models.gnn_model
# Checkpoints saved to: models/checkpoints/best_model_fold{0-4}.pt
# Logs metrics: Accuracy, F1, AUC, Confusion Matrix per fold
```

### OR: Run Full Pipeline (Recommended)

Use the unified pipeline runner to execute all stages:

```bash
# Full pipeline with diagnostics and safe harmonization
python src/run_pipeline.py --run-diagnostics --run-safe-harmonize --log-file logs/pipeline.log

# Skip download and data splitting (use existing data)
python src/run_pipeline.py --skip-split --run-manifest --run-safe-harmonize

# Force reset all intermediate files and rebuild
python src/run_pipeline.py --force-reset --run-diagnostics

# Dry run to see execution plan
python src/run_pipeline.py --dry-run
```

The pipeline orchestrates:
1. Environment validation (paths, CUDA, lobe mapping)
2. Optional: ABIDE download + preprocessing  
3. Stratified train/val/test split (70/15/15)
4. Master manifest generation
5. Optional: Atlas validation
6. Optional: Pipeline health diagnostics
7. YOLO ROI detection (or skip if weights exist)
8. Spatial feature extraction (5-lobe 3D coords)
9. Temporal feature extraction (6 stats per lobe)
10. Safe harmonization (neuroCombat with NaN handling)
11. Causal graph construction (5×5 directed)
12. GNN training (5-fold stratified CV)

## Current Results

**5-Fold Cross-Validation Performance (Full Training Set):**

| Metric | Mean ± Std | Range | Notes |
|--------|------------|-------|-------|
| **AUC** | 0.5354 ± 0.0562 | 0.4584 - 0.6056 | Near random; requires tuning |
| **F1** | 0.6586 ± 0.0164 | 0.6479 - 0.6911 | Reasonable for imbalanced data |
| **Accuracy** | 0.5193 ± 0.0454 | - | Slightly below random |
| **Optimal Threshold** | 0.588 | - | Learned from validation set |

**Interpretation:**
- AUC ~0.535 suggests the model is learning slightly better than random chance
- High F1 scores indicate good precision-recall balance
- Results indicate need for architectural improvements, class rebalancing, or enhanced feature engineering
- 5-fold consistency (low std in F1) suggests stable training process

**Per-Fold AUC Breakdown:**
- Fold 0: 0.4996 (worst)
- Fold 1: 0.5195
- Fold 2: 0.5937
- Fold 3: 0.6056 (best)
- Fold 4: 0.4584

**Next Steps for Improvement:**
1. Investigate class imbalance (ASD vs Control ratios)
2. Add class weights or focal loss to handle imbalance
3. Experiment with deeper GNN architectures or additional attention heads
4. Augment features with additional temporal/spectral measures
5. Perform ablation studies on harmonization impact
6. Validate on held-out test set (currently only using 5-fold CV on train)

## Project Structure

```
Neuro-CXG/
├── .github/
│   └── copilot-instructions.md    # AI agent guidelines
├── configs/
│   └── brain.yaml                 # YOLO configuration (5 lobe classes)
├── data/                          # Data directory (not tracked)
│   ├── raw/                       # Original fMRI/DTI downloads
│   ├── processed/                 # Processed time series
│   │   ├── causal_graphs/         # PyTorch graph objects (.pt)
│   │   └── metadata/              # CSVs: features, manifests
│   ├── final/                     # Train/val/test split images
│   └── atlases/                   # AAL3 reference atlas
├── src/
│   ├── config.py                  # Central configuration (SINGLE SOURCE OF TRUTH)
│   ├── run_pipeline.py            # Unified pipeline orchestrator
│   ├── pipeline_diagnostics.py    # Comprehensive health check
│   ├── safe_harmonization.py      # Robust harmonization with NaN handling
│   ├── atlas_validator.py         # AAL atlas validation tool
│   ├── data/                      # Data processing modules
│   │   ├── extract_features.py    # YOLO inference → spatial features
│   │   ├── harmonize.py           # neuroCombat batch effect removal
│   │   ├── construct_causal.py    # Graph construction (lagged correlation)
│   │   ├── graph_factory.py       # PyTorch Geometric dataset loader
│   │   ├── split.py               # Stratified splitting
│   │   └── abide_download.py      # ABIDE data download and preprocessing
│   ├── models/                    # GNN architecture and training
│   │   ├── causal_gnn.py          # GAT-based model (2 heads, skip connections)
│   │   └── gnn_model.py           # k-fold training loop
│   ├── pipelines/
│   │   └── roi_detection.py       # YOLO training entry point
│   └── utils/                     # Utility functions
│       ├── manifest.py            # Manifest generation
│       ├── compute_roi.py         # Temporal feature extraction
│       ├── integrity_check.py     # Data integrity validation
│       └── annotate.py            # Atlas-based label annotation
├── notebooks/
│   └── eda1.ipynb                 # Exploratory data analysis
├── results/                       # YOLO training outputs
├── models/checkpoints/            # Best GNN models per fold
├── requirements.txt               # Pinned package versions
├── ROADMAP.md                     # Development phases
├── TODO.md                        # 2-day refactoring sprint
└── README.md                      # This file
```

## Configuration

All project constants defined in [src/config.py](src/config.py) (single source of truth):

### Critical Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| NUM_LOBES | 5 | Frontal, Temporal, Parietal, Occipital, Limbic |
| LOBE_MAPPING | 170→5 | AAL3 atlas ROI aggregation (1-indexed to 0-indexed) |
| GNN_IN_CHANNELS | 9 | 6 temporal + 3 spatial (x,y,z) features |
| GNN_HIDDEN_CHANNELS | 64 | Hidden dimension for GATv2Conv |
| GNN_NUM_HEADS | 2 | Attention heads per GAT layer (sufficient for 5 nodes) |
| GNN_DROPOUT | 0.5 | High dropout to prevent site-specific memorization |
| K_FOLDS | 5 | Cross-validation folds |
| YOLO_BATCH_SIZE | 24 | (32+ causes OOM on RTX 4060 8GB) |
| YOLO_EPOCHS | 100 | YOLO training epochs |
| GNN_EPOCHS | 100 | GNN training epochs |
| CAUSAL_LAG | 1 | Time lag for temporal precedence (TRs) |
| SPARSITY_QUANTILE | 0.80 | Keep top 20% causal connections (~5 edges/graph) |

See [src/config.py](src/config.py) for all 60+ parameters.

## Data Format

### Time Series Input
- **Shape**: `(timepoints, 170)` - fMRI signal from 170 AAL ROIs
- **Processing**: Bandpass filtered (0.01-0.08 Hz), z-normalized

### Graph Construction Output

**Intermediate format** (saved by construct_causal.py):
```python
{
  'adj': torch.Tensor(5, 5),        # 5×5 directed adjacency matrix
  'subject_id': str,                # Subject identifier
  'lobe_order': list                # ['Frontal', 'Temporal', ...]
}
```

**Final format** (loaded by graph_factory.py into PyTorch Geometric):
```python
Data(
  x=torch.Tensor(5, 9),          # 5 lobes × (6 temporal + 3 spatial features)
  edge_index=torch.Tensor(2, K), # K directed edges (typically ~5 after sparsification)
  edge_attr=torch.Tensor(K,),    # Causal correlation weights [-1, 1]
  y=torch.Tensor([0 or 1]),      # Label: 0=Control, 1=ASD
  pos=torch.Tensor(5, 3),        # 3D spatial coordinates (from YOLO detections)
  sub_id=str                      # Subject identifier
)
```

**Key differences:**
- `construct_causal.py` outputs raw adjacency matrices as dictionaries
- `graph_factory.py` (ABIDECausalDataset) converts to PyG Data objects on-the-fly
- Edge sparsification: Only top 20% of correlations by absolute value are kept
```

## Validation & Testing

### Pipeline Health Check

```bash
# Run comprehensive diagnostics on all pipeline stages
python src/pipeline_diagnostics.py

# Outputs:
# - Environment validation (paths, CUDA, dependencies)
# - Data integrity checks (file counts, splits)
# - Feature matrix validation (shapes, NaN detection)
# - Graph construction validation (node/edge counts)
# - Atlas validation (ROI coverage, mapping correctness)
```

### Manual Validation Commands

```bash
# Validate environment and config
python -c "from src.core.config import validate_environment; validate_environment()"

# Check lobe mapping integrity
python -c "from src.core.config import validate_lobe_mapping; validate_lobe_mapping()"

# Test dataset loading
python -c "from src.features.graph_factory import ABIDECausalDataset; \
ds = ABIDECausalDataset('train'); \
print(f'Loaded {len(ds)} graphs, node features: {ds[0].x.shape}')"

# Verify graph structure
python -c "import torch; \
g = torch.load('data/processed/causal_graphs/Caltech_0051456_graph.pt'); \
print(f'Nodes: {g[\"adj\"].shape[0]}, Lobe order: {g[\"lobe_order\"]}')"
```

## Medical Context

- **Disorder**: Autism Spectrum Disorder (ASD)
- **Control**: Neurotypical development (TD)
- **Data Source**: ABIDE initiative (multi-site, n≈1000)
- **Modality**: resting-state fMRI (RS-fMRI)
- **Label Convention**: 0=Control, 1=ASD
- **Statistical Design**: 2D stratified by site + diagnosis (journal Q1 requirement)
- **Anatomical Framework**: AAL3v1 atlas (170 ROIs → 5 brain lobes)

## Key Design Decisions

### Why 5 Lobes Instead of 170 ROIs?
- **Computational**: Reduces graph from 170×170 (28,900 edges) to 5×5 (25 edges)
- **Interpretability**: Lobes have clear anatomical meaning for clinicians
- **Noise reduction**: Averaging within lobes reduces scanner-specific noise
- **Statistical power**: Fewer parameters = less overfitting on limited data (n~1000)

### Why GATv2 with 2 Heads?
- **Small graphs**: 5-node graphs don't benefit from excessive heads (4+ leads to redundancy)
- **Edge weights**: GAT naturally incorporates causal correlation weights via edge_attr
- **Attention**: Learns which lobe-lobe connections matter most for classification
- **Skip connections**: Prevent over-smoothing in shallow 5-node graphs

### Why Lagged Correlation Instead of Granger Causality?
- **Simplicity**: Pearson correlation is robust and interpretable
- **Temporal precedence**: Lag=1 TR enforces directionality (t-1 → t)
- **Scale**: Granger causality is expensive for 170×170 matrices; lagged correlation handles it
- **Sparsification**: Top 20% quantile keeps only strong connections