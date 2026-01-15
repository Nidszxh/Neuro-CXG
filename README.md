# Neuro-CXG

**Causal Graph Neural Networks for Brain Disorder Classification from fMRI**

A Graph Neural Network framework for brain disorder classification (ASD vs Control) using causal inference and explainable AI (XAI) on fMRI data. Combines YOLO-based ROI detection, causal graph construction, and Graph Attention Networks for interpretable neuroimaging analysis.

## Key Features

- **YOLO-based ROI Detection**: Automated detection of 5 brain anatomical lobes in 2D MRI slices
- **Causal Graph Construction**: Directed graphs from fMRI time series using lagged partial correlation
- **Graph Neural Networks**: GAT-based architecture for classification with interpretable edge weights
- **Batch Effect Harmonization**: neuroCombat integration for multi-site data harmonization
- **Stratified k-fold Validation**: Balanced by diagnosis and scanner site
- **Explainability**: Gradient-based node importance and edge weight analysis


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
│   ├── data/                      # Data processing modules
│   │   ├── extract_features.py    # YOLO inference → spatial features
│   │   ├── harmonize.py           # neuroCombat batch effect removal
│   │   ├── construct_causal.py    # Graph construction (lagged correlation)
│   │   ├── graph_factory.py       # PyTorch Geometric dataset loader
│   │   └── split.py               # Stratified splitting
│   ├── models/                    # GNN architecture and training
│   │   ├── causal_gnn.py          # GAT-based model (4 heads, skip connections)
│   │   └── gnn_model.py           # k-fold training loop
│   ├── pipelines/
│   │   └── roi_detection.py       # YOLO training entry point
│   ├── utils/                     # Utility functions
│   │   ├── manifest.py            # Manifest generation
│   │   └── compute_roi.py         # Temporal feature extraction
│   ├── test_suite.py              # Comprehensive pytest suite
│   └── test.py                    # Existing tests (legacy)
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
| LOBE_MAPPING | 170→5 | AAL3 atlas ROI aggregation |
| GNN_HIDDEN_CHANNELS | 64 | Hidden dimension for GATv2Conv |
| K_FOLDS | 5 | Cross-validation folds |
| YOLO_BATCH_SIZE | 24 |
| CAUSAL_LAG | 1 | Time lag for temporal precedence (TRs) |
| SPARSITY_QUANTILE | 0.80 | Keep top 20% causal connections |

See [src/config.py](src/config.py) for all 60+ parameters.

## Data Format

### Time Series Input
- **Shape**: `(timepoints, 170)` - fMRI signal from 170 AAL ROIs
- **Processing**: Bandpass filtered (0.01-0.08 Hz), z-normalized

### Graph Output (PyTorch Geometric Data)
```python
Data(
  x=torch.Tensor(5, 9),          # 5 lobes × (6 temporal + 3 spatial features)
  edge_index=torch.Tensor(2, K), # K directed edges
  edge_attr=torch.Tensor(K,),    # Causal correlation weights [-1, 1]
  y=torch.Tensor([0 or 1]),      # Label: 0=Control, 1=ASD
  pos=torch.Tensor(5, 3),        # 3D spatial coordinates
  sub_id=str                      # Subject identifier
)
```

## Validation & Testing

Run comprehensive test suite:

```bash
# All tests
pytest src/test_suite.py -v

# Configuration tests
pytest src/test_suite.py::TestConfiguration -v

# Data integrity tests
pytest src/test_suite.py::TestDataIntegrity -v

# Coverage report
pytest src/test_suite.py --cov=src --cov-report=html
```

## Medical Context

- **Disorder**: Autism Spectrum Disorder (ASD)
- **Control**: Neurotypical development (TD)
- **Data Source**: ABIDE initiative (multi-site, n≈1000)
- **Modality**: resting-state fMRI (RS-fMRI)
- **Label Convention**: 0=Control, 1=ASD
- **Statistical Design**: Balanced by site + diagnosis (journal Q1 requirement)