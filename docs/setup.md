# Setup

## Prerequisites

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| **OS** | Linux (Ubuntu 20.04+) | Linux |
| **Python** | 3.10 | 3.10+ |
| **RAM** | 24 GB | 64 GB |
| **GPU** | None (CPU OK) | CUDA 12.1, 8GB VRAM |
| **Disk** | 200 GB | 500 GB SSD |

## 1) Create And Activate A Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

## 2) Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` is pinned and includes:

- PyTorch / torchvision / torchaudio
- torch-geometric
- nilearn / nibabel / neuroHarmonize
- ultralytics (YOLO)
- scikit-learn / scipy / statsmodels
- captum (explainability)
- pytest / pytest-cov

## 3) Validate The Runtime Environment

```bash
python -c "from src.core.config import validate_environment; validate_environment()"
```

This check validates core paths and lobe mapping invariants before pipeline execution.

## 4) Run A Dry Pipeline Plan

```bash
python src/run_pipeline.py --dry-run
```

Dry-run confirms stage planning logic, skip decisions, and current artifact readiness without executing stages.

## 5) Optional Fast Sanity Checks

```bash
pytest tests/unit/
python -m src.features.extract_temporal --n-jobs 1 --no-frequency
```

## First Real Run Patterns

```bash
# Full non-interactive execution
python src/run_pipeline.py --auto

# Reuse existing download/split artifacts
python src/run_pipeline.py --auto --skip-download --skip-split

# Analysis stages only (requires trained checkpoints)
python src/run_pipeline.py --analysis-only
```

## Troubleshooting

### 1) CUDA not detected

Symptom:

- `torch.cuda.is_available()` is false

Actions:

- verify NVIDIA driver and CUDA runtime
- ensure the active environment has the expected torch build
- continue on CPU for functional testing if needed

### 2) Missing training prerequisites

Symptom:

- training fails with missing harmonized fold files

Actions:

- run harmonization stage:

```bash
python -m src.features.fold_safe_harmonization
```

- verify files exist under `data/metadata/harmonized_folds_cv/`

### 3) Atlas or manifest errors

Symptom:

- atlas validation or split/manifests fail

Actions:

- verify `data/raw/atlases/AAL3v1.nii`
- verify `data/processed/Phenotypic_V1_0b_preprocessed1.csv`
- rerun split/manifest stages through the orchestrator

### 4) YOLO weights not found for spatial extraction

Symptom:

- `extract_spatial.py` reports missing model weights

Actions:

- run label generation and ROI training stages
- or switch to atlas-based spatial extraction path for debugging

## Reproducibility Rules

- Import constants from `src/core/config.py`; avoid hardcoded paths and dimensions.
- Keep `FEATURE_GROUPS`, `ALL_FEATURE_NAMES`, and `GNN_IN_CHANNELS` in sync when changing features.
- Keep fold-safe behavior intact (train-fit/apply discipline in harmonization and training).

---

## Usage

### Primary Entry Point

Use the orchestrator for most workflows:

```bash
python src/run_pipeline.py
```

By default, `run_pipeline.py` is interactive. Use `--auto` for non-interactive execution.

### Core Pipeline Commands

```bash
# Show stage plan only
python src/run_pipeline.py --dry-run

# Full non-interactive run
python src/run_pipeline.py --auto

# Reuse existing downloaded and split data
python src/run_pipeline.py --auto --skip-download --skip-split

# Run only post-training analysis stages
python src/run_pipeline.py --analysis-only
```

### High-Impact Flags

| Flag | What It Changes | When To Use |
|------|-----------------|-------------|
| `--auto` | Run all stages non-interactively | Production runs |
| `--skip-download` | Skip ABIDE download | Already have data |
| `--skip-split` | Skip train/val/test split | Reuse existing split |
| `--multiview` | Generate multiview graphs | Invariance experiments |
| `--site-stratified-cv` | Regenerate cv_fold by site | Cross-site robustness |
| `--force-reset` | Clear intermediates | Full rebuild |
| `--analysis-only` | Run reporting only | With existing checkpoints |

### Expected Output Artifacts Per Stage

| Stage | Output Path | Format | Description |
|-------|-------------|--------|-------------|
| download | `data/metadata/download_log.csv` | CSV | Download status per subject |
| split | `data/final/{train,val,test}/` | dir | Split time series and images |
| manifest | `data/metadata/master_manifest.csv` | CSV | Subject metadata with cv_fold |
| yolo | `models/checkpoints/best.pt` | .pt | Trained YOLO weights |
| spatial_features | `data/metadata/node_features_3d.csv` | CSV | 3D coordinates per lobe |
| temporal_features | `data/metadata/node_attributes_temporal.csv` | CSV | Temporal features per ROI |
| harmonization | `data/metadata/node_attributes_harmonized.csv` | CSV | ComBat-harmonized features |
| causal_graphs | `data/processed/causal_graphs/*.pt` | .pt | Per-subject directed graphs |
| gnn_training | `models/checkpoints/best_model_fold*.pt` | .pt | 5-fold model checkpoints |
| evaluation | `results/evaluation/comprehensive_results.json` | JSON | Test metrics, bootstrap CI |
| explainability | `results/explainability/summary.json` | JSON | Node/edge importance |

### Stage-Level Script Commands

```bash
# Data
python -m src.data.abide_download
python -m src.data.split

# Labels and detection
python -m src.pipelines.generate_labels
python -m src.pipelines.roi_detection

# Features and harmonization
python -m src.features.extract_spatial
python -m src.features.extract_temporal --n-jobs -1
python -m src.features.fold_safe_harmonization

# Graph construction
python -m src.features.construct_causal --n-jobs -1
python -m src.features.construct_causal --multiview

# Training
python -m src.models.gnn_model
```

### Post-Training Commands

```bash
python src/run_evaluation.py
python src/run_explainability.py
python src/run_result_analysis.py
```

### Script-Specific Options

#### Evaluation

```bash
python src/run_evaluation.py --no-permutation
python src/run_evaluation.py --n-permutations 200
python src/run_evaluation.py --no-baselines --no-subgroups
python src/run_evaluation.py --output-dir results/evaluation_custom
```

#### Explainability

```bash
python src/run_explainability.py --fold 2
python src/run_explainability.py --phases node edge
python src/run_explainability.py --no-masking
python src/run_explainability.py --output-dir results/explainability_custom
```

#### Result Analysis

```bash
python src/run_result_analysis.py --n-cases 3
python src/run_result_analysis.py --no-heatmap
python src/run_result_analysis.py --no-severity
python src/run_result_analysis.py --output-dir results/analysis_custom
```

### Typical Workflows

```bash
# Full rebuild from current workspace state
python src/run_pipeline.py --auto --force-reset

# Recompute only analysis from existing checkpoints
python src/run_pipeline.py --analysis-only

# Enable multiview artifacts for invariance experiments
python src/run_pipeline.py --auto --multiview
```

---

## Walkthrough

### End-to-End Sequence

**Step 0: Preflight**

```bash
pip install -r requirements.txt
python -c "from src.core.config import validate_environment; validate_environment()"
python src/run_pipeline.py --dry-run
```

Expected: environment validation passes, dry-run prints stage plan without execution.

**Step 1: Build Data, Features, Graphs, and Train**

```bash
# Standard non-interactive run
python src/run_pipeline.py --auto

# Faster iteration (reuse existing download/split)
python src/run_pipeline.py --auto --skip-download --skip-split
```

**Step 2: Checkpoint Verification**

Verify core artifacts exist:

```bash
ls data/metadata/master_manifest.csv
ls data/metadata/node_attributes_harmonized.csv
ls data/metadata/harmonized_folds_cv/harmonized_fold_0.csv
ls data/processed/causal_graphs | head
ls models/checkpoints/best_model_fold0.pt
```

Expected: manifest and harmonized metadata present, per-subject graph files present, fold checkpoints present.

**Step 3: Run Post-Training Reports**

```bash
python src/run_evaluation.py
python src/run_explainability.py
python src/run_result_analysis.py
```

Or via orchestrator:

```bash
python src/run_pipeline.py --analysis-only
```

**Step 4: Verify Reporting Artifacts**

```bash
ls results/evaluation/comprehensive_results.json
ls results/explainability/summary.json
ls results/analysis/result_analysis_summary.json
```

Expected: all three summary JSON artifacts exist.

### Optional Branches

#### Site-Stratified CV Protocol

```bash
python -m src.data.split --site-stratified-cv
python -m src.features.fold_safe_harmonization
python -m src.models.gnn_model
```

#### Multiview Graph Path

```bash
python src/run_pipeline.py --auto --multiview
```

### Fast Debug Commands

```bash
# Rebuild feature/graph stack without full reset
python src/run_pipeline.py --auto --regenerate-features

# Skip expensive evaluation permutations
python src/run_evaluation.py --no-permutation

# Skip slow edge masking
python src/run_explainability.py --no-masking
```

### Minimal Reproducible Command Set

For a compact reproducible run with existing data:

```bash
python src/run_pipeline.py --auto --skip-download --skip-split
python src/run_evaluation.py
python src/run_explainability.py --no-masking
python src/run_result_analysis.py
```

---

## ABIDE Data Acquisition

### Primary Download Method

ABIDE I data is hosted on the FCP-INDI S3 bucket:

```bash
python -m src.data.abide_download
```

This script:
- Connects to fcp-indi bucket: `s3://fcp-indi/resources/project-specific/ABIDE/`
- Downloads preprocessed fMRI time series and phenotype CSV
- Handles network retries and partial downloads
- Validates downloaded artifacts

### Fallback Methods

| Method | Steps | When To Use |
|--------|-------|-------------|
| Manual download | Download from http://fcon_1000.projects.nitrc.org/indi/abide.html | S3 access issues |
| rsync | `rsync -avz user@server:/path/ data/raw/` | Local network copy |
| GridEngine | Submit download as batch job | Large-scale distributed runs |

### Post-Download Verification

```bash
python -c "from src.core.config import validate_environment; validate_environment()"
python src/run_pipeline.py --dry-run
```

Verify that download stage shows as complete in dry-run output.

### Troubleshooting

#### S3 Access Denied

- Verify AWS credentials configured: `aws configure`
- Check bucket public access policy
- Try manual download from Nitrc as fallback

#### Slow Downloads

- Use `--n-jobs` to parallelize downloads
- Consider regional S3 endpoints
- Check network bandwidth

#### Data Corruption

- Re-run download stage with `--force-reset`
- Verify MD5 checksums if available
- Use manual download fallback
