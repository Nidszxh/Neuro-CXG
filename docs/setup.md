# Setup

This guide covers installation, CLI usage, end-to-end workflow, and ABIDE data acquisition.

---
## Quick Start (From Zero to Training in <20 Minutes)

**Prerequisite**: Existing ABIDE data in `data/raw/` and existing train/val/test split.

```bash
# Step 1: Activate environment (or create one with Python 3.10+)
source ~/.zangestu/bin/activate

# Step 2: Verify environment (5 seconds)
python -c "from src.core.config import validate_environment; validate_environment()"

# Step 3: Run full pipeline with existing data (10-15 minutes)
python src/run_pipeline.py --auto --skip-download --skip-split

# Step 4: Verify training completed (should show 5 trained models)
ls models/checkpoints/best_model_fold*.pt

# Step 5: Run evaluation (<30 seconds)
python src/run_evaluation.py

# Expected: Test AUC ≈ 0.8819 [0.83, 0.93] (48ch/4hd/3L/0.33)
```

**What if I don't have data yet?** See Part D for ABIDE download (requires ~2 hours for full dataset).

---

## Part A — Installation

### Prerequisites

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| **OS** | Linux (Ubuntu 20.04+) | Linux |
| **Python** | 3.10 | 3.10+ |
| **RAM** | 24 GB | 64 GB |
| **GPU** | None (CPU OK) | CUDA 12.1, 8GB VRAM |
| **Disk** | 200 GB | 500 GB SSD |

### Numbered Install Steps

1. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   `requirements.txt` includes: PyTorch/torchvision/torchaudio, torch-geometric, nilearn/nibabel/neuroHarmonize, ultralytics (YOLO), scikit-learn/scipy/statsmodels, captum (explainability), pytest/pytest-cov

3. **Validate runtime environment**
   ```bash
   python -c "from src.core.config import validate_environment; validate_environment()"
   ```
   This validates core paths and lobe mapping invariants before pipeline execution.

4. **Run a dry pipeline plan**
   ```bash
   python src/run_pipeline.py --dry-run
   ```
   Dry-run confirms stage planning logic, skip decisions, and current artifact readiness without executing stages.

5. **Optional fast sanity checks**
   ```bash
   pytest tests/unit/
   python -m src.features.extract_temporal --n-jobs 1 --no-frequency
   ```

### Troubleshooting Table

| Symptom | Cause | Fix |
|---------|-------|-----|
| `torch.cuda.is_available()` returns false | CUDA not properly configured | Verify NVIDIA driver and CUDA runtime; ensure active environment has expected torch build; continue on CPU for functional testing if needed |
| Training fails with missing harmonized fold files | Harmonization stage not run | Run `python -m src.features.fold_safe_harmonization`; verify files exist under `data/metadata/harmonized_folds_cv/` |
| Atlas validation or split/manifests fail | Missing input files | Verify `data/raw/atlases/AAL3v1.nii` and `data/processed/Phenotypic_V1_0b_preprocessed1.csv` exist; rerun split/manifest stages through orchestrator |
| YOLO weights not found for spatial extraction | YOLO training stage not completed | Run label generation and ROI training stages, or switch to atlas-based spatial extraction path for debugging |

### Reproducibility Rules

- Import constants from `src/core/config.py`; avoid hardcoded paths and dimensions
- Keep `FEATURE_GROUPS`, `ALL_FEATURE_NAMES`, and `GNN_IN_CHANNELS` in sync when changing features
- Keep fold-safe behavior intact (train-fit/apply discipline in harmonization and training)

---

## Part B — CLI Usage

### Primary Entry Point

Use the orchestrator for most workflows:

```bash
python src/run_pipeline.py
```

By default, `run_pipeline.py` is interactive. Use `--auto` for non-interactive execution.

### Run Mode Table

| Mode | Command | What It Does | Expected Outputs |
|------|---------|--------------|------------------|
| Dry-run | `python src/run_pipeline.py --dry-run` | Show stage plan without execution | Stage plan printout |
| Full non-interactive | `python src/run_pipeline.py --auto` | Execute all stages | All stage artifacts |
| Reuse data | `python src/run_pipeline.py --auto --skip-download --skip-split` | Skip download/split stages | Feature → graph → train → report |
| Analysis only | `python src/run_pipeline.py --analysis-only` | Run reporting stages only | Evaluation/explainability results |

### High-Impact Flags

| Flag | Effect | When To Use |
|------|--------|-------------|
| `--auto` | Run all stages non-interactively | Production runs |
| `--skip-download` | Skip ABIDE download | Already have data |
| `--skip-split` | Skip train/val/test split | Reuse existing split |
| `--skip-yolo` | Skip YOLO training (ROI detection stage 9) | Already have YOLO weights or iterating on later stages |
| `--multiview` | Generate multiview graphs | Invariance experiments |
| `--site-stratified-cv` | Regenerate cv_fold by site | Cross-site robustness |
| `--force-reset` | Clear intermediates | Full rebuild |
| `--analysis-only` | Run reporting only | With existing checkpoints |
| `--regenerate-features` | Rebuild feature/graph without full reset | Fast iteration |
| `--visualizations-only` | Run visualization stage only | Generate plots |
| `--skip-evaluation` | Skip evaluation stage | Faster runs |
| `--skip-explainability` | Skip explainability stage | Faster runs |

### Stage-Level Script Commands

**Data:**
```bash
python -m src.data.abide_download
python -m src.data.split
python -m src.data.split --site-stratified-cv
```

**Labels and detection:**
```bash
python -m src.detection.generate_labels
python -m src.detection.roi_detection
```

**Features and harmonization:**
```bash
python -m src.features.extract_spatial
python -m src.features.extract_spatial_atlas
python -m src.features.extract_temporal --n-jobs -1
python -m src.features.fold_safe_harmonization
```

**Graph construction:**
```bash
python -m src.features.construct_causal --n-jobs -1
python -m src.features.construct_causal --multiview
```

**Training:**
```bash
python -m src.models.gnn_model
```

**Post-training reporting:**
```bash
python src/run_evaluation.py
python src/run_explainability.py
python src/run_result_analysis.py
```

### Script-Specific Options

#### `run_evaluation.py`

| Flag | Effect |
|------|--------|
| `--no-permutation` | Skip permutation test |
| `--n-permutations 200` | Set permutation count |
| `--no-baselines` | Skip baseline comparisons |
| `--no-subgroups` | Skip subgroup analysis |
| `--output-dir results/evaluation_custom` | Custom output directory |

#### `run_explainability.py`

| Flag | Effect |
|------|--------|
| `--fold 2` | Use specific fold for explainability |
| `--phases node edge` | Run only node and edge phases |
| `--no-masking` | Skip edge masking |
| `--output-dir results/explainability_custom` | Custom output directory |

#### `run_result_analysis.py`

| Flag | Effect |
|------|--------|
| `--n-cases 3` | Number of case studies |
| `--no-heatmap` | Skip heatmap generation |
| `--no-severity` | Skip severity analysis |
| `--output-dir results/analysis_custom` | Custom output directory |

### Expected Output Artifacts Per Stage

| Stage | Output Path | Format | Description |
|-------|-------------|--------|-------------|
| 1 download | `data/metadata/download_log.csv` | CSV | Download status per subject |
| 2 split | `data/final/{train,val,test}/` | dir | Split time series and images |
| 3 manifest | `data/metadata/master_manifest.csv` | CSV | Subject metadata with cv_fold |
| 4 site_stratified_cv | `data/metadata/master_manifest.csv` | CSV | Site-stratified fold assignment |
| 5 atlas_validation | `data/metadata/atlas_metadata.json` | JSON | Atlas validation report |
| 6 post_download_integrity | — | log | Download integrity check |
| 7 annotate | `data/processed/final_train/labels/` | txt | YOLO labels |
| 8 yolo | `results/experiments/detection/ROI_Detection_v29/weights/best.pt` | .pt | Trained YOLO weights |
| 9 spatial_features | `data/metadata/node_features_3d.csv` | CSV | 3D coordinates per lobe |
| 10 temporal_features | `data/metadata/node_attributes_temporal.csv` | CSV | Temporal features per ROI |
| 11 harmonization | `data/metadata/node_attributes_harmonized.csv` | CSV | ComBat-harmonized features |
| 12 pre_gnn_integrity | — | log | Pre-GNN validation |
| 13 causal_graphs | `data/processed/causal_graphs/*.pt` | .pt | Per-subject directed graphs |
| 14 multiview_graphs | `data/processed/causal_graphs_multiview/` | dir | Optional multiview graphs |
| 15 diagnostics | — | log | Pipeline health report |
| 16 quality_validation | — | log | Quality gates check |
| 17 gnn_training | `models/checkpoints/best_model_fold*.pt` | .pt | 5-fold model checkpoints |
| 18 visualizations | `results/visualizations/` | dir | Training/feature plots |
| 19 graph_visualization | `results/visualizations/causal_graph_comparison.png` | PNG | Causal graph comparison |
| 20 circular_connectome | `results/paper_figures/circular_connectome_ASD.png` | PNG | Connectome ring visualization |
| 21 brain_3d_visualization | `results/paper_figures/brain_3d/` | dir | Nilearn 3D brain rendering |
| 22 evaluation | `results/evaluation/comprehensive_results.json` | JSON | Test metrics, bootstrap CI |
| 23 explainability | `results/explainability/summary.json` | JSON | Node/edge importance |
| 24 result_analysis | `results/analysis/result_analysis_summary.json` | JSON | Per-subject predictions |
| 25 subject_analysis | `results/subject_analysis/` | CSV/txt | Per-subject diagnostics |

See `docs/architecture.md` for full data flow.

### Typical Workflow Patterns

**Full rebuild:**
```bash
python src/run_pipeline.py --auto --force-reset
```

**Analysis-only (with existing checkpoints):**
```bash
python src/run_pipeline.py --analysis-only
```

**Multiview (invariance experiments):**
```bash
python src/run_pipeline.py --auto --multiview
```

**Site-stratified CV:**
```bash
python -m src.data.split --site-stratified-cv
python -m src.features.fold_safe_harmonization
python -m src.models.gnn_model
```

---

## Part C — End-to-End Walkthrough

### Numbered Sequence with Verification Checkpoints

**Step 0: Preflight**
```bash
pip install -r requirements.txt
python -c "from src.core.config import validate_environment; validate_environment()"
python src/run_pipeline.py --dry-run
```
Expected: Environment validation passes, dry-run prints stage plan without execution.

**Step 1: Build Data, Features, Graphs, and Train**
```bash
python src/run_pipeline.py --auto
```
Or faster iteration (reuse existing download/split):
```bash
python src/run_pipeline.py --auto --skip-download --skip-split
```

**Step 2: Checkpoint Verification**
```bash
ls data/metadata/master_manifest.csv
ls data/metadata/node_attributes_harmonized.csv
ls data/metadata/harmonized_folds_cv/harmonized_fold_0.csv
ls data/processed/causal_graphs | head
ls models/checkpoints/best_model_fold0.pt
```
Expected: Manifest and harmonized metadata present, per-subject graph files present, fold checkpoints present.

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
Expected: All three summary JSON artifacts exist.

### Optional Branch Commands

**Site-Stratified CV Protocol:**
```bash
python -m src.data.split --site-stratified-cv
python -m src.features.fold_safe_harmonization
python -m src.models.gnn_model
```

**Multiview Graph Path:**
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

# Check training prerequisites
python -c "from src.core.config import validate_gnn_training_inputs; validate_gnn_training_inputs()"

# Validate pipeline
python -m src.validation.pipeline_checks
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

## Part D — ABIDE Data Acquisition

### Primary S3 Download Method

**Bucket:** `indiana-public-data` (public access, no credentials required)  
**Region:** `us-east-1`  
**Prefix:** `ABIDE_Initiative/`

```bash
python -m src.data.abide_download
```

This script:
- Connects to the public S3 bucket
- Downloads preprocessed fMRI time series and phenotype CSV
- Handles network retries and partial downloads
- Validates downloaded artifacts

Total size: ~150GB compressed.

### Fallback Methods Table

| Method | Steps | When To Use |
|--------|-------|-------------|
| Pre-downloaded dataset | Use `--skip-download` flag | Already have ABIDE data locally |
| Manual download from OpenNeuro | 1. Go to https://openneuro.org/datasets/ds000031 2. Download ~16GB zipped 3. Extract to `data/abide/` 4. Update config in `src/core/config.py` | S3 access issues |
| Pre-processed time series only | Request from ABIDE-initiative.org, use PCA/FV extraction, place in `data/processed/` | Only need time-series data |
| GridEngine batch job | Submit download as batch job | Large-scale distributed runs |
| rsync | `rsync -avz user@server:/path/ data/raw/` | Local network copy |

### Dataset Version Control

**CPAC Version Used:** CPAC 1.0 (filt_global preprocessing)

**Preprocessing Details:**
- Bandpass filtering: 0.01–0.1 Hz
- Global signal regression: Yes
- Spatial normalization: MNI152

**Phenotype File:**
- Version: 1.0b
- File: `Phenotypic_V1_0b_preprocessed1.csv`
- ⚠️ Checksum verification: **NOT IMPLEMENTED** — pending implementation in `abide_download.py`

### Post-Download Verification

```bash
python -c "from src.core.config import validate_environment; validate_environment()"
python src/run_pipeline.py --dry-run
```

Verify that download stage shows as complete in dry-run output.

### Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| S3 access denied | IP blocked or network restriction | Check IP not blocked; try different network; use VPN if behind corporate firewall |
| Slow downloads | Network bandwidth or regional latency | Use `--n-jobs` to parallelize downloads; consider regional AWS endpoints |
| Data corruption | Incomplete download or network issue | Re-run download stage with `--force-reset`; verify MD5 checksums if available; use manual download fallback |

### Checksum Generation (Future)

```bash
# After downloading phenotype CSV
md5sum data/raw/Phenotypic_V1_0b_preprocessed1.csv
```

To verify programmatically:
```python
import hashlib

def verify_phenotype_checksum(csv_path, expected_md5):
    md5 = hashlib.md5(csv_path.read_bytes()).hexdigest()
    return md5 == expected_md5
```

⚠️ **Action Required:** Generate MD5 checksum after first clean download and update `EXPECTED_PHENO_CHECKSUMS` in `abide_download.py`.