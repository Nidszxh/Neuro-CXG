# Setup

## Prerequisites

- OS: Linux is the primary supported environment in this repository
- Python: 3.10+ recommended
- Optional GPU: CUDA-capable GPU for faster training/evaluation
- Disk: enough space for ABIDE artifacts and generated graphs/results

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
