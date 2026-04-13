# Setup Guide

## Prerequisites
- OS: Linux or macOS recommended
- Python: 3.10+
- GPU: CUDA-capable GPU recommended for training (11GB+ VRAM preferred)
- RAM: 32GB recommended for full pipeline and harmonization stages

## 1) Clone and Create Environment
```bash
git clone <repo-url>
cd Neuro-CXG
python -m venv .venv
source .venv/bin/activate
```

## 2) Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 3) Verify Configuration and Environment
```bash
python -c "from src.core.config import validate_environment; validate_environment()"
```

## 4) Optional: Run Unit Tests
```bash
pytest tests/unit/
```

## 5) Optional: End-to-End Dry Run
```bash
python src/run_pipeline.py --dry-run
```

## Hardware Notes
- CPU-only training is possible but much slower.
- I/O-heavy stages (download, extraction) benefit from SSD storage.
- If memory pressure occurs, run stages separately instead of full auto mode.

## Common Setup Issues
1. Torch/CUDA mismatch
- Symptom: torch cannot detect GPU.
- Fix: reinstall torch build compatible with your CUDA runtime.

2. Missing atlas file
- Symptom: atlas validation fails.
- Fix: check data/raw/atlases and rerun atlas validation stage.

3. neuroHarmonize install issues
- Symptom: harmonization import error.
- Fix: reinstall requirements and ensure consistent Python environment.

## Reproducibility Checklist
- Keep requirements pinned.
- Use config constants from src/core/config.py rather than hardcoded values.
- Keep seeds fixed in training scripts.
