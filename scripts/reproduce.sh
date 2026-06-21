#!/bin/bash
# Neuro-CXG Reproducibility Script
# Usage: ./reproduce.sh [--skip-download]

set -euo pipefail

cd "$(dirname "$0")/.."

echo "Neuro-CXG Reproducibility Script"
echo "=============================="

# Hardware: 32GB RAM, 8GB VRAM, CUDA 12.1+
# Wall-clock: 6-12 hours full pipeline

# ---- Environment Validation ----
echo "Validating environment..."

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python version: $PYTHON_VERSION"
if [[ ! "$PYTHON_VERSION" =~ ^3\.(10|11|12)$ ]]; then
    echo "WARNING: Python 3.10-3.12 recommended"
fi

# Check CUDA
if python3 -c "import torch" 2>/dev/null; then
    CUDA_AVAIL=$(python3 -c "import torch; print(torch.cuda.is_available())")
    if [[ "$CUDA_AVAIL" == "True" ]]; then
        echo "CUDA: $(python3 -c "import torch; print(torch.cuda.get_device_name(0))")"
    else
        echo "WARNING: CUDA not available - will use CPU (slow)"
    fi
else
    echo "WARNING: PyTorch not installed"
fi

# Validate config
echo "Validating config..."
python3 -c "from src.core.config import validate_environment" || {
    echo "ERROR: Environment validation failed"
    exit 1
}

# ---- Run Pipeline ----
echo "Running pipeline..."
SKIP_DOWNLOAD=false
SKIP_SPLIT=false
for arg in "$@"; do
    case $arg in
        --skip-download) SKIP_DOWNLOAD=true ;;
        --skip-split) SKIP_SPLIT=true ;;
        -h|--help)
            echo "Usage: $0 [--skip-download] [--skip-split]"
            exit 0
            ;;
    esac
done

PIPELINE_ARGS="--auto"
if [[ "$SKIP_DOWNLOAD" == true ]]; then
    PIPELINE_ARGS="$PIPELINE_ARGS --skip-download"
    echo "Skipping download..."
fi
if [[ "$SKIP_SPLIT" == true ]]; then
    PIPELINE_ARGS="$PIPELINE_ARGS --skip-split"
    echo "Skipping split..."
fi
python3 src/run_pipeline.py $PIPELINE_ARGS

echo "Done!"
echo ""
echo "Verify results in: results/evaluation/comprehensive_results.json"
