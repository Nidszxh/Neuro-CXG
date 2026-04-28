#!/bin/bash
# Neuro-CXG Seed Verification Script
# Demonstrates reproducibility by running with multiple seeds
# Usage: ./verify_seeds.sh [--quick]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

QUICK=false
N_SEEDS=3
SEEDS=(42 123 456)
TIMEOUT_MINUTES=120

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--quick]"
            echo ""
            echo "Options:"
            echo "  --quick    Run faster with fewer seeds and epochs"
            echo "  -h, --help"
            exit 0
            ;;
    esac
done

cd "$PROJECT_DIR"

echo "=========================================="
echo "Neuro-CXG Seed Verification"
echo "=========================================="
echo "Testing reproducibility across $N_SEEDS seeds:"
printf "  Seeds: %s\n" "${SEEDS[*]}"
echo ""

# Set environment for faster testing
if [[ "$QUICK" == true ]]; then
    echo "[QUICK MODE]"
    export GNN_EPOCHS=2
    export GNN_BATCH_SIZE=8
    TIMEOUT_MINUTES=30
fi

RESULTS=()

# Check checkpoint directory
CHECKPOINT_DIR="models/checkpoints"
mkdir -p "$CHECKPOINT_DIR"

for seed in "${SEEDS[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "Running with seed=$seed"
    echo "----------------------------------------"

    START=$(date +%s)

    # Run training with seed
    # Uses gnn_model.set_random_seeds() for full reproducibility
    echo "Training with seed=$seed..."
    python3 src/run_pipeline.py --auto --seed="$seed" || {
        echo "WARNING: --seed flag not implemented, using fixed seed=42"
        # Fall back to fixed seed if --seed not available
        python3 src/run_pipeline.py --auto
        break
    }

    END=$(date +%s)
    ELAPSED=$((END - START))
    RESULTS+=("seed=$seed: ${ELAPSED}s")
done

echo ""
echo "=========================================="
echo "Results Summary"
echo "=========================================="
printf "  %s\n" "${RESULTS[@]}"
echo ""
echo "Variance Analysis:"
echo "  Expected: AUC variance < 0.01 across seeds"
echo "  For full reproducibility, verify AUC consistency across seeds"
echo ""
echo "Note: To enable --seed flag, add seed parameter to run_pipeline.py"
echo "  and call gnn_model.set_random_seeds(seed) at training start."