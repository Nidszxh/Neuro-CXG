#!/bin/bash
# Neuro-CXG Seed Verification Script
# Demonstrates reproducibility by running with multiple seeds
# Usage: ./verify_seeds.sh [--quick]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

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
AUC_VALUES=()

# Check checkpoint directory
CHECKPOINT_DIR="models/checkpoints"
mkdir -p "$CHECKPOINT_DIR"

for seed in "${SEEDS[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "Running with seed=$seed"
    echo "----------------------------------------"

    START=$(date +%s)

    # Run training + evaluation only (skip slow post-training stages)
    echo "Training with seed=$seed..."
    python3 src/run_pipeline.py --auto --seed="$seed" \
        --skip-download --skip-split --skip-yolo \
        --skip-visualizations --skip-graph-visualization \
        --skip-explainability --skip-result-analysis \
        --skip-subject-analysis --skip-ablations \
        --skip-paper-figures --skip-data-quality --skip-audit-check

    # Extract final test AUC from evaluation results
    EVAL_FILE="results/evaluation/comprehensive_results.json"
    if [[ -f "$EVAL_FILE" ]]; then
        AUC=$(python3 -c "import json; print(json.load(open('$EVAL_FILE'))['ensemble_metrics']['auc'])" 2>/dev/null || echo "N/A")
        RESULTS+=("seed=$seed: AUC=$AUC")
        if [[ "$AUC" != "N/A" ]]; then
            AUC_VALUES+=("$AUC")
        fi
    else
        RESULTS+=("seed=$seed: completed")
    fi

    END=$(date +%s)
    ELAPSED=$((END - START))
    echo "Seed $seed completed in ${ELAPSED}s"
done

echo ""
echo "=========================================="
echo "Results Summary"
echo "=========================================="
printf "  %s\n" "${RESULTS[@]}"
echo ""

echo "=========================================="
echo "Variance Analysis"
echo "=========================================="

if [[ ${#AUC_VALUES[@]} -gt 1 ]]; then
    # Calculate mean, std, variance using Python (requires numpy)
    python3 - <<END
import numpy as np
aucs = [$(printf "%s," "${AUC_VALUES[@]}")]
aucs = np.array(aucs)
mean = np.mean(aucs)
std = np.std(aucs, ddof=1)  # Sample standard deviation
var = np.var(aucs, ddof=1)   # Sample variance
print(f"  Seeds tested: {len(aucs)}")
print(f"  AUC values: {np.round(aucs, 4)}")
print(f"  Mean AUC: {mean:.4f}")
print(f"  Std Dev (sample): {std:.4f}")
print(f"  Variance (sample): {var:.6f}")
if var < 0.01:
    print("  ✅ Variance < 0.01 (excellent reproducibility)")
else:
    print("  ⚠️  Variance >= 0.01 (review seed propagation)")
END
else
    echo "  Insufficient AUC values to compute variance"
fi

echo ""
echo "Seed verification complete."