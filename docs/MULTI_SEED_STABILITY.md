# Multi-Seed Stability Analysis: Framework & Protocol

**Status**: Framework for multi-seed evaluation  
**Date**: April 29, 2026  
**Purpose**: Document how to assess stability across random seeds

---

## Motivation

Machine learning models can exhibit significant variance due to:
1. Random weight initialization
2. Random data shuffling within batches
3. Random dropout patterns
4. Random site/subject sampling in validation

To ensure Neuro-CXG results are reproducible and not artifacts of random chance, we recommend evaluating on **3 different seeds**: **42** (canonical), **123**, and **456**.

---

## Current Single-Seed Performance (Seed=42)

| Metric | Value | Notes |
|--------|-------|-------|
| CV AUC (5-fold) | 0.8104 ± 0.0301 | Current canonical |
| Test AUC | 0.8413 | Ridge Granger |
| Fold Variance | 0.0301 | Stable across folds |

---

## Multi-Seed Protocol

### Step 1: Run Pipeline with Different Seeds

```bash
# Seed 42 (canonical - already done)
python src/run_pipeline.py --auto --seed 42

# Seed 123
python src/run_pipeline.py --auto --seed 123

# Seed 456
python src/run_pipeline.py --auto --seed 456
```

### Step 2: Collect Results

After each run, extract metrics:

```bash
# Extract CV metrics (mean ± std across 5 folds)
python -c "
import json
for seed in [42, 123, 456]:
    with open(f'results/seed_{seed}/evaluation/comprehensive_results.json') as f:
        data = json.load(f)
        print(f'Seed {seed}: Test AUC = {data[\"ensemble_metrics\"][\"auc\"]:.4f}')
"
```

### Step 3: Compute Statistics

```bash
python src/analysis/multi_seed_analysis.py
```

---

## Expected Results (Based on Single-Seed Analysis)

### Predicted Performance Range

Based on the fold variance observed in seed=42 (CV std 0.0301), we expect:

| Seed | Expected Test AUC | 95% CI |
|------|-------------------|--------|
| 42 | 0.8413 (observed) | [0.7759, 0.8976] |
| 123 | 0.83–0.86 | [0.77, 0.90] |
| 456 | 0.83–0.86 | [0.77, 0.90] |

### Predicted Multi-Seed Statistics

| Statistic | Expected Value |
|-----------|----------------|
| **Mean Test AUC** | 0.84–0.85 |
| **Std Test AUC** | 0.01–0.02 |
| **Min Test AUC** | ~0.82 |
| **Max Test AUC** | ~0.86 |
| **Range** | ~0.04 |

### Stability Classification

| Stability Level | Std Range | Status |
|-----------------|-----------|--------|
| **Excellent** | < 0.01 | Expected for ridge_granger |
| **Good** | 0.01–0.02 | Likely |
| **Acceptable** | 0.02–0.03 | Possible |
| **Poor** | > 0.03 | Unlikely |

---

## Multi-Seed Analysis Script

Create `src/analysis/multi_seed_analysis.py`:

```python
"""
Multi-Seed Stability Analysis

Analyzes performance across multiple random seeds to assess
reproducibility and stability of Neuro-CXG results.
"""
import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import SEEDS_TO_TEST

def load_results(seed: int) -> dict:
    """Load evaluation results for a given seed."""
    eval_dir = Path(f"results/evaluation_seed_{seed}")
    if not eval_dir.exists():
        return None
    
    with open(eval_dir / "comprehensive_results.json") as f:
        return json.load(f)

def compute_statistics(results: list) -> dict:
    """Compute mean, std, min, max across seeds."""
    aucs = [r["ensemble_metrics"]["auc"] for r in results if r]
    
    return {
        "n_seeds": len(aucs),
        "mean_auc": np.mean(aucs),
        "std_auc": np.std(aucs),
        "min_auc": np.min(aucs),
        "max_auc": np.max(aucs),
        "range_auc": np.max(aucs) - np.min(aucs),
    }

def main():
    seeds = [42, 123, 456]
    results = []
    
    for seed in seeds:
        r = load_results(seed)
        if r:
            results.append(r)
            print(f"Seed {seed}: Test AUC = {r['ensemble_metrics']['auc']:.4f}")
        else:
            print(f"Seed {seed}: No results found")
    
    if results:
        stats = compute_statistics(results)
        print(f"\n=== Multi-Seed Statistics ===")
        print(f"Mean AUC: {stats['mean_auc']:.4f} ± {stats['std_auc']:.4f}")
        print(f"Range: [{stats['min_auc']:.4f}, {stats['max_auc']:.4f}]")
        
        # Classification
        if stats['std_auc'] < 0.01:
            stability = "EXCELLENT"
        elif stats['std_auc'] < 0.02:
            stability = "GOOD"
        elif stats['std_auc'] < 0.03:
            stability = "ACCEPTABLE"
        else:
            stability = "POOR"
        
        print(f"Stability: {stability}")

if __name__ == "__main__":
    main()
```

---

## Running Multi-Seed Analysis

### Full Evaluation (Takes 6–12 hours per seed)

```bash
# Run all 3 seeds
for seed in 42 123 456; do
    python src/run_pipeline.py --auto --seed $seed \
        --output-dir results/seed_$seed
done

# Analyze
python src/analysis/multi_seed_analysis.py
```

### Quick Check (Using Existing Checkpoints)

If only seed=42 has been run, use bootstrap resampling to estimate variance:

```python
# Bootstrap on seed=42 results
from sklearn.utils import resample
# Resample test subjects n_bootstrap times
# Compute AUC distribution
# Estimate expected std across seeds
```

---

## Interpretation Guidelines

### What to Look For

1. **Mean AUC**: Should be ~0.84 (within 0.02 of seed=42)
2. **Std AUC**: Should be < 0.02 for stable results
3. **Range**: Should be < 0.04 (not too variable)

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| High variance (>0.03) | Random seed affects site split | Use `--site-stratified-cv` |
| Very low AUC with new seed | Data loading bug | Verify data integrity |
| AUC increases with seed | Test set composition | Report full distribution |

### Reporting Format

```
Multi-Seed Stability Analysis (Neuro-CXG, Ridge Granger)
============================================================
Seeds tested: 42, 123, 456
Mean Test AUC: 0.8420 ± 0.0150
Range: [0.8300, 0.8550]
Stability: GOOD

Interpretation: Results are stable across random seeds.
The model achieves consistent performance regardless of
random initialization, indicating robust generalization.
```

---

## CI Configuration for Multi-Seed

Add multi-seed stability to `.github/workflows/tests.yml`:

```yaml
multi-seed-test:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    
    - name: Run multi-seed analysis
      run: |
        # Only run if checkpoints exist
        if [ -d "models/checkpoints" ]; then
          python src/analysis/multi_seed_analysis.py
        fi
    
    - name: Upload results
      uses: actions/upload-artifact@v4
      with:
        name: multi-seed-results
        path: results/multi_seed/
```

---

## Summary

| Aspect | Status | Notes |
|--------|--------|-------|
| Seed 42 (canonical) | ✅ Complete | Test AUC 0.8413 |
| Seed 123 | ⏳ Pending | Run to verify stability |
| Seed 456 | ⏳ Pending | Run to verify stability |
| Expected std | ~0.015 | Based on fold variance |
| Target | AUC > 0.80 across all seeds | Publication requirement |

**Recommendation**: Run seeds 123 and 456 to confirm stability before final submission. If std > 0.03, investigate causes (likely data splitting or initialization).

---

## References

- Current results: `results/evaluation_rg/comprehensive_results.json`
- Configuration: `src/core/config.py` (SEEDS_TO_TEST)
- Analysis script: `src/analysis/multi_seed_analysis.py` (to be created)