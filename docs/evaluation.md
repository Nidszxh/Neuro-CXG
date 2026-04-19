# GPU-Accelerated Granger Causality Testing

## Overview
`tests/unit/test_granger_gpu.py` provides a comprehensive test suite for GPU-accelerated Granger causality computation, verifying numerical equivalence between CPU and GPU implementations and benchmarking performance.

---

## Purpose
Granger causality testing is a computational bottleneck in the Neuro-CXG pipeline:
- **Baseline (CPU)**: ~42 minutes for all subjects with multiview construction
- **GPU-accelerated**: 5–10x speedup with proper batching

This test suite validates that GPU acceleration:
1. Produces numerically equivalent results to CPU implementation
2. Handles edge cases (NaN, Inf, short time series)
3. Achieves claimed performance improvements
4. Falls back gracefully on errors

---

## Running Tests

### Single Test Suite
```bash
python tests/unit/test_granger_gpu.py
```

### Individual Tests
```bash
# Synthetic causal signal
python -c "from tests.unit.test_granger_gpu import test_synthetic_causal_signal; test_synthetic_causal_signal()"

# Speed benchmark
python -c "from tests.unit.test_granger_gpu import benchmark_speed; benchmark_speed()"
```

### With Pytest Integration
```bash
pytest tests/unit/test_granger_gpu.py -v
pytest tests/unit/test_granger_gpu.py::test_synthetic_causal_signal -v
```

---

## Test Cases

### 1. Synthetic Causal Signal (`test_synthetic_causal_signal`)

**Objective**: Verify GPU produces correct causal discovery on known data.

**Setup**
- Create synthetic time series with **known causal relationship**: $X \to Y$
- $X_t$: random normal noise
- $Y_t = 0.5 \cdot X_{t-1} + 0.3 \cdot \epsilon_t$ (depends on lagged X)
- $Z_t$: independent noise (control)

**Validation**
- Compute Granger causality on CPU vs GPU
- **Expected**: GC(X→Y) > GC(Y→X) (X Granger-causes Y)
- **Tolerance**: max element-wise difference < 0.5, >90% sparsity pattern agreement
- **Output**: Side-by-side metrics table

---

### 2. Random Data (`test_random_data`)

**Objective**: Verify that GPU correctly identifies **sparse/zero edges** on non-causal data.

**Setup**
- 12 brain regions, 150 time points
- Completely independent random normal noise

**Validation**
- Both CPU and GPU should produce near-zero adjacency matrices
- Non-zero edge counts should be similar between implementations
- Max difference < 0.005

---

### 3. Short Time Series (`test_short_timeseries`)

**Objective**: Handle edge case where T < max_lag + 10 (insufficient samples for reliable Granger test).

**Expected Behavior**
- Both CPU and GPU return all-zero adjacency (graceful degradation)
- Numerically equal fallback behavior

---

### 4. NaN/Inf Handling (`test_nan_handling`)

**Objective**: Verify robust preprocessing on corrupted data.

**Setup**
- Insert NaN at (10, 5) and Inf at (20, 3)
- Both CPU and GPU must handle without crashes

**Expected Behavior**
- Pre-filter corrupted rows before Granger computation
- Return all-zero matrix (safe fallback)

---

### 5. Speed Benchmark (`benchmark_speed`)

**Objective**: Measure GPU speedup over CPU baseline.

**Setup**
- 200 time points, 12 regions
- Average over 10 repetitions

**Output Metrics**
- CPU time (ms per iteration)
- GPU time (ms per iteration)
- Speedup factor (CPU time / GPU time)

---

## Implementation Details

### GPU Implementation (`compute_granger_causality_gpu_impl`)
Located in `src/features/causal_inference.py`.

**Algorithm**
```
Input: time_series matrix (T, n_regions), max_lag
1. Construct lagged design matrix X ∈ ℝ^((T-max_lag) × n_regions*max_lag)
2. For each region pair (i, j):
   - Fit reduced model: y = β₀ (baseline)
   - Fit full model: y = β₀ + Σ(lagged_j)
   - Compute F-statistic: F = (RSS_r - RSS_f) / (RSS_f / dof)
   - Granger weight: max(0, -log₁₀(p_value))
3. Return n_regions × n_regions adjacency matrix
```

**Optimization Tricks**
- Vectorize across all lags using `torch.unfold()`
- Batch QR decomposition for numerical stability
- GPU memory pooling to reduce allocation overhead

---

## Troubleshooting

### Test Fails: "CUDA out of memory"
- **Fix**: Reduce batch size or set `GRANGER_USE_GPU=False`

### Test Fails: "Numerical differences too large"
- **Fix**: Ensure consistent dtypes across CPU/GPU paths (float64 vs float32)

---

## Integration with Pipeline

- `src/features/construct_causal.py` — calls `compute_granger_causality(use_gpu=True)`
- Tests part of standard `pytest tests/unit/`
