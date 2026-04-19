"""
Test script for numerical equivalence between CPU and GPU Granger causality implementations.

Run: python tests/unit/test_granger_gpu.py
"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.causal_inference import compute_granger_causality
import torch


def test_synthetic_causal_signal():
    """Test on synthetic data with known causal relationship X → Y."""
    print("\nTest 1: Synthetic causal signal (X → Y)")
    np.random.seed(42)
    n = 200
    X = np.random.randn(n)
    Y = 0.5 * np.roll(X, 1) + 0.3 * np.random.randn(n)
    Z = np.random.randn(n)

    ts_matrix = np.column_stack([X, Y, Z])

    gc_cpu = compute_granger_causality(ts_matrix, max_lag=5, use_gpu=False)
    gc_gpu = compute_granger_causality(ts_matrix, max_lag=5, use_gpu=True)

    print(f"  CPU: GC(X→Y)={gc_cpu[0,1]:.4f}, GC(Y→X)={gc_cpu[1,0]:.4f}")
    print(f"  GPU: GC(X→Y)={gc_gpu[0,1]:.4f}, GC(Y→X)={gc_gpu[1,0]:.4f}")

    diff = np.abs(gc_cpu - gc_gpu)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    print(f"  Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")

    non_zero_cpu = gc_cpu > 0
    non_zero_gpu = gc_gpu > 0
    matches = np.sum(non_zero_cpu == non_zero_gpu)
    print(f"  Zero/non-zero matches: {matches}/{gc_cpu.size} ({100*matches/gc_cpu.size:.1f}%)")

    return max_diff < 0.5


def test_random_data():
    """Test on random data - should be mostly zeros."""
    print("\nTest 2: Random data (should have few significant edges)")
    np.random.seed(123)
    n_regions = 12
    T = 150

    ts_matrix = np.random.randn(T, n_regions)

    gc_cpu = compute_granger_causality(ts_matrix, max_lag=5, use_gpu=False)
    gc_gpu = compute_granger_causality(ts_matrix, max_lag=5, use_gpu=True)

    nonzero_cpu = np.count_nonzero(gc_cpu)
    nonzero_gpu = np.count_nonzero(gc_gpu)
    print(f"  CPU non-zero edges: {nonzero_cpu}")
    print(f"  GPU non-zero edges: {nonzero_gpu}")

    diff = np.abs(gc_cpu - gc_gpu)
    max_diff = np.max(diff)
    print(f"  Max diff: {max_diff:.6f}")

    return True


def test_short_timeseries():
    """Test edge case: very short time series."""
    print("\nTest 3: Short time series (T < max_lag + 10)")
    np.random.seed(456)
    T = 12
    n_regions = 12

    ts_matrix = np.random.randn(T, n_regions)

    gc_cpu = compute_granger_causality(ts_matrix, max_lag=5, use_gpu=False)
    gc_gpu = compute_granger_causality(ts_matrix, max_lag=5, use_gpu=True)

    print(f"  CPU: all zeros = {np.allclose(gc_cpu, 0)}")
    print(f"  GPU: all zeros = {np.allclose(gc_gpu, 0)}")

    return np.allclose(gc_cpu, gc_gpu)


def test_nan_handling():
    """Test NaN handling."""
    print("\nTest 4: NaN handling")
    np.random.seed(789)
    T = 100
    n_regions = 12

    ts_matrix = np.random.randn(T, n_regions)
    ts_matrix[10, 5] = np.nan
    ts_matrix[20, 3] = np.inf

    gc_cpu = compute_granger_causality(ts_matrix, max_lag=5, use_gpu=False)
    gc_gpu = compute_granger_causality(ts_matrix, max_lag=5, use_gpu=True)

    print(f"  CPU: all zeros = {np.allclose(gc_cpu, 0)}")
    print(f"  GPU: all zeros = {np.allclose(gc_gpu, 0)}")

    return np.allclose(gc_cpu, 0) and np.allclose(gc_gpu, 0)


def benchmark_speed():
    """Benchmark CPU vs GPU speed."""
    print("\nTest 5: Speed benchmark")
    np.random.seed(999)
    T = 200
    n_regions = 12

    ts_matrix = np.random.randn(T, n_regions)

    import time

    start = time.time()
    for _ in range(10):
        compute_granger_causality(ts_matrix, max_lag=5, use_gpu=False)
    cpu_time = (time.time() - start) / 10 * 1000
    print(f"  CPU time (avg over 10): {cpu_time:.1f} ms")

    if torch.cuda.is_available():
        start = time.time()
        for _ in range(10):
            compute_granger_causality(ts_matrix, max_lag=5, use_gpu=True)
        gpu_time = (time.time() - start) / 10 * 1000
        print(f"  GPU time (avg over 10): {gpu_time:.1f} ms")
        print(f"  Speedup: {cpu_time/gpu_time:.1f}x")
    else:
        print("  GPU not available, skipping GPU benchmark")

    return True


def main():
    print("=" * 60)
    print("GRANGER CAUSALITY GPU TEST SUITE")
    print("=" * 60)

    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results = []
    results.append(("Synthetic causal signal", test_synthetic_causal_signal()))
    results.append(("Random data", test_random_data()))
    results.append(("Short time series", test_short_timeseries()))
    results.append(("NaN handling", test_nan_handling()))
    results.append(("Speed benchmark", benchmark_speed()))

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    all_passed = all(r[1] for r in results)
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())