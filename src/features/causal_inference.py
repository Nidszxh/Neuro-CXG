"""
Causal Inference for Brain Connectivity

Implements Granger causality for directed graph construction.

**GPU Implementation (June 2026):**
- Added GPU-accelerated Granger causality via `_compute_granger_causality_gpu_impl()`
- Uses batched linear regression with vectorized F-test
- Auto-detects CUDA availability via GRANGER_USE_GPU config flag
- Falls back to CPU on any error

Removed in Task 6 (DD-014):
- compute_granger_causality_gpu: unmaintained GPU path; CPU Granger is fast enough
  for 12-node graphs and avoids CUDA memory fragmentation issues.
- compute_transfer_entropy / _compute_te_pair / _conditional_entropy: placeholder TE
  implementation superseded by multi-view causal invariance loss (DD-010).
- compute_multilag_causality: superseded by construct_multiview_graphs() (DD-010).

References:
- Granger (1969): Investigating Causal Relations by Econometric Models
- Barnett & Seth (2014): The MVGC toolbox for Granger causality
"""

import numpy as np
import logging
from typing import Tuple, Dict, Optional
from joblib import Parallel, delayed
from statsmodels.tsa.stattools import grangercausalitytests
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_granger_causality(
    ts_matrix: np.ndarray,
    max_lag: int = 5,
    significance_level: float = 0.05,
    n_jobs: int = -1,
    use_gpu: Optional[bool] = None,
) -> np.ndarray:
    """
    Compute multivariate Granger causality matrix.

    Tests: Does past of region i improve prediction of region j beyond past of j alone?

    Args:
        ts_matrix: Time series matrix (shape: [timepoints, n_regions])
        max_lag: Maximum lag to test (default: 5 TRs)
        significance_level: Statistical significance threshold
        n_jobs: Number of parallel workers for pairwise region tests (-1 = all cores).
                Only used when use_gpu=False.
        use_gpu: If True, use GPU-accelerated implementation. If False, use CPU.
                 If None, auto-detect: use GPU if GRANGER_USE_GPU=True and CUDA available.

    Returns:
        Causality matrix (shape: [n_regions, n_regions])
        Values: -log10(p-value) where higher = stronger causality
        Diagonal is zero (no self-causation)

    Example:
        >>> ts = np.random.randn(200, 12)  # 200 timepoints, 12 regions
        >>> gc_matrix = compute_granger_causality(ts, max_lag=5)
        >>> print(gc_matrix.shape)  # (12, 12)
    """
    # Auto-detect GPU usage
    if use_gpu is None:
        try:
            from src.core.hyperparams import GRANGER_USE_GPU
            use_gpu = GRANGER_USE_GPU and torch.cuda.is_available()
        except ImportError:
            use_gpu = torch.cuda.is_available()

    if use_gpu:
        try:
            return _compute_granger_causality_gpu_impl(
                ts_matrix, max_lag, significance_level
            )
        except Exception as e:
            logger.warning(f"GPU computation failed: {e}, falling back to CPU")
            # Fall back to CPU

    # Rest of the CPU implementation
    n_timepoints, n_regions = ts_matrix.shape

    # Validate input
    if n_timepoints < max_lag + 10:
        logger.warning(f"Time series too short ({n_timepoints} points) for lag {max_lag}")
        return np.zeros((n_regions, n_regions))

    if np.isnan(ts_matrix).any() or np.isinf(ts_matrix).any():
        logger.warning("Time series contains NaN/Inf, returning zero matrix")
        return np.zeros((n_regions, n_regions))

    # Initialize causality matrix
    gc_matrix = np.zeros((n_regions, n_regions))

    def _test_pair(i: int, j: int) -> Tuple[int, int, float]:
        if i == j:
            return i, j, 0.0

        try:
            # Prepare data: [target, source]
            # Test if region i Granger-causes region j
            data = np.column_stack([ts_matrix[:, j], ts_matrix[:, i]])

            results = grangercausalitytests(data, maxlag=max_lag, verbose=False)
            p_values = [results[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1)]

            # Use minimum p-value across lags (strongest evidence)
            min_p_value = min(p_values)

            # Bonferroni correction for multiple lag tests
            n_tests = len(p_values)
            corrected_p = min(min_p_value * n_tests, 1.0)

            if corrected_p <= significance_level:
                score = -np.log10(corrected_p + 1e-10)
                return i, j, float(min(score, 10.0))
            return i, j, 0.0
        except Exception as e:
            logger.debug(f"Granger test failed for {i}→{j}: {e}")
            return i, j, 0.0

    pairs = [(i, j) for i in range(n_regions) for j in range(n_regions) if i != j]
    if n_regions <= 15:
        # Process overhead dominates at 12-region scale; sequential is faster.
        results = [_test_pair(i, j) for i, j in pairs]
    else:
        results = Parallel(n_jobs=n_jobs, prefer="processes", backend="loky")(
            delayed(_test_pair)(i, j) for i, j in pairs
        )

    for i, j, score in results:
        gc_matrix[i, j] = score

    return gc_matrix


def _compute_granger_causality_gpu_impl(
    ts_matrix: np.ndarray,
    max_lag: int = 5,
    significance_level: float = 0.05,
) -> np.ndarray:
    """
    GPU-accelerated multivariate Granger causality using batched linear regression.

    Computes all (i → j) pairs simultaneously using vectorized F-test.
    Based on the SSR (Sum of Squared Residuals) F-test from statsmodels.

    Args:
        ts_matrix: Time series matrix (shape: [timepoints, n_regions])
        max_lag: Maximum lag to test (default: 5 TRs)
        significance_level: Statistical significance threshold

    Returns:
        Causality matrix (shape: [n_regions, n_regions])
    """
    import scipy.stats as stats

    n_timepoints, n_regions = ts_matrix.shape

    if n_timepoints < max_lag + 10:
        return np.zeros((n_regions, n_regions))

    if np.isnan(ts_matrix).any() or np.isinf(ts_matrix).any():
        return np.zeros((n_regions, n_regions))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ts_gpu = torch.as_tensor(ts_matrix, dtype=torch.float32, device=device)

    T = n_timepoints
    L = max_lag
    n_effect = T - L

    gc_matrix = np.zeros((n_regions, n_regions))

    for target_idx in range(n_regions):
        y_full = ts_gpu[:, target_idx]
        y_valid = y_full[L:]

        Y_lags = torch.zeros(n_effect, L, dtype=torch.float32, device=device)
        for lag in range(1, L + 1):
            if lag < L:
                Y_lags[:, lag - 1] = y_full[L - lag:-lag]
            else:
                Y_lags[:, lag - 1] = y_full[:-L]

        X_pinv_Y = torch.linalg.pinv(Y_lags)
        y_pred_restricted = Y_lags @ X_pinv_Y @ y_valid
        residuals_restricted = y_valid - y_pred_restricted
        RSS_restricted = (residuals_restricted ** 2).sum().item()

        for source_idx in range(n_regions):
            if source_idx == target_idx:
                continue

            x_full = ts_gpu[:, source_idx]

            X_design = torch.zeros(n_effect, 2 * L, dtype=torch.float32, device=device)
            for lag in range(1, L + 1):
                if lag < L:
                    x_lag = x_full[L - lag:-lag]
                    y_lag = y_full[L - lag:-lag]
                else:
                    x_lag = x_full[:-L]
                    y_lag = y_full[:-L]
                X_design[:, L + lag - 1] = x_lag
                X_design[:, lag - 1] = y_lag

            X_pinv = torch.linalg.pinv(X_design)
            beta = X_pinv @ y_valid
            y_pred = X_design @ beta
            residuals_full = y_valid - y_pred
            RSS_full = (residuals_full ** 2).sum().item()

            df1 = L
            df2 = n_effect - 2 * L - 1
            if df2 <= 0 or RSS_full < 1e-10:
                continue

            f_stat = ((RSS_restricted - RSS_full) / df1) / (RSS_full / df2)

            if f_stat > 0:
                p_value = 1 - stats.f.cdf(f_stat, df1, df2)
                p_value = min(p_value * L, 1.0)

                if p_value <= significance_level:
                    score = -np.log10(p_value + 1e-10)
                    gc_matrix[source_idx, target_idx] = min(score, 10.0)

    return gc_matrix


def validate_causality_matrix(causal_matrix: np.ndarray) -> Dict[str, float]:
    """
    Validate and analyze causality matrix.

    Args:
        causal_matrix: Causality matrix (shape: [n_regions, n_regions])

    Returns:
        Dictionary with validation metrics
    """
    n_regions = causal_matrix.shape[0]

    # Check for symmetry (should NOT be symmetric for causal graphs)
    symmetry = np.abs(causal_matrix - causal_matrix.T).mean()

    # Check for directionality (X→Y should differ from Y→X)
    directionality_ratio = []
    for i in range(n_regions):
        for j in range(i + 1, n_regions):
            if causal_matrix[i, j] + causal_matrix[j, i] > 0:
                ratio = abs(causal_matrix[i, j] - causal_matrix[j, i]) / (
                    causal_matrix[i, j] + causal_matrix[j, i]
                )
                directionality_ratio.append(ratio)

    mean_directionality = np.mean(directionality_ratio) if directionality_ratio else 0.0

    # Check for non-zero edges
    non_zero = (causal_matrix != 0).sum() - n_regions  # Exclude diagonal

    metrics = {
        'symmetry': float(symmetry),
        'directionality': float(mean_directionality),
        'non_zero_edges': int(non_zero),
        'mean_strength': float(np.abs(causal_matrix).mean()),
        'max_strength': float(np.abs(causal_matrix).max()),
    }

    return metrics


if __name__ == "__main__":
    """Test Granger causality."""
    print("=" * 60)
    print("TESTING GRANGER CAUSALITY")
    print("=" * 60)

    # Synthetic causal signal: X → Y
    print("\nTest 1: Granger Causality (X → Y)")
    np.random.seed(42)
    n = 200
    X = np.random.randn(n)
    Y = 0.5 * np.roll(X, 1) + 0.3 * np.random.randn(n)
    Z = np.random.randn(n)

    ts_matrix = np.column_stack([X, Y, Z])
    gc_matrix = compute_granger_causality(ts_matrix, max_lag=5)

    print(f"  GC(X→Y): {gc_matrix[0, 1]:.4f} (should be high)")
    print(f"  GC(Y→X): {gc_matrix[1, 0]:.4f} (should be low)")
    print(f"  GC(X→Z): {gc_matrix[0, 2]:.4f} (should be low)")
    print(f"  Directionality: {gc_matrix[0, 1] > gc_matrix[1, 0]}")

    # Validation
    print("\nTest 2: Causality Matrix Validation")
    metrics = validate_causality_matrix(gc_matrix)
    print(f"  Symmetry:        {metrics['symmetry']:.4f} (lower = more directional)")
    print(f"  Directionality:  {metrics['directionality']:.4f} (higher = better)")
    print(f"  Non-zero edges:  {metrics['non_zero_edges']}")
    print(f"  Mean strength:   {metrics['mean_strength']:.4f}")

    print("\n" + "=" * 60)
    print("✅ ALL TESTS COMPLETED")
    print("=" * 60)
