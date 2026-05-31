"""
Causal Inference for Brain Connectivity

Implements Granger causality for directed graph construction.

References:
- Granger (1969): Investigating Causal Relations by Econometric Models
- Barnett & Seth (2014): The MVGC toolbox for Granger causality
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def compute_granger_causality(
    ts_matrix: np.ndarray,
    max_lag: int = 5,
    significance_level: float = 0.05,
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Compute multivariate Granger causality matrix.

    Tests: Does past of region i improve prediction of region j beyond past of j alone?

    Args:
        ts_matrix: Time series matrix (shape: [timepoints, n_regions])
        max_lag: Maximum lag to test (default: 5 TRs)
        significance_level: Statistical significance threshold
        n_jobs: Number of parallel workers for pairwise region tests (-1 = all cores).

    Returns:
        Causality matrix (shape: [n_regions, n_regions])
        Values: -log10(p-value) where higher = stronger causality
        Diagonal is zero (no self-causation)

    Example:
        >>> ts = np.random.randn(200, 12)  # 200 timepoints, 12 regions
        >>> gc_matrix = compute_granger_causality(ts, max_lag=5)
        >>> print(gc_matrix.shape)  # (12, 12)
    """
    from joblib import Parallel, delayed
    from statsmodels.tsa.stattools import grangercausalitytests

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

    def _test_pair(i: int, j: int) -> tuple[int, int, float]:
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


def validate_causality_matrix(causal_matrix: np.ndarray) -> dict[str, float]:
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
