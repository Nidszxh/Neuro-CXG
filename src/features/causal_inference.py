"""
Causal Inference for Brain Connectivity

Implements proper causal inference methods for directed graph construction:
1. Granger Causality: Multivariate F-test for temporal precedence
2. Transfer Entropy: Information-theoretic causality (nonlinear)
3. Multi-lag aggregation: Combine causality across multiple timescales

References:
- Granger (1969): Investigating Causal Relations by Econometric Models
- Barnett & Seth (2014): The MVGC toolbox for Granger causality
- Schreiber (2000): Measuring Information Transfer
"""

import numpy as np
import logging
from typing import Tuple, Dict
from joblib import Parallel, delayed
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.stats import f as f_dist
import torch

logging.basicConfig(level=logging.INFO)
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
        n_jobs: Number of parallel workers for pairwise region tests (-1 = all cores)
    
    Returns:
        Causality matrix (shape: [n_regions, n_regions])
        Values: -log10(p-value) where higher = stronger causality
        Diagonal is zero (no self-causation)
    
    Example:
        >>> ts = np.random.randn(200, 12)  # 200 timepoints, 12 regions
        >>> gc_matrix = compute_granger_causality(ts, max_lag=5)
        >>> print(gc_matrix.shape)  # (12, 12)
    """
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
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_test_pair)(i, j) for i, j in pairs
    )

    for i, j, score in results:
        gc_matrix[i, j] = score

    return gc_matrix


def compute_granger_causality_gpu(
    ts_matrix: np.ndarray,
    max_lag: int = 5,
    device: str = 'cuda'
) -> np.ndarray:
    """
    Compute Granger causality using GPU-accelerated OLS (PyTorch).
    
    Args:
        ts_matrix: Time series matrix (shape: [timepoints, n_regions])
        max_lag: Maximum lag to test
        device: 'cuda' or 'cpu'
    
    Returns:
        Causality matrix (shape: [n_regions, n_regions])
    """
    n, n_regions = ts_matrix.shape
    
    if n < max_lag + 10:
        return np.zeros((n_regions, n_regions))
    
    # Move to GPU
    try:
        data = torch.from_numpy(ts_matrix).float().to(device)
    except Exception as e:
        logger.warning(f"Failed to move data to {device}: {e}. Falling back to CPU Granger.")
        return compute_granger_causality(ts_matrix, max_lag)
        
    gc_matrix = np.zeros((n_regions, n_regions))
    
    # 1. Prepare Lagged Data (All regions)
    # Shape: (n - max_lag, n_regions * max_lag)
    effective_n = n - max_lag
    
    if effective_n < 2 * max_lag + 2:
        return np.zeros((n_regions, n_regions))

    # Pre-compute all lags for all regions
    # X_lags[t] = [r0_t-1...r0_t-p, r1_t-1...r1_t-p, ...]
    X_lags = torch.zeros(effective_n, n_regions * max_lag, device=device)
    
    for r in range(n_regions):
        for l in range(1, max_lag + 1):
            # Column index: r * max_lag + (l-1)
            col_idx = r * max_lag + (l - 1)
            X_lags[:, col_idx] = data[max_lag-l : -l, r]
            
    # Target variables (current time)
    Y_targets = data[max_lag:, :]  # Shape: (effective_n, n_regions)
    
    # Add bias term (column of ones)
    ones = torch.ones(effective_n, 1, device=device)
    
    # Loop over pairs
    # Optimize: Could batch this, but strict pair loop is safer for OLS stability
    for j in range(n_regions): # Target
        
        # Restricted Model: Y_j ~ Y_j_past
        # Design matrix: Bias + Lags of Y_j
        start_idx = j * max_lag
        end_idx = (j + 1) * max_lag
        X_restricted = torch.cat([ones, X_lags[:, start_idx:end_idx]], dim=1)
        
        # Fit Restricted
        # w = (X^T X)^-1 X^T y
        try:
            # RSS_restricted
            # Use ridge regression (solve) for stability instead of raw lstsq
            XtX_r = X_restricted.T @ X_restricted
            ridge_r = torch.eye(XtX_r.shape[0], device=device) * 1e-4
            w_r = torch.linalg.solve(XtX_r + ridge_r, X_restricted.T @ Y_targets[:, j])
            preds_r = X_restricted @ w_r
            resid_r = Y_targets[:, j] - preds_r
            rss_r = (resid_r ** 2).sum()
        except RuntimeError:
            continue

        for i in range(n_regions): # Source
            if i == j: continue
            
            # Unrestricted Model: Y_j ~ Y_j_past + X_i_past
            # Design matrix: Bias + Lags of Y_j + Lags of X_i
            i_start = i * max_lag
            i_end = (i + 1) * max_lag
            
            X_unrestricted = torch.cat([
                X_restricted, 
                X_lags[:, i_start:i_end]
            ], dim=1)
            
            try:
                # RSS_unrestricted
                XtX_u = X_unrestricted.T @ X_unrestricted
                ridge_u = torch.eye(XtX_u.shape[0], device=device) * 1e-4
                w_u = torch.linalg.solve(XtX_u + ridge_u, X_unrestricted.T @ Y_targets[:, j])
                preds_u = X_unrestricted @ w_u
                resid_u = Y_targets[:, j] - preds_u
                rss_u = (resid_u ** 2).sum()
                
                # F-test
                # p1 = max_lag (params in restricted excluding bias) -> actually max_lag + 1
                # p2 = 2*max_lag (params in unrestricted)
                # Number of restrictions = p2 - p1 = max_lag
                num_params_u = 2 * max_lag + 1
                dof_u = effective_n - num_params_u
                
                if dof_u <= 0 or rss_u < 1e-6:
                    continue
                    
                mse_u = rss_u / dof_u
                f_stat = ((rss_r - rss_u) / max_lag) / mse_u
                
                # P-value (using scipy on CPU as PyTorch doesn't have f.cdf)
                f_val = f_stat.item()
                if f_val > 0:
                    p_val = f_dist.sf(f_val, max_lag, dof_u)
                    if p_val > 0:
                        gc_matrix[i, j] = -np.log10(p_val + 1e-10)
                    else:
                        gc_matrix[i, j] = 10.0
                
            except RuntimeError:
                continue

    return gc_matrix


def compute_transfer_entropy(
    ts_matrix: np.ndarray,
    k: int = 1,
    bins: int = 10
) -> np.ndarray:
    """
    Compute transfer entropy matrix (information-theoretic causality).
    
    TE(X→Y) = I(Y_future; X_past | Y_past)
    Measures information flow from X to Y.
    
    Args:
        ts_matrix: Time series matrix (shape: [timepoints, n_regions])
        k: History length (default: 1)
        bins: Number of bins for discretization
    
    Returns:
        Transfer entropy matrix (shape: [n_regions, n_regions])
        Values: bits of information transferred
    
    Note: This is a simplified implementation. For production use,
          consider using specialized libraries like pyinform or IDTxl.
    """
    n_timepoints, n_regions = ts_matrix.shape
    
    # Validate input
    if n_timepoints < k + 10:
        logger.warning(f"Time series too short ({n_timepoints} points) for history {k}")
        return np.zeros((n_regions, n_regions))
    
    # Discretize time series
    ts_discrete = np.zeros_like(ts_matrix, dtype=int)
    for i in range(n_regions):
        ts_discrete[:, i] = np.digitize(
            ts_matrix[:, i],
            bins=np.linspace(ts_matrix[:, i].min(), ts_matrix[:, i].max(), bins)
        )
    
    # Initialize TE matrix
    te_matrix = np.zeros((n_regions, n_regions))
    
    # Compute pairwise transfer entropy
    for i in range(n_regions):
        for j in range(n_regions):
            if i == j:
                continue
            
            try:
                te_matrix[i, j] = _compute_te_pair(
                    ts_discrete[:, i],  # Source
                    ts_discrete[:, j],  # Target
                    k=k
                )
            except Exception as e:
                logger.debug(f"TE computation failed for {i}→{j}: {e}")
                te_matrix[i, j] = 0.0
    
    return te_matrix


def _compute_te_pair(source: np.ndarray, target: np.ndarray, k: int = 1) -> float:
    """
    Compute transfer entropy for a single pair of time series.
    
    TE(X→Y) = H(Y_t | Y_{t-1:t-k}) - H(Y_t | Y_{t-1:t-k}, X_{t-1:t-k})
    """
    n = len(source)
    
    # Create lagged variables
    y_future = target[k:]
    y_past = np.column_stack([target[i:n-k+i] for i in range(k)])
    x_past = np.column_stack([source[i:n-k+i] for i in range(k)])
    
    # Compute entropies
    h_y_given_y_past = _conditional_entropy(y_future, y_past)
    h_y_given_both = _conditional_entropy(y_future, np.column_stack([y_past, x_past]))
    
    # Transfer entropy
    te = h_y_given_y_past - h_y_given_both
    
    return max(te, 0.0)  # TE should be non-negative


def _conditional_entropy(y: np.ndarray, x: np.ndarray) -> float:
    """
    Compute conditional entropy H(Y|X) using empirical probabilities.
    """
    # Convert to tuples for hashing
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    
    x_tuples = [tuple(row) for row in x]
    
    # Count joint occurrences
    joint_counts = {}
    x_counts = {}
    
    for i in range(len(y)):
        x_val = x_tuples[i]
        y_val = y[i]
        
        joint_key = (x_val, y_val)
        joint_counts[joint_key] = joint_counts.get(joint_key, 0) + 1
        x_counts[x_val] = x_counts.get(x_val, 0) + 1
    
    # Compute conditional entropy
    h = 0.0
    n = len(y)
    
    for (x_val, y_val), joint_count in joint_counts.items():
        p_xy = joint_count / n
        p_x = x_counts[x_val] / n
        p_y_given_x = joint_count / x_counts[x_val]
        
        if p_y_given_x > 0:
            h -= p_xy * np.log2(p_y_given_x)
    
    return h


def compute_multilag_causality(
    ts_matrix: np.ndarray,
    lags: list = [1, 2, 3, 5, 10],
    method: str = 'pearson'
) -> np.ndarray:
    """
    Compute causality aggregated across multiple lags.
    
    Args:
        ts_matrix: Time series matrix (shape: [timepoints, n_regions])
        lags: List of lags to test
        method: 'pearson' (correlation) or 'partial' (partial correlation)
    
    Returns:
        Causality matrix (shape: [n_regions, n_regions])
        Values: mean absolute causality across lags
    """
    n_timepoints, n_regions = ts_matrix.shape
    causal_matrices = []
    
    for lag in lags:
        if n_timepoints <= lag:
            logger.warning(f"Skipping lag {lag} (time series too short)")
            continue
        
        # Split into past and current
        ts_past = ts_matrix[:-lag]
        ts_current = ts_matrix[lag:]
        
        if method == 'pearson':
            # Simple lagged correlation
            causal_matrix = np.corrcoef(ts_past.T, ts_current.T)[:n_regions, n_regions:]
        
        elif method == 'partial':
            # Partial correlation (controls for confounders)
            from sklearn.covariance import GraphicalLassoCV
            
            try:
                # Combine past and current
                combined = np.column_stack([ts_past, ts_current])
                
                # Fit graphical lasso
                model = GraphicalLassoCV(cv=3, max_iter=100)
                model.fit(combined)
                
                # Extract cross-lag precision matrix
                precision = model.precision_
                partial_corr = -precision[:n_regions, n_regions:] / np.sqrt(
                    np.outer(
                        np.diag(precision[:n_regions, :n_regions]),
                        np.diag(precision[n_regions:, n_regions:])
                    )
                )
                
                causal_matrix = partial_corr
            
            except Exception as e:
                logger.warning(f"Partial correlation failed at lag {lag}: {e}")
                causal_matrix = np.zeros((n_regions, n_regions))
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        causal_matrices.append(causal_matrix)
    
    if not causal_matrices:
        return np.zeros((n_regions, n_regions))
    
    # Aggregate: mean absolute causality across lags
    aggregated = np.mean(np.abs(causal_matrices), axis=0)
    
    return aggregated


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
        for j in range(i+1, n_regions):
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
        'max_strength': float(np.abs(causal_matrix).max())
    }
    
    return metrics


if __name__ == "__main__":
    """Test causal inference methods."""
    print("="*60)
    print("TESTING CAUSAL INFERENCE METHODS")
    print("="*60)
    
    # Test 1: Granger causality on synthetic causal signal
    print("\nTest 1: Granger Causality (X → Y)")
    np.random.seed(42)
    n = 200
    X = np.random.randn(n)
    Y = 0.5 * np.roll(X, 1) + 0.3 * np.random.randn(n)  # Y depends on past of X
    Z = np.random.randn(n)  # Independent
    
    ts_matrix = np.column_stack([X, Y, Z])
    gc_matrix = compute_granger_causality(ts_matrix, max_lag=5)
    
    print(f"  GC(X→Y): {gc_matrix[0, 1]:.4f} (should be high)")
    print(f"  GC(Y→X): {gc_matrix[1, 0]:.4f} (should be low)")
    print(f"  GC(X→Z): {gc_matrix[0, 2]:.4f} (should be low)")
    print(f"  Directionality: {gc_matrix[0, 1] > gc_matrix[1, 0]}")
    
    # Test 2: Transfer entropy
    print("\nTest 2: Transfer Entropy")
    te_matrix = compute_transfer_entropy(ts_matrix, k=1, bins=10)
    print(f"  TE(X→Y): {te_matrix[0, 1]:.4f}")
    print(f"  TE(Y→X): {te_matrix[1, 0]:.4f}")
    print(f"  TE(X→Z): {te_matrix[0, 2]:.4f}")
    
    # Test 3: Multi-lag causality
    print("\nTest 3: Multi-lag Causality")
    multilag_matrix = compute_multilag_causality(ts_matrix, lags=[1, 2, 3, 5])
    print(f"  Multi-lag(X→Y): {multilag_matrix[0, 1]:.4f}")
    print(f"  Multi-lag(Y→X): {multilag_matrix[1, 0]:.4f}")
    
    # Test 4: Validation
    print("\nTest 4: Causality Matrix Validation")
    metrics = validate_causality_matrix(gc_matrix)
    print(f"  Symmetry: {metrics['symmetry']:.4f} (lower = more directional)")
    print(f"  Directionality: {metrics['directionality']:.4f} (higher = better)")
    print(f"  Non-zero edges: {metrics['non_zero_edges']}")
    print(f"  Mean strength: {metrics['mean_strength']:.4f}")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS COMPLETED")
    print("="*60)
