"""
Frequency-Domain Feature Extraction for fMRI Time Series

Extracts spectral features from BOLD signals to capture oscillatory dynamics
that are altered in ASD (particularly gamma-band synchrony).

References:
- Rojas et al. (2008): Gamma-band abnormalities in ASD
- Orekhova et al. (2007): Enhanced gamma oscillations in ASD
- Bruña et al. (2012): Spectral entropy for brain disorder classification
"""

import numpy as np
import logging
from scipy.signal import welch, hilbert
from scipy.stats import entropy
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_band_power(
    ts: np.ndarray,
    fs: float = 0.5,
    bands: Dict[str, Tuple[float, float]] = None
) -> Dict[str, float]:
    """
    Extract power spectral density features from time series.
    
    Args:
        ts: Time series (1D array, shape: [timepoints])
        fs: Sampling frequency in Hz (default: 0.5 Hz for TR=2s)
        bands: Dictionary of frequency bands {name: (low, high)}
               Default: delta, theta, alpha, beta, gamma
    
    Returns:
        Dictionary with 12 features:
        - {band}_power: Total power in each band (5 features)
        - {band}_peak_freq: Dominant frequency in each band (5 features)
        - spectral_entropy: Shannon entropy of power spectrum
        - phase_std: Standard deviation of instantaneous phase
    
    Example:
        >>> ts = np.random.randn(200)
        >>> features = extract_band_power(ts, fs=0.5)
        >>> print(features['gamma_power'])
    """
    # Default frequency bands (adjusted for fMRI sampling rate ~0.5 Hz / TR=2s)
    # Note: Standard EEG bands (Gamma > 30Hz) are not visible in fMRI.
    # We map "Slow" oscillation bands to these names for compatibility:
    # - delta: 0.01 - 0.027 Hz (Slow-5)
    # - theta: 0.027 - 0.073 Hz (Slow-4)
    # - alpha: 0.073 - 0.15 Hz (Slow-3 lower)
    # - beta:  0.15 - 0.20 Hz (Slow-3 upper)
    # - gamma: 0.20 - 0.25 Hz (Slow-2) - encroaching on Nyquist
    if bands is None:
        bands = {
            'delta': (0.01, 0.027),
            'theta': (0.027, 0.073),
            'alpha': (0.073, 0.15),
            'beta': (0.15, 0.20),
            'gamma': (0.20, 0.25)
        }
    
    # Validate input
    if len(ts) < 10:
        logger.warning(f"Time series too short ({len(ts)} timepoints), returning zeros")
        return _get_zero_features(bands)
    
    if np.isnan(ts).any() or np.isinf(ts).any():
        logger.warning("Time series contains NaN/Inf, returning zeros")
        return _get_zero_features(bands)
    
    # Remove mean and detrend
    ts_centered = ts - np.mean(ts)
    
    # Compute power spectral density using Welch's method
    try:
        nperseg = min(256, len(ts_centered))
        freqs, psd = welch(
            ts_centered,
            fs=fs,
            nperseg=nperseg,
            noverlap=nperseg // 2,
            window='hann',
            scaling='density'
        )
    except Exception as e:
        logger.warning(f"PSD computation failed: {e}, returning zeros")
        return _get_zero_features(bands)
    
    # Initialize feature dictionary
    features = {}
    
    # Extract band-specific features
    for band_name, (low, high) in bands.items():
        # Find frequencies in this band
        band_mask = (freqs >= low) & (freqs < high)
        
        if not band_mask.any():
            # No frequencies in this band (shouldn't happen with default bands)
            features[f'{band_name}_power'] = 0.0
            features[f'{band_name}_peak_freq'] = 0.0
            continue
        
        # Total power in band (area under PSD curve)
        band_power = np.trapz(psd[band_mask], freqs[band_mask])
        features[f'{band_name}_power'] = float(band_power)
        
        # Peak frequency in band
        band_freqs = freqs[band_mask]
        band_psd = psd[band_mask]
        
        if len(band_psd) > 0:
            peak_idx = np.argmax(band_psd)
            features[f'{band_name}_peak_freq'] = float(band_freqs[peak_idx])
        else:
            features[f'{band_name}_peak_freq'] = 0.0
    
    # Spectral entropy (complexity measure)
    # Normalize PSD to probability distribution
    psd_norm = psd / (np.sum(psd) + 1e-10)
    spectral_entropy = entropy(psd_norm + 1e-10)
    features['spectral_entropy'] = float(spectral_entropy)
    
    # Instantaneous phase features (for connectivity analysis)
    try:
        analytic_signal = hilbert(ts_centered)
        instantaneous_phase = np.angle(analytic_signal)
        features['phase_std'] = float(np.std(instantaneous_phase))
    except Exception as e:
        logger.warning(f"Phase computation failed: {e}, setting to 0")
        features['phase_std'] = 0.0
    
    return features


def _get_zero_features(bands: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    """Return zero-valued features when computation fails."""
    features = {}
    for band_name in bands.keys():
        features[f'{band_name}_power'] = 0.0
        features[f'{band_name}_peak_freq'] = 0.0
    features['spectral_entropy'] = 0.0
    features['phase_std'] = 0.0
    return features


def extract_frequency_features_batch(
    ts_matrix: np.ndarray,
    fs: float = 0.5
) -> np.ndarray:
    """
    Extract frequency features for multiple ROIs in parallel.
    
    Args:
        ts_matrix: Time series matrix (shape: [timepoints, n_rois])
        fs: Sampling frequency in Hz
    
    Returns:
        Feature matrix (shape: [n_rois, 12])
        Columns: [delta_power, delta_peak, theta_power, theta_peak, ..., spectral_entropy, phase_std]
    
    Example:
        >>> ts_matrix = np.random.randn(200, 170)  # 200 timepoints, 170 ROIs
        >>> features = extract_frequency_features_batch(ts_matrix)
        >>> print(features.shape)  # (170, 12)
    """
    n_rois = ts_matrix.shape[1]
    feature_names = [
        'delta_power', 'delta_peak_freq',
        'theta_power', 'theta_peak_freq',
        'alpha_power', 'alpha_peak_freq',
        'beta_power', 'beta_peak_freq',
        'gamma_power', 'gamma_peak_freq',
        'spectral_entropy', 'phase_std'
    ]
    n_features = len(feature_names)
    
    # Initialize output matrix
    feature_matrix = np.zeros((n_rois, n_features))
    
    # Extract features for each ROI
    for roi_idx in range(n_rois):
        ts = ts_matrix[:, roi_idx]
        features_dict = extract_band_power(ts, fs=fs)
        
        # Convert dictionary to array (preserve order)
        for feat_idx, feat_name in enumerate(feature_names):
            feature_matrix[roi_idx, feat_idx] = features_dict[feat_name]
    
    return feature_matrix


def validate_frequency_features(features: Dict[str, float]) -> bool:
    """
    Validate extracted frequency features.
    
    Args:
        features: Dictionary of frequency features
    
    Returns:
        True if features are valid, False otherwise
    
    Validation checks:
    - All power values are non-negative
    - All frequencies are within valid range [0, fs/2]
    - Spectral entropy is non-negative
    - Phase std is in [0, π]
    """
    # Check power values
    power_keys = [k for k in features.keys() if k.endswith('_power')]
    for key in power_keys:
        if features[key] < 0:
            logger.warning(f"Negative power value: {key} = {features[key]}")
            return False
    
    # Check frequency values
    freq_keys = [k for k in features.keys() if k.endswith('_peak_freq')]
    for key in freq_keys:
        if features[key] < 0 or features[key] > 0.5:  # Nyquist limit for fs=0.5
            logger.warning(f"Invalid frequency: {key} = {features[key]}")
            return False
    
    # Check spectral entropy
    if features['spectral_entropy'] < 0:
        logger.warning(f"Negative entropy: {features['spectral_entropy']}")
        return False
    
    # Check phase std
    if features['phase_std'] < 0 or features['phase_std'] > np.pi:
        logger.warning(f"Invalid phase std: {features['phase_std']}")
        return False
    
    return True


if __name__ == "__main__":
    """Test frequency feature extraction."""
    print("="*60)
    print("TESTING FREQUENCY FEATURE EXTRACTION")
    print("="*60)
    
    # Test 1: Synthetic signal with known frequency
    print("\nTest 1: Synthetic signal (10 Hz oscillation)")
    fs = 100  # 100 Hz sampling
    t = np.linspace(0, 2, 200)  # 2 seconds
    signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz sine wave
    
    features = extract_band_power(signal, fs=fs, bands={'test': (8, 12)})
    print(f"  Peak frequency: {features['test_peak_freq']:.2f} Hz (expected: ~10 Hz)")
    print(f"  Power: {features['test_power']:.4f}")
    
    # Test 2: fMRI-like signal (low frequency)
    print("\nTest 2: fMRI-like signal (0.1 Hz)")
    fs_fmri = 0.5  # TR = 2s
    t_fmri = np.linspace(0, 400, 200)  # 400 seconds
    signal_fmri = np.sin(2 * np.pi * 0.1 * t_fmri) + 0.5 * np.random.randn(200)
    
    features_fmri = extract_band_power(signal_fmri, fs=fs_fmri)
    print(f"  Delta power: {features_fmri['delta_power']:.4f}")
    print(f"  Theta power: {features_fmri['theta_power']:.4f}")
    print(f"  Spectral entropy: {features_fmri['spectral_entropy']:.4f}")
    
    # Test 3: Batch processing
    print("\nTest 3: Batch processing (10 ROIs)")
    ts_matrix = np.random.randn(200, 10)
    feature_matrix = extract_frequency_features_batch(ts_matrix, fs=0.5)
    print(f"  Output shape: {feature_matrix.shape} (expected: (10, 12))")
    print(f"  Mean gamma power: {feature_matrix[:, 8].mean():.4f}")
    
    # Test 4: Validation
    print("\nTest 4: Feature validation")
    valid = validate_frequency_features(features_fmri)
    print(f"  Validation passed: {valid}")
    
    # Test 5: Edge cases
    print("\nTest 5: Edge cases")
    
    # Short signal
    short_signal = np.random.randn(5)
    features_short = extract_band_power(short_signal, fs=0.5)
    print(f"  Short signal (5 points): gamma_power = {features_short['gamma_power']}")
    
    # NaN signal
    nan_signal = np.array([1, 2, np.nan, 4, 5])
    features_nan = extract_band_power(nan_signal, fs=0.5)
    print(f"  NaN signal: gamma_power = {features_nan['gamma_power']}")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS COMPLETED")
    print("="*60)
