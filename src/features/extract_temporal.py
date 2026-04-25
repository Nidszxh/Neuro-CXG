import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import welch, hilbert
from tqdm import tqdm
from joblib import Parallel, delayed

import torch

torch.set_num_threads(min(4, torch.get_num_threads()))

# Setup paths and config
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import (
    ACTIVE_FREQ_BANDS,
    DATA_FINAL,
    FEATURE_GROUPS,
    MASTER_MANIFEST,
    NODE_ATTRIBUTES_TEMPORAL,
    DEFAULT_TR,
    NYQUIST_EPS,
    UNRELIABLE_FREQ_BANDS_AT_NYQUIST,
)

# Expected ROI count range for validation (AAL3v1 atlas)
# Note: Some AAL3v1 templates have 2 unused/empty ROIs, so 164-170 are all valid
VALID_ROI_RANGE = (164, 170)
MAX_ROIS = 170

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
_NYQUIST_NOTE_EMITTED = False


# ============================================================================
# FREQUENCY-DOMAIN FEATURE EXTRACTION
# ============================================================================


def _get_zero_features(bands: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    """Return zeroed feature dict for edge cases."""
    features = {}
    for band_name in bands.keys():
        features[f"{band_name}_power"] = 0.0
        features[f"{band_name}_peak_freq"] = 0.0
    features["spectral_entropy"] = 0.0
    features["phase_std"] = 0.0
    return features


def _get_unreliable_bands(fs: float) -> set[str]:
    """Return bands that should be zeroed for the given sampling rate."""
    nyquist = fs / 2.0
    unreliable = set(UNRELIABLE_FREQ_BANDS_AT_NYQUIST)
    for band_name, (_, high) in ACTIVE_FREQ_BANDS.items():
        if high >= nyquist:
            unreliable.add(band_name)
    return unreliable


def extract_band_power(
    ts: np.ndarray, fs: float = 0.5, bands: Dict[str, Tuple[float, float]] = None
) -> Dict[str, float]:
    """
    Extract power spectral density features from time series.

    Args:
        ts: Time series (1D array, shape: [timepoints])
        fs: Sampling frequency in Hz (default: 0.5 Hz for TR=2s)
         bands: Dictionary of frequency bands {name: (low, high)}
             Default: ACTIVE_FREQ_BANDS from config

    Returns:
        Dictionary with 12 features:
        - {band}_power: Total power in each band (5 features)
        - {band}_peak_freq: Dominant frequency in each band (5 features)
        - spectral_entropy: Shannon entropy of power spectrum
        - phase_std: Standard deviation of instantaneous phase

    Note: fMRI frequency bands differ from EEG:
    - delta: 0.01 - 0.027 Hz (Slow-5)
    - theta: 0.027 - 0.073 Hz (Slow-4)
    - alpha: 0.073 - 0.15 Hz (Slow-3 lower)
    - beta:  0.15 - 0.20 Hz (Slow-3 upper)
    - gamma: 0.20 - 0.25 Hz (Slow-2)
    """
    if bands is None:
        bands = ACTIVE_FREQ_BANDS  # Imported from config.py — single source of truth

    # Nyquist-safe adjustment: keep feature shape stable while avoiding aliasing warning spam.
    global _NYQUIST_NOTE_EMITTED
    nyquist = fs / 2.0
    nyquist_eps = nyquist - NYQUIST_EPS
    unreliable_bands = _get_unreliable_bands(fs)
    safe_bands = {}
    for band_name, (low, high) in bands.items():
        if band_name in unreliable_bands and high >= nyquist:
            safe_bands[band_name] = (0.0, 0.0)
            continue
        safe_low = max(0.0, low)
        safe_high = min(high, nyquist_eps)
        if safe_low >= safe_high:
            safe_bands[band_name] = (0.0, 0.0)
        else:
            safe_bands[band_name] = (safe_low, safe_high)

    if not _NYQUIST_NOTE_EMITTED and "gamma" in bands and "gamma" in unreliable_bands:
        tr = 1.0 / fs
        logger.warning(
            "Gamma band marked unreliable at Nyquist for TR=%.1fs; gamma features are zeroed.",
            tr,
        )
        _NYQUIST_NOTE_EMITTED = True

    bands = safe_bands

    # Validate input
    if len(ts) < 10 or np.isnan(ts).any() or np.isinf(ts).any():
        return _get_zero_features(bands)

    # Remove mean and detrend
    ts_centered = ts - np.mean(ts)

    # Compute power spectral density using Welch's method
    try:
        nperseg = min(256, len(ts_centered))
        freqs, psd = welch(
            ts_centered, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, window="hann", scaling="density"
        )
    except Exception:
        return _get_zero_features(bands)

    # Initialize feature dictionary
    features = {}

    # Extract band-specific features
    for band_name, (low, high) in bands.items():
        band_mask = (freqs >= low) & (freqs < high)

        if not np.any(band_mask):
            features[f"{band_name}_power"] = 0.0
            features[f"{band_name}_peak_freq"] = 0.0
            continue

        # Total power in band
        band_power = np.trapz(psd[band_mask], freqs[band_mask])
        features[f"{band_name}_power"] = float(band_power)

        # Peak frequency
        peak_idx = np.argmax(psd[band_mask])
        features[f"{band_name}_peak_freq"] = float(freqs[band_mask][peak_idx])

    # Global spectral features
    # 1. Spectral entropy (complexity measure)
    psd_normalized = psd / np.sum(psd) if np.sum(psd) > 0 else psd
    spectral_entropy = entropy(psd_normalized + 1e-12)  # Add epsilon for stability
    features["spectral_entropy"] = float(spectral_entropy)

    # 2. Phase coherence via Hilbert transform
    try:
        analytic_signal = hilbert(ts_centered)
        instantaneous_phase = np.angle(analytic_signal)
        phase_std = np.std(np.diff(instantaneous_phase))
        features["phase_std"] = float(phase_std)
    except Exception:
        features["phase_std"] = 0.0

    return features


def extract_frequency_features_batch(ts_matrix: np.ndarray, fs: float = 0.5) -> np.ndarray:
    """
    Extract frequency features for multiple ROIs in parallel.

    Args:
        ts_matrix: Time series matrix (shape: [timepoints, n_rois])
        fs: Sampling frequency in Hz

    Returns:
        Feature matrix (shape: [n_rois, 12])
        Columns: [delta_power, delta_peak, theta_power, theta_peak, ..., spectral_entropy, phase_std]
    """
    n_rois = ts_matrix.shape[1]
    feature_names = []
    for band in ACTIVE_FREQ_BANDS:
        feature_names.extend([f"{band}_power", f"{band}_peak_freq"])
    feature_names.extend(["spectral_entropy", "phase_std"])
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


# ============================================================================
# TIME-DOMAIN FEATURE EXTRACTION
# ============================================================================


def calculate_psd(ts: np.ndarray, tr: float) -> float:
    """
    Calculates the mean Power Spectral Density in the 0.01-0.1Hz band.

    Uses log-compression on the raw band power to avoid hard saturation.
    A fixed hard clip can collapse informative between-subject differences when
    many subjects hit the cap (observed in forensic audit for several *_psd
    channels). log1p preserves ordering while keeping scale bounded.
    """
    if len(ts) < 10 or np.all(ts == 0):
        return 0.0
    ts = ts - np.mean(ts)
    psd = np.abs(np.fft.fft(ts)) ** 2
    freqs = np.fft.fftfreq(len(ts), d=tr)
    mask = (freqs > 0.01) & (freqs < 0.1)
    if not np.any(mask):
        return 0.0

    psd_mean = float(np.mean(psd[mask]))
    if not np.isfinite(psd_mean) or psd_mean <= 0:
        return 0.0

    # Soft-compress heavy-tailed PSD values while preserving rank information.
    return float(np.log1p(psd_mean))


def calculate_autocorr(ts: np.ndarray, lag: int = 1) -> float:
    """Autocorrelation at specified lag (default lag=1 for temporal persistence)."""
    if len(ts) < lag + 1:
        return 0.0
    ts = ts - np.mean(ts)
    c0 = np.dot(ts, ts) / len(ts)
    c_lag = np.dot(ts[:-lag], ts[lag:]) / len(ts)
    return float(c_lag / c0) if c0 > 0 else 0.0


def calculate_band_power(ts: np.ndarray, tr: float, freq_band: tuple) -> float:
    """
    Compute relative power in specific frequency band using Welch's method.

    Frequency bands for brain analysis:
    - Delta (0.5-4 Hz): Deep sleep, unconscious processes
    - Theta (4-8 Hz): Drowsiness, creativity, meditation
    - Alpha (8-13 Hz): Relaxation, attention
    - Beta (13-30 Hz): Active thinking, problem solving
    """
    if len(ts) < 10 or np.std(ts) < 1e-6:
        return 0.0

    fs = 1.0 / tr
    freqs, psd = welch(ts, fs=fs, nperseg=min(len(ts), 64))

    band_mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
    if not np.any(band_mask):
        return 0.0

    band_power = np.trapz(psd[band_mask], freqs[band_mask])
    total_power = np.trapz(psd, freqs)

    return float(band_power / total_power) if total_power > 0 else 0.0


def extract_single_roi_features(ts: np.ndarray, tr: float, include_frequency: bool = True):
    """
    Computes temporal metrics for one ROI.

    Base features (8): mean, std, skew, kurtosis, psd, mssd, range, autocorr
    Frequency features (12): 5 bands x (power + peak_freq) + spectral_entropy + phase_std

    Args:
        ts: Time series signal
        tr: Repetition time
        include_frequency: If True, adds 12 frequency features (20 total)

    Returns:
        List of features (8 or 20 depending on include_frequency)
    """
    if not np.isfinite(ts).all() or np.std(ts) < 1e-6:
        n_features = len(FEATURE_GROUPS["temporal"]) + (
            len(FEATURE_GROUPS["frequency"]) if include_frequency else 0
        )
        return [0.0] * n_features

    # Compute statistics with bounds checking to prevent extreme outliers
    # from skew/kurtosis on near-constant or pathological distributions
    mean_val = float(np.mean(ts))
    std_val = float(np.std(ts))
    
    # Clip skewness and kurtosis to [-1e3, 1e3] to prevent extreme outliers
    # fMRI signals are typically moderate in shape parameters
    skew_val = np.clip(float(skew(ts, bias=False)), -1e3, 1e3)
    kurt_val = np.clip(float(kurtosis(ts, bias=False)), -1e3, 1e3)
    
    base_features = [
        mean_val,
        std_val,
        skew_val,
        kurt_val,
        calculate_psd(ts, tr),
        float(np.mean(np.diff(ts) ** 2)),
        float(np.max(ts) - np.min(ts)),
        calculate_autocorr(ts, lag=1),
    ]

    if include_frequency:
        fs = 1.0 / tr
        freq_features = extract_band_power(ts, fs=fs)

        frequency_values = []
        for feat_name in FEATURE_GROUPS["frequency"]:
            if feat_name.endswith("_peak"):
                band = feat_name[:-5]
                frequency_values.append(float(freq_features.get(f"{band}_peak_freq", 0.0)))
            else:
                frequency_values.append(float(freq_features.get(feat_name, 0.0)))

        return base_features + frequency_values

    return base_features


def _extract_temporal_vectorized(
    ts_data: np.ndarray,
    tr: float,
    add_frequency: bool = True,
) -> np.ndarray:
    """
    Fully vectorized temporal feature extraction for all ROIs at once.
    Processes all 170 ROIs in parallel using NumPy/SciPy vectorized operations.

    Args:
        ts_data: Time series array of shape (time_points, n_rois)
        tr: Repetition time
        add_frequency: Whether to include frequency features

    Returns:
        Array of shape (n_rois, features_per_roi) with temporal + frequency features
    """
    n_timepoints, n_rois = ts_data.shape

    temporal_feature_names = list(FEATURE_GROUPS["temporal"])
    freq_feature_names = list(FEATURE_GROUPS["frequency"]) if add_frequency else []
    n_temporal = len(temporal_feature_names)
    n_freq = len(freq_feature_names)
    features_per_roi = n_temporal + n_freq

    ts_clean = ts_data.copy()
    std_vals = np.std(ts_clean, axis=0, keepdims=True)

    bad_rois = (std_vals < 1e-6).flatten() | ~np.all(np.isfinite(ts_clean), axis=0)
    ts_clean[:, bad_rois] = 0.0

    clip_bounds = 1e3
    mean_vals = np.mean(ts_clean, axis=0)
    std_vals = np.std(ts_clean, axis=0)
    skew_vals = np.clip(skew(ts_clean, bias=False, axis=0), -clip_bounds, clip_bounds)
    kurt_vals = np.clip(kurtosis(ts_clean, bias=False, axis=0), -clip_bounds, clip_bounds)

    ts_centered = ts_clean - mean_vals
    psd_vals = _compute_psd_vectorized(ts_centered, tr)
    psd_vals = np.log1p(psd_vals)

    mssd_vals = np.mean(np.diff(ts_clean, axis=0) ** 2, axis=0)
    range_vals = np.ptp(ts_clean, axis=0)
    autocorr_vals = _compute_autocorr_vectorized(ts_clean)

    base_features = np.stack([
        mean_vals,
        std_vals,
        skew_vals,
        kurt_vals,
        psd_vals,
        mssd_vals,
        range_vals,
        autocorr_vals,
    ], axis=1)

    output_features = base_features.copy()

    if add_frequency:
        fs = 1.0 / tr
        safe_bands = {}
        nyquist = fs / 2.0
        nyquist_eps = nyquist - NYQUIST_EPS

        for band_name, (low, high) in ACTIVE_FREQ_BANDS.items():
            if high >= nyquist:
                safe_bands[band_name] = (low, high)
            else:
                safe_low = max(0.0, low)
                safe_high = min(high, nyquist_eps)
                if safe_low >= safe_high:
                    safe_bands[band_name] = (0.0, 0.0)
                else:
                    safe_bands[band_name] = (safe_low, safe_high)

        n_bands = len(ACTIVE_FREQ_BANDS)
        freq_feature_arr = np.zeros((n_rois, n_freq))

        try:
            nperseg = min(256, n_timepoints)
            freqs_full, psd_full = welch(
                ts_clean, fs=fs, nperseg=nperseg, noverlap=nperseg // 2,
                window="hann", scaling="density", axis=0
            )
        except Exception:
            return np.zeros((n_rois, features_per_roi))

        total_power = np.trapz(psd_full, freqs_full, axis=0)
        total_power = np.where(total_power > 0, total_power, 1.0)

        ordered_bands = ["delta", "theta", "alpha", "beta", "gamma"]
        for idx, band_name in enumerate(ordered_bands):
            if band_name not in safe_bands:
                continue
            low, high = safe_bands[band_name]
            if low >= high:
                continue
            band_mask = (freqs_full >= low) & (freqs_full < high)
            band_power = np.trapz(psd_full[band_mask, :], freqs_full[band_mask], axis=0)
            freq_feature_arr[:, idx] = band_power / total_power

            freq_feature_arr[:, idx + n_bands] = _compute_peak_freqs_vectorized(psd_full, freqs_full, band_mask)

        freq_feature_arr[:, 2 * n_bands] = _compute_spectral_entropy_vectorized(psd_full, total_power)
        freq_feature_arr[:, 2 * n_bands + 1] = _compute_phase_std_vectorized(ts_clean)

        output_features = np.concatenate([base_features, freq_feature_arr], axis=1)

    output_features[:, bad_rois] = 0.0

    return output_features


def _compute_psd_vectorized(ts_centered: np.ndarray, tr: float) -> np.ndarray:
    """Vectorized PSD: mean power in 0.01-0.1 Hz band via FFT."""
    n_timepoints, n_rois = ts_centered.shape
    freqs = np.fft.fftfreq(n_timepoints, d=tr)
    mask = (freqs > 0.01) & (freqs < 0.1)
    if not np.any(mask):
        return np.zeros(n_rois)

    fft_vals = np.fft.fft(ts_centered, axis=0)
    psd = np.abs(fft_vals) ** 2
    psd_mean = np.mean(psd[mask, :], axis=0)
    psd_mean = np.where(np.isfinite(psd_mean) & (psd_mean > 0), psd_mean, 0.0)
    return psd_mean


def _compute_autocorr_vectorized(ts: np.ndarray, lag: int = 1) -> np.ndarray:
    """Vectorized autocorrelation at lag-1 across all ROIs."""
    n_timepoints, n_rois = ts.shape
    if n_timepoints <= lag:
        return np.zeros(n_rois)

    ts_centered = ts - np.mean(ts, axis=0, keepdims=True)
    c0 = np.sum(ts_centered ** 2, axis=0) / n_timepoints
    c_lag = np.sum(ts_centered[:-lag, :] * ts_centered[lag:, :], axis=0) / n_timepoints

    c0 = np.where(c0 > 0, c0, 1.0)
    return c_lag / c0


def _compute_peak_freqs_vectorized(
    psd_full: np.ndarray,
    freqs_full: np.ndarray,
    band_mask: np.ndarray,
) -> np.ndarray:
    """Vectorized peak frequency extraction within band."""
    if not np.any(band_mask):
        return np.zeros(psd_full.shape[1])

    psd_band = psd_full[band_mask, :]
    freqs_band = freqs_full[band_mask]
    peak_indices = np.argmax(psd_band, axis=0)
    peak_freqs = freqs_band[peak_indices]
    peak_freqs = np.where(np.isfinite(peak_freqs), peak_freqs, 0.0)
    return peak_freqs


def _compute_spectral_entropy_vectorized(
    psd_full: np.ndarray,
    total_power: np.ndarray,
) -> np.ndarray:
    """Vectorized spectral entropy across all ROIs."""
    psd_norm = psd_full / total_power
    psd_norm = np.where(psd_norm > 0, psd_norm, np.nan)
    spectral_entropy = -np.nansum(psd_norm * np.log(psd_norm + 1e-10), axis=0)
    spectral_entropy = np.where(np.isfinite(spectral_entropy), spectral_entropy, 0.0)
    return spectral_entropy


def _compute_phase_std_vectorized(ts: np.ndarray) -> np.ndarray:
    """Vectorized instantaneous phase std via Hilbert transform."""
    n_timepoints, n_rois = ts.shape
    analytic = hilbert(ts, axis=0)
    instantaneous_phase = np.angle(analytic)
    phase_std = np.std(instantaneous_phase, axis=0)
    phase_std = np.where(np.isfinite(phase_std), phase_std, 0.0)
    return phase_std


def _process_single_subject(
    sub_id: str,
    split: str,
    tr: float,
    ts_path: Path,
    features_per_roi: int,
    add_frequency: bool,
    use_gpu: bool = False,
    max_rois: int = MAX_ROIS,
) -> Optional[List]:
    """
    Process a single subject's time series to extract temporal features.
    Uses vectorized extraction when possible, falls back to per-ROI loop on error.

    Args:
        sub_id: Subject ID
        split: Data split (train/val/test)
        tr: Repetition time
        ts_path: Path to time series .npy file
        features_per_roi: Number of features per ROI
        add_frequency: Whether to include frequency features
        use_gpu: Whether to use GPU acceleration
        max_rois: Maximum number of ROIs

    Returns:
        List of [sub_id, feat1, feat2, ...] or None if failed
    """
    try:
        ts_data = np.load(ts_path)
        if ts_data.size == 0:
            return None

        original_num_rois = ts_data.shape[1]
        if original_num_rois != max_rois:
            return None

        try:
            roi_features = _extract_temporal_vectorized(ts_data, tr, add_frequency=add_frequency)
            if roi_features.shape[0] != max_rois or roi_features.shape[1] != features_per_roi:
                raise ValueError(f"Shape mismatch: {roi_features.shape}")
        except Exception:
            roi_features = np.zeros((max_rois, features_per_roi))
            for i in range(max_rois):
                roi_signal = ts_data[:, i]
                try:
                    roi_feats = extract_single_roi_features(roi_signal, tr, include_frequency=add_frequency)
                    roi_features[i] = roi_feats
                except Exception:
                    roi_features[i] = [0.0] * features_per_roi

        subject_features = [sub_id]
        for i in range(max_rois):
            subject_features.extend(roi_features[i].tolist())

        return subject_features

    except Exception:
        return None


def main(add_frequency: bool = True, n_jobs: int = -1, use_gpu: bool = False) -> None:
    """Extract temporal features for all subjects.

    Args:
        add_frequency: Include frequency features (20 vs 8 per ROI)
        n_jobs: Number of parallel workers (-1 = all cores, default: -1)
    """
    if not MASTER_MANIFEST.exists():
        logger.error("Master manifest missing. Run manifestor.py first.")
        return

    try:
        manifest = pd.read_csv(MASTER_MANIFEST)
    except FileNotFoundError:
        logger.error(f"File not found: {MASTER_MANIFEST}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing failed for {MASTER_MANIFEST}: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load manifest: {e}")
        raise

    features_per_roi = len(FEATURE_GROUPS["temporal"]) + (
        len(FEATURE_GROUPS["frequency"]) if add_frequency else 0
    )
    gpu_info = "GPU" if use_gpu else "CPU"
    logger.info(f"Extracting temporal features for {len(manifest)} subjects (n_jobs={n_jobs}, device={gpu_info})...")
    logger.info(
        f"Features per ROI: {features_per_roi} ({'with' if add_frequency else 'without'} frequency features)"
    )

    subject_tasks = []
    for _, row in manifest.iterrows():
        sub_id = str(row["subject_id"])
        split = row["split"]
        tr = row.get("TR", DEFAULT_TR)
        if pd.isna(tr) or tr <= 0:
            tr = DEFAULT_TR

        ts_path = DATA_FINAL / split / "time_series" / f"{sub_id}_ts.npy"
        if ts_path.exists():
            subject_tasks.append((sub_id, split, tr, ts_path))

    logger.info(f"Found {len(subject_tasks)} valid subjects to process")

    results = Parallel(n_jobs=n_jobs, prefer="threads", verbose=0)(
        delayed(_process_single_subject)(
            sub_id, split, tr, ts_path, features_per_roi, add_frequency, use_gpu
        )
        for sub_id, split, tr, ts_path in tqdm(subject_tasks, desc="Processing", mininterval=10.0)
    )

    all_subject_data = [r for r in results if r is not None]
    failed_count = len(results) - len(all_subject_data)

    if failed_count > 0:
        logger.warning(f"Failed to process {failed_count} subjects")

    if not all_subject_data:
        logger.error("No valid subjects processed!")
        return

    # Create columns
    columns = ["subject_id"]
    stats = list(FEATURE_GROUPS["temporal"])
    if add_frequency:
        stats.extend(FEATURE_GROUPS["frequency"])

    for r in range(1, 171):
        for s in stats:
            columns.append(f"roi{r}_{s}")

    try:
        df = pd.DataFrame(all_subject_data, columns=columns)
        df.to_csv(NODE_ATTRIBUTES_TEMPORAL, index=False)
        logger.info(f"Saved temporal features to {NODE_ATTRIBUTES_TEMPORAL}")
    except Exception as e:
        logger.error(f"Failed to save temporal features: {e}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Temporal feature extraction")
    parser.add_argument(
        "--add-frequency",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-frequency",
        action="store_true",
        default=False,
        help="Disable frequency features (default keeps frequency features enabled)",
    )
    parser.add_argument("--n-jobs", type=int, default=-1, help="Number of parallel workers (-1=all cores)")
    parser.add_argument("--use-gpu", action="store_true", default=False, help="Use GPU acceleration if available")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(add_frequency=not args.no_frequency, n_jobs=args.n_jobs, use_gpu=args.use_gpu)
