import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import welch, hilbert
from tqdm import tqdm

# Setup paths and config
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import (
    DATA_FINAL,
    MASTER_MANIFEST,
    NODE_ATTRIBUTES_TEMPORAL,
    DEFAULT_TR,
)

# Expected ROI count range for validation (AAL3v1 atlas)
# Note: Some AAL3v1 templates have 2 unused/empty ROIs, so 164-170 are all valid
VALID_ROI_RANGE = (164, 170)
MAX_ROIS = 170

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


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


def extract_band_power(
    ts: np.ndarray, fs: float = 0.5, bands: Dict[str, Tuple[float, float]] = None
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

    Note: fMRI frequency bands differ from EEG:
    - delta: 0.01 - 0.027 Hz (Slow-5)
    - theta: 0.027 - 0.073 Hz (Slow-4)
    - alpha: 0.073 - 0.15 Hz (Slow-3 lower)
    - beta:  0.15 - 0.20 Hz (Slow-3 upper)
    - gamma: 0.20 - 0.25 Hz (Slow-2)
    """
    if bands is None:
        bands = {
            "delta": (0.01, 0.027),
            "theta": (0.027, 0.073),
            "alpha": (0.073, 0.15),
            "beta": (0.15, 0.20),
            "gamma": (0.20, 0.25),
        }

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
    feature_names = [
        "delta_power",
        "delta_peak_freq",
        "theta_power",
        "theta_peak_freq",
        "alpha_power",
        "alpha_peak_freq",
        "beta_power",
        "beta_peak_freq",
        "gamma_power",
        "gamma_peak_freq",
        "spectral_entropy",
        "phase_std",
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


# ============================================================================
# TIME-DOMAIN FEATURE EXTRACTION
# ============================================================================


def calculate_psd(ts: np.ndarray, tr: float) -> float:
    """Calculates the mean Power Spectral Density in the 0.01-0.1Hz band."""
    if len(ts) < 10 or np.all(ts == 0):
        return 0.0
    ts = ts - np.mean(ts)
    psd = np.abs(np.fft.fft(ts)) ** 2
    freqs = np.fft.fftfreq(len(ts), d=tr)
    mask = (freqs > 0.01) & (freqs < 0.1)
    return float(np.mean(psd[mask])) if np.any(mask) else 0.0


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
        n_features = 20 if include_frequency else 8
        return [0.0] * n_features

    base_features = [
        float(np.mean(ts)),
        float(np.std(ts)),
        float(skew(ts, bias=False)),
        float(kurtosis(ts, bias=False)),
        calculate_psd(ts, tr),
        float(np.mean(np.diff(ts) ** 2)),
        float(np.max(ts) - np.min(ts)),
        calculate_autocorr(ts, lag=1),
    ]

    if include_frequency:
        fs = 1.0 / tr
        freq_features = extract_band_power(ts, fs=fs)

        frequency_values = [
            freq_features["delta_power"],
            freq_features["delta_peak_freq"],
            freq_features["theta_power"],
            freq_features["theta_peak_freq"],
            freq_features["alpha_power"],
            freq_features["alpha_peak_freq"],
            freq_features["beta_power"],
            freq_features["beta_peak_freq"],
            freq_features["gamma_power"],
            freq_features["gamma_peak_freq"],
            freq_features["spectral_entropy"],
            freq_features["phase_std"],
        ]

        return base_features + frequency_values

    return base_features


def main(add_frequency: bool = True) -> None:
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

    all_subject_data = []
    failed_subjects = []

    features_per_roi = 20 if add_frequency else 8
    logger.info(f"Extracting temporal features for {len(manifest)} subjects...")
    logger.info(
        f"Features per ROI: {features_per_roi} ({'with' if add_frequency else 'without'} frequency features)"
    )

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Subjects"):
        sub_id = str(row["subject_id"])
        split = row["split"]
        tr = row.get("TR", DEFAULT_TR)
        if pd.isna(tr) or tr <= 0:
            tr = DEFAULT_TR

        ts_path = DATA_FINAL / split / "time_series" / f"{sub_id}_ts.npy"
        if not ts_path.exists():
            logger.debug(f"Time series not found for {sub_id}")
            continue

        try:
            ts_data = np.load(ts_path)
            if ts_data.size == 0:
                logger.warning(f"{sub_id}: Empty time series array")
                failed_subjects.append(sub_id)
                continue

            original_num_rois = ts_data.shape[1]

            # abide_download.py now saves fixed 170-column arrays
            if original_num_rois != MAX_ROIS:
                logger.error(
                    f"{sub_id}: Expected {MAX_ROIS} ROIs but got {original_num_rois}. "
                    f"Re-run abide_download.py to regenerate standardized time series."
                )
                failed_subjects.append(sub_id)
                continue

            subject_features = [sub_id]

            for i in range(MAX_ROIS):
                roi_signal = ts_data[:, i]
                try:
                    roi_feats = extract_single_roi_features(roi_signal, tr, include_frequency=add_frequency)
                    subject_features.extend(roi_feats)
                except Exception as e:
                    logger.warning(f"{sub_id} ROI {i}: Feature extraction failed: {e}")
                    subject_features.extend([0.0] * features_per_roi)

            all_subject_data.append(subject_features)

        except FileNotFoundError:
            logger.error(f"Time series file not found: {ts_path}")
            failed_subjects.append(sub_id)
        except ValueError as e:
            logger.error(f"{sub_id}: Invalid array format: {e}")
            failed_subjects.append(sub_id)
        except Exception as e:
            logger.error(f"Error processing {sub_id}: {e}")
            failed_subjects.append(sub_id)

    if failed_subjects:
        logger.warning(f"Failed to process {len(failed_subjects)} subjects: {failed_subjects[:5]}...")

    if not all_subject_data:
        logger.error("No valid subjects processed!")
        return

    # Create columns
    columns = ["subject_id"]
    stats = ["mean", "std", "skew", "kurt", "psd", "mssd", "range", "autocorr"]
    if add_frequency:
        stats.extend(
            [
                "delta_power",
                "delta_peak",
                "theta_power",
                "theta_peak",
                "alpha_power",
                "alpha_peak",
                "beta_power",
                "beta_peak",
                "gamma_power",
                "gamma_peak",
                "spectral_entropy",
                "phase_std",
            ]
        )

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
    parser.add_argument("--add-frequency", action="store_true", help="Include frequency features")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(add_frequency=args.add_frequency)
