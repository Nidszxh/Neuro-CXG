import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from tqdm import tqdm

# Setup paths and config
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import (
    DATA_FINAL,
    MASTER_MANIFEST,
    NODE_ATTRIBUTES_TEMPORAL,
    DEFAULT_TR
)

# Expected ROI count range for validation (AAL3v1 atlas)
# Note: Some AAL3v1 templates have 2 unused/empty ROIs, so 164-170 are all valid
VALID_ROI_RANGE = (164, 170)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def calculate_psd(ts: np.ndarray, tr: float) -> float:
    """Calculates the mean Power Spectral Density in the 0.01-0.1Hz band."""
    if len(ts) < 10 or np.all(ts == 0):
        return 0.0
    ts = ts - np.mean(ts)
    psd = np.abs(np.fft.fft(ts)) ** 2
    freqs = np.fft.fftfreq(len(ts), d=tr)
    # Focus on the low-frequency fluctuations typical of fMRI (0.01 - 0.1 Hz)
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
    
    These are established ASD biomarkers in neuroimaging literature.
    """
    if len(ts) < 10 or np.std(ts) < 1e-6:
        return 0.0
    
    fs = 1.0 / tr  # Sampling frequency
    freqs, psd = welch(ts, fs=fs, nperseg=min(len(ts), 64))
    
    # Extract power in specified band
    band_mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
    if not np.any(band_mask):
        return 0.0
    
    band_power = np.trapz(psd[band_mask], freqs[band_mask])
    total_power = np.trapz(psd, freqs)
    
    return float(band_power / total_power) if total_power > 0 else 0.0

def extract_single_roi_features(ts: np.ndarray, tr: float, include_bands: bool = False):
    """
    Computes temporal metrics for one ROI.
    
    Base features (8): mean, std, skew, kurtosis, psd, mssd, range, autocorr
    Optional band features (4): delta, theta, alpha, beta power
    
    Args:
        ts: Time series signal
        tr: Repetition time
        include_bands: If True, adds 4 frequency band features (12 total)
    """
    if not np.isfinite(ts).all() or np.std(ts) < 1e-6:
        n_features = 12 if include_bands else 8
        return [0.0] * n_features

    base_features = [
        float(np.mean(ts)),
        float(np.std(ts)),
        float(skew(ts, bias=False)),
        float(kurtosis(ts, bias=False)),
        calculate_psd(ts, tr),
        float(np.mean(np.diff(ts) ** 2)),  # MSSD
        float(np.max(ts) - np.min(ts)),     # Range
        calculate_autocorr(ts, lag=1)       # Autocorr
    ]
    
    if include_bands:
        # Add frequency band powers (ASD biomarkers)
        band_features = [
            calculate_band_power(ts, tr, (0.5, 4)),    # Delta
            calculate_band_power(ts, tr, (4, 8)),      # Theta
            calculate_band_power(ts, tr, (8, 13)),     # Alpha
            calculate_band_power(ts, tr, (13, 30))     # Beta
        ]
        return base_features + band_features
    
    return base_features

def main(add_bands: bool = False):
    if not MASTER_MANIFEST.exists():
        logger.error("Master manifest missing. Run manifest.py first.")
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
    
    features_per_roi = 12 if add_bands else 8
    logger.info(f"🚀 Extracting temporal features for {len(manifest)} subjects...")
    logger.info(f"Features per ROI: {features_per_roi} ({'with' if add_bands else 'without'} frequency bands)")

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Subjects"):
        sub_id = str(row["subject_id"])
        split = row["split"]
        tr = row.get("TR", DEFAULT_TR)
        if pd.isna(tr) or tr <= 0: tr = DEFAULT_TR

        ts_path = DATA_FINAL / split / "time_series" / f"{sub_id}_ts.npy"
        if not ts_path.exists():
            logger.debug(f"Time series not found for {sub_id}")
            continue

        try:
            ts_data = np.load(ts_path) # Expected Shape: (Timepoints, ROIs)
            if ts_data.size == 0:
                logger.warning(f"{sub_id}: Empty time series array")
                failed_subjects.append(sub_id)
                continue
            
            num_rois = ts_data.shape[1]
            
            # Validate ROI count within expected range (AAL3v1 variant: 164-170)
            if not (VALID_ROI_RANGE[0] <= num_rois <= VALID_ROI_RANGE[1]):
                logger.error(f"{sub_id}: ROI count {num_rois} outside valid range {VALID_ROI_RANGE}. Skipping.")
                failed_subjects.append(sub_id)
                continue
            
            # Log if less than 170 (informational)
            if num_rois < 170:
                logger.debug(f"{sub_id}: Using {num_rois} ROIs (AAL3v1 variant with unused ROIs)")
            
            subject_features = [sub_id]
            
            for i in range(num_rois):
                roi_signal = ts_data[:, i]
                try:
                    roi_feats = extract_single_roi_features(roi_signal, tr, include_bands=add_bands)
                    subject_features.extend(roi_feats)
                except Exception as e:
                    logger.warning(f"{sub_id} ROI {i}: Feature extraction failed: {e}")
                    subject_features.extend([0.0] * features_per_roi)  # Fallback: 8 features per ROI (mean, std, skew, kurt, psd, mssd, range, autocorr)
            
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

    # --- STANDARDIZE TO 170 ROIs (Handle AAL3v1 variants with fewer ROIs) ---
    # Subjects may have 164-170 ROIs; we pad to 170 for consistent downstream processing
    all_subject_data_normalized = []
    max_rois = 170
    features_per_roi = 12 if add_bands else 8
    expected_features = 1 + (max_rois * features_per_roi)  # subject_id + (170 * features_per_roi)
    
    for row in all_subject_data:
        if len(row) == expected_features:
            # Already has 170 ROIs, keep as-is
            all_subject_data_normalized.append(row)
        else:
            # Has fewer ROIs; pad with zeros to match 170 columns
            actual_rois = (len(row) - 1) // features_per_roi
            logger.debug(f"{row[0]}: Padding from {actual_rois} to {max_rois} ROIs")
            padded_row = row.copy() if isinstance(row, list) else list(row)
            # Append zeros for missing ROIs (features_per_roi values per missing ROI)
            padding_needed = (max_rois - actual_rois) * features_per_roi
            padded_row.extend([0.0] * padding_needed)
            all_subject_data_normalized.append(padded_row)
    
    all_subject_data = all_subject_data_normalized

    # Create Fixed Column Names (always 170 ROIs after normalization)
    columns = ['subject_id']
    stats = ["mean", "std", "skew", "kurt", "psd", "mssd", "range", "autocorr"]
    if add_bands:
        stats.extend(["delta", "theta", "alpha", "beta"])
    
    for r in range(1, 171):  # Always 170 ROIs, even if some are zero-padded
        for s in stats:
            columns.append(f"roi{r}_{s}")

    try:
        df = pd.DataFrame(all_subject_data, columns=columns)
        
        # Final cleanup: ensure no NaNs remain before saving
        # (NeuroCombat will fail if NaNs exist)
        df = df.fillna(0.0)

        NODE_ATTRIBUTES_TEMPORAL.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(NODE_ATTRIBUTES_TEMPORAL, index=False)
        
        logger.info(f"✅ Extracted features for {len(df)} subjects.")
        logger.info(f"ROI Count: 170 (standardized; subjects with <170 ROIs zero-padded) | Features per subject: {len(columns)-1}")
        logger.info(f"Output saved to: {NODE_ATTRIBUTES_TEMPORAL}")
    except Exception as e:
        logger.error(f"Failed to save output: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract temporal features from fMRI time series")
    parser.add_argument(
        "--add-bands",
        action="store_true",
        help="Add frequency band features (delta, theta, alpha, beta) - increases features from 8 to 12 per ROI"
    )
    args = parser.parse_args()
    
    if args.add_bands:
        logger.info("🎵 Including frequency band features (delta, theta, alpha, beta)")
        logger.info("This will increase features per ROI: 8 → 12")
    
    main(add_bands=args.add_bands)