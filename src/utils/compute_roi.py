import argparse
import json
import logging
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from tqdm import tqdm

# Setup paths and config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.core.config import (
    DATA_FINAL,
    MASTER_MANIFEST,
    NODE_ATTRIBUTES_TEMPORAL,
    DEFAULT_TR,
    NUM_TEMPORAL_FEATURES,
    NUM_LOBES
)

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

def extract_single_roi_features(ts: np.ndarray, tr: float):
    """Computes 8 standardized temporal metrics for one ROI."""
    if not np.isfinite(ts).all() or np.std(ts) < 1e-6:
        return [0.0] * 8 # Return zeros for dead signals to prevent NaN propagation

    return [
        float(np.mean(ts)),
        float(np.std(ts)),
        float(skew(ts, bias=False)),
        float(kurtosis(ts, bias=False)),
        calculate_psd(ts, tr),
        float(np.mean(np.diff(ts) ** 2)),  # MSSD (Mean Squared Successive Difference)
        float(np.max(ts) - np.min(ts)),     # Range (NEW: outlier sensitivity)
        calculate_autocorr(ts, lag=1)       # Autocorr at lag-1 (NEW: temporal persistence)
    ]

def main():
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
    
    logger.info(f"🚀 Extracting temporal features for {len(manifest)} subjects...")

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
            
            subject_features = [sub_id]
            
            for i in range(num_rois):
                roi_signal = ts_data[:, i]
                try:
                    roi_feats = extract_single_roi_features(roi_signal, tr)
                    subject_features.extend(roi_feats)
                except Exception as e:
                    logger.warning(f"{sub_id} ROI {i}: Feature extraction failed: {e}")
                    subject_features.extend([0.0] * 6)  # Fallback
            
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

    # Create Dynamic Column Names
    # We use a flat list: roi1_mean, roi1_std... roiN_autocorr
    first_sub_roi_count = (len(all_subject_data[0]) - 1) // 8
    columns = ['subject_id']
    stats = ["mean", "std", "skew", "kurt", "psd", "mssd", "range", "autocorr"]
    for r in range(1, first_sub_roi_count + 1):
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
        logger.info(f"ROI Count: {first_sub_roi_count} | Features per subject: {len(columns)-1}")
        logger.info(f"Output saved to: {NODE_ATTRIBUTES_TEMPORAL}")
    except Exception as e:
        logger.error(f"Failed to save output: {e}")
        raise

if __name__ == "__main__":
    main()