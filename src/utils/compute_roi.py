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
from src.config import (
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

def extract_single_roi_features(ts: np.ndarray, tr: float):
    """Computes 6 standardized temporal metrics for one ROI."""
    if not np.isfinite(ts).all() or np.std(ts) < 1e-6:
        return [0.0] * 6 # Return zeros for dead signals to prevent NaN propagation

    return [
        float(np.mean(ts)),
        float(np.std(ts)),
        float(skew(ts, bias=False)),
        float(kurtosis(ts, bias=False)),
        calculate_psd(ts, tr),
        float(np.mean(np.diff(ts) ** 2)) # MSSD (Mean Squared Successive Difference)
    ]

def main():
    if not MASTER_MANIFEST.exists():
        logger.error("Master manifest missing. Run manifest.py first.")
        return

    manifest = pd.read_csv(MASTER_MANIFEST)
    all_subject_data = []
    
    logger.info(f"🚀 Extracting temporal features for {len(manifest)} subjects...")

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Subjects"):
        sub_id = str(row["subject_id"])
        split = row["split"]
        tr = row.get("TR", DEFAULT_TR)
        if pd.isna(tr) or tr <= 0: tr = DEFAULT_TR

        ts_path = DATA_FINAL / split / "time_series" / f"{sub_id}_ts.npy"
        if not ts_path.exists():
            continue

        try:
            ts_data = np.load(ts_path) # Expected Shape: (Timepoints, ROIs)
            num_rois = ts_data.shape[1]
            
            subject_features = [sub_id]
            
            for i in range(num_rois):
                roi_signal = ts_data[:, i]
                roi_feats = extract_single_roi_features(roi_signal, tr)
                subject_features.extend(roi_feats)
            
            all_subject_data.append(subject_features)

        except Exception as e:
            logger.error(f"Error processing {sub_id}: {e}")

    # Create Dynamic Column Names
    # We use a flat list: roi1_mean, roi1_std... roiN_mssd
    first_sub_roi_count = (len(all_subject_data[0]) - 1) // 6
    columns = ['subject_id']
    stats = ["mean", "std", "skew", "kurt", "psd", "mssd"]
    for r in range(1, first_sub_roi_count + 1):
        for s in stats:
            columns.append(f"roi{r}_{s}")

    df = pd.DataFrame(all_subject_data, columns=columns)
    
    # Final cleanup: ensure no NaNs remain before saving
    # (NeuroCombat will fail if NaNs exist)
    df = df.fillna(0.0)

    NODE_ATTRIBUTES_TEMPORAL.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(NODE_ATTRIBUTES_TEMPORAL, index=False)
    
    logger.info(f"✅ Extracted features for {len(df)} subjects.")
    logger.info(f"ROI Count: {first_sub_roi_count} | Features per subject: {len(columns)-1}")
    logger.info(f"Output saved to: {NODE_ATTRIBUTES_TEMPORAL}")

if __name__ == "__main__":
    main()