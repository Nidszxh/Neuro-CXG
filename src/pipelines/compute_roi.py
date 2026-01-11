import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from pathlib import Path
from tqdm import tqdm

# --- CONFIG ---
PROJECT_ROOT = Path("./data")
# We use the .npy files from the split directories
DATASET_ROOT = PROJECT_ROOT / "final"
MANIFEST_PATH = PROJECT_ROOT / "metadata" / "master_manifest.csv"
OUTPUT_PATH = PROJECT_ROOT / "metadata" / "node_attributes_temporal.csv"

# TR is essential for PSD calculation (ABIDE usually ~2.0s)
DEFAULT_TR = 2.0 

def calculate_psd(ts, tr):
    """Calculates Power Spectral Density in the 0.01-0.1Hz band (Phase 4.2)."""
    n = len(ts)
    if n < 10: return 0
    freqs = np.fft.fftfreq(n, d=tr)
    psd = np.abs(np.fft.fft(ts - np.mean(ts)))**2
    # Filter for typical resting-state fluctuations
    mask = (freqs > 0.01) & (freqs < 0.1)
    return np.mean(psd[mask]) if np.any(mask) else 0

def extract_features_from_npy(ts_array, tr):
    """Extracts Q1-standard features for each ROI."""
    # ts_array shape: (timepoints, 170 ROIs)
    feats = []
    for i in range(ts_array.shape[1]):
        roi_ts = ts_array[:, i]
        feats.append({
            'mean': np.mean(roi_ts),
            'std': np.std(roi_ts),
            'skew': skew(roi_ts),
            'kurt': kurtosis(roi_ts),
            'psd': calculate_psd(roi_ts, tr),
            'mssd': np.mean(np.diff(roi_ts)**2) # Successive difference (Variability)
        })
    return feats

def main():
    manifest = pd.read_csv(MANIFEST_PATH)
    all_features = []

    print(f"🚀 Extracting temporal attributes from {len(manifest)} subjects...")

    for _, row in tqdm(manifest.iterrows(), total=len(manifest)):
        sub_id = row['subject_id']
        split = row['split']
        tr = row.get('TR', DEFAULT_TR)
        
        ts_path = DATASET_ROOT / split / "time_series" / f"{sub_id}_ts.npy"
        
        if not ts_path.exists():
            continue
            
        # Load pre-extracted AAL3 time series
        ts_data = np.load(ts_path)
        
        # We consolidate 170 ROIs into the 5 Lobe-based features for the Causal Graph
        # (Phase 7.1: Frontal, Temporal, Parietal, Occipital, Limbic)
        # For simplicity in this step, we aggregate the features
        roi_features = extract_features_from_npy(ts_data, tr)
        
        # Convert to a flat dictionary for this subject
        sub_entry = {'subject_id': sub_id}
        for i, f in enumerate(roi_features):
            # We save every ROI feature to keep the GNN flexible
            for k, v in f.items():
                sub_entry[f'roi{i}_{k}'] = v
        
        all_subject_features.append(sub_entry)

    # Save final attributes
    df = pd.DataFrame(all_subject_features)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved temporal attributes for {len(df)} subjects to {OUTPUT_PATH}")

if __name__ == "__main__":
    all_subject_features = []
    main()