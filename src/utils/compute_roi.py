import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from pathlib import Path
from tqdm import tqdm
import sys

# Add src to path for config import
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    DATA_FINAL, MASTER_MANIFEST, NODE_ATTRIBUTES_TEMPORAL,
    DEFAULT_TR, NUM_TEMPORAL_FEATURES
)


def calculate_psd(ts, tr):
    """
    Calculates Power Spectral Density in the 0.01-0.1Hz band.
    This frequency range captures typical resting-state fluctuations.
    
    Args:
        ts: Time series array (1D)
        tr: Repetition time in seconds
    
    Returns:
        Mean PSD in the target frequency band
    """
    n = len(ts)
    if n < 10:
        return 0.0
    
    freqs = np.fft.fftfreq(n, d=tr)
    psd = np.abs(np.fft.fft(ts - np.mean(ts)))**2
    
    # Filter for resting-state frequency range
    mask = (freqs > 0.01) & (freqs < 0.1)
    return float(np.mean(psd[mask])) if np.any(mask) else 0.0


def extract_features_from_npy(ts_array, tr):
    """
    Extracts 6 temporal features for each ROI.
    
    Features:
    1. Mean: Average signal intensity
    2. Std: Signal variability
    3. Skew: Asymmetry of distribution
    4. Kurt: Tailedness of distribution
    5. PSD: Power in resting-state band
    6. MSSD: Mean squared successive difference (temporal variability)
    
    Args:
        ts_array: Array of shape (timepoints, n_rois)
        tr: Repetition time in seconds
    
    Returns:
        List of feature dictionaries, one per ROI
    """
    feats = []
    for i in range(ts_array.shape[1]):
        roi_ts = ts_array[:, i]
        
        # Handle potential issues with the time series
        if np.isnan(roi_ts).any() or np.isinf(roi_ts).any():
            print(f"Warning: Invalid values in ROI {i}, using zeros")
            roi_ts = np.nan_to_num(roi_ts, nan=0.0, posinf=0.0, neginf=0.0)
        
        feats.append({
            'mean': float(np.mean(roi_ts)),
            'std': float(np.std(roi_ts)),
            'skew': float(skew(roi_ts)) if len(roi_ts) > 2 else 0.0,
            'kurt': float(kurtosis(roi_ts)) if len(roi_ts) > 3 else 0.0,
            'psd': calculate_psd(roi_ts, tr),
            'mssd': float(np.mean(np.diff(roi_ts)**2))
        })
    return feats


def main():
    """Main execution function."""
    # Validate inputs
    if not MASTER_MANIFEST.exists():
        print(f"❌ Error: Master manifest not found at {MASTER_MANIFEST}")
        print("   Run manifest.py first!")
        return
    
    manifest = pd.read_csv(MASTER_MANIFEST)
    all_subject_features = []  # Fixed: Initialize the list
    
    print(f"🚀 Extracting temporal attributes from {len(manifest)} subjects...")
    
    processed_count = 0
    missing_count = 0
    error_count = 0
    
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Processing subjects"):
        sub_id = row['subject_id']
        split = row['split']
        tr = row.get('TR', DEFAULT_TR)
        
        # Validate TR
        if pd.isna(tr) or tr <= 0:
            tr = DEFAULT_TR
        
        ts_path = DATA_FINAL / split / "time_series" / f"{sub_id}_ts.npy"
        
        if not ts_path.exists():
            missing_count += 1
            continue
        
        try:
            # Load pre-extracted AAL time series
            ts_data = np.load(ts_path)
            
            # Validate shape
            if ts_data.ndim != 2:
                print(f"Warning: Unexpected shape for {sub_id}: {ts_data.shape}")
                error_count += 1
                continue
            
            # Extract features for all ROIs
            roi_features = extract_features_from_npy(ts_data, tr)
            
            # Convert to flat dictionary for this subject
            sub_entry = {'subject_id': sub_id}
            for i, f in enumerate(roi_features):
                for k, v in f.items():
                    sub_entry[f'roi{i}_{k}'] = v
            
            all_subject_features.append(sub_entry)
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {sub_id}: {e}")
            error_count += 1
            continue
    
    # Report statistics
    print(f"\n{'='*60}")
    print(f"Processing Summary:")
    print(f"  Successfully processed: {processed_count}")
    print(f"  Missing time series:    {missing_count}")
    print(f"  Errors:                 {error_count}")
    print(f"{'='*60}\n")
    
    if not all_subject_features:
        print("❌ No subjects were successfully processed!")
        return
    
    # Save final attributes
    NODE_ATTRIBUTES_TEMPORAL.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_subject_features)
    df.to_csv(NODE_ATTRIBUTES_TEMPORAL, index=False)
    
    # Report feature statistics
    feature_cols = [c for c in df.columns if c != 'subject_id']
    print(f"✅ Saved temporal attributes for {len(df)} subjects to:")
    print(f"   {NODE_ATTRIBUTES_TEMPORAL}")
    print(f"\nFeature Statistics:")
    print(f"  Total features per subject: {len(feature_cols)}")
    print(f"  Expected ROIs: {len(feature_cols) // NUM_TEMPORAL_FEATURES}")
    print(f"  Features per ROI: {NUM_TEMPORAL_FEATURES}")
    
    # Quick sanity check
    sample_stats = df[feature_cols].describe()
    print(f"\nSample Feature Statistics (first few features):")
    print(sample_stats.iloc[:, :5])


if __name__ == "__main__":
    main()