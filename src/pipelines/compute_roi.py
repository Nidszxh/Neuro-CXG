import os
import nibabel as nib
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from nilearn.maskers import NiftiSpheresMasker
from tqdm import tqdm
import warnings

# --- CONFIG ---
ROI_FEATURES_CSV = "./data/metadata/roi_features_final.csv"
RAW_NIFTI_DIR = "./data/raw_nifti"
OUTPUT_PATH = "./data/metadata/node_signals_temporal.csv"
RADIUS = 8  # 8mm sphere around the centroid
TR = 2.0    # Repetition Time in seconds (will try to read from header)

# Node mapping
NODE_NAMES = {0: 'frontal', 1: 'temporal', 2: 'parietal', 3: 'occipital'}

def calculate_psd(ts, sampling_rate):
    """Calculates Power Spectral Density in low frequency band."""
    if len(ts) < 4:
        return np.nan
    
    n = len(ts)
    f_hat = np.fft.fft(ts - np.mean(ts), n)  # Remove DC component
    psd = (f_hat * np.conj(f_hat) / n).real
    freqs = np.fft.fftfreq(n, d=sampling_rate)
    
    # Low frequency band (0.01-0.1 Hz) - key for resting-state fMRI
    low_freq_mask = (freqs > 0.01) & (freqs < 0.1)
    
    if not np.any(low_freq_mask):
        return np.nan
    
    return np.mean(psd[low_freq_mask])

def calculate_signal_variability(ts):
    """Measures signal predictability via first-order difference ratio."""
    if len(ts) < 2:
        return np.nan
    
    diff_std = np.std(np.diff(ts))
    ts_std = np.std(ts)
    
    if ts_std == 0 or diff_std == 0:
        return np.nan
    
    # Higher values = more unpredictable/complex signal
    return diff_std / ts_std

def yolo_to_mni(yolo_coords, img_shape, affine):
    """
    Improved: Converts YOLO normalized to Voxel, then Voxel to MNI.
    """
    # 1. Normalized -> Voxel (Indices)
    # YOLO (x,y) is usually (col, row). Nifti is usually (x, y, z)
    # Ensure these mappings match your 'abide_download.py' slice extraction logic
    vx = yolo_coords[0] * img_shape[0]
    vy = yolo_coords[1] * img_shape[1]
    vz = yolo_coords[2] * img_shape[2]
    
    # 2. Voxel -> MNI (mm)
    # Using nibabel's apply_affine is safer than manual matrix multiplication
    mni_coords = nib.affines.apply_affine(affine, [vx, vy, vz])
    
    return tuple(mni_coords)

def extract_temporal_features(time_series):
    """Extract all temporal features from a time series."""
    features = {}
    
    if len(time_series) == 0:
        return {k: np.nan for k in ['mean', 'std', 'skew', 'kurtosis', 'psd_low', 'variability']}
    
    features['mean'] = np.mean(time_series)
    features['std'] = np.std(time_series)
    features['skew'] = skew(time_series)
    features['kurtosis'] = kurtosis(time_series)
    features['psd_low'] = calculate_psd(time_series, TR)
    features['variability'] = calculate_signal_variability(time_series)
    
    return features

def extract_node_signals():
    # 1. Load the YOLO ROI centroids
    if not os.path.exists(ROI_FEATURES_CSV):
        print(f"Error: {ROI_FEATURES_CSV} not found. Run extract_features.py first.")
        return
    
    roi_df = pd.read_csv(ROI_FEATURES_CSV)
    print(f"Loaded ROI features: {roi_df.shape}")
    
    # 2. Validate required columns
    required_cols = ['ID_MATCH', 'roi_class']
    if not all(col in roi_df.columns for col in required_cols):
        print(f"Error: Missing required columns. Expected: {required_cols}")
        return
    
    all_subject_features = []
    subjects_processed = 0
    subjects_failed = 0

    # 3. Group by subject (handle multiple nodes per subject)
    print(f"Extracting temporal features for {roi_df['ID_MATCH'].nunique()} subjects...")
    
    for sub_id, sub_data in tqdm(roi_df.groupby('ID_MATCH')):
        # FIX: Flexible naming for NIfTI files
        potential_files = [
            os.path.join(RAW_NIFTI_DIR, f"{sub_id}.nii.gz"),
            os.path.join(RAW_NIFTI_DIR, f"{sub_id}_func_preproc.nii.gz")
        ]
        
        nifti_file = next((f for f in potential_files if os.path.exists(f)), None)
        
        if not nifti_file:
            subjects_failed += 1
            continue
        
        try:
            img = nib.load(nifti_file)
            affine = img.affine
            img_shape = img.shape[:3]  # (x, y, z) dimensions
            
            # FIX: Ensure TR is pulled from header if TR=2.0 is just a placeholder
            try:
                header_tr = img.header.get_zooms()[3]
                actual_tr = header_tr if header_tr > 0 else TR
            except:
                actual_tr = TR
            
            # Prepare coordinate list for all nodes in this subject
            coords_list = []
            node_labels = []
            
            for _, row in sub_data.iterrows():
                node_id = int(row['roi_class'])
                
                # Adjust column names based on your actual CSV structure
                # Option 1: If columns are 'x_centroid', 'y_centroid', 'z_centroid'
                if 'x_centroid' in row:
                    yolo_coords = (row['x_centroid'], row['y_centroid'], row['z_centroid'])
                # Option 2: If columns are 'x_0', 'y_0', etc. (update as needed)
                else:
                    yolo_coords = (row['x'], row['y'], row['z'])
                
                # Convert to MNI space
                mni_coords = yolo_to_mni(yolo_coords, img_shape, affine)
                coords_list.append(mni_coords)
                node_labels.append(NODE_NAMES[node_id])
            
            # Extract time series for all nodes at once (more efficient)
            masker = NiftiSpheresMasker(
                seeds=coords_list,
                radius=RADIUS,
                detrend=True,
                standardize='zscore',  # Explicitly use z-score for causal model nodes
                low_pass=0.1,
                high_pass=0.01,
                t_r=actual_tr,
                memory='nilearn_cache',  # Adds caching for speed
                verbose=0
            )
            
            time_series_all = masker.fit_transform(img)  # Shape: (n_timepoints, n_nodes)
            
            # Build feature dictionary
            sub_features = {'sub_id': sub_id}
            
            for i, node_name in enumerate(node_labels):
                ts = time_series_all[:, i]
                features = extract_temporal_features(ts)
                
                # Add features with node-specific prefix
                for feat_name, feat_val in features.items():
                    sub_features[f'{node_name}_{feat_name}'] = feat_val
            
            all_subject_features.append(sub_features)
            subjects_processed += 1
            
        except Exception as e:
            print(f"\nError processing {sub_id}: {e}")
            subjects_failed += 1
            continue

    # 4. Create final DataFrame
    if not all_subject_features:
        print("Error: No subjects were successfully processed!")
        return
    
    final_df = pd.DataFrame(all_subject_features)
    print(f"\nProcessed {subjects_processed} subjects ({subjects_failed} failed)")
    print(f"Final feature shape: {final_df.shape}")
    
    # 5. Quality Control: Z-score normalization
    feature_cols = [c for c in final_df.columns if c != 'sub_id']
    
    # Check for columns with zero variance
    zero_var_cols = final_df[feature_cols].columns[final_df[feature_cols].std() == 0]
    if len(zero_var_cols) > 0:
        print(f"Warning: Removing {len(zero_var_cols)} zero-variance features")
        feature_cols = [c for c in feature_cols if c not in zero_var_cols]
    
    # Z-score normalization with safety check
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_df[feature_cols] = final_df[feature_cols].apply(
            lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
        )
    
    # Replace any remaining inf/nan with 0
    final_df[feature_cols] = final_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # 6. Save results
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"\nPhase 4 Complete!")
    print(f"Temporal features saved to {OUTPUT_PATH}")
    print(f"Features per subject: {len(feature_cols)}")
    
    # Print summary statistics
    print("\nFeature Summary:")
    print(final_df[feature_cols].describe().loc[['mean', 'std', 'min', 'max']])

if __name__ == "__main__":
    extract_node_signals()