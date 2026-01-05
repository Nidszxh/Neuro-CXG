import os, json, tempfile, numpy as np, pandas as pd, nibabel as nib
from pathlib import Path
from PIL import Image
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from nilearn.maskers import NiftiLabelsMasker
from nilearn.image import resample_to_img

# --- PATHS ---
PROJECT_ROOT = Path(__file__).resolve().parents[0] # Adjusted for local testing
PNG_OUTPUT   = PROJECT_ROOT / "data" / "images"
TS_OUTPUT    = PROJECT_ROOT / "data" / "processed"
META_DIR     = PROJECT_ROOT / "data" / "metadata"
ATLAS_PATH   = PROJECT_ROOT / "data" / "atlases" / "AAL3v1.nii"
PHENO_PATH   = PROJECT_ROOT / "data" / "processed" / "Phenotypic_V1_0b_preprocessed1.csv"

# --- HELPER: ATLAS PREP ---
def save_atlas_metadata():
    """Extracts ROI centroids once to avoid 4D overhead later."""
    if not ATLAS_PATH.exists():
        raise FileNotFoundError(f"Atlas not found at {ATLAS_PATH}")
    
    atlas_img = nib.load(str(ATLAS_PATH))
    data = atlas_img.get_fdata()
    affine = atlas_img.affine
    labels = np.unique(data)[1:] 
    
    coords = []
    for label in labels:
        # Get voxel indices
        indices = np.argwhere(data == label)
        # Convert voxel indices to MNI space (mm)
        mean_vox = indices.mean(axis=0)
        # Add 1 for the affine transformation math
        mni_coord = affine @ np.append(mean_vox, 1) 
        coords.append({
            "roi_id": int(label), 
            "x": float(mni_coord[0]), 
            "y": float(mni_coord[1]), 
            "z": float(mni_coord[2])
        })
    
    META_DIR.mkdir(parents=True, exist_ok=True)
    with open(META_DIR / "roi_centroids.json", 'w') as f:
        json.dump(coords, f)
    return atlas_img

# --- THE CORE PROCESS ---
def process_subject(sub_id, tr_val):
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            f_p = tmp_path / f"{sub_id}_func.nii.gz"
            a_p = tmp_path / f"{sub_id}_alff.nii.gz"
            
            # Download
            s3.download_file("fcp-indi", f"data/Projects/ABIDE_Initiative/Outputs/cpac/filt_global/func_preproc/{sub_id}_func_preproc.nii.gz", str(f_p))
            s3.download_file("fcp-indi", f"data/Projects/ABIDE_Initiative/Outputs/cpac/filt_global/alff/{sub_id}_alff.nii.gz", str(a_p))

            # 1. Load and Fix Orientation
            func_img = nib.load(str(f_p))
            func_img = nib.as_closest_canonical(func_img)
            
            # 2. Resample Atlas to Functional Space (Critical Step)
            # This ensures the masks align with the processed brain
            resampled_atlas = resample_to_img(str(ATLAS_PATH), func_img, interpolation='nearest')

            # 3. Time Series Extraction
            masker = NiftiLabelsMasker(
                labels_img=resampled_atlas, 
                t_r=float(tr_val), 
                standardize='zscore_sample', 
                detrend=True,
                low_pass=0.08, 
                high_pass=0.01,
                memory_level=0 # Saves RAM by not caching to disk
            )
            
            ts = masker.fit_transform(func_img)
            np.save(TS_OUTPUT / f"{sub_id}_ts.npy", ts.astype(np.float32))

            # 4. ALFF Slice Export (YOLO)
            alff_img = nib.as_closest_canonical(nib.load(str(a_p)))
            alff_data = alff_img.get_fdata()
            
            for p in [0.3, 0.4, 0.5, 0.6, 0.7]:
                z = int(alff_data.shape[2] * p)
                slice_arr = np.rot90(alff_data[:, :, z])
                
                # Robust Normalization
                p2, p98 = np.percentile(slice_arr, [2, 98])
                norm = np.clip((slice_arr - p2) / (p98 - p2 + 1e-8), 0, 1)
                
                img = Image.fromarray((norm * 255).astype(np.uint8))
                img.resize((640, 640)).save(PNG_OUTPUT / f"{sub_id}_z{z}.png")
            
            return sub_id, "Success", None
            
    except Exception as e:
        return sub_id, "Failed", str(e)

# --- EXECUTION ---
if __name__ == "__main__":
    # Setup folders
    for d in [PNG_OUTPUT, TS_OUTPUT, META_DIR]: 
        d.mkdir(parents=True, exist_ok=True)
    
    print("Pre-calculating Atlas Metadata...")
    save_atlas_metadata()
    
    # Load Phenotypic data
    df = pd.read_csv(PHENO_PATH)
    df['TR'] = pd.to_numeric(df['TR'], errors='coerce').fillna(2.0)
    
    # Filter valid subjects
    subjects_df = df[df["FILE_ID"] != "no_filename"].dropna(subset=["FILE_ID"])
    tasks = subjects_df[["FILE_ID", "TR"]].drop_duplicates().values
    
    print(f"Starting processing for {len(tasks)} subjects...")
    
    # Use max_workers=6 or 8 (even if you have 12-16 threads) 
    # fMRI processing is often RAM-bound, not CPU-bound.
    results = []
    with ProcessPoolExecutor(max_workers=8) as exe:
        futures = [exe.submit(process_subject, row[0], row[1]) for row in tasks]
        
        for fut in tqdm(as_completed(futures), total=len(tasks)):
            sub_id, status, err = fut.result()
            if status == "Failed":
                print(f"Error on {sub_id}: {err}")