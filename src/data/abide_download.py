import os
import pandas as pd
import nibabel as nib
import numpy as np
from PIL import Image
import tempfile
from concurrent.futures import ThreadPoolExecutor
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm

PHENO_PATH = "./data/processed/Phenotypic_V1_0b_preprocessed1.csv"
BUCKET_NAME = "fcp-indi"
S3_PREFIX = "data/Projects/ABIDE_Initiative/Outputs/cpac/filt_global/func_preproc/"
BASE_DIR = os.getcwd()
PNG_OUTPUT = os.path.join(BASE_DIR, "data", "images")
LOG_OUTPUT = os.path.join(BASE_DIR, "data", "metadata", "download_log.csv")

# 5 slices evenly spaced between 20 and 60
Z_SLICES = [20, 30, 40, 50, 60]
IMG_SIZE = (640, 640) 
N_WORKERS = 10      

# Initialize S3 Client once (Thread-safe)
S3_CLIENT = boto3.client('s3', config=Config(signature_version=UNSIGNED))

def process_subject(sub_id):
    """
    Downloads a NIfTI file, extracts specific slices, normalizes intensity, 
    resizes for YOLO, and saves as PNG.
    """
    status = {"subject_id": sub_id, "status": "Pending", "message": ""}
    
    # 1. Pre-check: Skip if all 5 slices already exist
    expected_paths = [os.path.join(PNG_OUTPUT, f"{sub_id}_z{z}.png") for z in Z_SLICES]
    if all(os.path.exists(p) for p in expected_paths):
        return {"subject_id": sub_id, "status": "Skipped", "message": "All slices exist"}

    fname = f"{sub_id}_func_preproc.nii.gz"
    s_key = f"{S3_PREFIX}{fname}"

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, fname)
            
            # 2. Download from S3 (Public bucket)
            S3_CLIENT.download_file(BUCKET_NAME, s_key, local_path)

            # 3. Load Image and fix Orientation
            # nib.as_closest_canonical ensures 'Z' is actually the vertical axis
            raw_img = nib.load(local_path)
            img = nib.as_closest_canonical(raw_img)
            data_shape = img.shape
            saved_count = 0
            
            # 4. Extract and Process Slices
            for z_idx in Z_SLICES:
                # Boundary check
                if z_idx >= data_shape[2]: 
                    continue

                # Pull data lazily using dataobj to save RAM
                if len(data_shape) == 4:
                    # [x, y, z, time] -> Mean across time
                    slice_data = np.array(img.dataobj[:, :, z_idx, :])
                    mean_slice = np.mean(slice_data, axis=-1)
                else:
                    # [x, y, z]
                    mean_slice = np.array(img.dataobj[:, :, z_idx])

                # 5. Signal Filtering (Skip empty/dark slices)
                if np.percentile(mean_slice, 98) < 1e-3:
                    continue 

                # 6. Robust Normalization (2nd-98th percentile for contrast)
                p_min, p_max = np.percentile(mean_slice, [2, 98])
                if p_max <= p_min: 
                    continue

                norm_slice = np.clip((mean_slice - p_min) / (p_max - p_min + 1e-8), 0, 1)

                # 7. Convert to Image and Resize
                # Convert to 8-bit grayscale
                img_out = Image.fromarray((norm_slice * 255).astype(np.uint8))
                
                # Standardize orientation (may need adjustment depending on specific site)
                img_out = img_out.transpose(Image.ROTATE_90)
                
                # High-quality resize to 640x640 for YOLO
                img_out = img_out.resize(IMG_SIZE, resample=Image.Resampling.LANCZOS)
                
                out_path = os.path.join(PNG_OUTPUT, f"{sub_id}_z{z_idx}.png")
                img_out.save(out_path)
                saved_count += 1
            
            # Explicit cleanup
            img.uncache()
            del raw_img, img

        status.update({"status": "Success", "message": f"Saved {saved_count}/{len(Z_SLICES)}"})
        
    except Exception as e:
        status.update({"status": "Failed", "message": str(e)})
    
    return status

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs(PNG_OUTPUT, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_OUTPUT), exist_ok=True)

    print(f"ABIDE Preprocessing convert to YOLO (640x640)")       
    
    if not os.path.exists(PHENO_PATH):
        print(f"Error: Phenotypic file not found at {PHENO_PATH}")
        exit()

    # Load subject list
    df = pd.read_csv(PHENO_PATH)
    subject_list = df[df['FILE_ID'] != 'no_filename']['FILE_ID'].dropna().unique().tolist()
    
    total_subjects = len(subject_list)
    print(f"Starting processing for {total_subjects} subjects...")
    
    # Process using a ThreadPool for S3 I/O efficiency
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        # tqdm creates a nice progress bar
        results = list(tqdm(executor.map(process_subject, subject_list), total=total_subjects))

    # Save execution log
    log_df = pd.DataFrame(results)
    log_df.to_csv(LOG_OUTPUT, index=False)
    
    print(f"\nProcessing Complete!")
    print(f"Log saved to: {LOG_OUTPUT}")
    print(f"Images saved to: {PNG_OUTPUT}")