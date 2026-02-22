import gc
import json
import logging
import os
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from PIL import Image
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from nilearn.maskers import NiftiLabelsMasker
from nilearn.image import resample_to_img
from nilearn.datasets import load_mni152_brain_mask

# Suppress nilearn FutureWarnings (already addressed in code with explicit parameters)
warnings.filterwarnings("ignore", category=FutureWarning, module="nilearn")

# --- PATHS ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.core.config import LOBE_MAPPING, LOBE_NAMES 
PNG_OUTPUT   = PROJECT_ROOT / "data" / "images"
TS_OUTPUT    = PROJECT_ROOT / "data" / "processed"
META_DIR     = PROJECT_ROOT / "data" / "metadata"
ATLAS_PATH   = PROJECT_ROOT / "data" / "raw" / "atlases" / "AAL3v1.nii"
PHENO_PATH   = PROJECT_ROOT / "data" / "processed" / "Phenotypic_V1_0b_preprocessed1.csv"
MASK_S3_TEMPLATE = "data/Projects/ABIDE_Initiative/Outputs/cpac/filt_global/brain_mask/{sub_id}_brain_mask.nii.gz"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_S3_CLIENT = None
_MNI_MASK_TEMPLATE = None

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
def init_worker():
    global _S3_CLIENT, _MNI_MASK_TEMPLATE
    _S3_CLIENT = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    _MNI_MASK_TEMPLATE = load_mni152_brain_mask()


def get_s3_client():
    if _S3_CLIENT is None:
        return boto3.client("s3", config=Config(signature_version=UNSIGNED))
    return _S3_CLIENT


def process_subject(sub_id, tr_val):
    # 1. Skip if files already exist (Idempotency)
    final_ts_path = TS_OUTPUT / f"{sub_id}_ts.npy"
    if final_ts_path.exists():
        return sub_id, "Skipped", None

    s3 = get_s3_client()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            f_p = tmp_path / f"{sub_id}_func.nii.gz"
            a_p = tmp_path / f"{sub_id}_alff.nii.gz"
            m_p = tmp_path / f"{sub_id}_mask.nii.gz"
            
            # Download only if necessary
            s3.download_file("fcp-indi", f"data/Projects/ABIDE_Initiative/Outputs/cpac/filt_global/func_preproc/{sub_id}_func_preproc.nii.gz", str(f_p))
            s3.download_file("fcp-indi", f"data/Projects/ABIDE_Initiative/Outputs/cpac/filt_global/alff/{sub_id}_alff.nii.gz", str(a_p))

            # 1. Load and Fix Orientation
            func_img = nib.as_closest_canonical(nib.load(str(f_p)))
            atlas_img = nib.as_closest_canonical(nib.load(str(ATLAS_PATH)))
            
            # 2. Resample Atlas to Functional Space (Critical Step)
            # This ensures the masks align with the processed brain
            resampled_atlas = resample_to_img(
                atlas_img,
                func_img,
                interpolation="nearest",
                force_resample=True,  # Suppress future warning
                copy_header=True      # Use new header behavior
            )

            # 3. Time Series Extraction with finite check
            try:
                s3.download_file(
                    "fcp-indi",
                    MASK_S3_TEMPLATE.format(sub_id=sub_id),
                    str(m_p)
                )
                mask_img_obj = resample_to_img(
                    nib.load(str(m_p)),
                    func_img,
                    interpolation="nearest"
                )
            except Exception as e:
                logger.warning(
                    "Mask download failed for %s; using MNI mask fallback (%s)",
                    sub_id,
                    e,
                )
                mask_img_obj = resample_to_img(
                    _MNI_MASK_TEMPLATE,
                    func_img,
                    interpolation="nearest",
                    force_resample=True,
                    copy_header=True,
                )

            # Temporal filtering: 0.01-0.08 Hz (standard resting-state connectivity)
            # Current setting follows ABIDE preprocessing guidelines for broad connectivity
            masker = NiftiLabelsMasker(
                labels_img=resampled_atlas, 
                mask_img=mask_img_obj,
                t_r=float(tr_val), 
                standardize="zscore",
                detrend=True,
                low_pass=0.08,   # Standard resting-state upper bound
                high_pass=0.01,  # Standard resting-state lower bound
                ensure_finite=True,  # Safety for NaN values
                strategy="mean",
                memory_level=0  # Saves RAM by not caching to disk
            )
            
            ts = masker.fit_transform(func_img)

            label_ids = getattr(masker, "labels_", None)
            if label_ids is None or len(label_ids) == 0:
                label_ids = np.unique(resampled_atlas.get_fdata())
                label_ids = label_ids[label_ids != 0]  # Remove background
            
            label_ids = np.asarray(label_ids, dtype=int)
            # Background (id=0) will be skipped in the loop below via 1 <= roi_id check
            if label_ids.shape[0] != ts.shape[1]:
                logger.debug(
                    "Subject %s: label count %s != ts columns %s (ROI dropped by masker)",
                    sub_id,
                    label_ids.shape[0],
                    ts.shape[1],
                )
            
            # Validate ROI count (AAL3v1 variant: 164-170 ROIs)
            # Some AAL3v1 templates have 2 unused/empty ROIs
            VALID_ROI_RANGE = (164, 170)
            if not (VALID_ROI_RANGE[0] <= ts.shape[1] <= VALID_ROI_RANGE[1]):
                raise ValueError(
                    f"ROI count mismatch: extracted {ts.shape[1]} ROIs, expected {VALID_ROI_RANGE[0]}-{VALID_ROI_RANGE[1]}. "
                    f"Atlas resampling may have failed for subject {sub_id}"
                )
            
            # Additional validation: ensure no NaN/Inf after masker processing
            if not np.isfinite(ts).all():
                raise ValueError(f"Non-finite values detected in time series for {sub_id}")

            # --- CRITICAL FIX FOR AAL3v1 INDEX SHIFT ---
            full_ts = np.zeros((ts.shape[0], 170), dtype=np.float32)
            for col_idx, roi_id in enumerate(label_ids[: ts.shape[1]]):
                if 1 <= roi_id <= 170:
                    full_ts[:, roi_id - 1] = ts[:, col_idx]
                else:
                    logger.warning("Subject %s: ROI id %s out of range", sub_id, roi_id)

            if np.any(np.all(full_ts == 0, axis=0)):
                zero_rois = np.where(np.all(full_ts == 0, axis=0))[0]
                
                # Use deterministic Gaussian noise to avoid flat signal artifacts
                # (1e-6 constant creates artificial autocorrelation and flat PSD)
                rng = np.random.default_rng(seed=hash(sub_id) % (2**32))
                for roi in zero_rois:
                    full_ts[:, roi] = rng.normal(0, 1e-6, size=full_ts.shape[0])
                
                # Map empty ROIs to their lobe assignments for better diagnostics
                empty_roi_ids = [r + 1 for r in zero_rois]  # Convert to 1-indexed
                
                affected_lobes = {}
                for lobe_id, roi_indices in LOBE_MAPPING.items():
                    affected = [r + 1 for r in roi_indices if (r + 1) in empty_roi_ids]
                    if affected:
                        affected_lobes[LOBE_NAMES[lobe_id]] = affected
                
                lobe_summary = ", ".join([f"{lobe}: {rois}" for lobe, rois in affected_lobes.items()])
                logger.warning(
                    "Subject %s: patched %d empty ROIs | %s",
                    sub_id,
                    len(zero_rois),
                    lobe_summary if lobe_summary else "unknown lobes"
                )

            np.save(final_ts_path, full_ts)
            np.save(TS_OUTPUT / f"{sub_id}_roi_labels.npy", label_ids)

            # 4. ALFF Slice Export (YOLO)
            alff_img = nib.as_closest_canonical(nib.load(str(a_p)))
            alff_data = alff_img.get_fdata()
            
            # Slice percentiles: 0.2 captures cerebellum/brainstem (regions 10-11 in LOBE_MAPPING)
            # 0.3-0.8 cover mid-brain structures (thalamus, basal ganglia, cortex)
            for p in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                z = int(alff_data.shape[2] * p)
                slice_arr = np.rot90(alff_data[:, :, z])
                
                # Robust Normalization
                p2, p98 = np.percentile(slice_arr, [2, 98])
                denom = p98 - p2
                norm = np.clip((slice_arr - p2) / (denom if denom > 0 else 1e-8), 0, 1)
                
                img = Image.fromarray((norm * 255).astype(np.uint8))
                # Using Lanczos for better downsampling quality if reducing size
                img.resize((640, 640), resample=Image.LANCZOS).save(PNG_OUTPUT / f"{sub_id}_z{z}.png")
            
            # Manual RAM cleanup to reduce peak usage per worker.
            del func_img, atlas_img, resampled_atlas, mask_img_obj, masker, ts, alff_img
            gc.collect()

            return sub_id, "Success", None
            
    except Exception as e:
        return sub_id, "Failed", str(e)

# --- EXECUTION ---
if __name__ == "__main__":
    # Setup folders
    for d in [PNG_OUTPUT, TS_OUTPUT, META_DIR]: 
        d.mkdir(parents=True, exist_ok=True)
    
    logger.info("Pre-calculating Atlas Metadata...")
    save_atlas_metadata()
    
    # Load Phenotypic data
    if not PHENO_PATH.exists():
        logger.info("Downloading Phenotypic data...")
        import urllib.request
        url = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv"
        urllib.request.urlretrieve(url, PHENO_PATH)
        logger.info("Phenotypic data downloaded.")

    df = pd.read_csv(PHENO_PATH)
    # Strip whitespace from FILE_ID to prevent match failures
    df['FILE_ID'] = df['FILE_ID'].astype(str).str.strip()
    
    # ABIDE I site-specific TRs (from ABIDE documentation)
    # Different scanners have different repetition times
    # CRITICAL: Keys must match SITE_ID column format (uppercase)
    SITE_TR_MAP = {
        'CALTECH': 2.0, 'CMU': 2.0, 'KKI': 2.5, 'LEUVEN_1': 1.656,
        'LEUVEN_2': 1.656, 'MAX_MUN': 3.0, 'NYU': 2.0, 'OHSU': 2.5,
        'OLIN': 1.5, 'PITT': 1.5, 'SBL': 2.5, 'SDSU': 2.0,
        'STANFORD': 2.0, 'TRINITY': 2.0, 'UCLA_1': 3.0, 'UCLA_2': 3.0,
        'UM_1': 2.0, 'UM_2': 2.0, 'USM': 2.0, 'YALE': 2.0
    }
    
    # Handle TR column (may not exist in all phenotype versions)
    if 'TR' not in df.columns:
        logger.warning("'TR' column not found in phenotype CSV. Using site-specific TRs.")
        df['TR'] = df['SITE_ID'].map(SITE_TR_MAP).fillna(2.0)
        logger.info("Assigned site-specific TRs: %s", df.groupby('SITE_ID')['TR'].first().to_dict())
    else:
        df['TR'] = pd.to_numeric(df['TR'], errors='coerce').fillna(2.0)
    
    # Filter valid subjects
    subjects_df = df[df["FILE_ID"] != "no_filename"].dropna(subset=["FILE_ID"])
    tasks = subjects_df[["FILE_ID", "TR"]].drop_duplicates().values
    limit = int(os.environ.get("ABIDE_SUBJECT_LIMIT", "0"))
    if limit > 0:
        tasks = tasks[:limit]
        logger.info("Limiting to %s subjects via ABIDE_SUBJECT_LIMIT", limit)
    
    logger.info("Starting processing for %s subjects...", len(tasks))
    
    DOWNLOAD_LOG = META_DIR / "download_log.csv"

    # fMRI processing is often RAM-bound, not CPU-bound.
    with ProcessPoolExecutor(max_workers=6, initializer=init_worker) as exe:
        futures = [exe.submit(process_subject, row[0], row[1]) for row in tasks]
        
        with open(DOWNLOAD_LOG, 'w') as log_file:
            log_file.write("subject_id,status,error\n")
            
            for fut in tqdm(as_completed(futures), total=len(tasks)):
                sub_id, status, err = fut.result()
                clean_err = str(err).replace(',', ';') if err else ""
                log_file.write(f"{sub_id},{status.lower()},{clean_err}\n")
                log_file.flush()  # Ensure real-time logging
                
                if status == "Failed":
                    logger.error("Error on %s: %s", sub_id, err)
    
    logger.info("Download log saved to %s", DOWNLOAD_LOG)