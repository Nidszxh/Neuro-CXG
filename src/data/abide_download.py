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
# Suppress torch CUDA UserWarning on hardware without matching CUDA driver support
warnings.filterwarnings("ignore", message=".*CUDA initialization.*", category=UserWarning)
# Suppress scipy RuntimeWarning for empty ROIs (NiftiLabelsMasker mean of zero-voxel regions)
warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning, module="scipy")

# PATHS - Import from config to ensure single source of truth
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import (
    LOBE_MAPPING, LOBE_NAMES,
    DATA_IMAGES, DATA_PROCESSED, DATA_TIME_SERIES, DATA_METADATA,
    ATLAS_PATH, PHENO_PATH, ROI_CENTROIDS_PATH, AAL3_VALID_ROI_RANGE,
    SITE_TR_MAP, ALFF_SLICE_PERCENTILES,
    BANDPASS_LOW, BANDPASS_HIGH,
    EXCLUDED_SUBJECTS,
)
PNG_OUTPUT   = DATA_IMAGES
# Prefer dedicated time-series directory when available, keep legacy fallback.
TS_OUTPUT    = DATA_TIME_SERIES if DATA_TIME_SERIES.exists() else DATA_PROCESSED
META_DIR     = DATA_METADATA
MASK_S3_TEMPLATE = "data/Projects/ABIDE_Initiative/Outputs/cpac/filt_global/func_mask/{sub_id}_func_mask.nii.gz"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_S3_CLIENT = None
_MNI_MASK_TEMPLATE = None

# HELPER: ATLAS PREP 
def save_atlas_metadata():
    """Extracts ROI centroids once to avoid 4D overhead later. Idempotent."""
    if not ATLAS_PATH.exists():
        raise FileNotFoundError(f"Atlas not found at {ATLAS_PATH}")
    
    # Idempotent: skip if already exists
    if ROI_CENTROIDS_PATH.exists():
        logger.info("ROI centroids already exist, skipping computation.")
        return nib.load(str(ATLAS_PATH))
    
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
    with open(ROI_CENTROIDS_PATH, 'w') as f:
        json.dump(coords, f)
    logger.info(f"Saved ROI centroids to {ROI_CENTROIDS_PATH}")
    return atlas_img

# THE CORE PROCESS 
def init_worker():
    global _S3_CLIENT, _MNI_MASK_TEMPLATE
    _S3_CLIENT = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    _MNI_MASK_TEMPLATE = load_mni152_brain_mask()


def get_s3_client():
    if _S3_CLIENT is None:
        logger.warning(
            "S3 client not initialized via init_worker(). "
            "Creating unauthenticated client - may fail outside process pool."
        )
        return boto3.client("s3", config=Config(signature_version=UNSIGNED))
    return _S3_CLIENT


def process_subject(sub_id, tr_val):
    # 1. Skip only when the subject is fully complete (TS + ROI labels + all slices).
    # This allows re-runs to fill missing artifacts without reprocessing everything.
    final_ts_path = TS_OUTPUT / f"{sub_id}_ts.npy"
    final_roi_labels_path = TS_OUTPUT / f"{sub_id}_roi_labels.npy"
    existing_png_count = len(list(PNG_OUTPUT.glob(f"{sub_id}_z*.png")))
    expected_png_count = len(ALFF_SLICE_PERCENTILES)

    subject_complete = (
        final_ts_path.exists()
        and final_roi_labels_path.exists()
        and existing_png_count >= expected_png_count
    )
    if subject_complete:
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
                    interpolation="nearest",
                    force_resample=True,
                    copy_header=True,
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

            # Temporal filtering: 0.01-0.08 Hz (standard resting-state connectivity).
            # standardize=False: a single z-score is applied downstream in
            # construct_causal.py::construct_graph() so that the normalisation
            # happens once, in one place, and is fully auditable.
            masker = NiftiLabelsMasker(
                labels_img=resampled_atlas, 
                mask_img=mask_img_obj,
                t_r=float(tr_val), 
                standardize=False,  # z-score applied once in construct_causal.py
                detrend=True,
                low_pass=BANDPASS_HIGH,  # config.BANDPASS_HIGH = 0.15 Hz (expanded from 0.08 Hz)
                high_pass=BANDPASS_LOW,  # config.BANDPASS_LOW  = 0.01 Hz
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
            if not (AAL3_VALID_ROI_RANGE[0] <= ts.shape[1] <= AAL3_VALID_ROI_RANGE[1]):
                raise ValueError(
                    f"ROI count mismatch: extracted {ts.shape[1]} ROIs, expected {AAL3_VALID_ROI_RANGE[0]}-{AAL3_VALID_ROI_RANGE[1]}. "
                    f"Atlas resampling may have failed for subject {sub_id}"
                )
            
            # Additional validation: ensure no NaN/Inf after masker processing
            if not np.isfinite(ts).all():
                raise ValueError(f"Non-finite values detected in time series for {sub_id}")

            # CRITICAL FIX FOR AAL3v1 INDEX SHIFT 
            full_ts = np.zeros((ts.shape[0], 170), dtype=np.float32)
            for col_idx, roi_id in enumerate(label_ids[: ts.shape[1]]):
                if 1 <= roi_id <= 170:
                    full_ts[:, roi_id - 1] = ts[:, col_idx]
                else:
                    # ROI id 0 is the atlas background label — skip silently
                    logger.debug("Subject %s: ROI id %s out of range (background label)", sub_id, roi_id)

            if np.any(np.all(full_ts == 0, axis=0)):
                zero_rois = np.where(np.all(full_ts == 0, axis=0))[0]

                # Fill empty ROIs with np.nan so downstream code can detect and
                # skip/impute them cleanly. Gaussian noise is scientifically unsound
                # because it introduces artificial autocorrelation and flat PSD.
                for roi in zero_rois:
                    full_ts[:, roi] = np.nan

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
            
            # Slice percentiles from config (single source of truth)
            # CRITICAL: 0.21 captures brainstem (region 11, ROIs 167-170 starting at z=38)
            # Must match generate_labels.py exactly for atlas-to-image alignment
            for p in ALFF_SLICE_PERCENTILES:
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

# EXECUTION 
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
    
    # Handle TR column: ALWAYS APPLY SITE-SPECIFIC MAPPING (imported from config)
    # Even if phenotype CSV has a TR column, we override with site-specific values
    # (phenotype CSVs often have generic/default TR values, not actual site-specific ones)
    logger.info("Applying site-specific TR mapping based on SITE_ID...")
    df['TR'] = df['SITE_ID'].map(SITE_TR_MAP).fillna(2.0)
    logger.info("Assigned site-specific TRs: %s", df.groupby('SITE_ID')['TR'].first().to_dict())
    
    # Filter valid subjects and enforce curated 1035->1015 exclusion policy.
    subjects_df = df[df["FILE_ID"] != "no_filename"].dropna(subset=["FILE_ID"])
    excluded_upper = {s.upper() for s in EXCLUDED_SUBJECTS}
    subjects_df = subjects_df[
        ~subjects_df["FILE_ID"].astype(str).str.upper().isin(excluded_upper)
    ]
    logger.info(
        "Applied EXCLUDED_SUBJECTS filter: %d subject(s) excluded, %d remaining",
        len(EXCLUDED_SUBJECTS),
        len(subjects_df["FILE_ID"].drop_duplicates()),
    )
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