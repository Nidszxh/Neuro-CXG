import os, logging, sys
import pandas as pd
from pathlib import Path

# Setup paths from config
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import (
    DATA_PROCESSED, DATA_FINAL, DATA_METADATA, MASTER_MANIFEST,
    PHENO_PATH, SITE_TR_MAP
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_manifest():
    if not PHENO_PATH.exists():
        logger.error(f"[Error]: Phenotypic file not found at {PHENO_PATH}")
        return
        
    # 1. Load and clean phenotypic data (Phase 2.1)
    df = pd.read_csv(PHENO_PATH)
    df.columns = df.columns.str.strip()
    # Strip whitespace from FILE_ID to prevent match failures
    df['FILE_ID'] = df['FILE_ID'].astype(str).str.strip() 
    
    # 2. Map processed files to their specific splits (Phase 2.2)
    manifest_data = []
    splits = ['train', 'val', 'test']
    
    for split in splits:
        # Check time_series folder specifically as it's the core for the GNN
        ts_path = DATA_FINAL / split / 'time_series'
        
        if ts_path.exists():
            # Extract subject IDs from time series files only (not roi_labels)
            subjects = [f.replace('_ts.npy', '') for f in os.listdir(ts_path) if f.endswith('_ts.npy')]
            for s in subjects:
                manifest_data.append({'subject_id': s, 'split': split})
    
    if not manifest_data:
        logger.error("[Error]: No processed data found. Ensure split.py was successful.")
        return

    manifest_df = pd.DataFrame(manifest_data)

    # Enforce one subject -> one split mapping; stale files across splits can violate this.
    duplicate_subjects = manifest_df['subject_id'].value_counts()
    duplicate_subjects = duplicate_subjects[duplicate_subjects > 1]
    if not duplicate_subjects.empty:
        logger.warning(
            f"Detected {len(duplicate_subjects)} subject IDs present in multiple split folders. "
            "Keeping first occurrence per subject and marking dataset as inconsistent."
        )
        logger.warning(
            f"Example duplicate subject IDs: {duplicate_subjects.index.tolist()[:10]}"
        )
        logger.warning(
            "Please re-run src/data/split.py to rebuild clean train/val/test directories."
        )

    # Stable priority if duplicates exist: train > val > test
    split_priority = {'train': 0, 'val': 1, 'test': 2}
    manifest_df['_split_priority'] = manifest_df['split'].map(split_priority).fillna(99)
    manifest_df = (
        manifest_df
        .sort_values(['subject_id', '_split_priority'])
        .drop_duplicates(subset=['subject_id'], keep='first')
        .drop(columns=['_split_priority'])
    )
    
    # 3. Select Causal & Clinical variables (Phase 7.1 & 8.4)
    # We include IQ and Handedness as they are major confounders in ASD research
    # TR is intentionally excluded here because phenotype CSV variants may not
    # contain site-correct TR; we derive TR from SITE_TR_MAP below.
    required_cols = [
        'FILE_ID', 'DX_GROUP', 'AGE_AT_SCAN', 'SEX',
        'SITE_ID', 'FIQ', 'HANDEDNESS_CATEGORY'
    ]
    
    # Filter only available columns to avoid merge errors
    available_cols = [c for c in required_cols if c in df.columns]
    
    # 4. Final Merge
    final_df = pd.merge(
        manifest_df, 
        df[available_cols], 
        left_on='subject_id', 
        right_on='FILE_ID',
        how='inner'
    ).drop(columns=['FILE_ID'])

    # Final safety: guarantee unique subject IDs in manifest output.
    final_df = final_df.drop_duplicates(subset=['subject_id'], keep='first')

    pheno_unique = df['FILE_ID'].nunique()
    manifest_unique = final_df['subject_id'].nunique()
    if manifest_unique > pheno_unique:
        logger.error(
            f"Manifest unique subject count ({manifest_unique}) exceeds phenotype unique FILE_ID count ({pheno_unique})."
        )
    else:
        logger.info(
            f"Subject count sanity check passed: manifest={manifest_unique}, phenotype={pheno_unique}"
        )
    
    # 5. Data Integrity Check: Ensure no missing targets
    final_df = final_df.dropna(subset=['DX_GROUP'])

    # 6. Apply Site-Specific TR Mapping (imported from config)
    logger.info("Applying site-specific TR mapping based on SITE_ID...")
    final_df['TR'] = final_df['SITE_ID'].map(SITE_TR_MAP).fillna(2.0)
    tr_summary = final_df.groupby('SITE_ID')['TR'].first().to_dict()
    logger.info(f"Assigned site-specific TRs: {tr_summary}")
    
    DATA_METADATA.mkdir(parents=True, exist_ok=True)

    # Preserve cv_fold column written by split.py — overwriting without it causes
    # GNN training to crash with "cv_fold column not found".
    if MASTER_MANIFEST.exists():
        try:
            _prev = pd.read_csv(MASTER_MANIFEST)
            if 'cv_fold' in _prev.columns:
                _fold_map = _prev.set_index('subject_id')['cv_fold']
                final_df['cv_fold'] = final_df['subject_id'].map(_fold_map).fillna(-1).astype(int)
                logger.info("Preserved cv_fold assignments from existing manifest.")
        except Exception as _e:
            logger.warning("Could not preserve cv_fold from existing manifest: %s", _e)

    final_df.to_csv(MASTER_MANIFEST, index=False)
    
    logger.info(f"Manifest successfully synchronized with {len(final_df)} subjects.")
    logger.info(f"Breakdown:\n{final_df.groupby(['split', 'DX_GROUP']).size().unstack(fill_value=0)}")

if __name__ == "__main__":
    create_manifest()