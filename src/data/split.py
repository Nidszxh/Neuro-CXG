import logging
import os
import random
import shutil
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

# Setup paths from config
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import DATA_ROOT, DATA_PROCESSED, DATA_TIME_SERIES, DATA_FINAL, MASTER_MANIFEST, PHENO_PATH

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- CONFIG ---
TRAIN_RATIO, VAL_RATIO = 0.70, 0.15 
random.seed(42)
SOURCE_IMG   = DATA_ROOT / "images"
# Prefer dedicated time-series directory when available, keep legacy fallback.
SOURCE_TS    = DATA_TIME_SERIES if DATA_TIME_SERIES.exists() else DATA_PROCESSED
SOURCE_LBL   = DATA_ROOT / "labels"


def _move_with_dedup(src: Path, dst: Path):
    """Move file to destination, dropping duplicate source if destination already exists."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        logger.warning(f"Duplicate file detected, keeping destination copy and dropping source: {dst.name}")
        src.unlink(missing_ok=True)
        return
    shutil.move(src, dst)


def consolidate_split_back_to_source():
    """Bring previously split files back to source pools so re-splitting uses full dataset."""
    moved_images = 0
    moved_labels = 0
    moved_ts = 0
    moved_roi_labels = 0

    split_names = ["train", "val", "test"]
    for split_name in split_names:
        split_root = DATA_FINAL / split_name
        if not split_root.exists():
            continue

        split_img = split_root / "images"
        split_lbl = split_root / "labels"
        split_ts = split_root / "time_series"

        if split_img.exists():
            for f in split_img.glob("*.png"):
                _move_with_dedup(f, SOURCE_IMG / f.name)
                moved_images += 1

        if split_lbl.exists():
            for f in split_lbl.glob("*.txt"):
                _move_with_dedup(f, SOURCE_LBL / f.name)
                moved_labels += 1

        if split_ts.exists():
            for f in split_ts.glob("*_ts.npy"):
                _move_with_dedup(f, SOURCE_TS / f.name)
                moved_ts += 1
            for f in split_ts.glob("*_roi_labels.npy"):
                _move_with_dedup(f, SOURCE_TS / f.name)
                moved_roi_labels += 1

    if moved_images or moved_labels or moved_ts or moved_roi_labels:
        logger.info(
            "Consolidated previous splits back to source pools: images=%d, labels=%d, ts=%d, roi_labels=%d",
            moved_images,
            moved_labels,
            moved_ts,
            moved_roi_labels,
        )
    else:
        logger.info("No existing split files needed consolidation")

def run_stratified_split():
    # 0. Consolidate existing split outputs back to source pools before rebuilding splits
    consolidate_split_back_to_source()

    # 1. Load labels to ensure stratification (Phase 2.2)
    df = pd.read_csv(PHENO_PATH)
    # Strip whitespace from FILE_ID to prevent match failures
    df['FILE_ID'] = df['FILE_ID'].astype(str).str.strip()
    # Only include subjects we actually have files for
    all_images = [f for f in os.listdir(SOURCE_IMG) if f.endswith('.png')]
    valid_ids = set([f.rsplit('_z', 1)[0] for f in all_images])
    df = df[df['FILE_ID'].isin(valid_ids)]
    
    # Check if master_manifest has pre-assigned splits (from prior runs)
    # If so, use those as authoritative source
    manifest_has_splits = False
    if MASTER_MANIFEST.exists():
        try:
            manifest_df = pd.read_csv(MASTER_MANIFEST)
            if 'split' in manifest_df.columns:
                manifest_has_splits = True
                logger.info("Found 'split' column in manifest — using pre-assigned splits")
                # Merge split assignments from manifest (supports subject_id or FILE_ID)
                manifest_id_col = 'FILE_ID' if 'FILE_ID' in manifest_df.columns else 'subject_id'
                if manifest_id_col not in manifest_df.columns:
                    logger.warning("Manifest has split column but no FILE_ID/subject_id column; ignoring pre-assigned splits")
                    manifest_has_splits = False
                else:
                    manifest_df = manifest_df.copy()
                    manifest_df['FILE_ID'] = manifest_df[manifest_id_col].astype(str).str.strip()
                    df = df.merge(manifest_df[['FILE_ID', 'split']], on='FILE_ID', how='left')
                # Any subjects not in manifest get None in split column
                df['split'] = df['split'].fillna('unknown')
        except Exception as e:
            logger.debug(f"Could not load splits from manifest: {e}")
    
    # If manifest doesn't have splits, perform 2D stratification from scratch
    if not manifest_has_splits:
        # Remove singleton groups that would break stratification
        # Group subjects to find groups with too few samples
        group_counts = df.groupby(['DX_GROUP', 'SITE_ID']).size()
        
        # Adaptive strategy: start with min_group_size=7, but fall back to lower values if needed
        min_group_size = 7
        for attempted_min in [7, 5, 3, 2, 1]:
            valid_groups = group_counts[group_counts >= attempted_min].index
            if len(valid_groups) > 0:
                min_group_size = attempted_min
                break
        
        if len(valid_groups) == 0:
            logger.error(f"No valid stratification groups found. All DX_GROUP×SITE_ID combinations are empty.")
            logger.error(f"Group size distribution: {group_counts.to_dict()}")
            raise ValueError("Cannot proceed with stratification: no valid groups remain after filtering")
        
        if min_group_size < 7:
            logger.warning(f"⚠️  Data too fragmented for strict 2D stratification. Using relaxed minimum of {min_group_size} members per group.")
        
        if len(valid_groups) < len(group_counts):
            removed_count = len(df.groupby(['DX_GROUP', 'SITE_ID'])) - len(valid_groups)
            logger.warning(f"Removed {removed_count} DX_GROUP×SITE_ID groups with < {min_group_size} members")
        
        df = df.set_index(['DX_GROUP', 'SITE_ID']).loc[valid_groups].reset_index()
        
        if len(df) < 100:
            logger.warning(f"⚠️  Only {len(df)} subjects remain after filtering. This may indicate a data availability issue.")
        
        logger.info(f"Filtered to {len(df)} subjects using stratification groups (min {min_group_size} per DX_GROUP×SITE_ID)")

    # 2. Perform stratification or use pre-assigned splits
    # This ensures a 'site-balanced' split, which is a Q1 Journal requirement
    
    if manifest_has_splits:
        # Use pre-assigned splits from manifest
        logger.info(f"Using pre-assigned splits from manifest ({len(df[df['split']=='train'])} train, {len(df[df['split']=='val'])} val, {len(df[df['split']=='test'])} test)")
        splits = {
            'train': df[df['split'] == 'train'].copy(),
            'val': df[df['split'] == 'val'].copy(),
            'test': df[df['split'] == 'test'].copy(),
        }
    else:
        # Perform 2D stratified split
        train_df, rem_df = train_test_split(
            df, train_size=TRAIN_RATIO, stratify=df[['DX_GROUP', 'SITE_ID']], random_state=42
        )
        
        # Split the remaining 30% into half (15% Val, 15% Test)
        val_df, test_df = train_test_split(
            rem_df, train_size=0.5, stratify=rem_df[['DX_GROUP', 'SITE_ID']], random_state=42
        )

        splits = {'train': train_df, 'val': val_df, 'test': test_df}
        
        # Save split assignments to DataFrame
        df['split'] = 'unknown'
        df.loc[df['FILE_ID'].isin(train_df['FILE_ID']), 'split'] = 'train'
        df.loc[df['FILE_ID'].isin(val_df['FILE_ID']), 'split'] = 'val'
        df.loc[df['FILE_ID'].isin(test_df['FILE_ID']), 'split'] = 'test'
    
    # Generate cv_fold for train split (always regenerate for harmonization alignment)
    if 'cv_fold' not in df.columns or df['cv_fold'].isna().any():
        df['cv_fold'] = -1
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx_overall = df.index[df['split'] == 'train'].tolist()
        
        if len(train_idx_overall) > 0:
            strat_col = df.loc[train_idx_overall, 'DX_GROUP'].astype(str) + "_" + df.loc[train_idx_overall, 'SITE_ID'].astype(str)
            
            if (strat_col.value_counts() < 2).any():
                logger.warning("Some DX_GROUP×SITE_ID combinations have <2 samples, falling back to DX_GROUP-only stratification for CV folds")
                strat_col = df.loc[train_idx_overall, 'DX_GROUP'].astype(str)
                
            for fold, (_, val_idx) in enumerate(skf.split(train_idx_overall, strat_col)):
                actual_val_indices = [train_idx_overall[i] for i in val_idx]
                df.loc[actual_val_indices, 'cv_fold'] = fold
            
            logger.info(f"Generated 5-fold CV assignments for {len(train_idx_overall)} training subjects")
    
    # Save updated manifest with splits and cv_fold
    MASTER_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    if 'subject_id' not in df.columns:
        df['subject_id'] = df['FILE_ID']
    df.to_csv(MASTER_MANIFEST, index=False)
    logger.info(f"Saved splits and CV folds to {MASTER_MANIFEST}")


    # 3. Clean previous split outputs to prevent cross-split duplicates on re-runs
    for split_name in ['train', 'val', 'test']:
        split_root = DATA_FINAL / split_name
        if split_root.exists():
            logger.warning(f"Clearing existing split directory before rewrite: {split_root}")
            shutil.rmtree(split_root)

    # 4. Execute Move
    for name, split_df in splits.items():
        logger.info(f"📦 Organizing {name} set ({len(split_df)} subjects)...")
        
        img_dst = DATA_FINAL / name / 'images'
        lbl_dst = DATA_FINAL / name / 'labels'
        ts_dst  = DATA_FINAL / name / 'time_series'
        
        for d in [img_dst, lbl_dst, ts_dst]: d.mkdir(parents=True, exist_ok=True)

        for sub_id in split_df['FILE_ID']:
            # Move all slices
            for f in [img for img in all_images if img.startswith(sub_id + "_z")]:
                shutil.move(SOURCE_IMG / f, img_dst / f)
                # Move label if it exists (Phase 3.2)
                lbl_f = f.replace('.png', '.txt')
                if (SOURCE_LBL / lbl_f).exists():
                    shutil.move(SOURCE_LBL / lbl_f, lbl_dst / lbl_f)
            
            # Move Time Series (Phase 4.1)
            ts_f = f"{sub_id}_ts.npy"
            if (SOURCE_TS / ts_f).exists():
                shutil.move(SOURCE_TS / ts_f, ts_dst / ts_f)

            roi_labels_f = f"{sub_id}_roi_labels.npy"
            if (SOURCE_TS / roi_labels_f).exists():
                shutil.move(SOURCE_TS / roi_labels_f, ts_dst / roi_labels_f)

    logger.info(f"\n✅ SUCCESS: Stratified split complete. Saved to {DATA_FINAL}")

if __name__ == "__main__":
    run_stratified_split()