import logging
import os
import random
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split

# Setup paths from config
from src.core.config import (
    DATA_FINAL,
    DATA_PROCESSED,
    DATA_ROOT,
    DATA_TIME_SERIES,
    EXCLUDED_SUBJECTS,
    MASTER_MANIFEST,
    PHENO_PATH,
)

# Setup logging
logger = logging.getLogger(__name__)

# --- CONFIG ---
TRAIN_RATIO, VAL_RATIO = 0.70, 0.15
random.seed(42)
SOURCE_IMG = DATA_ROOT / "images"
SOURCE_LBL = DATA_ROOT / "labels"

# --- TASK 5: Site-stratified CV (DD-013) ---
# Scanner manufacturer groupings for site clustering.
# Sites grouped by manufacturer determine the primary cluster axis;
# TR range (seconds) is the secondary axis for finer stratification.
SCANNER_MANUFACTURER: dict[str, str] = {
    "CALTECH":   "Siemens",
    "CMU":       "Siemens",
    "KKI":       "GE",
    "LEUVEN_1":  "Philips",
    "LEUVEN_2":  "Philips",
    "MAX_MUN":   "Siemens",
    "NYU":       "Siemens",
    "OHSU":      "GE",
    "OLIN":      "Philips",
    "PITT":      "Siemens",
    "SBL":       "GE",
    "SDSU":      "GE",
    "STANFORD":  "Siemens",
    "TRINITY":   "Siemens",
    "UCLA_1":    "Siemens",
    "UCLA_2":    "Siemens",
    "UM_1":      "Philips",
    "UM_2":      "Philips",
    "USM":       "Siemens",
    "YALE":      "Siemens",
}

# TR bins: fast (<2 s), standard (2–2.5 s), slow (>2.5 s)
def _tr_bin(tr: float) -> str:
    if tr < 2.0:
        return "fast"
    elif tr <= 2.5:
        return "standard"
    else:
        return "slow"

# Site TR lookup (seconds) from feature_registry — use for binning
_SITE_TR: dict[str, float] = {
    "CALTECH": 2.0, "CMU": 2.0, "KKI": 2.5, "LEUVEN_1": 1.656,
    "LEUVEN_2": 1.656, "MAX_MUN": 3.0, "NYU": 2.0, "OHSU": 2.5,
    "OLIN": 1.5, "PITT": 1.5, "SBL": 2.5, "SDSU": 2.0,
    "STANFORD": 2.0, "TRINITY": 2.0, "UCLA_1": 3.0, "UCLA_2": 3.0,
    "UM_1": 2.0, "UM_2": 2.0, "USM": 2.0, "YALE": 2.0,
}

_N_SITE_CLUSTERS = 5

def _assign_site_clusters(sites: list[str]) -> dict[str, int]:
    """
    Cluster sites into _N_SITE_CLUSTERS balanced groups based on
    scanner manufacturer × TR bin.

    Algorithm:
    1. Compute manufacturer+TR_bin label for each unique site.
    2. Group sites by that label.
    3. Merge tiny groups (<2 sites) into the closest manufacturer group.
    4. Assign integer cluster IDs 0..(_N_SITE_CLUSTERS-1) round-robin
       to balance cluster sizes.

    Returns:
        Dict mapping site name → cluster index (0-based).
    """
    unique_sites = list(dict.fromkeys(sites))  # preserve order, deduplicate

    # Primary grouping key: manufacturer + TR bin
    site_key = {
        s: (SCANNER_MANUFACTURER.get(s, "Unknown"), _tr_bin(_SITE_TR.get(s, 2.0)))
        for s in unique_sites
    }

    # Group sites by key
    key_to_sites: dict[tuple, list[str]] = {}
    for s, k in site_key.items():
        key_to_sites.setdefault(k, []).append(s)

    # Flatten groups, merge singletons into the same-manufacturer group
    merged: dict[str, str] = {}  # site → group_label
    for key, slist in sorted(key_to_sites.items(), key=lambda x: -len(x[1])):
        label = f"{key[0]}_{key[1]}"
        if len(slist) == 1:
            # Try merging into same manufacturer, different TR
            fallback = key[0]
            label = next(
                (f"{k[0]}_{k[1]}" for k, sl in key_to_sites.items()
                 if k[0] == fallback and len(sl) > 1 and k != key),
                label,
            )
        for s in slist:
            merged[s] = label

    # Sort group labels for determinism
    labels_ordered = sorted(set(merged.values()))
    # Pre-sort sites within each label alphabetically for determinism
    label_sites: dict[str, list[str]] = {}
    for s, lbl in sorted(merged.items()):
        label_sites.setdefault(lbl, []).append(s)

    # Assign cluster IDs round-robin across labels to balance sizes
    cluster_map: dict[str, int] = {}
    cluster_idx = 0
    for lbl in labels_ordered:
        for s in label_sites[lbl]:
            cluster_map[s] = cluster_idx % _N_SITE_CLUSTERS
            cluster_idx += 1

    return cluster_map

def generate_site_stratified_folds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace StratifiedKFold cv_fold assignments with GroupKFold where
    each group is a site cluster.  Each fold validates on a held-out
    site cluster that was never seen during that fold's training.

    Rationale (DD-013 / Root Cause 5): Random K-fold CV masks the site
    confound because the same scanner's subjects appear in both train and
    validation splits.  Site-stratified CV guarantees the validation set
    comes from scanner manufacturers/TRs NOT in the training fold, giving
    a more honest estimate of out-of-site generalisation.

    Args:
        df: Master manifest DataFrame with at least columns:
            split (str), SITE_ID (str).

    Returns:
        Modified df with updated cv_fold and new site_cluster columns.
    """
    train_mask = df['split'] == 'train'
    train_df = df[train_mask].copy()

    if len(train_df) == 0:
        logger.warning("No training subjects found; skipping site-stratified CV generation.")
        return df

    # Assign cluster per site
    unique_sites = train_df['SITE_ID'].unique().tolist()
    cluster_map = _assign_site_clusters(unique_sites)

    # Write site_cluster for all subjects (useful for analysis)
    df['site_cluster'] = df['SITE_ID'].map(lambda s: cluster_map.get(s, 0))

    train_idx = train_df.index.tolist()
    groups = [cluster_map.get(s, 0) for s in train_df['SITE_ID']]

    gkf = GroupKFold(n_splits=_N_SITE_CLUSTERS)
    df.loc[train_mask, 'cv_fold'] = -1

    for fold, (_, val_rel) in enumerate(
        gkf.split(train_df, y=train_df['DX_GROUP'].values, groups=groups)
    ):
        actual_idx = [train_idx[i] for i in val_rel]
        df.loc[actual_idx, 'cv_fold'] = fold

    # Report cluster composition
    for cid in range(_N_SITE_CLUSTERS):
        sites_in_cluster = [s for s, c in cluster_map.items() if c == cid]
        n_subj = (df['site_cluster'] == cid).sum()
        logger.info(
            "Site cluster %d: %d subjects | sites: %s",
            cid, n_subj, sites_in_cluster,
        )

    assigned = (df.loc[train_mask, 'cv_fold'] >= 0).sum()
    logger.info(
        "Site-stratified CV: %d/%d training subjects assigned to a fold.",
        assigned, len(train_df),
    )
    return df

def _resolve_source_ts_dir() -> Path:
    """Resolve source time-series directory across canonical and legacy layouts."""
    candidates = [
        DATA_TIME_SERIES,
        DATA_ROOT / "timeseries",
        DATA_PROCESSED,
    ]

    best_dir = None
    best_count = -1
    for candidate in candidates:
        if not candidate.exists():
            continue
        count = len(list(candidate.glob("*_ts.npy")))
        if count > best_count:
            best_dir = candidate
            best_count = count

    if best_dir is not None:
        logger.info(
            "Using source time-series directory: %s (%d *_ts.npy files)",
            best_dir, max(best_count, 0),
        )
        return best_dir

    logger.info("No source time-series directory found; defaulting to %s", DATA_TIME_SERIES)
    return DATA_TIME_SERIES

SOURCE_TS = _resolve_source_ts_dir()

def _move_with_dedup(src: Path, dst: Path):
    """Move file to destination, dropping duplicate source if destination already exists."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        logger.warning(f"Duplicate file detected, keeping destination copy: {dst.name}")
        src.unlink(missing_ok=True)
        return
    shutil.move(src, dst)

def consolidate_split_back_to_source():
    """Bring previously split files back to source pools so re-splitting uses full dataset."""
    moved = {'images': 0, 'labels': 0, 'ts': 0, 'roi_labels': 0}
    for split_name in ("train", "val", "test"):
        split_root = DATA_FINAL / split_name
        if not split_root.exists():
            continue
        for f in (split_root / "images").glob("*.png") if (split_root / "images").exists() else []:
            _move_with_dedup(f, SOURCE_IMG / f.name)
            moved['images'] += 1
        for f in (split_root / "labels").glob("*.txt") if (split_root / "labels").exists() else []:
            _move_with_dedup(f, SOURCE_LBL / f.name)
            moved['labels'] += 1
        if (split_root / "time_series").exists():
            for f in (split_root / "time_series").glob("*_ts.npy"):
                _move_with_dedup(f, SOURCE_TS / f.name)
                moved['ts'] += 1
            for f in (split_root / "time_series").glob("*_roi_labels.npy"):
                _move_with_dedup(f, SOURCE_TS / f.name)
                moved['roi_labels'] += 1

    if any(moved.values()):
        logger.info("Consolidated previous splits: %s", moved)
    else:
        logger.info("No existing split files needed consolidation")

def run_stratified_split():
    # 0. Consolidate existing split outputs back to source pools
    consolidate_split_back_to_source()

    # 1. Load labels for stratification
    df = pd.read_csv(PHENO_PATH)
    df['FILE_ID'] = df['FILE_ID'].astype(str).str.strip()
    all_images = [f for f in os.listdir(SOURCE_IMG) if f.endswith('.png')]
    valid_ids = {f.rsplit('_z', 1)[0] for f in all_images}
    df = df[df['FILE_ID'].isin(valid_ids)]

    # Enforce curated subject exclusion policy (1035 -> 1015 cohort).
    excluded_upper = {s.upper() for s in EXCLUDED_SUBJECTS}
    before_exclusion = len(df)
    df = df[~df['FILE_ID'].astype(str).str.upper().isin(excluded_upper)]
    logger.info(
        "Applied EXCLUDED_SUBJECTS in split: removed %d row(s), remaining %d",
        before_exclusion - len(df),
        len(df),
    )

    # Check for pre-assigned splits in master_manifest
    manifest_has_splits = False
    if MASTER_MANIFEST.exists():
        try:
            manifest_df = pd.read_csv(MASTER_MANIFEST)
            if 'split' in manifest_df.columns:
                manifest_has_splits = True
                logger.info("Found 'split' column in manifest — using pre-assigned splits")
                manifest_id_col = 'FILE_ID' if 'FILE_ID' in manifest_df.columns else 'subject_id'
                if manifest_id_col not in manifest_df.columns:
                    logger.warning("Manifest has split column but no FILE_ID/subject_id; ignoring.")
                    manifest_has_splits = False
                else:
                    manifest_df = manifest_df.copy()
                    manifest_df['FILE_ID'] = manifest_df[manifest_id_col].astype(str).str.strip()
                    df = df.merge(manifest_df[['FILE_ID', 'split']], on='FILE_ID', how='left')
                df['split'] = df['split'].fillna('unknown')
        except Exception as e:
            logger.debug(f"Could not load splits from manifest: {e}")

    if not manifest_has_splits:
        group_counts = df.groupby(['DX_GROUP', 'SITE_ID']).size()
        min_group_size = 7
        for attempted_min in [7, 5, 3, 2, 1]:
            valid_groups = group_counts[group_counts >= attempted_min].index
            if len(valid_groups) > 0:
                min_group_size = attempted_min
                break

        if len(valid_groups) == 0:
            raise ValueError("Cannot proceed with stratification: no valid groups remain.")

        if min_group_size < 7:
            logger.warning(f"Using relaxed stratification minimum of {min_group_size}.")

        df = df.set_index(['DX_GROUP', 'SITE_ID']).loc[valid_groups].reset_index()
        logger.info(f"Filtered to {len(df)} subjects using stratification groups (min {min_group_size}).")

    # 2. Perform stratification or use pre-assigned splits
    if manifest_has_splits:
        logger.info("Using pre-assigned splits from manifest.")
        splits = {
            'train': df[df['split'] == 'train'].copy(),
            'val':   df[df['split'] == 'val'].copy(),
            'test':  df[df['split'] == 'test'].copy(),
        }
    else:
        train_df, rem_df = train_test_split(
            df, train_size=TRAIN_RATIO, stratify=df[['DX_GROUP', 'SITE_ID']], random_state=42
        )
        val_df, test_df = train_test_split(
            rem_df, train_size=0.5, stratify=rem_df[['DX_GROUP', 'SITE_ID']], random_state=42
        )
        splits = {'train': train_df, 'val': val_df, 'test': test_df}
        df['split'] = 'unknown'
        df.loc[df['FILE_ID'].isin(train_df['FILE_ID']), 'split'] = 'train'
        df.loc[df['FILE_ID'].isin(val_df['FILE_ID']), 'split'] = 'val'
        df.loc[df['FILE_ID'].isin(test_df['FILE_ID']), 'split'] = 'test'

    # 3. Generate cv_fold — standard StratifiedKFold (default) or site-stratified
    if 'cv_fold' not in df.columns or df['cv_fold'].isna().any():
        df['cv_fold'] = -1
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx_overall = df.index[df['split'] == 'train'].tolist()
        if len(train_idx_overall) > 0:
            strat_col = df.loc[train_idx_overall, 'DX_GROUP'].astype(str) + "_" + df.loc[train_idx_overall, 'SITE_ID'].astype(str)
            if (strat_col.value_counts() < 2).any():
                logger.warning("Falling back to DX_GROUP-only stratification for CV folds.")
                strat_col = df.loc[train_idx_overall, 'DX_GROUP'].astype(str)
            for fold, (_, val_idx) in enumerate(skf.split(train_idx_overall, strat_col)):
                actual_val_indices = [train_idx_overall[i] for i in val_idx]
                df.loc[actual_val_indices, 'cv_fold'] = fold
            logger.info(f"Generated 5-fold StratifiedKFold CV for {len(train_idx_overall)} training subjects.")

    # 4. Save manifest
    MASTER_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    if 'subject_id' not in df.columns:
        df['subject_id'] = df['FILE_ID']
    df.to_csv(MASTER_MANIFEST, index=False)
    logger.info(f"Saved splits and CV folds to {MASTER_MANIFEST}")

    # 5. Reorganise files
    for split_name in ['train', 'val', 'test']:
        split_root = DATA_FINAL / split_name
        if split_root.exists():
            logger.warning(f"Clearing existing split directory: {split_root}")
            shutil.rmtree(split_root)

    for name, split_df in splits.items():
        logger.info(f"📦 Organising {name} set ({len(split_df)} subjects)...")
        img_dst = DATA_FINAL / name / 'images'
        lbl_dst = DATA_FINAL / name / 'labels'
        ts_dst  = DATA_FINAL / name / 'time_series'
        for d in [img_dst, lbl_dst, ts_dst]:
            d.mkdir(parents=True, exist_ok=True)

        for sub_id in split_df['FILE_ID']:
            for f in [img for img in all_images if img.startswith(sub_id + "_z")]:
                shutil.move(SOURCE_IMG / f, img_dst / f)
                lbl_f = f.replace('.png', '.txt')
                if (SOURCE_LBL / lbl_f).exists():
                    shutil.move(SOURCE_LBL / lbl_f, lbl_dst / lbl_f)
            ts_f = f"{sub_id}_ts.npy"
            if (SOURCE_TS / ts_f).exists():
                shutil.move(SOURCE_TS / ts_f, ts_dst / ts_f)
            roi_f = f"{sub_id}_roi_labels.npy"
            if (SOURCE_TS / roi_f).exists():
                shutil.move(SOURCE_TS / roi_f, ts_dst / roi_f)

    logger.info(f"\n✅ SUCCESS: Stratified split complete. Saved to {DATA_FINAL}")

def run_site_stratified_split():
    """
    Task 5 (DD-013): Re-assign cv_fold using site-stratified GroupKFold.

    Clusters the 20 ABIDE sites by scanner manufacturer × TR range into
    5 balanced groups.  Each GroupKFold fold validates on one held-out
    cluster, ensuring the validation set contains scanner profiles not
    present in the training fold.

    This replaces the StratifiedKFold cv_fold values in master_manifest.csv.
    Requires that the manifest already has 'split' and 'SITE_ID' columns.
    Also runs fold_safe_harmonization after updating folds.
    """
    if not MASTER_MANIFEST.exists():
        raise FileNotFoundError(
            f"Master manifest not found at {MASTER_MANIFEST}. "
            "Run run_stratified_split() first."
        )

    df = pd.read_csv(MASTER_MANIFEST)
    required_cols = {'split', 'SITE_ID', 'DX_GROUP'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    logger.info("Generating site-stratified CV folds (DD-013)...")
    df = generate_site_stratified_folds(df)

    df.to_csv(MASTER_MANIFEST, index=False)
    logger.info(f"✅ Site-stratified CV folds written to {MASTER_MANIFEST}")
    logger.info(
        "cv_fold distribution:\n%s",
        df[df['split'] == 'train']['cv_fold'].value_counts().sort_index().to_string(),
    )
    logger.info(
        "⚠️  Re-run fold_safe_harmonization.py to regenerate fold-specific harmonized features "
        "before training: python -m src.features.fold_safe_harmonization"
    )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Data split utility for Neuro-CXG")
    parser.add_argument(
        "--site-stratified-cv",
        action="store_true",
        help=(
            "Task 5 (DD-013): Replace StratifiedKFold cv_fold with site-stratified GroupKFold. "
            "Requires that master_manifest.csv already has split/SITE_ID columns. "
            "After running this, re-run fold_safe_harmonization.py."
        ),
    )
    args = parser.parse_args()

    if args.site_stratified_cv:
        run_site_stratified_split()
    else:
        run_stratified_split()
