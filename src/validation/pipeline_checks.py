import argparse
import logging
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image

# Setup paths from config
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import (
    ALL_FEATURE_NAMES,
    ATLAS_PATH,
    CAUSAL_GRAPHS_DIR,
    CHECKPOINT_DIR,
    DATA_FINAL,
    DATA_METADATA,
    DATA_PROCESSED,
    DATA_ROOT,
    EXCLUDED_SUBJECTS,
    GNN_IN_CHANNELS,
    K_FOLDS,
    LOBE_MAPPING,
    MASTER_MANIFEST,
    MIN_EDGES_PER_GRAPH,
    NODE_ATTRIBUTES_HARMONIZED,
    NODE_ATTRIBUTES_TEMPORAL,
    NODE_FEATURES_3D,
    NUM_LOBES,
    NUM_SPATIAL_FEATURES,
    NUM_TEMPORAL_FEATURES,
    PHENO_PATH,
    SITE_TR_MAP,
    SPARSITY_QUANTILE,
    HARMONIZED_FOLDS_DIR,
)
from src.core.hyperparams import GNN_MAX_DEGENERATE_GRAPH_RATE
from src.core.validators import summarize_graph_degeneracy_from_adj

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Integrity defaults
# abide_download.py saves 7 z-slices per subject (percentiles 0.2-0.8)
TARGET_SLICES = 7
VALID_ROI_RANGE = (164, 170)  # AAL3v1 atlas variants
PNG_DIR = DATA_ROOT / "images"
TS_DIR = DATA_PROCESSED


def _redownload_npy(corrupted_npy_paths: list, incomplete_subs: list) -> None:
    """
    Re-download time-series NPY files for corrupted/incomplete subjects.

    Keeps PNGs intact. Deletes existing bad NPY files, then re-runs
    process_subject() from abide_download.py for each affected subject.
    TR values are loaded from the phenotype CSV (with site-based fallback).
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm
    from src.data.abide_download import process_subject, init_worker, PHENO_PATH

    # Collect subject IDs that need re-download
    subjects_to_fix: set = set()
    for path in corrupted_npy_paths:
        # both _ts.npy and _roi_labels.npy map to the same subject ID
        sub_id = path.name.replace("_ts.npy", "").replace("_roi_labels.npy", "")
        subjects_to_fix.add(sub_id)
    for sub in incomplete_subs:
        # incomplete because ts is missing (PNG exists but no ts.npy)
        if not (TS_DIR / f"{sub}_ts.npy").exists():
            subjects_to_fix.add(sub)

    # Respect curated cohort exclusions; these subjects are intentionally removed.
    excluded_upper = {s.upper() for s in EXCLUDED_SUBJECTS}
    subjects_to_fix = {s for s in subjects_to_fix if s.upper() not in excluded_upper}

    if not subjects_to_fix:
        logger.info("No subjects require NPY re-download.")
        return

    logger.info(f"Re-downloading NPY for {len(subjects_to_fix)} subject(s)...")

    # Delete all existing NPY files for these subjects so the idempotency
    # check in process_subject() doesn't skip them
    for sub_id in subjects_to_fix:
        for pattern in (f"{sub_id}_ts.npy", f"{sub_id}_roi_labels.npy"):
            p = TS_DIR / pattern
            if p.exists():
                os.remove(p)
                logger.info(f"  Removed stale: {p.name}")

    # Build (subject_id, tr) task list from phenotype CSV
    tr_lookup: dict = {}
    if PHENO_PATH.exists():
        pheno_df = pd.read_csv(PHENO_PATH)
        pheno_df['FILE_ID'] = pheno_df['FILE_ID'].astype(str).str.strip()
        if 'TR' not in pheno_df.columns:
            pheno_df['TR'] = pheno_df['SITE_ID'].map(SITE_TR_MAP).fillna(2.0)
        else:
            pheno_df['TR'] = pd.to_numeric(pheno_df['TR'], errors='coerce').fillna(2.0)
        tr_lookup = dict(zip(pheno_df['FILE_ID'], pheno_df['TR']))

    tasks = [
        (sub_id, tr_lookup.get(sub_id, 2.0))
        for sub_id in sorted(subjects_to_fix)
    ]

    # Re-run extraction (same worker pool as abide_download.py)
    with ProcessPoolExecutor(max_workers=6, initializer=init_worker) as exe:
        futures = {exe.submit(process_subject, sub_id, tr): sub_id for sub_id, tr in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Re-downloading NPY"):
            sub_id = futures[fut]
            try:
                _, status, err = fut.result()
                if status == "Failed":
                    logger.error(f"  FAILED {sub_id}: {err}")
                else:
                    logger.info(f"  OK {sub_id} ({status})")
            except Exception as exc:
                logger.error(f"  EXCEPTION {sub_id}: {exc}")

    logger.info("NPY re-download complete. Re-run --dataset to verify.")


def check_dataset_integrity() -> None:
    """
    Post-download integrity check.

    Verifies:
    - PNG files are not corrupted
    - NPY time-series files are valid (no NaNs)
    - Subjects have all required data (slices + time series)

    Offers interactive cleanup options for corrupted/incomplete data.
    """
    logger.info("Starting post-download integrity check...")

    # 1. Check PNG Integrity
    # After split.py runs it moves all PNGs into data/final/{split}/images/,
    # leaving the source-pool dir (PNG_DIR) empty.  Prefer the split dirs.
    _split_img_dirs = [DATA_FINAL / sp / "images" for sp in ("train", "val", "test")]
    _split_pngs = [p for d in _split_img_dirs if d.exists() for p in d.glob("*.png")]
    png_files = _split_pngs if _split_pngs else list(PNG_DIR.glob("*.png"))
    corrupted_pngs = []
    subject_counts = Counter()

    if not png_files:
        logger.warning(
            "No PNG images found in split dirs (%s) or source pool (%s) — "
            "integrity check skipped.  Run split stage first.",
            ", ".join(str(d) for d in _split_img_dirs),
            PNG_DIR,
        )
    else:
        logger.info(f"Scanning {len(png_files)} images for integrity...")
        for path in png_files:
            try:
                with Image.open(path) as img:
                    img.verify()
                sub_id = path.name.rsplit("_z", 1)[0]
                subject_counts[sub_id] += 1
            except Exception:
                logger.warning(f" [!] Corrupted PNG: {path.name}")
                corrupted_pngs.append(path)

    # 2. Check NPY Integrity
    # Time-series files live in data/final/{split}/time_series/ after split.
    _split_ts_dirs = [DATA_FINAL / sp / "time_series" for sp in ("train", "val", "test")]
    _split_npys = [p for d in _split_ts_dirs if d.exists() for p in d.glob("*.npy")]
    npy_files = _split_npys if _split_npys else list(TS_DIR.glob("*.npy"))
    corrupted_npys = []
    ts_subjects = set()

    logger.info(f"Scanning {len(npy_files)} time-series files...")
    for path in npy_files:
        try:
            data = np.load(path)

            if path.name.endswith("_roi_labels.npy"):
                # 1D integer label array — just check dimensionality
                if data.ndim != 1:
                    raise ValueError(f"Expected 1D array, got {data.ndim}D")
                # No NaN check: label IDs are integers, NaN doesn't apply here

            elif path.name.endswith("_ts.npy"):
                # 2D time-series array: (timepoints, num_rois)
                if data.ndim != 2:
                    raise ValueError(f"Expected 2D array, got {data.ndim}D")
                num_rois = data.shape[1]
                if not (VALID_ROI_RANGE[0] <= num_rois <= VALID_ROI_RANGE[1]):
                    raise ValueError(
                        f"ROI mismatch: Found {num_rois}, expected {VALID_ROI_RANGE[0]}-{VALID_ROI_RANGE[1]}"
                    )
                # Whole-column NaNs are intentional: abide_download marks empty ROIs
                # (zero voxels after atlas resampling) with NaN so downstream can
                # detect and impute them. Only flag *scattered* NaNs within a column,
                # which indicate a real extraction failure.
                nan_mask = np.isnan(data)
                empty_roi_cols = int(np.all(nan_mask, axis=0).sum())   # fully-NaN columns
                scattered_nan_cols = int(np.any(nan_mask, axis=0).sum()) - empty_roi_cols
                if scattered_nan_cols > 0:
                    raise ValueError(
                        f"Scattered NaNs in {scattered_nan_cols} ROI(s) — extraction error"
                    )
                # >50% empty ROIs means the masker extracted almost nothing useful.
                # Treat these as corrupted so the interactive prompt can delete them.
                MAX_EMPTY_ROI_FRACTION = 0.50
                max_allowed_empty = int(num_rois * MAX_EMPTY_ROI_FRACTION)
                if empty_roi_cols > max_allowed_empty:
                    raise ValueError(
                        f"{empty_roi_cols}/{num_rois} empty ROIs (>{int(MAX_EMPTY_ROI_FRACTION*100)}%) "
                        f"— masker extraction failed, subject unusable"
                    )
                if empty_roi_cols > 10:
                    logger.warning(
                        " [W] %s: %d empty ROIs (whole-column NaN) — may affect feature quality",
                        path.name, empty_roi_cols,
                    )
                ts_subjects.add(path.name.replace("_ts.npy", ""))

        except Exception as e:
            logger.warning(f" [!] Invalid NPY: {path.name} ({e})")
            corrupted_npys.append(path)

    # 3. Identify Incomplete Subjects
    all_subjects = set(subject_counts.keys()) | ts_subjects
    incomplete_subs = []
    for sub in all_subjects:
        if subject_counts[sub] < TARGET_SLICES or sub not in ts_subjects:
            incomplete_subs.append(sub)

    logger.info("\n" + "=" * 40)
    logger.info("POST-DOWNLOAD INTEGRITY REPORT")
    logger.info("=" * 40)
    logger.info(f"Corrupted PNGs:      {len(corrupted_pngs)}")
    logger.info(f"Corrupted NPYs:      {len(corrupted_npys)}")
    logger.info(f"Incomplete Subjects: {len(incomplete_subs)} (Missing slices or TS)")
    logger.info("-" * 40)

    if corrupted_pngs or corrupted_npys or incomplete_subs:
        logger.info(
            "OPTIONS: [1] Delete Corrupted | [2] Purge Incomplete Subjects | "
            "[3] Exit | [4] Re-download NPY only (keep PNGs)"
        )

        # In non-interactive mode (pipeline --auto flag, or CI/scripted runs)
        # automatically delete corrupted/invalid files and skip the interactive
        # menu entirely.  The flag is propagated via the NEURO_CXG_NONINTERACTIVE
        # environment variable set by run_pipeline.py.
        _is_auto = (
            os.environ.get("NEURO_CXG_NONINTERACTIVE", "").strip() == "1"
            or not sys.stdout.isatty()
        )
        if _is_auto:
            if corrupted_npys:
                logger.info(
                    "Non-interactive mode: auto-deleting %d corrupted NPY file(s).",
                    len(corrupted_npys),
                )
                for path in corrupted_npys:
                    os.remove(path)
                    logger.info("Auto-deleted: %s", path.name)
            if corrupted_pngs:
                logger.info(
                    "Non-interactive mode: auto-deleting %d corrupted PNG file(s).",
                    len(corrupted_pngs),
                )
                for path in corrupted_pngs:
                    os.remove(path)
                    logger.info("Auto-deleted: %s", path.name)
            if incomplete_subs:
                logger.info(
                    "Non-interactive mode: %d incomplete subject(s) detected but NOT "
                    "auto-purged (re-run interactively to choose option 2 if desired).",
                    len(incomplete_subs),
                )
            return

        choice = input("Select (1/2/3/4): ")
        if choice == "1":
            for path in corrupted_pngs + corrupted_npys:
                os.remove(path)
                logger.info(f"Deleted: {path}")

        elif choice == "2":
            # After split.py runs, subject files live in the split dirs, not in the
            # source-pool dirs (PNG_DIR / TS_DIR).  Search all split dirs for a
            # complete purge of the requested subjects.
            _split_img_dirs_d2 = [DATA_FINAL / sp / "images" for sp in ("train", "val", "test")]
            _split_ts_dirs_d2  = [DATA_FINAL / sp / "time_series" for sp in ("train", "val", "test")]
            deleted = 0
            for sub in incomplete_subs:
                for img_d in _split_img_dirs_d2:
                    for path in img_d.glob(f"{sub}_z*.png"):
                        os.remove(path)
                        deleted += 1
                for ts_d in _split_ts_dirs_d2:
                    for pattern in (f"{sub}_ts.npy", f"{sub}_roi_labels.npy", f"{sub}_qc.json"):
                        for path in ts_d.glob(pattern):
                            os.remove(path)
                            deleted += 1
            if deleted:
                logger.info("Purge complete: %d file(s) removed.", deleted)
            else:
                logger.warning(
                    "Purge ran but found 0 files to delete.  "
                    "Subjects may already be absent from split directories."
                )

        elif choice == "4":
            _redownload_npy(corrupted_npys, incomplete_subs)


def check_distribution() -> None:
    """
    Pre-GNN integrity check.

    Validates dataset completeness across train/val/test splits:
    - Checks each subject has target number of slices
    - Verifies image/label file pairing for YOLO training
    - Reports distribution of slice counts per split
    """
    logger.info("Starting pre-GNN integrity check...")

    # Images are in data/final/{train,val,test}/images/ after splitting
    final_root = DATA_FINAL

    if not final_root.exists():
        logger.error(f"Path {final_root} does not exist. Run split.py first.")
        return

    splits = ["train", "val", "test"]

    logger.info(f"Dataset Completeness Report (Target: {TARGET_SLICES} slices/subject)")

    for split in splits:
        img_path = final_root / split / "images"
        lbl_path = final_root / split / "labels"

        if not img_path.exists():
            logger.warning(f"[!] Split '{split}' images folder missing.")
            continue

        png_paths = list(img_path.glob("*.png"))
        n_files = len(png_paths)
        subject_counts = Counter(p.name.rsplit("_z", 1)[0] for p in png_paths)

        # Analyze the distribution of slice counts
        dist = Counter(subject_counts.values())

        logger.info(f"\nSplit: {split.upper()}")
        logger.info(f"  Total Subjects: {len(subject_counts)}")

        for num_slices in sorted(dist.keys()):
            status = "✓" if num_slices == TARGET_SLICES else "X"
            logger.info(f"  {status} {num_slices} slices: {dist[num_slices]} subjects")

        # Check for matching labels (Critical for YOLO training)
        if lbl_path.exists():
            n_labels = len(list(lbl_path.glob("*.txt")))
            if n_labels != n_files:
                logger.warning(
                    f"  [!] ALERT: Image/Label Mismatch! ({n_files} images vs {n_labels} labels)"
                )
            else:
                logger.info(f"  ✓ Image/Label count matches ({n_files} files)")

    logger.info("\nPre-GNN integrity check complete.")


def analyze_class_distribution() -> None:
    """
    Comprehensive class distribution analysis.

    Analyzes:
    - Overall dataset class balance (ASD vs Control)
    - Per-split distribution (train/val/test)
    - Per-site distribution (top 10 sites)
    - Graph availability and its impact on class balance
    - Provides actionable recommendations based on imbalance severity
    """
    logger.info("=" * 70)
    logger.info("CLASS DISTRIBUTION ANALYSIS")
    logger.info("=" * 70)

    # Load manifest
    if not MASTER_MANIFEST.exists():
        logger.error(f"Manifest not found: {MASTER_MANIFEST}")
        return

    try:
        df = pd.read_csv(MASTER_MANIFEST)
    except FileNotFoundError:
        logger.error(f"File not found: {MASTER_MANIFEST}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing failed for {MASTER_MANIFEST}: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to load manifest: {e}")
        raise

    # Overall distribution
    logger.info("\n1. OVERALL DATASET:")
    logger.info("-" * 70)
    dx_counts = df["DX_GROUP"].value_counts()
    total = len(df)

    control_count = dx_counts.get(2, 0)
    asd_count = dx_counts.get(1, 0)

    logger.info(f"Total subjects: {total}")
    logger.info(f"Control (0): {control_count} ({control_count / total * 100:.1f}%)")
    logger.info(f"ASD (1): {asd_count} ({asd_count / total * 100:.1f}%)")
    if asd_count > 0:
        logger.info(f"Imbalance ratio: {control_count / asd_count:.2f}:1")

    # Per-split distribution
    logger.info("\n2. DISTRIBUTION BY SPLIT:")
    logger.info("-" * 70)
    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split]
        if len(split_df) == 0:
            continue

        split_dx = split_df["DX_GROUP"].value_counts()
        split_control = split_dx.get(2, 0)
        split_asd = split_dx.get(1, 0)
        split_total = len(split_df)

        logger.info(f"\n{split.upper()}:")
        logger.info(f"  Total: {split_total}")
        logger.info(f"  Control: {split_control} ({split_control / split_total * 100:.1f}%)")
        logger.info(f"  ASD: {split_asd} ({split_asd / split_total * 100:.1f}%)")
        if split_asd > 0:
            logger.info(f"  Ratio: {split_control / split_asd:.2f}:1")

    # Per-site distribution
    logger.info("\n3. DISTRIBUTION BY SITE (Top 10):")
    logger.info("-" * 70)

    if "SITE_ID" in df.columns:
        site_dx = (
            df.groupby(["SITE_ID", "DX_GROUP"])
            .size()
            .unstack(fill_value=0)
        )
        for col in (1, 2):  # ensure both ASD (1) and Control (2) columns exist
            if col not in site_dx.columns:
                site_dx[col] = 0
        site_dx = site_dx.rename(columns={1: "asd", 2: "control"})
        site_dx["total"] = site_dx["asd"] + site_dx["control"]
        site_dx["ratio"] = site_dx.apply(
            lambda r: f"{r['control'] / r['asd']:.2f}:1" if r["asd"] > 0 else "N/A", axis=1
        )
        for site, row in site_dx.sort_values("total", ascending=False).head(10).iterrows():
            logger.info(
                f"{str(site):20} | Total: {int(row['total']):4} | "
                f"Control: {int(row['control']):4} | ASD: {int(row['asd']):4} | "
                f"Ratio: {row['ratio']}"
            )

    # Check which subjects have graphs
    logger.info("\n4. SUBJECTS WITH CAUSAL GRAPHS:")
    logger.info("-" * 70)

    if CAUSAL_GRAPHS_DIR.exists():
        try:
            graph_files = list(CAUSAL_GRAPHS_DIR.glob("*_graph.pt"))
            graph_subjects = [f.stem.replace("_graph", "") for f in graph_files]

            # Find subjects with graphs
            df["has_graph"] = df["subject_id"].astype(str).isin(graph_subjects)

            graph_df = df[df["has_graph"]]

            if len(graph_df) > 0:
                graph_dx = graph_df["DX_GROUP"].value_counts()
                graph_control = graph_dx.get(2, 0)
                graph_asd = graph_dx.get(1, 0)

                logger.info(f"Subjects with graphs: {len(graph_df)}/{len(df)}")
                logger.info(f"  Control: {graph_control} ({graph_control / len(graph_df) * 100:.1f}%)")
                logger.info(f"  ASD: {graph_asd} ({graph_asd / len(graph_df) * 100:.1f}%)")
                if graph_asd > 0:
                    logger.info(f"  Ratio: {graph_control / graph_asd:.2f}:1")

                # Check if imbalance worsened after filtering
                original_ratio = control_count / asd_count if asd_count > 0 else 0
                graph_ratio = graph_control / graph_asd if graph_asd > 0 else 0

                if graph_ratio > original_ratio * 1.1:
                    logger.warning("\n  WARNING: Imbalance worsened after graph filtering!")
                    logger.warning(f"     Original: {original_ratio:.2f}:1 -> Graphs: {graph_ratio:.2f}:1")
            else:
                logger.warning("No subjects with graphs found")
        except Exception as e:
            logger.error(f"Failed to check graph files: {e}")

    # Recommendations
    logger.info("\n5. RECOMMENDATIONS:")
    logger.info("-" * 70)

    if asd_count > 0:
        ratio = control_count / asd_count

        if ratio > 3.0:
            logger.error("SEVERE imbalance (ratio > 3:1)")
            logger.error("   -> MUST use: Focal Loss + Threshold Optimization")
            logger.error("   -> CONSIDER: SMOTE oversampling or undersampling")
        elif ratio > 2.0:
            logger.warning("MODERATE imbalance (ratio 2-3:1)")
            logger.warning("   -> USE: Focal Loss + Threshold Optimization")
            logger.warning("   -> Class weights may help")
        elif ratio > 1.5:
            logger.info("MILD imbalance (ratio 1.5-2:1)")
            logger.info("   -> USE: Threshold Optimization")
            logger.info("   -> Focal Loss or class weights optional")
        else:
            logger.info("BALANCED dataset (ratio < 1.5:1)")
            logger.info("   -> Standard training should work")

    logger.info("=" * 70)


def generate_health_report(
    pheno_path: Optional[Path] = None,
    sample_png: int = 20,
    sample_ts: int = 10,
    run_deep_checks: bool = False,
) -> bool:
    """
    Generate comprehensive dataset health report.

    Args:
        pheno_path: Path to phenotype CSV (default: from config)
        sample_png: Number of PNG files to sample for deep validation
        sample_ts: Number of time series files to sample for deep validation
        run_deep_checks: If True, performs intensive file validation (slower)
    """
    pheno_path = pheno_path or (DATA_PROCESSED / "Phenotypic_V1_0b_preprocessed1.csv")

    # Load metadata
    if not pheno_path.exists():
        logger.error(f"Error: {pheno_path} not found.")
        return False

    df = pd.read_csv(pheno_path)
    df["FILE_ID"] = df["FILE_ID"].astype(str).str.strip()
    logger.info(f"Loaded metadata: {len(df)} records")

    # Collect PNGs from split directories (data/final/{train,val,test}/images/)
    # Fall back to legacy data/images/ if splits don't exist yet.
    split_image_dirs = [DATA_FINAL / split / "images" for split in ("train", "val", "test")]
    split_images_exist = any(d.exists() for d in split_image_dirs)

    if split_images_exist:
        downloaded_files = []
        for d in split_image_dirs:
            if d.exists():
                downloaded_files.extend([f.name for f in d.glob("*.png")])
        png_dir = DATA_FINAL  # used only for deep-check path resolution below
    else:
        # Legacy fallback
        png_dir = DATA_ROOT / "images"
        if not png_dir.exists():
            logger.error(f"Error: Image folder {png_dir} not found.")
            return False
        downloaded_files = [f for f in os.listdir(png_dir) if f.endswith(".png")]

    completed_subs = set([f.rsplit("_z", 1)[0] for f in downloaded_files])
    logger.info(f"Found {len(downloaded_files)} PNG slices from {len(completed_subs)} subjects")

    # Match metadata to images
    current_df = df[df["FILE_ID"].isin(completed_subs)].copy()

    if current_df.empty:
        logger.warning("[!] No matching metadata found for downloaded images.")
        return False

    logger.info(f"Matched {len(current_df)} subjects to metadata")

    # Generate report sections
    logger.info("\n" + "=" * 40)
    logger.info(f"{'DATASET HEALTH REPORT':^40}")
    logger.info("=" * 40)
    logger.info(f"Unique Subjects:   {len(completed_subs)}")
    logger.info(f"Total PNG Slices:  {len(downloaded_files)}")

    if len(completed_subs) > 0:
        avg_slices = len(downloaded_files) / len(completed_subs)
        logger.info(f"Avg Slices/Sub:    {avg_slices:.1f} (Target: {TARGET_SLICES})")

    # Class balance
    logger.info("-" * 40)
    logger.info("CLASS BALANCE")
    stats = current_df["DX_GROUP"].value_counts().to_dict()
    asd = stats.get(1, 0)
    tc = stats.get(2, 0)
    logger.info(f"  Autism (ASD):     {asd}")
    logger.info(f"  Controls (TC):    {tc}")
    if tc > 0:
        ratio = asd / tc
        logger.info(f"  Ratio (ASD/TC):   {ratio:.2f}")

    # Demographics
    logger.info("-" * 40)
    logger.info("DEMOGRAPHICS")
    if "AGE_AT_SCAN" in current_df.columns:
        valid_age = current_df[current_df["AGE_AT_SCAN"] > 0]["AGE_AT_SCAN"]
        if not valid_age.empty:
            logger.info(f"  Avg Age:          {valid_age.mean():.1f} years")

    if "SEX" in current_df.columns:
        sex_stats = current_df["SEX"].value_counts().to_dict()
        males = sex_stats.get(1, 0)
        females = sex_stats.get(2, 0)
        logger.info(f"  Sex Ratio (M/F):  {males}/{females}")

    # Top sites
    logger.info("-" * 40)
    logger.info("TOP SITES")
    if "SITE_ID" in current_df.columns:
        site_stats = current_df["SITE_ID"].value_counts().head(5)
        for site, count in site_stats.items():
            logger.info(f"  {str(site):<15}: {count} subjects")

    # Data completeness
    logger.info("-" * 40)
    logger.info("DATA COMPLETENESS")
    # Filter out malformed rows (empty/null FILE_ID values from the phenotype CSV).
    _invalid = {"", "no_filename", "nan", "none"}
    excluded_upper = {s.upper() for s in EXCLUDED_SUBJECTS}
    metadata_ids = {
        fid for fid in df["FILE_ID"].unique()
        if (
            pd.notna(fid)
            and str(fid).strip().lower() not in _invalid
            and str(fid).strip().upper() not in excluded_upper
        )
    }
    missing_metadata = completed_subs - metadata_ids

    if missing_metadata:
        logger.warning(f"  WARNING: Subjects with images but no metadata: {len(missing_metadata)}")
    else:
        logger.info("  All downloaded subjects have metadata")

    missing_images = metadata_ids - completed_subs
    if missing_images:
        logger.warning(f"  WARNING: Subjects with metadata but no images: {len(missing_images)}")
    else:
        logger.info("  All metadata subjects have images")

    # Slice distribution
    logger.info("-" * 40)
    logger.info("SLICE DISTRIBUTION")
    slice_counts = {}
    for filename in downloaded_files:
        subject_id = filename.rsplit("_z", 1)[0]
        slice_counts[subject_id] = slice_counts.get(subject_id, 0) + 1

    incomplete = {subid: count for subid, count in slice_counts.items() if count != TARGET_SLICES}
    complete = sum(1 for count in slice_counts.values() if count == TARGET_SLICES)
    logger.info(f"  Complete ({TARGET_SLICES} slices): {complete}/{len(slice_counts)}")

    if incomplete:
        logger.warning(f"  WARNING: Subjects with incomplete slices: {len(incomplete)}")
    else:
        logger.info(f"  All subjects have complete slice sets ({TARGET_SLICES}/{TARGET_SLICES})")

    # Time series files — search across split directories
    logger.info("-" * 40)
    logger.info("TIME SERIES FILES")
    split_ts_dirs = [DATA_FINAL / split / "time_series" for split in ("train", "val", "test")]
    all_ts_files: List[Path] = []
    for _td in split_ts_dirs:
        if _td.exists():
            all_ts_files.extend(_td.glob("*_ts.npy"))
    # Fallback to legacy DATA_PROCESSED root
    if not all_ts_files:
        all_ts_files = list(DATA_PROCESSED.glob("*_ts.npy"))
    ts_dir = DATA_FINAL  # for deep-check reference below

    logger.info(f"  Time series files:  {len(all_ts_files)}")
    ts_subjects = set([f.stem.replace("_ts", "") for f in all_ts_files])
    missing_ts = completed_subs - ts_subjects

    if missing_ts:
        logger.warning(f"  WARNING: Downloaded subjects missing time series: {len(missing_ts)}")
    else:
        logger.info("  All downloaded subjects have time series")

    # Feature files
    logger.info("-" * 40)
    logger.info("FEATURE EXTRACTION STATUS")
    feature_files = {
        "Spatial Features": DATA_METADATA / "node_features_3d.csv",
        "Temporal Features": DATA_METADATA / "node_attributes_temporal.csv",
        "Harmonized Features": DATA_METADATA / "node_attributes_harmonized.csv",
    }

    for feature_name, feature_path in feature_files.items():
        if feature_path.exists():
            try:
                feat_df = pd.read_csv(feature_path)
                logger.info(f"  {feature_name:<25}: {len(feat_df)} subjects")
            except Exception as e:
                logger.warning(f"  {feature_name:<25}: ERROR - {str(e)[:50]}")
        else:
            logger.warning(f"  {feature_name:<25}: NOT FOUND")

    # Graph files
    logger.info("-" * 40)
    logger.info("GRAPH CONSTRUCTION STATUS")
    graph_dir = DATA_PROCESSED / "causal_graphs"
    if graph_dir.exists():
        graph_files = list(graph_dir.glob("*_graph.pt"))
        logger.info(f"  Graph files:        {len(graph_files)}")
        if len(graph_files) > 0:
            logger.info("  Status:             Graphs constructed")
        else:
            logger.warning("  Status:             No graphs found")
    else:
        logger.warning(f"  WARNING: Graph directory not found: {graph_dir}")

    # Deep integrity checks (optional, slower)
    if run_deep_checks:
        logger.info("\n" + "=" * 40)
        logger.info("DEEP INTEGRITY CHECKS")
        logger.info("=" * 40)

        # PNG validation — build a name→Path lookup across all split dirs
        _png_lookup: dict = {}
        for _d in (split_image_dirs if split_images_exist else [DATA_ROOT / "images"]):
            if _d.exists():
                for _p in _d.glob("*.png"):
                    _png_lookup[_p.name] = _p

        logger.info("\nPNG FILE INTEGRITY (sampling)")
        corrupted_pngs = []
        wrong_size = []
        sample_files = random.sample(downloaded_files, min(sample_png, len(downloaded_files)))

        for png_file in sample_files:
            png_path = _png_lookup.get(png_file)
            if png_path is None:
                corrupted_pngs.append((png_file, "file not found in lookup"))
                continue
            try:
                img = Image.open(png_path)
                if img.size != (640, 640):
                    wrong_size.append((png_file, img.size))
            except Exception as e:
                corrupted_pngs.append((png_file, str(e)[:40]))

        if not corrupted_pngs and not wrong_size:
            logger.info(f"  PNG files valid (sampled {len(sample_files)} files)")
        else:
            if corrupted_pngs:
                logger.warning(f"  Corrupted PNG files: {len(corrupted_pngs)}")
            if wrong_size:
                logger.warning(f"  Wrong dimensions: {len(wrong_size)}")

        # Time series validation
        if all_ts_files:
            logger.info("\nTIME SERIES VALIDATION (sampling)")
            invalid_ts = []
            wrong_shape = []
            sample_ts_files = random.sample(all_ts_files, min(sample_ts, len(all_ts_files)))

            for ts_file in sample_ts_files:
                try:
                    data = np.load(ts_file)
                    if data.ndim != 2 or data.shape[1] != 170:
                        wrong_shape.append((ts_file.name, data.shape))
                    if np.isnan(data).any() or np.isinf(data).any():
                        invalid_ts.append((ts_file.name, "contains NaN/Inf"))
                except Exception as e:
                    invalid_ts.append((ts_file.name, str(e)[:40]))

            if not invalid_ts and not wrong_shape:
                logger.info(f"  Time series files valid (sampled {len(sample_ts_files)} files)")
            else:
                if invalid_ts:
                    logger.warning(f"  Invalid time series files: {len(invalid_ts)}")
                if wrong_shape:
                    logger.warning(f"  Wrong shape: {len(wrong_shape)}")

    logger.info("\n" + "=" * 40)
    logger.info("Health report complete.")
    return True


def _sample_graphs(graph_files: List[Path], sample_size: int = 200) -> Dict:
    """
    Sample up to *sample_size* graph .pt files and return validity statistics.

    Shared helper used by PipelineValidator.validate_graphs() to eliminate
    duplicated sampling code across classes.

    Returns a dict with keys::

        total, valid, corrupted, zero_edges, wrong_shape, edge_counts,
        mean_edges (present when edge_counts is non-empty),
        median_edges (present when edge_counts is non-empty).
    """
    stats: Dict = {
        "total": len(graph_files),
        "valid": 0,
        "corrupted": 0,
        "zero_edges": 0,
        "degenerate": 0,
        "dead_lobes": [],
        "wrong_shape": 0,
        "edge_counts": [],
        "edge_weights": [],
        "sparsification_fallback_triggered": 0,
        "min_edge_fallback": 0,
        "dead_lobe_repair": 0,
        "missing_sparsification_metadata": 0,
    }
    sample = list(np.random.choice(graph_files, min(sample_size, len(graph_files)), replace=False))
    for gf in sample:
        try:
            data = torch.load(gf, weights_only=False)
            if "adj" not in data:
                stats["corrupted"] += 1
                continue
            adj = data["adj"]
            if adj.shape != (NUM_LOBES, NUM_LOBES):
                stats["wrong_shape"] += 1
                continue
            if torch.isnan(adj).any() or torch.isinf(adj).any():
                stats["corrupted"] += 1
                continue

            # Collect edge-weight statistics for graph usefulness checks.
            offdiag = ~torch.eye(NUM_LOBES, dtype=torch.bool, device=adj.device)
            nz = (adj != 0) & offdiag
            if int(nz.sum().item()) > 0:
                stats["edge_weights"].extend(adj[nz].detach().cpu().numpy().tolist())

            graph_stats = data.get("stats", {}) if isinstance(data, dict) else {}
            if isinstance(graph_stats, dict) and "sparsification_fallback_triggered" in graph_stats:
                if bool(graph_stats.get("sparsification_fallback_triggered", False)):
                    stats["sparsification_fallback_triggered"] += 1
                if bool(graph_stats.get("min_edge_fallback", False)):
                    stats["min_edge_fallback"] += 1
                if bool(graph_stats.get("dead_lobe_repair", False)):
                    stats["dead_lobe_repair"] += 1
            else:
                stats["missing_sparsification_metadata"] += 1

            deg = summarize_graph_degeneracy_from_adj(adj, min_edges=MIN_EDGES_PER_GRAPH)
            n_edges = int(deg["edge_count"])
            stats["edge_counts"].append(n_edges)
            stats["dead_lobes"].append(int(deg["dead_lobes"]))
            if n_edges == 0:
                stats["zero_edges"] += 1
            if bool(deg["is_degenerate"]):
                stats["degenerate"] += 1
            else:
                stats["valid"] += 1
        except Exception:
            stats["corrupted"] += 1
    if stats["edge_counts"]:
        arr = np.array(stats["edge_counts"])
        stats["mean_edges"] = float(arr.mean())
        stats["median_edges"] = float(np.median(arr))
        stats["mean_dead_lobes"] = float(np.mean(stats["dead_lobes"])) if stats["dead_lobes"] else 0.0
    if stats["edge_weights"]:
        w = np.array(stats["edge_weights"], dtype=float)
        stats["edge_weight_mean"] = float(w.mean())
        stats["edge_weight_std"] = float(w.std())
        stats["edge_weight_abs_mean"] = float(np.abs(w).mean())
        stats["edge_weight_abs_std"] = float(np.abs(w).std())
    return stats


@dataclass
class ValidationResult:
    """Structured validation result."""

    stage: str
    passed: bool
    message: str
    severity: str  # 'critical', 'warning', 'info'
    fix_suggestion: Optional[str] = None
    metrics: Optional[Dict] = None


class PipelineValidator:
    """
    Unified pipeline validation system.

    Validates data integrity, feature quality, and model outputs
    at each stage of the pipeline.
    """

    def __init__(self, visualize: bool = False):
        self.results: List[ValidationResult] = []
        self.metrics: Dict = {}
        self.visualize = visualize
        self.output_dir = Path("./results/validation_outputs")
        if visualize:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def add_result(self, result: ValidationResult) -> None:
        self.results.append(result)

        if result.severity == "critical":
            logger.error(f"[{result.stage}] {result.message}")
            if result.fix_suggestion:
                logger.error(f"  Fix: {result.fix_suggestion}")
        elif result.severity == "warning":
            logger.warning(f"[{result.stage}] {result.message}")
        else:
            logger.info(f"[{result.stage}] {result.message}")

    # STAGE 1: ENVIRONMENT VALIDATION

    def validate_environment(self) -> bool:
        logger.info("=" * 70)
        logger.info("STAGE 1: ENVIRONMENT VALIDATION")
        logger.info("=" * 70)

        all_passed = True

        if sys.version_info < (3, 8):
            self.add_result(
                ValidationResult(
                    stage="Environment",
                    passed=False,
                    message=f"Python {sys.version_info.major}.{sys.version_info.minor} detected",
                    severity="critical",
                    fix_suggestion="Upgrade to Python 3.8+",
                )
            )
            all_passed = False
        else:
            self.add_result(
                ValidationResult(
                    stage="Environment",
                    passed=True,
                    message=f"Python {sys.version_info.major}.{sys.version_info.minor}",
                    severity="info",
                )
            )

        critical_dirs = {
            "DATA_ROOT": DATA_ROOT,
            "DATA_PROCESSED": DATA_PROCESSED,
            "DATA_FINAL": DATA_FINAL,
        }

        for name, path in critical_dirs.items():
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                self.add_result(
                    ValidationResult(
                        stage="Environment",
                        passed=True,
                        message=f"Created {name}: {path}",
                        severity="info",
                    )
                )

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            self.add_result(
                ValidationResult(
                    stage="Environment",
                    passed=True,
                    message=f"CUDA available: {device_name}",
                    severity="info",
                )
            )
        else:
            self.add_result(
                ValidationResult(
                    stage="Environment",
                    passed=True,
                    message="CPU-only mode (training will be slow)",
                    severity="warning",
                )
            )

        if len(LOBE_MAPPING) != NUM_LOBES:
            self.add_result(
                ValidationResult(
                    stage="Environment",
                    passed=False,
                    message=f"Config mismatch: LOBE_MAPPING has {len(LOBE_MAPPING)} lobes but NUM_LOBES={NUM_LOBES}",
                    severity="critical",
                    fix_suggestion="Fix LOBE_MAPPING in src/core/config.py",
                )
            )
            all_passed = False

        # GNN_IN_CHANNELS is derived as len(ALL_FEATURE_NAMES) in config.py
        # (temporal=8, frequency=12, internal=2, spatial=6 = 28 total)
        expected_features = len(ALL_FEATURE_NAMES)
        if GNN_IN_CHANNELS != expected_features:
            self.add_result(
                ValidationResult(
                    stage="Environment",
                    passed=False,
                    message=f"Config mismatch: GNN_IN_CHANNELS={GNN_IN_CHANNELS} but len(ALL_FEATURE_NAMES)={expected_features}",
                    severity="critical",
                    fix_suggestion="Ensure GNN_IN_CHANNELS = len(ALL_FEATURE_NAMES) in config.py",
                )
            )
            all_passed = False
        else:
            self.add_result(
                ValidationResult(
                    stage="Environment",
                    passed=True,
                    message=f"Feature dims: GNN_IN_CHANNELS={GNN_IN_CHANNELS} ({NUM_TEMPORAL_FEATURES} temporal + {NUM_SPATIAL_FEATURES} spatial + internal)",
                    severity="info",
                )
            )

        return all_passed

    # STAGE 2: DATA VALIDATION

    def validate_downloaded_data(self) -> bool:
        logger.info("\n" + "=" * 70)
        logger.info("STAGE 2: DATA VALIDATION")
        logger.info("=" * 70)

        all_passed = True

        # Images may live in split dirs (post-split) or the legacy pre-split dir
        split_image_dirs = [DATA_FINAL / s / "images" for s in ("train", "val", "test")]
        png_files = [p for d in split_image_dirs if d.exists() for p in d.glob("*.png")]
        if not png_files:
            legacy_dir = DATA_ROOT / "images"
            png_files = list(legacy_dir.glob("*.png")) if legacy_dir.exists() else []

        if not png_files:
            self.add_result(
                ValidationResult(
                    stage="Data",
                    passed=False,
                    message="No PNG images found in split dirs or legacy data/images/",
                    severity="critical",
                    fix_suggestion="Run: python -m src.data.abide_download",
                )
            )
            return False

        subjects = set(f.stem.rsplit("_z", 1)[0] for f in png_files)

        # Time series may live in split dirs or legacy DATA_PROCESSED root
        split_ts_dirs = [DATA_FINAL / s / "time_series" for s in ("train", "val", "test")]
        ts_files = [p for d in split_ts_dirs if d.exists() for p in d.glob("*_ts.npy")]
        if not ts_files:
            ts_files = list(DATA_PROCESSED.glob("*_ts.npy"))
        ts_subjects = set(f.stem.replace("_ts", "") for f in ts_files)

        missing_ts = subjects - ts_subjects
        missing_img = ts_subjects - subjects

        if missing_ts:
            self.add_result(
                ValidationResult(
                    stage="Data",
                    passed=False,
                    message=f"{len(missing_ts)} subjects have images but no time series",
                    severity="warning",
                    metrics={"missing_subjects": list(missing_ts)[:5]},
                )
            )

        if missing_img:
            self.add_result(
                ValidationResult(
                    stage="Data",
                    passed=False,
                    message=f"{len(missing_img)} subjects have time series but no images",
                    severity="warning",
                )
            )

        complete_subjects = subjects & ts_subjects
        self.add_result(
            ValidationResult(
                stage="Data",
                passed=True,
                message=f"{len(complete_subjects)} subjects with complete data",
                severity="info",
                metrics={"total_images": len(png_files), "total_subjects": len(complete_subjects)},
            )
        )

        # Exclude known-bad subjects so they don't pollute the sample check.
        excluded_upper = {s.upper() for s in EXCLUDED_SUBJECTS}
        valid_ts_files = [
            f for f in ts_files
            if f.stem.replace("_ts", "").upper() not in excluded_upper
        ] or ts_files  # fall back to full list if all were excluded
        sample_size = min(10, len(valid_ts_files))
        corrupted = 0
        wrong_shape = 0
        total_nan_ratio = 0.0

        for ts_file in np.random.choice(valid_ts_files, sample_size, replace=False):
            try:
                data = np.load(ts_file)
                if data.ndim != 2:
                    wrong_shape += 1
                else:
                    # Check NaN/Inf ratio - sparse NaN is acceptable in fMRI data
                    nan_inf_count = np.isnan(data).sum() + np.isinf(data).sum()
                    nan_ratio = nan_inf_count / data.size
                    total_nan_ratio += nan_ratio
                    
                    # Only flag if > 10% of data is NaN/Inf (severe corruption)
                    if nan_ratio > 0.10:
                        corrupted += 1
            except Exception:
                corrupted += 1

        # Calculate average NaN ratio across sample
        avg_nan_ratio = total_nan_ratio / sample_size if sample_size > 0 else 0
        
        if corrupted > 0 or wrong_shape > 0:
            # Treat as warning if only sparse NaN (< 10% corruption rate)
            severity = "warning" if corrupted < sample_size * 0.5 else "critical"
            self.add_result(
                ValidationResult(
                    stage="Data",
                    passed=False,
                    message=f"Sample validation: {corrupted} corrupted, {wrong_shape} wrong shape (avg NaN: {100*avg_nan_ratio:.1f}%)",
                    severity=severity,
                    fix_suggestion="Re-run data download if corruption rate > 50%" if severity == "critical" else "Sparse NaN is acceptable in fMRI data",
                )
            )
            if severity == "critical":
                all_passed = False

        return all_passed

    # STAGE 3: FEATURE VALIDATION

    def validate_features(self) -> bool:
        logger.info("\n" + "=" * 70)
        logger.info("STAGE 3: FEATURE VALIDATION")
        logger.info("=" * 70)

        all_passed = True

        if not NODE_ATTRIBUTES_TEMPORAL.exists():
            self.add_result(
                ValidationResult(
                    stage="Features",
                    passed=False,
                    message="Temporal features not found",
                    # Not critical: file will be generated in a subsequent pipeline stage.
                    severity="warning",
                    fix_suggestion="Run: python -m src.features.extract_temporal",
                )
            )
            return False

        try:
            temporal_df = pd.read_csv(NODE_ATTRIBUTES_TEMPORAL)
            feature_cols = [c for c in temporal_df.columns if c != "subject_id"]

            nan_count = temporal_df[feature_cols].isna().sum().sum()
            total_values = len(temporal_df) * len(feature_cols)
            nan_pct = (nan_count / total_values) * 100

            if nan_pct > 5:
                self.add_result(
                    ValidationResult(
                        stage="Features",
                        passed=False,
                        message=f"High NaN rate: {nan_pct:.1f}%",
                        severity="critical" if nan_pct > 20 else "warning",
                        fix_suggestion="Check atlas alignment and time series quality",
                    )
                )
                if nan_pct > 20:
                    all_passed = False
            else:
                self.add_result(
                    ValidationResult(
                        stage="Features",
                        passed=True,
                        message=f"Temporal features: {len(temporal_df)} subjects, {nan_pct:.2f}% NaN",
                        severity="info",
                    )
                )

            numeric_data = temporal_df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
            if len(numeric_data) > 0:
                extreme_values = (np.abs(numeric_data.values) > 1e6).sum()
                if extreme_values > 0:
                    self.add_result(
                        ValidationResult(
                            stage="Features",
                            passed=False,
                            message=f"{extreme_values} extreme values detected (|x| > 1e6)",
                            severity="warning",
                            fix_suggestion="Check feature extraction for numerical issues",
                        )
                    )

        except Exception as e:
            self.add_result(
                ValidationResult(
                    stage="Features",
                    passed=False,
                    message=f"Error loading temporal features: {e}",
                    severity="critical",
                    fix_suggestion="Regenerate temporal features",
                )
            )
            return False

        if not NODE_FEATURES_3D.exists():
            self.add_result(
                ValidationResult(
                    stage="Features",
                    passed=False,
                    message="Spatial features not found",
                    # Not critical: file will be generated in a subsequent pipeline stage.
                    severity="warning",
                    fix_suggestion="Run: python -m src.features.extract_spatial",
                )
            )
            return False

        try:
            spatial_df = pd.read_csv(NODE_FEATURES_3D)

            brainstem_cols = [
                c for c in spatial_df.columns
                if c.startswith("Brainstem_") and c not in {"Brainstem_conf_std", "Brainstem_detection_count"}
            ]
            if brainstem_cols:
                const_brainstem = all(
                    pd.to_numeric(spatial_df[col], errors="coerce").fillna(0.0).nunique(dropna=False) <= 1
                    for col in brainstem_cols
                )
                if const_brainstem:
                    self.add_result(
                        ValidationResult(
                            stage="Features",
                            passed=False,
                            message=(
                                "Brainstem spatial features are constant across all subjects "
                                "(global detection fallback active)"
                            ),
                            severity="warning",
                            fix_suggestion="Audit YOLO class-11 detections and atlas fallback behavior before publication runs",
                        )
                    )

            lobe_cols = [
                c
                for c in spatial_df.columns
                if any(c.startswith(f"{lobe}_") for lobe in ["Frontal", "Temporal", "Parietal", "Occipital", "Limbic"])
            ]

            complete_detections = 0
            for _, row in spatial_df.iterrows():
                has_all = all(pd.notna(row[col]) for col in lobe_cols[:5])
                if has_all:
                    complete_detections += 1

            survival_rate = (complete_detections / len(spatial_df)) * 100 if len(spatial_df) > 0 else 0

            if survival_rate < 80:
                self.add_result(
                    ValidationResult(
                        stage="Features",
                        passed=False,
                        message=f"Low YOLO survival rate: {survival_rate:.1f}%",
                        severity="warning",
                        fix_suggestion="Check YOLO model quality or detection threshold",
                    )
                )
            else:
                self.add_result(
                    ValidationResult(
                        stage="Features",
                        passed=True,
                        message=f"Spatial features: {complete_detections}/{len(spatial_df)} complete ({survival_rate:.1f}%)",
                        severity="info",
                    )
                )

        except Exception as e:
            self.add_result(
                ValidationResult(
                    stage="Features",
                    passed=False,
                    message=f"Error loading spatial features: {e}",
                    severity="critical",
                )
            )
            all_passed = False

        if NODE_ATTRIBUTES_HARMONIZED.exists():
            try:
                harm_df = pd.read_csv(NODE_ATTRIBUTES_HARMONIZED)
                harm_cols = [c for c in harm_df.columns if c != "subject_id"]

                harm_nan = harm_df[harm_cols].isna().sum().sum()
                harm_inf = np.isinf(harm_df[harm_cols].values).sum()

                if harm_nan > 0 or harm_inf > 0:
                    self.add_result(
                        ValidationResult(
                            stage="Features",
                            passed=False,
                            message=f"Harmonized features have {harm_nan} NaN, {harm_inf} Inf",
                            severity="critical",
                            fix_suggestion="Re-run harmonization with fold_safe_harmonization.py",
                        )
                    )
                    all_passed = False
                else:
                    self.add_result(
                        ValidationResult(
                            stage="Features",
                            passed=True,
                            message=f"Harmonized features: {len(harm_df)} subjects, clean",
                            severity="info",
                        )
                    )
            except Exception as e:
                self.add_result(
                    ValidationResult(
                        stage="Features",
                        passed=False,
                        message=f"Error loading harmonized features: {e}",
                        severity="warning",
                    )
                )

        return all_passed

    # STAGE 4: GRAPH VALIDATION

    def validate_graphs(self) -> bool:
        logger.info("\n" + "=" * 70)
        logger.info("STAGE 4: GRAPH VALIDATION")
        logger.info("=" * 70)

        if not CAUSAL_GRAPHS_DIR.exists() or not list(CAUSAL_GRAPHS_DIR.glob("*.pt")):
            self.add_result(
                ValidationResult(
                    stage="Graphs",
                    passed=False,
                    message="No graph files found",
                    # Not critical: graphs are built in a subsequent pipeline stage.
                    severity="warning",
                    fix_suggestion="Run: python -m src.features.construct_causal",
                )
            )
            return False

        graph_files = list(CAUSAL_GRAPHS_DIR.glob("*_graph.pt"))
        # Full scan avoids sample blind spots for sparse/degenerate tails.
        sample_size = len(graph_files) if len(graph_files) <= 5000 else 5000
        stats = _sample_graphs(graph_files, sample_size=sample_size)
        all_passed = True

        if stats["corrupted"] > sample_size * 0.05:
            self.add_result(ValidationResult(
                stage="Graphs",
                passed=False,
                message=f"{stats['corrupted']}/{sample_size} graphs corrupted or missing 'adj' key",
                severity="critical",
                fix_suggestion="Re-run graph construction",
            ))
            all_passed = False

        if stats.get("wrong_shape", 0) > 0:
            self.add_result(ValidationResult(
                stage="Graphs",
                passed=False,
                message=f"{stats['wrong_shape']} graphs have wrong shape (expected {NUM_LOBES}×{NUM_LOBES})",
                severity="critical",
                fix_suggestion="Clear graph directory and rebuild",
            ))
            all_passed = False

        if stats["zero_edges"] > sample_size * 0.1:
            self.add_result(ValidationResult(
                stage="Graphs",
                passed=False,
                message=f"{stats['zero_edges']}/{sample_size} graphs have zero edges",
                severity="warning",
                fix_suggestion=f"Lower SPARSITY_QUANTILE from {SPARSITY_QUANTILE}",
            ))

        if stats.get("degenerate", 0) > sample_size * GNN_MAX_DEGENERATE_GRAPH_RATE:
            self.add_result(ValidationResult(
                stage="Graphs",
                passed=False,
                message=(
                    f"{stats['degenerate']}/{sample_size} graphs are degenerate "
                    f"(edge_count < {MIN_EDGES_PER_GRAPH} or dead lobe present)"
                ),
                severity="critical",
                fix_suggestion="Inspect atlas coverage gaps and sparsification thresholds",
            ))
            all_passed = False

        if stats["edge_counts"]:
            mean_e = stats.get("mean_edges", 0.0)
            median_e = stats.get("median_edges", 0.0)
            self.add_result(ValidationResult(
                stage="Graphs",
                passed=True,
                message=f"{len(graph_files)} graphs — mean edges: {mean_e:.1f}, median: {median_e:.0f}",
                severity="info",
                metrics={
                    "total_graphs": len(graph_files),
                    "graphs_checked": sample_size,
                    "mean_edges": mean_e,
                    "median_edges": median_e,
                    "mean_dead_lobes": stats.get("mean_dead_lobes", 0.0),
                    "max_edges": max(stats["edge_counts"]),
                    "min_edges": min(stats["edge_counts"]),
                    "degenerate_sample": stats.get("degenerate", 0),
                    "degenerate_rate": stats.get("degenerate", 0) / max(sample_size, 1),
                    "edge_weight_mean": stats.get("edge_weight_mean"),
                    "edge_weight_std": stats.get("edge_weight_std"),
                    "edge_weight_abs_mean": stats.get("edge_weight_abs_mean"),
                    "edge_weight_abs_std": stats.get("edge_weight_abs_std"),
                    "sparsification_fallback_rate": stats.get("sparsification_fallback_triggered", 0) / max(sample_size, 1),
                    "min_edge_fallback_rate": stats.get("min_edge_fallback", 0) / max(sample_size, 1),
                    "dead_lobe_repair_rate": stats.get("dead_lobe_repair", 0) / max(sample_size, 1),
                    "missing_sparsification_metadata": stats.get("missing_sparsification_metadata", 0),
                    "edge_counts_sample": stats["edge_counts"],
                },
            ))

        if stats.get("missing_sparsification_metadata", 0) > 0:
            self.add_result(ValidationResult(
                stage="Graphs",
                passed=False,
                message=(
                    f"{stats['missing_sparsification_metadata']}/{sample_size} sampled graphs "
                    "lack sparsification metadata"
                ),
                severity="warning",
                fix_suggestion="Rebuild causal graphs with updated construct_causal.py to expose fallback telemetry",
            ))

        fallback_rate = stats.get("sparsification_fallback_triggered", 0) / max(sample_size, 1)
        if fallback_rate > 0.5:
            self.add_result(ValidationResult(
                stage="Graphs",
                passed=False,
                message=(
                    f"High sparsification fallback trigger rate: {fallback_rate:.1%} "
                    "(graph quality currently relies heavily on fallback/repair)"
                ),
                severity="warning",
                fix_suggestion="Audit sparsification thresholding and lobe-isolation dynamics",
            ))

        return all_passed

    # STAGE 5: MODEL VALIDATION

    def validate_trained_models(self) -> bool:
        """Check that trained model checkpoints exist (no AUC reporting to avoid ambiguity)."""
        logger.info("\n" + "=" * 70)
        logger.info("STAGE 5: MODEL VALIDATION")
        logger.info("=" * 70)

        if not CHECKPOINT_DIR.exists():
            self.add_result(
                ValidationResult(
                    stage="Models",
                    passed=False,
                    message="Checkpoint directory not found",
                    severity="warning",
                    fix_suggestion="Train models first",
                )
            )
            return False

        fold_checkpoints = list(CHECKPOINT_DIR.glob("best_model_fold*.pt"))

        if len(fold_checkpoints) == 0:
            self.add_result(
                ValidationResult(
                    stage="Models",
                    passed=False,
                    message="No trained models found",
                    severity="warning",
                    fix_suggestion="Run: python -m src.models.gnn_model",
                )
            )
            return False

        if len(fold_checkpoints) < K_FOLDS:
            self.add_result(
                ValidationResult(
                    stage="Models",
                    passed=False,
                    message=f"Incomplete training: {len(fold_checkpoints)}/{K_FOLDS} folds",
                    severity="warning",
                )
            )

        msg = f"{len(fold_checkpoints)} trained models found (validation results reported during training)"
        self.add_result(
            ValidationResult(
                stage="Models",
                passed=True,
                message=msg,
                severity="info",
            )
        )
        return True

    # ATLAS & CONFIG CHECKS (consolidated from PipelineHealthCheck)

    def check_atlas(self) -> bool:
        """Verify the brain atlas file exists and is loadable by nibabel."""
        logger.info("Checking atlas...")
        if not ATLAS_PATH.exists():
            self.add_result(ValidationResult(
                stage="Atlas", passed=False,
                message=f"Atlas missing: {ATLAS_PATH}",
                severity="critical",
                fix_suggestion="Run: python -m src.validation.atlas_validator",
            ))
            return False
        try:
            import nibabel as nib
            data = nib.load(str(ATLAS_PATH)).get_fdata()
            num_rois = len(np.unique(data)) - 1
            valid_counts = {116, 117, 164, 166, 170}
            severity = "info" if num_rois in valid_counts else "warning"
            self.add_result(ValidationResult(
                stage="Atlas", passed=True,
                message=f"Atlas loaded: {num_rois} ROIs",
                severity=severity,
            ))
            return True
        except Exception as e:
            self.add_result(ValidationResult(
                stage="Atlas", passed=False,
                message=f"Atlas load error: {e}",
                severity="critical",
                fix_suggestion="Re-download: python -m src.validation.atlas_validator",
            ))
            return False

    def check_lobe_mapping(self) -> bool:
        """Validate LOBE_MAPPING completeness, uniqueness, and ROI range."""
        logger.info("Checking LOBE_MAPPING...")
        try:
            if len(LOBE_MAPPING) != NUM_LOBES:
                raise ValueError(f"Expected {NUM_LOBES} lobes, got {len(LOBE_MAPPING)}")
            seen: set = set()
            for lobe_id, roi_list in LOBE_MAPPING.items():
                for roi in roi_list:
                    if roi in seen:
                        raise ValueError(f"Duplicate ROI {roi} in lobe {lobe_id}")
                    seen.add(roi)
            missing = set(range(170)) - seen
            extra = seen - set(range(170))
            if missing or extra:
                self.add_result(ValidationResult(
                    stage="Config", passed=True,
                    message=f"LOBE_MAPPING gap: missing={len(missing)}, extra={len(extra)}",
                    severity="warning",
                ))
            self.add_result(ValidationResult(
                stage="Config", passed=True,
                message=f"LOBE_MAPPING valid: {NUM_LOBES} lobes, {len(seen)} ROIs",
                severity="info",
            ))
            return True
        except ValueError as e:
            self.add_result(ValidationResult(
                stage="Config", passed=False,
                message=f"LOBE_MAPPING invalid: {e}",
                severity="critical",
                fix_suggestion="Fix LOBE_MAPPING in src/core/config.py",
            ))
            return False

    def check_manifest(self) -> Tuple[bool, Optional[pd.DataFrame]]:
        """Check manifest exists and contains all required columns."""
        logger.info("Checking manifest...")
        if not MASTER_MANIFEST.exists():
            self.add_result(ValidationResult(
                stage="Manifest", passed=False,
                message=f"Manifest missing: {MASTER_MANIFEST}",
                severity="critical",
                fix_suggestion="Run: python -m src.data.manifestor",
            ))
            return False, None
        try:
            df = pd.read_csv(MASTER_MANIFEST)
            missing_cols = [c for c in ("subject_id", "split", "DX_GROUP", "SITE_ID") if c not in df.columns]
            if missing_cols:
                self.add_result(ValidationResult(
                    stage="Manifest", passed=False,
                    message=f"Missing columns: {missing_cols}",
                    severity="critical",
                    fix_suggestion="Regenerate: python -m src.data.manifestor",
                ))
                return False, None
            splits = set(df["split"].unique())
            if not {"train", "val", "test"}.issubset(splits):
                self.add_result(ValidationResult(
                    stage="Manifest", passed=True,
                    message=f"Incomplete splits: {splits}",
                    severity="warning",
                ))
            self.add_result(ValidationResult(
                stage="Manifest", passed=True,
                message=f"{len(df)} subjects across {len(splits)} splits",
                severity="info",
            ))
            return True, df
        except Exception as e:
            self.add_result(ValidationResult(
                stage="Manifest", passed=False,
                message=f"Error reading manifest: {e}",
                severity="critical",
                fix_suggestion="Check or regenerate manifest",
            ))
            return False, None

    def check_stratification(self, manifest: Optional[pd.DataFrame] = None) -> None:
        """Assert no subject appears in more than one split (data leakage)."""
        logger.info("Checking stratification...")
        if manifest is None:
            ok, manifest = self.check_manifest()
            if not ok or manifest is None:
                return
        leakage = int((manifest.groupby("subject_id")["split"].nunique() > 1).sum())
        if leakage > 0:
            self.add_result(ValidationResult(
                stage="Stratification", passed=False,
                message=f"DATA LEAKAGE: {leakage} subjects appear in multiple splits",
                severity="critical",
                fix_suggestion="Re-run split.py with subject-level stratification",
            ))
        else:
            self.add_result(ValidationResult(
                stage="Stratification", passed=True,
                message=f"No data leakage across {manifest['split'].nunique()} splits",
                severity="info",
            ))

    def check_cv_fold_balance(self, manifest: Optional[pd.DataFrame] = None) -> None:
        """Validate CV fold-size balance on training split and flag severe skew."""
        logger.info("Checking CV fold balance...")
        if manifest is None:
            ok, manifest = self.check_manifest()
            if not ok or manifest is None:
                return

        if "cv_fold" not in manifest.columns:
            self.add_result(
                ValidationResult(
                    stage="Stratification",
                    passed=False,
                    message="cv_fold column missing in manifest",
                    severity="warning",
                    fix_suggestion="Regenerate manifest/folds with split.py",
                )
            )
            return

        train_df = manifest[manifest["split"] == "train"].copy()
        if train_df.empty:
            self.add_result(
                ValidationResult(
                    stage="Stratification",
                    passed=False,
                    message="No training rows found for CV fold balance check",
                    severity="warning",
                )
            )
            return

        fold_series = pd.to_numeric(train_df["cv_fold"], errors="coerce")
        if fold_series.isna().any():
            self.add_result(
                ValidationResult(
                    stage="Stratification",
                    passed=False,
                    message="cv_fold contains non-numeric/missing values",
                    severity="warning",
                    fix_suggestion="Regenerate cv_fold assignments",
                )
            )
            return

        fold_counts = fold_series.astype(int).value_counts().sort_index()
        if fold_counts.empty:
            self.add_result(
                ValidationResult(
                    stage="Stratification",
                    passed=False,
                    message="No CV folds found in training split",
                    severity="warning",
                )
            )
            return

        min_fold = int(fold_counts.min())
        max_fold = int(fold_counts.max())
        ratio = float(max_fold / max(min_fold, 1))
        summary = ", ".join([f"{int(k)}:{int(v)}" for k, v in fold_counts.items()])

        if ratio > 2.0:
            self.add_result(
                ValidationResult(
                    stage="Stratification",
                    passed=False,
                    message=(
                        f"Severely unbalanced CV folds (max/min={ratio:.2f}; counts={summary})"
                    ),
                    severity="warning",
                    fix_suggestion="Use balanced StratifiedKFold folds for publication training",
                )
            )
        else:
            self.add_result(
                ValidationResult(
                    stage="Stratification",
                    passed=True,
                    message=f"CV fold balance OK (max/min={ratio:.2f}; counts={summary})",
                    severity="info",
                    metrics={
                        "fold_counts": {str(int(k)): int(v) for k, v in fold_counts.items()},
                        "max_min_ratio": ratio,
                    },
                )
            )

        unseen_audit = HARMONIZED_FOLDS_DIR / "fold_unseen_site_audit.csv"
        if unseen_audit.exists():
            try:
                audit_df = pd.read_csv(unseen_audit)
                if {"unseen_row_count", "val_row_count"}.issubset(audit_df.columns) and not audit_df.empty:
                    unseen_rows = pd.to_numeric(audit_df["unseen_row_count"], errors="coerce").fillna(0).astype(int)
                    val_rows = pd.to_numeric(audit_df["val_row_count"], errors="coerce").fillna(0).astype(int)
                    all_unseen = bool((val_rows > 0).all() and (unseen_rows == val_rows).all())
                    if all_unseen:
                        self.add_result(
                            ValidationResult(
                                stage="Stratification",
                                passed=False,
                                message=(
                                    "All validation rows are unseen sites across CV folds "
                                    "(site-stratified mismatch with fold-safe harmonization)"
                                ),
                                severity="warning",
                                fix_suggestion="Use standard StratifiedKFold CV (Option A) and regenerate fold harmonization",
                            )
                        )
            except Exception as exc:
                self.add_result(
                    ValidationResult(
                        stage="Stratification",
                        passed=False,
                        message=f"Failed to parse fold unseen-site audit: {exc}",
                        severity="warning",
                    )
                )

    # VISUALIZATION (OPTIONAL)

    def visualize_results(self) -> None:
        """Render graph-sparsity and stratification plots when visualize=True."""
        if not self.visualize:
            return
        try:
            graph_result = next(
                (r for r in self.results if r.metrics and "edge_counts_sample" in (r.metrics or {})),
                None,
            )
            if graph_result and graph_result.metrics:
                edge_counts = graph_result.metrics.get("edge_counts_sample", [])
                if edge_counts:
                    self._plot_graph_sparsity(edge_counts)
            if MASTER_MANIFEST.exists():
                try:
                    self._plot_stratification(pd.read_csv(MASTER_MANIFEST))
                except Exception:
                    pass
            logger.info(f"Visualizations saved to: {self.output_dir}")
        except ImportError:
            logger.warning("matplotlib not available — skipping visualizations")

    def _plot_graph_sparsity(self, edge_counts: List[int]) -> None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(edge_counts, bins=range(0, max(edge_counts) + 2), edgecolor="black", alpha=0.7)
        ax.axvline(np.mean(edge_counts), color="red", linestyle="--", label=f"Mean: {np.mean(edge_counts):.1f}")
        ax.set_title("Graph Edge Count Distribution")
        ax.set_xlabel("Number of Edges")
        ax.set_ylabel("Frequency")
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "graph_sparsity.png", dpi=150)
        plt.close()

    def _plot_stratification(self, manifest: pd.DataFrame) -> None:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax = axes[0]
        split_dx = manifest.groupby(["split", "DX_GROUP"]).size().unstack(fill_value=0)
        split_dx.columns = ["ASD" if c == 1 else "Control" for c in split_dx.columns]
        split_dx.plot(kind="bar", ax=ax, color=["#e74c3c", "#3498db"])
        ax.set_title("Distribution by Split")
        ax.set_ylabel("Number of Subjects")
        ax.legend(title="Diagnosis")
        ax = axes[1]
        split_ratios = [
            (manifest[manifest["split"] == s]["DX_GROUP"] == 1).sum()
            / max(len(manifest[manifest["split"] == s]), 1)
            for s in ["train", "val", "test"]
        ]
        ax.bar(["Train", "Val", "Test"], split_ratios)
        ax.axhline(0.5, color="black", linestyle="--", label="Perfect Balance")
        ax.set_title("ASD Ratio per Split")
        ax.set_ylabel("Proportion")
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / "stratification.png", dpi=150)
        plt.close()

    # REPORTING

    def generate_report(self) -> Tuple[bool, Dict]:
        logger.info("\n" + "=" * 70)
        logger.info("VALIDATION REPORT")
        logger.info("=" * 70)

        critical = [r for r in self.results if r.severity == "critical" and not r.passed]
        warnings = [r for r in self.results if r.severity == "warning"]
        passed = [r for r in self.results if r.passed and r.severity == "info"]

        if critical:
            logger.error(f"\nCRITICAL ISSUES ({len(critical)}):")
            logger.error("-" * 70)
            for result in critical:
                logger.error(f"  [{result.stage}] {result.message}")
                if result.fix_suggestion:
                    logger.error(f"    -> Fix: {result.fix_suggestion}")

        if warnings:
            logger.warning(f"\nWARNINGS ({len(warnings)}):")
            logger.warning("-" * 70)
            for result in warnings:
                logger.warning(f"  [{result.stage}] {result.message}")
                if result.fix_suggestion:
                    logger.warning(f"    -> Suggestion: {result.fix_suggestion}")

        if passed:
            logger.info(f"\nPASSED CHECKS ({len(passed)}):")
            logger.info("-" * 70)
            for result in passed:
                logger.info(f"  {result.message}")

        logger.info("\n" + "=" * 70)
        logger.info("SUMMARY:")
        logger.info(f"  Passed: {len(passed)}")
        logger.info(f"  Warnings: {len(warnings)}")
        logger.info(f"  Critical: {len(critical)}")
        logger.info("=" * 70)

        is_healthy = len(critical) == 0

        if is_healthy:
            if warnings:
                logger.info("\nPipeline functional with warnings")
            else:
                logger.info("\nPIPELINE FULLY HEALTHY")
        else:
            logger.error("\nPIPELINE HAS CRITICAL ISSUES")

        report = {
            "healthy": is_healthy,
            "passed": len(passed),
            "warnings": len(warnings),
            "critical": len(critical),
            "results": self.results,
            "metrics": self.metrics,
        }

        return is_healthy, report

    def run_full_validation(self) -> bool:
        logger.info("Starting comprehensive pipeline validation...")

        self.validate_environment()
        self.check_atlas()
        self.check_lobe_mapping()
        self.validate_downloaded_data()
        manifest_ok, manifest = self.check_manifest()
        if manifest_ok:
            self.validate_features()
            self.validate_graphs()
            self.check_stratification(manifest)
            self.check_cv_fold_balance(manifest)
        self.validate_trained_models()
        self.visualize_results()

        is_healthy, _ = self.generate_report()
        return is_healthy


def run_quality_validation(visualize: bool = False, strict: bool = False) -> bool:
    checker = PipelineValidator(visualize=visualize)
    is_healthy = checker.run_full_validation()
    if strict and not is_healthy:
        sys.exit(1)
    return is_healthy


def run_pipeline_validation(strict: bool = False) -> bool:
    validator = PipelineValidator()
    is_healthy = validator.run_full_validation()
    if strict and not is_healthy:
        sys.exit(1)
    return is_healthy


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified validation and integrity checks")
    parser.add_argument("--dataset", action="store_true", help="Post-download integrity check")
    parser.add_argument("--distribution", action="store_true", help="Pre-GNN dataset distribution check")
    parser.add_argument("--class-analysis", action="store_true", help="Class imbalance analysis")
    parser.add_argument("--health", action="store_true", help="Dataset health report")
    parser.add_argument("--deep", action="store_true", help="Deep checks for health report")
    parser.add_argument("--quality", action="store_true", help="Run quality validation suite")
    parser.add_argument("--visualize", action="store_true", help="Generate validation visualizations")
    parser.add_argument("--strict", action="store_true", help="Exit with error code if issues found")

    args = parser.parse_args()

    if args.dataset:
        check_dataset_integrity()
        return

    if args.distribution:
        check_distribution()
        return

    if args.class_analysis:
        analyze_class_distribution()
        return

    if args.health:
        generate_health_report(run_deep_checks=args.deep)
        return

    if args.quality:
        run_quality_validation(visualize=args.visualize, strict=args.strict)
        return

    run_pipeline_validation(strict=args.strict)


if __name__ == "__main__":
    main()
