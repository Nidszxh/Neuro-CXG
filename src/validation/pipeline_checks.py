"""
Unified validation and integrity checks for Neuro-CXG.

This module consolidates:
- integrity checks (post-download, pre-GNN, health reports)
- quality validation (YOLO, graphs, stratification)
- full pipeline validation (environment, data, features, graphs, models)
"""

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
    ATLAS_PATH,
    CAUSAL_GRAPHS_DIR,
    CHECKPOINT_DIR,
    DATA_FINAL,
    DATA_METADATA,
    DATA_PROCESSED,
    DATA_ROOT,
    GNN_IN_CHANNELS,
    K_FOLDS,
    LOBE_MAPPING,
    MASTER_MANIFEST,
    NODE_ATTRIBUTES_HARMONIZED,
    NODE_ATTRIBUTES_TEMPORAL,
    NODE_FEATURES_3D,
    NUM_LOBES,
    NUM_SPATIAL_FEATURES,
    NUM_TEMPORAL_FEATURES,
    SPARSITY_QUANTILE,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Integrity defaults
TARGET_SLICES = 5
VALID_ROI_RANGE = (164, 170)  # AAL3v1 atlas variants
PNG_DIR = DATA_ROOT / "images"
TS_DIR = DATA_PROCESSED


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
    png_files = list(PNG_DIR.glob("*.png"))
    corrupted_pngs = []
    subject_counts = Counter()

    if not png_files:
        logger.info("No images found to check.")
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
    npy_files = list(TS_DIR.glob("*.npy"))
    corrupted_npys = []
    ts_subjects = set()

    logger.info(f"Scanning {len(npy_files)} time-series files...")
    for path in npy_files:
        try:
            data = np.load(path)
            if np.isnan(data).any():
                raise ValueError("NaNs detected")
            num_rois = data.shape[1]
            if not (VALID_ROI_RANGE[0] <= num_rois <= VALID_ROI_RANGE[1]):
                raise ValueError(
                    f"ROI mismatch: Found {num_rois}, expected {VALID_ROI_RANGE[0]}-{VALID_ROI_RANGE[1]}"
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
        logger.info("OPTIONS: [1] Delete Corrupted | [2] Purge Incomplete Subjects | [3] Exit")
        choice = input("Select (1/2/3): ")
        if choice == "1":
            for path in corrupted_pngs + corrupted_npys:
                os.remove(path)
                logger.info(f"Deleted: {path}")
        elif choice == "2":
            for sub in incomplete_subs:
                # Remove PNGs
                for path in PNG_DIR.glob(f"{sub}_z*.png"):
                    os.remove(path)
                # Remove NPYs
                for path in TS_DIR.glob(f"{sub}_ts.npy"):
                    os.remove(path)
                for path in TS_DIR.glob(f"{sub}_qc.json"):
                    os.remove(path)
            logger.info("Purge complete.")


def check_distribution() -> None:
    """
    Pre-GNN integrity check.

    Validates dataset completeness across train/val/test splits:
    - Checks each subject has target number of slices
    - Verifies image/label file pairing for YOLO training
    - Reports distribution of slice counts per split
    """
    logger.info("Starting pre-GNN integrity check...")

    processed_root = str(DATA_PROCESSED)

    if not os.path.exists(processed_root):
        logger.error(f"Path {processed_root} does not exist. Run split.py first.")
        return

    splits = ["train", "val", "test"]

    logger.info("Dataset Completeness Report (Target: 5 slices/subject)")

    for split in splits:
        img_path = os.path.join(processed_root, split, "images")
        lbl_path = os.path.join(processed_root, split, "labels")

        if not os.path.exists(img_path):
            logger.warning(f"[!] Split '{split}' images folder missing.")
            continue

        files = [f for f in os.listdir(img_path) if f.endswith(".png")]
        subject_counts = {}

        for file_name in files:
            sub_id = file_name.rsplit("_z", 1)[0]
            subject_counts[sub_id] = subject_counts.get(sub_id, 0) + 1

        # Analyze the distribution of slice counts
        dist = Counter(subject_counts.values())

        logger.info(f"\nSplit: {split.upper()}")
        logger.info(f"  Total Subjects: {len(subject_counts)}")

        for num_slices in sorted(dist.keys()):
            status = "✓" if num_slices == TARGET_SLICES else "X"
            logger.info(f"  {status} {num_slices} slices: {dist[num_slices]} subjects")

        # Check for matching labels (Critical for YOLO training)
        if os.path.exists(lbl_path):
            labels = [f for f in os.listdir(lbl_path) if f.endswith(".txt")]
            if len(labels) != len(files):
                logger.warning(
                    f"  [!] ALERT: Image/Label Mismatch! ({len(files)} images vs {len(labels)} labels)"
                )
            else:
                logger.info(f"  ✓ Image/Label count matches ({len(files)} files)")

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
        site_stats = []
        for site in df["SITE_ID"].value_counts().head(10).index:
            site_df = df[df["SITE_ID"] == site]
            site_dx = site_df["DX_GROUP"].value_counts()
            site_control = site_dx.get(2, 0)
            site_asd = site_dx.get(1, 0)

            site_stats.append(
                {
                    "site": site,
                    "total": len(site_df),
                    "control": site_control,
                    "asd": site_asd,
                    "ratio": f"{site_control / site_asd:.2f}:1" if site_asd > 0 else "N/A",
                }
            )

        for stat in site_stats:
            logger.info(
                f"{stat['site']:20} | Total: {stat['total']:4} | "
                f"Control: {stat['control']:4} | ASD: {stat['asd']:4} | "
                f"Ratio: {stat['ratio']}"
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
    png_dir = DATA_ROOT / "images"

    # Load metadata
    if not pheno_path.exists():
        logger.error(f"Error: {pheno_path} not found.")
        return False

    df = pd.read_csv(pheno_path)
    df["FILE_ID"] = df["FILE_ID"].astype(str).str.strip()
    logger.info(f"Loaded metadata: {len(df)} records")

    # Load images
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
        logger.info(f"Avg Slices/Sub:    {avg_slices:.1f} (Target: 5.0)")

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
    metadata_ids = set(df["FILE_ID"].unique())
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

    incomplete = {subid: count for subid, count in slice_counts.items() if count != 5}
    complete = sum(1 for count in slice_counts.values() if count == 5)
    logger.info(f"  Complete (5 slices): {complete}/{len(slice_counts)}")

    if incomplete:
        logger.warning(f"  WARNING: Subjects with incomplete slices: {len(incomplete)}")
    else:
        logger.info("  All subjects have complete slice sets (5/5)")

    # Time series files
    logger.info("-" * 40)
    logger.info("TIME SERIES FILES")
    ts_dir = DATA_PROCESSED
    if ts_dir.exists():
        ts_files = list(ts_dir.glob("*_ts.npy"))
        logger.info(f"  Time series files:  {len(ts_files)}")
        ts_subjects = set([f.stem.replace("_ts", "") for f in ts_files])
        missing_ts = completed_subs - ts_subjects

        if missing_ts:
            logger.warning(f"  WARNING: Downloaded subjects missing time series: {len(missing_ts)}")
        else:
            logger.info("  All downloaded subjects have time series")
    else:
        logger.warning(f"  WARNING: Time series directory not found: {ts_dir}")

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

        # PNG validation
        logger.info("\nPNG FILE INTEGRITY (sampling)")
        corrupted_pngs = []
        wrong_size = []
        sample_files = random.sample(downloaded_files, min(sample_png, len(downloaded_files)))

        for png_file in sample_files:
            png_path = png_dir / png_file
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
        if ts_dir.exists():
            logger.info("\nTIME SERIES VALIDATION (sampling)")
            ts_files = list(ts_dir.glob("*_ts.npy"))
            invalid_ts = []
            wrong_shape = []
            sample_ts_files = random.sample(ts_files, min(sample_ts, len(ts_files)))

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


class PipelineHealthCheck:
    """Pipeline validation and diagnostics (YOLO, graphs, stratification)."""

    def __init__(self, visualize: bool = False):
        self.visualize = visualize
        self.output_dir = Path("./results/validation_outputs")
        if visualize:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.issues = []
        self.warnings = []
        self.passed_checks = []
        self.results = {}

    # RESULT TRACKING

    def add_issue(self, stage: str, message: str, fix: str) -> None:
        self.issues.append({"stage": stage, "message": message, "fix": fix})

    def add_warning(self, stage: str, message: str) -> None:
        self.warnings.append({"stage": stage, "message": message})

    def add_pass(self, stage: str, message: str) -> None:
        self.passed_checks.append({"stage": stage, "message": message})

    # VALIDATION CHECKS

    def check_atlas(self) -> bool:
        logger.info("Checking atlas...")

        if not ATLAS_PATH.exists():
            self.add_issue(
                "Atlas",
                f"Atlas missing: {ATLAS_PATH}",
                "Run: python -m src.validation.atlas_validator",
            )
            return False

        try:
            import nibabel as nib

            atlas_img = nib.load(str(ATLAS_PATH))
            data = atlas_img.get_fdata()
            num_rois = len(np.unique(data)) - 1

            valid_counts = {116, 117, 164, 166, 170}
            if num_rois not in valid_counts:
                self.add_warning("Atlas", f"Unexpected ROI count: {num_rois} (expected {valid_counts})")

            self.add_pass("Atlas", f"Valid atlas with {num_rois} ROIs")
            return True

        except Exception as e:
            self.add_issue("Atlas", f"Atlas corrupted: {e}", "Re-download atlas")
            return False

    def check_lobe_mapping(self) -> bool:
        logger.info("Checking LOBE_MAPPING...")

        try:
            if len(LOBE_MAPPING) != NUM_LOBES:
                raise ValueError(f"Expected {NUM_LOBES} lobes, got {len(LOBE_MAPPING)}")

            # Check completeness
            all_rois = set()
            for lobe_id, roi_list in LOBE_MAPPING.items():
                for roi in roi_list:
                    if roi in all_rois:
                        raise ValueError(f"Duplicate ROI {roi} in lobe {lobe_id}")
                    all_rois.add(roi)

            # Verify range (AAL3: 0-169 after 0-based conversion)
            expected_rois = set(range(170))
            if all_rois != expected_rois:
                missing = expected_rois - all_rois
                extra = all_rois - expected_rois
                if missing or extra:
                    self.add_warning("Config", f"ROI coverage: missing={len(missing)}, extra={len(extra)}")

            self.add_pass("Config", "LOBE_MAPPING valid")
            return True

        except ValueError as e:
            self.add_issue("Config", f"LOBE_MAPPING invalid: {e}", "Fix LOBE_MAPPING in src/core/config.py")
            return False

    def check_manifest(self) -> Tuple[bool, Optional[pd.DataFrame]]:
        logger.info("Checking manifest...")

        if not MASTER_MANIFEST.exists():
            self.add_issue("Manifest", f"Manifest missing: {MASTER_MANIFEST}", "Run: python -m src.utils.manifestor")
            return False, None

        try:
            df = pd.read_csv(MASTER_MANIFEST)

            # Check required columns
            required = ["subject_id", "split", "DX_GROUP", "SITE_ID"]
            missing = [c for c in required if c not in df.columns]

            if missing:
                self.add_issue("Manifest", f"Missing columns: {missing}", "Regenerate manifest")
                return False, None

            # Verify splits
            splits = set(df["split"].unique())
            expected_splits = {"train", "val", "test"}
            if not expected_splits.issubset(splits):
                self.add_warning("Manifest", f"Missing splits: {expected_splits - splits}")

            self.add_pass("Manifest", f"{len(df)} subjects across {len(splits)} splits")
            return True, df

        except Exception as e:
            self.add_issue("Manifest", f"Error reading manifest: {e}", "Check manifest file")
            return False, None

    def check_temporal_features(self) -> bool:
        logger.info("Checking temporal features...")

        if not NODE_ATTRIBUTES_TEMPORAL.exists():
            self.add_issue(
                "Features",
                f"Temporal features missing: {NODE_ATTRIBUTES_TEMPORAL}",
                "Run: python -m src.features.extract_temporal",
            )
            return False

        try:
            df = pd.read_csv(NODE_ATTRIBUTES_TEMPORAL)
            feature_cols = [c for c in df.columns if c != "subject_id"]

            # Check for NaNs
            nan_count = df[feature_cols].isna().sum().sum()
            total_values = len(df) * len(feature_cols)
            nan_pct = (nan_count / total_values) * 100

            if nan_pct > 20:
                self.add_issue("Features", f"CRITICAL: {nan_pct:.1f}% NaN values!", "Check feature extraction pipeline")
                return False
            if nan_pct > 5:
                self.add_warning("Features", f"{nan_pct:.1f}% NaN values detected")

            # Estimate ROI count
            expected_rois = len(feature_cols) // NUM_TEMPORAL_FEATURES if NUM_TEMPORAL_FEATURES > 0 else 170

            self.add_pass("Features", f"Temporal features: {len(df)} subjects, ~{expected_rois} ROIs")
            return True

        except Exception as e:
            self.add_issue("Features", f"Error reading features: {e}", "Regenerate temporal features")
            return False

    def check_harmonization(self) -> bool:
        logger.info("Checking harmonization...")

        if not NODE_ATTRIBUTES_HARMONIZED.exists():
            self.add_issue(
                "Harmonization",
                f"Harmonized features missing: {NODE_ATTRIBUTES_HARMONIZED}",
                "Run: python -m src.features.fold_safe_harmonization",
            )
            return False

        try:
            df = pd.read_csv(NODE_ATTRIBUTES_HARMONIZED)
            feature_cols = [c for c in df.columns if c != "subject_id"]

            # Check for NaNs (should be ZERO)
            nan_count = df[feature_cols].isna().sum().sum()
            if nan_count > 0:
                self.add_issue(
                    "Harmonization",
                    f"CRITICAL: {nan_count} NaNs after harmonization!",
                    "Re-run fold_safe_harmonization.py",
                )
                return False

            # Check for infinites
            inf_count = np.isinf(df[feature_cols].values).sum()
            if inf_count > 0:
                self.add_warning("Harmonization", f"{inf_count} infinite values detected")

            self.add_pass("Harmonization", f"Clean harmonized features: {len(df)} subjects")
            return True

        except Exception as e:
            self.add_issue("Harmonization", f"Error reading harmonized features: {e}", "Re-run harmonization")
            return False

    def check_spatial_features(self) -> bool:
        logger.info("Checking spatial features...")

        if not NODE_FEATURES_3D.exists():
            self.add_issue(
                "Spatial Features",
                f"3D features missing: {NODE_FEATURES_3D}",
                "Run: python -m src.features.extract_spatial",
            )
            return False

        try:
            df = pd.read_csv(NODE_FEATURES_3D)

            # Check detection completeness
            if "node_count" in df.columns:
                complete = (df["node_count"] == NUM_LOBES).sum()
                survival_rate = complete / len(df) * 100

                self.results["yolo_survival_rate"] = survival_rate

                if survival_rate < 80:
                    self.add_warning(
                        "Spatial Features",
                        f"LOW survival rate: {survival_rate:.1f}% ({complete}/{len(df)})",
                    )
                else:
                    self.add_pass(
                        "Spatial Features",
                        f"{complete}/{len(df)} subjects with all {NUM_LOBES} lobes ({survival_rate:.1f}%)",
                    )

            return True

        except Exception as e:
            self.add_issue("Spatial Features", f"Error reading spatial features: {e}", "Re-run feature extraction")
            return False

    def check_causal_graphs(self, manifest: Optional[pd.DataFrame] = None) -> Dict:
        logger.info("Checking causal graphs...")

        if not CAUSAL_GRAPHS_DIR.exists():
            self.add_issue(
                "Graphs",
                f"Graph directory missing: {CAUSAL_GRAPHS_DIR}",
                "Run: python -m src.features.construct_causal",
            )
            return {"available": 0, "missing": 0, "corrupted": 0}

        graph_files = list(CAUSAL_GRAPHS_DIR.glob("*_graph.pt"))

        if not graph_files:
            self.add_issue("Graphs", "No graph files found", "Run: python -m src.features.construct_causal")
            return {"available": 0, "missing": 0, "corrupted": 0}

        stats = {
            "total_files": len(graph_files),
            "valid": 0,
            "corrupted": 0,
            "zero_edges": 0,
            "edge_counts": [],
        }

        # Sample graphs for analysis
        sample_size = min(200, len(graph_files))
        for graph_file in np.random.choice(graph_files, sample_size, replace=False):
            try:
                graph_data = torch.load(graph_file, weights_only=False)

                if "adj" not in graph_data:
                    stats["corrupted"] += 1
                    continue

                adj = graph_data["adj"]

                # Check for NaN/Inf
                if torch.isnan(adj).any() or torch.isinf(adj).any():
                    stats["corrupted"] += 1
                    continue

                # Count edges
                num_edges = (adj != 0).sum().item()
                stats["edge_counts"].append(num_edges)

                if num_edges == 0:
                    stats["zero_edges"] += 1
                else:
                    stats["valid"] += 1

            except Exception:
                stats["corrupted"] += 1

        # Calculate statistics
        if stats["edge_counts"]:
            stats["mean_edges"] = np.mean(stats["edge_counts"])
            stats["median_edges"] = np.median(stats["edge_counts"])

        # Report findings
        if stats["corrupted"] > sample_size * 0.05:
            self.add_warning("Graphs", f"{stats['corrupted']}/{sample_size} graphs corrupted")

        if stats["zero_edges"] > sample_size * 0.05:
            self.add_warning("Graphs", f"{stats['zero_edges']}/{sample_size} graphs have zero edges")
            self.add_warning("Graphs", f"Consider lowering SPARSITY_QUANTILE from {SPARSITY_QUANTILE}")

        if stats["valid"] > 0:
            self.add_pass(
                "Graphs",
                f"{len(graph_files)} graphs available, mean edges: {stats.get('mean_edges', 0):.1f}/{NUM_LOBES * NUM_LOBES}",
            )

        self.results["graph_stats"] = stats
        return stats

    def check_stratification(self, manifest: Optional[pd.DataFrame] = None) -> Dict:
        logger.info("Checking stratification...")

        if manifest is None:
            _, manifest = self.check_manifest()
            if manifest is None:
                return {}

        stats = {"total": len(manifest), "splits": {}}

        # Per-split analysis
        for split in ["train", "val", "test"]:
            split_data = manifest[manifest["split"] == split]

            if len(split_data) == 0:
                continue

            dx_counts = split_data["DX_GROUP"].value_counts()

            stats["splits"][split] = {
                "total": len(split_data),
                "control": dx_counts.get(2, 0),
                "asd": dx_counts.get(1, 0),
                "num_sites": split_data["SITE_ID"].nunique(),
            }

        # Check data leakage
        subject_counts = manifest.groupby("subject_id")["split"].nunique()
        leakage = (subject_counts > 1).sum()

        if leakage > 0:
            self.add_issue(
                "Stratification",
                f"DATA LEAKAGE: {leakage} subjects in multiple splits!",
                "Re-run split.py with proper stratification",
            )
        else:
            self.add_pass("Stratification", f"No data leakage, {len(stats['splits'])} splits")

        self.results["stratification"] = stats
        return stats

    # VISUALIZATION (OPTIONAL)

    def visualize_results(self) -> None:
        if not self.visualize:
            return

        try:
            # Graph sparsity visualization
            if "graph_stats" in self.results:
                stats = self.results["graph_stats"]
                if stats.get("edge_counts"):
                    self._plot_graph_sparsity(stats["edge_counts"])

            # Stratification visualization
            if "stratification" in self.results:
                manifest = pd.read_csv(MASTER_MANIFEST)
                self._plot_stratification(manifest)

            logger.info(f"Visualizations saved to: {self.output_dir}")

        except ImportError:
            logger.warning("matplotlib/seaborn not available, skipping visualizations")

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

        # Split distribution
        ax = axes[0]
        split_dx = manifest.groupby(["split", "DX_GROUP"]).size().unstack(fill_value=0)
        split_dx.columns = ["ASD", "Control"]
        split_dx.plot(kind="bar", ax=ax, color=["#e74c3c", "#3498db"])
        ax.set_title("Distribution by Split")
        ax.set_ylabel("Number of Subjects")
        ax.legend(title="Diagnosis")

        # Class balance
        ax = axes[1]
        split_ratios = []
        for split in ["train", "val", "test"]:
            split_data = manifest[manifest["split"] == split]
            asd = (split_data["DX_GROUP"] == 1).sum()
            ratio = asd / len(split_data) if len(split_data) > 0 else 0
            split_ratios.append(ratio)

        ax.bar(["Train", "Val", "Test"], split_ratios)
        ax.axhline(0.5, color="black", linestyle="--", label="Perfect Balance")
        ax.set_title("ASD Ratio per Split")
        ax.set_ylabel("Proportion")
        ax.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / "stratification.png", dpi=150)
        plt.close()

    # REPORTING

    def generate_report(self) -> bool:
        print("\n" + "=" * 70)
        print("NEURO-CXG PIPELINE HEALTH CHECK")
        print("=" * 70)

        # Passed checks
        if self.passed_checks:
            print("\nPASSED CHECKS:")
            print("-" * 70)
            for check in self.passed_checks:
                print(f"  [{check['stage']}] {check['message']}")

        # Warnings
        if self.warnings:
            print("\nWARNINGS:")
            print("-" * 70)
            for warn in self.warnings:
                print(f"  [{warn['stage']}] {warn['message']}")

        # Critical issues
        if self.issues:
            print("\nCRITICAL ISSUES:")
            print("-" * 70)
            for issue in self.issues:
                print(f"\n  [{issue['stage']}]")
                print(f"  Problem: {issue['message']}")
                print(f"  Fix: {issue['fix']}")

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY:")
        print(f"  Passed: {len(self.passed_checks)}")
        print(f"  Warnings: {len(self.warnings)}")
        print(f"  Critical Issues: {len(self.issues)}")
        print("=" * 70)

        if self.issues:
            print("\nPIPELINE HAS CRITICAL ISSUES")
            return False
        if self.warnings:
            print("\nPipeline functional with warnings")
            return True
        print("\nPIPELINE FULLY HEALTHY")
        return True

    # MAIN EXECUTION

    def run_full_check(self) -> bool:
        self.check_atlas()
        self.check_lobe_mapping()
        manifest_ok, manifest = self.check_manifest()

        if manifest_ok:
            self.check_temporal_features()
            self.check_harmonization()
            self.check_spatial_features()
            self.check_causal_graphs(manifest)
            self.check_stratification(manifest)

        self.visualize_results()
        return self.generate_report()


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

    def __init__(self):
        self.results: List[ValidationResult] = []
        self.metrics: Dict = {}

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

        expected_features = NUM_TEMPORAL_FEATURES + NUM_SPATIAL_FEATURES
        if GNN_IN_CHANNELS != expected_features:
            self.add_result(
                ValidationResult(
                    stage="Environment",
                    passed=False,
                    message=f"Config mismatch: GNN_IN_CHANNELS={GNN_IN_CHANNELS} but expected {expected_features}",
                    severity="critical",
                    fix_suggestion="Set GNN_IN_CHANNELS = NUM_TEMPORAL_FEATURES + NUM_SPATIAL_FEATURES",
                )
            )
            all_passed = False

        return all_passed

    # STAGE 2: DATA VALIDATION

    def validate_downloaded_data(self) -> bool:
        logger.info("\n" + "=" * 70)
        logger.info("STAGE 2: DATA VALIDATION")
        logger.info("=" * 70)

        all_passed = True

        image_dir = DATA_ROOT / "images"
        if not image_dir.exists() or not list(image_dir.glob("*.png")):
            self.add_result(
                ValidationResult(
                    stage="Data",
                    passed=False,
                    message="No PNG images found",
                    severity="critical",
                    fix_suggestion="Run: python -m src.data.abide_download",
                )
            )
            return False

        png_files = list(image_dir.glob("*.png"))
        subjects = set(f.stem.rsplit("_z", 1)[0] for f in png_files)

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

        sample_size = min(10, len(ts_files))
        corrupted = 0
        wrong_shape = 0

        for ts_file in np.random.choice(ts_files, sample_size, replace=False):
            try:
                data = np.load(ts_file)
                if data.ndim != 2:
                    wrong_shape += 1
                elif np.isnan(data).any() or np.isinf(data).any():
                    corrupted += 1
            except Exception:
                corrupted += 1

        if corrupted > 0 or wrong_shape > 0:
            self.add_result(
                ValidationResult(
                    stage="Data",
                    passed=False,
                    message=f"Sample validation: {corrupted} corrupted, {wrong_shape} wrong shape",
                    severity="critical",
                    fix_suggestion="Re-run data download",
                )
            )
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
                    severity="critical",
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
                    severity="critical",
                    fix_suggestion="Run: python -m src.features.extract_spatial",
                )
            )
            return False

        try:
            spatial_df = pd.read_csv(NODE_FEATURES_3D)

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
                    severity="critical",
                    fix_suggestion="Run: python -m src.features.construct_causal",
                )
            )
            return False

        graph_files = list(CAUSAL_GRAPHS_DIR.glob("*_graph.pt"))

        sample_size = min(50, len(graph_files))
        stats = {"valid": 0, "corrupted": 0, "zero_edges": 0, "wrong_shape": 0, "edge_counts": []}

        for graph_file in np.random.choice(graph_files, sample_size, replace=False):
            try:
                graph_data = torch.load(graph_file, weights_only=False)

                if "adj" not in graph_data:
                    stats["corrupted"] += 1
                    continue

                adj = graph_data["adj"]

                if adj.shape != (NUM_LOBES, NUM_LOBES):
                    stats["wrong_shape"] += 1
                    continue

                if torch.isnan(adj).any() or torch.isinf(adj).any():
                    stats["corrupted"] += 1
                    continue

                num_edges = (adj != 0).sum().item()
                stats["edge_counts"].append(num_edges)

                if num_edges == 0:
                    stats["zero_edges"] += 1
                else:
                    stats["valid"] += 1

            except Exception:
                stats["corrupted"] += 1

        all_passed = True

        if stats["corrupted"] > sample_size * 0.05:
            self.add_result(
                ValidationResult(
                    stage="Graphs",
                    passed=False,
                    message=f"{stats['corrupted']}/{sample_size} graphs corrupted",
                    severity="critical",
                    fix_suggestion="Re-run graph construction",
                )
            )
            all_passed = False

        if stats["wrong_shape"] > 0:
            self.add_result(
                ValidationResult(
                    stage="Graphs",
                    passed=False,
                    message=f"{stats['wrong_shape']} graphs have wrong shape (expected {NUM_LOBES}x{NUM_LOBES})",
                    severity="critical",
                    fix_suggestion="Clear graph directory and rebuild",
                )
            )
            all_passed = False

        if stats["zero_edges"] > sample_size * 0.1:
            self.add_result(
                ValidationResult(
                    stage="Graphs",
                    passed=False,
                    message=f"{stats['zero_edges']}/{sample_size} graphs have zero edges",
                    severity="warning",
                    fix_suggestion=f"Lower SPARSITY_QUANTILE from {SPARSITY_QUANTILE} to 0.70 or 0.60",
                )
            )

        if stats["edge_counts"]:
            mean_edges = np.mean(stats["edge_counts"])
            median_edges = np.median(stats["edge_counts"])

            self.add_result(
                ValidationResult(
                    stage="Graphs",
                    passed=True,
                    message=f"{len(graph_files)} graphs, mean edges: {mean_edges:.1f}, median: {median_edges:.0f}",
                    severity="info",
                    metrics={
                        "total_graphs": len(graph_files),
                        "mean_edges": mean_edges,
                        "median_edges": median_edges,
                        "max_edges": max(stats["edge_counts"]),
                        "min_edges": min(stats["edge_counts"]),
                    },
                )
            )

        return all_passed

    # STAGE 5: MODEL VALIDATION

    def validate_trained_models(self) -> bool:
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

        fold_metrics = []
        for ckpt_path in fold_checkpoints:
            try:
                ckpt = torch.load(ckpt_path, weights_only=False)

                required_keys = ["model_state", "epoch"]
                missing = [k for k in required_keys if k not in ckpt]

                if missing:
                    self.add_result(
                        ValidationResult(
                            stage="Models",
                            passed=False,
                            message=f"{ckpt_path.name} missing keys: {missing}",
                            severity="warning",
                        )
                    )
                else:
                    metrics = {
                        "fold": ckpt_path.stem.replace("best_model_fold", ""),
                        "epoch": ckpt.get("epoch", -1),
                        "auc": ckpt.get("auc", 0.0),
                        "f1": ckpt.get("f1", 0.0),
                    }
                    fold_metrics.append(metrics)

            except Exception as e:
                self.add_result(
                    ValidationResult(
                        stage="Models",
                        passed=False,
                        message=f"Error loading {ckpt_path.name}: {e}",
                        severity="warning",
                    )
                )

        if fold_metrics:
            mean_auc = np.mean([m["auc"] for m in fold_metrics])
            mean_f1 = np.mean([m["f1"] for m in fold_metrics])

            self.add_result(
                ValidationResult(
                    stage="Models",
                    passed=True,
                    message=f"{len(fold_checkpoints)} trained models, mean AUC: {mean_auc:.4f}, F1: {mean_f1:.4f}",
                    severity="info",
                    metrics={"fold_metrics": fold_metrics},
                )
            )

        return True

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
        self.validate_downloaded_data()
        self.validate_features()
        self.validate_graphs()
        self.validate_trained_models()

        is_healthy, _ = self.generate_report()
        return is_healthy


def run_quality_validation(visualize: bool = False, strict: bool = False) -> bool:
    checker = PipelineHealthCheck(visualize=visualize)
    is_healthy = checker.run_full_check()
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
