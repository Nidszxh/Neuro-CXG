"""
Combined data integrity validation module.

Provides two main functions:
1. check_dataset_integrity() - Post-download validation (checks PNGs, NPYs, incomplete subjects)
2. check_distribution() - Pre-GNN validation (checks dataset completeness across train/val/test splits)
"""

import os
import sys
import logging
import numpy as np
from PIL import Image
from collections import Counter
from pathlib import Path

# Setup paths from config
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import DATA_PROCESSED, DATA_ROOT

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
TARGET_SLICES = 5
EXPECTED_ROIS = 170  # AAL3 atlas ROI count
PNG_DIR = DATA_ROOT / "images"
TS_DIR = DATA_PROCESSED


def check_dataset_integrity():
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
                sub_id = path.name.rsplit('_z', 1)[0]
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
            if data.shape[1] != EXPECTED_ROIS:
                raise ValueError(f"ROI mismatch: Found {data.shape[1]}, expected {EXPECTED_ROIS}")
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

    logger.info("\n" + "="*40)
    logger.info(f"POST-DOWNLOAD INTEGRITY REPORT")
    logger.info("="*40)
    logger.info(f"Corrupted PNGs:      {len(corrupted_pngs)}")
    logger.info(f"Corrupted NPYs:      {len(corrupted_npys)}")
    logger.info(f"Incomplete Subjects: {len(incomplete_subs)} (Missing slices or TS)")
    logger.info("-" * 40)

    if corrupted_pngs or corrupted_npys or incomplete_subs:
        logger.info("OPTIONS: [1] Delete Corrupted | [2] Purge Incomplete Subjects | [3] Exit")
        choice = input("Select (1/2/3): ")
        if choice == '1':
            for p in corrupted_pngs + corrupted_npys:
                os.remove(p)
                logger.info(f"Deleted: {p}")
        elif choice == '2':
            for sub in incomplete_subs:
                # Remove PNGs
                for p in PNG_DIR.glob(f"{sub}_z*.png"):
                    os.remove(p)
                # Remove NPYs
                for p in TS_DIR.glob(f"{sub}_ts.npy"):
                    os.remove(p)
                for p in TS_DIR.glob(f"{sub}_qc.json"):
                    os.remove(p)
            logger.info("Purge complete.")


def check_distribution():
    """
    Pre-GNN integrity check.
    
    Validates dataset completeness across train/val/test splits:
    - Checks each subject has target number of slices
    - Verifies image/label file pairing for YOLO training
    - Reports distribution of slice counts per split
    """
    logger.info("Starting pre-GNN integrity check...")
    
    PROCESSED_ROOT = str(DATA_PROCESSED)
    
    if not os.path.exists(PROCESSED_ROOT):
        logger.error(f"Path {PROCESSED_ROOT} does not exist. Run split.py first.")
        return

    splits = ['train', 'val', 'test']
    overall_stats = {}

    logger.info(f"Dataset Completeness Report (Target: {TARGET_SLICES} slices/subject)")
    
    for split in splits:
        img_path = os.path.join(PROCESSED_ROOT, split, 'images')
        lbl_path = os.path.join(PROCESSED_ROOT, split, 'labels')
        
        if not os.path.exists(img_path):
            logger.warning(f"[!] Split '{split}' images folder missing.")
            continue

        files = [f for f in os.listdir(img_path) if f.endswith('.png')]
        subject_counts = {}
        
        for f in files:
            # Consistent splitting logic used throughout codebase
            sub_id = f.rsplit('_z', 1)[0]
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
            labels = [f for f in os.listdir(lbl_path) if f.endswith('.txt')]
            if len(labels) != len(files):
                logger.warning(f"  [!] ALERT: Image/Label Mismatch! ({len(files)} images vs {len(labels)} labels)")
            else:
                logger.info(f"  ✓ Image/Label count matches ({len(files)} files)")
    
    logger.info("\nPre-GNN integrity check complete.")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--distribution":
        check_distribution()
    elif len(sys.argv) > 1 and sys.argv[1] == "--dataset":
        check_dataset_integrity()
    else:
        # Run both by default
        check_dataset_integrity()
        print("\n")
        check_distribution()
