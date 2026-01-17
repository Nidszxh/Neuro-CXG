import os, logging, sys
import numpy as np
from PIL import Image
from collections import Counter
from pathlib import Path

# Setup paths from config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.core.config import DATA_PROCESSED, DATA_ROOT

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
TARGET_SLICES = 5
PNG_DIR = DATA_ROOT / "images"
TS_DIR = DATA_PROCESSED

def check_dataset_integrity():
    png_files = list(PNG_DIR.glob("*.png"))
    corrupted_pngs = []
    subject_counts = Counter()
    
    if not png_files:
        logger.info("No images found to check.")
    else:
        logger.info(f"Scanning {len(png_files)} images for integrity...")
        for path in png_files:
            # 1. Check physical corruption
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
    logger.info(f"FINAL INTEGRITY REPORT")
    logger.info("="*40)
    logger.info(f"Corrupted PNGs:      {len(corrupted_pngs)}")
    logger.info(f"Corrupted NPYs:      {len(corrupted_npys)}")
    logger.info(f"Incomplete Subjects: {len(incomplete_subs)} (Missing slices or TS)")
    logger.info("-" * 40)

    if corrupted_pngs or corrupted_npys or incomplete_subs:
        logger.info("OPTIONS: [1] Delete Corrupted | [2] Purge Incomplete Subjects | [3] Exit")
        choice = input("Select (1/2/3): ")
        if choice == '1':
            for p in corrupted_pngs + corrupted_npys: os.remove(p)
        elif choice == '2':
            for sub in incomplete_subs:
                # Remove PNGs
                for p in PNG_DIR.glob(f"{sub}_z*.png"): os.remove(p)
                # Remove NPYs
                for p in TS_DIR.glob(f"{sub}_ts.npy"): os.remove(p)
                for p in TS_DIR.glob(f"{sub}_qc.json"): os.remove(p)
            logger.info("Purge complete.")

if __name__ == "__main__":
    check_dataset_integrity()