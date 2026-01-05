import os
import numpy as np
from PIL import Image
from collections import Counter
from pathlib import Path

# Updated Paths
PNG_DIR = Path("./data/images")
TS_DIR = Path("./data/processed")
TARGET_SLICES = 5 

def check_dataset_integrity():
    png_files = list(PNG_DIR.glob("*.png"))
    corrupted_pngs = []
    subject_counts = Counter()
    
    if not png_files:
        print("No images found to check.")
    else:
        print(f"Scanning {len(png_files)} images for integrity...")
        for path in png_files:
            # 1. Check physical corruption
            try:
                with Image.open(path) as img:
                    img.verify()
                sub_id = path.name.rsplit('_z', 1)[0]
                subject_counts[sub_id] += 1
            except Exception:
                print(f" [!] Corrupted PNG: {path.name}")
                corrupted_pngs.append(path)

    # 2. Check NPY Integrity
    npy_files = list(TS_DIR.glob("*.npy"))
    corrupted_npys = []
    ts_subjects = set()
    
    print(f"Scanning {len(npy_files)} time-series files...")
    for path in npy_files:
        try:
            data = np.load(path)
            if np.isnan(data).any():
                raise ValueError("NaNs detected")
            ts_subjects.add(path.name.replace("_ts.npy", ""))
        except Exception as e:
            print(f" [!] Invalid NPY: {path.name} ({e})")
            corrupted_npys.append(path)

    # 3. Identify Incomplete Subjects
    all_subjects = set(subject_counts.keys()) | ts_subjects
    incomplete_subs = []
    for sub in all_subjects:
        if subject_counts[sub] < TARGET_SLICES or sub not in ts_subjects:
            incomplete_subs.append(sub)

    print("\n" + "="*40)
    print(f"FINAL INTEGRITY REPORT")
    print("="*40)
    print(f"Corrupted PNGs:      {len(corrupted_pngs)}")
    print(f"Corrupted NPYs:      {len(corrupted_npys)}")
    print(f"Incomplete Subjects: {len(incomplete_subs)} (Missing slices or TS)")
    print("-" * 40)

    if corrupted_pngs or corrupted_npys or incomplete_subs:
        print("OPTIONS: [1] Delete Corrupted | [2] Purge Incomplete Subjects | [3] Exit")
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
            print("Purge complete.")

if __name__ == "__main__":
    check_dataset_integrity()