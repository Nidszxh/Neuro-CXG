import os, shutil, random, pandas as pd
from pathlib import Path

# --- PATHS ---
PROJECT_ROOT = Path("./data")
SOURCE_IMG   = PROJECT_ROOT / "images"
SOURCE_TS    = PROJECT_ROOT / "processed"  # Where .npy files are
SOURCE_LBL   = PROJECT_ROOT / "labels"
TARGET_ROOT  = PROJECT_ROOT / "final"
PHENO_PATH   = PROJECT_ROOT / "processed" / "Phenotypic_V1_0b_preprocessed1.csv"

# --- CONFIG ---
TRAIN_RATIO, VAL_RATIO = 0.70, 0.15 
random.seed(42)

def run_stratified_split():
    # 1. Load labels to ensure stratification (Phase 2.2)
    df = pd.read_csv(PHENO_PATH)
    # Only include subjects we actually have files for
    all_images = [f for f in os.listdir(SOURCE_IMG) if f.endswith('.png')]
    valid_ids = set([f.rsplit('_z', 1)[0] for f in all_images])
    df = df[df['FILE_ID'].isin(valid_ids)]

    # 2. Group by Site and Diagnosis for stratification
    # This ensures a 'site-balanced' split, which is a Q1 Journal requirement
    from sklearn.model_selection import train_test_split
    
    train_df, rem_df = train_test_split(
        df, train_size=TRAIN_RATIO, stratify=df[['DX_GROUP', 'SITE_ID']], random_state=42
    )
    
    # Split the remaining 30% into half (15% Val, 15% Test)
    val_df, test_df = train_test_split(
        rem_df, train_size=0.5, stratify=rem_df[['DX_GROUP', 'SITE_ID']], random_state=42
    )

    splits = {'train': train_df, 'val': val_df, 'test': test_df}

    # 3. Execute Move
    for name, split_df in splits.items():
        print(f"📦 Organizing {name} set ({len(split_df)} subjects)...")
        
        img_dst = TARGET_ROOT / name / 'images'
        lbl_dst = TARGET_ROOT / name / 'labels'
        ts_dst  = TARGET_ROOT / name / 'time_series'
        
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

    print(f"\n✅ SUCCESS: Stratified split complete. Saved to {TARGET_ROOT}")

if __name__ == "__main__":
    run_stratified_split()