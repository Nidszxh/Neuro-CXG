import os
import logging
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from pathlib import Path
import sys

# Setup paths and config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    DATA_FINAL,
    MASTER_MANIFEST,
    LOBE_NAMES,
    RESULTS_DIR,
    NODE_FEATURES_3D,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- CONFIG ---
# Ensure this matches your latest successful YOLO run
MODEL_PATH    = RESULTS_DIR / "ROI_Detection_v222" / "weights" / "best.pt"
SPLIT_ROOT    = DATA_FINAL
MANIFEST_PATH = MASTER_MANIFEST
OUTPUT_PATH   = NODE_FEATURES_3D

def extract_features():
    if not MODEL_PATH.exists():
        logger.error(f"Model weights not found at {MODEL_PATH}. Rerun YOLO training first.")
        return

    model = YOLO(MODEL_PATH)
    all_detections = []
    
    for split in ['train', 'val', 'test']:
        img_dir = SPLIT_ROOT / split / "images"
        if not img_dir.exists(): continue
            
        logger.info(f"Processing {split} set...")
        results = model(str(img_dir), stream=True, conf=0.35)
        
        for res in tqdm(results, desc=f"Inference {split}"):
            file_name = Path(res.path).stem
            try:
                # Extract subject and slice Z-coordinate
                subject_id, z_str = file_name.rsplit('_z', 1)
                z_coord = int(z_str)
            except: continue 

            for box in res.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                c = box.xywhn[0].cpu().numpy() # [x_center, y_center, width, height]
                
                all_detections.append({
                    'subject_id': subject_id,
                    'roi_class': cls,
                    'x': c[0],
                    'y': c[1],
                    'z': z_coord,
                    'w': c[2],
                    'h': c[3],
                    'conf': conf
                })

    raw_df = pd.DataFrame(all_detections)
    
    # --- STICKY AGGREGATION LOGIC ---
    # We must convert 2D slice detections into ONE 3D entry per lobe
    logger.info("Aggregating 2D detections into 3D Lobe nodes...")
    
    processed_subjects = []
    subject_ids = raw_df['subject_id'].unique()

    for sub_id in tqdm(subject_ids, desc="Building Subject Nodes"):
        sub_group = raw_df[raw_df['subject_id'] == sub_id]
        
        # Check if we found all 5 lobes for this subject
        if sub_group['roi_class'].nunique() < 5:
            continue # Skip incomplete subjects to prevent NaNs in the GNN

        subject_row = {'subject_id': sub_id}
        
        # Aggregate each lobe (0-4)
        for lobe_id in range(5):
            lobe_name = LOBE_NAMES[lobe_id]
            lobe_data = sub_group[sub_group['roi_class'] == lobe_id]
            
            # 3D Centroid calculation
            subject_row[f"{lobe_name}_x"] = lobe_data['x'].mean()
            subject_row[f"{lobe_name}_y"] = lobe_data['y'].mean()
            subject_row[f"{lobe_name}_z_depth"] = lobe_data['z'].mean()
            
            # Physical dimension averages
            subject_row[f"{lobe_name}_w"] = lobe_data['w'].mean()
            subject_row[f"{lobe_name}_h"] = lobe_data['h'].mean()
            
            # Confidence metric
            subject_row[f"{lobe_name}_conf"] = lobe_data['conf'].max()

        processed_subjects.append(subject_row)

    final_df = pd.DataFrame(processed_subjects)
    
    # Merge with Master Manifest to get Diagnosis (DX_GROUP) and Split
    manifest = pd.read_csv(MANIFEST_PATH)
    manifest['subject_id'] = manifest['subject_id'].astype(str)
    
    # Final data assembly
    output_df = pd.merge(final_df, manifest, on='subject_id', how='inner')

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)
    
    logger.info(f"✓ Success: {len(output_df)} subjects aligned to 5-node architecture.")
    logger.info(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    extract_features()