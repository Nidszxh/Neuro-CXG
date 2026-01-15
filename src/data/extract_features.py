import os
import logging
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from pathlib import Path
import sys

# Setup paths and config
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
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
MODEL_PATH    = RESULTS_DIR / "ROI_Detection_v20_Final2" / "weights" / "best.pt"
SPLIT_ROOT    = DATA_FINAL
MANIFEST_PATH = MASTER_MANIFEST
OUTPUT_PATH   = NODE_FEATURES_3D

def extract_features():
    if not MODEL_PATH.exists():
        logger.error(f"Model weights not found at {MODEL_PATH}")
        return

    model = YOLO(MODEL_PATH)
    all_detections = []
    
    # Process each split (Phase 2.2)
    for split in ['train', 'val', 'test']:
        img_dir = SPLIT_ROOT / split / "images"
        if not img_dir.exists(): 
            logger.warning(f"Image directory not found: {img_dir}")
            continue
            
        logger.info(f"Processing {split} set from {img_dir}")
        # stream=True is essential for RAM management during batch inference
        results = model(str(img_dir), stream=True, conf=0.35)
        
        for res in tqdm(results, desc=f"Inference {split}"):
            file_name = Path(res.path).stem
            try:
                subject_id, z_str = file_name.rsplit('_z', 1)
                z_coord = int(z_str)
            except: 
                continue 

            for box in res.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                # xywhn: [x_center, y_center, width, height] normalized
                c = box.xywhn[0].cpu().numpy() 
                
                all_detections.append({
                    'subject_id': subject_id,
                    'roi_class': cls,
                    'x': c[0], 'y': c[1], 'z_depth': z_coord,
                    'w': c[2], 'h': c[3],
                    'conf': conf
                })

    if not all_detections:
        logger.error("No detections found across all splits")
        return
    
    raw_df = pd.DataFrame(all_detections)
    logger.info(f"Collected {len(raw_df)} detections across splits")

    # --- Q1 AGGREGATION LOGIC (Phase 4.2) ---
    # We aggregate the 5 slices into 1 set of 3D features per ROI per subject
    agg_funcs = {
        'x': 'mean', 
        'y': 'mean', 
        'z_depth': 'mean',  # This represents the weighted depth centroid
        'w': 'mean', 
        'h': 'mean',
        'conf': 'max'  # We take the highest confidence detection for each lobe
    }
    
    df_pivot = raw_df.groupby(['subject_id', 'roi_class']).agg(agg_funcs).unstack()
    
    # Flatten columns: e.g., (x, 0) -> frontal_x
    df_pivot.columns = [f"{LOBE_NAMES.get(c[1], c[1])}_{c[0]}" for c in df_pivot.columns]
    
    # --- ANATOMICAL NORMALIZATION ---
    # Calculate Total Detected Area to normalize individual lobe sizes
    area_cols = [c for c in df_pivot.columns if '_w' in c]
    
    # Verify Graph Completeness (Must have 5 nodes for Causal Model)
    df_pivot['node_count'] = df_pivot.filter(like='_conf').notna().sum(axis=1)
    
    # Q1 Filter: Only keep subjects where all 5 lobes were detected across the slices
    final_subjects = df_pivot[df_pivot['node_count'] == 5].copy()
    logger.info(f"Filtered to {len(final_subjects)} subjects with all 5 lobes detected")
    
    # Merge with Master Manifest (Phase 2.2)
    manifest = pd.read_csv(MANIFEST_PATH)
    manifest['subject_id'] = manifest['subject_id'].astype(str)
    
    final_df = pd.merge(final_subjects, manifest, on='subject_id', how='inner')

    # Save for GNN Node Initialization (Phase 6.3)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)
    
    logger.info(f"Generated 3D Node Features for {len(final_df)} complete graphs saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    extract_features()