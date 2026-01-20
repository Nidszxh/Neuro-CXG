import os
import logging
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from pathlib import Path
import sys

# Setup paths and config
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import (
    DATA_FINAL,
    MASTER_MANIFEST,
    LOBE_NAMES,
    RESULTS_DIR,
    NODE_FEATURES_3D,
    NUM_LOBES,
    NUM_SPATIAL_FEATURES
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- CONFIG ---
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
        if not img_dir.exists(): 
            continue
        
        try:
            logger.info(f"Processing {split} set...")
            results = model(str(img_dir), stream=True, conf=0.35)
            
            for res in tqdm(results, desc=f"Inference {split}"):
                try:
                    file_name = Path(res.path).stem
                    try:
                        subject_id, z_str = file_name.rsplit('_z', 1)
                        z_coord = int(z_str)
                    except ValueError:
                        logger.debug(f"Skipped malformed filename: {file_name}")
                        continue

                    for box in res.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        c = box.xywhn[0].cpu().numpy()
                        
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
                except Exception as e:
                    logger.warning(f"Error processing result: {e}")
                    continue
        except Exception as e:
            logger.error(f"Error during inference on {split} set: {e}")
            raise

    raw_df = pd.DataFrame(all_detections)
    
    logger.info("Aggregating 2D detections into 3D Lobe nodes...")
    
    processed_subjects = []
    subject_ids = raw_df['subject_id'].unique()

    for sub_id in tqdm(subject_ids, desc="Building Subject Nodes"):
        sub_group = raw_df[raw_df['subject_id'] == sub_id]
        
        # Check if we found all 5 lobes
        if sub_group['roi_class'].nunique() < NUM_LOBES:
            continue

        subject_row = {'subject_id': sub_id}
        
        # Aggregate each lobe (0-4)
        for lobe_id in range(NUM_LOBES):
            lobe_name = LOBE_NAMES[lobe_id]
            lobe_data = sub_group[sub_group['roi_class'] == lobe_id]
            
            # SPATIAL FEATURE 1-3: 3D Centroid
            subject_row[f"{lobe_name}_x"] = lobe_data['x'].mean()
            subject_row[f"{lobe_name}_y"] = lobe_data['y'].mean()
            subject_row[f"{lobe_name}_z_depth"] = lobe_data['z'].mean()
            
            # SPATIAL FEATURE 4: Bounding box area (size)
            subject_row[f"{lobe_name}_size"] = (
                lobe_data['w'].mean() * lobe_data['h'].mean()
            )
            
            # SPATIAL FEATURE 5: Confidence consistency (how stable across slices)
            subject_row[f"{lobe_name}_conf_std"] = (
                lobe_data['conf'].std() if len(lobe_data) > 1 else 0.0
            )
            
            # SPATIAL FEATURE 6: Detection frequency
            subject_row[f"{lobe_name}_detection_count"] = len(lobe_data)
            
            # REMOVED: {lobe_name}_conf (redundant, already captured by conf_std)

        processed_subjects.append(subject_row)

    final_df = pd.DataFrame(processed_subjects)
    
    # Merge with Master Manifest
    manifest = pd.read_csv(MANIFEST_PATH)
    manifest['subject_id'] = manifest['subject_id'].astype(str)
    
    output_df = pd.merge(final_df, manifest, on='subject_id', how='inner')

    # Validate feature count
    lobe_cols = [col for col in output_df.columns if any(
        col.startswith(f"{lobe}_") for lobe in LOBE_NAMES.values()
    )]
    
    expected_cols = NUM_LOBES * NUM_SPATIAL_FEATURES
    if len(lobe_cols) != expected_cols:
        logger.warning(
            f"Feature count mismatch! Expected {expected_cols} spatial feature columns, "
            f"got {len(lobe_cols)}"
        )
    
    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)
    
    logger.info(f"✓ Success: {len(output_df)} subjects with {NUM_LOBES}-node architecture")
    logger.info(f"✓ {NUM_SPATIAL_FEATURES} spatial features per lobe")
    logger.info(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    extract_features()