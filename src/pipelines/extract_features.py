import os
import pandas as pd
from ultralytics import YOLO
import numpy as np
from tqdm import tqdm

MODEL_PATH = "results/ROI_Detection_v20/weights/best.pt"
SPLIT_ROOT = "data/processed" 
OUTPUT_PATH = "data/metadata/roi_features_final.csv"
MANIFEST_PATH = "data/metadata/manifest_v1.csv"
CONF_THRESHOLD = 0.30 

def extract():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    model = YOLO(MODEL_PATH)
    all_detections = []
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        img_dir = os.path.join(SPLIT_ROOT, split, "images")
        if not os.path.exists(img_dir):
            continue
            
        print(f"🚀 Extracting Node Features from {split} split...")
        images = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        
        # Using stream=True for memory efficiency with large ABIDE datasets
        results = model(img_dir, stream=True, conf=CONF_THRESHOLD)
        
        for res in tqdm(results, total=len(images)):
            file_name = os.path.basename(res.path).replace(".png", "")
            
            # Use the consistent rsplit logic
            try:
                subject_id, z_str = file_name.rsplit('_z', 1)
                z_coord = int(z_str)
            except: 
                continue 
                
            if len(res.boxes) == 0:
                continue 

            for box in res.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                # xywhn provides normalized coordinates (0-1), perfect for GNN nodes
                c = box.xywhn[0].tolist() 
                
                all_detections.append({
                    'subject_id': subject_id,
                    'roi_class': cls,
                    'conf': conf,
                    'x': c[0], 'y': c[1], 'z': z_coord,
                    'w': c[2], 'h': c[3],
                    'area': c[2] * c[3]
                })

    if not all_detections:
        print("❌ No ROIs detected. Check image quality or model weights.")
        return

    df = pd.DataFrame(all_detections)

    # 1. Aggregate Slices into a 3D Representation for each Subject
    # We take the mean position across the 5 slices to define the node's 3D coordinate
    df_pivot = df.pivot_table(
        index='subject_id', 
        columns='roi_class', 
        values=['x', 'y', 'z', 'area', 'conf'],
        aggfunc='mean'
    )
    
    # Flatten multi-index columns: e.g., ('x', 0) -> 'x_frontal'
    node_names = {0: 'frontal', 1: 'temporal', 2: 'parietal', 3: 'occipital'}
    df_pivot.columns = [f"{col[0]}_{node_names.get(int(col[1]), col[1])}" for col in df_pivot.columns]
    
    # 2. Filtering for Causal Integrity
    # A causal graph requires all nodes to exist. Let's count present nodes per subject.
    area_cols = [c for c in df_pivot.columns if 'area' in c]
    df_pivot['node_count'] = df_pivot[area_cols].notna().sum(axis=1)
    
    print(f"Total subjects with at least one detection: {len(df_pivot)}")
    # We only keep subjects where all 4 lobes were successfully identified
    df_pivot = df_pivot[df_pivot['node_count'] == 4]
    print(f"Subjects with complete 4-node graphs: {len(df_pivot)}")

    # 3. Metadata Merge
    if not os.path.exists(MANIFEST_PATH):
        print(f"Error: Manifest not found at {MANIFEST_PATH}")
        return

    manifest = pd.read_csv(MANIFEST_PATH)
    # Ensure ID match
    manifest['ID_MATCH'] = manifest['ID_MATCH'].astype(str)
    
    final_df = pd.merge(df_pivot, manifest, left_index=True, right_on='ID_MATCH')

    # 4. Save Features
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Success! Extracted Graph Node Features to: {OUTPUT_PATH}")

if __name__ == "__main__":
    extract()