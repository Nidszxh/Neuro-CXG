import os, numpy as np, nibabel as nib, pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# --- CONFIGURATION ---
PROJECT_ROOT = Path("./data")
DATASET_ROOT = PROJECT_ROOT / "final" 
ATLAS_PATH   = PROJECT_ROOT / "atlases" / "AAL3v1.nii"
IMG_SIZE     = (640, 640)

ROI_MAP = {
    0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], # Frontal
    1: [79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90],                                               # Temporal
    2: [57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70],                                       # Parietal
    3: [43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],                                               # Occipital
    4: [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 71, 72, 73, 74, 75, 76, 77, 78, 91, 92, 93, 94] # Subcortical/Limbic
}

def calculate_yolo_bbox(mask, size):
    rows, cols = np.where(mask)
    if len(rows) == 0: return None
    y_min, y_max = rows.min(), rows.max()
    x_min, x_max = cols.min(), cols.max()
    w = (x_max - x_min)
    h = (y_max - y_min)
    x_center = (x_min + w / 2.0) / size[0]
    y_center = (y_min + h / 2.0) / size[1]
    return f"{x_center:.6f} {y_center:.6f} {(w/size[0]):.6f} {(h/size[1]):.6f}"

def generate_atlas_labels():
    atlas_img = nib.as_closest_canonical(nib.load(str(ATLAS_PATH)))
    data = atlas_img.get_fdata()
    z_dim = data.shape[2]
    atlas_labels = {}

    print("🧩 Pre-calculating Atlas Bounding Boxes...")
    for z in range(z_dim):
        bboxes = []
        slice_data = data[:, :, z]
        for class_id, aal_ids in ROI_MAP.items():
            mask = np.isin(slice_data, aal_ids)
            if not np.any(mask): continue
            processed_mask = np.rot90(mask)
            mask_img = Image.fromarray(processed_mask).resize(IMG_SIZE, Image.NEAREST)
            bbox = calculate_yolo_bbox(np.array(mask_img), IMG_SIZE)
            if bbox: bboxes.append(f"{class_id} {bbox}")
        if bboxes: atlas_labels[z] = bboxes
    return atlas_labels

def main():
    if not DATASET_ROOT.exists():
        print(f"❌ Error: {DATASET_ROOT} not found. Run split.py first!")
        return

    atlas_anno = generate_atlas_labels()
    splits = ['train', 'val', 'test']
    
    for split in splits:
        img_dir = DATASET_ROOT / split / "images"
        lbl_dir = DATASET_ROOT / split / "labels"
        
        if not img_dir.exists():
            print(f"⚠️ Warning: Split folder {split} is missing images. Skipping.")
            continue
            
        lbl_dir.mkdir(parents=True, exist_ok=True)
        img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        
        print(f"🏷️  Annotating {split} split ({len(img_files)} images)...")
        for img_name in tqdm(img_files):
            try:
                # Extract Z from filename (e.g., sub-001_z45.png)
                z_idx = int(img_name.split('_z')[1].split('.')[0])
                if z_idx in atlas_anno:
                    with open(lbl_dir / img_name.replace('.png', '.txt'), 'w') as f:
                        f.write("\n".join(atlas_anno[z_idx]))
            except Exception as e:
                continue

    print(f"✅ Annotation Complete. Labels synced with split images.")

if __name__ == "__main__":
    main()