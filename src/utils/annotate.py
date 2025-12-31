import os
import numpy as np
import nibabel as nib
from PIL import Image
from tqdm import tqdm

Z_SLICES = [20, 30, 40, 50, 60] # 5 slices
IMAGE_DIR = "./data/images" 
OUTPUT_LBL_DIR = "./data/labels" 
AAL_PATH = "./data/atlases/AAL3v1.nii"
IMG_SIZE = (640, 640)

# ROI Mapping for Brain Lobes (Nodes in your Causal Graph)
ROI_MAP = {
    0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], # Frontal
    1: [79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90],                                               # Temporal
    2: [57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70],                                       # Parietal
    3: [43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]                                                # Occipital
}

def get_yolo_bbox(mask, target_size):
    """Calculates normalized YOLO format bbox from a binary mask."""
    rows, cols = np.where(mask)
    if len(rows) == 0: return None
    
    y_min, y_max = rows.min(), rows.max()
    x_min, x_max = cols.min(), cols.max()
    
    W, H = target_size
    
    # Calculate box width and height
    w = (x_max - x_min)
    h = (y_max - y_min)
    
    # Calculate normalized center and dimensions
    x_center = (x_min + w / 2.0) / W
    y_center = (y_min + h / 2.0) / H
    norm_w = w / W
    norm_h = h / H
    
    return f"{x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"

def process_atlas():
    """Maps atlas ROIs to YOLO coordinates for each Z-slice percentage."""
    if not os.path.exists(AAL_PATH):
        raise FileNotFoundError(f"Atlas not found at {AAL_PATH}")
    
    raw_atlas = nib.load(AAL_PATH)
    atlas_img = nib.as_closest_canonical(raw_atlas)
    atlas_data = atlas_img.get_fdata()
    depth = atlas_data.shape[2] 

    # Map percentages to actual Z-indices in the Atlas
    target_indices = Z_SLICES
    slice_labels = {}

    print(f"Generating Atlas Nodes for Z-indices: {target_indices}")
    for z in target_indices:
        slice_data = atlas_data[:, :, z]
        bboxes = []
        
        for class_id, aal_indices in ROI_MAP.items():
            mask = np.isin(slice_data, aal_indices).astype(np.uint8)
            if not np.any(mask): continue
            
            # Match image preprocessing: Rotate 90 -> Resize
            mask_img = Image.fromarray(mask * 255)
            mask_img = mask_img.transpose(Image.ROTATE_90)
            mask_img = mask_img.resize(IMG_SIZE, resample=Image.Resampling.NEAREST)
            
            final_mask = np.array(mask_img) > 0
            bbox = get_yolo_bbox(final_mask, IMG_SIZE)
            if bbox:
                bboxes.append(f"{class_id} {bbox}")
        
        slice_labels[z] = bboxes
    return slice_labels

if __name__ == "__main__":
    os.makedirs(OUTPUT_LBL_DIR, exist_ok=True)
    
    # 1. Generate standard coordinates from AAL3 Atlas
    atlas_annotations = process_atlas()
    
    # 2. Assign these coordinates to every subject
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.png')]
    print(f"Applying Atlas-based labels to {len(image_files)} subject images...")

    for img_file in tqdm(image_files):
        try:
            # Extract Z-index from filename (e.g., 'sub101_z45.png' -> 45)
            z_val = int(img_file.rsplit('_z', 1)[1].split('.')[0])
            
            # Match the subject's Z-slice to the corresponding Atlas Z-slice
            if z_val in atlas_annotations:
                lbl_name = img_file.replace('.png', '.txt')
                with open(os.path.join(OUTPUT_LBL_DIR, lbl_name), 'w') as f:
                    f.write("\n".join(atlas_annotations[z_val]))
        except Exception as e:
            continue 

    print(f"\nAnnotation Complete. Labels saved to: {OUTPUT_LBL_DIR}")