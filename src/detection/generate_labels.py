import logging
import os
from collections import defaultdict

import nibabel as nib
import numpy as np
from PIL import Image
from tqdm import tqdm

# Setup paths from config
from src.core.config import (
    ALFF_SLICE_PERCENTILES,
    ATLAS_PATH,
    DATA_FINAL,
    LOBE_MAPPING,
    NUM_LOBES,
    YOLO_IMGSZ,
)

# Setup logging
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
IMG_SIZE = (YOLO_IMGSZ, YOLO_IMGSZ)

def calculate_yolo_bbox(mask, size):
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    y_min, y_max = rows.min(), rows.max()
    x_min, x_max = cols.min(), cols.max()
    w = x_max - x_min
    h = y_max - y_min
    x_center = (x_min + w / 2.0) / size[0]
    y_center = (y_min + h / 2.0) / size[1]
    return f"{x_center:.6f} {y_center:.6f} {(w / size[0]):.6f} {(h / size[1]):.6f}"

def generate_atlas_labels_for_percentiles():
    """
    Generate atlas labels for specific z-percentiles that match ALFF slice export.
    Returns labels indexed by percentile index (0-6) to handle variable subject z-dims.
    """
    if not ATLAS_PATH.exists():
        logger.error(f"Atlas not found at {ATLAS_PATH}")
        return {}

    atlas_img = nib.as_closest_canonical(nib.load(str(ATLAS_PATH)))
    data = atlas_img.get_fdata()
    atlas_z_dim = data.shape[2]

    # Use percentiles from config (single source of truth with abide_download.py)
    atlas_labels = {}

    logger.info(f"Pre-calculating atlas bounding boxes for {len(ALFF_SLICE_PERCENTILES)} percentile slices (atlas z_dim={atlas_z_dim})...")

    for idx, p in enumerate(ALFF_SLICE_PERCENTILES):
        z = int(atlas_z_dim * p)  # Atlas z-index for this percentile

        if z >= atlas_z_dim:
            logger.warning(f"Percentile {p} maps to z={z} which exceeds atlas z_dim={atlas_z_dim}")
            continue

        bboxes = []
        slice_data = data[:, :, z]

        for class_id in range(NUM_LOBES):
            aal_ids = [roi_id + 1 for roi_id in LOBE_MAPPING[class_id]]
            mask = np.isin(slice_data, aal_ids)
            if not np.any(mask):
                continue
            processed_mask = np.rot90(mask)
            mask_img = Image.fromarray(processed_mask).resize(IMG_SIZE, Image.NEAREST)
            bbox = calculate_yolo_bbox(np.array(mask_img), IMG_SIZE)
            if bbox:
                bboxes.append(f"{class_id} {bbox}")

        if bboxes:
            # Key by percentile index (0-6) so it works for all subject z-dimensions
            atlas_labels[idx] = bboxes
            logger.debug(f"Percentile {p} (idx={idx}, z={z}): {len(bboxes)} boxes")

    return atlas_labels

def main():
    if not DATA_FINAL.exists():
        logger.error(f"Error: {DATA_FINAL} not found. Run split.py first!")
        return

    # Generate atlas annotations indexed by percentile (0-6)
    atlas_anno = generate_atlas_labels_for_percentiles()

    if not atlas_anno:
        logger.error("Failed to generate atlas annotations. Check atlas file exists.")
        return

    logger.info(f"Generated annotations for {len(atlas_anno)} percentile slices")

    splits = ["train", "val", "test"]
    total_images = 0
    total_labels = 0

    for split in splits:
        img_dir = DATA_FINAL / split / "images"
        lbl_dir = DATA_FINAL / split / "labels"

        if not img_dir.exists():
            logger.warning(f"Split folder {split} is missing images. Skipping.")
            continue

        lbl_dir.mkdir(parents=True, exist_ok=True)
        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])

        logger.info(f"Annotating {split} split ({len(img_files)} images)...")

        # Group images by subject to map z-indices to percentiles

        subject_slices = defaultdict(list)

        for img_name in img_files:
            try:
                # Extract subject_id and z-index from filename like "Caltech_0051456_z12.png"
                parts = img_name.rsplit("_z", 1)
                if len(parts) != 2:
                    continue
                subject_id = parts[0]
                z_idx = int(parts[1].split(".")[0])
                subject_slices[subject_id].append((z_idx, img_name))
            except Exception as e:
                logger.warning(f"Failed to parse {img_name}: {e}")
                continue

        # Generate labels for each subject
        for subject_id, slice_list in tqdm(subject_slices.items(), desc=f"Subjects in {split}"):
            # Sort by z-index to map to percentiles
            slice_list_sorted = sorted(slice_list, key=lambda x: x[0])

            if len(slice_list_sorted) != len(ALFF_SLICE_PERCENTILES):
                logger.warning(f"{subject_id}: Expected {len(ALFF_SLICE_PERCENTILES)} slices, got {len(slice_list_sorted)}")
                continue

            # Map each slice to its corresponding percentile index (0-6)
            for percentile_idx, (_, img_name) in enumerate(slice_list_sorted):
                if percentile_idx in atlas_anno:
                    label_path = lbl_dir / img_name.replace(".png", ".txt")
                    with open(label_path, "w") as f:
                        f.write("\n".join(atlas_anno[percentile_idx]))
                    total_labels += 1
                else:
                    logger.warning(f"No annotations for percentile index {percentile_idx}")

            total_images += len(slice_list_sorted)

    logger.info(f"Annotation complete. Created {total_labels} labels for {total_images} images across all splits.")
    logger.info(f"Coverage: {100 * total_labels / total_images:.1f}%" if total_images > 0 else "No images found")

if __name__ == "__main__":
    main()
