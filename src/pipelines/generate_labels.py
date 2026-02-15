import logging
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image
from tqdm import tqdm

# Setup paths from config
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import DATA_FINAL, DATA_ATLASES, LOBE_MAPPING, NUM_LOBES

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
ATLAS_PATH = DATA_ATLASES / "AAL3v1.nii"
IMG_SIZE = (640, 640)


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


def generate_atlas_labels():
    atlas_img = nib.as_closest_canonical(nib.load(str(ATLAS_PATH)))
    data = atlas_img.get_fdata()
    z_dim = data.shape[2]
    atlas_labels = {}

    logger.info("Pre-calculating atlas bounding boxes...")
    for z in range(z_dim):
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
            atlas_labels[z] = bboxes
    return atlas_labels


def main():
    if not DATA_FINAL.exists():
        logger.error(f"Error: {DATA_FINAL} not found. Run split.py first!")
        return

    atlas_anno = generate_atlas_labels()
    splits = ["train", "val", "test"]

    for split in splits:
        img_dir = DATA_FINAL / split / "images"
        lbl_dir = DATA_FINAL / split / "labels"

        if not img_dir.exists():
            logger.warning(f"Split folder {split} is missing images. Skipping.")
            continue

        lbl_dir.mkdir(parents=True, exist_ok=True)
        img_files = [f for f in os.listdir(img_dir) if f.endswith(".png")]

        logger.info(f"Annotating {split} split ({len(img_files)} images)...")
        for img_name in tqdm(img_files):
            try:
                z_idx = int(img_name.split("_z")[1].split(".")[0])
                if z_idx in atlas_anno:
                    with open(lbl_dir / img_name.replace(".png", ".txt"), "w") as f:
                        f.write("\n".join(atlas_anno[z_idx]))
            except Exception:
                continue

    logger.info("Annotation complete. Labels synced with split images.")


if __name__ == "__main__":
    main()
