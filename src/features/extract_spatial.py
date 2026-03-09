import logging
import sys
from pathlib import Path

import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm

# Setup paths and config
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import (
    DATA_FINAL,
    MASTER_MANIFEST,
    LOBE_NAMES,
    YOLO_WEIGHTS_PATH,
    NODE_FEATURES_3D,
    NUM_LOBES,
    NUM_SPATIAL_FEATURES,
    SPATIAL_MIN_REQUIRED_REGIONS,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# --- CONFIG ---
MODEL_PATH = YOLO_WEIGHTS_PATH
SPLIT_ROOT = DATA_FINAL
MANIFEST_PATH = MASTER_MANIFEST
OUTPUT_PATH = NODE_FEATURES_3D


def extract_spatial():
    if not MODEL_PATH.exists():
        logger.error(f"Model weights not found at {MODEL_PATH}. Rerun YOLO training first.")
        return

    model = YOLO(MODEL_PATH)
    all_detections = []

    for split in ["train", "val", "test"]:
        img_dir = SPLIT_ROOT / split / "images"
        if not img_dir.exists():
            continue

        try:
            logger.info(f"Processing {split} set...")
            # Use config confidence threshold (not hardcoded 0.35)
            from src.core.config import YOLO_CONF_THRESHOLD
            results = model(str(img_dir), stream=True, conf=YOLO_CONF_THRESHOLD, verbose=False)

            for res in tqdm(results, desc=f"Inference {split}", disable=not sys.stdout.isatty()):
                try:
                    file_name = Path(res.path).stem
                    try:
                        subject_id, z_str = file_name.rsplit("_z", 1)
                        z_coord = int(z_str)
                    except ValueError:
                        logger.debug(f"Skipped malformed filename: {file_name}")
                        continue

                    for box in res.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        c = box.xywhn[0].cpu().numpy()

                        all_detections.append(
                            {
                                "subject_id": subject_id,
                                "roi_class": cls,
                                "x": c[0],
                                "y": c[1],
                                "z": z_coord,
                                "w": c[2],
                                "h": c[3],
                                "conf": conf,
                            }
                        )
                except Exception as e:
                    logger.warning(f"Error processing result: {e}")
                    continue
        except Exception as e:
            logger.error(f"Error during inference on {split} set: {e}")
            raise

    raw_df = pd.DataFrame(all_detections)
    
    if raw_df.empty:
        logger.error("No detections found! Check if YOLO model is working correctly.")
        return

    logger.info(f"Total detections: {len(raw_df)}")
    logger.info(f"Unique subjects: {raw_df['subject_id'].nunique()}")
    logger.info(f"Unique ROI classes detected: {sorted(raw_df['roi_class'].unique().tolist())}")
    logger.info("Aggregating 2D detections into 3D lobe nodes...")

    processed_subjects = []
    # Allow partial detections (min 9/12 regions) to maximize dataset size
    # Subjects with fewer regions are filtered here
    MIN_REQUIRED_REGIONS = SPATIAL_MIN_REQUIRED_REGIONS
    
    filtered_count = 0
    partial_count = 0
    subject_ids = raw_df["subject_id"].unique()

    for sub_id in tqdm(
        subject_ids, desc="Building Subject Nodes",
        miniters=max(1, len(subject_ids) // 20), mininterval=10.0
    ):
        sub_group = raw_df[raw_df["subject_id"] == sub_id]

        # RELAXED FILTER: Require at least 9/12 regions detected
        # This allows subjects with up to 3 missing regions to proceed to feature extraction
        detected_regions = sub_group["roi_class"].nunique()
        if detected_regions < MIN_REQUIRED_REGIONS:
            filtered_count += 1
            logger.debug(f"Subject {sub_id}: only {detected_regions}/{NUM_LOBES} regions detected (REJECTED)")
            continue

        # Track whether subject has complete detection (all 12 regions)
        is_complete = detected_regions == NUM_LOBES
        if not is_complete:
            partial_count += 1
            logger.debug(f"Subject {sub_id}: {detected_regions}/{NUM_LOBES} regions detected (PARTIAL - kept for ranking)")

        subject_row = {"subject_id": sub_id, "spatial_complete": is_complete}

        # Aggregate spatial features for each of the 12 detected regions
        for lobe_id in range(NUM_LOBES):
            lobe_name = LOBE_NAMES[lobe_id]
            lobe_data = sub_group[sub_group["roi_class"] == lobe_id]

            # LENIENT: If region not detected, fill with zeros (YOLO sometimes misses regions like Frontal_Orbital)
            if len(lobe_data) == 0:
                logger.debug(f"Subject {sub_id}: Region {lobe_name} not detected, filling with zeros")
                subject_row[f"{lobe_name}_x"] = 0.0
                subject_row[f"{lobe_name}_y"] = 0.0
                subject_row[f"{lobe_name}_z_depth"] = 0.0
                subject_row[f"{lobe_name}_size"] = 0.0
                subject_row[f"{lobe_name}_conf_std"] = 0.0
                subject_row[f"{lobe_name}_detection_count"] = 0.0
            else:
                subject_row[f"{lobe_name}_x"] = lobe_data["x"].mean()
                subject_row[f"{lobe_name}_y"] = lobe_data["y"].mean()
                subject_row[f"{lobe_name}_z_depth"] = lobe_data["z"].mean()
                subject_row[f"{lobe_name}_size"] = lobe_data["w"].mean() * lobe_data["h"].mean()
                subject_row[f"{lobe_name}_conf_std"] = (
                    lobe_data["conf"].std() if len(lobe_data) > 1 else 0.0
                )
                subject_row[f"{lobe_name}_detection_count"] = len(lobe_data)
        
        # Append if region aggregation succeeded (processing didn't encounter corruption)
        if subject_row is not None:
            processed_subjects.append(subject_row)

    logger.info(f"Subjects processed: {len(subject_ids)}")
    logger.info(f"Subjects filtered (< {MIN_REQUIRED_REGIONS} regions): {filtered_count}")
    logger.info(f"Subjects with partial detection (9-11 regions): {partial_count}")
    logger.info(f"Subjects with complete detection (all {NUM_LOBES} regions): {len(processed_subjects) - partial_count}")
    logger.info(f"Subjects kept (>= {MIN_REQUIRED_REGIONS} regions): {len(processed_subjects)}")
    logger.warning(f"RELAXED FILTER: Subjects with >= {MIN_REQUIRED_REGIONS} regions kept. Final filtering to complete {NUM_LOBES}-region subjects happens in variance ranking stage.")
    
    final_df = pd.DataFrame(processed_subjects)
    
    # Handle case where no subjects passed the filter
    if final_df.empty:
        logger.error(
            f"No subjects detected with >= {MIN_REQUIRED_REGIONS}/{NUM_LOBES} brain regions! "
            f"Check YOLO model quality or detection confidence threshold."
        )
        # Create empty output with proper schema
        manifest = pd.read_csv(MANIFEST_PATH)
        manifest["subject_id"] = manifest["subject_id"].astype(str)
        
        # Create minimal schema for empty output (includes spatial_complete tracking column)
        empty_cols = ["subject_id", "spatial_complete"] + [
            f"{lobe}_{feat}" 
            for lobe in LOBE_NAMES.values() 
            for feat in ["x", "y", "z_depth", "size", "conf_std", "detection_count"]
        ]
        final_df = pd.DataFrame(columns=empty_cols)
        output_df = pd.merge(final_df, manifest, on="subject_id", how="inner")
    else:
        manifest = pd.read_csv(MANIFEST_PATH)
        manifest["subject_id"] = manifest["subject_id"].astype(str)
        output_df = pd.merge(final_df, manifest, on="subject_id", how="inner")

    lobe_cols = [
        col for col in output_df.columns if any(col.startswith(f"{lobe}_") for lobe in LOBE_NAMES.values())
    ]

    expected_cols = NUM_LOBES * NUM_SPATIAL_FEATURES
    if len(lobe_cols) != expected_cols:
        logger.warning(
            f"Feature count mismatch! Expected {expected_cols} spatial feature columns, "
            f"got {len(lobe_cols)}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)

    logger.info(f"Success: {len(output_df)} subjects with {NUM_LOBES}-node architecture")
    logger.info(f"{NUM_SPATIAL_FEATURES} spatial features per lobe")
    logger.info(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    extract_spatial()
