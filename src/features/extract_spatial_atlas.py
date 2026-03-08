"""
Extract spatial features from atlas ROI centroids.

This is a lightweight alternative to YOLO-based detection that uses precomputed
atlas centroids from AAL3v1. It provides 6 spatial features per lobe:
  - x, y, z_depth: Centroid coordinates (mm in MNI space)
  - size: Relative size based on voxel count
  - conf_std: Confidence std (fixed to 0.0 for atlas-based)
  - detection_count: Number of detections per lobe (fixed to 1.0 for atlas)

Pipeline: Extract centroids → Group by lobe → Compute lobe-level statistics
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Setup paths and config
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import (
    MASTER_MANIFEST,
    NODE_FEATURES_3D,
    LOBE_MAPPING,
    LOBE_NAMES,
    NUM_LOBES,
    NUM_SPATIAL_FEATURES,
    DATA_METADATA,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

CENTROIDS_PATH = DATA_METADATA / "roi_centroids.json"


def load_centroids():
    """Load precomputed atlas ROI centroids."""
    if not CENTROIDS_PATH.exists():
        raise FileNotFoundError(f"Centroids not found at {CENTROIDS_PATH}")
    
    with open(CENTROIDS_PATH) as f:
        centroids_list = json.load(f)
    
    # Convert to dict keyed by roi_id (1-indexed)
    centroids = {c["roi_id"]: c for c in centroids_list}
    logger.info(f"Loaded {len(centroids)} ROI centroids")
    return centroids


def compute_roi_sizes():
    """
    Compute relative size of each ROI based on voxel count.
    Derived from AAL3v1 atlas dimensions.
    """
    try:
        import nibabel as nib
        atlas_path = DATA_METADATA.parent / "raw" / "atlases" / "AAL3v1.nii"
        if atlas_path.exists():
            atlas_img = nib.load(str(atlas_path))
            atlas_data = atlas_img.get_fdata()
            
            roi_sizes = {}
            for roi_id in range(1, 171):
                count = np.sum(atlas_data == roi_id)
                roi_sizes[roi_id] = float(count)
            
            # Normalize by max size
            max_size = max(roi_sizes.values())
            roi_sizes = {roi_id: size / max_size for roi_id, size in roi_sizes.items()}
            return roi_sizes
    except Exception as e:
        logger.warning(f"Failed to compute ROI sizes from atlas: {e}")
    
    # Fallback: uniform sizes
    return {roi_id: 1.0 for roi_id in range(1, 171)}


def extract_lobe_features(lobe_id, roi_indices, centroids, roi_sizes):
    """
    Aggregate spatial features for one lobe from its constituent ROIs.
    
    Returns: [x, y, z_depth, size, conf_std, detection_count]
    """
    # Get centroids for ROIs in this lobe (1-indexed)
    lobe_centroids = []
    lobe_sizes = []
    
    for roi_idx_0 in roi_indices:
        roi_id = roi_idx_0 + 1  # Convert 0-indexed to 1-indexed
        if roi_id in centroids:
            c = centroids[roi_id]
            lobe_centroids.append([c['x'], c['y'], c['z']])
            lobe_sizes.append(roi_sizes.get(roi_id, 1.0))
    
    if not lobe_centroids:
        # Fallback for missing lobe (shouldn't happen with AAL)
        logger.warning(f"Lobe {lobe_id} ({LOBE_NAMES[lobe_id]}) has no centroids")
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    lobe_centroids = np.array(lobe_centroids)
    lobe_sizes = np.array(lobe_sizes)
    
    # Compute lobe-level statistics
    mean_x = float(np.mean(lobe_centroids[:, 0]))
    mean_y = float(np.mean(lobe_centroids[:, 1]))
    mean_z = float(np.mean(lobe_centroids[:, 2]))
    mean_size = float(np.mean(lobe_sizes))
    
    # conf_std: 0.0 for atlas (no detection variance)
    conf_std = 0.0
    
    # detection_count: 1.0 for atlas (1 unique lobe definition)
    detection_count = 1.0
    
    return [mean_x, mean_y, mean_z, mean_size, conf_std, detection_count]


def extract_spatial():
    """Main extraction: compute spatial features for all subjects."""
    
    logger.info("Loading atlas centroids and ROI sizes...")
    centroids = load_centroids()
    roi_sizes = compute_roi_sizes()
    
    # Load manifest to get subject list
    if not MASTER_MANIFEST.exists():
        logger.error("Master manifest missing. Run manifestor.py first.")
        return
    
    manifest = pd.read_csv(MASTER_MANIFEST)
    logger.info(f"Extracting spatial features for {len(manifest)} subjects...")
    
    # Extract features for each subject
    all_features = []
    
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Subjects"):
        sub_id = str(row["subject_id"])
        
        # Compute spatial features for each lobe
        subject_row = [sub_id]
        
        for lobe_id in range(NUM_LOBES):
            roi_indices = LOBE_MAPPING[lobe_id]
            lobe_feats = extract_lobe_features(lobe_id, roi_indices, centroids, roi_sizes)
            subject_row.extend(lobe_feats)
        
        all_features.append(subject_row)
    
    if not all_features:
        logger.error("No features extracted!")
        return
    
    # Create feature names: subject_id + roi1_x, roi1_y, ..., roi12_z, etc.
    columns = ["subject_id"]
    spatial_names = ["x", "y", "z_depth", "size", "conf_std", "detection_count"]
    
    for lobe_id in range(NUM_LOBES):
        for feat_name in spatial_names:
            columns.append(f"roi{lobe_id+1}_{feat_name}")
    
    # Save to CSV
    df = pd.DataFrame(all_features, columns=columns)
    df.to_csv(NODE_FEATURES_3D, index=False)
    
    logger.info(f"Saved spatial features to {NODE_FEATURES_3D}")
    logger.info(f"Features shape: {df.shape} ({len(df)} subjects × {len(df.columns)-1} spatial features)")


if __name__ == "__main__":
    extract_spatial()
