"""
Extract spatial features from atlas ROI centroids.

This is a lightweight alternative to YOLO-based detection that uses precomputed
atlas centroids from AAL3v1. It provides 4 anatomical spatial features per lobe:
    - x, y, z_depth: Centroid coordinates (mm in MNI space)
    - size: Relative size based on voxel count

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
    ATLAS_PATH,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

CENTROIDS_PATH = DATA_METADATA / "roi_centroids.json"


def _compute_and_save_centroids_from_atlas() -> dict:
    """Compute ROI centroids from atlas and persist roi_centroids.json."""
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError(
            "nibabel is required to compute atlas centroids when roi_centroids.json is missing"
        ) from exc

    if not ATLAS_PATH.exists():
        raise FileNotFoundError(
            f"Atlas not found at {ATLAS_PATH}. Cannot auto-generate {CENTROIDS_PATH}."
        )

    atlas_img = nib.load(str(ATLAS_PATH))
    atlas_data = atlas_img.get_fdata()
    affine = atlas_img.affine

    labels = np.unique(atlas_data)
    labels = labels[labels > 0]

    centroids_list = []
    for label in labels:
        roi_id = int(label)
        voxels = np.argwhere(atlas_data == label)
        if voxels.size == 0:
            continue
        mean_vox = voxels.mean(axis=0)
        mni = affine @ np.append(mean_vox, 1.0)
        centroids_list.append(
            {
                "roi_id": roi_id,
                "x": float(mni[0]),
                "y": float(mni[1]),
                "z": float(mni[2]),
            }
        )

    if not centroids_list:
        raise RuntimeError(f"No ROI centroids computed from atlas at {ATLAS_PATH}")

    CENTROIDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CENTROIDS_PATH, "w") as f:
        json.dump(centroids_list, f)

    logger.info("Generated %d ROI centroids at %s", len(centroids_list), CENTROIDS_PATH)
    return {c["roi_id"]: c for c in centroids_list}


def load_centroids():
    """Load precomputed atlas ROI centroids."""
    if not CENTROIDS_PATH.exists():
        logger.warning(
            "Centroids not found at %s. Attempting auto-generation from atlas...",
            CENTROIDS_PATH,
        )
        return _compute_and_save_centroids_from_atlas()
    
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
    
    Returns: [x, y, z_depth, size]
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
        return [0.0, 0.0, 0.0, 0.0]
    
    lobe_centroids = np.array(lobe_centroids)
    lobe_sizes = np.array(lobe_sizes)
    
    # Compute lobe-level statistics
    mean_x = float(np.mean(lobe_centroids[:, 0]))
    mean_y = float(np.mean(lobe_centroids[:, 1]))
    mean_z = float(np.mean(lobe_centroids[:, 2]))
    mean_size = float(np.mean(lobe_sizes))
    
    return [mean_x, mean_y, mean_z, mean_size]


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
    
    # Create feature names matching graph_factory expectations:
    # subject_id + {LOBE_NAME}_{feature}
    columns = ["subject_id"]
    spatial_names = ["x", "y", "z_depth", "size"]
    
    for lobe_id in range(NUM_LOBES):
        lobe_name = LOBE_NAMES[lobe_id]
        for feat_name in spatial_names:
            columns.append(f"{lobe_name}_{feat_name}")
    
    # Save to CSV
    df = pd.DataFrame(all_features, columns=columns)
    df.to_csv(NODE_FEATURES_3D, index=False)
    
    logger.info(f"Saved spatial features to {NODE_FEATURES_3D}")
    logger.info(f"Features shape: {df.shape} ({len(df)} subjects × {len(df.columns)-1} spatial features)")


if __name__ == "__main__":
    extract_spatial()
