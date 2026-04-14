"""Atlas and neuroanatomical mapping constants for Neuro-CXG."""

from src.core.paths import DATA_METADATA

# --- ATLAS VALIDATION CONSTANTS ---
AAL3_VALID_ROI_RANGE = (164, 170)  # AAL3v1: some variants have 2 unused ROIs
ROI_CENTROIDS_PATH = DATA_METADATA / "roi_centroids.json"  # ROI 3D centroids for visualization


# --- ANATOMICAL MAPPING (12-Region Neuroanatomical Subdivision) ---
# Note: AAL ROI IDs are 1-indexed; convert to 0-indexed for array access.
# Updated January 2026: Expanded from 5 lobes to 12 functionally-distinct brain regions

def _idx(ids):
    return [i - 1 for i in ids]


LOBE_MAPPING = {
    0: _idx([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),  # Frontal_Superior (Left+Right)
    1: _idx([21, 22, 25, 26, 27, 28]),  # Frontal_Orbital (Left+Right)
    2: _idx([17, 18, 19, 20, 23, 24]),  # Motor_Premotor (Central, includes 23-24)
    3: _idx([29, 30, 31, 32]),  # Insula (Left+Right, 29-30 missing previously)
    4: _idx([33, 34, 35, 36, 37, 38, 151, 152, 153, 154, 155, 156]),  # Cingulate + ACC subdivisions
    5: _idx([39, 40, 41, 42, 91, 92, 93, 94]),  # Limbic (Hippocampus, Amygdala)
    6: _idx([43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56]),  # Occipital
    7: _idx([57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70]),  # Parietal
    8: _idx([79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90]),  # Temporal
    9: _idx([71, 72, 73, 74, 75, 76, 77, 78, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150]),  # Subcortical
    10: _idx(list(range(95, 121)) + list(range(157, 167))),  # Cerebellum (Vermis + Hemispheres)
    11: _idx([167, 168, 169, 170]),  # Brainstem (Midbrain, Pons, Medulla)
}

LOBE_NAMES = {
    0: "Frontal_Superior",
    1: "Frontal_Orbital",
    2: "Motor_Premotor",
    3: "Insula",
    4: "Cingulate",
    5: "Limbic",
    6: "Occipital",
    7: "Parietal",
    8: "Temporal",
    9: "Subcortical",
    10: "Cerebellum",
    11: "Brainstem",
}
NUM_LOBES = 12  # Updated from 5 to 12 regions
SPATIAL_MIN_REQUIRED_REGIONS = 9  # relaxed gate; final golden filter enforces complete 12-region subjects

# --- FOUR-NETWORK HIERARCHY FOR ANATOMICAL HIERARCHICAL POOLING (Task 3 — DD-011) ---
# Maps each of the 12 lobe indices to one of 4 functional networks.
# Based on fMRI functional connectivity literature (Power et al. 2011, Yeo et al. 2011).
# ASD literature: DMN and Salience are the primary affected networks.
LOBE_TO_NETWORK = {
    0: 0,   # Frontal_Superior  → DMN (default mode)
    1: 0,   # Frontal_Orbital   → DMN
    4: 0,   # Cingulate         → DMN (ACC is a DMN hub)
    7: 0,   # Parietal          → DMN (angular/precuneus)
    8: 0,   # Temporal          → DMN (inferior temporal)
    2: 1,   # Motor_Premotor    → Salience (SMA/motor cingulate)
    3: 1,   # Insula            → Salience (anterior insula hub)
    9: 1,   # Subcortical       → Salience (thalamus/striatum)
    6: 2,   # Occipital         → Visual/Cerebellar
    10: 2,  # Cerebellum        → Visual/Cerebellar
    5: 3,   # Limbic            → Limbic (hippocampus/amygdala)
    11: 3,  # Brainstem         → Limbic (brainstem modulation)
}

NUM_NETWORKS = 4

NETWORK_NAMES = {
    0: "DMN",
    1: "Salience",
    2: "Visual_Cerebellar",
    3: "Limbic",
}

# Reverse mapping: network index → list of lobe indices
NETWORK_TO_LOBES = {
    0: [0, 1, 4, 7, 8],   # DMN
    1: [2, 3, 9],          # Salience
    2: [6, 10],            # Visual/Cerebellar
    3: [5, 11],            # Limbic
}
