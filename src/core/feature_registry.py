"""Feature and acquisition constants for Neuro-CXG."""

# --- fMRI ACQUISITION DEFAULTS ---
# Default TR (seconds) when missing in manifest metadata.
DEFAULT_TR = 2.0
NYQUIST_EPS = 1e-6
UNRELIABLE_FREQ_BANDS_AT_NYQUIST = ("gamma",)
EXCLUDE_NYQUIST_BANDS = True

# fMRI frequency bands (single source of truth — import in extract_temporal.py)
# Note: fMRI bands differ from EEG; these are slow hemodynamic oscillations.
# gamma is at the Nyquist limit for TR=2s; see UNRELIABLE_FREQ_BANDS_AT_NYQUIST.
FREQ_BANDS = {
    "delta": (0.01, 0.027),  # Slow-5 (well below Nyquist)
    "theta": (0.027, 0.073),  # Slow-4 (well below Nyquist)
    "alpha": (0.073, 0.15),  # Slow-3 (well below Nyquist)
    "beta": (0.15, 0.20),  # Upper Slow-3 (safe, ~3x below Nyquist @ 0.25 Hz)
    "gamma": (0.20, 0.25),  # Slow-2/Gamma (AT Nyquist limit — aliasing risk)
}

# Runtime-effective frequency bands used by feature extraction and feature registry.
ACTIVE_FREQ_BANDS = {
    name: bounds
    for name, bounds in FREQ_BANDS.items()
    if not (EXCLUDE_NYQUIST_BANDS and name in UNRELIABLE_FREQ_BANDS_AT_NYQUIST)
}

# Bandpass filter bounds used by NiftiLabelsMasker in abide_download.py.
# Raising BANDPASS_HIGH from 0.08 -> 0.15 Hz retains beta-band oscillations
# that are physiologically relevant but were previously filtered out.
BANDPASS_LOW = 0.01  # High-pass (BOLD low-frequency cut-off)
BANDPASS_HIGH = 0.15  # Low-pass (expanded from 0.08 Hz to retain beta band)

# ABIDE I site-specific TRs from scanner specifications (single source of truth)
# Different scanners have different repetition times - CRITICAL for multi-site studies
SITE_TR_MAP = {
    "CALTECH": 2.0,
    "CMU": 2.0,
    "KKI": 2.5,
    "LEUVEN_1": 1.656,
    "LEUVEN_2": 1.656,
    "MAX_MUN": 3.0,
    "NYU": 2.0,
    "OHSU": 2.5,
    "OLIN": 1.5,
    "PITT": 1.5,
    "SBL": 2.5,
    "SDSU": 2.0,
    "STANFORD": 2.0,
    "TRINITY": 2.0,
    "UCLA_1": 3.0,
    "UCLA_2": 3.0,
    "UM_1": 2.0,
    "UM_2": 2.0,
    "USM": 2.0,
    "YALE": 2.0,
}

# ALFF slice extraction percentiles (single source of truth)
# CRITICAL: 0.21 captures brainstem (ROIs 167-170 starting at z=38), fixes missing class 11
# These percentiles must match exactly between abide_download.py and generate_labels.py
ALFF_SLICE_PERCENTILES = [0.21, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# --- FEATURE REGISTRY (The Golden Standard) ---
# Explicit feature definitions. GNN_IN_CHANNELS is calculated dynamically from this.
_BASE_TEMPORAL_FEATURES = ["mean", "std", "skew", "kurt", "psd", "mssd", "range", "autocorr"]
_RUNTIME_FREQ_FEATURES = [
    f"{band}_{suffix}"
    for band in ACTIVE_FREQ_BANDS
    for suffix in ("power", "peak")
]
FEATURE_GROUPS = {
    "temporal": _BASE_TEMPORAL_FEATURES,
    "frequency": _RUNTIME_FREQ_FEATURES + ["spectral_entropy", "phase_std"],
    "internal": ["coherence", "spatial_variance"],  # NEW: PCA/ReHo features from Phase 2
    "spatial": ["x", "y", "z_depth", "size"],  # conf_std/detection_count excluded (site leakage)
}

NUM_FREQUENCY_FEATURES = len(FEATURE_GROUPS["frequency"])
NUM_TEMPORAL_FEATURES = len(FEATURE_GROUPS["temporal"]) + NUM_FREQUENCY_FEATURES
NUM_SPATIAL_FEATURES = len(FEATURE_GROUPS["spatial"])

# ALL_FEATURE_NAMES: Concatenation order used everywhere (temporal + frequency + internal + spatial)
ALL_FEATURE_NAMES = (
    FEATURE_GROUPS["temporal"]
    + FEATURE_GROUPS["frequency"]
    + FEATURE_GROUPS["internal"]
    + FEATURE_GROUPS["spatial"]
)

# Dynamic calculation — currently 24 when gamma excluded at runtime (4 active bands × 2
# + 8 base temporal + spectral_entropy + phase_std + 2 internal + 4 spatial).
GNN_IN_CHANNELS = len(ALL_FEATURE_NAMES)

# Task 4 sentinel: conf_std and detection_count must NOT appear in spatial features.
# They encode site/scanner identity (RF AUC=1.000 in run 3) — pure site leakage.
assert NUM_SPATIAL_FEATURES == 4, (
    f"NUM_SPATIAL_FEATURES must be 4 (x, y, z_depth, size). "
    f"Got {NUM_SPATIAL_FEATURES}. conf_std and detection_count are forbidden."
)
