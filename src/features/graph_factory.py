from collections import OrderedDict
import functools
import logging
import hashlib
import torch
from torch_geometric.data import Data, Dataset
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Optional, Any

# Setup paths and config
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import (
    NUM_LOBES, LOBE_NAMES,
    MASTER_MANIFEST, NODE_ATTRIBUTES_HARMONIZED, 
    NODE_FEATURES_3D, NODE_FEATURES_3D_HARMONIZED, CAUSAL_GRAPHS_DIR,
    NUM_TEMPORAL_FEATURES, NUM_SPATIAL_FEATURES, GNN_IN_CHANNELS,
    EXCLUDED_SUBJECTS, MAX_NAN_ROIS,
)
from src.core.hyperparams import GNN_MAX_DEGENERATE_GRAPH_RATE, DEMO_AGE_CENTER, DEMO_AGE_SCALE, DEMO_SEX_CENTER, DEMO_FIQ_CENTER, DEMO_FIQ_SCALE

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

_exclusion_logged = False  # Suppress repeated exclusion log messages


def _log_exclusions_once():
    """Log excluded subjects only once per session."""
    global _exclusion_logged
    if not _exclusion_logged:
        logger.info(
            "Excluded %d hard-coded corrupted subjects: %s",
            len(EXCLUDED_SUBJECTS), sorted(EXCLUDED_SUBJECTS),
        )
        _exclusion_logged = True


def _trim_to_num_lobes(tensor: torch.Tensor, name: str) -> Optional[torch.Tensor]:
    """Handle lobe count mismatch (12→11) for compatibility."""
    if tensor.shape[0] == NUM_LOBES:
        return tensor
    if tensor.shape[0] == 12 and NUM_LOBES == 11:
        logger.debug(f"{name} has 12 lobes, trimming to {NUM_LOBES} (excluding Brainstem)")
        return tensor[:NUM_LOBES] if tensor.dim() == 1 else tensor[:NUM_LOBES, ...]
    logger.warning(f"{name} shape {tensor.shape} mismatches NUM_LOBES={NUM_LOBES}")
    return None


@functools.lru_cache(maxsize=64)
def _load_csv_cached(csv_path_str: str, index_col: Optional[str] = None) -> pd.DataFrame:
    """Load CSV with in-memory caching for faster repeated reads."""
    csv_path = Path(csv_path_str)
    df = pd.read_csv(csv_path)
    if index_col is not None:
        df = df.set_index(index_col)
    return df


def _stable_subject_seed(subject_id: str) -> int:
    """Derive a reproducible 32-bit seed from subject_id."""
    return int(hashlib.md5(subject_id.encode()).hexdigest()[:8], 16)

class ABIDECausalDataset(Dataset):

    def __init__(
        self,
        split='train',
        transform=None,
        pre_transform=None,
        temporal_features_path: Optional[Path] = None,
        graph_cache_limit: int = 256,
    ):
        super().__init__(None, transform, pre_transform)
        self.split = split
        self.augment_graphs = split == 'train'  # Only augment training data
        self.temporal_features_path = temporal_features_path or NODE_ATTRIBUTES_HARMONIZED
        self._cache_limit = max(int(graph_cache_limit), 16)
        self._load_data_sources()
        self._validate_subjects()
        self.subject_ids = self.manifest['subject_id'].astype(str).tolist()
        
        # Validate feature counts match config
        self._validate_feature_dimensions()
        
        logger.info(f"Initialized {split} dataset with {len(self.manifest)} subjects")
        logger.info(f"  Node features: {GNN_IN_CHANNELS} ({NUM_TEMPORAL_FEATURES} temporal+internal + {NUM_SPATIAL_FEATURES} spatial)")
    
    def _validate_feature_dimensions(self):
        """Ensure loaded features match config expectations."""
        if len(self.node_attr) > 0:
            sample_sub = self.node_attr.index[0]
            temporal = self._get_subject_temporal(sample_sub)
            spatial = self._get_subject_spatial(sample_sub)
            
            if temporal is not None and temporal.shape != (NUM_LOBES, NUM_TEMPORAL_FEATURES):
                raise ValueError(
                    f"Temporal feature mismatch! Expected ({NUM_LOBES}, {NUM_TEMPORAL_FEATURES}), "
                    f"got {temporal.shape}. Check extract_temporal.py output."
                )
            
            if spatial is not None and spatial.shape != (NUM_LOBES, NUM_SPATIAL_FEATURES):
                raise ValueError(
                    f"Spatial feature mismatch! Expected ({NUM_LOBES}, {NUM_SPATIAL_FEATURES}), "
                    f"got {spatial.shape}. Check extract_spatial.py output."
                )
            
            logger.info(f"✓ Feature dimensions validated")
    
    def _load_data_sources(self):
        """Load the harmonized 12-region features and spatial coordinates."""
        # 1. Master manifest
        self.manifest_raw = _load_csv_cached(str(MASTER_MANIFEST))
        
        # 2. Harmonized temporal features (aggregated to 12 regions)
        self.node_attr = _load_csv_cached(str(self.temporal_features_path), index_col='subject_id')
        logger.info("  Temporal features: %s", Path(self.temporal_features_path).name)
        
        # 3. Spatial coordinates and geometric features (6 per lobe).
        # Prefer the ComBat-harmonized version (conf_std / detection_count corrected for
        # site effects) when available; fall back to raw YOLO output otherwise.
        _spatial_path = (
            NODE_FEATURES_3D_HARMONIZED
            if NODE_FEATURES_3D_HARMONIZED.exists()
            else NODE_FEATURES_3D
        )
        logger.info("  Spatial features: %s", _spatial_path.name)
        self.coords = _load_csv_cached(str(_spatial_path), index_col='subject_id')
        self._spatial_missing_cols = [
            f"{name}_spatial_missing" for name in LOBE_NAMES.values()
        ]
        self._has_spatial_missing_mask = all(
            c in self.coords.columns for c in self._spatial_missing_cols
        )
        if self._has_spatial_missing_mask:
            logger.info("  Spatial missing-mask columns detected and will be merged into zero_lobe_mask")
        
        # 4. Adjacency matrices directory
        self.adj_dir = CAUSAL_GRAPHS_DIR
    
    def _validate_subjects(self):
        """Find subjects present in all data files and the physical graph folder."""
        manifest_subs = set(self.manifest_raw['subject_id'].astype(str).unique())
        attr_subs = set(self.node_attr.index.astype(str).unique())
        coord_subs = set(self.coords.index.astype(str).unique())
        
        available_subs = manifest_subs.intersection(attr_subs).intersection(coord_subs)

        # 1. Remove known-corrupted subjects (near-100% NaN coverage).
        excluded_upper = {s.upper() for s in EXCLUDED_SUBJECTS}
        available_subs = {
            s for s in available_subs
            if s.upper() not in excluded_upper
        }
        if excluded_upper:
            _log_exclusions_once()

        # 2. Remove subjects where too many temporal feature columns are NaN.
        # Any column whose name starts with a lobe index (0-11) is a feature column.
        feat_cols = [c for c in self.node_attr.columns if c != 'subject_id']
        nan_counts = self.node_attr[feat_cols].isna().sum(axis=1)
        high_nan_subs = set(nan_counts[nan_counts > MAX_NAN_ROIS].index.astype(str))
        if high_nan_subs:
            logger.warning(
                "Removing %d subjects with >%d NaN feature columns (likely brainstem/coverage gaps): %s",
                len(high_nan_subs), MAX_NAN_ROIS, sorted(high_nan_subs)[:10],
            )
        available_subs -= high_nan_subs

        valid_subs = []
        invalid_count = 0
        missing_graph_count = 0
        self._subject_edge_counts = {}
        self._graph_cache: OrderedDict = OrderedDict()
        self._graph_stats = {}
        
        for sub in available_subs:
            graph_path = self.adj_dir / f"{sub}_graph.pt"
            if graph_path.exists():
                try:
                    graph_data = torch.load(graph_path, map_location='cpu', weights_only=True)
                    if 'adj' not in graph_data:
                        invalid_count += 1
                        continue
                    
                    adj = graph_data['adj']
                    num_edges = (adj != 0).sum().item()
                    
                    if num_edges == 0:
                        logger.warning(f"Subject {sub}: Graph has zero edges - skipping")
                        invalid_count += 1
                        continue
                    
                    valid_subs.append(sub)
                    self._subject_edge_counts[sub] = int(num_edges)
                    self._graph_stats[sub] = {
                        'num_edges': int(num_edges),
                        'path': graph_path,
                    }
                except Exception as e:
                    logger.warning(f"Subject {sub}: Failed to validate graph: {e}")
                    invalid_count += 1
            else:
                missing_graph_count += 1
        
        if invalid_count > 0:
            logger.warning(f"Excluded {invalid_count} subjects due to invalid graphs")
        if missing_graph_count > 0:
            logger.warning(
                "Excluded %d subjects because graph artifacts are missing in %s",
                missing_graph_count,
                self.adj_dir,
            )

        total_available = len(available_subs)
        dropped_subjects = total_available - len(valid_subs)
        dropped_rate = (dropped_subjects / total_available) if total_available else 0.0
        logger.info(
            "Subject alignment summary: available=%d, valid=%d, dropped=%d (%.1f%%)",
            total_available,
            len(valid_subs),
            dropped_subjects,
            dropped_rate * 100.0,
        )

        if dropped_rate > GNN_MAX_DEGENERATE_GRAPH_RATE:
            raise ValueError(
                "Subject drop rate exceeds quality gate: "
                f"{dropped_rate:.1%} > {GNN_MAX_DEGENERATE_GRAPH_RATE:.1%}. "
                "Investigate graph construction outputs before training."
            )
        
        self.manifest = self.manifest_raw[
            (self.manifest_raw['subject_id'].astype(str).isin(valid_subs)) & 
            (self.manifest_raw['split'] == self.split)
        ].copy()
        
        self.manifest = self.manifest.sort_values('subject_id').reset_index(drop=True)
    
    def len(self):
        return len(self.manifest)
    
    def get(self, idx: int) -> Optional[Any]:
        """
        Build a PyTorch Geometric ``Data`` object for the subject at *idx*.

        Loads and combines three data sources:
        * **Causal adjacency matrix** – ``{subject_id}_graph.pt`` (12×12 directed
          graph with -log10(p) or Pearson correlation weights).
        * **Temporal + internal features** – 20 harmonised temporal features per
          region + 2 ReHo/PCA internal features from the graph dict.
        * **Spatial features** – 4 anatomical coordinates per region
          (x, y, z_depth, size). conf_std and detection_count are excluded
          (site leakage — RF AUC=1.000 in run 3; DD-012).

        The three sources are concatenated along the feature axis to produce
        node features ``x`` of shape ``(NUM_LOBES, GNN_IN_CHANNELS)`` = ``(12, 28)``.

        Args:
            idx (int): Index into the split-specific manifest (0-based).

        Returns:
            torch_geometric.data.Data: A graph object with::n

                x             – Node features            (12, 28)  float32
                edge_index    – COO edge connectivity    (2, E)    int64
                edge_attr     – Edge weights             (E, 1)    float32
                y             – Diagnosis label          (1,)      int64  (0=Control, 1=ASD)
                pos           – Node positions (x,y,z)  (12, 3)   float32
                sub_id        – Subject identifier string
                site_id       – Site index tensor        (1,)      int64
                age           – Normalised age           (1,)      float32
                sex           – Normalised sex           (1,)      float32
                fiq           – Normalised FIQ           (1,)      float32

            ``None`` if any required data source is missing, contains NaN/Inf, or
            the resulting graph has zero edges after sparsification.

        Notes:
            * Training-split graphs undergo random augmentation (50% probability):
              ±5 % Gaussian noise on node features and 30 % edge weight dropout.
            * DX_GROUP encoding: 1 → 0 (Control), 2 → 1 (ASD).
        """
        sub_id = str(self.manifest.iloc[idx]['subject_id'])
        dx_group = self.manifest.iloc[idx]['DX_GROUP']
        label = 1 if dx_group == 2 else 0  # DX_GROUP: 1=Control, 2=ASD → labels: 0=Control, 1=ASD
        
        try:
            # 1. Load 12×12 Causal Adjacency Matrix
            graph_dict = self._graph_cache.get(sub_id)
            if graph_dict is None:
                graph_path = self.adj_dir / f"{sub_id}_graph.pt"
                raw_graph = torch.load(graph_path, weights_only=True)
                graph_dict = {
                    'adj': raw_graph['adj'].clone().to(torch.float32),
                    'internal_features': raw_graph.get('internal_features'),
                    'zero_lobe_mask': raw_graph.get(
                        'zero_lobe_mask',
                        torch.zeros(NUM_LOBES, dtype=torch.bool),
                    ).bool(),
                }

                if len(self._graph_cache) >= self._cache_limit:
                    self._graph_cache.popitem(last=False)  # LRU: remove oldest (FIFO)
                self._graph_cache[sub_id] = graph_dict
            else:
                self._graph_cache.move_to_end(sub_id)  # LRU: mark as recently used

            adj = graph_dict['adj'].clone()  # Should be (NUM_LOBES, NUM_LOBES) or (12, 12) in older graphs
            
            adj_trimmed = _trim_to_num_lobes(adj, f"Subject {sub_id}: Adjacency")
            if adj_trimmed is None:
                return None
            adj = adj_trimmed
            
            if torch.isnan(adj).any() or torch.isinf(adj).any():
                logger.error(f"Subject {sub_id}: Adjacency matrix contains NaN/Inf")
                return None

            # 2. Load 12-Region Temporal Features (20 per region after encoding)
            temporal_features = self._get_subject_temporal(sub_id)
            
            # 2b. Load Internal Features from graph (Coherence + Variance)
            internal_features = graph_dict.get('internal_features')  # (12, 2) or (NUM_LOBES, 2)
            if internal_features is None:
                logger.warning(f"Subject {sub_id}: Missing internal_features in graph, using zeros")
                internal_features = torch.zeros((NUM_LOBES, 2), dtype=torch.float32)
            else:
                # SAFETY: Clean NaN/Inf in internal features
                internal_features = internal_features.float()
                
                internal_trimmed = _trim_to_num_lobes(internal_features, f"Subject {sub_id}: Internal features")
                if internal_trimmed is None:
                    internal_features = torch.zeros((NUM_LOBES, 2), dtype=torch.float32)
                else:
                    internal_features = internal_trimmed
                
                if torch.isnan(internal_features).any() or torch.isinf(internal_features).any():
                    logger.warning(f"Subject {sub_id}: Internal features contain NaN/Inf, replacing with 0")
                    internal_features = torch.where(
                        torch.isnan(internal_features) | torch.isinf(internal_features),
                        torch.tensor(0.0, dtype=torch.float32),
                        internal_features
                    )

            # Load zero-signal lobe mask (True = atlas gap / zero-signal fallback).
            # Graphs built before this field was introduced get an all-False default.
            zero_lobe_mask = graph_dict.get(
                'zero_lobe_mask',
                torch.zeros(NUM_LOBES, dtype=torch.bool)
            ).bool()
            
            zero_lobe_mask_trimmed = _trim_to_num_lobes(zero_lobe_mask.float(), f"Subject {sub_id}: Zero mask")
            if zero_lobe_mask_trimmed is None:
                zero_lobe_mask = torch.zeros(NUM_LOBES, dtype=torch.bool)
            else:
                zero_lobe_mask = zero_lobe_mask_trimmed.bool()

            spatial_missing_mask = self._get_subject_spatial_missing_mask(sub_id)
            if spatial_missing_mask is not None and spatial_missing_mask.numel() == NUM_LOBES:
                zero_lobe_mask = (zero_lobe_mask | spatial_missing_mask.bool()).bool()
            
            # 3. Load 12-Region Spatial Features (6 per region)
            spatial_features = self._get_subject_spatial(sub_id)
            
            if temporal_features is None or spatial_features is None:
                logger.error(f"Subject {sub_id}: Missing features")
                return None

            # 4. Combine all features: (12, 18+) temporal/freq + (12, 2) internal + (12, 4) spatial = (12, GNN_IN_CHANNELS)
            x = torch.cat([
                torch.tensor(temporal_features, dtype=torch.float32),  # (12, NUM_TEMPORAL_FEATURES)
                internal_features,                                       # (12, 2)
                torch.tensor(spatial_features, dtype=torch.float32)     # (12, NUM_SPATIAL_FEATURES)
            ], dim=1)
            
            # SAFETY: Check final feature tensor for NaN/Inf
            if torch.isnan(x).any() or torch.isinf(x).any():
                logger.error(f"Subject {sub_id}: Combined features contain NaN/Inf after concatenation")
                nan_mask = torch.isnan(x) | torch.isinf(x)
                x = torch.where(nan_mask, torch.tensor(0.0, dtype=torch.float32), x)
                logger.warning(f"Subject {sub_id}: Replaced {nan_mask.sum().item()} NaN/Inf values with 0")
            
            # Validate final shape
            if x.shape != (NUM_LOBES, GNN_IN_CHANNELS):
                logger.error(
                    f"Subject {sub_id}: Feature shape mismatch! "
                    f"Expected ({NUM_LOBES}, {GNN_IN_CHANNELS}), got {x.shape}"
                )
                return None
            
            # 5. Create Edge Index with validation
            edge_index = adj.nonzero().t().contiguous()
            
            if self._subject_edge_counts.get(sub_id, edge_index.shape[1]) == 0:
                logger.warning(f"Subject {sub_id}: Zero edges detected")
                return None
            
            edge_attr = adj[edge_index[0], edge_index[1]].unsqueeze(1).to(torch.float32)
            
            # Normalize edge weights for stable GAT attention.
            # Raw graph edges can be either:
            # - positive Granger-like strengths, or
            # - signed lagged-Pearson values.
            # Use standardization + tanh to preserve relative ordering while bounding.
            # This avoids sigmoid collapse when values are high (5-10 -> ~1.0)
            edge_mean = edge_attr.mean()
            edge_std = edge_attr.std() + 1e-8
            edge_attr = (edge_attr - edge_mean) / edge_std
            edge_attr = torch.tanh(edge_attr)  # Bound to [-1, 1] range
            
            if torch.isnan(edge_attr).any() or torch.isinf(edge_attr).any():
                logger.error(f"Subject {sub_id}: Edge attributes contain NaN/Inf")
                return None
            
            # 6. Build position tensor (first 3 spatial features: x, y, z)
            pos = torch.tensor(spatial_features[:, :3], dtype=torch.float32)
            
            # 7. Extract site and demographic covariates (NEW: for conditioning)
            site_id = self.manifest.iloc[idx]['SITE_ID']
            age = self.manifest.iloc[idx].get('AGE_AT_SCAN', 0)
            sex = self.manifest.iloc[idx].get('SEX', 0)  # 1=M, 2=F typically
            fiq = self.manifest.iloc[idx].get('FIQ', 100)
            
            # Map site names to indices (0-19 for 20 sites)
            site_idx = self._encode_site(site_id)
            
            # Normalize covariates
            age_norm = (age - DEMO_AGE_CENTER) / DEMO_AGE_SCALE if pd.notna(age) else 0
            sex_norm = (sex - DEMO_SEX_CENTER) if pd.notna(sex) else 0
            fiq_norm = (fiq - DEMO_FIQ_CENTER) / DEMO_FIQ_SCALE if pd.notna(fiq) and fiq > 0 else 0
            
            data_obj = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor([label], dtype=torch.long),
                pos=pos,
                sub_id=sub_id,
                site_id=torch.tensor([site_idx], dtype=torch.long),
                age=torch.tensor([age_norm], dtype=torch.float32),
                sex=torch.tensor([sex_norm], dtype=torch.float32),
                fiq=torch.tensor([fiq_norm], dtype=torch.float32),
                zero_lobe_mask=zero_lobe_mask,
            )

            # Apply augmentation to training data
            if self.augment_graphs:
                fold_rng = np.random.default_rng(_stable_subject_seed(sub_id))
                data_obj = self._augment_graph(data_obj, rng=fold_rng)

            return data_obj
            
        except Exception as e:
            logger.error(f"Failed to build graph for {sub_id}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def _get_subject_temporal(self, sub_id):
        """
        Extracts temporal features and reshapes to (NUM_LOBES, NUM_TEMPORAL_FEATURES).
        Handles both 12-lobe and 11-lobe CSVs (automatically trims Brainstem when needed).
        Returns None if subject not found or features are invalid.
        """
        try:
            row = self.node_attr.loc[sub_id].values

            # Handle both old (12-lobe) and new (11-lobe) CSVs
            features_per_lobe = NUM_TEMPORAL_FEATURES
            num_lobes_in_row = len(row) // features_per_lobe

            if num_lobes_in_row not in (11, 12):
                logger.warning(
                    f"Subject {sub_id}: Invalid temporal feature count "
                    f"({len(row)} values, expected {11*features_per_lobe} or {12*features_per_lobe})"
                )
                return None

            # Reshape based on available lobes and target NUM_LOBES
            if num_lobes_in_row == 12 and NUM_LOBES == 11:
                # Old CSV with 12 lobes, trim Brainstem (last lobe)
                features = row[:11 * features_per_lobe].reshape(11, features_per_lobe)
            elif num_lobes_in_row == NUM_LOBES:
                # CSV matches target lobe count
                features = row.reshape(NUM_LOBES, features_per_lobe)
            else:
                logger.warning(
                    f"Subject {sub_id}: Temporal lobe count mismatch "
                    f"(CSV has {num_lobes_in_row}, NUM_LOBES={NUM_LOBES})"
                )
                return None

            # Validate no NaNs
            if np.isnan(features).any():
                logger.warning(f"Subject {sub_id}: Temporal features contain NaN")
                return None

            return features
            
        except KeyError:
            logger.warning(f"Subject {sub_id}: Not found in temporal features")
            return None
        except Exception as e:
            logger.error(f"Subject {sub_id}: Error loading temporal features: {e}")
            return None

    def _get_subject_spatial(self, sub_id):
        """
        Extracts spatial features for NUM_LOBES regions and reshapes to (NUM_LOBES, NUM_SPATIAL_FEATURES).

        Extracts 4 anatomical spatial features per region (conf_std and
        detection_count are intentionally excluded — they are YOLO scanner-quality
        metrics that encode site identity, not brain structure, confirmed by
        RF AUC=1.000 in run 3):
        1. x (centroid x-coordinate)
        2. y (centroid y-coordinate)
        3. z_depth (centroid z-coordinate)
        4. size (bounding box area)

        In 11-lobe mode, skips Brainstem (lobe 11) automatically.
        Returns None if subject not found or features are invalid.
        """
        try:
            spatial_data = []

            # In 11-lobe mode, skip Brainstem (lobe 11, last in LOBE_NAMES)
            lobes_to_process = NUM_LOBES if NUM_LOBES == 12 else 11

            for lobe_id in range(lobes_to_process):
                lobe_name = LOBE_NAMES[lobe_id]

                try:
                    # Extract 4 anatomical spatial features (conf_std and
                    # detection_count excluded — encode site/scanner, not anatomy)
                    x = self.coords.loc[sub_id, f"{lobe_name}_x"]
                    y = self.coords.loc[sub_id, f"{lobe_name}_y"]
                    z = self.coords.loc[sub_id, f"{lobe_name}_z_depth"]
                    size = self.coords.loc[sub_id, f"{lobe_name}_size"]

                    # Validate all features are finite
                    features = [x, y, z, size]
                    if not all(np.isfinite(features)):
                        logger.warning(
                            f"Subject {sub_id}: Invalid spatial features for {lobe_name}"
                        )
                        return None

                    spatial_data.append(features)

                except KeyError as e:
                    logger.warning(
                        f"Subject {sub_id}: Missing spatial feature for {lobe_name}: {e}"
                    )
                    return None

            spatial_array = np.array(spatial_data)

            # Validate final shape
            if spatial_array.shape != (NUM_LOBES, NUM_SPATIAL_FEATURES):
                logger.error(
                    f"Subject {sub_id}: Spatial features shape mismatch! "
                    f"Expected ({NUM_LOBES}, {NUM_SPATIAL_FEATURES}), got {spatial_array.shape}"
                )
                return None

            return spatial_array
            
        except Exception as e:
            logger.error(f"Subject {sub_id}: Error loading spatial features: {e}")
            return None

    def _get_subject_spatial_missing_mask(self, sub_id):
        """Return per-lobe spatial-missing mask from spatial feature CSV.

        The extractor writes `<Lobe>_spatial_missing` for lobes with no global
        YOLO detections (explicit zero fallback). We merge this into the graph
        `zero_lobe_mask` so training/explainability can treat these lobes as
        structurally missing signals.
        """
        if not getattr(self, "_has_spatial_missing_mask", False):
            return torch.zeros(NUM_LOBES, dtype=torch.bool)

        try:
            row = self.coords.loc[sub_id]
        except KeyError:
            return torch.zeros(NUM_LOBES, dtype=torch.bool)

        mask = []
        for lobe_id in range(NUM_LOBES):
            col = self._spatial_missing_cols[lobe_id]
            val = 0.0
            try:
                raw = row[col]
                if isinstance(raw, pd.Series):
                    raw = raw.iloc[0]
                val = float(raw)
            except Exception:
                val = 0.0
            mask.append(bool(np.isfinite(val) and val > 0.5))

        return torch.tensor(mask, dtype=torch.bool)

    def _encode_site(self, site_name):
        """
        Encodes site name to integer index (0-19).
        Creates mapping on-the-fly if not cached.
        Uses full manifest (all splits) to guarantee a consistent site→index
        mapping across train, val, and test so the site embedding lookup is
        identical at training and inference time.
        """
        if not hasattr(self, '_site_mapping'):
            unique_sites = sorted(self.manifest_raw['SITE_ID'].unique())
            self._site_mapping = {site: idx for idx, site in enumerate(unique_sites)}

        return self._site_mapping.get(site_name, 0)  # Default to site 0 if unknown

    def _augment_graph(self, data, rng: Optional[np.random.Generator] = None):
        """
        Applies light augmentation to training graphs (feature noise, edge dropout).
        Only applied to training set to improve generalization.
        """
        if not self.augment_graphs:
            return data
        if rng is None:
            rng = np.random.default_rng()
        if rng.random() > 0.5:  # 50% augmentation rate
            return data

        data = data.clone()
        
        # Light feature noise (5% Gaussian noise on node features)
        noise = torch.tensor(
            rng.normal(loc=0.0, scale=0.05, size=data.x.shape),
            dtype=data.x.dtype,
            device=data.x.device,
        )
        data.x = data.x + noise
        
        # Edge weight dropout (30% chance to drop edge weights, but keep edge)
        edge_dropout = 0.3
        keep_mask = torch.from_numpy(rng.random(data.edge_attr.shape[0]) > edge_dropout).to(data.edge_attr.device)
        if keep_mask.sum() > 0:
            data.edge_attr = data.edge_attr * keep_mask.unsqueeze(1).float()
        
        return data


if __name__ == "__main__":
    # Test with comprehensive validation
    logger.info("="*60)
    logger.info("TESTING FIXED GRAPH FACTORY")
    logger.info("="*60)
    
    train_set = ABIDECausalDataset(split='train')
    
    if len(train_set) == 0:
        logger.error("No valid graphs found in training set!")
    else:
        logger.info(f"✓ Loaded {len(train_set)} valid graphs")
        
        # Test first 10 subjects
        valid_count = 0
        null_count = 0
        
        for i in range(min(10, len(train_set))):
            sample = train_set[i]
            if sample is None:
                null_count += 1
                continue
            
            valid_count += 1
            
            # Validate shapes
            assert sample.x.shape == (NUM_LOBES, GNN_IN_CHANNELS), \
                f"Wrong node features shape: {sample.x.shape}"
            assert sample.edge_index.shape[0] == 2, \
                f"Wrong edge_index shape: {sample.edge_index.shape}"
            assert sample.edge_index.shape[1] > 0, "Empty edge_index!"
            assert sample.edge_attr.shape[0] == sample.edge_index.shape[1], \
                "Edge attr mismatch"
            assert sample.y.shape == (1,), f"Wrong label shape: {sample.y.shape}"
        
        logger.info(f"✓ Validated {valid_count} graphs successfully")
        if null_count > 0:
            logger.warning(f"⚠️  {null_count} graphs returned None")
        
        # Print sample statistics
        sample = train_set[0]
        if sample is not None:
            logger.info("\nSample Graph Statistics:")
            logger.info(f"  Nodes: {sample.x.shape[0]}")
            logger.info(f"  Edges: {sample.edge_index.shape[1]}")
            logger.info(f"  Node Features: {sample.x.shape[1]}")
            logger.info(f"  Label: {sample.y.item()} ({'ASD' if sample.y.item() == 1 else 'Control'})")
            logger.info(f"  Subject ID: {sample.sub_id}")
