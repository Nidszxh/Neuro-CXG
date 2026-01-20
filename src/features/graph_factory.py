import logging
import torch
from torch_geometric.data import Data, Dataset
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Setup paths and config
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.config import (
    NUM_LOBES, LOBE_NAMES, DATA_ROOT,
    MASTER_MANIFEST, NODE_ATTRIBUTES_HARMONIZED, 
    NODE_FEATURES_3D, CAUSAL_GRAPHS_DIR,
    NUM_TEMPORAL_FEATURES, NUM_SPATIAL_FEATURES, GNN_IN_CHANNELS
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ABIDECausalDataset(Dataset):

    def __init__(self, split='train', transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.split = split
        self._load_data_sources()
        self._validate_subjects()
        
        # Validate feature counts match config
        self._validate_feature_dimensions()
        
        logger.info(f"Initialized {split} dataset with {len(self.manifest)} subjects")
        logger.info(f"  Node features: {GNN_IN_CHANNELS} ({NUM_TEMPORAL_FEATURES} temporal + {NUM_SPATIAL_FEATURES} spatial)")
    
    def _validate_feature_dimensions(self):
        """Ensure loaded features match config expectations."""
        if len(self.node_attr) > 0:
            sample_sub = self.node_attr.index[0]
            temporal = self._get_subject_temporal(sample_sub)
            spatial = self._get_subject_spatial(sample_sub)
            
            if temporal is not None and temporal.shape != (NUM_LOBES, NUM_TEMPORAL_FEATURES):
                raise ValueError(
                    f"Temporal feature mismatch! Expected ({NUM_LOBES}, {NUM_TEMPORAL_FEATURES}), "
                    f"got {temporal.shape}. Check compute_roi.py output."
                )
            
            if spatial is not None and spatial.shape != (NUM_LOBES, NUM_SPATIAL_FEATURES):
                raise ValueError(
                    f"Spatial feature mismatch! Expected ({NUM_LOBES}, {NUM_SPATIAL_FEATURES}), "
                    f"got {spatial.shape}. Check extract_features.py output."
                )
            
            logger.info(f"✓ Feature dimensions validated")
    
    def _load_data_sources(self):
        """Load the harmonized 5-lobe features and spatial coordinates."""
        # 1. Master manifest
        self.manifest_raw = pd.read_csv(MASTER_MANIFEST)
        
        # 2. Harmonized temporal features (aggregated to 5 lobes)
        self.node_attr = pd.read_csv(NODE_ATTRIBUTES_HARMONIZED).set_index('subject_id')
        
        # 3. Spatial coordinates and geometric features (6 per lobe)
        self.coords = pd.read_csv(NODE_FEATURES_3D).set_index('subject_id')
        
        # 4. Adjacency matrices directory
        self.adj_dir = CAUSAL_GRAPHS_DIR
    
    def _validate_subjects(self):
        """Find subjects present in all data files and the physical graph folder."""
        manifest_subs = set(self.manifest_raw['subject_id'].astype(str).unique())
        attr_subs = set(self.node_attr.index.astype(str).unique())
        coord_subs = set(self.coords.index.astype(str).unique())
        
        available_subs = manifest_subs.intersection(attr_subs).intersection(coord_subs)
        
        valid_subs = []
        invalid_count = 0
        
        for sub in available_subs:
            graph_path = self.adj_dir / f"{sub}_graph.pt"
            if graph_path.exists():
                try:
                    graph_data = torch.load(graph_path)
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
                except Exception as e:
                    logger.warning(f"Subject {sub}: Failed to validate graph: {e}")
                    invalid_count += 1
        
        if invalid_count > 0:
            logger.warning(f"Excluded {invalid_count} subjects due to invalid graphs")
        
        self.manifest = self.manifest_raw[
            (self.manifest_raw['subject_id'].astype(str).isin(valid_subs)) & 
            (self.manifest_raw['split'] == self.split)
        ].copy()
        
        self.manifest = self.manifest.sort_values('subject_id').reset_index(drop=True)
    
    def len(self):
        return len(self.manifest)
    
    def get(self, idx):

        sub_id = str(self.manifest.iloc[idx]['subject_id'])
        dx_group = self.manifest.iloc[idx]['DX_GROUP']
        label = 1 if dx_group == 1 else 0  # 1=ASD, 0=Control
        
        try:
            # 1. Load 5x5 Causal Adjacency Matrix
            graph_path = self.adj_dir / f"{sub_id}_graph.pt"
            graph_dict = torch.load(graph_path)
            adj = graph_dict['adj']  # Should be (5, 5)
            
            if torch.isnan(adj).any() or torch.isinf(adj).any():
                logger.error(f"Subject {sub_id}: Adjacency matrix contains NaN/Inf")
                return None

            # 2. Load 5-Lobe Temporal Features (FIXED: 8 per lobe)
            temporal_features = self._get_subject_temporal(sub_id)
            
            # 3. Load 5-Lobe Spatial Features (FIXED: 6 per lobe)
            spatial_features = self._get_subject_spatial(sub_id)
            
            if temporal_features is None or spatial_features is None:
                logger.error(f"Subject {sub_id}: Missing features")
                return None

            # 4. Combine (5, 8) and (5, 6) -> (5, 14) ✓ FIXED
            x = torch.cat([
                torch.tensor(temporal_features, dtype=torch.float32),
                torch.tensor(spatial_features, dtype=torch.float32)
            ], dim=1)
            
            # Validate final shape
            if x.shape != (NUM_LOBES, GNN_IN_CHANNELS):
                logger.error(
                    f"Subject {sub_id}: Feature shape mismatch! "
                    f"Expected ({NUM_LOBES}, {GNN_IN_CHANNELS}), got {x.shape}"
                )
                return None
            
            # 5. Create Edge Index with validation
            edge_index = adj.nonzero().t().contiguous()
            
            if edge_index.shape[1] == 0:
                logger.warning(f"Subject {sub_id}: Zero edges detected")
                return None
            
            edge_attr = adj[edge_index[0], edge_index[1]].unsqueeze(1).to(torch.float32)
            
            if torch.isnan(edge_attr).any() or torch.isinf(edge_attr).any():
                logger.error(f"Subject {sub_id}: Edge attributes contain NaN/Inf")
                return None
            
            # 6. Build position tensor (first 3 spatial features: x, y, z)
            pos = torch.tensor(spatial_features[:, :3], dtype=torch.float32)
            
            return Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor([label], dtype=torch.long),
                pos=pos,
                sub_id=sub_id
            )
            
        except Exception as e:
            logger.error(f"Failed to build graph for {sub_id}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def _get_subject_temporal(self, sub_id):
        """
        Extracts temporal features and reshapes to (5, 8).        
        Returns None if subject not found or features are invalid.
        """
        try:
            row = self.node_attr.loc[sub_id].values
            
            # Expected: 5 lobes * 8 features = 40 values
            expected_features = NUM_LOBES * NUM_TEMPORAL_FEATURES
            
            if len(row) < expected_features:
                logger.warning(
                    f"Subject {sub_id}: Insufficient temporal features "
                    f"({len(row)} < {expected_features})"
                )
                return None
            
            features = row[:expected_features].reshape(NUM_LOBES, NUM_TEMPORAL_FEATURES)
            
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
        Extracts spatial features for 5 lobes and reshapes to (5, 6).
        
        FIXED: Now extracts all 6 spatial features per lobe:
        1. x (centroid x-coordinate)
        2. y (centroid y-coordinate)
        3. z_depth (centroid z-coordinate)
        4. size (bounding box area)
        5. conf_std (detection confidence consistency)
        6. detection_count (number of slices with detection)
        
        Returns None if subject not found or features are invalid.
        """
        try:
            spatial_data = []
            
            for lobe_id in range(NUM_LOBES):
                lobe_name = LOBE_NAMES[lobe_id]
                
                try:
                    # Extract all 6 spatial features
                    x = self.coords.loc[sub_id, f"{lobe_name}_x"]
                    y = self.coords.loc[sub_id, f"{lobe_name}_y"]
                    z = self.coords.loc[sub_id, f"{lobe_name}_z_depth"]
                    size = self.coords.loc[sub_id, f"{lobe_name}_size"]
                    conf_std = self.coords.loc[sub_id, f"{lobe_name}_conf_std"]
                    detection_count = self.coords.loc[sub_id, f"{lobe_name}_detection_count"]
                    
                    # Validate all features are finite
                    features = [x, y, z, size, conf_std, detection_count]
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