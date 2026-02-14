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
    NUM_LOBES, LOBE_NAMES,
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
        self.augment_graphs = split == 'train'  # Only augment training data
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
        self.manifest_raw = pd.read_csv(MASTER_MANIFEST)
        
        # 2. Harmonized temporal features (aggregated to 12 regions)
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
        label = 1 if dx_group == 2 else 0  # DX_GROUP: 1=Control, 2=ASD → labels: 0=Control, 1=ASD
        
        try:
            # 1. Load 12×12 Causal Adjacency Matrix
            graph_path = self.adj_dir / f"{sub_id}_graph.pt"
            graph_dict = torch.load(graph_path)
            adj = graph_dict['adj']  # Should be (12, 12)
            
            if torch.isnan(adj).any() or torch.isinf(adj).any():
                logger.error(f"Subject {sub_id}: Adjacency matrix contains NaN/Inf")
                return None

            # 2. Load 12-Region Temporal Features (8 per region)
            temporal_features = self._get_subject_temporal(sub_id)
            
            # 3. Load 12-Region Spatial Features (6 per region)
            spatial_features = self._get_subject_spatial(sub_id)
            
            if temporal_features is None or spatial_features is None:
                logger.error(f"Subject {sub_id}: Missing features")
                return None

            # 4. Combine (12, 8) temporal and (12, 6) spatial -> (12, 14) ✓
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
            
            # 7. Extract site and demographic covariates (NEW: for conditioning)
            site_id = self.manifest.iloc[idx]['SITE_ID']
            age = self.manifest.iloc[idx].get('AGE_AT_SCAN', 0)
            sex = self.manifest.iloc[idx].get('SEX', 0)  # 1=M, 2=F typically
            fiq = self.manifest.iloc[idx].get('FIQ', 100)
            
            # Map site names to indices (0-19 for 20 sites)
            site_idx = self._encode_site(site_id)
            
            # Normalize covariates
            age_norm = (age - 15) / 20 if pd.notna(age) else 0  # Roughly 5-35 years
            sex_norm = (sex - 1.5) if pd.notna(sex) else 0  # Normalize to ~[-0.5, 0.5]
            fiq_norm = (fiq - 100) / 30 if pd.notna(fiq) and fiq > 0 else 0  # Normalize IQ
            
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
                fiq=torch.tensor([fiq_norm], dtype=torch.float32)
            )

            # Apply augmentation to training data
            data_obj = self._augment_graph(data_obj)

            return data_obj
            
        except Exception as e:
            logger.error(f"Failed to build graph for {sub_id}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def _get_subject_temporal(self, sub_id):
        """
        Extracts temporal features and reshapes to (12, 8).        
        Returns None if subject not found or features are invalid.
        """
        try:
            row = self.node_attr.loc[sub_id].values
            
            # Expected: 12 regions * 8 features = 96 values
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
        Extracts spatial features for 12 regions and reshapes to (12, 6).
        
        FIXED: Now extracts all 6 spatial features per region:
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

    def _encode_site(self, site_name):
        """
        Encodes site name to integer index (0-19).
        Creates mapping on-the-fly if not cached.
        """
        if not hasattr(self, '_site_mapping'):
            unique_sites = sorted(self.manifest['SITE_ID'].unique())
            self._site_mapping = {site: idx for idx, site in enumerate(unique_sites)}

        return self._site_mapping.get(site_name, 0)  # Default to site 0 if unknown

    def _augment_graph(self, data):
        """
        Applies light augmentation to training graphs (feature noise, edge dropout).
        Only applied to training set to improve generalization.
        """
        if not self.augment_graphs or np.random.random() > 0.5:  # 50% augmentation rate
            return data
        
        # Light feature noise (5% Gaussian noise on node features)
        noise = torch.randn_like(data.x) * 0.05
        data.x = data.x + noise
        
        # Edge weight dropout (30% chance to drop edge weights, but keep edge)
        edge_dropout = 0.3
        keep_mask = torch.rand(data.edge_attr.shape[0]) > edge_dropout
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