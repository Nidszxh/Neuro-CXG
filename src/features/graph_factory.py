import logging
import torch
from torch_geometric.data import Data, Dataset
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Setup paths and config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.core.config import (
    NUM_LOBES, LOBE_NAMES, DATA_ROOT,
    MASTER_MANIFEST, NODE_ATTRIBUTES_HARMONIZED, 
    NODE_FEATURES_3D, CAUSAL_GRAPHS_DIR
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ABIDECausalDataset(Dataset):
    """
    Optimized Dataset Factory for 5-Lobe Macro-Anatomy Graphs.
    Matches the output of the ROI Feature Extractor.
    
    CRITICAL FIX: Handles edge cases where sparsification creates graphs with zero edges.
    """
  
    def __init__(self, split='train', transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.split = split
        self._load_data_sources()
        self._validate_subjects()
        
        logger.info(f"Initialized {split} dataset with {len(self.manifest)} subjects (5-node architecture)")
    
    def _load_data_sources(self):
        """Load the harmonized 5-lobe features and 3D coordinates."""
        # 1. Master manifest
        self.manifest_raw = pd.read_csv(MASTER_MANIFEST)
        
        # 2. Harmonized temporal features (Already aggregated to 5 lobes by extractor)
        self.node_attr = pd.read_csv(NODE_ATTRIBUTES_HARMONIZED).set_index('subject_id')
        
        # 3. Spatial coordinates (x, y, z for 5 lobes)
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
                # NEW: Pre-validate graph has edges
                try:
                    graph_data = torch.load(graph_path)
                    if 'adj' not in graph_data:
                        invalid_count += 1
                        continue
                    
                    adj = graph_data['adj']
                    num_edges = (adj != 0).sum().item()
                    
                    if num_edges == 0:
                        logger.warning(f"Subject {sub}: Graph has zero edges after sparsification - skipping")
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
        """
        Construct PyTorch Geometric Data object with enhanced error handling.
        
        CRITICAL FIX: Validates edge_index is non-empty before creating Data object.
        """
        sub_id = str(self.manifest.iloc[idx]['subject_id'])
        dx_group = self.manifest.iloc[idx]['DX_GROUP']
        label = 1 if dx_group == 1 else 0  # 1=ASD, 0=Control
        
        try:
            # 1. Load 5x5 Causal Adjacency Matrix
            graph_path = self.adj_dir / f"{sub_id}_graph.pt"
            graph_dict = torch.load(graph_path)
            adj = graph_dict['adj']  # Should be (5, 5)
            
            # CRITICAL FIX: Validate adjacency matrix
            if torch.isnan(adj).any() or torch.isinf(adj).any():
                logger.error(f"Subject {sub_id}: Adjacency matrix contains NaN/Inf")
                return None

            # 2. Load 5-Lobe Temporal Features
            temporal_features = self._get_subject_temporal(sub_id)
            
            # 3. Load 5-Lobe Spatial Features
            spatial_features = self._get_subject_spatial(sub_id)
            
            if temporal_features is None or spatial_features is None:
                logger.error(f"Subject {sub_id}: Missing features")
                return None

            # 4. Combine (5, 6) and (5, 3) -> (5, 9)
            x = torch.cat([
                torch.tensor(temporal_features, dtype=torch.float32),
                torch.tensor(spatial_features, dtype=torch.float32)
            ], dim=1)
            
            # 5. Create Edge Index with CRITICAL VALIDATION
            edge_index = adj.nonzero().t().contiguous()
            
            # ========== CRITICAL FIX START ==========
            if edge_index.shape[1] == 0:
                # This subject has ZERO edges after sparsification
                # Log detailed info for debugging
                adj_stats = {
                    'max': float(adj.abs().max()),
                    'min': float(adj.abs().min()),
                    'mean': float(adj.abs().mean()),
                    'non_zero_count': int((adj != 0).sum())
                }
                logger.warning(
                    f"Subject {sub_id}: Zero edges detected | "
                    f"Adj stats: max={adj_stats['max']:.4f}, "
                    f"mean={adj_stats['mean']:.4f}, "
                    f"non_zero={adj_stats['non_zero_count']}"
                )
                return None
            # ========== CRITICAL FIX END ==========
            
            edge_attr = adj[edge_index[0], edge_index[1]].unsqueeze(1).to(torch.float32)
            
            # 6. Validate edge attributes
            if torch.isnan(edge_attr).any() or torch.isinf(edge_attr).any():
                logger.error(f"Subject {sub_id}: Edge attributes contain NaN/Inf")
                return None
            
            return Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor([label], dtype=torch.long),
                pos=torch.tensor(spatial_features, dtype=torch.float32),
                sub_id=sub_id
            )
            
        except Exception as e:
            logger.error(f"Failed to build graph for {sub_id}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def _get_subject_temporal(self, sub_id):
        """
        Extracts the 30 temporal features and reshapes to (5, 6).
        
        Returns None if subject not found or features are invalid.
        """
        try:
            row = self.node_attr.loc[sub_id].values
            # Ensure we only have the feature columns (no metadata)
            # 5 lobes * 6 stats = 30 values
            if len(row) < 30:
                logger.warning(f"Subject {sub_id}: Insufficient temporal features ({len(row)} < 30)")
                return None
            
            features = row[:30].reshape(5, 6)
            
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
        Extracts x, y, z for the 5 lobes and reshapes to (5, 3).
        
        Returns None if subject not found or coordinates are invalid.
        """
        try:
            pos_data = []
            for lobe_id in range(5):
                lobe_name = LOBE_NAMES[lobe_id]
                try:
                    x = self.coords.loc[sub_id, f"{lobe_name}_x"]
                    y = self.coords.loc[sub_id, f"{lobe_name}_y"]
                    z = self.coords.loc[sub_id, f"{lobe_name}_z_depth"]
                    
                    # Validate coordinates are finite
                    if not all(np.isfinite([x, y, z])):
                        logger.warning(f"Subject {sub_id}: Invalid coordinates for {lobe_name}")
                        return None
                    
                    pos_data.append([x, y, z])
                except KeyError:
                    logger.warning(f"Subject {sub_id}: Missing coordinates for {lobe_name}")
                    return None
            
            return np.array(pos_data)
            
        except Exception as e:
            logger.error(f"Subject {sub_id}: Error loading spatial features: {e}")
            return None


if __name__ == "__main__":
    # Test logic with detailed validation
    logger.info("="*60)
    logger.info("TESTING GRAPH FACTORY WITH EMPTY EDGE FIX")
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
            assert sample.x.shape == (5, 9), f"Wrong node features shape: {sample.x.shape}"
            assert sample.edge_index.shape[0] == 2, f"Wrong edge_index shape: {sample.edge_index.shape}"
            assert sample.edge_index.shape[1] > 0, "Empty edge_index!"
            assert sample.edge_attr.shape[0] == sample.edge_index.shape[1], "Edge attr mismatch"
            assert sample.y.shape == (1,), f"Wrong label shape: {sample.y.shape}"
        
        logger.info(f"✓ Validated {valid_count} graphs successfully")
        if null_count > 0:
            logger.warning(f"⚠️  {null_count} graphs returned None (expected for graphs with zero edges)")
        
        # Print sample statistics
        sample = train_set[0]
        if sample is not None:
            logger.info("\nSample Graph Statistics:")
            logger.info(f"  Nodes: {sample.x.shape[0]}")
            logger.info(f"  Edges: {sample.edge_index.shape[1]}")
            logger.info(f"  Node Features: {sample.x.shape[1]}")
            logger.info(f"  Label: {sample.y.item()} ({'ASD' if sample.y.item() == 1 else 'Control'})")
            logger.info(f"  Subject ID: {sample.sub_id}")