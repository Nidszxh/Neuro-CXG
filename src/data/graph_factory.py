import logging
import torch
from torch_geometric.data import Data, Dataset
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Setup paths and config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
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
        for sub in available_subs:
            graph_path = self.adj_dir / f"{sub}_graph.pt"
            if graph_path.exists():
                valid_subs.append(sub)
        
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
            adj = graph_dict['adj'] # Should be (5, 5)

            # 2. Load 5-Lobe Temporal Features
            temporal_features = self._get_subject_temporal(sub_id)
            
            # 3. Load 5-Lobe Spatial Features
            spatial_features = self._get_subject_spatial(sub_id)
            
            if temporal_features is None or spatial_features is None:
                return None

            # 4. Combine (5, 6) and (5, 3) -> (5, 9)
            x = torch.cat([
                torch.tensor(temporal_features, dtype=torch.float32),
                torch.tensor(spatial_features, dtype=torch.float32)
            ], dim=1)
            
            # 5. Create Edge Index
            edge_index = adj.nonzero().t().contiguous()
            edge_attr = adj[edge_index[0], edge_index[1]].unsqueeze(1).to(torch.float32)
            
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
            return None

    def _get_subject_temporal(self, sub_id):
        """Extracts the 30 temporal features and reshapes to (5, 6)."""
        row = self.node_attr.loc[sub_id].values
        # Ensure we only have the feature columns (no metadata)
        # 5 lobes * 6 stats = 30 values
        if len(row) < 30:
            return None
        return row[:30].reshape(5, 6)

    def _get_subject_spatial(self, sub_id):
        """Extracts x, y, z for the 5 lobes and reshapes to (5, 3)."""
        # We need to ensure the columns are picked in the same order as LOBE_NAMES (0-4)
        pos_data = []
        for lobe_id in range(5):
            lobe_name = LOBE_NAMES[lobe_id]
            try:
                x = self.coords.loc[sub_id, f"{lobe_name}_x"]
                y = self.coords.loc[sub_id, f"{lobe_name}_y"]
                z = self.coords.loc[sub_id, f"{lobe_name}_z_depth"]
                pos_data.append([x, y, z])
            except KeyError:
                pos_data.append([0.0, 0.0, 0.0]) # Fallback for missing detection
        
        return np.array(pos_data)

if __name__ == "__main__":
    # Test logic
    train_set = ABIDECausalDataset(split='train')
    if len(train_set) > 0:
        sample = train_set[0]
        print(f"Node Features Shape: {sample.x.shape}") # Should be (5, 9)
        print(f"Edge Index Shape: {sample.edge_index.shape}")