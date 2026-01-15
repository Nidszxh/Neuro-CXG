"""
Graph Dataset Factory for ABIDE Causal Graphs

Loads and prepares PyTorch Geometric graph objects for GNN training.
Combines temporal features, spatial coordinates, and causal adjacency matrices.
"""

import logging
import torch
from torch_geometric.data import Data, Dataset
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Setup paths
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (
    LOBE_MAPPING, NUM_LOBES, DATA_ROOT,
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
    PyTorch Geometric dataset for ABIDE causal brain graphs.
    
    Loads causal graphs constructed from fMRI time series, combining:
    - Harmonized temporal features (6 per ROI, aggregated to 5 lobes)
    - Spatial coordinates from YOLO detections
    - Directed causal adjacency matrices
    
    Args:
        split: Data split to load ('train', 'val', or 'test')
        transform: Optional transform to apply to graphs
        pre_transform: Optional pre-transform to apply once
        
    Attributes:
        manifest: DataFrame with subject metadata
        node_attr: Harmonized node features
        coords: 3D spatial coordinates
        adj_dir: Directory containing causal graph files
        
    Example:
        >>> dataset = ABIDECausalDataset(split='train')
        >>> print(f"Loaded {len(dataset)} subjects")
        >>> sample = dataset[0]
        >>> print(f"Graph: {sample.x.shape[0]} nodes, {sample.edge_index.shape[1]} edges")
    """
    
    def __init__(self, split='train', transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        
        self.split = split
        self.root = DATA_ROOT
        
        # Load required dataframes
        self._load_data_sources()
        
        # Perform intersection to find valid subjects
        self._validate_subjects()
        
        logger.info(f"Initialized {split} dataset with {len(self.manifest)} subjects")
    
    def _load_data_sources(self):
        """Load all required data sources."""
        # 1. Master manifest
        if not MASTER_MANIFEST.exists():
            raise FileNotFoundError(f"Master manifest not found: {MASTER_MANIFEST}")
        
        self.manifest_raw = pd.read_csv(MASTER_MANIFEST)
        logger.debug(f"Loaded manifest with {len(self.manifest_raw)} total subjects")
        
        # 2. Harmonized temporal features
        if not NODE_ATTRIBUTES_HARMONIZED.exists():
            raise FileNotFoundError(
                f"Harmonized features not found: {NODE_ATTRIBUTES_HARMONIZED}"
            )
        
        self.node_attr = pd.read_csv(NODE_ATTRIBUTES_HARMONIZED).set_index('subject_id')
        logger.debug(f"Loaded temporal features for {len(self.node_attr)} subjects")
        
        # 3. Spatial coordinates from YOLO
        if not NODE_FEATURES_3D.exists():
            raise FileNotFoundError(
                f"Spatial features not found: {NODE_FEATURES_3D}"
            )
        
        self.coords = pd.read_csv(NODE_FEATURES_3D).set_index('subject_id')
        logger.debug(f"Loaded spatial coords for {len(self.coords)} subjects")
        
        # 4. Causal graphs directory
        if not CAUSAL_GRAPHS_DIR.exists():
            raise FileNotFoundError(
                f"Causal graphs directory not found: {CAUSAL_GRAPHS_DIR}"
            )
        
        self.adj_dir = CAUSAL_GRAPHS_DIR
    
    def _validate_subjects(self):
        """
        Validate subject availability across all data sources.
        
        Only includes subjects that have:
        1. Entry in manifest
        2. Harmonized temporal features
        3. Spatial coordinates
        4. Physical .pt causal graph file
        """
        # Get subject sets
        manifest_subs = set(self.manifest_raw['subject_id'].unique())
        attr_subs = set(self.node_attr.index.unique())
        coord_subs = set(self.coords.index.unique())
        
        # Find intersection
        available_subs = manifest_subs.intersection(attr_subs).intersection(coord_subs)
        
        # Check for physical graph files
        valid_subs = []
        for sub in available_subs:
            graph_path = self.adj_dir / f"{sub}_graph.pt"
            if graph_path.exists():
                valid_subs.append(sub)
        
        # Filter manifest to valid subjects for this split
        self.manifest = self.manifest_raw[
            (self.manifest_raw['subject_id'].isin(valid_subs)) & 
            (self.manifest_raw['split'] == self.split)
        ].copy()
        
        self.manifest = self.manifest.sort_values('subject_id').reset_index(drop=True)
        
        # Report statistics
        total_split = len(self.manifest_raw[self.manifest_raw['split'] == self.split])
        dropped = total_split - len(self.manifest)
        
        if dropped > 0:
            logger.warning(
                f"{self.split.upper()}: Dropped {dropped}/{total_split} subjects "
                f"due to missing data"
            )
        else:
            logger.info(
                f"{self.split.upper()}: All {len(self.manifest)} subjects valid"
            )
    
    def len(self):
        """Return number of subjects in dataset."""
        return len(self.manifest)
    
    def get(self, idx):
        """
        Load and prepare a single graph.
        
        Args:
            idx: Index of subject to load
            
        Returns:
            PyTorch Geometric Data object with:
            - x: Node features (temporal + spatial)
            - edge_index: Edge connectivity
            - edge_attr: Edge weights (causal strengths)
            - y: Label (0=control, 1=ASD)
            - pos: 3D spatial coordinates
            - sub_id: Subject identifier
        """
        # Get subject info
        sub_id = self.manifest.iloc[idx]['subject_id']
        dx_group = self.manifest.iloc[idx]['DX_GROUP']
        label = 1 if dx_group == 1 else 0  # DX_GROUP: 1=ASD, 2=Control
        
        try:
            # 1. Load causal adjacency matrix
            graph_path = self.adj_dir / f"{sub_id}_graph.pt"
            if not graph_path.exists():
                logger.error(f"Graph file missing for {sub_id}")
                return None
            
            graph_dict = torch.load(graph_path)
            adj = graph_dict['adj']
            
            # 2. Load and process temporal features
            temporal_features = self._load_temporal_features(sub_id)
            if temporal_features is None:
                return None
            
            # 3. Load spatial coordinates
            spatial_features = self._load_spatial_features(sub_id)
            if spatial_features is None:
                return None
            
            # 4. Combine features
            x = torch.cat([
                torch.tensor(temporal_features, dtype=torch.float32),
                torch.tensor(spatial_features, dtype=torch.float32)
            ], dim=1)
            
            # 5. Create edge index and attributes from adjacency matrix
            edge_index = adj.nonzero().t().contiguous()
            edge_attr = adj[edge_index[0], edge_index[1]].unsqueeze(1).to(torch.float32)
            
            # 6. Create Data object
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor([label], dtype=torch.long),
                pos=torch.tensor(spatial_features, dtype=torch.float32),
                sub_id=str(sub_id)
            )
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading graph for {sub_id}: {e}", exc_info=True)
            return None
    
    def _load_temporal_features(self, sub_id: str) -> np.ndarray:
        """
        Load and aggregate temporal features to 5 lobes.
        
        Args:
            sub_id: Subject identifier
            
        Returns:
            Array of shape (5, 6) with aggregated features, or None if error
        """
        try:
            raw_row = self.node_attr.loc[sub_id].values
        except KeyError:
            logger.warning(f"Subject {sub_id} missing from temporal features")
            return None
        
        # Constants
        NUM_FEATS_PER_ROI = 6  # mean, std, skew, kurt, psd, mssd
        
        # Remove any non-feature columns (handle case where extra cols exist)
        # Features should be divisible by 6
        feature_only = raw_row[:-(len(raw_row) % NUM_FEATS_PER_ROI)] \
                      if len(raw_row) % NUM_FEATS_PER_ROI != 0 else raw_row
        
        num_rois = len(feature_only) // NUM_FEATS_PER_ROI
        
        # Validate ROI count
        if num_rois not in [116, 117, 170]:
            logger.warning(
                f"Unexpected ROI count {num_rois} for {sub_id} "
                f"(expected 116/117/170)"
            )
            # Continue anyway, but log the issue
        
        try:
            ts_feats_raw = feature_only.reshape(num_rois, NUM_FEATS_PER_ROI)
        except ValueError as e:
            logger.error(f"Cannot reshape features for {sub_id}: {e}")
            return None
        
        # Aggregate to 5 lobes
        lobe_feats = []
        for lobe_id in range(NUM_LOBES):
            # Get valid indices for this lobe (accounting for different atlas versions)
            valid_indices = [
                i-1 for i in LOBE_MAPPING[lobe_id] 
                if i <= num_rois
            ]
            
            if not valid_indices:
                logger.warning(
                    f"No valid ROIs for lobe {lobe_id} in {sub_id}, using zeros"
                )
                avg_feat = np.zeros(NUM_FEATS_PER_ROI)
            else:
                avg_feat = ts_feats_raw[valid_indices].mean(axis=0)
            
            lobe_feats.append(avg_feat)
        
        return np.stack(lobe_feats)  # Shape: (5, 6)
    
    def _load_spatial_features(self, sub_id: str) -> np.ndarray:
        """
        Load 3D spatial coordinates for 5 lobes.
        
        Args:
            sub_id: Subject identifier
            
        Returns:
            Array of shape (5, 3) with x, y, z coordinates, or None if error
        """
        try:
            # Identify coordinate columns
            pos_cols = [
                c for c in self.coords.columns 
                if any(x in c for x in ['_x', '_y', '_z_depth'])
            ]
            
            # Extract and force numeric conversion
            spatial_data = self.coords.loc[sub_id][pos_cols]
            spatial_numeric = pd.to_numeric(
                spatial_data, 
                errors='coerce'
            ).values.astype(np.float32)
            
            # Handle NaNs (replace with 0 or mean)
            spatial_numeric = np.nan_to_num(spatial_numeric, nan=0.0)
            
            # Reshape to (5, 3) - expect 15 values (5 lobes × 3 coords)
            if len(spatial_numeric) != 15:
                logger.warning(
                    f"Expected 15 spatial values for {sub_id}, "
                    f"got {len(spatial_numeric)}"
                )
                # Pad or truncate as needed
                if len(spatial_numeric) < 15:
                    spatial_numeric = np.pad(
                        spatial_numeric, 
                        (0, 15 - len(spatial_numeric))
                    )
                else:
                    spatial_numeric = spatial_numeric[:15]
            
            spatial_feats = spatial_numeric.reshape(5, 3)
            
            return spatial_feats
            
        except KeyError:
            logger.warning(f"Subject {sub_id} missing from spatial features")
            return None
        except Exception as e:
            logger.error(f"Error loading spatial features for {sub_id}: {e}")
            return None


# Example usage and testing
if __name__ == "__main__":
    logger.info("Testing ABIDECausalDataset...")
    
    try:
        # Test dataset loading
        train_set = ABIDECausalDataset(split='train')
        logger.info(f"Successfully loaded {len(train_set)} training subjects")
        
        # Test sample loading
        if len(train_set) > 0:
            sample = train_set[0]
            
            if sample is not None:
                logger.info(f"Sample graph for subject {sample.sub_id}:")
                logger.info(f"  Nodes: {sample.x.shape[0]}")
                logger.info(f"  Node features: {sample.x.shape[1]}")
                logger.info(f"  Edges: {sample.edge_index.shape[1]}")
                logger.info(f"  Label: {sample.y.item()}")
                
                # Validate structure
                assert sample.x.shape[0] == 5, "Expected 5 nodes (lobes)"
                assert sample.x.shape[1] == 9, "Expected 9 features (6 temporal + 3 spatial)"
                assert sample.y.item() in [0, 1], "Label should be 0 or 1"
                
                logger.info("✓ Dataset structure validation passed")
            else:
                logger.error("Failed to load sample")
        
    except Exception as e:
        logger.error(f"Dataset test failed: {e}", exc_info=True)