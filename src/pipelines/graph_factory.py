import torch
from torch_geometric.data import Data, Dataset
import pandas as pd
import numpy as np
from pathlib import Path

class ABIDECausalDataset(Dataset):
    
    def __init__(self, split='train', transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.root = Path("./data")
        
        # 1. Load Dataframes
        manifest_raw = pd.read_csv(self.root / "metadata" / "master_manifest.csv")
        self.node_attr = pd.read_csv(self.root / "metadata" / "node_attributes_harmonized.csv").set_index('subject_id')
        self.coords = pd.read_csv(self.root / "metadata" / "node_features_3d.csv").set_index('subject_id')
        self.adj_dir = self.root / "processed" / "causal_graphs"
        
        # 2. PERFORM INTERSECTION (The Fix)
        # Check which subjects exist in ALL required data sources
        manifest_subs = set(manifest_raw['subject_id'].unique())
        attr_subs = set(self.node_attr.index.unique())
        coord_subs = set(self.coords.index.unique())
        
        # Find subjects that have entries in CSVs AND a physical .pt file on disk
        available_subs = manifest_subs.intersection(attr_subs).intersection(coord_subs)
        
        valid_subs = []
        for sub in available_subs:
            if (self.adj_dir / f"{sub}_graph.pt").exists():
                valid_subs.append(sub)
        
        # 3. Filter the manifest to only include these verified subjects for the requested split
        self.manifest = manifest_raw[
            (manifest_raw['subject_id'].isin(valid_subs)) & 
            (manifest_raw['split'] == split)
        ].copy()
        
        self.manifest = self.manifest.sort_values('subject_id').reset_index(drop=True)
        
        # Reporting for Journal Methods section
        dropped = len(manifest_raw[manifest_raw['split'] == split]) - len(self.manifest)
        print(f"✅ {split.upper()} set initialized: {len(self.manifest)} subjects.")
        if dropped > 0:
            print(f"⚠️ Dropped {dropped} subjects due to missing 3D coords or causal graphs.")

    def len(self):
        return len(self.manifest)

    def get(self, idx):
        sub_id = self.manifest.iloc[idx]['subject_id']
        label = 1 if self.manifest.iloc[idx]['DX_GROUP'] == 1 else 0
        
        # 1. Load Causal Adjacency
        graph_path = self.adj_dir / f"{sub_id}_graph.pt"
        if not graph_path.exists(): return None
        graph_dict = torch.load(graph_path)
        adj = graph_dict['adj'] 
        
        # 2. Get Harmonized Features
        raw_row = self.node_attr.loc[sub_id].values
        
        # DYNAMIC ROI DETECTION (Fixes the ValueError)
        # We know we have 6 temporal features: mean, std, skew, kurt, psd, mssd
        num_feats_per_roi = 6 
        num_rois = len(raw_row) // num_feats_per_roi
        
        if len(raw_row) % num_feats_per_roi != 0:
            # Handle cases where there might be a 'site' or 'age' column at the end
            # by trimming the array to the nearest multiple of 6
            raw_row = raw_row[:(len(raw_row) // num_feats_per_roi) * num_feats_per_roi]
            num_rois = len(raw_row) // num_feats_per_roi

        ts_feats_raw = raw_row.reshape(num_rois, num_feats_per_roi)
        
        # 3. Aggregate to 5 Lobes
        # Important: AAL3 has indices up to 170, but AAL1/2 go to 116.
        # This list comprehension ensures we don't try to access index 169 if num_rois is 116.
        ROI_MAP = {
            0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], # Frontal
            1: [79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90], # Temporal
            2: [57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70], # Parietal
            3: [43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54], # Occipital
            4: [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 71, 72, 73, 74, 75, 76, 77, 78, 91, 92, 93, 94] # Limbic
        }
        
        lobe_feats = []
        for lobe_id in range(5):
            # i-1 because AAL indices are 1-based, but numpy is 0-based
            valid_indices = [i-1 for i in ROI_MAP[lobe_id] if (i-1) < num_rois]
            
            if not valid_indices:
                # Fallback: if a lobe has no ROIs in this atlas version, use zeros
                avg_feat = np.zeros(num_feats_per_roi)
            else:
                avg_feat = ts_feats_raw[valid_indices].mean(axis=0)
            lobe_feats.append(avg_feat)
            
        temp_feats = np.stack(lobe_feats)

        # 4. Spatial Coordinates (Already 5 nodes from YOLO)
        # 1. Identify coordinate columns
        pos_cols = [c for c in self.coords.columns if any(x in c for x in ['_x', '_y', '_z_depth'])]
        
        # 2. Extract and FORCE numeric conversion (The Fix)
        # errors='coerce' turns strings/nans into NaNs, then we fill with 0 or mean
        spatial_data = self.coords.loc[sub_id][pos_cols]
        spatial_numeric = pd.to_numeric(spatial_data, errors='coerce').values.astype(np.float32)
        
        # 3. Reshape and handle any missing values
        spatial_feats = np.nan_to_num(spatial_numeric).reshape(5, 3)
        
        # Combine using explicit float32 tensors
        x = torch.cat([
            torch.tensor(temp_feats, dtype=torch.float32),
            torch.tensor(spatial_feats, dtype=torch.float32)
        ], dim=1)

        edge_index = adj.nonzero().t().contiguous()
        edge_attr = adj[edge_index[0], edge_index[1]].unsqueeze(1).to(torch.float32)
        
        return Data(
            x=x, 
            edge_index=edge_index, 
            edge_attr=edge_attr, 
            y=torch.tensor([label], dtype=torch.long),
            pos=torch.tensor(spatial_feats, dtype=torch.float32),
            sub_id=str(sub_id)
        )   

# Example usage for Phase 6.1
if __name__ == "__main__":
    train_set = ABIDECausalDataset(split='train')
    sample = train_set[0]
    print(f"Graph Structure for Subject {sample.sub_id}:")
    print(f"Nodes: {sample.x.shape} (Features: Temporal + Spatial)")
    print(f"Edges: {sample.edge_index.shape[1]} Directed Connections")