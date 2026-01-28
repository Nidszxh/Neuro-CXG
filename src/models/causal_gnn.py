import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool, global_add_pool
from torch.nn import Linear, Sequential, ReLU, Dropout, LayerNorm

class CausalBrainGNN(torch.nn.Module):
    """
    GNN for 5-Node Lobe Graphs with architectural improvements.
    
    Input: 14 Features (8 Temporal + 6 Spatial)
    Architecture: 
    - 3 GATv2 layers with skip connections
    - Layer normalization after each layer
    - Dropout between layers
    - Multi-scale pooling (mean + max + sum)
    - Learnable edge weight transformation
    """
    def __init__(
        self, 
        num_node_features=14,  
        hidden_channels=64, 
        num_classes=2,
        dropout=0.5,
        num_heads=2,
        num_sites=20,              # Site conditioning
        use_site_embedding=True,   # Toggle site conditioning
        use_demographics=True,     # age/sex/fiq conditioning
        strip_yolo_metadata=False, # If True, drop YOLO metadata (size/conf/count) and keep coords only
        yolo_metadata_dim=3        # Number of YOLO metadata features to optionally strip
    ):
        super(CausalBrainGNN, self).__init__()
        torch.manual_seed(42)

        self.use_site_embedding = use_site_embedding
        self.use_demographics = use_demographics
        self.strip_yolo_metadata = strip_yolo_metadata
        self.yolo_metadata_dim = yolo_metadata_dim

        # Allow two modes:
        # 1) YOLO coords-only: strip_yolo_metadata=True → drop last yolo_metadata_dim features
        # 2) YOLO full: strip_yolo_metadata=False → use all features (default)
        effective_in_feats = num_node_features - (yolo_metadata_dim if strip_yolo_metadata else 0)
        if effective_in_feats <= 0:
            raise ValueError("Effective input features must be positive. Check num_node_features/yolo_metadata_dim.")
        
        # NEW: Site embedding (reduces site-specific bias)
        if use_site_embedding:
            self.site_embedding = torch.nn.Embedding(num_sites, 16)
            site_embed_dim = 16
        else:
            site_embed_dim = 0

        # 1. Input Embedding with normalization
        self.lin_in = Linear(effective_in_feats + site_embed_dim, hidden_channels)
        self.norm_in = LayerNorm(hidden_channels)
        
        # 2. Learnable Edge Weight Transformation
        # This allows the model to learn which edge correlations matter most
        self.edge_encoder = Sequential(
            Linear(1, 16),  # Edge attr is scalar correlation
            ReLU(),
            Linear(16, 1)
        )

        # 3. Multi-Head Causal Attention Layers (3 layers for depth)
        # Layer 1
        self.conv1 = GATv2Conv(
            hidden_channels, 
            hidden_channels, 
            heads=num_heads, 
            edge_dim=1,  # Transformed edge weight
            concat=True
        )
        self.norm1 = LayerNorm(hidden_channels * num_heads)
        self.dropout1 = Dropout(dropout)
        self.skip1 = Linear(hidden_channels, hidden_channels * num_heads)
        
        # Layer 2
        self.conv2 = GATv2Conv(
            hidden_channels * num_heads,
            hidden_channels,
            heads=num_heads,
            edge_dim=1,
            concat=True
        )
        self.norm2 = LayerNorm(hidden_channels * num_heads)
        self.dropout2 = Dropout(dropout)
        self.skip2 = Linear(hidden_channels * num_heads, hidden_channels * num_heads)
        
        # Layer 3 (NEW: increased depth)
        self.conv3 = GATv2Conv(
            hidden_channels * num_heads,
            hidden_channels,
            heads=num_heads,
            edge_dim=1,
            concat=False  # Final layer averages heads
        )
        self.norm3 = LayerNorm(hidden_channels)
        self.dropout3 = Dropout(dropout)
        self.skip3 = Linear(hidden_channels * num_heads, hidden_channels)

        # 4. Multi-Scale Pooling
        # Combines mean (global brain state), max (pathological hub), and sum (total activation)
        demo_dim = 3 if use_demographics else 0
        pooling_dim = hidden_channels * 3 + demo_dim  # mean + max + sum + demographics
        
        # 5. Final Classification Head
        self.classifier = Sequential(
            Linear(pooling_dim, hidden_channels * 2),
            LayerNorm(hidden_channels * 2),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_channels * 2, hidden_channels),
            LayerNorm(hidden_channels),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_channels, num_classes)
        )
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Kaiming initialization for stable training."""
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, edge_attr, batch, site_id=None, age=None, sex=None, fiq=None):
        """
        Forward pass.
        
        Args:
            x: Node features. Two supported layouts:
                - YOLO full (default): temporal + spatial coords + metadata (size/conf/count)
                - YOLO coords-only: temporal + spatial coords (set strip_yolo_metadata=True)
            edge_index: Edge connectivity (2, num_edges)
            edge_attr: Edge weights (num_edges, 1)
            batch: Batch assignment (num_nodes,)
        """
        # Optionally drop YOLO metadata (size/conf/count) to run coords-only mode
        if self.strip_yolo_metadata:
            if x.shape[1] == self.lin_in.in_features:
                pass  # already stripped before call
            elif x.shape[1] == self.lin_in.in_features + self.yolo_metadata_dim:
                x = x[:, :-self.yolo_metadata_dim]
            else:
                raise ValueError(
                    f"Unexpected feature dim {x.shape[1]} for coords-only mode. "
                    f"Expected {self.lin_in.in_features} (stripped) or "
                    f"{self.lin_in.in_features + self.yolo_metadata_dim} (full)."
                )
        # Optional: site_id tensor (num_graphs,) for site conditioning
        if self.use_site_embedding and site_id is not None:
            # Map site embeddings to nodes using batch index
            site_emb = self.site_embedding(site_id)  # (num_graphs, 16)
            site_per_node = site_emb[batch]          # (num_nodes, 16)
            x = torch.cat([x, site_per_node], dim=1)

        # A. Input Projection
        h = self.norm_in(F.relu(self.lin_in(x)))
        
        # B. Transform Edge Weights (learnable importance)
        edge_attr_transformed = self.edge_encoder(edge_attr)
        
        # C. Layer 1: Causal Message Passing + Skip + Norm + Dropout
        h_res = self.skip1(h)
        h = self.conv1(h, edge_index, edge_attr_transformed)
        h = self.norm1(h + h_res)
        h = F.elu(h)
        h = self.dropout1(h)
        
        # D. Layer 2: Causal Message Passing + Skip + Norm + Dropout
        h_res = self.skip2(h)
        h = self.conv2(h, edge_index, edge_attr_transformed)
        h = self.norm2(h + h_res)
        h = F.elu(h)
        h = self.dropout2(h)
        
        # E. Layer 3: Causal Message Passing + Skip + Norm + Dropout
        h_res = self.skip3(h)
        h = self.conv3(h, edge_index, edge_attr_transformed)
        h = self.norm3(h + h_res)
        h = F.elu(h)
        h = self.dropout3(h)

        # F. Multi-Scale Hub-Aware Pooling
        g_mean = global_mean_pool(h, batch)  # Global brain state
        g_max = global_max_pool(h, batch)    # Pathological hub detection
        g_sum = global_add_pool(h, batch)    # Total activation level
        
        g = torch.cat([g_mean, g_max, g_sum], dim=1)

        # Append demographics (age, sex, fiq) if available
        if self.use_demographics and age is not None and sex is not None and fiq is not None:
            demo = torch.stack([age, sex, fiq], dim=1)  # (num_graphs, 3)
            g = torch.cat([g, demo], dim=1)

        # G. Final Classification
        return self.classifier(g)

    def get_node_importance(self, x, edge_index, edge_attr, batch):
        """
        Explainability: Returns node importance via gradient-based saliency.
        
        Returns:
            Tensor (num_nodes,): Importance score per node (lobe)
        """
        self.eval()
        x = x.clone().detach().requires_grad_(True)
        
        out = self.forward(x, edge_index, edge_attr, batch)
        
        # Back propagate through predicted class
        score = out.max()
        score.backward()
        
        # Saliency = absolute gradient magnitude
        return x.grad.abs().sum(dim=1)
    
    def get_edge_importance(self, x, edge_index, edge_attr, batch):
        """
        Explainability: Returns edge importance via gradient-based saliency.
        
        Returns:
            Tensor (num_edges,): Importance score per edge
        """
        self.eval()
        edge_attr_clone = edge_attr.clone().detach().requires_grad_(True)
        
        out = self.forward(x, edge_index, edge_attr_clone, batch)
        
        score = out.max()
        score.backward()
        
        return edge_attr_clone.grad.abs().squeeze()


if __name__ == "__main__":
    """Test the  architecture."""
    print("="*60)
    print("TESTING  GNN ARCHITECTURE")
    print("="*60)
    
    # Create dummy data
    num_nodes = 5
    num_edges = 8
    batch_size = 4
    
    x = torch.randn(num_nodes * batch_size, 14)
    edge_index = torch.randint(0, num_nodes * batch_size, (2, num_edges * batch_size))
    edge_attr = torch.randn(num_edges * batch_size, 1)
    batch = torch.repeat_interleave(torch.arange(batch_size), num_nodes)
    
    # Initialize model
    model = CausalBrainGNN(
        num_node_features=14,
        hidden_channels=64,
        num_classes=2,
        dropout=0.5,
        num_heads=2
    )
    
    # Forward pass
    out = model(x, edge_index, edge_attr, batch)
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {out.shape}")
    print(f"✓ Expected: ({batch_size}, 2)")
    
    assert out.shape == (batch_size, 2), "Output shape mismatch!"
    
    # Test explainability
    node_importance = model.get_node_importance(x, edge_index, edge_attr, batch)
    edge_importance = model.get_edge_importance(x, edge_index, edge_attr, batch)
    
    print(f"✓ Node importance shape: {node_importance.shape}")
    print(f"✓ Edge importance shape: {edge_importance.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    
    print("="*60)
    print("✅ ALL TESTS PASSED")
    print("="*60)