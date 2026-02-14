import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool, global_add_pool
from torch.nn import Linear, Sequential, GELU, Dropout, LayerNorm

class CausalBrainGNN(torch.nn.Module):
    """
    Simplified GNN for 12-Node Lobe Graphs (Phase 3: Regularized).
    
    Architecture:
    - Dynamic feature input (28 features: 20 temporal + 2 internal + 6 spatial)
    - 2 GAT layers (simplified for 12-node graphs, prevents overfitting)
    - GELU activations (smooth, well-behaved gradients)
    - Residual connections and LayerNorm
    - Multi-scale pooling (mean + max + sum)
    - Optional site conditioning (16-dim embeddings)
    """
    def __init__(
        self, 
        num_node_features,  # Dynamic: should be 28
        hidden_channels=64,  # Reduced from 256
        num_classes=2,
        dropout=0.6,  # Increased from 0.5
        num_heads=4,
        num_sites=20,
        use_site_embedding=True,
        use_demographics=True
    ):
        super(CausalBrainGNN, self).__init__()
        torch.manual_seed(42)

        self.use_site_embedding = use_site_embedding
        self.use_demographics = use_demographics

        # Site embedding for scanner bias reduction
        if use_site_embedding:
            self.site_embedding = torch.nn.Embedding(num_sites, 16)
            site_embed_dim = 16
        else:
            site_embed_dim = 0

        # 1. Input Projection with LayerNorm
        self.lin_in = Linear(num_node_features + site_embed_dim, hidden_channels)
        self.norm_in = LayerNorm(hidden_channels)
        
        # 2. GAT Layer 1
        self.conv1 = GATv2Conv(
            hidden_channels,
            hidden_channels,
            heads=num_heads,
            edge_dim=1,
            concat=True
        )
        self.norm1 = LayerNorm(hidden_channels * num_heads)
        self.skip1 = Linear(hidden_channels, hidden_channels * num_heads)
        self.dropout1 = Dropout(dropout)
        
        # 3. GAT Layer 2 (Final)
        self.conv2 = GATv2Conv(
            hidden_channels * num_heads,
            hidden_channels,
            heads=num_heads,
            edge_dim=1,
            concat=False  # Average heads
        )
        self.norm2 = LayerNorm(hidden_channels)
        self.skip2 = Linear(hidden_channels * num_heads, hidden_channels)
        self.dropout2 = Dropout(dropout)

        # 4. Multi-Scale Pooling
        # Captures: global brain state (mean), pathological hubs (max), total activation (sum)
        demo_dim = 3 if use_demographics else 0
        pooling_dim = hidden_channels * 3 + demo_dim
        
        # 5. Classification Head
        self.classifier = Sequential(
            Linear(pooling_dim, hidden_channels),
            GELU(),
            Dropout(dropout),
            Linear(hidden_channels, num_classes)
        )

    def forward(self, x, edge_index, edge_attr, batch, site_id=None, age=None, sex=None, fiq=None):
        """
        Forward pass: Treat all features as unified input (no conditional logic).
        
        Args:
            x: Node features (num_nodes, num_node_features)
            edge_index: Edge connectivity (2, num_edges)
            edge_attr: Edge weights (num_edges, 1)
            batch: Batch assignment (num_nodes,)
            site_id: Site ID per graph (num_graphs,) for conditioning
            age, sex, fiq: Demographics (num_graphs,)
        """
        # 1. Optionally add site embeddings
        if self.use_site_embedding and site_id is not None:
            site_emb = self.site_embedding(site_id)  # (num_graphs, 16)
            site_per_node = site_emb[batch]          # (num_nodes, 16)
            x = torch.cat([x, site_per_node], dim=1)

        # 2. Input projection with activation
        h = self.norm_in(F.gelu(self.lin_in(x)))
        
        # 3. GAT Layer 1 with residual
        h_res = self.skip1(h)
        h = self.conv1(h, edge_index, edge_attr)
        h = self.norm1(h + h_res)
        h = F.gelu(h)
        h = self.dropout1(h)
        
        # 4. GAT Layer 2 with residual
        h_res = self.skip2(h)
        h = self.conv2(h, edge_index, edge_attr)
        h = self.norm2(h + h_res)
        h = F.gelu(h)
        h = self.dropout2(h)

        # 5. Multi-scale global pooling
        g_mean = global_mean_pool(h, batch)
        g_max = global_max_pool(h, batch)
        g_sum = global_add_pool(h, batch)
        g = torch.cat([g_mean, g_max, g_sum], dim=1)

        # 6. Append demographics if enabled
        if self.use_demographics and age is not None:
            demo = torch.stack([age.squeeze(), sex.squeeze(), fiq.squeeze()], dim=1)
            g = torch.cat([g, demo], dim=1)

        # 7. Classification
        return self.classifier(g)
