import torch
import torch.nn.functional as F
from torch_geometric.nn import (
    GATv2Conv,
    GlobalAttention,
    global_max_pool,
    global_mean_pool,
    global_add_pool,
)
from torch.nn import Linear, Sequential, GELU, Dropout, LayerNorm


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class CausalBrainGNN(torch.nn.Module):
    """
    GNN for 12-Node Lobe Graphs (Phase 3: Balanced Capacity).
    
    Architecture:
    - Dynamic feature input (28 features: 20 temporal + 2 internal + 6 spatial)
    - 2-3 GAT layers (configurable for 12-node graphs)
    - GELU activations (smooth, well-behaved gradients)
    - Residual connections and LayerNorm
    - Multi-scale pooling (mean + max + sum)
    - Optional site conditioning (16-dim embeddings)
    """
    def __init__(
        self, 
        num_node_features,  # Dynamic: should be 28
        hidden_channels=128,  # Increased capacity for 28-feature inputs
        num_classes=2,
        dropout=0.4,  # Reduced to prevent underfitting
        num_heads=4,
        num_layers=3,
        pooling="mean_max_sum",
        num_sites=20,
        use_site_embedding=True,
        use_demographics=True,
        use_grl=False,
        grl_alpha=1.0,
        edge_gate=True
    ):
        super(CausalBrainGNN, self).__init__()
        torch.manual_seed(42)

        self.use_site_embedding = use_site_embedding
        self.use_demographics = use_demographics
        self.pooling = pooling
        self.use_grl = use_grl
        self.grl_alpha = grl_alpha
        self.edge_gate = edge_gate

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

        self.num_layers = num_layers
        conv2_concat = False
        conv2_out = hidden_channels
        self.conv2 = GATv2Conv(
            hidden_channels * num_heads,
            hidden_channels,
            heads=num_heads,
            edge_dim=1,
            concat=conv2_concat
        )
        self.norm2 = LayerNorm(conv2_out)
        self.skip2 = Linear(hidden_channels * num_heads, conv2_out)
        self.dropout2 = Dropout(dropout)

        if num_layers > 2:
            self.conv3 = GATv2Conv(
                hidden_channels,
                hidden_channels,
                heads=num_heads,
                edge_dim=1,
                concat=False
            )
            self.norm3 = LayerNorm(hidden_channels)
            self.skip3 = Linear(hidden_channels, hidden_channels)
            self.dropout3 = Dropout(dropout)

        # Edge gating to reduce noisy causal links
        if edge_gate:
            self.edge_gate_nn = Sequential(
                Linear(1, 1)
            )

        # 4. Pooling
        demo_dim = 3 if use_demographics else 0
        if pooling == "attention":
            self.att_pool = GlobalAttention(
                gate_nn=Sequential(
                    Linear(hidden_channels, hidden_channels // 2),
                    GELU(),
                    Linear(hidden_channels // 2, 1)
                )
            )
            pooling_dim = hidden_channels + demo_dim
        else:
            # Multi-scale pooling: mean + max + sum
            pooling_dim = hidden_channels * 3 + demo_dim

        if use_demographics:
            self.post_fusion_norm = LayerNorm(pooling_dim)

        if use_grl:
            self.site_classifier = Sequential(
                Linear(pooling_dim, 32),
                GELU(),
                Linear(32, num_sites)
            )
        
        # 5. Classification Head
        self.classifier = Sequential(
            Linear(pooling_dim, hidden_channels),
            GELU(),
            Dropout(dropout),
            Linear(hidden_channels, num_classes)
        )

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
        site_id=None,
        age=None,
        sex=None,
        fiq=None,
        return_site_logits=False
    ):
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
        
        # 3. Soft edge gating
        if self.edge_gate:
            edge_gate = torch.sigmoid(self.edge_gate_nn(edge_attr))
            edge_attr = edge_attr * edge_gate

        # 4. GAT Layer 1 with residual
        h_res = self.skip1(h)
        h = self.conv1(h, edge_index, edge_attr)
        h = self.norm1(h + h_res)
        h = F.gelu(h)
        h = self.dropout1(h)
        
        # 5. GAT Layer 2 with residual
        h_res = self.skip2(h)
        h = self.conv2(h, edge_index, edge_attr)
        h = self.norm2(h + h_res)
        h = F.gelu(h)
        h = self.dropout2(h)

        # 6. GAT Layer 3 with residual (optional)
        if self.num_layers > 2:
            h_res = self.skip3(h)
            h = self.conv3(h, edge_index, edge_attr)
            h = self.norm3(h + h_res)
            h = F.gelu(h)
            h = self.dropout3(h)

        # 7. Global pooling
        if self.pooling == "attention":
            g = self.att_pool(h, batch)
        else:
            g_mean = global_mean_pool(h, batch)
            g_max = global_max_pool(h, batch)
            g_sum = global_add_pool(h, batch)
            g = torch.cat([g_mean, g_max, g_sum], dim=1)

        # 8. Append demographics if enabled
        if self.use_demographics and age is not None:
            demo = torch.stack([age.squeeze(), sex.squeeze(), fiq.squeeze()], dim=1)
            g = torch.cat([g, demo], dim=1)
            g = self.post_fusion_norm(g)

        # 9. Classification
        class_logits = self.classifier(g)

        if self.use_grl and return_site_logits:
            grl_out = GradientReversal.apply(g, self.grl_alpha)
            site_logits = self.site_classifier(grl_out)
            return class_logits, site_logits

        return class_logits
