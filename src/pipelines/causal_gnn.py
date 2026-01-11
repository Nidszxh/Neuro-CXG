import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, global_add_pool, global_mean_pool
from torch.nn import Linear, Sequential, ReLU, Dropout, BatchNorm1d, LayerNorm

class CausalBrainGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes=2):
        super(CausalBrainGNN, self).__init__()
        torch.manual_seed(42)

        # 1. Feature Transformation (Phase 6.3)
        # We use LayerNorm instead of BatchNorm for smaller graph batches common in ABIDE
        self.lin_in = Linear(num_node_features, hidden_channels)
        self.norm_in = LayerNorm(hidden_channels)

        # 2. Directed Causal Convolutions
        # We use GATv2 with edge_dim=1 to ingest the directed weights directly
        # We add 'heads' to capture different frequency bands of causal influence
        self.conv1 = GATv2Conv(hidden_channels, hidden_channels, heads=4, edge_dim=1, concat=True)
        self.conv2 = GATv2Conv(hidden_channels * 4, hidden_channels, heads=4, edge_dim=1, concat=True)
        
        # Skip connections to prevent "Over-smoothing" in small brain graphs
        self.skip1 = Linear(hidden_channels, hidden_channels * 4)
        self.skip2 = Linear(hidden_channels * 4, hidden_channels * 4)

        # 3. Explainable Readout Layer (Phase 8.1)
        # We combine Mean and Max pooling - a standard trick for Q1 neuroimaging papers
        # to capture both global and peak local causal activity.
        self.fc_layer = Sequential(
            Linear(hidden_channels * 4 * 2, hidden_channels * 2),
            ReLU(),
            Dropout(0.5),
            Linear(hidden_channels * 2, num_classes)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        # A. Initial Embedding
        h = self.norm_in(F.relu(self.lin_in(x)))
        
        # B. First Causal Block
        h_conv = self.conv1(h, edge_index, edge_attr)
        h = F.elu(h_conv + self.skip1(h)) # Residual addition
        
        # C. Second Causal Block
        h_conv = self.conv2(h, edge_index, edge_attr)
        h = F.elu(h_conv + self.skip2(h)) # Residual addition

        # D. Readout: Hierarchical Feature Fusion
        # Mean pooling captures overall brain state
        # Max pooling captures the most significant 'causal hub' activity
        g_mean = global_mean_pool(h, batch)
        g_max = global_add_pool(h, batch)
        g = torch.cat([g_mean, g_max], dim=1)

        # E. Classification
        return self.fc_layer(g)

    def get_node_importance(self, x, edge_index, edge_attr, batch):
        """
        Special hook for Phase 8 (Explainability). 
        Calculates the sensitivity of the output to each lobe.
        """
        self.eval()
        x.requires_grad = True
        out = self.forward(x, edge_index, edge_attr, batch)
        # Gradient-based importance (Saliency Map)
        out.max().backward()
        return x.grad.abs().mean(dim=1)