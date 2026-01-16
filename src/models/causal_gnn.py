import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool
from torch.nn import Linear, Sequential, ReLU, Dropout, LayerNorm

class CausalBrainGNN(torch.nn.Module):
    """
    Tuned GNN for 5-Node Lobe Graphs.
    Input: 9 Features (6 Temporal + 3 Spatial)
    Architecture: GATv2 with Mean-Max Hub Fusion
    """
    def __init__(self, num_node_features=9, hidden_channels=64, num_classes=2):
        super(CausalBrainGNN, self).__init__()
        torch.manual_seed(42)

        # 1. Input Embedding
        # LayerNorm is critical here because it stabilizes the 9 features (which vary in scale)
        self.lin_in = Linear(num_node_features, hidden_channels)
        self.norm_in = LayerNorm(hidden_channels)

        # 2. Multi-Head Causal Attention
        # 2 heads is sufficient for 5 nodes; 4 heads often leads to redundancy on this scale.
        self.conv1 = GATv2Conv(hidden_channels, hidden_channels, heads=2, edge_dim=1, concat=True)
        # The output of conv1 is hidden_channels * 2 (due to concat)
        self.conv2 = GATv2Conv(hidden_channels * 2, hidden_channels, heads=2, edge_dim=1, concat=True)
        
        # Simple skip connections to maintain feature identity
        self.skip1 = Linear(hidden_channels, hidden_channels * 2)
        self.skip2 = Linear(hidden_channels * 2, hidden_channels * 2)

        # 3. Final Classification (Hierarchical Fusion)
        # (hidden_channels * 2) * 2 because we concat Mean and Max pooling
        self.classifier = Sequential(
            Linear(hidden_channels * 2 * 2, hidden_channels),
            ReLU(),
            Dropout(0.5), # High dropout to prevent memorizing site-specific noise
            Linear(hidden_channels, num_classes)
        )
        
        # Initialize weights for medical stability
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, edge_attr, batch):
        # A. Feature Projection
        h = self.norm_in(F.relu(self.lin_in(x)))
        
        # B. Causal Message Passing Layer 1
        h_res = self.skip1(h)
        h = F.elu(self.conv1(h, edge_index, edge_attr) + h_res)
        
        # C. Causal Message Passing Layer 2
        h_res = self.skip2(h)
        h = F.elu(self.conv2(h, edge_index, edge_attr) + h_res)

        # D. Hub-Aware Pooling
        # global_mean_pool: Captures the 'Global Brain State'
        # global_max_pool: Captures the 'Pathological Lobe Hub'
        g_mean = global_mean_pool(h, batch)
        g_max = global_max_pool(h, batch)
        g = torch.cat([g_mean, g_max], dim=1)

        # E. Final Logits
        return self.classifier(g)

    def get_node_importance(self, x, edge_index, edge_attr, batch):
        """
        Explainability Hook: Returns which lobe (0-4) drove the classification.
        """
        self.eval()
        x = x.clone().detach().requires_grad_(True)
        out = self.forward(x, edge_index, edge_attr, batch)
        # Backpropagate through the winning class
        score = out.max()
        score.backward()
        # Saliency = Absolute value of gradients normalized across features
        return x.grad.abs().sum(dim=1)