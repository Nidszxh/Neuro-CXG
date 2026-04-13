import torch
import torch.nn.functional as F
from torch_geometric.nn import (
    GATv2Conv,
    global_max_pool,
    global_mean_pool,
    global_add_pool,
)
from torch_geometric.nn.aggr import AttentionalAggregation
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
    - Optional per-lobe identity embeddings (16-dim by default)
    """
    def __init__(
        self, 
        num_node_features,  # Dynamic: should be 28
        hidden_channels=128,  # Increased capacity for 28-feature inputs
        num_classes=2,
        dropout=0.4,  # Reduced to prevent underfitting
        num_heads=4,
        num_layers=2,
        pooling="mean_max_sum",
        num_sites=20,
        use_site_embedding=True,
        use_demographics=True,
        use_grl=False,
        grl_alpha=1.0,
        edge_gate=True,
        num_nodes=12,        # Number of graph nodes (lobes) — used for identity embedding
        node_emb_dim=16,     # Learnable per-lobe identity embedding size (0 = disabled)
    ):
        super(CausalBrainGNN, self).__init__()
        torch.manual_seed(42)

        self.use_site_embedding = use_site_embedding
        self.use_demographics = use_demographics
        self.pooling = pooling
        self.use_grl = use_grl
        self.grl_alpha = grl_alpha
        self.edge_gate = edge_gate
        self.num_nodes = num_nodes
        self.node_emb_dim = node_emb_dim

        # Learnable per-lobe identity embedding (analogous to positional embedding).
        # Gives the GNN a stable anatomical identity for each of the 12 brain lobes.
        if node_emb_dim > 0:
            self.node_embedding = torch.nn.Embedding(num_nodes, node_emb_dim)
            torch.nn.init.xavier_uniform_(self.node_embedding.weight.unsqueeze(0))
        else:
            node_emb_dim = 0  # ensure correct lin_in size

        # Site embedding for scanner bias reduction
        if use_site_embedding:
            self.site_embedding = torch.nn.Embedding(num_sites, 16)
            site_embed_dim = 16
        else:
            site_embed_dim = 0

        # 1. Input Projection with LayerNorm
        self.lin_in = Linear(num_node_features + site_embed_dim + node_emb_dim, hidden_channels)
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
                Linear(2 * hidden_channels + 1, hidden_channels // 2),
                GELU(),
                Linear(hidden_channels // 2, 1)
            )

        # 4. Pooling
        demo_dim = 3 if use_demographics else 0
        if pooling == "attention":
            self.att_pool = AttentionalAggregation(
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

    def set_grl_alpha(self, progress: float, alpha_max: float = 0.1) -> None:
        """Anneal GRL alpha with warmup and capped adversarial strength.

        Args:
            progress: Training progress in [0, 1] (current_epoch / total_epochs).
            alpha_max: Maximum GRL strength after warmup.
        """
        import math

        p = min(max(float(progress), 0.0), 1.0)
        if p < 0.2:
            self.grl_alpha = 0.0
            return

        adjusted_progress = (p - 0.2) / 0.8
        alpha = 2.0 / (1.0 + math.exp(-5.0 * adjusted_progress)) - 1.0
        self.grl_alpha = alpha * max(float(alpha_max), 0.0)

    def forward_batch(self, batch) -> torch.Tensor:
        """Convenience wrapper that forwards a PyG batch object."""
        return self.forward(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            site_id=getattr(batch, "site_id", None),
            age=getattr(batch, "age", None),
            sex=getattr(batch, "sex", None),
            fiq=getattr(batch, "fiq", None),
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
        Forward pass through the GATv2-based brain connectivity classifier.

        Processing pipeline:
            1. [Optional] Site embedding: append 16-dim site vector to node features.
               When ``site_id`` is None the embedding column is zero-padded so that
               ``lin_in`` always receives the same input width.
            2. Input projection: ``lin_in`` (Linear) + LayerNorm + GELU activation.
            3. Soft edge gating: learnable sigmoid gate applied to ``edge_attr``.
            4. GATv2 layer 1 with skip connection + LayerNorm + GELU + Dropout.
            5. GATv2 layer 2 with skip connection + LayerNorm + GELU + Dropout.
            6. [Optional] GATv2 layer 3 with skip connection (when ``num_layers >= 3``).
            7. Global graph pooling (attention or mean+max+sum).
            8. [Optional] Append demographics (age, sex, fiq) before classifier.
            9. Classifier head → class logits.
           10. [Optional] Adversarial site head via GRL.

        Args:
            x (Tensor): Node feature matrix of shape ``(num_nodes, in_channels)``.
                        ``in_channels = GNN_IN_CHANNELS`` (default 28).
            edge_index (LongTensor): COO edge connectivity, shape ``(2, num_edges)``.
            edge_attr (Tensor): Edge weights, shape ``(num_edges, 1)``.
                                Values are -log10(p-value) for Granger causality or
                                Pearson correlation coefficients.
            batch (LongTensor): Graph assignment vector, shape ``(num_nodes,)``.
                                Maps each node to a graph within the mini-batch.
            site_id (LongTensor, optional): Site index per graph, shape ``(num_graphs,)``.
                                            Integer in ``[0, num_sites)``.  Pass ``None``
                                            to disable site conditioning (uses zero-padding).
            age (Tensor, optional): Normalised age per graph, shape ``(num_graphs, 1)``.
                                    Normalisation: ``(age - 15) / 20``.
            sex (Tensor, optional): Normalised sex per graph, shape ``(num_graphs, 1)``.
                                    Normalisation: ``sex - 1.5``.  (1=M → -0.5, 2=F → 0.5)
            fiq (Tensor, optional): Normalised FIQ per graph, shape ``(num_graphs, 1)``.
                                    Normalisation: ``(fiq - 100) / 30``.
            return_site_logits (bool): If ``True`` and GRL is active, also return the
                                       site classification logits. Default ``False``.

        Returns:
            Tensor: Classification logits, shape ``(num_graphs, num_classes)``.
                    Apply ``softmax`` for probabilities or use directly with
                    ``F.cross_entropy`` / ``FocalLoss``.
            Tensor (optional): Site logits ``(num_graphs, num_sites)`` — only returned
                               when ``return_site_logits=True`` and ``use_grl=True``.

        Note:
            * Setting ``return_site_logits=True`` without enabling GRL at init-time
              returns only the classification logits (the GRL head is absent).
            * During Captum attribution, pass ``site_id=None`` so the 28-feature input
              projection stays correctly dimensioned; the site embedding column is
              automatically zero-padded.
        """
        # 1. Optionally add site embeddings
        # When site_id is None (e.g., during attribution/inference without site info),
        # concatenate zeros so lin_in always receives the same input dimensionality.
        if self.use_site_embedding:
            if site_id is not None:
                site_emb = self.site_embedding(site_id)  # (num_graphs, 16)
                site_per_node = site_emb[batch]          # (num_nodes, 16)
            else:
                site_per_node = torch.zeros(
                    x.shape[0], self.site_embedding.embedding_dim,
                    device=x.device, dtype=x.dtype
                )
            x = torch.cat([x, site_per_node], dim=1)

        # 1b. Per-lobe identity embedding.
        # Each graph has exactly self.num_nodes nodes in a fixed lobe order,
        # so local lobe index = global node index modulo num_nodes.
        if self.node_emb_dim > 0:
            lobe_idx = torch.arange(x.shape[0], device=x.device) % self.num_nodes
            node_emb = self.node_embedding(lobe_idx)  # (num_nodes_in_batch, node_emb_dim)
            x = torch.cat([x, node_emb], dim=1)

        # 2. Input projection with activation
        h = self.norm_in(F.gelu(self.lin_in(x)))
        
        # 3. Soft edge gating (Dynamic based on source/target nodes and original edge weight)
        if self.edge_gate:
            row, col = edge_index
            gate_input = torch.cat([h[row], h[col], edge_attr], dim=-1)
            edge_gate = torch.sigmoid(self.edge_gate_nn(gate_input))
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
            # Ensure tensors are 1D (batch,) not scalar or multi-dimensional
            age_flat = age.view(-1) if age.dim() > 0 else age.unsqueeze(0)
            sex_flat = sex.view(-1) if sex.dim() > 0 else sex.unsqueeze(0)
            fiq_flat = fiq.view(-1) if fiq.dim() > 0 else fiq.unsqueeze(0)
            demo = torch.stack([age_flat, sex_flat, fiq_flat], dim=1)  # (batch_size, 3)
            g = torch.cat([g, demo], dim=1)
            g = self.post_fusion_norm(g)

        # 9. Classification
        class_logits = self.classifier(g)

        if self.use_grl and return_site_logits:
            grl_out = GradientReversal.apply(g, self.grl_alpha)
            site_logits = self.site_classifier(grl_out)
            return class_logits, site_logits

        return class_logits
