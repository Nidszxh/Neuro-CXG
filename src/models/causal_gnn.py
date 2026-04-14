import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GATv2Conv,
    global_max_pool,
    global_mean_pool,
    global_add_pool,
)
from torch_geometric.nn.aggr import AttentionalAggregation
from torch.nn import Linear, Sequential, GELU, Dropout, LayerNorm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.core.atlas_config import LOBE_TO_NETWORK, NETWORK_TO_LOBES, NUM_NETWORKS, NUM_LOBES


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


# ─── TASK 3: Anatomical Hierarchical Pooling (DD-011) ──────────────────────────

class AnatomicalHierarchyPool(nn.Module):
    """
    Two-level anatomical pooling replacing global mean/max/sum pooling.

    Rationale (DD-011): Global pooling collapses 12 lobe embeddings into a single
    vector without respecting the known functional hierarchy of the brain. This
    discards structured information about which *networks* (DMN, Salience, etc.)
    drive classification. Hierarchical pooling forces the model to first summarise
    lobes within each functional network and then aggregate networks into a graph
    vector, matching the known two-level organisation of resting-state fMRI.

    Level 1: Attention-weighted aggregation of lobes within each of 4 networks.
             Produces network_embeddings of shape (batch, NUM_NETWORKS, hidden_dim).

    Level 2: Attention-weighted aggregation of NUM_NETWORKS embeddings → graph vector
             of shape (batch, hidden_dim).

    The intermediate ``last_network_embeddings`` is stored as an instance attribute
    after every forward call so that explainability code can access it without
    re-running the model.

    Args:
        hidden_dim: Size of node embeddings coming from the last GATv2Conv layer.
        num_networks: Number of functional networks (default 4).
        lobe_to_network: Dict mapping lobe index → network index.
        network_to_lobes: Dict mapping network index → list of lobe indices.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_networks: int = NUM_NETWORKS,
        lobe_to_network: dict = None,
        network_to_lobes: dict = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_networks = num_networks
        self.lobe_to_network = lobe_to_network or LOBE_TO_NETWORK
        self.network_to_lobes = network_to_lobes or NETWORK_TO_LOBES

        # Level-1 attention gate: scores each lobe within its network
        self.lobe_gate = Linear(hidden_dim, 1)

        # Level-2 attention gate: scores each network embedding
        self.network_gate = Linear(hidden_dim, 1)

        # Stored during forward for explainability access
        self.last_network_embeddings: torch.Tensor = None

    def forward(
        self,
        h: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        """
        Args:
            h: Node embeddings after GATv2 layers, shape (total_nodes, hidden_dim).
               total_nodes = num_graphs * NUM_LOBES (fixed 12-node graphs).
            batch: Graph-assignment vector, shape (total_nodes,). Maps each node
                   to a graph index in [0, num_graphs).
            num_graphs: Number of graphs in the mini-batch.

        Returns:
            graph_emb: shape (num_graphs, hidden_dim)
        """
        device = h.device

        # Build network embeddings: (num_graphs, num_networks, hidden_dim)
        network_embs = torch.zeros(
            num_graphs, self.num_networks, self.hidden_dim, device=device
        )

        for net_idx, lobe_list in self.network_to_lobes.items():
            # For each graph, gather lobe embeddings belonging to this network.
            # node global index = graph_idx * NUM_LOBES + lobe_local_idx
            lobe_tensor = torch.tensor(lobe_list, device=device)  # (L,)

            # Get global node indices for all graphs × lobes in this network
            # Shape: (num_graphs, L)
            global_ids = (
                torch.arange(num_graphs, device=device).unsqueeze(1) * NUM_LOBES
                + lobe_tensor.unsqueeze(0)
            )  # (num_graphs, L)

            flat_ids = global_ids.view(-1)  # (num_graphs * L,)

            # Guard against out-of-range (should not happen with fixed 12-node graphs)
            valid_mask = flat_ids < h.size(0)
            if not valid_mask.all():
                flat_ids = flat_ids[valid_mask]

            lobe_embs_flat = h[flat_ids]  # (num_graphs * L, hidden_dim)
            lobe_embs = lobe_embs_flat.view(num_graphs, len(lobe_list), self.hidden_dim)

            # Level-1 attention: (num_graphs, L, 1) → softmax over L
            gates = self.lobe_gate(lobe_embs)  # (num_graphs, L, 1)
            attn = torch.softmax(gates, dim=1)  # (num_graphs, L, 1)
            net_emb = (attn * lobe_embs).sum(dim=1)  # (num_graphs, hidden_dim)

            network_embs[:, net_idx, :] = net_emb

        # Store for explainability before Level-2 aggregation
        self.last_network_embeddings = network_embs.detach()

        # Level-2 attention: collapse (num_graphs, num_networks, hidden_dim) → (num_graphs, hidden_dim)
        gates2 = self.network_gate(network_embs)          # (num_graphs, num_networks, 1)
        attn2 = torch.softmax(gates2, dim=1)              # (num_graphs, num_networks, 1)
        graph_emb = (attn2 * network_embs).sum(dim=1)     # (num_graphs, hidden_dim)

        return graph_emb


# ─── MAIN GNN MODEL ────────────────────────────────────────────────────────────

class CausalBrainGNN(torch.nn.Module):
    """
    GNN for 12-Node Lobe Graphs (Phase 3: Balanced Capacity).

    Architecture:
    - Dynamic feature input (24 features: 18 temporal+freq + 2 internal + 4 spatial)
    - 2-3 GAT layers (configurable for 12-node graphs)
    - GELU activations (smooth, well-behaved gradients)
    - Residual connections and LayerNorm
    - Anatomical hierarchical pooling (Task 3) or attention / mean+max+sum
    - Optional site conditioning (16-dim embeddings)
    - Optional per-lobe identity embeddings (16-dim by default)
    """
    def __init__(
        self,
        num_node_features,
        hidden_channels=128,
        num_classes=2,
        dropout=0.4,
        num_heads=4,
        num_layers=2,
        pooling="anatomical",   # changed default from "mean_max_sum" to "anatomical" (Task 3)
        num_sites=20,
        use_site_embedding=True,
        use_demographics=True,
        use_grl=False,
        grl_alpha=1.0,
        edge_gate=True,
        num_nodes=12,
        node_emb_dim=16,
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
        if node_emb_dim > 0:
            self.node_embedding = torch.nn.Embedding(num_nodes, node_emb_dim)
            torch.nn.init.xavier_uniform_(self.node_embedding.weight.unsqueeze(0))
        else:
            node_emb_dim = 0

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
        if pooling == "anatomical":
            # Task 3: two-level anatomical hierarchical pooling
            self.anatomical_pool = AnatomicalHierarchyPool(hidden_channels)
            pooling_dim = hidden_channels + demo_dim
        elif pooling == "attention":
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
        """Anneal GRL alpha with warmup and capped adversarial strength."""
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

    def forward_multiview(self, views: list) -> tuple:
        """
        Forward pass over multiple PyG Batch objects (different causal graph views
        of the same subjects). Returns class logits and graph embeddings for each view.

        Args:
            views: List of PyG Batch objects, each representing one causal graph view.

        Returns:
            logits_list: List of logit tensors, one per view.
            embeddings_list: List of graph embedding tensors (before classifier),
                             shape (batch, hidden_dim) per view, used for CausalInvarianceLoss.
        """
        logits_list = []
        embeddings_list = []
        for batch in views:
            logits, emb = self._forward_with_embedding(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
                site_id=getattr(batch, "site_id", None),
                age=getattr(batch, "age", None),
                sex=getattr(batch, "sex", None),
                fiq=getattr(batch, "fiq", None),
            )
            logits_list.append(logits)
            embeddings_list.append(emb)
        return logits_list, embeddings_list

    def _forward_with_embedding(
        self, x, edge_index, edge_attr, batch,
        site_id=None, age=None, sex=None, fiq=None,
    ):
        """Internal: returns (logits, graph_embedding) for contrastive/multiview use."""
        g = self._encode(x, edge_index, edge_attr, batch, site_id, age, sex, fiq)
        return self.classifier(g), g

    def _encode(self, x, edge_index, edge_attr, batch, site_id, age, sex, fiq):
        """Shared encoder body used by both forward() and _forward_with_embedding()."""
        # 1. Optionally add site embeddings
        if self.use_site_embedding:
            if site_id is not None:
                site_emb = self.site_embedding(site_id)
                site_per_node = site_emb[batch]
            else:
                site_per_node = torch.zeros(
                    x.shape[0], self.site_embedding.embedding_dim,
                    device=x.device, dtype=x.dtype
                )
            x = torch.cat([x, site_per_node], dim=1)

        # 1b. Per-lobe identity embedding
        if self.node_emb_dim > 0:
            lobe_idx = torch.arange(x.shape[0], device=x.device) % self.num_nodes
            node_emb = self.node_embedding(lobe_idx)
            x = torch.cat([x, node_emb], dim=1)

        # 2. Input projection
        h = self.norm_in(F.gelu(self.lin_in(x)))

        # 3. Edge gating
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

        # 6. GAT Layer 3 (optional)
        if self.num_layers > 2:
            h_res = self.skip3(h)
            h = self.conv3(h, edge_index, edge_attr)
            h = self.norm3(h + h_res)
            h = F.gelu(h)
            h = self.dropout3(h)

        # 7. Pooling
        num_graphs = int(batch.max().item()) + 1
        if self.pooling == "anatomical":
            g = self.anatomical_pool(h, batch, num_graphs)
        elif self.pooling == "attention":
            g = self.att_pool(h, batch)
        else:
            g_mean = global_mean_pool(h, batch)
            g_max = global_max_pool(h, batch)
            g_sum = global_add_pool(h, batch)
            g = torch.cat([g_mean, g_max, g_sum], dim=1)

        # 8. Demographics
        if self.use_demographics and age is not None:
            age_flat = age.view(-1) if age.dim() > 0 else age.unsqueeze(0)
            sex_flat = sex.view(-1) if sex.dim() > 0 else sex.unsqueeze(0)
            fiq_flat = fiq.view(-1) if fiq.dim() > 0 else fiq.unsqueeze(0)
            demo = torch.stack([age_flat, sex_flat, fiq_flat], dim=1)
            g = torch.cat([g, demo], dim=1)
            g = self.post_fusion_norm(g)

        return g

    def forward(
        self,
        x, edge_index, edge_attr, batch,
        site_id=None, age=None, sex=None, fiq=None,
        return_site_logits=False,
    ):
        """
        Forward pass through the GATv2-based brain connectivity classifier.

        Processing pipeline:
            1. [Optional] Site embedding appended to node features.
            2. Input projection: lin_in + LayerNorm + GELU.
            3. Soft edge gating with learned sigmoid gate.
            4. GATv2 layer 1 with skip connection + LayerNorm + GELU + Dropout.
            5. GATv2 layer 2 with skip connection + LayerNorm + GELU + Dropout.
            6. [Optional] GATv2 layer 3.
            7. Anatomical hierarchical pooling (or attention / mean+max+sum).
            8. [Optional] Demographics concatenated before classifier.
            9. Classifier head → class logits.
           10. [Optional] Adversarial site head via GRL.

        Args:
            x:        Node features (num_nodes, GNN_IN_CHANNELS).
            edge_index: COO connectivity  (2, E).
            edge_attr:  Edge weights      (E, 1).
            batch:      Graph assignment  (num_nodes,).
            site_id:    Site index        (num_graphs,) or None.
            age/sex/fiq: Demographics    (num_graphs, 1) each.
            return_site_logits: Also return site logits when GRL is active.

        Returns:
            class_logits: (num_graphs, num_classes)
            site_logits (optional): (num_graphs, num_sites)
        """
        g = self._encode(x, edge_index, edge_attr, batch, site_id, age, sex, fiq)
        class_logits = self.classifier(g)

        if self.use_grl and return_site_logits:
            grl_out = GradientReversal.apply(g, self.grl_alpha)
            site_logits = self.site_classifier(grl_out)
            return class_logits, site_logits

        return class_logits
