import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Dict, List, Any
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
from src.core.config import LOBE_TO_NETWORK, NETWORK_TO_LOBES, NUM_NETWORKS, NUM_LOBES
from src.core.hyperparams import GRL_ANNEAL_STEEPNESS

logger = logging.getLogger(__name__)


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
        lobe_to_network: Optional[Dict[int, int]] = None,
        network_to_lobes: Optional[Dict[int, List[int]]] = None,
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

        expected_network_ids = set(range(self.num_networks))
        actual_network_ids = set(self.network_to_lobes.keys())
        if actual_network_ids != expected_network_ids:
            raise ValueError(
                f"network_to_lobes keys {sorted(actual_network_ids)} do not match expected "
                f"network ids {sorted(expected_network_ids)}"
            )

        # Precompute and pad per-network lobe indices to avoid rebuilding tensors
        # in every forward pass.
        max_lobes_per_network = max(len(v) for v in self.network_to_lobes.values())
        
        lobe_idx_padded = torch.full(
            (self.num_networks, max_lobes_per_network),
            fill_value=-1,
            dtype=torch.long,
        )
        lobe_counts = torch.zeros(self.num_networks, dtype=torch.long)
        for net_idx, lobe_list in self.network_to_lobes.items():
            lobe_tensor = torch.as_tensor(lobe_list, dtype=torch.long)
            lobe_idx_padded[net_idx, : lobe_tensor.numel()] = lobe_tensor
            lobe_counts[net_idx] = int(lobe_tensor.numel())

        self.register_buffer("lobe_idx_padded", lobe_idx_padded)
        self.register_buffer("lobe_counts", lobe_counts)

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
               total_nodes = num_graphs * NUM_LOBES (multi-lobe graphs, configurable per atlas_config).
            batch: Graph-assignment vector, shape (total_nodes,). Maps each node
                   to a graph index in [0, num_graphs).
            num_graphs: Number of graphs in the mini-batch.

        Returns:
            graph_emb: shape (num_graphs, hidden_dim)
        """
        expected_nodes = num_graphs * NUM_LOBES
        if h.size(0) != expected_nodes:
            logger.warning(
                "AnatomicalHierarchyPool expected %d nodes (%d graphs x %d lobes) but got %d; "
                "falling back to mean pooling for this batch.",
                expected_nodes,
                num_graphs,
                NUM_LOBES,
                h.size(0),
            )
            self.last_network_embeddings = None
            return global_mean_pool(h, batch)

        h_3d = h.reshape(num_graphs, NUM_LOBES, self.hidden_dim)

        # Build network embeddings: (num_graphs, num_networks, hidden_dim)
        network_embs = torch.zeros(
            num_graphs, self.num_networks, self.hidden_dim, device=h.device
        )

        for net_idx in range(self.num_networks):
            lobe_count = int(self.lobe_counts[net_idx].item())
            if lobe_count <= 0:
                continue

            lobe_indices = self.lobe_idx_padded[net_idx, :lobe_count]
            valid_mask = lobe_indices < h_3d.size(1)
            if not bool(valid_mask.any()):
                continue
            if not bool(valid_mask.all()):
                lobe_indices = lobe_indices[valid_mask]
            lobe_embs = h_3d[:, lobe_indices, :]

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
    GNN for Multi-Lobe Brain Graphs (configurable architecture, 11 or 12 lobes).

    Architecture:
    - Dynamic feature input (24 features: 18 temporal+freq + 2 internal + 4 spatial)
    - 2-3 GAT layers (configurable for multi-lobe graphs)
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
        pooling="mean_max_sum",
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
        alpha = 2.0 / (1.0 + math.exp(-GRL_ANNEAL_STEEPNESS * adjusted_progress)) - 1.0
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
        """Forward multiple causal views of the same subjects.

        Args:
            views: List of PyG Batch objects, one per causal graph view.
                   ``views[0]`` is treated as the base view for classification.

        Returns:
            logits_base: Class logits from the base view, shape (batch, num_classes).
            embeddings_list: List of graph embeddings, one per view, each shape (batch, hidden_dim).
                             Used by CausalInvarianceLoss during training.
        """
        if not views:
            raise ValueError("forward_multiview requires at least one view batch")

        embeddings_list = []
        for batch in views:
            _, emb = self._forward_with_embedding(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
                site_id=getattr(batch, "site_id", None),
                age=getattr(batch, "age", None),
                sex=getattr(batch, "sex", None),
                fiq=getattr(batch, "fiq", None),
            )
            embeddings_list.append(emb)

        logits_base = self.classifier(embeddings_list[0])
        return logits_base, embeddings_list

    def get_last_network_embeddings(self):
        """Return most recent (batch, NUM_NETWORKS, hidden_dim) embeddings when anatomical pooling is active."""
        if hasattr(self, "anatomical_pool"):
            return self.anatomical_pool.last_network_embeddings
        return None

    def _forward_with_embedding(
        self, x, edge_index, edge_attr, batch,
        site_id=None, age=None, sex=None, fiq=None,
    ):
        """Internal: returns (logits, graph_embedding) for contrastive/multiview use."""
        g = self._encode(x, edge_index, edge_attr, batch, site_id, age, sex, fiq)
        return self.classifier(g), g

    def _encode(self, x, edge_index, edge_attr, batch, site_id, age, sex, fiq):
        """Shared encoder body used by both forward() and _forward_with_embedding()."""
        if x.dim() != 2:
            raise ValueError(f"Expected x shape (N, F), got {tuple(x.shape)}")

        num_nodes = int(x.size(0))
        if num_nodes <= 0:
            raise ValueError("Received empty node feature tensor")

        if batch is None:
            raise ValueError("Batch tensor is required for graph pooling")
        batch = batch.view(-1).long()
        if batch.numel() != num_nodes:
            raise ValueError(
                f"Batch length mismatch: len(batch)={batch.numel()} vs num_nodes={num_nodes}"
            )
        if torch.any(batch < 0):
            raise ValueError("Batch tensor contains negative graph indices")

        if edge_index is None or edge_index.dim() != 2 or int(edge_index.size(0)) != 2:
            raise ValueError(f"Expected edge_index shape (2, E), got {tuple(edge_index.shape)}")
        edge_index = edge_index.long()
        if edge_index.numel() > 0:
            e_min = int(edge_index.min().item())
            e_max = int(edge_index.max().item())
            if e_min < 0 or e_max >= num_nodes:
                raise ValueError(
                    f"edge_index out of range: min={e_min}, max={e_max}, num_nodes={num_nodes}"
                )

        preprocessing_mode = str(getattr(self, "_preprocessing_mode", "legacy_global")).strip().lower()
        site_norm_mode = str(getattr(self, "_site_normalization_mode", "global")).strip().lower()

        # Optional fold-internal MI feature mask loaded from checkpoint.
        feature_mask = getattr(self, "_feature_mask", None)
        if feature_mask is not None:
            try:
                mask_t = torch.as_tensor(feature_mask, dtype=x.dtype, device=x.device).view(1, -1)
                if mask_t.shape[1] == x.shape[1]:
                    x = x * mask_t
            except Exception:
                pass

        # Optional fold-internal within-site normalization loaded from checkpoint.
        site_means = getattr(self, "_site_feature_means", None)
        site_stds = getattr(self, "_site_feature_stds", None)
        site_norm_applied = False
        if (
            preprocessing_mode != "legacy_global"
            and site_norm_mode == "within_site"
            and isinstance(site_means, dict)
            and isinstance(site_stds, dict)
            and site_means
            and site_stds
            and site_id is not None
        ):
            try:
                site_vec = site_id.view(-1)
                if site_vec.numel() > 0:
                    x_norm = x.clone()
                    global_mean = getattr(self, "_feature_mean", None)
                    global_std = getattr(self, "_feature_std", None)
                    global_mean_t = None
                    global_std_t = None
                    if global_mean is not None and global_std is not None:
                        global_mean_t = torch.as_tensor(global_mean, dtype=x.dtype, device=x.device).view(1, -1)
                        global_std_t = torch.as_tensor(global_std, dtype=x.dtype, device=x.device).view(1, -1).clamp_min(1e-6)
                        if global_mean_t.shape[1] != x.shape[1] or global_std_t.shape[1] != x.shape[1]:
                            global_mean_t = None
                            global_std_t = None

                    unique_sites = torch.unique(site_vec).tolist()
                    for sid_val in unique_sites:
                        sid_int = int(sid_val)
                        mean = site_means.get(sid_int)
                        std = site_stds.get(sid_int)
                        if mean is None or std is None:
                            mean_t = global_mean_t
                            std_t = global_std_t
                        else:
                            mean_t = torch.as_tensor(mean, dtype=x.dtype, device=x.device).view(1, -1)
                            std_t = torch.as_tensor(std, dtype=x.dtype, device=x.device).view(1, -1).clamp_min(1e-6)

                        if mean_t is None or std_t is None:
                            continue
                        if mean_t.shape[1] != x.shape[1] or std_t.shape[1] != x.shape[1]:
                            continue
                        node_mask = (site_vec[batch] == sid_int)
                        if node_mask.any():
                            x_norm[node_mask] = (x_norm[node_mask] - mean_t) / std_t
                            site_norm_applied = True
                    x = x_norm
            except Exception:
                pass

        # Optional fold-wise feature scaling loaded from checkpoint.
        # Keeps inference-time preprocessing consistent with train-fold scaling.
        feature_mean = getattr(self, "_feature_mean", None)
        feature_std = getattr(self, "_feature_std", None)
        should_apply_global = not (
            preprocessing_mode != "legacy_global"
            and site_norm_mode == "within_site"
            and site_norm_applied
        )
        if should_apply_global and feature_mean is not None and feature_std is not None:
            try:
                mean_t = torch.as_tensor(feature_mean, dtype=x.dtype, device=x.device).view(1, -1)
                std_t = torch.as_tensor(feature_std, dtype=x.dtype, device=x.device).view(1, -1).clamp_min(1e-6)
                if mean_t.shape[1] == x.shape[1] and std_t.shape[1] == x.shape[1]:
                    x = (x - mean_t) / std_t
            except Exception:
                # Never fail the forward pass because scaler metadata is malformed.
                pass

        # 1. Optionally add site embeddings
        if self.use_site_embedding:
            if site_id is not None:
                site_id_safe = site_id.view(-1).long()
                num_graphs = int(batch.max().item()) + 1
                if site_id_safe.numel() < num_graphs:
                    pad = torch.zeros(
                        num_graphs - site_id_safe.numel(),
                        device=site_id_safe.device,
                        dtype=site_id_safe.dtype,
                    )
                    site_id_safe = torch.cat([site_id_safe, pad], dim=0)
                num_sites = int(self.site_embedding.num_embeddings)
                if torch.any((site_id_safe < 0) | (site_id_safe >= num_sites)):
                    site_id_safe = site_id_safe.clone()
                    site_id_safe = torch.where(
                        (site_id_safe < 0) | (site_id_safe >= num_sites),
                        torch.zeros_like(site_id_safe),
                        site_id_safe,
                    )
                site_emb = self.site_embedding(site_id_safe)
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
