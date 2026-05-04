"""Unit tests for Task 3 — AnatomicalHierarchyPool (DD-011).

Tests:
1. Output shape is (batch, hidden_dim)
2. last_network_embeddings shape is (batch, NUM_NETWORKS, hidden_dim)
3. Pooling is differentiable (backward pass runs)
4. Different graphs produce different pooled embeddings (non-trivial function)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.core.atlas_config import (
    LOBE_TO_NETWORK,
    NETWORK_TO_LOBES,
    NUM_LOBES,
    NUM_NETWORKS,
)
from src.models.causal_gnn import AnatomicalHierarchyPool


def _make_batch_tensor(num_graphs: int, num_lobes: int = NUM_LOBES, hidden_dim: int = 64):
    """Create (N, hidden_dim) node embedding tensor with batch assignment vector."""
    N = num_graphs * num_lobes
    h = torch.randn(N, hidden_dim, requires_grad=True)
    batch = torch.repeat_interleave(
        torch.arange(num_graphs), num_lobes
    )
    return h, batch


class TestAnatomicalHierarchyPool:
    def setup_method(self):
        self.hidden_dim = 64
        self.pool = AnatomicalHierarchyPool(
            hidden_dim=self.hidden_dim,
            num_networks=NUM_NETWORKS,
            lobe_to_network=LOBE_TO_NETWORK,
            network_to_lobes=NETWORK_TO_LOBES,
        )

    def test_output_shape(self):
        """Output must be (num_graphs, hidden_dim)."""
        for num_graphs in (1, 4, 16):
            h, batch = _make_batch_tensor(num_graphs, hidden_dim=self.hidden_dim)
            out = self.pool(h, batch, num_graphs)
            assert out.shape == (num_graphs, self.hidden_dim), (
                f"Expected ({num_graphs}, {self.hidden_dim}), got {out.shape}"
            )

    def test_network_embeddings_shape(self):
        """last_network_embeddings must be (num_graphs, NUM_NETWORKS, hidden_dim)."""
        num_graphs = 8
        h, batch = _make_batch_tensor(num_graphs, hidden_dim=self.hidden_dim)
        _ = self.pool(h, batch, num_graphs)
        net_emb = self.pool.last_network_embeddings
        assert net_emb is not None, "last_network_embeddings should be set after forward"
        assert net_emb.shape == (num_graphs, NUM_NETWORKS, self.hidden_dim), (
            f"Expected ({num_graphs}, {NUM_NETWORKS}, {self.hidden_dim}), got {net_emb.shape}"
        )

    def test_backward_runs(self):
        """Backward pass must not raise and gradients must flow to h."""
        num_graphs = 4
        h, batch = _make_batch_tensor(num_graphs, hidden_dim=self.hidden_dim)
        out = self.pool(h, batch, num_graphs)
        loss = out.sum()
        loss.backward()
        assert h.grad is not None, "Gradient must flow to node embeddings"
        assert not torch.isnan(h.grad).any(), "Gradients must not contain NaN"

    def test_different_inputs_different_outputs(self):
        """Different node embeddings must produce different graph embeddings."""
        num_graphs = 2
        h1, batch = _make_batch_tensor(num_graphs, hidden_dim=self.hidden_dim)
        h2 = torch.randn_like(h1)

        with torch.no_grad():
            out1 = self.pool(h1.detach(), batch, num_graphs)
            out2 = self.pool(h2, batch, num_graphs)

        assert not torch.allclose(out1, out2), (
            "Different inputs should produce different pooled embeddings"
        )

    def test_network_cluster_coverage(self):
        """All 12 lobe indices must be assigned to a network."""
        all_lobe_ids = set(range(NUM_LOBES))
        assigned = set(LOBE_TO_NETWORK.keys())
        assert all_lobe_ids == assigned, (
            f"Missing lobe-to-network assignments for: {all_lobe_ids - assigned}"
        )
        # Reverse mapping must cover the same lobes
        covered = set(lobe for lobes in NETWORK_TO_LOBES.values() for lobe in lobes)
        assert all_lobe_ids == covered, (
            f"NETWORK_TO_LOBES missing: {all_lobe_ids - covered}"
        )
