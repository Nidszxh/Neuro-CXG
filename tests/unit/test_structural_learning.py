"""Unit tests for Task 1 — Structural Learning Enforcement (DD-009).

Tests:
1. _apply_structural_dropout zeros ~30% of graphs' node features
2. EdgeStructureContrastiveLoss is lower for same-graph views than cross-graph pairs
3. Structural dropout preserves edge_index and edge_attr unchanged
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pytest
import torch
from torch_geometric.data import Data, Batch

from src.models.training_utils import _apply_structural_dropout, EdgeStructureContrastiveLoss


def _make_synthetic_batch(num_graphs: int = 8, num_nodes: int = 12, num_features: int = 24) -> Batch:
    """Create a synthetic PyG Batch for testing."""
    data_list = []
    for g in range(num_graphs):
        x = torch.randn(num_nodes, num_features)
        edge_index = torch.randint(0, num_nodes, (2, 20))
        edge_attr = torch.rand(20, 1)
        y = torch.tensor([g % 2], dtype=torch.long)
        data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
    return Batch.from_data_list(data_list)


class TestStructuralDropout:
    def test_zeroes_approximately_correct_fraction(self):
        """~30% of graphs should have all-zero node features after dropout."""
        torch.manual_seed(0)
        num_graphs = 200
        num_nodes = 12
        batch = _make_synthetic_batch(num_graphs=num_graphs, num_nodes=num_nodes)

        modified = _apply_structural_dropout(batch, dropout_prob=0.30, training=True)

        zero_graph_count = 0
        for g in range(num_graphs):
            node_mask = modified.batch == g
            if modified.x[node_mask].abs().sum().item() == 0.0:
                zero_graph_count += 1

        fraction = zero_graph_count / num_graphs
        # Allow generous tolerance: binomial std ~ sqrt(200 * 0.3 * 0.7) / 200 ≈ 0.032
        assert 0.15 <= fraction <= 0.50, (
            f"Expected ~30% graphs zeroed, got {fraction:.2%}"
        )

    def test_no_dropout_at_eval(self):
        """training=False must return batch unchanged."""
        batch = _make_synthetic_batch(num_graphs=8)
        original_x = batch.x.clone()
        modified = _apply_structural_dropout(batch, dropout_prob=0.30, training=False)
        assert torch.allclose(modified.x, original_x), "Eval batch should be unchanged"

    def test_edge_attrs_preserved(self):
        """Edge index and edge attributes must not be modified by structural dropout."""
        batch = _make_synthetic_batch(num_graphs=8)
        orig_ei = batch.edge_index.clone()
        orig_ea = batch.edge_attr.clone()
        modified = _apply_structural_dropout(batch, dropout_prob=1.0, training=True)
        assert torch.allclose(modified.edge_index, orig_ei), "edge_index should be unchanged"
        assert torch.allclose(modified.edge_attr, orig_ea), "edge_attr should be unchanged"

    def test_original_batch_not_mutated(self):
        """The original batch object must not be modified (clone semantics)."""
        batch = _make_synthetic_batch(num_graphs=8)
        orig_x = batch.x.clone()
        _apply_structural_dropout(batch, dropout_prob=1.0, training=True)
        assert torch.allclose(batch.x, orig_x), "Original batch.x should be unchanged after dropout"

    def test_zero_prob_leaves_batch_unchanged(self):
        """dropout_prob=0.0 should return batch unmodified."""
        batch = _make_synthetic_batch(num_graphs=8)
        orig_x = batch.x.clone()
        modified = _apply_structural_dropout(batch, dropout_prob=0.0, training=True)
        assert torch.allclose(modified.x, orig_x)


class TestEdgeStructureContrastiveLoss:
    def setup_method(self):
        self.loss_fn = EdgeStructureContrastiveLoss(temperature=0.5)

    def test_same_graph_lower_loss_than_random(self):
        """Loss should be lower when z_full and z_edge are from the same graphs."""
        torch.manual_seed(42)
        B, D = 16, 64
        z_base = torch.randn(B, D)

        # Similar embeddings (same graphs, minor noise)
        z_same = z_base + 0.01 * torch.randn(B, D)
        loss_same = self.loss_fn(z_base, z_same)

        # Completely random embeddings (different graphs)
        z_random = torch.randn(B, D)
        loss_random = self.loss_fn(z_base, z_random)

        assert loss_same.item() < loss_random.item(), (
            f"Same-graph loss ({loss_same:.4f}) should be < random loss ({loss_random:.4f})"
        )

    def test_returns_scalar(self):
        z_full = torch.randn(8, 32)
        z_edge = torch.randn(8, 32)
        loss = self.loss_fn(z_full, z_edge)
        assert loss.dim() == 0, "Loss must be a scalar"
        assert loss.requires_grad, "Loss must be differentiable"

    def test_single_sample_returns_zero(self):
        """With B=1 there are no negatives; loss should be 0."""
        z = torch.randn(1, 32)
        loss = self.loss_fn(z, z.clone())
        assert loss.item() == pytest.approx(0.0)

    def test_gradient_flows(self):
        """Verify backward pass runs without error."""
        z_full = torch.randn(8, 32, requires_grad=True)
        z_edge = torch.randn(8, 32, requires_grad=True)
        loss = self.loss_fn(z_full, z_edge)
        loss.backward()
        assert z_full.grad is not None
        assert z_edge.grad is not None
