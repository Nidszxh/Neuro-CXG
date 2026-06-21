"""Unit tests for Task 1 — Structural Learning Enforcement (DD-009).

Tests:
1. _apply_structural_dropout zeros ~30% of graphs' node features
2. EdgeStructureContrastiveLoss is lower for same-graph views than cross-graph pairs
3. Structural dropout preserves edge_index and edge_attr unchanged
"""

import copy

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from src.analysis.edge_importance import GradientEdgeAttributor
from src.models.training_utils import (
    EdgeStructureContrastiveLoss,
    _apply_structural_dropout,
    make_loader,
    train_one_epoch_with_accumulation,
)


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
        z_full = torch.randn(8, 32, requires_grad=True)
        z_edge = torch.randn(8, 32, requires_grad=True)
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

class _ToyStructuralModel(nn.Module):
    """Small differentiable graph model exposing _forward_with_embedding for training_utils."""

    def __init__(self):
        super().__init__()
        self.node_scale = nn.Parameter(torch.tensor(1.0))
        self.edge_scale = nn.Parameter(torch.tensor(1.0))
        self.classifier = nn.Linear(2, 2, bias=False)

    def _encode(self, x, edge_index, edge_attr, batch):
        num_graphs = int(batch.max().item()) + 1
        edge_batch = batch[edge_index[1]]

        node_means = []
        edge_means = []
        for g_idx in range(num_graphs):
            node_mask = batch == g_idx
            edge_mask = edge_batch == g_idx
            node_means.append(x[node_mask, 0].mean())
            edge_means.append(edge_attr[edge_mask].mean() if edge_mask.any() else torch.tensor(0.0, device=x.device))

        node_means = torch.stack(node_means)
        edge_means = torch.stack(edge_means)
        return torch.stack(
            [self.node_scale * node_means, self.edge_scale * edge_means],
            dim=1,
        )

    def _forward_with_embedding(self, x, edge_index, edge_attr, batch, **kwargs):
        emb = self._encode(x, edge_index, edge_attr, batch)
        logits = self.classifier(emb)
        return logits, emb

    def forward(self, x, edge_index, edge_attr, batch, **kwargs):
        logits, _ = self._forward_with_embedding(x, edge_index, edge_attr, batch, **kwargs)
        return logits

def _make_structural_learning_graphs(
    num_graphs: int,
    num_nodes: int = 12,
    node_corr: float = 0.92,
    node_scale: float = 3.0,
    edge_scale: float = 0.35,
) -> list[Data]:
    """Generate graphs where labels are easy to memorize from nodes unless regularized."""
    graphs = []
    src = torch.arange(num_nodes, dtype=torch.long)
    dst = torch.roll(src, shifts=-1)
    edge_index = torch.stack([src, dst], dim=0)

    for _ in range(num_graphs):
        y = int(torch.randint(0, 2, (1,)).item())
        label_sign = 1.0 if y == 1 else -1.0

        # Node cue: intentionally dominant magnitude so baseline prefers nodes.
        node_sign = label_sign if torch.rand(1).item() < node_corr else -label_sign
        x = node_scale * node_sign + 0.10 * torch.randn(num_nodes, 1)

        # Edge cue: weaker magnitude but perfectly label-aligned.
        edge_attr = edge_scale * label_sign + 0.02 * torch.randn(num_nodes, 1)

        graphs.append(
            Data(
                x=x.float(),
                edge_index=edge_index.clone(),
                edge_attr=edge_attr.float(),
                y=torch.tensor([y], dtype=torch.long),
            )
        )

    return graphs

def _train_toy_model(
    train_graphs: list[Data],
    use_structural_dropout: bool,
    init_state: dict,
) -> _ToyStructuralModel:
    model = _ToyStructuralModel()
    model.load_state_dict(copy.deepcopy(init_state))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.CrossEntropyLoss()
    loader = make_loader(train_graphs, batch_size=32, shuffle=False, num_workers=0)

    for _ in range(16):
        train_one_epoch_with_accumulation(
            model=model,
            loader=loader,
            optimizer=optimizer,
            criterion=criterion,
            device=torch.device("cpu"),
            gradient_accumulation_steps=1,
            structural_dropout_prob=1.0 if use_structural_dropout else 0.0,
            edge_contrastive_weight=1.0 if use_structural_dropout else 0.0,
        )

    return model

def _edge_to_node_importance_ratio(model: _ToyStructuralModel, eval_batch: Batch) -> float:
    model.eval()
    eval_batch = eval_batch.to(torch.device("cpu"))

    attributor = GradientEdgeAttributor(model, target_class=1)
    edge_scores = attributor.compute(
        eval_batch.x,
        eval_batch.edge_index,
        eval_batch.edge_attr,
        eval_batch.batch,
    )
    edge_mean = float(edge_scores.mean().item())

    model.zero_grad(set_to_none=True)
    x = eval_batch.x.clone().detach().requires_grad_(True)
    out = model(x, eval_batch.edge_index, eval_batch.edge_attr, eval_batch.batch)
    out[:, 1].sum().backward()
    node_mean = float(x.grad.abs().mean().item())

    return edge_mean / (node_mean + 1e-8)

class TestStructuralLearningBehavior:
    def test_structural_dropout_increases_edge_to_node_ratio(self):
        """Structural dropout should increase edge importance relative to node importance."""
        torch.manual_seed(7)
        np.random.seed(7)

        train_graphs = _make_structural_learning_graphs(
            num_graphs=192,
            node_corr=0.92,
            node_scale=3.0,
            edge_scale=0.35,
        )
        eval_graphs = _make_structural_learning_graphs(
            num_graphs=64,
            node_corr=0.92,
            node_scale=3.0,
            edge_scale=0.35,
        )
        eval_loader = make_loader(eval_graphs, batch_size=64, shuffle=False, num_workers=0)
        eval_batch = next(iter(eval_loader))

        init_state = _ToyStructuralModel().state_dict()
        baseline_model = _train_toy_model(
            train_graphs,
            use_structural_dropout=False,
            init_state=init_state,
        )
        structural_model = _train_toy_model(
            train_graphs,
            use_structural_dropout=True,
            init_state=init_state,
        )

        baseline_ratio = _edge_to_node_importance_ratio(baseline_model, eval_batch)
        structural_ratio = _edge_to_node_importance_ratio(structural_model, eval_batch)

        assert structural_ratio > baseline_ratio, (
            f"Expected structural dropout to raise edge-to-node importance ratio; "
            f"baseline={baseline_ratio:.4f}, structural={structural_ratio:.4f}"
        )
