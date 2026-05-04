"""Unit tests for Task 2 — CausalInvarianceLoss (DD-010).

Tests:
1. Loss is lower for same-subject view pairs than cross-subject pairs
2. Returns scalar with valid grad
3. Handles V=1 gracefully (returns 0)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
import torch

from src.models.losses import CausalInvarianceLoss


class TestCausalInvarianceLoss:
    def setup_method(self):
        self.loss_fn = CausalInvarianceLoss(temperature=0.07)

    def test_same_subject_views_lower_loss(self):
        """Loss must be lower when views are from the same subjects (close embeddings)."""
        torch.manual_seed(0)
        B, D = 12, 64

        # Same-subject views: small noise perturbation
        z_base = torch.randn(B, D)
        z_view_same = z_base + 0.01 * torch.randn(B, D)
        loss_same = self.loss_fn([z_base, z_view_same])

        # Cross-subject views: independent random embeddings
        z_random = torch.randn(B, D)
        loss_random = self.loss_fn([z_base, z_random])

        assert loss_same.item() < loss_random.item(), (
            f"Same-view loss ({loss_same:.4f}) must be < random-view loss ({loss_random:.4f})"
        )

    def test_scalar_output(self):
        B, D = 8, 32
        z0 = torch.randn(B, D)
        z1 = torch.randn(B, D)
        loss = self.loss_fn([z0, z1])
        assert loss.dim() == 0, "Loss must be scalar"

    def test_gradient_flows(self):
        B, D = 8, 32
        z0 = torch.randn(B, D, requires_grad=True)
        z1 = torch.randn(B, D, requires_grad=True)
        loss = self.loss_fn([z0, z1])
        loss.backward()
        assert z0.grad is not None
        assert z1.grad is not None
        assert not torch.isnan(z0.grad).any()

    def test_single_view_returns_zero(self):
        """V=1 (no second view to contrast) should return 0."""
        z0 = torch.randn(8, 32)
        loss = self.loss_fn([z0])
        assert loss.item() == pytest.approx(0.0)

    def test_multiview_V3(self):
        """Should handle V=3 views without error."""
        B, D = 8, 32
        views = [torch.randn(B, D) for _ in range(3)]
        loss = self.loss_fn(views)
        assert loss.dim() == 0
        assert torch.isfinite(loss), "Loss must be finite for V=3"
