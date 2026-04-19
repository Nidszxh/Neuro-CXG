from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional positive-class reweighting.

    Args:
        alpha: Weight assigned to class 1 (ASD). Class 0 receives (1 - alpha).
        gamma: Focusing parameter for hard-example mining.
        pos_weight: Optional multiplicative weight for class 1 examples.
    """

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 3.0,
        pos_weight: Optional[float] = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss from logits and class targets."""
        probs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        pt = (probs * targets_one_hot).sum(dim=1)

        focal_weight = (1.0 - pt) ** self.gamma
        alpha_weight = (
            targets_one_hot[:, 1] * self.alpha
            + targets_one_hot[:, 0] * (1.0 - self.alpha)
        )

        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        if self.pos_weight is not None:
            class_weight = targets_one_hot[:, 1] * self.pos_weight + targets_one_hot[:, 0]
            ce_loss = ce_loss * class_weight

        return (alpha_weight * focal_weight * ce_loss).mean()
