from __future__ import annotations

from typing import List, Optional

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
        if inputs.dim() != 2:
            raise ValueError(f"FocalLoss expects logits shape (B, C), got {tuple(inputs.shape)}")

        num_classes = int(inputs.size(1))
        if num_classes < 2:
            raise ValueError(f"FocalLoss requires at least 2 classes, got C={num_classes}")

        targets = targets.view(-1).long()
        if targets.numel() != int(inputs.size(0)):
            raise ValueError(
                "FocalLoss target/logit batch mismatch: "
                f"targets={targets.numel()} logits={int(inputs.size(0))}"
            )

        # Validate on CPU first so invalid labels produce a clear Python error
        # instead of an opaque CUDA device-side assert.
        t_cpu = targets.detach().to(device="cpu")
        bad = (t_cpu < 0) | (t_cpu >= num_classes)
        if bool(bad.any()):
            bad_vals = sorted({int(v) for v in t_cpu[bad].tolist()})
            raise ValueError(
                f"FocalLoss received out-of-range targets {bad_vals} for num_classes={num_classes}. "
                "Expected labels in [0, num_classes-1]."
            )

        probs = F.softmax(inputs, dim=1).clamp(min=1e-8, max=1.0)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
        pt = (probs * targets_one_hot).sum(dim=1).clamp(min=1e-8, max=1.0)

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


def build_criterion(
    train_labels,
    device: torch.device,
    use_focal_loss: bool = True,
    use_class_weights: bool = False,
    focal_alpha: float = 0.75,
    focal_gamma: float = 3.0,
) -> nn.Module:
    """Build loss criterion based on config and training label distribution.
    
    Args:
        train_labels: List or array of labels (0=Control, 1=ASD)
        device: Torch device for class weights tensor
        use_focal_loss: Whether to use FocalLoss (else CrossEntropyLoss)
        use_class_weights: Whether to apply class reweighting
        focal_alpha: Focal loss alpha parameter
        focal_gamma: Focal loss gamma parameter
    
    Returns:
        Initialized loss module
    """
    labels_arr = np.array(train_labels)
    n_control = max(int((labels_arr == 0).sum()), 1)
    n_asd = max(int((labels_arr == 1).sum()), 1)
    
    class_weight_tensor = None
    if use_class_weights:
        total = max(len(labels_arr), 1)
        class_weight_tensor = torch.tensor(
            [total / (2 * n_control), total / (2 * n_asd)],
            dtype=torch.float32,
            device=device,
        )
    
    if use_focal_loss:
        pos_weight = float(n_control / n_asd) if use_class_weights else None
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma, pos_weight=pos_weight)
    else:
        return nn.CrossEntropyLoss(weight=class_weight_tensor)


# ── Causal Invariance Loss (DD-010) ───────────────────────────────────────────────

class CausalInvarianceLoss(nn.Module):
    """NT-Xent contrastive loss across multiple causal graph views of the same subject.

    Args:
        temperature: Softmax temperature τ. Default 0.07.
    """

    _VIEW_ORDER = (
        "base",
        "extended_lag",
        "bootstrap_0",
        "bootstrap_1",
        "bootstrap_2",
        "high_confidence",
    )

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings_list: List[torch.Tensor]) -> torch.Tensor:
        """Compute NT-Xent invariance loss across views.

        Args:
            embeddings_list: List of V tensors each of shape (B, D), one per view.
                            B graphs must correspond to the same subjects across views.
        Returns:
            Scalar NT-Xent invariance loss.
        """
        V = len(embeddings_list)
        if V < 2:
            return torch.tensor(0.0, device=embeddings_list[0].device, requires_grad=True)

        B = embeddings_list[0].size(0)
        if B < 2:
            return torch.tensor(0.0, device=embeddings_list[0].device, requires_grad=True)

        zs = [F.normalize(e, dim=1) for e in embeddings_list]

        pair_losses = []
        labels = torch.arange(B, device=zs[0].device)
        for i in range(V):
            for j in range(i + 1, V):
                sim_ij = torch.mm(zs[i], zs[j].t()) / self.temperature
                pair_losses.append(F.cross_entropy(sim_ij, labels))
                pair_losses.append(F.cross_entropy(sim_ij.t(), labels))

        if not pair_losses:
            return torch.tensor(0.0, device=zs[0].device, requires_grad=True)

        return torch.stack(pair_losses).mean()


# ── Spatial Invariance Loss (DD-012) ──────────────────────────────────────

class SpatialInvarianceLoss(nn.Module):
    """Gradient reversal applied to spatial feature slice for site invariance.

    Args:
        spatial_start_idx: First column index of the spatial feature block in x.
        num_sites: Number of acquisition sites for the site classifier head.
        reversal_weight: Gradient reversal strength (λ). Default 0.1.
    """

    def __init__(
        self,
        spatial_start_idx: int,
        num_sites: int = 20,
        reversal_weight: float = 0.1,
    ):
        super().__init__()
        self.spatial_start_idx = spatial_start_idx
        self.reversal_weight = reversal_weight
        self.site_head = nn.Sequential(
            nn.Linear(4, 16),
            nn.GELU(),
            nn.Linear(16, num_sites),
        )

    def forward(self, x: torch.Tensor, site_targets: torch.Tensor) -> torch.Tensor:
        """Compute site classification loss on reversed-gradient spatial features.

        Args:
            x: Node feature matrix (N, F).
            site_targets: Integer site labels per node (N,).
        Returns:
            Scalar site classification loss.
        """
        from src.models.causal_gnn import GradientReversal

        spatial = x[:, self.spatial_start_idx:]
        spatial_rev = GradientReversal.apply(spatial, self.reversal_weight)
        site_logits = self.site_head(spatial_rev)
        return F.cross_entropy(site_logits, site_targets)


# ── Edge Structure Contrastive Loss (DD-009) ───────────────────────────

class EdgeStructureContrastiveLoss(nn.Module):
    """NT-Xent style contrastive loss for structural learning.

    Args:
        temperature: Softmax temperature τ. Default 0.5.
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_full: torch.Tensor,
        z_edge_only: torch.Tensor,
    ) -> torch.Tensor:
        """Compute alignment loss between full-feature and edge-only views.

        Args:
            z_full: Embeddings from full-feature pass, shape (B, D).
            z_edge_only: Embeddings from edge-only pass, shape (B, D).
        Returns:
            Scalar contrastive loss.
        """
        B = z_full.size(0)
        if B < 2:
            return torch.tensor(0.0, device=z_full.device, requires_grad=True)

        z_f = F.normalize(z_full, dim=1)
        z_e = F.normalize(z_edge_only, dim=1)

        z = torch.cat([z_f, z_e], dim=0)

        sim = torch.mm(z, z.t()) / self.temperature

        eye = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(eye, float('-inf'))

        labels = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(0, B, device=z.device),
        ])

        return F.cross_entropy(sim, labels)
