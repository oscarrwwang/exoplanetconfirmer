"""Loss functions for imbalanced binary classification. All take raw logits."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

LOSSES = ("bce", "weighted_bce", "focal")


class WeightedBCELoss(nn.Module):
    """BCE with a positive-class weight: pos_weight = n_neg / n_pos balances the classes."""

    def __init__(self, pos_weight: float = 1.0) -> None:
        super().__init__()
        if pos_weight <= 0:
            raise ValueError(f"pos_weight must be positive, got {pos_weight}")
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)


class FocalLoss(nn.Module):
    """Binary focal loss (Lin et al. 2017): alpha_t * (1 - p_t)^gamma * BCE.

    gamma down-weights well-classified examples; gamma=0 reduces to (alpha-weighted) BCE.
    alpha is the weight on positives (negatives get 1 - alpha); None disables it.
    """

    def __init__(self, gamma: float = 2.0, alpha: float | None = None) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        if alpha is not None and not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1) or None, got {alpha}")
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        # p_t = exp(-bce) is the probability assigned to the true class.
        p_t = torch.exp(-bce)
        loss = (1 - p_t).pow(self.gamma) * bce
        if self.alpha is not None:
            loss = (self.alpha * targets + (1 - self.alpha) * (1 - targets)) * loss
        return loss.mean()


def class_counts(targets: torch.Tensor) -> tuple[int, int]:
    n_pos = int((targets >= 0.5).sum())
    return n_pos, int(targets.numel()) - n_pos


def build_loss(
    name: str,
    targets: torch.Tensor,
    pos_weight: str | float = "auto",
    focal_gamma: float = 2.0,
    focal_alpha: str | float | None = "auto",
) -> nn.Module:
    """Create a loss. "auto" values are derived from the training-label class balance."""
    n_pos, n_neg = class_counts(targets)
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "weighted_bce":
        if pos_weight == "auto":
            if n_pos == 0 or n_neg == 0:
                raise ValueError("pos_weight=auto needs both classes in the training set")
            pos_weight = n_neg / n_pos
        return WeightedBCELoss(float(pos_weight))
    if name == "focal":
        if focal_alpha == "auto":
            if n_pos == 0 or n_neg == 0:
                raise ValueError("focal_alpha=auto needs both classes in the training set")
            focal_alpha = n_neg / (n_pos + n_neg)
        return FocalLoss(focal_gamma, None if focal_alpha is None else float(focal_alpha))
    raise ValueError(f"Unknown loss {name!r}; choose from {LOSSES}")
