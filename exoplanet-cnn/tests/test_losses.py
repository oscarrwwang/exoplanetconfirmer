import math

import pytest
import torch
import torch.nn.functional as F

from src.training.losses import FocalLoss, WeightedBCELoss, build_loss


def test_weighted_bce_matches_torch():
    logits = torch.tensor([2.0, -1.0, 0.5])
    y = torch.tensor([1.0, 0.0, 1.0])
    expected = F.binary_cross_entropy_with_logits(logits, y, pos_weight=torch.tensor(3.0))
    assert torch.allclose(WeightedBCELoss(3.0)(logits, y), expected)


def test_focal_gamma_zero_equals_bce():
    logits = torch.randn(20)
    y = (torch.rand(20) > 0.5).float()
    assert torch.allclose(FocalLoss(gamma=0)(logits, y), F.binary_cross_entropy_with_logits(logits, y))


def test_focal_downweights_easy_examples():
    y = torch.tensor([1.0])
    easy, hard = torch.tensor([4.0]), torch.tensor([-1.0])
    bce_ratio = F.binary_cross_entropy_with_logits(easy, y) / F.binary_cross_entropy_with_logits(hard, y)
    focal_ratio = FocalLoss(gamma=2)(easy, y) / FocalLoss(gamma=2)(hard, y)
    assert focal_ratio < bce_ratio / 10


def test_focal_alpha_weights_classes():
    loss = FocalLoss(gamma=0, alpha=0.75)
    pos = loss(torch.tensor([0.0]), torch.tensor([1.0]))
    neg = loss(torch.tensor([0.0]), torch.tensor([0.0]))
    assert math.isclose(pos.item() / neg.item(), 3.0, rel_tol=1e-6)


def test_focal_is_finite_for_extreme_logits():
    out = FocalLoss()(torch.tensor([100.0, -100.0]), torch.tensor([0.0, 1.0]))
    assert torch.isfinite(out)


def test_build_loss_auto_values():
    y = torch.tensor([1.0, 0.0, 0.0, 0.0])
    assert build_loss("weighted_bce", y).pos_weight.item() == 3.0
    assert build_loss("focal", y).alpha == 0.75
    assert build_loss("focal", y, focal_alpha=None).alpha is None
    assert build_loss("weighted_bce", y, pos_weight=2.5).pos_weight.item() == 2.5
    assert isinstance(build_loss("bce", y), torch.nn.BCEWithLogitsLoss)


def test_build_loss_rejects_bad_config():
    with pytest.raises(ValueError):
        build_loss("weighted_bce", torch.zeros(4))
    with pytest.raises(ValueError):
        build_loss("focal", torch.tensor([1.0, 0.0]), focal_alpha=1.5)
    with pytest.raises(ValueError):
        build_loss("hinge", torch.tensor([1.0, 0.0]))
