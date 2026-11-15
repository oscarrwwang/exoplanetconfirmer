import pytest
import torch

from src.models.astronet_style import AstronetStyle
from src.models.baseline_cnn import count_parameters


def test_forward_returns_raw_logits_and_predict_proba_returns_probabilities():
    model = AstronetStyle().eval()
    g, loc = torch.randn(3, 1, 2001), torch.randn(3, 1, 201)
    logits = model(g, loc)
    assert logits.shape == (3,)
    with torch.no_grad():
        model.classifier[-1].bias.fill_(10.0)
    assert (model(g, loc) > 1).all()
    probs = model.predict_proba(g, loc)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_accepts_inputs_without_channel_dim():
    model = AstronetStyle().eval()
    assert model(torch.randn(2, 2001), torch.randn(2, 201)).shape == (2,)


def test_feature_dims_and_parameter_count():
    model = AstronetStyle()
    assert (model.global_dim, model.local_dim) == (256 * 59, 32 * 46)
    assert count_parameters(model) == 9_940_193


def test_both_branches_receive_gradients():
    model = AstronetStyle()
    model.logits(torch.randn(2, 1, 2001), torch.randn(2, 1, 201)).sum().backward()
    assert model.global_branch[0].weight.grad.abs().sum() > 0
    assert model.local_branch[0].weight.grad.abs().sum() > 0


def test_batch_norm_variant_runs():
    model = AstronetStyle(batch_norm=True).train()
    assert model(torch.randn(4, 1, 2001), torch.randn(4, 1, 201)).shape == (4,)


def test_rejects_too_short_input():
    with pytest.raises(ValueError):
        AstronetStyle(local_length=10)
