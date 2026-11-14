import pytest
import torch

from src.models.baseline_cnn import BaselineCNN, count_parameters


def test_forward_returns_raw_logits_and_predict_proba_returns_probabilities():
    model = BaselineCNN().eval()
    x = torch.randn(3, 1, 2001)
    logits = model(x)
    assert logits.shape == (3,)
    probs = model.predict_proba(x)
    assert ((probs >= 0) & (probs <= 1)).all()
    assert torch.allclose(probs, torch.sigmoid(logits))


def test_forward_is_not_squashed():
    model = BaselineCNN().eval()
    with torch.no_grad():
        model.classifier[-1].bias.fill_(10.0)
    assert (model(torch.randn(2, 1, 2001)) > 1).all()


def test_accepts_inputs_without_channel_dim():
    model = BaselineCNN().eval()
    assert model(torch.randn(2, 2001)).shape == (2,)


def test_parameter_count():
    assert count_parameters(BaselineCNN()) == 300_705


def test_logits_alias_matches_forward():
    model = BaselineCNN().eval()
    x = torch.randn(2, 1, 2001)
    assert torch.allclose(model.logits(x), model(x))


def test_rejects_too_short_input():
    with pytest.raises(ValueError):
        BaselineCNN(input_length=100)
