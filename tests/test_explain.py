import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from src.data.build_dataset import save_split
from src.evaluation import explain as ex
from src.models.astronet_style import AstronetStyle
from src.models.baseline_cnn import BaselineCNN
from src.preprocessing.views import ViewConfig
from src.training import train as train_module

G, L = 2001, 201


class DipProbe(nn.Module):
    """logit = -sum(flux[100:110]): a dip in bins 100-109 means 'planet'; nothing else matters."""

    def forward(self, x_global, x_local):
        return -x_global[:, 0, 100:110].sum(-1) + 0 * x_local.sum((1, 2))


def inputs(n=3, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(n, 1, G, generator=gen) * 0.1, torch.randn(n, 1, L, generator=gen) * 0.1


@pytest.mark.parametrize("model", [AstronetStyle(), BaselineCNN()])
def test_grad_cam_shapes_range_and_no_side_effects(model):
    model.eval()
    xg, xl = inputs()
    cam = ex.grad_cam(model, xg, xl)
    assert set(cam) == set(ex.model_views(model))
    assert cam["global"].shape == (3, G)
    if "local" in cam:
        assert cam["local"].shape == (3, L)
    for a in cam.values():
        assert torch.isfinite(a).all() and a.min() >= 0 and a.max() <= 1
    assert all(p.grad is None for p in model.parameters())
    assert all(not layer._forward_hooks for layer in ex.gradcam_layers(model).values())


def test_saliency_and_occlusion_localize_the_dip():
    model = DipProbe()
    xg, xl = inputs()
    xg[:, 0, 100:110] -= 1.0
    for samples in (0, 5):
        sal = ex.saliency(model, xg, xl, smooth_samples=samples)["global"]
        assert torch.allclose(sal[:, 100:110], torch.ones(3, 10))
        assert sal[:, :100].max() == 0 and sal[:, 110:].max() == 0
    occ = ex.occlusion(model, xg, xl, {"global": (5, 1), "local": (5, 1)})
    assert (occ["global"][:, 100:110] > 0).all()
    assert occ["global"][:, :95].abs().max() == 0 and occ["global"][:, 115:].abs().max() == 0
    assert occ["local"].abs().max() == 0
    blank = ex.view_importance(model, xg, xl)
    assert (blank["global"] > 0).all() and (blank["local"] == 0).all()


def test_region_masks_match_view_geometry():
    config = ViewConfig()
    masks = ex.region_masks("global", G, period=10.0, duration=0.5, config=config)
    assert masks["transit"].mean() == pytest.approx(0.05, abs=0.002)
    assert masks["secondary"].mean() == pytest.approx(0.05, abs=0.002)
    assert masks["transit"][G // 2] and masks["secondary"][0] and masks["secondary"][-1]
    local = ex.region_masks("local", L, period=10.0, duration=0.5, config=config)
    # Local view spans +-4 durations, so the transit covers 1/8 of it.
    assert local["transit"].mean() == pytest.approx(1 / 8, abs=0.01)
    assert not local["secondary"].any()

    a = np.where(masks["transit"], 1.0, 0.0)
    shares = ex.region_shares(a, masks)
    assert shares["transit_share"] == pytest.approx(1.0)
    assert shares["transit_enrichment"] == pytest.approx(1 / masks["transit"].mean())


def test_select_examples_picks_most_confident():
    labels = np.array([1, 1, 0, 0, 1, 0])
    probs = np.array([0.9, 0.6, 0.8, 0.1, 0.2, 0.3])
    cats = ex.categorize(labels, probs, 0.5)
    assert cats.tolist() == ["TP", "TP", "FP", "TN", "FN", "TN"]
    picked = ex.select_examples(cats, probs, 1)
    assert picked["TP"].tolist() == [0] and picked["TN"].tolist() == [3]
    assert picked["FP"].tolist() == [2] and picked["FN"].tolist() == [4]


@pytest.fixture
def trained(tmp_path):
    rng = np.random.default_rng(0)
    views_dir = tmp_path / "views"
    views_dir.mkdir()
    for split, n in (("train", 24), ("val", 8)):
        y = np.arange(n) % 2
        g = rng.normal(0, 0.1, (n, G)).astype(np.float32)
        loc = rng.normal(0, 0.1, (n, L)).astype(np.float32)
        g[y == 1, 990:1010] -= 1.0
        loc[y == 1, 88:113] -= 1.0
        names = [f"{split}{i}" for i in range(n)]
        save_split(tmp_path / "dataset" / split, g, loc,
                   pd.DataFrame({"kepoi_name": names, "kepid": range(n), "label": y}))
        for name in names[:-1]:  # last KOI has no view file
            np.savez(views_dir / f"{name}.npz", period=10.0, duration=0.2)
    args = train_module.parse_args([
        "--data-dir", str(tmp_path / "dataset"), "--checkpoint-dir", str(tmp_path / "ckpt"),
        "--log-dir", str(tmp_path / "runs"), "--model", "astronet", "--base-filters", "4",
        "--epochs", "1", "--device", "cpu", "--run-name", "r", "--skip-test",
    ])
    train_module.train(args)
    return tmp_path


def test_explain_cli_writes_outputs(trained):
    out = trained / "explain"
    result = ex.main([
        "--checkpoint", str(trained / "ckpt" / "r" / "best.pt"),
        "--data-dir", str(trained / "dataset"), "--views-dir", str(trained / "views"),
        "--out-dir", str(out), "--device", "cpu", "--examples", "1", "--smooth-samples", "2",
        "--occlusion-global", "101", "200", "--occlusion-local", "21", "20",
    ])
    assert result["metrics"]["n"] == 8
    for name in ("metrics.json", "class_mean_attribution.png", "region_importance.png",
                 "region_importance.csv", "region_summary.csv", "view_importance.csv",
                 "attributions.npz"):
        assert (out / name).exists(), name
    assert list((out / "examples").glob("*.png"))
    regions = pd.read_csv(out / "region_importance.csv")
    assert regions.kepoi_name.nunique() == 7
    assert set(regions.method) == set(ex.METHODS) and set(regions.view) == {"global", "local"}
    with np.load(out / "attributions.npz") as data:
        assert data["gradcam_global"].shape == (8, G)
        assert data["occlusion_local"].shape == (8, L)


def test_test_split_requires_final_flag():
    with pytest.raises(SystemExit):
        ex.parse_args(["--checkpoint", "x.pt", "--split", "test"])
    assert ex.parse_args(["--checkpoint", "x.pt", "--split", "test", "--final-test"]).final_test
