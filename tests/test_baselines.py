import numpy as np
import pandas as pd
import pytest

from src.baselines import compare as cmp
from src.baselines.features import FEATURE_COLUMNS, bls_statistics, view_features
from src.data.build_dataset import save_split
from src.preprocessing.views import Ephemeris
from src.training import train as train_module

EPH = Ephemeris(period=5.0, t0=1.0, duration=0.2)


def light_curve(depth_odd=1e-3, depth_even=1e-3, secondary=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, 90.0, 0.02043)
    f = 1.0 + rng.normal(0, 1e-4, t.size)
    phase = (t - EPH.t0 + 0.5 * EPH.period) % EPH.period - 0.5 * EPH.period
    epoch = np.round((t - EPH.t0) / EPH.period).astype(int)
    in_transit = np.abs(phase) < 0.5 * EPH.duration
    f[in_transit] -= np.where(epoch[in_transit] % 2 == 0, depth_even, depth_odd)
    f[np.abs(np.abs(phase) - 0.5 * EPH.period) < 0.5 * EPH.duration] -= secondary
    return t, f


def test_bls_statistics_flag_eclipsing_binaries():
    planet = bls_statistics(*light_curve(), EPH)
    assert planet["bls_depth_ppm"] == pytest.approx(1000, rel=0.05)
    assert planet["bls_snr"] > 50
    assert planet["bls_n_transits"] == 18

    odd_even = bls_statistics(*light_curve(depth_odd=2e-3), EPH)
    assert odd_even["bls_odd_even_sigma"] > 5 * max(planet["bls_odd_even_sigma"], 1)

    secondary = bls_statistics(*light_curve(secondary=5e-4), EPH)
    assert secondary["bls_secondary_snr"] > 20
    assert abs(planet["bls_secondary_snr"]) < 5


def test_view_features_shape_statistics():
    local = np.zeros(201)
    box = local.copy()
    box[88:113] = -1.0
    vee = local.copy()
    vee[88:113] = -(1 - np.abs(np.linspace(-1, 1, 25)))
    glob = np.zeros(2001)
    glob[990:1011] = -1.0
    glob[:10] = glob[-10:] = -0.3
    f_box = view_features(glob, box, period=10.0, duration=0.1)
    f_vee = view_features(glob, vee, period=10.0, duration=0.1)
    assert f_box["local_v_shape"] == pytest.approx(1.0)
    assert f_vee["local_v_shape"] > f_box["local_v_shape"]  # V: deeper centre than mean
    assert f_box["local_asymmetry"] == pytest.approx(0.0)
    assert f_box["global_secondary_depth"] == pytest.approx(-0.3)


def test_rule_scores_rank_missing_fits_lowest():
    feats = pd.DataFrame({
        "bls_snr": [10.0, np.nan, 3.0],
        "bls_odd_even_sigma": [0.5, np.nan, 8.0],
        "bls_secondary_snr": [-1.0, np.nan, 2.0],
        "bls_status": ["ok", "failed", "ok"],
    })
    snr = cmp.bls_snr_score(feats)
    assert snr[0] > snr[2] > snr[1]
    vet = cmp.bls_vetting_score(feats)
    assert vet[0] == pytest.approx(-0.5) and vet[2] == pytest.approx(-8.0) and vet[1] < -8.0


@pytest.fixture
def setup(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for split, n in (("train", 40), ("val", 16), ("test", 16)):
        y = np.arange(n) % 2
        g = rng.normal(0, 0.1, (n, 2001)).astype(np.float32)
        loc = rng.normal(0, 0.1, (n, 201)).astype(np.float32)
        g[y == 1, 990:1010] -= 1.0
        names = [f"{split}{i}" for i in range(n)]
        save_split(tmp_path / "dataset" / split, g, loc,
                   pd.DataFrame({"kepoi_name": names, "kepid": range(n), "label": y}))
        for name, label in zip(names, y):
            f = {c: rng.normal() for c in FEATURE_COLUMNS}
            f["bls_odd_even_sigma"] = abs(rng.normal()) + (0 if label else 4)
            f["bls_snr"] = 10 + 5 * label + rng.normal()
            rows.append({"kepoi_name": name, "kepid": 0, **f, "bls_status": "ok", "bls_error": ""})
    pd.DataFrame(rows).to_csv(tmp_path / "features.csv", index=False)
    args = train_module.parse_args([
        "--data-dir", str(tmp_path / "dataset"), "--checkpoint-dir", str(tmp_path / "ckpt"),
        "--log-dir", str(tmp_path / "runs"), "--model", "baseline", "--epochs", "1",
        "--device", "cpu", "--run-name", "r", "--skip-test",
    ])
    train_module.train(args)
    return tmp_path


def run(setup, *extra):
    return cmp.main([
        "--data-dir", str(setup / "dataset"), "--features", str(setup / "features.csv"),
        "--koi-table", str(setup / "unused.csv"), "--views-dir", str(setup / "views"),
        "--checkpoint", str(setup / "ckpt" / "r" / "best.pt"), "--out-dir", str(setup / "out"),
        "--device", "cpu", "--bootstrap", "50", "--workers", "1", *extra,
    ])


def test_compare_val_produces_results_table(setup):
    results = run(setup)
    assert results.model.tolist() == ["BLS SNR", "BLS vetting", "Random Forest", "XGBoost", "CNN"]
    assert (results.n == 16).all()
    # Synthetic features separate the classes perfectly.
    assert results.set_index("model").loc["BLS vetting", "roc_auc"] == pytest.approx(1.0)
    assert (results.pr_auc_lo <= results.pr_auc + 1e-12).all()
    out = setup / "out"
    for name in ("results.csv", "results.md", "predictions.csv", "pr_curves.png",
                 "roc_curves.png", "feature_importance.csv", "feature_importance.png"):
        assert (out / name).exists(), name
    assert "| CNN |" in (out / "results.md").read_text(encoding="utf-8")
    importance = pd.read_csv(out / "feature_importance.csv")
    assert set(importance.feature) == set(FEATURE_COLUMNS)


def test_compare_with_view_features(setup):
    results = run(setup, "--feature-set", "both")
    assert len(results) == 5
    assert len(pd.read_csv(setup / "out" / "feature_importance.csv")) == len(FEATURE_COLUMNS) + 2202


def test_missing_features_are_reported(setup):
    feats = pd.read_csv(setup / "features.csv")
    with pytest.raises(ValueError, match="No features"):
        cmp.design_matrix(setup / "dataset" / "val", feats[feats.kepoi_name != "val3"],
                          "engineered")


def test_test_split_requires_final_flag():
    with pytest.raises(SystemExit):
        cmp.parse_args(["--split", "test"])
    assert cmp.parse_args(["--split", "test", "--final-test"]).final_test
