import json

import numpy as np
import pandas as pd
import pytest
import torch
from astropy.io import fits

from src.inference import predict as pr
from src.models.astronet_style import AstronetStyle
from src.preprocessing.views import ViewConfig

P, T0, DUR = 3.7, 2.0, 0.15
GLOBAL_BINS, LOCAL_BINS = 201, 61


def write_quarter(path, kepid, quarter, t_start, secondary=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(t_start, t_start + 30, 0.0204)
    f = 1.0 + rng.normal(0, 2e-4, t.size)
    phase = (t - T0 + 0.5 * P) % P - 0.5 * P
    f[np.abs(phase) < 0.5 * DUR] -= 3e-3
    f[np.abs(np.abs(phase) - 0.5 * P) < 0.5 * DUR] -= secondary
    f[5:8] = np.nan
    primary = fits.PrimaryHDU()
    primary.header["KEPLERID"] = kepid
    primary.header["QUARTER"] = quarter
    cols = fits.ColDefs([
        fits.Column(name="TIME", format="D", array=t),
        fits.Column(name="PDCSAP_FLUX", format="E", array=f * 1e4),
        fits.Column(name="SAP_QUALITY", format="J", array=np.zeros(t.size, np.int32)),
    ])
    lc = fits.BinTableHDU.from_columns(cols, name="LIGHTCURVE")
    fits.HDUList([primary, lc]).writeto(path)


@pytest.fixture
def setup(tmp_path):
    star = tmp_path / "kplr000001234"
    star.mkdir()
    for q in range(3):
        write_quarter(star / f"q{q}_llc.fits", 1234, q + 1, 30.0 * q, seed=q)
    eb = tmp_path / "kplr000005678"
    eb.mkdir()
    for q in range(3):
        write_quarter(eb / f"q{q}_llc.fits", 5678, q + 1, 30.0 * q, secondary=1.5e-3, seed=q)

    config = ViewConfig(global_bins=GLOBAL_BINS, global_bin_width_factor=1 / GLOBAL_BINS,
                        local_bins=LOCAL_BINS)
    (tmp_path / "views_config.json").write_text(json.dumps(config.__dict__))
    torch.manual_seed(0)
    model = AstronetStyle(GLOBAL_BINS, LOCAL_BINS, global_channels=(4, 8, 16, 32, 64),
                          local_channels=(4, 8))
    torch.save({
        "model": model.state_dict(), "epoch": 1, "threshold": 0.5,
        "config": {"model": "astronet", "dropout": None, "batch_norm": False,
                   "kernel_size": None, "base_filters": 4, "run_name": "unit"},
    }, tmp_path / "best.pt")
    pd.DataFrame([
        {"kepid": 1234, "kepoi_name": "K01234.01", "koi_period": P, "koi_time0bk": T0,
         "koi_duration": DUR * 24},
        {"kepid": 1234, "kepoi_name": "K01234.02", "koi_period": 11.3, "koi_time0bk": 5.1,
         "koi_duration": 2.0},
        {"kepid": 9999, "kepoi_name": "K09999.01", "koi_period": 1.0, "koi_time0bk": 0.5,
         "koi_duration": 1.0},
    ]).to_csv(tmp_path / "kois.csv", index=False)
    meta_dir = tmp_path / "dataset" / "train"
    meta_dir.mkdir(parents=True)
    pd.DataFrame({"kepoi_name": ["K01234.01"], "kepid": [1234], "label": [1]}).to_csv(
        meta_dir / "meta.csv", index=False)
    return tmp_path


def common(setup):
    return ["--checkpoint", str(setup / "best.pt"), "--views-config",
            str(setup / "views_config.json"), "--data-dir", str(setup / "dataset"),
            "--device", "cpu"]


def test_explicit_ephemeris_report(setup):
    out = setup / "out"
    report = pr.main([str(setup / "kplr000001234"), *common(setup), "--period", str(P),
                      "--t0", str(T0), "--duration-hours", str(DUR * 24), "--out-dir", str(out)])
    (cand,) = report["candidates"]
    assert cand["status"] == "ok"
    assert 0 <= cand["planet_probability"] <= 1
    assert cand["model_classification"] == (
        "PLANET" if cand["planet_probability"] >= 0.5 else "FALSE POSITIVE")
    assert 0 <= cand["confidence_score"] <= 1
    assert cand["disposition"] in ("PLANET CANDIDATE", "FALSE POSITIVE", "NEEDS REVIEW")
    assert cand["features"]["bls_snr"] > 20
    assert "SECONDARY_ECLIPSE" not in cand["reason_codes"]
    assert report["light_curve"]["kepid"] == 1234 and report["light_curve"]["n_files"] == 3
    assert report["in_dataset"] == {"train": ["K01234.01"]}
    saved = json.loads((out / "report.json").read_text())
    assert saved["candidates"][0]["planet_probability"] == pytest.approx(
        cand["planet_probability"])
    text = (out / "report.txt").read_text(encoding="utf-8")
    assert "FINAL DISPOSITION" in text and "already in the model's dataset" in text
    assert (out / "candidate.png").exists()


def test_koi_table_masks_companions_and_selects_targets(setup):
    out = setup / "out_table"
    report = pr.main([str(setup / "kplr000001234"), *common(setup), "--koi-table",
                      str(setup / "kois.csv"), "--kepoi-name", "K01234.01", "--no-explain",
                      "--out-dir", str(out)])
    assert [c["candidate"] for c in report["candidates"]] == ["K01234.01"]
    assert "transit_evidence_share" not in report["candidates"][0]["features"]
    assert not list(out.glob("*.png"))


def test_eclipsing_binary_gets_secondary_eclipse_code(setup):
    report = pr.main([str(setup / "kplr000005678"), *common(setup), "--period", str(P),
                      "--t0", str(T0), "--duration-hours", str(DUR * 24), "--no-explain",
                      "--out-dir", str(setup / "out_eb")])
    cand = report["candidates"][0]
    assert "SECONDARY_ECLIPSE" in cand["reason_codes"]
    assert cand["disposition"] in ("FALSE POSITIVE", "NEEDS REVIEW")


def test_rejects_mixed_stars_and_bad_arguments(setup):
    files = [setup / "kplr000001234" / "q0_llc.fits", setup / "kplr000005678" / "q0_llc.fits"]
    with pytest.raises(ValueError, match="different stars"):
        pr.read_light_curve(files, "PDCSAP_FLUX")
    with pytest.raises(SystemExit):
        pr.parse_args(["x", "--checkpoint", "c.pt", "--period", "1"])
    with pytest.raises(SystemExit):
        pr.parse_args(["x", "--checkpoint", "c.pt", "--koi-table", "k.csv", "--period", "1",
                       "--t0", "0", "--duration-hours", "2"])


def test_wrong_view_config_is_explained(setup):
    with pytest.raises(ValueError, match="views_config"):
        pr.load_predictor(setup / "best.pt", setup / "missing.json", "cpu")
