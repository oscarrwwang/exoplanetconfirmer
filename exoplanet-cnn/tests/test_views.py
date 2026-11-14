from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from src.preprocessing.preprocess import KEPLER_LONG_CADENCE_DAYS, PreprocessingError, Segment
from src.preprocessing.views import (
    Ephemeris,
    ViewConfig,
    bin_view,
    fill_empty_bins,
    generate_views,
    normalize_views,
    phase_fold,
    process_star_views,
    transit_mask,
)

PLANET = Ephemeris(period=3.7, t0=1.3, duration=0.15)
DEPTH = 0.005


def box_transit(time: np.ndarray, eph: Ephemeris, depth: float) -> np.ndarray:
    return np.where(transit_mask(time, eph, factor=0.5), 1.0 - depth, 1.0)


def synthetic_star(
    days: float = 90.0, others: tuple[tuple[Ephemeris, float], ...] = (), seed: int = 0
) -> Segment:
    rng = np.random.default_rng(seed)
    time = np.arange(0.0, days, KEPLER_LONG_CADENCE_DAYS)
    stellar = 1000.0 * (1 + 0.001 * time + 0.002 * np.sin(2 * np.pi * time / 10.0))
    signal = box_transit(time, PLANET, DEPTH)
    for eph, depth in others:
        signal = signal * box_transit(time, eph, depth)
    noise = 1 + rng.normal(0, 1e-5, time.size)
    return Segment(time=time, flux=stellar * signal * noise)


def test_phase_fold_centers_transits_at_zero():
    epochs = PLANET.t0 + PLANET.period * np.arange(-3, 4)
    assert np.allclose(phase_fold(epochs, PLANET.period, PLANET.t0), 0.0, atol=1e-12)
    phase = phase_fold(np.linspace(-50, 50, 10_001), PLANET.period, PLANET.t0)
    assert phase.min() >= -PLANET.period / 2
    assert phase.max() < PLANET.period / 2


def test_transit_mask_width():
    t = PLANET.t0 + np.array([0.0, 0.07, 0.08, 0.2])
    assert transit_mask(t, PLANET, factor=0.5).tolist() == [True, True, False, False]


def test_ephemeris_validation():
    with pytest.raises(ValueError):
        Ephemeris(period=1.0, t0=0.0, duration=2.0)
    with pytest.raises(ValueError):
        Ephemeris(period=np.nan, t0=0.0, duration=0.1)


def test_bin_view_edges_and_medians():
    phase = np.array([-0.95, -0.9, 0.0, 0.05, 0.9])
    flux = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    view, counts = bin_view(phase, flux, num_bins=3, bin_width=0.2, x_min=-1.0, x_max=1.0)
    # Bins: [-1, -0.8), [-0.1, 0.1), [0.8, 1.0)
    assert counts.tolist() == [2, 2, 1]
    assert view.tolist() == [2.0, 6.0, 9.0]


def test_bin_view_rejects_bad_width():
    with pytest.raises(ValueError):
        bin_view(np.zeros(3), np.zeros(3), 10, bin_width=3.0, x_min=-1.0, x_max=1.0)


def test_fill_empty_bins():
    out = fill_empty_bins(np.array([np.nan, 1.0, np.nan, 3.0, np.nan]))
    assert out.tolist() == [1.0, 1.0, 2.0, 3.0, 3.0]
    with pytest.raises(PreprocessingError):
        fill_empty_bins(np.full(3, np.nan))


def test_normalize_views_shared_and_per_view():
    g = np.array([1.0, 1.0, 0.996, 1.0, 1.0])
    loc = np.array([1.0, 0.995, 0.995, 1.0, 1.0])
    gn, ln, depth = normalize_views(g, loc, "shared")
    assert depth == pytest.approx(0.005)
    assert ln.min() == pytest.approx(-1.0)
    assert gn.min() == pytest.approx(-0.8)
    gn, _, _ = normalize_views(g, loc, "per_view")
    assert gn.min() == pytest.approx(-1.0)


def test_normalize_flat_view_does_not_divide_by_zero():
    gn, ln, depth = normalize_views(np.ones(5), np.ones(5))
    assert depth == 0.0
    assert np.all(gn == 0) and np.all(ln == 0)


def test_generate_views_shapes_and_centering():
    views = generate_views([synthetic_star()], PLANET)
    g, loc = views.global_flux, views.local_flux
    assert g.shape == (2001,) and g.dtype == np.float32
    assert loc.shape == (201,) and loc.dtype == np.float32

    assert np.argmin(loc) in range(85, 116)
    assert loc[100] == pytest.approx(-1.0, abs=0.1)
    assert abs(np.median(loc)) < 0.05
    assert abs(loc[0]) < 0.1 and abs(loc[-1]) < 0.1

    assert g[1000] < -0.8
    assert abs(np.median(g)) < 0.05
    assert views.depth == pytest.approx(DEPTH, rel=0.1)
    assert views.local_coverage == 1.0


def test_generate_views_masks_other_planets():
    other = Ephemeris(period=5.3, t0=0.7, duration=0.12)
    star = synthetic_star(others=((other, 0.01),))
    views = generate_views([star], PLANET, others=[other])
    away_from_transit = np.r_[views.global_flux[:950], views.global_flux[1051:]]
    assert away_from_transit.min() > -0.3
    assert views.local_flux[100] == pytest.approx(-1.0, abs=0.1)


def test_generate_views_without_transit_coverage():
    seg = synthetic_star(days=3.0)
    far = Ephemeris(period=50.0, t0=30.0, duration=0.2)
    with pytest.raises(PreprocessingError):
        generate_views([seg], far)


def test_process_star_views_writes_npz(tmp_path: Path):
    raw, out = tmp_path / "raw", tmp_path / "views"
    seg = synthetic_star()
    star = raw / "kplr000000042"
    star.mkdir(parents=True)
    cols = [
        fits.Column(name="TIME", format="D", array=seg.time),
        fits.Column(name="PDCSAP_FLUX", format="E", array=seg.flux),
    ]
    hdu = fits.BinTableHDU.from_columns(cols, name="LIGHTCURVE")
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(star / "q1.fits")

    koi = {
        "kepid": 42,
        "kepoi_name": "K00042.01",
        "label": 1,
        "koi_period": PLANET.period,
        "koi_time0bk": PLANET.t0,
        "koi_duration": PLANET.duration * 24.0,
        "koi_depth": DEPTH * 1e6,
    }
    bad = {**koi, "kepoi_name": "K00042.02", "koi_period": np.nan}
    rows = process_star_views(42, [koi, bad], raw, out, ViewConfig())
    status = {r["kepoi_name"]: r["status"] for r in rows}
    assert status == {"K00042.01": "ok", "K00042.02": "bad_ephemeris"}

    with np.load(out / "K00042.01.npz") as data:
        assert data["global_flux"].shape == (2001,)
        assert data["local_flux"].shape == (201,)
        assert int(data["label"]) == 1
        assert str(data["kepoi_name"]) == "K00042.01"

    again = process_star_views(42, [koi], raw, out, ViewConfig())
    assert again[0]["status"] == "skipped"
    missing = process_star_views(7, [{**koi, "kepid": 7}], raw, tmp_path / "v2", ViewConfig())
    assert missing[0]["status"] == "missing"
