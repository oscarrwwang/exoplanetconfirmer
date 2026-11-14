from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from src.preprocessing.preprocess import (
    KEPLER_LONG_CADENCE_DAYS,
    PreprocessConfig,
    PreprocessingError,
    Segment,
    detrend,
    median_normalize,
    pad_sequence,
    preprocess_segments,
    process_star,
    remove_invalid,
    resample,
    sigma_clip,
)

CADENCE = KEPLER_LONG_CADENCE_DAYS


def synthetic_quarter(
    t0: float = 0.0,
    days: float = 10.0,
    level: float = 1000.0,
    slope: float = 0.01,
    depth: float = 0.01,
    transit_center: float = 5.0,
    transit_duration: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = t0 + np.arange(0, days, CADENCE)
    trend = level * (1 + slope * (time - t0))
    in_transit = np.abs(time - (t0 + transit_center)) < transit_duration / 2
    flux = trend * np.where(in_transit, 1 - depth, 1.0)
    return time, flux, in_transit


def test_remove_invalid_drops_nans_and_flags():
    time = np.array([0.0, 1.0, np.nan, 3.0, 4.0])
    flux = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
    quality = np.array([0, 0, 0, 0, 32])
    t, f = remove_invalid(time, flux, quality, quality_bitmask=32)
    assert t.tolist() == [0.0, 3.0]
    assert f.tolist() == [1.0, 4.0]
    t, _ = remove_invalid(time, flux, quality, quality_bitmask=0)
    assert t.tolist() == [0.0, 3.0, 4.0]


def test_remove_invalid_shape_mismatch():
    with pytest.raises(ValueError):
        remove_invalid(np.zeros(3), np.zeros(4))


def test_sigma_clip_removes_spikes_keeps_dips():
    rng = np.random.default_rng(0)
    flux = 1.0 + rng.normal(0, 1e-4, 1000)
    flux[100] = 1.1
    flux[200] = 0.9
    time = np.arange(flux.size, dtype=float)
    t, f = sigma_clip(time, flux, sigma_upper=3.0, sigma_lower=None)
    assert 100.0 not in t
    assert 200.0 in t
    t, _ = sigma_clip(time, flux, sigma_upper=3.0, sigma_lower=3.0)
    assert 200.0 not in t


def test_detrend_removes_trend_preserves_transit():
    time, flux, in_transit = synthetic_quarter()
    t, f = detrend(time, flux, window_days=1.0)
    assert t.size == time.size
    assert np.allclose(f[~in_transit], 1.0, atol=2e-3)
    assert np.mean(f[in_transit]) == pytest.approx(0.99, abs=2e-3)


def test_detrend_splits_on_gaps():
    t1, f1, _ = synthetic_quarter(t0=0.0, level=1000.0, depth=0.0)
    t2, f2, _ = synthetic_quarter(t0=20.0, level=3000.0, depth=0.0)
    t, f = detrend(np.concatenate([t1, t2]), np.concatenate([f1, f2]), gap_days=0.5)
    assert np.allclose(f, 1.0, atol=1e-3)


def test_median_normalize():
    out = median_normalize(np.array([2.0, 2.0, 1.0, 3.0]))
    assert np.median(out) == pytest.approx(0.0)
    assert out.tolist() == pytest.approx([0.0, 0.0, -0.5, 0.5])
    with pytest.raises(PreprocessingError):
        median_normalize(np.zeros(3))


def test_resample_uniform_grid_with_gap_mask():
    time = np.array([0.0, 1.0, 2.0, 5.0]) * CADENCE
    flux = np.array([1.0, 2.0, 3.0, 4.0])
    out, mask, t_start = resample(time, flux, CADENCE, fill_value=-9.0)
    assert t_start == 0.0
    assert out.tolist() == [1.0, 2.0, 3.0, -9.0, -9.0, 4.0]
    assert mask.tolist() == [True, True, True, False, False, True]


def test_resample_averages_duplicates():
    out, mask, _ = resample(np.array([0.0, 0.0]), np.array([1.0, 3.0]), CADENCE)
    assert out.tolist() == [2.0]
    assert mask.tolist() == [True]


def test_pad_sequence_pads_and_truncates():
    flux = np.arange(5, dtype=np.float32)
    mask = np.ones(5, bool)
    out, m = pad_sequence(flux, mask, 7, pad_value=-1)
    assert out.tolist() == [0, 1, 2, 3, 4, -1, -1]
    assert m.tolist() == [True] * 5 + [False] * 2
    out, m = pad_sequence(flux, mask, 3)
    assert out.tolist() == [0, 1, 2]
    assert m.all()


def test_config_validation():
    with pytest.raises(ValueError):
        PreprocessConfig(seq_len=0)
    with pytest.raises(ValueError):
        PreprocessConfig(sigma_lower=-1.0)


def test_pipeline_end_to_end():
    t1, f1, _ = synthetic_quarter(t0=0.0, level=1000.0)
    t2, f2, _ = synthetic_quarter(t0=12.0, level=5000.0, slope=-0.005)
    f1 = f1.copy()
    f1[10] = np.nan
    f1[20] = f1[20] * 1.5
    config = PreprocessConfig(seq_len=2000)
    result = preprocess_segments([Segment(t1, f1), Segment(t2, f2)], config)

    assert result.flux.shape == (2000,)
    assert result.flux.dtype == np.float32
    assert result.mask.shape == (2000,)
    assert result.n_input == t1.size + t2.size
    assert result.n_valid == result.n_input - 1
    assert result.n_clipped >= 1
    assert np.isfinite(result.flux).all()
    assert result.flux[result.mask].min() == pytest.approx(-0.01, abs=2e-3)
    assert np.median(result.flux[result.mask]) == pytest.approx(0.0, abs=1e-3)
    # Gap between quarters and trailing padding are masked out.
    assert not result.mask.all()
    assert result.n_resampled < 2000


def test_pipeline_rejects_empty():
    with pytest.raises(PreprocessingError):
        preprocess_segments([Segment(np.array([0.0]), np.array([np.nan]))])


def write_fits(path: Path, time: np.ndarray, flux: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        fits.Column(name="TIME", format="D", array=time),
        fits.Column(name="PDCSAP_FLUX", format="E", array=flux),
        fits.Column(name="SAP_QUALITY", format="J", array=np.zeros(time.size, np.int32)),
    ]
    hdu = fits.BinTableHDU.from_columns(cols, name="LIGHTCURVE")
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)


def test_process_star_writes_and_skips(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "processed"
    time, flux, _ = synthetic_quarter()
    write_fits(raw / "kplr000000042" / "q1.fits", time, flux)
    (raw / "kplr000000042" / "broken.fits").write_bytes(b"garbage")
    config = PreprocessConfig(seq_len=1000)

    row = process_star(42, 1, raw, out, config)
    assert row["status"] == "ok"
    assert row["n_files"] == 2
    assert row["n_bad_files"] == 1
    with np.load(out / "kplr000000042.npz") as data:
        assert data["flux"].shape == (1000,)
        assert data["mask"].dtype == bool
        assert int(data["label"]) == 1
        assert int(data["kepid"]) == 42

    assert process_star(42, 1, raw, out, config)["status"] == "skipped"
    assert process_star(42, 1, raw, out, config, overwrite=True)["status"] == "ok"
    assert process_star(7, 0, raw, out, config)["status"] == "missing"
