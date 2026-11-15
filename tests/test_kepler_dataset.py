from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from astropy.io import fits

from src.data.kepler_dataset import (
    KeplerDataset,
    pad_or_truncate,
    stitch_quarters,
    star_dir,
)


def write_lightcurve(
    raw_dir: Path, kepid: int, quarter: int, time: np.ndarray, flux: np.ndarray
) -> Path:
    directory = star_dir(raw_dir, kepid)
    directory.mkdir(parents=True, exist_ok=True)
    columns = [
        fits.Column(name="TIME", format="D", array=time),
        fits.Column(name="PDCSAP_FLUX", format="E", array=flux),
    ]
    table = fits.BinTableHDU.from_columns(columns, name="LIGHTCURVE")
    path = directory / f"kplr{kepid:09d}-q{quarter:02d}_llc.fits"
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path)
    return path


def write_labels(path: Path, rows: list[tuple[int, int]]) -> Path:
    pd.DataFrame(
        {
            "kepid": [k for k, _ in rows],
            "label": [l for _, l in rows],
            "source": "test",
            "disposition": ["CONFIRMED" if l else "FALSE POSITIVE" for _, l in rows],
        }
    ).to_csv(path, index=False)
    return path


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[Path, Path]:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    t = np.arange(100, dtype=np.float64)
    write_lightcurve(raw_dir, 1, 1, t, np.full(100, 1000.0))
    write_lightcurve(raw_dir, 2, 1, t, np.full(100, 500.0))
    labels = write_labels(tmp_path / "labels.csv", [(1, 1), (2, 0), (3, 1)])
    return labels, raw_dir


def test_shapes_and_dtypes(workspace):
    labels, raw_dir = workspace
    ds = KeplerDataset(labels, raw_dir, seq_len=64)
    flux, label = ds[0]
    assert flux.shape == (1, 64)
    assert flux.dtype == torch.float32
    assert label.shape == ()
    assert label.dtype == torch.float32


def test_labels_loaded(workspace):
    labels, raw_dir = workspace
    ds = KeplerDataset(labels, raw_dir, seq_len=64)
    assert ds.kepids == [1, 2]
    assert ds.labels.tolist() == [1.0, 0.0]
    assert [ds[i][1].item() for i in range(len(ds))] == [1.0, 0.0]


def test_flux_is_median_normalized(workspace):
    labels, raw_dir = workspace
    flux, _ = KeplerDataset(labels, raw_dir, seq_len=64)[0]
    assert torch.allclose(flux, torch.zeros_like(flux))


def test_missing_skip(workspace):
    labels, raw_dir = workspace
    ds = KeplerDataset(labels, raw_dir, missing="skip")
    assert len(ds) == 2
    assert ds.missing_kepids == [3]


def test_missing_zeros(workspace):
    labels, raw_dir = workspace
    ds = KeplerDataset(labels, raw_dir, seq_len=32, missing="zeros")
    assert len(ds) == 3
    flux, label = ds[2]
    assert torch.equal(flux, torch.zeros(1, 32))
    assert label.item() == 1.0


def test_missing_error(workspace):
    labels, raw_dir = workspace
    with pytest.raises(FileNotFoundError):
        KeplerDataset(labels, raw_dir, missing="error")


def test_invalid_missing_policy(workspace):
    labels, raw_dir = workspace
    with pytest.raises(ValueError):
        KeplerDataset(labels, raw_dir, missing="ignore")


def test_missing_label_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"kepid": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        KeplerDataset(path, tmp_path)


def test_nans_are_dropped(tmp_path):
    raw_dir = tmp_path / "raw"
    flux = np.array([100.0, np.nan, 110.0, 90.0], dtype=np.float64)
    write_lightcurve(raw_dir, 7, 1, np.arange(4, dtype=np.float64), flux)
    labels = write_labels(tmp_path / "labels.csv", [(7, 1)])
    out, _ = KeplerDataset(labels, raw_dir, seq_len=4)[0]
    assert torch.isfinite(out).all()
    np.testing.assert_allclose(out[0, :3].numpy(), [0.0, 0.1, -0.1], atol=1e-6)
    assert out[0, 3].item() == 0.0


def test_quarters_stitched_in_time_order(tmp_path):
    raw_dir = tmp_path / "raw"
    write_lightcurve(raw_dir, 5, 2, np.array([10.0, 11.0]), np.array([200.0, 220.0]))
    write_lightcurve(raw_dir, 5, 1, np.array([0.0, 1.0]), np.array([100.0, 100.0]))
    labels = write_labels(tmp_path / "labels.csv", [(5, 0)])
    flux, _ = KeplerDataset(labels, raw_dir, seq_len=4)[0]
    np.testing.assert_allclose(
        flux[0].numpy(), [0.0, 0.0, 200 / 210 - 1, 220 / 210 - 1], atol=1e-6
    )


def test_corrupt_file_is_skipped(tmp_path):
    raw_dir = tmp_path / "raw"
    write_lightcurve(raw_dir, 9, 1, np.arange(3, dtype=np.float64), np.full(3, 50.0))
    (star_dir(raw_dir, 9) / "kplr000000009-q02_llc.fits").write_bytes(b"not a fits file")
    labels = write_labels(tmp_path / "labels.csv", [(9, 1)])
    flux, _ = KeplerDataset(labels, raw_dir, seq_len=5)[0]
    assert torch.equal(flux, torch.zeros(1, 5))


def test_transform_applied(workspace):
    labels, raw_dir = workspace
    ds = KeplerDataset(labels, raw_dir, seq_len=8, transform=lambda x: x + 1)
    flux, _ = ds[0]
    assert torch.allclose(flux, torch.ones(1, 8))


def test_works_with_dataloader(workspace):
    labels, raw_dir = workspace
    ds = KeplerDataset(labels, raw_dir, seq_len=16)
    flux, label = next(iter(torch.utils.data.DataLoader(ds, batch_size=2)))
    assert flux.shape == (2, 1, 16)
    assert label.shape == (2,)


def test_pad_or_truncate():
    x = np.arange(5, dtype=np.float32)
    assert pad_or_truncate(x, 3).tolist() == [0, 1, 2]
    assert pad_or_truncate(x, 7, pad_value=-1).tolist() == [0, 1, 2, 3, 4, -1, -1]


def test_stitch_quarters_empty():
    assert stitch_quarters([]).size == 0
    assert stitch_quarters([(np.array([]), np.array([]))]).size == 0
