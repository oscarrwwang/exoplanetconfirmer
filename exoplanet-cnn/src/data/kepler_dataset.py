"""PyTorch Dataset for Kepler light curves stored under data/raw/kplr<kepid>/."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd
import torch
from astropy.io import fits
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_LABELS_CSV = PROJECT_ROOT / "data" / "labels" / "labels.csv"

MissingPolicy = Literal["skip", "zeros", "error"]

log = logging.getLogger(__name__)


def star_dir(raw_dir: Path, kepid: int) -> Path:
    return Path(raw_dir) / f"kplr{int(kepid):09d}"


def find_fits_files(raw_dir: Path, kepid: int) -> list[Path]:
    directory = star_dir(raw_dir, kepid)
    if not directory.is_dir():
        return []
    return sorted(directory.rglob("*.fits"))


def read_flux(path: Path, flux_column: str = "PDCSAP_FLUX") -> tuple[np.ndarray, np.ndarray]:
    """Read finite (time, flux) pairs from a Kepler light curve FITS file."""
    with fits.open(path, memmap=False) as hdul:
        data = hdul["LIGHTCURVE"].data
        time = np.asarray(data["TIME"], dtype=np.float64)
        flux = np.asarray(data[flux_column], dtype=np.float64)
    mask = np.isfinite(time) & np.isfinite(flux)
    return time[mask], flux[mask]


def stitch_quarters(segments: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Normalize each quarter to zero median and concatenate in time order."""
    times, fluxes = [], []
    for time, flux in segments:
        if flux.size == 0:
            continue
        median = np.median(flux)
        if median == 0:
            continue
        times.append(time)
        fluxes.append(flux / median - 1.0)
    if not fluxes:
        return np.empty(0, dtype=np.float32)
    time = np.concatenate(times)
    flux = np.concatenate(fluxes)
    return flux[np.argsort(time, kind="stable")].astype(np.float32)


def pad_or_truncate(flux: np.ndarray, length: int, pad_value: float = 0.0) -> np.ndarray:
    if flux.size >= length:
        return flux[:length]
    out = np.full(length, pad_value, dtype=np.float32)
    out[: flux.size] = flux
    return out


class KeplerDataset(Dataset):
    """Yields (flux, label) where flux is (1, seq_len) float32 and label is a float32 scalar.

    missing:
        "skip"  - drop stars with no FITS files when the dataset is built.
        "zeros" - keep them and return an all-zero flux tensor.
        "error" - raise FileNotFoundError when the dataset is built.
    """

    def __init__(
        self,
        labels_csv: str | Path = DEFAULT_LABELS_CSV,
        raw_dir: str | Path = DEFAULT_RAW_DIR,
        seq_len: int = 4000,
        flux_column: str = "PDCSAP_FLUX",
        missing: MissingPolicy = "skip",
        pad_value: float = 0.0,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        if missing not in ("skip", "zeros", "error"):
            raise ValueError(f"Invalid missing policy: {missing!r}")
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")

        self.raw_dir = Path(raw_dir)
        self.seq_len = seq_len
        self.flux_column = flux_column
        self.missing = missing
        self.pad_value = pad_value
        self.transform = transform

        labels = pd.read_csv(labels_csv)
        required = {"kepid", "label"}
        if not required.issubset(labels.columns):
            raise ValueError(f"{labels_csv} must contain columns {sorted(required)}")
        labels = labels.dropna(subset=["kepid", "label"])

        self.samples: list[tuple[int, int, list[Path]]] = []
        missing_ids: list[int] = []
        for kepid, label in zip(labels["kepid"].astype(int), labels["label"].astype(int)):
            files = find_fits_files(self.raw_dir, kepid)
            if not files:
                missing_ids.append(kepid)
                if missing == "skip":
                    continue
            self.samples.append((kepid, label, files))

        if missing_ids:
            if missing == "error":
                raise FileNotFoundError(
                    f"No FITS files for {len(missing_ids)} stars, e.g. KIC {missing_ids[:5]}"
                )
            log.warning(
                "%d of %d stars have no FITS files (policy=%s)",
                len(missing_ids),
                len(labels),
                missing,
            )
        self.missing_kepids = missing_ids

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def kepids(self) -> list[int]:
        return [kepid for kepid, _, _ in self.samples]

    @property
    def labels(self) -> torch.Tensor:
        return torch.tensor([label for _, label, _ in self.samples], dtype=torch.float32)

    def load_flux(self, kepid: int, files: list[Path]) -> np.ndarray:
        segments = []
        for path in files:
            try:
                segments.append(read_flux(path, self.flux_column))
            except (OSError, KeyError, ValueError) as exc:
                log.warning("KIC %s: skipping unreadable file %s (%s)", kepid, path, exc)
        flux = stitch_quarters(segments)
        if flux.size == 0 and files:
            log.warning("KIC %s: no usable flux, returning padding", kepid)
        return pad_or_truncate(flux, self.seq_len, self.pad_value)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        kepid, label, files = self.samples[index]
        flux = torch.from_numpy(self.load_flux(kepid, files)).unsqueeze(0)
        if self.transform is not None:
            flux = self.transform(flux)
        return flux, torch.tensor(label, dtype=torch.float32)
