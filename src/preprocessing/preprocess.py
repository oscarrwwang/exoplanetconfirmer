"""Kepler light curve preprocessing pipeline.

Per quarter (and per gap-separated segment):
    1. Remove NaNs (and optionally quality-flagged cadences)
    2. Sigma clipping (upper-only by default so transit dips are preserved)
    3. Detrending (binned running median divided out)
    4. Median normalization (flux / median - 1, baseline at 0)
Then across the stitched light curve:
    5. Resampling onto a uniform cadence grid
    6. Padding / truncation to a fixed sequence length

Usage:
    python -m src.preprocessing.preprocess --help
    python -m src.preprocessing.preprocess
    python -m src.preprocessing.preprocess --seq-len 8000 --detrend-window 0.75 --workers 8
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clip as astropy_sigma_clip
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_LABELS_CSV = PROJECT_ROOT / "data" / "labels" / "labels.csv"

KEPLER_LONG_CADENCE_DAYS = 29.4243814 / (60.0 * 24.0)

log = logging.getLogger(__name__)


class PreprocessingError(ValueError):
    """Raised when a light curve has no usable data after preprocessing."""


@dataclass(frozen=True)
class PreprocessConfig:
    flux_column: str = "PDCSAP_FLUX"
    # 0 disables quality filtering; PDCSAP flux is already NaN for most severe flags.
    quality_bitmask: int = 0
    sigma_upper: float = 3.0
    sigma_lower: float | None = None
    clip_maxiters: int = 5
    # Should be several times longer than the longest transit of interest.
    detrend_window_days: float = 1.0
    gap_days: float = 0.5
    min_points_per_bin: int = 10
    min_points_per_segment: int = 50
    cadence_days: float = KEPLER_LONG_CADENCE_DAYS
    seq_len: int = 4000
    pad_value: float = 0.0
    fill_value: float = 0.0

    def __post_init__(self) -> None:
        positive = {
            "sigma_upper": self.sigma_upper,
            "detrend_window_days": self.detrend_window_days,
            "gap_days": self.gap_days,
            "cadence_days": self.cadence_days,
            "seq_len": self.seq_len,
            "clip_maxiters": self.clip_maxiters,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.sigma_lower is not None and self.sigma_lower <= 0:
            raise ValueError(f"sigma_lower must be positive or None, got {self.sigma_lower}")
        if self.min_points_per_bin < 1 or self.min_points_per_segment < 1:
            raise ValueError("min_points_per_bin and min_points_per_segment must be >= 1")


@dataclass
class Segment:
    time: np.ndarray
    flux: np.ndarray
    quality: np.ndarray | None = None


@dataclass
class PreprocessResult:
    flux: np.ndarray  # float32, (seq_len,)
    mask: np.ndarray  # bool, (seq_len,), True where the grid point has real data
    t_start: float
    n_input: int
    n_valid: int
    n_clipped: int
    n_resampled: int


# ---------------------------------------------------------------------------
# Step 1: remove NaNs
# ---------------------------------------------------------------------------
def remove_invalid(
    time: np.ndarray,
    flux: np.ndarray,
    quality: np.ndarray | None = None,
    quality_bitmask: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    if time.shape != flux.shape:
        raise ValueError(f"time and flux shapes differ: {time.shape} vs {flux.shape}")
    keep = np.isfinite(time) & np.isfinite(flux)
    if quality is not None and quality_bitmask:
        keep &= (np.asarray(quality, dtype=np.int64) & quality_bitmask) == 0
    return time[keep], flux[keep]


# ---------------------------------------------------------------------------
# Step 2: sigma clipping
# ---------------------------------------------------------------------------
def sigma_clip(
    time: np.ndarray,
    flux: np.ndarray,
    sigma_upper: float = 3.0,
    sigma_lower: float | None = None,
    maxiters: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Clip outliers using median / MAD statistics. sigma_lower=None keeps all dips."""
    if flux.size == 0:
        return time, flux
    clipped = astropy_sigma_clip(
        flux,
        sigma_lower=np.inf if sigma_lower is None else sigma_lower,
        sigma_upper=sigma_upper,
        maxiters=maxiters,
        cenfunc="median",
        stdfunc="mad_std",
        masked=True,
    )
    keep = ~np.ma.getmaskarray(clipped)
    return time[keep], flux[keep]


# ---------------------------------------------------------------------------
# Step 3: detrending
# ---------------------------------------------------------------------------
def _binned_median_trend(
    time: np.ndarray, flux: np.ndarray, window_days: float, min_points: int
) -> np.ndarray:
    bins = np.floor((time - time[0]) / window_days).astype(np.int64)
    splits = np.flatnonzero(np.diff(bins)) + 1
    centers, medians = [], []
    for t_chunk, f_chunk in zip(np.split(time, splits), np.split(flux, splits)):
        if t_chunk.size >= min_points:
            centers.append(np.median(t_chunk))
            medians.append(np.median(f_chunk))
    if len(centers) < 2:
        level = medians[0] if medians else np.median(flux)
        return np.full_like(flux, level)
    centers_arr = np.asarray(centers)
    medians_arr = np.asarray(medians)
    trend = np.interp(time, centers_arr, medians_arr)
    # Extrapolate linearly past the outer bin centers instead of holding them constant.
    for edge, (i, j) in ((time < centers_arr[0], (0, 1)), (time > centers_arr[-1], (-2, -1))):
        if edge.any():
            slope = (medians_arr[j] - medians_arr[i]) / (centers_arr[j] - centers_arr[i])
            trend[edge] = medians_arr[i] + slope * (time[edge] - centers_arr[i])
    return trend


def split_segments(time: np.ndarray, gap_days: float) -> list[slice]:
    bounds = np.concatenate(([0], np.flatnonzero(np.diff(time) > gap_days) + 1, [time.size]))
    return [slice(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:])]


def detrend(
    time: np.ndarray,
    flux: np.ndarray,
    window_days: float = 1.0,
    gap_days: float = 0.5,
    min_points_per_bin: int = 10,
    exclude: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Divide out a running-median trend, fitted independently across data gaps.

    `exclude` marks points (e.g. known transits) left out of the trend fit; the
    trend is interpolated across them.
    """
    if flux.size == 0:
        return time, flux
    order = np.argsort(time, kind="stable")
    time, flux = time[order], flux[order]
    fit = np.ones(time.size, bool) if exclude is None else ~np.asarray(exclude, bool)[order]
    trend = np.full_like(flux, np.nan)
    for seg in split_segments(time, gap_days):
        t, f, m = time[seg], flux[seg], fit[seg]
        if not m.any():
            continue
        fitted = _binned_median_trend(t[m], f[m], window_days, min_points_per_bin)
        trend[seg] = fitted if m.all() else np.interp(t, t[m], fitted)
    keep = np.isfinite(trend) & (trend > 0)
    return time[keep], flux[keep] / trend[keep]


# ---------------------------------------------------------------------------
# Step 4: median normalization
# ---------------------------------------------------------------------------
def median_normalize(flux: np.ndarray) -> np.ndarray:
    """Scale to relative flux with baseline 0 (a 1% transit becomes -0.01)."""
    if flux.size == 0:
        return flux
    median = np.median(flux)
    if not np.isfinite(median) or median == 0:
        raise PreprocessingError(f"Cannot normalize by median {median}")
    return flux / median - 1.0


# ---------------------------------------------------------------------------
# Step 5: resampling
# ---------------------------------------------------------------------------
def resample(
    time: np.ndarray,
    flux: np.ndarray,
    cadence_days: float = KEPLER_LONG_CADENCE_DAYS,
    fill_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Average onto a uniform grid starting at min(time); empty bins get fill_value."""
    if time.size == 0:
        return np.empty(0, np.float32), np.empty(0, bool), float("nan")
    t_start = float(time.min())
    idx = np.floor((time - t_start) / cadence_days + 0.5).astype(np.int64)
    n = int(idx.max()) + 1
    counts = np.bincount(idx, minlength=n)
    sums = np.bincount(idx, weights=flux, minlength=n)
    mask = counts > 0
    out = np.full(n, fill_value, dtype=np.float64)
    out[mask] = sums[mask] / counts[mask]
    return out.astype(np.float32), mask, t_start


# ---------------------------------------------------------------------------
# Step 6: sequence padding
# ---------------------------------------------------------------------------
def pad_sequence(
    flux: np.ndarray, mask: np.ndarray, length: int, pad_value: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Right-pad with pad_value or truncate to `length`."""
    out = np.full(length, pad_value, dtype=np.float32)
    out_mask = np.zeros(length, dtype=bool)
    n = min(length, flux.size)
    out[:n] = flux[:n]
    out_mask[:n] = mask[:n]
    return out, out_mask


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def preprocess_segments(
    segments: Iterable[Segment], config: PreprocessConfig = PreprocessConfig()
) -> PreprocessResult:
    times, fluxes = [], []
    n_input = n_valid = n_clipped = 0

    for seg in segments:
        n_input += seg.time.size
        time, flux = remove_invalid(seg.time, seg.flux, seg.quality, config.quality_bitmask)
        n_valid += time.size
        if time.size < config.min_points_per_segment:
            continue

        before = time.size
        time, flux = sigma_clip(
            time, flux, config.sigma_upper, config.sigma_lower, config.clip_maxiters
        )
        n_clipped += before - time.size

        time, flux = detrend(
            time,
            flux,
            config.detrend_window_days,
            config.gap_days,
            config.min_points_per_bin,
        )
        if time.size < config.min_points_per_segment:
            continue

        times.append(time)
        fluxes.append(median_normalize(flux))

    if not times:
        raise PreprocessingError("No segment has enough valid points")

    time = np.concatenate(times)
    flux = np.concatenate(fluxes)
    order = np.argsort(time, kind="stable")

    grid_flux, grid_mask, t_start = resample(
        time[order], flux[order], config.cadence_days, config.fill_value
    )
    padded, padded_mask = pad_sequence(grid_flux, grid_mask, config.seq_len, config.pad_value)
    return PreprocessResult(
        flux=padded,
        mask=padded_mask,
        t_start=t_start,
        n_input=n_input,
        n_valid=n_valid,
        n_clipped=n_clipped,
        n_resampled=int(grid_flux.size),
    )


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def read_kepler_fits(path: Path, flux_column: str = "PDCSAP_FLUX") -> Segment:
    with fits.open(path, memmap=False) as hdul:
        data = hdul["LIGHTCURVE"].data
        names = data.columns.names
        quality = np.asarray(data["SAP_QUALITY"]) if "SAP_QUALITY" in names else None
        return Segment(
            time=np.asarray(data["TIME"], dtype=np.float64),
            flux=np.asarray(data[flux_column], dtype=np.float64),
            quality=quality,
        )


def find_fits_files(raw_dir: Path, kepid: int) -> list[Path]:
    directory = Path(raw_dir) / f"kplr{int(kepid):09d}"
    return sorted(directory.rglob("*.fits")) if directory.is_dir() else []


def output_path(processed_dir: Path, kepid: int) -> Path:
    return Path(processed_dir) / f"kplr{int(kepid):09d}.npz"


def save_result(path: Path, kepid: int, label: int, result: PreprocessResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(
            fh,
            flux=result.flux,
            mask=result.mask,
            kepid=np.int64(kepid),
            label=np.int8(label),
            t_start=np.float64(result.t_start),
        )
    os.replace(tmp, path)


def process_star(
    kepid: int,
    label: int,
    raw_dir: Path,
    processed_dir: Path,
    config: PreprocessConfig,
    overwrite: bool = False,
) -> dict:
    out = output_path(processed_dir, kepid)
    row = {
        "kepid": kepid,
        "label": label,
        "path": out.name,
        "status": "",
        "n_files": 0,
        "n_bad_files": 0,
        "n_input": 0,
        "n_valid": 0,
        "n_clipped": 0,
        "n_resampled": 0,
        "n_grid_valid": 0,
        "error": "",
    }
    if out.exists() and not overwrite:
        row["status"] = "skipped"
        return row

    files = find_fits_files(raw_dir, kepid)
    row["n_files"] = len(files)
    if not files:
        row["status"] = "missing"
        return row

    segments = []
    for path in files:
        try:
            segments.append(read_kepler_fits(path, config.flux_column))
        except (OSError, KeyError, ValueError) as exc:
            row["n_bad_files"] += 1
            row["error"] = f"{path.name}: {type(exc).__name__}: {exc}"

    try:
        result = preprocess_segments(segments, config)
    except PreprocessingError as exc:
        row["status"] = "empty"
        row["error"] = str(exc)
        return row

    save_result(out, kepid, label, result)
    row.update(
        status="ok",
        n_input=result.n_input,
        n_valid=result.n_valid,
        n_clipped=result.n_clipped,
        n_resampled=result.n_resampled,
        n_grid_valid=int(result.mask.sum()),
    )
    return row


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = PreprocessConfig()
    p = argparse.ArgumentParser(
        description="Preprocess raw Kepler light curves into fixed-length arrays.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--labels", type=Path, default=DEFAULT_LABELS_CSV)
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    p.add_argument("--flux-column", default=defaults.flux_column)
    p.add_argument("--quality-bitmask", type=int, default=defaults.quality_bitmask)
    p.add_argument("--sigma-upper", type=float, default=defaults.sigma_upper)
    p.add_argument(
        "--sigma-lower", type=float, default=defaults.sigma_lower,
        help="Lower clip threshold; omit to keep all dips (recommended for transits).",
    )
    p.add_argument("--detrend-window", type=float, default=defaults.detrend_window_days,
                   help="Detrending window in days.")
    p.add_argument("--gap-days", type=float, default=defaults.gap_days)
    p.add_argument("--cadence-days", type=float, default=defaults.cadence_days)
    p.add_argument("--seq-len", type=int, default=defaults.seq_len)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--limit", type=int, default=None, help="Process only the first N stars.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> PreprocessConfig:
    return PreprocessConfig(
        flux_column=args.flux_column,
        quality_bitmask=args.quality_bitmask,
        sigma_upper=args.sigma_upper,
        sigma_lower=args.sigma_lower,
        detrend_window_days=args.detrend_window,
        gap_days=args.gap_days,
        cadence_days=args.cadence_days,
        seq_len=args.seq_len,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config = config_from_args(args)

    labels = pd.read_csv(args.labels).dropna(subset=["kepid", "label"])
    labels = labels.drop_duplicates("kepid")
    if args.limit is not None:
        labels = labels.head(args.limit)
    targets = list(zip(labels["kepid"].astype(int), labels["label"].astype(int)))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "preprocess_config.json").write_text(json.dumps(asdict(config), indent=2))
    log.info("Preprocessing %d stars from %s -> %s", len(targets), args.raw_dir, args.out_dir)

    rows = []
    if args.workers <= 1:
        for kepid, label in tqdm(targets, desc="Preprocess"):
            rows.append(
                process_star(kepid, label, args.raw_dir, args.out_dir, config, args.overwrite)
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    process_star, kepid, label, args.raw_dir, args.out_dir, config, args.overwrite
                ): kepid
                for kepid, label in targets
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Preprocess"):
                kepid = futures[future]
                try:
                    rows.append(future.result())
                except Exception as exc:  # keep the batch going on unexpected failures
                    log.exception("KIC %s failed", kepid)
                    rows.append({"kepid": kepid, "status": "error", "error": repr(exc)})

    index = pd.DataFrame(rows).sort_values("kepid")
    index_path = args.out_dir / "index.csv"
    index.to_csv(index_path, index=False)
    log.info("Status: %s", index["status"].value_counts().to_dict())
    log.info("Index written to %s", index_path)


if __name__ == "__main__":
    main()
