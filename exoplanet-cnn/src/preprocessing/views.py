"""Transit-focused global / local views (Astronet / ExoMiner style).

For each KOI with ephemeris (period P, epoch t0, duration D):

    1. Clean and detrend the light curve with known transits masked out of the trend fit.
    2. Phase fold on P so every transit lands at phase 0.
    3. Bin the folded curve into
         global view: 2001 bins over the full orbit   [-P/2, P/2]
         local view:   201 bins around the transit   [-4D, 4D]
    4. Normalize so the out-of-transit baseline is 0 and the transit bottom is -1.

Inputs:
    data/labels/kepler_koi_labels.csv  (from `python -m src.data.download_kepler --labels-only`)
    data/raw/kplr<kepid>/*.fits        (from `python -m src.data.download_kepler`)
Outputs:
    data/processed/views/<kepoi_name>.npz  with global_flux (2001,), local_flux (201,), label, ...
    data/processed/views/index.csv

Usage:
    python -m src.preprocessing.views --help
    python -m src.preprocessing.views --limit 100 --workers 8
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.preprocessing.preprocess import (
    PROJECT_ROOT,
    DEFAULT_RAW_DIR,
    PreprocessingError,
    Segment,
    detrend,
    find_fits_files,
    read_kepler_fits,
    remove_invalid,
    sigma_clip,
)

DEFAULT_KOI_TABLE = PROJECT_ROOT / "data" / "labels" / "kepler_koi_labels.csv"
DEFAULT_VIEWS_DIR = PROJECT_ROOT / "data" / "processed" / "views"

Normalization = Literal["shared", "per_view"]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Ephemeris:
    """Transit ephemeris. All values in days; t0 in the same time system as the light curve (BKJD)."""

    period: float
    t0: float
    duration: float

    def __post_init__(self) -> None:
        for name in ("period", "t0", "duration"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.period <= 0 or self.duration <= 0:
            raise ValueError("period and duration must be positive")
        if self.duration >= self.period:
            raise ValueError("duration must be shorter than period")


@dataclass(frozen=True)
class ViewConfig:
    flux_column: str = "PDCSAP_FLUX"
    quality_bitmask: int = 0
    min_points_per_segment: int = 50
    detrend_window_days: float = 1.0
    gap_days: float = 0.5
    min_points_per_bin: int = 10
    # Half-width of the masked transit window, in units of the transit duration.
    transit_mask_factor: float = 1.0
    sigma_upper: float = 3.0
    clip_maxiters: int = 5
    global_bins: int = 2001
    global_bin_width_factor: float = 1.0 / 2001  # x period
    local_bins: int = 201
    local_bin_width_factor: float = 0.16  # x duration
    local_num_durations: float = 4.0
    normalization: Normalization = "shared"

    def __post_init__(self) -> None:
        if self.global_bins < 2 or self.local_bins < 2:
            raise ValueError("global_bins and local_bins must be >= 2")
        if not 0 < self.global_bin_width_factor < 1:
            raise ValueError("global_bin_width_factor must be in (0, 1)")
        if self.local_bin_width_factor <= 0 or self.local_num_durations <= 0:
            raise ValueError("local_bin_width_factor and local_num_durations must be positive")
        if self.transit_mask_factor <= 0:
            raise ValueError("transit_mask_factor must be positive")
        if self.normalization not in ("shared", "per_view"):
            raise ValueError(f"Invalid normalization: {self.normalization!r}")


@dataclass
class TransitViews:
    global_flux: np.ndarray  # float32, (global_bins,)
    local_flux: np.ndarray  # float32, (local_bins,)
    depth: float  # fractional transit depth measured from the local view
    n_points: int
    global_coverage: float  # fraction of global bins that contained data
    local_coverage: float


# ---------------------------------------------------------------------------
# Folding
# ---------------------------------------------------------------------------
def phase_fold(time: np.ndarray, period: float, t0: float) -> np.ndarray:
    """Map time to phase (days) in [-P/2, P/2) with transit centers at 0."""
    return np.mod(time - t0 + 0.5 * period, period) - 0.5 * period


def transit_mask(time: np.ndarray, ephemeris: Ephemeris, factor: float = 1.0) -> np.ndarray:
    """True for points within factor * duration of a transit center."""
    phase = phase_fold(time, ephemeris.period, ephemeris.t0)
    return np.abs(phase) < factor * ephemeris.duration


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------
def bin_view(
    phase: np.ndarray,
    flux: np.ndarray,
    num_bins: int,
    bin_width: float,
    x_min: float,
    x_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Median-bin (phase, flux) into `num_bins` evenly spaced, possibly overlapping bins.

    Bin i covers [x_min + i*s, x_min + i*s + bin_width) with
    s = (x_max - x_min - bin_width) / (num_bins - 1), so the first bin starts at
    x_min and the last ends at x_max. Empty bins are NaN.
    """
    if num_bins < 2:
        raise ValueError("num_bins must be >= 2")
    if x_max <= x_min:
        raise ValueError("x_max must be greater than x_min")
    if not 0 < bin_width < x_max - x_min:
        raise ValueError(f"bin_width {bin_width} must be in (0, {x_max - x_min})")

    order = np.argsort(phase, kind="stable")
    phase, flux = phase[order], flux[order]

    spacing = (x_max - x_min - bin_width) / (num_bins - 1)
    lo = x_min + spacing * np.arange(num_bins)
    hi = lo + bin_width
    start = np.searchsorted(phase, lo, side="left")
    stop = np.searchsorted(phase, hi, side="left")
    counts = stop - start

    view = np.full(num_bins, np.nan)
    for i in np.flatnonzero(counts):
        view[i] = np.median(flux[start[i] : stop[i]])
    return view, counts


def fill_empty_bins(view: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN bins from their neighbours (edges held constant)."""
    valid = np.isfinite(view)
    if not valid.any():
        raise PreprocessingError("View has no populated bins")
    if valid.all():
        return view
    x = np.arange(view.size)
    out = view.copy()
    out[~valid] = np.interp(x[~valid], x[valid], view[valid])
    return out


def global_view(
    phase: np.ndarray, flux: np.ndarray, ephemeris: Ephemeris, config: ViewConfig
) -> tuple[np.ndarray, np.ndarray]:
    half = 0.5 * ephemeris.period
    width = ephemeris.period * config.global_bin_width_factor
    return bin_view(phase, flux, config.global_bins, width, -half, half)


def local_view(
    phase: np.ndarray, flux: np.ndarray, ephemeris: Ephemeris, config: ViewConfig
) -> tuple[np.ndarray, np.ndarray]:
    half = min(0.5 * ephemeris.period, config.local_num_durations * ephemeris.duration)
    width = ephemeris.duration * config.local_bin_width_factor
    return bin_view(phase, flux, config.local_bins, width, -half, half)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
def normalize_views(
    global_v: np.ndarray, local_v: np.ndarray, mode: Normalization = "shared"
) -> tuple[np.ndarray, np.ndarray, float]:
    """Shift each view's median to 0 and scale so the transit bottom is -1.

    shared:   both views divided by the depth measured on the local view.
    per_view: each view divided by its own |min| (original Astronet behaviour).
    Returns (global, local, depth) where depth is the local-view depth in relative flux.
    """
    g = global_v - np.median(global_v)
    loc = local_v - np.median(local_v)
    depth = float(-loc.min())

    def safe(scale: float) -> float:
        return scale if np.isfinite(scale) and scale > 0 else 1.0

    local_scale = safe(depth)
    global_scale = local_scale if mode == "shared" else safe(float(-g.min()))
    return g / global_scale, loc / local_scale, max(depth, 0.0)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def clean_and_detrend(
    segments: Iterable[Segment],
    config: ViewConfig,
    masked: Sequence[Ephemeris],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (time, relative flux ~ 1) with all `masked` transits excluded from the trend fit."""
    times, fluxes = [], []
    for seg in segments:
        time, flux = remove_invalid(seg.time, seg.flux, seg.quality, config.quality_bitmask)
        if time.size < config.min_points_per_segment:
            continue
        in_transit = np.zeros(time.size, bool)
        for eph in masked:
            in_transit |= transit_mask(time, eph, config.transit_mask_factor)
        time, flux = detrend(
            time,
            flux,
            config.detrend_window_days,
            config.gap_days,
            config.min_points_per_bin,
            exclude=in_transit,
        )
        # Upper-only clip after detrending: removes flares / cosmic rays, keeps dips.
        time, flux = sigma_clip(time, flux, config.sigma_upper, None, config.clip_maxiters)
        if time.size >= config.min_points_per_segment:
            times.append(time)
            fluxes.append(flux)
    if not times:
        raise PreprocessingError("No segment has enough valid points")
    time = np.concatenate(times)
    flux = np.concatenate(fluxes)
    order = np.argsort(time, kind="stable")
    return time[order], flux[order]


def views_from_detrended(
    time: np.ndarray,
    flux: np.ndarray,
    ephemeris: Ephemeris,
    config: ViewConfig = ViewConfig(),
    others: Sequence[Ephemeris] = (),
) -> TransitViews:
    """Build normalized global/local views from a detrended light curve."""
    if others:
        keep = np.ones(time.size, bool)
        for eph in others:
            keep &= ~transit_mask(time, eph, config.transit_mask_factor)
        time, flux = time[keep], flux[keep]

    phase = phase_fold(time, ephemeris.period, ephemeris.t0)
    g, g_counts = global_view(phase, flux, ephemeris, config)
    loc, l_counts = local_view(phase, flux, ephemeris, config)
    if not l_counts.any():
        raise PreprocessingError("No data inside the local (transit) window")

    g, loc, depth = normalize_views(fill_empty_bins(g), fill_empty_bins(loc), config.normalization)
    return TransitViews(
        global_flux=g.astype(np.float32),
        local_flux=loc.astype(np.float32),
        depth=depth,
        n_points=int(time.size),
        global_coverage=float((g_counts > 0).mean()),
        local_coverage=float((l_counts > 0).mean()),
    )


def generate_views(
    segments: Iterable[Segment],
    ephemeris: Ephemeris,
    config: ViewConfig = ViewConfig(),
    others: Sequence[Ephemeris] = (),
) -> TransitViews:
    """Full pipeline for one KOI. `others` are other known transiting objects on the same star."""
    time, flux = clean_and_detrend(segments, config, [ephemeris, *others])
    return views_from_detrended(time, flux, ephemeris, config, others)


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = ("kepid", "kepoi_name", "koi_period", "koi_time0bk", "koi_duration", "label")


def ephemeris_from_row(row: dict) -> Ephemeris:
    """KOI table: period [days], time0bk [BKJD], duration [hours]."""
    return Ephemeris(
        period=float(row["koi_period"]),
        t0=float(row["koi_time0bk"]),
        duration=float(row["koi_duration"]) / 24.0,
    )


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(name))


def save_views(path: Path, row: dict, eph: Ephemeris, views: TransitViews) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(
            fh,
            global_flux=views.global_flux,
            local_flux=views.local_flux,
            label=np.int8(row["label"]),
            kepid=np.int64(row["kepid"]),
            kepoi_name=np.str_(row["kepoi_name"]),
            period=np.float64(eph.period),
            t0=np.float64(eph.t0),
            duration=np.float64(eph.duration),
            depth=np.float64(views.depth),
        )
    os.replace(tmp, path)


def process_star_views(
    kepid: int,
    kois: list[dict],
    raw_dir: Path,
    out_dir: Path,
    config: ViewConfig,
    overwrite: bool = False,
) -> list[dict]:
    """Generate views for every KOI on one star, reading its FITS files once."""
    results = []
    pending = []
    ephemerides: dict[str, Ephemeris] = {}
    for koi in kois:
        name = koi["kepoi_name"]
        out = Path(out_dir) / f"{safe_name(name)}.npz"
        result = {
            "kepoi_name": name,
            "kepid": kepid,
            "label": koi["label"],
            "path": out.name,
            "status": "",
            "depth_ppm": np.nan,
            "koi_depth_ppm": koi.get("koi_depth", np.nan),
            "n_points": 0,
            "global_coverage": np.nan,
            "local_coverage": np.nan,
            "error": "",
        }
        try:
            ephemerides[name] = ephemeris_from_row(koi)
        except (TypeError, ValueError) as exc:
            result.update(status="bad_ephemeris", error=str(exc))
            results.append(result)
            continue
        if out.exists() and not overwrite:
            result["status"] = "skipped"
            results.append(result)
            continue
        pending.append((koi, out, result))

    if not pending:
        return results

    files = find_fits_files(raw_dir, kepid)
    segments, read_errors = [], []
    for path in files:
        try:
            segments.append(read_kepler_fits(path, config.flux_column))
        except (OSError, KeyError, ValueError) as exc:
            read_errors.append(f"{path.name}: {type(exc).__name__}: {exc}")

    for koi, out, result in pending:
        name = koi["kepoi_name"]
        if not segments:
            result.update(status="missing", error="; ".join(read_errors))
            results.append(result)
            continue
        eph = ephemerides[name]
        others = [e for n, e in ephemerides.items() if n != name]
        try:
            views = generate_views(segments, eph, config, others)
        except (PreprocessingError, ValueError) as exc:
            result.update(status="empty", error=str(exc))
            results.append(result)
            continue
        save_views(out, koi, eph, views)
        result.update(
            status="ok",
            depth_ppm=views.depth * 1e6,
            n_points=views.n_points,
            global_coverage=views.global_coverage,
            local_coverage=views.local_coverage,
        )
        results.append(result)
    return results


def load_koi_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(
            f"{path} is missing columns {missing}; "
            "generate it with `python -m src.data.download_kepler --labels-only`"
        )
    return table.dropna(subset=["kepid", "kepoi_name", "label"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    d = ViewConfig()
    p = argparse.ArgumentParser(
        description="Generate phase-folded global/local transit views for Kepler KOIs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--koi-table", type=Path, default=DEFAULT_KOI_TABLE)
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_VIEWS_DIR)
    p.add_argument("--global-bins", type=int, default=d.global_bins)
    p.add_argument("--local-bins", type=int, default=d.local_bins)
    p.add_argument("--local-durations", type=float, default=d.local_num_durations,
                   help="Local view half-width in transit durations.")
    p.add_argument("--detrend-window", type=float, default=d.detrend_window_days,
                   help="Detrending window in days.")
    p.add_argument("--transit-mask-factor", type=float, default=d.transit_mask_factor,
                   help="Masked half-width around transits, in durations.")
    p.add_argument("--normalization", choices=("shared", "per_view"), default=d.normalization)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--limit", type=int, default=None, help="Process only the first N stars.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config = ViewConfig(
        global_bins=args.global_bins,
        global_bin_width_factor=1.0 / args.global_bins,
        local_bins=args.local_bins,
        local_num_durations=args.local_durations,
        detrend_window_days=args.detrend_window,
        transit_mask_factor=args.transit_mask_factor,
        normalization=args.normalization,
    )

    table = load_koi_table(args.koi_table)
    stars = [
        (int(kepid), group.to_dict("records"))
        for kepid, group in table.groupby("kepid", sort=True)
    ]
    if args.limit is not None:
        stars = stars[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "views_config.json").write_text(json.dumps(asdict(config), indent=2))
    log.info("Generating views for %d KOIs on %d stars", sum(len(k) for _, k in stars), len(stars))

    rows: list[dict] = []
    if args.workers <= 1:
        for kepid, kois in tqdm(stars, desc="Views"):
            rows.extend(
                process_star_views(kepid, kois, args.raw_dir, args.out_dir, config, args.overwrite)
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    process_star_views, kepid, kois, args.raw_dir, args.out_dir, config,
                    args.overwrite,
                ): kepid
                for kepid, kois in stars
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Views"):
                kepid = futures[future]
                try:
                    rows.extend(future.result())
                except Exception as exc:  # keep the batch going on unexpected failures
                    log.exception("KIC %s failed", kepid)
                    rows.append({"kepid": kepid, "status": "error", "error": repr(exc)})

    index = pd.DataFrame(rows).sort_values(["kepid", "kepoi_name"], na_position="last")
    index_path = args.out_dir / "index.csv"
    index.to_csv(index_path, index=False)
    log.info("Status: %s", index["status"].value_counts().to_dict())
    log.info("Index written to %s", index_path)


if __name__ == "__main__":
    main()
