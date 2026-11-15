"""Hand-engineered transit features for the classical baselines (BLS, Random Forest, XGBoost).

Per KOI, from the same detrended light curve and ephemeris used to build the CNN views:

    BLS statistics (astropy BoxLeastSquares evaluated at the catalog period / epoch / duration)
        bls_depth_ppm, bls_snr            transit depth and its significance
        bls_odd_even_sigma                |depth_odd - depth_even| / error  (eclipsing binary at 2P)
        bls_secondary_snr                 depth at phase 0.5 / error        (secondary eclipse)
        bls_harmonic_dll                  sinusoid minus box log-likelihood (> 0: sinusoid fits better)
        bls_n_transits                    transits with data
        bls_max_transit_ll_frac           share of the signal from the strongest single event
    View shape (from the normalized global / local views, transit bottom = -1)
        local_v_shape                     mean flux of the inner half of the transit / full transit
        local_asymmetry                   ingress-half minus egress-half mean flux
        local_oot_std, global_oot_std     out-of-transit scatter relative to the transit depth
        global_secondary_depth            mean flux in the secondary-eclipse window
        global_min_oot                    deepest out-of-transit bin (other dips)
    Ephemeris
        log10_period, duration_hours, duty_cycle

The BLS search is not blind: like the CNN, it is evaluated at the catalog ephemeris.

Usage (normally called through `python -m src.baselines.compare`):
    python -m src.baselines.features --raw-dir data/raw_small \
        --koi-table data/labels/kepler_koi_labels_small.csv
"""

from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares
from tqdm import tqdm

from src.data.build_dataset import DEFAULT_DATASET_DIR, SPLITS, load_split
from src.evaluation.explain import load_view_config, region_masks
from src.preprocessing.preprocess import (
    DEFAULT_RAW_DIR,
    PROJECT_ROOT,
    PreprocessingError,
    find_fits_files,
    read_kepler_fits,
)
from src.preprocessing.views import (
    DEFAULT_KOI_TABLE,
    DEFAULT_VIEWS_DIR,
    Ephemeris,
    ViewConfig,
    clean_and_detrend,
    ephemeris_from_row,
    load_koi_table,
    transit_mask,
)

DEFAULT_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "baseline_features.csv"

BLS_FEATURES = (
    "bls_depth_ppm",
    "bls_snr",
    "bls_odd_even_sigma",
    "bls_secondary_snr",
    "bls_harmonic_dll",
    "bls_n_transits",
    "bls_max_transit_ll_frac",
)
VIEW_FEATURES = (
    "local_v_shape",
    "local_asymmetry",
    "local_oot_std",
    "global_oot_std",
    "global_secondary_depth",
    "global_min_oot",
)
EPHEMERIS_FEATURES = ("log10_period", "duration_hours", "duty_cycle")
FEATURE_COLUMNS = BLS_FEATURES + VIEW_FEATURES + EPHEMERIS_FEATURES

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BLS
# ---------------------------------------------------------------------------
def _ratio(value: float, error: float) -> float:
    return float(value / error) if np.isfinite(error) and error > 0 else np.nan


def robust_std(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    mad = 1.4826 * np.median(np.abs(x - np.median(x)))
    return float(mad) if mad > 0 else float(np.std(x))


def bls_statistics(time: np.ndarray, flux: np.ndarray, eph: Ephemeris) -> dict:
    """BLS depth / vetting statistics at a known ephemeris. `flux` is relative (~1)."""
    sigma = robust_std(flux[~transit_mask(time, eph)])
    if not np.isfinite(sigma) or sigma <= 0:
        raise PreprocessingError("Cannot estimate flux uncertainty")
    bls = BoxLeastSquares(time, flux, dy=np.full_like(flux, sigma))
    s = bls.compute_stats(eph.period, eph.duration, eph.t0)

    depth, depth_err = s["depth"]
    (odd, odd_err), (even, even_err) = s["depth_odd"], s["depth_even"]
    counts = np.asarray(s["per_transit_count"])
    ll = np.asarray(s["per_transit_log_likelihood"])[counts > 0]
    return {
        "bls_depth_ppm": float(depth) * 1e6,
        "bls_snr": _ratio(depth, depth_err),
        "bls_odd_even_sigma": abs(_ratio(odd - even, float(np.hypot(odd_err, even_err)))),
        "bls_secondary_snr": _ratio(*s["depth_phased"]),
        "bls_harmonic_dll": float(s["harmonic_delta_log_likelihood"]),
        "bls_n_transits": int((counts > 0).sum()),
        "bls_max_transit_ll_frac": float(ll.max() / ll.sum()) if ll.size and ll.sum() > 0
        else np.nan,
    }


# ---------------------------------------------------------------------------
# View shape
# ---------------------------------------------------------------------------
def view_features(
    global_view: np.ndarray,
    local_view: np.ndarray,
    period: float,
    duration: float,
    config: ViewConfig = ViewConfig(),
) -> dict:
    g_masks = region_masks("global", global_view.size, period, duration, config)
    l_masks = region_masks("local", local_view.size, period, duration, config)

    in_transit = np.flatnonzero(l_masks["transit"])
    out = {name: np.nan for name in VIEW_FEATURES}
    if in_transit.size >= 2:
        transit = local_view[in_transit]
        quarter = max(1, in_transit.size // 4)
        inner = transit[quarter : in_transit.size - quarter] if in_transit.size > 2 else transit
        full_mean = transit.mean()
        out["local_v_shape"] = float(inner.mean() / full_mean) if full_mean < 0 else np.nan
        half = in_transit.size // 2
        out["local_asymmetry"] = float(transit[:half].mean() - transit[-half:].mean())
    if l_masks["out_of_transit"].any():
        out["local_oot_std"] = robust_std(local_view[l_masks["out_of_transit"]])
    oot = g_masks["out_of_transit"]
    if oot.any():
        out["global_oot_std"] = robust_std(global_view[oot])
        out["global_min_oot"] = float(global_view[oot].min())
    if g_masks["secondary"].any():
        out["global_secondary_depth"] = float(global_view[g_masks["secondary"]].mean())
    return out


def ephemeris_features(eph: Ephemeris) -> dict:
    return {
        "log10_period": float(np.log10(eph.period)),
        "duration_hours": eph.duration * 24.0,
        "duty_cycle": eph.duration / eph.period,
    }


# ---------------------------------------------------------------------------
# Per star / dataset
# ---------------------------------------------------------------------------
def star_bls_features(
    kepid: int, kois: list[dict], raw_dir: Path, config: ViewConfig = ViewConfig()
) -> list[dict]:
    """BLS statistics for every KOI on one star (FITS read and detrended once)."""
    ephems = {}
    for k in kois:
        try:
            ephems[k["kepoi_name"]] = ephemeris_from_row(k)
        except (TypeError, ValueError):
            continue  # such KOIs have no views either
    segments = []
    for path in find_fits_files(raw_dir, kepid):
        try:
            segments.append(read_kepler_fits(path, config.flux_column))
        except (OSError, KeyError, ValueError):
            continue
    return bls_features_from_segments(segments, ephems, config)


def bls_features_from_segments(
    segments: list, ephems: dict[str, Ephemeris], config: ViewConfig = ViewConfig()
) -> list[dict]:
    """BLS statistics per candidate; companions on the same star are masked out."""
    base = [{"kepoi_name": name, "bls_status": "", "bls_error": ""} for name in ephems]
    if not segments:
        return [{**b, "bls_status": "missing"} for b in base]
    try:
        # Same cleaning as the views: every known transit on the star masked from the trend fit.
        time, flux = clean_and_detrend(segments, config, list(ephems.values()))
    except (PreprocessingError, ValueError) as exc:
        return [{**b, "bls_status": "empty", "bls_error": str(exc)} for b in base]

    rows = []
    for b in base:
        eph = ephems[b["kepoi_name"]]
        keep = np.ones(time.size, bool)
        for other_name, other in ephems.items():
            if other_name != b["kepoi_name"]:
                keep &= ~transit_mask(time, other, config.transit_mask_factor)
        try:
            rows.append({**b, "bls_status": "ok", **bls_statistics(time[keep], flux[keep], eph)})
        except (PreprocessingError, ValueError) as exc:
            rows.append({**b, "bls_status": "failed", "bls_error": str(exc)})
    return rows


def compute_features(
    meta: pd.DataFrame,
    x_global: np.ndarray,
    x_local: np.ndarray,
    koi_table: pd.DataFrame,
    raw_dir: Path,
    config: ViewConfig = ViewConfig(),
    workers: int = 1,
) -> pd.DataFrame:
    """Feature rows for the KOIs in `meta` (aligned with the view arrays)."""
    table = koi_table.set_index("kepoi_name")
    missing = sorted(set(meta.kepoi_name) - set(table.index))
    if missing:
        raise ValueError(f"{len(missing)} KOIs missing from the KOI table, e.g. {missing[:3]}")

    # BLS needs every KOI on the star (to mask companions), not only those in `meta`.
    stars = {
        int(kepid): koi_table[koi_table.kepid == kepid].to_dict("records")
        for kepid in meta.kepid.unique()
    }
    bls_rows: list[dict] = []
    if workers <= 1:
        for kepid, kois in tqdm(stars.items(), desc="BLS"):
            bls_rows += star_bls_features(kepid, kois, raw_dir, config)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(star_bls_features, k, v, raw_dir, config)
                       for k, v in stars.items()]
            for f in tqdm(as_completed(futures), total=len(futures), desc="BLS"):
                bls_rows += f.result()
    bls = pd.DataFrame(bls_rows).set_index("kepoi_name")

    rows = []
    for i, r in enumerate(meta.itertuples(index=False)):
        eph = ephemeris_from_row(table.loc[r.kepoi_name].to_dict())
        row = {"kepoi_name": r.kepoi_name, "kepid": int(r.kepid)}
        row.update(ephemeris_features(eph))
        row.update(view_features(x_global[i], x_local[i], eph.period, eph.duration, config))
        row.update(bls.loc[r.kepoi_name].to_dict())
        rows.append(row)
    out = pd.DataFrame(rows)
    for col in FEATURE_COLUMNS:
        if col not in out:
            out[col] = np.nan
    return out[["kepoi_name", "kepid", *FEATURE_COLUMNS, "bls_status", "bls_error"]]


def load_or_compute_features(
    splits: tuple[str, ...],
    data_dir: Path,
    views_dir: Path,
    koi_table_path: Path,
    raw_dir: Path,
    cache_path: Path,
    workers: int = 1,
) -> pd.DataFrame:
    """Features for all KOIs in `splits`, computing only those not already in `cache_path`."""
    config = load_view_config(views_dir)
    cache = pd.read_csv(cache_path) if Path(cache_path).exists() else pd.DataFrame()
    known = set(cache.kepoi_name) if not cache.empty else set()

    new = []
    koi_table = None
    for split in splits:
        split_dir = Path(data_dir) / split
        meta = pd.read_csv(split_dir / "meta.csv")
        todo = ~meta.kepoi_name.isin(known)
        if not todo.any():
            continue
        if koi_table is None:
            koi_table = load_koi_table(koi_table_path)
        x_global, x_local, _ = load_split(split_dir)
        idx = np.flatnonzero(todo.to_numpy())
        log.info("Computing features for %d %s KOIs", idx.size, split)
        new.append(compute_features(
            meta.iloc[idx].reset_index(drop=True), x_global[:, 0].numpy()[idx],
            x_local[:, 0].numpy()[idx], koi_table, raw_dir, config, workers,
        ))
        known |= set(meta.kepoi_name)
    if new:
        cache = pd.concat([cache, *new], ignore_index=True)
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        cache.to_csv(cache_path, index=False)
        log.info("Feature cache updated: %s (%d KOIs)", cache_path, len(cache))
    return cache


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute BLS / view-shape features for the baseline classifiers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATASET_DIR)
    p.add_argument("--views-dir", type=Path, default=DEFAULT_VIEWS_DIR)
    p.add_argument("--koi-table", type=Path, default=DEFAULT_KOI_TABLE)
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument("--out", type=Path, default=DEFAULT_FEATURES_PATH)
    p.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> pd.DataFrame:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    feats = load_or_compute_features(tuple(args.splits), args.data_dir, args.views_dir,
                                     args.koi_table, args.raw_dir, args.out, args.workers)
    log.info("BLS status: %s", feats.bls_status.value_counts().to_dict())
    return feats


if __name__ == "__main__":
    main()
