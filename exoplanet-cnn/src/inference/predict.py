"""Production inference: Kepler light curve + candidate ephemeris -> vetted prediction report.

Pipeline (each step reuses the training code, so inference matches training exactly):
    1. Load the trained checkpoint, its validation-selected threshold and the view settings.
    2. Read the Kepler long-cadence FITS files of one star (*_llc.fits).
    3. Preprocess with src.preprocessing.views.generate_views (clean, detrend with transits
       masked, phase-fold, bin into global / local views, normalize).
    4. Run the CNN -> planet probability, classification and confidence score.
    5. Compute BLS / transit-shape features (src.baselines.features) and occlusion evidence
       (src.evaluation.explain), then apply false-positive vetting (src.inference.vetting).
    6. Write report.json, report.txt and one explanation figure per candidate.

The CNN classifies a *given* periodic transit signal; it does not search for one. Each
candidate needs an ephemeris (period, mid-transit time in BKJD = BJD - 2454833, duration),
either on the command line or from a KOI/TCE table. Other KOIs on the same star listed in the
table are masked out, exactly as during training.

Usage:
    python -m src.inference.predict data/raw_small/kplr010593626 \
        --checkpoint checkpoints/<run>/best.pt --period 289.86 --t0 133.29 --duration-hours 7.4
    python -m src.inference.predict path/to/kplr*.fits --checkpoint checkpoints/<run>/best.pt \
        --koi-table data/labels/kepler_koi_labels.csv [--kepoi-name K00087.01]

Outputs (default outputs/inference/kic<kepid>/): report.json  report.txt  <candidate>.png
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from astropy.io import fits

from src.baselines.features import (
    bls_features_from_segments,
    ephemeris_features,
    view_features,
)
from src.data.build_dataset import DEFAULT_DATASET_DIR, SPLITS
from src.evaluation.evaluate import load_model
from src.evaluation.explain import (
    grad_cam,
    model_views,
    occlusion,
    plot_axis,
    plot_example,
    region_masks,
    region_shares,
    saliency,
)
from src.inference.vetting import VettingConfig, decide, vet
from src.preprocessing.preprocess import PROJECT_ROOT, PreprocessingError, Segment, read_kepler_fits
from src.preprocessing.views import (
    DEFAULT_VIEWS_DIR,
    Ephemeris,
    ViewConfig,
    ephemeris_from_row,
    generate_views,
    safe_name,
)
from src.training.train import compute_logits, load_checkpoint, select_device

DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "inference"
DEFAULT_VIEWS_CONFIG = DEFAULT_VIEWS_DIR / "views_config.json"
CALIBRATION_NOTE = (
    "Probabilities come from a sigmoid on the CNN logit and are not calibrated; the confidence "
    "score measures distance from the validation-selected threshold, not a likelihood."
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Model
# ---------------------------------------------------------------------------
@dataclass
class Predictor:
    model: torch.nn.Module
    threshold: float
    view_config: ViewConfig
    device: torch.device
    checkpoint: Path
    run_name: str
    model_name: str


def load_predictor(
    checkpoint: Path, views_config: Path | None = DEFAULT_VIEWS_CONFIG, device: str | None = None
) -> Predictor:
    dev = select_device(device)
    ckpt = load_checkpoint(checkpoint, dev)
    threshold = ckpt.get("threshold")
    if threshold is None:
        raise ValueError(f"{checkpoint} has no validation threshold; use the run's best.pt")
    if views_config is not None and Path(views_config).exists():
        config = ViewConfig(**json.loads(Path(views_config).read_text()))
    else:
        log.warning("View config %s not found; using ViewConfig defaults", views_config)
        config = ViewConfig()
    try:
        model = load_model(ckpt, config.global_bins, config.local_bins, dev)
    except RuntimeError as exc:
        raise ValueError(
            f"Checkpoint does not accept {config.global_bins}/{config.local_bins}-bin views; "
            "pass the views_config.json used to build its training data"
        ) from exc
    cfg = ckpt["config"]
    return Predictor(model, float(threshold), config, dev, Path(checkpoint),
                     cfg.get("run_name", Path(checkpoint).parent.name), cfg["model"])


# ---------------------------------------------------------------------------
# 2. Light curve
# ---------------------------------------------------------------------------
def collect_fits(inputs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in map(Path, inputs):
        if p.is_dir():
            files += sorted(p.rglob("*.fits"))
        elif p.is_file():
            files.append(p)
        else:
            raise FileNotFoundError(p)
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No .fits files in {[str(p) for p in inputs]}")
    return files


def read_light_curve(files: list[Path], flux_column: str) -> tuple[list[Segment], dict]:
    """Read one star's quarters; refuses files from different targets."""
    segments, kepids, quarters, errors = [], set(), [], []
    for path in files:
        try:
            header = fits.getheader(path, 0)
            segments.append(read_kepler_fits(path, flux_column))
        except (OSError, KeyError, ValueError) as exc:
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
            continue
        if header.get("KEPLERID") is not None:
            kepids.add(int(header["KEPLERID"]))
        if header.get("QUARTER") is not None:
            quarters.append(int(header["QUARTER"]))
    if len(kepids) > 1:
        raise ValueError(f"Input files belong to different stars: KIC {sorted(kepids)}")
    if not segments:
        raise ValueError("No readable Kepler light-curve files:\n" + "\n".join(errors))
    times = np.concatenate([s.time[np.isfinite(s.time)] for s in segments])
    info = {
        "kepid": next(iter(kepids), None),
        "n_files": len(segments),
        "quarters": sorted(set(quarters)),
        "n_cadences": int(sum(s.time.size for s in segments)),
        "time_span_days": float(times.max() - times.min()) if times.size else 0.0,
        "unreadable_files": errors,
    }
    return segments, info


# ---------------------------------------------------------------------------
# Ephemerides
# ---------------------------------------------------------------------------
def ephemeris_from_args(period: float, t0: float, duration_hours: float) -> Ephemeris:
    return Ephemeris(period=period, t0=t0, duration=duration_hours / 24.0)


def ephemerides_from_table(path: Path, kepid: int) -> dict[str, Ephemeris]:
    """Every usable KOI of `kepid` (all are needed to mask companions)."""
    table = pd.read_csv(path)
    rows = table[table.kepid == kepid]
    out = {}
    for row in rows.to_dict("records"):
        try:
            out[row["kepoi_name"]] = ephemeris_from_row(row)
        except (KeyError, TypeError, ValueError) as exc:
            log.warning("Skipping %s: invalid ephemeris (%s)", row.get("kepoi_name"), exc)
    if not out:
        raise ValueError(f"No usable ephemeris for KIC {kepid} in {path}")
    return out


def dataset_membership(data_dir: Path, kepid: int | None) -> dict[str, list[str]]:
    """KOIs of this star already in the training / validation / test data."""
    out = {}
    for split in SPLITS:
        meta_path = Path(data_dir) / split / "meta.csv"
        if kepid is None or not meta_path.exists():
            continue
        meta = pd.read_csv(meta_path)
        names = meta.kepoi_name[meta.kepid == kepid].tolist()
        if names:
            out[split] = names
    return out


# ---------------------------------------------------------------------------
# 3-5. Preprocess, infer, vet
# ---------------------------------------------------------------------------
def _clean(value):
    """JSON-safe copy: numpy scalars -> Python, NaN/inf -> None."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def predict_candidates(
    predictor: Predictor,
    segments: list[Segment],
    ephems: dict[str, Ephemeris],
    targets: list[str] | None = None,
    explain: bool = True,
    vetting_config: VettingConfig = VettingConfig(),
) -> list[dict]:
    """One result dict per target candidate. Keys starting with '_' hold arrays for plotting."""
    config, device, model = predictor.view_config, predictor.device, predictor.model
    targets = targets or list(ephems)
    unknown = sorted(set(targets) - set(ephems))
    if unknown:
        raise ValueError(f"No ephemeris for {unknown}")
    bls = {r["kepoi_name"]: r for r in bls_features_from_segments(segments, ephems, config)}

    results = []
    for name in targets:
        eph = ephems[name]
        base = {
            "candidate": name,
            "ephemeris": {"period_days": eph.period, "t0_bkjd": eph.t0,
                          "duration_hours": eph.duration * 24.0},
        }
        others = [e for n, e in ephems.items() if n != name]
        try:
            views = generate_views(segments, eph, config, others)
        except (PreprocessingError, ValueError) as exc:
            results.append({**base, "status": "preprocessing_failed", "error": str(exc)})
            continue

        xg = torch.from_numpy(views.global_flux)[None, None].to(device)
        xl = torch.from_numpy(views.local_flux)[None, None].to(device)
        with torch.no_grad():
            prob = float(torch.sigmoid(compute_logits(model, xg, xl))[0])

        features = {
            **ephemeris_features(eph),
            **view_features(views.global_flux, views.local_flux, eph.period, eph.duration,
                            config),
            **{k: v for k, v in bls[name].items() if k != "kepoi_name"},
            "local_coverage": views.local_coverage,
        }
        attributions = None
        if explain:
            raw = {
                "gradcam": grad_cam(model, xg, xl),
                "saliency": saliency(model, xg, xl, smooth_samples=10),
                "occlusion": occlusion(model, xg, xl),
            }
            attributions = {m: {v: a[0].cpu().numpy() for v, a in per.items()}
                            for m, per in raw.items()}
            view = "local" if "local" in attributions["occlusion"] else "global"
            evidence = attributions["occlusion"][view]
            masks = region_masks(view, evidence.size, eph.period, eph.duration, config)
            features["transit_evidence_share"] = region_shares(evidence, masks)["transit_share"]

        decision = decide(prob, predictor.threshold, vet(features, vetting_config),
                          vetting_config)
        results.append({
            **base,
            "status": "ok",
            **decision,
            "data_quality": {
                "n_points_used": views.n_points,
                "global_coverage": views.global_coverage,
                "local_coverage": views.local_coverage,
                "n_transits_with_data": features.get("bls_n_transits"),
            },
            "features": features,
            "_views": {"global": views.global_flux, "local": views.local_flux},
            "_attributions": attributions,
        })
    return results


# ---------------------------------------------------------------------------
# 6. Report
# ---------------------------------------------------------------------------
SEVERITY_TAGS = {"false_positive": "FP  ", "warning": "WARN", "info": "INFO"}


def format_text_report(report: dict) -> str:
    lc, model = report["light_curve"], report["model"]
    lines = [
        f"Exoplanet candidate report - KIC {lc['kepid']}",
        f"Generated {report['generated_utc']} UTC",
        f"Model     : {model['architecture']} (run {model['run_name']}), threshold "
        f"{model['threshold']:.4f} selected on the validation split",
        f"Input     : {lc['n_files']} FITS files, quarters {lc['quarters'] or 'unknown'}, "
        f"{lc['n_cadences']:,} cadences over {lc['time_span_days']:.0f} days",
    ]
    if lc["unreadable_files"]:
        lines.append(f"Unreadable: {len(lc['unreadable_files'])} file(s) skipped")
    if report["in_dataset"]:
        lines.append("NOTE      : this star is already in the model's dataset "
                     f"{report['in_dataset']}; the prediction is not independent.")
    for r in report["candidates"]:
        e = r["ephemeris"]
        lines += [
            "",
            "=" * 78,
            f"Candidate {r['candidate']}: P = {e['period_days']:.5f} d, t0 = "
            f"{e['t0_bkjd']:.4f} BKJD, duration = {e['duration_hours']:.2f} h",
            "-" * 78,
        ]
        if r["status"] != "ok":
            lines.append(f"Preprocessing failed: {r['error']}")
            continue
        q = r["data_quality"]
        lines += [
            f"  Planet probability   : {r['planet_probability']:.4f}",
            f"  Model classification : {r['model_classification']}",
            f"  Confidence score     : {r['confidence_score']:.3f} ({r['confidence_level']})",
            f"  FINAL DISPOSITION    : {r['disposition']}",
            f"  Data quality         : {q['n_points_used']:,} points, view coverage "
            f"global {q['global_coverage']:.0%} / local {q['local_coverage']:.0%}, "
            f"{q['n_transits_with_data'] if q['n_transits_with_data'] is not None else '?'} "
            "transits",
        ]
        share = r["features"].get("transit_evidence_share")
        if share is not None:
            lines.append(f"  Model evidence       : {share:.0%} of occlusion evidence inside the "
                         "transit window")
        lines.append("  Reason codes:" if r["flags"] else "  Reason codes: none")
        for fl in r["flags"]:
            lines.append(f"    [{SEVERITY_TAGS[fl['severity']]}] {fl['code']}: {fl['message']}")
    lines += ["", CALIBRATION_NOTE]
    return "\n".join(lines) + "\n"


def write_report(out_dir: Path, report: dict, results: list[dict], config: ViewConfig) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(_clean(report), indent=2))
    (out_dir / "report.txt").write_text(format_text_report(_clean(report)), encoding="utf-8")
    for r in results:
        if r.get("_attributions") is None:
            continue
        e = r["ephemeris"]
        period, duration = e["period_days"], e["duration_hours"] / 24.0
        views = list(r["_attributions"]["gradcam"])
        plot_example(
            out_dir / f"{safe_name(r['candidate'])}.png",
            {v: r["_views"][v] for v in views},
            r["_attributions"],
            {v: plot_axis(v, r["_views"][v].size, period, duration, config) for v in views},
            f"{r['candidate']} - P(planet) = {r['planet_probability']:.3f}, "
            f"{r['disposition']} ({', '.join(r['reason_codes']) or 'no reason codes'})",
        )


def run(
    inputs: list[Path],
    predictor: Predictor,
    ephems: dict[str, Ephemeris] | None = None,
    koi_table: Path | None = None,
    targets: list[str] | None = None,
    data_dir: Path | None = DEFAULT_DATASET_DIR,
    out_dir: Path | None = None,
    explain: bool = True,
) -> dict:
    files = collect_fits(inputs)
    segments, lc_info = read_light_curve(files, predictor.view_config.flux_column)
    if koi_table is not None:
        if lc_info["kepid"] is None:
            raise ValueError("FITS headers have no KEPLERID; pass the ephemeris explicitly")
        ephems = ephemerides_from_table(koi_table, lc_info["kepid"])
    if not ephems:
        raise ValueError("No candidate ephemeris given")

    results = predict_candidates(predictor, segments, ephems, targets, explain)
    report = {
        "generated_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "model": {
            "checkpoint": str(predictor.checkpoint),
            "run_name": predictor.run_name,
            "architecture": predictor.model_name,
            "threshold": predictor.threshold,
            "view_config": asdict(predictor.view_config),
        },
        "light_curve": {**lc_info, "files": [str(f) for f in files]},
        "in_dataset": dataset_membership(data_dir, lc_info["kepid"]) if data_dir else {},
        "candidates": [{k: v for k, v in r.items() if not k.startswith("_")} for r in results],
        "notes": [CALIBRATION_NOTE],
    }
    target_name = f"kic{lc_info['kepid']:09d}" if lc_info["kepid"] else "target"
    out_dir = Path(out_dir or DEFAULT_OUT_DIR / target_name)
    write_report(out_dir, report, results, predictor.view_config)
    report["out_dir"] = str(out_dir)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Classify transit candidates in a Kepler light curve and write a report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("inputs", nargs="+", type=Path,
                   help="Kepler *_llc.fits files or directories with one star's files.")
    p.add_argument("--checkpoint", type=Path, required=True, help="Trained best.pt.")
    p.add_argument("--views-config", type=Path, default=DEFAULT_VIEWS_CONFIG,
                   help="views_config.json used to build the model's training views.")
    eph = p.add_argument_group("candidate ephemeris (give either all three values or --koi-table)")
    eph.add_argument("--period", type=float, help="Orbital period [days].")
    eph.add_argument("--t0", type=float, help="Mid-transit time [BKJD = BJD - 2454833].")
    eph.add_argument("--duration-hours", type=float, help="Transit duration [hours].")
    eph.add_argument("--name", default="candidate", help="Label for an explicit ephemeris.")
    eph.add_argument("--koi-table", type=Path, default=None,
                     help="CSV with kepid, kepoi_name, koi_period, koi_time0bk, koi_duration.")
    eph.add_argument("--kepoi-name", nargs="+", default=None,
                     help="Only report these KOIs (others on the star are still masked).")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATASET_DIR,
                   help="Dataset used to flag stars the model has already seen.")
    p.add_argument("--no-explain", action="store_true",
                   help="Skip Grad-CAM / saliency / occlusion (faster, no figure).")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Defaults to outputs/inference/kic<kepid>.")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    explicit = [args.period, args.t0, args.duration_hours]
    if args.koi_table is None and any(v is None for v in explicit):
        p.error("give --period, --t0 and --duration-hours, or --koi-table")
    if args.koi_table is not None and any(v is not None for v in explicit):
        p.error("use either an explicit ephemeris or --koi-table, not both")
    if args.kepoi_name and args.koi_table is None:
        p.error("--kepoi-name requires --koi-table")
    return args


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    predictor = load_predictor(args.checkpoint, args.views_config, args.device)
    ephems = None
    if args.koi_table is None:
        try:
            ephems = {args.name: ephemeris_from_args(args.period, args.t0, args.duration_hours)}
        except ValueError as exc:
            raise SystemExit(f"Invalid ephemeris: {exc}") from exc
    report = run(args.inputs, predictor, ephems, args.koi_table, args.kepoi_name,
                 args.data_dir, args.out_dir, not args.no_explain)
    print(format_text_report(_clean(report)))
    log.info("Report written to %s", report["out_dir"])
    return report


if __name__ == "__main__":
    main()
