"""Rule-based false-positive vetting with human-readable reason codes.

Consumes the per-candidate features from `src.baselines.features` (BLS statistics and view
shape), the CNN probability, and optionally the occlusion evidence from
`src.evaluation.explain`. It does not compute features itself.

Reason codes (severity):
    ODD_EVEN_DEPTH          false_positive  odd vs even transit depths differ  -> EB at 2P
    SECONDARY_ECLIPSE       false_positive  significant dip at phase 0.5       -> EB
    V_SHAPED                false_positive  centre much deeper than the mean   -> grazing EB
    TOO_DEEP                false_positive  depth too large for a planet
    SINUSOIDAL              false_positive  sinusoid fits better than a box    -> variability
    OUT_OF_TRANSIT_DIPS     warning         other dips in the folded curve
    LOW_SNR                 warning         transit not significant
    FEW_TRANSITS            warning         too few events to fold reliably
    SINGLE_EVENT            warning         one event dominates the signal     -> systematic
    ASYMMETRIC              warning         ingress / egress differ
    NOISY_FOLD              warning         out-of-transit scatter comparable to the depth
    LOW_COVERAGE            warning         many empty bins in the local view
    NO_BLS_FIT              warning         BLS statistics unavailable
    MODEL_OFF_TRANSIT       warning         model evidence mostly outside the transit window
    MODEL_LOW_CONFIDENCE    info            probability close to the decision threshold

Limits are fixed, physically motivated defaults (VettingConfig), not fitted to labels.
`python -m src.inference.vetting` reports how often each code fires per class on TRAIN / VAL.

Usage:
    python -m src.inference.vetting --checkpoint checkpoints/<run>/best.pt \
        --raw-dir data/raw_small --koi-table data/labels/kepler_koi_labels_small.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from src.baselines.features import DEFAULT_FEATURES_PATH, load_or_compute_features
from src.data.build_dataset import DEFAULT_DATASET_DIR
from src.evaluation.evaluate import predict_split
from src.preprocessing.preprocess import DEFAULT_RAW_DIR, PROJECT_ROOT
from src.preprocessing.views import DEFAULT_KOI_TABLE, DEFAULT_VIEWS_DIR
from src.training.train import load_checkpoint, select_device

DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "vetting"
SEVERITIES = ("false_positive", "warning", "info")

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class VettingConfig:
    max_odd_even_sigma: float = 3.0
    max_secondary_snr: float = 3.0
    # Inner-half / full-transit mean depth: box = 1.0, triangle (V) = 1.5.
    max_v_shape: float = 1.3
    # 3% depth is ~1.7 Jupiter radii on a Sun-like star.
    max_depth_ppm: float = 30_000.0
    # > 0 means a sinusoid fits better than a box.
    max_harmonic_dll: float = 0.0
    # Kepler pipeline detection threshold (7.1 sigma).
    min_snr: float = 7.1
    min_transits: int = 3
    max_single_event_frac: float = 0.5
    # Global view is normalized so the transit bottom is -1.
    max_oot_dip: float = 0.5
    max_asymmetry: float = 0.25
    max_oot_noise: float = 0.3
    min_local_coverage: float = 0.5
    min_transit_evidence: float = 0.3
    high_confidence: float = 0.6
    low_confidence: float = 0.25


@dataclass(frozen=True)
class Flag:
    code: str
    severity: str
    message: str
    value: float | None = None
    limit: float | None = None


MESSAGES = {
    "ODD_EVEN_DEPTH": ("false_positive", "Odd and even transit depths differ by {value:.1f} sigma "
                       "(limit {limit:g}): eclipsing binary at twice the period?"),
    "SECONDARY_ECLIPSE": ("false_positive", "Dip at orbital phase 0.5 with SNR {value:.1f} "
                          "(limit {limit:g}): secondary eclipse of an eclipsing binary?"),
    "V_SHAPED": ("false_positive", "V-shaped transit (centre/mean depth {value:.2f} > {limit:g}): "
                 "grazing eclipsing binary?"),
    "TOO_DEEP": ("false_positive", "Transit depth {value:,.0f} ppm > {limit:,.0f} ppm: companion "
                 "likely too large to be a planet"),
    "SINUSOIDAL": ("false_positive", "A sinusoid fits better than a transit box (delta logL "
                   "{value:.1f}): stellar variability or ellipsoidal modulation?"),
    "OUT_OF_TRANSIT_DIPS": ("warning", "Out-of-transit dip reaching {value:.0%} of the transit "
                            "depth: additional eclipses or systematics"),
    "LOW_SNR": ("warning", "Transit SNR {value:.1f} is below {limit:g}: the signal may not be real"),
    "FEW_TRANSITS": ("warning", "Only {value:.0f} transits with data (< {limit:g})"),
    "SINGLE_EVENT": ("warning", "One event carries {value:.0%} of the transit signal: possible "
                     "systematic or single-event artefact"),
    "ASYMMETRIC": ("warning", "Asymmetric transit: ingress and egress halves differ by "
                   "{value:.2f} of the depth"),
    "NOISY_FOLD": ("warning", "Out-of-transit scatter is {value:.0%} of the transit depth"),
    "LOW_COVERAGE": ("warning", "Only {value:.0%} of local-view bins contain data"),
    "NO_BLS_FIT": ("warning", "BLS statistics unavailable ({value}); transit tests skipped"),
    "MODEL_OFF_TRANSIT": ("warning", "Only {value:.0%} of the model's occlusion evidence lies in "
                          "the transit window (limit {limit:.0%})"),
    "MODEL_LOW_CONFIDENCE": ("info", "Planet probability {value:.3f} is close to the decision "
                             "threshold {limit:.3f}"),
}


def make_flag(code: str, value, limit=None) -> Flag:
    severity, template = MESSAGES[code]
    return Flag(code, severity, template.format(value=value, limit=limit),
                value if isinstance(value, (int, float)) else None, limit)


def _finite(features: Mapping, key: str) -> float | None:
    value = features.get(key)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def vet(features: Mapping, config: VettingConfig = VettingConfig()) -> list[Flag]:
    """Reason codes from candidate features. Missing / NaN features skip their check.

    Recognised optional keys beyond src.baselines.features: local_coverage,
    transit_evidence_share (occlusion share in the local transit window).
    """
    flags: list[Flag] = []
    f = lambda key: _finite(features, key)  # noqa: E731
    c = config

    status = features.get("bls_status", "ok")
    if status != "ok":
        flags.append(make_flag("NO_BLS_FIT", status if isinstance(status, str) else "missing"))
    else:
        checks_above = (
            ("ODD_EVEN_DEPTH", "bls_odd_even_sigma", c.max_odd_even_sigma),
            ("SECONDARY_ECLIPSE", "bls_secondary_snr", c.max_secondary_snr),
            ("TOO_DEEP", "bls_depth_ppm", c.max_depth_ppm),
            ("SINUSOIDAL", "bls_harmonic_dll", c.max_harmonic_dll),
        )
        for code, key, limit in checks_above:
            if (v := f(key)) is not None and v > limit:
                flags.append(make_flag(code, v, limit))
        if (v := f("bls_snr")) is not None and v < c.min_snr:
            flags.append(make_flag("LOW_SNR", v, c.min_snr))
        n_transits = f("bls_n_transits")
        if n_transits is not None and n_transits < c.min_transits:
            flags.append(make_flag("FEW_TRANSITS", n_transits, c.min_transits))
        elif (v := f("bls_max_transit_ll_frac")) is not None and v > c.max_single_event_frac:
            flags.append(make_flag("SINGLE_EVENT", v, c.max_single_event_frac))

    if (v := f("local_v_shape")) is not None and v > c.max_v_shape:
        flags.append(make_flag("V_SHAPED", v, c.max_v_shape))
    if (v := f("global_min_oot")) is not None and -v > c.max_oot_dip:
        flags.append(make_flag("OUT_OF_TRANSIT_DIPS", -v, c.max_oot_dip))
    if (v := f("local_asymmetry")) is not None and abs(v) > c.max_asymmetry:
        flags.append(make_flag("ASYMMETRIC", abs(v), c.max_asymmetry))
    if (v := f("local_oot_std")) is not None and v > c.max_oot_noise:
        flags.append(make_flag("NOISY_FOLD", v, c.max_oot_noise))
    if (v := f("local_coverage")) is not None and v < c.min_local_coverage:
        flags.append(make_flag("LOW_COVERAGE", v, c.min_local_coverage))
    if (v := f("transit_evidence_share")) is not None and v < c.min_transit_evidence:
        flags.append(make_flag("MODEL_OFF_TRANSIT", v, c.min_transit_evidence))
    return flags


# ---------------------------------------------------------------------------
# Model confidence and final disposition
# ---------------------------------------------------------------------------
def confidence_score(prob: float, threshold: float) -> float:
    """Distance from the decision threshold, scaled to [0, 1] on each side."""
    if prob >= threshold:
        return (prob - threshold) / (1 - threshold) if threshold < 1 else 0.0
    return (threshold - prob) / threshold if threshold > 0 else 0.0


def confidence_level(score: float, config: VettingConfig = VettingConfig()) -> str:
    if score >= config.high_confidence:
        return "high"
    return "medium" if score >= config.low_confidence else "low"


def decide(
    prob: float,
    threshold: float,
    flags: list[Flag],
    config: VettingConfig = VettingConfig(),
) -> dict:
    """Combine the CNN probability with the vetting flags into a final disposition.

    PLANET CANDIDATE  model says planet, confident, and no false-positive flag
    FALSE POSITIVE    model says false positive (confidently or backed by a flag)
    NEEDS REVIEW      model says planet but a false-positive test fires, or low confidence
    """
    score = confidence_score(prob, threshold)
    level = confidence_level(score, config)
    flags = list(flags)
    if level == "low":
        flags.append(make_flag("MODEL_LOW_CONFIDENCE", prob, threshold))
    fp_codes = [fl.code for fl in flags if fl.severity == "false_positive"]
    model_planet = prob >= threshold

    if model_planet:
        disposition = "NEEDS REVIEW" if fp_codes or level == "low" else "PLANET CANDIDATE"
    else:
        disposition = "NEEDS REVIEW" if level == "low" and not fp_codes else "FALSE POSITIVE"
    return {
        "planet_probability": float(prob),
        "threshold": float(threshold),
        "model_classification": "PLANET" if model_planet else "FALSE POSITIVE",
        "confidence_score": float(score),
        "confidence_level": level,
        "disposition": disposition,
        "reason_codes": [fl.code for fl in flags],
        "flags": [asdict(fl) for fl in flags],
    }


# ---------------------------------------------------------------------------
# Batch validation on existing splits (TRAIN / VAL only)
# ---------------------------------------------------------------------------
def vet_table(
    features: pd.DataFrame, probs: np.ndarray, threshold: float,
    config: VettingConfig = VettingConfig(),
) -> pd.DataFrame:
    rows = []
    for (_, feat), prob in zip(features.iterrows(), probs):
        d = decide(prob, threshold, vet(feat.to_dict(), config), config)
        rows.append({"kepoi_name": feat["kepoi_name"], **{k: d[k] for k in (
            "planet_probability", "model_classification", "confidence_score",
            "confidence_level", "disposition")}, "reason_codes": ";".join(d["reason_codes"])})
    return pd.DataFrame(rows)


def flag_summary(table: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """How often each code fires per class, and the false-positive fraction among flagged."""
    labels = np.asarray(labels).astype(int)
    codes = table.reason_codes.str.split(";")
    rows = []
    for code, (severity, _) in MESSAGES.items():
        hit = codes.apply(lambda cs: code in cs).to_numpy()
        n_fp, n_pl = int((hit & (labels == 0)).sum()), int((hit & (labels == 1)).sum())
        rows.append({
            "code": code, "severity": severity,
            "flagged_false_positives": n_fp, "flagged_planets": n_pl,
            "fp_rate_flagged": n_fp / max(1, int((labels == 0).sum())),
            "planet_rate_flagged": n_pl / max(1, int((labels == 1).sum())),
            "fp_fraction_of_flagged": n_fp / (n_fp + n_pl) if n_fp + n_pl else np.nan,
        })
    return pd.DataFrame(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run reason-code vetting on TRAIN / VAL candidates and summarise per class.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=Path, required=True, help="CNN best.pt.")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATASET_DIR)
    p.add_argument("--views-dir", type=Path, default=DEFAULT_VIEWS_DIR)
    p.add_argument("--koi-table", type=Path, default=DEFAULT_KOI_TABLE)
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument("--features", type=Path, default=DEFAULT_FEATURES_PATH)
    p.add_argument("--splits", nargs="+", choices=("train", "val"), default=["train", "val"],
                   help="The test split is deliberately not available here.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--device", default=None)
    p.add_argument("--workers", type=int, default=1)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, pd.DataFrame]:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = select_device(args.device)
    ckpt = load_checkpoint(args.checkpoint, device)
    threshold = ckpt.get("threshold")
    if threshold is None:
        raise ValueError(f"{args.checkpoint} has no validation threshold; use best.pt")
    features = load_or_compute_features(tuple(args.splits), args.data_dir, args.views_dir,
                                        args.koi_table, args.raw_dir, args.features, args.workers)
    config = VettingConfig()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "vetting_config.json").write_text(json.dumps(asdict(config), indent=2))

    summaries = {}
    for split in args.splits:
        split_dir = Path(args.data_dir) / split
        meta = pd.read_csv(split_dir / "meta.csv")
        feats = meta[["kepoi_name"]].merge(features, on="kepoi_name", how="left")
        probs, _ = predict_split(ckpt, split_dir, device)
        table = vet_table(feats, probs, threshold, config)
        table.insert(1, "label", meta.label.to_numpy())
        table.to_csv(args.out_dir / f"{split}_vetting.csv", index=False)
        summary = flag_summary(table, meta.label.to_numpy())
        summary.to_csv(args.out_dir / f"{split}_flag_summary.csv", index=False)
        summaries[split] = summary
        crosstab = pd.crosstab(table.disposition,
                               np.where(meta.label == 1, "planet", "false positive"))
        log.info("%s dispositions vs labels:\n%s", split, crosstab.to_string())
        fired = summary[(summary.flagged_false_positives + summary.flagged_planets) > 0]
        log.info("%s reason codes:\n%s", split, fired[[
            "code", "severity", "flagged_false_positives", "flagged_planets",
            "fp_fraction_of_flagged"]].to_string(index=False))
    log.info("Vetting written to %s", args.out_dir)
    return summaries


if __name__ == "__main__":
    main()
