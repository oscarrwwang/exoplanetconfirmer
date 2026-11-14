"""Build star-level transit labels from the NASA Exoplanet Archive.

Planet (CONFIRMED) = 1, False Positive (FALSE POSITIVE) = 0.
Output columns: kepid, label, source, disposition.

Usage:
    python -m src.data.build_labels
    python -m src.data.build_labels --output data/labels/labels.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "labels" / "labels.csv"

TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
TABLE = "cumulative"
SOURCE = f"NASA Exoplanet Archive ({TABLE} KOI table)"
LABEL_MAP = {"CONFIRMED": 1, "FALSE POSITIVE": 0}

log = logging.getLogger("build_labels")


def fetch_dispositions() -> pd.DataFrame:
    """Return one row per KOI with kepid and koi_disposition."""
    in_clause = ", ".join(f"'{d}'" for d in LABEL_MAP)
    query = f"select kepid, koi_disposition from {TABLE} where koi_disposition in ({in_clause})"
    url = f"{TAP_URL}?{urlencode({'query': query, 'format': 'csv'})}"
    log.info("Querying NASA Exoplanet Archive: %s", query)
    return pd.read_csv(url)


def build_labels(kois: pd.DataFrame) -> pd.DataFrame:
    """Collapse KOIs to one label per star; any confirmed planet makes the star positive."""
    kois = kois.dropna(subset=["kepid", "koi_disposition"]).copy()
    kois["kepid"] = kois["kepid"].astype(int)
    kois["label"] = kois["koi_disposition"].map(LABEL_MAP)

    labels = (
        kois.sort_values(["kepid", "label"], ascending=[True, False])
        .drop_duplicates("kepid")
        .rename(columns={"koi_disposition": "disposition"})
    )
    labels["source"] = SOURCE
    return labels[["kepid", "label", "source", "disposition"]].reset_index(drop=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build labels.csv (planet=1, false positive=0) from the NASA Exoplanet Archive.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    kois = fetch_dispositions()
    log.info("Fetched %d KOIs: %s", len(kois), kois["koi_disposition"].value_counts().to_dict())

    labels = build_labels(kois)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(args.output, index=False)
    log.info(
        "Wrote %d stars to %s (planet=%d, false_positive=%d)",
        len(labels),
        args.output,
        int((labels["label"] == 1).sum()),
        int((labels["label"] == 0).sum()),
    )


if __name__ == "__main__":
    main()
