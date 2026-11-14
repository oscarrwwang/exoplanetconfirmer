"""Download Kepler light curves from MAST for labeled KOIs.

Labels come from the NASA Exoplanet Archive cumulative KOI table. Light curves
are fetched with Lightkurve and stored per star under data/raw/.

Usage:
    python -m src.data.download_kepler --help
    python -m src.data.download_kepler --limit 50
    python -m src.data.download_kepler --dispositions CONFIRMED --limit 100 --workers 8
    python -m src.data.download_kepler --labels-only
    python -m src.data.download_kepler --koi-table data/labels/kepler_koi_labels.csv --limit 100
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlencode

import lightkurve as lk
import pandas as pd
from astropy.io import fits
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_LABELS_DIR = PROJECT_ROOT / "data" / "labels"

TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
VALID_DISPOSITIONS = ("CONFIRMED", "FALSE POSITIVE")
KOI_COLUMNS = (
    "kepid",
    "kepoi_name",
    "kepler_name",
    "koi_disposition",
    "koi_pdisposition",
    "koi_period",
    "koi_time0bk",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_model_snr",
)

log = logging.getLogger("download_kepler")


def fetch_koi_labels(dispositions: tuple[str, ...]) -> pd.DataFrame:
    """Query the Exoplanet Archive cumulative KOI table for the given dispositions."""
    in_clause = ", ".join(f"'{d}'" for d in dispositions)
    query = (
        f"select {', '.join(KOI_COLUMNS)} from cumulative "
        f"where koi_disposition in ({in_clause})"
    )
    url = f"{TAP_URL}?{urlencode({'query': query, 'format': 'csv'})}"
    log.info("Querying NASA Exoplanet Archive: %s", query)
    df = pd.read_csv(url)
    df["label"] = (df["koi_disposition"] == "CONFIRMED").astype(int)
    return df


def select_targets(
    labels: pd.DataFrame, limit: int | None, seed: int
) -> pd.DataFrame:
    """Pick unique stars per disposition, optionally capped at `limit` per class."""
    # A star hosting any confirmed planet is treated as a confirmed-planet host.
    per_star = (
        labels.sort_values("label", ascending=False)
        .drop_duplicates("kepid")[["kepid", "koi_disposition"]]
    )
    if limit is None:
        return per_star.reset_index(drop=True)
    rng = random.Random(seed)
    groups = []
    for _, group in per_star.groupby("koi_disposition"):
        idx = list(group.index)
        rng.shuffle(idx)
        groups.append(group.loc[idx[:limit]])
    return pd.concat(groups).reset_index(drop=True)


def star_dir(raw_dir: Path, kepid: int) -> Path:
    return raw_dir / f"kplr{int(kepid):09d}"


def quarantine_unreadable(directory: Path) -> list[Path]:
    """Rename truncated / corrupt FITS to *.fits.partial so they are downloaded again.

    Lightkurve reuses any existing file, so without this a cut-off download fails forever.
    """
    bad = []
    for path in Path(directory).rglob("*.fits"):
        try:
            with fits.open(path, memmap=False) as hdul:
                hdul["LIGHTCURVE"].data["PDCSAP_FLUX"]
        except Exception:  # any read failure means the file is unusable
            path.replace(path.with_name(path.name + ".partial"))
            bad.append(path)
    return bad


def download_star(
    kepid: int,
    raw_dir: Path,
    exptime: str,
    quarters: list[int] | None,
    overwrite: bool,
    retries: int,
) -> dict:
    out_dir = star_dir(raw_dir, kepid)
    # Written only after every file arrived, so interrupted downloads are resumed, not skipped.
    complete = out_dir / ".complete"
    if complete.exists() and not overwrite:
        n_existing = len(list(out_dir.rglob("*.fits")))
        return {"kepid": kepid, "status": "skipped", "n_files": n_existing, "error": ""}

    out_dir.mkdir(parents=True, exist_ok=True)
    last_error = ""
    for attempt in range(1, retries + 1):
        if bad := quarantine_unreadable(out_dir):
            log.debug("KIC %s: %d truncated file(s) set aside", kepid, len(bad))
        try:
            search = lk.search_lightcurve(
                f"KIC {kepid}",
                mission="Kepler",
                author="Kepler",
                exptime=exptime,
                quarter=quarters,
            )
            if len(search) == 0:
                return {"kepid": kepid, "status": "not_found", "n_files": 0, "error": ""}
            search.download_all(download_dir=str(out_dir))
            if bad := quarantine_unreadable(out_dir):
                raise RuntimeError(f"{len(bad)} truncated file(s), e.g. {bad[0].name}")
            n_files = len(list(out_dir.rglob("*.fits")))
            if n_files < len(search):
                raise RuntimeError(f"only {n_files} of {len(search)} files downloaded")
            complete.touch()
            return {"kepid": kepid, "status": "ok", "n_files": n_files, "error": ""}
        except Exception as exc:  # network / MAST errors are transient
            last_error = f"{type(exc).__name__}: {exc}"
            log.debug("KIC %s attempt %d failed: %s", kepid, attempt, last_error)
            time.sleep(2**attempt)
    return {"kepid": kepid, "status": "failed", "n_files": 0, "error": last_error}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Kepler light curves for confirmed planets and false positives.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dispositions",
        nargs="+",
        default=list(VALID_DISPOSITIONS),
        choices=VALID_DISPOSITIONS,
        help="KOI dispositions to download.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Max stars per disposition (default: all)."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument(
        "--exptime", choices=("long", "short"), default="long", help="Cadence (long=30 min)."
    )
    parser.add_argument(
        "--quarters", type=int, nargs="+", default=None, help="Restrict to Kepler quarters."
    )
    parser.add_argument("--workers", type=int, default=4, help="Parallel downloads.")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42, help="Seed for --limit sampling.")
    parser.add_argument("--overwrite", action="store_true", help="Re-download existing stars.")
    parser.add_argument(
        "--labels-only", action="store_true", help="Only fetch and save the label table."
    )
    parser.add_argument(
        "--koi-table", type=Path, default=None,
        help="Reuse a saved KOI table instead of querying the archive (reproducible targets).",
    )
    parser.add_argument(
        "--system-certs", action="store_true",
        help="Verify HTTPS with the OS certificate store via `truststore` (TLS-inspecting "
             "proxies); ignores REQUESTS_CA_BUNDLE / SSL_CERT_FILE.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def use_system_certificates() -> None:
    try:
        import truststore
    except ImportError as exc:
        raise SystemExit("--system-certs needs `pip install truststore`") from exc
    for var in ("REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "CURL_CA_BUNDLE"):
        os.environ.pop(var, None)
    truststore.inject_into_ssl()


def load_koi_table(path: Path, dispositions: tuple[str, ...]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["koi_disposition"].isin(dispositions)].copy()
    df["label"] = (df["koi_disposition"] == "CONFIRMED").astype(int)
    return df


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    args.labels_dir.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    if args.system_certs:
        use_system_certificates()

    labels_path = args.labels_dir / "kepler_koi_labels.csv"
    if args.koi_table is not None:
        labels = load_koi_table(args.koi_table, tuple(args.dispositions))
        log.info("Loaded %d KOI labels from %s", len(labels), args.koi_table)
    else:
        labels = fetch_koi_labels(tuple(args.dispositions))
        labels.to_csv(labels_path, index=False)
        log.info("Saved %d KOI labels to %s", len(labels), labels_path)
    log.info("Dispositions: %s", labels["koi_disposition"].value_counts().to_dict())
    if args.labels_only:
        return

    targets = select_targets(labels, args.limit, args.seed)
    log.info("Downloading light curves for %d stars to %s", len(targets), args.raw_dir)

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                download_star,
                int(row.kepid),
                args.raw_dir,
                args.exptime,
                args.quarters,
                args.overwrite,
                args.retries,
            ): row
            for row in targets.itertuples(index=False)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Stars"):
            row = futures[future]
            result = future.result()
            result["koi_disposition"] = row.koi_disposition
            results.append(result)

    manifest = pd.DataFrame(results).sort_values("kepid")
    manifest_path = args.raw_dir / "download_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    log.info("Status: %s", manifest["status"].value_counts().to_dict())
    log.info("Manifest written to %s", manifest_path)


if __name__ == "__main__":
    main()
