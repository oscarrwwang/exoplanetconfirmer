"""Assemble phase-folded global/local views into train/val/test arrays.

Input:
    data/processed/views/<kepoi_name>.npz   (from `python -m src.preprocessing.views`)
Output, for each split in {train, val, test}:
    data/processed/dataset/<split>/X_global.npy   float32 (N, global_bins)
    data/processed/dataset/<split>/X_local.npy    float32 (N, local_bins)
    data/processed/dataset/<split>/y.npy          float32 (N,)
    data/processed/dataset/<split>/meta.csv       kepoi_name, kepid, label
    data/processed/dataset/summary.json

The split is stratified by label and grouped by star (kepid) so KOIs on the same star
never leak between splits.

Usage:
    python -m src.data.build_dataset
    python -m src.data.build_dataset --train 0.7 --val 0.15 --test 0.15 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIEWS_DIR = PROJECT_ROOT / "data" / "processed" / "views"
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "processed" / "dataset"

SPLITS = ("train", "val", "test")

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_views(views_dir: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Stack all view files into (X_global, X_local, meta). Skips malformed or non-finite views."""
    files = sorted(Path(views_dir).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz view files in {views_dir}")

    globals_, locals_, rows = [], [], []
    n_bad = 0
    for path in tqdm(files, desc="Loading views"):
        try:
            with np.load(path, allow_pickle=False) as data:
                g = np.asarray(data["global_flux"], dtype=np.float32)
                loc = np.asarray(data["local_flux"], dtype=np.float32)
                row = {
                    "kepoi_name": str(data["kepoi_name"]),
                    "kepid": int(data["kepid"]),
                    "label": int(data["label"]),
                }
        except (OSError, KeyError, ValueError) as exc:
            log.warning("Skipping unreadable %s (%s)", path.name, exc)
            n_bad += 1
            continue

        if g.ndim != 1 or loc.ndim != 1:
            log.warning("Skipping %s: views must be 1-D, got %s/%s", path.name, g.shape, loc.shape)
            n_bad += 1
            continue
        if not (np.isfinite(g).all() and np.isfinite(loc).all()):
            log.warning("Skipping %s: non-finite values", path.name)
            n_bad += 1
            continue
        if row["label"] not in (0, 1):
            log.warning("Skipping %s: label %s not in {0, 1}", path.name, row["label"])
            n_bad += 1
            continue

        globals_.append(g)
        locals_.append(loc)
        rows.append(row)

    if not rows:
        raise ValueError(f"No usable views in {views_dir}")

    # Use the majority shape so one odd file (whatever its sort order) can't reject the rest.
    shapes = [(g.size, loc.size) for g, loc in zip(globals_, locals_)]
    expected = max(set(shapes), key=shapes.count)
    keep = [i for i, s in enumerate(shapes) if s == expected]
    for i in sorted(set(range(len(rows))) - set(keep)):
        log.warning("Skipping %s: view lengths %s != %s", rows[i]["kepoi_name"], shapes[i], expected)
    n_bad += len(rows) - len(keep)
    globals_ = [globals_[i] for i in keep]
    locals_ = [locals_[i] for i in keep]
    rows = [rows[i] for i in keep]

    if n_bad:
        log.warning("Skipped %d of %d view files", n_bad, len(files))
    return np.stack(globals_), np.stack(locals_), pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def split_by_group(
    groups: np.ndarray,
    labels: np.ndarray,
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Stratified group split. Returns sorted sample indices per split.

    Each group (star) is assigned the max label of its samples, groups are shuffled
    within each class, and each class is cut by `fractions`.
    """
    fractions = np.asarray(fractions, dtype=np.float64)
    if fractions.shape != (3,) or (fractions < 0).any() or not np.isclose(fractions.sum(), 1.0):
        raise ValueError(f"fractions must be 3 non-negative values summing to 1, got {fractions}")
    groups = np.asarray(groups)
    labels = np.asarray(labels)
    if groups.shape != labels.shape:
        raise ValueError("groups and labels must have the same shape")

    group_label = pd.Series(labels).groupby(groups).max()
    rng = np.random.default_rng(seed)
    assignment: dict[object, int] = {}
    for _, members in group_label.groupby(group_label):
        ids = rng.permutation(members.index.to_numpy())
        n_train = int(round(fractions[0] * ids.size))
        n_val = int(round(fractions[1] * ids.size))
        n_val = min(n_val, ids.size - n_train)
        for split_id, chunk in enumerate(np.split(ids, [n_train, n_train + n_val])):
            assignment.update(dict.fromkeys(chunk.tolist(), split_id))

    split_of_sample = np.array([assignment[g] for g in groups.tolist()])
    return {name: np.flatnonzero(split_of_sample == i) for i, name in enumerate(SPLITS)}


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------
def save_split(
    out_dir: Path, x_global: np.ndarray, x_local: np.ndarray, meta: pd.DataFrame
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X_global.npy", x_global.astype(np.float32, copy=False))
    np.save(out_dir / "X_local.npy", x_local.astype(np.float32, copy=False))
    np.save(out_dir / "y.npy", meta["label"].to_numpy(dtype=np.float32))
    meta.to_csv(out_dir / "meta.csv", index=False)


def build_dataset(
    views_dir: Path = DEFAULT_VIEWS_DIR,
    out_dir: Path = DEFAULT_DATASET_DIR,
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> dict:
    x_global, x_local, meta = load_views(views_dir)
    splits = split_by_group(meta["kepid"].to_numpy(), meta["label"].to_numpy(), fractions, seed)

    summary = {
        "views_dir": str(views_dir),
        "seed": seed,
        "fractions": dict(zip(SPLITS, map(float, fractions))),
        "global_bins": int(x_global.shape[1]),
        "local_bins": int(x_local.shape[1]),
        "splits": {},
    }
    for name, idx in splits.items():
        split_meta = meta.iloc[idx].reset_index(drop=True)
        save_split(Path(out_dir) / name, x_global[idx], x_local[idx], split_meta)
        n_pos = int(split_meta["label"].sum())
        summary["splits"][name] = {
            "n_samples": int(idx.size),
            "n_stars": int(split_meta["kepid"].nunique()),
            "n_planet": n_pos,
            "n_false_positive": int(idx.size - n_pos),
        }
        log.info("%-5s %s", name, summary["splits"][name])

    (Path(out_dir) / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# PyTorch access
# ---------------------------------------------------------------------------
def load_split(
    split_dir: str | Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (X_global (N, 1, G), X_local (N, 1, L), y (N,)) float32 tensors."""
    split_dir = Path(split_dir)
    x_global = torch.from_numpy(np.load(split_dir / "X_global.npy")).unsqueeze(1)
    x_local = torch.from_numpy(np.load(split_dir / "X_local.npy")).unsqueeze(1)
    y = torch.from_numpy(np.load(split_dir / "y.npy"))
    if not (x_global.shape[0] == x_local.shape[0] == y.shape[0]):
        raise ValueError(f"Inconsistent sample counts in {split_dir}")
    return x_global, x_local, y


class TransitViewsDataset(Dataset):
    """Yields (global (1, G), local (1, L), label scalar) float32 tensors for one split."""

    def __init__(self, split_dir: str | Path) -> None:
        self.x_global, self.x_local, self.y = load_split(split_dir)

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x_global[index], self.x_local[index], self.y[index]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build train/val/test X_global.npy, X_local.npy, y.npy from transit views.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--views-dir", type=Path, default=DEFAULT_VIEWS_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_DATASET_DIR)
    p.add_argument("--train", type=float, default=0.7)
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_dataset(args.views_dir, args.out_dir, (args.train, args.val, args.test), args.seed)
    log.info("Dataset written to %s", args.out_dir)


if __name__ == "__main__":
    main()
