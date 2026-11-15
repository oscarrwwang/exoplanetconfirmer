import json

import numpy as np
import pytest
import torch

from src.data.build_dataset import TransitViewsDataset, build_dataset, load_split, split_by_group


def write_view(path, name, kepid, label, g_len=21, l_len=7, value=0.0):
    np.savez_compressed(
        path / f"{name}.npz",
        global_flux=np.full(g_len, value, np.float32),
        local_flux=np.full(l_len, value, np.float32),
        label=np.int8(label),
        kepid=np.int64(kepid),
        kepoi_name=np.str_(name),
    )


def test_split_by_group_fractions_and_no_leakage():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(200), 2)
    labels = np.repeat(rng.integers(0, 2, 200), 2)
    splits = split_by_group(groups, labels, (0.7, 0.15, 0.15), seed=1)

    all_idx = np.concatenate(list(splits.values()))
    assert np.array_equal(np.sort(all_idx), np.arange(groups.size))
    star_sets = [set(groups[idx]) for idx in splits.values()]
    assert sum(len(s) for s in star_sets) == len(set.union(*star_sets))
    sizes = np.array([len(s) for s in star_sets]) / 200
    assert np.allclose(sizes, [0.7, 0.15, 0.15], atol=0.02)


def test_split_by_group_is_deterministic():
    groups = np.arange(50)
    labels = groups % 2
    a = split_by_group(groups, labels, seed=3)
    b = split_by_group(groups, labels, seed=3)
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_split_by_group_rejects_bad_fractions():
    with pytest.raises(ValueError):
        split_by_group(np.arange(4), np.zeros(4), (0.5, 0.5, 0.5))


def test_build_dataset_end_to_end(tmp_path):
    views = tmp_path / "views"
    views.mkdir()
    for i in range(40):
        write_view(views, f"K{i:05d}.01", kepid=i, label=i % 2, value=float(i))
    write_view(views, "K00000.02", kepid=0, label=0)  # second KOI on star 0
    write_view(views, "bad", kepid=999, label=1, g_len=5)  # wrong shape -> skipped

    out = tmp_path / "dataset"
    summary = build_dataset(views, out, seed=0)

    assert sum(s["n_samples"] for s in summary["splits"].values()) == 41
    assert json.loads((out / "summary.json").read_text())["global_bins"] == 21
    for split in ("train", "val", "test"):
        assert (out / split / "X_global.npy").exists()
        g, loc, y = load_split(out / split)
        assert g.shape[1:] == (1, 21) and loc.shape[1:] == (1, 7)
        assert g.dtype == torch.float32 and y.dtype == torch.float32

    ds = TransitViewsDataset(out / "train")
    g, loc, y = ds[0]
    assert g.shape == (1, 21) and loc.shape == (1, 7) and y.ndim == 0
