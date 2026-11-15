import json
import math

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.build_dataset import save_split
from src.training import train as train_module
from src.training.train import EarlyStopping, main

REQUIRED_METRICS = {"pr_auc", "roc_auc", "precision", "recall", "f1", "confusion_matrix"}
# Different class ratios per split so leakage into pos_weight would change its value.
SPLIT_LABELS = {
    "train": lambda i: int(i % 3 == 0),  # 8 pos / 16 neg -> pos_weight 2.0
    "val": lambda i: i % 2,
    "test": lambda i: int(i % 4 == 0),
}


@pytest.fixture
def dataset_dir(tmp_path):
    rng = np.random.default_rng(0)
    for split, n in (("train", 24), ("val", 8), ("test", 8)):
        y = np.array([SPLIT_LABELS[split](i) for i in range(n)])
        g = rng.normal(0, 0.1, (n, 2001)).astype(np.float32)
        loc = rng.normal(0, 0.1, (n, 201)).astype(np.float32)
        g[y == 1, 990:1010] -= 1.0
        loc[y == 1, 90:110] -= 1.0
        meta = pd.DataFrame({"kepoi_name": [f"K{i}" for i in range(n)], "kepid": range(n),
                             "label": y})
        save_split(tmp_path / "dataset" / split, g, loc, meta)
    return tmp_path


def run(tmp_path, *extra):
    return main([
        "--data-dir", str(tmp_path / "dataset"),
        "--checkpoint-dir", str(tmp_path / "ckpt"),
        "--log-dir", str(tmp_path / "runs"),
        "--run-name", "test",
        "--batch-size", "8",
        "--device", "cpu",
        *extra,
    ])


def test_early_stopping():
    stopper = EarlyStopping(patience=2, mode="min")
    assert stopper.step(1.0)
    assert stopper.step(0.5)
    assert not stopper.step(0.6) and not stopper.should_stop
    assert not stopper.step(0.7) and stopper.should_stop


@pytest.mark.parametrize("model", ["baseline", "astronet"])
def test_training_writes_checkpoints_logs_and_metrics(dataset_dir, model):
    results = run(dataset_dir, "--model", model, "--epochs", "2", "--lr", "1e-3")
    run_dir = dataset_dir / "ckpt" / "test"
    assert (run_dir / "best.pt").exists() and (run_dir / "last.pt").exists()
    assert any((dataset_dir / "runs" / "test").glob("events.out.tfevents.*"))
    saved = json.loads((run_dir / "metrics.json").read_text())
    assert saved["best_epoch"] in (1, 2)
    for split in ("train", "val", "test"):
        assert REQUIRED_METRICS <= set(results[split])
        assert results[split]["threshold"] == results["threshold"]
    best = torch.load(run_dir / "best.pt", weights_only=True)
    assert best["threshold"] == results["threshold"]


def test_pos_weight_uses_training_split_only(dataset_dir):
    results = run(dataset_dir, "--model", "baseline", "--epochs", "1")
    assert results["pos_weight"] == 2.0
    assert results["train_class_counts"] == {"positive": 8, "negative": 16}


def test_threshold_selected_on_validation_only(dataset_dir, monkeypatch):
    calls = []
    real = train_module.select_threshold

    def spy(labels, probs, *args, **kwargs):
        calls.append(np.asarray(labels).copy())
        return real(labels, probs, *args, **kwargs)

    monkeypatch.setattr(train_module, "select_threshold", spy)
    results = run(dataset_dir, "--model", "baseline", "--epochs", "1")
    val_y = np.load(dataset_dir / "dataset" / "val" / "y.npy")
    assert len(calls) == 1 and np.array_equal(calls[0], val_y)
    assert results["threshold_source"] == "validation split, max F1"


def test_fixed_threshold_skips_selection(dataset_dir):
    results = run(dataset_dir, "--model", "baseline", "--epochs", "1", "--threshold", "0.3")
    assert results["threshold"] == 0.3
    assert results["val"]["threshold"] == 0.3


def test_val_and_test_sets_are_not_modified(dataset_dir):
    before = {s: np.load(dataset_dir / "dataset" / s / "y.npy") for s in ("val", "test")}
    results = run(dataset_dir, "--model", "baseline", "--epochs", "1")
    for split, y in before.items():
        assert results[split]["n"] == y.size
        assert results[split]["n_positive"] == int(y.sum())


def test_baseline_learns_synthetic_transits(dataset_dir):
    results = run(dataset_dir, "--model", "baseline", "--epochs", "15", "--lr", "1e-3",
                  "--scheduler", "none")
    assert results["val"]["roc_auc"] >= 0.9


@pytest.mark.parametrize("loss_args", [
    ["--loss", "bce"],
    ["--loss", "weighted_bce", "--pos-weight", "2.0"],
    ["--loss", "focal", "--focal-gamma", "1.5"],
    ["--loss", "focal", "--focal-alpha", "none"],
])
def test_loss_switch(dataset_dir, loss_args):
    results = run(dataset_dir, "--model", "baseline", "--epochs", "1", *loss_args)
    assert all(math.isfinite(results[s]["loss"]) for s in ("train", "val", "test"))


def test_rejects_bad_loss_option(dataset_dir):
    with pytest.raises(SystemExit):
        run(dataset_dir, "--loss", "focal", "--focal-alpha", "high")


def test_early_stopping_and_resume(dataset_dir):
    # lr=0 with no BatchNorm keeps val loss constant, so stopping happens after `patience` epochs.
    results = run(dataset_dir, "--model", "astronet", "--epochs", "30", "--lr", "0",
                  "--patience", "2", "--scheduler", "none")
    assert results["last_epoch"] == 3

    last = dataset_dir / "ckpt" / "test" / "last.pt"
    resumed = run(dataset_dir, "--model", "astronet", "--epochs", "5", "--lr", "0",
                  "--patience", "0", "--scheduler", "none", "--resume", str(last))
    assert resumed["last_epoch"] == 5
