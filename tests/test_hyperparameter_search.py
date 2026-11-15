import json

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.build_dataset import save_split
from src.training import hyperparameter_search as hpo
from src.training import train as train_module


@pytest.fixture
def dataset_dir(tmp_path):
    rng = np.random.default_rng(0)
    for split, n in (("train", 24), ("val", 8), ("test", 8)):
        y = np.arange(n) % 2
        g = rng.normal(0, 0.1, (n, 2001)).astype(np.float32)
        loc = rng.normal(0, 0.1, (n, 201)).astype(np.float32)
        g[y == 1, 990:1010] -= 1.0
        loc[y == 1, 90:110] -= 1.0
        meta = pd.DataFrame({"kepoi_name": [f"K{i}" for i in range(n)], "kepid": range(n),
                             "label": y})
        save_split(tmp_path / "dataset" / split, g, loc, meta)
    return tmp_path


def run_search(tmp_path, *extra):
    return hpo.main([
        "--n-trials", "3",
        "--storage", "memory",
        "--study-name", "t",
        "--optuna-dir", str(tmp_path / "optuna"),
        "--checkpoint-dir", str(tmp_path / "ckpt"),
        "--log-dir", str(tmp_path / "runs"),
        "--no-prune",
        "--data-dir", str(tmp_path / "dataset"),
        "--model", "baseline",
        "--epochs", "2",
        "--device", "cpu",
        *extra,
    ])


def test_trial_args_override_forwarded_options(tmp_path):
    params = {"lr": 3e-4, "dropout": 0.2, "kernel_size": 7, "batch_size": 32, "base_filters": 8}
    args = hpo.trial_train_args(["--lr", "1.0", "--batch-size", "4"], params, "r",
                                tmp_path, tmp_path)
    assert (args.lr, args.dropout, args.kernel_size, args.batch_size, args.base_filters) == (
        3e-4, 0.2, 7, 32, 8)
    assert args.monitor == "val_pr_auc" and args.skip_test


def test_build_model_uses_kernel_size_and_filters():
    model = train_module.build_model("astronet", 2001, 201, 0.1, False, 3, 8)
    assert model.global_branch[0].kernel_size == (3,)
    assert model.global_branch[0].out_channels == 8
    assert model.local_branch[0].out_channels == 8
    baseline = train_module.build_model("baseline", 2001, 201, None, False, 7, 32)
    assert baseline.features[0][0].kernel_size == (7,)
    assert baseline.features[-1][0].out_channels == 256


def test_search_selects_by_val_pr_auc_without_touching_test(tmp_path, dataset_dir, monkeypatch):
    include_test_flags = []
    real = train_module.make_loaders

    def spy(*args, include_test=True, **kwargs):
        include_test_flags.append(include_test)
        return real(*args, include_test=include_test, **kwargs)

    monkeypatch.setattr(train_module, "make_loaders", spy)
    summary = run_search(dataset_dir)

    assert include_test_flags == [False, False, False]
    assert summary["objective"] == "validation PR-AUC"
    assert set(summary["best_params"]) == set(hpo.SEARCH_SPACE)
    assert summary["trial_counts"]["COMPLETE"] == 3
    assert "test_metrics_of_selected_model" not in summary

    trials = pd.read_csv(dataset_dir / "optuna" / "t" / "trials.csv")
    assert summary["best_val_pr_auc"] == pytest.approx(trials["value"].max())
    remaining = list((dataset_dir / "ckpt" / "t").iterdir())
    assert [p.name for p in remaining] == [f"trial_{summary['best_trial']:04d}"]
    saved = json.loads((dataset_dir / "optuna" / "t" / "best.json").read_text())
    assert saved["best_trial"] == summary["best_trial"]


def test_evaluate_test_scores_only_the_selected_model(dataset_dir):
    summary = run_search(dataset_dir, "--evaluate-test")
    test = summary["test_metrics_of_selected_model"]
    assert test["threshold"] == summary["threshold_from_validation"]
    assert test["n"] == 8
    ckpt = torch.load(summary["best_checkpoint"], weights_only=True)
    assert ckpt["threshold"] == summary["threshold_from_validation"]
