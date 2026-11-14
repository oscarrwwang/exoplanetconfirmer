import json
import math

import numpy as np
import pandas as pd
import pytest

from src.data.build_dataset import save_split
from src.evaluation import evaluate as ev
from src.training import train as train_module


def test_operating_points():
    labels = np.array([1, 1, 1, 0, 1, 0, 0, 0])
    probs = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    # Top 3 are all planets: recall 3/4 at precision 1.
    assert ev.recall_at_precision(labels, probs, 1.0) == pytest.approx(0.75)
    # Recall 1 needs the top 5 (4 planets, 1 FP): precision 0.8.
    assert ev.precision_at_recall(labels, probs, 1.0) == pytest.approx(0.8)
    assert math.isnan(ev.recall_at_precision(np.ones(4), np.ones(4), 0.9))


def test_evaluate_predictions_adds_specificity_and_prevalence():
    labels = np.array([1, 1, 0, 0, 0])
    probs = np.array([0.9, 0.2, 0.8, 0.1, 0.1])
    m = ev.evaluate_predictions(labels, probs, 0.5)
    assert m["confusion_matrix"] == {"tn": 2, "fp": 1, "fn": 1, "tp": 1}
    assert m["specificity"] == pytest.approx(2 / 3)
    assert m["false_positive_rate"] == pytest.approx(1 / 3)
    assert m["prevalence"] == pytest.approx(0.4)
    assert set(m["recall_at_precision"]) == {"0.9", "0.95", "0.99"}


@pytest.fixture
def trained(tmp_path):
    rng = np.random.default_rng(0)
    for split, n in (("train", 24), ("val", 8), ("test", 8)):
        y = np.arange(n) % 2
        g = rng.normal(0, 0.1, (n, 2001)).astype(np.float32)
        loc = rng.normal(0, 0.1, (n, 201)).astype(np.float32)
        g[y == 1, 990:1010] -= 1.0
        meta = pd.DataFrame({"kepoi_name": [f"K{i}" for i in range(n)], "kepid": range(n),
                             "label": y})
        save_split(tmp_path / "dataset" / split, g, loc, meta)
    args = train_module.parse_args([
        "--data-dir", str(tmp_path / "dataset"), "--checkpoint-dir", str(tmp_path / "ckpt"),
        "--log-dir", str(tmp_path / "runs"), "--model", "baseline", "--epochs", "1",
        "--device", "cpu", "--run-name", "r", "--skip-test",
    ])
    train_module.train(args)
    return tmp_path


def test_evaluate_val_writes_outputs(trained):
    out = trained / "eval"
    m = ev.main(["--checkpoint", str(trained / "ckpt" / "r" / "best.pt"),
                 "--data-dir", str(trained / "dataset"), "--out-dir", str(out),
                 "--device", "cpu"])
    assert m["split"] == "val" and m["n"] == 8
    for name in ("metrics.json", "predictions.csv", "roc_curve.png", "pr_curve.png",
                 "confusion_matrix.png"):
        assert (out / name).exists()
    saved = json.loads((out / "metrics.json").read_text())
    assert saved["threshold"] == pytest.approx(m["threshold"])
    preds = pd.read_csv(out / "predictions.csv")
    assert list(preds.columns) == ["kepoi_name", "kepid", "label", "prob_planet", "predicted"]


def test_test_split_requires_final_flag():
    with pytest.raises(SystemExit):
        ev.parse_args(["--checkpoint", "x.pt", "--split", "test"])
    assert ev.parse_args(["--checkpoint", "x.pt", "--split", "test", "--final-test"]).final_test
