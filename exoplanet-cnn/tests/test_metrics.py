import math

import numpy as np
import pytest

from src.evaluation.metrics import binary_metrics, select_threshold


def test_binary_metrics_values():
    labels = np.array([1, 1, 0, 0, 1])
    probs = np.array([0.9, 0.4, 0.6, 0.1, 0.8])
    m = binary_metrics(labels, probs, threshold=0.5)
    assert m["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 2}
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["accuracy"] == pytest.approx(3 / 5)
    assert m["roc_auc"] == pytest.approx(5 / 6)
    assert 0 < m["pr_auc"] <= 1


def test_binary_metrics_no_positive_predictions():
    m = binary_metrics(np.array([1, 0]), np.array([0.1, 0.2]), threshold=0.5)
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0


def test_binary_metrics_single_class_auc_is_nan():
    m = binary_metrics(np.array([0, 0]), np.array([0.1, 0.2]), threshold=0.5)
    assert math.isnan(m["pr_auc"]) and math.isnan(m["roc_auc"])


def test_select_threshold_maximises_f1_not_fixed_half():
    # Weighted-BCE-like shift: positives score 0.7-0.9, negatives 0.55-0.65.
    labels = np.array([1, 1, 1, 0, 0, 0])
    probs = np.array([0.9, 0.8, 0.7, 0.65, 0.6, 0.55])
    t = select_threshold(labels, probs)
    assert t == pytest.approx(0.7)
    assert binary_metrics(labels, probs, t)["f1"] == 1.0
    assert binary_metrics(labels, probs, 0.5)["f1"] < 1.0


def test_select_threshold_single_class_falls_back():
    assert select_threshold(np.array([1, 1]), np.array([0.3, 0.9]), default=0.5) == 0.5
