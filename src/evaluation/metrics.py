"""Binary classification metrics and validation-based threshold selection."""

from __future__ import annotations

import logging
import math

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

log = logging.getLogger(__name__)


def binary_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    """PR-AUC, ROC-AUC (threshold-free) and thresholded precision/recall/F1/accuracy/confusion."""
    labels = np.asarray(labels).astype(bool)
    probs = np.asarray(probs, dtype=np.float64)
    if labels.shape != probs.shape:
        raise ValueError(f"labels {labels.shape} and probs {probs.shape} differ in shape")

    pred = probs >= threshold
    tp = int(np.sum(pred & labels))
    fp = int(np.sum(pred & ~labels))
    tn = int(np.sum(~pred & ~labels))
    fn = int(np.sum(~pred & labels))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    both_classes = 0 < labels.sum() < labels.size
    return {
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(labels, probs)) if both_classes else math.nan,
        "roc_auc": float(roc_auc_score(labels, probs)) if both_classes else math.nan,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / labels.size if labels.size else math.nan,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "n": int(labels.size),
        "n_positive": int(labels.sum()),
    }


def select_threshold(labels: np.ndarray, probs: np.ndarray, default: float = 0.5) -> float:
    """Return the probability threshold that maximises F1. Call with VALIDATION data only."""
    labels = np.asarray(labels).astype(bool)
    if not 0 < labels.sum() < labels.size:
        log.warning("Threshold selection needs both classes; falling back to %.2f", default)
        return default
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    # The last precision/recall pair has no threshold.
    p, r = precision[:-1], recall[:-1]
    denom = p + r
    f1 = np.divide(2 * p * r, denom, out=np.zeros_like(denom), where=denom > 0)
    return float(thresholds[int(np.argmax(f1))])


def format_metrics(name: str, m: dict) -> str:
    cm = m["confusion_matrix"]
    return (
        f"{name}: PR-AUC {m['pr_auc']:.4f} | ROC-AUC {m['roc_auc']:.4f} | "
        f"precision {m['precision']:.4f} | recall {m['recall']:.4f} | F1 {m['f1']:.4f} | "
        f"accuracy {m['accuracy']:.4f} @ threshold {m['threshold']:.4f} | "
        f"confusion [[TN {cm['tn']}, FP {cm['fp']}], [FN {cm['fn']}, TP {cm['tp']}]]"
    )
