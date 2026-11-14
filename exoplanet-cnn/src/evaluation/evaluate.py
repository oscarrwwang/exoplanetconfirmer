"""Evaluate a trained checkpoint on one split: metrics, per-KOI predictions and plots.

Threshold-dependent metrics (accuracy, precision, recall, F1, confusion matrix) use the
threshold stored in the checkpoint, which `src.training.train` selects on the validation
split. Scoring the test split requires --final-test and should be done once, for the single
model already selected on validation.

Outputs (default outputs/evaluation/<run_name>/<split>/):
    metrics.json  predictions.csv  roc_curve.png  pr_curve.png  confusion_matrix.png

Usage:
    python -m src.evaluation.evaluate --checkpoint checkpoints/<run>/best.pt
    python -m src.evaluation.evaluate --checkpoint checkpoints/<run>/best.pt --split test --final-test
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.data.build_dataset import DEFAULT_DATASET_DIR, TransitViewsDataset  # noqa: E402
from src.evaluation.metrics import binary_metrics, format_metrics  # noqa: E402
from src.training.train import (  # noqa: E402
    PROJECT_ROOT,
    build_model,
    compute_logits,
    load_checkpoint,
    select_device,
)

DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"
DEFAULT_TARGETS = (0.9, 0.95, 0.99)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def recall_at_precision(labels: np.ndarray, probs: np.ndarray, target: float) -> float:
    """Highest recall achievable while keeping precision >= target (NaN if one class)."""
    labels = np.asarray(labels).astype(bool)
    if not 0 < labels.sum() < labels.size:
        return math.nan
    precision, recall, _ = precision_recall_curve(labels, probs)
    ok = precision >= target
    return float(recall[ok].max()) if ok.any() else 0.0


def precision_at_recall(labels: np.ndarray, probs: np.ndarray, target: float) -> float:
    """Highest precision achievable while keeping recall >= target (NaN if one class)."""
    labels = np.asarray(labels).astype(bool)
    if not 0 < labels.sum() < labels.size:
        return math.nan
    precision, recall, _ = precision_recall_curve(labels, probs)
    return float(precision[recall >= target].max())


def evaluate_predictions(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    targets: tuple[float, ...] = DEFAULT_TARGETS,
) -> dict:
    """All metrics for one split. `threshold` must come from the validation split."""
    m = binary_metrics(labels, probs, threshold)
    cm = m["confusion_matrix"]
    negatives = cm["tn"] + cm["fp"]
    m["specificity"] = cm["tn"] / negatives if negatives else math.nan
    m["false_positive_rate"] = cm["fp"] / negatives if negatives else math.nan
    # PR-AUC of a random classifier equals the positive fraction.
    m["prevalence"] = m["n_positive"] / m["n"] if m["n"] else math.nan
    m["recall_at_precision"] = {str(t): recall_at_precision(labels, probs, t) for t in targets}
    m["precision_at_recall"] = {str(t): precision_at_recall(labels, probs, t) for t in targets}
    return m


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def load_model(
    checkpoint: dict, global_len: int, local_len: int, device: torch.device
) -> torch.nn.Module:
    """Rebuild the checkpoint's model in eval mode."""
    cfg = checkpoint["config"]
    model = build_model(
        cfg["model"], global_len, local_len, cfg.get("dropout"),
        cfg.get("batch_norm", False), cfg.get("kernel_size"), cfg.get("base_filters"),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval()


@torch.no_grad()
def predict_split(
    checkpoint: dict, split_dir: Path, device: torch.device, batch_size: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Return (probabilities, labels) for every example in `split_dir`."""
    ds = TransitViewsDataset(split_dir)
    model = load_model(checkpoint, ds.x_global.shape[-1], ds.x_local.shape[-1], device)
    probs = []
    for x_global, x_local, _ in DataLoader(ds, batch_size=batch_size):
        logits = compute_logits(model, x_global.to(device), x_local.to(device))
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs), ds.y.numpy()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def save_plots(labels: np.ndarray, probs: np.ndarray, m: dict, out_dir: Path, title: str) -> None:
    labels = np.asarray(labels).astype(bool)
    if 0 < labels.sum() < labels.size:
        fpr, tpr, _ = roc_curve(labels, probs)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(fpr, tpr, label=f"ROC-AUC = {m['roc_auc']:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="random")
        ax.set(xlabel="False positive rate", ylabel="True positive rate (recall)",
               title=f"ROC - {title}")
        ax.legend(loc="lower right")
        fig.savefig(out_dir / "roc_curve.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

        precision, recall, _ = precision_recall_curve(labels, probs)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(recall, precision, label=f"PR-AUC = {m['pr_auc']:.3f}")
        ax.axhline(m["prevalence"], color="k", ls="--", lw=1,
                   label=f"random = {m['prevalence']:.3f}")
        ax.scatter([m["recall"]], [m["precision"]], color="r", zorder=3,
                   label=f"threshold {m['threshold']:.3f}")
        ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02),
               title=f"Precision-Recall - {title}")
        ax.legend(loc="lower left")
        fig.savefig(out_dir / "pr_curve.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    else:
        log.warning("Only one class present; skipping ROC/PR plots")

    cm = m["confusion_matrix"]
    grid = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(grid, cmap="Blues")
    for (i, j), v in np.ndenumerate(grid):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > grid.max() / 2 else "black")
    ticks = ["False positive", "Planet"]
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=ticks, yticklabels=ticks,
           xlabel="Predicted", ylabel="Actual",
           title=f"{title} @ {m['threshold']:.3f}")
    fig.savefig(out_dir / "confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a checkpoint: metrics, predictions and ROC/PR/confusion plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=Path, required=True, help="best.pt from train.py.")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATASET_DIR)
    p.add_argument("--split", choices=("train", "val", "test"), default="val")
    p.add_argument("--final-test", action="store_true",
                   help="Required for --split test: confirms this is the one final evaluation.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Defaults to outputs/evaluation/<run_name>/<split>.")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    if args.split == "test" and not args.final_test:
        p.error("--split test requires --final-test; select models and thresholds on val only")
    return args


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = select_device(args.device)
    ckpt = load_checkpoint(args.checkpoint, device)
    threshold = ckpt.get("threshold")
    if threshold is None:
        raise ValueError(f"{args.checkpoint} has no validation threshold; use the run's best.pt")

    split_dir = Path(args.data_dir) / args.split
    probs, labels = predict_split(ckpt, split_dir, device)
    m = evaluate_predictions(labels, probs, threshold)
    m.update(split=args.split, checkpoint=str(args.checkpoint),
             threshold_source="validation split (stored in checkpoint)")

    run_name = ckpt["config"].get("run_name", args.checkpoint.parent.name)
    out_dir = args.out_dir or DEFAULT_OUT_DIR / run_name / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(split_dir / "meta.csv")
    meta.assign(prob_planet=probs, predicted=(probs >= threshold).astype(int)).to_csv(
        out_dir / "predictions.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(m, indent=2))
    save_plots(labels, probs, m, out_dir, f"{run_name} ({args.split})")

    log.info(format_metrics(args.split, m))
    log.info("Specificity %.4f | prevalence (random PR-AUC) %.4f", m["specificity"], m["prevalence"])
    for t, v in m["recall_at_precision"].items():
        log.info("Recall at precision >= %s: %.4f", t, v)
    log.info("Results written to %s", out_dir)
    return m


if __name__ == "__main__":
    main()
