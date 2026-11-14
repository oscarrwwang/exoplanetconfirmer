"""Compare classical baselines with the CNN on the same star-grouped splits.

Models
    BLS SNR          transit significance at the catalog ephemeris. No training.
    BLS vetting      -max(odd/even depth sigma, secondary-eclipse SNR): the classic Kepler DV
                     eclipsing-binary tests applied to the BLS fit. No training.
    Random Forest    engineered features (src.baselines.features), class_weight from TRAIN.
    XGBoost          same features, scale_pos_weight = n_neg / n_pos from TRAIN.
    CNN              a trained checkpoint (src.training.train), optional.

Protocol (identical to the CNN's)
    * fit on TRAIN only, fixed hyperparameters (no tuning);
    * decision threshold = max F1 on VAL (the CNN's threshold is stored in its checkpoint);
    * report --split val by default; --split test requires --final-test and should be run once;
    * 95% bootstrap intervals for PR-AUC / ROC-AUC over the evaluated split.

Outputs (default outputs/baselines/<split>/)
    results.csv  results.md  predictions.csv  pr_curves.png  roc_curves.png
    feature_importance.csv  feature_importance.png

Usage
    python -m src.baselines.compare --raw-dir data/raw_small \
        --koi-table data/labels/kepler_koi_labels_small.csv \
        --checkpoint checkpoints/<run>/best.pt
    python -m src.baselines.compare ... --split test --final-test
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import make_pipeline  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from src.baselines.features import (  # noqa: E402
    DEFAULT_FEATURES_PATH,
    FEATURE_COLUMNS,
    load_or_compute_features,
)
from src.data.build_dataset import DEFAULT_DATASET_DIR, load_split  # noqa: E402
from src.evaluation.evaluate import predict_split, recall_at_precision  # noqa: E402
from src.evaluation.metrics import binary_metrics, select_threshold  # noqa: E402
from src.preprocessing.preprocess import DEFAULT_RAW_DIR, PROJECT_ROOT  # noqa: E402
from src.preprocessing.views import DEFAULT_KOI_TABLE, DEFAULT_VIEWS_DIR  # noqa: E402
from src.training.train import load_checkpoint, select_device  # noqa: E402

DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "baselines"
FEATURE_SETS = ("engineered", "views", "both")

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rule-based BLS scores (higher = more planet-like)
# ---------------------------------------------------------------------------
def _fill_low(score: np.ndarray) -> np.ndarray:
    """Replace NaN (no usable BLS fit) with a score below every finite one."""
    score = np.asarray(score, dtype=np.float64)
    finite = np.isfinite(score)
    floor = score[finite].min() - 1.0 if finite.any() else 0.0
    return np.where(finite, score, floor)


def bls_snr_score(features: pd.DataFrame) -> np.ndarray:
    return _fill_low(features["bls_snr"].to_numpy())


def bls_vetting_score(features: pd.DataFrame) -> np.ndarray:
    """-max(odd/even sigma, secondary SNR, 0); 0 = passes both eclipsing-binary tests."""
    tests = features[["bls_odd_even_sigma", "bls_secondary_snr"]].to_numpy(dtype=np.float64)
    penalty = np.nanmax(np.column_stack([np.zeros(len(tests)), tests]), axis=1)
    score = -penalty
    score[features["bls_status"].to_numpy() != "ok"] = np.nan
    return _fill_low(score)


# ---------------------------------------------------------------------------
# Learned baselines
# ---------------------------------------------------------------------------
def class_balance(y: np.ndarray) -> tuple[int, int]:
    n_pos = int((y == 1).sum())
    return n_pos, int(y.size - n_pos)


def make_random_forest(y_train: np.ndarray, seed: int):
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced", random_state=seed, n_jobs=-1,
        ),
    )


def make_xgboost(y_train: np.ndarray, seed: int) -> XGBClassifier:
    n_pos, n_neg = class_balance(y_train)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("XGBoost scale_pos_weight needs both classes in TRAIN")
    return XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=1.0, reg_lambda=1.0,
        scale_pos_weight=n_neg / n_pos, eval_metric="aucpr", random_state=seed,
        n_jobs=os.cpu_count() or 1,
    )


LEARNED: dict[str, Callable] = {"Random Forest": make_random_forest, "XGBoost": make_xgboost}


def feature_importances(model, names: list[str]) -> np.ndarray:
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    return np.asarray(est.feature_importances_, dtype=np.float64)[: len(names)]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def design_matrix(
    split_dir: Path, features: pd.DataFrame, feature_set: str
) -> tuple[np.ndarray, list[str], pd.DataFrame, np.ndarray]:
    """(X, column names, aligned feature rows, labels) for one split."""
    meta = pd.read_csv(split_dir / "meta.csv")
    rows = meta[["kepoi_name"]].merge(features, on="kepoi_name", how="left")
    if rows.bls_status.isna().any():
        missing = rows.kepoi_name[rows.bls_status.isna()].tolist()
        raise ValueError(f"No features for {len(missing)} KOIs in {split_dir}: {missing[:3]}")
    parts, names = [], []
    if feature_set in ("engineered", "both"):
        parts.append(rows[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64))
        names += list(FEATURE_COLUMNS)
    if feature_set in ("views", "both"):
        x_global, x_local, _ = load_split(split_dir)
        parts += [x_global[:, 0].numpy(), x_local[:, 0].numpy()]
        names += [f"global_{i}" for i in range(x_global.shape[-1])]
        names += [f"local_{i}" for i in range(x_local.shape[-1])]
    return np.hstack(parts), names, rows, meta["label"].to_numpy(dtype=int)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def bootstrap_ci(
    labels: np.ndarray, scores: np.ndarray, metric: Callable, n: int = 1000, seed: int = 0
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    values = []
    for _ in range(n):
        idx = rng.integers(0, labels.size, labels.size)
        y = labels[idx]
        if 0 < y.sum() < y.size:
            values.append(metric(y, scores[idx]))
    if not values:
        return np.nan, np.nan
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def result_row(
    model: str, labels: np.ndarray, scores: np.ndarray, threshold: float, n_boot: int, seed: int
) -> dict:
    m = binary_metrics(labels, scores, threshold)
    cm = m["confusion_matrix"]
    pr_lo, pr_hi = bootstrap_ci(labels, scores, average_precision_score, n_boot, seed)
    roc_lo, roc_hi = bootstrap_ci(labels, scores, roc_auc_score, n_boot, seed)
    return {
        "model": model,
        "pr_auc": m["pr_auc"], "pr_auc_lo": pr_lo, "pr_auc_hi": pr_hi,
        "roc_auc": m["roc_auc"], "roc_auc_lo": roc_lo, "roc_auc_hi": roc_hi,
        "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
        "accuracy": m["accuracy"],
        "recall_at_p90": recall_at_precision(labels, scores, 0.9),
        "tn": cm["tn"], "fp": cm["fp"], "fn": cm["fn"], "tp": cm["tp"],
        "threshold": threshold, "n": m["n"], "n_positive": m["n_positive"],
    }


def markdown_table(results: pd.DataFrame, split: str) -> str:
    head = ("| Model | PR-AUC [95% CI] | ROC-AUC [95% CI] | Precision | Recall | F1 | Accuracy "
            "| Recall@P>=0.9 | TN | FP | FN | TP |")
    lines = [f"Split: **{split}** (n={int(results.n.iloc[0])}, "
             f"planets={int(results.n_positive.iloc[0])}); thresholds chosen on VAL.", "",
             head, "|" + "---|" * 12]
    for r in results.itertuples(index=False):
        lines.append(
            f"| {r.model} | {r.pr_auc:.3f} [{r.pr_auc_lo:.3f}, {r.pr_auc_hi:.3f}] "
            f"| {r.roc_auc:.3f} [{r.roc_auc_lo:.3f}, {r.roc_auc_hi:.3f}] "
            f"| {r.precision:.3f} | {r.recall:.3f} | {r.f1:.3f} | {r.accuracy:.3f} "
            f"| {r.recall_at_p90:.3f} | {r.tn} | {r.fp} | {r.fn} | {r.tp} |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_curves(out_dir: Path, labels: np.ndarray, scores: dict[str, np.ndarray],
                results: pd.DataFrame, split: str) -> None:
    if not 0 < labels.sum() < labels.size:
        log.warning("Only one class in %s; skipping curves", split)
        return
    auc = results.set_index("model")
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, s in scores.items():
        p, r, _ = precision_recall_curve(labels, s)
        ax.plot(r, p, label=f"{name} ({auc.loc[name, 'pr_auc']:.3f})")
    ax.axhline(labels.mean(), color="k", ls="--", lw=0.8, label=f"random ({labels.mean():.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02),
           title=f"Precision-recall ({split})")
    ax.legend(fontsize=8, loc="lower left")
    fig.savefig(out_dir / "pr_curves.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(labels, s)
        ax.plot(fpr, tpr, label=f"{name} ({auc.loc[name, 'roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="random")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=f"ROC ({split})")
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(out_dir / "roc_curves.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_importance(out_dir: Path, importance: pd.DataFrame, top: int = 20) -> None:
    models = importance.columns.drop("feature").tolist()
    ranked = importance.assign(mean=importance[models].mean(axis=1)).nlargest(top, "mean")
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(ranked) + 1.2))
    y = np.arange(len(ranked))[::-1]
    height = 0.8 / len(models)
    for k, m in enumerate(models):
        ax.barh(y + (k - (len(models) - 1) / 2) * height, ranked[m], height, label=m)
    ax.set_yticks(y, ranked.feature)
    ax.set(xlabel="Importance (impurity / gain, normalized)", title="Feature importance")
    ax.legend(fontsize=8)
    fig.savefig(out_dir / "feature_importance.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def compare(args: argparse.Namespace) -> pd.DataFrame:
    splits = ("train", "val", "test") if args.split == "test" else ("train", "val")
    features = load_or_compute_features(splits, args.data_dir, args.views_dir, args.koi_table,
                                        args.raw_dir, args.features, args.workers)
    data = {s: design_matrix(Path(args.data_dir) / s, features, args.feature_set)
            for s in splits}
    names = data["train"][1]
    y_train, y_val = data["train"][3], data["val"][3]
    rows_eval, labels = data[args.split][2], data[args.split][3]
    n_pos, n_neg = class_balance(y_train)
    log.info("TRAIN %d planets / %d false positives; evaluating on %s (n=%d)",
             n_pos, n_neg, args.split, labels.size)
    n_failed = int((rows_eval.bls_status != "ok").sum())
    if n_failed:
        log.warning("%d %s KOIs have no BLS fit; rule scores rank them lowest",
                    n_failed, args.split)

    scores: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}
    for name, fn in (("BLS SNR", bls_snr_score), ("BLS vetting", bls_vetting_score)):
        thresholds[name] = select_threshold(y_val, fn(data["val"][2]))
        scores[name] = fn(rows_eval)

    importance = pd.DataFrame({"feature": names})
    for name, factory in LEARNED.items():
        model = factory(y_train, args.seed)
        model.fit(data["train"][0], y_train)
        thresholds[name] = select_threshold(y_val, model.predict_proba(data["val"][0])[:, 1])
        scores[name] = model.predict_proba(data[args.split][0])[:, 1]
        importance[name] = feature_importances(model, names)

    if args.checkpoint is not None:
        device = select_device(args.device)
        ckpt = load_checkpoint(args.checkpoint, device)
        if ckpt.get("threshold") is None:
            raise ValueError(f"{args.checkpoint} has no validation threshold; use best.pt")
        trained_on = Path(ckpt["config"].get("data_dir", "")).resolve()
        if trained_on != Path(args.data_dir).resolve():
            log.warning("CNN was trained on %s, comparing on %s", trained_on, args.data_dir)
        probs, cnn_labels = predict_split(ckpt, Path(args.data_dir) / args.split, device)
        if not np.array_equal(cnn_labels.astype(int), labels):
            raise ValueError("CNN split labels do not match the baseline split")
        scores["CNN"], thresholds["CNN"] = probs, float(ckpt["threshold"])

    results = pd.DataFrame([
        result_row(name, labels, s, thresholds[name], args.bootstrap, args.seed)
        for name, s in scores.items()
    ])

    out_dir = args.out_dir or DEFAULT_OUT_DIR / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "results.csv", index=False)
    table = markdown_table(results, args.split)
    (out_dir / "results.md").write_text(table, encoding="utf-8")
    preds = rows_eval[["kepoi_name", "kepid"]].assign(label=labels)
    for name, s in scores.items():
        preds[f"score_{name.lower().replace(' ', '_')}"] = s
    preds.to_csv(out_dir / "predictions.csv", index=False)
    importance.sort_values(list(LEARNED)[0], ascending=False).to_csv(
        out_dir / "feature_importance.csv", index=False)
    plot_importance(out_dir, importance)
    plot_curves(out_dir, labels.astype(bool).astype(int), scores, results, args.split)
    (out_dir / "config.json").write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2))

    print(table)
    log.info("Results written to %s", out_dir)
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare BLS, Random Forest and XGBoost baselines with the CNN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATASET_DIR)
    p.add_argument("--views-dir", type=Path, default=DEFAULT_VIEWS_DIR)
    p.add_argument("--koi-table", type=Path, default=DEFAULT_KOI_TABLE)
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument("--features", type=Path, default=DEFAULT_FEATURES_PATH,
                   help="Feature cache CSV; missing KOIs are computed and appended.")
    p.add_argument("--feature-set", choices=FEATURE_SETS, default="engineered",
                   help="RF/XGBoost inputs: engineered features, raw view bins, or both.")
    p.add_argument("--checkpoint", type=Path, default=None, help="CNN best.pt to include.")
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--final-test", action="store_true",
                   help="Required for --split test: confirms this is the one final evaluation.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Defaults to outputs/baselines/<split>.")
    p.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap resamples for CIs.")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    if args.split == "test" and not args.final_test:
        p.error("--split test requires --final-test; select models and thresholds on val only")
    return args


def main(argv: list[str] | None = None) -> pd.DataFrame:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return compare(parse_args(argv))


if __name__ == "__main__":
    main()
