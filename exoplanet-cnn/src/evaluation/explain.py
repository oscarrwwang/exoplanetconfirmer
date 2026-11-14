"""Explain CNN predictions: 1D Grad-CAM, saliency maps and occlusion feature importance.

Every attribution explains the *planet* logit, so it answers "which parts of the light curve
pushed the model towards calling this signal a planet?".

    Grad-CAM    Gradient-weighted activations of the last conv layer of each branch,
                upsampled to the input length. Coarse, class-discriminative, in [0, 1].
    Saliency    SmoothGrad |d logit / d flux|: bins where a small flux change would move the
                logit most. Fine-grained, in [0, 1]. --smooth-samples 0 gives vanilla gradients.
    Occlusion   Drop in planet probability when a sliding window is replaced by the
                out-of-transit baseline (0). Signed: > 0 supports "planet", < 0 argues against.
                Model-agnostic and in probability units, so it is the feature-importance measure.

Feature importance is also summarised physically, using each KOI's period and duration from
the views directory:
    view importance     probability drop when an entire view (global / local) is blanked
    region importance   share of attribution in the transit window, the secondary-eclipse
                        window (phase 0.5) and the remaining out-of-transit baseline, plus
                        enrichment = share / fraction of bins (> 1 means the model focuses there)

As in `src.evaluation.evaluate`, the decision threshold comes from the checkpoint (selected on
validation) and the test split requires --final-test.

Outputs (default outputs/explain/<run_name>/<split>/):
    metrics.json                 evaluation metrics for the split
    examples/<cat>_<koi>.png     per-example Grad-CAM / saliency / occlusion (TP, FN, FP, TN)
    class_mean_attribution.png   mean flux and attribution per class
    region_importance.png/.csv   attribution share and enrichment per region (per KOI)
    region_summary.csv           mean region enrichment by class
    view_importance.csv          per-KOI probability drop when blanking each view
    attributions.npz             all attribution arrays

Usage:
    python -m src.evaluation.explain --checkpoint checkpoints/<run>/best.pt
    python -m src.evaluation.explain --checkpoint checkpoints/<run>/best.pt --examples 5
    python -m src.evaluation.explain --checkpoint checkpoints/<run>/best.pt --split test --final-test
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from src.data.build_dataset import DEFAULT_DATASET_DIR, TransitViewsDataset  # noqa: E402
from src.evaluation.evaluate import evaluate_predictions, load_model  # noqa: E402
from src.evaluation.metrics import format_metrics  # noqa: E402
from src.models.baseline_cnn import BaselineCNN  # noqa: E402
from src.preprocessing.views import DEFAULT_VIEWS_DIR, ViewConfig, safe_name  # noqa: E402
from src.training.train import (  # noqa: E402
    PROJECT_ROOT,
    compute_logits,
    load_checkpoint,
    select_device,
)

DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "explain"
METHODS = ("gradcam", "saliency", "occlusion")
METHOD_TITLES = {
    "gradcam": "Grad-CAM",
    "saliency": "Saliency (SmoothGrad)",
    "occlusion": "Occlusion (prob. drop)",
}
REGIONS = ("transit", "secondary", "out_of_transit")
CATEGORIES = ("TP", "FN", "FP", "TN")
# (window, stride) in bins; windows roughly match a typical transit width in each view.
DEFAULT_OCCLUSION = {"global": (41, 10), "local": (11, 2)}

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Attribution methods
# ---------------------------------------------------------------------------
def model_views(model: nn.Module) -> tuple[str, ...]:
    return ("global",) if isinstance(model, BaselineCNN) else ("global", "local")


def _last_relu(module: nn.Module) -> nn.Module:
    relus = [m for m in module.modules() if isinstance(m, nn.ReLU)]
    if not relus:
        raise ValueError(f"{type(module).__name__} has no ReLU layer for Grad-CAM")
    return relus[-1]


def gradcam_layers(model: nn.Module) -> dict[str, nn.Module]:
    """Last conv activation (post-ReLU) of each convolutional branch."""
    if isinstance(model, BaselineCNN):
        return {"global": _last_relu(model.features)}
    return {"global": _last_relu(model.global_branch), "local": _last_relu(model.local_branch)}


def _scale_to_unit(x: torch.Tensor) -> torch.Tensor:
    """Divide each row by its maximum; all-zero rows stay zero."""
    peak = x.amax(dim=-1, keepdim=True)
    return torch.where(peak > 0, x / peak.clamp_min(1e-12), torch.zeros_like(x))


def grad_cam(
    model: nn.Module, x_global: torch.Tensor, x_local: torch.Tensor
) -> dict[str, torch.Tensor]:
    """1D Grad-CAM of the planet logit. Returns {view: (N, L)} in [0, 1]."""
    layers = gradcam_layers(model)
    names = list(layers)
    acts: dict[str, torch.Tensor] = {}
    handles = [
        layer.register_forward_hook(lambda _m, _i, out, name=name: acts.__setitem__(name, out))
        for name, layer in layers.items()
    ]
    try:
        with torch.enable_grad():
            logits = compute_logits(
                model, x_global.detach().requires_grad_(True), x_local.detach().requires_grad_(True)
            )
            # autograd.grad leaves parameter .grad untouched.
            grads = torch.autograd.grad(logits.sum(), [acts[n] for n in names])
    finally:
        for h in handles:
            h.remove()

    lengths = {"global": x_global.shape[-1], "local": x_local.shape[-1]}
    out = {}
    for name, grad in zip(names, grads):
        weights = grad.mean(dim=-1, keepdim=True)
        cam = F.relu((weights * acts[name].detach()).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=lengths[name], mode="linear", align_corners=False)
        out[name] = _scale_to_unit(cam[:, 0])
    return out


def _input_gradients(
    model: nn.Module, x_global: torch.Tensor, x_local: torch.Tensor, views: tuple[str, ...]
) -> dict[str, torch.Tensor]:
    inputs = {
        "global": x_global.detach().clone().requires_grad_(True),
        "local": x_local.detach().clone().requires_grad_(True),
    }
    with torch.enable_grad():
        logits = compute_logits(model, inputs["global"], inputs["local"])
        grads = torch.autograd.grad(logits.sum(), [inputs[v] for v in views])
    return {v: g[:, 0].detach() for v, g in zip(views, grads)}


def saliency(
    model: nn.Module,
    x_global: torch.Tensor,
    x_local: torch.Tensor,
    smooth_samples: int = 25,
    noise_level: float = 0.1,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """SmoothGrad saliency |d logit / d flux| in [0, 1]; smooth_samples=0 gives vanilla gradients.

    Noise std is noise_level * (max - min) of each example's view.
    """
    views = model_views(model)
    if smooth_samples <= 0:
        grads = _input_gradients(model, x_global, x_local, views)
        return {v: _scale_to_unit(g.abs()) for v, g in grads.items()}

    xs = {"global": x_global, "local": x_local}
    sigma = {v: noise_level * (x.amax(-1, keepdim=True) - x.amin(-1, keepdim=True))
             for v, x in xs.items()}
    gen = torch.Generator().manual_seed(seed)
    total = {v: torch.zeros(xs[v].shape[0], xs[v].shape[-1], device=xs[v].device) for v in views}
    for _ in range(smooth_samples):
        noisy = {v: x + sigma[v] * torch.randn(x.shape, generator=gen).to(x.device)
                 for v, x in xs.items()}
        for v, g in _input_gradients(model, noisy["global"], noisy["local"], views).items():
            total[v] += g.abs()
    return {v: _scale_to_unit(t / smooth_samples) for v, t in total.items()}


@torch.no_grad()
def occlusion(
    model: nn.Module,
    x_global: torch.Tensor,
    x_local: torch.Tensor,
    windows: dict[str, tuple[int, int]] | None = None,
    fill: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Planet-probability drop per bin, averaged over all occluding windows that cover it.

    fill=0 is the normalized out-of-transit baseline, so occluding a transit "erases" it.
    """
    windows = {**DEFAULT_OCCLUSION, **(windows or {})}
    xs = {"global": x_global, "local": x_local}
    base = torch.sigmoid(compute_logits(model, x_global, x_local))
    out = {}
    for v in model_views(model):
        length = xs[v].shape[-1]
        width = min(windows[v][0], length)
        stride = max(1, windows[v][1])
        starts = list(range(0, length - width + 1, stride))
        if starts[-1] != length - width:
            starts.append(length - width)
        importance = torch.zeros(xs[v].shape[0], length, device=xs[v].device)
        counts = torch.zeros(length, device=xs[v].device)
        for s in starts:
            occluded = dict(xs)
            occluded[v] = xs[v].clone()
            occluded[v][..., s : s + width] = fill
            prob = torch.sigmoid(compute_logits(model, occluded["global"], occluded["local"]))
            importance[:, s : s + width] += (base - prob)[:, None]
            counts[s : s + width] += 1
        out[v] = importance / counts
    return out


@torch.no_grad()
def view_importance(
    model: nn.Module, x_global: torch.Tensor, x_local: torch.Tensor, fill: float = 0.0
) -> dict[str, torch.Tensor]:
    """Planet-probability drop when an entire view is replaced by the baseline."""
    base = torch.sigmoid(compute_logits(model, x_global, x_local))
    out = {}
    for v in model_views(model):
        xg = torch.full_like(x_global, fill) if v == "global" else x_global
        xl = torch.full_like(x_local, fill) if v == "local" else x_local
        out[v] = base - torch.sigmoid(compute_logits(model, xg, xl))
    return out


# ---------------------------------------------------------------------------
# Physical regions of the views
# ---------------------------------------------------------------------------
def load_view_config(views_dir: Path) -> ViewConfig:
    path = Path(views_dir) / "views_config.json"
    return ViewConfig(**json.loads(path.read_text())) if path.exists() else ViewConfig()


def view_offsets(
    view: str, n_bins: int, period: float, duration: float, config: ViewConfig
) -> np.ndarray:
    """Bin centres as time from mid-transit (days), matching `src.preprocessing.views`."""
    if view == "global":
        half = 0.5 * period
        factor = config.global_bin_width_factor if config.global_bins == n_bins else 1.0 / n_bins
        width = period * factor
    else:
        half = min(0.5 * period, config.local_num_durations * duration)
        width = duration * config.local_bin_width_factor
    spacing = (2 * half - width) / (n_bins - 1)
    return -half + spacing * np.arange(n_bins) + 0.5 * width


def region_masks(
    view: str, n_bins: int, period: float, duration: float, config: ViewConfig
) -> dict[str, np.ndarray]:
    """Boolean masks for the transit, secondary-eclipse (phase 0.5) and other bins."""
    t = np.abs(view_offsets(view, n_bins, period, duration, config))
    transit = t < 0.5 * duration
    secondary = (t > 0.5 * period - 0.5 * duration) & ~transit
    return {"transit": transit, "secondary": secondary, "out_of_transit": ~(transit | secondary)}


def region_shares(attribution: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, float]:
    """Share of non-negative attribution per region and enrichment = share / area fraction."""
    a = np.clip(attribution, 0, None)
    total = a.sum()
    out = {}
    for name, mask in masks.items():
        share = float(a[mask].sum() / total) if total > 0 else np.nan
        area = float(mask.mean())
        out[f"{name}_share"] = share
        out[f"{name}_enrichment"] = share / area if area > 0 else np.nan
    return out


def load_ephemerides(views_dir: Path, names: list[str]) -> pd.DataFrame:
    """Period and duration (days) per KOI from the view files; NaN if a file is missing."""
    rows = []
    for name in names:
        path = Path(views_dir) / f"{safe_name(name)}.npz"
        period = duration = np.nan
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                period, duration = float(data["period"]), float(data["duration"])
        rows.append({"kepoi_name": name, "period": period, "duration": duration})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Example selection
# ---------------------------------------------------------------------------
def categorize(labels: np.ndarray, probs: np.ndarray, threshold: float) -> np.ndarray:
    pred = probs >= threshold
    lab = np.asarray(labels).astype(bool)
    return np.select([pred & lab, ~pred & lab, pred & ~lab], ["TP", "FN", "FP"], "TN")


def select_examples(
    categories: np.ndarray, probs: np.ndarray, k: int
) -> dict[str, np.ndarray]:
    """Most confident k per category: highest prob for TP/FP, lowest for FN/TN."""
    out = {}
    for cat in CATEGORIES:
        idx = np.flatnonzero(categories == cat)
        order = np.argsort(-probs[idx] if cat in ("TP", "FP") else probs[idx], kind="stable")
        out[cat] = idx[order[:k]]
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_axis(
    view: str, n_bins: int, period: float, duration: float, config: ViewConfig
) -> tuple[np.ndarray, str, float | None]:
    """x values, label and transit half-width (in x units) for one view."""
    if np.isfinite(period) and np.isfinite(duration):
        t = view_offsets(view, n_bins, period, duration, config)
        if view == "global":
            return t / period, "Orbital phase", 0.5 * duration / period
        return t / duration, "Time from mid-transit [transit durations]", 0.5
    return np.arange(n_bins), "Bin", None


def plot_example(
    path: Path,
    flux: dict[str, np.ndarray],
    attributions: dict[str, dict[str, np.ndarray]],
    axes_info: dict[str, tuple[np.ndarray, str, float | None]],
    title: str,
) -> None:
    views = list(flux)
    fig, axes = plt.subplots(len(METHODS), len(views), figsize=(6.5 * len(views), 7.5),
                             squeeze=False, sharex="col")
    for j, view in enumerate(views):
        x, xlabel, half = axes_info[view]
        y = flux[view]
        lo, hi = float(y.min()), float(y.max())
        pad = 0.1 * (hi - lo or 1.0)
        for i, method in enumerate(METHODS):
            ax = axes[i, j]
            a = attributions[method][view]
            if method == "occlusion":
                lim = max(float(np.abs(a).max()), 1e-6)
                cmap, vmin, vmax = "RdBu_r", -lim, lim
            else:
                cmap, vmin, vmax = "YlOrRd", 0.0, 1.0
            im = ax.imshow(a[None, :], aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                           extent=(x[0], x[-1], lo - pad, hi + pad), interpolation="nearest")
            ax.plot(x, y, color="k", lw=0.7)
            if half is not None:
                for edge in (-half, half):
                    ax.axvline(edge, color="tab:green", ls="--", lw=0.8)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_title(f"{view} view - {METHOD_TITLES[method]}", fontsize=9)
            ax.set_ylabel("Norm. flux", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        axes[-1, j].set_xlabel(xlabel)
    fig.suptitle(title + "\nred = pushes towards 'planet'; green dashes = transit window",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_class_means(
    path: Path,
    flux: dict[str, np.ndarray],
    attributions: dict[str, dict[str, np.ndarray]],
    labels: np.ndarray,
    config: ViewConfig,
) -> None:
    views = list(attributions["gradcam"])
    rows = ("flux", *METHODS)
    fig, axes = plt.subplots(len(rows), len(views), figsize=(6.5 * len(views), 9),
                             squeeze=False, sharex="col")
    lab = np.asarray(labels).astype(bool)
    for j, view in enumerate(views):
        n = flux[view].shape[-1]
        if view == "global":
            x, xlabel = np.linspace(-0.5, 0.5, n), "Orbital phase"
        else:
            k = config.local_num_durations
            x, xlabel = np.linspace(-k, k, n), "Time from mid-transit [~transit durations]"
        for i, row in enumerate(rows):
            ax = axes[i, j]
            data = flux[view] if row == "flux" else attributions[row][view]
            for mask, name, color in ((lab, "planet", "tab:blue"),
                                      (~lab, "false positive", "tab:orange")):
                if mask.any():
                    ax.plot(x, data[mask].mean(0), color=color, lw=1,
                            label=f"{name} (n={int(mask.sum())})")
            if row == "occlusion":
                ax.axhline(0, color="k", lw=0.5)
            title = "mean normalized flux" if row == "flux" else f"mean {METHOD_TITLES[row]}"
            ax.set_title(f"{view} view - {title}", fontsize=9)
        axes[0, j].legend(fontsize=8)
        axes[-1, j].set_xlabel(xlabel)
    fig.suptitle("Class-mean attribution (where the model looks, per class)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_region_summary(path: Path, summary: pd.DataFrame) -> None:
    panels = summary[["method", "view"]].drop_duplicates().to_numpy().tolist()
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.6),
                             squeeze=False, sharey=True)
    width = 0.38
    for ax, (method, view) in zip(axes[0], panels):
        sub = summary[(summary.method == method) & (summary.view == view)]
        for k, (cls, color) in enumerate((("planet", "tab:blue"),
                                          ("false positive", "tab:orange"))):
            row = sub[sub["class"] == cls]
            if row.empty:
                continue
            vals = [float(row[f"{r}_enrichment"].iloc[0]) for r in REGIONS]
            ax.bar(np.arange(len(REGIONS)) + (k - 0.5) * width, vals, width,
                   color=color, label=cls)
        ax.axhline(1, color="k", ls="--", lw=0.8)
        ax.set_xticks(range(len(REGIONS)), ["transit", "secondary", "out-of-\ntransit"],
                      fontsize=8)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_title(f"{METHOD_TITLES[method]}\n{view} view", fontsize=9)
    axes[0, 0].set_ylabel("Attribution enrichment\n(share / area; 1 = uniform)")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def explain_dataset(
    model: nn.Module,
    ds: TransitViewsDataset,
    device: torch.device,
    batch_size: int,
    smooth_samples: int,
    noise_level: float,
    windows: dict[str, tuple[int, int]],
    seed: int,
) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    """Returns (probs, {method: {view: (N, L)}}, {view: probability drop when blanked})."""
    probs, attrs, blank = [], {m: {} for m in METHODS}, {}
    for start in range(0, len(ds), batch_size):
        xg = ds.x_global[start : start + batch_size].to(device)
        xl = ds.x_local[start : start + batch_size].to(device)
        with torch.no_grad():
            probs.append(torch.sigmoid(compute_logits(model, xg, xl)).cpu())
        batch = {
            "gradcam": grad_cam(model, xg, xl),
            "saliency": saliency(model, xg, xl, smooth_samples, noise_level, seed + start),
            "occlusion": occlusion(model, xg, xl, windows),
        }
        for method, per_view in batch.items():
            for view, a in per_view.items():
                attrs[method].setdefault(view, []).append(a.cpu())
        for view, drop in view_importance(model, xg, xl).items():
            blank.setdefault(view, []).append(drop.cpu())
    cat = lambda xs: torch.cat(xs).numpy()  # noqa: E731
    return (
        cat(probs),
        {m: {v: cat(a) for v, a in per_view.items()} for m, per_view in attrs.items()},
        {v: cat(d) for v, d in blank.items()},
    )


def region_table(
    meta: pd.DataFrame,
    ephem: pd.DataFrame,
    attributions: dict[str, dict[str, np.ndarray]],
    config: ViewConfig,
) -> pd.DataFrame:
    rows = []
    for i, r in enumerate(meta.itertuples(index=False)):
        period, duration = ephem.period.iloc[i], ephem.duration.iloc[i]
        if not (np.isfinite(period) and np.isfinite(duration)):
            continue
        for method, per_view in attributions.items():
            for view, a in per_view.items():
                masks = region_masks(view, a.shape[-1], period, duration, config)
                rows.append({"kepoi_name": r.kepoi_name, "label": int(r.label),
                             "method": method, "view": view, **region_shares(a[i], masks)})
    return pd.DataFrame(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Grad-CAM, saliency and occlusion explanations for a trained checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=Path, required=True, help="best.pt from train.py.")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATASET_DIR)
    p.add_argument("--views-dir", type=Path, default=DEFAULT_VIEWS_DIR,
                   help="View .npz files, used for each KOI's period and duration.")
    p.add_argument("--split", choices=("train", "val", "test"), default="val")
    p.add_argument("--final-test", action="store_true",
                   help="Required for --split test: confirms this is the one final evaluation.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Defaults to outputs/explain/<run_name>/<split>.")
    p.add_argument("--examples", type=int, default=3, help="Plotted examples per TP/FN/FP/TN.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--smooth-samples", type=int, default=25,
                   help="SmoothGrad noise samples; 0 = vanilla gradient saliency.")
    p.add_argument("--noise-level", type=float, default=0.1,
                   help="SmoothGrad noise std as a fraction of each view's flux range.")
    p.add_argument("--occlusion-global", type=int, nargs=2, metavar=("WINDOW", "STRIDE"),
                   default=list(DEFAULT_OCCLUSION["global"]))
    p.add_argument("--occlusion-local", type=int, nargs=2, metavar=("WINDOW", "STRIDE"),
                   default=list(DEFAULT_OCCLUSION["local"]))
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.split == "test" and not args.final_test:
        p.error("--split test requires --final-test; select models and thresholds on val only")
    if min(*args.occlusion_global, *args.occlusion_local) < 1:
        p.error("occlusion window and stride must be >= 1")
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
    ds = TransitViewsDataset(split_dir)
    meta = pd.read_csv(split_dir / "meta.csv")
    model = load_model(ckpt, ds.x_global.shape[-1], ds.x_local.shape[-1], device)
    labels = ds.y.numpy()

    windows = {"global": tuple(args.occlusion_global), "local": tuple(args.occlusion_local)}
    probs, attrs, blank = explain_dataset(
        model, ds, device, args.batch_size, args.smooth_samples, args.noise_level, windows,
        args.seed,
    )
    metrics = evaluate_predictions(labels, probs, threshold)
    metrics.update(split=args.split, checkpoint=str(args.checkpoint),
                   threshold_source="validation split (stored in checkpoint)")
    log.info(format_metrics(args.split, metrics))
    if probs.std() < 1e-3:
        log.warning("P(planet) is almost constant (std %.1e): the model has not learned to "
                    "discriminate, so attributions are near zero and not interpretable",
                    probs.std())
    for view, cam in attrs["gradcam"].items():
        n_zero = int((cam.max(axis=1) == 0).sum())
        if n_zero:
            log.warning("Grad-CAM is all zero for %d/%d examples in the %s view (no positive "
                        "evidence for 'planet' in the last conv layer)", n_zero, len(cam), view)

    run_name = ckpt["config"].get("run_name", args.checkpoint.parent.name)
    out_dir = args.out_dir or DEFAULT_OUT_DIR / run_name / args.split
    (out_dir / "examples").mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    config = load_view_config(args.views_dir)
    ephem = load_ephemerides(args.views_dir, meta.kepoi_name.tolist())
    n_missing = int(ephem.period.isna().sum())
    if n_missing:
        log.warning("%d of %d KOIs have no view file in %s; region importance skips them",
                    n_missing, len(meta), args.views_dir)

    categories = categorize(labels, probs, threshold)
    views = model_views(model)
    flux = {"global": ds.x_global[:, 0].numpy(), "local": ds.x_local[:, 0].numpy()}

    blank_df = meta.assign(prob_planet=probs, category=categories)
    for v in ("global", "local"):
        blank_df[f"prob_drop_blank_{v}"] = blank.get(v, np.full(len(meta), np.nan))
    blank_df.to_csv(out_dir / "view_importance.csv", index=False)

    for cat, idx in select_examples(categories, probs, args.examples).items():
        for i in idx:
            name = meta.kepoi_name.iloc[i]
            period, duration = ephem.period.iloc[i], ephem.duration.iloc[i]
            axes_info = {v: plot_axis(v, flux[v].shape[-1], period, duration, config)
                         for v in views}
            title = (f"{name} (KIC {meta.kepid.iloc[i]}) - {cat}: label "
                     f"{'planet' if labels[i] else 'false positive'}, "
                     f"P(planet) = {probs[i]:.3f} (threshold {threshold:.3f})")
            plot_example(
                out_dir / "examples" / f"{cat}_{safe_name(name)}.png",
                {v: flux[v][i] for v in views},
                {m: {v: attrs[m][v][i] for v in views} for m in METHODS},
                axes_info,
                title,
            )

    plot_class_means(out_dir / "class_mean_attribution.png",
                     {v: flux[v] for v in views}, attrs, labels, config)

    regions = region_table(meta, ephem, attrs, config)
    summary = pd.DataFrame()
    if not regions.empty:
        regions.to_csv(out_dir / "region_importance.csv", index=False)
        summary = (
            regions.assign(**{"class": np.where(regions.label == 1, "planet", "false positive")})
            .groupby(["method", "view", "class"], sort=False)
            .mean(numeric_only=True)
            .drop(columns="label")
            .reset_index()
        )
        summary.to_csv(out_dir / "region_summary.csv", index=False)
        plot_region_summary(out_dir / "region_importance.png", summary)
        for _, r in summary.iterrows():
            log.info("%-9s %-6s %-14s transit share %.2f (x%.1f) | secondary %.2f (x%.1f)",
                     r["method"], r["view"], r["class"], r["transit_share"],
                     r["transit_enrichment"], r["secondary_share"], r["secondary_enrichment"])

    np.savez_compressed(
        out_dir / "attributions.npz",
        kepoi_name=meta.kepoi_name.to_numpy(dtype=str),
        label=labels,
        prob_planet=probs,
        **{f"{m}_{v}": a for m, per_view in attrs.items() for v, a in per_view.items()},
    )
    for v, drop in blank.items():
        log.info("Blanking the %s view lowers P(planet) by %.3f on average (planets %.3f, "
                 "false positives %.3f)", v, drop.mean(),
                 drop[labels == 1].mean() if (labels == 1).any() else np.nan,
                 drop[labels == 0].mean() if (labels == 0).any() else np.nan)
    log.info("Explanations written to %s", out_dir)
    return {"metrics": metrics, "region_summary": summary, "out_dir": out_dir}


if __name__ == "__main__":
    main()
