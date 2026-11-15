"""Train BaselineCNN or AstronetStyle on the train/val/test arrays from build_dataset.

Components: DataLoader, AdamW, LR scheduler (plateau / cosine), TensorBoard, checkpointing
(last.pt every epoch, best.pt on improvement), early stopping, resume.

The model outputs raw logits; sigmoid is applied only to compute metrics. pos_weight comes
from the training split only, and the decision threshold is selected on the validation split
only (max F1) unless --threshold is given. Val/test are never resampled or modified.

Inputs:
    data/processed/dataset/{train,val,test}/X_global.npy, X_local.npy, y.npy
Outputs:
    checkpoints/<run>/best.pt, last.pt, metrics.json, config.json
    outputs/runs/<run>/            TensorBoard logs (tensorboard --logdir outputs/runs)

Usage:
    python -m src.training.train --model astronet
    python -m src.training.train --model baseline --epochs 100 --lr 1e-3 --scheduler cosine
    python -m src.training.train --resume checkpoints/<run>/last.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data.build_dataset import DEFAULT_DATASET_DIR, TransitViewsDataset
from src.evaluation.metrics import binary_metrics, format_metrics, select_threshold
from src.models.astronet_style import AstronetStyle
from src.models.baseline_cnn import BaselineCNN, count_parameters
from src.training.losses import LOSSES, WeightedBCELoss, build_loss, class_counts

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DEFAULT_LOG_DIR = PROJECT_ROOT / "outputs" / "runs"

MONITOR_MODES = {"val_loss": "min", "val_pr_auc": "max", "val_roc_auc": "max"}
# Per-epoch thresholded metrics are logged at this fixed value for trend tracking only;
# reported metrics use the validation-selected threshold.
EPOCH_THRESHOLD = 0.5

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    seed: int,
    include_test: bool = True,
) -> dict[str, DataLoader]:
    loaders = {}
    for split in ("train", "val", "test") if include_test else ("train", "val"):
        split_dir = Path(data_dir) / split
        if not (split_dir / "y.npy").exists():
            if split == "test":
                continue
            raise FileNotFoundError(f"Missing {split_dir}; run `python -m src.data.build_dataset`")
        dataset = TransitViewsDataset(split_dir)
        if len(dataset) == 0:
            if split == "test":
                continue
            raise ValueError(f"{split_dir} is empty")
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            generator=torch.Generator().manual_seed(seed) if split == "train" else None,
        )
    return loaders


def build_model(
    name: str,
    global_len: int,
    local_len: int,
    dropout: float | None,
    batch_norm: bool,
    kernel_size: int | None = None,
    base_filters: int | None = None,
) -> nn.Module:
    """base_filters f gives channels (f, 2f, 4f, ...) per block; None keeps model defaults."""
    kwargs: dict = {}
    if dropout is not None:
        kwargs["dropout"] = dropout
    if kernel_size is not None:
        kwargs["kernel_size"] = kernel_size
    if name == "baseline":
        if base_filters is not None:
            kwargs["channels"] = tuple(base_filters * 2**i for i in range(4))
        return BaselineCNN(input_length=global_len, **kwargs)
    if name == "astronet":
        if base_filters is not None:
            kwargs["global_channels"] = tuple(base_filters * 2**i for i in range(5))
            kwargs["local_channels"] = tuple(base_filters * 2**i for i in range(2))
        return AstronetStyle(global_len, local_len, batch_norm=batch_norm, **kwargs)
    raise ValueError(f"Unknown model {name!r}")


def compute_logits(model: nn.Module, x_global: torch.Tensor, x_local: torch.Tensor) -> torch.Tensor:
    if isinstance(model, BaselineCNN):
        return model(x_global)
    return model(x_global, x_local)


def build_scheduler(
    name: str, optimizer: torch.optim.Optimizer, epochs: int, mode: str, patience: int
) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=0.5, patience=patience
        )
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "none":
        return None
    raise ValueError(f"Unknown scheduler {name!r}")


# ---------------------------------------------------------------------------
# Epoch loop
# ---------------------------------------------------------------------------
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float = 0.0,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Train for one epoch if `optimizer` is given, otherwise evaluate.

    Returns (mean loss, probabilities, labels). The loss is computed on raw logits.
    """
    training = optimizer is not None
    model.train(training)
    non_blocking = device.type == "cuda"
    loss_sum, n = 0.0, 0
    probs, labels = [], []
    with torch.set_grad_enabled(training):
        for x_global, x_local, y in loader:
            x_global = x_global.to(device, non_blocking=non_blocking)
            x_local = x_local.to(device, non_blocking=non_blocking)
            y = y.to(device, non_blocking=non_blocking)
            logits = compute_logits(model, x_global, x_local)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            loss_sum += float(loss.detach()) * y.numel()
            n += y.numel()
            probs.append(torch.sigmoid(logits.detach()).cpu().numpy())
            labels.append(y.cpu().numpy())
    if n == 0:
        raise ValueError("Empty data loader")
    return loss_sum / n, np.concatenate(probs), np.concatenate(labels)


def epoch_metrics(loss: float, probs: np.ndarray, labels: np.ndarray) -> dict:
    m = binary_metrics(labels, probs, EPOCH_THRESHOLD)
    m["loss"] = loss
    return m


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience: int = 10, mode: str = "min", min_delta: float = 0.0) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best: float | None = None
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        """Record `value`; return True if it is a new best."""
        if self.best is None or (
            value < self.best - self.min_delta
            if self.mode == "min"
            else value > self.best + self.min_delta
        ):
            self.best = value
            self.bad_epochs = 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.patience > 0 and self.bad_epochs >= self.patience

    def state_dict(self) -> dict:
        return {"best": self.best, "bad_epochs": self.bad_epochs}

    def load_state_dict(self, state: dict) -> None:
        self.best = state["best"]
        self.bad_epochs = state["bad_epochs"]


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(path: Path, **state) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    return torch.load(path, map_location=device, weights_only=True)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(
    args: argparse.Namespace,
    epoch_callback: Callable[[int, dict], None] | None = None,
) -> dict:
    """`epoch_callback(epoch, val_metrics)` runs after each epoch; it may raise to abort."""
    seed_everything(args.seed)
    device = select_device(args.device)
    if device.type == "cuda":
        log.info("Using GPU: %s", torch.cuda.get_device_name(device))
        torch.backends.cudnn.benchmark = True
    else:
        log.info("Using device: %s", device)

    loaders = make_loaders(
        args.data_dir, args.batch_size, args.num_workers, device, args.seed,
        include_test=not args.skip_test,
    )
    train_ds: TransitViewsDataset = loaders["train"].dataset
    model = build_model(
        args.model,
        train_ds.x_global.shape[-1],
        train_ds.x_local.shape[-1],
        args.dropout,
        args.batch_norm,
        args.kernel_size,
        args.base_filters,
    ).to(device)
    log.info("Model %s with %s trainable parameters", args.model, f"{count_parameters(model):,}")

    criterion = build_loss(
        args.loss, train_ds.y, args.pos_weight, args.focal_gamma, args.focal_alpha
    ).to(device)
    n_pos, n_neg = class_counts(train_ds.y)
    log.info("Train classes: %d planet / %d false positive. Loss: %s", n_pos, n_neg, criterion)
    pos_weight_used = (
        float(criterion.pos_weight) if isinstance(criterion, WeightedBCELoss) else None
    )
    if pos_weight_used is not None:
        log.info("pos_weight = %.4f (training split only: %d neg / %d pos)",
                 pos_weight_used, n_neg, n_pos)

    mode = MONITOR_MODES[args.monitor]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(
        args.scheduler, optimizer, args.epochs, mode, args.scheduler_patience
    )
    stopper = EarlyStopping(args.patience, mode, args.min_delta)

    start_epoch = 1
    run_name = args.run_name or f"{args.model}_{time.strftime('%Y%m%d-%H%M%S')}"
    if args.resume:
        ckpt = load_checkpoint(args.resume, device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler is not None and ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])
        stopper.load_state_dict(ckpt["early_stopping"])
        start_epoch = ckpt["epoch"] + 1
        run_name = ckpt["config"]["run_name"]
        log.info("Resumed %s from epoch %d", run_name, ckpt["epoch"])

    run_dir = Path(args.checkpoint_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(run_name=run_name, device=str(device), parameters=count_parameters(model))
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    writer = SummaryWriter(log_dir=str(Path(args.log_dir) / run_name))
    if start_epoch == 1:
        writer.add_text("config", f"```\n{json.dumps(config, indent=2)}\n```")

    best_path = run_dir / "best.pt"
    epoch = start_epoch - 1
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.perf_counter()
            train_m = epoch_metrics(*run_epoch(
                model, loaders["train"], criterion, device, optimizer, args.grad_clip
            ))
            val_m = epoch_metrics(*run_epoch(model, loaders["val"], criterion, device))
            lr = optimizer.param_groups[0]["lr"]

            for name in ("loss", "pr_auc", "roc_auc"):
                writer.add_scalar(f"{name}/train", train_m[name], epoch)
                writer.add_scalar(f"{name}/val", val_m[name], epoch)
            for name in ("accuracy", "precision", "recall", "f1"):
                writer.add_scalar(f"at_0.5/{name}/train", train_m[name], epoch)
                writer.add_scalar(f"at_0.5/{name}/val", val_m[name], epoch)
            writer.add_scalar("lr", lr, epoch)

            monitored = val_m[args.monitor.removeprefix("val_")]
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(monitored)
            elif scheduler is not None:
                scheduler.step()

            improved = stopper.step(monitored)
            state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "early_stopping": stopper.state_dict(),
                "config": config,
                "val_metrics": val_m,
            }
            if improved:
                save_checkpoint(best_path, **state)
            save_checkpoint(run_dir / "last.pt", **state)

            log.info(
                "epoch %3d | train loss %.4f PR-AUC %.4f | val loss %.4f PR-AUC %.4f"
                " ROC-AUC %.4f | lr %.2e | %.1fs%s",
                epoch, train_m["loss"], train_m["pr_auc"],
                val_m["loss"], val_m["pr_auc"], val_m["roc_auc"],
                lr, time.perf_counter() - t0, " *" if improved else "",
            )
            if epoch_callback is not None:
                epoch_callback(epoch, val_m)
            if stopper.should_stop:
                log.info("Early stopping: no %s improvement for %d epochs",
                         args.monitor, args.patience)
                break
    except BaseException:
        writer.close()
        raise

    best = load_checkpoint(best_path, device)
    model.load_state_dict(best["model"])
    outputs = {split: run_epoch(model, loader, criterion, device)
               for split, loader in loaders.items()}

    if args.threshold is None:
        _, val_probs, val_labels = outputs["val"]
        threshold = select_threshold(val_labels, val_probs)
        threshold_source = "validation split, max F1"
    else:
        threshold = args.threshold
        threshold_source = "fixed via --threshold"
    log.info("Decision threshold %.4f (%s)", threshold, threshold_source)

    results = {
        "best_epoch": best["epoch"],
        "last_epoch": epoch,
        "monitor": args.monitor,
        "loss_fn": args.loss,
        "pos_weight": pos_weight_used,
        "train_class_counts": {"positive": n_pos, "negative": n_neg},
        "threshold": threshold,
        "threshold_source": threshold_source,
    }
    for split, (loss, probs, labels) in outputs.items():
        m = binary_metrics(labels, probs, threshold)
        m["loss"] = loss
        results[split] = m
        log.info(format_metrics(split, m))
        for name in ("loss", "pr_auc", "roc_auc", "precision", "recall", "f1", "accuracy"):
            writer.add_scalar(f"best/{split}_{name}", m[name], best["epoch"])
    writer.close()

    best["threshold"] = threshold
    save_checkpoint(best_path, **best)
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    log.info("Best epoch %d; best model saved to %s", best["epoch"], best_path)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train an exoplanet transit classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", choices=("baseline", "astronet"), default="astronet")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATASET_DIR)
    p.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    p.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    p.add_argument("--run-name", default=None, help="Defaults to <model>_<timestamp>.")
    p.add_argument("--resume", type=Path, default=None, help="Path to a last.pt to continue from.")

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=0.0, help="Max grad norm; 0 disables.")
    p.add_argument("--dropout", type=float, default=None, help="Defaults to the model's own.")
    p.add_argument("--kernel-size", type=int, default=None,
                   help="Odd conv kernel size. Defaults to the model's own (5).")
    p.add_argument("--base-filters", type=int, default=None,
                   help="Filters in the first conv block, doubled per block. Default 16.")
    p.add_argument("--batch-norm", action="store_true", help="Astronet only.")
    p.add_argument("--loss", choices=LOSSES, default="weighted_bce")
    p.add_argument("--pos-weight", default="auto",
                   help="weighted_bce positive weight: 'auto' (n_neg / n_pos) or a number.")
    p.add_argument("--focal-gamma", type=float, default=2.0, help="Focal focusing parameter.")
    p.add_argument("--focal-alpha", default="auto",
                   help="Focal positive-class weight: 'auto' (n_neg / n), 'none', or in (0, 1).")
    p.add_argument("--threshold", type=float, default=None,
                   help="Fixed decision threshold. Default: select on the validation split.")

    p.add_argument("--scheduler", choices=("plateau", "cosine", "none"), default="plateau")
    p.add_argument("--scheduler-patience", type=int, default=3)
    p.add_argument("--monitor", choices=tuple(MONITOR_MODES), default="val_pr_auc",
                   help="Metric for best model, early stopping and plateau scheduler.")
    p.add_argument("--patience", type=int, default=10, help="Early stopping; 0 disables.")
    p.add_argument("--min-delta", type=float, default=1e-4)

    p.add_argument("--device", default=None, help="e.g. cuda, cuda:1, cpu. Auto by default.")
    p.add_argument("--skip-test", action="store_true",
                   help="Do not load or evaluate the test split (used during tuning).")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1:
        p.error("--epochs and --batch-size must be >= 1")
    if args.kernel_size is not None and (args.kernel_size < 1 or args.kernel_size % 2 == 0):
        p.error("--kernel-size must be a positive odd integer")
    if args.base_filters is not None and args.base_filters < 1:
        p.error("--base-filters must be >= 1")
    for name, allowed in (("pos_weight", ("auto",)), ("focal_alpha", ("auto", "none"))):
        value = getattr(args, name)
        if value in allowed:
            setattr(args, name, None if value == "none" else value)
            continue
        try:
            setattr(args, name, float(value))
        except ValueError:
            p.error(f"--{name.replace('_', '-')} must be one of {allowed} or a number")
    return args


def main(argv: list[str] | None = None) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return train(parse_args(argv))


if __name__ == "__main__":
    main()
