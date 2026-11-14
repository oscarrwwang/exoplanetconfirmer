"""Optuna hyperparameter search for the transit classifiers.

Tuned: learning rate, dropout, kernel size, batch size, number of filters (first block;
doubled per block). Objective: validation PR-AUC of each trial's best checkpoint.

The test split is never loaded during the search: trials run with --skip-test, so model
selection uses training and validation data only. Pass --evaluate-test to score the single
selected model on the test split once, after the search has finished.

Any unrecognised option is forwarded to src.training.train (e.g. --model, --epochs, --loss).
Tuned options passed this way are overridden by the trial's suggestions.

Usage:
    python -m src.training.hyperparameter_search --n-trials 50 --model astronet --epochs 30
    python -m src.training.hyperparameter_search --study-name astro1 --n-trials 20  # resumes
    python -m src.training.hyperparameter_search --n-trials 50 --evaluate-test
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
from pathlib import Path

import optuna

from src.data.build_dataset import TransitViewsDataset
from src.evaluation.metrics import binary_metrics, format_metrics
from src.training.losses import build_loss
from src.training.train import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_LOG_DIR,
    PROJECT_ROOT,
    build_model,
    load_checkpoint,
    make_loaders,
    parse_args as parse_train_args,
    run_epoch,
    select_device,
    train,
)

DEFAULT_OPTUNA_DIR = PROJECT_ROOT / "outputs" / "optuna"

SEARCH_SPACE = {
    "lr": (1e-5, 1e-2),
    "dropout": (0.0, 0.5),
    "kernel_size": [3, 5, 7, 9],
    "batch_size": [16, 32, 64, 128],
    "base_filters": [8, 16, 32],
}

log = logging.getLogger(__name__)


def suggest_params(trial: optuna.Trial) -> dict:
    return {
        "lr": trial.suggest_float("lr", *SEARCH_SPACE["lr"], log=True),
        "dropout": trial.suggest_float("dropout", *SEARCH_SPACE["dropout"]),
        "kernel_size": trial.suggest_categorical("kernel_size", SEARCH_SPACE["kernel_size"]),
        "batch_size": trial.suggest_categorical("batch_size", SEARCH_SPACE["batch_size"]),
        "base_filters": trial.suggest_categorical("base_filters", SEARCH_SPACE["base_filters"]),
    }


def trial_train_args(
    train_argv: list[str], params: dict, run_name: str, checkpoint_dir: Path, log_dir: Path
) -> argparse.Namespace:
    # Later flags win in argparse, so trial values override anything in train_argv.
    return parse_train_args([
        *train_argv,
        "--lr", str(params["lr"]),
        "--dropout", str(params["dropout"]),
        "--kernel-size", str(params["kernel_size"]),
        "--batch-size", str(params["batch_size"]),
        "--base-filters", str(params["base_filters"]),
        "--monitor", "val_pr_auc",
        "--skip-test",
        "--run-name", run_name,
        "--checkpoint-dir", str(checkpoint_dir),
        "--log-dir", str(log_dir),
    ])


def make_objective(train_argv: list[str], checkpoint_dir: Path, log_dir: Path):
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        run_name = f"trial_{trial.number:04d}"
        trial.set_user_attr("run_dir", str(checkpoint_dir / run_name))
        args = trial_train_args(train_argv, params, run_name, checkpoint_dir, log_dir)

        def report(epoch: int, val_metrics: dict) -> None:
            value = val_metrics["pr_auc"]
            if math.isfinite(value):
                trial.report(value, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned(f"pruned at epoch {epoch} (val PR-AUC {value:.4f})")

        results = train(args, epoch_callback=report)
        val = results["val"]
        trial.set_user_attr("best_epoch", results["best_epoch"])
        trial.set_user_attr("threshold", results["threshold"])
        for name in ("roc_auc", "precision", "recall", "f1", "loss"):
            trial.set_user_attr(f"val_{name}", val[name])
        if not math.isfinite(val["pr_auc"]):
            raise ValueError("Validation PR-AUC is undefined (validation split has one class)")
        return val["pr_auc"]

    return objective


def make_cleanup(checkpoint_dir: Path):
    """Delete checkpoints of every trial except the current best to bound disk usage."""
    root = checkpoint_dir.resolve()

    def cleanup(study: optuna.Study, _: optuna.trial.FrozenTrial) -> None:
        try:
            best_number = study.best_trial.number
        except ValueError:
            best_number = None
        for t in study.trials:
            run_dir = t.user_attrs.get("run_dir")
            if t.number == best_number or not run_dir or t.state == optuna.trial.TrialState.RUNNING:
                continue
            path = Path(run_dir).resolve()
            if path.parent == root and path.is_dir():
                shutil.rmtree(path)

    return cleanup


def evaluate_on_test(checkpoint_path: Path, data_dir: Path, device_name: str | None) -> dict:
    """Score a saved model on the test split using the threshold chosen on validation."""
    device = select_device(device_name)
    ckpt = load_checkpoint(checkpoint_path, device)
    cfg = ckpt["config"]
    loaders = make_loaders(Path(data_dir), cfg["batch_size"], 0, device, cfg["seed"])
    if "test" not in loaders:
        raise FileNotFoundError(f"No test split under {data_dir}")
    train_ds: TransitViewsDataset = loaders["train"].dataset
    model = build_model(
        cfg["model"], train_ds.x_global.shape[-1], train_ds.x_local.shape[-1],
        cfg["dropout"], cfg["batch_norm"], cfg["kernel_size"], cfg["base_filters"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    criterion = build_loss(
        cfg["loss"], train_ds.y, cfg["pos_weight"], cfg["focal_gamma"], cfg["focal_alpha"]
    ).to(device)
    loss, probs, labels = run_epoch(model, loaders["test"], criterion, device)
    metrics = binary_metrics(labels, probs, ckpt["threshold"])
    metrics["loss"] = loss
    return metrics


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        description="Optuna search over lr, dropout, kernel size, batch size and filters. "
        "Unrecognised options are forwarded to src.training.train.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--timeout", type=float, default=None, help="Seconds; stops starting trials.")
    p.add_argument("--study-name", default="transit_cnn")
    p.add_argument("--storage", default="sqlite",
                   help="'sqlite' (outputs/optuna/<study>.db, resumable), 'memory', or an "
                        "Optuna storage URL.")
    p.add_argument("--optuna-dir", type=Path, default=DEFAULT_OPTUNA_DIR)
    p.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR / "hpo")
    p.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR / "hpo")
    p.add_argument("--sampler-seed", type=int, default=42)
    p.add_argument("--pruner-warmup-epochs", type=int, default=5,
                   help="Epochs before a trial can be pruned.")
    p.add_argument("--no-prune", action="store_true")
    p.add_argument("--keep-all-checkpoints", action="store_true")
    p.add_argument("--evaluate-test", action="store_true",
                   help="After the search, evaluate only the selected model on the test split.")
    args, train_argv = p.parse_known_args(argv)
    if args.n_trials < 1:
        p.error("--n-trials must be >= 1")
    parse_train_args(train_argv)  # fail fast on invalid forwarded options
    return args, train_argv


def main(argv: list[str] | None = None) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args, train_argv = parse_args(argv)
    checkpoint_dir = args.checkpoint_dir / args.study_name
    log_dir = args.log_dir / args.study_name
    out_dir = args.optuna_dir / args.study_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.storage == "memory":
        storage = None
    elif args.storage == "sqlite":
        storage = f"sqlite:///{(args.optuna_dir / f'{args.study_name}.db').resolve().as_posix()}"
    else:
        storage = args.storage

    pruner = (
        optuna.pruners.NopPruner()
        if args.no_prune
        else optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=args.pruner_warmup_epochs)
    )
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.sampler_seed),
        pruner=pruner,
    )
    log.info("Study %s: %d existing trials; storage=%s", args.study_name, len(study.trials),
             storage or "memory")

    callbacks = [] if args.keep_all_checkpoints else [make_cleanup(checkpoint_dir)]
    study.optimize(
        make_objective(train_argv, checkpoint_dir, log_dir),
        n_trials=args.n_trials,
        timeout=args.timeout,
        callbacks=callbacks,
        catch=(ValueError, RuntimeError),
    )

    study.trials_dataframe().to_csv(out_dir / "trials.csv", index=False)
    states = optuna.trial.TrialState
    counts = {s.name: sum(t.state == s for t in study.trials)
              for s in (states.COMPLETE, states.PRUNED, states.FAIL)}
    if counts["COMPLETE"] == 0:
        raise RuntimeError(f"No trial completed: {counts}")

    best = study.best_trial
    best_ckpt = Path(best.user_attrs["run_dir"]) / "best.pt"
    summary = {
        "study_name": args.study_name,
        "objective": "validation PR-AUC",
        "trial_counts": counts,
        "best_trial": best.number,
        "best_val_pr_auc": best.value,
        "best_params": best.params,
        "best_val_metrics": {k: v for k, v in best.user_attrs.items() if k.startswith("val_")},
        "best_epoch": best.user_attrs.get("best_epoch"),
        "threshold_from_validation": best.user_attrs.get("threshold"),
        "best_checkpoint": str(best_ckpt),
        "train_args": train_argv,
    }
    log.info("Trials: %s", counts)
    log.info("Best trial %d: val PR-AUC %.4f, params %s", best.number, best.value, best.params)
    log.info("Best checkpoint: %s", best_ckpt)

    if args.evaluate_test:
        base = parse_train_args(train_argv)
        test = evaluate_on_test(best_ckpt, base.data_dir, base.device)
        summary["test_metrics_of_selected_model"] = test
        log.info(format_metrics("test (selected model, evaluated once)", test))

    (out_dir / "best.json").write_text(json.dumps(summary, indent=2))
    log.info("Summary written to %s", out_dir / "best.json")
    return summary


if __name__ == "__main__":
    main()
