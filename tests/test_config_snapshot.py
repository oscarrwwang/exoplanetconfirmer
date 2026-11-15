import inspect
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.baselines import compare
from src.data import build_dataset, download_kepler
from src.inference.vetting import VettingConfig
from src.models.astronet_style import AstronetStyle
from src.preprocessing.views import ViewConfig
from src.training import hyperparameter_search, train

SNAPSHOT = json.loads(
    (Path(__file__).resolve().parents[1] / "configs" / "reproducibility.json").read_text()
)


def test_views_and_vetting_match_dataclasses():
    views = asdict(ViewConfig())
    for key, value in SNAPSHOT["views"].items():
        if key != "module":
            assert views[key] == value, key
    vetting = asdict(VettingConfig())
    assert {k: v for k, v in SNAPSHOT["vetting"].items() if k != "module"} == vetting


def test_split_training_and_optuna_defaults():
    split = vars(build_dataset.parse_args([]))
    assert [split[k] for k in ("train", "val", "test", "seed")] == [
        SNAPSHOT["split"][k] for k in ("train", "val", "test")] + [SNAPSHOT["seed"]["dataset_split"]]

    t = vars(train.parse_args([]))
    cfg = SNAPSHOT["training"]
    for key in ("model", "lr", "weight_decay", "batch_size", "epochs", "grad_clip", "loss",
                "pos_weight", "scheduler", "scheduler_patience", "patience", "min_delta"):
        assert t[key] == cfg[key], key
    assert t["monitor"] == cfg["monitor_default"] and t["seed"] == SNAPSHOT["seed"]["training"]

    hpo, _ = hyperparameter_search.parse_args([])
    opt = SNAPSHOT["optuna"]
    assert hpo.n_trials == opt["n_trials"] and hpo.storage == opt["storage"]
    assert hpo.pruner_warmup_epochs == opt["pruner_warmup_epochs"]
    assert hpo.sampler_seed == SNAPSHOT["seed"]["optuna_sampler"]
    space = {k: list(v) for k, v in hyperparameter_search.SEARCH_SPACE.items()}
    assert space == opt["search_space"]


def test_cnn_download_and_baseline_defaults():
    params = {k: v.default for k, v in inspect.signature(AstronetStyle).parameters.items()}
    for key, value in SNAPSHOT["cnn"].items():
        if key in params:
            expected = tuple(value) if isinstance(value, list) else value
            assert params[key] == expected, key
    assert params["global_length"] == SNAPSHOT["views"]["global_bins"]
    assert params["local_length"] == SNAPSHOT["views"]["local_bins"]

    dl = vars(download_kepler.parse_args([]))
    assert dl["dispositions"] == SNAPSHOT["download"]["dispositions"]
    assert dl["exptime"] == SNAPSHOT["download"]["exptime"]
    assert dl["seed"] == SNAPSHOT["seed"]["download_target_sampling"]

    cmp = vars(compare.parse_args([]))
    assert cmp["feature_set"] == SNAPSHOT["baselines"]["feature_set"]
    assert cmp["bootstrap"] == SNAPSHOT["baselines"]["bootstrap"]
    assert cmp["seed"] == SNAPSHOT["seed"]["baselines"]
    rf = compare.make_random_forest(None, 42).steps[-1][1]
    assert rf.n_estimators == SNAPSHOT["baselines"]["random_forest_n_estimators"]
    xgb = compare.make_xgboost(np.array([0, 1]), 42)
    assert xgb.get_params()["max_depth"] == SNAPSHOT["baselines"]["xgboost_max_depth"]
