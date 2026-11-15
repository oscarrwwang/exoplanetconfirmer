# exoplanet-cnn

Classify NASA Kepler transit candidates as **planet** or **false positive** with an
Astronet-style two-view 1D CNN, compare it against classical baselines, explain its decisions,
vet it with physical false-positive tests, and run it on new light curves.

> **Detection vs classification.** The CNN does **not** search raw light curves for unknown
> transits. It classifies an *already identified* periodic transit signal, so every candidate
> needs a **period, mid-transit epoch and duration**. During training these come from the NASA
> KOI table; at inference they come from a KOI/TCE table (`--koi-table`) or are supplied on
> the command line (`--period --t0 --duration-hours`).

> **Development status.** The project currently runs on a ~200-star **development** dataset
> (100 confirmed-planet hosts + 100 false-positive hosts, 222 KOIs). It proves the software
> pipeline works end to end. The smoke-test CNN (2 epochs) and the Random Forest / XGBoost
> numbers from this dataset are **not** scientific results; final performance must come from
> the larger dataset (see [Development vs scientific workflow](#development-vs-scientific-workflow)).

---

## Contents

1. [Architecture](#architecture)
2. [Directory structure](#directory-structure)
3. [Setup](#setup)
4. [Data: labels and light curves](#data-labels-and-light-curves)
5. [Preprocessing and global/local views](#preprocessing-and-globallocal-views)
6. [Grouped dataset](#grouped-dataset)
7. [Training](#training)
8. [Hyperparameter search](#hyperparameter-search)
9. [Evaluation](#evaluation)
10. [Explainability](#explainability)
11. [Classical baselines](#classical-baselines)
12. [False-positive vetting](#false-positive-vetting)
13. [Inference on a star](#inference-on-a-star)
14. [Development vs scientific workflow](#development-vs-scientific-workflow)
15. [Configuration and reproducibility](#configuration-and-reproducibility)
16. [Outputs and checkpoints](#outputs-and-checkpoints)
17. [Testing](#testing)

---

## Architecture

```
NASA Exoplanet Archive (cumulative KOI table)        MAST (Kepler long-cadence FITS)
        │ src.data.download_kepler --labels-only              │ src.data.download_kepler
        ▼                                                     ▼
data/labels/kepler_koi_labels.csv  ──────────────►  data/raw[_small]/kplr<kepid>/…/*_llc.fits
  (CONFIRMED = 1, FALSE POSITIVE = 0, per KOI;                │
   period, epoch, duration)                                   │
        └──────────────────────────┬──────────────────────────┘
                                   ▼
        src.preprocessing.views   clean → detrend (transits masked) → phase-fold on the
                                  KOI ephemeris → global view (2001 bins, full orbit)
                                  + local view (201 bins, ±4 durations) → normalize
                                   ▼  data/processed/views/<kepoi>.npz, index.csv, views_config.json
        src.data.build_dataset    star-grouped (kepid), label-stratified 70/15/15 split
                                   ▼  data/processed/dataset/{train,val,test}/X_global.npy, X_local.npy, y.npy, meta.csv
        TransitViewsDataset / DataLoader  →  (B,1,2001), (B,1,201), (B,)
                                   ▼
        src.models.astronet_style.AstronetStyle   → raw logit (B,)
        src.training.train        weighted BCE-with-logits, pos_weight from TRAIN,
                                  AdamW, early stopping, val-selected threshold
                                   ▼  checkpoints/<run>/best.pt (+ threshold)
        src.training.hyperparameter_search   Optuna, objective = val PR-AUC
                                   ▼
        src.evaluation.evaluate   metrics + ROC/PR/confusion plots
        src.evaluation.explain    Grad-CAM, SmoothGrad, occlusion, region importance
        src.baselines.compare     BLS, BLS vetting, Random Forest, XGBoost vs CNN
        src.inference.vetting     reason codes + PLANET CANDIDATE / FALSE POSITIVE / NEEDS REVIEW
        src.inference.predict     FITS + ephemeris → probability, class, confidence, report
```

Every downstream step reuses the upstream code: inference calls the same `generate_views`
as training (verified to reproduce training views bit-for-bit), and vetting consumes the
BLS / shape features from `src.baselines.features` rather than recomputing them.

---

## Directory structure

Actual layout (data, checkpoint and output contents abbreviated):

```
exoplanet-cnn/
├── README.md
├── EXPERIMENT_PLAN.md            # training/tuning plan and ground rules
├── pyproject.toml                # package metadata, dependencies, pytest/ruff config
├── requirements.txt
├── configs/
│   └── reproducibility.json      # snapshot of all reproducibility settings (checked by a test)
├── scripts/
│   └── run_small_pipeline.ps1    # development workflow: views → split → checks → smoke training
├── src/
│   ├── data/
│   │   ├── download_kepler.py    # KOI labels (NASA Exoplanet Archive) + light curves (MAST)
│   │   ├── build_dataset.py      # grouped splits, TransitViewsDataset
│   │   ├── build_labels.py       # legacy: star-level labels.csv (not used by the CNN pipeline)
│   │   └── kepler_dataset.py     # legacy: full-light-curve Dataset (not used by the CNN pipeline)
│   ├── preprocessing/
│   │   ├── preprocess.py         # cleaning/detrending primitives (+ legacy full-curve CLI)
│   │   └── views.py              # phase folding, global/local views
│   ├── models/
│   │   ├── astronet_style.py     # two-branch CNN (primary)
│   │   └── baseline_cnn.py       # global-view-only CNN (reference)
│   ├── training/
│   │   ├── train.py              # training loop, checkpoints, threshold selection
│   │   ├── losses.py             # weighted BCE, focal loss (all on logits)
│   │   └── hyperparameter_search.py  # Optuna
│   ├── evaluation/
│   │   ├── metrics.py            # PR/ROC-AUC, P/R/F1, confusion, threshold selection
│   │   ├── evaluate.py           # checkpoint evaluation + plots
│   │   └── explain.py            # Grad-CAM, saliency, occlusion, region importance
│   ├── baselines/
│   │   ├── features.py           # BLS statistics + transit-shape features
│   │   └── compare.py            # BLS / RF / XGBoost vs CNN results table
│   └── inference/
│       ├── vetting.py            # false-positive reason codes and final disposition
│       └── predict.py            # production inference + report
├── tests/                        # pytest suite (one file per module)
├── notebooks/                    # (empty)
├── data/                         # git-ignored contents
│   ├── labels/                   # kepler_koi_labels.csv, kepler_koi_labels_small.csv, labels.csv
│   ├── raw/                      # light curves (production downloads)
│   ├── raw_small/                # light curves of the 200-star development sample
│   └── processed/                # views/, dataset/, baseline_features.csv
├── checkpoints/                  # git-ignored; <run>/best.pt, last.pt, metrics.json, config.json
└── outputs/                      # git-ignored; evaluation/, explain/, baselines/, vetting/,
                                  # inference/, runs/ (TensorBoard), optuna/, logs
```

`src/data/build_labels.py`, `src/data/kepler_dataset.py` and the CLI of
`src/preprocessing/preprocess.py` form an earlier star-level, full-light-curve path. They are
kept (and tested) but the CNN pipeline below uses per-KOI labels and phase-folded views instead.

---

## Setup

Python ≥ 3.10 (developed with 3.14).

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt     # or: pip install -e ".[dev]"  (adds pytest, ruff, jupyter)
```

For a GPU, install the matching PyTorch build from https://pytorch.org/get-started/locally/.
All commands below are run from the project root.

**TLS-inspecting proxies (e.g. Zscaler).** If HTTPS downloads fail or are very slow because of
a large corporate CA bundle in `SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE`, install `truststore`
(`pip install truststore`) and add `--system-certs` to `src.data.download_kepler`; it then
verifies certificates with the operating-system store.

---

## Data: labels and light curves

All data come from NASA:

* **Labels** – the *cumulative* KOI table of the NASA Exoplanet Archive. Each KOI with
  disposition `CONFIRMED` is labelled 1, `FALSE POSITIVE` 0; `CANDIDATE` KOIs are excluded.
  The table also provides each KOI's ephemeris (`koi_period` [d], `koi_time0bk` [BKJD],
  `koi_duration` [h]).
* **Light curves** – Kepler long-cadence (`*_llc.fits`) PDCSAP light curves from MAST via
  `lightkurve`, one directory per star: `kplr<kepid 9 digits>/mastDownload/Kepler/…/*.fits`.

```powershell
# 1. KOI labels + ephemerides -> data/labels/kepler_koi_labels.csv
python -m src.data.download_kepler --labels-only

# 2. Light curves. --limit N = N stars per disposition (random, --seed 42).
#    --koi-table reuses the saved table, so target selection is reproducible and offline.
python -m src.data.download_kepler --koi-table data/labels/kepler_koi_labels.csv `
    --limit 100 --seed 42 --raw-dir data/raw_small --workers 4
```

Downloads are resumable: a star is complete only when every file arrived and opens as a valid
light curve (a `.complete` marker is then written). Truncated files are renamed to
`*.fits.partial` and fetched again. `<raw-dir>/download_manifest.csv` records each star's
status (`ok`, `skipped` = already complete, `not_found`, `failed`). A star hosting any
confirmed KOI is sampled as a confirmed host; all of its KOIs (with their own labels) are used.

Other options (see `--help`): `--dispositions`, `--exptime {long,short}`, `--quarters`,
`--retries`, `--overwrite`, `--system-certs`.

---

## Preprocessing and global/local views

`src.preprocessing.views` builds, for every KOI in the table whose star has light curves:

1. read PDCSAP flux, drop NaNs; per gap-separated segment
2. detrend with a 1-day binned running median **fitted with all known transits on the star
   masked**, upper-only 3σ clipping (dips are kept)
3. remove the transits of the *other* KOIs on the same star
4. phase-fold on the KOI's period/epoch (transit at phase 0)
5. median-bin into a **global view** (2001 bins over the full orbit) and a **local view**
   (201 bins over ±4 transit durations), interpolate empty bins
6. normalize: out-of-transit baseline 0, transit bottom −1 (`shared` scaling)

```powershell
python -m src.preprocessing.views --koi-table data/labels/kepler_koi_labels.csv `
    --raw-dir data/raw --out-dir data/processed/views --workers 8
```

Writes `<out-dir>/<kepoi_name>.npz` (views, label, kepid, period, t0, duration, depth),
`index.csv` (status per KOI: `ok`, `missing`, `empty`, `bad_ephemeris`, `skipped`) and
`views_config.json` (needed later by inference). Options: `--global-bins`, `--local-bins`,
`--local-durations`, `--detrend-window`, `--transit-mask-factor`, `--normalization`,
`--limit`, `--overwrite`.

---

## Grouped dataset

```powershell
python -m src.data.build_dataset --views-dir data/processed/views `
    --out-dir data/processed/dataset --train 0.7 --val 0.15 --test 0.15 --seed 42
```

Stars (kepid) are assigned whole to one split, stratified by the star's maximum label, so KOIs
of the same star can never appear in two splits. Output per split: `X_global.npy` (N, 2001),
`X_local.npy` (N, 201), `y.npy`, `meta.csv`; plus `summary.json`. `TransitViewsDataset`
yields `(global (1, 2001), local (1, 201), label)` float32 tensors.

**Do not regenerate or modify the split between experiments.**

---

## Training

```powershell
python -m src.training.train --model astronet --epochs 50 --batch-size 64 --lr 1e-4 `
    --weight-decay 1e-4 --dropout 0.2 --kernel-size 5 --base-filters 16 `
    --loss weighted_bce --pos-weight auto --scheduler plateau --monitor val_pr_auc `
    --patience 10 --skip-test --run-name astronet_init
```

* **Model output** – both CNNs return **raw logits**; the loss uses
  `binary_cross_entropy_with_logits` (`BCEWithLogitsLoss`), sigmoid is applied only for metrics.
* **pos_weight** – `--pos-weight auto` computes `n_negative / n_positive` from the **TRAIN**
  split only and logs it (e.g. `pos_weight = 0.9125 (training split only: 73 neg / 80 pos)`).
  A number may be given instead. `--loss bce` (unweighted) and `--loss focal`
  (`--focal-gamma`, `--focal-alpha auto` = TRAIN negative fraction) are also available.
* **Monitoring** – the default is `--monitor val_pr_auc`; `val_loss` and `val_roc_auc` remain
  available through `--monitor`. The monitored
  value drives the best checkpoint, early stopping (`--patience`, `--min-delta`) and the
  `plateau` scheduler. Choose `--scheduler cosine|none` otherwise.
* **Threshold** – after training, the decision threshold maximizing F1 on **VALIDATION** is
  stored in `best.pt` (override with `--threshold`).
* **Checkpoints** – `checkpoints/<run>/best.pt` (best monitored epoch + threshold), `last.pt`
  (every epoch), `metrics.json`, `config.json`; TensorBoard logs in `outputs/runs/<run>`
  (`tensorboard --logdir outputs/runs`). Resume with `--resume checkpoints/<run>/last.pt`.
* **TEST** – use `--skip-test` while developing; without it the final epoch summary also scores
  the test split, which is only appropriate for the single final model.
* Other options: `--model baseline`, `--batch-norm`, `--grad-clip`, `--device`,
  `--num-workers`, `--seed`, `--data-dir`, `--checkpoint-dir`, `--log-dir`.
  Show the architecture with `python -m src.models.astronet_style`.

See [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) for the planned configurations and failure modes.

---

## Hyperparameter search

```powershell
python -m src.training.hyperparameter_search --n-trials 50 --model astronet --epochs 30
```

* Tunes `lr` (1e-5–1e-2, log), `dropout` (0–0.5), `kernel_size` {3,5,7,9},
  `batch_size` {16,32,64,128}, `base_filters` {8,16,32}.
* Objective: **validation PR-AUC** of each trial's best checkpoint (trials run with
  `--monitor val_pr_auc --skip-test`; TEST is never loaded during the search).
* TPE sampler (`--sampler-seed 42`), median pruner (`--pruner-warmup-epochs 5`, `--no-prune`),
  SQLite storage `outputs/optuna/<study>.db` – rerunning with the same `--study-name` resumes.
* Unrecognised options are forwarded to `src.training.train` (e.g. `--model`, `--epochs`,
  `--loss`, `--data-dir`).
* Results: `outputs/optuna/<study>/trials.csv`, `best.json`; only the best trial's checkpoint
  is kept (`--keep-all-checkpoints` to keep all).
* `--evaluate-test` scores the single selected model on TEST once, after the search.

---

## Evaluation

```powershell
python -m src.evaluation.evaluate --checkpoint checkpoints/<run>/best.pt            # VAL
python -m src.evaluation.evaluate --checkpoint checkpoints/<run>/best.pt `
    --split test --final-test                                                       # once, at the end
```

Writes `metrics.json`, `predictions.csv`, `roc_curve.png`, `pr_curve.png`,
`confusion_matrix.png` to `outputs/evaluation/<run>/<split>/`.

| Metric | Meaning for planet search |
|---|---|
| **PR-AUC** (average precision) | **Primary metric.** Purity of the ranked candidate list; robust to class imbalance. |
| Prevalence | Planet fraction = PR-AUC of a random classifier; compare PR-AUC against it. |
| ROC-AUC | Ranking quality over all thresholds; optimistic when negatives dominate. |
| Recall at precision ≥ 0.90 / 0.95 / 0.99 | Planets recovered while keeping follow-up lists pure. |
| Precision at recall ≥ … | Purity when a recall target must be met. |
| Precision, recall, F1, accuracy | At the validation-selected threshold. Accuracy is misleading under imbalance. |
| Confusion matrix | TN / FP / FN / TP counts (FN = missed planets, FP = wasted follow-up). |
| Specificity, false-positive rate | Fraction of false positives correctly rejected / wrongly accepted. |

**Threshold selection** happens on VALIDATION only (max F1, stored in the checkpoint).

**The TEST split must not be used for model selection, hyperparameter selection, threshold
selection or early stopping.** Every command that can read TEST requires an explicit
`--final-test` (`evaluate`, `explain`, `compare`) or `--evaluate-test` (Optuna); use it once,
for the single model already chosen on validation.

---

## Explainability

```powershell
python -m src.evaluation.explain --checkpoint checkpoints/<run>/best.pt --examples 3
```

All attributions explain the **planet** logit (red = pushes towards "planet").

| Method | What it shows |
|---|---|
| Grad-CAM (1D) | Gradient-weighted last-conv activations of each branch, upsampled; coarse, [0, 1] |
| Saliency (SmoothGrad) | Mean \|∂logit/∂flux\| over noisy copies (`--smooth-samples`, `0` = vanilla gradient) |
| Occlusion | Drop in P(planet) when a sliding window is set to the baseline (`--occlusion-global/-local WINDOW STRIDE`); signed |
| View importance | Probability drop when the whole global or local view is blanked |
| Region importance | Attribution share and enrichment (share / area) in the **transit** window, the **secondary-eclipse** window (phase 0.5) and the **out-of-transit baseline**, from each KOI's period/duration |

Outputs in `outputs/explain/<run>/<split>/`: per-example figures for the most confident TP /
FN / FP / TN, `class_mean_attribution.png`, `region_importance.png/.csv`,
`region_summary.csv`, `view_importance.csv`, `attributions.npz`, `metrics.json`.
`--split test` requires `--final-test`.

**Attributions from an untrained or poorly trained model are not scientifically meaningful.**
The current smoke-test checkpoint outputs an almost constant probability, so its Grad-CAM is
zero and its occlusion maps are ~1e-5; the command warns when this happens.

---

## Classical baselines

```powershell
python -m src.baselines.compare --raw-dir data/raw_small `
    --koi-table data/labels/kepler_koi_labels_small.csv `
    --checkpoint checkpoints/<run>/best.pt
```

| Model | Score | Training |
|---|---|---|
| BLS SNR | astropy `BoxLeastSquares` transit significance at the catalog ephemeris | none |
| BLS vetting | −max(odd/even depth σ, secondary-eclipse SNR) – classic eclipsing-binary tests | none |
| Random Forest | 16 engineered features (`src.baselines.features`), `class_weight="balanced"` | TRAIN |
| XGBoost | same features, `scale_pos_weight = n_neg / n_pos` from TRAIN | TRAIN |
| CNN | the given checkpoint | – |

Same protocol as the CNN: fit on TRAIN, fixed hyperparameters, thresholds chosen on VAL,
95 % bootstrap intervals (`--bootstrap`). Features (BLS depth/SNR/odd-even/secondary/harmonic,
transit shape, ephemeris) are cached in `data/processed/baseline_features.csv`.
`--feature-set views|both` feeds raw view bins to the tree models. Outputs:
`outputs/baselines/<split>/results.csv`, `results.md`, `predictions.csv`, `pr_curves.png`,
`roc_curves.png`, `feature_importance.csv/.png`. `--split test` requires `--final-test`.

BLS here is evaluated at the known ephemeris (no blind period search), like the CNN.
**Results on the development dataset (34 validation KOIs, wide confidence intervals, CNN
trained for only 2 epochs) are not a scientific comparison.**

---

## False-positive vetting

`src.inference.vetting` turns the existing features, the CNN probability and (in inference)
the occlusion evidence into reason codes. Limits are fixed, physically motivated defaults
(`VettingConfig`), not fitted to labels.

| Code | Severity | Fires when |
|---|---|---|
| `ODD_EVEN_DEPTH` | false_positive | odd/even depths differ > 3σ (EB at twice the period) |
| `SECONDARY_ECLIPSE` | false_positive | dip at phase 0.5 with SNR > 3 |
| `V_SHAPED` | false_positive | inner-half / full transit depth > 1.3 (grazing EB) |
| `TOO_DEEP` | false_positive | depth > 30 000 ppm |
| `SINUSOIDAL` | false_positive | a sinusoid fits better than a box (variability) |
| `OUT_OF_TRANSIT_DIPS` | warning | out-of-transit dip > 50 % of the transit depth |
| `LOW_SNR` | warning | BLS SNR < 7.1 |
| `FEW_TRANSITS` | warning | < 3 transits with data |
| `SINGLE_EVENT` | warning | one event > 50 % of the signal |
| `ASYMMETRIC` | warning | ingress/egress halves differ > 0.25 of depth |
| `NOISY_FOLD` | warning | out-of-transit scatter > 30 % of depth |
| `LOW_COVERAGE` | warning | < 50 % of local-view bins have data |
| `NO_BLS_FIT` | warning | BLS statistics unavailable |
| `MODEL_OFF_TRANSIT` | warning | < 30 % of occlusion evidence inside the transit window |
| `MODEL_LOW_CONFIDENCE` | info | confidence score < 0.25 |

**Confidence score** = distance of the probability from the validation threshold, scaled to
[0, 1] on each side (`high` ≥ 0.6, `medium` ≥ 0.25, else `low`).

**Final disposition**

| Disposition | Rule |
|---|---|
| `PLANET CANDIDATE` | CNN ≥ threshold, confidence not low, no false-positive-severity code |
| `FALSE POSITIVE` | CNN < threshold (confident, or backed by a false-positive code) |
| `NEEDS REVIEW` | CNN ≥ threshold but a false-positive test fires, or confidence is low |

**`PLANET CANDIDATE` does not mean a confirmed exoplanet.** Confirmation requires independent
follow-up (e.g. radial velocities, high-resolution imaging, statistical validation).

Summarize how often each code fires per class on TRAIN/VAL (TEST is not available here):

```powershell
python -m src.inference.vetting --checkpoint checkpoints/<run>/best.pt `
    --raw-dir data/raw_small --koi-table data/labels/kepler_koi_labels_small.csv
```

Outputs `outputs/vetting/{train,val}_vetting.csv`, `{train,val}_flag_summary.csv`,
`vetting_config.json`.

---

## Inference on a star

"Inference on a star" means:

```
Kepler FITS files of one star
 → candidate ephemeris (period, t0, duration)   from --koi-table or the command line
 → preprocessing (same code as training)         clean, detrend, mask other KOIs
 → global/local views                            using the training views_config.json
 → trained CNN (loaded, NOT retrained)           planet probability
 → classification + confidence                   validation threshold from the checkpoint
 → physical vetting                              BLS/shape features, occlusion evidence, reason codes
 → report                                        report.json, report.txt, <candidate>.png
```

```powershell
# every KOI of the star listed in a KOI table (others on the star are masked)
python -m src.inference.predict data/raw_small/kplr005357901 `
    --checkpoint checkpoints/<run>/best.pt --koi-table data/labels/kepler_koi_labels.csv

# only some KOIs of that star
python -m src.inference.predict data/raw_small/kplr005357901 `
    --checkpoint checkpoints/<run>/best.pt --koi-table data/labels/kepler_koi_labels.csv `
    --kepoi-name K00188.01

# an explicit ephemeris (t0 in BKJD = BJD - 2454833)
python -m src.inference.predict path\to\kplr*.fits --checkpoint checkpoints/<run>/best.pt `
    --period 3.79702 --t0 133.5088 --duration-hours 2.26 --name my_candidate
```

Inputs are FITS files or directories of **one** star (mixed stars are rejected). The report
lists planet probability, model classification, confidence score, final disposition,
reason codes, data quality and the fraction of model evidence inside the transit window, and
notes if the star is already in the model's dataset (`--data-dir`), in which case the
prediction is not independent. `--no-explain` skips the attribution figure;
`--views-config` must point to the `views_config.json` used to build the model's training
views (default `data/processed/views/views_config.json`). Output:
`outputs/inference/kic<kepid>/` or `--out-dir`.

The probability is a sigmoid of the CNN logit and **is not necessarily calibrated**; the
confidence score is a distance from the decision threshold, not a likelihood.

---

## Development vs scientific workflow

### Development workflow (current, ~200 stars)

Purpose: prove the pipeline end to end on real NASA data.

```powershell
python -m src.data.download_kepler --labels-only                       # once
python -m src.data.download_kepler --koi-table data/labels/kepler_koi_labels.csv `
    --limit 100 --seed 42 --raw-dir data/raw_small --workers 4         # writes download_manifest.csv
powershell -ExecutionPolicy Bypass -File scripts\run_small_pipeline.ps1
```

`scripts/run_small_pipeline.ps1`:

1. keeps stars whose manifest status is `ok`, or `skipped` with a `.complete` marker and the
   expected number of FITS files, and writes `data/labels/kepler_koi_labels_small.csv`
2. builds views from `data/raw_small` into `data/processed/views`
3. builds the grouped split into `data/processed/dataset`
4. verifies split sizes, zero kepid overlap, tensor shapes, finite values, logits and gradients
5. trains a 2-epoch smoke-test CNN on TRAIN/VAL only (`--skip-test`) into `checkpoints/smoke/`

Current development dataset: 200 stars → 222 KOIs (119 planets, 103 false positives);
TRAIN 153 KOIs / 140 stars, VAL 34 / 30, TEST 35 / 30; zero kepid overlap. The smoke-test
CNN is a plumbing check (its validation probabilities are nearly constant), and the
baseline/vetting runs on this dataset only verify that those components run correctly.
**None of these numbers are evidence of final model performance.** The TEST split of the
development dataset has not been used.

### Scientific workflow (larger dataset, not yet run)

```powershell
python -m src.data.download_kepler --koi-table data/labels/kepler_koi_labels.csv `
    --raw-dir data/raw --workers 8                                     # all labelled stars
python -m src.preprocessing.views --koi-table data/labels/kepler_koi_labels.csv `
    --raw-dir data/raw --out-dir data/processed/views_full --workers 8
python -m src.data.build_dataset --views-dir data/processed/views_full `
    --out-dir data/processed/dataset_full --seed 42
python -m src.training.train --data-dir data/processed/dataset_full --model astronet `
    --dropout 0.2 --monitor val_pr_auc --skip-test --run-name astronet_full
python -m src.training.hyperparameter_search --data-dir data/processed/dataset_full `
    --model astronet --epochs 30 --n-trials 50 --study-name astronet_full
python -m src.evaluation.evaluate --checkpoint checkpoints/hpo/astronet_full/trial_<NNNN>/best.pt `
    --data-dir data/processed/dataset_full                              # VAL
# ... explain / compare / vetting on VAL with the same --data-dir and the matching --views-dir ...
# Finally, once, for the single selected model:
python -m src.evaluation.evaluate --checkpoint <selected_run_dir>/best.pt `
    --data-dir data/processed/dataset_full --split test --final-test
```

Using separate `views_full` / `dataset_full` directories keeps the development artefacts
intact; pass the matching `--views-config data/processed/views_full/views_config.json` to
inference.

---

## Configuration and reproducibility

The CLIs are the source of truth; [configs/reproducibility.json](configs/reproducibility.json)
records the defaults in one place and `tests/test_config_snapshot.py` fails if they drift.

| Setting | Value |
|---|---|
| Random seeds | 42 for target sampling, split, training, Optuna sampler, baselines |
| Views | global 2001 bins (full orbit), local 201 bins (±4 durations, bin width 0.16 × duration), PDCSAP, 1-day detrending, shared normalization |
| Split | 70 / 15 / 15, grouped by kepid, stratified by label |
| CNN | AstronetStyle: global 5 blocks (16→256 ch, pool 5/2), local 2 blocks (16→32, pool 7/2), 2 convs/block, kernel 5, 4 × 512 dense, raw logit |
| Optimizer | AdamW, lr 1e-4, weight decay 1e-4 |
| Training | batch 64, ≤ 50 epochs, plateau scheduler (patience 3), early stopping (patience 10, min-delta 1e-4), monitor `val_pr_auc` (default; `val_loss` / `val_roc_auc` via `--monitor`); recommended dropout 0.2 |
| pos_weight | `auto` = n_neg / n_pos of TRAIN only |
| Threshold | max F1 on VAL, stored in `best.pt` |
| Primary metric | validation PR-AUC |
| Optuna | TPE (seed 42), median pruner (5 startup trials, 5 warm-up epochs), 30 trials default, SQLite |
| Vetting | limits in `VettingConfig` (table above) |

Every run also saves its exact settings: `checkpoints/<run>/config.json`,
`data/processed/views/views_config.json`, `data/processed/dataset/summary.json`,
`outputs/baselines/<split>/config.json`, `outputs/vetting/vetting_config.json`.

---

## Outputs and checkpoints

| Path | Produced by |
|---|---|
| `data/labels/kepler_koi_labels.csv` | `download_kepler --labels-only` |
| `data/raw*/kplr*/`, `download_manifest.csv` | `download_kepler` |
| `data/processed/views/` | `preprocessing.views` |
| `data/processed/dataset/` | `data.build_dataset` |
| `checkpoints/<run>/best.pt, last.pt, metrics.json, config.json` | `training.train` |
| `checkpoints/hpo/<study>/`, `outputs/optuna/<study>/` | `training.hyperparameter_search` |
| `outputs/runs/` | TensorBoard logs |
| `outputs/evaluation/<run>/<split>/` | `evaluation.evaluate` |
| `outputs/explain/<run>/<split>/` | `evaluation.explain` |
| `data/processed/baseline_features.csv`, `outputs/baselines/<split>/` | `baselines.compare` |
| `outputs/vetting/` | `inference.vetting` |
| `outputs/inference/kic<kepid>/` | `inference.predict` |

`data/`, `checkpoints/` and `outputs/` contents are git-ignored.

---

## Testing

```powershell
python -m pytest -q
```

The suite uses only temporary synthetic files (no network, no `data/`); it covers download
helpers, preprocessing, views, splitting, models, losses, training, Optuna, metrics,
evaluation, explainability, baselines, vetting, inference and the configuration snapshot.
