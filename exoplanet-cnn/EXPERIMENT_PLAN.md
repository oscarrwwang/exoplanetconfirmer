# Experiment Plan: Exoplanet Transit CNN

Scope: train and tune `AstronetStyle` (primary) and `BaselineCNN` (reference) on the
star-grouped 70/15/15 split produced by `src.data.build_dataset`.

Ground rules (apply to all stages):

- `pos_weight` is computed from the **train** split only (`--pos-weight auto`).
- The decision threshold is chosen on **validation** only.
- The **test** split is never used for tuning or model selection. It is scored once, at the end,
  on the single selected model (`--evaluate-test`).
- The split is not regenerated or modified between experiments.
- No metrics are recorded in this plan; all numbers come from real runs logged under `outputs/`.

---

## 1. Initial training configuration

| Setting | Value | Rationale |
|---|---|---|
| Model | `astronet` | Two-branch (global 2001 + local 201) view model |
| Optimizer | AdamW (fixed in `train.py`) | |
| Learning rate | `1e-4` | Safe default for Adam-family on 1D CNNs with large dense head |
| Weight decay | `1e-4` | |
| Batch size | `64` | Enough to usually contain positives at typical Kepler TCE imbalance |
| Base filters | `16` | Channels 16→256 (global), 16→32 (local); matches Astronet |
| Kernel size | `5` | Matches Astronet |
| Dropout | `0.2` | Model default is 0.0; dense head (~8.5M params in first FC) overfits easily |
| Batch norm | off | Keep architecture close to reference first |
| Loss | `weighted_bce` with `--pos-weight auto` | Handles class imbalance |
| Scheduler | `plateau`, patience 3 | |
| Epochs / early stopping | 50 / patience 10, `min-delta 1e-4` | |
| Monitor | `val_pr_auc` | More informative than ROC-AUC under heavy imbalance |
| Seed | 42 | |

Run:

```powershell
python -m src.training.train --model astronet --epochs 50 --batch-size 64 --lr 1e-4 `
    --weight-decay 1e-4 --dropout 0.2 --kernel-size 5 --base-filters 16 `
    --loss weighted_bce --pos-weight auto --scheduler plateau --monitor val_pr_auc `
    --patience 10 --skip-test --run-name astronet_init
```

Also run the same with `--model baseline` as a reference point.

---

## 2. Hyperparameter search strategy

### Search space

| Hyperparameter | Range | Scale | Notes |
|---|---|---|---|
| Learning rate | 1e-5 – 1e-2 (narrow to 1e-5 – 3e-3 if Stage B shows divergence above that) | log | Most influential parameter; tune first |
| Batch size | {16, 32, 64, 128} | categorical | Interacts with LR; small batches may contain no positives |
| Base filters | {8, 16, 32} | categorical | Doubles per block; mainly scales the first dense layer |
| Kernel size | {3, 5, 7, 9} | categorical | Odd only (same-padding) |
| Dropout | 0.0 – 0.5 | linear | Applied in the dense head |

This matches `SEARCH_SPACE` in `src/training/hyperparameter_search.py`.

### Stages

**Stage A: Sanity checks (before any tuning)**

1. Overfit a tiny subset (e.g. 64 examples, dropout 0): train loss should approach 0.
   If it does not, there is a bug in data, labels or model and tuning is pointless.
2. Confirm train/val/test have positives and that `pos_weight` printed at start is plausible
   (≈ n_neg / n_pos on train).
3. Run the initial config (Section 1) for both models.

**Stage B: Learning-rate sweep (one factor at a time)**

Fix everything else at the initial config; try LR ∈ {1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3}.
Purpose: find the stable region and set a sensible upper bound for the Optuna range.

**Stage C: Joint Optuna search (TPE + median pruning)**

```powershell
python -m src.training.hyperparameter_search --study-name astro_v1 --n-trials 50 `
    --model astronet --epochs 30 --patience 8 --loss weighted_bce --scheduler plateau
```

- Sampler: TPE (seed 42). First ~10 trials are effectively random exploration.
- Pruner: `MedianPruner(n_startup_trials=5, n_warmup_steps=5)` on val ROC-AUC per epoch.
  Increase `--pruner-warmup-epochs` if low-LR trials are being pruned before they ramp up.
- Objective: validation ROC-AUC of each trial's best checkpoint. Track `val_pr_auc` (stored as a
  trial user attribute) alongside it.
- Study is stored in SQLite, so it can be resumed with the same `--study-name`.
- Budget: 50 trials; extend by 20–30 if the best value is still improving in the last 15 trials.

**Stage D: Analysis**

- Inspect `outputs/optuna/astro_v1/trials.csv`: per-parameter importance
  (`optuna.importance.get_param_importances`), slice plots, and LR × batch-size interaction.
- If the best values sit on a range boundary (e.g. dropout = 0.5, LR at the top), widen that range
  and run a follow-up study.

**Stage E: Robustness re-runs**

Re-train the top 3 configurations with 3 seeds each (`--seed 1/2/3`, full 50 epochs,
`--skip-test`). Select the configuration with the best **mean** validation PR-AUC / ROC-AUC;
prefer the smaller model when the difference is within the seed-to-seed spread.
This reduces the risk of picking a configuration that won by validation noise.

**Stage F: Final evaluation (once)**

Score only the selected checkpoint on test using the validation threshold
(`--evaluate-test`, or `evaluate_on_test`). Report test PR-AUC, ROC-AUC, precision, recall, F1,
and the confusion matrix. Do not go back and tune after seeing test results.

---

## 3. Expected training curves

These are qualitative expectations used to judge whether a run is healthy, not predicted values.

| Curve | Healthy behaviour |
|---|---|
| Train loss | Decreases steadily; fastest in the first ~5–10 epochs, then slower |
| Val loss | Tracks train loss early, then flattens; a later rise signals overfitting |
| Val ROC-AUC | Rises quickly in early epochs, then plateaus; generally higher than PR-AUC |
| Val PR-AUC | Rises more slowly and is noisier (few positives in val) |
| Learning rate (plateau) | Step drops after `scheduler-patience` epochs without improvement; each drop often gives a small further decrease in val loss |

Notes:

- With weighted BCE, absolute loss values are inflated by `pos_weight` and are not comparable
  across runs with different loss settings. Compare runs by AUCs.
- The train–val gap should widen gradually. A gap that opens within a few epochs suggests too
  little regularisation for the model size (raise dropout, lower base filters).
- Epoch-to-epoch noise in val metrics is expected because the val split is small; judge trends,
  not single epochs.
- Larger batch sizes give smoother curves but may need a higher LR to converge at the same speed.

---

## 4. Common failure modes

| Symptom | Likely cause | Action |
|---|---|---|
| Loss is NaN / explodes, or oscillates wildly | LR too high (especially with small batches) | Lower LR; add `--grad-clip 1.0`; cap Optuna LR range |
| Loss barely moves for many epochs | LR too low, or pruned before warmup | Raise LR; increase `--pruner-warmup-epochs` |
| Val ROC-AUC ≈ 0.5, all predictions one class | Imbalance collapse; `pos_weight` not applied; label bug | Check `pos_weight` log line and label counts; run Stage A overfit test |
| Train loss → 0, val loss rises early | Overfitting (large dense head, dropout 0) | Raise dropout, lower base filters, raise weight decay, rely on early stopping |
| Train and val both plateau at poor values | Underfitting or bad preprocessing | Increase filters/kernel, check view normalisation and phase-folding |
| Val metrics jump a lot between epochs | Small val set / few positives | Use early-stopping patience ≥ 8; compare by multi-seed mean (Stage E) |
| Best trial does not reproduce with another seed | HPO overfit to validation noise | Stage E re-runs; prefer simpler configs within noise |
| High recall but very low precision (or vice versa) | Threshold not suited to the goal | Re-select threshold on val for the target operating point; never on test |
| CUDA out of memory | Large batch × base filters 32 × kernel 9 | Lower batch size or filters; Optuna catches `RuntimeError` and continues |
| Suspiciously high val/test scores | Leakage (same star across splits, test used in tuning) | Keep the star-grouped split; confirm test was never loaded during search |
| Unstable training with batch norm and small batches | Batch statistics from 16 samples, often with 0 positives | Keep batch norm off, or use batch size ≥ 64 with it |
