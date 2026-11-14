# Small real-data pipeline validation: views -> dataset -> verification -> one smoke training run.
# Never touches the test split for training/selection; no Optuna.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..
$py = ".\.venv\Scripts\python.exe"

Write-Host "=== 1. KOI table restricted to downloaded stars ==="
& $py -W ignore -c @"
import pandas as pd
from pathlib import Path
m = pd.read_csv('data/raw_small/download_manifest.csv')
print(m.groupby(['koi_disposition', 'status']).size().to_string())
def files_present(kepid, n_files):
    d = Path(f'data/raw_small/kplr{int(kepid):09d}')
    return (d / '.complete').exists() and n_files > 0 and len(list(d.rglob('*.fits'))) == n_files
skipped = m[m.status == 'skipped']
bad = skipped[[not files_present(k, n) for k, n in zip(skipped.kepid, skipped.n_files)]]
print(f'skipped entries: {len(skipped)}, missing/inconsistent (excluded): {len(bad)}')
if len(bad):
    print(bad[['kepid', 'koi_disposition', 'n_files']].to_string(index=False))
ok = m.loc[m.status.isin(['ok', 'skipped']) & (m.n_files > 0) & ~m.kepid.isin(bad.kepid), 'kepid']
t = pd.read_csv('data/labels/kepler_koi_labels.csv')
small = t[t.kepid.isin(ok)]
small.to_csv('data/labels/kepler_koi_labels_small.csv', index=False)
print(f'{ok.size} stars with light curves -> {len(small)} KOIs', small.koi_disposition.value_counts().to_dict())
"@
if ($LASTEXITCODE) { throw "step 1 failed" }

Write-Host "=== 2. Global/local views ==="
& $py -W ignore -m src.preprocessing.views --koi-table data/labels/kepler_koi_labels_small.csv `
    --raw-dir data/raw_small --out-dir data/processed/views --workers 4
if ($LASTEXITCODE) { throw "step 2 failed" }

Write-Host "=== 3. Star-grouped train/val/test split ==="
& $py -W ignore -m src.data.build_dataset --views-dir data/processed/views --out-dir data/processed/dataset
if ($LASTEXITCODE) { throw "step 3 failed" }

Write-Host "=== 4. Verification ==="
& $py -W ignore -c @"
import json, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from src.data.build_dataset import TransitViewsDataset
from src.training.train import build_model, compute_logits
from src.training.losses import build_loss
root = 'data/processed/dataset'
idx = pd.read_csv('data/processed/views/index.csv')
print('view status:', idx.status.value_counts().to_dict())
print(idx.loc[idx.status != 'ok', ['kepoi_name', 'status', 'error']].assign(error=lambda d: d.error.astype(str).str[:80]).to_string())
metas = {s: pd.read_csv(f'{root}/{s}/meta.csv') for s in ('train', 'val', 'test')}
tot = pd.concat(metas.values())
print(f'TOTAL {len(tot)} examples: planets={int(tot.label.sum())} false_positives={int((tot.label == 0).sum())} stars={tot.kepid.nunique()}')
for s, m in metas.items():
    print(f'{s:5s} n={len(m):3d} planets={int(m.label.sum()):3d} fp={int((m.label == 0).sum()):3d} stars={m.kepid.nunique()}')
ids = {s: set(m.kepid) for s, m in metas.items()}
leak = (ids['train'] & ids['val']) | (ids['train'] & ids['test']) | (ids['val'] & ids['test'])
print('kepid overlap between splits:', len(leak))
assert not leak, 'LEAKAGE'
for s in ('train', 'val'):
    ds = TransitViewsDataset(f'{root}/{s}')
    assert torch.isfinite(ds.x_global).all() and torch.isfinite(ds.x_local).all()
    print(f'{s} tensors: global {tuple(ds.x_global.shape)} local {tuple(ds.x_local.shape)} y {tuple(ds.y.shape)}')
train = TransitViewsDataset(f'{root}/train')
g, l, y = next(iter(DataLoader(train, batch_size=8, shuffle=True)))
print('batch shapes: global', tuple(g.shape), 'local', tuple(l.shape), 'y', tuple(y.shape))
model = build_model('astronet', g.shape[-1], l.shape[-1], 0.2, False, 5, 16)
crit = build_loss('weighted_bce', train.y, 'auto', 2.0, 'auto')
logits = compute_logits(model, g, l)
loss = crit(logits, y)
loss.backward()
grads = [p.grad for p in model.parameters() if p.grad is not None]
print('logits', tuple(logits.shape), 'loss', round(float(loss), 4),
      'params with grad', len(grads), '/', sum(1 for _ in model.parameters()),
      'all finite', all(torch.isfinite(gr).all() for gr in grads),
      'grad norm', round(float(torch.norm(torch.stack([gr.norm() for gr in grads]))), 4))
"@
if ($LASTEXITCODE) { throw "step 4 failed" }

Write-Host "=== 5. Smoke-test training (train + val only) ==="
& $py -W ignore -m src.training.train --model astronet --epochs 2 --batch-size 16 --lr 1e-4 `
    --dropout 0.2 --kernel-size 5 --base-filters 16 --monitor val_pr_auc --patience 0 `
    --skip-test --run-name smoke_small --checkpoint-dir checkpoints/smoke --log-dir outputs/runs/smoke
if ($LASTEXITCODE) { throw "step 5 failed" }
Write-Host "=== DONE ==="
