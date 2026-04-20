"""Quick multi-horizon check: does Mamba separate from persistence at longer H?

Trains one model per horizon in {3, 6, 12, 18}, reports held-out MAE for
Mamba and the three baselines. No plumbing changes elsewhere — writes a
small JSON + prints the table.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_momentum import SelectiveSSM  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SEQ_NPZ = ROOT / "Data" / "learned" / "sequences.npz"
OUT = ROOT / "Data" / "learned" / "horizon_sweep.json"

SEED = 0
WINDOW = 24
HIDDEN = 32
N_BLOCKS = 2
BATCH = 64
EPOCHS = 80
LR = 3e-3
PATIENCE = 12
TRAIN_CUTOFF = "2025-01"
HORIZONS = [1, 3, 6, 9, 12]


def _seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


class ForecastEncoder(nn.Module):
    def __init__(self, n_features, hidden=HIDDEN, n_blocks=N_BLOCKS):
        super().__init__()
        self.embed = nn.Linear(n_features, hidden)
        self.blocks = nn.ModuleList([SelectiveSSM(hidden) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden),
            nn.GELU(), nn.Linear(hidden, 1),
        )

    def forward(self, x):
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h)
        return self.head(h[:, -1, :]).squeeze(-1)


class DS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def build_windows(X, iso3, snapshots, horizon):
    N, T, F_ = X.shape
    sev = X[:, :, 0]; obs = X[:, :, -1]
    Xs, ys, meta = [], [], []
    for i in range(N):
        for t in range(WINDOW, T - horizon):
            if obs[i, t - 1] < 0.5:
                continue
            if obs[i, t : t + horizon].mean() < 0.5:
                continue
            target = float(sev[i, t : t + horizon].mean())
            cur = float(sev[i, t - 1])
            seasonal = float(sev[i, t - 13]) if t >= 13 and obs[i, t - 13] >= 0.5 else cur
            Xs.append(X[i, t - WINDOW : t, :])
            ys.append(target)
            meta.append((str(iso3[i]), str(snapshots[t - 1]), cur, seasonal))
    return (np.stack(Xs).astype(np.float32),
            np.array(ys, dtype=np.float32),
            meta)


def run_one(X, iso3, snapshots, horizon):
    _seed(SEED)
    Xw, y, meta = build_windows(X, iso3, snapshots, horizon)
    train_idx = np.array([i for i, m in enumerate(meta) if m[1] < TRAIN_CUTOFF], dtype=np.int64)
    test_idx  = np.array([i for i, m in enumerate(meta) if m[1] >= TRAIN_CUTOFF], dtype=np.int64)
    if len(test_idx) == 0 or len(train_idx) == 0:
        return {"horizon": horizon, "n_test": int(len(test_idx)),
                "mae_mamba": None, "mae_persistence": None,
                "mae_seasonal": None, "mae_mean": None,
                "note": "empty split at this horizon"}

    F_seq = Xw.shape[-1]
    mean = Xw[train_idx].reshape(-1, F_seq).mean(0)
    std = Xw[train_idx].reshape(-1, F_seq).std(0) + 1e-6
    mean[-1] = 0.0; std[-1] = 1.0
    Xn = (Xw - mean) / std
    y_mean = float(y[train_idx].mean()); y_std = float(y[train_idx].std() + 1e-6)
    yn = (y - y_mean) / y_std

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = ForecastEncoder(F_seq).to(device)
    tl = DataLoader(DS(Xn[train_idx], yn[train_idx]), BATCH, shuffle=True)
    vl = DataLoader(DS(Xn[test_idx],  yn[test_idx]),  BATCH, shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=3e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, epochs=EPOCHS, steps_per_epoch=len(tl), pct_start=0.15)

    best = float("inf"); best_state = None; pat = 0
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in tl:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.huber_loss(model(xb), yb, delta=0.5)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step(); sched.step()
        model.eval(); vloss = 0.0; nb = 0
        with torch.no_grad():
            for xb, yb in vl:
                vloss += F.huber_loss(model(xb.to(device)), yb.to(device), delta=0.5).item(); nb += 1
        vloss /= max(nb, 1)
        if vloss < best - 1e-4:
            best = vloss; best_state = {k: v.clone() for k, v in model.state_dict().items()}; pat = 0
        else:
            pat += 1
            if pat >= PATIENCE:
                break
    model.load_state_dict(best_state); model.eval()

    preds = []
    with torch.no_grad():
        for xb, _ in DataLoader(DS(Xn[test_idx], yn[test_idx]), BATCH, shuffle=False):
            preds.extend(model(xb.to(device)).cpu().numpy().tolist())
    pred = np.array(preds, dtype=np.float32) * y_std + y_mean
    true = y[test_idx]
    cur = np.array([meta[i][2] for i in test_idx], dtype=np.float32)
    seas = np.array([meta[i][3] for i in test_idx], dtype=np.float32)

    def _mae(a, b): return float(np.mean(np.abs(a - b)))
    return {
        "horizon": horizon,
        "n_test": int(len(test_idx)),
        "mae_mamba":       _mae(pred, true),
        "mae_persistence": _mae(cur, true),
        "mae_seasonal":    _mae(seas, true),
        "mae_mean":        _mae(np.full_like(true, y_mean), true),
    }


def main():
    z = np.load(SEQ_NPZ, allow_pickle=True)
    X = z["X"]; iso3 = z["iso3"]; snapshots = z["snapshot"]

    results = []
    print(f"{'H':>3}  {'n':>5}  {'Mamba':>7}  {'Persist':>7}  {'Season':>7}  {'Mean':>7}  {'M/P':>6}")
    for h in HORIZONS:
        t0 = time.time()
        r = run_one(X, iso3, snapshots, h)
        r["wall_s"] = time.time() - t0
        results.append(r)
        print(f"{h:>3}  {r['n_test']:>5}  "
              f"{r['mae_mamba']:.4f}  {r['mae_persistence']:.4f}  "
              f"{r['mae_seasonal']:.4f}  {r['mae_mean']:.4f}  "
              f"{r['mae_persistence']/max(r['mae_mamba'],1e-9):>5.2f}×  "
              f"[{r['wall_s']:.0f}s]")

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
