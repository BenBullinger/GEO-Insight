"""Train the monthly INFORM severity forecaster.

Target: 6-month-ahead severity value (absolute, 1–5 scale; mean of the next
six months per sliding window). Input: 24 months of the 8-channel monthly
INFORM panel. No theta, no Bayesian context.

Baselines reported on every run, because the right baseline for a slow-
moving ordinal is not 'mean of train':
    persistence : y_hat = severity[t-1]            — last observed value
    seasonal    : y_hat = severity[t-12] if obs    — same month, last year
    mean        : y_hat = train target mean        — weakest sanity baseline

Uplift is claimed only against the strongest applicable baseline.
Per-country MAE is broken down by stability class (flat / changing) so
the narrative is truthful about where the encoder helps.

Outputs
-------
Data/learned/severity_momentum.parquet
    per-country inference: iso3, predicted_severity_6m, persistence_pred,
    current_severity, end_snapshot, n_observed_steps
Data/learned/training_metrics.json
    forecast metrics for the test set (Mamba + baselines, overall + class)
analysis/learned/momentum_model.pt   — model weights
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Data"
SEQ_NPZ = DATA / "learned" / "sequences.npz"
OUT_PARQUET = DATA / "learned" / "severity_momentum.parquet"
OUT_METRICS = DATA / "learned" / "training_metrics.json"
OUT_PRED_TABLE = DATA / "learned" / "forecast_test_predictions.parquet"
OUT_WEIGHTS = Path(__file__).resolve().parent / "momentum_model.pt"

SEED = 0
WINDOW = 24
HORIZON = 12  # persistence breaks down past ~9 months; headline horizon is 12
HIDDEN = 32
STATE = 16
N_BLOCKS = 2
BATCH = 64
EPOCHS = 120
LR = 3e-3
PATIENCE = 15
TRAIN_CUTOFF = "2025-01"
STABILITY_THRESHOLD = 0.5  # severity std over trailing 12 months distinguishes flat vs changing


def _seed(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


# ════════════════════════════════════════════════════════════════════════════
# Selective-SSM block (CPU-friendly pure PyTorch)
# ════════════════════════════════════════════════════════════════════════════
class SelectiveSSM(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        d_inner = d_model * expand
        self.d_inner = d_inner
        self.d_state = d_state
        dt_rank = max(1, d_model // 16)
        self.dt_rank = dt_rank
        self.in_proj = nn.Linear(d_model, d_inner * 2 + d_state * 2 + dt_rank, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        nn.init.uniform_(self.dt_proj.bias, -4.0, -1.0)
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32), "n -> h n", h=d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv - 1, groups=d_inner, bias=True)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def _scan(self, u, delta, A, B, C, D):
        Bsz, L, d = u.shape; N = A.shape[1]
        dA = torch.exp(rearrange(delta, "b l d -> b l d 1") * rearrange(A, "d n -> 1 1 d n"))
        dBu = (rearrange(delta, "b l d -> b l d 1")
               * rearrange(B, "b l n -> b l 1 n")
               * rearrange(u, "b l d -> b l d 1"))
        h = torch.zeros(Bsz, d, N, device=u.device, dtype=u.dtype)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dBu[:, t]
            ys.append((h * rearrange(C[:, t], "b n -> b 1 n")).sum(-1))
        return torch.stack(ys, 1) + u * D.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        residual = x; B, L, _ = x.shape
        proj = self.in_proj(x)
        x_s, z, B_s, C_s, dt = torch.split(
            proj, [self.d_inner, self.d_inner, self.d_state, self.d_state, self.dt_rank], dim=-1)
        x_s = F.silu(self.conv1d(rearrange(x_s, "b l d -> b d l"))[..., :L])
        x_s = rearrange(x_s, "b d l -> b l d")
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log.float())
        y = self._scan(x_s, dt, A, B_s, C_s, self.D)
        y = y * F.silu(z)
        return self.norm(residual + self.out_proj(y))


class ForecastEncoder(nn.Module):
    """Reads the 8-channel monthly panel, predicts 6-month-ahead severity value."""

    def __init__(self, n_features: int, hidden: int = HIDDEN, n_blocks: int = N_BLOCKS, d_state: int = STATE):
        super().__init__()
        self.embed = nn.Linear(n_features, hidden)
        self.blocks = nn.ModuleList([SelectiveSSM(hidden, d_state=d_state) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h)
        pooled = h[:, -1, :]
        return self.head(pooled).squeeze(-1)


# ════════════════════════════════════════════════════════════════════════════
# Windows: target is the ABSOLUTE 6-month-ahead severity value
# ════════════════════════════════════════════════════════════════════════════
class WindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def build_windows(X, iso3, snapshots):
    """Yield (input_window, target) pairs.

    For each t with an observed step at t-1 and at least half of the
    HORIZON future steps observed:
        input  = X[:, t-WINDOW:t, :]
        target = mean of sev[t : t+HORIZON]       (absolute, 1-5 scale)
    Metadata carried per row: iso3, end_snapshot, current_sev (t-1),
    last-year severity (t-12), stability class (flat/changing),
    persistence prediction, seasonal prediction.
    """
    N, T, F_ = X.shape
    sev = X[:, :, 0]; obs = X[:, :, -1]

    wins_X, wins_y, meta_rows = [], [], []
    for i in range(N):
        for t in range(WINDOW, T - HORIZON):
            if obs[i, t - 1] < 0.5:
                continue
            if obs[i, t : t + HORIZON].mean() < 0.5:
                continue
            target = float(sev[i, t : t + HORIZON].mean())
            cur = float(sev[i, t - 1])
            seasonal = float(sev[i, t - 13]) if t >= 13 and obs[i, t - 13] >= 0.5 else cur
            trail_std = float(sev[i, max(0, t - 12) : t].std())
            wins_X.append(X[i, t - WINDOW : t, :])
            wins_y.append(target)
            meta_rows.append({
                "iso3": str(iso3[i]),
                "end_snapshot": str(snapshots[t - 1]),
                "current_sev": cur,
                "seasonal_sev": seasonal,
                "trailing_std_12m": trail_std,
                "stability": "flat" if trail_std < STABILITY_THRESHOLD else "changing",
            })
    return (
        np.stack(wins_X).astype(np.float32),
        np.array(wins_y, dtype=np.float32),
        pd.DataFrame(meta_rows),
    )


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
def main():
    _seed(SEED)
    z = np.load(SEQ_NPZ, allow_pickle=True)
    X = z["X"]; iso3 = z["iso3"]; snapshots = z["snapshot"]
    F_seq = X.shape[-1]

    X_windows, y_all, meta = build_windows(X, iso3, snapshots)
    train_idx = np.where(meta["end_snapshot"] < TRAIN_CUTOFF)[0]
    test_idx = np.where(meta["end_snapshot"] >= TRAIN_CUTOFF)[0]

    # Z-score features using train-only stats. Keep the observed-mask channel raw.
    mean = X_windows[train_idx].reshape(-1, F_seq).mean(0)
    std = X_windows[train_idx].reshape(-1, F_seq).std(0) + 1e-6
    mean[-1] = 0.0; std[-1] = 1.0
    X_norm = (X_windows - mean) / std

    y_mean = float(y_all[train_idx].mean())
    y_std = float(y_all[train_idx].std() + 1e-6)
    y_norm = (y_all - y_mean) / y_std

    print(f"Windows  train={len(train_idx)}  test={len(test_idx)}")
    print(f"Target   train mean={y_mean:.3f}  std={y_std:.3f}  (severity 1-5 scale)")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = ForecastEncoder(n_features=F_seq).to(device)
    print(f"Params   {sum(p.numel() for p in model.parameters()):,}   device={device}")

    train_loader = DataLoader(WindowDataset(X_norm[train_idx], y_norm[train_idx]), BATCH, shuffle=True)
    test_loader  = DataLoader(WindowDataset(X_norm[test_idx],  y_norm[test_idx]),  BATCH, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=3e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, epochs=EPOCHS, steps_per_epoch=len(train_loader), pct_start=0.15)

    best = float("inf"); best_state = None; patience_ct = 0
    for epoch in range(EPOCHS):
        model.train(); tloss = 0.0; nb = 0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            pred = model(xb)
            loss = F.huber_loss(pred, yb, delta=0.5)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step(); sched.step()
            tloss += loss.item(); nb += 1

        model.eval(); vloss = 0.0; vb = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                pred = model(xb.to(device))
                vloss += F.huber_loss(pred, yb.to(device), delta=0.5).item(); vb += 1
        vloss /= max(vb, 1); tloss /= max(nb, 1)

        improved = vloss < best - 1e-4
        if improved:
            best = vloss; best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ct = 0
        else:
            patience_ct += 1
        if epoch % 5 == 0 or improved:
            tag = " *" if improved else ""
            print(f"ep {epoch:3d}  train={tloss:.4f}  test={vloss:.4f}{tag}")
        if patience_ct >= PATIENCE:
            print(f"early stop at epoch {epoch}"); break
    model.load_state_dict(best_state)

    # ────────────────────────────────────────────────────────────────────────
    # Evaluation on the held-out set
    # ────────────────────────────────────────────────────────────────────────
    model.eval(); preds = []
    with torch.no_grad():
        for xb, _ in DataLoader(WindowDataset(X_norm[test_idx], y_norm[test_idx]), BATCH, shuffle=False):
            preds.extend(model(xb.to(device)).cpu().numpy().tolist())
    pred_sev = np.array(preds, dtype=np.float32) * y_std + y_mean
    true_sev = y_all[test_idx]
    meta_test = meta.iloc[test_idx].reset_index(drop=True).copy()
    meta_test["y_true"] = true_sev
    meta_test["y_mamba"] = pred_sev
    meta_test["y_persistence"] = meta_test["current_sev"].values
    meta_test["y_seasonal"] = meta_test["seasonal_sev"].values
    meta_test["y_mean"] = y_mean

    def _mae(a, b): return float(np.mean(np.abs(a - b)))
    mae_mamba   = _mae(meta_test["y_mamba"].values, true_sev)
    mae_persist = _mae(meta_test["y_persistence"].values, true_sev)
    mae_seasonal= _mae(meta_test["y_seasonal"].values, true_sev)
    mae_mean    = _mae(meta_test["y_mean"].values, true_sev)

    print(f"\n── Held-out {HORIZON}-month-ahead severity forecast (1–5 scale) ──")
    print(f"  Mamba MAE                : {mae_mamba:.4f}")
    print(f"  Persistence baseline MAE : {mae_persist:.4f}  (y = severity[t-1])")
    print(f"  Seasonal baseline MAE    : {mae_seasonal:.4f}  (y = severity[t-12])")
    print(f"  Mean baseline MAE        : {mae_mean:.4f}  (y = train mean)")
    strongest = min(mae_persist, mae_seasonal)
    print(f"  Mamba vs strongest       : {strongest / max(mae_mamba, 1e-9):.3f}×  "
          f"({'persistence' if mae_persist <= mae_seasonal else 'seasonal'})")

    # By stability class
    print("\n── Per-stability-class MAE ──")
    class_metrics = {}
    for cls, sub in meta_test.groupby("stability"):
        r = {
            "n": int(len(sub)),
            "mae_mamba":   _mae(sub["y_mamba"].values, sub["y_true"].values),
            "mae_persist": _mae(sub["y_persistence"].values, sub["y_true"].values),
            "mae_seasonal":_mae(sub["y_seasonal"].values, sub["y_true"].values),
        }
        class_metrics[cls] = r
        print(f"  {cls:<10} n={r['n']:<4} mamba={r['mae_mamba']:.4f}  "
              f"persist={r['mae_persist']:.4f}  seasonal={r['mae_seasonal']:.4f}")

    meta_test.to_parquet(OUT_PRED_TABLE)

    # ────────────────────────────────────────────────────────────────────────
    # Per-country scalar: latest 24-month window → 6-month-ahead forecast
    # ────────────────────────────────────────────────────────────────────────
    N, T, _ = X.shape
    sev = X[:, :, 0]; obs = X[:, :, -1]
    rows = []
    for i in range(N):
        t_last = None
        for t in range(T, WINDOW - 1, -1):
            if 0 <= t - 1 < T and obs[i, t - 1] >= 0.5:
                t_last = t; break
        if t_last is None:
            rows.append({
                "iso3": str(iso3[i]),
                "severity_momentum_learned": np.nan,
                "persistence_pred": np.nan,
                "current_severity": np.nan,
                "end_snapshot": None,
                "n_observed_steps": 0,
            })
            continue
        win = X[i, t_last - WINDOW : t_last, :].astype(np.float32)
        win_n = (win - mean) / std
        with torch.no_grad():
            xb = torch.tensor(win_n, dtype=torch.float32, device=device).unsqueeze(0)
            p = float(model(xb).cpu().numpy()[0]) * y_std + y_mean
        rows.append({
            "iso3": str(iso3[i]),
            "severity_momentum_learned": p,
            "persistence_pred": float(sev[i, t_last - 1]),
            "current_severity": float(sev[i, t_last - 1]),
            "end_snapshot": str(snapshots[t_last - 1]),
            "n_observed_steps": int(obs[i].sum()),
        })
    df = pd.DataFrame(rows).set_index("iso3")
    df.to_parquet(OUT_PARQUET)
    print(f"\nWrote {OUT_PARQUET.relative_to(ROOT)}  ({len(df)} countries)")
    print(df.head(10).round(3))

    metrics = {
        "task": f"{HORIZON}-month-ahead severity value (absolute, 1-5 scale)",
        "window": WINDOW,
        "horizon": HORIZON,
        "train_cutoff": TRAIN_CUTOFF,
        "n_train_windows": int(len(train_idx)),
        "n_test_windows": int(len(test_idx)),
        "params": int(sum(p.numel() for p in model.parameters())),
        "mae_mamba":     mae_mamba,
        "mae_persistence": mae_persist,
        "mae_seasonal":  mae_seasonal,
        "mae_mean":      mae_mean,
        "uplift_vs_persistence": float(mae_persist / max(mae_mamba, 1e-9)),
        "uplift_vs_seasonal":    float(mae_seasonal / max(mae_mamba, 1e-9)),
        "per_class": class_metrics,
    }
    OUT_METRICS.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {OUT_METRICS.relative_to(ROOT)}")

    torch.save(model.state_dict(), OUT_WEIGHTS)
    print(f"Wrote {OUT_WEIGHTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
