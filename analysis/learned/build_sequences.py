"""Build the monthly INFORM sequence tensor used by the Mamba encoder.

Inputs
------
Data/Third-Party/DRMKC-INFORM/inform_severity_long.csv   · N × T severity
Data/Third-Party/DRMKC-INFORM/inform_indicators_long.csv · N × T sub-indicators

Output
------
Data/learned/sequences.npz   with fields
    X        (N, T, F_seq)   float32 — feature matrix
    iso3     (N,)              object — country codes
    snapshot (T,)              object — YYYY-MM labels
    features (F_seq,)          object — feature names

Design
------
- NO theta / completeness / any Bayesian quantities. This is the input side of
  the architecture — the Mamba encoder MUST NOT see its own downstream target.
- Seven time-varying features per month:
    severity (1-5), category (1-5 ordinal),
    affected (log1p), displaced (log1p), fatalities (log1p),
    pin_level_5_share, access_restricted_share
- NaN handling: forward-fill within country, then back-fill, then zero.
  The mask is preserved as `X[..., -1]` so the model can learn to down-weight
  imputed steps if it chooses to.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Data"
INFORM = DATA / "Third-Party" / "DRMKC-INFORM"
OUT = DATA / "learned"

FEATURES = [
    "severity",
    "category",
    "affected_log1p",
    "displaced_log1p",
    "fatalities_log1p",
    "pin_level_5_share",
    "access_restricted_share",
    "observed_mask",
]


def build() -> None:
    sev = pd.read_csv(INFORM / "inform_severity_long.csv")
    ind = pd.read_csv(INFORM / "inform_indicators_long.csv")

    # Canonicalise snapshot format
    for df in (sev, ind):
        df["snapshot"] = df["snapshot"].astype(str).str.slice(0, 7)

    # Aggregate sub-indicators to country-snapshot (sum across crisis IDs)
    ind_agg = (
        ind.groupby(["ISO3", "snapshot"], as_index=False)
           .agg(
               affected=("affected", "sum"),
               displaced=("displaced", "sum"),
               fatalities=("fatalities", "sum"),
               pin_level_5=("pin_level_5", "sum"),
               pin_level_4=("pin_level_4", "sum"),
               pin_level_3=("pin_level_3", "sum"),
               pin_level_2=("pin_level_2", "sum"),
               pin_level_1=("pin_level_1", "sum"),
               access_restricted=("access_restricted", "sum"),
           )
    )
    ind_agg["pin_total"] = ind_agg[[f"pin_level_{i}" for i in range(1, 6)]].sum(axis=1)
    ind_agg["pin_level_5_share"] = np.where(
        ind_agg["pin_total"] > 0,
        ind_agg["pin_level_5"] / ind_agg["pin_total"],
        0.0,
    )
    ind_agg["access_restricted_share"] = np.where(
        ind_agg["affected"] > 0,
        ind_agg["access_restricted"] / ind_agg["affected"],
        0.0,
    )
    ind_agg["access_restricted_share"] = ind_agg["access_restricted_share"].clip(0, 1)
    for col in ("affected", "displaced", "fatalities"):
        ind_agg[f"{col}_log1p"] = np.log1p(ind_agg[col].clip(lower=0))

    # Aggregate severity to country-snapshot (mean across crisis IDs in same country)
    sev_agg = (
        sev.groupby(["ISO3", "snapshot"], as_index=False)
           .agg(severity=("severity", "mean"), category=("category", "mean"))
    )

    merged = sev_agg.merge(ind_agg, on=["ISO3", "snapshot"], how="outer")

    # Universe: all countries × all snapshots ever seen
    iso_order = sorted(merged["ISO3"].dropna().unique())
    snapshots = sorted(merged["snapshot"].dropna().unique())
    N, T = len(iso_order), len(snapshots)

    iso_idx = {c: i for i, c in enumerate(iso_order)}
    snap_idx = {s: j for j, s in enumerate(snapshots)}

    F = len(FEATURES)
    X = np.zeros((N, T, F), dtype=np.float32)
    observed = np.zeros((N, T), dtype=np.bool_)

    feat_cols = [
        "severity", "category",
        "affected_log1p", "displaced_log1p", "fatalities_log1p",
        "pin_level_5_share", "access_restricted_share",
    ]
    for row in merged.itertuples(index=False):
        i = iso_idx.get(row.ISO3)
        j = snap_idx.get(row.snapshot)
        if i is None or j is None:
            continue
        for k, col in enumerate(feat_cols):
            v = getattr(row, col, np.nan)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                X[i, j, k] = float(v)
                observed[i, j] = True

    # Fill missing: forward fill within country along time, then back fill, then 0
    for i in range(N):
        for k in range(len(feat_cols)):
            seq = X[i, :, k]
            mask = observed[i]
            if not mask.any():
                continue
            # forward fill
            last = 0.0
            ok = False
            for j in range(T):
                if mask[j]:
                    last = seq[j]
                    ok = True
                else:
                    if ok:
                        seq[j] = last
            # back fill
            last = 0.0
            ok = False
            for j in range(T - 1, -1, -1):
                if mask[j]:
                    last = seq[j]
                    ok = True
                else:
                    if ok and not mask[j]:
                        seq[j] = last
            X[i, :, k] = seq

    # Observed mask as the last channel
    X[:, :, -1] = observed.astype(np.float32)

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "sequences.npz"
    np.savez_compressed(
        out_path,
        X=X,
        iso3=np.array(iso_order, dtype=object),
        snapshot=np.array(snapshots, dtype=object),
        features=np.array(FEATURES, dtype=object),
    )
    print(f"Wrote {out_path.relative_to(ROOT)}")
    print(f"  shape: {X.shape}   observed rate: {observed.mean():.3f}")
    print(f"  countries: {N}   snapshots: {T}  ({snapshots[0]} … {snapshots[-1]})")
    print(f"  features: {FEATURES}")


if __name__ == "__main__":
    build()
