"""Level-4 temporal aggregates — baselines, deltas, volatility, trends, persistence.

All windows are trailing from each country's latest observed snapshot. We use
the INFORM severity **category** (1–5 ordinal) rather than the continuous
index to side-step the Feb-2026 rescaling discontinuity. See
`proposal/metric_cards.md` §§1–2 for the rationale.

Produced columns per country (indexed by ISO3):

  From INFORM severity monthly panel:
    severity_baseline_24m, severity_acute_delta_3m,
    severity_volatility_12m, severity_trend_12m

  From INFORM sub-indicators monthly panel (derived shares):
    phase_45_share_baseline_12m, phase_45_share_delta_3m,
    access_restricted_baseline_12m, access_restricted_delta_3m,
    persistence_P4_plus, displaced_growth_12m

  From FTS multi-year panel:
    coverage_baseline_3y, coverage_trend_3y
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent.parent / "Data"

# ─── panel loaders ──────────────────────────────────────────────────────────


def _load_severity_panel() -> pd.DataFrame:
    inf = pd.read_csv(
        DATA / "Third-Party" / "DRMKC-INFORM" / "inform_severity_long.csv"
    )
    inf["date"] = pd.to_datetime(inf["snapshot"] + "-01")
    inf["category"] = pd.to_numeric(inf["category"], errors="coerce")
    return inf.dropna(subset=["category"]).sort_values(["ISO3", "date"])


def _load_indicator_panel() -> pd.DataFrame:
    ind = pd.read_csv(
        DATA / "Third-Party" / "DRMKC-INFORM" / "inform_indicators_long.csv"
    )
    ind["date"] = pd.to_datetime(ind["snapshot"] + "-01")
    for c in ("affected", "pin_level_4", "pin_level_5", "access_restricted", "displaced"):
        if c in ind.columns:
            ind[c] = pd.to_numeric(ind[c], errors="coerce")
    aff = ind["affected"].where(ind["affected"] > 0)
    ind["phase_45_share"] = (
        ind.get("pin_level_4", 0).fillna(0) + ind.get("pin_level_5", 0).fillna(0)
    ) / aff
    ind["access_restricted_share"] = ind.get("access_restricted", 0) / aff
    return ind.sort_values(["ISO3", "date"])


def _load_fts_annual_panel() -> pd.DataFrame:
    fts = pd.read_csv(
        DATA / "fts" / "fts_requirements_funding_global.csv", skiprows=[1]
    )
    fts["year"] = pd.to_numeric(fts["year"], errors="coerce")
    fts = fts.dropna(subset=["year", "countryCode"])
    fts = fts.groupby(["countryCode", "year"], as_index=False)[
        ["requirements", "funding"]
    ].sum()
    fts["coverage"] = (fts["funding"] / fts["requirements"]).clip(upper=1.5)
    return fts.rename(columns={"countryCode": "iso3"}).sort_values(["iso3", "year"])


# ─── aggregation helpers ────────────────────────────────────────────────────


def _trailing_median(arr: np.ndarray, k: int) -> float:
    sub = arr[-k:]
    sub = sub[~np.isnan(sub)]
    return float(np.median(sub)) if len(sub) else np.nan


def _trailing_std(arr: np.ndarray, k: int) -> float:
    sub = arr[-k:]
    sub = sub[~np.isnan(sub)]
    return float(np.std(sub, ddof=1)) if len(sub) >= 2 else np.nan


def _acute_delta(arr: np.ndarray, k: int) -> float:
    """Latest value minus median of the k values immediately before it."""
    if len(arr) <= k:
        return np.nan
    latest = arr[-1]
    if np.isnan(latest):
        return np.nan
    prev = arr[-(k + 1):-1]
    prev = prev[~np.isnan(prev)]
    if not len(prev):
        return np.nan
    return float(latest - np.median(prev))


def _ols_slope(arr: np.ndarray, k: int) -> float:
    sub = arr[-k:]
    mask = ~np.isnan(sub)
    if mask.sum() < 3 or np.std(sub[mask]) == 0:
        return np.nan
    x = np.arange(len(sub), dtype=float)
    return float(np.polyfit(x[mask], sub[mask], 1)[0])


def _longest_run(mask: np.ndarray) -> int:
    """Max consecutive True count in a 1D boolean array."""
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        if cur > best:
            best = cur
    return int(best)


# ─── public API ────────────────────────────────────────────────────────────


def severity_aggregates(
    baseline_k: int = 24, delta_k: int = 3, vol_k: int = 12, trend_k: int = 12
) -> pd.DataFrame:
    panel = _load_severity_panel()
    rows = []
    for iso3, grp in panel.groupby("ISO3"):
        cat = grp["category"].to_numpy(dtype=float)
        if len(cat) < 2:
            continue
        rows.append(
            {
                "iso3": iso3,
                "severity_baseline_24m": _trailing_median(cat, baseline_k),
                "severity_acute_delta_3m": _acute_delta(cat, delta_k),
                "severity_volatility_12m": _trailing_std(cat, vol_k),
                "severity_trend_12m": _ols_slope(cat, trend_k),
            }
        )
    return pd.DataFrame(rows).set_index("iso3")


def indicator_temporal_aggregates(
    baseline_k_m: int = 12, delta_k_m: int = 3, persistence_threshold: float = 0.15
) -> pd.DataFrame:
    """Aggregates over INFORM sub-indicator monthly panels."""
    panel = _load_indicator_panel()
    rows = []
    for iso3, grp in panel.groupby("ISO3"):
        grp = grp.sort_values("date")
        phase = grp["phase_45_share"].to_numpy(dtype=float)
        access = grp["access_restricted_share"].to_numpy(dtype=float)
        disp = grp["displaced"].to_numpy(dtype=float)

        row = {"iso3": iso3}
        row["phase_45_share_baseline_12m"] = _trailing_median(phase, baseline_k_m)
        row["phase_45_share_delta_3m"] = _acute_delta(phase, delta_k_m)
        row["access_restricted_baseline_12m"] = _trailing_median(access, baseline_k_m)
        row["access_restricted_delta_3m"] = _acute_delta(access, delta_k_m)

        # persistence — longest run of months over threshold (not restricted to trailing)
        with np.errstate(invalid="ignore"):
            mask = phase > persistence_threshold
        row["persistence_P4_plus"] = _longest_run(mask)

        # displaced growth = latest / value 12 months ago (forward-fill-safe)
        if len(disp) >= 13:
            latest = disp[-1]
            old = disp[-13]
            if pd.notna(latest) and pd.notna(old) and old > 0:
                row["displaced_growth_12m"] = float(latest / old)
            else:
                row["displaced_growth_12m"] = np.nan
        else:
            row["displaced_growth_12m"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("iso3")


def fts_temporal_aggregates(baseline_years: int = 3, trend_years: int = 3) -> pd.DataFrame:
    panel = _load_fts_annual_panel()
    rows = []
    for iso3, grp in panel.groupby("iso3"):
        cov = grp["coverage"].to_numpy(dtype=float)
        if len(cov) < 2:
            continue
        row = {"iso3": iso3}
        row["coverage_baseline_3y"] = _trailing_median(cov, baseline_years)
        row["coverage_trend_3y"] = _ols_slope(cov, trend_years)
        rows.append(row)
    return pd.DataFrame(rows).set_index("iso3")


def build_temporal_frame(
    baseline_month_k: int = 24,
    delta_month_k: int = 3,
    vol_month_k: int = 12,
    trend_month_k: int = 12,
    indicator_baseline_k: int = 12,
    indicator_delta_k: int = 3,
    persistence_threshold: float = 0.15,
    baseline_years: int = 3,
    trend_years: int = 3,
) -> pd.DataFrame:
    sev = severity_aggregates(
        baseline_k=baseline_month_k,
        delta_k=delta_month_k,
        vol_k=vol_month_k,
        trend_k=trend_month_k,
    )
    ind = indicator_temporal_aggregates(
        baseline_k_m=indicator_baseline_k,
        delta_k_m=indicator_delta_k,
        persistence_threshold=persistence_threshold,
    )
    fts = fts_temporal_aggregates(
        baseline_years=baseline_years, trend_years=trend_years
    )
    return sev.join(ind, how="outer").join(fts, how="outer")
