"""Feature matrix construction for the unsupervised-analysis app.

Joins the five primary OCHA sources with the INFORM Severity panel + sub-
indicators into one per-country feature vector for the latest available year.

Also provides a trajectory matrix (country × month) used for temporal
archetype clustering.

Everything is cached via @st.cache_data in app.py; this module is pure data
manipulation, no streamlit deps.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "Data"


def _load_fts(year: int) -> pd.DataFrame:
    fts = pd.read_csv(DATA / "fts" / "fts_requirements_funding_global.csv", skiprows=[1])
    fts["year"] = pd.to_numeric(fts["year"], errors="coerce")
    sub = fts[fts["year"] == year].copy()
    sub = sub[["countryCode", "requirements", "funding"]].dropna()
    # aggregate across plan types (country-year may have multiple rows)
    sub = sub.groupby("countryCode", as_index=False).sum(numeric_only=True)
    sub["coverage"] = (sub["funding"] / sub["requirements"]).clip(upper=1.5)
    sub["coverage_shortfall"] = (1 - sub["coverage"].clip(upper=1)).clip(lower=0)
    sub["log_requirements"] = np.log1p(sub["requirements"])
    return sub.rename(columns={"countryCode": "iso3"}).set_index("iso3")


def _load_hno_pin(year: int) -> pd.Series:
    hno = pd.read_csv(
        DATA / "hno" / f"hpc_hno_{year}.csv", skiprows=[1], low_memory=False
    )
    hno = hno[hno["Country ISO3"].notna()]
    country_level = hno[hno["Admin 1 PCode"].isna() & hno["Admin 2 PCode"].isna()]
    if country_level.empty:
        country_level = hno
    pin = country_level.groupby("Country ISO3")["In Need"].sum()
    pin.index.name = "iso3"
    pin.name = "pin"
    return pin


def _load_population() -> pd.Series:
    pop = pd.read_csv(
        DATA / "cod-ps" / "cod_population_admin0.csv", skiprows=[1], low_memory=False
    )
    pop["Gender"] = pop["Gender"].astype(str).str.lower()
    pop["Age_range"] = pop["Age_range"].astype(str).str.lower()
    both = pop[
        pop["Gender"].isin({"all", "total", "t"})
        & pop["Age_range"].isin({"all", "total", "t"})
    ]
    if both.empty:
        both = pop
    out = both.groupby("ISO3")["Population"].sum()
    out.index.name = "iso3"
    out.name = "population"
    return out


def _load_inform_latest_severity() -> pd.DataFrame:
    inf = pd.read_csv(DATA / "Third-Party" / "DRMKC-INFORM" / "inform_severity_long.csv")
    # latest snapshot per country (some countries drop out in the latest snapshot)
    inf = inf.sort_values(["ISO3", "snapshot"])
    latest = inf.groupby("ISO3").tail(1).set_index("ISO3")[["category", "severity"]]
    latest.index.name = "iso3"
    return latest


def _load_inform_latest_indicators() -> pd.DataFrame:
    ind = pd.read_csv(DATA / "Third-Party" / "DRMKC-INFORM" / "inform_indicators_long.csv")
    ind = ind.sort_values(["ISO3", "snapshot"])
    latest = ind.groupby("ISO3").tail(1).set_index("ISO3")
    keep = [
        "affected", "displaced", "injured", "fatalities",
        "pin_level_4", "pin_level_5",
        "access_limited", "access_restricted",
    ]
    latest = latest[[c for c in keep if c in latest.columns]]
    latest.index.name = "iso3"
    return latest


def build_feature_matrix(year: int = 2025) -> pd.DataFrame:
    """Per-country feature matrix, latest snapshot per source.

    Returns a wide DataFrame indexed by ISO3. Features are in roughly comparable
    scales where possible (log transforms, ratios) but still need standardisation
    before PCA / k-means — the caller handles that.
    """
    fts = _load_fts(year)
    pin = _load_hno_pin(year)
    pop = _load_population()
    sev = _load_inform_latest_severity()
    ind = _load_inform_latest_indicators()

    df = pd.concat([fts, pin, pop, sev, ind], axis=1)

    # Derived ratios
    df["need_intensity"] = df["pin"] / df["population"]
    denom_affected = df["affected"].where(df["affected"] > 0)
    df["phase_45_share"] = (
        (df.get("pin_level_4", 0).fillna(0) + df.get("pin_level_5", 0).fillna(0))
        / denom_affected
    )
    pin_denom = df["pin"].where(df["pin"] > 0)
    df["displaced_share"] = df.get("displaced", 0) / pin_denom
    df["access_restricted_share"] = df.get("access_restricted", 0) / pin_denom
    df["log_fatalities"] = np.log1p(df.get("fatalities", 0).fillna(0))
    df["log_displaced"] = np.log1p(df.get("displaced", 0).fillna(0))

    feature_cols = [
        "coverage_shortfall",
        "log_requirements",
        "need_intensity",
        "category",
        "severity",
        "phase_45_share",
        "displaced_share",
        "access_restricted_share",
        "log_fatalities",
        "log_displaced",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols]

    # Keep rows with at least 60% feature coverage, then fill the rest by median
    keep_mask = X.notna().mean(axis=1) >= 0.6
    X = X.loc[keep_mask]
    X = X.fillna(X.median(numeric_only=True))
    return X


def build_trajectory_matrix(
    value: str = "category", min_snapshots: int = 50
) -> pd.DataFrame:
    """Country × snapshot matrix of a chosen INFORM field.

    `value` ∈ {"category", "severity"}. Countries with fewer than min_snapshots
    observed months are dropped. Remaining gaps are linearly interpolated.
    """
    inf = pd.read_csv(DATA / "Third-Party" / "DRMKC-INFORM" / "inform_severity_long.csv")
    traj = (
        inf.pivot_table(
            index="ISO3", columns="snapshot", values=value, aggfunc="mean"
        )
        .sort_index(axis=1)
    )
    traj = traj.loc[traj.notna().sum(axis=1) >= min_snapshots]
    # interpolate across time, then fill remaining with row mean
    traj = traj.T.interpolate(method="linear", limit_direction="both").T
    traj = traj.T.fillna(traj.mean(axis=1)).T
    return traj


def build_indicator_trajectory_matrix(indicator: str, min_snapshots: int = 50) -> pd.DataFrame:
    """Country × snapshot matrix for an INFORM sub-indicator."""
    ind = pd.read_csv(DATA / "Third-Party" / "DRMKC-INFORM" / "inform_indicators_long.csv")
    if indicator not in ind.columns:
        raise ValueError(f"indicator {indicator!r} not in INFORM indicator panel")
    traj = (
        ind.pivot_table(
            index="ISO3", columns="snapshot", values=indicator, aggfunc="mean"
        )
        .sort_index(axis=1)
    )
    traj = traj.loc[traj.notna().sum(axis=1) >= min_snapshots]
    traj = traj.T.interpolate(method="linear", limit_direction="both").T
    traj = traj.T.fillna(traj.mean(axis=1)).T
    return traj
