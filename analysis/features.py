"""Feature construction for the unsupervised-analysis app.

Builds an "enriched frame" — one row per country, columns at Levels 1-3
(Phase 1), later extended to Level 4 (temporal aggregates) and Level 5
(composite gap scores) in Phases 2-3.

Every column name in the enriched frame has a matching entry in
analysis/spec.yaml so that provenance can be looked up at display time.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from aggregations.concentration import (
    donor_concentration,
    cluster_inequality,
    phase_gini_latest,
    cbpf_reliance_latest,
)
from aggregations.temporal import build_temporal_frame

DATA = Path(__file__).resolve().parent.parent / "Data"


# ─── Level-1 loaders (cached as DataFrames, not Series, so joins are cleaner)
def _load_fts_country(year: int) -> pd.DataFrame:
    fts = pd.read_csv(DATA / "fts" / "fts_requirements_funding_global.csv", skiprows=[1])
    fts["year"] = pd.to_numeric(fts["year"], errors="coerce")
    sub = fts[fts["year"] == year][["countryCode", "requirements", "funding"]].dropna()
    sub = sub.groupby("countryCode", as_index=False).sum(numeric_only=True)
    return sub.rename(columns={"countryCode": "iso3"}).set_index("iso3")


def _load_hno_pin(year: int) -> pd.Series:
    hno = pd.read_csv(
        DATA / "hno" / f"hpc_hno_{year}.csv", skiprows=[1], low_memory=False
    )
    hno = hno[hno["Country ISO3"].notna()]
    country_lvl = hno[hno["Admin 1 PCode"].isna() & hno["Admin 2 PCode"].isna()]
    if country_lvl.empty:
        country_lvl = hno
    pin = country_lvl.groupby("Country ISO3")["In Need"].sum()
    pin.index.name = "iso3"
    pin.name = "pin_total"
    return pin


def _load_population() -> pd.Series:
    pop = pd.read_csv(
        DATA / "cod-ps" / "cod_population_admin0.csv", skiprows=[1], low_memory=False
    )
    pop["Gender"] = pop["Gender"].astype(str).str.lower()
    pop["Age_range"] = pop["Age_range"].astype(str).str.lower()
    all_rows = pop[
        pop["Gender"].isin({"all", "total", "t"})
        & pop["Age_range"].isin({"all", "total", "t"})
    ]
    if all_rows.empty:
        all_rows = pop
    out = all_rows.groupby("ISO3")["Population"].sum()
    out.index.name = "iso3"
    out.name = "population"
    return out


def _load_inform_severity_latest() -> pd.DataFrame:
    inf = pd.read_csv(
        DATA / "Third-Party" / "DRMKC-INFORM" / "inform_severity_long.csv"
    )
    inf = inf.sort_values(["ISO3", "snapshot"])
    latest = inf.groupby("ISO3").tail(1).set_index("ISO3")[["category", "severity"]]
    latest.index.name = "iso3"
    return latest.rename(
        columns={"category": "severity_category", "severity": "severity_index"}
    )


def _load_inform_indicators_latest() -> pd.DataFrame:
    ind = pd.read_csv(
        DATA / "Third-Party" / "DRMKC-INFORM" / "inform_indicators_long.csv"
    )
    ind = ind.sort_values(["ISO3", "snapshot"])
    latest = ind.groupby("ISO3").tail(1).set_index("ISO3")
    keep = {
        "affected": "affected",
        "displaced": "displaced",
        "injured": "injured",
        "fatalities": "fatalities",
        "pin_level_1": "pin_phase_1",
        "pin_level_2": "pin_phase_2",
        "pin_level_3": "pin_phase_3",
        "pin_level_4": "pin_phase_4",
        "pin_level_5": "pin_phase_5",
        "access_limited": "access_limited",
        "access_restricted": "access_restricted",
        "impediments_bureaucratic": "impediments",
    }
    latest = latest[[c for c in keep if c in latest.columns]].rename(columns=keep)
    latest.index.name = "iso3"
    return latest


# ─── Level-1 + Level-2 assembly ─────────────────────────────────────────────
def _assemble_level_1_and_2(year: int) -> pd.DataFrame:
    fts = _load_fts_country(year)
    pin = _load_hno_pin(year)
    pop = _load_population()
    sev = _load_inform_severity_latest()
    ind = _load_inform_indicators_latest()

    df = pd.concat([fts, pin, pop, sev, ind], axis=1)

    # ── Level 2 — ratios ──
    df["coverage"] = (df["funding"] / df["requirements"]).clip(upper=1.5)
    df["coverage_shortfall"] = (1 - df["coverage"].clip(upper=1)).clip(lower=0)
    df["per_pin_gap"] = (
        (df["requirements"] - df["funding"]).clip(lower=0) / df["pin_total"].replace({0: np.nan})
    )
    df["per_pin_allocated"] = df["funding"] / df["pin_total"].replace({0: np.nan})
    df["need_intensity"] = df["pin_total"] / df["population"].replace({0: np.nan})
    df["affected_intensity"] = df.get("affected", 0) / df["population"].replace({0: np.nan})
    df["displaced_intensity"] = df.get("displaced", 0) / df["population"].replace({0: np.nan})

    affected_den = df["affected"].where(df["affected"] > 0)
    p45 = df.get("pin_phase_4", 0).fillna(0) + df.get("pin_phase_5", 0).fillna(0)
    df["phase_45_share"] = p45 / affected_den
    df["phase_5_share"] = df.get("pin_phase_5", 0) / affected_den

    pin_den = df["pin_total"].where(df["pin_total"] > 0)
    df["displaced_share_of_pin"] = df.get("displaced", 0) / pin_den
    df["access_restricted_share"] = df.get("access_restricted", 0) / pin_den
    df["access_limited_share"] = df.get("access_limited", 0) / pin_den

    # Log transforms (Level 2 monotone)
    df["log_pin_total"] = np.log1p(df["pin_total"].fillna(0))
    df["log_requirements"] = np.log1p(df["requirements"].fillna(0))
    df["log_affected"] = np.log1p(df.get("affected", 0).fillna(0))
    df["log_displaced"] = np.log1p(df.get("displaced", 0).fillna(0))
    df["log_fatalities"] = np.log1p(df.get("fatalities", 0).fillna(0))

    # Rename FTS raw columns to match spec.yaml property names
    df = df.rename(columns={"funding": "funding_received"})
    return df


# ─── Public API ─────────────────────────────────────────────────────────────
def build_enriched_frame(year: int = 2025) -> pd.DataFrame:
    """One row per country, indexed by ISO3, Levels 1-3.

    Level 1: pin_total, requirements, funding_received, population,
             severity_{category,index}, pin_phase_{1..5}, affected, displaced,
             fatalities, injured, access_{limited,restricted}, impediments
    Level 2: coverage, coverage_shortfall, per_pin_{gap,allocated},
             need_intensity, affected_intensity, displaced_intensity,
             phase_{45,5}_share, displaced_share_of_pin,
             access_{restricted,limited}_share, log_*
    Level 3: donor_{hhi,top1_share,top3_share,entropy}, n_donors,
             cluster_{gini,min_coverage,min_name}
    """
    df = _assemble_level_1_and_2(year)

    # ── Level 3 ──
    donors = donor_concentration(year=year)
    clusters = cluster_inequality(year=year)
    phase_gini = phase_gini_latest()
    cbpf = cbpf_reliance_latest(year=year)
    df = (
        df.join(donors, how="left")
        .join(clusters, how="left")
        .join(phase_gini, how="left")
        .join(cbpf, how="left")
    )

    # ── Level 4 ──
    temporal = build_temporal_frame()
    df = df.join(temporal, how="left")

    # Eligibility: need at least requirements>0 OR a severity reading on the panel
    has_req = df["requirements"].fillna(0) > 0
    has_sev = df.get("severity_baseline_24m", pd.Series(dtype=float)).notna()
    df = df[has_req | has_sev]
    return df


# ─── Trajectory matrices (used in Phase 2 temporal lenses) ─────────────────
def build_trajectory_matrix(
    value: str = "category", min_snapshots: int = 50
) -> pd.DataFrame:
    inf = pd.read_csv(DATA / "Third-Party" / "DRMKC-INFORM" / "inform_severity_long.csv")
    traj = inf.pivot_table(
        index="ISO3", columns="snapshot", values=value, aggfunc="mean"
    ).sort_index(axis=1)
    traj = traj.loc[traj.notna().sum(axis=1) >= min_snapshots]
    traj = traj.T.interpolate(method="linear", limit_direction="both").T
    traj = traj.T.fillna(traj.mean(axis=1)).T
    return traj


def build_indicator_trajectory_matrix(
    indicator: str, min_snapshots: int = 50
) -> pd.DataFrame:
    ind = pd.read_csv(
        DATA / "Third-Party" / "DRMKC-INFORM" / "inform_indicators_long.csv"
    )
    if indicator not in ind.columns:
        raise ValueError(f"indicator {indicator!r} not in INFORM indicator panel")
    traj = ind.pivot_table(
        index="ISO3", columns="snapshot", values=indicator, aggfunc="mean"
    ).sort_index(axis=1)
    traj = traj.loc[traj.notna().sum(axis=1) >= min_snapshots]
    traj = traj.T.interpolate(method="linear", limit_direction="both").T
    traj = traj.T.fillna(traj.mean(axis=1)).T
    return traj
