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
from aggregations import sectoral
from aggregations.temporal import build_temporal_frame
from aggregations.composites import compute_gap_scores, four_cell_typology

DATA = Path(__file__).resolve().parent.parent / "Data"

# Disk-cache location for the assembled enriched frame (see spec.yaml →
# persistence.enriched_frame). Writing is opt-in via save_enriched_frame();
# build_enriched_frame() prefers a fresh on-disk cache if `max_age_hours` is
# respected, otherwise re-derives.
ENRICHED_CACHE = Path(__file__).resolve().parent / "enriched.parquet"


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


# ─── Public long-form helpers for multi-row L1/L2 properties ──────────────
# The country-indexed enriched frame can only hold scalar columns. Spec
# properties with country_sector_year or country_donor_year granularity are
# exposed here instead, for per-country drill-downs in the profile view.
def load_sector_breakdown(year: int = 2025) -> pd.DataFrame:
    """Long-form country × sector frame.

    Surfaces L1 `pin_by_sector`, `requirements_by_sector`, `funding_by_sector`
    and L2 `coverage_by_sector` — properties declared in spec.yaml with
    country_sector_year granularity, which don't fit the country-indexed
    enriched frame.
    """
    return sectoral.build_sector_coverage(year=year)


def load_donor_breakdown(year: int = 2025) -> pd.DataFrame:
    """Long-form country × donor frame.

    Surfaces L1 `funding_by_donor` (spec.yaml granularity: country_donor_year).
    Mirrors the filter used for donor concentration: only rows whose
    FTS `destLocations` is a single ISO3, and positive amounts.

    Returns a DataFrame with columns:
        iso3, donor, funding_by_donor
    """
    inc = pd.read_csv(
        DATA / "fts" / "fts_incoming_funding_global.csv",
        skiprows=[1],
        low_memory=False,
    )
    inc["budgetYear"] = pd.to_numeric(inc["budgetYear"], errors="coerce")
    inc["amountUSD"] = pd.to_numeric(inc["amountUSD"], errors="coerce")
    sub = inc[(inc["budgetYear"] == year) & inc["destLocations"].notna()].copy()
    sub["destLocations"] = sub["destLocations"].astype(str).str.strip()
    single = sub[sub["destLocations"].str.len() == 3]
    single = single.dropna(subset=["amountUSD", "srcOrganization"])
    single = single[single["amountUSD"] > 0]
    per_donor = (
        single.rename(columns={"destLocations": "iso3", "srcOrganization": "donor", "amountUSD": "funding_by_donor"})
        .groupby(["iso3", "donor"], as_index=False)["funding_by_donor"]
        .sum()
        .sort_values(["iso3", "funding_by_donor"], ascending=[True, False])
    )
    return per_donor


# ─── Public API ─────────────────────────────────────────────────────────────
def build_enriched_frame(year: int = 2025) -> pd.DataFrame:
    """One row per country, indexed by ISO3, with all scalar L1-L5 properties.

    Level 1 scalars: pin_total, requirements, funding_received, population,
             severity_{category,index}, pin_phase_{1..5}, affected, displaced,
             fatalities, access_{limited,restricted}, impediments,
             cbpf_allocation
    Level 2 scalars: coverage, coverage_shortfall, per_pin_{gap,allocated},
             need_intensity, affected_intensity, displaced_intensity,
             phase_{45,5}_share, displaced_share_of_pin,
             access_{restricted,limited}_share, log_*
    Level 3: donor_{hhi,top1_share,top3_share,entropy}, n_donors,
             cluster_{gini,min_coverage,min_name}, phase_gini, cbpf_reliance
    Level 4: severity_{baseline_24m,acute_delta_3m,volatility_12m,trend_12m},
             phase_45_share_{baseline_12m,delta_3m},
             access_restricted_{baseline_12m,delta_3m}, persistence_P4_plus,
             coverage_{baseline_3y,trend_3y}, displaced_growth_12m
    Level 5: gap_score_{balanced,cerf,echo,usaid,ngo}_profile, median_rank,
             rank_iqr, completeness, typology_cell

    Multi-row properties (pin_by_sector, requirements_by_sector,
    funding_by_sector, coverage_by_sector, funding_by_donor) are available
    via load_sector_breakdown() and load_donor_breakdown().
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

    # Eligibility gate BEFORE Level-5 so pool-based normalisation is clean
    has_req = df["requirements"].fillna(0) > 0
    has_sev = df.get("severity_baseline_24m", pd.Series(dtype=float)).notna()
    df = df[has_req | has_sev]

    # ── Level 5 ──
    gap = compute_gap_scores(df)
    df = df.join(gap, how="left")

    # Four-cell typology — classification based on cluster_gini × rank_iqr
    df["typology_cell"] = four_cell_typology(df)

    # inform_severity_index is the same value as severity_index, re-exported
    # at Level 5 because INFORM's native composite is itself a composite
    # score. Aliased here so every L5 property resolves in the frame.
    if "severity_index" in df.columns:
        df["inform_severity_index"] = df["severity_index"]
    return df


# ─── Disk-cached enriched frame (optional, for fast cold starts) ──────────
def save_enriched_frame(year: int = 2025, path: Path | None = None) -> Path:
    """Materialise the enriched frame to parquet for fast cold starts.

    The parquet is a byte-for-byte snapshot of what build_enriched_frame()
    returns — the authoritative source of truth remains the code path.
    """
    path = path or ENRICHED_CACHE
    df = build_enriched_frame(year=year)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return path


def load_cached_enriched_frame(
    path: Path | None = None, max_age_hours: float = 24.0
) -> pd.DataFrame | None:
    """Read the on-disk enriched frame if it exists and is fresh.

    Returns None when the cache is missing or older than `max_age_hours`.
    """
    import time

    path = path or ENRICHED_CACHE
    if not path.exists():
        return None
    age_hours = (time.time() - path.stat().st_mtime) / 3600.0
    if age_hours > max_age_hours:
        return None
    return pd.read_parquet(path)


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
