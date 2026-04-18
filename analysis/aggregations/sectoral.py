"""Country × sector joins for cluster-level coverage computation.

HNO and FTS cluster names drift (e.g. "Sanitation & Hygiene" vs "Water
Sanitation and Hygiene"). We normalise with a simple rule (lowercase +
collapse non-alnum → underscore) before joining. This is good enough for
Phase 1; a curated map can come later in src/taxonomies/cluster_map.csv.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent.parent / "Data"


def _norm_cluster(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def build_sector_coverage(year: int = 2025) -> pd.DataFrame:
    """Per-country × sector long-form frame joining FTS cluster funding/requirements with HNO PIN.

    Surfaces the four multi-row L1/L2 properties from spec.yaml that don't fit
    the country-indexed enriched frame:

        pin_by_sector            (L1, HNO)
        requirements_by_sector   (L1, FTS cluster)
        funding_by_sector        (L1, FTS cluster)
        coverage_by_sector       (L2, derived = funding / requirements per sector)

    Returns a DataFrame with columns:
        iso3, cluster, cluster_norm,
        requirements_by_sector, funding_by_sector,
        pin_by_sector, coverage_by_sector
    """
    fts = pd.read_csv(
        DATA / "fts" / "fts_requirements_funding_cluster_global.csv", skiprows=[1]
    )
    fts["year"] = pd.to_numeric(fts["year"], errors="coerce")
    fts_y = fts[fts["year"] == year][
        ["countryCode", "cluster", "requirements", "funding"]
    ].dropna(subset=["countryCode", "cluster"])
    fts_y = fts_y.rename(
        columns={
            "countryCode": "iso3",
            "requirements": "requirements_by_sector",
            "funding": "funding_by_sector",
        }
    ).copy()
    fts_y["cluster_norm"] = fts_y["cluster"].astype(str).apply(_norm_cluster)

    hno = pd.read_csv(
        DATA / "hno" / f"hpc_hno_{year}.csv", skiprows=[1], low_memory=False
    )
    hno = hno[hno["Country ISO3"].notna()]
    country_lvl = hno[hno["Admin 1 PCode"].isna() & hno["Admin 2 PCode"].isna()]
    if country_lvl.empty:
        country_lvl = hno
    pin = (
        country_lvl.groupby(["Country ISO3", "Cluster"], as_index=False)["In Need"]
        .sum()
        .rename(
            columns={
                "Country ISO3": "iso3",
                "Cluster": "cluster_hno",
                "In Need": "pin_by_sector",
            }
        )
    )
    pin["cluster_norm"] = pin["cluster_hno"].astype(str).apply(_norm_cluster)

    merged = pd.merge(
        fts_y[["iso3", "cluster", "cluster_norm", "requirements_by_sector", "funding_by_sector"]],
        pin[["iso3", "cluster_norm", "pin_by_sector"]],
        on=["iso3", "cluster_norm"],
        how="outer",
    )
    # Fill missing sector names where the join came only from the HNO side.
    missing_cluster = merged["cluster"].isna()
    if missing_cluster.any():
        merged.loc[missing_cluster, "cluster"] = merged.loc[missing_cluster, "cluster_norm"].str.replace("_", " ").str.title()
    merged["coverage_by_sector"] = (
        merged["funding_by_sector"] / merged["requirements_by_sector"]
    ).clip(upper=1.5)
    return merged
