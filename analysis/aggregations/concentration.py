"""Level-3 concentration and inequality properties.

- Donor concentration (HHI, top-1 share, top-3 share, n_donors, entropy)
  computed from FTS incoming flows, attributing multi-country rows only
  when destLocations is a single ISO3 (conservative; others excluded).
- Cluster coverage inequality (cluster_gini, min-coverage and starved
  cluster name) computed from the sector-coverage join in sectoral.py,
  weighted by PIN where available.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import sectoral

DATA = Path(__file__).resolve().parent.parent.parent / "Data"


# ─── Donor concentration ────────────────────────────────────────────────────
def donor_concentration(year: int = 2025) -> pd.DataFrame:
    inc = pd.read_csv(
        DATA / "fts" / "fts_incoming_funding_global.csv",
        skiprows=[1],
        low_memory=False,
    )
    inc["budgetYear"] = pd.to_numeric(inc["budgetYear"], errors="coerce")
    inc["amountUSD"] = pd.to_numeric(inc["amountUSD"], errors="coerce")
    sub = inc[(inc["budgetYear"] == year) & inc["destLocations"].notna()].copy()
    sub["destLocations"] = sub["destLocations"].astype(str).str.strip()

    # Only take rows where destLocations is a single ISO3 (conservative).
    single = sub[sub["destLocations"].str.len() == 3]
    single = single.dropna(subset=["amountUSD", "srcOrganization"])
    single = single[single["amountUSD"] > 0]

    per_donor = (
        single.rename(columns={"destLocations": "iso3"})
        .groupby(["iso3", "srcOrganization"], as_index=False)["amountUSD"]
        .sum()
    )

    rows = []
    for iso3, grp in per_donor.groupby("iso3"):
        amounts = grp["amountUSD"].to_numpy()
        total = amounts.sum()
        if total <= 0:
            continue
        shares = amounts / total
        s_sorted = np.sort(shares)[::-1]
        rows.append(
            {
                "iso3": iso3,
                "donor_hhi": float(np.sum(shares ** 2)),
                "donor_top1_share": float(s_sorted[0]),
                "donor_top3_share": float(s_sorted[:3].sum()),
                "n_donors": int(len(shares)),
                "donor_entropy": float(-np.sum(shares * np.log(shares + 1e-12))),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["donor_hhi", "donor_top1_share", "donor_top3_share",
                     "n_donors", "donor_entropy"]
        )
    return pd.DataFrame(rows).set_index("iso3")


# ─── Cluster inequality (PIN-weighted Gini over sector coverage) ────────────
def _weighted_gini(values: np.ndarray, weights: np.ndarray) -> float:
    """PIN-weighted Gini coefficient of a 1D array.

    Standard formula:
        G = (Σᵢ Σⱼ wᵢ wⱼ |vᵢ − vⱼ|) / (2 (Σw)² μ_w)
    Returns 0 for degenerate inputs rather than NaN so rows aren't dropped.
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if len(v) == 0 or w.sum() == 0:
        return 0.0
    mean = (w * v).sum() / w.sum()
    if mean == 0:
        return 0.0
    diffs = np.abs(v[:, None] - v[None, :])
    weight_prod = w[:, None] * w[None, :]
    num = (weight_prod * diffs).sum()
    denom = 2 * w.sum() ** 2 * mean
    return float(num / denom)


def cluster_inequality(year: int = 2025) -> pd.DataFrame:
    sc = sectoral.build_sector_coverage(year=year)
    sc = sc.dropna(subset=["coverage"])
    sc = sc[sc["requirements"] > 0]
    rows = []
    for iso3, grp in sc.groupby("iso3"):
        if len(grp) < 2:
            continue
        cov = grp["coverage"].to_numpy()
        pin_w = grp["pin"].fillna(0).to_numpy()
        if pin_w.sum() == 0:
            pin_w = np.ones_like(cov)
        gini = _weighted_gini(cov, pin_w)
        min_i = int(np.argmin(cov))
        rows.append(
            {
                "iso3": iso3,
                "cluster_gini": gini,
                "cluster_min_coverage": float(cov[min_i]),
                "cluster_min_name": str(grp["cluster"].iloc[min_i]),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["cluster_gini", "cluster_min_coverage", "cluster_min_name"]
        )
    return pd.DataFrame(rows).set_index("iso3")
