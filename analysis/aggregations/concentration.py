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


# ─── Phase-composition Gini (INFORM sub-indicators) ────────────────────────
def phase_gini_latest() -> pd.DataFrame:
    """Gini coefficient over the pin_phase_1..5 population counts per country."""
    ind = pd.read_csv(
        DATA / "Third-Party" / "DRMKC-INFORM" / "inform_indicators_long.csv"
    )
    ind = ind.sort_values(["ISO3", "snapshot"])
    latest = ind.groupby("ISO3").tail(1).set_index("ISO3")
    level_cols = [f"pin_level_{i}" for i in range(1, 6)]
    for c in level_cols:
        if c not in latest.columns:
            latest[c] = 0.0
        latest[c] = pd.to_numeric(latest[c], errors="coerce").fillna(0)

    rows = []
    for iso3, row in latest[level_cols].iterrows():
        vals = row.to_numpy(dtype=float)
        w = np.ones_like(vals)
        rows.append({"iso3": iso3, "phase_gini": _weighted_gini(vals, w)})
    return pd.DataFrame(rows).set_index("iso3")


# ─── CBPF reliance ─────────────────────────────────────────────────────────
def cbpf_reliance_latest(year: int = 2025) -> pd.DataFrame:
    """Share of a country's humanitarian funding that flows via CBPF.

    Maps CBPF projects to ISO3 by matching `PooledFundName` prefix. The
    PooledFundName takes forms like "Sudan Humanitarian Fund" or
    "Afghanistan Humanitarian Fund"; we match against a small hand-curated
    dictionary of fund → ISO3 for the recognised fund names. Funds we can't
    match are excluded.
    """
    fund_to_iso3 = {
        "Afghanistan": "AFG",
        "Central African Republic": "CAF",
        "Colombia": "COL",
        "Democratic Republic of the Congo": "COD",
        "DRC": "COD",
        "Ethiopia": "ETH",
        "Haiti": "HTI",
        "Iraq": "IRQ",
        "Jordan": "JOR",
        "Lebanon": "LBN",
        "Mali": "MLI",
        "Myanmar": "MMR",
        "Nigeria": "NGA",
        "occupied Palestinian territory": "PSE",
        "oPt": "PSE",
        "Pakistan": "PAK",
        "Somalia": "SOM",
        "South Sudan": "SSD",
        "Sudan": "SDN",
        "Syria": "SYR",
        "Syria Cross Border": "SYR",
        "Ukraine": "UKR",
        "Yemen": "YEM",
    }
    try:
        cbpf = pd.read_csv(DATA / "cbpf" / "cbpf_project_summary.csv", low_memory=False)
    except FileNotFoundError:
        return pd.DataFrame(columns=["cbpf_reliance"])
    cbpf["AllocationYear"] = pd.to_numeric(cbpf["AllocationYear"], errors="coerce")
    cbpf_y = cbpf[cbpf["AllocationYear"] == year].copy()
    cbpf_y["Budget"] = pd.to_numeric(cbpf_y["Budget"], errors="coerce").fillna(0)

    def map_iso(name: str) -> str | None:
        if not isinstance(name, str):
            return None
        for key, iso in fund_to_iso3.items():
            if key.lower() in name.lower():
                return iso
        return None

    cbpf_y["iso3"] = cbpf_y["PooledFundName"].apply(map_iso)
    cbpf_y = cbpf_y.dropna(subset=["iso3"])
    per_country = cbpf_y.groupby("iso3")["Budget"].sum().rename("cbpf_allocation")

    # FTS funding received for denominator
    fts = pd.read_csv(
        DATA / "fts" / "fts_requirements_funding_global.csv", skiprows=[1]
    )
    fts["year"] = pd.to_numeric(fts["year"], errors="coerce")
    fts_y = fts[fts["year"] == year].groupby("countryCode")["funding"].sum()
    fts_y.index.name = "iso3"

    joined = pd.concat([per_country, fts_y.rename("funding_received")], axis=1)
    joined["cbpf_reliance"] = (
        joined["cbpf_allocation"] / joined["funding_received"].replace({0: np.nan})
    ).clip(upper=1.0)
    return joined[["cbpf_reliance"]].dropna()

