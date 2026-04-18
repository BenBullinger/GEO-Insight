"""Level-5 composites — the GEO-Insight gap scores per donor profile.

Implements the MAUT-additive scoring from proposal §5:

    G_p(u) = Σᵢ w_{p,i} · Uᵢ(ãᵢ(u))

where ãᵢ is the min-max normalised attribute, Uᵢ the per-attribute utility
(linear by default, concave on coverage shortfall so that the low-coverage
tail is amplified), and w_p a donor-profile weight vector.

Produces seven columns per country:
    gap_score_balanced, gap_score_cerf_profile, gap_score_echo_profile,
    gap_score_usaid_profile, gap_score_ngo_profile,
    median_rank    — median rank across the four donor profiles
    rank_iqr       — IQR of those ranks  (proposal's "disagreement band")

Only countries with all six attributes non-null are scored. Others come back
with NaN in every gap_score column (preserving provenance via the ontology).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── Six attributes in canonical order (proposal §5.3) ────────────────────
ATTRIBUTES = [
    "coverage_shortfall",  # a_1
    "per_pin_gap",         # a_2
    "need_intensity",      # a_3
    "severity_category",   # a_4
    "donor_hhi",           # a_5
    "cluster_gini",        # a_6
]

# ── Donor-profile weights (proposal §5.6) — each row sums to 1.0 ────────
PROFILE_WEIGHTS = {
    "balanced": [1 / 6] * 6,
    "cerf":     [0.30, 0.15, 0.20, 0.25, 0.05, 0.05],
    "echo":     [0.25, 0.20, 0.15, 0.15, 0.05, 0.20],
    "usaid":    [0.20, 0.15, 0.15, 0.15, 0.25, 0.10],
    "ngo":      [0.20, 0.15, 0.15, 0.10, 0.10, 0.30],
}

GAMMA_COVERAGE = 2.0  # concavity of the coverage-shortfall utility (proposal §5.4)


def _utility(name: str, tilde: pd.Series) -> pd.Series:
    """Per-attribute utility transform (proposal §5.4)."""
    if name == "coverage_shortfall":
        # U₁(x) = 1 − (1 − x)^γ  — amplifies the low-coverage tail
        return 1.0 - (1.0 - tilde.clip(0, 1)) ** GAMMA_COVERAGE
    return tilde


def _min_max(col: pd.Series) -> pd.Series:
    a, b = col.min(), col.max()
    if b - a == 0:
        return pd.Series(np.zeros(len(col)), index=col.index)
    return (col - a) / (b - a)


def compute_gap_scores(enriched: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame indexed by ISO3 with seven Level-5 columns."""
    missing = [a for a in ATTRIBUTES if a not in enriched.columns]
    if missing:
        raise ValueError(f"enriched frame is missing required attributes: {missing}")

    # only score countries where every attribute is observed
    pool = enriched.loc[enriched[ATTRIBUTES].notna().all(axis=1), ATTRIBUTES].copy()
    if pool.empty:
        cols = (
            [f"gap_score_{p if p == 'balanced' else p + '_profile'}" for p in PROFILE_WEIGHTS]
            + ["median_rank", "rank_iqr"]
        )
        return pd.DataFrame(columns=cols)

    # normalise + transform
    tilde = pool.apply(_min_max)
    U = pd.DataFrame(
        {name: _utility(name, tilde[name]) for name in ATTRIBUTES},
        index=tilde.index,
    )

    # per-profile gap scores
    out = pd.DataFrame(index=pool.index)
    for name, weights in PROFILE_WEIGHTS.items():
        col = "gap_score_" + ("balanced" if name == "balanced" else f"{name}_profile")
        out[col] = U.values @ np.asarray(weights)

    # ranks across donor profiles (exclude balanced — it's the neutral anchor)
    donor_cols = [
        "gap_score_cerf_profile",
        "gap_score_echo_profile",
        "gap_score_usaid_profile",
        "gap_score_ngo_profile",
    ]
    rank_mat = out[donor_cols].rank(ascending=False, method="average")
    out["median_rank"] = rank_mat.median(axis=1)
    out["rank_iqr"] = rank_mat.quantile(0.75, axis=1) - rank_mat.quantile(0.25, axis=1)

    return out


def four_cell_typology(enriched: pd.DataFrame) -> pd.Series:
    """Classify each crisis into one of the proposal's four-cell typology buckets.

    Requires `cluster_gini` (Level-3) and `rank_iqr` (Level-5). Uses median
    splits of the eligible pool to define "high" vs "low" per proposal §5.7.
    Missing either input → "—".
    """
    if "cluster_gini" not in enriched.columns or "rank_iqr" not in enriched.columns:
        return pd.Series("—", index=enriched.index, dtype=object)
    gini = enriched["cluster_gini"]
    iqr = enriched["rank_iqr"]
    gini_thr = gini.quantile(0.5)
    iqr_thr = iqr.quantile(0.5)

    def cell(row: pd.Series) -> str:
        g = row["cluster_gini"]
        i = row["rank_iqr"]
        if pd.isna(g) or pd.isna(i):
            return "—"
        hi_g = g > gini_thr
        hi_i = i > iqr_thr
        return {
            (True,  False): "consensus-sector-starved",
            (True,  True):  "contested-sector-starved",
            (False, False): "consensus-overlooked",
            (False, True):  "contested-balanced",
        }[(hi_g, hi_i)]

    return enriched[["cluster_gini", "rank_iqr"]].apply(cell, axis=1).rename("typology_cell")
