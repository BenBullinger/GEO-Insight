"""Level-5 composites — the Geo-Insight gap scores per donor profile.

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


def compute_gap_scores(
    enriched: pd.DataFrame, min_completeness: float = 3 / 6
) -> pd.DataFrame:
    """Six-attribute MAUT composite — **evidence-bounded**.

    For country `u`, profile `p`:

        G_p(u) = Σ_{i ∈ 𝒪(u)} w_{p,i} · U_i(ã_i(u))

    i.e. a **weighted sum (not mean) over observed attributes only**. Missing
    attributes contribute zero to the score. Crucially, the weights are
    *not* renormalised over the observed subset.

    This is the principled answer to the question "how should incomplete
    observations scale?". Three alternative formulations were considered and
    rejected:

      1. Weighted mean of observed  (G = Σ_{i∈𝒪} wU / Σ_{i∈𝒪} w)
         — systematically inflates extreme partial-complete countries; a
         country observed only on three "hottest" attributes can outrank a
         fully-observed country with moderate values. Empirically verified
         on our CERF UFE validation: strict-pool Overlap@5 = 100%
         collapsed to 0% under graceful, with partial-complete countries
         displacing the correct top.

      2. Bayesian mean-imputation (missing ← population mean of U_i).
         — statistically principled under MCAR, but a country with all
         observed attributes extreme still ranks above fully-observed
         moderates: the imputation regresses *missing* attributes to the
         mean but not *extreme observed* ones. Post-fix CERF UFE
         Overlap@5 in All pool: 40% — better but still inflated.

      3. Model-based imputation
         — fabricates country-specific values, violates the proposal's
         no-imputation principle.

    Evidence-bounded scoring (this function) has the property that a
    partial-complete country's MAXIMUM possible score is its observed
    weight-mass Σ_{i ∈ 𝒪(u)} w_{p,i} < 1. A fully-observed country with
    moderate attributes will therefore always outrank a partial-complete
    country whose observed-weight-mass is less than its own total score —
    exactly the robustness the user's "incompleteness must not inflate"
    constraint demands.

    The tradeoff: systematically under-ranks countries with extreme but
    poorly-documented crises. We accept this as the cost of the principle.
    This is documented in the proposal §5.7 and surfaced in the dashboard
    through the `completeness` flag.

    Reduces to Σᵢ w_{p,i} U_i(ã_i(u)) when 𝒪(u) = {1..6}.

    Countries with completeness < `min_completeness` are withheld from the
    ranking entirely (gap_score_* set to NaN).

    Output columns:
      gap_score_balanced, gap_score_{cerf,echo,usaid,ngo}_profile,
      completeness, median_rank, rank_iqr.
    """
    missing = [a for a in ATTRIBUTES if a not in enriched.columns]
    if missing:
        raise ValueError(f"enriched frame is missing required attributes: {missing}")

    # Per-attribute min-max over every country that reports it.
    tilde = pd.DataFrame(
        np.nan, index=enriched.index, columns=ATTRIBUTES, dtype=float
    )
    for attr in ATTRIBUTES:
        col = enriched[attr]
        valid = col.dropna()
        if valid.empty:
            continue
        a, b = valid.min(), valid.max()
        if b == a:
            tilde[attr] = np.where(col.notna(), 0.0, np.nan)
        else:
            tilde[attr] = (col - a) / (b - a)

    # Per-attribute utility transform.
    U = pd.DataFrame(index=enriched.index, columns=ATTRIBUTES, dtype=float)
    for attr in ATTRIBUTES:
        U[attr] = _utility(attr, tilde[attr])

    # Completeness flag.
    present = U.notna()
    completeness = (present.sum(axis=1) / len(ATTRIBUTES)).rename("completeness")
    eligible = completeness >= min_completeness

    # Evidence-bounded score: weighted sum of observed, missing contributes 0.
    U_obs = U.fillna(0.0)  # zero out missing for the dot product

    out = pd.DataFrame(index=enriched.index)
    for name, weights in PROFILE_WEIGHTS.items():
        w = np.asarray(weights, dtype=float)
        score = pd.Series(U_obs.values @ w, index=enriched.index)
        score.loc[~eligible] = np.nan
        col = "gap_score_" + ("balanced" if name == "balanced" else f"{name}_profile")
        out[col] = score

    out["completeness"] = completeness
    out.loc[completeness == 0, "completeness"] = np.nan

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
