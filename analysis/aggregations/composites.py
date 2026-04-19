"""Level-5 composites — Bayesian posterior over the latent overlookedness θ.

For each HRP-eligible country (those with an observed `per_pin_gap`), we
fit the hierarchical latent-variable model in `analysis/bayesian/hierarchical.py`
and report the posterior median of θ together with a 90 % credible
interval. Countries outside the HRP-eligible pool — where the construct
"overlooked humanitarian crisis" is not well-defined, since CERF UFE
allocations themselves draw from this pool — are returned with NaN in
every posterior column.

Output columns per country:
    theta_median     posterior median of the latent (higher = more overlooked)
    theta_ci_lo      5th-percentile of the posterior
    theta_ci_hi      95th-percentile of the posterior
    theta_ci_width   posterior 90 % CI width (uncertainty proxy)
    completeness     fraction of the six attributes observed for the country

The Bayesian fit replaces the additive MAUT scoring that lived here
previously. The model is validated externally against CERF UFE and CARE
BTS — see `analysis/validation.py` and `analysis/bayesian/hierarchical.py`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── Six attributes in canonical order ──────────────────────────────────────
ATTRIBUTES = [
    "coverage_shortfall",  # a_1
    "per_pin_gap",         # a_2
    "need_intensity",      # a_3
    "severity_category",   # a_4
    "donor_hhi",           # a_5
    "cluster_gini",        # a_6
]


def compute_overlookedness_posterior(enriched: pd.DataFrame) -> pd.DataFrame:
    """Fit the hierarchical latent-variable model and return posterior summaries.

    Pool definition: HRP-eligible countries (per_pin_gap observed). This is
    the population CERF UFE picks from, so it is the only pool for which
    external validation is meaningful.

    Returns a DataFrame indexed identically to `enriched`, with NaN for
    countries outside the HRP-eligible pool.
    """
    # Imports are local: numpyro/jax pull in heavy deps and we don't want
    # them at module-import time (composites is imported by features which
    # is imported by every dashboard view).
    from analysis.bayesian import hierarchical
    from analysis.bayesian.mvp import prepare_inputs, fit

    missing = [a for a in ATTRIBUTES if a not in enriched.columns]
    if missing:
        raise ValueError(f"enriched frame is missing required attributes: {missing}")

    # Completeness over the six attributes (used for the diagnostic column
    # and as a sanity filter inside the HRP pool).
    present = enriched[ATTRIBUTES].notna()
    completeness = (present.sum(axis=1) / len(ATTRIBUTES)).rename("completeness")

    out = pd.DataFrame(
        {
            "theta_median":   np.nan,
            "theta_ci_lo":    np.nan,
            "theta_ci_hi":    np.nan,
            "theta_ci_width": np.nan,
            "completeness":   completeness,
        },
        index=enriched.index,
    )

    # Restrict to HRP-eligible countries
    hrp = enriched[enriched["per_pin_gap"].notna()].copy()
    if len(hrp) == 0:
        return out

    inputs = prepare_inputs(hrp)
    # 8000 steps gets the AutoMultivariateNormal guide to ~0.9 Spearman
    # against NUTS on theta medians, with CI widths within ~2x of NUTS.
    # Cheaper configurations under-converge the MVN covariance and degrade
    # the calibration meaningfully. See analysis/bayesian/hierarchical.py
    # main() for the SVI-vs-NUTS calibration check that justifies this.
    res = fit(inputs, model_fn=hierarchical.model, num_steps=8000, learning_rate=3e-3)

    iso = inputs["iso3"]
    theta_med = res["theta_median"]
    lo, hi = res["theta_ci90"]

    out.loc[iso, "theta_median"]   = theta_med
    out.loc[iso, "theta_ci_lo"]    = lo
    out.loc[iso, "theta_ci_hi"]    = hi
    out.loc[iso, "theta_ci_width"] = hi - lo

    return out


def four_cell_typology(enriched: pd.DataFrame) -> pd.Series:
    """Classify each crisis into a four-cell typology along
    (cluster_gini × posterior CI width).

    Axes:
        cluster_gini   — sectoral imbalance of funding (Level 3)
        theta_ci_width — uncertainty in the latent posterior (Level 5)

    Cells (median split of the eligible pool):
        consensus-overlooked       low gini   ·  narrow CI   (clear, broad-based)
        contested-balanced         low gini   ·  wide CI     (uncertain about a balanced crisis)
        consensus-sector-starved   high gini  ·  narrow CI   (clear, but lopsided)
        contested-sector-starved   high gini  ·  wide CI     (uncertain and lopsided)

    Missing either axis → "—".
    """
    if "cluster_gini" not in enriched.columns or "theta_ci_width" not in enriched.columns:
        return pd.Series("—", index=enriched.index, dtype=object)
    gini = enriched["cluster_gini"]
    width = enriched["theta_ci_width"]
    gini_thr = gini.quantile(0.5)
    width_thr = width.quantile(0.5)

    def cell(row: pd.Series) -> str:
        g = row["cluster_gini"]
        w = row["theta_ci_width"]
        if pd.isna(g) or pd.isna(w):
            return "—"
        hi_g = g > gini_thr
        hi_w = w > width_thr
        return {
            (True,  False): "consensus-sector-starved",
            (True,  True):  "contested-sector-starved",
            (False, False): "consensus-overlooked",
            (False, True):  "contested-balanced",
        }[(hi_g, hi_w)]

    return (
        enriched[["cluster_gini", "theta_ci_width"]]
        .apply(cell, axis=1)
        .rename("typology_cell")
    )
