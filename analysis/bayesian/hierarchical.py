"""Hierarchical cross-sectional model — Day 2 of the latent-variable build.

The Day-1 MVP in ``mvp.py`` uses a flat ``theta ~ Normal(0, 2)`` prior.
Under that regime, countries with few observed attributes can still get
large posterior |theta| — the likelihood of the one or two attributes
they do have is free to dominate, and there is no mechanism pulling
under-constrained countries back toward the rest of the population.

The fix is a population-level prior on theta with **short-tailed
shrinkage**:

    mu_theta    ~ Normal(0, 0.2)            (population mean, near 0)
    sigma_theta ~ HalfNormal(0.3)           (population scale, bounded)
    theta[u]    ~ Normal(mu_theta, sigma_theta)

The methodology originally specified Student-t(nu=4) for outlier
robustness on sigma_theta. In practice that produced heavy tails that
let under-constrained countries drift to large |theta| — the opposite
of what we want here. Gaussian shrinkage penalises excursions
quadratically, which is the right tradeoff when the goal is to pull
low-evidence countries back toward the population mean while still
letting well-evidenced countries separate.

Validation regime: the candidate pool is restricted to HRP-eligible
countries (those with an observed per_pin_gap), which is the population
CERF UFE allocations actually draw from. The latent's posterior median
ranking is scored against:

  * CERF UFE — the UN's twice-yearly underfunded-emergency allocations
    (independent expert consensus on which crises are underfunded).
  * CARE BTS — CARE's annual top-10 most under-reported crises.

These are both human-curated and methodologically independent of any
model in this repo, so they are valid external ground truth for the
"overlooked humanitarian crisis" construct.

Usage:
    dashboard/.venv/bin/python -m analysis.bayesian.hierarchical
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro import handlers
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import features  # noqa: E402

from analysis.bayesian.mvp import (  # noqa: E402
    BETA_REGR_ATTRS,
    LOGNORMAL_ATTR,
    ORDINAL_ATTR,
    ORDINAL_LEVELS,
    prepare_inputs,
    fit,
)
from analysis import validation as val  # noqa: E402


# ─── NumPyro model (hierarchical prior on theta) ───────────────────────────
def model(observed: dict, mask: dict, n: int):
    """Same observation model as ``mvp.model`` but with a pooled prior on theta.

    The only change is the three lines that draw theta. Everything else
    (attribute slopes, dispersions, cutpoints, masking) is identical.
    """
    # ── Hierarchical prior on theta (short-tailed shrinkage) ──
    mu_theta = numpyro.sample("mu_theta", dist.Normal(0.0, 0.3))
    sigma_theta = numpyro.sample("sigma_theta", dist.HalfNormal(0.5))
    theta = numpyro.sample(
        "theta",
        dist.Normal(loc=mu_theta, scale=sigma_theta).expand([n]).to_event(1),
    )

    # ── Beta-regressed attributes ──
    # All six slopes are constrained positive: every attribute is defined so
    # that higher values correspond to "more overlooked" (higher coverage
    # shortfall, higher per-PIN gap, more concentrated donors, sharper
    # cluster imbalance, greater need intensity, higher severity). Without
    # this, the sign of each slope is unidentifiable from the latent and
    # the inferred theta picks up an arbitrary linear combination.
    for attr in BETA_REGR_ATTRS:
        alpha = numpyro.sample(f"alpha_{attr}", dist.Normal(0.0, 1.0))
        beta = numpyro.sample(f"beta_{attr}", dist.HalfNormal(1.5))
        phi = numpyro.sample(f"phi_{attr}", dist.HalfNormal(10.0))
        mu = jax.nn.sigmoid(alpha + beta * theta)
        with handlers.mask(mask=mask[attr]):
            numpyro.sample(
                f"obs_{attr}",
                dist.Beta(mu * phi, (1.0 - mu) * phi),
                obs=observed[attr],
            )

    # ── Log-normal attribute (per_pin_gap) ──
    # beta_ln uses HalfNormal so the positivity constraint is enforced
    # without the TruncatedNormal boundary, which proved numerically
    # fragile when composed with the tight hierarchical prior on theta.
    alpha_ln = numpyro.sample("alpha_per_pin_gap", dist.Normal(4.0, 2.0))
    beta_ln = numpyro.sample("beta_per_pin_gap", dist.HalfNormal(1.5))
    sigma_ln = numpyro.sample("sigma_per_pin_gap", dist.HalfNormal(1.0))
    loc_ln = alpha_ln + beta_ln * theta
    with handlers.mask(mask=mask[LOGNORMAL_ATTR]):
        numpyro.sample(
            f"obs_{LOGNORMAL_ATTR}",
            dist.LogNormal(loc=loc_ln, scale=sigma_ln),
            obs=observed[LOGNORMAL_ATTR],
        )

    # ── Ordered logistic (severity_category) ──
    cut_base = numpyro.sample("cut_base", dist.Normal(0.0, 2.0))
    cut_delta = numpyro.sample(
        "cut_delta", dist.HalfNormal(1.5).expand([ORDINAL_LEVELS - 2]).to_event(1)
    )
    cutpoints = jnp.concatenate([cut_base[None], cut_base + jnp.cumsum(cut_delta)])

    alpha_sev = numpyro.sample("alpha_severity_category", dist.Normal(0.0, 1.0))
    beta_sev = numpyro.sample("beta_severity_category", dist.HalfNormal(1.5))
    predictor = alpha_sev + beta_sev * theta
    with handlers.mask(mask=mask[ORDINAL_ATTR]):
        numpyro.sample(
            f"obs_{ORDINAL_ATTR}",
            dist.OrderedLogistic(predictor, cutpoints=cutpoints),
            obs=observed[ORDINAL_ATTR],
        )


# ─── Entry point ───────────────────────────────────────────────────────────
def main() -> int:
    print("[hier] loading enriched frame…")
    df = features.load_cached_enriched_frame()
    if df is None:
        df = features.build_enriched_frame()

    # ── Candidate set ──
    # The construct "overlooked humanitarian crisis" is only well-defined for
    # countries with an active humanitarian response plan: per_pin_gap (= HRP
    # gap / PIN) is undefined when no HRP exists. CERF UFE allocations draw
    # exclusively from this pool, so external validation requires we do too.
    # Including non-HRP countries put coverage_shortfall = 1 and donor_hhi = 1
    # by mechanical default and pulled the latent toward those countries
    # spuriously; restricting to HRP-eligible countries removes that artefact.
    cols = BETA_REGR_ATTRS + [
        LOGNORMAL_ATTR,
        ORDINAL_ATTR,
        "gap_score_balanced",
        "completeness",
    ]
    scored = df[df["per_pin_gap"].notna()][cols].copy()

    print(f"[hier] HRP-eligible candidate pool: {len(scored)} countries")
    print(
        f"[hier] completeness: min={scored['completeness'].min():.2f} "
        f"mean={scored['completeness'].mean():.2f}"
    )

    attrs = BETA_REGR_ATTRS + [LOGNORMAL_ATTR, ORDINAL_ATTR]
    inputs = prepare_inputs(scored)
    print(f"[hier] prepared arrays: n={inputs['n']}")
    for attr in attrs:
        n_obs = int(np.asarray(inputs["mask"][attr]).sum())
        print(f"        {attr:<24s}  observed in {n_obs}/{inputs['n']}")

    print("[hier] fitting SVI (6000 steps, hierarchical prior)…")
    t0 = time.time()
    res = fit(inputs, model_fn=model, num_steps=6000, learning_rate=3e-3)
    print(
        f"[hier] SVI took {res['elapsed_sec']:.1f}s · "
        f"final ELBO loss = {res['losses'][-1]:.2f} "
        f"(wall total {time.time() - t0:.1f}s)"
    )

    # ── Population-level hyperparameters ──
    mu_samples = np.asarray(res["samples"]["mu_theta"])
    sig_samples = np.asarray(res["samples"]["sigma_theta"])
    print()
    print(
        f"[hier] mu_theta    posterior: median={np.median(mu_samples):+.3f}  "
        f"90% CI=[{np.percentile(mu_samples, 5):+.3f}, {np.percentile(mu_samples, 95):+.3f}]"
    )
    print(
        f"[hier] sigma_theta posterior: median={np.median(sig_samples):.3f}  "
        f"90% CI=[{np.percentile(sig_samples, 5):.3f}, {np.percentile(sig_samples, 95):.3f}]"
    )

    # ── External validation against CERF UFE and CARE BTS ──
    # The validation question is not "does the latent agree with MAUT" — both
    # are models. The validation question is "does the latent agree with the
    # human-curated, methodologically-independent benchmarks". CERF UFE picks
    # from the HRP-eligible pool exactly, so it is the cleanest external
    # signal of expert consensus on which crises are underfunded.
    iso = inputs["iso3"]
    theta_med = res["theta_median"]
    maut = scored["gap_score_balanced"].values

    bay_rank = pd.Series(-theta_med, index=iso).rank(method="average")
    maut_rank = pd.Series(-maut, index=iso).rank(method="average")

    benchmarks = [
        ("CERF UFE 2024 w2", set(val.load_cerf_ufe(2024).query("window == 2")["iso3"]), 10),
        ("CERF UFE 2025 w1", set(val.load_cerf_ufe(2025).query("window == 1")["iso3"]), 10),
        ("CERF UFE 2025 w2", set(val.load_cerf_ufe(2025).query("window == 2")["iso3"]), 7),
        ("CARE BTS 2024",    set(val.load_care_bts(2024)["iso3"]),                       10),
    ]

    print()
    print(
        f"{'benchmark':<22s} {'k':>3s}   {'Bayesian':>15s}   {'MAUT balanced':>15s}"
    )
    print("-" * 64)
    for name, bset, k in benchmarks:
        bay_top = set(bay_rank.nsmallest(k).index.astype(str))
        maut_top = set(maut_rank.nsmallest(k).index.astype(str))
        bay_p = len(bay_top & bset) / k
        maut_p = len(maut_top & bset) / k
        print(
            f"{name:<22s} {k:>3d}   "
            f"{bay_p:>5.2f} ({len(bay_top & bset)}/{k})    "
            f"{maut_p:>5.2f} ({len(maut_top & bset)}/{k})"
        )

    print()
    rho, _ = spearmanr(theta_med, maut)
    print(f"[hier] Spearman ρ (theta vs MAUT, internal cross-check): {rho:+.3f}")

    print()
    print("[hier] Bayesian top-10:")
    top10_bay = bay_rank.nsmallest(10).index.tolist()
    lo, hi = res["theta_ci90"]
    for iso3 in top10_bay:
        i = inputs["iso3"].index(iso3)
        print(
            f"  {iso3}   median θ = {theta_med[i]:+.3f}   "
            f"90% CI = [{lo[i]:+.3f}, {hi[i]:+.3f}]   "
            f"completeness = {scored.loc[iso3, 'completeness']:.2f}"
        )

    # ── Width-by-completeness diagnostic ──
    widths = hi - lo
    comp = scored["completeness"].values
    print()
    print("[hier] CI width by completeness bucket:")
    for lo_c, hi_c in [(0.5, 0.67), (0.67, 0.84), (0.84, 1.01)]:
        m = (comp >= lo_c) & (comp < hi_c)
        if m.sum() > 0:
            print(
                f"        completeness ∈ [{lo_c:.2f}, {hi_c:.2f})  n={m.sum():3d}  "
                f"mean CI width = {widths[m].mean():.2f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
