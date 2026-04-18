"""Hierarchical cross-sectional model — Day 2 of the latent-variable build.

The Day-1 MVP in ``mvp.py`` uses a flat ``theta ~ Normal(0, 2)`` prior.
Under that regime, countries with few observed attributes can still get
large posterior |theta| — the likelihood of the one or two attributes
they do have is free to dominate, and there is no mechanism pulling
under-constrained countries back toward the rest of the population.

The fix is the population-level prior on theta specified in
``proposal/methodology.md`` §5:

    mu_theta    ~ Normal(0, 0.5)            (population mean)
    sigma_theta ~ HalfCauchy(1)             (population scale)
    theta[u]    ~ StudentT(nu=4, mu_theta, sigma_theta)

Student-t with nu=4 keeps heavier-than-Gaussian tails so genuinely
extreme countries can still leave the pack — but countries with weak
evidence shrink toward ``mu_theta`` (close to 0 under the diffuse
consistency regime), because the t density penalises drifting away
from the population scale.

This is the consistency regime — the validation target is a high
Spearman rank correlation with the existing MAUT balanced score.

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


# ─── NumPyro model (hierarchical prior on theta) ───────────────────────────
def model(observed: dict, mask: dict, n: int):
    """Same observation model as ``mvp.model`` but with a pooled prior on theta.

    The only change is the three lines that draw theta. Everything else
    (attribute slopes, dispersions, cutpoints, masking) is identical.
    """
    # ── Hierarchical prior on theta ──
    mu_theta = numpyro.sample("mu_theta", dist.Normal(0.0, 0.5))
    sigma_theta = numpyro.sample("sigma_theta", dist.HalfCauchy(1.0))
    theta = numpyro.sample(
        "theta",
        dist.StudentT(df=4.0, loc=mu_theta, scale=sigma_theta)
        .expand([n])
        .to_event(1),
    )

    # ── Beta-regressed attributes ──
    for attr in BETA_REGR_ATTRS:
        alpha = numpyro.sample(f"alpha_{attr}", dist.Normal(0.0, 1.0))
        if attr == "coverage_shortfall":
            beta = numpyro.sample(
                f"beta_{attr}", dist.TruncatedNormal(1.0, 1.0, low=0.0)
            )
        else:
            beta = numpyro.sample(f"beta_{attr}", dist.Normal(0.5, 1.0))
        phi = numpyro.sample(f"phi_{attr}", dist.HalfNormal(10.0))
        mu = jax.nn.sigmoid(alpha + beta * theta)
        with handlers.mask(mask=mask[attr]):
            numpyro.sample(
                f"obs_{attr}",
                dist.Beta(mu * phi, (1.0 - mu) * phi),
                obs=observed[attr],
            )

    # ── Log-normal attribute ──
    alpha_ln = numpyro.sample("alpha_per_pin_gap", dist.Normal(4.0, 2.0))
    beta_ln = numpyro.sample("beta_per_pin_gap", dist.Normal(0.5, 1.0))
    sigma_ln = numpyro.sample("sigma_per_pin_gap", dist.HalfNormal(1.0))
    loc_ln = alpha_ln + beta_ln * theta
    with handlers.mask(mask=mask[LOGNORMAL_ATTR]):
        numpyro.sample(
            f"obs_{LOGNORMAL_ATTR}",
            dist.LogNormal(loc=loc_ln, scale=sigma_ln),
            obs=observed[LOGNORMAL_ATTR],
        )

    # ── Ordered logistic ──
    cut_base = numpyro.sample("cut_base", dist.Normal(0.0, 2.0))
    cut_delta = numpyro.sample(
        "cut_delta", dist.HalfNormal(1.5).expand([ORDINAL_LEVELS - 2]).to_event(1)
    )
    cutpoints = jnp.concatenate([cut_base[None], cut_base + jnp.cumsum(cut_delta)])

    alpha_sev = numpyro.sample("alpha_severity_category", dist.Normal(0.0, 1.0))
    beta_sev = numpyro.sample("beta_severity_category", dist.Normal(0.5, 1.0))
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

    eligible = df["completeness"].fillna(0) >= 0.5
    scored = df[eligible].copy()
    cols = BETA_REGR_ATTRS + [
        LOGNORMAL_ATTR,
        ORDINAL_ATTR,
        "gap_score_balanced",
        "completeness",
    ]
    scored = scored[cols]
    attrs = BETA_REGR_ATTRS + [LOGNORMAL_ATTR, ORDINAL_ATTR]
    obs_count = scored[attrs].notna().sum(axis=1)
    scored = scored[obs_count >= 1]

    print(f"[hier] scored pool: {len(scored)} countries")
    print(
        f"[hier] completeness: min={scored['completeness'].min():.2f} "
        f"mean={scored['completeness'].mean():.2f}"
    )

    inputs = prepare_inputs(scored)
    print(f"[hier] prepared arrays: n={inputs['n']}")
    for attr in attrs:
        n_obs = int(np.asarray(inputs["mask"][attr]).sum())
        print(f"        {attr:<24s}  observed in {n_obs}/{inputs['n']}")

    print("[hier] fitting SVI (4000 steps, hierarchical prior)…")
    t0 = time.time()
    res = fit(inputs, model_fn=model, num_steps=4000)
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

    # ── Consistency check against MAUT ──
    iso = inputs["iso3"]
    theta_med = res["theta_median"]
    maut = scored["gap_score_balanced"].values
    valid = ~np.isnan(maut)
    rho, _ = spearmanr(theta_med[valid], maut[valid])
    print()
    print(f"[hier] Spearman ρ  (theta median vs MAUT balanced): {rho:+.3f}")

    bay_rank = pd.Series(-theta_med, index=iso).rank(method="average")
    maut_rank = pd.Series(-maut, index=iso).rank(method="average")
    top10_bay = bay_rank.nsmallest(10).index.tolist()
    top10_maut = maut_rank.nsmallest(10).index.tolist()
    overlap = len(set(top10_bay) & set(top10_maut))
    print(f"[hier] top-10 overlap (Bayesian vs MAUT balanced): {overlap}/10")

    print()
    print("[hier] Bayesian top-10:")
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
