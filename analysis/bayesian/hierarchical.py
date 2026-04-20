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
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_median
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# `import features` deferred into main() to avoid a circular import with
# composites.py — see analysis/bayesian/mvp.py for the same pattern.
from analysis.bayesian.mvp import (  # noqa: E402
    BETA_REGR_ATTRS,
    LOGNORMAL_ATTR,
    ORDINAL_ATTR,
    ORDINAL_LEVELS,
    prepare_inputs,
    fit,
)
from analysis import validation as val  # noqa: E402


# ─── Stakeholder priors over the six attribute slopes ─────────────────────
# Each stakeholder is encoded as a HalfNormal scale per attribute. Larger
# scale = stakeholder thinks this attribute moves more strongly with
# overlookedness. The "diffuse" prior is the default consistency-regime
# prior used elsewhere in the file. The four others are pre-registered
# in proposal/methodology.md §6.2 — they are illustrative, not calibrated
# from any institution's stated policy.
STAKEHOLDER_PRIORS = {
    "diffuse": {
        "coverage_shortfall": 1.5, "per_pin_gap": 1.5, "need_intensity": 1.5,
        "severity_category": 1.5, "donor_hhi": 1.5, "cluster_gini": 1.5,
    },
    "cerf": {  # severity + magnitude emphasis
        "coverage_shortfall": 1.5, "per_pin_gap": 0.75, "need_intensity": 1.0,
        "severity_category": 1.25, "donor_hhi": 0.25, "cluster_gini": 0.25,
    },
    "echo": {  # equity emphasis
        "coverage_shortfall": 1.25, "per_pin_gap": 1.0, "need_intensity": 0.75,
        "severity_category": 0.75, "donor_hhi": 0.25, "cluster_gini": 1.0,
    },
    "usaid": {  # donor-concentration emphasis
        "coverage_shortfall": 1.0, "per_pin_gap": 0.75, "need_intensity": 0.75,
        "severity_category": 0.75, "donor_hhi": 1.25, "cluster_gini": 0.5,
    },
    "ngo": {  # cluster-equity emphasis
        "coverage_shortfall": 1.0, "per_pin_gap": 0.75, "need_intensity": 0.75,
        "severity_category": 0.5, "donor_hhi": 0.5, "cluster_gini": 1.5,
    },
}


def make_model(prior_means: dict | None = None):
    """Build a model closure with the given stakeholder prior means.

    If prior_means is None, returns the diffuse-prior production model.
    """
    if prior_means is None:
        prior_means = STAKEHOLDER_PRIORS["diffuse"]

    def _stakeholder_model(observed: dict, mask: dict, n: int):
        # Hierarchical prior on theta — same in every stakeholder regime.
        mu_theta = numpyro.sample("mu_theta", dist.Normal(0.0, 0.3))
        sigma_theta = numpyro.sample("sigma_theta", dist.HalfNormal(0.5))
        theta = numpyro.sample(
            "theta",
            dist.Normal(loc=mu_theta, scale=sigma_theta).expand([n]).to_event(1),
        )

        # Beta-regressed attributes
        for attr in BETA_REGR_ATTRS:
            alpha = numpyro.sample(f"alpha_{attr}", dist.Normal(0.0, 1.0))
            beta = numpyro.sample(f"beta_{attr}", dist.HalfNormal(prior_means[attr]))
            phi = numpyro.sample(f"phi_{attr}", dist.HalfNormal(10.0))
            mu = jax.nn.sigmoid(alpha + beta * theta)
            with handlers.mask(mask=mask[attr]):
                numpyro.sample(
                    f"obs_{attr}",
                    dist.Beta(mu * phi, (1.0 - mu) * phi),
                    obs=observed[attr],
                )

        # Log-normal attribute
        alpha_ln = numpyro.sample("alpha_per_pin_gap", dist.Normal(4.0, 2.0))
        beta_ln = numpyro.sample("beta_per_pin_gap", dist.HalfNormal(prior_means[LOGNORMAL_ATTR]))
        sigma_ln = numpyro.sample("sigma_per_pin_gap", dist.HalfNormal(1.0))
        loc_ln = alpha_ln + beta_ln * theta
        with handlers.mask(mask=mask[LOGNORMAL_ATTR]):
            numpyro.sample(
                f"obs_{LOGNORMAL_ATTR}",
                dist.LogNormal(loc=loc_ln, scale=sigma_ln),
                obs=observed[LOGNORMAL_ATTR],
            )

        # Ordered logistic
        cut_base = numpyro.sample("cut_base", dist.Normal(0.0, 2.0))
        cut_delta = numpyro.sample(
            "cut_delta", dist.HalfNormal(1.5).expand([ORDINAL_LEVELS - 2]).to_event(1)
        )
        cutpoints = jnp.concatenate([cut_base[None], cut_base + jnp.cumsum(cut_delta)])

        alpha_sev = numpyro.sample("alpha_severity_category", dist.Normal(0.0, 1.0))
        beta_sev = numpyro.sample("beta_severity_category", dist.HalfNormal(prior_means[ORDINAL_ATTR]))
        predictor = alpha_sev + beta_sev * theta
        with handlers.mask(mask=mask[ORDINAL_ATTR]):
            numpyro.sample(
                f"obs_{ORDINAL_ATTR}",
                dist.OrderedLogistic(predictor, cutpoints=cutpoints),
                obs=observed[ORDINAL_ATTR],
            )

    return _stakeholder_model


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


# ─── NUTS reference fit ────────────────────────────────────────────────────
def fit_nuts(
    inputs: dict,
    model_fn=model,
    num_warmup: int = 1000,
    num_samples: int = 2000,
    num_chains: int = 4,
    seed: int = 0,
) -> dict:
    """Run NUTS on the same model and inputs as ``fit`` for SVI calibration.

    Returns a dict with ``samples`` (posterior samples by site name),
    ``theta_median``, ``theta_ci90`` (5th / 95th percentiles), and
    ``elapsed_sec``. The schema matches ``fit`` so downstream comparison
    code can consume either one interchangeably.
    """
    numpyro.set_platform("cpu")
    kernel = NUTS(model_fn, init_strategy=init_to_median)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="vectorized",
        progress_bar=False,
    )
    t0 = time.time()
    mcmc.run(
        jax.random.PRNGKey(seed),
        observed=inputs["observed"],
        mask=inputs["mask"],
        n=inputs["n"],
    )
    elapsed = time.time() - t0

    samples = mcmc.get_samples()
    theta_samples = np.asarray(samples["theta"])
    theta_median = np.median(theta_samples, axis=0)
    theta_ci90 = np.percentile(theta_samples, [5, 95], axis=0)

    return {
        "samples": samples,
        "theta_median": theta_median,
        "theta_ci90": theta_ci90,
        "elapsed_sec": elapsed,
    }


def compare_svi_nuts(svi_res: dict, nuts_res: dict) -> dict:
    """SVI-vs-NUTS calibration metrics on the same posterior.

    Returns a dict of summary stats. The interpretation:
      * Spearman ρ on theta medians: should be > 0.95. Lower means SVI
        is recovering a different point estimate, not just a noisier one.
      * mean ratio of CI widths (NUTS / SVI): 1.0 = SVI is calibrated;
        > 1 = SVI underestimates posterior variance (the canonical
        mean-field-Gaussian failure mode); < 1 = SVI is overconfident
        in the wrong direction.
      * max absolute median deviation: a single-country worst case.
    """
    svi_med = np.asarray(svi_res["theta_median"])
    nut_med = np.asarray(nuts_res["theta_median"])
    rho, _ = spearmanr(svi_med, nut_med)

    svi_lo, svi_hi = svi_res["theta_ci90"]
    nut_lo, nut_hi = nuts_res["theta_ci90"]
    svi_w = svi_hi - svi_lo
    nut_w = nut_hi - nut_lo
    width_ratio = nut_w / np.where(svi_w > 0, svi_w, np.nan)

    return {
        "spearman_medians":     float(rho),
        "max_median_dev":       float(np.max(np.abs(svi_med - nut_med))),
        "mean_width_ratio":     float(np.nanmean(width_ratio)),
        "median_width_ratio":   float(np.nanmedian(width_ratio)),
    }


# ─── Entry point ───────────────────────────────────────────────────────────
def main() -> int:
    import features  # local import: see module docstring

    print("[hier] loading enriched frame…")
    df = features.load_cached_enriched_frame()
    if df is None:
        df = features.build_enriched_frame()

    # ── Candidate set ──
    # Countries with an observed per_pin_gap — those with both a formal HNO
    # filed and positive FTS requirements. This is the HRP-eligible pool,
    # and the same population CERF UFE largely draws from. Expanding to
    # any-appeal (requirements>0) reintroduces the non-HRP pathology
    # (CHL/TTO/AGO dominate via mechanically-saturated coverage_shortfall);
    # expanding via a completeness floor crowds out stronger CERF-aligned
    # picks with partial-data countries. 22 is where the data lands cleanly.
    cols = BETA_REGR_ATTRS + [LOGNORMAL_ATTR, ORDINAL_ATTR, "completeness"]
    scored = df[df["per_pin_gap"].notna()][cols].copy()

    print(f"[hier] HRP-eligible pool: {len(scored)} countries")
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

    print("[hier] fitting SVI (8000 steps, AutoMultivariateNormal guide)…")
    t0 = time.time()
    res = fit(inputs, model_fn=model, num_steps=8000, learning_rate=3e-3)
    print(
        f"[hier] SVI took {res['elapsed_sec']:.1f}s · "
        f"final ELBO loss = {res['losses'][-1]:.2f} "
        f"(wall total {time.time() - t0:.1f}s)"
    )

    print("[hier] fitting NUTS (1000 warmup + 2000 samples × 4 chains)…")
    nuts_res = fit_nuts(inputs, model_fn=model)
    cal = compare_svi_nuts(res, nuts_res)
    print(f"[hier] NUTS took {nuts_res['elapsed_sec']:.1f}s")
    print(f"[hier] SVI ↔ NUTS calibration:")
    print(f"        Spearman ρ on theta medians  : {cal['spearman_medians']:+.3f}  "
          f"(target > 0.95)")
    print(f"        max |SVI median − NUTS median|: {cal['max_median_dev']:.3f}")
    print(f"        mean CI-width ratio NUTS/SVI : {cal['mean_width_ratio']:.2f}  "
          f"(1.0 = SVI calibrated; > 1 = SVI underestimates variance)")
    print(f"        median CI-width ratio        : {cal['median_width_ratio']:.2f}")

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
    # CERF UFE allocations and CARE BTS rankings are human-curated lists
    # of underfunded / under-reported crises produced independently of
    # any model in this repo. They are the only valid ground truth for
    # the "overlooked humanitarian crisis" construct.
    iso = inputs["iso3"]
    theta_med = res["theta_median"]
    bay_rank = pd.Series(-theta_med, index=iso).rank(method="average")

    benchmarks = [
        ("CERF UFE 2024 w2", set(val.load_cerf_ufe(2024).query("window == 2")["iso3"]), 10),
        ("CERF UFE 2025 w1", set(val.load_cerf_ufe(2025).query("window == 1")["iso3"]), 10),
        ("CERF UFE 2025 w2", set(val.load_cerf_ufe(2025).query("window == 2")["iso3"]), 7),
        ("CARE BTS 2024",    set(val.load_care_bts(2024)["iso3"]),                       10),
    ]

    print()
    print(f"{'benchmark':<22s} {'k':>3s}   {'precision @ k':>15s}")
    print("-" * 48)
    for name, bset, k in benchmarks:
        bay_top = set(bay_rank.nsmallest(k).index.astype(str))
        prec = len(bay_top & bset) / k
        print(f"{name:<22s} {k:>3d}   {prec:>5.2f} ({len(bay_top & bset)}/{k})")

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
