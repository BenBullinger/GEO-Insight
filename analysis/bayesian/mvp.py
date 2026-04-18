"""Cross-sectional MVP — Day 1 of the latent-variable build.

One month, one stakeholder, six attributes, diffuse prior, no temporal
structure, no hierarchical pooling. The point is to verify that the
inference machinery works end-to-end before layering on complexity.

Model recap (see proposal/methodology.md):

    theta[u]    ~ Normal(0, 2)                  (latent per country)
    alpha_i     ~ Normal(0, 1)
    beta_i      ~ TruncNormal(1, 1, lower=0)    (a1 = coverage_shortfall,
                                                 sign convention)
                ~ Normal(0.5, 1.0)               (other beta regressors)
                ~ Normal(0.5, 1.0)               (log-normal)
                ~ Normal(0.5, 1.0)               (ordered logistic)
    phi_i       ~ HalfNormal(10)                 (beta regression only)
    sigma       ~ HalfNormal(1)                  (log-normal only)
    kappa       ~ TransformedDistribution(...)   (ordered logistic cutpoints)

Observation model per attribute, with missingness handled by
numpyro.handlers.mask:

    a1 (coverage_shortfall) in [0,1]  ~ Beta(mu*phi, (1-mu)*phi),
                                         mu = sigmoid(alpha + beta*theta)
    a2 (per_pin_gap)        in R+      ~ LogNormal(alpha + beta*theta, sigma)
    a3 (need_intensity)     in [0,1]  ~ Beta(...)
    a4 (severity_category)  in {0..4}  ~ OrderedLogistic(alpha + beta*theta, kappa)
    a5 (donor_hhi)          in [0,1]  ~ Beta(...)
    a6 (cluster_gini)       in [0,1]  ~ Beta(...)

Validation of this MVP:
    1. SVI converges (ELBO monotonically decreases, finite gradients)
    2. Rank order of posterior median(theta) correlates strongly with the
       existing MAUT balanced ranking (consistency test)
    3. Partial-completeness countries have visibly wider posteriors

Usage:
    dashboard/.venv/bin/python -m analysis.bayesian.mvp
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
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
from numpyro.infer.initialization import init_to_median
from scipy.stats import spearmanr

# Allow `import features` when run from the repo root or as a module
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import features  # noqa: E402


# ─── Attribute spec ────────────────────────────────────────────────────────
# Order matches composites.py, so downstream comparisons line up cleanly.
BETA_REGR_ATTRS = ["coverage_shortfall", "need_intensity", "donor_hhi", "cluster_gini"]
LOGNORMAL_ATTR  = "per_pin_gap"
ORDINAL_ATTR    = "severity_category"
ORDINAL_LEVELS  = 5  # INFORM severity 1..5 → 0..4 internally

EPS = 1e-4  # clip bounded attributes into (EPS, 1-EPS) so Beta is well-defined


# ─── Data preparation ──────────────────────────────────────────────────────
def prepare_inputs(df: pd.DataFrame) -> dict:
    """Turn the enriched frame into arrays ready for NumPyro.

    Returns a dict with:
        iso3:          list of country codes (length n)
        observed:      dict attr -> float jax array of length n (NaN filled with 0)
        mask:          dict attr -> bool jax array of length n (True when observed)
        n:             int, number of countries
    """
    iso3 = df.index.astype(str).tolist()
    n = len(iso3)

    observed = {}
    mask = {}

    # Beta-regressed attributes — values clipped to (EPS, 1-EPS)
    for attr in BETA_REGR_ATTRS:
        col = df[attr].astype(float)
        m = col.notna().values
        v = col.clip(lower=EPS, upper=1 - EPS).fillna(0.5).values
        observed[attr] = jnp.asarray(v, dtype=jnp.float32)
        mask[attr] = jnp.asarray(m)

    # Log-normal attribute — values must be strictly positive
    col = df[LOGNORMAL_ATTR].astype(float)
    m = col.notna().values & (col.fillna(-1).values > 0)
    v = col.where(col > 0, other=EPS).fillna(EPS).values
    observed[LOGNORMAL_ATTR] = jnp.asarray(v, dtype=jnp.float32)
    mask[LOGNORMAL_ATTR] = jnp.asarray(m)

    # Ordered attribute — severity 1..5 → 0..4
    col = df[ORDINAL_ATTR].astype(float)
    m = col.notna().values
    # Fill missing with 0 (arbitrary — the mask will suppress the likelihood)
    v = (col.fillna(1).clip(lower=1, upper=5).astype(int).values - 1)
    observed[ORDINAL_ATTR] = jnp.asarray(v, dtype=jnp.int32)
    mask[ORDINAL_ATTR] = jnp.asarray(m)

    return {"iso3": iso3, "observed": observed, "mask": mask, "n": n}


# ─── NumPyro model ─────────────────────────────────────────────────────────
def model(observed: dict, mask: dict, n: int):
    """Cross-sectional latent-variable model over the six attributes.

    Priors are deliberately diffuse. This is the consistency-test regime
    against the MAUT balanced ranking.
    """
    # Latent overlookedness, one per country. Prior centred at 0, unit scale.
    theta = numpyro.sample("theta", dist.Normal(0.0, 2.0).expand([n]).to_event(1))

    # ── Beta-regressed attributes ──
    for attr in BETA_REGR_ATTRS:
        alpha = numpyro.sample(f"alpha_{attr}", dist.Normal(0.0, 1.0))
        # Sign convention: coverage_shortfall slope is strictly positive.
        if attr == "coverage_shortfall":
            beta = numpyro.sample(
                f"beta_{attr}", dist.TruncatedNormal(1.0, 1.0, low=0.0)
            )
        else:
            beta = numpyro.sample(f"beta_{attr}", dist.Normal(0.5, 1.0))
        phi = numpyro.sample(f"phi_{attr}", dist.HalfNormal(10.0))
        mu = jax.nn.sigmoid(alpha + beta * theta)
        # Beta(mu*phi, (1-mu)*phi) requires both parameters > 0; phi > 0 from prior.
        with handlers.mask(mask=mask[attr]):
            numpyro.sample(
                f"obs_{attr}",
                dist.Beta(mu * phi, (1.0 - mu) * phi),
                obs=observed[attr],
            )

    # ── Log-normal attribute (per_pin_gap) ──
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

    # ── Ordered logistic (severity_category) ──
    # Cutpoints: four in strictly increasing order.
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


# ─── SVI fit + posterior sampling ──────────────────────────────────────────
def fit(
    inputs: dict,
    model_fn=None,
    num_steps: int = 4000,
    learning_rate: float = 1e-2,
    num_samples: int = 2000,
    seed: int = 0,
) -> dict:
    """Run ADVI, then draw posterior samples from the variational guide.

    Returns a dict with:
        losses:        ELBO trace over SVI steps
        samples:       dict[param_name -> array], posterior samples
        theta_median:  array of per-country posterior medians
        theta_ci90:    (2, n) array of 5th / 95th percentiles
    """
    if model_fn is None:
        model_fn = model
    numpyro.set_platform("cpu")
    rng = jax.random.PRNGKey(seed)
    # init_to_median + tight init_scale avoids the float-overflow blowup that
    # AutoNormal's default init_to_uniform produces once enough slopes are
    # positivity-constrained (HalfNormal/TruncatedNormal). See Day-2 notes.
    guide = AutoNormal(model_fn, init_loc_fn=init_to_median, init_scale=0.01)
    svi = SVI(model_fn, guide, numpyro.optim.Adam(learning_rate), Trace_ELBO())

    t0 = time.time()
    svi_result = svi.run(
        rng,
        num_steps,
        observed=inputs["observed"],
        mask=inputs["mask"],
        n=inputs["n"],
        progress_bar=False,
    )
    elapsed = time.time() - t0

    # Sample from the fitted guide
    predictive = numpyro.infer.Predictive(
        guide, params=svi_result.params, num_samples=num_samples
    )
    samples = predictive(
        jax.random.PRNGKey(seed + 1),
        observed=inputs["observed"],
        mask=inputs["mask"],
        n=inputs["n"],
    )
    theta_samples = np.asarray(samples["theta"])  # (num_samples, n)
    theta_median = np.median(theta_samples, axis=0)
    theta_ci90 = np.percentile(theta_samples, [5, 95], axis=0)

    return {
        "losses": np.asarray(svi_result.losses),
        "samples": samples,
        "theta_median": theta_median,
        "theta_ci90": theta_ci90,
        "elapsed_sec": elapsed,
    }


# ─── Smoke test entry ──────────────────────────────────────────────────────
def main() -> int:
    print("[mvp] loading enriched frame…")
    df = features.load_cached_enriched_frame()
    if df is None:
        df = features.build_enriched_frame()

    # Keep only countries scored under the existing MAUT (i.e., completeness >= 0.5)
    eligible = df["completeness"].fillna(0) >= 0.5
    scored = df[eligible].copy()

    # Restrict to columns we need
    cols = BETA_REGR_ATTRS + [LOGNORMAL_ATTR, ORDINAL_ATTR, "gap_score_balanced", "completeness"]
    scored = scored[cols]

    # Drop rows where ALL six attributes are missing (safety: shouldn't happen post-eligibility)
    attrs = BETA_REGR_ATTRS + [LOGNORMAL_ATTR, ORDINAL_ATTR]
    obs_count = scored[attrs].notna().sum(axis=1)
    scored = scored[obs_count >= 1]

    print(f"[mvp] scored pool: {len(scored)} countries")
    print(f"[mvp] completeness: min={scored['completeness'].min():.2f} "
          f"mean={scored['completeness'].mean():.2f}")

    inputs = prepare_inputs(scored)
    print(f"[mvp] prepared arrays: n={inputs['n']}")
    for attr in BETA_REGR_ATTRS + [LOGNORMAL_ATTR, ORDINAL_ATTR]:
        n_obs = int(np.asarray(inputs["mask"][attr]).sum())
        print(f"        {attr:<24s}  observed in {n_obs}/{inputs['n']}")

    print("[mvp] fitting SVI (4000 steps)…")
    res = fit(inputs, num_steps=4000)
    print(f"[mvp] SVI took {res['elapsed_sec']:.1f}s · "
          f"final ELBO loss = {res['losses'][-1]:.2f}")

    # ── Consistency check against the existing MAUT ranking ──
    iso = inputs["iso3"]
    theta_med = res["theta_median"]
    maut = scored["gap_score_balanced"].values

    # Higher theta → more overlooked; MAUT balanced score likewise (larger = more overlooked)
    valid = ~np.isnan(maut)
    rho, _ = spearmanr(theta_med[valid], maut[valid])
    print()
    print(f"[mvp] Spearman ρ  (theta median vs MAUT balanced): {rho:+.3f}")

    # ── Top-10 comparison ──
    bay_rank = pd.Series(-theta_med, index=iso).rank(method="average")
    maut_rank = pd.Series(-maut, index=iso).rank(method="average")
    top10_bay = bay_rank.nsmallest(10).index.tolist()
    top10_maut = maut_rank.nsmallest(10).index.tolist()
    overlap = len(set(top10_bay) & set(top10_maut))
    print(f"[mvp] top-10 overlap (Bayesian vs MAUT balanced): {overlap}/10")

    print()
    print("[mvp] Bayesian top-10:")
    lo, hi = res["theta_ci90"]
    for iso3 in top10_bay:
        i = inputs["iso3"].index(iso3)
        print(f"  {iso3}   median θ = {theta_med[i]:+.3f}   "
              f"90 % CI = [{lo[i]:+.3f}, {hi[i]:+.3f}]   "
              f"completeness = {scored.loc[iso3, 'completeness']:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
