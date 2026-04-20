"""6-attribute vs 7-attribute LVM ablation.

Fits the canonical 6-attribute Bayesian hierarchical model and the same
model with a 7th sign-constrained Gaussian attribute — the learned
severity momentum scalar from analysis/learned/train_momentum.py.
Reports CERF UFE and CARE BTS precision @ k for both fits on the HRP-
eligible pool.

No theta leakage: the Mamba encoder that produces the 7th attribute is
trained only on the monthly INFORM panel, never on theta itself. The
ablation isolates whether adding the learned scalar as a new L4
attribute moves the LVM's external-benchmark agreement at all.
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro import handlers
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoMultivariateNormal
from numpyro.infer.initialization import init_to_median

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from bayesian.mvp import (  # noqa: E402
    BETA_REGR_ATTRS, LOGNORMAL_ATTR, ORDINAL_ATTR, ORDINAL_LEVELS,
    prepare_inputs,
)
from bayesian.hierarchical import model as model_6attr_hier  # noqa: E402
import features as features_mod  # noqa: E402
import validation as val  # noqa: E402

LEARNED_PARQUET = ROOT / "Data" / "learned" / "severity_momentum.parquet"


# ════════════════════════════════════════════════════════════════════════════
# 7-attribute model
# ════════════════════════════════════════════════════════════════════════════
def prepare_inputs_with_learned(df: pd.DataFrame, learned: pd.Series) -> dict:
    """prepare_inputs + a standardised 'severity_momentum_learned' scalar."""
    base = prepare_inputs(df)
    iso3 = df.index.astype(str).tolist()
    col = learned.reindex(iso3).astype(float)
    m = col.notna().values
    mu, sd = col[m].mean(), col[m].std() if m.any() else 1.0
    sd = sd if sd and sd > 1e-6 else 1.0
    v = ((col - mu) / sd).fillna(0.0).values
    base["observed"]["severity_momentum_learned"] = jnp.asarray(v, dtype=jnp.float32)
    base["mask"]["severity_momentum_learned"] = jnp.asarray(m)
    return base


def model_7attr_hier(observed: dict, mask: dict, n: int):
    """Hierarchical 6-attribute model + Gaussian likelihood for the learned scalar.

    Slope beta_7 is sign-constrained (> 0) under a half-normal prior — the
    same discipline as the other slopes. This makes the 7th attribute move
    with theta in a declared direction: higher predicted momentum → more
    overlooked.
    """
    # Hierarchical prior on theta — matches hierarchical.model exactly
    mu_theta = numpyro.sample("mu_theta", dist.Normal(0.0, 0.3))
    sigma_theta = numpyro.sample("sigma_theta", dist.HalfNormal(0.5))
    theta = numpyro.sample(
        "theta",
        dist.Normal(loc=mu_theta, scale=sigma_theta).expand([n]).to_event(1),
    )

    # Diffuse stakeholder prior means (matches STAKEHOLDER_PRIORS['diffuse'])
    DIFFUSE = {
        "coverage_shortfall": 1.0, "need_intensity": 1.0, "donor_hhi": 1.0,
        "cluster_gini": 1.0, "per_pin_gap": 1.0, "severity_category": 1.0,
    }

    # Beta-regressed attributes
    for attr in BETA_REGR_ATTRS:
        alpha = numpyro.sample(f"alpha_{attr}", dist.Normal(0.0, 1.0))
        beta = numpyro.sample(f"beta_{attr}", dist.HalfNormal(DIFFUSE[attr]))
        phi = numpyro.sample(f"phi_{attr}", dist.HalfNormal(10.0))
        mu = jax.nn.sigmoid(alpha + beta * theta)
        with handlers.mask(mask=mask[attr]):
            numpyro.sample(f"obs_{attr}",
                           dist.Beta(mu * phi, (1.0 - mu) * phi),
                           obs=observed[attr])

    # Log-normal
    alpha_ln = numpyro.sample("alpha_per_pin_gap", dist.Normal(4.0, 2.0))
    beta_ln = numpyro.sample("beta_per_pin_gap", dist.HalfNormal(DIFFUSE[LOGNORMAL_ATTR]))
    sigma_ln = numpyro.sample("sigma_per_pin_gap", dist.HalfNormal(1.0))
    loc_ln = alpha_ln + beta_ln * theta
    with handlers.mask(mask=mask[LOGNORMAL_ATTR]):
        numpyro.sample(f"obs_{LOGNORMAL_ATTR}",
                       dist.LogNormal(loc=loc_ln, scale=sigma_ln),
                       obs=observed[LOGNORMAL_ATTR])

    # Ordinal
    cut_base = numpyro.sample("cut_base", dist.Normal(0.0, 2.0))
    cut_delta = numpyro.sample(
        "cut_delta", dist.HalfNormal(1.5).expand([ORDINAL_LEVELS - 2]).to_event(1))
    cutpoints = jnp.concatenate([cut_base[None], cut_base + jnp.cumsum(cut_delta)])
    alpha_sev = numpyro.sample("alpha_severity_category", dist.Normal(0.0, 1.0))
    beta_sev = numpyro.sample("beta_severity_category", dist.HalfNormal(DIFFUSE[ORDINAL_ATTR]))
    predictor = alpha_sev + beta_sev * theta
    with handlers.mask(mask=mask[ORDINAL_ATTR]):
        numpyro.sample(f"obs_{ORDINAL_ATTR}",
                       dist.OrderedLogistic(predictor, cutpoints=cutpoints),
                       obs=observed[ORDINAL_ATTR])

    # ── 7th attribute: learned severity momentum ──
    alpha_m = numpyro.sample("alpha_momentum", dist.Normal(0.0, 1.0))
    beta_m = numpyro.sample("beta_momentum", dist.HalfNormal(1.0))   # sign-constrained > 0
    sigma_m = numpyro.sample("sigma_momentum", dist.HalfNormal(1.0))
    mu_m = alpha_m + beta_m * theta
    with handlers.mask(mask=mask["severity_momentum_learned"]):
        numpyro.sample("obs_severity_momentum_learned",
                       dist.Normal(loc=mu_m, scale=sigma_m),
                       obs=observed["severity_momentum_learned"])


# ════════════════════════════════════════════════════════════════════════════
# Fit helper
# ════════════════════════════════════════════════════════════════════════════
def fit_svi(model_fn, inputs, seed: int = 0, num_steps: int = 8000,
            num_samples: int = 2000, lr: float = 3e-3):
    numpyro.set_platform("cpu")
    rng = jax.random.PRNGKey(seed)
    guide = AutoMultivariateNormal(model_fn, init_loc_fn=init_to_median, init_scale=0.01)
    svi = SVI(model_fn, guide, numpyro.optim.Adam(lr), Trace_ELBO())
    result = svi.run(rng, num_steps,
                     observed=inputs["observed"], mask=inputs["mask"], n=inputs["n"],
                     progress_bar=False)
    predictive = numpyro.infer.Predictive(guide, params=result.params, num_samples=num_samples)
    samples = predictive(jax.random.PRNGKey(seed + 1),
                         observed=inputs["observed"], mask=inputs["mask"], n=inputs["n"])
    theta = np.asarray(samples["theta"])
    return np.median(theta, axis=0), np.percentile(theta, [5, 95], axis=0)


# ════════════════════════════════════════════════════════════════════════════
# Evaluation
# ════════════════════════════════════════════════════════════════════════════
def precision_at_k(ranks: pd.Series, bench: set, k: int) -> tuple[int, float]:
    top = set(ranks.nsmallest(k).index.astype(str))
    hit = len(top & bench)
    return hit, hit / k


def main() -> None:
    enriched = features_mod.load_cached_enriched_frame()
    if enriched is None:
        enriched = features_mod.build_enriched_frame()

    # HRP-eligible pool: those with observed per_pin_gap
    hrp = enriched[enriched["per_pin_gap"].notna()].copy()
    print(f"HRP-eligible pool: {len(hrp)} countries")

    # Load the learned L4 feature
    learned = pd.read_parquet(LEARNED_PARQUET)["severity_momentum_learned"]
    overlap = learned.index.intersection(hrp.index)
    print(f"Learned scalars available for {len(overlap)} / {len(hrp)} HRP countries")

    # ────────────────────────────────────────────────────────────────────────
    # Fit 6-attribute model
    # ────────────────────────────────────────────────────────────────────────
    print("\n── Fitting 6-attribute LVM (hierarchical) ──")
    inputs6 = prepare_inputs(hrp)
    med6, ci6 = fit_svi(model_6attr_hier, inputs6)
    theta6 = pd.Series(med6, index=hrp.index, name="theta_6attr")

    # ────────────────────────────────────────────────────────────────────────
    # Fit 7-attribute model (with learned severity momentum)
    # ────────────────────────────────────────────────────────────────────────
    print("── Fitting 7-attribute LVM (6 + learned momentum, hierarchical) ──")
    inputs7 = prepare_inputs_with_learned(hrp, learned)
    med7, ci7 = fit_svi(model_7attr_hier, inputs7)
    theta7 = pd.Series(med7, index=hrp.index, name="theta_7attr")

    # Ranks: higher theta = more overlooked → rank by -theta
    ranks6 = (-theta6).rank(method="average")
    ranks7 = (-theta7).rank(method="average")

    # ────────────────────────────────────────────────────────────────────────
    # Benchmarks
    # ────────────────────────────────────────────────────────────────────────
    cerf = val.load_cerf_ufe()
    care = val.load_care_bts()

    def cerf_set(year, window):
        s = cerf[(cerf["year"] == year) & (cerf["window"] == window)]
        return set(s["iso3"].astype(str).unique())

    def care_set(year):
        return set(care[care["year"] == year]["iso3"].astype(str).unique())

    benchmarks = [
        ("CERF UFE 2024 w2", cerf_set(2024, 2), 10),
        ("CERF UFE 2025 w1", cerf_set(2025, 1), 10),
        ("CERF UFE 2025 w2", cerf_set(2025, 2), 7),
        ("CARE BTS 2024",    care_set(2024),    10),
    ]

    print("\n── Precision @ k ──")
    print(f"{'Benchmark':<22} {'k':>3}  {'6-attr':>8}  {'7-attr':>8}  {'Δ':>4}")
    rows = []
    for name, bset, k in benchmarks:
        if not bset:
            continue
        h6, p6 = precision_at_k(ranks6, bset, k)
        h7, p7 = precision_at_k(ranks7, bset, k)
        delta = h7 - h6
        print(f"{name:<22} {k:>3}  {h6}/{k} ({p6:.2f})  {h7}/{k} ({p7:.2f})  {delta:>+4d}")
        rows.append({"benchmark": name, "k": k,
                     "six_hit": h6, "six_precision": p6,
                     "seven_hit": h7, "seven_precision": p7,
                     "delta": delta})

    out = pd.DataFrame(rows)
    out_path = ROOT / "Data" / "learned" / "ablation_results.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path.relative_to(ROOT)}")

    # Top-10 diff for reporting
    top6 = set(ranks6.nsmallest(10).index.astype(str))
    top7 = set(ranks7.nsmallest(10).index.astype(str))
    added = top7 - top6
    dropped = top6 - top7
    print("\n── Top-10 churn (7-attr vs 6-attr) ──")
    print(f"Added   by 7-attr: {sorted(added) or '—'}")
    print(f"Dropped by 7-attr: {sorted(dropped) or '—'}")


if __name__ == "__main__":
    main()
