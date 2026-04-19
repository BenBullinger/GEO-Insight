"""Posterior predictive checks for the hierarchical model.

For each of the six observed attributes, draw replicates from the fitted
posterior and compare against the observed values. Three diagnostics
per attribute:

  1. Marginal distribution: posterior predictive density (across all
     country-month-replicate triples) overlaid with the observed histogram.
     A model that systematically misplaces probability mass on one
     attribute is misspecified for that attribute, and any rankings it
     produces inherit the misspecification.

  2. Per-country mean: predicted mean of a_i for each country plotted
     against the observed a_i. On a well-specified model the points
     scatter around y=x with the dispersion implied by the likelihood.

  3. 90 % prediction-interval coverage: fraction of observed values
     that fall within the 5th–95th percentile of the posterior
     predictive samples for that country. A correctly-calibrated model
     covers 90 %; substantial deviations expose mass-misallocation.

Usage:
    dashboard/.venv/bin/python -m analysis.bayesian.ppc
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
from numpyro.infer import Predictive

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.bayesian import hierarchical  # noqa: E402
from analysis.bayesian.mvp import (  # noqa: E402
    BETA_REGR_ATTRS,
    LOGNORMAL_ATTR,
    ORDINAL_ATTR,
    ORDINAL_LEVELS,
    prepare_inputs,
    fit,
)


# Attributes in plot order, with display config
ALL_ATTRS = BETA_REGR_ATTRS + [LOGNORMAL_ATTR, ORDINAL_ATTR]
PLOT_INFO = {
    "coverage_shortfall": {"label": "coverage shortfall",       "support": (0, 1),    "scale": "linear"},
    "need_intensity":     {"label": "need intensity",           "support": (0, 1),    "scale": "linear"},
    "donor_hhi":          {"label": "donor concentration (HHI)", "support": (0, 1),   "scale": "linear"},
    "cluster_gini":       {"label": "intra-crisis equity (Gini)", "support": (0, 1),  "scale": "linear"},
    "per_pin_gap":        {"label": "gap per PIN (USD)",        "support": (0.01, 100), "scale": "log"},
    "severity_category":  {"label": "INFORM severity (1–5)",    "support": (1, 5),    "scale": "linear"},
}


def draw_posterior_predictive(
    inputs: dict, model_fn, num_samples: int = 1000, seed: int = 0
) -> dict:
    """Fit the model and draw posterior predictive samples.

    Returns a dict mapping attribute name to an array of shape
    (num_samples, n_countries) — one predictive draw of a_i per country
    per posterior sample.
    """
    res = fit(inputs, model_fn=model_fn, num_steps=8000, learning_rate=3e-3,
              num_samples=num_samples, seed=seed)
    # Use AutoMultivariateNormal guide we just fitted, embedded in res via
    # the `samples` dict — but the cleanest way to get posterior predictive
    # is to wire a Predictive over the posterior samples back through the
    # model with no `obs=`. We construct that here by stripping the obs.
    posterior_samples = {k: v for k, v in res["samples"].items()
                         if not k.startswith("obs_")}
    pp = Predictive(model_fn, posterior_samples=posterior_samples,
                    num_samples=num_samples)
    # observed=None so NumPyro forward-samples the obs_* sites instead of
    # treating the placeholder values as observations. mask=ones so every
    # country is sampled (we want predictive draws everywhere).
    pp_samples = pp(jax.random.PRNGKey(seed + 100),
                    observed={k: None for k in inputs["observed"]},
                    mask={k: jnp.ones_like(v) for k, v in inputs["mask"].items()},
                    n=inputs["n"])
    out = {}
    for attr in ALL_ATTRS:
        site = f"obs_{attr}"
        if site in pp_samples:
            out[attr] = np.asarray(pp_samples[site])  # (num_samples, n)
    return out


def compute_diagnostics(inputs: dict, pp: dict) -> pd.DataFrame:
    """Per-attribute summary: coverage @ 90, mean predicted vs observed correlation."""
    rows = []
    for attr in ALL_ATTRS:
        if attr not in pp:
            continue
        obs = np.asarray(inputs["observed"][attr])
        mask = np.asarray(inputs["mask"][attr])
        rep = pp[attr]  # (S, n)
        lo = np.percentile(rep, 5, axis=0)
        hi = np.percentile(rep, 95, axis=0)
        within = (obs >= lo) & (obs <= hi)
        coverage = within[mask].mean() if mask.sum() > 0 else float("nan")
        # Predicted vs observed correlation on the masked pool
        mean_pred = rep.mean(axis=0)
        if mask.sum() >= 3:
            r = np.corrcoef(mean_pred[mask], obs[mask])[0, 1]
        else:
            r = float("nan")
        rows.append({
            "attribute": attr,
            "n_observed": int(mask.sum()),
            "coverage_90": coverage,
            "pearson_r": r,
        })
    return pd.DataFrame(rows)


def plot_ppc(inputs: dict, pp: dict, out_path: Path) -> None:
    """2×3 grid of marginal distributions: observed vs posterior predictive."""
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams["font.family"] = "sans-serif"
    rcParams["font.size"] = 9.5
    rcParams["axes.titlesize"] = 10.5
    rcParams["axes.titleweight"] = "regular"
    rcParams["axes.spines.top"] = False
    rcParams["axes.spines.right"] = False

    OXBLOOD = "#7A1F2A"
    INK = "#222222"
    GREY = "#888888"

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.0), constrained_layout=True)

    for ax, attr in zip(axes.flat, ALL_ATTRS):
        info = PLOT_INFO[attr]
        obs = np.asarray(inputs["observed"][attr])
        mask = np.asarray(inputs["mask"][attr])
        obs_obs = obs[mask]
        rep = pp[attr].ravel()  # all replicates flattened

        if attr == ORDINAL_ATTR:
            # Discrete: bar chart of observed vs predictive frequencies
            obs_int = obs_obs.astype(int)
            rep_int = np.clip(np.rint(rep), 0, ORDINAL_LEVELS - 1).astype(int)
            xs = np.arange(ORDINAL_LEVELS)
            obs_freq = np.array([(obs_int == k).mean() for k in xs])
            rep_freq = np.array([(rep_int == k).mean() for k in xs])
            w = 0.4
            ax.bar(xs - w/2, obs_freq, width=w, color=INK, label="observed", alpha=0.85)
            ax.bar(xs + w/2, rep_freq, width=w, color=OXBLOOD, label="predictive", alpha=0.85)
            ax.set_xticks(xs)
            ax.set_xticklabels([str(k + 1) for k in xs])
        elif info["scale"] == "log":
            # Log-x histograms (per_pin_gap)
            bins = np.logspace(np.log10(info["support"][0]),
                               np.log10(info["support"][1]), 30)
            ax.hist(rep[(rep > 0) & (rep < info["support"][1] * 5)],
                    bins=bins, color=OXBLOOD, alpha=0.55, density=True,
                    label="predictive")
            ax.hist(obs_obs, bins=bins, color=INK, alpha=0.85, density=True,
                    histtype="step", linewidth=1.6, label="observed")
            ax.set_xscale("log")
        else:
            # Linear histogram in [0, 1]
            bins = np.linspace(0, 1, 30)
            ax.hist(rep, bins=bins, color=OXBLOOD, alpha=0.55, density=True,
                    label="predictive")
            ax.hist(obs_obs, bins=bins, color=INK, alpha=0.85, density=True,
                    histtype="step", linewidth=1.6, label="observed")

        ax.set_title(info["label"], color=INK, loc="left")
        ax.tick_params(colors=GREY, labelsize=8.5)
        for s in ax.spines.values():
            s.set_color(GREY)
        if ax is axes.flat[0]:
            ax.legend(frameon=False, fontsize=8.5, loc="best")

    fig.suptitle(
        "Posterior predictive check — observed vs simulated marginals "
        f"(n = {inputs['n']} HRP-eligible countries)",
        fontsize=11.5, color=INK, y=1.02
    )
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    import features

    print("[ppc] loading enriched frame…")
    df = features.load_cached_enriched_frame()
    if df is None:
        df = features.build_enriched_frame()

    cols = ALL_ATTRS + ["completeness"]
    scored = df[df["per_pin_gap"].notna()][cols].copy()
    print(f"[ppc] HRP-eligible pool: {len(scored)} countries")

    inputs = prepare_inputs(scored)

    print("[ppc] fitting + drawing posterior predictive samples…")
    t0 = time.time()
    pp = draw_posterior_predictive(inputs, hierarchical.model, num_samples=1000)
    print(f"[ppc] done in {time.time() - t0:.1f}s")

    print()
    print("[ppc] per-attribute diagnostics:")
    diag = compute_diagnostics(inputs, pp)
    print(diag.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()
    print(
        "       coverage_90 — fraction of observed values within the model's "
        "90% predictive interval (target ≈ 0.90)."
    )
    print(
        "       pearson_r   — corr(predicted mean, observed) on the masked "
        "pool (target close to 1)."
    )

    out_path = Path(__file__).resolve().parents[2] / "proposal" / "figures" / "fig_ppc.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print()
    print(f"[ppc] writing figure to {out_path.relative_to(Path.cwd())}")
    plot_ppc(inputs, pp, out_path)
    print("[ppc] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
