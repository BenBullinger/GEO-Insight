"""Four stakeholder posteriors — consensus and contested crises.

Fits the hierarchical model four times, once per pre-registered stakeholder
prior (CERF, ECHO, USAID, NGO). Each stakeholder is a different set of
HalfNormal scales on the six attribute slopes — see STAKEHOLDER_PRIORS in
hierarchical.py. The same six observations and the same population prior
on theta apply throughout; only the prior on slopes changes.

For each country, the four posteriors give four medians of theta. We
identify two cells:

  * Consensus crises — countries that appear in the top-10 under all four
    stakeholder posteriors. The data-plus-model identification of these
    countries is robust to stakeholder preferences.

  * Contested crises — countries that appear in the top-10 under some
    stakeholders but not others. Their position depends on whose priors
    you apply.

The comparison figure renders all four posterior medians per country with
their 90 % credible intervals, top-10-under-any-stakeholder highlighted in
oxblood. Saved to landing/methodology/figures/fig_stakeholder_consensus.png.

Usage:
    dashboard/.venv/bin/python -m analysis.bayesian.stakeholders
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.bayesian import hierarchical  # noqa: E402
from analysis.bayesian.mvp import (  # noqa: E402
    BETA_REGR_ATTRS,
    LOGNORMAL_ATTR,
    ORDINAL_ATTR,
    prepare_inputs,
    fit,
)


STAKEHOLDERS = ["cerf", "echo", "usaid", "ngo"]
STAKEHOLDER_LABELS = {
    "cerf":  "CERF",
    "echo":  "ECHO",
    "usaid": "USAID",
    "ngo":   "NGO consortium",
}
# Single-accent palette: oxblood family + one cool counter-accent.
# Distinct hues, accessible against the white background.
STAKEHOLDER_COLORS = {
    "cerf":  "#5C1414",  # deep oxblood
    "echo":  "#C26D4E",  # warm coral
    "usaid": "#B8923B",  # ochre
    "ngo":   "#4A6E8A",  # slate blue
}


def fit_all_stakeholders(inputs: dict, num_steps: int = 8000, seed: int = 0) -> dict:
    """Run the hierarchical model four times, one per stakeholder prior.

    Returns a dict: stakeholder_id -> {theta_median, theta_ci_lo, theta_ci_hi, elapsed}.
    """
    out = {}
    for s in STAKEHOLDERS:
        priors = hierarchical.STAKEHOLDER_PRIORS[s]
        model_fn = hierarchical.make_model(prior_means=priors)
        t0 = time.time()
        res = fit(inputs, model_fn=model_fn, num_steps=num_steps,
                  learning_rate=3e-3, seed=seed)
        elapsed = time.time() - t0
        lo, hi = res["theta_ci90"]
        out[s] = {
            "theta_median": res["theta_median"],
            "theta_ci_lo": lo,
            "theta_ci_hi": hi,
            "elapsed_sec": elapsed,
        }
    return out


def consensus_contested(stakeholder_results: dict, iso: list, k: int = 10) -> dict:
    """Identify consensus and contested crises across the four posteriors."""
    top_per = {}
    for s, r in stakeholder_results.items():
        rank = pd.Series(-r["theta_median"], index=iso).rank(method="average")
        top_per[s] = set(rank.nsmallest(k).index.astype(str))
    all_top = set().union(*top_per.values())
    consensus = sorted(set.intersection(*top_per.values()))
    contested = sorted(c for c in all_top if c not in consensus)
    return {
        "consensus": consensus,
        "contested": contested,
        "any_top": sorted(all_top),
        "per_stakeholder_top": top_per,
    }


def plot_stakeholder_comparison(stakeholder_results: dict, iso: list, out_path: Path) -> None:
    """For each country in the union of top-10s, plot four dots + CIs by stakeholder."""
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams["font.family"] = "sans-serif"
    rcParams["font.size"] = 9.5
    rcParams["axes.spines.top"] = False
    rcParams["axes.spines.right"] = False

    INK = "#222222"
    MUTED = "#999999"

    cc = consensus_contested(stakeholder_results, iso, k=10)
    countries = cc["any_top"]
    # Order countries by mean posterior median across stakeholders (most overlooked first).
    mean_theta = {}
    for c in countries:
        i = iso.index(c)
        mean_theta[c] = float(np.mean([stakeholder_results[s]["theta_median"][i]
                                       for s in STAKEHOLDERS]))
    countries.sort(key=lambda c: mean_theta[c], reverse=True)

    n = len(countries)
    fig, ax = plt.subplots(figsize=(8.4, 0.42 * n + 1.5), constrained_layout=True)

    # Each stakeholder is centered on its OWN posterior median across the 22
    # countries. This is necessary because the absolute scale of theta is
    # not comparable across stakeholders — different prior scales on the
    # slopes shift the latent. Within-stakeholder centering preserves the
    # rank semantics and the CI widths, which is what the panel is about.
    centered = {}
    for s in STAKEHOLDERS:
        r = stakeholder_results[s]
        mu_s = float(np.median(r["theta_median"]))
        centered[s] = {
            "med": r["theta_median"] - mu_s,
            "lo":  r["theta_ci_lo"]  - mu_s,
            "hi":  r["theta_ci_hi"]  - mu_s,
        }

    highlight_idx = [iso.index(c) for c in countries]
    h_lo = np.concatenate([centered[s]["lo"][highlight_idx] for s in STAKEHOLDERS])
    h_hi = np.concatenate([centered[s]["hi"][highlight_idx] for s in STAKEHOLDERS])
    pad = 0.05
    x_min = h_lo.min() - pad
    x_max = h_hi.max() + 0.20  # space for country labels
    label_x = h_hi.max() + 0.04

    y_positions = np.arange(n)[::-1]  # top of the chart = most overlooked
    OFFSET = 0.18  # vertical spacing between the four stakeholders within a row

    for ci, c in enumerate(countries):
        i = iso.index(c)
        y_base = y_positions[ci]
        for si, s in enumerate(STAKEHOLDERS):
            med = float(centered[s]["med"][i])
            lo  = float(centered[s]["lo"][i])
            hi  = float(centered[s]["hi"][i])
            y = y_base + (si - 1.5) * OFFSET
            color = STAKEHOLDER_COLORS[s]
            ax.errorbar(
                med, y, xerr=[[med - lo], [hi - med]],
                fmt="o", color=color, ecolor=color, elinewidth=1.2, capsize=0,
                markersize=5.5, markeredgecolor="white", markeredgewidth=0.8, alpha=0.9,
            )
        # Country label + consensus marker
        is_consensus = c in cc["consensus"]
        marker = "✓ " if is_consensus else ""
        ax.text(
            label_x, y_base, f"{marker}{c}",
            va="center", ha="left", fontsize=9.5,
            color=("#5C1414" if is_consensus else INK),
            fontweight="700" if is_consensus else "500",
        )

    ax.set_xlim(x_min, x_max)

    ax.axvline(0, color=MUTED, linewidth=0.8, linestyle="--", alpha=0.6, zorder=1)

    ax.set_yticks([])
    ax.set_xlabel("Posterior median, recentered (less overlooked ←  → more overlooked)",
                  fontsize=9, color=INK)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8)
    for s in ax.spines.values():
        s.set_color(MUTED)

    # Legend
    handles = []
    for s in STAKEHOLDERS:
        handles.append(plt.Line2D([0], [0], marker="o", color="w",
                                   markerfacecolor=STAKEHOLDER_COLORS[s],
                                   markeredgecolor="white", markeredgewidth=0.8,
                                   markersize=7, label=STAKEHOLDER_LABELS[s]))
    ax.legend(
        handles=handles, loc="lower right", frameon=False,
        fontsize=8.5, ncol=4, bbox_to_anchor=(1.0, -0.16),
    )

    ax.set_title(
        f"Posterior medians under four stakeholder priors  ·  "
        f"✓ = top-10 under all four (consensus)",
        loc="left", fontsize=10.5, color=INK, pad=10,
    )

    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    import features

    print("[stakeholders] loading enriched frame…")
    df = features.load_cached_enriched_frame()
    if df is None:
        df = features.build_enriched_frame()

    cols = BETA_REGR_ATTRS + [LOGNORMAL_ATTR, ORDINAL_ATTR, "completeness"]
    scored = df[df["per_pin_gap"].notna()][cols].copy()
    inputs = prepare_inputs(scored)
    print(f"[stakeholders] HRP-eligible pool: {inputs['n']} countries")
    print()

    print("[stakeholders] fitting four stakeholder posteriors…")
    results = fit_all_stakeholders(inputs)
    for s in STAKEHOLDERS:
        print(f"  {s:<7s}: {results[s]['elapsed_sec']:.1f}s")
    print()

    iso = inputs["iso3"]
    cc = consensus_contested(results, iso, k=10)
    print(f"[stakeholders] consensus crises (top-10 under all four): "
          f"{len(cc['consensus'])}")
    print(f"  {cc['consensus']}")
    print()
    print(f"[stakeholders] contested crises (top-10 under some, not all): "
          f"{len(cc['contested'])}")
    print(f"  {cc['contested']}")
    print()

    print("[stakeholders] per-stakeholder top-10:")
    for s, top in cc["per_stakeholder_top"].items():
        print(f"  {s:<7s}: {sorted(top)}")
    print()

    out_path = Path(__file__).resolve().parents[2] / "landing" / "methodology" / "figures" / "fig_stakeholder_consensus.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_stakeholder_comparison(results, iso, out_path)
    print(f"[stakeholders] wrote {out_path.relative_to(Path.cwd())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
