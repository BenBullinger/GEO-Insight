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
red. Saved to landing/methodology/figures/fig_stakeholder_consensus.png.

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
# Single-accent palette: deep-red family + one cool counter-accent.
# Distinct hues, accessible against the white background.
STAKEHOLDER_COLORS = {
    "cerf":  "#5C1414",  # deep red
    "echo":  "#C26D4E",  # warm coral
    "usaid": "#B8923B",  # ochre
    "ngo":   "#4A6E8A",  # slate blue
}


def fit_all_stakeholders(inputs: dict, num_steps: int = 8000, seed: int = 0) -> dict:
    """Run the hierarchical model four times, one per stakeholder prior.

    Returns a dict: stakeholder_id -> {theta_median, theta_ci_lo, theta_ci_hi,
    theta_samples, elapsed}. theta_samples has shape (num_posterior_samples, n)
    and is what the overlaid-density plots consume.
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
            "theta_samples": np.asarray(res["samples"]["theta"]),  # (S, n)
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



def _contestation_scores(stakeholder_results: dict, iso: list, countries: list) -> dict:
    """Per-country contestation: the range of centered posterior medians across
    the four stakeholders. Larger range = more contested."""
    per_stakeholder_centre = {
        s: float(np.median(stakeholder_results[s]["theta_median"]))
        for s in STAKEHOLDERS
    }
    out = {}
    for c in countries:
        i = iso.index(c)
        centred = [
            float(stakeholder_results[s]["theta_median"][i]) - per_stakeholder_centre[s]
            for s in STAKEHOLDERS
        ]
        out[c] = max(centred) - min(centred)
    return out


def plot_top10_consensus(
    stakeholder_results: dict, iso: list, countries: list, out_path: Path
) -> None:
    """Single figure: each of the top-10 countries gets a horizontal row.
    Eight rows show four-stakeholder CI bars (1D). The most- and least-contested
    countries get taller rows with overlaid posterior density curves (2D KDE)
    to visualise the stakeholder disagreement structurally.

    The top-10 is taken from the canonical (diffuse-prior) Bayesian production
    fit, re-ordered here by average stakeholder posterior median so the most
    overlooked is at the top.
    """
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.gridspec import GridSpec
    from scipy.stats import gaussian_kde

    rcParams["font.family"] = "sans-serif"
    rcParams["font.size"] = 9.5
    rcParams["axes.spines.top"] = False
    rcParams["axes.spines.right"] = False
    rcParams["axes.spines.left"] = False

    INK = "#222222"
    MUTED = "#888888"

    per_stakeholder_centre = {
        s: float(np.median(stakeholder_results[s]["theta_median"]))
        for s in STAKEHOLDERS
    }

    # Centred posteriors for every country × stakeholder
    centred = {}
    for c in countries:
        i = iso.index(c)
        centred[c] = {}
        for s in STAKEHOLDERS:
            centred[c][s] = {
                "med": float(stakeholder_results[s]["theta_median"][i]) - per_stakeholder_centre[s],
                "lo":  float(stakeholder_results[s]["theta_ci_lo"][i])  - per_stakeholder_centre[s],
                "hi":  float(stakeholder_results[s]["theta_ci_hi"][i])  - per_stakeholder_centre[s],
                "samples": stakeholder_results[s]["theta_samples"][:, i] - per_stakeholder_centre[s],
            }

    # Order countries by mean posterior median across stakeholders (most
    # overlooked at the top of the figure).
    countries_ordered = sorted(
        countries,
        key=lambda c: -np.mean([centred[c][s]["med"] for s in STAKEHOLDERS]),
    )

    # Identify most- and least-contested within the top-10
    scores = _contestation_scores(stakeholder_results, iso, countries)
    most_contested = max(scores, key=scores.get)
    least_contested = min(scores, key=scores.get)

    # Shared x-range computed from all CIs across the top-10
    all_lo = []
    all_hi = []
    for c in countries_ordered:
        for s in STAKEHOLDERS:
            all_lo.append(centred[c][s]["lo"])
            all_hi.append(centred[c][s]["hi"])
    pad = 0.04
    x_min = min(all_lo) - pad
    x_max = max(all_hi) + 0.22  # extra right margin for labels

    # Layout: density rows get 4× the height of bar rows
    n = len(countries_ordered)
    row_heights = [
        4.0 if c in (most_contested, least_contested) else 1.0
        for c in countries_ordered
    ]
    fig_h = sum(row_heights) * 0.35 + 1.0
    fig = plt.figure(figsize=(8.5, fig_h), constrained_layout=True)
    gs = GridSpec(n, 1, figure=fig, height_ratios=row_heights, hspace=0.0)

    xs = np.linspace(x_min, x_max, 500)
    OFFSET = 0.18

    for row_idx, c in enumerate(countries_ordered):
        ax = fig.add_subplot(gs[row_idx, 0])
        is_blown = c in (most_contested, least_contested)

        if is_blown:
            # Overlaid KDE curves
            for s in STAKEHOLDERS:
                kde = gaussian_kde(centred[c][s]["samples"])
                ys = kde(xs)
                color = STAKEHOLDER_COLORS[s]
                ax.plot(xs, ys, color=color, linewidth=1.5)
                ax.fill_between(xs, 0, ys, color=color, alpha=0.12)
            ax.set_ylim(bottom=0)
        else:
            # Four horizontal error-bar rows
            for si, s in enumerate(STAKEHOLDERS):
                med = centred[c][s]["med"]
                lo = centred[c][s]["lo"]
                hi = centred[c][s]["hi"]
                y = (si - 1.5) * OFFSET
                color = STAKEHOLDER_COLORS[s]
                ax.errorbar(
                    med, y, xerr=[[med - lo], [hi - med]],
                    fmt="o", color=color, ecolor=color, elinewidth=1.2, capsize=0,
                    markersize=4.8, markeredgecolor="white", markeredgewidth=0.6,
                )
            ax.set_ylim(-0.5, 0.5)

        # Zero line (population centre under each stakeholder's own posterior)
        ax.axvline(0, color=MUTED, linewidth=0.7, linestyle="--", alpha=0.5, zorder=1)

        # Country label on the right + contestation tag where applicable
        tag = ""
        tag_color = INK
        if c == most_contested:
            tag = "  most contested"
            tag_color = STAKEHOLDER_COLORS["usaid"]
        elif c == least_contested:
            tag = "  most consensus"
            tag_color = STAKEHOLDER_COLORS["cerf"]
        y_centre = ax.get_ylim()
        y_label = (y_centre[0] + y_centre[1]) / 2.0
        ax.text(
            x_max - 0.2, y_label, c,
            va="center", ha="left", fontsize=10.5, color=INK, fontweight="700",
        )
        if tag:
            ax.text(
                x_max - 0.2, y_label - (y_centre[1] - y_centre[0]) * 0.18, tag,
                va="center", ha="left",
                fontsize=7.5, color=tag_color, fontweight="600",
                fontstyle="italic",
            )

        ax.set_xlim(x_min, x_max)
        ax.set_yticks([])
        ax.spines["bottom"].set_visible(row_idx == n - 1)
        ax.spines["bottom"].set_color(MUTED)
        if row_idx == n - 1:
            ax.tick_params(axis="x", colors=MUTED, labelsize=8)
            ax.set_xlabel(
                "Posterior of θ, recentered per stakeholder  "
                "(← less overlooked · more overlooked →)",
                fontsize=8.5, color=INK,
            )
        else:
            ax.set_xticks([])

    # Legend — four stakeholder swatches in a compact strip above the plot
    handles = []
    for s in STAKEHOLDERS:
        handles.append(plt.Line2D([0], [0], marker="o", color="w",
                                   markerfacecolor=STAKEHOLDER_COLORS[s],
                                   markeredgecolor="white", markeredgewidth=0.6,
                                   markersize=7, label=STAKEHOLDER_LABELS[s]))
    fig.legend(
        handles=handles, loc="upper center", frameon=False,
        fontsize=9, ncol=4, bbox_to_anchor=(0.5, 1.02),
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

    # Top-10 from the canonical (diffuse-prior) Bayesian production fit,
    # per proposal Table 2. Used as the unified row list for the consensus
    # figure — two of these ten get blown up to 2D densities.
    TOP10 = ["HND", "SLV", "MOZ", "SOM", "GTM", "NER", "HTI", "CMR", "VEN", "TCD"]

    out_path = Path(__file__).resolve().parents[2] / "landing" / "methodology" / "figures" / "fig_stakeholder_consensus.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_top10_consensus(results, iso, TOP10, out_path)
    print(f"[stakeholders] wrote {out_path.relative_to(Path.cwd())}")

    scores = _contestation_scores(results, iso, TOP10)
    print(f"[stakeholders] contestation range (centred medians, max − min):")
    for c, v in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"  {c}: {v:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
