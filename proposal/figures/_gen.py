"""Pedagogical figures for the methodology page.

Each figure is standalone and rebuilds from this script. Run:
    dashboard/.venv/bin/python proposal/figures/_gen.py
Output: proposal/figures/*.png at print-quality DPI.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "Data"

# ─── Design tokens (mirror landing/shared.css) ───────────────────────────
INK         = "#111111"
TEXT        = "#333333"
MUTED       = "#555555"
SUBTLE      = "#888888"
FAINT       = "#bcbcbc"
PALE        = "#e8e8e8"
SURFACE     = "#fafafa"
ACCENT      = "#7c1d1d"   # oxblood
ACCENT_2    = "#a13636"   # redder
ACCENT_3    = "#c9554f"   # rose
ACCENT_WASH = "#fcf5f5"

# Distinct-enough stakeholder colours: accent family + one teal counterpoint
# used only where discriminability is essential.
STAKE_COLS = {
    "CERF":  "#5c1414",
    "ECHO":  "#c9554f",
    "USAID": "#e8a76d",
    "NGO":   "#7a8fb8",
}

mpl.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size":         10,
    "axes.edgecolor":    PALE,
    "axes.linewidth":    0.8,
    "axes.labelcolor":   MUTED,
    "axes.labelsize":    9,
    "axes.titlecolor":   INK,
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "axes.titlepad":     10,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "xtick.labelsize":   8.5,
    "ytick.labelsize":   8.5,
    "xtick.major.size":  2,
    "ytick.major.size":  2,
    "grid.color":        PALE,
    "grid.linewidth":    0.4,
    "legend.frameon":    False,
    "legend.fontsize":   8.5,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.dpi":       180,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
})


def strip(ax, y_grid=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PALE)
    ax.tick_params(length=2)
    if y_grid:
        ax.grid(axis="y", linestyle="-", alpha=0.5)


# ═══════════════════════════════════════════════════════════════════════
# Figure 1 — Small-multiples of funding coverage, 2019–2025.
# Six active crises as six little panels. Same axes. The red shaded region
# below 50 % tells the story: most of them end in the red by 2024/25.
# ═══════════════════════════════════════════════════════════════════════
def fig1_coverage_collapse(out: Path) -> None:
    try:
        fts = pd.read_csv(DATA / "fts" / "fts_requirements_funding_global.csv", skiprows=[1])
    except FileNotFoundError:
        print("fig1: FTS data not found; skipping"); return
    fts["year"] = pd.to_numeric(fts["year"], errors="coerce")
    fts = fts.dropna(subset=["year"])
    agg = fts.groupby(["countryCode", "year"], as_index=False)[["requirements", "funding"]].sum()
    agg["coverage"] = (agg["funding"] / agg["requirements"]).clip(upper=1.2) * 100

    picks = [
        ("SDN", "Sudan"),
        ("YEM", "Yemen"),
        ("SOM", "Somalia"),
        ("AFG", "Afghanistan"),
        ("HTI", "Haiti"),
        ("SSD", "South Sudan"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(9.0, 4.6), sharex=True, sharey=True)
    for ax, (iso, name) in zip(axes.flat, picks):
        grp = agg[(agg["countryCode"] == iso) & (agg["year"].between(2019, 2025))].sort_values("year")
        if grp.empty:
            continue
        ax.axhspan(0, 50, facecolor=ACCENT_WASH, alpha=1.0, zorder=0)
        ax.axhline(50, color=ACCENT, linewidth=0.7, linestyle=(0, (4, 3)), zorder=1)
        ax.plot(grp["year"], grp["coverage"], color=ACCENT, linewidth=1.8, zorder=3)
        ax.scatter([grp["year"].iloc[-1]], [grp["coverage"].iloc[-1]],
                   s=28, color=ACCENT, zorder=4, edgecolor="white", linewidths=1.1)
        last_cov = grp["coverage"].iloc[-1]
        ax.set_title(name, loc="left", pad=6, fontsize=10.5, color=INK, fontweight="bold")
        ax.text(0.97, 0.92, f"{last_cov:.0f} % in 2025",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8.2, color=ACCENT, fontweight="bold")
        ax.set_xlim(2018.7, 2025.3)
        ax.set_xticks([2019, 2021, 2023, 2025])
        ax.set_ylim(0, 110)
        ax.set_yticks([0, 50, 100])
        ax.set_yticklabels(["0 %", "50 %", "100 %"])
        strip(ax)

    fig.suptitle("Funding coverage has crossed into the shaded zone for most active crises",
                 x=0.02, y=0.99, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.text(0.02, -0.02,
             "Shaded region: below 50 % coverage (severely underfunded). Source: OCHA FTS, country-year aggregates.",
             fontsize=8.5, color=SUBTLE, ha="left")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(out)
    plt.close(fig)
    print(f"wrote {out.name}")


# ═══════════════════════════════════════════════════════════════════════
# Figure 2 — The observation model, four likelihood families.
# ═══════════════════════════════════════════════════════════════════════
def fig2_observation_model(out: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.0))
    theta_colors = {-1: FAINT, 0: ACCENT_3, 1: ACCENT}
    theta_labels = {-1: r"low $\theta$", 0: r"mid $\theta$", 1: r"high $\theta$"}

    # (a) Beta regression
    ax = axes[0]
    alpha, beta, phi = -0.5, 1.5, 12
    x = np.linspace(0.001, 0.999, 400)
    for th in (-1, 0, 1):
        mu = 1 / (1 + np.exp(-(alpha + beta * th)))
        a, b = mu * phi, (1 - mu) * phi
        ax.plot(x, stats.beta.pdf(x, a, b), color=theta_colors[th], linewidth=1.7, label=theta_labels[th])
    ax.set_xlabel("observed coverage shortfall")
    ax.set_ylabel("likelihood")
    ax.set_title("(a) Beta regression  ·  bounded fractions", loc="left")
    ax.legend(loc="upper left", handlelength=1.4, frameon=False)
    ax.set_xlim(0, 1); ax.set_ylim(bottom=0); ax.set_yticks([])
    strip(ax, y_grid=False)

    # (b) Log-normal
    ax = axes[1]
    alpha_ln, beta_ln, sigma_ln = 4.0, 0.9, 0.7
    x = np.linspace(1, 600, 400)
    for th in (-1, 0, 1):
        mu = alpha_ln + beta_ln * th
        ax.plot(x, stats.lognorm.pdf(x, s=sigma_ln, scale=np.exp(mu)),
                color=theta_colors[th], linewidth=1.7, label=theta_labels[th])
    ax.set_xlabel("observed gap per person in need  (USD)")
    ax.set_title("(b) Log-normal  ·  positive reals", loc="left")
    ax.legend(loc="upper right", handlelength=1.4, frameon=False)
    ax.set_xlim(0, 600); ax.set_ylim(bottom=0); ax.set_yticks([])
    strip(ax, y_grid=False)

    # (c) Ordered probit
    ax = axes[2]
    cutpoints = np.array([-1.5, -0.5, 0.5, 1.5])
    thetas = np.linspace(-2.5, 2.5, 5)
    beta4, alpha4 = 1.0, 0.0
    bottoms = np.zeros(len(thetas))
    cat_cols = ["#f0d3d3", "#d69595", "#b75858", "#8a2f2f", "#5c1414"]
    for k in range(5):
        lo = cutpoints[k - 1] if k > 0 else -np.inf
        hi = cutpoints[k] if k < 4 else np.inf
        loc = alpha4 + beta4 * thetas
        prob = stats.norm.cdf(hi - loc) - stats.norm.cdf(lo - loc)
        bars = ax.bar(thetas, prob, bottom=bottoms, color=cat_cols[k], width=0.55,
                      edgecolor="white", linewidth=0.5)
        bottoms += prob
    # Category labels to the right of the rightmost bar (stacked probabilities)
    r_probs = []
    loc_r = alpha4 + beta4 * thetas[-1]
    for k in range(5):
        lo = cutpoints[k - 1] if k > 0 else -np.inf
        hi = cutpoints[k] if k < 4 else np.inf
        r_probs.append(stats.norm.cdf(hi - loc_r) - stats.norm.cdf(lo - loc_r))
    cum = 0.0
    for k, p in enumerate(r_probs):
        mid = cum + p / 2
        if p >= 0.06:
            ax.text(thetas[-1] + 0.45, mid, f"cat {k+1}",
                    va="center", fontsize=7.8, color=cat_cols[k], fontweight="bold")
        cum += p
    ax.set_xlabel(r"latent $\theta$")
    ax.set_ylabel("P(category)")
    ax.set_title("(c) Ordered probit  ·  $\{1,2,3,4,5\}$ severity", loc="left")
    ax.set_xticks(thetas)
    ax.set_xticklabels([f"{t:g}" for t in thetas])
    ax.set_ylim(0, 1); ax.set_xlim(-3.2, 3.6)
    strip(ax, y_grid=False)

    # (d) Missingness widens posterior
    ax = axes[3]
    x = np.linspace(-2.5, 2.5, 400)
    narrow = stats.norm.pdf(x, loc=0.9, scale=0.35)
    wide = stats.norm.pdf(x, loc=0.7, scale=0.9)
    ax.fill_between(x, 0, narrow, color=ACCENT, alpha=0.18, linewidth=0)
    ax.plot(x, narrow, color=ACCENT, linewidth=1.7, label="6 signals observed")
    ax.fill_between(x, 0, wide, color=SUBTLE, alpha=0.15, linewidth=0)
    ax.plot(x, wide, color=SUBTLE, linewidth=1.7, linestyle=(0, (3, 2)),
            label="3 signals observed")
    ax.set_xlabel(r"posterior on $\theta$ for one country")
    ax.set_title("(d) Missingness widens posterior", loc="left")
    ax.legend(loc="upper left", handlelength=1.6, frameon=False)
    ax.set_ylim(bottom=0); ax.set_yticks([])
    strip(ax, y_grid=False)

    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)
    print(f"wrote {out.name}")


# ═══════════════════════════════════════════════════════════════════════
# Figure 3 — AR(1) trajectory with posterior CI band.
# ═══════════════════════════════════════════════════════════════════════
def fig3_ar1(out: Path) -> None:
    rng = np.random.default_rng(7)
    T = 60
    t = np.arange(T)
    mu_u, rho_u, tau_u = 0.7, 0.92, 0.11
    theta = np.zeros(T); theta[0] = mu_u
    for k in range(1, T):
        theta[k] = mu_u + rho_u * (theta[k-1] - mu_u) + rng.normal(0, tau_u)
    ci_upper = theta + 0.22
    ci_lower = theta - 0.22

    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    ax.fill_between(t, ci_lower, ci_upper, color=ACCENT, alpha=0.1, linewidth=0, zorder=1)
    ax.axhline(mu_u, color=ACCENT, linestyle=(0, (4, 2.5)), linewidth=1.0, alpha=0.75, zorder=2)

    # Acute spike shading
    above = theta > mu_u
    ax.fill_between(t, theta, mu_u, where=above, color=ACCENT, alpha=0.18, linewidth=0, zorder=2)
    ax.fill_between(t, theta, mu_u, where=~above, color=SUBTLE, alpha=0.10, linewidth=0, zorder=2)

    ax.plot(t, theta, color=INK, linewidth=1.4, zorder=3)

    # Chronic baseline label on left, inside plot, not overlapping the line
    ax.text(1.5, mu_u + 0.04, r"chronic baseline $\mu_u$",
            va="bottom", ha="left", color=ACCENT, fontsize=9, fontweight="bold")

    # Acute deviation annotation on the right side spike
    peak = int(np.argmax(theta))
    ax.annotate("acute deviation",
                xy=(peak, theta[peak]), xytext=(peak + 7, theta[peak] - 0.15),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.7,
                                connectionstyle="arc3,rad=0.2"),
                fontsize=9, color=MUTED, fontweight="bold")

    # 90% CI label
    ax.text(T - 1.5, ci_upper[-1] + 0.05, "90 % CI",
            va="bottom", ha="right", color=ACCENT_3, fontsize=8.5, fontweight="bold")

    ax.set_xlabel("month")
    ax.set_ylabel(r"latent $\theta(u, t)$")
    ax.set_title(r"One country's latent trajectory — chronic baseline + acute deviations",
                 loc="left", pad=14)
    ax.set_xlim(0, T - 1)
    ax.set_ylim(ci_lower.min() - 0.08, ci_upper.max() + 0.18)
    strip(ax)
    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)
    print(f"wrote {out.name}")


# ═══════════════════════════════════════════════════════════════════════
# Figure 4 — Stakeholder priors over a single slope.
# ═══════════════════════════════════════════════════════════════════════
def fig4_stakeholder_priors(out: Path) -> None:
    stake = [
        ("CERF",  1.50, 0.30, STAKE_COLS["CERF"]),
        ("ECHO",  1.25, 0.35, STAKE_COLS["ECHO"]),
        ("USAID", 1.00, 0.40, STAKE_COLS["USAID"]),
        ("NGO",   1.00, 0.42, STAKE_COLS["NGO"]),
    ]
    x = np.linspace(-0.5, 3.0, 400)
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    for name, m, s, c in stake:
        y = stats.norm.pdf(x, loc=m, scale=s)
        ax.plot(x, y, color=c, linewidth=1.8, label=name)
        ax.fill_between(x, 0, y, color=c, alpha=0.08, linewidth=0)
    ax.axvline(0, color=FAINT, linewidth=0.7)
    ax.text(0, 1.3, "no effect", ha="center", va="bottom", color=SUBTLE, fontsize=8.5)
    ax.set_xlabel(r"slope $\beta_{\text{coverage}}$  —  how strongly coverage shortfall reflects $\theta$")
    ax.set_ylabel("prior density")
    ax.set_title(r"Each stakeholder is a prior over the attribute slopes",
                 loc="left", pad=14)
    ax.legend(loc="upper right", handlelength=1.6, frameon=False)
    ax.set_xlim(-0.5, 3.0); ax.set_ylim(bottom=0)
    strip(ax)
    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)
    print(f"wrote {out.name}")


# ═══════════════════════════════════════════════════════════════════════
# Figure 5 — Consensus vs contested posteriors.
# ═══════════════════════════════════════════════════════════════════════
def fig5_consensus_contested(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.4), sharey=True)

    names = ["CERF", "ECHO", "USAID", "NGO"]
    cols = [STAKE_COLS[n] for n in names]

    def panel(ax, title, means, scales):
        x = np.linspace(means[3] - 3*scales[3] - 0.2,
                        means[0] + 3*scales[0] + 0.2, 400)
        for m, s, c, n in zip(means, scales, cols, names):
            y = stats.norm.pdf(x, loc=m, scale=s)
            ax.plot(x, y, color=c, linewidth=1.6, label=n)
            ax.fill_between(x, 0, y, color=c, alpha=0.08, linewidth=0)
        ax.set_title(title, loc="left", pad=12)
        ax.set_xlabel(r"posterior on $\theta$")
        ax.legend(loc="upper right", handlelength=1.4, frameon=False)
        strip(ax)

    panel(axes[0], "Consensus  ·  SDN — all four posteriors overlap",
          means=[2.05, 2.00, 2.05, 1.95], scales=[0.28, 0.30, 0.30, 0.30])
    axes[0].set_ylabel("density")
    panel(axes[1], "Contested  ·  MWI — the four posteriors separate",
          means=[2.6, 1.8, 2.2, 0.9],  scales=[0.32, 0.35, 0.30, 0.35])
    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)
    print(f"wrote {out.name}")


# ═══════════════════════════════════════════════════════════════════════
# Figure 6 — Rank forest with CIs, note below the axis.
# ═══════════════════════════════════════════════════════════════════════
def fig6_rank_forest(out: Path) -> None:
    countries = [
        ("Sudan",        2.5, (1,  5), False),
        ("Honduras",     2.5, (1,  5), False),
        ("Afghanistan",  5.0, (3,  7), False),
        ("Zimbabwe",     5.0, (2, 12), True),
        ("Somalia",      5.5, (3,  8), False),
        ("Haiti",        6.0, (4,  9), False),
        ("Malawi",       7.5, (3, 15), True),
        ("Mozambique",   9.0, (6, 12), False),
        ("South Sudan",  9.0, (7, 11), False),
        ("Chad",         9.5, (6, 14), False),
    ]
    countries = list(reversed(countries))
    ys = np.arange(len(countries))

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    for y, (name, med, (lo, hi), partial) in zip(ys, countries):
        colour = ACCENT_3 if partial else ACCENT
        alpha_fill = 0.55 if partial else 0.45
        ax.hlines(y, lo, hi, color=colour, linewidth=4.2, alpha=alpha_fill)
        ax.plot(med, y, "o", color=ACCENT, markersize=7,
                markeredgecolor="white", markeredgewidth=1.2, zorder=3)
        label = name + (" *" if partial else "")
        ax.text(-0.5, y, label, va="center", ha="right", fontsize=9.2, color=INK)
        ax.text(hi + 0.4, y, f"{med:.1f}", va="center", ha="left",
                fontsize=8.5, color=MUTED)

    ax.set_xlim(-6, 18)
    ax.set_ylim(-0.6, len(countries) - 0.4)
    ax.set_yticks([])
    ax.set_xticks([1, 5, 10, 15])
    ax.set_xlabel("rank within the 57-country pool  (1 = most overlooked)")
    ax.set_title("Posterior rank with 90 % credible interval, per country",
                 loc="left", pad=14)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(PALE)
    ax.tick_params(axis="y", length=0)

    fig.text(0.02, -0.03,
             "*  partial-completeness countries — their credible intervals are visibly wider.",
             fontsize=8, color=SUBTLE, ha="left")
    plt.tight_layout(rect=(0, 0.02, 1, 1))
    plt.savefig(out)
    plt.close(fig)
    print(f"wrote {out.name}")


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    fig1_coverage_collapse(HERE / "fig1_coverage_collapse.png")
    fig2_observation_model(HERE / "fig2_observation_model.png")
    fig3_ar1(HERE / "fig3_temporal_dynamics.png")
    fig4_stakeholder_priors(HERE / "fig4_stakeholder_priors.png")
    fig5_consensus_contested(HERE / "fig5_consensus_contested.png")
    fig6_rank_forest(HERE / "fig6_rank_forest.png")


if __name__ == "__main__":
    main()
