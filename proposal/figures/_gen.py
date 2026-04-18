"""Generate pedagogical figures for the methodology whitepaper.

Each figure illustrates one concept and is standalone. Run from the repo root:
    dashboard/.venv/bin/python proposal/figures/_gen.py
Figures are written to proposal/figures/*.png at print-quality DPI.

The style matches the landing-page / deck design system: white ground,
oxblood-red single accent, Inter-equivalent sans, strict geometric
typography, minimal axes.
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
RULE        = "#e8e8e8"
ACCENT      = "#7c1d1d"
ACCENT_DIM  = "#9a2a2a"
ACCENT_SOFT = "#c9554f"
ACCENT_MIST = "#fcf5f5"

# Shared typography + axes
mpl.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size":          10,
    "axes.edgecolor":     RULE,
    "axes.linewidth":     0.8,
    "axes.labelcolor":    MUTED,
    "axes.labelsize":     9,
    "axes.titlecolor":    INK,
    "axes.titlesize":     11,
    "axes.titleweight":   "bold",
    "axes.titlepad":      10,
    "xtick.color":        MUTED,
    "ytick.color":        MUTED,
    "xtick.labelsize":    8.5,
    "ytick.labelsize":    8.5,
    "xtick.major.size":   2,
    "ytick.major.size":   2,
    "grid.color":         RULE,
    "grid.linewidth":     0.4,
    "legend.frameon":     False,
    "legend.fontsize":    8.5,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "savefig.dpi":        180,
    "savefig.bbox":       "tight",
    "savefig.facecolor":  "white",
})


def strip(ax):
    """Minimal-chrome axes: no top/right spines, muted grid on y only."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.tick_params(length=2)
    ax.grid(axis="y", linestyle="-", alpha=0.5)


# ═══════════════════════════════════════════════════════════════════════
# Figure 1 — "The problem"
# Coverage ratio over years for a handful of crises, most sliding below
# the 50 % line. Real data from FTS requirements/funding, so the chart
# speaks the literal story the methodology motivates.
# ═══════════════════════════════════════════════════════════════════════
def fig1_coverage_collapse(out: Path) -> None:
    try:
        fts = pd.read_csv(DATA / "fts" / "fts_requirements_funding_global.csv", skiprows=[1])
    except FileNotFoundError:
        print("fig1: FTS data not found; skipping")
        return
    fts["year"] = pd.to_numeric(fts["year"], errors="coerce")
    fts = fts.dropna(subset=["year"])
    agg = fts.groupby(["countryCode", "year"], as_index=False)[["requirements", "funding"]].sum()
    agg["coverage"] = (agg["funding"] / agg["requirements"]).clip(upper=1.5) * 100

    # Countries with multi-year data in 2019–2025
    spans = agg.groupby("countryCode")["year"].agg(["min", "max", "count"])
    active = spans[(spans["min"] <= 2020) & (spans["max"] >= 2025) & (spans["count"] >= 5)].index
    recent = agg[agg["countryCode"].isin(active) & agg["year"].between(2019, 2025)]

    focus = ["SDN", "YEM", "SOM", "AFG", "HTI"]
    focus = [c for c in focus if c in active]

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.axhspan(0, 50, facecolor=ACCENT_MIST, alpha=0.55, zorder=0)
    ax.axhline(50, color=ACCENT, linewidth=0.7, linestyle=(0, (3, 2)), zorder=1)
    ax.text(2025.85, 48, "50 %", va="top", ha="right",
            color=ACCENT, fontsize=8, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", pad=1))

    # Assign distinct shades so separate traces stay legible
    traces = {
        "SDN": (ACCENT,     "Sudan"),
        "YEM": ("#8f2a2a",  "Yemen"),
        "SOM": ("#a13636",  "Somalia"),
        "AFG": ("#b85555",  "Afghanistan"),
        "HTI": ("#cf7c7c",  "Haiti"),
    }

    endpoints = []  # (iso, label, last_year, last_cov, colour)
    for iso in focus:
        if iso not in traces:
            continue
        colour, label = traces[iso]
        grp = recent[recent["countryCode"] == iso].sort_values("year")
        if grp.empty:
            continue
        ax.plot(grp["year"], grp["coverage"], color=colour, linewidth=1.7,
                alpha=0.92, zorder=3)
        last = grp.iloc[-1]
        endpoints.append((iso, label, last["year"], last["coverage"], colour))

    # Arrange right-hand labels on a vertical ladder to avoid overlap
    endpoints.sort(key=lambda r: -r[3])
    label_x = 2025.6
    ladder_y = np.linspace(78, 10, len(endpoints))
    for (iso, label, lx, ly, colour), ly_fixed in zip(endpoints, ladder_y):
        ax.plot([lx, label_x - 0.02], [ly, ly_fixed], color=colour,
                linewidth=0.6, alpha=0.6, zorder=2)
        ax.text(label_x, ly_fixed, label, va="center", ha="left",
                fontsize=8.5, color=colour, fontweight="bold")

    ax.set_xlim(2018.7, 2026.8)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0 %", "25 %", "50 %", "75 %", "100 %"])
    ax.set_xticks(range(2019, 2026))
    ax.set_xlabel("year")
    ax.set_title("Funding coverage for five active crises, 2019 – 2025",
                 loc="left", pad=14)
    strip(ax)
    plt.savefig(out)
    plt.close(fig)
    print(f"wrote {out.name}")


# ═══════════════════════════════════════════════════════════════════════
# Figure 2 — The observation model, one panel per likelihood family
#   (a) Beta regression (coverage shortfall, Gini, HHI, need intensity)
#   (b) Log-normal (per-PIN gap)
#   (c) Ordered probit (INFORM severity category)
#   (d) Missing data → wider posterior
# ═══════════════════════════════════════════════════════════════════════
def fig2_observation_model(out: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.1))
    colours_theta = {-1: FAINT, 0: ACCENT_SOFT, 1: ACCENT}
    theta_label = {-1: r"$\theta=-1$", 0: r"$\theta=0$", 1: r"$\theta=+1$"}

    # ── (a) Beta regression: a_1 = coverage_shortfall in [0,1] ──────────
    ax = axes[0]
    alpha = -0.5
    beta = 1.5
    phi = 12
    x = np.linspace(0.001, 0.999, 400)
    for th in (-1, 0, 1):
        mu = 1 / (1 + np.exp(-(alpha + beta * th)))
        a, b = mu * phi, (1 - mu) * phi
        density = stats.beta.pdf(x, a, b)
        ax.plot(x, density, color=colours_theta[th], linewidth=1.6, label=theta_label[th])
    ax.set_xlabel("observed coverage shortfall $a_1$")
    ax.set_ylabel("likelihood density")
    ax.set_title("(a) Beta regression", loc="left")
    ax.legend(loc="upper center", handlelength=1.4)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    strip(ax)

    # ── (b) Log-normal: a_2 = per-PIN gap in USD ────────────────────────
    ax = axes[1]
    alpha_ln = 4.0
    beta_ln = 0.9
    sigma_ln = 0.7
    x = np.linspace(1, 600, 400)
    for th in (-1, 0, 1):
        mu = alpha_ln + beta_ln * th
        density = stats.lognorm.pdf(x, s=sigma_ln, scale=np.exp(mu))
        ax.plot(x, density, color=colours_theta[th], linewidth=1.6, label=theta_label[th])
    ax.set_xlabel("observed gap per PIN, USD")
    ax.set_title("(b) Log-normal", loc="left")
    ax.legend(loc="upper right", handlelength=1.4)
    ax.set_xlim(0, 600)
    ax.set_ylim(bottom=0)
    ax.set_yticks([])
    strip(ax)

    # ── (c) Ordered probit: a_4 = severity ∈ {1..5} ─────────────────────
    ax = axes[2]
    cutpoints = np.array([-1.5, -0.5, 0.5, 1.5])
    thetas = np.linspace(-2.5, 2.5, 5)
    beta4 = 1.0
    alpha4 = 0.0
    bottoms = np.zeros(len(thetas))
    colors_cat = ["#f0d3d3", "#c97c7c", "#a85050", "#832828", "#5c1414"]
    for k in range(5):
        lo = cutpoints[k - 1] if k > 0 else -np.inf
        hi = cutpoints[k] if k < 4 else np.inf
        loc = alpha4 + beta4 * thetas
        prob = stats.norm.cdf(hi - loc) - stats.norm.cdf(lo - loc)
        ax.bar(thetas, prob, bottom=bottoms, color=colors_cat[k], width=0.6,
               edgecolor="white", linewidth=0.5, label=f"category {k+1}")
        bottoms += prob
    ax.set_xlabel(r"latent $\theta$")
    ax.set_ylabel("P(category)")
    ax.set_title("(c) Ordered probit", loc="left")
    ax.set_xticks(thetas)
    ax.set_xticklabels([f"{t:g}" for t in thetas])
    ax.set_ylim(0, 1)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), handlelength=1.0)
    strip(ax)

    # ── (d) Missingness widens the posterior ─────────────────────────────
    ax = axes[3]
    x = np.linspace(-2.5, 2.5, 400)
    narrow = stats.norm.pdf(x, loc=0.9, scale=0.35)
    wide = stats.norm.pdf(x, loc=0.7, scale=0.9)
    ax.fill_between(x, 0, narrow, color=ACCENT, alpha=0.2, linewidth=0)
    ax.plot(x, narrow, color=ACCENT, linewidth=1.6, label="all six observed")
    ax.fill_between(x, 0, wide, color=SUBTLE, alpha=0.15, linewidth=0)
    ax.plot(x, wide, color=SUBTLE, linewidth=1.6, linestyle=(0, (3, 2)),
            label="three observed")
    ax.set_xlabel(r"posterior on $\theta$ for a country")
    ax.set_title("(d) Missingness widens, not shifts", loc="left")
    ax.legend(loc="upper left", handlelength=1.6)
    ax.set_ylim(bottom=0)
    ax.set_yticks([])
    strip(ax)

    fig.suptitle(
        r"Observation model — each attribute $a_i$ has a likelihood matched to its support, linked to $\theta$ through $(\alpha_i, \beta_i)$",
        y=1.03, fontsize=10, color=MUTED, ha="center",
    )
    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)
    print(f"wrote {out.name}")


# ═══════════════════════════════════════════════════════════════════════
# Figure 3 — Temporal dynamics: AR(1) with country-specific mean
# Chronic baseline μ_u falls out; acute = θ(t) − μ_u.
# ═══════════════════════════════════════════════════════════════════════
def fig3_ar1(out: Path) -> None:
    rng = np.random.default_rng(3)
    T = 60
    t = np.arange(T)
    mu_u = 0.7
    rho_u = 0.82
    tau_u = 0.22
    theta = np.zeros(T)
    theta[0] = mu_u
    for k in range(1, T):
        theta[k] = mu_u + rho_u * (theta[k - 1] - mu_u) + rng.normal(0, tau_u)

    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    ax.axhline(mu_u, color=ACCENT, linestyle=(0, (4, 2.5)), linewidth=1.1, alpha=0.8)
    ax.text(T - 0.2, mu_u + 0.03, r"  chronic baseline $\mu_u$",
            va="bottom", ha="right", color=ACCENT, fontsize=9, fontweight="bold")
    ax.fill_between(t, theta, mu_u,
                    where=(theta > mu_u), color=ACCENT, alpha=0.15, linewidth=0)
    ax.fill_between(t, theta, mu_u,
                    where=(theta <= mu_u), color=SUBTLE, alpha=0.12, linewidth=0)
    ax.plot(t, theta, color=INK, linewidth=1.4)

    # Mark an "acute" spike
    peak = int(np.argmax(theta))
    ax.annotate("acute deviation",
                xy=(peak, theta[peak]), xytext=(peak + 6, theta[peak] + 0.25),
                arrowprops=dict(arrowstyle="-", color=SUBTLE, lw=0.8),
                fontsize=8.5, color=MUTED)

    ax.set_xlabel("month")
    ax.set_ylabel(r"latent $\theta(u, t)$")
    ax.set_title(r"A country's posterior on $\theta(u, \cdot)$ decomposes into chronic baseline and acute deviation",
                 loc="left", pad=16)
    ax.set_xlim(0, T - 1)
    strip(ax)
    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)
    print(f"wrote {out.name}")


# ═══════════════════════════════════════════════════════════════════════
# Figure 4 — Stakeholders as priors: four Gaussians over a single
# attribute slope β_i. Overlap = agreement; separation = disagreement.
# ═══════════════════════════════════════════════════════════════════════
def fig4_stakeholder_priors(out: Path) -> None:
    stakeholders = [
        ("CERF",  1.50, 0.30, ACCENT),
        ("ECHO",  1.25, 0.35, "#a13636"),
        ("USAID", 1.00, 0.40, "#b85555"),
        ("NGO",   1.00, 0.40, "#d07979"),
    ]
    x = np.linspace(-0.5, 3.0, 400)
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    for name, m, s, c in stakeholders:
        y = stats.norm.pdf(x, loc=m, scale=s)
        ax.plot(x, y, color=c, linewidth=1.7, label=name)
        ax.fill_between(x, 0, y, color=c, alpha=0.08, linewidth=0)
    ax.axvline(0, color=FAINT, linewidth=0.7)
    ax.text(0, ax.get_ylim()[1] if False else 1.22, "no effect",
            ha="center", va="bottom", color=SUBTLE, fontsize=8.5)
    ax.set_xlabel(r"slope $\beta_{\text{coverage}}$   (how strongly coverage shortfall reflects $\theta$)")
    ax.set_ylabel("prior density")
    ax.set_title(r"Each stakeholder is a prior over the attribute slopes",
                 loc="left", pad=14)
    ax.legend(loc="upper right", handlelength=1.6)
    ax.set_xlim(-0.5, 3.0)
    ax.set_ylim(bottom=0)
    strip(ax)
    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)
    print(f"wrote {out.name}")


# ═══════════════════════════════════════════════════════════════════════
# Figure 5 — Consensus vs contested: four stakeholder posteriors on
# θ for two different countries.
# ═══════════════════════════════════════════════════════════════════════
def fig5_consensus_contested(out: Path) -> None:
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.6), sharey=True)

    names = ["CERF", "ECHO", "USAID", "NGO"]
    colours = [ACCENT, "#a13636", "#b85555", "#d07979"]

    # Consensus example (tight overlap)
    ax = axes[0]
    means = [1.9, 2.1, 2.0, 2.05]
    scales = [0.28, 0.30, 0.30, 0.32]
    x = np.linspace(0.5, 3.5, 400)
    for m, s, c, n in zip(means, scales, colours, names):
        y = stats.norm.pdf(x, loc=m, scale=s)
        ax.plot(x, y, color=c, linewidth=1.5, label=n)
        ax.fill_between(x, 0, y, color=c, alpha=0.08, linewidth=0)
    ax.set_title("Consensus country — all four posteriors overlap", loc="left", pad=12)
    ax.set_xlabel(r"posterior on $\theta$")
    ax.set_ylabel("density")
    ax.legend(loc="upper right", handlelength=1.4)
    strip(ax)

    # Contested example (wide spread)
    ax = axes[1]
    means = [2.5, 1.7, 2.2, 0.9]
    scales = [0.30, 0.32, 0.28, 0.33]
    x = np.linspace(-0.5, 3.5, 400)
    for m, s, c, n in zip(means, scales, colours, names):
        y = stats.norm.pdf(x, loc=m, scale=s)
        ax.plot(x, y, color=c, linewidth=1.5, label=n)
        ax.fill_between(x, 0, y, color=c, alpha=0.08, linewidth=0)
    ax.set_title("Contested country — they separate", loc="left", pad=12)
    ax.set_xlabel(r"posterior on $\theta$")
    ax.legend(loc="upper right", handlelength=1.4)
    strip(ax)
    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)
    print(f"wrote {out.name}")


# ═══════════════════════════════════════════════════════════════════════
# Figure 6 — Rank forest plot: each country's posterior rank with a
# 90 % credible interval. The honest output shape.
# ═══════════════════════════════════════════════════════════════════════
def fig6_rank_forest(out: Path) -> None:
    rng = np.random.default_rng(11)
    countries = [
        ("Sudan",        2.5, (1,  5)),
        ("Honduras",     2.5, (1,  5)),
        ("Afghanistan",  5.0, (3,  7)),
        ("Zimbabwe",     5.0, (2, 12)),   # partial completeness → wider
        ("Somalia",      5.5, (3,  8)),
        ("Haiti",        6.0, (4,  9)),
        ("Malawi",       7.5, (3, 15)),   # partial completeness → wider
        ("Mozambique",   9.0, (6, 12)),
        ("South Sudan",  9.0, (7, 11)),
        ("Chad",         9.5, (6, 14)),
    ]
    countries = list(reversed(countries))  # top at top
    ys = np.arange(len(countries))
    medians = np.array([c[1] for c in countries])
    lo = np.array([c[2][0] for c in countries])
    hi = np.array([c[2][1] for c in countries])

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.hlines(ys, lo, hi, color=ACCENT_SOFT, linewidth=4, alpha=0.45)
    ax.plot(medians, ys, "o", color=ACCENT, markersize=7, markeredgecolor="white",
            markeredgewidth=1.2, zorder=3)

    for y, (name, med, (l, h)) in zip(ys, countries):
        partial = " *" if name in ("Zimbabwe", "Malawi") else ""
        ax.text(-0.3, y, name + partial, va="center", ha="right",
                fontsize=9, color=INK)
        ax.text(h + 0.4, y, f"{med:.1f}", va="center", ha="left",
                fontsize=8.5, color=MUTED)

    ax.set_xlim(-5, 18)
    ax.set_ylim(-0.6, len(countries) - 0.4)
    ax.set_yticks([])
    ax.set_xticks([1, 5, 10, 15])
    ax.set_xlabel("rank within the 57-country pool   (1 = most overlooked)")
    ax.set_title("Posterior rank with 90 % credible interval, per country",
                 loc="left", pad=14)
    fig.text(0.02, 0.01, "*  partial completeness — wider credible interval",
             fontsize=8, color=SUBTLE, ha="left", va="bottom")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.tick_params(axis="y", length=0)
    plt.tight_layout()
    plt.savefig(out)
    plt.close(fig)
    print(f"wrote {out.name}")


# ═══════════════════════════════════════════════════════════════════════
# Entry
# ═══════════════════════════════════════════════════════════════════════
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
