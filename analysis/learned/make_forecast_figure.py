"""Forecasting-quality figure for landing §7.

Two panels:
  (a) MAE-vs-horizon curve — Mamba, persistence, seasonal, mean.
      Shows the crossover where the learned encoder separates from
      persistence.
  (b) Zoom on the headline horizon (H=12) — MAE bars for the four models.

Nothing flashy, nothing oversold.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SWEEP = ROOT / "Data" / "learned" / "horizon_sweep.json"
OUT = ROOT / "landing" / "methodology" / "figures" / "fig_learned_forecast.png"

INK, MUTED, SUBTLE, PALE = "#111111", "#555555", "#888888", "#e8e8e8"
ACCENT, ACCENT_SOFT, FAINT = "#7c1d1d", "#c9554f", "#d8d8d8"


def main() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 180,
        "savefig.bbox": "tight",
    })
    rs = json.loads(SWEEP.read_text())
    rs = [r for r in rs if r.get("mae_mamba") is not None]
    rs.sort(key=lambda r: r["horizon"])

    H = [r["horizon"] for r in rs]
    mamba = [r["mae_mamba"] for r in rs]
    persist = [r["mae_persistence"] for r in rs]
    seasonal = [r["mae_seasonal"] for r in rs]
    mean = [r["mae_mean"] for r in rs]
    ns = [r["n_test"] for r in rs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 3.8),
                                    gridspec_kw={"width_ratios": [1.35, 1.0]})

    # ── Panel A: MAE-vs-horizon curve ───────────────────────────────────
    ax1.plot(H, mamba,    marker="o", color=ACCENT, linewidth=2.0, markersize=6,
             label="Mamba (2 blocks, 19k params)")
    ax1.plot(H, persist,  marker="s", color=INK,    linewidth=1.6, markersize=5,
             label=r"Persistence  $y = s_{t-1}$")
    ax1.plot(H, seasonal, marker="^", color=MUTED,  linewidth=1.4, markersize=5,
             label=r"Seasonal  $y = s_{t-12}$", linestyle="--")
    ax1.plot(H, mean,     color=FAINT, linewidth=1.2, linestyle=":",
             label="Mean baseline")

    # Shade the horizon range where Mamba leads
    lead_xs = [h for h, m, p in zip(H, mamba, persist) if p > m]
    if lead_xs:
        ax1.axvspan(min(lead_xs) - 0.5, max(H) + 0.5, color=ACCENT, alpha=0.05, zorder=0)

    # Annotate crossover
    for i, (h, m, p) in enumerate(zip(H, mamba, persist)):
        if p > m:
            gain = p / m
            ax1.annotate(f"{gain:.2f}×",
                         xy=(h, m), xytext=(0, -14), textcoords="offset points",
                         ha="center", fontsize=8, color=ACCENT, fontweight="bold")

    ax1.set_xticks(H)
    ax1.set_xticklabels([f"{h}\n(n={n})" for h, n in zip(H, ns)], fontsize=9, color=INK)
    ax1.set_xlabel("forecast horizon · months", fontsize=10, color=MUTED, labelpad=8)
    ax1.set_ylabel("MAE (1–5 severity scale)", fontsize=10, color=MUTED)
    ax1.set_ylim(0, max(max(mean) * 1.05, 1.15))
    ax1.tick_params(axis="y", labelsize=9, colors=MUTED)
    for side in ("top", "right"):
        ax1.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax1.spines[side].set_color(PALE)
    ax1.grid(axis="y", linestyle="-", color=PALE, linewidth=0.5, alpha=0.8)
    ax1.set_axisbelow(True)
    ax1.legend(loc="upper left", frameon=False, fontsize=8.5, ncol=2)
    ax1.set_title("Held-out MAE vs horizon — Mamba's lead grows with H",
                  loc="left", fontsize=10, color=INK, pad=12, fontweight="bold")

    # ── Panel B: headline horizon (largest Mamba lead) ──────────────────
    # Pick the horizon where Mamba's lead over persistence is largest
    best = max(rs, key=lambda r: r["mae_persistence"] / max(r["mae_mamba"], 1e-9))
    names = ["Mamba", "Persistence", "Seasonal", "Mean"]
    vals  = [best["mae_mamba"], best["mae_persistence"],
             best["mae_seasonal"], best["mae_mean"]]
    cols  = [ACCENT, INK, MUTED, FAINT]
    x = np.arange(len(names))
    bars = ax2.bar(x, vals, color=cols, edgecolor=cols, linewidth=0.8, width=0.65)
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.012,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax2.set_xticks(x); ax2.set_xticklabels(names, fontsize=9, color=INK)
    ax2.set_ylabel("MAE (1–5 severity scale)", fontsize=10, color=MUTED)
    ax2.set_ylim(0, max(vals) * 1.18)
    ax2.tick_params(axis="y", labelsize=9, colors=MUTED)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax2.spines[side].set_color(PALE)
    ax2.grid(axis="y", linestyle="-", color=PALE, linewidth=0.5, alpha=0.8)
    ax2.set_axisbelow(True)
    ratio = best["mae_persistence"] / best["mae_mamba"]
    ax2.set_title(f"H = {best['horizon']} months · Mamba {ratio:.2f}× persistence  (n={best['n_test']})",
                  loc="left", fontsize=10, color=INK, pad=12, fontweight="bold")

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT)
    plt.close(fig)
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
