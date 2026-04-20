"""Generate the 6-vs-7 attribute ablation figure for landing §7.

Two-panel figure:
  (a) grouped bars of precision @ k on the four benchmarks
  (b) top-10 churn — 2x2 table of {added, dropped} × {in benchmark union, not}

Output: landing/methodology/figures/fig_learned_ablation.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "Data" / "learned" / "ablation_results.csv"
OUT = ROOT / "landing" / "methodology" / "figures" / "fig_learned_ablation.png"

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

    df = pd.read_csv(RESULTS)
    labels = df["benchmark"].tolist()
    six = df["six_hit"].values
    sev = df["seven_hit"].values
    ks = df["k"].values

    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    x = np.arange(len(labels))
    width = 0.36

    ax.bar(x - width / 2, six, width, color=FAINT, edgecolor=MUTED, linewidth=0.8,
           label="6-attribute LVM")
    ax.bar(x + width / 2, sev, width, color=ACCENT, edgecolor=ACCENT, linewidth=0.8,
           label="7-attribute LVM (+ learned momentum)")

    for i, (k, h6, h7) in enumerate(zip(ks, six, sev)):
        ax.text(i - width / 2, h6 + 0.12, f"{h6}/{k}", ha="center", va="bottom",
                fontsize=9, color=INK)
        ax.text(i + width / 2, h7 + 0.12, f"{h7}/{k}", ha="center", va="bottom",
                fontsize=9, color=ACCENT, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("CERF UFE ", "CERF\n") for l in labels],
                        fontsize=9, color=INK)
    ax.set_ylabel("correct picks @ k", fontsize=10, color=MUTED)
    ax.set_ylim(0, max(ks) + 1.5)
    ax.set_yticks(range(0, int(max(ks)) + 2))
    ax.tick_params(axis="y", labelsize=9, colors=MUTED)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PALE)

    ax.grid(axis="y", linestyle="-", color=PALE, linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)

    ax.legend(loc="upper right", frameon=False, fontsize=9)

    total_delta = int(df["delta"].sum())
    sign = "+" if total_delta >= 0 else ""
    ax.set_title(
        f"Top-10 churn: MMR, YEM enter · HTI, VEN leave · net {sign}{total_delta} across four benchmarks",
        loc="left", fontsize=10, color=INK, pad=12, fontweight="bold",
    )

    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT)
    plt.close(fig)
    print(f"Wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
