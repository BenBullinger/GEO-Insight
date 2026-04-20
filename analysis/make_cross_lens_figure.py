"""Generate the semantic-lens radar figure for the landing page §5.

Two side-by-side polar plots: one crisis whose rank is *consensus* across
the eight semantic lenses (all axes cluster tightly) and one *contested*
crisis whose rank depends sharply on which question you ask.

Output: landing/methodology/figures/fig_cross_lens_radar.png
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "dashboard"))

from ontology import Registry  # noqa: E402
import features  # noqa: E402
from views.cross_lens import _build_rank_matrix  # noqa: E402

INK, MUTED, SUBTLE, PALE = "#111111", "#555555", "#888888", "#e8e8e8"
ACCENT, ACCENT_DIM, ACCENT_SOFT = "#7c1d1d", "#a13636", "#c9554f"

SHORT_NAMES = {
    "Magnitude": "Magnitude",
    "Intensity": "Intensity",
    "Severity composition": "Severity",
    "Funding pressure": "Funding",
    "Donor fragility": "Donor",
    "Temporal dynamics": "Temporal",
    "Access & friction": "Access",
    "Geo-Insight score": "Geo-Insight",
}


def _radar(ax: plt.Axes, angles: np.ndarray, values: np.ndarray, *,
           labels: list[str], title: str, subtitle: str, color: str) -> None:
    values = np.concatenate([values, values[:1]])
    angles_closed = np.concatenate([angles, angles[:1]])

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(["0.25", "0.5", "0.75"], color=SUBTLE, fontsize=7)
    ax.set_rlabel_position(90)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, color=INK, fontsize=8.5)

    for spine in ax.spines.values():
        spine.set_color(PALE)
    ax.grid(color=PALE, linewidth=0.6, alpha=0.8)

    ax.plot(angles_closed, values, color=color, linewidth=1.8, zorder=3)
    ax.fill(angles_closed, values, color=color, alpha=0.18, zorder=2)
    ax.plot(angles_closed, values, "o", color=color, markersize=3.2,
            markeredgecolor="white", markeredgewidth=0.7, zorder=4)

    ax.set_title(title, color=INK, fontsize=12, fontweight="bold",
                 pad=18, loc="center")
    ax.text(0.5, 1.13, subtitle, transform=ax.transAxes,
            ha="center", va="bottom", color=MUTED, fontsize=8.5)


def main() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 180,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    })

    reg = Registry.load()
    enr = features.load_cached_enriched_frame()
    if enr is None:
        enr = features.build_enriched_frame()
    M = _build_rank_matrix(enr, reg).clip(lower=0, upper=1)

    # Order axes in canonical project order (L1→L5 flow), not alphabetic.
    canonical = [
        "Magnitude", "Intensity", "Severity composition",
        "Funding pressure", "Donor fragility", "Temporal dynamics",
        "Access & friction", "Geo-Insight score",
    ]
    cols = [c for c in canonical if c in M.columns]
    M = M[cols]
    labels = [SHORT_NAMES[c] for c in cols]
    n = len(cols)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    consensus_iso, consensus_sub = "SSD", "consensus — every lens agrees"
    contested_iso, contested_sub = "TCD", "contested — rank depends on the question"

    fig, axes = plt.subplots(
        1, 2, figsize=(9.8, 4.6), subplot_kw={"polar": True},
    )
    _radar(
        axes[0], angles, M.loc[consensus_iso].values,
        labels=labels, title=f"{consensus_iso} · South Sudan",
        subtitle=consensus_sub, color=ACCENT,
    )
    _radar(
        axes[1], angles, M.loc[contested_iso].values,
        labels=labels, title=f"{contested_iso} · Chad",
        subtitle=contested_sub, color=ACCENT_SOFT,
    )

    fig.suptitle(
        "Same crisis, eight lenses — rank fraction within each lens (1.0 = top)",
        color=INK, fontsize=11, fontweight="bold", y=1.02,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.96))

    out = ROOT / "landing" / "methodology" / "figures" / "fig_cross_lens_radar.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
