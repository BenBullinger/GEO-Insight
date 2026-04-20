"""Mode A — Atlas: ranked table scoped to the chosen lens.

For the Geo-Insight lens, the atlas is a Bayesian posterior forest plot with
credible intervals rather than a spreadsheet of numeric columns — the
uncertainty needs to be visible at a glance, not reconstructed from
separate `theta_ci_lo` / `theta_ci_hi` columns.

For every other lens, the atlas remains a ranked table with provenance
tooltips on every column.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Design tokens — imported lazily via _theme (sibling package)
try:
    from _theme import COLORS
except ImportError:  # pragma: no cover — defensive
    COLORS = {"accent": "#7c1d1d", "muted": "#6b6b6b", "faint": "#d8d8d8"}


def render(enriched: pd.DataFrame, lens, registry) -> None:
    if lens.id == "geo_insight_score":
        _render_bayesian_atlas(enriched, lens, registry)
    else:
        _render_default_atlas(enriched, lens, registry)


# ═══════════════════════════════════════════════════════════════════════════
# Bayesian posterior atlas — used for lens.id == geo_insight_score
# ═══════════════════════════════════════════════════════════════════════════
def _render_bayesian_atlas(enriched: pd.DataFrame, lens, registry) -> None:
    st.title("Atlas — Geo-Insight posterior")
    st.caption(lens.question)

    required = {"theta_median", "theta_ci_lo", "theta_ci_hi", "theta_ci_width"}
    if not required.issubset(enriched.columns):
        st.info(
            "Posterior columns not yet in the enriched frame. "
            f"Missing: {', '.join(sorted(required - set(enriched.columns)))}."
        )
        return

    # ── Pool breakdown banner ───────────────────────────────────────────
    total = len(enriched)
    fit = int(enriched["theta_median"].notna().sum())
    hrp_excluded = total - fit

    b1, b2 = st.columns([2, 3])
    with b1:
        c1, c2 = st.columns(2)
        c1.metric("Fitted", f"{fit} countries", help="HRP-eligible: per_pin_gap observed.")
        c2.metric("Not fitted", f"{hrp_excluded}", help=(
            "Countries in the enriched frame without per_pin_gap — no formal "
            "HRP or no FTS requirements. The observation model is undefined for them."
        ))
    with b2:
        st.markdown(
            """
            **The Bayesian model fits the HRP-eligible pool only.** Countries
            without an HNO people-in-need figure and FTS requirements return
            no posterior — not a low score, not a zero. See the data dashboard's
            *Candidate pool* page for the full breakdown of who sits in and out.
            """
        )

    show_all = st.toggle(
        "Include non-HRP countries (shown without posterior, for context)",
        value=False,
        help="Off: only the 22 countries the model fits. On: all 114, greyed out where θ is not estimated.",
    )

    # ── Data preparation ────────────────────────────────────────────────
    sub = enriched.copy()
    if not show_all:
        sub = sub[sub["theta_median"].notna()]
    sub = sub.sort_values("theta_median", ascending=True, na_position="first")

    # ── Forest plot ─────────────────────────────────────────────────────
    st.markdown("### Posterior median & 90 % credible interval")
    fig = _forest_figure(sub)
    st.plotly_chart(fig, use_container_width=True)

    # ── Companion data table (headline numbers only) ───────────────────
    st.markdown("### Country table")
    st.caption(
        "Posterior columns render as a progress bar for completeness and a "
        "text tag for the four-cell typology. The full property frame is in "
        "the expander below."
    )
    table_cols = [c for c in ["theta_median", "theta_ci_width", "completeness", "typology_cell"] if c in sub.columns]
    table = sub[table_cols].copy()
    if "theta_median" in table.columns:
        table = table.sort_values("theta_median", ascending=False, na_position="last")

    col_config: dict = {}
    if "theta_median" in table.columns:
        col_config["theta_median"] = st.column_config.NumberColumn(
            "θ median",
            help=registry.tooltip("theta_median"),
            format="%+.3f",
        )
    if "theta_ci_width" in table.columns:
        col_config["theta_ci_width"] = st.column_config.NumberColumn(
            "90 % CI width",
            help=registry.tooltip("theta_ci_width"),
            format="%.2f",
        )
    if "completeness" in table.columns:
        col_config["completeness"] = st.column_config.ProgressColumn(
            "completeness",
            help=registry.tooltip("completeness"),
            min_value=0.0,
            max_value=1.0,
            format="%.2f",
        )
    if "typology_cell" in table.columns:
        col_config["typology_cell"] = st.column_config.TextColumn(
            "typology",
            help=registry.tooltip("typology_cell"),
        )
    st.dataframe(table, use_container_width=True, height=420, column_config=col_config)

    with st.expander("Full posterior columns (theta_ci_lo, theta_ci_hi)"):
        wide_cols = [c for c in ["theta_median", "theta_ci_lo", "theta_ci_hi", "theta_ci_width", "completeness", "typology_cell"] if c in sub.columns]
        st.dataframe(
            sub[wide_cols].sort_values("theta_median", ascending=False, na_position="last"),
            use_container_width=True,
            height=420,
        )


def _forest_figure(sub: pd.DataFrame) -> go.Figure:
    """Forest plot: one row per country, posterior median dot + 90 % CI bar.

    Top-10 by posterior median drawn in the accent red; the remainder muted.
    Non-HRP countries (with NaN theta_median) render as grey markers at 0
    with no CI bar.
    """
    ACCENT = COLORS.get("accent", "#7c1d1d")
    MUTED = COLORS.get("muted", "#6b6b6b")
    FAINT = COLORS.get("faint", "#d8d8d8")

    # Compute top-10 by posterior median for styling
    fit = sub.dropna(subset=["theta_median"]).copy()
    top_10_iso = set(fit.nlargest(10, "theta_median").index.astype(str))

    rows = []
    for iso, r in sub.iterrows():
        iso3 = str(iso)
        med = r.get("theta_median")
        lo = r.get("theta_ci_lo")
        hi = r.get("theta_ci_hi")
        rows.append({
            "iso3": iso3,
            "median": med,
            "lo": lo,
            "hi": hi,
            "typology": r.get("typology_cell", "—"),
            "is_top10": iso3 in top_10_iso,
            "has_posterior": pd.notna(med),
        })
    df = pd.DataFrame(rows)
    y_labels = df["iso3"].tolist()
    y = list(range(len(df)))

    fig = go.Figure()

    # CI bars — draw as line segments
    has_post = df[df["has_posterior"]]
    for _, r in has_post.iterrows():
        i = y_labels.index(r["iso3"])
        color = ACCENT if r["is_top10"] else FAINT
        fig.add_trace(
            go.Scatter(
                x=[r["lo"], r["hi"]],
                y=[i, i],
                mode="lines",
                line={"color": color, "width": 2 if r["is_top10"] else 1.4},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Median dots
    for top_only in (False, True):
        mask = has_post["is_top10"] == top_only
        sub_df = has_post[mask]
        if sub_df.empty:
            continue
        ys = [y_labels.index(iso) for iso in sub_df["iso3"]]
        fig.add_trace(
            go.Scatter(
                x=sub_df["median"],
                y=ys,
                mode="markers",
                marker={
                    "color": ACCENT if top_only else MUTED,
                    "size": 9 if top_only else 6,
                    "line": {"color": "white", "width": 1},
                },
                text=sub_df["iso3"],
                customdata=sub_df[["lo", "hi", "typology"]].values,
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "θ median %{x:+.3f}<br>"
                    "90 % CI [%{customdata[0]:+.3f}, %{customdata[1]:+.3f}]<br>"
                    "typology %{customdata[2]}<extra></extra>"
                ),
                name="Top-10" if top_only else "Rest of pool",
            )
        )

    # Non-posterior markers at x=0
    no_post = df[~df["has_posterior"]]
    if not no_post.empty:
        ys = [y_labels.index(iso) for iso in no_post["iso3"]]
        fig.add_trace(
            go.Scatter(
                x=[0] * len(no_post),
                y=ys,
                mode="markers",
                marker={
                    "color": FAINT,
                    "size": 6,
                    "symbol": "x-thin",
                    "line": {"color": MUTED, "width": 1.2},
                },
                text=no_post["iso3"],
                hovertemplate="<b>%{text}</b><br>Not in HRP-eligible pool<extra></extra>",
                name="No posterior",
            )
        )

    fig.update_layout(
        height=max(400, 20 * len(df) + 80),
        margin={"l": 50, "r": 30, "t": 30, "b": 40},
        xaxis={
            "title": "posterior median θ (higher = more overlooked)",
            "zeroline": True,
            "zerolinecolor": MUTED,
            "zerolinewidth": 0.8,
            "gridcolor": "#eeeeee",
        },
        yaxis={
            "tickmode": "array",
            "tickvals": y,
            "ticktext": y_labels,
            "gridcolor": "#f5f5f5",
        },
        plot_bgcolor="white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Default atlas — every other lens
# ═══════════════════════════════════════════════════════════════════════════
def _render_default_atlas(enriched: pd.DataFrame, lens, registry) -> None:
    st.title(f"Atlas — {lens.name}")
    st.caption(lens.question)

    cols = [c for c in lens.properties if c in enriched.columns]
    missing = [c for c in lens.properties if c not in enriched.columns]
    if missing:
        st.info(
            f"Not yet in enriched frame: {', '.join(f'`{m}`' for m in missing)}. "
            "These will appear as Phase 2–4 aggregations are added."
        )
    if not cols:
        st.warning("Lens has no available columns yet. Phase 2+ lens.")
        return

    # ── Legend: what each column means, one line per property ──
    st.markdown("#### Properties in this lens")
    legend_rows = []
    for c in cols:
        p = registry.properties.get(c)
        legend_rows.append(
            {
                "property": c,
                "description": (p.description if p and p.description else "—"),
                "level": p.level if p else "?",
                "unit": (p.unit if p and p.unit else ""),
            }
        )
    st.dataframe(
        pd.DataFrame(legend_rows),
        use_container_width=True,
        hide_index=True,
        height=min(35 * (len(legend_rows) + 1) + 8, 320),
    )

    # ── Data table ──
    view = enriched[cols].dropna(how="all").copy()

    first_num = next(
        (c for c in cols if pd.api.types.is_numeric_dtype(view[c])), None
    )
    if first_num:
        view = view.sort_values(first_num, ascending=False)

    col_config = {}
    for c in cols:
        tip = registry.tooltip(c)
        if c == "completeness":
            col_config[c] = st.column_config.ProgressColumn(
                c, help=tip, min_value=0.0, max_value=1.0, format="%.2f"
            )
        elif pd.api.types.is_numeric_dtype(view[c]):
            col_config[c] = st.column_config.NumberColumn(c, help=tip, format="%.3g")
        else:
            col_config[c] = st.column_config.TextColumn(c, help=tip)

    st.markdown(
        f"#### Ranked table — **{len(view)} countries × {len(cols)} properties**"
    )
    st.caption("Hover any column header for formula, source, and failure modes.")
    st.dataframe(view, use_container_width=True, height=520, column_config=col_config)

    with st.expander("Full provenance for every column"):
        for c in cols:
            st.markdown(registry.tooltip(c))
            st.markdown("---")
