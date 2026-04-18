"""Mode E — Cross-lens view.

For each country and each lens, we compute a *lens rank fraction* — the
median rank across the lens's numeric properties, flipped so 1.0 = most
extreme within that lens, 0.0 = least extreme. This gives a scale-free radar
axis per lens.

Surfaces:
  - Radar per country with one axis per lens
  - Most contested crises (largest spread of lens-rank fractions)
  - Most consensus crises (smallest spread)

Lens rank is a Phase-3 concept and depends on lenses being wired; Phase-3
lens (`geo_insight_score`) is included automatically once gap_score columns
land in the enriched frame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def _lens_rank_fraction(enriched: pd.DataFrame, lens) -> pd.Series:
    """Median rank of each country across lens properties, flipped to [0,1].

    1 = top of the lens (most extreme), 0 = bottom.
    """
    props = [
        p for p in lens.properties
        if p in enriched.columns and pd.api.types.is_numeric_dtype(enriched[p])
    ]
    if not props:
        return pd.Series(dtype=float, index=enriched.index, name=lens.id)
    ranks = enriched[props].rank(ascending=False, na_option="bottom")
    med = ranks.median(axis=1)
    n = enriched[props].notna().any(axis=1).sum()
    if n == 0:
        return pd.Series(dtype=float, index=enriched.index, name=lens.id)
    return (1 - med / n).rename(lens.id)


def _build_rank_matrix(enriched: pd.DataFrame, registry) -> pd.DataFrame:
    cols = {}
    for lens_id, lens in registry.lenses.items():
        s = _lens_rank_fraction(enriched, lens)
        if s.notna().any():
            cols[lens.name] = s
    return pd.DataFrame(cols).dropna(how="all")


def render(enriched: pd.DataFrame, lens, registry) -> None:
    st.title("Cross-lens")
    st.caption(
        "One axis per lens, rank-within-lens as the radial coordinate. "
        "Surface *contested* crises — where the rank disagrees across lenses — "
        "and *consensus* crises — where every lens tells the same story."
    )

    M = _build_rank_matrix(enriched, registry)
    if M.empty:
        st.warning("No lens ranks available yet.")
        return

    n_lenses = M.shape[1]
    st.markdown(
        f"**{M.shape[0]} countries × {n_lenses} lenses.** Axes show rank fraction "
        "within each lens (1.0 = top rank, 0 = bottom). Each country's median "
        "rank across the lens's properties is used."
    )

    # ─── Radar comparison ─────────────────────────────────────────────
    iso_choices = sorted(M.index.tolist())
    defaults = [c for c in ("SDN", "YEM", "SOM", "COD", "AFG") if c in iso_choices][:4]
    picks = st.multiselect("Countries to compare", iso_choices, default=defaults)
    if picks:
        axes = list(M.columns)
        fig = go.Figure()
        for iso in picks:
            vals = M.loc[iso].tolist()
            fig.add_trace(
                go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=axes + [axes[0]],
                    fill="toself",
                    name=iso,
                    opacity=0.55,
                )
            )
        fig.update_layout(
            polar={"radialaxis": {"range": [0, 1], "tickformat": ".0%"}},
            height=600,
            margin={"l": 40, "r": 40, "t": 40, "b": 40},
        )
        st.plotly_chart(fig, use_container_width=True)

    # ─── Contested detection ─────────────────────────────────────────
    spread = (M.max(axis=1) - M.min(axis=1)).rename("spread_across_lenses")
    M_with = M.join(spread)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Most contested (highest spread)")
        st.caption(
            "Crises whose rank varies sharply across lenses — the single-number "
            "answer depends on which question you ask."
        )
        contested = M_with.sort_values("spread_across_lenses", ascending=False).head(15)
        st.dataframe(contested.round(2), use_container_width=True, height=420)

    with c2:
        st.markdown("#### Most consensus (lowest spread)")
        st.caption(
            "Crises where every lens gives a similar rank — either uniformly "
            "extreme or uniformly middle-of-the-pack."
        )
        consensus = M_with.sort_values("spread_across_lenses", ascending=True).head(15)
        st.dataframe(consensus.round(2), use_container_width=True, height=420)

    # ─── Four-cell typology (if Level-5 columns present) ─────────────
    if "typology_cell" in enriched.columns and enriched["typology_cell"].notna().any():
        st.markdown("#### Four-cell typology")
        st.caption(
            "Classification against sector-coverage inequality × donor-rank disagreement "
            "(median splits). See proposal §5.7."
        )

        def _prettify(raw: str) -> str:
            """consensus-sector-starved → consensus · sector-starved"""
            if not isinstance(raw, str):
                return raw
            # First hyphen separates {consensus, contested} from the rest.
            head, _, tail = raw.partition("-")
            return f"{head} · {tail}" if tail else head

        typ = enriched[["typology_cell"]].copy()
        typ["type"] = typ["typology_cell"].map(_prettify)
        counts = (
            typ["type"]
            .value_counts()
            .rename_axis("Type")
            .to_frame("Countries")
        )
        st.dataframe(counts, use_container_width=True)
        with st.expander("Typology assignments (full list)"):
            full = typ[["type"]].rename(columns={"type": "Type"}).sort_values("Type")
            st.dataframe(full, use_container_width=True, height=420)
