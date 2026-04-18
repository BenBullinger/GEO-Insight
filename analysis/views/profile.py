"""Mode D — Per-crisis profile card scoped to the lens.

All lens properties with values, ranks, provenance, plus a rank-based radar.
Rank used rather than raw values so axes are comparable regardless of unit.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render(enriched: pd.DataFrame, lens, registry) -> None:
    st.title(f"Profile — {lens.name}")
    st.caption(f"Lens: {lens.question}")

    cols = [c for c in lens.properties if c in enriched.columns]
    if not cols:
        st.warning("No lens properties available.")
        return

    iso_choices = sorted(enriched.index.astype(str).tolist())
    iso = st.selectbox("Country (ISO3)", iso_choices, key=f"profile_iso_{lens.id}")
    row = enriched.loc[iso, cols]

    # Ranks: higher rank = more extreme (descending for most metrics)
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(enriched[c])]
    ranks = enriched[numeric_cols].rank(ascending=False, na_option="bottom")
    n = len(enriched)

    # Value table — includes a one-line description next to each property
    st.markdown(f"### {iso}")
    info_rows = []
    for c in cols:
        val = row[c]
        rank = int(ranks.loc[iso, c]) if c in numeric_cols and pd.notna(ranks.loc[iso, c]) else None
        prop = registry.properties.get(c)
        info_rows.append(
            {
                "property": c,
                "description": (prop.description if prop and prop.description else "—"),
                "value": (f"{val:.3g}" if pd.api.types.is_numeric_dtype(enriched[c]) and pd.notna(val) else str(val)),
                "rank": f"{rank}/{n}" if rank else "—",
                "level": prop.level if prop else "?",
                "unit": prop.unit if prop else "",
            }
        )
    st.dataframe(pd.DataFrame(info_rows), use_container_width=True, height=300, hide_index=True)

    # Radar on rank fractions (1 = most extreme, 0 = least extreme)
    if numeric_cols:
        rank_frac = 1 - (ranks.loc[iso, numeric_cols] / n)
        fig = go.Figure(
            go.Scatterpolar(
                r=rank_frac.tolist() + [rank_frac.iloc[0]],
                theta=numeric_cols + [numeric_cols[0]],
                fill="toself",
                name=iso,
            )
        )
        fig.update_layout(
            polar={"radialaxis": {"range": [0, 1], "tickformat": ".0%"}},
            height=520,
            margin={"l": 40, "r": 40, "t": 40, "b": 40},
            showlegend=False,
        )
        st.markdown("#### Lens rank radar (outer = most extreme within lens)")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Property provenance"):
        for c in cols:
            st.markdown(registry.tooltip(c))
            st.markdown("---")
