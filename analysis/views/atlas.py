"""Mode A — Atlas: ranked table scoped to the chosen lens.

Every column carries a provenance tooltip (formula, source, unit, failure
modes) pulled from the ontology registry.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st


def render(enriched: pd.DataFrame, lens, registry) -> None:
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

    # Completeness filter, when this lens includes completeness (i.e. the
    # Facet lens). Lets the analyst toggle between strict / all-observed.
    if "completeness" in view.columns:
        mode = st.radio(
            "Confidence filter",
            [
                "Strict (completeness = 1.0 — all six attributes observed)",
                "High (≥ 2/3 = 4 of 6 observed)",
                "All (≥ 1/2 = 3 of 6; partials included)",
            ],
            index=2,
            horizontal=True,
        )
        thr = {"Strict": 1.0, "High": 4 / 6, "All": 0.5}[mode.split(" ", 1)[0]]
        view = view[view["completeness"] >= thr - 1e-9]

    # Sort: by median_rank ascending if present, else by first numeric column descending
    if "median_rank" in view.columns:
        view = view.sort_values("median_rank", ascending=True, na_position="last")
    else:
        first_num = next(
            (c for c in cols if pd.api.types.is_numeric_dtype(view[c])), None
        )
        if first_num:
            view = view.sort_values(first_num, ascending=False)

    col_config = {}
    for c in cols:
        tip = registry.tooltip(c)
        if c == "completeness":
            # Progress bar — completeness is bounded [0,1] and primary to interpretation
            col_config[c] = st.column_config.ProgressColumn(
                c, help=tip, min_value=0.0, max_value=1.0, format="%.2f"
            )
        elif c == "typology_cell":
            col_config[c] = st.column_config.TextColumn(c, help=tip)
        elif pd.api.types.is_numeric_dtype(view[c]):
            col_config[c] = st.column_config.NumberColumn(c, help=tip, format="%.3g")
        else:
            col_config[c] = st.column_config.TextColumn(c, help=tip)

    st.markdown(
        f"#### Ranked table — **{len(view)} countries × {len(cols)} properties**"
    )
    st.caption(
        "Hover any column header for formula, source, and failure modes. "
        "Where `completeness` is shown, the score is a weighted mean over observed "
        "attributes only — see `proposal/proposal.pdf` §5.8 for the graceful-"
        "degradation algorithm."
    )
    st.dataframe(view, use_container_width=True, height=520, column_config=col_config)

    with st.expander("Full provenance for every column"):
        for c in cols:
            st.markdown(registry.tooltip(c))
            st.markdown("---")
