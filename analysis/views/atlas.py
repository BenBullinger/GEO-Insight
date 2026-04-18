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
            "These will appear as Phase 2-4 aggregations are added."
        )
    if not cols:
        st.warning("Lens has no available columns yet. Phase 2+ lens.")
        return

    # Drop rows where all lens columns are NaN (uninformative)
    view = enriched[cols].dropna(how="all").copy()
    first_num = next(
        (c for c in cols if pd.api.types.is_numeric_dtype(view[c])), None
    )
    if first_num:
        view = view.sort_values(first_num, ascending=False)

    col_config = {}
    for c in cols:
        tip = registry.tooltip(c)
        col_config[c] = (
            st.column_config.NumberColumn(c, help=tip, format="%.3g")
            if pd.api.types.is_numeric_dtype(view[c])
            else st.column_config.TextColumn(c, help=tip)
        )

    st.markdown(
        f"**{len(view)} countries** × **{len(cols)} properties**. "
        "Hover any column header for its formula, source, and failure modes."
    )
    st.dataframe(view, use_container_width=True, height=520, column_config=col_config)

    with st.expander("Column provenance"):
        for c in cols:
            st.markdown(registry.tooltip(c))
            st.markdown("---")
