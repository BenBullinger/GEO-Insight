"""Mode E — Cross-lens view. Only valid in `all_features` selection.

Phase 1 placeholder; Phase 3 implements the full cross-lens radar + contested-
crisis detection, once all 8 lenses' aggregations are in the enriched frame.
"""
from __future__ import annotations

import streamlit as st


def render(enriched, lens, registry) -> None:
    st.title("Cross-lens")
    st.info(
        "Cross-lens comparison ships in **Phase 3**.\n\n"
        "It will produce a radar per crisis with one axis per lens, using "
        "rank-within-lens as the scale-free radial coordinate. Phase 3 also "
        "adds *contested-crisis detection* — crises whose rank varies sharply "
        "across lenses are flagged.\n\n"
        "Phase 1 ships only the Funding Pressure lens end-to-end. Other lenses "
        "come in Phase 2 once their Level-4 aggregations are wired."
    )
