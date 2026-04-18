"""GEO-Insight — Unsupervised Analysis (lens-first).

Thin orchestrator. Reads sidebar → loads the enriched frame (Levels 1-3 for
Phase 1) → dispatches to the chosen view mode with (enriched, lens, registry).

Phase 1 wires the Funding Pressure lens end-to-end. Other lenses show a
"ships in Phase 2/3" message so the shape of the final UI is visible.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Allow `import ontology`, `import features`, `from views import ...` from the
# analysis/ folder regardless of CWD
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ontology import Registry
import features
from views import atlas, pca as pca_view, cluster, profile, cross_lens

st.set_page_config(
    page_title="GEO-Insight — Unsupervised Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Registry + enriched frame (cached) ────────────────────────────────────
@st.cache_resource(show_spinner="Loading ontology…")
def get_registry() -> Registry:
    return Registry.load()


@st.cache_data(show_spinner="Building enriched frame (Level 1-3)…")
def get_enriched():
    return features.build_enriched_frame()


REGISTRY = get_registry()

# Phase 2: seven lenses wired (Level 1-4 aggregates landed). Only geo_insight_score
# (Lens 8 — our composite gap score) waits on Phase 3.
ACTIVE_LENSES = {
    "magnitude",
    "intensity",
    "severity_composition",
    "funding_pressure",
    "donor_fragility",
    "temporal_dynamics",
    "access_friction",
}


# ─── Sidebar ───────────────────────────────────────────────────────────────
st.sidebar.title("GEO-Insight")
st.sidebar.caption("Unsupervised analysis · Cache Me if You Can · Phase 1")

lens_order = ["funding_pressure"] + [
    lid for lid in REGISTRY.lenses if lid != "funding_pressure"
]


def _label(lid: str) -> str:
    lens = REGISTRY.lenses[lid]
    marker = "" if lid in ACTIVE_LENSES else " · Phase 3"
    return f"{lens.name}{marker}"


lens_id = st.sidebar.selectbox("Lens", lens_order, format_func=_label)
lens = REGISTRY.lenses[lens_id]

mode_id = st.sidebar.radio(
    "Mode",
    ["atlas", "pca", "cluster", "profile", "cross_lens"],
    format_func=lambda m: REGISTRY.modes[m].name,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Companion: data-exploration dashboard on **:8501**.\n\n"
    "Spec: `analysis/spec.yaml`.\n\n"
    "Metric provenance: `proposal/metric_cards.md`.\n\n"
    "Methodology: `proposal/proposal.pdf`."
)


# ─── Route ─────────────────────────────────────────────────────────────────
if lens_id not in ACTIVE_LENSES:
    st.warning(
        f"**{lens.name}** ships in Phase 3. Phase 2 wires seven lenses end-to-end; "
        "this one needs the Level-5 composite gap score which is Phase 3 work."
    )
    st.markdown(f"_Question this lens will answer:_ **{lens.question}**")
    import pandas as _pd
    rows = []
    for p in lens.properties:
        prop = REGISTRY.properties.get(p)
        if prop:
            in_frame = "✓ in enriched frame" if p in get_enriched().columns else "Phase 3"
            rows.append({
                "property": p,
                "description": (prop.description if prop.description else "—"),
                "level": prop.level,
                "status": in_frame,
            })
    st.markdown("#### Properties this lens will read")
    st.dataframe(_pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.stop()

enriched = get_enriched()

VIEWS = {
    "atlas": atlas.render,
    "pca": pca_view.render,
    "cluster": cluster.render,
    "profile": profile.render,
    "cross_lens": cross_lens.render,
}

VIEWS[mode_id](enriched, lens, REGISTRY)
