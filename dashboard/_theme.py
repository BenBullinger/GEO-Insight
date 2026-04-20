"""Geo-Insight shared Streamlit theme.

Injects Inter + JetBrains Mono, applies the deep-red accent across
Streamlit chrome that `.streamlit/config.toml` doesn't cover, and registers
a Plotly template so charts match the rest of the project.

Import once at the top of each Streamlit app:

    from _theme import apply_theme, page_header, COLORS
    apply_theme()

The module is intentionally small and side-effect-free until `apply_theme()`
is called. It's safe to import from any app entry.
"""
from __future__ import annotations

import streamlit as st

# ─── Design tokens (mirror landing/shared.css) ──────────────────────────────
COLORS = {
    "ink":          "#111111",
    "text":         "#222222",
    "muted":        "#555555",
    "subtle":       "#888888",
    "faint":        "#bcbcbc",
    "rule":         "#e8e8e8",
    "surface":      "#fafafa",
    "bg":           "#ffffff",
    "accent":       "#7c1d1d",
    "accent_dim":   "#9a2a2a",
    "accent_soft":  "#c9554f",
    "accent_wash":  "#f8ecec",
}

# Red-leaning sequential palette for continuous encodings
SEQUENTIAL = [
    "#fdf1f1", "#f5c8c7", "#e79593", "#d2625f",
    "#b73d3a", "#952626", "#7c1d1d", "#5c1414",
]

# Discrete palette for categorical encodings (reads as a single-accent family
# first, then neutrals). Avoids using multiple competing hues.
QUALITATIVE = [
    "#7c1d1d", "#9a2a2a", "#c9554f",
    "#555555", "#888888", "#bcbcbc",
]


# ─── CSS injection ──────────────────────────────────────────────────────────
_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"],
.stMarkdown, .stText, .stCaption, button[kind], .stButton button,
[data-baseweb="select"], [data-baseweb="tab"] {{
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
    letter-spacing: -0.005em;
}}

/* Restore the icon font for every kind of Material icon Streamlit emits.
   The broad `[class*="st-"]` override used to clobber these, so expander
   chevrons, dropdown caret, etc. rendered as raw ligature text
   ("keyboard_arrow_down"). */
span.material-icons, span.material-icons-outlined,
span[class*="material-symbols"], [data-testid="stIconMaterial"],
[class*="IconMaterial"], [class*="stIconMaterial"] {{
    font-family: "Material Symbols Outlined", "Material Icons", "Material Icons Outlined",
                 "Material Symbols Rounded", "Material Symbols Sharp" !important;
    letter-spacing: normal !important;
}}

code, kbd, pre, [class*="stCodeBlock"], [data-testid="stMetricValue"] {{
    font-family: "JetBrains Mono", "SF Mono", Menlo, monospace !important;
    font-variant-numeric: tabular-nums !important;
}}

/* Page canvas */
.main .block-container {{
    padding-top: 2.2rem;
    padding-bottom: 3rem;
    max-width: 1280px;
}}

/* Headings */
h1, h2, h3, h4 {{
    font-family: "Inter", sans-serif !important;
    color: {COLORS['ink']} !important;
    letter-spacing: -0.02em !important;
    font-weight: 600 !important;
}}
h1 {{ font-size: 1.9rem !important; margin-bottom: 0.4em !important; }}
h2 {{ font-size: 1.3rem !important; margin-top: 1.2em !important; }}
h3 {{ font-size: 1.05rem !important; }}

/* Body text */
.stMarkdown p {{
    color: {COLORS['text']};
    line-height: 1.65;
}}

/* Captions */
[data-testid="stCaptionContainer"] {{
    color: {COLORS['muted']} !important;
    font-size: 0.85rem !important;
    line-height: 1.55 !important;
}}

/* Metric cards */
[data-testid="stMetric"] {{
    background: {COLORS['surface']};
    border: 1px solid {COLORS['rule']};
    border-radius: 4px;
    padding: 14px 18px;
}}
[data-testid="stMetricLabel"] {{
    color: {COLORS['muted']} !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600 !important;
}}
[data-testid="stMetricValue"] {{
    color: {COLORS['ink']} !important;
    font-weight: 500 !important;
    letter-spacing: -0.02em !important;
}}
[data-testid="stMetricDelta"] {{
    color: {COLORS['subtle']} !important;
    font-size: 0.72rem !important;
}}

/* Sidebar */
[data-testid="stSidebar"] {{
    background: {COLORS['surface']};
    border-right: 1px solid {COLORS['rule']};
}}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] .stMarkdown h1 {{
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
    color: {COLORS['ink']} !important;
}}

/* Radio / select labels */
[data-testid="stWidgetLabel"] {{
    font-size: 0.78rem !important;
    color: {COLORS['muted']} !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}

/* Dataframe */
[data-testid="stDataFrame"] {{
    border: 1px solid {COLORS['rule']};
    border-radius: 4px;
}}

/* Dividers */
hr {{
    border: none !important;
    border-top: 1px solid {COLORS['rule']} !important;
    margin: 1.6rem 0 !important;
}}

/* Tabs */
button[data-baseweb="tab"] {{
    font-weight: 500 !important;
    font-family: "Inter", sans-serif !important;
}}

/* Streamlit's own small "footer" and running-man graphic we don't need */
footer {{ visibility: hidden; }}
#MainMenu {{ visibility: hidden; }}
[data-testid="stDecoration"] {{ display: none; }}
</style>
"""


def apply_theme() -> None:
    """Inject shared CSS + register the Plotly template.

    Safe to call more than once; Streamlit deduplicates by content.
    """
    st.markdown(_CSS, unsafe_allow_html=True)
    _register_plotly_template()


# ─── Plotly template ────────────────────────────────────────────────────────
def _register_plotly_template() -> None:
    """Register 'geo_insight' as the default Plotly template.

    Uses the red-leaning palettes above for both continuous and categorical
    encodings. Applied globally via plotly.io.templates.default so every chart
    in the app picks it up without per-call changes.
    """
    try:
        import plotly.io as pio
        import plotly.graph_objects as go
    except ImportError:
        return

    template = go.layout.Template(
        layout=go.Layout(
            font=dict(family="Inter, -apple-system, sans-serif", color=COLORS["text"], size=13),
            paper_bgcolor=COLORS["bg"],
            plot_bgcolor=COLORS["bg"],
            colorway=QUALITATIVE,
            xaxis=dict(
                gridcolor=COLORS["rule"],
                linecolor=COLORS["rule"],
                tickcolor=COLORS["subtle"],
                tickfont=dict(color=COLORS["muted"], size=11),
                title=dict(font=dict(color=COLORS["muted"], size=11)),
                zeroline=False,
            ),
            yaxis=dict(
                gridcolor=COLORS["rule"],
                linecolor=COLORS["rule"],
                tickcolor=COLORS["subtle"],
                tickfont=dict(color=COLORS["muted"], size=11),
                title=dict(font=dict(color=COLORS["muted"], size=11)),
                zeroline=False,
            ),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=COLORS["muted"], size=11)),
            margin=dict(l=40, r=24, t=30, b=40),
            colorscale=dict(sequential=[[i / (len(SEQUENTIAL) - 1), c] for i, c in enumerate(SEQUENTIAL)]),
        )
    )
    pio.templates["geo_insight"] = template
    pio.templates.default = "geo_insight"


# ─── Section helpers ────────────────────────────────────────────────────────
def page_header(title: str, sub: str | None = None, eyebrow: str | None = None) -> None:
    """Consistent h1 + caption pattern for every page.

    Renders an optional eyebrow (ALL-CAPS tag), the section title, and an
    optional single-line description underneath. Keeps prose terse by
    convention — anything longer than ~18 words belongs elsewhere.
    """
    if eyebrow:
        st.markdown(
            f"<div style='font-size: 0.7rem; text-transform: uppercase; "
            f"letter-spacing: 0.14em; color: {COLORS['accent']}; font-weight: 600; "
            f"margin-bottom: 6px;'>{eyebrow}</div>",
            unsafe_allow_html=True,
        )
    st.markdown(f"## {title}")
    if sub:
        st.caption(sub)


def accent_rule() -> None:
    """Small geometric accent bar — substitute for a heavy horizontal rule."""
    st.markdown(
        f"<div style='width: 42px; height: 3px; background: {COLORS['accent']}; "
        f"border-radius: 2px; margin: 4px 0 18px 0;'></div>",
        unsafe_allow_html=True,
    )


def back_to_landing(url: str = "http://localhost:7777/") -> None:
    """Render a '← Back to Geo-Insight' link at the top of the sidebar.

    The landing page at `url` is the canonical entry point; this helper
    gives every Streamlit surface a one-click route home.
    """
    st.sidebar.markdown(
        f"<a href='{url}' target='_self' style='"
        f"display: inline-block; font-size: 0.78rem; color: {COLORS['accent']}; "
        f"text-decoration: none; border-bottom: 1px solid {COLORS['accent_soft']}; "
        f"padding-bottom: 1px; margin-bottom: 14px; font-weight: 500;"
        f"'>← Back to Geo-Insight</a>",
        unsafe_allow_html=True,
    )
