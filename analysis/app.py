"""GEO-Insight — Unsupervised Analysis.

A second Streamlit site alongside the data-exploration dashboard. Applies
classical unsupervised methods to a per-country feature matrix joined from
all sources (FTS, HNO, CoD-PS, INFORM severity and sub-indicators), plus
temporal-archetype clustering on INFORM trajectories. Purpose: surface the
latent structure of the crisis space so our scoring and typology aren't
assumed — they're triangulated against the data's own geometry.

Sections:
  1. Feature matrix      — the raw per-country vector everything else reads
  2. PCA                 — scree + biplot + loadings
  3. Clustering (k-means)— k-selection, centroids, cluster scatter in PC space
  4. Hierarchical        — dendrogram + tight pairs
  5. t-SNE embedding     — nonlinear 2D
  6. Temporal archetypes — k-means on INFORM trajectories
  7. Correlation         — feature redundancy heatmap + high-correlation pairs

Shares the dashboard/.venv virtualenv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# allow `import features` from the same folder
sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as ft

from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ─── Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GEO-Insight — Unsupervised Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Cached loaders ────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Building feature matrix…")
def get_X(year: int = 2025) -> pd.DataFrame:
    return ft.build_feature_matrix(year=year)


@st.cache_data(show_spinner="Standardising features…")
def get_X_std(year: int = 2025) -> tuple[np.ndarray, list[str], list[str]]:
    X = get_X(year)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)
    return Xs, list(X.columns), list(X.index)


@st.cache_data(show_spinner="Fitting PCA…")
def get_pca(year: int = 2025) -> tuple[PCA, np.ndarray, list[str], list[str]]:
    Xs, cols, index = get_X_std(year)
    pca = PCA(n_components=min(len(cols), 8))
    Z = pca.fit_transform(Xs)
    return pca, Z, cols, index


@st.cache_data(show_spinner="Loading severity trajectory matrix…")
def get_trajectories(value: str = "category", min_months: int = 50) -> pd.DataFrame:
    return ft.build_trajectory_matrix(value=value, min_snapshots=min_months)


# ─── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.title("GEO-Insight")
st.sidebar.caption("Unsupervised analysis · Cache Me if You Can · Datathon 2026")
section = st.sidebar.radio(
    "Section",
    [
        "Feature matrix",
        "PCA",
        "Clustering (k-means)",
        "Hierarchical",
        "t-SNE embedding",
        "Temporal archetypes",
        "Correlation structure",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Companion to the data-exploration dashboard at **:8501**.\n\n"
    "Methodology: `proposal/proposal.pdf`.\n\n"
    "Metric definitions: `proposal/metric_cards.md`."
)


# ─── Page 1 — Feature matrix ───────────────────────────────────────────────
def page_matrix() -> None:
    st.title("Feature matrix")
    st.caption(
        "Per-country feature vector joined from FTS (coverage, requirements), "
        "HNO (PIN), CoD-PS (population), INFORM Severity (category + index), and "
        "INFORM sub-indicators (Phase 4+5 share, displaced, access constraints, "
        "fatalities). All downstream analysis runs on this matrix."
    )
    X = get_X()
    st.markdown(f"**Shape:** {X.shape[0]} countries × {X.shape[1]} features.")
    st.markdown(
        "All features are latest-snapshot values; missing cells filled with "
        "column median; countries with <60% feature coverage dropped."
    )
    st.dataframe(X.round(3), use_container_width=True, height=420)

    st.markdown("#### Descriptive statistics")
    st.dataframe(X.describe().T.round(3), use_container_width=True, height=340)

    st.markdown("#### Feature glossary")
    st.markdown(
        """
| Feature | Definition |
|---|---|
| `coverage_shortfall` | $1 - F/R$ (FTS, clipped at 1) |
| `log_requirements` | $\\log(1+R)$ (FTS) |
| `need_intensity` | PIN / population (HNO / CoD-PS) |
| `category` | INFORM severity category 1–5 (latest) |
| `severity` | INFORM continuous index (latest; note Feb-2026 rescaling in metric cards) |
| `phase_45_share` | (PIN in Phase 4 + Phase 5) / total affected (INFORM) |
| `displaced_share` | displaced / PIN (INFORM) |
| `access_restricted_share` | people facing restricted access / PIN (INFORM) |
| `log_fatalities` | $\\log(1+\\text{fatalities})$ (INFORM) |
| `log_displaced` | $\\log(1+\\text{displaced})$ (INFORM) |
        """
    )


# ─── Page 2 — PCA ───────────────────────────────────────────────────────────
def page_pca() -> None:
    st.title("Principal component analysis")
    st.caption(
        "Linear decomposition of the feature matrix. Orthogonal directions of "
        "variance, named by their loadings. First few components usually admit "
        "a clean semantic reading (magnitude, composition, access friction)."
    )
    pca, Z, cols, index = get_pca()

    # Variance explained
    var = pca.explained_variance_ratio_
    cum = np.cumsum(var)
    scree = pd.DataFrame({
        "Component": [f"PC{i+1}" for i in range(len(var))],
        "Variance explained": var,
        "Cumulative": cum,
    })
    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown("#### Scree")
        fig = px.bar(scree, x="Component", y="Variance explained")
        fig.add_scatter(
            x=scree["Component"], y=scree["Cumulative"], mode="lines+markers",
            name="cumulative", yaxis="y2",
        )
        fig.update_layout(
            yaxis={"tickformat": ".0%"},
            yaxis2={"overlaying": "y", "side": "right", "tickformat": ".0%"},
            height=360, margin={"l": 0, "r": 0, "t": 10, "b": 0},
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("#### Loadings")
        loadings = pd.DataFrame(
            pca.components_.T,
            index=cols,
            columns=[f"PC{i+1}" for i in range(pca.n_components_)],
        )
        k = st.slider("Components to show", 2, min(pca.n_components_, 6), 4, key="pca_k")
        st.dataframe(loadings.iloc[:, :k].round(3), use_container_width=True)

    # Biplot PC1 × PC2
    st.markdown("#### Biplot — PC1 × PC2")
    scores = pd.DataFrame(Z[:, :2], columns=["PC1", "PC2"], index=index).reset_index().rename(columns={"index": "ISO3"})
    colour_by = st.selectbox(
        "Colour points by",
        ["(none)"] + cols,
    )
    fig = px.scatter(
        scores,
        x="PC1",
        y="PC2",
        text="ISO3",
        color=None if colour_by == "(none)" else get_X().loc[scores["ISO3"], colour_by].values,
        color_continuous_scale="Viridis",
        labels={"color": colour_by if colour_by != "(none)" else ""},
    )
    fig.update_traces(textposition="top center", textfont={"size": 9})
    # Loading arrows (scaled to match points' range)
    load2 = pca.components_[:2].T
    arrow_scale = max(abs(scores["PC1"]).max(), abs(scores["PC2"]).max()) * 0.9
    for i, name in enumerate(cols):
        x, y = load2[i, 0] * arrow_scale, load2[i, 1] * arrow_scale
        fig.add_annotation(
            x=x, y=y, ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y",
            arrowhead=3, arrowsize=1, arrowwidth=1, arrowcolor="#E99C2D",
        )
        fig.add_annotation(
            x=x * 1.08, y=y * 1.08, text=name, showarrow=False,
            font=dict(size=10, color="#E99C2D"),
        )
    fig.update_layout(height=560, margin={"l": 0, "r": 0, "t": 10, "b": 0})
    st.plotly_chart(fig, use_container_width=True)


# ─── Page 3 — k-means ──────────────────────────────────────────────────────
def page_kmeans() -> None:
    st.title("k-means clustering")
    st.caption(
        "Partition crises by feature similarity. We show silhouette-vs-k for "
        "model selection, cluster centroids (what defines each group), and a "
        "scatter in PC space coloured by cluster."
    )
    Xs, cols, index = get_X_std()

    # Silhouette across k
    st.markdown("#### Silhouette analysis")
    ks = list(range(2, 11))
    scores = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(Xs)
        scores.append(silhouette_score(Xs, km.labels_))
    sil_df = pd.DataFrame({"k": ks, "silhouette": scores})
    fig = px.line(sil_df, x="k", y="silhouette", markers=True)
    fig.update_layout(height=280, margin={"l": 0, "r": 0, "t": 10, "b": 0})
    st.plotly_chart(fig, use_container_width=True)

    best_k = int(sil_df.loc[sil_df["silhouette"].idxmax(), "k"])
    st.info(f"Silhouette-optimal k = **{best_k}**.")

    k = st.slider("k", 2, 10, best_k, key="km_k")
    km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(Xs)
    labels = km.labels_

    # Centroids in ORIGINAL units (un-standardise via get_X().mean/std)
    X = get_X()
    orig_centroids = pd.DataFrame(
        km.cluster_centers_ * X.std().values + X.mean().values,
        columns=cols,
        index=[f"Cluster {i}" for i in range(k)],
    )
    orig_centroids["count"] = pd.Series(labels).value_counts().sort_index().values
    st.markdown("#### Cluster centroids (original units)")
    st.dataframe(orig_centroids.round(3), use_container_width=True)

    # PC1×PC2 scatter coloured by cluster
    pca, Z, _, _ = get_pca()
    scatter = pd.DataFrame({
        "PC1": Z[:, 0], "PC2": Z[:, 1], "cluster": labels, "ISO3": index,
    })
    fig = px.scatter(
        scatter, x="PC1", y="PC2", color=scatter["cluster"].astype(str),
        text="ISO3", hover_name="ISO3",
    )
    fig.update_traces(textposition="top center", textfont={"size": 9})
    fig.update_layout(
        height=560, legend_title_text="cluster",
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
    )
    st.plotly_chart(fig, use_container_width=True)

    # Membership table
    with st.expander("Cluster membership"):
        mem = pd.DataFrame({"ISO3": index, "cluster": labels}).sort_values(["cluster", "ISO3"])
        st.dataframe(mem, use_container_width=True, height=420)


# ─── Page 4 — Hierarchical ─────────────────────────────────────────────────
def page_hierarchical() -> None:
    st.title("Hierarchical clustering")
    st.caption(
        "Agglomerative clustering with Ward linkage. Use the dendrogram to see "
        "the nested structure; cut at a chosen number of clusters for membership."
    )
    Xs, cols, index = get_X_std()

    # Linkage matrix for dendrogram
    Z_link = hierarchy.linkage(Xs, method="ward")

    # Use Plotly's built-in dendrogram via scipy → matplotlib-free
    import plotly.figure_factory as ff
    fig = ff.create_dendrogram(
        Xs,
        orientation="bottom",
        labels=index,
        linkagefun=lambda x: hierarchy.linkage(x, method="ward"),
    )
    fig.update_layout(
        height=620, margin={"l": 0, "r": 0, "t": 30, "b": 120},
        xaxis={"tickfont": {"size": 9}},
    )
    st.plotly_chart(fig, use_container_width=True)

    k = st.slider("Cut for k clusters", 2, 10, 4, key="hier_k")
    agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = agg.fit_predict(Xs)
    st.markdown(f"#### Membership at k={k}")
    st.dataframe(
        pd.DataFrame({"ISO3": index, "cluster": labels}).sort_values(["cluster", "ISO3"]),
        use_container_width=True, height=420,
    )

    # Tight pairs
    st.markdown("#### 10 tightest country pairs (nearest-neighbour distance)")
    dist = pdist(Xs)
    from scipy.spatial.distance import squareform
    D = squareform(dist)
    np.fill_diagonal(D, np.inf)
    rows = []
    for i in range(D.shape[0]):
        j = int(np.argmin(D[i]))
        if i < j:
            rows.append((index[i], index[j], float(D[i, j])))
    rows.sort(key=lambda t: t[2])
    pairs_df = pd.DataFrame(rows[:10], columns=["ISO3_a", "ISO3_b", "euclidean_distance"])
    st.dataframe(pairs_df.round(3), use_container_width=True)


# ─── Page 5 — t-SNE ────────────────────────────────────────────────────────
def page_tsne() -> None:
    st.title("t-SNE embedding")
    st.caption(
        "Nonlinear projection into 2D. Preserves local neighbourhoods — useful "
        "when PCA's linear axes miss curved structure. Colour by any feature or "
        "by a k-means label."
    )
    Xs, cols, index = get_X_std()
    perplexity = st.slider("Perplexity", 5, 50, min(30, max(5, Xs.shape[0] // 4)), key="tsne_p")
    k = st.slider("k-means clusters (for colour)", 2, 10, 4, key="tsne_k")
    km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(Xs)
    tsne = TSNE(
        n_components=2, perplexity=perplexity, init="pca",
        learning_rate="auto", random_state=0,
    )
    with st.spinner("Fitting t-SNE…"):
        emb = tsne.fit_transform(Xs)
    df = pd.DataFrame({"x": emb[:, 0], "y": emb[:, 1], "cluster": km.labels_, "ISO3": index})
    fig = px.scatter(
        df, x="x", y="y", color=df["cluster"].astype(str), text="ISO3", hover_name="ISO3",
    )
    fig.update_traces(textposition="top center", textfont={"size": 9})
    fig.update_layout(
        height=600, legend_title_text="cluster",
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
    )
    st.plotly_chart(fig, use_container_width=True)


# ─── Page 6 — Temporal archetypes ──────────────────────────────────────────
def page_archetypes() -> None:
    st.title("Temporal archetypes")
    st.caption(
        "k-means over country trajectories of INFORM severity category. Each "
        "cluster centroid is an archetype over time. Each trajectory is "
        "z-scored per country so clusters reflect **shape**, not level."
    )
    traj = get_trajectories(value="category", min_months=50)
    # z-score per country → focus on shape
    means = traj.mean(axis=1)
    sds = traj.std(axis=1).replace(0, 1)
    T = ((traj.T - means) / sds).T.values

    k = st.slider("Number of archetypes", 2, 8, 4, key="arch_k")
    km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(T)
    labels = km.labels_

    # plot centroids (un-normalised back to absolute 1–5 scale for readability)
    centroids = km.cluster_centers_
    # map z-scored centroid back to the typical level: use mean of constituent trajectories
    cols = traj.columns
    plot_rows = []
    for c in range(k):
        mask = labels == c
        mean_traj = traj.loc[mask].mean(axis=0)
        for snap, val in mean_traj.items():
            plot_rows.append({"cluster": f"Arch {c}", "snapshot": snap, "category": val, "n": int(mask.sum())})
    plot = pd.DataFrame(plot_rows)
    plot["date"] = pd.to_datetime(plot["snapshot"] + "-01")

    fig = px.line(
        plot, x="date", y="category", color="cluster", markers=True,
        hover_data=["snapshot"],
    )
    fig.update_layout(
        height=420, yaxis={"range": [1, 5], "title": "INFORM category (1–5)"},
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
    )
    st.plotly_chart(fig, use_container_width=True)

    # Membership table
    mem = pd.DataFrame({"ISO3": traj.index, "cluster": labels}).sort_values(["cluster", "ISO3"])
    counts = mem.groupby("cluster").size().rename("count")
    st.markdown("#### Cluster sizes")
    st.dataframe(counts.to_frame(), use_container_width=True)
    with st.expander("Country membership"):
        st.dataframe(mem, use_container_width=True, height=420)


# ─── Page 7 — Correlation ──────────────────────────────────────────────────
def page_correlation() -> None:
    st.title("Correlation structure")
    st.caption(
        "Pearson correlation heatmap over the feature matrix. If two features "
        "are near-perfectly correlated, one of them is redundant for clustering."
    )
    X = get_X()
    C = X.corr().round(2)
    fig = go.Figure(go.Heatmap(
        z=C.values, x=C.columns, y=C.index,
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=C.values, texttemplate="%{text}",
    ))
    fig.update_layout(height=540, margin={"l": 0, "r": 0, "t": 10, "b": 0})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Feature pairs with |r| > 0.7")
    pairs = []
    for i in range(len(C.columns)):
        for j in range(i + 1, len(C.columns)):
            r = C.iloc[i, j]
            if abs(r) > 0.7:
                pairs.append((C.columns[i], C.columns[j], r))
    if pairs:
        st.dataframe(
            pd.DataFrame(pairs, columns=["feature_a", "feature_b", "r"]).sort_values("r", key=lambda s: s.abs(), ascending=False),
            use_container_width=True,
        )
    else:
        st.info("No feature pair exceeds |r| = 0.7 — low redundancy.")


# ─── Router ────────────────────────────────────────────────────────────────
PAGES = {
    "Feature matrix": page_matrix,
    "PCA": page_pca,
    "Clustering (k-means)": page_kmeans,
    "Hierarchical": page_hierarchical,
    "t-SNE embedding": page_tsne,
    "Temporal archetypes": page_archetypes,
    "Correlation structure": page_correlation,
}
PAGES[section]()
