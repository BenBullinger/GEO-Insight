# Facet — Unsupervised Analysis

Sister site to the **data-exploration** dashboard. Applies classical unsupervised methods (PCA, k-means, hierarchical, t-SNE, temporal clustering, correlation structure) to a per-country feature matrix joined from all sources — FTS, HNO, CoD-PS, INFORM Severity + sub-indicators.

**Purpose.** Triangulate against the data's own geometry: instead of assuming our six-attribute score vector is the right decomposition, let the data tell us how many independent directions there are and what the archetypes look like.

## Run

The analysis app shares the dashboard's virtualenv. `./run.sh` at the repo root launches it alongside everything else on port **8502**.

Manually:

```bash
cd GEO-Insight
dashboard/.venv/bin/streamlit run analysis/app.py --server.port 8502
```

## Files

```
analysis/
├── app.py         Streamlit UI · 7 sections
├── features.py    Feature matrix + trajectory matrix construction
└── README.md
```

No separate `.venv/` — relies on `dashboard/.venv/` having `scikit-learn` and `scipy` (added to `dashboard/requirements.txt`).

## Sections

| # | Section | What's there |
|---|---|---|
| 1 | Feature matrix | 10 features × ~50 countries. Glossary + descriptive stats. |
| 2 | PCA | Scree + loadings table + PC1×PC2 biplot with loading arrows. |
| 3 | Clustering (k-means) | Silhouette-vs-k, centroids in original units, PC-space scatter coloured by cluster. |
| 4 | Hierarchical | Ward-linkage dendrogram, cut slider, tightest country pairs. |
| 5 | t-SNE embedding | Nonlinear 2D, perplexity slider, coloured by k-means label. |
| 6 | Temporal archetypes | k-means over country × month INFORM trajectories (z-scored per country for shape). |
| 7 | Correlation structure | Pearson heatmap + high-correlation pair listing. |

## Feature matrix composition

Per ISO3 country, latest snapshot:

| Feature | Source |
|---|---|
| `coverage_shortfall` | FTS `fts_requirements_funding_global.csv` |
| `log_requirements` | FTS |
| `need_intensity` | HNO PIN / CoD-PS population |
| `category` (1–5), `severity` (1–10) | INFORM severity panel |
| `phase_45_share`, `displaced_share`, `access_restricted_share` | INFORM sub-indicators |
| `log_fatalities`, `log_displaced` | INFORM sub-indicators |

Rows with <60% feature coverage are dropped; remaining missing cells filled with the column median. See `proposal/metric_cards.md` for provenance of each input.
