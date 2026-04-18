# Geo-Insight

**Which humanitarian crises are most overlooked?**
Datathon 2026

A decision-support prototype that ranks active humanitarian crises by the mismatch between documented need (HNO/HRP) and available funding coverage (FTS/CBPF), with an intra-crisis equity term and a cross-stakeholder disagreement term. A typed five-level semantic layer (71 properties, declared formulas, declared failure modes) makes every score traceable to its source.

Research direction and methodology live in `proposal/proposal.pdf`; this README is the operational guide to getting everything running locally.

Everything runs on localhost — no public hosting.

---

## Prerequisites

- **Python 3.9+** (check with `python3 --version`). Standard library is enough for all downloads and the launcher; only the Streamlit dashboard needs pip-installed packages, and `run.sh` handles that for you.
- **A Unix-like shell** with `bash`. Tested on macOS. Linux works. Windows users: use WSL.
- **Disk space**: ~280 MB after the one-time data download.

No Node, no npm, no Docker.

## One-time setup

```bash
git clone git@github.com:BenBullinger/GEO-Insight.git
cd GEO-Insight
./run.sh
```

On the very first run, `./run.sh` will:
1. **Download ~270 MB of source data** via `Data/download.py` (HNO, HRP, CoD-PS, FTS, CBPF from the 5 official sources — see `Data/README.md`). ~2–3 minutes on broadband.
2. **Create a dashboard virtualenv** at `dashboard/.venv/` and install requirements from `dashboard/requirements.txt` (Streamlit, pandas, plotly, openpyxl).
3. **Download INFORM Severity** — 67 monthly xlsx snapshots from the EU JRC DRMKC (~130 MB, polite 1.2 s inter-request delay so 3–5 min total) and consolidate them into one CSV. This supplies global monthly severity data for attribute `a₄` and the temporal bonus.
4. **Silence Streamlit's first-run email prompt** by writing `~/.streamlit/credentials.toml`.

Subsequent runs skip all of the above and just start the three servers.

## Launch

```bash
./run.sh
```

Opens the landing page in your default browser. `Ctrl-C` in the terminal shuts everything down cleanly.

## What's running after `./run.sh`

| URL | What | Notes |
|---|---|---|
| **<http://localhost:7777>** | **Landing page** | Start here — clean page linking to everything else |
| <http://localhost:8000> | Presentation | reveal.js deck · `S` speaker · `F` fullscreen · `?` shortcuts · `?print-pdf` for PDF export |
| <http://localhost:8501> | Data exploration | Streamlit landscape explorer across the 5 primary sources + INFORM Severity |
| <http://localhost:8502> | Analysis (unsupervised) | Streamlit site: PCA, k-means, hierarchical, t-SNE, temporal archetypes, correlation |
| `proposal/proposal.pdf` | Proposal | Also reachable from the landing page card |

## Running pieces individually

If you only need one surface, or `./run.sh`'s ports conflict with something else:

```bash
# Download data (if not already done; idempotent)
python3 Data/download.py
python3 Data/download.py --check       # report what's on disk, no download

# Presentation on :8000
python3 -m http.server 8000 --directory presentation

# Data exploration on :8501 (requires one-time venv setup below)
dashboard/.venv/bin/streamlit run dashboard/app.py

# Analysis on :8502 (shares the same venv)
dashboard/.venv/bin/streamlit run analysis/app.py --server.port 8502

# Landing page on :7777
python3 -m http.server 7777 --directory landing
```

One-time dashboard venv (if not using `run.sh`):

```bash
python3 -m venv dashboard/.venv
dashboard/.venv/bin/pip install -r dashboard/requirements.txt
```

## If something breaks

- **`Port X already in use`** — another process is on 7777, 8000, or 8501. Either kill it (`lsof -nP -iTCP:<port> -sTCP:LISTEN` then `kill <pid>`) or edit the port at the top of `run.sh`.
- **Streamlit shows a blank page or "Connection error"** — the app crashed or was killed. Re-run `./run.sh`.
- **Downloads fail with 404** — HDX may have re-issued a resource. Regenerate URLs per instructions in `Data/README.md`.
- **Data/ directory exists but files are missing** — `python3 Data/download.py` re-downloads only what's missing.
- **Want to wipe the venv and start over** — `rm -rf dashboard/.venv` then `./run.sh`.

## Repository layout

```
GEO-Insight/
├── README.md                 this file
├── run.sh                    single-command launcher
├── .gitignore
├── task/
│   └── challenge.md          official challenge brief
├── Data/
│   ├── sources.txt           the 5 official source URLs
│   ├── download.py           reproducible downloader (stdlib only)
│   ├── README.md             per-dataset sizes, roles, audit findings
│   ├── Third-Party/IPCInfo/  IPC Population Tracking Tool export (committed)
│   └── {hno,hrp,cod-ps,fts,cbpf}/   downloaded source data (gitignored)
├── literature/
│   ├── applicable/           3 directly applicable papers
│   └── adjacent/             2 adjacent papers
├── proposal/
│   ├── proposal.tex          LaTeX source
│   └── proposal.pdf          compiled (committed for preview)
├── presentation/
│   ├── index.html            reveal.js deck
│   ├── css/theme-un.css      custom theme (oxblood-red single-accent)
│   ├── README.md             editing and export instructions
│   └── vendor/reveal.js/     vendored reveal.js 6.0.1
├── landing/
│   ├── index.html            localhost landing page
│   └── proposal.pdf          symlink to ../proposal/proposal.pdf
├── dashboard/
│   ├── app.py                Streamlit data-exploration dashboard (9 sections)
│   ├── _theme.py             shared UI theme + Plotly template (imported by both apps)
│   ├── requirements.txt      streamlit, pandas, plotly, openpyxl, scikit-learn, scipy
│   ├── README.md             dashboard-specific notes
│   └── .venv/                created by run.sh (gitignored; shared with analysis/)
├── analysis/
│   ├── app.py                Streamlit semantic-analysis app (lens × mode grid)
│   ├── spec.yaml             canonical ontology — all 71 properties, levels, lenses, modes
│   ├── ontology.py           Property + Lens + Mode registry with provenance helpers
│   ├── features.py           L1 loaders, L2 ratios, enriched-frame orchestration
│   ├── aggregations/         L3 concentration · L4 temporal · L5 composites
│   ├── views/                six render modes (atlas, PCA, cluster, profile, cross-lens, validation)
│   └── README.md
├── scripts/
│   ├── export-pdf.sh         slide-deck PDF exporter (experimental)
│   └── refresh_enriched.py   materialise analysis/enriched.parquet for fast cold starts
└── src/                      implementation — to be written
```

## What's in git vs. not

**Tracked**: source code, launcher, proposal (`.tex` + compiled `.pdf`), presentation, landing, dashboard code, literature PDFs (private repo), download script, IPC third-party export, source manifests.

**Not tracked** (reproduced locally on first run of `./run.sh`): `Data/hno/`, `Data/hrp/`, `Data/cod-ps/`, `Data/fts/`, `Data/cbpf/` (~270 MB), `dashboard/.venv/`, LaTeX build artefacts, `__pycache__`, `.DS_Store`.

## Research framing (brief)

Full write-up in [`proposal/proposal.pdf`](proposal/proposal.pdf). In short, we extend the MADM/AHP/MAUT framework of Rye & Aktas (2022) with two new axes — both anchored in explicit gaps in the 2015–2026 humanitarian-analytics literature:

1. **Intra-crisis equity** — cluster-coverage Gini within a single country's HRP. Unmasks aggregate-coverage blindness (Vargas Florez et al. 2015, Eqs. 9–12).
2. **Inter-stakeholder disagreement** — rank variance across four donor-preference profiles. Low variance = *consensus-overlooked*; high variance = *contested* (extends Rye & Aktas Table 13).

Temporal chronic/acute decomposition is the bonus third axis. Validated against two independent benchmarks: CERF Underfunded Emergencies (consensus axis) and CARE *Breaking the Silence* (sector-starved axis).

## Status & conventions

The semantic stack (L1 observations → L5 composites) is complete. Every property declared in `analysis/spec.yaml` resolves either as a scalar column on the enriched frame or via a long-form helper in `features.py` (sector-level, donor-level). The four donor-profile gap scores, four-cell typology, and median-rank / rank-IQR pair are all computed and validated against two independent benchmarks.

- **Single source of truth** — `analysis/spec.yaml` declares every property with its formula, source, inputs, unit, and known failure modes. No analytic column exists in code that isn't registered there.
- **Data-quality transparency** is a first-class design principle. Stale HNO, missing sector data, plan-less crises are flagged — never silently imputed.
- **No false precision**. Scores to two significant figures; inter-profile rank-IQR always shown alongside point estimates.
- **Decision support, not automation**. The tool ranks; humans decide.
- **Commits** go to `main` directly for now (small team, fast iteration). Branch/PR if a change is substantive or risky.
