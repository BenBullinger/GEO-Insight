# GEO-Insight

**Which humanitarian crises are most overlooked?**
Cache Me if You Can — Datathon 2026 · FS26

A decision-support prototype that ranks active humanitarian crises by the mismatch between documented need (HNO/HRP) and available funding coverage (FTS/CBPF), with an intra-crisis equity term and an inter-stakeholder disagreement term. Research direction and methodology live in `proposal/proposal.pdf`; this README is the operational guide to getting everything running locally.

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
2. **Create a dashboard virtualenv** at `dashboard/.venv/` and install requirements from `dashboard/requirements.txt` (Streamlit, pandas, plotly).
3. **Silence Streamlit's first-run email prompt** by writing `~/.streamlit/credentials.toml`.

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
| <http://localhost:8501> | Dashboard | Streamlit data-landscape explorer across the 5 sources |
| `proposal/proposal.pdf` | Proposal | Also reachable from the landing page card |

## Running pieces individually

If you only need one surface, or `./run.sh`'s ports conflict with something else:

```bash
# Download data (if not already done; idempotent)
python3 Data/download.py
python3 Data/download.py --check       # report what's on disk, no download

# Presentation on :8000
python3 -m http.server 8000 --directory presentation

# Dashboard on :8501 (requires one-time venv setup below)
dashboard/.venv/bin/streamlit run dashboard/app.py

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
│   ├── css/theme-un.css      UN-adjacent custom theme
│   ├── README.md             editing and export instructions
│   └── vendor/reveal.js/     vendored reveal.js 6.0.1
├── landing/
│   ├── index.html            localhost landing page
│   └── proposal.pdf          symlink to ../proposal/proposal.pdf
├── dashboard/
│   ├── app.py                Streamlit data-landscape dashboard
│   ├── requirements.txt      streamlit, pandas, plotly
│   ├── README.md             dashboard-specific notes
│   └── .venv/                created by run.sh (gitignored)
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

Implementation (`src/`) not yet started. Immediate next step: Phase 1 MVP — country-level six-attribute scoring under a balanced profile, validated against CERF UFE.

- **Commits** go to `main` directly for now (small team, fast iteration). Branch/PR if a change is substantive or risky.
- **Data-quality transparency** is a first-class design principle. Stale HNO, missing sector data, plan-less crises are flagged — never silently imputed.
- **No false precision**. Scores to two significant figures; inter-profile IQR always shown alongside point estimates.
- **Decision support, not automation**. The tool ranks; humans decide.
