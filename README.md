# GEO-Insight

**Which humanitarian crises are most overlooked?**
Datathon 2026 · ETH Zürich · FS26

A decision-support prototype that ranks active humanitarian crises by the mismatch between documented need (HNO/HRP) and available funding coverage (FTS/CBPF). Our framing (see §[Research direction](#research-direction)) goes beyond a scalar gap ratio: we report a *distribution* of ranks across donor-preference profiles plus an intra-crisis equity correction, producing a four-cell typology (consensus-overlooked / contested / sector-starved / combinations) that existing tools conflate.

## Quick start

```bash
git clone git@github.com:BenBullinger/GEO-Insight.git
cd GEO-Insight
./run.sh
```

That's it. `run.sh` is idempotent: on first run it downloads the ~270 MB of source data, creates the dashboard virtualenv, and installs Streamlit; on subsequent runs it just starts the servers. Ctrl-C shuts everything down cleanly.

Once it's running, the landing page opens in your default browser and links to everything:

- **Landing page** — <http://localhost:7777> (start here)
- **Presentation** — <http://localhost:8000> (reveal.js deck · `S` speaker · `F` fullscreen · `?` shortcuts)
- **Dashboard** — <http://localhost:8501> (data-landscape explorer across the 5 sources)
- **Proposal** — `proposal/proposal.pdf` (also reachable from the landing page)

Nothing is hosted publicly. Everything is localhost-only.

### Running pieces individually

If you want just one surface up (or `run.sh` conflicts with something else on 8000/8501):

```bash
python3 Data/download.py             # one-time: pull ~270 MB source data into Data/
cd presentation && python3 -m http.server 8000               # presentation only
cd dashboard && .venv/bin/streamlit run app.py               # dashboard only (needs venv set up once)
```

Dashboard venv setup (one-time, if not using `run.sh`):

```bash
cd dashboard
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Toolchain: Python 3 stdlib covers everything except the dashboard, which needs `streamlit + pandas + plotly` in a venv (handled automatically by `run.sh`).

## Research direction

Existing tools (CERF UFE, FTS coverage ratios, Palantir-Foundry–based systems like WFP's DOTS) score overlookedness as a **scalar**. We treat it as a **joint distribution** over stakeholder preferences, extending the MADM/AHP/MAUT framework of Rye & Aktas (2022) with two new axes — each anchored in an explicit gap in the 2015–2026 literature:

1. **Intra-crisis equity** — cluster-coverage Gini within a single country's HRP, following the fair-distribution formulation of Vargas Florez et al. (2015, Eqs. 9–12). Unmasks aggregate-coverage blindness.
2. **Inter-stakeholder disagreement** — rank variance across four donor-preference profiles (CERF-, ECHO-, USAID-, NGO-consortium-flavoured). Low variance = *consensus-overlooked*; high variance = *contested*.

Chronic/acute temporal decomposition remains the bonus task (third axis). Validated against **two independent benchmarks**: CERF Underfunded Emergencies (consensus axis) and CARE *Breaking the Silence* (sector-starved axis).

Literature convergence: Abdulrashid et al. 2026 (§7.3 + Table 12), Sahebjamnia et al. 2017 (§6), Vargas Florez et al. 2015 (Eqs. 9–12) all name equity-aware modelling as open.

Full write-up with equations and validation strategy: [`proposal/proposal.pdf`](proposal/proposal.pdf).

## Why it matters now (April 2026)

- 2026 GHO asks **$33B** for 135M people; 2025 raised only **$12B** — the lowest in a decade.
- Coverage: DRC ~22%, Yemen ~24%, Somalia ~24%, Sudan 69%→35% year-on-year.
- **19 contexts** severely underfunded (<50%), up from 8 in 2021.

When almost every response is underfunded, a single coverage ratio stops being discriminative.

## Repository layout

```
GEO-Insight/
├── README.md                 this file
├── run.sh                    single-command launcher (landing + presentation + dashboard)
├── landing/
│   ├── index.html            clean landing page linking to all surfaces
│   └── proposal.pdf          symlink to ../proposal/proposal.pdf
├── .gitignore
├── task/
│   └── challenge.md          official challenge brief
├── Data/
│   ├── sources.txt           the 5 official source URLs
│   ├── download.py           reproducible downloader (stdlib only)
│   ├── README.md             per-dataset details, sizes, which files the methodology needs
│   └── {hno, hrp, cod-ps, fts, cbpf}/      downloaded CSVs/JSONs (gitignored)
├── literature/
│   ├── applicable/           3 directly applicable papers (MADM, AI-DSS review, hybrid DSS)
│   └── adjacent/             2 adjacent papers (robust facility location, embedded analytics)
├── proposal/
│   ├── proposal.tex          LaTeX source
│   └── proposal.pdf          compiled (committed for preview convenience)
├── presentation/
│   ├── index.html            reveal.js deck (slide content lives here)
│   ├── css/theme-un.css      UN-adjacent custom theme, no UN branding
│   ├── README.md             how to run / export / edit slides
│   └── vendor/reveal.js/     vendored reveal.js 6.0.1 — do not edit
├── dashboard/
│   ├── app.py                Streamlit data-landscape dashboard (8 sections)
│   ├── requirements.txt      streamlit, pandas, plotly
│   └── README.md             run instructions + per-page notes
└── src/                      implementation (empty — to be written)
```

## Data sources

| # | Dataset | Role | Local path |
|---|---|---|---|
| 1 | [HNO](https://data.humdata.org/dataset/global-hpc-hno) | People in need by country × sector | `Data/hno/` |
| 2 | [HRP](https://data.humdata.org/dataset/humanitarian-response-plans) | Plan targets, status, requirements | `Data/hrp/` |
| 3 | [CoD-PS](https://data.humdata.org/dataset/cod-ps-global) | Population denominators (admin0–4) | `Data/cod-ps/` |
| 4 | [FTS](https://data.humdata.org/dataset/global-requirements-and-funding-data) | Funding requested/pledged/received (incl. cluster breakdown — load-bearing for the Gini) | `Data/fts/` |
| 5 | [CBPF](https://cbpf.data.unocha.org) | Pooled-fund allocations (resolved to the HDX dataset `cbpf-allocations-and-contributions`) | `Data/cbpf/` |

Validation benchmarks (fetched separately, not via `download.py`): [CERF Underfunded Emergencies](https://cerf.un.org/apply-for-a-grant/underfunded-emergencies) and [CARE Breaking the Silence](https://reliefweb.int/report/angola/breaking-silence-10-most-under-reported-humanitarian-crises-2023).

Per-dataset sizes and which specific files the methodology needs: see [`Data/README.md`](Data/README.md).

## What's in git, what isn't

**Tracked**: source code, the proposal, the deck, the 5 source URLs, the download script, literature PDFs (private repo so paywalled PDFs are OK).
**Not tracked** (reproduced locally): everything under `Data/hno/`, `Data/hrp/`, `Data/cod-ps/`, `Data/fts/`, `Data/cbpf/` (~270 MB). LaTeX build artefacts, `__pycache__`, `.DS_Store`.

## Status & next steps

Phase 1 (MVP) not yet started — `src/` is empty. Immediate work:

1. Country-level ingestion of HNO 2025 + FTS requirements/funding + cluster-level FTS
2. Six-attribute score vector (incl. cluster-coverage Gini), single balanced profile
3. Validation: Overlap@10 against CERF UFE
4. Then: four-profile driver + disagreement IQR + four-cell typology (Phase 2)
5. Then: temporal chronic/acute + NL query layer (Phase 3 / bonus)

## Conventions

- **Commits** go to `main` directly for now (small team, fast iteration). Branch/PR if a change is substantive or risky.
- **Honest-about-data-quality** is a first-class design principle. Stale HNO, missing sector data, plan-less crises are flagged — never silently imputed.
- **No false precision.** Scores to two significant figures; inter-profile IQR always shown alongside point estimates.
- **Decision support, not automation.** The tool ranks; humans decide.
