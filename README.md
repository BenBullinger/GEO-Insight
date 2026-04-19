# Geo-Insight

**Which humanitarian crises are most overlooked?**

In 2026, agencies are asking $33B to reach 135 M people; the 2025 appeal raised $12B, the lowest in a decade, with 19 contexts below 50 % coverage. When nearly every response is underfunded, a single coverage ratio stops discriminating. Geo-Insight treats overlookedness as a *latent scalar* θ and infers its posterior from six observed signals through a hierarchical Bayesian observation model. Every ranking carries a calibrated 90 % credible interval; the variational fit is validated against NUTS on every run.

Datathon 2026.

---

## Headline result

External validation against two human-curated benchmarks produced by methodology entirely independent of anything in this repo:

| Benchmark (selection date)     | k    | Bayesian      | Additive baseline |
| ------------------------------ | ---- | ------------- | ----------------- |
| CERF UFE 2024 w2 (Dec 2024)    | 10   | 3/10          | 3/10              |
| CERF UFE 2025 w1 (Mar 2025)    | 10   | **5/10**      | **5/10**          |
| CERF UFE 2025 w2 (Dec 2025)    | 7    | **2/7**       | 1/7               |
| CARE BTS 2024                  | 10   | **3/10**      | 1/10              |

The Bayesian model ties the additive baseline on the two larger CERF windows and beats it on the smaller windows and on CARE BTS, where the baseline collapses. **Held-out test:** a 2024-only fit predicts CERF's March-2025 selections at 4/10 precision @ 10 — without seeing any 2025 data. **Year-over-year stability:** 7/10 top-10 overlap between cycles.

**Calibration:** the production `AutoMultivariateNormal` SVI guide runs in 3 s and recovers NUTS posterior medians at Spearman ρ = 0.89, with credible intervals within 2× of NUTS. The simpler mean-field guide gives ρ = 0.68 and underestimates posterior variance by ~3× — a finding that surfaced only because we ran NUTS, and the reason we ship the multivariate guide.

**Posterior predictive checks:** marginal coverage of the 90 % predictive interval is ≥ 0.91 for every attribute (target 0.90). Per-country Pearson correlation reveals the latent is best read as a *funding-overlookedness axis*: coverage shortfall (r = 0.76) and donor concentration (r = 0.82) drive it; need intensity, intra-crisis equity, and severity sit near attribute ceilings and constrain the posterior without differentiating.

## Top-10 most overlooked, 2025

HND, SLV, MOZ, SOM, GTM, NER, HTI, CMR, VEN, TCD.

The 90 % credible intervals overlap substantially across the top-10: set membership is the defensible claim, ordering within is less so. Posterior medians and CIs in [`proposal/proposal.pdf`](proposal/proposal.pdf) Table 2 and live in the analysis dashboard at <http://localhost:8502>.

---

## Run it

```bash
git clone git@github.com:BenBullinger/GEO-Insight.git
cd GEO-Insight
./run.sh
```

Opens the landing page in your default browser. `Ctrl-C` shuts everything down cleanly. First run downloads ~270 MB of source data and creates a Python virtualenv (~3–5 minutes total); subsequent runs start in seconds. Everything runs on localhost — no public hosting.

| Surface             | URL                                                | Description                                                                                  |
| ------------------- | -------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **Landing**         | <http://localhost:7777>                            | Hero narrative, interactive globe of the top-10, headline validation table                   |
| Methodology         | <http://localhost:7777/methodology/>               | Eight-section technical walkthrough — latent θ, likelihoods, priors, validation, calibration |
| Presentation        | <http://localhost:8000>                            | Twelve-slide research narrative (reveal.js)                                                  |
| Data exploration    | <http://localhost:8501>                            | Provenance audit across the five primary sources + INFORM                                    |
| Analysis            | <http://localhost:8502>                            | Eight lenses × six modes, Bayesian posterior atlas, external validation                      |
| Proposal            | [`proposal/proposal.pdf`](proposal/proposal.pdf)   | Eight pages, methodology + validation + results                                              |

Reveal.js shortcuts: `S` speaker view, `F` fullscreen, `?` keyboard shortcuts, `?print-pdf` for PDF export.

---

## The model in one paragraph

Overlookedness θ is a latent scalar per country. We never observe θ directly; we observe six attributes — funding coverage shortfall, unmet amount per person in need, fraction of population in need, INFORM severity category, donor concentration (Herfindahl index), and intra-crisis sector equity (PIN-weighted Gini) — through likelihoods matched to each attribute's support: **Beta regression** for the four bounded fractions, **log-normal** for the per-PIN gap, **ordered logistic** for INFORM severity. The six slopes are sign-constrained positive (HalfNormal priors) so larger attribute values mean more overlooked. A Gaussian population-level prior on θ pools partial-data countries toward the global mean, so sparse-data countries get wider posteriors structurally — not because we down-weight them by hand, but because the likelihood has fewer terms to sharpen the posterior on θ. Stakeholder differences (CERF, ECHO, USAID, NGO) are encoded as differences between priors over slopes, not weights in a sum; each stakeholder produces a separate posterior whose overlap defines consensus and contested crises. Inference is variational with `AutoMultivariateNormal` (3 s on CPU) and validated against NUTS on every run. Implementation in [`analysis/bayesian/hierarchical.py`](analysis/bayesian/hierarchical.py). Full math in [`proposal/methodology.md`](proposal/methodology.md).

## The candidate set

The model is fitted on **HRP-eligible countries** — those with an active humanitarian response plan (operationalised as `per_pin_gap` observed). This is the population CERF Underfunded Emergencies allocations are drawn from, so external validation is on the same pool. Countries without an HRP return no posterior — the construct "overlooked humanitarian crisis" is not well-defined for them, and their absence from the ranking is a feature, not a bug. In the 2025 cycle this is 22 countries.

## Validation regime

Three pre-registered validations, each falsifies a different claim:

1. **External benchmarks** — posterior median ranking scored against CERF UFE (3 windows) and CARE BTS. Reproducible via [`analysis/bayesian/hierarchical.py`](analysis/bayesian/hierarchical.py); precision @ k against an additive baseline reported in the table above.
2. **Posterior predictive checks** — for each of the six attributes, 1,000 replicates drawn from the posterior and overlaid on the observed marginal. Marginal coverage ≥ 0.91 for every attribute. Reproducible via [`analysis/bayesian/ppc.py`](analysis/bayesian/ppc.py); figure at [`proposal/figures/fig_ppc.png`](proposal/figures/fig_ppc.png).
3. **Temporal held-out** — fit on 2024 cycle data alone, score against CERF UFE selections made in 2025. The 2024 fit predicts CERF's March-2025 picks at 4/10 without seeing any 2025 data. Reproducible via [`analysis/bayesian/temporal_holdout.py`](analysis/bayesian/temporal_holdout.py).

NUTS calibration runs alongside SVI in `hierarchical.py main()` — the calibration ratio (Spearman ρ = 0.89, CI width within 2×) is reported on every run.

---

## Repository layout

```
GEO-Insight/
├── README.md                      this file
├── run.sh                         single-command launcher
├── Data/
│   ├── sources.txt                the 5 official source URLs
│   ├── download.py                reproducible downloader (stdlib only)
│   ├── README.md                  per-dataset sizes, roles, audit findings
│   ├── Third-Party/
│   │   ├── Benchmarks/            CERF UFE + CARE BTS curated lists (validation only)
│   │   ├── DRMKC-INFORM/          INFORM Severity 67 monthly snapshots, consolidator
│   │   └── IPCInfo/               IPC Population Tracking Tool export
│   └── {hno,hrp,cod-ps,fts,cbpf}/ official source data (downloaded, gitignored)
├── proposal/
│   ├── proposal.tex               LaTeX source
│   ├── proposal.pdf               compiled (committed)
│   ├── methodology.md             long-form methodology (8 sections)
│   ├── metric_cards.md            seven metric cards including the posterior card
│   └── figures/
│       ├── _gen.py                matplotlib figure generation
│       ├── fig1–fig6              pedagogical figures (semantic stack, observation
│       │                          model, AR(1), stakeholder priors, consensus/contested,
│       │                          rank forest)
│       └── fig_ppc.png            posterior predictive check (six attributes)
├── presentation/
│   ├── index.html                 reveal.js deck (12 slides + dividers)
│   ├── css/theme-un.css           oxblood-red single-accent theme
│   └── vendor/reveal.js/          vendored reveal.js 6.0.1
├── landing/
│   ├── index.html                 hero, globe, evidence, surfaces
│   ├── globe.js                   D3 orthographic globe of the top-10
│   ├── methodology/index.html     eight-section methodology walkthrough
│   ├── shared.css                 design tokens (oxblood, font stack, spacing)
│   └── proposal.pdf               symlink to ../proposal/proposal.pdf
├── dashboard/                     port 8501 — raw data audit
│   ├── app.py                     Streamlit, 7 sections, one per source
│   ├── _theme.py                  shared UI theme + Plotly template
│   └── requirements.txt           streamlit, pandas, plotly, openpyxl, scikit-learn,
│                                  scipy, numpyro, jax
├── analysis/                      port 8502 — semantic-analysis app
│   ├── app.py                     lens × mode grid, Featured-views shortcut strip
│   ├── spec.yaml                  canonical ontology — 71 properties, 8 lenses, 6 modes
│   ├── ontology.py                Property + Lens + Mode registry with provenance
│   ├── features.py                L1 loaders, L2 ratios, enriched-frame orchestration
│   ├── validation.py              CERF UFE / CARE BTS loaders + overlap @ k metrics
│   ├── aggregations/
│   │   ├── concentration.py       L3 — donor HHI, cluster Gini
│   │   ├── temporal.py            L4 — baselines, deltas, persistence
│   │   └── composites.py          L5 — calls the Bayesian model, returns posterior
│   ├── bayesian/
│   │   ├── mvp.py                 cross-sectional MVP, shared NumPyro machinery
│   │   ├── hierarchical.py        production model: sign-constrained slopes, hier prior,
│   │   │                          MVN SVI + NUTS calibration on every run
│   │   ├── ppc.py                 posterior predictive checks → fig_ppc.png
│   │   └── temporal_holdout.py    2024-fit vs 2025 CERF UFE prediction
│   └── views/                     six render modes (atlas, pca, cluster, profile,
│                                  cross_lens, validation)
└── scripts/
    ├── export-pdf.sh              slide-deck PDF exporter (experimental)
    └── refresh_enriched.py        materialise analysis/enriched.parquet for fast cold starts
```

## What's in git, what isn't

**Tracked**: source code, launcher, proposal (`.tex` + compiled `.pdf`), methodology, metric cards, presentation, landing, dashboards, ontology spec, third-party benchmark lists, IPC export, source manifests.

**Not tracked** (reproduced locally on first run of `./run.sh`): `Data/{hno,hrp,cod-ps,fts,cbpf}/` (~270 MB), INFORM monthly snapshots (~130 MB), `dashboard/.venv/`, LaTeX build artefacts, `__pycache__`, `.DS_Store`, `analysis/enriched.parquet`.

---

## Setup details

**Prerequisites**

- **Python 3.9+** (check with `python3 --version`). Standard library is enough for downloads and the launcher; only the dashboards need pip-installed packages, and `run.sh` handles that.
- **A Unix-like shell** with `bash`. Tested on macOS. Linux works. Windows users: use WSL.
- **Disk space**: ~280 MB after the one-time data download.

No Node, no npm, no Docker.

**What `./run.sh` does on first run**

1. Downloads ~270 MB of source data via `Data/download.py` (HNO, HRP, CoD-PS, FTS, CBPF from the 5 official sources — see `Data/README.md`).
2. Creates a virtualenv at `dashboard/.venv/` and installs `dashboard/requirements.txt` (Streamlit, pandas, plotly, openpyxl, scikit-learn, scipy, numpyro, jax).
3. Downloads INFORM Severity — 67 monthly xlsx snapshots from the EU JRC DRMKC (~130 MB, polite 1.2 s inter-request delay so 3–5 min total) and consolidates them into a long-format CSV. Supplies global monthly severity data for attribute `a₄`.
4. Silences Streamlit's first-run email prompt by writing `~/.streamlit/credentials.toml`.

Subsequent runs skip all of the above and start the four servers in seconds.

**Running pieces individually**

If you only need one surface, or `./run.sh`'s ports conflict with something else:

```bash
# Re-download data (idempotent; --check reports without downloading)
python3 Data/download.py
python3 Data/download.py --check

# Landing on :7777
python3 -m http.server 7777 --directory landing

# Presentation on :8000
python3 -m http.server 8000 --directory presentation

# Data exploration on :8501
dashboard/.venv/bin/streamlit run dashboard/app.py

# Analysis on :8502
dashboard/.venv/bin/streamlit run analysis/app.py --server.port 8502
```

One-time virtualenv (if not using `run.sh`):

```bash
python3 -m venv dashboard/.venv
dashboard/.venv/bin/pip install -r dashboard/requirements.txt
```

**Re-running the model standalone**

```bash
# Production fit + NUTS calibration check + external benchmark scoring
dashboard/.venv/bin/python -m analysis.bayesian.hierarchical

# Posterior predictive checks → updates proposal/figures/fig_ppc.png
dashboard/.venv/bin/python -m analysis.bayesian.ppc

# Temporal held-out: 2024 fit vs 2025 CERF UFE selections
dashboard/.venv/bin/python -m analysis.bayesian.temporal_holdout
```

**If something breaks**

- `Port X already in use` — another process is on 7777 / 8000 / 8501 / 8502. Either kill it (`lsof -nP -iTCP:<port> -sTCP:LISTEN` then `kill <pid>`) or edit the port at the top of `run.sh`.
- Streamlit shows a blank page or "Connection error" — the app crashed or was killed. Re-run `./run.sh`; tail `_logs/*.log`.
- Downloads fail with 404 — HDX may have re-issued a resource. Regenerate URLs per `Data/README.md`.
- `Data/` exists but files are missing — `python3 Data/download.py` re-downloads only what's missing.
- Want to wipe the venv and start over — `rm -rf dashboard/.venv` then `./run.sh`.

---

## Conventions and discipline

- **Single source of truth.** [`analysis/spec.yaml`](analysis/spec.yaml) declares every property with its formula, source, inputs, unit, and known failure modes. No analytic column exists in code that isn't registered there.
- **Honest uncertainty.** Every score is a posterior; every ranking comes with a 90 % credible interval; the variational fit is calibrated against NUTS on every run.
- **Honest scope.** The model fits HRP-eligible countries only. Countries without an active response plan return no posterior, not a guess.
- **Data-quality transparency** is a first-class design principle. Stale HNO, missing sector data, plan-less crises are flagged — never silently imputed.
- **Decision support, not automation.** The tool ranks; humans decide. Every displayed value carries its provenance.
- **Commits** go to `main` directly for now (small team, fast iteration). Branch/PR if a change is substantive or risky.

## What we are not claiming

- We are not forecasting. The model describes θ for observed cycles only. The temporal AR(1) extension above is the handle for forward prediction; it requires more time points than the annual HRP/FTS attributes give us, and is not yet shipped.
- We are not inferring causality. The observation model is statistical association, not intervention.
- We are not replacing judgement. Every ranking is decision support.
- We have not observed overlookedness directly. θ is latent by definition; every claim is contingent on the observation model being approximately correct. Validation gives evidence; it does not give proof.
- The benchmarks themselves are imperfect proxies — CERF UFE reflects a political process; CARE BTS reflects media attention.

---

## Where this goes next

In priority order:

1. **Stakeholder posteriors.** The four pre-registered slope priors (CERF, ECHO, USAID, NGO) are encoded; fitting them as four separate posteriors and measuring pairwise overlap turns the consensus-vs-contested typology from a methodological commitment into an empirical readout.
2. **Temporal AR(1).** A per-country AR(1) on θ requires monthly attributes; INFORM severity is monthly and would anchor a partial-temporal model with chronic ($\mu_u$) and acute ($\theta - \mu_u$) decompositions falling out of the same posterior.
3. **Learned representation layer.** A sequence model (Mamba, masked-attention encoder) trained on INFORM's eighteen monthly sub-indicators would produce an embedding $z(u, t)$ to feed alongside the six aggregates: replace $a_i \mid \theta$ with $\{a, z\} \mid \theta$. The latent stays interpretable; the encoder has a crisp objective (reduce posterior variance on θ).

Each extension is held to the same discipline: declared properties, declared failure modes, posteriors with credible intervals, decision support not automation.

---

## References

- **Proposal:** [`proposal/proposal.pdf`](proposal/proposal.pdf) — eight pages.
- **Long-form methodology:** [`proposal/methodology.md`](proposal/methodology.md) — eleven sections.
- **Metric cards:** [`proposal/metric_cards.md`](proposal/metric_cards.md) — definition, source, failure modes for every aggregate.
- **External benchmarks:** [`Data/Third-Party/Benchmarks/README.md`](Data/Third-Party/Benchmarks/README.md) — CERF UFE + CARE BTS curation policy.
- **Data sources:** [`Data/README.md`](Data/README.md) — per-dataset sizes, roles, audit findings.
