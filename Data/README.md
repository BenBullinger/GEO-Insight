# Data

Raw source datasets for GEO-Insight. None of the CSV/JSON files in this folder are tracked in git (too large for GitHub; ~270 MB total). The repository ships the **download script** and the **source manifest**; everyone reproduces their local data directory with one command.

## Reproducing the data

```bash
cd GEO-Insight
python3 Data/download.py          # download missing files
python3 Data/download.py --check  # list what's present / missing
python3 Data/download.py --force  # re-download everything
```

Python 3 standard library only — no `pip install` step.

## Sources

Five official sources from the challenge brief, listed in `sources.txt`:

1. [HNO — Humanitarian Needs Overview](https://data.humdata.org/dataset/global-hpc-hno) → `hno/`
2. [HRP — Humanitarian Response Plans](https://data.humdata.org/dataset/humanitarian-response-plans) → `hrp/`
3. [CoD-PS — Global subnational population](https://data.humdata.org/dataset/cod-ps-global) → `cod-ps/`
4. [FTS — Global requirements & funding](https://data.humdata.org/dataset/global-requirements-and-funding-data) → `fts/`
5. [CBPF Pooled Funds Data Hub](https://cbpf.data.unocha.org) → `cbpf/` (resolved to the HDX dataset `cbpf-allocations-and-contributions`, which is what the portal serves)

## Local folder layout after a full download

```
Data/
├── sources.txt                                            the 5 URLs from the brief
├── download.py                                            this script
├── README.md                                              this file
├── hno/
│   ├── hpc_hno_2024.csv              ~31 MB   PIN by country × sector
│   ├── hpc_hno_2025.csv              ~26 MB
│   └── hpc_hno_2026.csv              ~  7 KB   (current cycle, still thin)
├── hrp/
│   ├── humanitarian-response-plans.csv                ~128 KB
│   └── hpc_tools_plans.json                           ~3.8 MB   (live API snapshot)
├── cod-ps/
│   ├── cod_population_admin0.csv     ~786 KB   country totals          ← we use this
│   ├── cod_population_admin1.csv     ~11 MB    province / state
│   ├── cod_population_admin2.csv     ~144 MB   district
│   ├── cod_population_admin3.csv     ~35 MB    sub-district
│   └── cod_population_admin4.csv     ~4.6 MB   lowest unit
├── fts/
│   ├── fts_requirements_funding_global.csv            ~272 KB   country × plan × year
│   ├── fts_requirements_funding_cluster_global.csv    ~1.0 MB   ← critical: cluster-level for the equity Gini
│   ├── fts_requirements_funding_globalcluster_global.csv ~1.3 MB   global-cluster rollups
│   ├── fts_requirements_funding_covid_global.csv      ~24 KB
│   ├── fts_incoming_funding_global.csv                ~3.9 MB   donor flows in
│   ├── fts_internal_funding_global.csv                ~460 KB
│   └── fts_outgoing_funding_global.csv                ~1.3 MB
└── cbpf/
    ├── cbpf_project_summary.csv      ~9.1 MB   project-level allocations
    └── cbpf_contributions.csv        ~812 KB   donor contributions to CBPFs
```

## Which files the methodology actually needs

Critical for the core scoring (Phase 1 MVP):
- `hno/hpc_hno_2025.csv` (and `2024.csv` for temporal comparisons) — people in need
- `fts/fts_requirements_funding_global.csv` — country-level funding gap
- `fts/fts_requirements_funding_cluster_global.csv` — **load-bearing** for the intra-crisis equity Gini term
- `hrp/humanitarian-response-plans.csv` — plan status and requirements
- `cod-ps/cod_population_admin0.csv` — per-capita normalisation

Used for enrichment / bonus task:
- `fts/fts_incoming_funding_global.csv` — donor concentration (HHI)
- `cbpf/cbpf_project_summary.csv` — pooled fund allocations
- `cod-ps/cod_population_admin1.csv` — sub-national analysis (stretch goal)

The admin2/3/4 population files are pulled in completeness but are not required for the country-level methodology.

## Known data-audit findings

Audit run 2026-04-18, full mapping in `proposal/proposal.pdf` §4.1–4.4:

- **Severity (methodology attribute $a_4$) is NOT in HNO.** HNO CSVs have PIN/Targeted/Reached but no structured 1–5 severity score — only a free-text `Description` field. Source for $a_4$: IPC phase classification from [ipcinfo.org](https://www.ipcinfo.org/) via their free registered API. Non-food crises degrade to a plan-typology flag.
- **Cluster names disagree between HNO and FTS.** E.g. HNO `"Sanitation & Hygiene"` vs. FTS `"Water Sanitation and Hygiene"` vs. French `"EHA - Eau Hygiène Assainissement"`. A manual mapping table will live at `src/taxonomies/cluster_map.csv` (~20–30 entries) and is load-bearing for the equity Gini.
- **FTS incoming records sometimes span multiple destination countries** via pipe-delimited `destLocations`. HHI computation applies a lead-country rule or splits equally across listed destinations and flags the crisis as `estimated`.
- **HNO 2026 is a stub** (~7 KB, ~100 rows) — plan cycles not finalised yet. Primary analysis uses 2024 + 2025.
- **Validation benchmarks** (CERF UFE, CARE *Breaking the Silence*) are not included in the core download — they are scraped separately once per cycle.

Verdict: all six attributes recoverable from public sources; no kill blocker.

## If a download 404s

HDX resource IDs are generally stable, but if OCHA re-issues a dataset the URL hash changes. To regenerate URLs:

```bash
curl -sL "https://data.humdata.org/api/3/action/package_show?id=<slug>" | \
    python3 -c "import json,sys; [print(r['url']) for r in json.load(sys.stdin)['result']['resources']]"
```

Slugs to try: `global-hpc-hno`, `humanitarian-response-plans`, `cod-ps-global`, `global-requirements-and-funding-data`, `cbpf-allocations-and-contributions`.

Then edit the `DATASETS` list in `download.py` and re-run.
