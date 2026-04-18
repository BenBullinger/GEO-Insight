# GEO-Insight

**Which humanitarian crises are most overlooked?**

GEO-Insight is a decision-support prototype developed for the ETH Zürich Datathon 2026. Given a query or geographic scope, the system is intended to rank active humanitarian crises by the mismatch between documented need (from HNO / HRP data) and available funding coverage (from OCHA FTS and CBPF pooled funds), with an explanation of why each top-ranked crisis appears overlooked.

This repository is work in progress. The current contents scope the problem, curate relevant literature, and sketch a provisional methodology. An implementation will follow; nothing here is final.

## The problem

Humanitarian coordinators and donor advisors need to answer questions like:

- Which crises have the highest people-in-need but the lowest fund allocations?
- Are there countries with active HRPs where funding is absent or negligible?
- Which regions are consistently underfunded relative to need across multiple years?
- Show me acute food insecurity hotspots that have received less than 10% of their requested funding.

The analytical challenge is that humanitarian data blends two very different signals: *objective severity* (the scale and urgency of a crisis) and *funding coverage* (what has actually been resourced). A defensible ranking has to separate these layers and then recombine them with weights that can be audited.

The full task brief is in [`task/challenge.md`](task/challenge.md).

## Why this matters now (April 2026)

- The 2026 Global Humanitarian Overview asks **$33B to reach 135M people**, with a hyper-prioritised floor of $23B for 87M lives.
- The 2025 appeal raised only **$12B** — the lowest in a decade.
- Headline coverage ratios: DRC ~22%, Yemen ~24%, Somalia ~24%, Sudan dropped from 69% (2024) to 35% (2025).
- **19 country contexts** are now severely underfunded (<50% coverage) — up from 8 in 2021.

When nearly everything is underfunded, the useful question is no longer *which crises are funded?* but *relative to documented need, which crises are most overlooked?*

## Primary data sources

| Dataset | Role |
|---|---|
| [HNO — Humanitarian Needs Overview](https://data.humdata.org/dataset/global-hpc-hno) | People in need by country and sector |
| [HRP — Humanitarian Response Plans](https://data.humdata.org/dataset/humanitarian-response-plans) | Plan targets, status, requirements |
| [FTS — Requirements & Funding](https://data.humdata.org/dataset/global-requirements-and-funding-data) | Requested, pledged, received funding |
| [CoD-PS — Global population](https://data.humdata.org/dataset/cod-ps-global) | Denominator for per-capita normalisation |
| [CBPF Pooled Funds](https://cbpf.data.unocha.org/) | Country-based pooled fund allocations |
| [HDX HAPI](https://data.humdata.org/dataset/hdx-hapi-funding) | Unified, daily-refresh API over the above |

**Validation benchmark:** OCHA's own [CERF Underfunded Emergencies](https://cerf.un.org/apply-for-a-grant/underfunded-emergencies) list — an independent, human-curated source for which crises are considered neglected.

Optional enrichment (declared in outputs if used): ACLED (conflict events), IPC (food security phases), UNHCR (displacement).

## Repository layout

```
GEO-Insight/
├── task/          Challenge brief
├── literature/    Curated reading
│   ├── applicable/   Directly applicable methods
│   └── adjacent/     Adjacent / partial relevance
├── proposal/      Methodology proposal (LaTeX + compiled PDF)
├── src/           Implementation — to be written
└── README.md
```

## Current direction (provisional)

A full write-up is in [`proposal/proposal.pdf`](proposal/proposal.pdf). In short, the working plan draws on a multi-attribute decision-making framework (Rye & Aktas, 2022) — AHP-elicited weights over attributes covering coverage shortfall, per-PIN funding gap, need intensity, severity, and donor concentration, composed via a MAUT additive score — with a temporal decomposition separating chronic neglect from acute deterioration for the bonus task.

The design principles are firmer than the specific methodology:

- **Decision support, not automation.** The tool produces rankings; humans decide.
- **Explainable by construction.** Every weight is inspectable; every score decomposes into per-attribute contributions.
- **Honest about data quality.** Stale HNO figures, missing sector breakdowns, and plan-less crises are flagged, not imputed.
- **No false precision.** Scores reported to two significant figures; weight-sensitivity bands shown alongside point values.

Details, validation strategy, failure modes, and phased delivery plan are in the proposal.

## Status

Work in progress. Core implementation (`src/`) has not yet started. Next step is a Phase 1 MVP: country-level ranking notebook validated against CERF UFE overlap on the current plan year.

## Team

ETH Zürich — Datathon 2026 (FS26).
