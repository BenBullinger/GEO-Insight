# NOTICE

Geo-Insight is a hackathon proof-of-concept built for **Datathon 2026**. The
code in this repository is released under the MIT license (see `LICENSE`).
The data *inputs* come from third parties and retain their own licences and
terms of use, summarised below.

## Scope disclaimer

This project is a student hackathon submission. It is **not** a policy
recommendation, **not** a production decision-support system, **not**
peer-reviewed, and **not** endorsed by any of the data providers listed
below. Rankings are statistical estimates with calibrated uncertainty
intervals; they should not be used to allocate humanitarian funding or to
substitute for the judgement of humanitarian coordinators, donors, or
affected communities.

## Third-party code

| Component | Path | Licence |
| --- | --- | --- |
| reveal.js 6.0.1 | `presentation/vendor/reveal.js/` | MIT (see bundled `LICENSE`) |
| D3 7, topojson-client 3, MathJax 3 | loaded from jsDelivr at runtime | BSD / Apache 2.0 / Apache 2.0 |

## Third-party data

All datasets are downloaded at build time from their official sources by the
scripts in `Data/download.py` and `Data/Third-Party/DRMKC-INFORM/download.py`.
Redistributed data in this repo is limited to facts-based derivatives used
for validation.

| Dataset | Provider | Use | Licence / terms |
| --- | --- | --- | --- |
| Humanitarian Needs Overview (HNO) | UN OCHA via HDX | Attribute inputs | Per dataset on HDX; not redistributed |
| Humanitarian Response Plan (HRP) | UN OCHA via HDX | Eligibility gate | Per dataset on HDX; not redistributed |
| Common Operational Dataset — Population Statistics (CoD-PS) | UN OCHA via HDX | Population denominators | Per dataset on HDX; not redistributed |
| Financial Tracking Service (FTS) | UN OCHA | Funding flows | FTS public API; not redistributed |
| Country-Based Pooled Funds (CBPF) | UN OCHA | Pooled fund context | UN OCHA public data; not redistributed |
| INFORM Severity | EU JRC DRMKC | Attribute `a₄` | CC-BY 4.0 — long-format CSV derivatives in `Data/Third-Party/DRMKC-INFORM/` with attribution. Monthly xlsx snapshots are downloaded locally and not redistributed. |
| CERF Underfunded Emergencies allocations | UN OCHA CERF | External validation benchmark | Curated ISO3 lists (`Data/Third-Party/Benchmarks/cerf_ufe.csv`) are facts compiled from public CERF press releases; attribution at <https://cerf.un.org/what-we-do/underfunded-emergencies>. |
| CARE "Breaking the Silence" annual ranking | CARE International | External validation benchmark | Curated ISO3 list (`Data/Third-Party/Benchmarks/care_bts.csv`) is a fact compiled from the publicly-released ranking; attribution at <https://www.care.org/resources/breaking-the-silence-2024/>. "Breaking the Silence" is a trademark of CARE International. |

If you are a rights-holder and believe any content in this repository
exceeds fair use or your licence terms, please open an issue and the
material will be removed.

## Items removed from the public repository

The following categories of material are excluded from the public repo and
listed in `.gitignore`:

- **Academic literature PDFs** (`literature/`) — copyrighted third-party
  articles used only as private reading material during the hackathon.
- **Challenge brief** (`task/`) — Datathon 2026 organiser-owned.
- **IPC Population Analysis export** (`Data/Third-Party/IPCInfo/`) — IPC
  terms prefer redistribution from source; users should download directly
  from <https://www.ipcinfo.org/ipc-country-analysis/population-tracking-tool/en/>.
