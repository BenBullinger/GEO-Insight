# External validation benchmarks

Two independently-curated, human-produced lists of "overlooked" humanitarian crises. They cover different dimensions of *overlooked-ness*:

| File | Axis | Produced by | Signal | Update frequency |
|---|---|---|---|---|
| `cerf_ufe.csv` | **Underfunding** (consensus axis) | OCHA · Central Emergency Response Fund | CERF UFE allocations ($110M + $100M rounds) go to "protracted or neglected humanitarian crises where funding is low but vulnerability and risk levels are high" | Twice yearly (March + December allocation windows) |
| `care_bts.csv` | **Under-reporting** (sector-starved / forgotten axis) | CARE International | Annual ranking of the top-10 humanitarian crises that received the least media attention the previous year | Annual (January report covering previous year) |

These are both *human-curated* and *use methodology independent of ours*. We can therefore validate our composite gap score (Level-5) against them without circularity: when our `median_rank` top-K overlaps with a benchmark, that's genuine cross-method agreement.

## Current contents

### `cerf_ufe.csv`
- **2024, window 2** — late-2024 $110M allocation: BFA, BDI, CMR, ETH, HTI, MWI, MLI, MOZ, MMR, YEM (10 countries)
- **2025, window 1** — March 2025 $110M allocation: SDN, TCD, AFG, CAF, HND, MRT, NER, SOM, VEN, ZMB (10 countries)
- **2025, window 2** — December 2025 $100M allocation: BFA, COD, HTI, MLI, MOZ, MMR, SYR (7 countries)

Source: CERF press releases and the OCHA page at <https://cerf.un.org/what-we-do/underfunded-emergencies>.

### `care_bts.csv`
- **2024** — the 10 most under-reported humanitarian crises of 2024 (all African): AGO, CAF, MDG, BFA, BDI, MOZ, CMR, MWI, ZMB, NER.

Source: CARE International's "Breaking the Silence" 2024 report at <https://www.care.org/resources/breaking-the-silence-2024/>.

## How to update

When the next CERF UFE window is announced:

1. Find the press release under <https://cerf.un.org/what-we-do/underfunded-emergencies>.
2. Append rows to `cerf_ufe.csv` with `year`, `window` (1 = first half, 2 = second half), `iso3`, `country_name`.

When CARE publishes a new Breaking the Silence report (typically January):

1. Fetch the ranked top-10 from the new annual report.
2. Append rows to `care_bts.csv` with `year`, `rank` (1 = most under-reported), `iso3`, `country_name`.

Both files are small and human-maintainable; no scraping code is needed. If either file is missing or empty, the validation view in the analysis app simply reports the benchmark as unavailable.
