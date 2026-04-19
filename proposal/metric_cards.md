# Metric Cards

Every aggregate metric used anywhere in the Geo-Insight pipeline — whether consumed from source data or produced by our scoring — has a card here. Each card captures the formal definition, authoritative source, known discontinuities, failure modes, and our specific handling.

The purpose of this discipline is **defensibility**: any claim we make from an aggregate must be auditable back to a clearly-defined primitive.

---

## 1. INFORM Severity Index (continuous 1–10)

**Source.** EU Joint Research Centre, Disaster Risk Management Knowledge Centre. [INFORM Severity — About](https://drmkc.jrc.ec.europa.eu/inform-index/INFORM-Severity) · [Methodology](https://drmkc.jrc.ec.europa.eu/inform-index/INFORM-Severity/Methodology). Monthly updates; indicator manual revised periodically.

**Definition.** Weighted composite of three pillars — *Impact of the crisis*, *Conditions of people affected*, *Complexity of the crisis* — each itself a weighted composite of roughly 10–15 indicators. Full indicator list and weights in the `Indicator Metadata` sheet of each monthly xlsx.

**Known discontinuities.**
- **Feb 2026 methodology update** rescaled the continuous index from 1–5 to 0–10 while preserving the 1–5 ordinal category. In our consolidated panel, this shows as a jump of the per-snapshot mean from ≈3.07 (pre) to ≈6.34 (post) and maximum from 5.0 to 9.7 — the same week for every country. **This is a scale change, not a real-world deterioration.**
- Indicator additions/revisions over 2020–2026 logged in the xlsx `Log` sheet.
- Some indicator values imputed when source data unavailable (`Imputed and missing data hidden` sheet).

**Failure modes.**
- Face-value time-series comparison across 2026-01 → 2026-02 is invalid for the continuous index.
- Weighted composite is opaque without pillar-level decomposition.
- Monthly re-assessments can shift a crisis's score by ~0.5 on minor indicator revisions.

**Our handling.** We treat the continuous index as **secondary**. Primary severity signal is the ordinal category (see next card), which feeds attribute $a_4$ in the Bayesian model via an ordered-logistic likelihood (§7). For sub-analyses we decompose further — see *INFORM Sub-indicators* card below. In any across-time analysis we annotate the Feb-2026 discontinuity and refuse to compute deltas that straddle it.

---

## 2. INFORM Severity Category (ordinal 1–5)

**Source.** Same as above. Category is the ordinal classification of the continuous index.

**Definition.** 1 = Very Low · 2 = Low · 3 = Medium · 4 = High · 5 = Very High. The binning thresholds on the continuous score were recalibrated in Feb 2026 *to preserve the category distribution* — a crisis classified "4 (High)" before the rescaling typically remains "4 (High)" after.

**Known discontinuities.** None observable in our data: category distribution is stable across the Feb-2026 index rescaling.

**Failure modes.**
- Five-level ordinal loses granularity compared to the continuous score.
- Tie-breaking between (e.g.) `3.4` and `3.6` is not meaningful — ordinal transitions are the only semantically valid comparison.

**Our handling.** Primary severity signal. Feeds attribute $a_4$ in the Bayesian model via an ordered-logistic likelihood (§7), with cut-points inferred from data rather than assumed equally spaced. Stable across the Feb-2026 boundary.

---

## 3. INFORM Sub-indicators (auditable primitives)

**Source.** Each monthly xlsx `Crisis Indicator Data` sheet. Extracted by `Data/Third-Party/DRMKC-INFORM/consolidate_indicators.py` → `inform_indicators_long.csv` (9,590 rows × 109 countries × 65 snapshots, no missingness).

**Primitives extracted.** Per crisis × month:

| Key | Source column | Unit |
|---|---|---|
| `affected` | Total # of people affected by the crisis | people |
| `displaced` | Total # of crisis related displaced people | people |
| `injured`, `fatalities` | Total # of crisis related injuries / fatalities | people |
| `pin_level_1` … `pin_level_5` | # of people affected facing {minimal, stressed, moderate, severe, extreme} humanitarian needs | people |
| `access_limited`, `access_restricted` | People in need facing {limited, restricted} access constraints | people |
| `impediments_bureaucratic` | Impediments to entry into country | categorical/count |

**Known discontinuities.** None observed — these are reported directly by the national teams feeding INFORM. The Feb-2026 composite rescaling did not affect primitives.

**Failure modes.**
- Definitions of "affected" and "displaced" vary slightly across country teams (harmonised by ACAPS guidelines but not perfectly).
- Imputation where source data missing (flagged in the `Data Reliability` sheet per indicator).

**Our handling.** Treated as first-class features for explanation and audit, not as direct inputs to the Bayesian model (the six modelled attributes are higher-level aggregates). When we claim "deterioration" or "sector starvation", we show the sub-indicator that justifies it, not the composite. `pin_level_4 + pin_level_5` is our preferred "severe+ humanitarian conditions" count. These primitives anchor the planned learned representation layer (proposal §7).

---

## 4. FTS Coverage Ratio

**Source.** OCHA [Financial Tracking Service](https://fts.unocha.org). Underlying data in `Data/fts/`.

**Definition.**

$$
C(u, t) \;=\; \frac{F(u, t)}{R(u, t)}
$$

where $F(u,t)$ = reported funding received for crisis $u$ in year $t$, $R(u,t)$ = current plan requirements (revised, not original). `fts_requirements_funding_global.csv` has all three (`requirements`, `funding`, `percentFunded`).

**Known discontinuities.**
- Requirements are revised mid-cycle → early-year snapshots can show higher $C$ than end-of-year.
- "Received" lags disbursement by weeks to months (reporting delay).
- Flash appeals vs. structured HRPs follow different reporting conventions.
- Private-sector contributions are systematically under-reported.

**Failure modes.**
- $C$ near 100% can hide sector-level starvation → cluster Gini ($a_6$, §6) is the model's complementary signal.
- $C > 100%$ occurs (over-funding) → capped at 1.0 before being passed to the Beta-regression likelihood for $a_1$.
- Plan revisions mid-year can make $C_t > C_{t-1}$ without any real funding change.

**Our handling.** Coverage shortfall $a_1 = 1 - \min(C, 1)$ feeds the Bayesian model via a Beta-regression likelihood (§7). Posterior predictive coverage on $a_1$ is calibrated; per-country Pearson $r = 0.76$ between predicted and observed values — coverage shortfall is one of the two attributes the latent θ actively recovers. Display $F$, $R$, and $C$ side-by-side; never $C$ alone.

---

## 5. HHI — Donor Concentration

**Source.** Herfindahl-Hirschman Index, standard in industrial organisation. [OECD glossary](https://stats.oecd.org/glossary/detail.asp?ID=3235). Computed over per-donor contributions from `fts_incoming_funding_global.csv`.

**Definition.**

$$
\mathrm{HHI}(u) \;=\; \sum_{d \in D(u)} s_d^2, \quad s_d = \frac{a_d(u)}{\sum_{d'} a_{d'}(u)}
$$

Range $[1/|D|, 1]$. 1 = single donor, lower = diversified.

**Known discontinuities.**
- Undefined for $|D| < 2$.
- Sensitive to definition of "donor" (government vs. executing agency vs. private foundation).
- FTS aggregates some donors as "Various" — inflates implicit concentration.

**Failure modes.**
- Hides which donor matters. Three donors at 50/30/20 and 80/10/10 can both return similar HHI values — but the stories are different.
- Zero-reporting bias: a crisis with one reported donor and many pledged-but-unreported donors looks overconcentrated.

**Our handling.** Donor HHI feeds attribute $a_5$ in the Bayesian model via a Beta-regression likelihood (§7). Posterior predictive Pearson $r = 0.82$ — donor concentration is the other attribute (alongside coverage shortfall) that the latent θ actively recovers. Always report **HHI + top-1 donor share + number of donors** together for explanation; flag `single-donor-risk` when $|D| = 1$.

---

## 6. Cluster-Coverage Gini — Intra-Crisis Equity ($a_6$)

**Source.** Gini coefficient, standard inequality measure. Applied to cluster-level coverage ratios within a single crisis. Formal template from Vargas Florez et al. (2015, *Engineering Applications of Artificial Intelligence* 46:326–335, Eqs. 9–12).

**Definition.**

$$
G_{\text{cluster}}(u) \;=\; \frac{\displaystyle \sum_{i,j \in \mathcal{C}(u)} P_i(u)\, P_j(u)\, |\,r_i(u) - r_j(u)\,|}{\displaystyle 2 \left(\sum_{c \in \mathcal{C}(u)} P_c(u)\right)^2 \bar r(u)}
$$

PIN-weighted Gini of cluster coverage ratios $r_c = F_c / R_c$.

**Known discontinuities.**
- Cluster-taxonomy mismatch between HNO (e.g., "Sanitation & Hygiene") and FTS (e.g., "Water Sanitation and Hygiene") can silently inflate or deflate the Gini. Harmonised via `src/taxonomies/cluster_map.csv`.
- Clusters with zero requirements are dropped (cannot compute $r_c$).
- Undefined for crises with only one active cluster.

**Failure modes.**
- Summary number hides *which* cluster is starved.
- A crisis with one overfunded cluster ($r=1.5$) and many uniformly underfunded ($r \approx 0.3$) can return similar Gini to one with moderate, varied coverage.
- Does not account for absolute PIN magnitude of the affected cluster.

**Our handling.** Cluster Gini feeds attribute $a_6$ in the Bayesian model via a Beta-regression likelihood (§7). Posterior predictive checks confirm marginal calibration but reveal weak per-country correlation (Pearson $r = 0.26$) — Gini constrains the latent posterior without strongly differentiating between countries; for ranking explanations, always render cluster-by-cluster coverage alongside the Gini and name the lowest-$r_c$ cluster.

---

## 7. Posterior over latent overlookedness $\theta$ — the Geo-Insight readout

**Source.** Defined in `proposal/proposal.pdf` §3 and `proposal/methodology.md` §3–6. Implemented in `analysis/bayesian/hierarchical.py`; cached at Level 5 of the typed semantic layer (`theta_median`, `theta_ci_lo`, `theta_ci_hi`, `theta_ci_width`).

**Definition.** For each HRP-eligible country $u$, $\theta(u)$ is the latent scalar overlookedness, inferred from a hierarchical Bayesian model. The six observed attributes link to $\theta$ through generalised-linear likelihoods matched to each attribute's support:

$$
a_i(u) \mid \theta(u) \;\sim\; \begin{cases}
\text{Beta}(\sigma(\alpha_i + \beta_i\theta)\phi_i,\; (1-\sigma(\alpha_i+\beta_i\theta))\phi_i) & i \in \{a_1, a_3, a_5, a_6\} \\
\text{LogNormal}(\alpha_2 + \beta_2\theta,\; \sigma_2^2) & i = a_2 \\
\text{OrderedLogistic}(\alpha_4 + \beta_4\theta,\; \kappa) & i = a_4
\end{cases}
$$

Slopes $\beta_i$ are sign-constrained positive ($\beta_i > 0$) so that the latent has the natural interpretation: higher $\theta$ = more overlooked. A population-level prior on $\theta(u)$ pools partial-data countries toward the global mean. Inference is variational (NumPyro `AutoNormal` + `init_to_median`), validated against NUTS on a sub-sample.

**Outputs per country.** Posterior median, 5th and 95th percentiles (90 % CI), CI width.

**Candidate pool.** HRP-eligible countries only — those with an observed `per_pin_gap` (i.e., an active humanitarian response plan). Other countries return NaN, since the construct "overlooked humanitarian crisis" is not well-defined for them and CERF UFE allocations themselves draw from this pool.

**Validation.** External, against CERF UFE and CARE BTS. Precision @ 10: 5/10 on CERF UFE 2025 w1 (tied with additive baseline); 3/10 on CERF UFE 2024 w2 (tied); 2/7 on CERF UFE 2025 w2 (vs 1/7); 3/10 on CARE BTS 2024 (vs 1/10). Posterior calibrated against NUTS: Spearman ρ = 0.89 on theta medians, CI widths within 2× of NUTS.

**Known discontinuities.**
- Changes to the prior, model structure, or candidate pool shift the posterior.
- The model is fit on the current snapshot; adding new countries refits the population-level prior, which can move other countries' posteriors.
- Variational inference may underestimate posterior variance; NUTS validation is the calibration reference.

**Failure modes.**
- Sign constraints encode the assumption that all six attributes co-vary positively with overlookedness. A new attribute that does not satisfy this would need an opposite-sign prior or a different observation likelihood.
- Countries with very few observations get wide CIs structurally — but a wide CI is not a flag of "noise"; it is an honest statement of uncertainty that downstream users should respect.
- The "overlookedness" label is itself a modelling commitment; the posterior is a posterior over what the chosen six observations *imply*, not a direct measurement of the political concept.

**Our handling.** Always report posterior median *with* 90 % CI; never present the median alone. Rankings derived from the posterior should carry the CI as a column. Where two countries' CIs overlap heavily, "A is more overlooked than B" is not a defensible claim from the model.

---

## Discipline going forward

1. **Before adding any new metric to the pipeline, write its card here.** If the card can't be filled in cleanly, the metric doesn't go in.
2. **In proposal text and dashboard views, cite the card** (`see Metric Cards §X`) alongside any quantitative claim that depends on an aggregate.
3. **When an upstream source revises methodology** (like INFORM Feb-2026), update the card's "Known discontinuities" section before incorporating the revised data.
4. **Prefer the decomposed primitive over the composite** where the primitive answers the analytical question.
