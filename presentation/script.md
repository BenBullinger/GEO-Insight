# Geo-Insight · presenter script

Ten slides, roughly six to seven minutes. Names the latent, the six attributes,
the four stakeholders, and the forecast numbers without hand-waving.

---

## Slide 1 · Title  (~10 s)

Geo-Insight. Which humanitarian crises are most overlooked? In 2026 nearly
every humanitarian response is underfunded — and when everything is
underfunded, the funded-over-requested ratio stops telling us anything.
This is our answer.

---

## Slide 2 · The problem  (~40 s)

Three numbers frame it. Agencies asked for **$33 billion** to reach
**135 million people** in 2026. In 2025 they raised **$12 billion** —
the lowest in a decade. **Nineteen** country responses are now below
fifty percent covered, up from eight in 2021.

The useful variation is no longer *whether* a crisis is underfunded but
*how*. Three failure modes that a single ratio cannot separate:

- **Forgotten** — low funding, low attention. Angola 2023: 7 M affected,
  ~1,000 news articles.
- **Contested** — high attention, persistent decline. Sudan 2025: world's
  largest displacement, still fell to 35 %.
- **Sector-starved** — adequate aggregate, one cluster at 20 %.

---

## Slide 3 · The model  (~75 s)

We stop scoring and start inferring. Overlookedness is a latent scalar
**θ** — one real number per country, never observed directly. Higher θ
means more overlooked. We infer θ's posterior from **six attributes**,
each with a likelihood matched to its natural support:

1. **a₁ coverage shortfall** — one minus the fraction funded, bounded [0, 1].
   Beta regression.
2. **a₂ per-PIN gap** — dollars of unmet need per person in need, positive
   real. Log-normal.
3. **a₃ need intensity** — people in need divided by population, bounded [0, 1].
   Beta.
4. **a₄ severity** — INFORM's 1-to-5 ordinal. Ordered-logistic link, so the
   step from 3 to 4 is not forced to equal the step from 4 to 5.
5. **a₅ donor concentration** — Herfindahl over single-destination FTS
   incoming flows. Bounded [0, 1]. Beta.
6. **a₆ intra-crisis equity** — PIN-weighted Gini over sector coverage
   ratios. Catches a country with 80 % on one cluster and 20 % on another.
   Beta.

Every slope β is positivity-constrained by domain knowledge. A Gaussian
population-level prior pools partial-data countries toward the global
mean — so the model is honest about what it doesn't know.

---

## Slide 4 · Semantic layer  (~35 s)

All **71 properties** live in a typed ontology, five levels deep:
**L1** raw observations (HNO, HRP, FTS, INFORM), **L2** scale-normalised
ratios, **L3** inequality indices, **L4** windowed trends, **L5** the
posterior on θ itself. Every derived value declares its formula, source,
and failure modes *before* it's computed. Nothing renders on screen that
isn't registered.

---

## Slide 5 · The posterior  (~45 s)

The fit runs on **22 countries** — those with an active humanitarian
response plan and both HNO and FTS entries. That's the same population
CERF's Underfunded Emergencies list draws from, so validation is on the
same pool.

Inference is variational with a full-covariance Gaussian guide,
**~3 seconds on a CPU**, calibrated against NUTS on every run. Top ten
by posterior median: **Honduras, El Salvador, Mozambique, Somalia,
Guatemala, Niger, Haiti, Cameroon, Venezuela, Chad**. The 90 % credible
intervals overlap substantially — the defensible claim is *membership*
in this set, not ordering within it.

---

## Slide 6 · Stakeholders  (~60 s)

Stakeholders don't disagree on θ — they disagree on *what should move θ*.
So we encode each as a prior over the six slopes, not as weights in a sum.

Four are pre-registered:

- **CERF** — the UN Emergency Relief Coordinator's pooled fund, weights
  **severity** (a₄).
- **ECHO** — the European Commission's humanitarian office, weights
  **sectoral equity** (a₆).
- **USAID** — the US bilateral humanitarian agency, weights
  **donor concentration** (a₅).
- **NGO consortia** — weights **cluster-level equity** (a₆), but with a
  different tilt than ECHO.

Each produces its own posterior on θ. **Honduras** is the most contested:
three densities cluster close, USAID's sits visibly to the right — all
four still keep it in the top-10, so they disagree on *how* overlooked,
not *whether*. **Haiti** is the most consensus: all four stack almost
perfectly. Across the pool, **5 of 22** are full consensus, **11** are
contested under at least one lens.

---

## Slide 7 · Validation  (~70 s)

Two independent human-curated benchmarks, neither used in fitting.
**CERF UFE** — the UN's twice-yearly allocations to underfunded
crises. **CARE Breaking the Silence** — the annual top-10 most
under-reported.

| Benchmark              | Hits      |
| ---------------------- | --------- |
| CERF UFE Mar 2025      | **5 / 10**|
| CERF UFE Dec 2025      | **2 / 7** |
| CERF UFE Dec 2024      | 3 / 10    |
| CARE BTS 2024          | **3 / 10**|

On CERF's March 2025 list: half their picks appear in our top-10,
without a single CERF label in training. SVI-versus-NUTS agreement:
Spearman **ρ = 0.89**. Posterior-predictive coverage **≥ 0.91** on
every attribute. And the genuine out-of-sample check: a 2024-only
fit predicts March-2025 CERF picks at **4 / 10**, without ever seeing
2025 data.

---

## Slide 8 · Forecasting the monthly panel  (~60 s)

We also forecast. A small Mamba — two selective-SSM blocks, **19 k
parameters** — reads 24 months of the INFORM panel and predicts mean
severity *H* months ahead. No θ, no funding, no stakeholder priors.

At short horizons, **persistence wins**: severity is a slow 1–5 ordinal
and "tomorrow looks like today" is unbeatable on average. At **H = 9**
Mamba crosses below persistence; at **H = 12** it leads by **1.38×**
— 0.294 MAE vs 0.407 — on 78 held-out windows. We do not claim a
universal win; the horizon curve is the result.

The forecast feeds the Bayesian LVM as a seventh sign-constrained
attribute. That ablation shifts the top-10 by three countries without
a net precision gain — honest extension, not uplift.

---

## Slide 9 · Scope and what's next  (~45 s)

**Twenty-two countries, not more.** We tested expanding the gate to
countries with positive FTS requirements but no HRP; precision against
CERF dropped. That tells us the HRP gate is the right scientific scope.

Three things next:

1. **Calibrated stakeholder priors** — regress actual allocation
   histories on the six attributes to recover effective priors
   empirically.
2. **Temporal AR(1) on θ** — chronic and acute decomposition from the
   same posterior, anchored by monthly INFORM.
3. **Longer monthly panel** — the Mamba closes to persistence only at
   12 months; more history per country is the unlock, not a bigger
   model.

---

## Slide 10 · Close  (~15 s)

θ, six observations, one Bayesian scaffold, four stakeholders as priors
not weights, external validation on CERF and CARE, a learned forecaster
that knows when to hand the microphone back to persistence. Methodology
is online at `localhost:7777/methodology/`. Questions?
