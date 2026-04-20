# Geo-Insight — Methodology

*A latent-variable model for ranking humanitarian crises under incompleteness, temporal drift, and multi-stakeholder ambiguity.*

---

## 1 · What we are doing, and why

Humanitarian agencies must decide, each funding cycle, where a limited budget does the most good. In 2026 this question has become harder: every active response is now underfunded, which means a single coverage ratio no longer separates the worst-off from the rest. Nineteen country contexts sit below fifty per cent coverage. We have moved from a regime in which some crises were adequately funded and some were not, to a regime in which the only useful question is *how* a crisis is underfunded and *for whom*.

This document describes the mathematics of a system that addresses that question. The goal is not a scoring algorithm — it is a statistical inference problem. The quantity of interest is only ever observed indirectly, through six noisy signals, some of which are missing, all of which drift over time, and about which different humanitarian stakeholders hold different priors. We treat this as the inference problem it actually is.

Two commitments carry the entire design.

**Overlookedness is a latent scalar.** For each country $u$ and time $t$, there is a single unobserved quantity $\theta(u, t) \in \mathbb{R}$ describing how overlooked that crisis is. We never observe $\theta$ directly; we observe six attributes that are noisy, possibly missing, functions of $\theta$.

**Stakeholder disagreement is a prior difference.** Each humanitarian stakeholder — say CERF, ECHO, USAID, NGO consortia — has an a priori belief about how strongly each attribute reflects $\theta$. These priors are not weights in a weighted-sum formula; they are Bayesian priors over the parameters of an observation model. When two stakeholders' posteriors on $\theta(u, t)$ overlap, we have consensus on $u$; when they separate, $u$ is contested. This is not a metaphor — it is what the mathematics says, and every operational output we produce inherits this semantics.

The artefact the system delivers is, per country, per month, per stakeholder: a posterior distribution over $\theta$, from which we read ranks with honest uncertainty.

---

## 2 · What we observe

The upstream data is a panel of six attributes collected from five OCHA-registered sources and one EU JRC source. Each attribute answers a different question about the state of a crisis.

| Attribute | Symbol | Support | What it measures |
|---|---|---|---|
| Coverage shortfall | $a_1$ | $[0, 1]$ | Fraction of the appeal still unfunded |
| Gap per person in need | $a_2$ | $[0, \infty)$ | Unmet requirement in US dollars per person in need |
| Need intensity | $a_3$ | $[0, 1]$ | Fraction of the country's population in humanitarian need |
| Severity category | $a_4$ | $\{1,\ldots,5\}$ | INFORM severity ordinal (Very Low → Very High) |
| Donor concentration | $a_5$ | $[0, 1]$ | Herfindahl–Hirschman index of donor-share distribution |
| Intra-crisis equity | $a_6$ | $[0, 1]$ | PIN-weighted Gini over sector coverage ratios |

These six were not chosen to cover every facet of overlookedness — they were chosen because they satisfy three conditions simultaneously: (i) each is reproducible from a declared, publicly-available source, (ii) each maps cleanly onto a distinct facet of how a crisis can be overlooked (financial, temporal, structural, intra-crisis), and (iii) together they span the two new dimensions our methodology adds — sector-equity via $a_6$, stakeholder disagreement implicitly via the four profile posteriors we will construct in §6.

**What is missing, and why it matters.** Not every country-month has all six attributes. HNO publishes annually, FTS nearly continuously, CBPF quarterly, INFORM monthly. Even within a single cycle, some sectors lack cluster-level reporting; some crises have no distinct pooled-fund presence. A crisis with fewer observed attributes should produce a *wider* posterior, not a shifted one. The machinery below delivers this property structurally, not procedurally.

**Normalisation note.** Each bounded attribute is left on its natural support; no min–max rescaling is used. The observation model links $\theta$ to each attribute through a generalised-linear-model parameterisation with an appropriate link function, so the raw scales speak directly. The unbounded $a_2$ is handled on a logarithmic scale.

---

## 3 · The latent: what $\theta$ means, and what it does not

$\theta(u, t) \in \mathbb{R}$ is a real-valued quantity. Larger $\theta$ means "more overlooked relative to documented need." We deliberately make four commitments about its nature.

**$\theta$ is scalar, not vector.** A country has exactly one overlookedness value at a given time. A vector-valued latent (say, separate axes for "neglected financially" and "neglected in attention") would be more expressive, but at the cost of requiring a stakeholder-specific combination rule to obtain a scalar ranking, which re-introduces the multi-attribute-utility problem we are trying to dissolve. Scalar $\theta$ commits to *one* notion of overlookedness; stakeholders differ on *how attributes project onto that notion*, not on the dimensionality of the notion itself.

**$\theta$ is unbounded.** There is no finite maximum overlookedness, so we do not impose one. $\theta$ is only ever consumed through ranks and through posterior credible intervals; its numerical scale is book-keeping and has no operational interpretation.

**$\theta$ is not identified in absolute terms.** The model is invariant under the transformation
$$
\theta(u, t) \;\to\; \theta(u, t) + c, \qquad \alpha_i \;\to\; \alpha_i - \beta_i \, c
$$
for any constant $c$, where $(\alpha_i, \beta_i)$ are the intercept and slope in the observation model for attribute $i$. It is also invariant under the sign flip $(\theta, \beta) \to (-\theta, -\beta)$. We fix this ambiguity by two constraints: location, $\mathbb{E}_u\bigl[\theta(u, t_{\text{latest}})\bigr] = 0$; sign, $\beta_{\text{coverage}} > 0$ (an a priori natural direction — higher coverage shortfall should reflect greater overlookedness).

**$\theta$ describes the present and the past, not a prediction.** We fit $\theta(u, t)$ on observed data; we do not extrapolate forward. A separate learning layer, described in §11, will later map raw monthly panels forward in time, but nothing in the present methodology claims to forecast.

---

## 4 · The observation model

Here the meaning of each attribute becomes concrete. For each attribute $i$, we specify a likelihood: given the latent $\theta(u, t)$, what is the probability distribution of the observed $a_i(u, t)$? Each likelihood is chosen to match the attribute's support. Missing observations contribute no likelihood term at all — not a zero, not an imputed value, simply nothing.

### 4.1 · Beta regression for bounded fractions: $a_1, a_3, a_5, a_6$

Four of the six attributes live on $[0, 1]$. For these we use **beta regression**. The beta distribution's two parameters reparameterise naturally as (mean, precision):
$$
a_i(u, t) \mid \theta(u, t) \;\sim\; \text{Beta}\Bigl(\mu_i(u,t)\,\phi_i,\; \bigl(1 - \mu_i(u,t)\bigr)\phi_i\Bigr),
$$
$$
\mu_i(u, t) \;=\; \sigma\bigl(\alpha_i + \beta_i\,\theta(u, t)\bigr),
$$
where $\sigma$ is the logistic function and $\phi_i > 0$ is a precision parameter common across countries but specific to the attribute. This gives exactly the right behaviour: as $\theta$ increases, the expected observation shifts toward one (or toward zero, depending on the sign of $\beta_i$), while remaining bounded in $[0, 1]$.

**Why not a Gaussian on a transformed scale?** Beta regression respects the floor and ceiling of the attribute's support — there is no probability mass outside $[0, 1]$. A logit-transformed Gaussian would be similar in spirit but would complicate likelihood inference with the change-of-variables Jacobian for no gain.

**What the parameters mean.** $\beta_i$ is the sensitivity of attribute $i$ to overlookedness on the logit scale; a larger $|\beta_i|$ means the attribute moves more sharply with $\theta$. $\phi_i$ controls dispersion; a larger $\phi_i$ means observations are tightly clustered around $\mu_i(u, t)$.

### 4.2 · Log-normal for the per-PIN gap: $a_2$

The gap per person in need is a positive real number spanning many orders of magnitude across crises. We model it on the logarithmic scale:
$$
\log a_2(u, t) \;\sim\; \mathcal{N}\bigl(\alpha_2 + \beta_2\,\theta(u, t),\; \sigma_2^2\bigr).
$$
This is a standard log-linear model. Multiplicative rather than additive shifts in $\theta$ produce additive shifts in the observed $\log a_2$, which matches the empirical fact that financial gaps scale roughly multiplicatively across crises (a gap of 50 \$/person in need is very different from a gap of 500 \$/person in need, not because of a factor of ten in $\theta$ but because of a factor of ten in the underlying financial pressure).

### 4.3 · Ordered probit for severity: $a_4$

INFORM's severity category is an ordinal on $\{1, 2, 3, 4, 5\}$. The natural likelihood is an **ordered probit**. Introduce a latent continuous variable
$$
y(u, t) \;=\; \alpha_4 + \beta_4\,\theta(u, t) + \varepsilon,\qquad \varepsilon \sim \mathcal{N}(0, 1),
$$
and four cut-points $\kappa_1 < \kappa_2 < \kappa_3 < \kappa_4$. The observed severity category is
$$
a_4(u, t) \;=\; k \;\iff\; \kappa_{k-1} < y(u, t) \le \kappa_k,
$$
with $\kappa_0 = -\infty$ and $\kappa_5 = +\infty$. The cut-points are parameters we infer jointly with everything else; they encode the empirical spacing between INFORM's category boundaries, which is not guaranteed to be uniform.

**Why not treat $a_4$ as continuous 1–5?** Doing so implies the steps 1→2 and 4→5 are equally large, which is exactly what INFORM's qualitative descriptions (Very Low → Low → Moderate → High → Very High) suggest is *false*. Ordered probit lets the data tell us how the steps are actually spaced.

### 4.4 · Missing observations

For each (country, month) and each attribute, we record whether the attribute is observed. If not, there is no likelihood contribution for that (country, month, attribute) triple. The posterior on $\theta(u, t)$ for a country-month with only two attributes observed will be wider than the posterior for a country-month with six. No imputation is performed; the conditional distribution of $\theta$ given the data simply reflects less evidence.

Sparse-data countries get wider posteriors structurally — not because we down-weight them by hand, but because the likelihood has fewer terms to sharpen the posterior on $\theta$.

---

## 5 · Temporal dynamics

A crisis does not reshuffle itself every month. If Sudan was ranked second in March 2025, it is likely still highly ranked in April 2025. We encode this expectation through a temporal prior on $\theta(u, \cdot)$.

For each country $u$, the monthly trajectory of overlookedness is modelled as an AR(1) process with country-specific mean, persistence, and innovation scale:
$$
\theta(u, t) \;=\; \mu_u \;+\; \rho_u\bigl(\theta(u, t-1) - \mu_u\bigr) \;+\; \eta_u(t),\qquad \eta_u(t) \sim \mathcal{N}(0,\tau_u^2).
$$

$\mu_u$ is the country's long-run baseline; $\rho_u \in (0, 1)$ controls how quickly shocks decay; $\tau_u$ controls the size of month-to-month innovations.

**Chronic and acute, for free.** The quantity $\mu_u$ is precisely the "chronic" overlookedness of country $u$ — its long-run baseline. The quantity $\theta(u, t) - \mu_u$ is precisely the "acute" component — the deviation of the current state from that baseline. No separate chronic/acute computation is needed; both fall out of the same posterior.

**Why AR(1) and not a Gaussian process.** A GP would give more flexible smoothing, at $O(T^3)$ inference cost per country and more subtle identifiability. AR(1) fits within an HMM-style forward–backward pass in linear time and is the right parsimony for a monthly panel. If residuals suggest structured departures (trends, regime shifts), we escalate to a GP with an RBF+Matérn kernel.

**A known weakness.** AR(1) with constant $\mu_u$ assumes the baseline is stable over time. Countries in sustained decline (Venezuela, Sudan post-2023, Afghanistan post-2021) have *rising* baselines, and the AR(1) residual absorbs this drift rather than representing it faithfully. If this becomes empirically visible in residual diagnostics, the fix is to let $\mu_u$ itself carry a linear trend or to replace AR(1) with a local-level state-space model.

---

## 6 · Cross-country hierarchy and the stakeholders

Two distinct hierarchical structures live above the observation model. The first pools countries; the second pools stakeholders.

### 6.1 · Partial pooling across countries

The AR(1) parameters $(\mu_u, \rho_u, \tau_u)$ are drawn from population-level distributions:
$$
\mu_u \sim \mathcal{T}_\nu(\mu_0,\sigma_\mu^2), \qquad
\rho_u \sim \text{Beta}(a_\rho, b_\rho), \qquad
\tau_u \sim \text{HalfCauchy}(\tau_0).
$$

We use a Student-$t$ with $\nu = 4$ degrees of freedom on $\mu_u$, not a Gaussian. This is deliberate — countries like Sudan or Yemen have genuinely exceptional baselines, and a Gaussian hyper-prior would shrink them toward the global mean in a way that a researcher would find unconvincing. The heavier tails of the $t$-distribution allow these countries to stand out.

Partial pooling has a simple operational effect: countries with sparse data borrow strength from the population hyperparameters; countries with dense data are barely affected by them. This is exactly the behaviour we want.

### 6.2 · Stakeholders as priors over attribute slopes

Here the model acquires its novelty. A stakeholder $s$ is a prior on the six slopes $\beta = (\beta_1, \ldots, \beta_6)$:
$$
\beta_i \;\sim\; \mathcal{N}\bigl(m_i^{(s)},\; v_i^{(s)}\bigr), \qquad i = 1, \ldots, 6.
$$

Four stakeholder priors are pre-registered. Each is a pair of (mean, variance) vectors encoding "what this stakeholder believes about how each attribute reflects overlookedness". Plausible table:

| Stakeholder | $m_1^{(s)}$ | $m_2^{(s)}$ | $m_3^{(s)}$ | $m_4^{(s)}$ | $m_5^{(s)}$ | $m_6^{(s)}$ |
|---|---|---|---|---|---|---|
| CERF  | 1.50 | 0.75 | 1.00 | 1.25 | 0.25 | 0.25 |
| ECHO  | 1.25 | 1.00 | 0.75 | 0.75 | 0.25 | 1.00 |
| USAID | 1.00 | 0.75 | 0.75 | 0.75 | 1.25 | 0.50 |
| NGO   | 1.00 | 0.75 | 0.75 | 0.50 | 0.50 | 1.50 |

Prior variances $v_i^{(s)}$ are of order $0.3$–$0.5$ — tight enough that stakeholder priors differ meaningfully, diffuse enough that data can move the posterior away from the prior if it strongly disagrees.

**Three properties of stakeholders-as-priors.**

1. *Falsifiability.* Each stakeholder prior is a pre-registered statistical claim. The posterior either corroborates or refutes it. If CERF's prior says $\beta_{\text{coverage}} \approx 1.5$ but the data (pooled across all crises CERF actually selects) says $\approx 0.8$, we have learned something concrete about CERF's stated-vs-effective preference.

2. *Uncertainty quantification.* We have a posterior distribution on $\beta$, not a point estimate. Stakeholder disagreement is not a four-point spread; it is four overlapping (or non-overlapping) posterior clouds.

3. *Consistency with data density.* A stakeholder prior is a soft statement. When a country has ample data, the posterior on $\theta$ is driven by the data; when a country has sparse data, the posterior on $\theta$ is shaped more strongly by the prior — and thus by the stakeholder. Consensus vs contested thus becomes context-sensitive in a principled way.

Each stakeholder's posterior is a separate inference: we fit four posteriors $p(\theta \mid D, s)$, not one.

---

## 7 · Inference

At the declared scale — roughly 100 active countries over 84 months with 6 attributes of heterogeneous availability, per stakeholder — we have on the order of $5 \times 10^4$ likelihood evaluations. The joint posterior is not analytically tractable.

We use **variational inference** in two stages.

### 7.1 · Stage 1: mean-field SVI

We approximate the posterior with a factorised Gaussian guide, one per latent variable, fitted by stochastic variational inference (ADVI) with Adam. This is fast — minutes per stakeholder on a CPU — and gives calibrated posterior medians for ranking purposes. Implementation: NumPyro with a `numpyro.infer.SVI` loop.

The mean-field Gaussian guide is known to underestimate posterior variance when variables are strongly correlated. For our ranking outputs this is acceptable — the point estimates (medians) are the load-bearing numbers; variance underestimation makes our uncertainty statements mildly optimistic, which we will correct in stage 2.

### 7.2 · Stage 2: NUTS validation on a sub-sample

For a representative sub-sample of ~20 countries spanning the posterior-rank range, we refit with the No-U-Turn Sampler. NUTS is slow (an hour per stakeholder) but gives a near-exact posterior. We compare SVI medians and 90% credible intervals against NUTS on this sub-sample. If they diverge by more than a small tolerance, we escalate the guide family (to a low-rank multivariate normal, or a normalising flow) rather than continuing with mean-field.

### 7.3 · Caching

Posterior samples — $S \approx 5000$ per (stakeholder, month, country) — are stored as parquet in `analysis/posterior/`. The dashboard reads the cache; re-fitting is an offline operation triggered by `scripts/refit_posterior.py`.

---

## 8 · Outputs

Per country-month, per stakeholder, from $S$ posterior samples of $\theta(u, t)$:

- **Posterior median** $\widehat\theta$ — the point estimate.
- **90% credible interval** — the honest statement of uncertainty.
- **Posterior rank distribution** — in each sample, compute the rank of $u$ across the pool; report the median rank and the 90% CI on the rank.
- **Attribute contribution** — for interpretability, project the posterior on $\theta$ back through the observation model to show, per attribute, how much $a_i(u, t)$ moved the posterior. This is the direct analogue of SHAP attributions for linear models and is cheap to compute from the posterior.

Aggregated across stakeholders:

- **Consensus score** — the fraction of posterior samples in which country $u$ is in the top-$K$ under *all four* stakeholder posteriors. Very high consensus scores correspond to "nobody disagrees this country is overlooked."
- **Contested score** — the maximum of $\widehat\theta_s$ across $s$ minus the minimum, or equivalently the Bhattacharyya distance between the most and least concordant stakeholder posteriors. Very high contested scores correspond to "this country's rank depends heavily on whose preferences you use."

---

## 9 · Validation

Three tests against independent external ground truth.

**External benchmarks (cross-sectional).** The model's posterior median is scored against two human-curated lists of overlooked crises. CERF Underfunded Emergencies (UN's twice-yearly allocations to underfunded crises) and CARE *Breaking the Silence* (annual top-ten under-reported crises). Both are produced by methodology independent of anything in this repo.

| benchmark | k | precision @ k |
|---|---|---|
| CERF UFE 2024 w2 | 10 | 3/10 |
| CERF UFE 2025 w1 | 10 | 5/10 |
| CERF UFE 2025 w2 | 7 | 2/7 |
| CARE BTS 2024 | 10 | 3/10 |

The variational posterior is calibrated against NUTS on every fit. The production guide is `AutoMultivariateNormal` (Spearman ρ = 0.89 on theta medians vs NUTS; CI widths within 2× of NUTS); the simpler `AutoNormal` mean-field guide gives ρ = 0.68 and underestimates posterior variance by ~3×, so its sharper rankings are not trustworthy.

The candidate pool is restricted to HRP-eligible countries (those with an observed `per_pin_gap`) — the population CERF UFE picks from.

**Temporal held-out.** Fit on the 2024 cycle (HRP 2024, FTS through end-2024, INFORM through end-2024); compare the 2024-fit ranking against CERF UFE selections made in 2025. Result (run via `analysis/bayesian/temporal_holdout.py`):

| benchmark (selection date) | k | 2024-fit (held-out) | 2025-fit (concurrent) |
|---|---|---|---|
| CERF UFE 2024 w2 (Dec 2024) | 10 | 4/10 | 3/10 |
| CERF UFE 2025 w1 (Mar 2025) | 10 | 4/10 | 5/10 |
| CERF UFE 2025 w2 (Dec 2025) | 7 | 2/7 | 2/7 |
| CARE BTS 2024 | 10 | 1/10 | 3/10 |

The 2024 fit predicts CERF's March-2025 selection at 4/10 precision @ 10 without seeing any 2025 data. Year-over-year top-10 overlap is 7/10 — HND, HTI, MOZ, SLV, SOM, TCD, VEN persist; ETH, MLI, SYR drop out; CMR, GTM, NER enter. The model identifies the same crisis cluster across cycles.

The full AR(1) trajectory model in §5 is not yet identified by our data: the six attributes are annual (HRP, FTS), so we have at most two time points per country across 2024–2025, well below what AR(1) inference needs to discriminate persistence from noise. INFORM severity is monthly and could anchor a partial-temporal extension; that is on the explicit roadmap.

**Attribute masking.** For a country with full six-attribute observation (say Sudan or Yemen), mask one attribute at a time and refit. The posterior median should change only modestly; the 90 % CI should *widen* — by an amount predictable from the information content of the masked attribute. If masking narrows the CI, the model is miscalibrated and we investigate.

**Posterior predictive checks.** For each of the six attributes, draw 1{,}000 replicates from the fitted posterior and compare the marginal distribution of the simulated values against the observed values. Result (run via `analysis/bayesian/ppc.py`): coverage of the 90 % predictive interval is ≥ 0.91 for every attribute (target 0.90), so the model's marginals are well-calibrated. Per-country Pearson correlation between predicted mean and observed value is high for `coverage_shortfall` (0.76) and `donor_hhi` (0.82) — these are the attributes the latent θ actively explains — and weak for `need_intensity`, `cluster_gini`, and `severity_category`, which sit near their attribute ceilings (most HRP countries have PIN ≈ population and severity ∈ {4, 5}) and carry little cross-country variation for the latent to recover. The latent is best read as a *funding-overlookedness* axis; the saturating attributes constrain the posterior but do not drive its discrimination.

---

## 10 · What we are not claiming

Several things, plainly.

**We are not forecasting.** The model describes $\theta(u, t)$ for observed $t$ only. Forward prediction is the subject of the separate learning phase described in §11.

**We are not inferring causality.** The observation model is a statistical association — it says nothing about whether a change in $a_i$ would cause a change in $\theta$, or vice versa. No counterfactual claims of the form "if this country had more donors, it would be less overlooked" are supported by this machinery.

**We are not replacing judgement.** The system ranks; humans decide. Every displayed posterior carries the attribute contributions that produced it, so a coordinator can see exactly what drove a given ranking.

**We have not observed overlookedness directly, ever.** $\theta$ is latent by definition. All our claims about the latent are contingent on the observation model being approximately correct. §9's validation tests are the main evidence that it is; they are not proof.

**The stakeholder priors are judgement, not calibration.** The table in §6.2 represents what we think CERF, ECHO, USAID, and NGO consortia broadly care about. It is a starting point. A serious subsequent step would be to calibrate each stakeholder's prior by regressing the selections they have actually made against our attributes. That calibration is not part of the present model; it is a natural extension.

**Benchmarks are imperfect proxies.** CERF UFE selections reflect both genuine overlookedness and a human political process. CARE's *Breaking the Silence* reflects media attention, not funding. Agreement with either list is informative but not conclusive.

---

## 11 · Where this goes next

The methodology above produces a principled, uncertainty-quantified ranking over six hand-chosen attributes computed from structured upstream data. The next step is to ask whether the attributes themselves are what we should be conditioning on.

A richer view of a crisis is embedded in the raw monthly time-series: INFORM's eighteen sub-indicators on displacement, casualties, access, and phase-level need, rather than the six aggregates we currently derive. A learned representation of these raw signals — by a sequence model such as Mamba, or a masked-attention encoder trained by self-supervision on the INFORM panel — would give us an embedding $z(u, t) \in \mathbb{R}^d$ that compresses the raw history into a form the latent-variable model can consume.

The extension is straightforward, architecturally: replace the observation model $a_i \mid \theta$ by a learned map $(a, z) \mid \theta$, where $z$ is produced by the encoder and carries information the hand-designed attributes do not. The learned encoder trains on the full monthly panel; the latent-variable model, unchanged above the encoder, provides the interpretable posterior. The typed semantic layer, unchanged below, provides the provenance.

Three properties of this staged architecture matter.

*The latent $\theta$ stays interpretable.* It is the same scalar it has always been, with the same ranking semantics. The only thing that changes is what we condition on.

*The learned encoder has a target.* Self-supervision on raw signals is often under-specified; here the encoder is trained to produce features that help reconstruct $a_i$ from $z$, or equivalently to produce features that reduce posterior variance on $\theta$. That is a crisp, measurable objective.

*The provenance discipline carries through.* Each learned dimension of $z$ is registered as a new Level-6 property in the ontology, with its training data declared, its checkpoint hash declared, its known failure modes declared, and its attribution decomposition available on request. This is the discipline the rest of the system already operates under; we do not relax it for a learned component.

When that layer is in place, the system will produce posterior ranks conditioned on both structured attributes and learned representations, both provenance-traceable, both uncertainty-quantified, and both honest about what they do not know.
