# Parameter sourcing memo (WO-000)

**Date of writing: 2026-09-07.** Item 8 is date-stamped because its facts decay.

## What this memo is, and is not

PLAN section 3 tabulates every environment parameter with an arm, a Phase-1 value and a sweep
range. Six rows are marked provisional: their Phase-1 values are placeholders that gate G1 replaces
with values a human picks from the interior of the DP regime map (PLAN sections 5 and 13), recorded
in `runs/G1_decision.md`. This memo is the evidence base for that pick, and the source of the
`source` column of `gosplan/params.py` (WO-001).

It is **not** a decision document. It does not choose Phase-1 values and it does not rank arms.

Two rules govern every line below, from PLAN section 15 and the WO-000 card:

- A range with a citation is evidence. A range without one is a **prior**, and is labelled a prior
  in those words. No item ends with a point value presented as historical fact.
- A citation is only written where the document was actually retrieved and the quoted sentence read.
  Where retrieval failed, the memo says **not obtained** and falls back to the unsourced sentence.
  The Phase-1 defaults already sitting in `gosplan/params.py` and `spec/spec.py` are placeholders
  and are **not** evidence; using them here would close the loop on itself.

**Outcome of this pass: one item of eight (item 8) is sourced. Seven are unsourced priors.** That
is a low yield, and it is the honest one — see "What was tempted and not written" at the end, which
is the card's stated primary quality signal.

---

### Item 1 - Historical bonus schedules

**(a) The question.** The size of the manager's bonus at the plan target relative to base pay (the
notch), the per-percentage-point increment for overfulfilment above target, and any cap on accrual.
Feeds `notch_height` (beta), `overfulfilment_slope` (s), `overfulfilment_cap` (rho_cap). What
matters for the design is s and rho_cap *relative to* beta, since beta is the normalising unit of
the reward scale (PLAN section 2.9.1) — not any of the three in absolute roubles.

**(b) What was found.** Not obtained. Berliner's *Factory and Manager in the USSR* (1957) is a
monograph and no retrievable full text was located. A search surfaced a Berkeley *California
Management Review* item (1962) whose snippet carried percentage figures for maximum and average
bonus as a share of salary; the page redirects to `haas.berkeley.edu` and returns **HTTP 404**, so
the figures could not be read in context and are not cited here. See the tempted list.

**(c) What it implies.** Nothing sourced. The design question — s and rho_cap relative to beta —
is in any case a *shape* question that the DP regime map of PLAN section 5 sweeps directly, and
beta is fixed at 1 by the reward-scale normalisation rather than by history.

This item also carries `effort_cost` (kappa), which is covered by **no** research item in PLAN
section 15. That is deliberate and worth stating: kappa is a modelling primitive of the agent's cost
function with no historical counterpart to source. It is recorded here so that criterion A2 is met
for all six provisional parameters rather than silently omitting one.

**(d)** unsourced; prior = [beta = 1 by normalisation; s in 0 to 2; rho_cap in {1.1, 1.2, infinity}; kappa in 0.05 to 0.5]; sensitivity handled by regime map / Sobol

---

### Item 2 - Growth directives

**(a) The question.** Annual plan growth targets by period of Soviet history, as a range **across
periods** rather than one national figure. Feeds `growth_directive` (g), registry range [0, 0.07]
per plan period. g is a treatment variable and the forcing term behind review finding F1: at g = 0
with reports at target the target rule has a fixed point (test T-B2), which is precisely why g
cannot be a constant.

**(b) What was found.** The *form* is attested, the *magnitude* is not. Harrison (2007) describes
the target rule as planning

> "from the achieved level," that is by planning in the next period to achieve the same results as
> in the period before, plus an increment to allow for growth.

— Mark Harrison, "Gosplan", in *Dizionario del comunismo nel XX secolo*, vol. 1, pp. 338-340,
Einaudi 2007; author's postprint retrieved from
<https://warwick.ac.uk/fac/soc/economics/staff/mharrison/public/gosplan2007.pdf>.

That sentence establishes that a positive growth increment was a structural feature of target
setting. It attaches **no number** to the increment, and no retrievable source giving directive
(as opposed to realised) growth rates by period was obtained. The distinction matters and is the
main hazard in this item: published Soviet growth series overwhelmingly report *realised* output
growth, while g is the *directive* — what enterprises were told to achieve.

**(c) What it implies.** The functional form of the target rule (PLAN section 2.7.1) is supported.
The magnitude is not.

**(d)** unsourced; prior = [g in 0 to 0.07 per plan period, the registry range, treated as a treatment variable and swept]; sensitivity handled by regime map / Sobol

---

### Item 3 - Ratchet

**(a) The question.** Two separate questions. First: how strongly did next period's target respond
to this period's fulfilment? Feeds `ratchet_lambda` (lambda), range [0, 1]. Second, kept separate:
did tolerance bands exist — a band of fulfilment around 1 within which the target did not move?
Feeds `ratchet_deadband` (delta), range [0, 0.03], Phase-1 value 0.0 and inert.

**(b) What was found.** The theory is sourced and is explicitly theory. Weitzman (1980), abstract
retrieved verbatim from RePEc/IDEAS
(<https://ideas.repec.org/a/rje/bellje/v11y1980ispringp302-308.html>):

> "The use of current performance as a partial basis for setting future targets is an almost
> universal feature of economic planning. This 'ratchet principle,' as it is sometimes called,
> creates a dynamic incentive problem for the enterprise. Higher rewards from better current
> performance must be weighed against the future assignment of more ambitious targets. In this
> paper I formulate the problem of the enterprise as a multiperiod stochastic optimization model
> incorporating an explicit feedback mechanism for target setting."

— Martin L. Weitzman, "The 'Ratchet Principle' and Performance Incentives", *Bell Journal of
Economics* 11(1), 1980, pp. 302-308.

The abstract describes a multiperiod stochastic optimisation model and offers **no empirical
estimate** of the feedback coefficient. The full text was not obtained: the author's copy at
`scholar.harvard.edu` returns HTTP 403 and a SciSpace mirror returned an empty body.

Harrison (2007), cited in full under item 2, independently attests the same mechanism in practice
and its consequence:

> "They also learned to manipulate planners' expectations; as a result plans became less ambitious
> and were increasingly likely to be fulfilled by lowering plans to match performance rather than
> by improving performance."

The canonical empirical reference on the mechanism, identified from Harrison's reference list but
**not obtained**, is Igor Birman, "From the Achieved Level", *Soviet Studies* 30(2), 1978,
pp. 153-172. It is recorded here as the first thing a later pass should retrieve.

On the **second** question — tolerance bands — nothing was found either way. The record retrieved
supports neither the presence nor the absence of a deadband.

**(c) What it implies.** The *form* of the ratchet in PLAN section 2.7.1 is well supported by both a
theoretical treatment and an archival historian's description. The *coefficient* lambda has no
sourced value, which is exactly the case PLAN section 15 anticipates ("If the record supports the
FORM but no value, say so and use the unsourced sentence"). delta likewise has neither form nor
value attested.

**(d)** unsourced; prior = [lambda in 0 to 1, form attested by Weitzman 1980 and Harrison 2007 but no coefficient estimate retrieved; delta in 0 to 0.03, form not attested either way]; sensitivity handled by regime map / Sobol

---

### Item 4 - Inspection frequency and sanctions for pripiski

**(a) The question.** How often enterprises were inspected, what the sanction was, and — the part
that matters most — whether the sanction **scaled with the size of the discrepancy** or was a fixed
consequence of being caught at all. Feeds `audit_rate` (a, range [0.01, 0.30]), `penalty_scale`
(pen, range [5, 200]) and the two shape parameters `penalty_form` (proportional vs fixed) and
`penalty_arg` (positive_part vs absolute). A separate evidential question: did inspection compare
the claim to **stock on hand** or to **production**?

**(b) What was found.** Not obtained. No retrievable source was located giving an inspection rate,
a sanction schedule, or the scaling of sanction with discrepancy size. The existence of pripiski as
a recognised offence is not in doubt — it is named in the material retrieved under item 7 — but
naming an offence is not sourcing an audit rate or a penalty function.

**(c) What it implies.** Nothing sourced, on either the level or the shape.

The stock-versus-production question is therefore **an unsourced modelling assumption, and is
flagged as one**. PLAN section 2.8 has the audit compare the claim to stock on hand, which is what
makes hidden reserves protective against audit. No evidence was retrieved for or against that being
how Soviet inspection actually worked. This note exists so that the assumption is visible at gate G1
rather than inherited silently.

Note for the regime map: `a` and `pen` enter the G2 padding-elasticity criterion (PLAN section 4.5)
only through the compound `a * pen`, so the sweep is over the product and the two need not be
separately identified.

**(d)** unsourced; prior = [a in 0.01 to 0.30; pen in 5 to 200, swept as the compound a*pen; penalty_form and penalty_arg treated as shape toggles, not sourced]; sensitivity handled by regime map / Sobol

---

### Item 5 - Managerial tenure and rotation

**(a) The question.** How long an enterprise director typically held the post, and how rotation was
decided. Feeds `tenure` (psi), the per-plan-period continuation probability of the geometric horizon
(PLAN section 2.12), registry range [0.7, 0.98].

**(b) What was found.** Not obtained. No retrievable source giving a distribution or central
tendency of Soviet enterprise director tenure was located in this pass.

**(c) What it implies.** Nothing sourced. Two things are nonetheless worth fixing in writing,
because both are places a later pass could go wrong:

Unit conversion assumption, stated on its own line as the card requires:
> a per-year survival hazard is **not** a per-plan-period hazard unless the plan period is one year;
> any figure retrieved later must be converted with the plan period stated explicitly.

Second: psi is an **economic** parameter (the INC arm) and is deliberately distinct from the
technical PPO discount gamma = 0.99. They are not to be conflated, identified, or swept together.

**(d)** unsourced; prior = [psi in 0.7 to 0.98 per plan period, the registry range]; sensitivity handled by regime map / Sobol

---

### Item 6 - Enterprise-level fulfilment distributions, and the level of the archival series

**(a) The question.** Two outputs, both load-bearing. (i) The **target shape** for phenomenon 1 of
PLAN section 4.1: what a histogram of plan fulfilment actually looked like, so the bunching
estimator has something to be compared against. (ii) The **level** at which archival Soviet
fulfilment series are observed — enterprise, ministry or republic — because a series aggregated
above the enterprise cannot speak to an enterprise-level notch. Criterion A5 requires the level to
be stated explicitly.

**(b) What was found.** On (i), **not obtained**: no retrievable source reporting a distribution or
histogram of enterprise plan fulfilment percentages was located. Harrison (2007) speaks to the
*trend* but not the *distribution*:

> "Plan fulfillment appears to have improved through time, it seems unlikely that the reason was
> that Soviet producers became more obedient. Rather, they learned to manipulate plan indicators to
> show fulfillment."

On (ii), the same source speaks directly to the level, and this is the one substantive finding of
the item:

> "In total there were twelve five year plans... quarterly plans that were binding on the economy
> as a whole and its ministerial sub-branches. But these plans were too aggregated and preliminary
> to have much influence on what happened in particular factories and offices."

— Harrison 2007, retrieved as cited under item 2.

**(c) What it implies.** **The level of the Gosplan-held series is the ministry / ministerial
sub-branch, not the enterprise.** Enterprise-level plans existed but sat below the level Gosplan
itself held and aggregated.

The consequence for PLAN section 7.3 is direct and restrictive: a ministry-level series **cannot**
speak to an enterprise-level notch, because aggregation destroys exactly the bunching the estimator
looks for. On the evidence retrieved here, **the coupling to the forensic-stats Soviet series is
unpromised at the enterprise level.** It may still be promised at the ministry level, which is what
PLAN section 7.3 already scopes ("ministry-level series only from P2"), and that scoping is
consistent with what was found.

No target shape for phenomenon 1 is available, so the bunching estimator has, for now, no external
distribution to be compared against — only the DP's own predicted distribution (PLAN section 5).

**(d)** unsourced; prior = [no external target shape for phenomenon 1; level established as ministry / sub-branch, so the enterprise-level coupling stays unpromised]; sensitivity handled by regime map / Sobol

---

### Item 7 - The Uzbek cotton affair

**(a) The question.** Magnitude, level and mechanism: was the padding done at the enterprise layer,
the ministry layer, or the republic layer? The output is a **judgement** on whether the episode is
admissible as an anchor for the Phase-2 ministry layer (PLAN section 2.14) at all. The card states
that "not an anchor" is a perfectly good answer and is preferable to a stretched one.

**(b) What was found.** Only tertiary material was retrievable, and it does not answer the question
that matters. The English Wikipedia article on the Uzbek cotton scandal
(<https://en.wikipedia.org/wiki/Uzbek_cotton_scandal>) states:

> "In that year alone, 981,000 tonnes of cotton were reported as being harvested when they did not
> exist."

The article attributes that figure to a single Russian-language secondary source and, on inspection,
**does not state where in the administrative hierarchy the padding was organised**, nor the
mechanism of concealment. Two better sources were attempted and failed: Cucciolla's PhD thesis at
`unora.unior.it` returns HTTP 403, and an OhioLink MA thesis PDF was retrieved but its text could
not be extracted.

**(c) What it implies.** The magnitude is large and the episode is real, but **the level — the one
thing that would make it an anchor for a ministry layer — is not established by any source retrieved
here.** A famous scandal is not automatically a calibration target.

**Judgement: not an anchor.** The Uzbek cotton affair is not admissible as an empirical anchor for
the Phase-2 ministry layer on this evidence. It may be cited as a *motivating example* that padding
at layers above the enterprise occurred, and nothing more. Revisiting this requires the Cucciolla
thesis or equivalent archival work, and a specific finding on the layer at which figures were
altered.

**(d)** unsourced; prior = [not admissible as a ministry-layer anchor; ministry_passthrough and kappa_m stay design parameters swept over their registry ranges]; sensitivity handled by regime map / Sobol

---

### Item 8 - Tooling: availability, versions and prices

**(a) The question.** Current availability and versions of the reference PPO implementation WO-017
will pin, JAX multi-agent scaffolds for the WO-029 port, `rliable` for the interval estimates of
PLAN section 4.3, and open-source MIP solvers for the PLAN section 6.2 oracle. The reference-PPO
version, the solver version and the solver's optimality gap are all CONTRACT rule 10 manifest
fields, so these are operational facts the build depends on, not background.

**(b) What was found.** All figures below were read from the package index or repository API on
**2026-09-07**.

| Component | Version seen | Source |
|---|---|---|
| `rliable` | 1.2.0 | PyPI JSON API |
| `ortools` | 9.15.6755 | PyPI JSON API |
| `highspy` (HiGHS bindings) | 1.15.1 | PyPI JSON API |
| `pyomo` | 6.10.1 | PyPI JSON API |
| `jax` / `jaxlib` | 0.11.1 / 0.11.1 | PyPI JSON API |
| HiGHS (upstream) | v1.15.1, released 2026-07-02 | GitHub `ERGO-Code/HiGHS` releases |
| OR-Tools (upstream) | v9.15, released 2026-01-12 | GitHub `google/or-tools` releases |
| CleanRL | latest **release** v1.0.0 (2022-11-14); repository last pushed 2026-04-20 | GitHub `vwxyzjn/cleanrl` |
| JaxMARL | no release tag read; repository last pushed 2026-09-04 | GitHub `FLAIROx/JaxMARL` |
| PureJaxRL | no release tag read; repository last pushed 2024-09-09 | GitHub `luchris429/purejaxrl` |

**One finding here is load-bearing for WO-017.** CleanRL's most recent tagged release is v1.0.0 from
2022-11-14, while the repository itself has been pushed as recently as 2026-04-20. A release tag is
therefore **not** a usable pin: WO-017 must pin the reference PPO **by commit SHA**, and record that
SHA in the manifest under CONTRACT rule 10. Pinning `cleanrl==1.0.0` would pin something almost four
years older than the code a reader would find on the default branch.

PureJaxRL's last push is 2024-09-09, roughly two years before the date of writing; JaxMARL is
actively maintained. For the WO-029 port, JaxMARL is the better-maintained scaffold on this evidence.

**On the model-tier mapping.** The card also asks for the specific current models behind the
`MID-fast` and `MID-strong` tiers and their prices. **This memo deliberately omits model names.**
The repository carries no AI model or vendor names by the owner's standing instruction, and PLAN
section 12.1 in any case requires the *lead* to record the tier-to-model mapping in the manifest at
issue time — which is the right place for a fact that decays this fast. What is recorded here is the
method: at issue time the lead selects one model per tier, records the identifier and version in the
run manifest alongside the reference-PPO SHA, and budgets against the then-current per-token price.
This is a **deliberate deviation from the card's wording**, recorded as such.

**(c) What it implies.** WO-017 pins CleanRL by commit SHA. WO-029 targets JaxMARL. WO-032 uses
`rliable` 1.2.0. WO-027 uses HiGHS 1.15.1 or OR-Tools 9.15 and records the optimality gap.

**(d)** Sourced, as of 2026-09-07: the versions in the table above, each read from the named package index or repository API on that date. These facts decay and must be re-read at WO-017, WO-027, WO-029 and WO-032 rather than trusted from this memo.

---

## What was tempted and not written

The card calls this the primary quality signal, and asks for every parameter where a
plausible-sounding number was available but not defensible. Listed in full, not trimmed.

**Item 1, bonus schedules — maximum and average bonus as a share of salary.** A search snippet
carried specific percentage figures for the cap on a Soviet manager's bonus and for the average
bonus, attributed to a 1962 Berkeley *California Management Review* piece. The page 404s on the
current host. The figures were **not written**, in any form, because the card forbids "attaching a
citation to a figure that source does not contain" and a snippet is not the source. Item 1 says
"not obtained" instead. This is the single most tempting omission in the memo: the numbers are
memorable, widely repeated, and would have looked authoritative.

**Item 1 — the overfulfilment increment per percentage point.** Nothing retrieved. Not written.

**Item 2, growth directives — a headline five-year-plan growth rate.** Several such figures are
easy to recall and would have read as a sourced range. They are also almost always *realised* output
growth rather than the *directive*, which is the quantity g actually models. Not written; the item
records the form from Harrison and the distinction, and nothing more.

**Item 3, ratchet — a numeric value for lambda.** Weitzman 1980 is a theory paper and its abstract
says so; no empirical coefficient was retrieved from any source. Not written. The Birman 1978
reference is recorded as the retrieval target for a later pass rather than cited as if read.

**Item 3 — tolerance bands (delta).** No evidence either way. Not written as either present or
absent; the item says the record retrieved supports neither.

**Item 4, pripiski — an inspection rate and a sanction schedule.** Nothing retrieved. Not written.
Related and more subtle: it would have been easy to assert that inspection compared the claim to
production rather than stock, or vice versa, since the simulation has to pick one. The memo instead
flags PLAN section 2.8's stock comparison as an **unsourced modelling assumption**, so it is visible
at gate G1.

**Item 5, tenure — a typical director tenure in years.** Nothing retrieved. Not written. The
per-year to per-plan-period conversion is recorded as an assumption so that a later retrieval cannot
skip it.

**Item 6 — a described shape for the fulfilment histogram.** It would have been easy to assert
bunching just above 100% with a hole below, because that is exactly what the simulation is built to
produce and what the literature's qualitative account implies. Asserting it here would have made the
project's own hypothesis into its own evidence. Not written.

**Item 7 — the level of the cotton padding.** The scandal is republic-scale and the temptation was
to conclude from that scale that the padding was organised at the republic or ministry layer. Scale
is not level. Not written; the item concludes "not an anchor" instead, which is the answer the card
says it prefers.

**Item 8 — model names and per-token prices.** Omitted deliberately, for the reason given in the
item, and flagged there as a deviation from the card's wording rather than a silent gap.

## What a later pass should retrieve first

In priority order, judged by how much each would change the G1 decision:

1. Birman, "From the Achieved Level", *Soviet Studies* 30(2), 1978 — the empirical ratchet, and the
   only identified path to a sourced lambda.
2. Berliner, *Factory and Manager in the USSR* (1957), chapters on the bonus system — s and rho_cap
   relative to beta.
3. Any archival study reporting an enterprise-level fulfilment distribution — the target shape for
   phenomenon 1, and the only way to promise the enterprise-level forensics coupling.
4. Cucciolla's thesis on the Uzbek cotton affair — the layer at which figures were altered, and the
   only route to reopening item 7's "not an anchor".
