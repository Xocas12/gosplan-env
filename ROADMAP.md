# ROADMAP

The order in which this repository gets built, and why that order.

This document is a map, not a specification. Where it summarises `PLAN.md`, the plan wins; where it
summarises `CONTRACT.md`, the contract wins. Every figure below is taken from `PLAN.md` or from a
task task in `ROADMAP.md`, and nothing here is a result.

Read alongside:

| File | What it gives you |
|---|---|
| [`README.md`](README.md) | orientation: what the environment is, what is and is not claimed |
| [`CONTRACT.md`](CONTRACT.md) | the 13 binding rules (PLAN section 9 verbatim) |
| `PLAN.md` | the authoritative build plan; git-ignored, supplied out of band, never committed |
| [`ROADMAP.md`](ROADMAP.md) | the task list, the gate conditions and the dependency order |

---

## 1. Framing

### 1.1 The two claims

The programme supports two claims that are stated, tested and reported **separately** (PLAN
section 1.1). No artefact may bear on both without saying which part is which.

**Claim A - emergence.** Under a fixed rule-based planner, learning enterprises rewarded only
through the five-term reward of PLAN section 2.9 produce (i) hidden reserves and report shaving,
(ii) input hoarding and propagated shortage, (iii) horizontal barter, and (iv) storming *in excess
of what input timing forces* - none of which is written into any transition rule or reward term
(CONTRACT rule 7).

Bunching, padding and quality degradation are **not** part of Claim A. They are direct optima of the
reward under agent control and serve as pipeline checks: if they fail to appear, the optimiser is
broken.

**Claim B - counterfactual.** For a pre-registered baseline configuration, the welfare gap to a
full-information oracle closed by an "OGAS" information contrast, by an incentive-reform contrast,
and by both together (with their interaction) is estimated with common random numbers and bootstrap
confidence intervals. Claim B is a set of **named contrasts** (C0, C_OGAS, C_AUDIT, C_INC, C_BOTH;
PLAN section 4.3), not a variance decomposition. A sentence of the form "X% of the welfare loss is
informational" appears in no output unless it is attached to a named contrast and a CI.

### 1.2 What each phase can establish

Reproduced from PLAN section 1.2. The "cannot establish" column is binding.

| Phase | Establishes | Cannot establish |
|---|---|---|
| P1 | Training stack recovers the exactly-solved single-enterprise optimum; 20-enterprise system bunches under a notch and not under a smooth bonus; padding responds to expected penalty as the DP predicts | Anything about Claim A or B |
| P2 | Claim A phenomena on pre-registered mechanism parameters; equilibrium verification; oracle; JAX parity | Claim B |
| P3 | Claim B contrasts; estimator-bias study; LLM study | - |

A Phase-1 artefact is not evidence for Claim A, however suggestive it looks. A Phase-2 artefact is
not evidence for Claim B.

### 1.3 What is not claimed

From PLAN section 1.1, unchanged:

- **Necessity.** Not that these institutional rules are the only way to produce these behaviours.
- **Causality about the historical USSR.** This is a model, not an identification strategy.
- **Transfer of any absolute number to archival data.** No quantity produced here is an estimate of
  anything that happened.

The coupling to [`forensics-core`](https://github.com/Xocas12/forensics-core) and the `gosplan` project in
[`forensic-economy`](https://github.com/Xocas12/forensic-economy) is scoped to **estimator robustness**
(PLAN section 7.3) and to nothing else.

### 1.4 Standing warning: nothing has been run

**No result exists.** Nothing in this repository has been executed. There are no trained agents, no
runs, no figures, no estimates, no findings. `runs/` is empty. Gates G0-G4 are all unsigned. Every
function body in `gosplan/`, `spec/`, `ref/` and `tests/` raises `NotImplementedError`, with one
exception: `scripts/contract_guard.py`, its test `tests/unit/test_contract_guard.py` and
`tests/conftest.py`'s `pytest_configure` are fully implemented CI infrastructure and must pass.
A failure there is a real failure, never expected skeleton behaviour.

Consequently, every number visible in the repository today is either a provisional parameter default
from PLAN section 3 (six of them carry a dagger and are **replaced at gate G1**: `audit_rate` a,
`ratchet_lambda` lambda, `growth_directive` g, `overfulfilment_slope` s, `penalty_scale` pen,
`effort_cost` kappa) or a template placeholder reading `TBD`. **No number in this repository may be
cited as a result, an estimate, or a historical fact.**

Four phenomena are **held out** and are not computed, plotted or tested before the Phase-2
acceptance run (PLAN section 4.1): storming excess (row 2), hoarding to shortage (row 5), blat
(row 6), hidden reserves and shaving (row 7). Their mechanism parameters are locked now, in advance,
by PLAN section 4.2.

---

## 2. The critical path

The DAG of PLAN section 12.2, drawn with the dependency edges the tasks actually carry. Gate nodes
are hexagons; the maintainer-owned join points (a later task and the P2 spec revision)
are the places where the graph narrows to a single owner.

```mermaid
flowchart TD
  WO000["a later task param sourcing"] --> WO001["a later task spec v0 + contract"]
  WO001 --> WO002["a later task ref + frozen tests"]
  WO001 --> WO003["a later task config"]
  WO001 --> WO004["a later task rng"]
  WO003 --> WO005["a later task production"]
  WO003 --> WO006["a later task planner"]
  WO003 --> WO007["a later task reporting/reward"]
  WO003 --> WO008["a later task obs"]
  WO004 --> WO005
  WO004 --> WO006
  WO004 --> WO007
  WO004 --> WO008
  WO007 --> WO008
  WO005 --> WO009["a later task step/env maintainer"]
  WO006 --> WO009
  WO007 --> WO009
  WO008 --> WO009
  WO009 --> WO010["a later task heuristics"]
  WO009 --> WO011["a later task ledger"]
  WO010 --> WO012["a later task MC sanity"]
  WO011 --> WO012
  WO012 --> G0{{"G0 Scaffold and sanity - maintainer"}}

  G0 --> WO013["a later task spec v1 freeze maintainer"]
  WO013 --> WO014["a later task single-enterprise DP"]
  WO014 --> WO015["a later task regime map"]
  WO015 --> G1{{"G1 Regime map and freeze - human picks P1 values"}}

  G1 --> WO016["a later task P1 metrics"]
  G1 --> WO017["a later task PPO adapter maintainer"]
  WO017 --> WO018["a later task train harness"]
  WO011 --> WO018
  WO014 --> WO019["a later task DP vs PPO"]
  WO018 --> WO019
  WO016 --> WO020["a later task P1 gate exp"]
  WO018 --> WO020
  WO019 --> G2{{"G2 Phase 1 gate - human + maintainer"}}
  WO020 --> G2

  G2 --> P2SPEC["P2 spec revision maintainer"]
  P2SPEC --> WO021["a later task quality"]
  P2SPEC --> WO022["a later task delivery timing"]
  P2SPEC --> WO023["a later task holding, audit, soft budget"]
  P2SPEC --> WO024["a later task trade"]
  P2SPEC --> WO025["a later task ministry"]
  P2SPEC --> WO027["a later task oracle"]
  P2SPEC --> WO028["a later task exploitability"]
  WO025 --> WO026["a later task LLM ministry adapter"]
  WO021 --> WO029["a later task JAX port maintainer"]
  WO022 --> WO029
  WO023 --> WO029
  WO024 --> WO029
  WO025 --> WO029
  WO029 --> WO028
  WO021 --> WO030["a later task held-out metrics"]
  WO022 --> WO030
  WO023 --> WO030
  WO024 --> WO030
  WO025 --> WO030
  WO016 --> WO030
  WO026 --> WO031["a later task P2 acceptance"]
  WO027 --> WO031
  WO028 --> WO031
  WO029 --> WO031
  WO030 --> WO031
  WO031 --> G3{{"G3 Phase 2 acceptance - human + maintainer"}}

  G3 --> WO032["a later task contrasts"]
  G3 --> WO034["a later task estimator bias"]
  G3 --> WO035["a later task LLM study"]
  WO032 --> WO033["a later task Sobol optional"]
  WO032 --> WO036["a later task price sensitivity"]
  WO033 --> WO036
  WO034 --> WO036
  WO035 --> WO036
  WO036 --> WO037["a later task report generation"]
  WO037 --> G4{{"G4 Final report - human"}}

  classDef gate fill:#1f2933,stroke:#0b0f14,stroke-width:2px,color:#ffffff;
  classDef lead stroke-width:3px;
  class G0,G1,G2,G3,G4 gate;
  class WO009,WO013,WO017,WO029,P2SPEC lead;
```

Two edges are easy to misread:

- **a later task does not block a later task through a later task.** It writes the frozen suites those tasks must
  pass, so it must land before they can be *checked*, but its own dependency is a later task only, and
  no task - a later task included - names it under *Depends on*.
- **a later task point forward only.** a later task produces the G3 artefact and a later task re-scores
  headline tables afterwards; PLAN section 12.2 places no P3 task before G3, so the reverse edge
  would close a cycle and does not exist.

---

## 3. Milestones

Pass conditions are PLAN section 13; artefacts and sign-off likewise. Compute is PLAN section 14 -
those are the plan's budget estimates, not measurements.

| Gate | Work orders | Pass condition (PLAN section 13) | Artefacts | Sign-off | Compute (PLAN section 14) |
|---|---|---|---|---|---|
| **G0 Scaffold and sanity** | a later task - a later task | Full frozen suite green; MC sanity report clean; maintainer's diff review of `env/` against CONTRACT rule 7 | `runs/mc_sanity/report.md` | maintainer | MC sanity: 3 agents x 2,000 episodes x 21 configs at approx. 50 agent-steps each - CPU minutes |
| **G1 Regime map and freeze** | a later task - a later task | Regime map produced; human selects the P1 dagger values from the interior of the bunching region; three `a*pen` levels and the `b_hat_DP` thresholds recorded **before** any training | `runs/G1_decision.md`, `spec` v1.0.0 | Human | DP regime map: 500 configs at approx. 1 CPU-min each - approx. 1 h on 8 cores |
| **G2 Phase 1 gate** | a later task - a later task | PLAN section 4.5 criteria 1-4: DP recovery, bunching present/absent, padding elasticity, hygiene | `runs/dp_vs_ppo/report.md`, `runs/phase1_gate/report.md` | Human + maintainer | DP-vs-PPO: 3 levels x 10 seeds at `N=1`, approx. 2M env steps - approx. 1 GPU-h or 8 CPU-h. P1 gate: 2 arms x 30 seeds at `N=20`, approx. 5M env steps at approx. 3k steps/s in NumPy - approx. 30 h CPU, approx. 4 h on 8 cores |
| **G3 Phase 2 acceptance** | a later task - a later task | Held-out phenomena 2, 5, 6, 7 evaluated on the PLAN section 4.2 values (pass or reported failure); exploitability below threshold on all arms used; oracle gap recorded; JAX parity | `runs/phase2_acceptance/report.md` | Human + maintainer | P2 acceptance: approx. 8 configs x 30 seeds - approx. 1-2 days CPU, or hours on JAX. Exploitability doubles the runs it audits - budget for it |
| **G4 Final report** | a later task - a later task | Contrasts with CIs; estimator-bias curves; LLM study; price sensitivity on every headline table | final report | Human | Contrasts: 5 x 30 on JAX - hours. Sobol (optional): 1,500-2,800 runs, JAX only - approx. 1-3 GPU-days. Estimator bias: 10 x 30 plus the DP - hours. LLM study: approx. 4M tokens - tens of dollars |

The five strings in the Gate column - `G0 Scaffold and sanity`, `G1 Regime map and freeze`,
`G2 Phase 1 gate`, `G3 Phase 2 acceptance`, `G4 Final report` - are the GitHub **milestone**
names, and GitHub matches a milestone by exact string. They are used verbatim here, in the
section 4 headings below, in `CONTRIBUTING.md` section 9 and in the `Gate` dropdown of
`.github/ISSUE_TEMPLATE/gate.yml`. Do not paraphrase one.

Programme-wide, PLAN section 14 also budgets **delegation**: approx. 38 tasks at approx. 40k
tokens each including retries, at the mid tier, so approx. 1.5M tokens - dollars, with the maintainer's
spec, test and task writing plus ambiguity resolutions dominating the real cost.

JAX is required for Sobol and helpful from P2 onward. Phase 1 is deliberately achievable in NumPy,
so the JAX port is not on the critical path to the first real check.

**A gate that fails produces a written failure report.** The next task is then a lead
diagnosis - never a parameter change made in order to pass the gate. Parameter changes after G1
create a new, labelled study with its own pre-registration (PLAN section 13). Failure of G2
criterion 1 is a training-stack failure and blocks everything; failure of criterion 2 with
criterion 1 passing is a multi-agent effect and is a **result**, reported and not tuned away
(PLAN section 4.5).

---

## 4. Tasks by milestone

"Diff." is the 1-5 difficulty estimate. Every task below is built; the status column names the
artefact, and where a gate was not passed it says so. The gate records under `runs/` are the
authority on what each result does and does not show.

### G0 Scaffold and sanity - signed off (`runs/G0_signoff.md`)

| Title | Diff. | Status |
|---|---|---|
| Parameter sourcing memo | 3 | **done** - `docs/params_sources.md` |
| Spec v0, CONTRACT, registry | 4 | **done** - `spec/spec.py`, `CONTRACT.md`, `gosplan/params.py` |
| Reference dynamics and frozen tests | 5 | **done** - `ref/ref_step.py`; `make golden` generates 30 trajectories |
| Config | 2 | **done** - `gosplan/config.py`, with cross-field validation and config hashing |
| RNG | 2 | **done** - `gosplan/rng.py` |
| Production | 3 | **done** - `gosplan/env/production.py` |
| Planner | 3 | **done** - `gosplan/env/planner.py` |
| Reporting and reward | 3 | **done** - `gosplan/env/reporting.py`, `reward.py`, `prices.py` |
| Observation | 2 | **done** - `gosplan/env/obs.py` |
| Step function and env wrapper | 5 | **done** - `gosplan/env/step.py`, `env.py`; golden parity exact on all 30 cells |
| Heuristic agents | 2 | **done** - `gosplan/agents/heuristic.py` |
| Ledger and manifest | 2 | **done** - `gosplan/metrics/ledger.py` |
| Monte-Carlo sanity harness | 3 | **done** - `runs/mc_sanity/report.md`, all assertions held |

The sanity harness computes and plots no held-out quantity (PLAN section 4.1, rows 2, 5, 6 and
7); its assertions are conservation and boundedness only, never direction.

### G1 Regime map and freeze - values recorded (`runs/G1_decision.md`)

| Title | Diff. | Status |
|---|---|---|
| Spec v1 freeze | 3 | **done** - `SPEC_VERSION` 1.0.0, then post-freeze revisions to 2.1.0 in `spec/CHANGELOG.md` |
| Single-enterprise DP | 4 | **done** - `gosplan/agents/dp.py` |
| Regime map experiment | 3 | **done** - `runs/regime_map/` (500 points, 71 interior bunching candidates) |

G1 is where the six provisional parameters were fixed from the interior of the bunching region,
with the thresholds recorded **before** any training run. The owner delegated the choice.

### G2 Phase 1 gate - **not passed** (`runs/G2_record.md`)

| Title | Diff. | Status |
|---|---|---|
| Phase-1 metrics (rows 1 and 4 only) | 3 | **done** - `gosplan/metrics/phenomena.py`, `_fallback.py` |
| PPO adapter | 5 | **done** - `gosplan/agents/ppo/adapter.py` |
| Training harness | 3 | **done** - `gosplan/agents/ppo/train.py` |
| DP-vs-PPO recovery (G2 criterion 1) | 3 | **run; not met** in three attempts - `runs/dp_vs_ppo/` (limitation L1) |
| Phase-1 gate experiment (G2 criteria 2-4) | 3 | **run as a labelled study** - `runs/phase1_gate/report.md` (limitation L2) |

Nothing in this block implements phenomena rows 2, 5, 6 and 7. They arrive at G3.

### G3 Phase 2 acceptance - **not passed** (`runs/G3_record.md`)

The Phase-2 spec revision is `spec/P2_REVISION.md` (R1-R16).

| Title | Status |
|---|---|
| Quality mechanism on | **done** (R5) |
| Delivery timing (`stochastic`, `backloaded`) | **done** (R6) |
| Input holding loss, targeted audits, soft budget | **done** (R7, R4, R8) |
| Trade matching | **done** (R9); the offer timing fixed by R15 after G3 found no learner could post an offer (D1) |
| Rule-based ministry, `n_ministries` | **done** (R10) - `gosplan/env/ministry.py` |
| LLM ministry adapter | **done** - `gosplan/agents/llm_ministry.py`, `llm_client.py` |
| Oracle (expected-value MIP) | **done** - `gosplan/oracle/kantorovich.py` (HiGHS) |
| Exploitability harness | **done** - `gosplan/experiments/exploitability.py`; revised by R16 after G3 found the audit inconclusive (D2) |
| JAX port | **done** - `gosplan/jax/`, NumPy parity about 1e-13 |
| Held-out phenomena metrics (rows 2, 3, 5, 6, 7) and the Phase-2 heuristic agents | **done** - first computed in the acceptance run |
| Phase-2 acceptance experiment | **run** - `runs/phase2_acceptance/report.md`; the labelled G3b study (R15, R16) is `runs/G3b_record.md`: row 6 fails behaviourally, populations non-converged; the learner-convergence study LC (R17) is `runs/LC_record.md`; the coordination-trap test CT (R18) is `runs/CT_record.md`; the budget-versus-schedule study ES (R19) is `runs/ES_record.md` |

The four held-out phenomena were measured for the first time in the acceptance run only.
`scripts/contract_guard.py` still fails CI if a Phase-1 module so much as calls one.

### G4 Final report - not met until the LLM study runs (`runs/final_report/report.md`)

The Phase-3 revision is `spec/P3_REVISION.md` (S1-S7); the record is `runs/P3_record.md`.

| Title | Status |
|---|---|
| Contrasts harness with `rliable` | **run** - `runs/contrasts/report.md` (15 seeds per new arm, reduced for compute) |
| Saltelli/Sobol design and total-order indices (optional) | **not run** by decision (S3): out of reach without a JAX trainer |
| Estimator-bias study | **run** - `runs/estimator_bias/report.md` |
| LLM ministry study | **implemented; NOT RUN** - needs model credentials; the command is in `runs/llm_study/report.md` |
| Price-vector sensitivity on every headline table | **run** - `runs/price_sensitivity/report.md` |
| Report generation: figures, tables, manifest roll-up | **done** - `gosplan/experiments/report.py` |

## 5. Parallelism and serialisation

### 5.1 What can be worked concurrently

| After | Concurrent set | Note |
|---|---|---|
| - | a later task | a later task writes the suites the other two must pass, so it should land first in wall-clock terms even though it is not their dependency |
| - | **a later task** | The Phase-1 fan-out. Four separate sessions, disjoint "Write only" lists, all checked against a later task's frozen suite |
| - | a later task | Both single-file tasks on disjoint paths |
| G1 | - | Both depend only on a later task |
| - | a later task | Two independent experiments, one per G2 criterion group |
| the P2 spec revision | **a later task** | The Phase-2 fan-out. a later task waits on a later task; a later task wait on a later task - a later task; a later task additionally waits on a later task if the arms it audits run on JAX |
| G3 | - | a later task waits on a later task; a later task waits on every headline table |

### 5.2 Strict serialisation points

- **a later task -> a later task -> a later task.** The trunk: sourcing before registry, registry before reference
  dynamics and frozen tests. No task in the fan-out can be *checked* until a later task exists.
- a later task, the maintainer-owned join. Five upstream tasks converge on the step function and the env
  wrapper; nothing downstream starts until it lands. maintainer-owned by design (PLAN finding F14).
- a later task, the spec v1 freeze. Everything before it works against a provisional interface;
  everything after works against a frozen one.
- **The P2 spec revision**, maintainer-owned, between G2 and a later task. a later task's matching rule and surplus
  definition, and a later task's MIP formulation, are written here - not by the contributors.
- later tasks, the two roll-up tasks. Each waits on its whole block.
- **Every gate.** A gate is a written sign-off on named artefacts, not a vibe. No task in the next
  block is issued before the previous gate is signed.

### 5.3 Why the fan-outs are safe

`tests/unit`, `tests/behavioural` and `tests/golden` are read-only for contributors (CONTRACT
rule 2). `tests/acceptance/` is lead-run, is never on a must-pass list and is never executed by an
contributor (CONTRACT rule 13). An contributor reads only the files on its task's whitelist
and writes only the files it names (CONTRACT rule 12). That is what makes the concurrent sets above
collision-free: disjoint write sets, one shared frozen suite, no shared editable state.

---

## 6. Human decision points

### 6.1 G1 parameter selection - the single most consequential human act in the programme

At gate G1 a human reads the regime map produced by a later task - a Latin-hypercube sample of 500 points
over `(beta, s, a*pen, lambda, psi, g, kappa, w)` with `w` in `{0, 0.25}`, heat maps of the regime
label and of `b_hat_DP`, and a candidate list of at least five interior bunching-region points - and
records, in `runs/G1_decision.md`:

1. The Phase-1 values for the six provisional dagger parameters, chosen **from the interior of the
   bunching region**, not from its edge.
2. The **three `a*pen` levels** used by G2 criterion 1, chosen to span the bunching region.
3. The **`b_hat_DP` thresholds** that G2 criterion 2 is scored against.

All three are recorded **before any training run**. That ordering is the whole point of the gate:
the acceptance thresholds are fixed before the thing being accepted exists. `spec/spec.py` is frozen
at v1.0.0 in the same gate.

Parameter changes after G1 do not amend this decision - they create a new, labelled study with its
own pre-registration (PLAN section 13).

### 6.2 Gate sign-offs

| Gate | Signs | What the signature asserts |
|---|---|---|
| G0 | maintainer | Frozen suite green, MC sanity report clean, and the maintainer has personally reviewed every `env/` diff against CONTRACT rule 7 (no hard-coded pathology) |
| G1 | Human | The regime map exists, and the values in 6.1 are recorded before any training run |
| G2 | Human + maintainer | PLAN section 4.5 criteria 1-4, each with an explicit pass/fail line in the report |
| G3 | Human + maintainer | Held-out phenomena 2, 5, 6 and 7 evaluated on the section 4.2 locked values - pass **or reported failure**; exploitability below threshold on all arms used; oracle gap recorded; JAX parity |
| G4 | Human | Contrasts with CIs; estimator-bias curves; LLM study; price sensitivity on every headline table |

A failed gate is signed as failed and produces a written failure report. The next task is a
lead diagnosis.

### 6.3 Open questions

None are open. The two filed against the skeleton were answered in the build:
`tests/acceptance/` is maintainer-authored and holds the five gate harnesses, and the P2 and P3
modules the PLAN section 8 tree does not draw are `gosplan/experiments/phase2_acceptance.py` and
`gosplan/experiments/report.py`. The lead rulings made during the build are in `docs/rulings/`.

---|---|---|---|
| [`open question 001`](open question 001) | Which task is authorised to author `tests/acceptance/`, given that no task's "Write only" list names it? | (A) add `tests/acceptance/*` to a maintainer task's "Write only", a later task being the natural home since it already re-runs the full suite; (B) declare the directory maintainer-authored outside the whitelist system, exempt from CONTRACT rule 12 | Nothing. The five harnesses are excluded from `testpaths` |
| [`open question 002`](open question 002) | What are the module names for the P2 acceptance harness and the P3 report generator, which PLAN section 8's tree does not draw? | (A) add both names to the section 8 tree now and create the stubs; (B) note that P2/P3 harness module names are fixed at the P2 spec revision, and leave the two files uncreated until then | Nothing. Both modules are P2/P3 and no Phase-1 test imports them |

Neither answer changes any dynamics, metric or result. Both are recorded here so that the next
session does not silently re-derive them - which is exactly what CONTRACT rule 3 exists to prevent.

---

## 7. Definition of done for the programme

The programme is done when **gate G4 is signed by the human**, which requires all of:

1. **Contrasts with CIs.** C0, C_OGAS, C_AUDIT, C_INC and C_BOTH at 30 seeds with common random
   numbers, reported as IQM with stratified bootstrap 95% CIs over `welfare_ratio`, `padding_index`
   and `specification_gap`; the gap closed `Delta_X` per contrast, and the interaction
   `I = Delta_BOTH - Delta_OGAS - Delta_INC` (PLAN section 4.3). C_AUDIT is dual-classified and
   reported separately, never folded into C_OGAS.
2. **Estimator-bias curves** from a later task, scored against the DP's exact no-manipulation
   counterfactual.
3. **The LLM study** from a later task, with its payoff-variation arms and its manipulation check.
4. **Price sensitivity on every headline table** from a later task - every table, not a sample of them.
5. **The final report** from a later task, rolling up figures, tables and manifests.

And, as standing conditions rather than deliverables:

- Every run under `runs/<hash>/` carries the CONTRACT rule 10 manifest: config hash, spec version,
  git hash, seeds, reference-PPO version, estimator version, LLM model ids and versions, solver
  version and optimality gap, flags.
- Any arm whose exploitability exceeds the G3 threshold is labelled **non-converged**, is not
  reported as a result, and keeps that label in the final report.
- Any run with more than 1% of reports at `rho_max = 10` carries the `BOUND_BINDING` flag and is
  reported with it. Bounds are results and are never silently widened or narrowed
  (CONTRACT rule 8).
- Claim A and Claim B are reported separately, and the "not claimed" list of section 1.3 above is
  intact in the final text: no necessity claim, no causal claim about the historical USSR, no
  absolute number transferred to archival data.
- Held-out phenomena that failed to appear are reported **as failures**, with the locked
  section 4.2 parameter values stated alongside.

---

## 8. Risks and known unknowns

Each of these is a way the programme could produce a null or an uninterpretable result. None of them
is a reason to change the design mid-flight.

**The provisional parameter values are unsourced until a later task.** Six Phase-1 defaults carry a dagger
in PLAN section 3, and several more are marked "unsourced" outright. a later task ends each research item
with either a sourced range or the explicit sentence "unsourced; prior = [...]; sensitivity handled
by regime map / Sobol", and no item may end with a point value presented as historical fact. Until
that memo exists, no number in `gosplan/params.py` has evidentiary standing - and note that what
actually chooses the Phase-1 values at G1 is the regime map, not a literature citation.

**The multi-agent training may not converge.** Independent PPO in a 20-agent economy has no
convergence guarantee. That is why the exploitability harness exists: freeze `N-1` learned
policies, train one fresh best-responder with the same PPO configuration and budget, and report
`(R_BR - R_pop) / |R_pop|` per seed. The threshold is **provisional at 5%** and is finalised by the
lead at G3, using the DP-recovery tolerance as a guide. Arms above the threshold are reported as
non-converged rather than as results. The harness roughly doubles the compute of the runs it audits,
and PLAN section 14 says to plan for that.

**The held-out phenomena may simply fail to appear.** Storming excess, hoarding to shortage, blat and
hidden reserves are held out precisely so that their absence is informative. If a held-out
phenomenon does not appear on the PLAN section 4.2 locked values, **the result is reported as a
failure**. It is not a bug to be tuned away, and the locked mechanism parameters may be changed only
in a new, labelled study with its own pre-registration. The same discipline applies one level up: if
Phase-1 bunching fails while DP recovery passes, that is a multi-agent effect and a result
(PLAN section 4.5).

**The `forensics-core` interface is not agreed until G1.** PLAN section 7.3 scopes the coupling to
estimator robustness - bunching bias, reconciliation, dispersion - excludes digit tests, and admits
ministry-level series only from P2. a later task must ship `gosplan/metrics/_fallback.py` with signatures
identical to `forensics_core`'s so either can be swapped in, and a later task research item 6 decides the
level of the Soviet series and therefore what section 7.3 can promise at all. Until that lands,
treat every `forensics_core` call site as provisional.

**The pipeline checks are load-bearing.** Bunching, padding and quality degradation are not claims;
they are the instruments that say whether the optimiser works. If they fail, nothing downstream of
them means anything, and the correct move is a lead diagnosis, not a parameter search.

**Placeholders are not specifications.** Every task from a later task to a later task is marked NOT ISSUABLE YET
and carries a difficulty the maintainer sets at issue. Their dependency lists are real; their
implementation notes are provisional and are rewritten by the P2 spec revision (for the G3 block) or
after G3 (for the G4 block). Do not start one because it looks ready.
