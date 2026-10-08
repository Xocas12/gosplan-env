# gosplan-env

A multi-agent reinforcement-learning environment in which learning **enterprises** face a fixed
**rule-based planner**: the planner sets targets, allocates inputs from reported claims, audits a
random sample of reports, and ratchets next period's targets on the reported fulfilment ratio.
Enterprises choose effort and what to report. Nothing else is scripted.

The repository exists to support two claims that are stated, tested and reported **separately**
(PLAN §1.1), and to make it obvious which one a given artefact does or does not bear on.

---

## What is built

Phases 1 to 3 of the build plan are implemented, and each has been run. The only stubs left are
the optional Sobol design (`gosplan/experiments/sobol.py`, not run by decision: see
`spec/P3_REVISION.md` S3) and `spec/spec.py`, which is the frozen interface surface by design.

| Component | What it is |
|---|---|
| `ref/` | The pure-Python reference dynamics and the golden-trajectory generator: the executable specification the environment is checked against (golden parity exact on all 30 cells) |
| `gosplan/env/` | The environment: production, planner, reporting, audits, reward, observation, prices, and the Phase-2 mechanisms (horizontal trade, ministries, quality, storming inputs) |
| `gosplan/agents/` | Heuristic agents (Random, TruthfulMyopic, Padder, DPGreedy, Berliner, Weitzman, Kornai), the exact single-enterprise DP, the independent-PPO learner and its training harness, and the LLM ministry adapter |
| `gosplan/metrics/` | The ledger and run manifests, the PLAN section 4.1 phenomenon operationalisations, and the bunching and reconciliation estimators |
| `gosplan/oracle/`, `gosplan/jax/` | The full-information Kantorovich oracle (HiGHS) and the JAX port of the step function, with a NumPy parity check |
| `gosplan/experiments/` | Every experiment driver: MC sanity, regime map, DP vs PPO, the Phase-1 gate study, exploitability, the Phase-2 acceptance run, contrasts, estimator bias, price sensitivity, the LLM study and the final report |

Spec revisions after the v1 freeze are pre-registered in `spec/P2_REVISION.md` (R1-R16) and
`spec/P3_REVISION.md` (S1-S7) and logged in `spec/CHANGELOG.md`; the spec is at 2.1.0. The lead
rulings behind them are in `docs/rulings/`.

## Where the project stands

Every result is in `runs/`, and each gate's record says exactly what it does and does not show.
Read the records before quoting a number: several results are reported failures or labelled
studies, and that is how they must be cited.

| Gate | Record | Status |
|---|---|---|
| G0 | `runs/G0_signoff.md` | signed off (lead) |
| G1 | `runs/G1_decision.md` | Phase-1 values chosen (the owner delegated the decision) |
| G2 | `runs/G2_record.md` | **not passed**: PPO did not recover the single-enterprise DP optimum in three attempts. Criteria 2-4 were run afterwards as a labelled study |
| G3 | `runs/G3_record.md` | **not passed**: rows 2 and 5 appear; row 7 fails its null test; row 3 and hygiene fail; row 6 was not evaluable because of a trade defect (D1); the exploitability audit was inconclusive (D2) |
| G3b | `runs/G3b_record.md` | labelled study after the D1 fix (R15) and the revised audit (R16): row 6 **fails behaviourally** (learners post only buy offers, so nothing trades), and the revised audit finds the populations **non-converged** (median exploitability 0.80) |
| LC | `runs/LC_record.md` | labelled study (R17): tripling the training budget lowers exploitability (median 0.80 to 0.00) but the populations do not converge, and the C0 economy **collapses to near-zero output** (effort 0.02, welfare 0 on every seed) |
| CT | `runs/CT_record.md` | labelled study (R18, evaluation only): the 3M collapse **is a coordination trap**. A lone producer loses (median -1.72 [-2.24, -1.44]), and everyone producing leaves every seat better off (median +0.87 [+0.49, +1.30]). The 1M populations are not a trap: they are too far from equilibrium |
| ES | `runs/ES_record.md` | labelled study (R19): with the 1M runs' exact entropy schedule, 3M-step training **still collapses** (median effort 0.019 [0.012, 0.023]) into the same coordination trap. The collapse is driven by training length, not the stretched schedule |
| P3 | `runs/P3_record.md` | contrasts, estimator bias and price sensitivity complete as a labelled study; the LLM study is NOT RUN (it needs model credentials) |
| G4 | `runs/final_report/report.md` | not met until the LLM study runs; human sign-off pending |

Three limitations travel with every learned-agent number:
- **L1.** The PPO learner does not recover the single-enterprise DP's mixed under-reporting
  strategy (G2 criterion 1).
- **L2.** The pre-registered bunching estimator is undefined on degenerate distributions and
  over-confident on sharply peaked ones (G2 criterion 2).
- **L3.** Every Phase-2 and Phase-3 number before G3b comes from an economy in which no learner
  could post a trade offer (G3 D1).
- **L4.** Under the revised audit (R16) the learned C0 populations are not approximate equilibria:
  a warm-started best responder gains on 9 of 10 seeds (G3b).
- **L5.** Learned-economy outcomes depend on the training budget: at 3M agent-steps the C0
  economy collapses to near-zero output (LC), under either entropy schedule (ES). Every
  learned-agent number here comes from 1M-step populations and describes a point on a learning
  trajectory, not a steady state.

The learned Phase-2 economy is heavily degraded: `welfare_ratio` is about 0.03, where the
truthful-myopic baseline reaches about 0.53. Every contrast is a movement within that economy.

The four held-out phenomena (storming excess, hoarding to shortage, blat, hidden reserves) were
first computed in the Phase-2 acceptance run, as pre-registered, and `scripts/contract_guard.py`
still fails CI if a Phase-1 module calls one of them.

---

## What is claimed

**Claim A - emergence.** Under a fixed rule-based planner, learning enterprises rewarded only
through the five-term reward of PLAN §2.9 produce (i) hidden reserves and report shaving,
(ii) input hoarding and propagated shortage, (iii) horizontal barter, and (iv) storming *in excess
of what input timing forces* - none of which is written into any transition rule or reward term
(CONTRACT rule 7).

Bunching, padding and quality degradation are **not** part of Claim A. They are direct optima of the
reward under agent control, and serve as pipeline checks: if they fail to appear, the optimiser is
broken.

**Claim B - counterfactual.** For a pre-registered baseline configuration, the welfare gap to a
full-information oracle closed by an "OGAS" information contrast, by an incentive-reform contrast,
and by both together (with their interaction) is estimated with common random numbers and bootstrap
confidence intervals. Claim B is a set of **named contrasts** (PLAN §4.3), not a variance
decomposition. A sentence of the form "X% of the welfare loss is informational" appears in no output
unless it is attached to a named contrast and a CI.

## What is *not* claimed

- **Necessity.** That these institutional rules are the only way to produce these behaviours.
- **Causality about the historical USSR.** This is a model, not an identification strategy.
- **Transfer of any absolute number to archival data.** No quantity produced here is an estimate of
  anything that happened.

The coupling to [`forensics-core`](https://github.com/Xocas12/forensics-core) and the `gosplan` project in
[`forensic-economy`](https://github.com/Xocas12/forensic-economy) is scoped to **estimator robustness**
(PLAN §7.3) and to nothing else.

## What each phase can establish

| Phase | Establishes | Cannot establish |
|---|---|---|
| P1 | Training stack recovers the exactly-solved single-enterprise optimum; 20-enterprise system bunches under a notch and not under a smooth bonus; padding responds to expected penalty as the DP predicts | Anything about Claim A or B |
| P2 | Claim A phenomena on pre-registered mechanism parameters; equilibrium verification; oracle; JAX parity | Claim B |
| P3 | Claim B contrasts; estimator-bias study; LLM study | - |

(PLAN §1.2. Read the "cannot establish" column as binding: a Phase-1 artefact is not evidence for
Claim A, however suggestive it looks.)

---

## Repository layout

```
gosplan-env/
  PLAN.md                     # build plan; git-ignored, never committed
  CONTRACT.md                 # §9 verbatim
  pyproject.toml
  spec/
    spec.py                   # v0 provisional → v1 frozen at G1 (§10)
    CHANGELOG.md              # every change after v1: version, reason, approver
  gosplan/
    config.py                 # EnvConfig dataclasses, validation, hashing
    params.py                 # registry (§3): name, arm, default, range, source, phase
    rng.py                    # key-based RNG (§2.15)
    env/
      state.py
      production.py           # §2.6
      planner.py              # §2.7 (targets, allocation, delivery, audits, planner view)
      reporting.py            # §2.8 (report processing, audit, penalty, bonus)
      reward.py               # §2.9 (reward, scale, val, welfare, headline metrics)
      obs.py                  # §2.4
      prices.py               # §2.10
      trade.py                # §2.13 (P2)
      ministry.py             # §2.14 (P2)
      step.py                 # maintainer: period schedule §2.5, assembles modules
      env.py                  # reset/step wrapper, specs, info/ledger hookup
    agents/
      base.py                 # Agent protocol
      heuristic.py            # Random, TruthfulMyopic, Padder, DPGreedy (P2: Berliner, Weitzman, Kornai)
      dp.py                   # §5
      ppo/adapter.py          # maintainer: thin adapter over reference PPO
      ppo/train.py            # training harness, checkpoints, eval
      llm_ministry.py         # P2
    oracle/kantorovich.py     # P2
    metrics/
      ledger.py               # StepRecord, Ledger, run manifest
      phenomena.py            # §4.1 operationalisations
      _fallback.py            # vendored estimator signatures (§7.3)
    experiments/
      mc_sanity.py            # G0 Monte-Carlo sanity sweep
      regime_map.py           # G1 DP regime map
      dp_vs_ppo.py            # G2 criterion 1: PPO vs the DP
      phase1_gate.py          # G2 criteria 2-4
      exploitability.py       # P2 best-responder audit
      phase2_acceptance.py    # P2 acceptance run (G3, G3b)
      contrasts.py            # P3
      sobol.py                # P3 optional
      estimator_bias.py       # P3
      llm_study.py            # P3
      price_sensitivity.py    # P3
      report.py               # P3 final report (G4 artefact)
    jax/                      # maintainer, P2: port + parity
  ref/
    ref_step.py               # maintainer: slow pure-Python reference dynamics — the test oracle
    gen_golden.py             # generates tests/golden/*.json from ref
  tests/
    unit/                     # property and conservation tests (frozen)
    behavioural/              # heuristic-agent behavioural tests incl. no-hardcoded-pathology (frozen)
    golden/                   # generated from ref (frozen)
    acceptance/               # lead-run experiments; NOT in CI; not "tests" in the contract sense
  runs/                       # manifests + results, one directory per run hash
```

(PLAN section 8.)

---

## PLAN.md is the build instruction

`PLAN.md` ("gosplan-env - Build Plan v1.0") is the authoritative design document. Everything in this
repository is derived from it, and every module docstring, task task and test cites the PLAN
section it realises.

- It is **deliberately git-ignored** (see `.gitignore`) and is never committed. Do not un-ignore it,
  do not vendor it, do not copy large prose blocks of it into code or docs. Cite section numbers.
- It **must be present locally at the repository root** for any task to be executed. Without
  it, a session cannot check its work against the spec, and the correct action is to stop rather
  than to reconstruct the missing sections from the code.
- Where this README and `PLAN.md` disagree, `PLAN.md` wins - except for `CONTRACT.md`, which is
  PLAN §9 reproduced verbatim and is the binding text.

## Documents in this repository

| File | What it is | Owner |
|---|---|---|
| `CONTRACT.md` | PLAN §9 verbatim: the 13 rules that bind every session, human or model | maintainer |
| `README.md` | this file | maintainer |
| `spec/CHANGELOG.md` | every change to `spec/spec.py` after the v1 freeze: version, reason, affected tasks, approver | maintainer |
| `docs/params_sources.md` | sourced range or explicit prior for every provisional parameter | maintainer |
| `docs/ref_worked_example.md` | the hand-checked 2-enterprise, 2-sector validation of `ref/ref_step.py` | maintainer |
| `ROADMAP.md` | the task list, the gate conditions and the dependency order | maintainer |
| `spec/P2_REVISION.md`, `spec/P3_REVISION.md` | the pre-registered Phase-2 and Phase-3 designs (R1-R16, S1-S7) | maintainer |
| `docs/rulings/` | the lead rulings AMBIGUITY-003 to 023 | maintainer |
| `runs/G*_record.md`, `runs/P3_record.md` | the gate records | maintainer |

---

## Setup

Python 3.12. Dependencies and tooling are managed with [uv](https://docs.astral.sh/uv/); the
`Makefile` targets are thin wrappers so that CI and a human run the same commands.

```
uv sync          # create .venv and install the pinned dependency set
make setup       # uv sync, plus the pre-commit hook
make test        # the frozen suites: tests/unit, tests/behavioural, tests/golden
make lint        # ruff check and ruff format --check
make format      # ruff check --fix and ruff format
make spec-check  # spec/spec.py imports and exposes every PLAN §10 public symbol
```

`make test` runs **only** the frozen suites. `tests/acceptance/` holds the maintainer-run gate experiments
and is never collected: not by CI, not by a contributor, not on any task's must-pass
list (CONTRACT rule 13). `make gate` exists and deliberately refuses - a gate is run by the maintainer, on
purpose, and writes its artefacts under `runs/` with the manifest of CONTRACT rule 10.

The golden-parity tests need the generated trajectories: run `make golden` first, or those
tests skip. A green suite means the implementation matches the reference and the stated
invariants. It is never evidence about the phenomena, which are measured by the experiments
under `runs/`.

Lint and line length: `ruff`, `line-length = 100`, `target-version = py312`.

## Test architecture

| Category | Location | Written by | Frozen | In CI | Purpose |
|---|---|---|---|---|---|
| Unit / property | `tests/unit` | maintainer | yes | yes | conservation, monotonicity, invariances, bounds, scale |
| Behavioural | `tests/behavioural` | maintainer | yes | yes | heuristic-agent dynamics; no-hard-coded-pathology; information invariants |
| Golden | `tests/golden` | generated from `ref/` | yes | yes | implementation == reference to 1e-9 on seeded trajectories |
| Acceptance | `tests/acceptance` | maintainer | n/a | **no** | gates G0-G4; experiments, not tests |

Test ids referenced throughout the docstrings are `T-U#` (unit / property) and `T-B#` (behavioural),
enumerated in PLAN §11. The first three categories are read-only for contributors: if a test looks
wrong, file an OPEN QUESTION (CONTRACT rules 2 and 3).

---

## Gate sequence

Work proceeds through five gates. A gate is a written sign-off on named artefacts, not a vibe.

| Gate | After | Pass condition | Artefacts | Sign-off |
|---|---|---|---|---|
| **G0** | `mc_sanity` | Full frozen suite green; MC sanity report clean; maintainer's diff review of `env/` against CONTRACT rule 7 | `runs/mc_sanity/report.md` | maintainer |
| **G1** | `regime_map` | Regime map produced; human selects the P1 provisional values from the interior of the bunching region; three `a·pen` levels and the `b̂_DP` thresholds recorded **before** any training | `runs/G1_decision.md`, `spec` v1.0.0 | Human |
| **G2** | `dp_vs_ppo`, `phase1_gate` | PLAN §4.5 criteria 1-4 | `runs/dp_vs_ppo/report.md`, `runs/phase1_gate/report.md` | Human + maintainer |
| **G3** | `exploitability`, phase-2 acceptance | Held-out phenomena 2, 5, 6, 7 evaluated on the PLAN §4.2 values (pass or reported failure); exploitability below threshold on all arms used; oracle gap recorded; JAX parity | `runs/phase2_acceptance/report.md` | Human + maintainer |
| **G4** | `contrasts`, `estimator_bias`, `llm_study` | Contrasts with CIs; estimator-bias curves; LLM study; price sensitivity on every headline table | final report | Human |

**A gate that fails produces a written failure report.** The next task is then a lead
diagnosis - never a parameter change made in order to pass the gate. Parameter changes after G1
create a new, labelled study with its own pre-registration. (PLAN §13.)

The build order between gates is the DAG of PLAN §12.2: parameter sourcing, the spec v0,
contract and registry, and the reference dynamics and frozen tests come first; the
environment modules, step function, heuristics, ledger and MC sanity harness reach G0; the spec
freeze, DP and regime map reach G1; metrics, the PPO adapter, the training harness and the two
Phase-1 experiments reach G2.

---

## The contract

[`CONTRACT.md`](CONTRACT.md) is PLAN §9 reproduced verbatim and binds every session, human or model.
Read it before writing a line. Its rules cover, in order: the frozen spec, the frozen tests, the
duty to stop and report rather than invent, the exhaustive list of reward terms, planner blindness,
welfare blindness, the prohibition on hard-coded pathology, bounds as results, RNG discipline, the
run manifest, parameter-arm classification, task scope, and the separation of tests from
experiments.

Violations invalidate the session's output. The shape was fixed first, in writing, so that the filling-in
could not quietly redefine the question.
