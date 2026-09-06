# gosplan-env

A multi-agent reinforcement-learning environment in which learning **enterprises** face a fixed
**rule-based planner**: the planner sets targets, allocates inputs from reported claims, audits a
random sample of reports, and ratchets next period's targets on the reported fulfilment ratio.
Enterprises choose effort and what to report. Nothing else is scripted.

The repository exists to support two claims that are stated, tested and reported **separately**
(PLAN §1.1), and to make it obvious which one a given artefact does or does not bear on.

---

## THIS REPOSITORY IS A SKELETON

Every function and method body in `gosplan/`, `spec/`, `ref/` and `tests/` is exactly

```python
raise NotImplementedError("<PLAN section> - implemented in WO-###")
```

There is no logic anywhere. The only real content is prose and declarations: the markdown
documents, `CONTRACT.md`, the parameter registry data in `gosplan/params.py`, the work-order cards
in `workorders/`, packaging and lint configuration, type aliases, enums, `Protocol` definitions,
and dataclass field declarations carrying the Phase-1 defaults of PLAN §3.

Bodies are filled in one at a time by the work orders in `workorders/`, in the DAG order of
PLAN §12.2, under the rules of [`CONTRACT.md`](CONTRACT.md). Do not "just implement" something
because it looks small: an implementer writes only the files its card names, and files an
AMBIGUITY REPORT instead of inventing (CONTRACT rules 3 and 12).

## NO RESULT EXISTS YET

Nothing in this repository has been run. There are no trained agents, no runs, no figures, no
estimates, and no findings. `runs/` is empty. Gates G0-G4 are all unsigned.

This is deliberate, not incidental. PLAN §4 pre-registers the phenomena, their operationalisations,
the estimator settings and the Phase-1 acceptance criteria **before** any learning run, and PLAN §4.2
locks the mechanism parameters behind the held-out phenomena now. Four phenomena (storming excess,
hoarding to shortage, blat, hidden reserves) are held out: no plot, table or test of them is
produced before the Phase-2 acceptance run.

Consequently: every number visible in this repository today is either a provisional parameter
default from PLAN §3 (those marked with a dagger are replaced at gate G1) or a template placeholder
reading `TBD`. **No number here may be cited as a result, an estimate, or a historical fact.**

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

The coupling to `forensic-stats` / `forensics_core` is scoped to **estimator robustness**
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
      step.py                 # LEAD: period schedule §2.5, assembles modules
      env.py                  # reset/step wrapper, specs, info/ledger hookup
    agents/
      base.py                 # Agent protocol
      heuristic.py            # Random, TruthfulMyopic, Padder, DPGreedy (P2: Berliner, Weitzman, Kornai)
      dp.py                   # §5
      ppo/adapter.py          # LEAD: thin adapter over reference PPO
      ppo/train.py            # training harness, checkpoints, eval
      llm_ministry.py         # P2
    oracle/kantorovich.py     # P2
    metrics/
      ledger.py               # StepRecord, Ledger, run manifest
      phenomena.py            # §4.1 operationalisations
      _fallback.py            # vendored estimator signatures (§7.3)
    experiments/
      mc_sanity.py            # WO-012
      regime_map.py           # WO-015
      dp_vs_ppo.py            # WO-019
      phase1_gate.py          # WO-020
      exploitability.py       # P2
      contrasts.py            # P3
      sobol.py                # P3 optional
      estimator_bias.py       # P3
      llm_study.py            # P3
      price_sensitivity.py    # P3
    jax/                      # LEAD, P2: port + parity
  ref/
    ref_step.py               # LEAD: slow pure-Python reference dynamics — the test oracle
    gen_golden.py             # generates tests/golden/*.json from ref
  tests/
    unit/                     # property and conservation tests (frozen)
    behavioural/              # heuristic-agent behavioural tests incl. no-hardcoded-pathology (frozen)
    golden/                   # generated from ref (frozen)
    acceptance/               # lead-run experiments; NOT in CI; not "tests" in the contract sense
  workorders/
    WO-000.md … WO-037.md
    TEMPLATE.md
    AMBIGUITY_TEMPLATE.md
  runs/                       # manifests + results, one directory per run hash
```

(PLAN §8. Directories exist in the skeleton even where every file in them is still a stub.)

---

## PLAN.md is the build instruction

`PLAN.md` ("gosplan-env - Build Plan v1.0") is the authoritative design document. Everything in this
repository is derived from it, and every module docstring, work-order card and test cites the PLAN
section it realises.

- It is **deliberately git-ignored** (see `.gitignore`) and is never committed. Do not un-ignore it,
  do not vendor it, do not copy large prose blocks of it into code or docs. Cite section numbers.
- It **must be present locally at the repository root** for any work order to be executed. Without
  it, a session cannot check its work against the spec, and the correct action is to stop rather
  than to reconstruct the missing sections from the code.
- Where this README and `PLAN.md` disagree, `PLAN.md` wins - except for `CONTRACT.md`, which is
  PLAN §9 reproduced verbatim and is the binding text.

## Documents in this repository

| File | What it is | Owner |
|---|---|---|
| `CONTRACT.md` | PLAN §9 verbatim: the 13 rules that bind every session, human or model | LEAD (WO-001) |
| `README.md` | this file | LEAD |
| `spec/CHANGELOG.md` | every change to `spec/spec.py` after the v1 freeze: version, reason, affected work orders, approver | LEAD |
| `docs/params_sources.md` | WO-000 deliverable: sourced range or explicit prior for every provisional parameter | LEAD |
| `docs/ref_worked_example.md` | WO-002 deliverable: the hand-checked 2-enterprise, 2-sector validation of `ref/ref_step.py` | LEAD |
| `workorders/` | one card per unit of work, plus `TEMPLATE.md` and `AMBIGUITY_TEMPLATE.md` (PLAN §12.1) | LEAD |

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

`make test` runs **only** the frozen suites. `tests/acceptance/` holds the lead-run gate experiments
and is never collected: not by CI, not by an implementer session, not on any work order's must-pass
list (CONTRACT rule 13). `make gate` exists and deliberately refuses - a gate is run by the lead, on
purpose, and writes its artefacts under `runs/` with the manifest of CONTRACT rule 10.

While the repository is a skeleton, the only tests present are stubs that assert the interface
surface exists; they make no behavioural claim, and any path into real dynamics stops at
`NotImplementedError`. A green suite today means the *shape* is right - it is never evidence that
anything works.

Lint and line length: `ruff`, `line-length = 100`, `target-version = py312`.

## Test architecture

| Category | Location | Written by | Frozen | In CI | Purpose |
|---|---|---|---|---|---|
| Unit / property | `tests/unit` | LEAD | yes | yes | conservation, monotonicity, invariances, bounds, scale |
| Behavioural | `tests/behavioural` | LEAD | yes | yes | heuristic-agent dynamics; no-hard-coded-pathology; information invariants |
| Golden | `tests/golden` | generated from `ref/` | yes | yes | implementation == reference to 1e-9 on seeded trajectories |
| Acceptance | `tests/acceptance` | LEAD | n/a | **no** | gates G0-G4; experiments, not tests |

Test ids referenced throughout the docstrings are `T-U#` (unit / property) and `T-B#` (behavioural),
enumerated in PLAN §11. The first three categories are read-only for implementers: if a test looks
wrong, file an AMBIGUITY REPORT (CONTRACT rules 2 and 3).

---

## Gate sequence

Work proceeds through five gates. A gate is a written sign-off on named artefacts, not a vibe.

| Gate | After | Pass condition | Artefacts | Sign-off |
|---|---|---|---|---|
| **G0** | WO-012 | Full frozen suite green; MC sanity report clean; lead's diff review of `env/` against CONTRACT rule 7 | `runs/mc_sanity/report.md` | LEAD |
| **G1** | WO-015 | Regime map produced; human selects the P1 provisional values from the interior of the bunching region; three `a·pen` levels and the `b̂_DP` thresholds recorded **before** any training | `runs/G1_decision.md`, `spec` v1.0.0 | Human |
| **G2** | WO-020 | PLAN §4.5 criteria 1-4 | `runs/dp_vs_ppo/report.md`, `runs/phase1_gate/report.md` | Human + LEAD |
| **G3** | WO-031 | Held-out phenomena 2, 5, 6, 7 evaluated on the PLAN §4.2 values (pass or reported failure); exploitability below threshold on all arms used; oracle gap recorded; JAX parity | `runs/phase2_acceptance/report.md` | Human + LEAD |
| **G4** | WO-037 | Contrasts with CIs; estimator-bias curves; LLM study; price sensitivity on every headline table | final report | Human |

**A gate that fails produces a written failure report.** The next work order is then a lead
diagnosis - never a parameter change made in order to pass the gate. Parameter changes after G1
create a new, labelled study with its own pre-registration. (PLAN §13.)

The build order between gates is the DAG of PLAN §12.2: WO-000 (parameter sourcing) and WO-001
(spec v0, contract, registry) and WO-002 (reference dynamics and frozen tests) come first; the
environment modules, step function, heuristics, ledger and MC sanity harness reach G0; the spec
freeze, DP and regime map reach G1; metrics, the PPO adapter, the training harness and the two
Phase-1 experiments reach G2.

---

## The contract

[`CONTRACT.md`](CONTRACT.md) is PLAN §9 reproduced verbatim and binds every session, human or model.
Read it before writing a line. Its rules cover, in order: the frozen spec, the frozen tests, the
duty to stop and report rather than invent, the exhaustive list of reward terms, planner blindness,
welfare blindness, the prohibition on hard-coded pathology, bounds as results, RNG discipline, the
run manifest, parameter-arm classification, work-order scope, and the separation of tests from
experiments.

Violations invalidate the session's output. That is the whole point of a skeleton: the shape is
fixed first, in writing, so that the filling-in cannot quietly redefine the question.
