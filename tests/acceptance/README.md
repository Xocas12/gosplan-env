# `tests/acceptance/` — the gates. **These are experiments, not tests.**

## CONTRACT rule 13, verbatim

> **13. TESTS ARE NOT EXPERIMENTS.** `tests/acceptance/` holds lead-run experiments (gates).
> Nothing there is a unit test, nothing there is on any work order's must-pass list, and no
> implementer session runs it.

That rule is why nothing in this directory is named `test_*.py`. Every file here is `gate_*.py`, so
pytest's default collection never picks it up, and `pyproject.toml` excludes the directory a second
time by naming only the three frozen categories in `testpaths`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests/unit", "tests/behavioural", "tests/golden"]
```

Two independent guards, on purpose. A gate is a multi-hour, multi-seed experiment whose result is a
*finding*; a test is a fast, deterministic assertion whose result is *green or a bug*. Collecting a
gate into CI would turn a finding into a build failure and create exactly the pressure this design
exists to remove — the pressure to move a parameter until the build goes green. `make gate` refuses
to run them for the same reason, and prints why.

Each `gate_*.py` here is a **harness stub**: a module docstring naming the gate, its pass condition
from PLAN section 13, its artefacts, its sign-off, and a `main()` that raises `NotImplementedError`.
The experiments the gates *drive* live in `gosplan/experiments/` and are implemented by their own
work orders; a gate module is the lead's entry point, the place the run is parameterised and its
artefacts collected, and it is never on a work order's must-pass list.

## The gates (PLAN section 13)

| Gate | After | Pass condition | Artefacts | Sign-off |
|---|---|---|---|---|
| **G0** | WO-012 | Full frozen suite green; MC sanity report clean; lead's diff review of `env/` against CONTRACT rule 7 | `runs/mc_sanity/report.md` | LEAD |
| **G1** | WO-015 | Regime map produced; human selects the Phase-1 daggered values from the *interior* of the bunching region; three `a·pen` levels and the `b̂_DP` thresholds recorded **before** any training | `runs/G1_decision.md`, `spec` v1.0.0 | Human |
| **G2** | WO-020 | PLAN section 4.5 criteria 1–4 | `runs/dp_vs_ppo/report.md`, `runs/phase1_gate/report.md` | Human + LEAD |
| **G3** | WO-031 | Held-out phenomena 2, 5, 6, 7 evaluated on the PLAN section 4.2 values (pass or reported failure); exploitability below threshold on all arms used; oracle gap recorded; JAX parity | `runs/phase2_acceptance/report.md` | Human + LEAD |
| **G4** | WO-037 | Contrasts with CIs; estimator-bias curves; LLM study; price sensitivity on every headline table | final report | Human |

| File | Gate | Drives |
|---|---|---|
| `gate_g0_mc_sanity.py` | G0 | `gosplan.experiments.mc_sanity` (WO-012) + the lead's `env/` diff review |
| `gate_g1_regime_map.py` | G1 | `gosplan.experiments.regime_map` (WO-015) + the human's `runs/G1_decision.md` |
| `gate_g2_phase1.py` | G2 | `gosplan.experiments.dp_vs_ppo` (WO-019), `gosplan.experiments.phase1_gate` (WO-020) |
| `gate_g3_phase2.py` | G3 | the WO-031 Phase-2 acceptance harness, `gosplan.experiments.exploitability` (WO-028), the oracle (WO-027), the JAX parity check (WO-029) |
| `gate_g4_final.py` | G4 | `gosplan.experiments.contrasts` (WO-032), `estimator_bias` (WO-034), `llm_study` (WO-035), `price_sensitivity` (WO-036), the WO-037 report roll-up |

## What a failed gate produces — PLAN section 13

> A gate that fails produces a written failure report; the next work order is a **lead diagnosis**,
> never a parameter change to make the gate pass. **Parameter changes after G1 create a new,
> labelled study.**

Read that as three separate prohibitions, because they fail in three different ways:

1. **A failed gate is written down.** The failure report is an artefact under `runs/`, with the
   configuration hash, the seeds, the manifest of CONTRACT rule 10 and the criterion that failed. A
   gate that is re-run until it passes, with only the passing run recorded, has produced no
   evidence at all.
2. **The successor is a diagnosis, not a tune.** The next work order asks *why* the criterion
   failed. It is written by the lead. It may not be "set `penalty_scale` to 90 and re-run".
3. **After G1 the parameters are fixed.** The Phase-1 values of the daggered PLAN section 3 rows
   are chosen once, by a human, from the interior of the DP regime map, and recorded in
   `runs/G1_decision.md`. Changing one afterwards does not amend the study — it *starts a new one*,
   with its own label, its own pre-registration and its own gates, reported separately. Rerunning
   the old study's numbers under new parameters and reporting them as the old study is the specific
   fraud this rule prevents.

PLAN section 4.5 adds the reading that matters most at G2: failure of criterion 1 (DP recovery) is a
training-stack failure and blocks everything, while **failure of criterion 2 with criterion 1
passing is a multi-agent effect and is a RESULT** — reported, with its CIs and seed counts, and
never tuned away.

## G1 is human-first, and the order is load-bearing

`gate_g1_regime_map.py` produces the regime map. It does **not** choose the Phase-1 values.

The human reads the map, selects the daggered values (`ratchet_lambda`, `growth_directive`,
`overfulfilment_slope`, `penalty_scale`, `effort_cost`, `audit_rate`) from the *interior* of the
bunching region, fixes the three `a·pen` levels for G2 criterion 1, records the `b̂_DP` thresholds,
and writes all of it into `runs/G1_decision.md` — **before any training run exists**. Only then does
`spec/spec.py` go to v1.0.0 (WO-013, CONTRACT rule 1) and only then may WO-018 train anything.

The order is the pre-registration. Values picked after seeing a training result are values chosen to
produce that result, and no amount of later reporting repairs it. `gosplan/experiments/dp_vs_ppo.py`
and `gosplan/experiments/phase1_gate.py` therefore *read* `runs/G1_decision.md` and never re-derive
its numbers.

## Held-out phenomena — PLAN sections 4.1 and 4.2

Rows 2 (storming), 5 (hoarding), 6 (blat) and 7 (hidden reserves) of PLAN section 4.1 are
**emergence** claims and are held out: no plot, table or test of them is produced before the Phase-2
acceptance run — not during Phase 1, and not while debugging their mechanisms. G0's Monte-Carlo
harness may assert conservation and boundedness on the same mechanisms, never a direction. Their
mechanism parameters are locked now in PLAN section 4.2 and may be changed only in a new, labelled
study. `gate_g3_phase2.py` is the first place those four are computed at all, and a phenomenon that
fails to appear there is **reported as a failure**, not investigated until it appears.

## Running one

A gate is run deliberately, by the lead, from the repository root:

```sh
python -m tests.acceptance.gate_g0_mc_sanity     # after WO-012, before G1
```

`make gate` refuses and explains; there is no target that runs a gate for you. Every run writes its
artefacts under `runs/` with the manifest of CONTRACT rule 10 (config hash, spec version, git hash,
seeds, reference-PPO version, estimator version, LLM model ids, solver version and optimality gap,
flags). A gate's exit code reports whether the *experiment* completed, never whether the criterion
passed: pass and fail are lines in the report, signed off by the people named in the table above.
