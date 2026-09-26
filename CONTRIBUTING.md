# Contributing to gosplan-env

This project asks whether Soviet-style reporting pathologies — padding, ratchet-avoidance,
storming — emerge from a plan-fulfilment incentive structure alone, with no pathological
behaviour written into the environment. That question is easy to answer by accident and hard
to answer honestly, so the process below exists to make the accidental answer difficult.

Read [`CONTRACT.md`](CONTRACT.md) first. It is thirteen rules and it is short.

## What is built

463 of 657 functions are implemented. `ref/ref_step.py` is a complete reference
implementation of the environment dynamics and it runs today: `make golden` rolls it forward
and writes 30 deterministic trajectories. What is not built is the vectorised environment in
`gosplan/env/`, the agents, the metrics and the experiment drivers.

Nothing has been run as an experiment. `runs/` is empty, gates G0 to G4 are unsigned, and no
number in this repository is a result. See the README for why that is deliberate.

## Setup

```sh
make setup     # uv sync + pre-commit
make test      # the frozen suites
make lint
make golden    # regenerate tests/golden/ from the reference implementation
```

`PLAN.md` is the authoritative design document. It is deliberately not committed, and it must
be present at the repository root before you can check your work against the specification.
Without it, stop.

## The rules that matter most

**Stop rather than guess.** When the plan, the tests and the contract do not determine a
choice, open an issue describing the choice and what turns on it, and stop. Picking "the
reasonable default" is the failure mode this project is organised against, and it is
dangerous precisely because the invented choice usually is reasonable.

**The tests are frozen.** `tests/unit`, `tests/behavioural` and `tests/golden` carry their
assertions already and skip while the symbol they name is a stub; they activate the moment it
lands. Do not edit a test to make code pass. If a test is genuinely wrong, say so in writing
first, and record in the commit message what it asserted, why that was wrong, and what
replaces it.

**Nothing pathological may be written into the environment.** The whole claim collapses if
padding is implemented rather than learned. `scripts/contract_guard.py` enforces this in CI,
along with nine of the other contract rules. Read
[`scripts/README.md`](scripts/README.md) before arguing with it.

**Four phenomena are held out.** Storming excess, hoarding-to-shortage, blat and hidden
reserves may not be plotted, tested, scored or summarised before the Phase-2 acceptance run.
The guard fails CI if a Phase-1 module so much as calls one of them. Looking is the violation.

**Bounds and failures are results.** A phenomenon that does not appear, reported as not
appearing, is a finding. Quietly retuning until it appears is not.

## Scope of a change

One task, one branch, one pull request, touching only the files that task needs. A change set
whose scope is declared before the work starts is reviewable; one that grows as it goes is
not. If you find an unrelated bug, open an issue for it rather than fixing it here.

## Gates

Work is organised into phases, and each gate is a checkpoint that must be signed before the
next phase starts. The gate conditions and their artefacts are in [`ROADMAP.md`](ROADMAP.md).

| Gate | What it establishes |
|---|---|
| G0 | The environment is implemented and behaves sanely |
| G1 | The regime map is produced and the free parameters are fixed **before** any training |
| G2 | Phase-1 acceptance: learned behaviour matches the exactly-solved optimum |
| G3 | Phase-2 acceptance: the held-out phenomena are evaluated, pass or fail |
| G4 | The final report |

The ordering is the point. G1 fixes the parameters before any learning run, so that the
result cannot be produced by tuning. A clean G2 still establishes nothing about emergence;
only G3 can speak to that.

## Pull requests

State what the change does, which files it touches and why, and what you checked. If you
stopped short of something because the specification did not determine it, say so and link
the issue. `make test` and `make lint` must pass.
