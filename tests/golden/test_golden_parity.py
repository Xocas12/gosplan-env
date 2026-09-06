"""T-B7 - golden parity: `gosplan` reproduces `ref/` to 1e-9 on the pre-registered matrix.

Realises: PLAN section 11 (the golden category and behavioural test T-B7), read against PLAN
sections 2.5-2.11 (the dynamics the reference implements), 2.4 (the observation layout the files
carry), 2.9.1 (the reward the files carry), 2.15 (keyed RNG, which is what makes two independent
implementations agree at all) and 3 (the configuration hash). Owning work order: **WO-002** (the
LEAD writes `ref/ref_step.py`, `ref/gen_golden.py` and this test); on the must-pass list of
**WO-009** (`gosplan/env/step.py`, `env.py`, `state.py`).

What T-B7 asserts: for every golden document under `tests/golden/*.json`, replaying the same
configuration, seeds, policy and number of agent-steps through `gosplan.env.env.GosplanEnv`
reproduces the reference's observations and rewards to `GOLDEN_TOLERANCE` and its state digest
exactly. The files are the oracle; hand-computed expectations are what finding F14 rules out.

    THE FILES ARE GENERATED, AND THIS TEST SKIPS WHEN THEY ARE ABSENT.

    `tests/golden/*.json` is git-ignored (see `.gitignore`) and produced by `make golden`, i.e.
    `python -m ref.gen_golden`. A fresh checkout therefore has none, and a missing golden set is a
    *missing input*, not a failing implementation: this module skips, loudly, with
    `NO_GOLDEN_REASON`, which names the command that produces them. The skip plumbing below -
    discovery, the module-level `skipif`, the parametrisation ids - is written for real, because it
    is test plumbing rather than environment logic. Everything downstream of a loaded document is
    supplied by WO-002.

Frozen (CONTRACT rule 2). `tests/unit`, `tests/behavioural` and `tests/golden` are read-only for
implementers: an implementer never edits this file, never edits or deletes a golden document, and
never regenerates one to make a failure go away. If a golden file looks wrong, the response is an
AMBIGUITY REPORT (CONTRACT rule 3). Regeneration is a lead action at **WO-013** with a
`spec/CHANGELOG.md` entry (CONTRACT rule 1) - see `README.md` beside this file.

The matrix (PLAN section 11, fixed): 5 configurations x 3 seeds x 2 policies (`Random`,
`TruthfulMyopic`) x 30 agent-steps = `GOLDEN_MATRIX_SIZE` documents. A partial set is not a golden
set, which is why `gen_golden.py` refuses a filtered run without `--force` and why
`test_golden_matrix_is_complete` asserts the count.

Document layout: `GOLDEN_SCHEMA` in `ref/gen_golden.py` defines it, but this test reads a file
without consulting that module - the file is self-contained by design, so the two can never drift
into agreement by sharing code.
"""

from __future__ import annotations

from pathlib import Path

import pytest

GOLDEN_DIR = Path(__file__).resolve().parent
"""The directory this test lives in; `ref/gen_golden.py` writes its documents here
(`GOLDEN_DIR` there resolves to the same path)."""

GOLDEN_GLOB = "*.json"
"""Every golden document, whatever its cell name. The parity test parses nothing out of a file
name: the name is for humans, and every field the test needs is inside the document."""

GOLDEN_TOLERANCE = 1e-9
"""Tolerance on observations and rewards (PLAN section 11: "implementation == reference to 1e-9").
The state digest is compared for exact string equality, not to this tolerance."""

GOLDEN_SCHEMA_VERSION = "1"
"""The only document layout this test understands. A file carrying anything else is refused rather
than guessed at, because a missing key would otherwise read as a passing comparison."""

GOLDEN_MATRIX_SIZE = 30
"""5 configurations x 3 seeds x 2 policies (PLAN section 11). Asserted, so a partially generated
set fails loudly instead of silently testing five cells."""

GOLDEN_N_STEPS = 30
"""Agent-steps per rollout (PLAN section 11). Read back from each document and asserted, so a
`--force --steps` generation cannot be mistaken for the pre-registered matrix."""

REGENERATION_COMMAND = "make golden  (equivalently: python -m ref.gen_golden)"
"""How the files are produced. Named in every skip message so a reader never has to look it up."""


def golden_paths() -> list[Path]:
    """Return every golden document present, sorted, or an empty list.

    Takes: nothing. Returns: `sorted(GOLDEN_DIR.glob(GOLDEN_GLOB))` - deterministic order, so
    parametrisation ids are stable across machines and a failure names the same cell everywhere.

    Real plumbing, not environment logic: the module-level `skipif` below depends on it, and it is
    what makes an absent golden set a skip rather than a collection error.
    """
    return sorted(GOLDEN_DIR.glob(GOLDEN_GLOB))


GOLDEN_PATHS = golden_paths()
"""The documents found at import time."""

NO_GOLDEN_REASON = (
    "no golden files in tests/golden/: they are generated from ref/ref_step.py, not committed "
    "(tests/golden/*.json is git-ignored). Generate them with `" + REGENERATION_COMMAND + "`, "
    "then re-run. T-B7 is not evidence of anything until they exist - see tests/golden/README.md."
)
"""The skip message. It says what is missing, why it is missing, how to produce it, and what the
absence does and does not prove."""

pytestmark = pytest.mark.skipif(not GOLDEN_PATHS, reason=NO_GOLDEN_REASON)
"""Module-level skip: with no documents on disk every test below is reported as skipped with
`NO_GOLDEN_REASON`, rather than failing, erroring at collection, or silently collecting nothing."""

GOLDEN_PARAMS = GOLDEN_PATHS or [None]
"""Parametrisation argument. The `or [None]` keeps the parametrised tests collectable when the
directory is empty - they are skipped by `pytestmark` before the argument is ever used, and an
empty parametrisation would report "no tests ran" instead of a skip with a reason."""

GOLDEN_IDS = [p.name for p in GOLDEN_PATHS] or ["<no-golden-files>"]
"""Test ids: the document's file name, so a failure names its cell - configuration, policy and
seed - without the reader opening anything."""

SKIP_REASON = (
    "skeleton: T-B7 comparisons are written by WO-002 (frozen tests); they bind WO-009 "
    "(GosplanEnv.step) against the reference dynamics of ref/ref_step.py"
)
"""Reason attached to every `@pytest.mark.skip` below; the bodies arrive with WO-002. It is
independent of `NO_GOLDEN_REASON`, which is about the *inputs* being absent."""


@pytest.mark.skeleton
@pytest.mark.skip(reason=SKIP_REASON)
def test_golden_matrix_is_complete() -> None:
    """The generated set is the pre-registered matrix: `GOLDEN_MATRIX_SIZE` documents, no more.

    Assert `len(GOLDEN_PATHS) == GOLDEN_MATRIX_SIZE`, and that the documents cover exactly five
    distinct `config_name` values, three distinct `seed_env` values and the two policy names
    `Random` and `TruthfulMyopic`, with one document per (configuration, seed, policy) cell and no
    duplicates. Assert every document's `n_steps` equals `GOLDEN_N_STEPS`.

    PLAN section 11 fixes the matrix; `ref/gen_golden.py` refuses a filtered or re-sized generation
    without `--force`. A set that is not the matrix is not a golden set, and this test is what
    stops a `--config`-filtered regeneration from quietly reducing the suite's coverage. Owning WO:
    **WO-002**.
    """
    raise NotImplementedError("PLAN section 11 (T-B7) - implemented in WO-002")


@pytest.mark.skeleton
@pytest.mark.skip(reason=SKIP_REASON)
@pytest.mark.parametrize("path", GOLDEN_PARAMS, ids=GOLDEN_IDS)
def test_golden_document_is_readable_and_current(path: Path) -> None:
    """Every document has the schema this test understands and the spec version in force.

    Load `path` with `json.loads`. Assert `document["schema_version"] == GOLDEN_SCHEMA_VERSION`,
    that every key of the layout is present (including the `null`-valued period scalars on PRODUCE
    steps - a missing key is a bug, never an ambiguity), that `document["generator"]` names
    `ref/gen_golden.py`, and that `document["tolerance"] == GOLDEN_TOLERANCE`.

    Then compare `document["spec_version"]` with `spec.SPEC_VERSION`. If they differ, call
    `pytest.skip` with a message naming both versions and stating that a stale golden set is
    regenerated by the lead at **WO-013** with a `spec/CHANGELOG.md` entry (CONTRACT rule 1) and is
    never silently accepted - a *loud skip*, exactly as `ref/gen_golden.read_spec_version`'s
    docstring specifies, because a comparison against an older interface proves nothing either way.

    Owning WO: **WO-002**.
    """
    raise NotImplementedError("PLAN section 11 (T-B7) - implemented in WO-002")


@pytest.mark.skeleton
@pytest.mark.skip(reason=SKIP_REASON)
@pytest.mark.parametrize("path", GOLDEN_PARAMS, ids=GOLDEN_IDS)
def test_golden_config_hash_and_obs_layout_match(path: Path) -> None:
    """The configuration rebuilds, hashes identically, and lays the observation out the same way.

    Rebuild the configuration from `document["config"]` with `load_config` (decoding the string
    `"inf"` back to `float("inf")`, the inverse of `EnvConfig.hash`'s encoding) and assert:

        cfg.hash() == document["config_hash"]
        obs_spec(cfg) == document["obs_names"]

    The first links a golden file to a production run of the same configuration, and cross-checks
    the independent hash implementation in `ref/gen_golden.config_hash` against `EnvConfig.hash`
    (PLAN section 3). The second is a schema check that must pass **before any number is compared**:
    if the observation layout moved, comparing entry `d` against entry `d` compares two different
    quantities, and the resulting failure would point at the dynamics instead of the layout.

    Owning WO: **WO-002**; binds **WO-003** and **WO-008**.
    """
    raise NotImplementedError("PLAN section 11 (T-B7) - implemented in WO-002")


@pytest.mark.skeleton
@pytest.mark.skip(reason=SKIP_REASON)
@pytest.mark.parametrize("path", GOLDEN_PARAMS, ids=GOLDEN_IDS)
def test_observations_and_rewards_match_reference(path: Path) -> None:
    """Replaying the cell through `GosplanEnv` reproduces the reference to 1e-9.

    Build `GosplanEnv(cfg)`, `reset(document["seed_env"], document["seed_policy"])`, and drive it
    for `document["n_steps"]` agent-steps with the production agent named by `document["agent"]`
    (`Random` or `TruthfulMyopic` from `gosplan.agents.heuristic`), taking policy randomness from a
    `numpy.random.Generator` seeded with `seed_policy` and drawing in the dimension order
    `ref/gen_golden.ref_random_policy` documents - effort, then report ratio, then input request,
    one call per dimension over all enterprises - because that ordering is part of the golden
    contract.

    At each step `n`, assert against `document["steps"][n]`:

        abs(obs - expected_obs)       <= GOLDEN_TOLERANCE      elementwise, shape (N, d)
        abs(reward - expected_reward) <= GOLDEN_TOLERANCE      elementwise, shape (N,)
        done == expected_done
        t_period, k_step, phase all equal

    Report the first mismatching (step, enterprise, component) with both values and their
    difference: the whole point of storing full observations and rewards is that a mismatch
    localises. Owning WO: **WO-002**; binds **WO-009**.
    """
    raise NotImplementedError("PLAN section 11 (T-B7) - implemented in WO-002")


@pytest.mark.skeleton
@pytest.mark.skip(reason=SKIP_REASON)
@pytest.mark.parametrize("path", GOLDEN_PARAMS, ids=GOLDEN_IDS)
def test_state_digest_matches_reference(path: Path) -> None:
    """The hidden state agrees exactly, not merely the observable interface.

    After each replayed step, rebuild the reference's canonical rendering from the production
    `State` - fields in declaration order, `name=value;`, floats as `repr(float(x))`, ints as
    `repr(int(x))`, bools as `"true"`/`"false"`, lists comma-separated inside brackets and
    row-major for nested arrays, UTF-8 encoded, SHA-256 hex - exactly as
    `ref.ref_step.ref_state_digest` documents it, and assert string equality with
    `document["steps"][n]["state_digest"]`.

    Compare the digest **only after** the observation and reward assertions of the previous test
    have passed for that cell: a digest mismatch with matching observations is a hidden-state
    divergence - stock, input stocks or pending investment - and is reported as such, never waived.
    Any change to the rendering invalidates every golden file and is a regeneration under WO-013
    with a `spec/CHANGELOG.md` entry. Owning WO: **WO-002**; binds **WO-009**.
    """
    raise NotImplementedError("PLAN section 11 (T-B7) - implemented in WO-002")


@pytest.mark.skeleton
@pytest.mark.skip(reason=SKIP_REASON)
@pytest.mark.parametrize("path", GOLDEN_PARAMS, ids=GOLDEN_IDS)
def test_logged_period_scalars_match_reference(path: Path) -> None:
    """`val_measured`, `val_true` and `welfare` agree at every REPORT step - logged, never observed.

    At each REPORT step assert `abs(value - expected) <= GOLDEN_TOLERANCE` for the three period
    scalars carried by `StepInfo` (PLAN section 2.9.3), and assert they are `None` in the document
    and absent from `StepInfo`'s period fields at PRODUCE steps, where the reference writes JSON
    `null` rather than omitting the key.

    These three exist in the golden files **for parity only**. CONTRACT rule 6 keeps them out of
    every observation, reward and agent input, which T-B5 asserts with sentinels; checking them
    here catches a welfare or `val` bug that never reaches an agent and so could otherwise sit
    undetected until a headline metric was computed from it (PLAN section 2.9.4). Owning WO:
    **WO-002**; binds **WO-007** and **WO-009**.
    """
    raise NotImplementedError("PLAN section 11 (T-B7) - implemented in WO-002")
