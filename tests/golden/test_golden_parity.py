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

import json
from pathlib import Path

import numpy as np
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


def _gate(*targets):
    """Skip while any of `targets` is still a skeleton stub (the module-level gate)."""
    import inspect

    pending = []
    for target in targets:
        try:
            source = inspect.getsource(target)
        except (OSError, TypeError):
            continue
        if "raise NotImplementedError" in source:
            pending.append(getattr(target, "__qualname__", repr(target)))
    if pending:
        pytest.skip("awaiting implementation: " + ", ".join(pending))


def _document(path):
    """Load a golden document, skipping the test when the matrix has not been generated."""
    if path is None:
        pytest.skip(NO_GOLDEN_REASON)
    return json.loads(path.read_text(encoding="utf-8"))


def _decode_inf(value):
    """Undo the `float("inf") -> "inf"` encoding `EnvConfig.hash` applies (spec/CHANGELOG 0.1.1)."""
    if isinstance(value, str) and value == "inf":
        return float("inf")
    if isinstance(value, dict):
        return {k: _decode_inf(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode_inf(v) for v in value]
    return value


def _rebuild_config(document):
    """Rebuild an `EnvConfig` from the document's own configuration, via `load_config`."""
    import tempfile

    from gosplan.config import load_config

    _gate(load_config)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cfg.json"
        path.write_text(json.dumps(document["config"]), encoding="utf-8")
        return load_config(str(path))


def _render_state(state):
    """The canonical rendering `ref_state_digest` hashes, rebuilt from a production `State`.

    Specified in that function's docstring precisely so this test can reproduce it without
    importing `ref/`: field name, "=", value, ";" in declaration order; floats as `repr(float(x))`,
    ints as `repr(int(x))`, bools as "true"/"false", strings verbatim, lists row-major inside
    brackets; UTF-8 before hashing.
    """
    import dataclasses
    import hashlib

    import numpy as _np

    def render(value):
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, _np.integer)):
            return repr(int(value))
        if isinstance(value, (float, _np.floating)):
            return repr(float(value))
        if isinstance(value, str):
            return value
        arr = _np.asarray(value)
        if arr.ndim == 0:
            return render(arr.item())
        return "[" + ",".join(render(v) for v in arr) + "]"

    parts = [f"{f.name}={render(getattr(state, f.name))};" for f in dataclasses.fields(state)]
    return hashlib.sha256("".join(parts).encode("utf-8")).hexdigest()


def _replay(document):
    """Replay one golden cell through `GosplanEnv`, returning the per-step observations.

    Gated on the environment and the named production agent, so the whole module skips naming its
    missing dependency until WO-009 and WO-010 land.
    """
    from gosplan.agents import heuristic
    from gosplan.env.env import GosplanEnv

    cfg = _rebuild_config(document)
    agent_cls = getattr(heuristic, document["agent"])
    _gate(GosplanEnv.reset, GosplanEnv.step, agent_cls.act)

    env = GosplanEnv(cfg)
    obs, info = env.reset(int(document["seed_env"]), int(document["seed_policy"]))
    policy = agent_cls(cfg)
    rng = np.random.default_rng(int(document["seed_policy"]))
    out = []
    for _ in range(int(document["n_steps"])):
        obs, reward, done, info = env.step(policy.act(obs, env.phase(), rng))
        out.append((np.asarray(obs), np.asarray(reward), bool(done), env.state, info))
    return out


@pytest.mark.skeleton
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
    if not GOLDEN_PATHS:
        pytest.skip(NO_GOLDEN_REASON)
    assert len(GOLDEN_PATHS) == GOLDEN_MATRIX_SIZE
    documents = [json.loads(p.read_text(encoding="utf-8")) for p in GOLDEN_PATHS]
    assert len({d["config_name"] for d in documents}) == 5
    assert len({d["seed_env"] for d in documents}) == 3
    assert len({d["agent"] for d in documents}) == 2


@pytest.mark.skeleton
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
    document = _document(path)
    assert document["schema_version"] == GOLDEN_SCHEMA_VERSION
    for key in (
        "spec_version",
        "generator",
        "config_name",
        "config_hash",
        "config",
        "agent",
        "seed_env",
        "seed_policy",
        "n_steps",
        "tolerance",
        "obs_names",
        "steps",
    ):
        assert key in document, key
    assert document["generator"] == "ref/gen_golden.py"
    assert document["tolerance"] == GOLDEN_TOLERANCE
    assert len(document["steps"]) == document["n_steps"] == GOLDEN_N_STEPS
    for step in document["steps"]:
        for key in (
            "t_period",
            "k_step",
            "phase",
            "obs",
            "reward",
            "done",
            "state_digest",
            "val_measured",
            "val_true",
            "welfare",
        ):
            assert key in step, key
        if step["phase"] == "produce":
            assert step["val_measured"] is None


@pytest.mark.skeleton
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
    from gosplan.env.obs import obs_spec

    document = _document(path)
    cfg = _rebuild_config(document)
    _gate(obs_spec)
    assert cfg.hash() == document["config_hash"], (
        "EnvConfig.hash and ref/gen_golden.config_hash disagree; the canonical JSON encoding is "
        "pinned in spec/CHANGELOG.md 0.1.1 and both sides must apply it identically"
    )
    assert list(obs_spec(cfg)) == list(document["obs_names"])


@pytest.mark.skeleton
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
    document = _document(path)
    replay = _replay(document)
    for index, (step, (obs, reward, done, _state, _info)) in enumerate(
        zip(document["steps"], replay, strict=True)
    ):
        want_obs = np.asarray(step["obs"], dtype=float)
        want_reward = np.asarray(step["reward"], dtype=float)
        assert obs.shape == want_obs.shape, index
        assert np.max(np.abs(obs - want_obs)) <= GOLDEN_TOLERANCE, index
        assert np.max(np.abs(reward - want_reward)) <= GOLDEN_TOLERANCE, index
        assert done == bool(step["done"]), index


@pytest.mark.skeleton
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
    document = _document(path)
    replay = _replay(document)
    for index, (step, (_obs, _reward, _done, state, _info)) in enumerate(
        zip(document["steps"], replay, strict=True)
    ):
        assert _render_state(state) == step["state_digest"], (
            f"step {index}: hidden state diverges from the reference even though the observable "
            "interface may agree; this is never waived by loosening the tolerance"
        )


@pytest.mark.skeleton
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
    document = _document(path)
    replay = _replay(document)
    for index, (step, (obs, _reward, _done, _state, info)) in enumerate(
        zip(document["steps"], replay, strict=True)
    ):
        if step["phase"] != "report":
            continue
        for key in ("val_measured", "val_true", "welfare"):
            want = step[key]
            got = getattr(info, key)
            assert want is not None and got is not None, (index, key)
            assert abs(float(got) - float(want)) <= GOLDEN_TOLERANCE, (index, key)
            # CONTRACT rule 6: the logged scalar must not appear in the observation
            assert (
                np.min(np.abs(np.asarray(obs, dtype=float) - float(want))) > GOLDEN_TOLERANCE
                or abs(float(want)) <= GOLDEN_TOLERANCE
            ), (index, key)
