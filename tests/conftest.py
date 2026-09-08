"""Shared fixtures and marker registration for the frozen test suite (PLAN section 11).

Realises: PLAN section 11 (test architecture - the unit/property, behavioural and golden categories,
their owners and their frozen status) and PLAN section 12.3 (the "Must pass" line of every Phase-1
work order card). Owning work order: **WO-002** (reference dynamics and frozen tests; LEAD).

FROZEN BY CONTRACT RULE 2. `tests/unit`, `tests/behavioural` and `tests/golden` are read-only for
implementers. An implementer session may not edit a test, may not skip one, and may not
special-case an implementation to make one pass; a test that looks wrong is an AMBIGUITY REPORT
(`workorders/AMBIGUITY_TEMPLATE.md`, CONTRACT rule 3), never an edit. This file is part of that
frozen surface: the fixtures below are the configuration objects the frozen tests run at, so that
"the configuration this result was produced under" is a single auditable fact rather than a literal
repeated in twenty modules.

CONTRACT RULE 13 keeps `tests/acceptance/` out of this suite entirely: it holds the lead-run gate
experiments G0-G4 of PLAN section 13, nothing there is a unit test, and `pyproject.toml`'s
`testpaths` never collects it.

SKELETON STATUS. The repository is a skeleton. Every fixture body here raises `NotImplementedError`
naming the work order that fills it, and every test in `tests/unit` except `test_spec_imports.py`
carries `@pytest.mark.skeleton` plus `@pytest.mark.skip`, with a docstring stating the exact
assertion the eventual test must make. The skips lift work order by work order as PLAN section 12.3
lands the implementations; the assertions are already frozen by CONTRACT rule 2 and are not
renegotiable at the moment they start to run.
"""

from __future__ import annotations

import inspect

import pytest

SKELETON_MARKER = (
    "skeleton: stub test -- asserts only that the skeleton imports and exposes the PLAN "
    "section 10 surface; no behavioural expectation"
)
"""The `skeleton` marker description, verbatim from the `markers` entry of `pyproject.toml`. Held
here as data and re-registered by `pytest_configure` so the suite also collects under
`--strict-markers` when it is run with a different ini file (for example from a work order's
completion command)."""


def pytest_configure(config: pytest.Config) -> None:
    """Register the `skeleton` marker with pytest.

    Takes: `config`, the pytest configuration object of the running session. Returns: `None`; it
    appends `SKELETON_MARKER` to the session's `markers` ini lines.

    This is configuration data, not an implementation: it mirrors the `markers` entry that
    `pyproject.toml` already declares, and it must keep a real body, because a hook that raised
    would abort collection of the whole frozen suite. It is the one exception in this file to the
    skeleton rule that every body raises `NotImplementedError`.

    Binds: collection of `tests/unit`, `tests/behavioural` and `tests/golden` under
    `--strict-markers`; every skeleton stub in `tests/unit` carries this marker.
    """
    config.addinivalue_line("markers", SKELETON_MARKER)


def _is_stub(target: object) -> bool:
    """True when `target` is still a skeleton body that raises `NotImplementedError`.

    Takes: any callable or class. Returns: `True` if its source contains a
    `raise NotImplementedError`, `False` if it does not or if the source cannot be read.

    Source inspection rather than a call: calling the target would need arguments the caller does
    not have, and a stub that raised for a different reason would be indistinguishable.
    """
    try:
        source = inspect.getsource(target)
    except (OSError, TypeError):
        return False
    return "raise NotImplementedError" in source


def _require_implemented(*targets: object) -> None:
    """Skip the calling test while any of `targets` is still a skeleton stub.

    THIS IS THE MECHANISM THAT KEEPS CONTRACT RULE 2 AND A GREEN CI COMPATIBLE, and it replaces the
    hand-lifted `@pytest.mark.skip` markers the skeleton shipped with.

    The problem it solves. A frozen test asserts the behaviour of an implementation that does not
    exist yet, so it must not fail the build today; but rule 2 forbids an implementer from editing a
    frozen test, which is exactly what "lift the skip when your card lands" would require. A
    hand-lifted marker also means a card can be reported complete while its must-pass tests are
    still skipping - the failure mode where a green completion command proves nothing.

    With this helper the test carries its real assertion from the day it is written, skips itself
    for a stated reason while its dependency is a stub, and STARTS RUNNING THE INSTANT the
    implementing card lands - with no edit to any frozen file, by anyone.

    Takes: the callables or classes the test's assertion depends on. Returns: `None`, or raises
    `Skipped` naming every dependency that is still unimplemented.
    """
    pending = [t for t in targets if _is_stub(t)]
    if pending:
        names = ", ".join(getattr(t, "__qualname__", repr(t)) for t in pending)
        pytest.skip(f"awaiting implementation: {names}")


@pytest.fixture
def implemented():
    """Expose `_require_implemented` to a frozen test.

    Takes: nothing. Returns: the `_require_implemented` callable. A test calls it with the symbols
    its assertion exercises, as its first statement.
    """
    return _require_implemented


@pytest.fixture
def p1_cfg():
    """The Phase-1 configuration every frozen unit test runs at.

    Takes: nothing. Returns: `gosplan.config.p1_default_config()` - the `EnvConfig` whose every
    field equals the "P1 value" column of the PLAN section 3 registry, already validated by
    `EnvConfig.validate()`. In Phase 1 that means `N = 20` enterprises in `J = 5` sectors, four per
    sector; `M = 4` PRODUCE steps per period; the 5-cycle-with-chords I-O matrix of PLAN section
    2.10 with every entry 0.2; `phi_j = 0.5`; `theta = 8`; `h = 0.02`; the notched bonus
    (`beta = 1`, `w = 0`, `rho_cap = 1.2`); `penalty_form = "proportional"`,
    `penalty_arg = "positive_part"`; `audit_mode = "random"`; `aggregation_level = "enterprise"`;
    `report_lag = 0`; `channel_noise = self_obs_noise = audit_noise = 0`;
    `horizon_mode = "geometric"` with `psi = 0.9`, `P_min = 4`, `P_max = 20`; `seed_env = 0`,
    `seed_policy = 0`.

    The daggered rows of PLAN section 3 (`ratchet_lambda`, `growth_directive`,
    `overfulfilment_slope`, `penalty_scale`, `effort_cost`, `audit_rate`) are provisional
    placeholders until gate G1 records the values the human picks from the DP regime map in
    `runs/G1_decision.md` (PLAN sections 5, 13). A frozen test may therefore assert a *shape* under
    this fixture - a formula, an invariance, a bound - but never a numeric level that only holds at
    a placeholder value.

    Binds: every `tests/unit` module below; the Phase-1 identity cases of `test_planner.py`
    (`report_lag = 0`, `channel_noise = 0`, `aggregation_level = "enterprise"`) and of
    `test_obs.py` (`self_obs_noise = 0`). Owning WO: **WO-003** (`gosplan/config.py`).
    """
    from gosplan.config import p1_default_config

    _require_implemented(p1_default_config)
    cfg = p1_default_config()
    cfg.validate()
    return cfg


@pytest.fixture
def tiny_cfg():
    """The smallest configuration that still exercises every branch: `N = 2`, `J = 2`.

    Takes: nothing. Returns: an `EnvConfig` equal to `p1_default_config()` except for the sizing
    fields, resized to the hand-checked case of `docs/ref_worked_example.md` section 1 - two
    enterprises, two sectors, one enterprise per sector (`sector_of = (0, 1)`), a 2x2 `io_matrix`
    whose two off-diagonal entries are non-zero so each sector depends on the other (coverage,
    delivery and shortage propagation are live in both directions), and `final_demand_share`,
    `productivity`, `yield_sigma`, `ces_alpha` all resized to length 2. Every row of the I-O matrix
    must satisfy `sum_k a[j][k] < 1` (`EnvConfig.validate`, WO-003) and, more tightly,
    `(1 + m) * sum_k a[j][k] < 1`, or the cost-plus price fixed point of PLAN section 2.10 does not
    converge. The configuration must validate; it is `validate()`d before being returned.

    The numeric values are not invented here: they are the ones WO-002 records in
    `docs/ref_worked_example.md` section 1 when it validates `ref/ref_step.py` by hand, so the
    fixture, the reference oracle and the worked example are the same case and the golden files can
    be traced to a hand computation (PLAN section 11, finding F14).

    Why it exists: at `N = 2, J = 2` the conservation identity of test T-U1, the allocation and
    delivery arithmetic of PLAN section 2.7.2-2.7.3, and the coverage aggregator of PLAN section 2.6
    are all checkable by hand, and a failure names one enterprise rather than twenty.

    Binds: `test_conservation.py`, `test_planner.py`, `test_production.py`, `test_env_api.py`.
    Owning WO: **WO-003** (`gosplan/config.py`), with the numeric case from **WO-002**.
    """
    import dataclasses

    from gosplan.config import p1_default_config

    _require_implemented(p1_default_config)
    base = p1_default_config()
    supply = dataclasses.replace(
        base.supply,
        n_enterprises=2,
        n_sectors=2,
        sector_of=(0, 1),
        io_matrix=((0.0, 0.2), (0.2, 0.0)),
        final_demand_share=(0.5, 0.5),
        productivity=(1.0, 1.0),
        yield_sigma=(0.05, 0.08),
        ces_alpha=(0.5, 0.5),
    )
    cfg = dataclasses.replace(base, supply=supply)
    cfg.validate()
    return cfg


@pytest.fixture
def rng_seed():
    """The root environment seed every frozen test draws with.

    Takes: nothing. Returns: the `int` root seed passed as `seed_env` to `gosplan.rng.draw` and to
    `GosplanEnv.reset`, namely `TechConfig.seed_env` - 0 in Phase 1 (PLAN sections 2.15, 3).

    Sharing one `seed_env` across the cases of a test is what makes common random numbers hold by
    construction (PLAN sections 2.15, 4.3): two configurations that differ only in a treatment
    parameter see identical draws, so a difference between them is the treatment and not the noise.
    The draw is keyed on `(seed_env, purpose, *indices)` and never on call order, which is what test
    T-U6 asserts.

    CONTRACT rule 9: this is the *environment* stream only. The policy stream is `seed_policy`,
    kept strictly separate, and a test that needs policy randomness builds its own
    `numpy.random.Generator` from `cfg.tech.seed_policy` rather than reusing this value.

    Binds: `test_rng.py` (T-U6), `test_production.py` (the yield-shock moments),
    `test_planner.py` (audit selection), `test_conservation.py` and `test_env_api.py`. Owning WO:
    **WO-004** (`gosplan/rng.py`).
    """
    return 0
