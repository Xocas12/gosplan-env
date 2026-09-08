"""T-B2 - the target rule's fixed point under a constant `rho = 1` report.

Realises: PLAN section 11 (behavioural test T-B2), read against PLAN sections 2.7.1 (the target
rule), 2.9.2 (the fulfilment measure the ratchet keys on), 2.5 (the period schedule, TARGET after
REPORT) and 1.3/3 (finding F1 - the growth directive is the forcing term, and it must be a
treatment variable *because* the map has a fixed point without it). Owning work order: **WO-002**;
on the must-pass list of **WO-010** (heuristic agents) and binding **WO-006** (`update_targets`).

What T-B2 asserts (PLAN section 11, verbatim): *`Padder` at `g = 0` keeps `T` constant; at `g > 0`
`T` grows at exactly `(1 + g)`.*

Why the fixed point exists. `Padder` reports `rho = PADDER_REPORT_RATIO = 1.0` every period
whatever it produced, so with `objective_metric = "val"` the planner's measure is `m_i = R_i =
rho * T_i = T_i` and `rho_measure = m_i / T_i = 1` exactly. The target rule of PLAN section 2.7.1
is then

    step_i = clip(rho_i - 1, -c_dn, +c_up) = 0
    step_i = 0                                   also under the deadband, which is inert at 0
    T_i   <- max(T_min, (1 + g) * T_i * (1 + lambda * step_i)) = max(T_min, (1 + g) * T_i)

so at `g = 0` the map is the identity on `T` and at `g > 0` it is multiplication by `(1 + g)`, with
the floor `T_min = target_floor_frac * T_0` inactive because the sequence never decreases. Nothing
in that derivation depends on realised output, on the yield draws or on `lambda`, which is what
makes this a *fixed point* test rather than a numerical one: the assertions below are exact to
`FIXED_POINT_TOL`, and `ratchet_lambda` is deliberately swept to show the map is invariant to it.

`Padder` is a sanity probe, never a baseline (PLAN section 6.1 and the class docstring in
`gosplan/agents/heuristic.py`): its two uses are this test and T-B3, plus the Monte-Carlo sanity
harness of WO-012. No table, plot or claim may use it as a comparison point for padding (PLAN
section 4.1 row 4) - its padding is assumed, not learned.

Held-out phenomena (PLAN section 4.1): nothing here computes rows 2, 5, 6 or 7. The quantity under
test is the planner's target path, which is planner-side by construction.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

SKIP_REASON = (
    "skeleton: T-B2 assertions are written by WO-002 (frozen tests); they bind WO-006 "
    "(update_targets), WO-009 (the period schedule) and WO-010 (Padder)"
)
"""Reason attached to every `@pytest.mark.skip` below; the bodies arrive with WO-002."""

TB2_AGENT = "Padder"
"""The probe of PLAN section 11's T-B2 clause: `gosplan.agents.heuristic.Padder`, which reports
`PADDER_REPORT_RATIO = 1.0` every period at `PADDER_EFFORT = 0.3`."""

ZERO_GROWTH = 0.0
"""`g = 0`: the fixed-point case. `growth_directive` is an INC parameter (PLAN section 3)."""

POSITIVE_GROWTH_VALUES: tuple[float, ...] = (0.01, 0.02, 0.05)
"""`g > 0` cases. `0.02` is the provisional Phase-1 value of the daggered `growth_directive` row of
PLAN section 3; the other two are interior points of its declared range [0, 0.07]. Three values,
not one, because the assertion is an exact growth factor and a single value could be matched by an
unrelated rule."""

RATCHET_LAMBDA_VALUES: tuple[float, ...] = (0.0, 0.5, 1.0)
"""`lambda` values swept in both cases. At `rho = 1` the ratchet step is zero, so the target path
must be identical for every `lambda` in the range [0, 1] of PLAN section 3; a dependence on
`lambda` means the step was not zero."""

TB2_SEEDS: tuple[int, ...] = (0, 1, 2)
"""Environment seeds, one episode each. A WO-002 test-design constant. Three seeds, because the
claim is that the target path does not depend on the yield draws at all - `test_target_path_is_
independent_of_yield_draws` compares the paths across these seeds."""

FIXED_POINT_TOL = 1e-12
"""Tolerance on the target identities. Tighter than the 1e-9 of the golden files (PLAN section 11):
these are exact arithmetic identities on one multiplication per period, not trajectory agreement
between two implementations."""


def _gate(*targets):
    """Skip the calling test while any of `targets` is still a skeleton stub.

    The module-level twin of the `implemented` fixture in `tests/conftest.py`. A module-level
    helper cannot request a fixture, so the check is repeated here rather than the helper being
    called before the gate - which would raise `NotImplementedError` and FAIL the test instead of
    skipping it.
    """
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


def _cfg(**sections):
    """`p1_default_config()` with per-section overrides applied, validated."""
    import dataclasses

    from gosplan.config import p1_default_config

    _gate(p1_default_config)
    cfg = p1_default_config()
    for section, changes in sections.items():
        cfg = dataclasses.replace(
            cfg, **{section: dataclasses.replace(getattr(cfg, section), **changes)}
        )
    cfg.validate()
    return cfg


def _episode(cfg, agent_name, seed_env, implemented, max_periods=None):
    """Drive one episode of `GosplanEnv` with a named heuristic; return the ledger records.

    Gated on the environment (WO-009), the heuristic agents (WO-010) and the ledger (WO-011), so a
    behavioural module skips naming its missing dependency rather than failing.
    """
    from gosplan.agents import heuristic
    from gosplan.config import p1_default_config
    from gosplan.env.env import GosplanEnv
    from gosplan.metrics.ledger import Ledger

    agent_cls = getattr(heuristic, agent_name)
    implemented(p1_default_config, GosplanEnv.reset, GosplanEnv.step, agent_cls.act, Ledger.append)

    env = GosplanEnv(cfg)
    ledger = Ledger()
    env.attach_ledger(ledger)
    obs, _info = env.reset(seed_env, cfg.tech.seed_policy)
    policy = agent_cls(cfg)
    rng = np.random.default_rng(cfg.tech.seed_policy)
    m = cfg.incentive.steps_per_period
    cap = (max_periods or cfg.tech.max_periods) * (m + 1)
    done = False
    steps = 0
    while not done and steps < cap:
        obs, _r, done, _i = env.step(policy.act(obs, env.phase(), rng))
        steps += 1
    return ledger.records


def _report_rows(records):
    """Only the REPORT-step rows, which is where period-level quantities are written."""
    return [r for r in records if r.phase == "report"]


@pytest.mark.skeleton
@pytest.mark.parametrize("ratchet_lambda", RATCHET_LAMBDA_VALUES)
def test_targets_constant_at_zero_growth(ratchet_lambda: float, implemented) -> None:
    """At `g = 0` a constant `rho = 1` report leaves every target exactly where it started.

    Roll out `TB2_AGENT` at `p1_default_config()` with `incentive.growth_directive = ZERO_GROWTH`
    and `incentive.ratchet_lambda = ratchet_lambda`, one episode per seed in `TB2_SEEDS`, recording
    the target after every TARGET step (PLAN section 2.5 step 6). Assert, for every enterprise `i`
    and every period `t` of the episode,

        abs(T_i(t + 1) - T_i(t)) <= FIXED_POINT_TOL
        abs(T_i(t) - T_i(0))     <= FIXED_POINT_TOL

    i.e. the map's fixed point, and no drift accumulated over the episode. Assert also that the
    floor never bound (`T_i(t) > target_floor_frac * T_0` throughout), so the constancy is the
    ratchet's fixed point and not the floor clamping a falling sequence.

    The `lambda` sweep is the point of the parametrisation: at `rho = 1` the step is zero, so the
    path must be identical for `lambda in RATCHET_LAMBDA_VALUES`. Owning WO: **WO-002**; binds
    **WO-006** (`update_targets`, T-U4's behavioural counterpart) and **WO-010**.
    """
    cfg = _cfg(incentive=dict(growth_directive=ZERO_GROWTH, ratchet_lambda=ratchet_lambda))
    floor = cfg.tech.target_floor_frac * cfg.tech.initial_target_frac
    for seed in TB2_SEEDS:
        rows = _report_rows(_episode(cfg, TB2_AGENT, seed, implemented))
        by_ent: dict[int, list[float]] = {}
        for r in rows:
            by_ent.setdefault(r.enterprise, []).append(float(r.target))
        for i, path in by_ent.items():
            for a, b in itertools.pairwise(path):
                assert abs(b - a) <= FIXED_POINT_TOL, (seed, i)
            for t in path:
                assert abs(t - path[0]) <= FIXED_POINT_TOL, (seed, i)
                assert t > floor, (seed, i)


@pytest.mark.skeleton
@pytest.mark.parametrize("growth", POSITIVE_GROWTH_VALUES)
def test_targets_grow_at_exactly_one_plus_g(growth: float, implemented) -> None:
    """At `g > 0` the same report makes every target grow by exactly `(1 + g)` per period.

    Roll out `TB2_AGENT` at `p1_default_config()` with `incentive.growth_directive = growth`,
    sweeping `incentive.ratchet_lambda` over `RATCHET_LAMBDA_VALUES`, one episode per seed in
    `TB2_SEEDS`. Assert, for every enterprise `i` and every period `t`,

        abs(T_i(t + 1) - (1.0 + growth) * T_i(t)) <= FIXED_POINT_TOL * max(1.0, T_i(t))
        abs(T_i(t) - (1.0 + growth) ** t * T_i(0)) <= FIXED_POINT_TOL * max(1.0, T_i(t))

    - the one-step factor and the compounded path, the second catching an error that cancels
    between consecutive steps. Assert that the growth factor is identical across enterprises,
    across sectors and across `RATCHET_LAMBDA_VALUES`.

    This is the forcing term of finding F1 (PLAN sections 1.3, 2.7.1): because the map has a fixed
    point at `g = 0`, `g` must be a treatment variable rather than a constant, and this test is
    what pins its arithmetic. Owning WO: **WO-002**; binds **WO-006** and **WO-010**.
    """
    cfg = _cfg(incentive=dict(growth_directive=growth))
    for seed in TB2_SEEDS:
        rows = _report_rows(_episode(cfg, TB2_AGENT, seed, implemented))
        by_ent: dict[int, list[float]] = {}
        for r in rows:
            by_ent.setdefault(r.enterprise, []).append(float(r.target))
        for i, path in by_ent.items():
            for a, b in itertools.pairwise(path):
                assert abs(b - a * (1.0 + growth)) <= FIXED_POINT_TOL * max(1.0, abs(a)), (seed, i)


@pytest.mark.skeleton
def test_target_path_is_independent_of_yield_draws(implemented) -> None:
    """The target path under `Padder` does not depend on the environment's draws at all.

    Roll out `TB2_AGENT` at `p1_default_config()` once per seed in `TB2_SEEDS`, at
    `growth_directive = ZERO_GROWTH` and again at each value of `POSITIVE_GROWTH_VALUES`, and
    assert that the recorded target path `T_i(t)` is identical across seeds to
    `FIXED_POINT_TOL` - while checking that the same rollouts' `cum_output` paths are *not*
    identical across seeds, so the invariance is a property of the target rule and not of a
    degenerate environment in which nothing is random.

    The report is a constant, the fulfilment measure `val` is the claim (PLAN section 2.9.2) and
    the claim is `rho * T`, so no yield shock, no coverage shortfall and no audit outcome enters
    the ratchet. A seed-dependent target path means some true quantity leaked into the target rule,
    which is a CONTRACT rule 5 failure as well as a T-B2 failure - `update_targets` takes a
    `PlannerView`. Owning WO: **WO-002**; binds **WO-006**.
    """
    cfg = _cfg()
    paths = []
    for seed in TB2_SEEDS:
        rows = _report_rows(_episode(cfg, TB2_AGENT, seed, implemented))
        paths.append([float(r.target) for r in rows if r.enterprise == 0])
    shortest = min(len(p) for p in paths)
    for other in paths[1:]:
        for a, b in zip(paths[0][:shortest], other[:shortest], strict=True):
            assert abs(a - b) <= FIXED_POINT_TOL
