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


@pytest.mark.skeleton
@pytest.mark.skip(reason=SKIP_REASON)
@pytest.mark.parametrize("ratchet_lambda", RATCHET_LAMBDA_VALUES)
def test_targets_constant_at_zero_growth(ratchet_lambda: float) -> None:
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
    raise NotImplementedError("PLAN section 11 (T-B2) - implemented in WO-002")


@pytest.mark.skeleton
@pytest.mark.skip(reason=SKIP_REASON)
@pytest.mark.parametrize("growth", POSITIVE_GROWTH_VALUES)
def test_targets_grow_at_exactly_one_plus_g(growth: float) -> None:
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
    raise NotImplementedError("PLAN section 11 (T-B2) - implemented in WO-002")


@pytest.mark.skeleton
@pytest.mark.skip(reason=SKIP_REASON)
def test_target_path_is_independent_of_yield_draws() -> None:
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
    raise NotImplementedError("PLAN section 11 (T-B2) - implemented in WO-002")
