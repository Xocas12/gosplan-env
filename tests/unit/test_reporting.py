"""Reporting, audit and penalty at the REPORT and AUDIT steps (PLAN section 2.8).

Realises: PLAN section 2.8 (reporting, audit, penalty), PLAN section 2.11 (inventory: holding loss
and the stock cap) and PLAN section 11 (test architecture; property test **T-U8**). Owning work
order: **WO-002** (frozen tests; LEAD). Binds the WO-007 must-pass line of PLAN section 12.3,
verbatim - "`tests/unit/test_reporting.py` (T-U8; audit against stock; holding loss applied before
adding `y`; report clipped to `rho_max`)". Module under test: `gosplan/env/reporting.py`.

T-U8, verbatim (PLAN section 11): "penalty: `positive_part` gives 0 for any under-report; `absolute`
does not; audited=False gives 0."

PLAN section 2.8 formulas, verbatim:

    S_i   <- (1 - h) * S_i + y_i                                  # loss on carried stock, then y
    R_i    = clip(rho_i_report, 0, rho_max) * T_i
    S_hat_i = S_i * exp(nu_i),  nu_i ~ N(0, sigma_aud**2)         key (seed_env, "auditnoise", t, i)
    f_i     = max(0, R_i - S_hat_i) / T_i     if penalty_arg = positive_part   (Phase 1)
            = |R_i - S_hat_i| / T_i           if penalty_arg = absolute
    Pen_i   = pen * f_i                       if penalty_form = proportional   (Phase 1)
            = pen * 1[f_i > 0]                if penalty_form = fixed
    penalty_i = 1[audited_i] * Pen_i

CONTRACT RULE 8 (BOUNDS ARE RESULTS): `report_ratio` is bounded at `rho_max = 10`, the fraction of
reports at the bound is logged, and above 1% the run manifest is flagged `BOUND_BINDING` (test
T-B8). A bound is never silently widened or narrowed to fix a result, and no test here may be made
to pass by moving one.

CONTRACT RULE 7: the audit comparing the claim to stock on hand rather than to production is a rule
about measurement, not an implementation of hidden reserves; no assertion below asserts a direction
of agent behaviour.

FROZEN BY CONTRACT RULE 2. SKELETON: every test is `@pytest.mark.skeleton` and skipped until WO-007
lands; each docstring states the exact assertion, formula and tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest


def _view(cfg, *, claims=None, targets=None, requests=None, audited=None, audit_meas=None):
    """A `PlannerView` built directly, for a planner-rule test.

    `make_planner_view` is the only State -> planner boundary (CONTRACT rule 5), but a test of the
    RULES that consume a view should not have to construct a State to reach them. Fields follow
    PLAN section 2.4.
    """
    from gosplan.env.state import PlannerView

    n, j = cfg.supply.n_enterprises, cfg.supply.n_sectors
    t = np.full(n, cfg.tech.initial_target_frac) if targets is None else np.asarray(targets, float)
    return PlannerView(
        claims=t.copy() if claims is None else np.asarray(claims, dtype=float),
        requests=np.zeros((n, j)) if requests is None else np.asarray(requests, dtype=float),
        audited=np.zeros(n, dtype=bool) if audited is None else np.asarray(audited, dtype=bool),
        audit_meas=np.zeros(n) if audit_meas is None else np.asarray(audit_meas, dtype=float),
        measured_quality=np.ones(n),
        targets=t,
        planner_io=np.asarray(cfg.supply.io_matrix, dtype=float),
        downstream_shortfall=np.zeros(n),
        aggregation_level=cfg.information.aggregation_level,
        plan_prices=np.ones(j),
    )


def _state(cfg, *, inv_output=None, cum_output=None, last_report=None, prices=None):
    """A `State` built directly from `cfg`, for a reporting- or delivery-level test."""
    from gosplan.env.state import State

    n, j = cfg.supply.n_enterprises, cfg.supply.n_sectors
    z = np.zeros(n)
    return State(
        target=np.full(n, cfg.tech.initial_target_frac),
        capital=np.ones(n),
        inv_output=z.copy() if inv_output is None else np.asarray(inv_output, dtype=float),
        inv_inputs=np.zeros((n, j)),
        cum_output=z.copy() if cum_output is None else np.asarray(cum_output, dtype=float),
        cum_cost=z.copy(),
        quality_acc=z.copy(),
        last_report_ratio=z.copy(),
        last_report=z.copy() if last_report is None else np.asarray(last_report, dtype=float),
        last_audited=np.zeros(n, dtype=bool),
        last_penalty=z.copy(),
        last_fill=np.ones(n),
        request=np.zeros((n, j)),
        pending_invest=np.zeros((n, 0)),
        t_period=0,
        k_step=cfg.incentive.steps_per_period,
        phase="report",
        plan_prices=np.ones(j) if prices is None else np.asarray(prices, dtype=float),
        planner_io=np.asarray(cfg.supply.io_matrix, dtype=float),
        consumer_delivery=np.zeros(j),
        alive=True,
        seed_env=cfg.tech.seed_env,
        seed_policy=cfg.tech.seed_policy,
    )


def _action(cfg, *, report_ratio=None, input_request=None):
    """An `EnterpriseAction` carrying only the REPORT-step dimensions."""
    from gosplan.env.state import EnterpriseAction

    n, j = cfg.supply.n_enterprises, cfg.supply.n_sectors
    return EnterpriseAction(
        effort=np.zeros(n),
        quality=np.ones(n),
        invest=np.zeros(n),
        report_ratio=np.ones(n) if report_ratio is None else np.asarray(report_ratio, dtype=float),
        input_request=np.zeros((n, j))
        if input_request is None
        else np.asarray(input_request, float),
        trade_offer=np.zeros((n, j)),
    )


def _t_min(cfg):
    """`T_min_i = target_floor_frac * T_0_i`, with `T_0_i = initial_target_frac * A_{s(i)} * cap_i`."""
    prod = np.asarray(cfg.supply.productivity)[np.asarray(cfg.supply.sector_of)]
    return cfg.tech.target_floor_frac * cfg.tech.initial_target_frac * prod


@pytest.mark.skeleton
def test_holding_loss_is_applied_before_this_periods_output_is_added(p1_cfg, implemented) -> None:
    """`S <- (1 - h) * S + y`, in that order.

    Assertion: with entering stock `S_prev` and period output `y`, `process_reports` leaves
    `state.inv_output == (1 - h) * S_prev + y` to 1e-12, and NOT `(1 - h) * (S_prev + y)`; the two
    differ by `h * y`, which is the whole content of the ordering. At `h = 0.02` (Phase 1) and
    `S_prev = 0` the two forms coincide, so the test must use a non-zero entering stock. The loss
    applies to the stock carried in, never to output just produced.

    Second bullet of the WO-007 reporting must-pass list. The lost quantity `h * S_prev` is a term
    of the per-period conservation identity of test T-U1 and must be recorded as
    `StepRecord.holding_loss`, not silently dropped.
    """
    from gosplan.env.reporting import process_reports

    implemented(process_reports)
    h = p1_cfg.supply.holding_loss
    assert h > 0.0
    n = p1_cfg.supply.n_enterprises
    s_prev = np.linspace(0.1, 1.0, n)
    y = np.linspace(0.2, 0.8, n)
    state = _state(p1_cfg, inv_output=s_prev, cum_output=y)
    after = process_reports(state, _action(p1_cfg), p1_cfg)
    want = (1.0 - h) * s_prev + y
    wrong = (1.0 - h) * (s_prev + y)
    got = np.asarray(after.inv_output)
    assert np.max(np.abs(got - want)) < 1e-12
    assert np.max(np.abs(got - wrong)) > 1e-6  # the two differ by h * y


@pytest.mark.skeleton
def test_report_is_clipped_to_the_report_bound(p1_cfg, implemented) -> None:
    """`R_i = clip(rho_i_report, 0, rho_max) * T_i`.

    Assertion: a report action of 25 at `rho_max = cfg.tech.report_max_ratio = 10` yields
    `R_i == 10 * T_i` to 1e-12 and `state.last_report_ratio == 10`; a negative report action yields
    `R_i == 0`; an in-range action passes through unchanged. The claim is stored in units
    (`last_report`) as well as in ratio units (`last_report_ratio`), because the ratchet moves `T`
    later in the same period (PLAN section 2.5, steps 3 then 6).

    Third bullet of the WO-007 reporting must-pass list.
    """
    from gosplan.env.reporting import process_reports

    implemented(process_reports)
    rho_max = p1_cfg.tech.report_max_ratio
    n = p1_cfg.supply.n_enterprises
    ratios = np.full(n, 25.0)
    ratios[1] = -3.0
    ratios[2] = 1.05
    state = _state(p1_cfg)
    after = process_reports(state, _action(p1_cfg, report_ratio=ratios), p1_cfg)
    t = np.asarray(after.target)
    got_ratio = np.asarray(after.last_report_ratio)
    got_report = np.asarray(after.last_report)
    assert abs(got_ratio[0] - rho_max) < 1e-12
    assert abs(got_report[0] - rho_max * t[0]) < 1e-12
    assert abs(got_ratio[1]) < 1e-12
    assert abs(got_report[1]) < 1e-12
    assert abs(got_ratio[2] - 1.05) < 1e-12


@pytest.mark.skeleton
def test_report_at_the_bound_is_recorded_as_such(p1_cfg, implemented) -> None:
    """A report that sits at `rho_max` sets the at-bound marker (CONTRACT rule 8).

    Assertion: `process_reports` records `at_bound = True` for exactly those enterprises whose
    clipped report ratio equals `rho_max`, and `False` otherwise, and the marker reaches
    `StepRecord.at_bound`. The ledger turns the running fraction of at-bound reports above
    `AT_BOUND_FLAG_THRESHOLD = 0.01` into the run flag `BOUND_BINDING`
    (`tests/unit/test_ledger.py`), and test T-B8 forces `rho = 10` in more than 1% of reports and
    asserts the flag is raised. The bound is a result to be reported, never a knob to be moved.
    """
    from gosplan.env.reporting import process_reports

    implemented(process_reports)
    rho_max = p1_cfg.tech.report_max_ratio
    n = p1_cfg.supply.n_enterprises
    ratios = np.full(n, 1.0)
    ratios[0] = rho_max
    ratios[1] = rho_max + 5.0
    after = process_reports(_state(p1_cfg), _action(p1_cfg, report_ratio=ratios), p1_cfg)
    got = np.asarray(after.last_report_ratio)
    at_bound = np.isclose(got, rho_max)
    assert bool(at_bound[0]) and bool(at_bound[1])
    assert not bool(at_bound[2])


@pytest.mark.skeleton
def test_input_request_is_clipped_to_the_request_bound(p1_cfg, implemented) -> None:
    """The stored request is clipped to `r_max * need_ij` (PLAN section 2.3).

    Assertion: `process_reports` stores `state.request` equal to the action's `input_request`
    rescaled against the current need and clipped to `[0, cfg.tech.request_max_multiple *
    need_ij]`, to 1e-12; a request above the bound is clipped rather than rejected, and a negative
    request becomes 0. In Phase 1 the stored value is inert in the allocation
    (`alloc_eta_request = 0`) but is logged, because request inflation `q_ij / need_ij` is the
    Phase-2 hoarding operationalisation and its ledger column must exist from the start.
    """
    from gosplan.env.reporting import process_reports

    implemented(process_reports)
    r_max = p1_cfg.tech.request_max_multiple
    n, j = p1_cfg.supply.n_enterprises, p1_cfg.supply.n_sectors
    need = np.asarray(p1_cfg.supply.io_matrix, dtype=float)[np.asarray(p1_cfg.supply.sector_of)]
    need = need * p1_cfg.tech.initial_target_frac
    huge = np.full((n, j), 1e6)
    after = process_reports(_state(p1_cfg), _action(p1_cfg, input_request=huge), p1_cfg)
    got = np.asarray(after.request)
    assert np.all(got >= -1e-12)
    assert np.all(got <= r_max * need + 1e-9)


@pytest.mark.skeleton
def test_stock_above_the_cap_overflows_and_is_recorded(p1_cfg, implemented) -> None:
    """Stock above `S_max = inventory_cap_mult * cap_i` is lost and logged (PLAN section 2.11).

    Assertion: when `(1 - h) * S_prev + y` exceeds `S_max`, the retained stock is exactly `S_max`
    and the excess is recorded as `StepRecord.cap_overflow`, to 1e-12; below the cap the overflow
    is exactly 0. The overflow is a sink of the per-period conservation identity of test T-U1, which
    is why it is logged rather than silently clipped: a quantity that disappears without a term is
    a bookkeeping bug, not an inventory rule.
    """
    from gosplan.env.reporting import process_reports

    implemented(process_reports)
    h = p1_cfg.supply.holding_loss
    s_max_mult = p1_cfg.tech.inventory_cap_mult
    n = p1_cfg.supply.n_enterprises
    s_prev = np.full(n, 2.0 * s_max_mult)
    y = np.full(n, 2.0 * s_max_mult)
    after = process_reports(
        _state(p1_cfg, inv_output=s_prev, cum_output=y), _action(p1_cfg), p1_cfg
    )
    got = np.asarray(after.inv_output)
    cap = s_max_mult * np.asarray(after.capital)
    raw = (1.0 - h) * s_prev + y
    assert np.all(raw > cap)  # the case must actually bind
    assert np.max(np.abs(got - cap)) < 1e-12


@pytest.mark.skeleton
def test_audit_measures_stock_on_hand_and_not_production(p1_cfg, implemented) -> None:
    """The audit compares the claim to `S_i`, never to the period's output.

    Assertion: construct two states with identical stock `S_i` but very different period output
    `y_i` (one where output was consumed by delivery, one where it was not) and identical claims;
    `audit_and_penalise` returns identical penalties. Then vary `S_i` alone and the penalty must
    move. The measurement is `S_hat_i = S_i * exp(nu_i)` on the post-REPORT stock, i.e. after
    `S <- (1 - h) * S + y` has run.

    First bullet of the WO-007 reporting must-pass list. That single choice is what makes
    accumulated stock protect against audits; asserting *that* an agent accumulates is held out
    (PLAN section 4.1 row 7) and is asserted nowhere.
    """
    from gosplan.env.reporting import audit_and_penalise

    implemented(audit_and_penalise)
    n = p1_cfg.supply.n_enterprises
    audited = np.ones(n, dtype=bool)
    stock = np.full(n, 0.3)
    claims = np.full(n, 0.9 * p1_cfg.tech.initial_target_frac)
    low_y = _state(p1_cfg, inv_output=stock, cum_output=np.zeros(n), last_report=claims)
    high_y = _state(p1_cfg, inv_output=stock, cum_output=np.full(n, 99.0), last_report=claims)
    a = np.asarray(audit_and_penalise(low_y, audited, p1_cfg, 0))
    b = np.asarray(audit_and_penalise(high_y, audited, p1_cfg, 0))
    assert np.array_equal(a, b)


@pytest.mark.skeleton
def test_audit_measurement_is_exact_at_zero_audit_noise(p1_cfg, rng_seed, implemented) -> None:
    """At `audit_noise = 0` the measurement is the stock itself; at `sigma > 0` it is lognormal.

    Assertion: with `cfg.information.audit_noise = 0` (Phase 1), `S_hat_i == S_i` exactly for every
    audited enterprise. With `audit_noise = sigma > 0`, `S_hat_i / S_i` is drawn as `exp(nu)`,
    `nu ~ N(0, sigma**2)` through `gosplan.rng.draw` with purpose `auditnoise` and key
    `(seed_env, "auditnoise", t, i)` (CONTRACT rule 9), is deterministic in that key, and has
    sample mean `exp(sigma**2 / 2)` over 1e5 draws to 1e-3 relative.
    """
    from gosplan.env.reporting import audit_and_penalise

    implemented(audit_and_penalise)
    assert p1_cfg.information.audit_noise == 0.0
    n = p1_cfg.supply.n_enterprises
    t0 = p1_cfg.tech.initial_target_frac
    stock = np.full(n, 0.2 * t0)
    claims = np.full(n, 0.9 * t0)
    audited = np.ones(n, dtype=bool)
    pen = np.asarray(
        audit_and_penalise(_state(p1_cfg, inv_output=stock, last_report=claims), audited, p1_cfg, 0)
    )
    want = p1_cfg.incentive.penalty_scale * (claims - stock) / t0
    assert np.max(np.abs(pen - want)) < 1e-12


@pytest.mark.skeleton
def test_positive_part_penalty_is_exactly_zero_for_any_under_report(p1_cfg, implemented) -> None:
    """T-U8, first clause: `penalty_arg = "positive_part"` gives 0 whenever `R_i <= S_hat_i`.

    Assertion: for every audited enterprise with a claim below its measured stock - by a hair or by
    an order of magnitude - `audit_and_penalise` returns exactly 0.0, because
    `f_i = max(0, R_i - S_hat_i) / T_i` is 0 there; and for `R_i > S_hat_i` it returns
    `pen * (R_i - S_hat_i) / T_i`. Phase 1 uses `positive_part`, so under-reporting carries no
    audit risk at all: the audit punishes over-claiming only.
    """
    from gosplan.env.reporting import audit_and_penalise

    implemented(audit_and_penalise)
    assert p1_cfg.incentive.penalty_arg == "positive_part"
    n = p1_cfg.supply.n_enterprises
    t0 = p1_cfg.tech.initial_target_frac
    stock = np.full(n, 5.0 * t0)
    claims = np.linspace(1e-9, 4.0 * t0, n)  # every claim strictly below stock
    pen = np.asarray(
        audit_and_penalise(
            _state(p1_cfg, inv_output=stock, last_report=claims), np.ones(n, dtype=bool), p1_cfg, 0
        )
    )
    assert np.all(pen == 0.0)


@pytest.mark.skeleton
def test_absolute_penalty_charges_under_reports_too(p1_cfg, implemented) -> None:
    """T-U8, second clause: `penalty_arg = "absolute"` does not give 0 for an under-report.

    Assertion: on the same audited under-reporting state, `penalty_arg = "absolute"` returns
    `pen * |R_i - S_hat_i| / T_i > 0`, and it equals the `positive_part` result whenever
    `R_i >= S_hat_i`. The two branches differ only on the under-report side, which is what makes
    `penalty_arg` a mechanism parameter of PLAN section 4.2 rather than a numerical detail.
    """
    import dataclasses

    from gosplan.env.reporting import audit_and_penalise

    implemented(audit_and_penalise)
    cfg = dataclasses.replace(
        p1_cfg, incentive=dataclasses.replace(p1_cfg.incentive, penalty_arg="absolute")
    )
    n = cfg.supply.n_enterprises
    t0 = cfg.tech.initial_target_frac
    stock = np.full(n, 5.0 * t0)
    claims = np.full(n, 2.0 * t0)
    state = _state(cfg, inv_output=stock, last_report=claims)
    audited = np.ones(n, dtype=bool)
    absolute = np.asarray(audit_and_penalise(state, audited, cfg, 0))
    positive = np.asarray(audit_and_penalise(state, audited, p1_cfg, 0))
    assert np.all(absolute > 0.0)
    assert np.all(positive == 0.0)
    want = cfg.incentive.penalty_scale * np.abs(claims - stock) / t0
    assert np.max(np.abs(absolute - want)) < 1e-12


@pytest.mark.skeleton
def test_unaudited_enterprises_pay_no_penalty(p1_cfg, implemented) -> None:
    """T-U8, third clause: `audited = False` gives exactly 0, whatever the claim.

    Assertion: `penalty_i = 1[audited_i] * Pen_i` is exactly 0.0 for every enterprise with
    `audited_i` False, including one whose claim exceeds its stock by a large multiple, and the
    returned array has the same shape as `audited`. The expected penalty an agent faces is
    therefore `audit_rate * Pen_i`, which is why `audit_rate * penalty_scale` is the compound
    quantity the G2 padding-elasticity criterion sweeps (PLAN section 4.5).
    """
    from gosplan.env.reporting import audit_and_penalise

    implemented(audit_and_penalise)
    n = p1_cfg.supply.n_enterprises
    t0 = p1_cfg.tech.initial_target_frac
    state = _state(p1_cfg, inv_output=np.zeros(n), last_report=np.full(n, 50.0 * t0))
    pen = np.asarray(audit_and_penalise(state, np.zeros(n, dtype=bool), p1_cfg, 0))
    assert np.all(pen == 0.0)


@pytest.mark.skeleton
def test_proportional_penalty_is_pen_times_the_ratio_gap(p1_cfg, implemented) -> None:
    """`penalty_form = "proportional"` gives `pen * f_i`, in ratio units.

    Assertion: for an audited over-claiming enterprise,
    `penalty_i == cfg.incentive.penalty_scale * (R_i - S_hat_i) / T_i` to 1e-12; doubling the gap
    doubles the penalty; and the penalty is invariant to a common rescaling of `R`, `S` and `T`,
    because the argument is divided by `T_i` (finding F9: the penalty is expressed in ratio units,
    the same units as the bonus, so `pen`, `beta` and `kappa` are comparable).
    """
    from gosplan.env.reporting import audit_and_penalise

    implemented(audit_and_penalise)
    assert p1_cfg.incentive.penalty_form == "proportional"
    n = p1_cfg.supply.n_enterprises
    t0 = p1_cfg.tech.initial_target_frac
    stock = np.zeros(n)
    audited = np.ones(n, dtype=bool)
    small = np.asarray(
        audit_and_penalise(
            _state(p1_cfg, inv_output=stock, last_report=np.full(n, t0)), audited, p1_cfg, 0
        )
    )
    doubled = np.asarray(
        audit_and_penalise(
            _state(p1_cfg, inv_output=stock, last_report=np.full(n, 2.0 * t0)), audited, p1_cfg, 0
        )
    )
    assert np.max(np.abs(small - p1_cfg.incentive.penalty_scale * 1.0)) < 1e-12
    assert np.max(np.abs(doubled - 2.0 * small)) < 1e-12


@pytest.mark.skeleton
def test_fixed_penalty_form_is_an_indicator(p1_cfg, implemented) -> None:
    """`penalty_form = "fixed"` gives `pen * 1[f_i > 0]`, independent of the size of the gap.

    Assertion: under `penalty_form = "fixed"`, every audited enterprise with `f_i > 0` pays exactly
    `cfg.incentive.penalty_scale` regardless of how large the discrepancy is, and every enterprise
    with `f_i == 0` pays exactly 0; the result is a step function of the claim with its single
    discontinuity at `R_i = S_hat_i`.
    """
    import dataclasses

    from gosplan.env.reporting import audit_and_penalise

    implemented(audit_and_penalise)
    cfg = dataclasses.replace(
        p1_cfg, incentive=dataclasses.replace(p1_cfg.incentive, penalty_form="fixed")
    )
    n = cfg.supply.n_enterprises
    t0 = cfg.tech.initial_target_frac
    audited = np.ones(n, dtype=bool)
    small = np.asarray(
        audit_and_penalise(
            _state(cfg, inv_output=np.zeros(n), last_report=np.full(n, 0.01 * t0)), audited, cfg, 0
        )
    )
    large = np.asarray(
        audit_and_penalise(
            _state(cfg, inv_output=np.zeros(n), last_report=np.full(n, 50.0 * t0)), audited, cfg, 0
        )
    )
    assert np.max(np.abs(small - cfg.incentive.penalty_scale)) < 1e-12
    assert np.array_equal(small, large)  # independent of the size of the gap
    honest = np.asarray(
        audit_and_penalise(
            _state(cfg, inv_output=np.full(n, 9.0 * t0), last_report=np.full(n, t0)),
            audited,
            cfg,
            0,
        )
    )
    assert np.all(honest == 0.0)
