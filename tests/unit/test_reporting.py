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

import pytest


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-007")
def test_holding_loss_is_applied_before_this_periods_output_is_added(p1_cfg) -> None:
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
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-007")
def test_report_is_clipped_to_the_report_bound(p1_cfg) -> None:
    """`R_i = clip(rho_i_report, 0, rho_max) * T_i`.

    Assertion: a report action of 25 at `rho_max = cfg.tech.report_max_ratio = 10` yields
    `R_i == 10 * T_i` to 1e-12 and `state.last_report_ratio == 10`; a negative report action yields
    `R_i == 0`; an in-range action passes through unchanged. The claim is stored in units
    (`last_report`) as well as in ratio units (`last_report_ratio`), because the ratchet moves `T`
    later in the same period (PLAN section 2.5, steps 3 then 6).

    Third bullet of the WO-007 reporting must-pass list.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-007")
def test_report_at_the_bound_is_recorded_as_such(p1_cfg) -> None:
    """A report that sits at `rho_max` sets the at-bound marker (CONTRACT rule 8).

    Assertion: `process_reports` records `at_bound = True` for exactly those enterprises whose
    clipped report ratio equals `rho_max`, and `False` otherwise, and the marker reaches
    `StepRecord.at_bound`. The ledger turns the running fraction of at-bound reports above
    `AT_BOUND_FLAG_THRESHOLD = 0.01` into the run flag `BOUND_BINDING`
    (`tests/unit/test_ledger.py`), and test T-B8 forces `rho = 10` in more than 1% of reports and
    asserts the flag is raised. The bound is a result to be reported, never a knob to be moved.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-007")
def test_input_request_is_clipped_to_the_request_bound(p1_cfg) -> None:
    """The stored request is clipped to `r_max * need_ij` (PLAN section 2.3).

    Assertion: `process_reports` stores `state.request` equal to the action's `input_request`
    rescaled against the current need and clipped to `[0, cfg.tech.request_max_multiple *
    need_ij]`, to 1e-12; a request above the bound is clipped rather than rejected, and a negative
    request becomes 0. In Phase 1 the stored value is inert in the allocation
    (`alloc_eta_request = 0`) but is logged, because request inflation `q_ij / need_ij` is the
    Phase-2 hoarding operationalisation and its ledger column must exist from the start.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-007")
def test_stock_above_the_cap_overflows_and_is_recorded(p1_cfg) -> None:
    """Stock above `S_max = inventory_cap_mult * cap_i` is lost and logged (PLAN section 2.11).

    Assertion: when `(1 - h) * S_prev + y` exceeds `S_max`, the retained stock is exactly `S_max`
    and the excess is recorded as `StepRecord.cap_overflow`, to 1e-12; below the cap the overflow
    is exactly 0. The overflow is a sink of the per-period conservation identity of test T-U1, which
    is why it is logged rather than silently clipped: a quantity that disappears without a term is
    a bookkeeping bug, not an inventory rule.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-007")
def test_audit_measures_stock_on_hand_and_not_production(p1_cfg) -> None:
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
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-007")
def test_audit_measurement_is_exact_at_zero_audit_noise(p1_cfg, rng_seed) -> None:
    """At `audit_noise = 0` the measurement is the stock itself; at `sigma > 0` it is lognormal.

    Assertion: with `cfg.information.audit_noise = 0` (Phase 1), `S_hat_i == S_i` exactly for every
    audited enterprise. With `audit_noise = sigma > 0`, `S_hat_i / S_i` is drawn as `exp(nu)`,
    `nu ~ N(0, sigma**2)` through `gosplan.rng.draw` with purpose `auditnoise` and key
    `(seed_env, "auditnoise", t, i)` (CONTRACT rule 9), is deterministic in that key, and has
    sample mean `exp(sigma**2 / 2)` over 1e5 draws to 1e-3 relative.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-007")
def test_positive_part_penalty_is_exactly_zero_for_any_under_report(p1_cfg) -> None:
    """T-U8, first clause: `penalty_arg = "positive_part"` gives 0 whenever `R_i <= S_hat_i`.

    Assertion: for every audited enterprise with a claim below its measured stock - by a hair or by
    an order of magnitude - `audit_and_penalise` returns exactly 0.0, because
    `f_i = max(0, R_i - S_hat_i) / T_i` is 0 there; and for `R_i > S_hat_i` it returns
    `pen * (R_i - S_hat_i) / T_i`. Phase 1 uses `positive_part`, so under-reporting carries no
    audit risk at all: the audit punishes over-claiming only.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-007")
def test_absolute_penalty_charges_under_reports_too(p1_cfg) -> None:
    """T-U8, second clause: `penalty_arg = "absolute"` does not give 0 for an under-report.

    Assertion: on the same audited under-reporting state, `penalty_arg = "absolute"` returns
    `pen * |R_i - S_hat_i| / T_i > 0`, and it equals the `positive_part` result whenever
    `R_i >= S_hat_i`. The two branches differ only on the under-report side, which is what makes
    `penalty_arg` a mechanism parameter of PLAN section 4.2 rather than a numerical detail.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-007")
def test_unaudited_enterprises_pay_no_penalty(p1_cfg) -> None:
    """T-U8, third clause: `audited = False` gives exactly 0, whatever the claim.

    Assertion: `penalty_i = 1[audited_i] * Pen_i` is exactly 0.0 for every enterprise with
    `audited_i` False, including one whose claim exceeds its stock by a large multiple, and the
    returned array has the same shape as `audited`. The expected penalty an agent faces is
    therefore `audit_rate * Pen_i`, which is why `audit_rate * penalty_scale` is the compound
    quantity the G2 padding-elasticity criterion sweeps (PLAN section 4.5).
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-007")
def test_proportional_penalty_is_pen_times_the_ratio_gap(p1_cfg) -> None:
    """`penalty_form = "proportional"` gives `pen * f_i`, in ratio units.

    Assertion: for an audited over-claiming enterprise,
    `penalty_i == cfg.incentive.penalty_scale * (R_i - S_hat_i) / T_i` to 1e-12; doubling the gap
    doubles the penalty; and the penalty is invariant to a common rescaling of `R`, `S` and `T`,
    because the argument is divided by `T_i` (finding F9: the penalty is expressed in ratio units,
    the same units as the bonus, so `pen`, `beta` and `kappa` are comparable).
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-007")
def test_fixed_penalty_form_is_an_indicator(p1_cfg) -> None:
    """`penalty_form = "fixed"` gives `pen * 1[f_i > 0]`, independent of the size of the gap.

    Assertion: under `penalty_form = "fixed"`, every audited enterprise with `f_i > 0` pays exactly
    `cfg.incentive.penalty_scale` regardless of how large the discrepancy is, and every enterprise
    with `f_i == 0` pays exactly 0; the result is a step function of the claim with its single
    discontinuity at `R_i = S_hat_i`.
    """
    assert False
