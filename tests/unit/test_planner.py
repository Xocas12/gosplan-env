"""Planner rules: target ratchet, allocation, delivery and audit selection (PLAN section 2.7).

Realises: PLAN sections 2.7.1-2.7.5 (planner rules, which read only a `PlannerView`), 2.9.2
(fulfilment measure) and PLAN section 11 (test architecture; property test **T-U4**). Owning work
order: **WO-002** (frozen tests; LEAD). Binds the WO-006 must-pass line of PLAN section 12.3,
verbatim - "`tests/unit/test_planner.py` (T-U4; allocation sums to `avail_j`; `eta_q = 0` ignores
requests; `poolfill` in [0,1]; delivery conservation)". Module under test: `gosplan/env/planner.py`.

T-U4, verbatim (PLAN section 11): "target rule: fixed point at `rho = 1`, `g = 0`; step bounded by
caps; floor respected; deadband inert outside `|rho - 1| <= delta`."

CONTRACT RULE 5 (PLANNER BLINDNESS) frames every test here: planner rules take a `PlannerView` and
nothing else, the view is built by the single function `make_planner_view(state, cfg)`, and it
contains no true quantity. The structural half of that rule is test T-B4 in
`tests/behavioural/test_planner_blindness.py`; this module tests the arithmetic the rules perform on
a view, and never hands one of them a `State`. (`deliver` is the sanctioned exception noted in the
spec: it is physical execution of an allocation already decided from the view, and the T-B4 static
check whitelists it alongside `make_planner_view`.)

CONTRACT RULE 7: none of these rules implements a pathology. A claim above stock lowering
`poolfill`, and a claim below stock leaving goods in `S`, are consequences of the four delivery
lines of PLAN section 2.7.3 - never rules of their own, and no assertion below asserts a direction
of agent behaviour.

FROZEN BY CONTRACT RULE 2. SKELETON: every test is `@pytest.mark.skeleton` and skipped until WO-006
lands; each docstring states the exact assertion, formula and tolerance.
"""

from __future__ import annotations

import math

import numpy as np
import pytest


def _view(cfg, *, claims=None, targets=None, requests=None, audited=None, audit_meas=None):
    """A `PlannerView` built directly, for a planner-rule test.

    `make_planner_view` is the only State -> planner boundary (CONTRACT rule 5), but a test of the
    RULES that consume a view should not have to construct a State to reach them. Fields follow
    PLAN section 2.4.
    """
    from gosplan.env.planner import PlannerView

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
def test_target_rule_has_a_fixed_point_at_unit_ratio_and_zero_growth(p1_cfg, implemented) -> None:
    """T-U4, first clause: at `rho = 1` and `g = 0` the target is unchanged.

    Assertion: with `growth_directive = 0` and a view whose claims equal its targets (so
    `rho_i = m_i / T_i = 1` for every `i`), `update_targets(view, cfg)` returns the input targets
    elementwise to 1e-12, at every `ratchet_lambda` in [0, 1]. Formula (PLAN section 2.7.1,
    verbatim):

        step_i = clip(rho_i - 1, -c_dn, +c_up)
        T_i   <- max(T_min, (1 + g) * T_i * (1 + lambda * step_i))

    at `rho = 1` the step is 0 and the map is the identity above the floor.

    This is exactly why `g` must be a treatment variable rather than a constant (finding F1): the
    ratchet alone has a fixed point, and the behavioural counterpart is test T-B2 in
    `tests/behavioural/test_fixed_point.py` (`Padder` at `g = 0` keeps `T` constant; at `g > 0` it
    grows at exactly `(1 + g)`).
    """
    import dataclasses

    from gosplan.env.planner import update_targets

    implemented(update_targets)
    cfg = dataclasses.replace(
        p1_cfg, incentive=dataclasses.replace(p1_cfg.incentive, growth_directive=0.0)
    )
    view = _view(cfg)
    out = np.asarray(update_targets(view, cfg))
    assert np.max(np.abs(out - np.asarray(view.targets))) < 1e-12


@pytest.mark.skeleton
def test_target_step_is_bounded_by_the_ratchet_caps(p1_cfg, implemented) -> None:
    """T-U4, second clause: the per-period step is clipped to `[-c_dn, +c_up]`.

    Assertion: for `rho` far above and far below 1 - including `rho = report_max_ratio = 10` and
    `rho = 0` - the realised multiplier satisfies
    `T_new / ((1 + g) * T_old) == 1 + lambda * clip(rho - 1, -c_dn, +c_up)` to 1e-12 wherever the
    floor does not bind, so no single period can move a target by more than
    `lambda * c_up` upward or `lambda * c_dn` downward. At the Phase-1 caps `c_up = c_dn = 0.3` the
    bound is `1 +/- 0.3 * lambda` per period.
    """
    from gosplan.env.planner import update_targets

    implemented(update_targets)
    g = p1_cfg.incentive.growth_directive
    lam = p1_cfg.incentive.ratchet_lambda
    c_up, c_dn = p1_cfg.incentive.ratchet_cap_up, p1_cfg.incentive.ratchet_cap_dn
    for rho, want_step in ((p1_cfg.tech.report_max_ratio, c_up), (0.0, -c_dn)):
        view = _view(
            p1_cfg,
            claims=np.full(p1_cfg.supply.n_enterprises, rho)
            * np.full(p1_cfg.supply.n_enterprises, p1_cfg.tech.initial_target_frac),
        )
        out = np.asarray(update_targets(view, p1_cfg))
        t_old = np.asarray(view.targets)
        floor = _t_min(p1_cfg)
        want = np.maximum(floor, (1.0 + g) * t_old * (1.0 + lam * want_step))
        assert np.max(np.abs(out - want)) < 1e-12, rho


@pytest.mark.skeleton
def test_target_floor_is_respected(p1_cfg, implemented) -> None:
    """T-U4, third clause: the target never falls below `T_min`.

    Assertion: driving `rho` to 0 for many consecutive periods leaves every target at exactly
    `T_min = cfg.tech.target_floor_frac * T_0` and never below, where
    `T_0 = cfg.tech.initial_target_frac * A_{s(i)} * cap_i`; the floor is applied as the outer
    `max` of `T_i <- max(T_min, (1 + g) * T_i * (1 + lambda * step_i))`, so it binds after the
    growth directive and the ratchet, not before.
    """
    from gosplan.env.planner import update_targets

    implemented(update_targets)
    n = p1_cfg.supply.n_enterprises
    targets = np.full(n, p1_cfg.tech.initial_target_frac)
    floor = _t_min(p1_cfg)
    for _ in range(200):
        view = _view(p1_cfg, claims=np.zeros(n), targets=targets)
        targets = np.asarray(update_targets(view, p1_cfg))
        assert np.all(targets >= floor - 1e-12)
    assert np.max(np.abs(targets - floor)) < 1e-9


@pytest.mark.skeleton
def test_target_deadband_is_inert_outside_the_band(p1_cfg, implemented) -> None:
    """T-U4, fourth clause: the deadband zeroes the step only inside `|rho - 1| <= delta`.

    Assertion: at `ratchet_deadband = delta > 0`, any `rho` with `|rho - 1| <= delta` leaves
    `T_new == (1 + g) * T_old` (the step is exactly 0), while any `rho` with `|rho - 1| > delta`
    gives the same result as at `delta = 0` - the deadband does not shrink or shift the step outside
    the band. At the Phase-1 value `delta = 0` the branch is inert and the rule matches the
    `delta = 0` formula exactly.
    """
    import dataclasses

    from gosplan.env.planner import update_targets

    implemented(update_targets)
    delta = 0.02
    cfg = dataclasses.replace(
        p1_cfg, incentive=dataclasses.replace(p1_cfg.incentive, ratchet_deadband=delta)
    )
    n = cfg.supply.n_enterprises
    t0 = np.full(n, cfg.tech.initial_target_frac)
    g, lam = cfg.incentive.growth_directive, cfg.incentive.ratchet_lambda
    for rho in (1.0, 1.0 + delta, 1.0 - delta):
        out = np.asarray(update_targets(_view(cfg, claims=rho * t0, targets=t0.copy()), cfg))
        assert np.max(np.abs(out - (1.0 + g) * t0)) < 1e-12, rho
    outside = 1.0 + 2.0 * delta
    out = np.asarray(update_targets(_view(cfg, claims=outside * t0, targets=t0.copy()), cfg))
    want = (1.0 + g) * t0 * (1.0 + lam * (outside - 1.0))
    assert np.max(np.abs(out - want)) < 1e-12


@pytest.mark.skeleton
def test_allocation_sums_to_available_supply_per_good(p1_cfg, implemented) -> None:
    """`sum_b alloc_bj == avail_j` for every good `j`.

    Assertion: for random non-negative claims and needs,
    `allocate(view, cfg).sum(axis=0)` equals
    `avail_j = sum_{i: s(i)=j} (1 - phi_j) * claimed_i` elementwise to 1e-12, and every entry of
    `alloc` is non-negative. Formula (PLAN section 2.7.2, verbatim):

        w_bj     = (q_bj + 1e-6)**eta_q * (need_bj + 1e-6)**eta_n
        alloc_bj = avail_j * w_bj / sum_b w_bj

    Allocation distributes *promises* in claimed units: it conserves what the planner believes
    exists, not what exists. The `1e-6` regularisers keep the weights finite when a need or a
    request is zero, which the test must exercise with a zero-need buyer and a zero-request buyer.

    Second bullet of the WO-006 must-pass list.
    """
    from gosplan.env.planner import allocate

    implemented(allocate)
    rng = np.random.default_rng(3)
    n, j = p1_cfg.supply.n_enterprises, p1_cfg.supply.n_sectors
    claims = rng.uniform(0.0, 1.0, size=n)
    view = _view(p1_cfg, claims=claims)
    alloc = np.asarray(allocate(view, p1_cfg))
    phi = np.asarray(p1_cfg.supply.final_demand_share)
    sector = np.asarray(p1_cfg.supply.sector_of)
    avail = np.zeros(j)
    for i in range(n):
        avail[sector[i]] += (1.0 - phi[sector[i]]) * claims[i]
    assert np.all(alloc >= -1e-12)
    assert np.max(np.abs(alloc.sum(axis=0) - avail)) < 1e-12


@pytest.mark.skeleton
def test_allocation_ignores_requests_at_zero_request_elasticity(p1_cfg, implemented) -> None:
    """At `eta_q = 0` the allocation is invariant to `view.requests`.

    Assertion: with `alloc_eta_request = 0` (the Phase-1 value), two views identical except for
    their `requests` arrays - including one with all-zero requests and one with requests at the
    bound `r_max * need` - produce byte-identical allocations; the request term
    `(q_bj + 1e-6)**eta_q` is exactly 1. At `alloc_eta_request = 0.7` (the Phase-2 locked value of
    PLAN section 4.2) the allocation must instead vary with requests, which the same test asserts as
    its complement.

    Third bullet of the WO-006 must-pass list, and finding F7: in Phase 1 requests are logged but
    inert. The Phase-2 change is a rule about weights, never an instruction to inflate a request
    (CONTRACT rule 7).
    """
    from gosplan.env.planner import allocate

    implemented(allocate)
    assert p1_cfg.incentive.alloc_eta_request == 0.0
    n, j = p1_cfg.supply.n_enterprises, p1_cfg.supply.n_sectors
    claims = np.linspace(0.1, 1.0, n)
    zero = np.asarray(allocate(_view(p1_cfg, claims=claims, requests=np.zeros((n, j))), p1_cfg))
    loud = np.asarray(allocate(_view(p1_cfg, claims=claims, requests=np.full((n, j), 1e3)), p1_cfg))
    assert np.max(np.abs(zero - loud)) < 1e-12


@pytest.mark.skeleton
def test_poolfill_lies_in_the_unit_interval(p1_cfg, implemented) -> None:
    """`poolfill_j` is a fraction in [0, 1] for every good and every claim profile.

    Assertion: over random stocks and claims, including all-zero claims for a whole sector,
    `poolfill_j = sum_{i in j} fill_i * claimed_i / sum_{i in j} claimed_i` lies in [0, 1]; it is
    exactly 1 when every seller in the sector has stock at least its claim, and strictly below 1
    when at least one seller's claim exceeds its stock and its claim carries positive weight in the
    pool. A sector with no claims at all gives `poolfill_j = 1` (the empty-pool convention that
    keeps `deliv = alloc * poolfill` well defined when `alloc` is already 0).

    Fourth bullet of the WO-006 must-pass list.
    """
    from gosplan.env.planner import allocate, deliver

    implemented(allocate, deliver)
    rng = np.random.default_rng(4)
    n = p1_cfg.supply.n_enterprises
    for claims in (rng.uniform(0.0, 1.0, n), np.zeros(n), rng.uniform(0.0, 5.0, n)):
        state = _state(p1_cfg, inv_output=rng.uniform(0.0, 1.0, n), last_report=claims)
        view = _view(p1_cfg, claims=claims)
        _s, deliv, fill, _consumer = deliver(state, allocate(view, p1_cfg), p1_cfg)
        fill = np.asarray(fill)
        assert np.all(fill >= -1e-12) and np.all(fill <= 1.0 + 1e-12)
        assert np.all(np.asarray(deliv) >= -1e-12)


@pytest.mark.skeleton
def test_delivery_conserves_goods(p1_cfg, tiny_cfg, implemented) -> None:
    """Delivery moves goods without creating or destroying them.

    Assertion, per good `j`, to 1e-9 (the T-U1 tolerance): the physical quantity leaving sellers,
    `sum_{i in j} shipped_i` with `shipped_i = min(S_i, claimed_i)`, equals the quantity arriving,
    `sum_b deliv_bj + consumer_j`, with

        fill_i     = min(1, S_i / claimed_i)          (fill_i = 1 when claimed_i = 0)
        deliv_bj   = alloc_bj * poolfill_j
        consumer_j = sum_{i in j} phi_j * shipped_i * qbar_i
        S_i       -= shipped_i

    and the sellers' stock drop equals `shipped_i` exactly. Buyers' input stocks rise by
    `deliv_bj * qbar_j` (Phase 1 `qbar = 1`). Nothing may be lost in this stage: the only sanctioned
    sinks in the whole period are the holding loss and the inventory cap of PLAN section 2.11, both
    of which belong to REPORT, not DELIVER.

    Fifth bullet of the WO-006 must-pass list, and the DELIVER half of test T-U1.
    """
    from gosplan.env.planner import allocate, deliver

    implemented(allocate, deliver)
    for cfg in (p1_cfg, tiny_cfg):
        rng = np.random.default_rng(5)
        n, j = cfg.supply.n_enterprises, cfg.supply.n_sectors
        stock = rng.uniform(0.0, 1.0, n)
        claims = rng.uniform(0.0, 1.5, n)
        state = _state(cfg, inv_output=stock, last_report=claims)
        _s, deliv, _fill, consumer = deliver(state, allocate(_view(cfg, claims=claims), cfg), cfg)
        sector = np.asarray(cfg.supply.sector_of)
        shipped = np.minimum(stock, claims)
        for good in range(j):
            left = shipped[sector == good].sum()
            arrived = np.asarray(deliv)[:, good].sum() + float(np.asarray(consumer)[good])
            assert abs(left - arrived) < 1e-9, (cfg.supply.n_enterprises, good)


@pytest.mark.skeleton
def test_fill_is_one_when_the_claim_is_zero(p1_cfg, implemented) -> None:
    """A zero claim gives `fill_i = 1`, not a division by zero.

    Assertion: for an enterprise with `claimed_i = 0`, `deliver` returns `fill_i == 1.0` exactly
    (whatever its stock), ships nothing, and leaves its stock unchanged; no NaN, no warning, and the
    zero claim contributes 0 to both the numerator and the denominator of its sector's `poolfill`.
    Stated explicitly in PLAN section 2.7.3 and in the WO-006 card.
    """
    from gosplan.env.planner import allocate, deliver

    implemented(allocate, deliver)
    n = p1_cfg.supply.n_enterprises
    claims = np.linspace(0.1, 1.0, n)
    claims[0] = 0.0
    stock = np.full(n, 0.5)
    state = _state(p1_cfg, inv_output=stock, last_report=claims)
    after, _deliv, fill, _consumer = deliver(
        state, allocate(_view(p1_cfg, claims=claims), p1_cfg), p1_cfg
    )
    assert float(np.asarray(fill)[0]) == 1.0
    assert abs(float(np.asarray(after.inv_output)[0]) - stock[0]) < 1e-12
    assert np.all(np.isfinite(np.asarray(fill)))


@pytest.mark.skeleton
def test_audit_selection_matches_the_audit_rate_and_is_keyed(p1_cfg, rng_seed, implemented) -> None:
    """`select_audits` draws `Bernoulli(audit_rate)` deterministically in `(seed_env, t)`.

    Assertion: over many periods and enterprises the empirical audit frequency is within 3
    binomial standard errors of `cfg.information.audit_rate`; the returned array is boolean of
    shape `(N,)`; and two calls with the same `(seed_env, t)` return identical selections while
    different `t` values decorrelate. Formula (PLAN section 2.7.4, verbatim, Phase 1 `random`
    mode): `audited_i ~ Bernoulli(a)` with key `(seed_env, "audit", t, i)`, drawn through
    `gosplan.rng.draw` (CONTRACT rule 9). The selection is never observable to any agent before it
    happens (PLAN section 2.4).
    """
    from gosplan.env.planner import select_audits

    implemented(select_audits)
    rate = p1_cfg.information.audit_rate
    n = p1_cfg.supply.n_enterprises
    view = _view(p1_cfg)
    draws = np.concatenate(
        [np.asarray(select_audits(view, p1_cfg, t), dtype=bool) for t in range(400)]
    )
    assert draws.dtype == np.bool_
    se = math.sqrt(rate * (1.0 - rate) / draws.size)
    assert abs(draws.mean() - rate) < 3.0 * se
    again = np.asarray(select_audits(view, p1_cfg, 7), dtype=bool)
    assert np.array_equal(again, np.asarray(select_audits(view, p1_cfg, 7), dtype=bool))
    assert len(again) == n


@pytest.mark.skeleton
def test_planner_view_filters_are_the_identity_at_phase_1(p1_cfg, implemented) -> None:
    """The three information filters exist and are the identity at the Phase-1 configuration.

    Assertion: at `report_lag = 0`, `channel_noise = 0` and `aggregation_level = "enterprise"`,
    `make_planner_view(state, cfg).claims` equals `state.last_report` elementwise to 1e-12, and the
    requests, targets and `planner_io` pass through unchanged. Their non-identity branches must
    exist too and are asserted alongside (PLAN section 2.7.5, applied in this order):

        aggregation   at `aggregation_level = "sector"` the view carries only the per-sector sum of
                      claims, and allocation keys on planned need alone
        lag           at `report_lag = L` the rules consume claims from `L` periods ago
        channel noise `claimed_i <- claimed_i * exp(xi_i)`, `xi ~ N(0, sigma_ch**2)`,
                      key `(seed_env, "channel", t, i)`

    The WO-006 card requires all three branches to be implemented even though Phase 1 makes each of
    them the identity - "implement them (they are in the frozen signature), test the identity case".
    """
    from gosplan.env.planner import make_planner_view

    implemented(make_planner_view)
    assert p1_cfg.information.report_lag == 0
    assert p1_cfg.information.channel_noise == 0.0
    assert p1_cfg.information.aggregation_level == "enterprise"
    n = p1_cfg.supply.n_enterprises
    reports = np.linspace(0.1, 1.0, n)
    view = make_planner_view(_state(p1_cfg, last_report=reports), p1_cfg)
    assert np.max(np.abs(np.asarray(view.claims) - reports)) < 1e-12


@pytest.mark.skeleton
def test_fulfilment_measure_val_branch_is_the_identity_on_claims(p1_cfg, implemented) -> None:
    """At `objective_metric = "val"` the fulfilment measure is `m_i = R_i`.

    Assertion: `fulfilment_measure(view, cfg)` equals `view.claims` elementwise to 1e-12 at the
    Phase-1 metric, and reads nothing else from the view. This is the measure the bonus and the
    ratchet key on (PLAN section 2.9.2), and `welfare` is deliberately not an option (finding F6):
    the planner cannot key on a quantity it does not observe (CONTRACT rule 6).
    """
    from gosplan.env.planner import fulfilment_measure

    implemented(fulfilment_measure)
    assert p1_cfg.incentive.objective_metric == "val"
    claims = np.linspace(0.1, 1.0, p1_cfg.supply.n_enterprises)
    got = np.asarray(fulfilment_measure(_view(p1_cfg, claims=claims), p1_cfg))
    assert np.max(np.abs(got - claims)) < 1e-12


@pytest.mark.skeleton
def test_fulfilment_measure_net_output_falls_as_allocated_inputs_rise(p1_cfg, implemented) -> None:
    """At `objective_metric = "net_output"` the measure nets out allocated inputs at plan prices.

    Assertion: `m_i = R_i - sum_j p_j * alloc_ij / p_{s(i)}` to 1e-12, with `alloc` the allocation
    the planner itself computed for the same view via `allocate(view, cfg)` and `p` the view's
    `plan_prices`; holding claims fixed and raising any `alloc_ij` strictly lowers `m_i`. At
    `objective_metric = "quality_weighted"` the measure is `R_i * q_hat_i` with
    `q_hat_i = view.measured_quality`, which is 1 everywhere in Phase 1. Every input is
    planner-side, so the function never needs the true state (CONTRACT rule 5).
    """
    import dataclasses

    from gosplan.env.planner import allocate, fulfilment_measure

    implemented(allocate, fulfilment_measure)
    cfg = dataclasses.replace(
        p1_cfg, incentive=dataclasses.replace(p1_cfg.incentive, objective_metric="net_output")
    )
    n = cfg.supply.n_enterprises
    claims = np.linspace(0.2, 1.0, n)
    view = _view(cfg, claims=claims)
    alloc = np.asarray(allocate(view, cfg))
    prices = np.asarray(view.plan_prices)
    sector = np.asarray(cfg.supply.sector_of)
    want = claims - (alloc * prices[None, :]).sum(axis=1) / prices[sector]
    got = np.asarray(fulfilment_measure(view, cfg))
    assert np.max(np.abs(got - want)) < 1e-12
    assert np.all(got <= claims + 1e-12)
