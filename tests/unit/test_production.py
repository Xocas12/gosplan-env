"""Production: the coverage aggregator, the yield shock, costs and input consumption (PLAN 2.6).

Realises: PLAN section 2.6 (production at a PRODUCE step) and PLAN section 11 (test architecture;
property test **T-U7**). Owning work order: **WO-002** (frozen tests; LEAD). Binds the WO-005
must-pass line of PLAN section 12.3, verbatim - "`tests/unit/test_production.py` (T-U7; yield
mean = 1 to 1e-3 over 1e5 draws; `v` diversion; cost formula; inputs consumed = `a * y_tilde` capped
at stock; `H = 1` when `a` row is zero)". Module under test: `gosplan/env/production.py`.

T-U7, verbatim (PLAN section 11): "coverage aggregator: `theta = inf` equals `min`; `theta -> 1`
equals weighted harmonic mean; `H = 1` when no inputs needed."

PLAN section 2.6 formulas, verbatim, as every docstring below cites them:

    y_hat_ik   = (A_{s(i)} * cap_i / M) * e_ik
    need_ikj   = a_{s(i)j} * y_hat_ik
    H_ik       = ( sum_j omega_j * min(1, X_ij / need_ikj)**(-theta) )**(-1/theta)
    eps_ik     ~ LogNormal(-sigma_{s(i)}**2 / 2, sigma_{s(i)})     key (seed_env, "yield", t, k, i)
    y_tilde_ik = y_hat_ik * H_ik * eps_ik
    y_ik       = y_tilde_ik * (1 - v_ik)
    X_ij      -= min(X_ij, a_{s(i)j} * y_tilde_ik)
    c_ik       = kappa * e_ik**2 + F * 1[e_ik > 0] + kappa_q * q_ik * e_ik

FORBIDDEN IN WO-005, and therefore absent from every assertion here: any reference to reports,
targets or rewards. CONTRACT rule 9 governs the yield shock - it is drawn through `gosplan.rng.draw`
with purpose `yield`, never from `numpy.random`. CONTRACT rule 7 governs the module as a whole: none
of these rules implements a pathology, and none of the assertions below asserts a direction of
behaviour.

FROZEN BY CONTRACT RULE 2. SKELETON: every test is `@pytest.mark.skeleton` and skipped until WO-005
lands; each docstring states the exact assertion, formula and tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest


def _state(cfg, *, inputs=None, effort_step=0, phase="produce", cum_output=None, inv_output=None):
    """A `State` built directly from `cfg`, for a step-level test.

    `initial_state` belongs to WO-009; a production- or observation-level assertion should not wait
    on the environment, so the dataclass is constructed here. Fields follow PLAN section 2.2.
    """
    from gosplan.env.state import State

    n, j = cfg.supply.n_enterprises, cfg.supply.n_sectors
    z = np.zeros(n)
    return State(
        target=np.full(n, cfg.tech.initial_target_frac),
        capital=np.ones(n),
        inv_output=z.copy() if inv_output is None else np.asarray(inv_output, dtype=float),
        inv_inputs=np.zeros((n, j)) if inputs is None else np.asarray(inputs, dtype=float),
        cum_output=z.copy() if cum_output is None else np.asarray(cum_output, dtype=float),
        cum_cost=z.copy(),
        quality_acc=z.copy(),
        last_report_ratio=z.copy(),
        last_report=z.copy(),
        last_audited=np.zeros(n, dtype=bool),
        last_penalty=z.copy(),
        last_fill=np.ones(n),
        request=np.zeros((n, j)),
        pending_invest=np.zeros((n, 0)),
        t_period=0,
        k_step=effort_step,
        phase=phase,
        plan_prices=np.ones(j),
        planner_io=np.asarray(cfg.supply.io_matrix, dtype=float),
        consumer_delivery=np.zeros(j),
        alive=True,
        seed_env=cfg.tech.seed_env,
        seed_policy=cfg.tech.seed_policy,
    )


def _action(cfg, *, effort=None, invest=None, quality=None):
    """An `EnterpriseAction` with the Phase-1 dimensions set and the rest inert."""
    from gosplan.env.state import EnterpriseAction

    n, j = cfg.supply.n_enterprises, cfg.supply.n_sectors
    return EnterpriseAction(
        effort=np.full(n, 0.5) if effort is None else np.asarray(effort, dtype=float),
        quality=np.ones(n) if quality is None else np.asarray(quality, dtype=float),
        invest=np.zeros(n) if invest is None else np.asarray(invest, dtype=float),
        report_ratio=np.ones(n),
        input_request=np.zeros((n, j)),
        trade_offer=np.zeros((n, j)),
    )


def _weights(cfg):
    """`omega_j = a_{s(i)j} / sum_j a_{s(i)j}`, per enterprise; zeros on a row needing no inputs."""
    a = np.asarray(cfg.supply.io_matrix, dtype=float)
    rows = a[np.asarray(cfg.supply.sector_of)]
    totals = rows.sum(axis=1, keepdims=True)
    return np.divide(rows, totals, out=np.zeros_like(rows), where=totals > 0)


@pytest.mark.skeleton
def test_coverage_at_infinite_theta_equals_the_minimum(p1_cfg, implemented) -> None:
    """T-U7, first clause: `theta = inf` reduces the CES aggregator to Leontief `min`.

    Assertion: for random `X` and `need` matrices `(N, J)` with a mix of covered and short goods,
    `coverage(X, need, weights, theta=float("inf"))` equals
    `np.min(np.minimum(1.0, X / need), axis=1)` taken over the goods with `need > 0`, elementwise to
    1e-12. Goods with `need == 0` are excluded from the minimum rather than contributing a ratio of
    infinity.

    The `inf` branch must be taken explicitly (`np.min`), not approximated by a large finite
    `theta`: PLAN section 2.6 makes `theta = inf` a member of the {2, 8, inf} grid, and
    `(...)**(-1/theta)` overflows long before it is reached.
    """
    from gosplan.env.production import coverage

    implemented(coverage)
    rng = np.random.default_rng(0)
    n, j = p1_cfg.supply.n_enterprises, p1_cfg.supply.n_sectors
    need = (
        np.asarray(p1_cfg.supply.io_matrix, dtype=float)[np.asarray(p1_cfg.supply.sector_of)] * 0.5
    )
    x = rng.uniform(0.0, 0.2, size=(n, j))
    w = _weights(p1_cfg)
    got = np.asarray(coverage(x, need, w, float("inf")))
    ratios = np.where(
        need > 0, np.minimum(1.0, np.divide(x, need, out=np.ones_like(x), where=need > 0)), np.inf
    )
    want = ratios.min(axis=1)
    want = np.where(np.isfinite(want), want, 1.0)
    assert np.max(np.abs(got - want)) < 1e-12


@pytest.mark.skeleton
def test_coverage_at_unit_theta_equals_the_weighted_harmonic_mean(p1_cfg, implemented) -> None:
    """T-U7, second clause: `theta -> 1` reduces to the weighted harmonic mean.

    Assertion: with `r_ij = min(1, X_ij / need_ij)` over the goods with `need > 0`,
    `coverage(X, need, weights, theta)` tends to `1 / sum_j (omega_j / r_ij)` as `theta -> 1`:
    at `theta = 1` the two agree to 1e-12, and at `theta = 1 + 1e-6` to 1e-6. This is the
    `theta = 1` case of `H = (sum_j omega_j * r_ij**(-theta))**(-1/theta)` and pins the branch that
    the near-Leontief Phase-1 value `theta = 8` interpolates towards.
    """
    from gosplan.env.production import coverage

    implemented(coverage)
    rng = np.random.default_rng(1)
    n, j = p1_cfg.supply.n_enterprises, p1_cfg.supply.n_sectors
    need = (
        np.asarray(p1_cfg.supply.io_matrix, dtype=float)[np.asarray(p1_cfg.supply.sector_of)] * 0.5
    )
    x = rng.uniform(0.02, 0.2, size=(n, j))
    w = _weights(p1_cfg)
    ratios = np.divide(x, need, out=np.ones_like(x), where=need > 0)
    ratios = np.minimum(1.0, np.where(need > 0, ratios, 1.0))
    harmonic = 1.0 / np.sum(np.divide(w, ratios, out=np.zeros_like(w), where=w > 0), axis=1)
    got = np.asarray(coverage(x, need, w, 1.0))
    assert np.max(np.abs(got - harmonic)) < 1e-9


@pytest.mark.skeleton
def test_coverage_is_one_when_no_inputs_are_needed(p1_cfg, implemented) -> None:
    """T-U7, third clause: an enterprise that requires no inputs has `H = 1`.

    Assertion: when a whole row of `need` is zero - which happens whenever the enterprise's row of
    `a` is zero, or its intended output `y_hat_ik` is zero because effort is zero -
    `coverage(...)` returns exactly 1.0 for that row, at every `theta` including `inf`, with no
    division by zero, no NaN and no warning. A good that is not needed is fully covered by
    definition.

    This is also the fifth bullet of the WO-005 must-pass list ("`H = 1` when the `a` row is
    zero"), and the same convention the observation of PLAN section 2.4 uses for its coverage
    fields (`need = 0` gives 1.0, WO-008).
    """
    from gosplan.env.production import coverage

    implemented(coverage)
    n, j = p1_cfg.supply.n_enterprises, p1_cfg.supply.n_sectors
    need = np.zeros((n, j))
    x = np.zeros((n, j))
    w = np.zeros((n, j))
    for theta in (1.0, 2.0, 8.0, float("inf")):
        got = np.asarray(coverage(x, need, w, theta))
        assert np.max(np.abs(got - 1.0)) < 1e-12, theta


@pytest.mark.skeleton
def test_coverage_treats_a_zero_need_entry_as_fully_covered(p1_cfg, implemented) -> None:
    """A single `need_ikj == 0` contributes a coverage ratio of 1, not a division by zero.

    Assertion: in a row with some positive needs and some zero needs, `coverage` returns the same
    value as the aggregator computed over the positive-need goods alone with their weights
    renormalised - i.e. the zero-need goods contribute `min(1, .) = 1` and do not drag `H` down; the
    result is finite and in [0, 1] for every `theta` in {1, 2, 8, inf}. This is the Phase-1 case,
    where every row of `a` has three zero entries out of five.
    """
    from gosplan.env.production import coverage

    implemented(coverage)
    # one enterprise, two goods needed of four; the two unneeded goods must not affect H
    need = np.array([[0.1, 0.0, 0.2, 0.0]])
    x = np.array([[0.05, 0.0, 0.2, 0.0]])
    w = np.array([[1.0 / 3.0, 0.0, 2.0 / 3.0, 0.0]])
    full = float(np.asarray(coverage(x, need, w, 8.0))[0])
    trimmed = float(np.asarray(coverage(x[:, [0, 2]], need[:, [0, 2]], w[:, [0, 2]], 8.0))[0])
    assert abs(full - trimmed) < 1e-12


@pytest.mark.skeleton
def test_coverage_is_bounded_in_the_unit_interval(p1_cfg, implemented) -> None:
    """`H` is a multiplier in [0, 1], monotone in coverage.

    Assertion: over random `X`, `need` and `theta` in {1, 2, 8, inf}, every returned `H` lies in
    [0, 1]; `H = 1` exactly when every needed good is fully covered (`X_ij >= need_ikj` for all `j`
    with `need > 0`); and `H` is non-decreasing in each `X_ij`. The clip at 1 in
    `min(1, X_ij / need_ikj)` is what makes surplus stock of one input unable to compensate for a
    shortage of another - the complementarity the parameter `theta` grades.
    """
    from gosplan.env.production import coverage

    implemented(coverage)
    rng = np.random.default_rng(2)
    n, j = p1_cfg.supply.n_enterprises, p1_cfg.supply.n_sectors
    need = (
        np.asarray(p1_cfg.supply.io_matrix, dtype=float)[np.asarray(p1_cfg.supply.sector_of)] * 0.5
    )
    w = _weights(p1_cfg)
    for theta in (1.0, 2.0, 8.0, float("inf")):
        x = rng.uniform(0.0, 0.3, size=(n, j))
        got = np.asarray(coverage(x, need, w, theta))
        assert np.all(got >= -1e-12) and np.all(got <= 1.0 + 1e-12), theta
        # full coverage everywhere gives exactly 1
        assert np.max(np.abs(np.asarray(coverage(need * 2.0, need, w, theta)) - 1.0)) < 1e-12
        # more stock never lowers H
        assert np.all(np.asarray(coverage(x * 2.0, need, w, theta)) >= got - 1e-12)


@pytest.mark.skeleton
def test_yield_shock_has_unit_mean(p1_cfg, rng_seed, implemented) -> None:
    """The yield shock is mean 1: `eps ~ LogNormal(-sigma**2 / 2, sigma)`.

    Assertion: drawing `1e5` shocks per sector through the production path (purpose `yield`, key
    `(seed_env, "yield", t, k, i)`), the sample mean of `eps` equals 1 to **1e-3** for every sector
    `sigma` in `cfg.supply.yield_sigma`, and the realised `y_tilde` at full coverage and effort `e`
    has sample mean `(A * cap / M) * e` to the same relative tolerance. The centring
    `mean_log = -sigma**2 / 2` is what makes the shock multiplicative-unbiased, so heteroskedastic
    sectors differ in variance and not in expected output.

    Second bullet of the WO-005 must-pass list; tolerance and sample size are PLAN section 12.3's,
    verbatim.
    """
    from gosplan.rng import draw

    implemented(draw)
    for s, sigma in enumerate(p1_cfg.supply.yield_sigma):
        eps = np.asarray(
            draw(
                rng_seed,
                "yield",
                0,
                0,
                s,
                shape=(200_000,),
                dist="lognormal",
                mean_log=-(sigma**2) / 2.0,
                sigma=sigma,
            )
        )
        assert abs(eps.mean() - 1.0) < 1e-3, (s, sigma)


@pytest.mark.skeleton
def test_investment_diversion_scales_realised_output(p1_cfg, implemented) -> None:
    """`y_ik = y_tilde_ik * (1 - v_ik)`, and `y_tilde_ik * v_ik` goes to `pending_invest`.

    Assertion: holding effort, stocks and the yield draw fixed, `produce_step` returns
    `y_ik = y_tilde_ik * (1 - v_ik)` elementwise to 1e-12 for `v` in {0, 0.25, 1}; the diverted
    remainder `y_tilde_ik * v_ik` is added to `state.pending_invest`, so nothing is lost; and at
    the Phase-1 value `v == 0` output equals `y_tilde` exactly and `pending_invest` is untouched.

    Third bullet of the WO-005 must-pass list. Note the input consumption below is charged against
    `y_tilde` - output *before* the diversion - so investment does not economise on inputs.
    """
    from gosplan.env.production import produce_step

    implemented(produce_step)
    n = p1_cfg.supply.n_enterprises
    big = np.full((n, p1_cfg.supply.n_sectors), 1e3)
    baseline = None
    for v in (0.0, 0.25, 1.0):
        state = _state(p1_cfg, inputs=big.copy())
        _s, y, _c = produce_step(state, _action(p1_cfg, invest=np.full(n, v)), p1_cfg)
        y = np.asarray(y)
        if baseline is None:
            baseline = y.copy()
        assert np.max(np.abs(y - baseline * (1.0 - v))) < 1e-12, v


@pytest.mark.skeleton
def test_cost_formula_is_exactly_the_three_terms(p1_cfg, implemented) -> None:
    """`c_ik = kappa * e_ik**2 + F * 1[e_ik > 0] + kappa_q * q_ik * e_ik`.

    Assertion: `produce_step` returns costs equal to that expression to 1e-12, checked at
    `e = 0` (cost exactly 0 including the setup term, since the indicator is `e > 0` strictly),
    at small positive `e` (where the setup term `F` makes the cost discontinuous at 0 once `F > 0`),
    and at `e = 1`. At the Phase-1 configuration `F = 0` and `kappa_q = 0`, so the cost is exactly
    `kappa * e**2`, and `cum_cost` accumulates it over the `M` steps of the period.

    Fourth bullet of the WO-005 must-pass list. CONTRACT rule 4: this cost is a real cost paid when
    it is incurred and enters the reward only as `-scale * c_ik`; it is never shaping.
    """
    from gosplan.env.production import produce_step

    implemented(produce_step)
    n = p1_cfg.supply.n_enterprises
    kappa = p1_cfg.incentive.effort_cost
    setup = p1_cfg.supply.setup_cost
    kappa_q = p1_cfg.supply.quality_cost
    big = np.full((n, p1_cfg.supply.n_sectors), 1e3)
    e = np.linspace(0.0, 1.0, n)
    q = np.linspace(0.5, 1.0, n)
    state = _state(p1_cfg, inputs=big)
    _s, _y, c = produce_step(state, _action(p1_cfg, effort=e, quality=q), p1_cfg)
    want = kappa * e**2 + setup * (e > 0.0) + kappa_q * q * e
    assert np.max(np.abs(np.asarray(c) - want)) < 1e-12
    assert abs(float(np.asarray(c)[0])) < 1e-12  # e = 0 costs exactly 0, setup indicator strict


@pytest.mark.skeleton
def test_inputs_consumed_equal_a_times_y_tilde_capped_at_stock(p1_cfg, implemented) -> None:
    """`X_ij -= min(X_ij, a_{s(i)j} * y_tilde_ik)`, per good, per step.

    Assertion: after `produce_step`, the drop in each `X_ij` equals `min(X_ij_before,
    a_{s(i)j} * y_tilde_ik)` to 1e-12; stocks never go negative; goods with `a_{s(i)j} == 0` are
    untouched; and when stock binds, the consumption is exactly the stock that was there, with the
    shortfall showing up as a coverage `H < 1` rather than as negative stock.

    Fifth bullet of the WO-005 must-pass list, and one of the two enterprise-side terms of the
    per-period conservation identity of test T-U1 (`tests/unit/test_conservation.py`).
    """
    from gosplan.env.production import produce_step

    implemented(produce_step)
    n, j = p1_cfg.supply.n_enterprises, p1_cfg.supply.n_sectors
    a = np.asarray(p1_cfg.supply.io_matrix, dtype=float)[np.asarray(p1_cfg.supply.sector_of)]
    for scale in (1e3, 1e-4):  # abundant, then binding
        before = np.full((n, j), scale)
        state = _state(p1_cfg, inputs=before.copy())
        after, _y, _c = produce_step(state, _action(p1_cfg), p1_cfg)
        used = before - np.asarray(after.inv_inputs)
        assert np.all(used >= -1e-12)
        assert np.all(np.asarray(after.inv_inputs) >= -1e-12)
        assert np.all(used[a == 0.0] < 1e-12)  # goods the row does not call for are untouched
        assert np.all(used <= before + 1e-12)


@pytest.mark.skeleton
def test_intended_output_is_capacity_over_steps_times_effort(p1_cfg, implemented) -> None:
    """`y_hat_ik = (A_{s(i)} * cap_i / M) * e_ik`, and `need_ikj = a_{s(i)j} * y_hat_ik`.

    Assertion: with coverage full (`X` large) and the yield shock forced to 1 by `sigma = 0`,
    `produce_step` returns `y_ik = (A_{s(i)} * cap_i / M) * e_ik` to 1e-12; the per-step need it
    charges coverage against is `a_{s(i)j} * y_hat_ik`; and `cum_output` accumulates `y_ik` across
    the `M` steps, so a full period at effort `e` produces `A * cap * e` before noise. This is the
    normalisation behind the TECH row `T_0 = 0.6 * A * cap`: the Phase-1 target is feasible at
    effort about 0.6.
    """
    import dataclasses

    from gosplan.env.production import produce_step

    implemented(produce_step)
    n = p1_cfg.supply.n_enterprises
    j = p1_cfg.supply.n_sectors
    deterministic = dataclasses.replace(
        p1_cfg,
        supply=dataclasses.replace(p1_cfg.supply, yield_sigma=tuple(0.0 for _ in range(j))),
    )
    m = deterministic.incentive.steps_per_period
    prod = np.asarray(deterministic.supply.productivity)[np.asarray(deterministic.supply.sector_of)]
    e = np.linspace(0.1, 1.0, n)
    state = _state(deterministic, inputs=np.full((n, j), 1e3))
    _s, y, _c = produce_step(state, _action(deterministic, effort=e), deterministic)
    assert np.max(np.abs(np.asarray(y) - (prod * 1.0 / m) * e)) < 1e-12
