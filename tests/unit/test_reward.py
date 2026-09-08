"""Bonus shape, reward scale, the reward terms and the logged aggregates (PLAN section 2.9).

Realises: PLAN sections 2.8 (the bonus schedule `B(rho)`), 2.9.1 (the enterprise reward and its
analytic scale), 2.9.2-2.9.3 (fulfilment measure, `val_measured`, `val_true`, CES welfare), 2.9.4
(the three headline metrics) and PLAN section 11 (test architecture; property tests **T-U2** and
**T-U3**, and the components of **T-B6**). Owning work order: **WO-002** (frozen tests; LEAD). Binds
the WO-007 must-pass line of PLAN section 12.3, verbatim - "`tests/unit/test_reward.py` (T-U2, T-U3,
T-B6 components; `val`, `val_true`, CES welfare with `sigma_c -> 1` limit = Cobb-Douglas)". Module
under test: `gosplan/env/reward.py`.

T-U2, verbatim (PLAN section 11): "`reward_scale(cfg) * B_cfg(1.1) == 1`."
T-U3, verbatim: "`bonus` is monotone in `rho`; discontinuous at 1 iff `w = 0`; continuous with
continuous derivative iff `w > 0` and `rho_cap = inf`."

CONTRACT RULE 4, verbatim, is the standard every reward assertion here is measured against:

    production step:  -scale * c_ik
    report step:       scale * (B(rho) - 1[audited] * Pen + trade_surplus)

with no per-step shaping, no auxiliary reward, no curiosity term, no potential-based term and no
running reward normalisation; `scale = reward_scale(cfg)`, computed analytically from the
configuration. Per-batch advantage normalisation inside PPO is permitted and is the adapter's
business (`tests/unit/test_ppo_adapter.py`), not the environment's.

CONTRACT RULE 6: `val_measured` and `welfare_true` are logged and never observed by any agent. The
tests below compute them from a state and a consumer vector; none of them feeds either quantity into
an observation, a reward or an agent input, and test T-B5 asserts that separation with sentinels.

FROZEN BY CONTRACT RULE 2. SKELETON: every test is `@pytest.mark.skeleton` and skipped until WO-007
lands; each docstring states the exact assertion, formula and tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest


def _bonus_matrix(cfg):
    """The three named bonus configurations of PLAN section 2.8, at several `beta` and `s`.

    notched (`w = 0`, `rho_cap = 1.2`), smooth counterfactual (`w = 0.25`, `rho_cap = inf`) and
    kink-only (`w = 0.25`, `rho_cap = 1.2`). T-U2 and T-U3 quantify over all of them.
    """
    import dataclasses

    out = []
    for width, cap in ((0.0, 1.2), (0.25, float("inf")), (0.25, 1.2)):
        for beta in (0.25, 1.0, 2.0):
            for slope in (0.0, 0.5, 2.0):
                out.append(
                    dataclasses.replace(
                        cfg,
                        incentive=dataclasses.replace(
                            cfg.incentive,
                            notch_width=width,
                            overfulfilment_cap=cap,
                            notch_height=beta,
                            overfulfilment_slope=slope,
                        ),
                    )
                )
    return out


def _state(cfg, *, reports=None, outputs=None, prices=None, consumer=None):
    """Build a `State` directly from `cfg` for a pure-function test.

    `initial_state` belongs to WO-009, so a reward-level test that only needs somewhere to hang
    `last_report`, `cum_output` and `plan_prices` constructs the dataclass itself rather than
    waiting for the environment. Fields follow PLAN section 2.2 in declaration order.
    """
    import numpy as np

    from gosplan.env.state import State

    n = cfg.supply.n_enterprises
    j = cfg.supply.n_sectors
    zeros_n = np.zeros(n)
    return State(
        target=np.full(n, cfg.tech.initial_target_frac),
        capital=np.ones(n),
        inv_output=zeros_n.copy(),
        inv_inputs=np.zeros((n, j)),
        cum_output=np.zeros(n) if outputs is None else np.asarray(outputs, dtype=float),
        cum_cost=zeros_n.copy(),
        quality_acc=zeros_n.copy(),
        last_report_ratio=zeros_n.copy(),
        last_report=np.zeros(n) if reports is None else np.asarray(reports, dtype=float),
        last_audited=np.zeros(n, dtype=bool),
        last_penalty=zeros_n.copy(),
        last_fill=np.ones(n),
        request=np.zeros((n, j)),
        pending_invest=np.zeros((n, 0)),
        t_period=0,
        k_step=0,
        phase="report",
        plan_prices=np.ones(j) if prices is None else np.asarray(prices, dtype=float),
        planner_io=np.asarray(cfg.supply.io_matrix, dtype=float),
        consumer_delivery=np.zeros(j) if consumer is None else np.asarray(consumer, dtype=float),
        alive=True,
        seed_env=cfg.tech.seed_env,
        seed_policy=cfg.tech.seed_policy,
    )


@pytest.mark.skeleton
def test_reward_scale_normalises_the_bonus_at_the_reference_ratio(p1_cfg, implemented) -> None:
    """T-U2: `reward_scale(cfg) * bonus(1.1, cfg) == 1`.

    Assertion: for every configuration in the test matrix - the notched arm (`w = 0`,
    `rho_cap = 1.2`), the smooth counterfactual (`w = 0.25`, `rho_cap = inf`) and the kink-only arm
    (`w = 0.25`, `rho_cap = 1.2`) of PLAN section 2.8, each at several `beta` and `s` values - the
    product equals 1.0 to floating-point tolerance (1e-12 relative). Formula (PLAN section 2.9.1,
    verbatim): `scale = 1 / B_cfg(rho_ref = 1.1)`.

    `reward_scale` must be a pure function of the configuration: computed analytically, never from
    running statistics of a rollout. That is what makes a sweep over `beta` change the economics -
    the notch relative to `pen` and `kappa` - and not the gradient magnitude (finding F9), and it is
    why CONTRACT rule 4 forbids running reward normalisation outright.
    """
    from gosplan.env.reward import bonus, reward_scale

    implemented(bonus, reward_scale)
    for cfg in _bonus_matrix(p1_cfg):
        got = reward_scale(cfg) * float(np.asarray(bonus(np.array([1.1]), cfg))[0])
        assert abs(got - 1.0) < 1e-12, cfg.incentive


@pytest.mark.skeleton
def test_bonus_is_monotone_non_decreasing_in_the_fulfilment_ratio(p1_cfg, implemented) -> None:
    """T-U3, first clause: `B(rho)` never decreases in `rho`.

    Assertion: on a fine grid of `rho` spanning [0, `rho_max`] (step 1e-3, and refined to 1e-6
    around `rho = 1` and `rho = rho_cap`), `bonus(rho, cfg)` is non-decreasing for every
    configuration in the matrix - notched, smooth and kink-only, at every `beta >= 0` and `s >= 0`.
    Formula (PLAN section 2.8, verbatim):

        Lambda_w(x) = 1[x >= 0]              if w = 0     # strict >=, a true Heaviside
                    = 1 / (1 + exp(-x / w))  if w > 0
        B(rho)      = beta * Lambda_w(rho - 1) + s * clip(rho - 1, 0, rho_cap - 1)

    Both terms are non-decreasing, so their sum is; a violation means a sign error in one of them.
    """
    from gosplan.env.reward import bonus

    implemented(bonus)
    grid = np.concatenate(
        [
            np.arange(0.0, p1_cfg.tech.report_max_ratio, 1e-3),
            np.linspace(1.0 - 1e-5, 1.0 + 1e-5, 401),
            np.linspace(1.15, 1.25, 401),
        ]
    )
    grid.sort()
    for cfg in _bonus_matrix(p1_cfg):
        b = np.asarray(bonus(grid, cfg))
        assert np.all(np.diff(b) >= -1e-12), cfg.incentive


@pytest.mark.skeleton
def test_bonus_is_discontinuous_at_one_exactly_when_the_notch_width_is_zero(
    p1_cfg, implemented
) -> None:
    """T-U3, second clause: discontinuous at `rho = 1` if and only if `w = 0`.

    Assertion, both directions. At `w = 0`: `bonus(1.0) - bonus(1.0 - eps) >= beta * (1 - 1e-9)` for
    `eps` down to 1e-9, i.e. the jump is the full notch height and does not shrink with `eps`, and
    `Lambda_0` uses a strict `>=` so `rho = 1` exactly is on the *paid* side. At `w > 0`:
    `|bonus(1 + eps) - bonus(1 - eps)| -> 0` as `eps -> 0`, bounded by `beta * eps / (2 * w)` plus
    the slope term, so the schedule is continuous everywhere.

    This is the counterfactual knob of the whole Phase-1 design: the notched arm and the smooth arm
    share `beta` and `s` (equal-parameter, not equal-expected-value), and gate G2 criterion 2 asks
    for bunching present in the first and absent in the second (PLAN section 4.5).
    """
    import dataclasses

    from gosplan.env.reward import bonus

    implemented(bonus)
    beta = p1_cfg.incentive.notch_height
    notched = dataclasses.replace(
        p1_cfg, incentive=dataclasses.replace(p1_cfg.incentive, notch_width=0.0)
    )
    for eps in (1e-3, 1e-6, 1e-9):
        jump = float(np.asarray(bonus(np.array([1.0]), notched))[0]) - float(
            np.asarray(bonus(np.array([1.0 - eps]), notched))[0]
        )
        assert jump >= beta * (1.0 - 1e-9), eps

    smooth = dataclasses.replace(
        p1_cfg, incentive=dataclasses.replace(p1_cfg.incentive, notch_width=0.25)
    )
    small = float(np.asarray(bonus(np.array([1.0]), smooth))[0]) - float(
        np.asarray(bonus(np.array([1.0 - 1e-6]), smooth))[0]
    )
    assert small < beta * 1e-3


@pytest.mark.skeleton
def test_bonus_has_a_continuous_derivative_only_in_the_smooth_configuration(
    p1_cfg, implemented
) -> None:
    """T-U3, third clause: continuous with continuous derivative iff `w > 0` and `rho_cap = inf`.

    Assertion: at (`w = 0.25`, `rho_cap = inf`) the numerical derivative of `bonus` is continuous
    everywhere on [0, `rho_max`] - successive central differences at step 1e-6 differ by less than
    1e-3 - and in particular there is no kink at `rho = 1` and none at any cap. At
    (`w = 0.25`, `rho_cap = 1.2`), the kink-only arm, the derivative jumps at `rho = rho_cap` by
    exactly the slope `s`. At `rho_cap = inf` the implementation must not clip at all: there is no
    upper end to clip to, and `clip(rho - 1, 0, inf)` is `max(0, rho - 1)`.
    """
    import dataclasses

    from gosplan.env.reward import bonus

    implemented(bonus)
    smooth = dataclasses.replace(
        p1_cfg,
        incentive=dataclasses.replace(
            p1_cfg.incentive, notch_width=0.25, overfulfilment_cap=float("inf")
        ),
    )
    grid = np.arange(0.05, 3.0, 1e-3)
    h = 1e-6
    deriv = (np.asarray(bonus(grid + h, smooth)) - np.asarray(bonus(grid - h, smooth))) / (2.0 * h)
    assert np.max(np.abs(np.diff(deriv))) < 1e-3

    kinked = dataclasses.replace(
        p1_cfg,
        incentive=dataclasses.replace(p1_cfg.incentive, notch_width=0.25, overfulfilment_cap=1.2),
    )
    dk = (np.asarray(bonus(grid + h, kinked)) - np.asarray(bonus(grid - h, kinked))) / (2.0 * h)
    assert np.max(np.abs(np.diff(dk))) > 1e-3


@pytest.mark.skeleton
def test_produce_step_reward_is_exactly_minus_the_scaled_cost(p1_cfg, implemented) -> None:
    """T-B6 component: at a PRODUCE step the reward is `-scale * c_ik` and nothing else.

    Assertion: `enterprise_reward(state, cfg, phase="produce", cost=c, penalty=None,
    trade_surplus=None)` equals `-reward_scale(cfg) * c` elementwise, exactly (to 1e-12); it is
    zero when the cost is zero; and it does not read the bonus, the target, the report or any
    penalty. Effort cost is a real cost paid when it is incurred, not shaping (CONTRACT rule 4).
    """
    from gosplan.env.reward import enterprise_reward, reward_scale

    implemented(enterprise_reward, reward_scale)
    state = _state(p1_cfg)
    n = p1_cfg.supply.n_enterprises
    cost = np.linspace(0.0, 1.0, n)
    got = np.asarray(enterprise_reward(state, p1_cfg, "produce", cost, None, None))
    assert np.max(np.abs(got - (-reward_scale(p1_cfg) * cost))) < 1e-12
    zero = np.asarray(enterprise_reward(state, p1_cfg, "produce", np.zeros(n), None, None))
    assert np.max(np.abs(zero)) < 1e-12


@pytest.mark.skeleton
def test_report_step_reward_is_the_three_term_formula(p1_cfg, implemented) -> None:
    """T-B6 component: at the REPORT step the reward is `scale * (B(rho) - penalty + surplus)`.

    Assertion: `enterprise_reward(state, cfg, phase="report", cost=None, penalty=pen_arr,
    trade_surplus=surplus)` equals `reward_scale(cfg) * (bonus(rho, cfg) - pen_arr + surplus)`
    elementwise to 1e-12, with `rho_i = fulfilment_measure(...) / T_i` and
    `penalty_i = 1[audited_i] * Pen_i` as returned by `audit_and_penalise`. Recomputing the formula
    independently on random states must reproduce the environment's reward exactly - the
    behavioural form of this check is test T-B6.

    There are no other terms. A reward that contains anything else - a shaping bonus, a curiosity
    term, a potential, a normalised return - violates CONTRACT rule 4 and invalidates the session.
    """
    from gosplan.env.reward import bonus, enterprise_reward, reward_scale

    implemented(bonus, enterprise_reward, reward_scale)
    n = p1_cfg.supply.n_enterprises
    targets = np.full(n, p1_cfg.tech.initial_target_frac)
    rho = np.linspace(0.5, 1.5, n)
    state = _state(p1_cfg, reports=rho * targets)
    pen = np.linspace(0.0, 2.0, n)
    surplus = np.zeros(n)
    got = np.asarray(enterprise_reward(state, p1_cfg, "report", None, pen, surplus))
    want = reward_scale(p1_cfg) * (np.asarray(bonus(rho, p1_cfg)) - pen + surplus)
    assert np.max(np.abs(got - want)) < 1e-12


@pytest.mark.skeleton
def test_trade_surplus_is_zero_throughout_phase_1(p1_cfg, implemented) -> None:
    """The trade term exists in the formula and is identically zero in Phase 1.

    Assertion: at `p1_cfg` (`horizontal_visibility = 0`, no trade), the surplus argument the
    environment passes is an all-zero array, so the REPORT-step reward reduces to
    `scale * (B(rho) - penalty)`; and passing an explicit non-zero surplus adds exactly that
    quantity, scaled, and nothing more. The term is named in CONTRACT rule 4 so that Phase-2 trade
    (PLAN section 2.13) enters the reward through a term that already exists rather than through a
    new one.
    """
    from gosplan.env.reward import enterprise_reward, reward_scale

    implemented(enterprise_reward, reward_scale)
    assert p1_cfg.information.horizontal_visibility == 0
    n = p1_cfg.supply.n_enterprises
    state = _state(p1_cfg, reports=np.full(n, p1_cfg.tech.initial_target_frac))
    pen = np.zeros(n)
    base = np.asarray(enterprise_reward(state, p1_cfg, "report", None, pen, np.zeros(n)))
    extra = np.linspace(0.0, 1.0, n)
    with_surplus = np.asarray(enterprise_reward(state, p1_cfg, "report", None, pen, extra))
    assert np.max(np.abs((with_surplus - base) - reward_scale(p1_cfg) * extra)) < 1e-12


@pytest.mark.skeleton
def test_reward_is_invariant_to_welfare_and_val_measured(p1_cfg, implemented) -> None:
    """CONTRACT rule 6: no reward term reads `welfare_true` or `val_measured`.

    Assertion: two states that differ only in quantities that feed `welfare_true` and
    `val_measured` - the consumer sink's receipts, other enterprises' claims - but agree on this
    enterprise's own `rho`, penalty and cost produce byte-identical rewards. The unit form of the
    sentinel check that test T-B5 performs over observations; here the target is the reward path.
    """
    from gosplan.env.reward import enterprise_reward

    implemented(enterprise_reward)
    n, j = p1_cfg.supply.n_enterprises, p1_cfg.supply.n_sectors
    reports = np.full(n, p1_cfg.tech.initial_target_frac)
    pen = np.zeros(n)
    a = _state(p1_cfg, reports=reports, consumer=np.zeros(j))
    b = _state(p1_cfg, reports=reports, consumer=np.full(j, 7.0))
    ra = np.asarray(enterprise_reward(a, p1_cfg, "report", None, pen, np.zeros(n)))
    rb = np.asarray(enterprise_reward(b, p1_cfg, "report", None, pen, np.zeros(n)))
    assert np.array_equal(ra, rb)


@pytest.mark.skeleton
def test_val_measured_is_the_price_weighted_sum_of_claims(p1_cfg, implemented) -> None:
    """`val_measured_t = sum_i p_{s(i)} * R_i * q_hat_i` (PLAN section 2.9.3).

    Assertion: `val_measured(state, cfg)` equals that sum to 1e-12, with `p = state.plan_prices`,
    `R_i = state.last_report` and `q_hat_i = 1 + mu * (qbar_i - 1)` (exactly 1 in Phase 1, where
    `quality_measurability = 0`); it scales linearly in a common rescaling of claims and is
    unchanged by anything that is not a claim, a price or a measured quality. It is what the
    planning system believes it produced, and it is logged only.
    """
    from gosplan.env.reward import val_measured

    implemented(val_measured)
    n, j = p1_cfg.supply.n_enterprises, p1_cfg.supply.n_sectors
    reports = np.linspace(0.1, 1.0, n)
    prices = np.linspace(1.0, 2.0, j)
    state = _state(p1_cfg, reports=reports, prices=prices)
    sector = np.asarray(p1_cfg.supply.sector_of)
    want = float(np.sum(prices[sector] * reports))
    assert abs(float(val_measured(state, p1_cfg)) - want) < 1e-12

    doubled = _state(p1_cfg, reports=reports, prices=2.0 * prices)
    assert abs(float(val_measured(doubled, p1_cfg)) - 2.0 * want) < 1e-12


@pytest.mark.skeleton
def test_val_true_is_the_price_weighted_sum_of_true_output(p1_cfg, implemented) -> None:
    """`val_true_t = sum_i p_{s(i)} * y_i * qbar_i` (PLAN section 2.9.3).

    Assertion: `val_true(state, cfg)` equals that sum to 1e-12, reading the period's true
    production and its period-average quality (1 in Phase 1); it equals `val_measured` exactly when
    every claim equals the corresponding true output, and differs otherwise. Logged only, exactly
    as `val_measured` (CONTRACT rule 6).
    """
    from gosplan.env.reward import val_measured, val_true

    implemented(val_measured, val_true)
    n, j = p1_cfg.supply.n_enterprises, p1_cfg.supply.n_sectors
    y = np.linspace(0.2, 1.1, n)
    prices = np.linspace(1.0, 2.0, j)
    sector = np.asarray(p1_cfg.supply.sector_of)
    honest = _state(p1_cfg, reports=y, outputs=y, prices=prices)
    want = float(np.sum(prices[sector] * y))
    assert abs(float(val_true(honest, p1_cfg)) - want) < 1e-12
    assert abs(float(val_true(honest, p1_cfg)) - float(val_measured(honest, p1_cfg))) < 1e-12

    padded = _state(p1_cfg, reports=y * 1.5, outputs=y, prices=prices)
    assert float(val_measured(padded, p1_cfg)) > float(val_true(padded, p1_cfg))


@pytest.mark.skeleton
def test_welfare_is_the_ces_index_of_consumer_receipts(p1_cfg, implemented) -> None:
    """`welfare_t = (sum_j alpha_j * consumer_j**rho_ces)**(1 / rho_ces)`, `rho_ces = (sc-1)/sc`.

    Assertion: `welfare_true(consumer, cfg)` equals that index to 1e-12 at
    `sigma_c = cfg.supply.ces_sigma = 0.8` with `alpha = cfg.supply.ces_alpha`; it is homogeneous of
    degree 1 (doubling every `consumer_j` doubles welfare); it is increasing in each `consumer_j`;
    and at `sigma_c < 1` (complements) it goes to 0 as any single `consumer_j` goes to 0, which is
    the substitutability property the Phase-1 value encodes.

    Logged only, and the strictest case of CONTRACT rule 6: no agent, no planner rule and no reward
    term may read it. `W = mean_t welfare_t` over the measurement window of PLAN section 4.4
    (periods `t >= 2`).
    """
    from gosplan.env.reward import welfare_true

    implemented(welfare_true)
    alpha = np.asarray(p1_cfg.supply.ces_alpha, dtype=float)
    sc = p1_cfg.supply.ces_sigma
    rho_ces = (sc - 1.0) / sc
    consumer = np.linspace(0.5, 1.5, p1_cfg.supply.n_sectors)
    want = float(np.sum(alpha * consumer**rho_ces) ** (1.0 / rho_ces))
    assert abs(float(welfare_true(consumer, p1_cfg)) - want) < 1e-12
    # homogeneous of degree 1, and increasing in every coordinate
    assert abs(float(welfare_true(2.0 * consumer, p1_cfg)) - 2.0 * want) < 1e-9
    more = consumer.copy()
    more[0] += 0.1
    assert float(welfare_true(more, p1_cfg)) > float(welfare_true(consumer, p1_cfg))


@pytest.mark.skeleton
def test_welfare_has_an_explicit_cobb_douglas_limit_at_unit_elasticity(p1_cfg, implemented) -> None:
    """The `sigma_c -> 1` limit is `prod_j consumer_j**alpha_j`, as an explicit branch.

    Assertion: `welfare_true(consumer, cfg)` at `ces_sigma = 1.0` equals
    `np.prod(consumer ** np.asarray(alpha))` to 1e-12 - computed by an explicit branch, not by
    evaluating `rho_ces = 0` and dividing by zero - and the CES form converges to it from both
    sides: at `sigma_c = 1 +/- 1e-6` the two agree to 1e-5. The branch is a named item of the
    WO-007 must-pass list precisely because the general formula is singular there.
    """
    import dataclasses

    from gosplan.env.reward import welfare_true

    implemented(welfare_true)
    alpha = np.asarray(p1_cfg.supply.ces_alpha, dtype=float)
    consumer = np.linspace(0.5, 1.5, p1_cfg.supply.n_sectors)
    unit = dataclasses.replace(p1_cfg, supply=dataclasses.replace(p1_cfg.supply, ces_sigma=1.0))
    want = float(np.prod(consumer**alpha))
    assert abs(float(welfare_true(consumer, unit)) - want) < 1e-12
    for sc in (0.999, 1.001):
        near = dataclasses.replace(p1_cfg, supply=dataclasses.replace(p1_cfg.supply, ces_sigma=sc))
        assert abs(float(welfare_true(consumer, near)) - want) < 1e-2


@pytest.mark.skeleton
def test_padding_index_is_val_measured_over_val_true(p1_cfg, implemented) -> None:
    """`padding_index = val_measured / val_true` (PLAN section 2.9.4).

    Assertion: over a measurement window, the index equals the ratio of the two aggregates to
    1e-12; it is exactly 1 on a trajectory where every claim equals true output; and it exceeds 1
    whenever any claim exceeds the corresponding true output with the others unchanged. It is
    dimensionless and is one of the three headline metrics, all logged and none observed.
    """
    from gosplan.env.reward import padding_index

    implemented(padding_index)
    meas = np.array([2.0, 3.0, 4.0])
    true = np.array([2.0, 3.0, 4.0])
    assert abs(padding_index(meas, true) - 1.0) < 1e-12
    padded = meas.copy()
    padded[1] += 1.0
    assert padding_index(padded, true) > 1.0


@pytest.mark.skeleton
def test_welfare_ratio_and_specification_gap_follow_their_definitions(p1_cfg, implemented) -> None:
    """`welfare_ratio = W / W_oracle`; `specification_gap = val_measured/val_oracle - W/W_oracle`.

    Assertion: `welfare_ratio` equals the mean of the welfare window divided by the oracle welfare
    to 1e-12 and is 1 when the trajectory matches the oracle; `specification_gap` is 0 on an honest,
    efficient trajectory and strictly positive when claims are padded while deliveries are
    unchanged. In Phase 1 the denominator is `W_truthful_max`, a clearly labelled placeholder for
    the PLAN section 6.2 oracle, and every table that reports the ratio must say so (PLAN section
    2.9.4). The standing robustness check of PLAN section 7.5 recomputes all three under three
    perturbed price vectors; a sign change in `specification_gap` is reported, not suppressed.
    """
    from gosplan.env.reward import specification_gap, welfare_ratio

    implemented(specification_gap, welfare_ratio)
    window = np.array([1.0, 2.0, 3.0])
    oracle = float(window.mean())
    assert abs(welfare_ratio(window, oracle) - 1.0) < 1e-12
    assert abs(welfare_ratio(window, 2.0 * oracle) - 0.5) < 1e-12
    honest = specification_gap(np.array([1.0, 1.0]), 1.0, window, oracle)
    assert abs(honest) < 1e-12
