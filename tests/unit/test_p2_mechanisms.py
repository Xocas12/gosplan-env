"""Unit tests for the Phase-2 environment mechanisms (spec/P2_REVISION.md, spec 2.0.0).

Realises: P2 revision R12 ("each work order adds unit tests for its mechanism") for WO-021 to
WO-025: report lag (R2), downstream shortfall (R3), targeted audits (R4), quality (R5), delivery
timing (R6), input holding loss (R7), soft budget (R8), trade (R9) and the rule-based ministry
(R10), plus a smoke run of `p2_default_config()` (R11).

Every check here is a conservation, boundedness or formula check. None asserts the direction of a
held-out phenomenon (PLAN section 4.1 rows 2, 5, 6, 7: storming, hoarding, blat, hidden reserves);
R12 forbids that before WO-031. Hand-worked cases are stated in the docstrings.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from gosplan.config import p1_default_config, p2_default_config
from gosplan.rng import draw


def _replace(cfg, **arms):
    """`cfg` with fields of its arms replaced: `_replace(cfg, supply={...}, information={...})`."""
    new = {name: dataclasses.replace(getattr(cfg, name), **fields) for name, fields in arms.items()}
    out = dataclasses.replace(cfg, **new)
    out.validate()
    return out


def _tiny(**arms):
    """The two-enterprise, two-sector case of `docs/ref_worked_example.md` (`tiny_cfg`)."""
    base = p1_default_config()
    supply = dict(
        n_enterprises=2,
        n_sectors=2,
        sector_of=(0, 1),
        io_matrix=((0.0, 0.2), (0.2, 0.0)),
        final_demand_share=(0.5, 0.5),
        productivity=(1.0, 1.0),
        yield_sigma=(0.05, 0.08),
        ces_alpha=(0.5, 0.5),
    )
    supply.update(arms.pop("supply", {}))
    return _replace(base, supply=supply, **arms)


def _four(**arms):
    """Four enterprises, two per sector, each sector needing the other's good."""
    base = p1_default_config()
    supply = dict(
        n_enterprises=4,
        n_sectors=2,
        sector_of=(0, 0, 1, 1),
        io_matrix=((0.0, 0.2), (0.2, 0.0)),
        final_demand_share=(0.5, 0.5),
        productivity=(1.0, 1.0),
        yield_sigma=(0.05, 0.08),
        ces_alpha=(0.5, 0.5),
    )
    supply.update(arms.pop("supply", {}))
    return _replace(base, supply=supply, **arms)


def _state(cfg):
    from gosplan.env.state import initial_state

    return initial_state(cfg)


def _zero_action(cfg):
    from gosplan.env.state import EnterpriseAction

    n, j = cfg.supply.n_enterprises, cfg.supply.n_sectors
    return EnterpriseAction(
        effort=np.zeros(n),
        quality=np.zeros(n),
        invest=np.zeros(n),
        report_ratio=np.zeros(n),
        input_request=np.zeros((n, j)),
        trade_offer=np.zeros((n, j)),
    )


# ---------- R2: report lag ----------


@pytest.mark.parametrize("lag", [1, 2])
def test_report_lag_serves_the_right_history_column(lag: int) -> None:
    """R2: with `report_lag = L > 0` the view's claims are `claim_history[:, L - 1]`.

    Channel noise 0 and enterprise aggregation, so the served claims equal the column exactly; at
    `L = 0` the current claim `last_report` is served instead.
    """
    from gosplan.env.planner import make_planner_view

    cfg = _replace(p1_default_config(), information={"report_lag": lag})
    state = _state(cfg)
    n = cfg.supply.n_enterprises
    state.last_report = np.full(n, 9.0)
    state.claim_history = np.stack([np.arange(n) + 1.0, np.arange(n) + 100.0], axis=1)
    view = make_planner_view(state, cfg)
    np.testing.assert_array_equal(view.claims, state.claim_history[:, lag - 1])

    cfg0 = p1_default_config()
    np.testing.assert_array_equal(make_planner_view(state, cfg0).claims, state.last_report)


def test_claim_history_opens_on_plan_and_shifts_at_target() -> None:
    """R2: `claim_history` starts at `T_0`; after a period's TARGET, column 0 holds this period's
    forwarded claim (`R` at `pi = 1`) and column 1 the old column 0."""
    from gosplan.env.env import GosplanEnv
    from gosplan.env.state import initial_targets

    cfg = _replace(p1_default_config(), information={"report_lag": 1})
    env = GosplanEnv(cfg)
    env.reset(3, 4)
    t0 = initial_targets(cfg)
    np.testing.assert_array_equal(env.state.claim_history, np.stack([t0, t0], axis=1))
    action = _zero_action(cfg)
    action.effort[:] = 0.6
    for _ in range(cfg.incentive.steps_per_period):
        env.step(action)
    action.report_ratio[:] = np.linspace(0.5, 1.5, cfg.supply.n_enterprises)
    env.step(action)
    np.testing.assert_array_equal(env.state.claim_history[:, 0], env.state.last_report)
    np.testing.assert_array_equal(env.state.claim_history[:, 1], t0)


# ---------- R3 / R4: downstream shortfall and targeted audits ----------


def test_downstream_shortfall_formula() -> None:
    """R3: `ds_i = sv * (1 - fill_i) * exp(xi_i)`, `xi ~ N(0, channel_noise**2)` at key
    `(seed_env, "complaint", t, i)`; exactly `sv * (1 - fill)` at zero noise, 0 when fill = 1."""
    from gosplan.env.planner import make_planner_view

    sv, sigma = 0.5, 0.05
    cfg = _replace(p1_default_config(), information={"shortfall_visibility": sv})
    state = _state(cfg)
    n = cfg.supply.n_enterprises
    state.t_period = 3
    fill = np.linspace(0.0, 1.0, n)
    state.last_fill = fill
    np.testing.assert_allclose(
        make_planner_view(state, cfg).downstream_shortfall, sv * (1 - fill), atol=0.0
    )
    noisy = _replace(cfg, information={"channel_noise": sigma})
    xi = draw(state.seed_env, "complaint", 3, shape=(n,), dist="normal", mean=0.0, sigma=sigma)
    got = make_planner_view(state, noisy).downstream_shortfall
    np.testing.assert_allclose(got, sv * (1 - fill) * np.exp(xi), rtol=1e-15)
    assert got[-1] == 0.0
    assert np.all(got >= 0.0)


def test_downstream_shortfall_sector_mean_under_sector_aggregation() -> None:
    """R3: under `aggregation_level = "sector"` each entry is its sector's mean."""
    from gosplan.env.planner import make_planner_view

    cfg = _replace(
        p1_default_config(),
        information={"shortfall_visibility": 1.0, "aggregation_level": "sector"},
    )
    state = _state(cfg)
    n = cfg.supply.n_enterprises
    state.last_fill = np.linspace(0.0, 1.0, n)
    got = make_planner_view(state, cfg).downstream_shortfall
    sector = np.asarray(cfg.supply.sector_of)
    for s in set(sector.tolist()):
        np.testing.assert_allclose(got[sector == s], np.mean(1 - state.last_fill[sector == s]))


def test_targeted_audit_probability_formula() -> None:
    """R4: `p_i = clip(a * (1 + kappa_t * ds_i), 0, 1)`, drawn at key `(seed_env, "audit", t, i)`.

    The selection equals the Bernoulli draw at those probabilities; `p = 1` (here `a = 0.5`,
    `kappa_t = 10`, `ds = 0.5`) audits for sure.
    """
    from gosplan.env.planner import make_planner_view, select_audits

    a, kappa = 0.2, 4.0
    cfg = _replace(
        p1_default_config(),
        information={
            "audit_mode": "targeted",
            "shortfall_visibility": 0.5,
            "audit_rate": a,
            "audit_target_gain": kappa,
        },
    )
    state = _state(cfg)
    n = cfg.supply.n_enterprises
    state.last_fill = np.linspace(0.0, 1.0, n)
    view = make_planner_view(state, cfg)
    p = np.clip(a * (1 + kappa * view.downstream_shortfall), 0, 1)
    for t in range(6):
        expected = draw(cfg.tech.seed_env, "audit", t, shape=(n,), dist="bernoulli", p=p)
        np.testing.assert_array_equal(select_audits(view, cfg, t), expected)

    sure = _replace(cfg, information={"audit_rate": 0.5, "audit_target_gain": 10.0})
    state.last_fill = np.zeros(n)
    view = make_planner_view(state, sure)
    assert all(select_audits(view, sure, t).all() for t in range(5))


def test_targeted_audits_reduce_to_random_at_zero_visibility() -> None:
    """R4: with `shortfall_visibility = 0` targeted audits are exactly random audits."""
    from gosplan.env.planner import make_planner_view, select_audits

    rnd = _replace(p1_default_config(), information={"audit_rate": 0.3})
    tgt = _replace(rnd, information={"audit_mode": "targeted"})
    state = _state(rnd)
    state.last_fill = np.linspace(0.0, 1.0, rnd.supply.n_enterprises)
    for t in range(10):
        np.testing.assert_array_equal(
            select_audits(make_planner_view(state, rnd), rnd, t),
            select_audits(make_planner_view(state, tgt), tgt, t),
        )


# ---------- R5: quality ----------


def test_qbar_is_period_mean_and_one_when_quality_off() -> None:
    """R5: `qbar = quality_acc / M` with `quality_matters`, else exactly 1."""
    from gosplan.env.production import period_quality

    on = _tiny(supply={"quality_matters": True})
    acc = np.array([2.0, 3.0])
    np.testing.assert_array_equal(period_quality(acc, on), acc / on.incentive.steps_per_period)
    np.testing.assert_array_equal(period_quality(acc, _tiny()), np.ones(2))


def test_bundle_routing_hand_worked_two_enterprise_case() -> None:
    """R5, the tiny case (enterprise 0 sells good 0, enterprise 1 sells good 1; M = 4):

    R = (0.6, 0.4), S = (0.6, 0.2), quality_acc = (2.0, 3.0)  ->  qbar = (0.5, 0.75)
    fill = (1, 0.5), shipped = (0.6, 0.2), poolfill = (1, 0.5)
    alloc = [[0, 0.2], [0.3, 0]]  ->  deliv = [[0, 0.1], [0.3, 0]]
    X += deliv * qbar_j:  X_01 += 0.1 * 0.75 = 0.075,  X_10 += 0.3 * 0.5 = 0.15
    consumer = phi * shipped * qbar = (0.5*0.6*0.5, 0.5*0.2*0.75) = (0.15, 0.075)
    """
    from gosplan.env.planner import deliver

    cfg = _tiny(supply={"quality_matters": True})
    state = _state(cfg)
    state.last_report = np.array([0.6, 0.4])
    state.inv_output = np.array([0.6, 0.2])
    state.quality_acc = np.array([2.0, 3.0])
    x0 = np.array(state.inv_inputs)
    alloc = np.array([[0.0, 0.2], [0.3, 0.0]])
    new, deliv, fill, consumer = deliver(state, alloc, cfg)
    np.testing.assert_allclose(fill, [1.0, 0.5])
    np.testing.assert_allclose(deliv, [[0.0, 0.1], [0.3, 0.0]])
    np.testing.assert_allclose(new.inv_inputs - x0, [[0.0, 0.075], [0.15, 0.0]])
    np.testing.assert_allclose(consumer, [0.15, 0.075])
    np.testing.assert_allclose(new.inv_output, [0.0, 0.0])


def test_bundle_quality_is_claim_weighted_over_sellers() -> None:
    """R5: `qbar_j = sum_i claimed_i qbar_i / sum_i claimed_i` over good `j`'s sellers.

    Sellers 0 and 1 of good 0 claim 0.2 and 0.6 with qbar 1.0 and 0.5 and full stock, so
    `qbar_0 = (0.2 * 1 + 0.6 * 0.5) / 0.8 = 0.625`; buyer 2 promised 0.4 of good 0 gets 0.25.
    """
    from gosplan.env.planner import deliver

    cfg = _four(supply={"quality_matters": True})
    state = _state(cfg)
    state.last_report = np.array([0.2, 0.6, 0.0, 0.0])
    state.inv_output = np.array([1.0, 1.0, 0.0, 0.0])
    state.quality_acc = np.array([4.0, 2.0, 0.0, 0.0])
    x0 = np.array(state.inv_inputs)
    alloc = np.zeros((4, 2))
    alloc[2, 0] = 0.4
    new, _deliv, _fill, _consumer = deliver(state, alloc, cfg)
    assert new.inv_inputs[2, 0] - x0[2, 0] == pytest.approx(0.4 * 0.625, abs=1e-15)


# ---------- R6: delivery timing ----------


def test_arrival_steps_follow_the_keyed_draw() -> None:
    """R6: `k_bj ~ Categorical(pi)` at key `(seed_env, "arrival", t, b, j)` (trailing `j` through
    `shape`); `backloaded` uses `pi_k` proportional to `(k + 1)**2`; `uniform` is all zeros."""
    from gosplan.env.planner import arrival_steps

    probs = (0.1, 0.2, 0.3, 0.4)
    cfg = _tiny(supply={"delivery_timing": "stochastic", "arrival_probs": probs})
    k = arrival_steps(7, 5, cfg)
    for b in range(2):
        np.testing.assert_array_equal(
            k[b], draw(7, "arrival", 5, b, shape=(2,), dist="categorical", probs=probs)
        )
    back = _tiny(supply={"delivery_timing": "backloaded", "arrival_probs": probs})
    w = (np.arange(4) + 1.0) ** 2
    pi = tuple(float(v) for v in w / w.sum())
    kb = arrival_steps(7, 5, back)
    for b in range(2):
        np.testing.assert_array_equal(
            kb[b], draw(7, "arrival", 5, b, shape=(2,), dist="categorical", probs=pi)
        )
    np.testing.assert_array_equal(arrival_steps(7, 5, _tiny()), np.zeros((2, 2), dtype=int))


@pytest.mark.parametrize("timing", ["stochastic", "backloaded"])
def test_pending_deliveries_conserve_quantity_and_arrive_at_drawn_step(timing: str) -> None:
    """R6: what DELIVER credits now plus what waits in `pending_deliv` equals `deliv * qbar`;
    each buyer-good quantity reaches `X` exactly at the start of its drawn step, and nothing is
    left pending after the last PRODUCE step."""
    from gosplan.env.planner import arrival_steps, deliver
    from gosplan.env.production import credit_arrivals

    cfg = _four(supply={"delivery_timing": timing})
    state = _state(cfg)
    state.t_period = 2
    state.last_report = np.array([0.3, 0.3, 0.3, 0.3])
    state.inv_output = np.array([0.3, 0.1, 0.3, 0.3])
    x0 = np.array(state.inv_inputs)
    alloc = np.array([[0.0, 0.1], [0.0, 0.2], [0.15, 0.0], [0.05, 0.0]])
    new, deliv, _fill, _c = deliver(state, alloc, cfg)
    k_arr = arrival_steps(new.seed_env, 2, cfg)
    total = new.inv_inputs - x0 + new.pending_deliv.sum(axis=2)
    np.testing.assert_allclose(total, deliv, atol=1e-15)
    np.testing.assert_allclose(new.inv_inputs - x0, np.where(k_arr == 0, deliv, 0.0), atol=1e-15)
    for k in range(1, cfg.incentive.steps_per_period):
        before = np.array(new.inv_inputs)
        new.k_step = k
        new = credit_arrivals(new, cfg)
        np.testing.assert_allclose(
            new.inv_inputs - before, np.where(k_arr == k, deliv, 0.0), atol=1e-15
        )
    np.testing.assert_array_equal(new.pending_deliv, 0.0)
    np.testing.assert_allclose(new.inv_inputs - x0, deliv, atol=1e-15)


def test_observation_reports_deliveries_received_so_far() -> None:
    """R6: fields `12+2J:12+3J` are `deliv_ij / need_ij` over the deliveries whose arrival step is
    at or before the step just executed (1.0 where `need = 0`, as in Phase 1); at the REPORT step
    the whole period's delivery is shown."""
    from gosplan.agents.heuristic import TruthfulMyopic
    from gosplan.env.env import GosplanEnv
    from gosplan.env.obs import _coverage
    from gosplan.env.planner import arrival_steps

    cfg = _replace(p1_default_config(), supply={"delivery_timing": "stochastic"})
    env = GosplanEnv(cfg)
    obs, _ = env.reset(11, 12)
    agent = TruthfulMyopic(cfg)
    rng = np.random.default_rng(0)
    j = cfg.supply.n_sectors
    lo = 12 + 2 * j
    m = cfg.incentive.steps_per_period
    sector = np.asarray(cfg.supply.sector_of)
    for _period in range(3):
        k_arr = deliv = None
        for step in range(m + 1):
            obs, _r, _d, info = env.step(agent.act(obs, env.phase(), rng))
            need = np.asarray(env.state.planner_io)[sector] * np.asarray(env.state.target)[:, None]
            if step == 0:
                k_arr = arrival_steps(env.state.seed_env, info.t_period, cfg)
                deliv = np.array([r.deliv for r in info.records])
            received = np.where(k_arr <= step, deliv, 0.0)
            np.testing.assert_array_equal(obs[:, lo : lo + j], _coverage(received, need))


# ---------- R7: input holding loss ----------


def test_input_holding_loss_applied_once_per_period_at_report() -> None:
    """R7: at REPORT, `X <- (1 - h_X) X`; PRODUCE steps are unaffected, so a period with
    `h_X = 0.1` matches the `h_X = 0` period on every PRODUCE row and differs at REPORT by exactly
    the factor 0.9."""
    from gosplan.agents.heuristic import TruthfulMyopic
    from gosplan.env.env import GosplanEnv

    def rows(h_x):
        cfg = _replace(p1_default_config(), supply={"input_holding_loss": h_x})
        env = GosplanEnv(cfg)
        obs, _ = env.reset(5, 6)
        agent = TruthfulMyopic(cfg)
        out = []
        for _ in range(cfg.incentive.steps_per_period + 1):
            obs, _r, _d, info = env.step(agent.act(obs, env.phase(), np.random.default_rng(0)))
            out.append(np.array([r.inv_inputs for r in info.records]))
        return out

    base, lossy = rows(0.0), rows(0.1)
    for k in range(len(base) - 1):
        np.testing.assert_array_equal(base[k], lossy[k])
    np.testing.assert_allclose(lossy[-1], 0.9 * base[-2], rtol=1e-15)
    np.testing.assert_array_equal(base[-1], base[-2])


# ---------- R8: soft budget ----------


def test_bailout_zeroes_penalty_only_when_fill_below_one() -> None:
    """R8: at `soft_budget = 1` every enterprise with `last_fill < 1` has its penalty set to 0;
    one with `last_fill = 1` keeps it. At `soft_budget = 0` nobody is bailed out."""
    from gosplan.env.planner import make_planner_view
    from gosplan.env.step import soft_budget_bailouts, stage_audit

    cfg = _tiny(
        information={"audit_rate": 1.0}, incentive={"soft_budget": 1.0, "penalty_scale": 2.0}
    )
    state = _state(cfg)
    state.last_report = np.array([1.0, 1.0])
    state.inv_output = np.array([0.1, 0.1])
    state.last_fill = np.array([1.0, 0.5])
    np.testing.assert_array_equal(soft_budget_bailouts(state, cfg, 0), [False, True])
    _s, audited, penalty = stage_audit(state, make_planner_view(state, cfg), cfg, 0)
    assert audited.all()
    assert penalty[0] > 0.0 and penalty[1] == 0.0

    off = _replace(cfg, incentive={"soft_budget": 0.0})
    state.last_fill = np.array([0.0, 0.5])
    assert not soft_budget_bailouts(state, off, 0).any()
    _s, _a, penalty = stage_audit(state, make_planner_view(state, off), off, 0)
    assert (penalty > 0).all()


def test_bailout_draw_is_keyed() -> None:
    """R8: the bailout draw is `Bernoulli(soft_budget)` at key `(seed_env, "bailout", t, i)`."""
    from gosplan.env.step import soft_budget_bailouts

    cfg = _replace(p1_default_config(), incentive={"soft_budget": 0.4})
    state = _state(cfg)
    n = cfg.supply.n_enterprises
    state.last_fill = np.zeros(n)
    for t in range(4):
        expected = draw(state.seed_env, "bailout", t, shape=(n,), dist="bernoulli", p=0.4)
        np.testing.assert_array_equal(soft_budget_bailouts(state, cfg, t), expected)


# ---------- R9: trade ----------


def _random_trade_case(seed, n=6, j=3):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=(n, j))
    offers = rng.uniform(-1.0, 1.0, size=(n, j))
    need = rng.uniform(0.0, 1.0, size=(n, j))
    return x, offers, need


@pytest.mark.parametrize("seed", range(5))
def test_trade_conserves_inputs_up_to_tau(seed: int) -> None:
    """R9.3 / R12: per good, `sum_b X_bj` falls by exactly `tau` times the executed volume; no
    stock goes negative; nothing is sold beyond the offered fraction of stock."""
    from gosplan.env.trade import execute_trades

    tau = 0.05
    x, offers, need = _random_trade_case(seed)
    n, j = x.shape
    visible = np.ones((n, n), dtype=bool)
    after, sold = execute_trades(x, offers, need, visible, tau)
    assert np.all(after >= 0.0)
    for g in range(j):
        only = np.zeros_like(offers)
        only[:, g] = offers[:, g]
        _a, sold_g = execute_trades(x, only, need, visible, tau)
        assert x[:, g].sum() - after[:, g].sum() == pytest.approx(tau * sold_g.sum(), abs=1e-12)
    cap = (np.maximum(0.0, offers) * x).sum(axis=1)
    assert np.all(sold <= cap + 1e-12)


def test_trade_respects_visibility_and_never_self_pairs() -> None:
    """R9.1: only visible pairs trade (either direction suffices); the diagonal never trades."""
    from gosplan.env.trade import execute_trades

    x = np.array([[1.0], [0.0], [0.0]])
    offers = np.array([[1.0], [-1.0], [-1.0]])
    need = np.array([[0.0], [0.5], [0.8]])
    only_self = np.eye(3, dtype=bool)
    after, sold = execute_trades(x, offers, need, only_self, 0.05)
    np.testing.assert_array_equal(after, x)
    assert sold.sum() == 0.0

    one_way = np.zeros((3, 3), dtype=bool)
    one_way[1, 0] = True  # buyer 1 sees seller 0: enough for the pair (0, 1)
    after, sold = execute_trades(x, offers, need, one_way, 0.05)
    assert after[2, 0] == 0.0
    assert after[1, 0] == pytest.approx(0.95 * 0.5)
    assert sold[0] == pytest.approx(0.5)


def test_trade_takes_largest_pair_first_ties_to_lowest_index() -> None:
    """R9.3: seller 0 (supply 0.5) faces buyers 1 and 2 with equal demand 0.3: the tie goes to
    buyer 1 (lowest `(i, b)`), buyer 2 gets the remaining 0.2."""
    from gosplan.env.trade import execute_trades

    x = np.array([[0.5], [0.0], [0.0]])
    offers = np.array([[1.0], [-1.0], [-1.0]])
    need = np.array([[0.0], [0.3], [0.3]])
    after, _sold = execute_trades(x, offers, need, np.ones((3, 3), dtype=bool), 0.0)
    np.testing.assert_allclose(after[:, 0], [0.0, 0.3, 0.2])


def test_visible_counterparties_draw_k_others() -> None:
    """R9.1: each row has exactly `K = round(hv * (N - 1))` True entries, a False diagonal, and
    is the first K of the argsort of the keyed normal draw."""
    from gosplan.env.trade import visible_counterparties

    for hv in (0.0, 0.3, 0.5, 1.0):
        cfg = _replace(p1_default_config(), information={"horizontal_visibility": hv})
        state = _state(cfg)
        n = cfg.supply.n_enterprises
        mask = visible_counterparties(state, cfg, 4)
        k = round(hv * (n - 1))
        assert not mask.diagonal().any()
        np.testing.assert_array_equal(mask.sum(axis=1), np.full(n, k))
        if 0 < k < n - 1:
            z = draw(
                state.seed_env,
                "trade_visibility",
                4,
                0,
                shape=(n - 1,),
                dist="normal",
                mean=0.0,
                sigma=1.0,
            )
            others = np.arange(1, n)
            np.testing.assert_array_equal(
                np.flatnonzero(mask[0]), np.sort(others[np.argsort(z)][:k])
            )


def test_wash_trade_is_never_profitable_at_positive_tau() -> None:
    """R9.5: a round trip of good 1 between enterprises 0 and 1 (both sector 0, both short of good
    1, effort 1) leaves 1's stock where it was and 0's lower by the two `tau` losses, so the summed
    surplus over both legs is negative and neither party gains.

        need at e = 1:  y_hat = 1/4, need_i1 = 0.2 * 0.25 = 0.05 > X = 0.04 (binding)
        leg 1: 0 sells 0.02 to 1 -> X_01 = 0.02, X_11 = 0.04 + 0.019
        leg 2: 1 sells 0.019 back -> X_11 = 0.04, X_01 = 0.02 + 0.95 * 0.019 = 0.03805
    """
    from gosplan.env.trade import execute_trades, trade_surplus

    cfg = _four(supply={"trade_tau": 0.05})
    state = _state(cfg)
    x0 = np.zeros((4, 2))
    x0[0, 1] = x0[1, 1] = 0.04
    need = np.zeros((4, 2))
    need[0, 1] = need[1, 1] = 1.0
    vis = np.ones((4, 4), dtype=bool)
    e = np.ones(4)
    leg1 = np.zeros((4, 2))
    leg1[0, 1], leg1[1, 1] = 0.5, -1.0
    x1, _ = execute_trades(x0, leg1, need, vis, 0.05)
    leg2 = np.zeros((4, 2))
    leg2[1, 1], leg2[0, 1] = (x1[1, 1] - 0.04) / x1[1, 1], -1.0
    x2, _ = execute_trades(x1, leg2, need, vis, 0.05)
    assert x2[1, 1] == pytest.approx(0.04, abs=1e-15)
    assert x2[0, 1] == pytest.approx(0.03805, abs=1e-15)
    total = trade_surplus(state, x0, x1, e, cfg) + trade_surplus(state, x1, x2, e, cfg)
    assert total[0] < 0.0
    assert abs(total[1]) < 1e-12
    assert total.sum() < 0.0


def test_trade_stage_surplus_is_zero_for_non_traders_and_paid_at_report() -> None:
    """R9.5: enterprises whose stocks do not move get surplus exactly 0; the accumulated surplus is
    added to the REPORT reward once (scaled) and the accumulator is then reset."""
    from gosplan.env.reward import enterprise_reward, reward_scale
    from gosplan.env.trade import trade_stage

    cfg = _replace(p1_default_config(), information={"horizontal_visibility": 1.0})
    state = _state(cfg)
    n, j = cfg.supply.n_enterprises, cfg.supply.n_sectors
    offers = np.zeros((n, j))
    # Enterprise 0 (sector 0) sells good 1 to enterprise 16 (sector 4, which also uses good 1).
    offers[0, 1], offers[16, 1] = 1.0, -1.0
    new, surplus, sold = trade_stage(state, offers, cfg, 0, effort=np.full(n, 0.6))
    moved = np.any(new.inv_inputs != state.inv_inputs, axis=1)
    np.testing.assert_array_equal(np.flatnonzero(moved), [0, 16])
    assert np.all(surplus[~moved] == 0.0)
    assert sold[0] > 0.0 and sold[1:].sum() == 0.0

    state.last_report = np.asarray(state.target) * 1.1
    zero = enterprise_reward(state, cfg, "report", None, np.zeros(n), np.zeros(n))
    paid = enterprise_reward(state, cfg, "report", None, np.zeros(n), surplus)
    np.testing.assert_allclose(paid - zero, reward_scale(cfg) * surplus, atol=1e-15)


def test_peer_block_holds_sector_peers_report_ratios() -> None:
    """R9.6: with `horizontal_visibility > 0` the observation gains `G - 1` fields after
    `12 + 3J`: the last report ratios of the agent's sector peers in index order, zero-padded."""
    from gosplan.env.obs import build_observation, obs_spec

    cfg = _replace(p1_default_config(), information={"horizontal_visibility": 0.5})
    n, j = cfg.supply.n_enterprises, cfg.supply.n_sectors
    names = obs_spec(cfg)
    assert len(names) == 12 + 3 * j + 3  # sectors of size 4
    assert len(obs_spec(p1_default_config())) == 12 + 3 * j
    state = _state(cfg)
    state.last_report_ratio = np.arange(n) / 10.0
    obs = build_observation(state, cfg, np.zeros((n, j)), np.ones((n, j)))
    assert obs.shape == (n, len(names))
    sector = np.asarray(cfg.supply.sector_of)
    for i in range(n):
        peers = [b for b in range(n) if b != i and sector[b] == sector[i]]
        np.testing.assert_allclose(obs[i, 12 + 3 * j :], np.arange(n)[peers] / 10.0)


# ---------- R10: ministry ----------


def test_ministry_partition_is_contiguous_floor_rule() -> None:
    """R10.1: enterprise `i` belongs to ministry `floor(i * n_m / N)`; the views partition N."""
    from gosplan.env.ministry import make_ministry_views, ministry_of

    cfg = _replace(p1_default_config(), information={"n_ministries": 3})
    n = cfg.supply.n_enterprises
    np.testing.assert_array_equal(ministry_of(cfg), [(i * 3) // n for i in range(n)])
    views = make_ministry_views(_state(cfg), cfg)
    ids = np.concatenate([v.enterprise_ids for v in views])
    np.testing.assert_array_equal(np.sort(ids), np.arange(n))
    assert [v.ministry_id for v in views] == [0, 1, 2]


def test_ministry_is_transparent_at_passthrough_one() -> None:
    """R10.4: `pi = 1` forwards every claim unchanged, and a whole trajectory with five
    transparent ministries is identical to one with a single one."""
    from gosplan.agents.heuristic import Random
    from gosplan.env.env import GosplanEnv
    from gosplan.env.ministry import forward_all, make_ministry_views

    cfg5 = _replace(p1_default_config(), information={"n_ministries": 5})
    state = _state(cfg5)
    state.last_report = np.linspace(0.1, 2.0, cfg5.supply.n_enterprises)
    np.testing.assert_array_equal(
        forward_all(make_ministry_views(state, cfg5), cfg5), state.last_report
    )

    def run(cfg):
        env = GosplanEnv(cfg)
        obs, _ = env.reset(21, 22)
        agent, rng = Random(cfg), np.random.default_rng(1)
        out = []
        for _ in range(3 * (cfg.incentive.steps_per_period + 1)):
            obs, r, _d, _i = env.step(agent.act(obs, env.phase(), rng))
            out.append((obs.copy(), r.copy()))
        return out

    for (o1, r1), (o5, r5) in zip(run(p1_default_config()), run(cfg5), strict=True):
        np.testing.assert_array_equal(o1, o5)
        np.testing.assert_array_equal(r1, r5)


def test_ministry_forward_formula_and_one_sided_pad() -> None:
    """R10.2: `Rtilde = pi R + (1 - pi) [prev + kappa_m max(0, T - R)]`; the pad is 0 when R >= T.

    Hand-worked with pi = 0.75, kappa_m = 0.5, T = 1, prev = 0.8:
        R = 0.6  ->  0.45 + 0.25 * (0.8 + 0.2) = 0.70
        R = 1.2  ->  0.90 + 0.25 * 0.8         = 1.10
    """
    from gosplan.env.ministry import MinistryView, ministry_forward

    cfg = _replace(
        p1_default_config(), information={"ministry_passthrough": 0.75, "ministry_pad": 0.5}
    )
    view = MinistryView(
        ministry_id=0,
        enterprise_ids=np.array([0, 1]),
        claims=np.array([0.6, 1.2]),
        targets=np.array([1.0, 1.0]),
        prev_forward=np.array([0.8, 0.8]),
        passthrough=0.75,
        t_period=0,
    )
    np.testing.assert_allclose(ministry_forward(view, cfg), [0.70, 1.10], atol=1e-15)


def test_planner_sees_forward_while_bonus_and_audit_use_own_claim() -> None:
    """R10.3: after REPORT the planner's claims are `Rtilde` (= `ministry_prev`), delivery
    obligations follow `Rtilde`, and the bonus is evaluated at the enterprise's own `R / T`."""
    from gosplan.env.planner import make_planner_view
    from gosplan.env.reward import bonus, enterprise_reward, reward_scale
    from gosplan.env.step import stage_ministry

    cfg = _replace(
        p1_default_config(),
        information={"ministry_passthrough": 0.75, "n_ministries": 5, "ministry_pad": 0.5},
    )
    state = _state(cfg)
    n = cfg.supply.n_enterprises
    target = np.asarray(state.target, dtype=float)
    state.last_report = target * np.linspace(0.5, 1.5, n)
    prev = np.array(state.ministry_prev)
    state = stage_ministry(state, cfg)
    expected = 0.75 * state.last_report + 0.25 * (
        prev + 0.5 * np.maximum(0.0, target - state.last_report)
    )
    np.testing.assert_allclose(state.ministry_prev, expected, rtol=1e-15)
    np.testing.assert_allclose(make_planner_view(state, cfg).claims, expected, rtol=1e-15)
    r = enterprise_reward(state, cfg, "report", None, np.zeros(n), np.zeros(n))
    np.testing.assert_allclose(
        r, reward_scale(cfg) * np.asarray(bonus(state.last_report / target, cfg)), rtol=1e-15
    )


# ---------- R11: the P2 default configuration ----------


class _TradingRandom:
    """Uniform random actions including `quality` and `trade_offer` on every step (test only)."""

    def __init__(self, cfg):
        from gosplan.agents.heuristic import Random

        self.inner = Random(cfg)
        self.cfg = cfg

    def act(self, obs, phase, rng):
        action = self.inner.act(obs, phase, rng)
        n, j = self.cfg.supply.n_enterprises, self.cfg.supply.n_sectors
        action.trade_offer = rng.uniform(-1.0, 1.0, size=(n, j))
        action.quality = rng.uniform(0.0, 1.0, size=n)
        return action


@pytest.mark.parametrize("agent_name", ["Random", "TruthfulMyopic", "TradingRandom"])
def test_p2_default_config_runs_a_full_episode(agent_name: str) -> None:
    """R11: `p2_default_config()` runs whole episodes without error, with finite rewards and
    observations of the declared width, non-negative stocks, and no delivery left pending at the
    REPORT step."""
    from gosplan.agents import heuristic
    from gosplan.env.env import GosplanEnv

    cfg = p2_default_config()
    cfg.validate()
    env = GosplanEnv(cfg, records=False)
    agent = (
        _TradingRandom(cfg)
        if agent_name == "TradingRandom"
        else getattr(heuristic, agent_name)(cfg)
    )
    rng = np.random.default_rng(3)
    d = len(env.obs_spec())
    assert d == 12 + 3 * cfg.supply.n_sectors + 3
    for episode in range(2):
        obs, _ = env.reset(100 + episode, 7)
        done, steps = False, 0
        while not done:
            obs, reward, done, _info = env.step(agent.act(obs, env.phase(), rng))
            steps += 1
            assert obs.shape == (cfg.supply.n_enterprises, d)
            assert np.all(np.isfinite(obs)) and np.all(np.isfinite(reward))
            assert np.all(env.state.inv_inputs >= 0.0) and np.all(env.state.inv_output >= 0.0)
            if env.state.phase == "produce" and env.state.k_step == 0:
                assert np.all(env.state.pending_deliv == 0.0)
        assert steps % (cfg.incentive.steps_per_period + 1) == 0
