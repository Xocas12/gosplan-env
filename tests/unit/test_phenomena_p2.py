"""WO-030 estimators on SYNTHETIC ledgers with known answers (spec/P2_REVISION.md R13).

Authored by the LEAD. FROZEN BY CONTRACT RULE 2 once landed. Rows 2, 5, 6 and 7 are held out (PLAN
section 4.1): no test here runs the environment or an agent. Every ledger below is written by hand,
so these tests check the arithmetic of the estimators and nothing about the direction of any
phenomenon.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest


def _cfg(**supply):
    from gosplan.config import p1_default_config

    base = p1_default_config()
    two = dict(
        n_enterprises=2,
        n_sectors=2,
        sector_of=(0, 1),
        io_matrix=((0.0, 0.5), (0.25, 0.0)),
        final_demand_share=(0.5, 0.5),
        productivity=(1.0, 1.0),
        yield_sigma=(0.1, 0.1),
        ces_alpha=(0.5, 0.5),
    )
    two.update(supply)
    cfg = dataclasses.replace(base, supply=dataclasses.replace(base.supply, **two))
    cfg.validate()
    return cfg


def _rec(cfg, **over):
    from gosplan.metrics.ledger import StepRecord

    j = cfg.supply.n_sectors
    z = tuple(0.0 for _ in range(j))
    base = dict(
        run_hash="x",
        episode=0,
        t_period=2,
        k_step=cfg.incentive.steps_per_period,
        phase="report",
        enterprise=0,
        sector=0,
        target=1.0,
        capital=1.0,
        inv_output_pre=0.0,
        inv_output_post=0.0,
        inv_inputs=z,
        cum_output=0.0,
        cum_cost=0.0,
        quality_acc=0.0,
        last_report_ratio=1.0,
        last_penalty=0.0,
        last_fill=1.0,
        request=z,
        need=z,
        effort=0.0,
        quality=0.0,
        invest=0.0,
        output=0.0,
        cost=0.0,
        coverage=1.0,
        reward=0.0,
        report=1.0,
        report_ratio=1.0,
        at_bound=False,
        audited=False,
        audit_meas=0.0,
        penalty_arg=0.0,
        penalty=0.0,
        fill=1.0,
        shipped=0.0,
        alloc=z,
        deliv=z,
        input_consumed=z,
        holding_loss=0.0,
        cap_overflow=0.0,
        trade_volume=0.0,
        consumer=z,
        val_measured=0.0,
        val_true=0.0,
        welfare=0.0,
    )
    base.update(over)
    return StepRecord(**base)


def _ledger(records):
    from gosplan.metrics.ledger import Ledger

    ledger = Ledger()
    for rec in records:
        ledger.append(rec)
    return ledger


def _produce(cfg, efforts, t=2, i=0):
    return [
        _rec(cfg, phase="produce", k_step=k, t_period=t, enterprise=i, effort=e)
        for k, e in enumerate(efforts)
    ]


# ---------- row 2 ----------


def test_storming_gini_known_values() -> None:
    from gosplan.metrics.phenomena import phenomenon_storming

    cfg = _cfg()
    # All effort in one of 4 steps: Gini = (M - 1) / M = 0.75. Uniform effort: 0.
    run = _ledger(_produce(cfg, [0, 0, 0, 1.0]) + _produce(cfg, [0, 0, 0, 0], t=3))
    base = _ledger(_produce(cfg, [0.5] * 4) + _produce(cfg, [1.0, 0, 0, 0], t=1))
    out = phenomenon_storming(run, base, cfg)
    assert out["gini"] == pytest.approx(0.75)
    assert out["gini_baseline"] == pytest.approx(0.0)  # t = 1 is outside the window
    assert out["excess"] == pytest.approx(0.75)
    assert out["n_zero_effort_periods"] == 1


# ---------- row 3 ----------


def test_quality_means() -> None:
    from gosplan.metrics.phenomena import phenomenon_quality

    cfg = _cfg(quality_matters=True)
    m = cfg.incentive.steps_per_period
    ledger = _ledger([_rec(cfg, quality_acc=0.5 * m), _rec(cfg, quality_acc=1.0 * m)])
    out = phenomenon_quality(ledger, cfg)
    mu = cfg.information.quality_measurability
    assert out["mean_quality"] == pytest.approx(0.75)
    assert out["mean_quality_weighted"] == pytest.approx(1.0 + mu * (0.75 - 1.0))


# ---------- row 5 ----------


def test_hoarding_inflation_and_correlation() -> None:
    from gosplan.metrics.phenomena import phenomenon_hoarding

    cfg = _cfg()
    recs = []
    for t, (stock, fill) in enumerate([(0.1, 1.0), (0.5, 0.8), (0.9, 0.6)], start=2):
        recs.append(
            _rec(
                cfg,
                t_period=t,
                enterprise=0,
                request=(0.0, 2.0),
                need=(0.0, 1.0),
                inv_inputs=(0.0, stock),
            )
        )
        # next period's DELIVER row for good 1's seller (enterprise 1)
        recs.append(
            _rec(
                cfg,
                phase="produce",
                k_step=0,
                t_period=t + 1,
                enterprise=1,
                sector=1,
                fill=fill,
                shipped=1.0,
            )
        )
    base = [
        _rec(cfg, t_period=2, enterprise=0, request=(0.0, 1.0), need=(0.0, 1.0)),
    ]
    out = phenomenon_hoarding(_ledger(recs), _ledger(base), cfg)
    assert out["request_inflation"] == pytest.approx(2.0)
    assert out["request_inflation_baseline"] == pytest.approx(1.0)
    assert out["corr_stock_shortfall"] == pytest.approx(1.0)
    assert math.isnan(out["corr_stock_shortfall_baseline"])


def test_cross_section_fallback() -> None:
    from gosplan.metrics._fallback import cross_section

    same = cross_section(np.ones(6), np.array([0, 0, 0, 1, 1, 1]))
    assert (same.statistic, same.p_value) == (0.0, 1.0)
    split = cross_section(np.array([1.0, 1.1, 1.2, 5.0, 5.1, 5.2]), np.array([1, 1, 1, 0, 0, 0]))
    assert split.p_value < 0.1
    np.testing.assert_allclose(split.group_stats, [1.1, 5.1])
    assert split.n_groups == 2 and split.n_obs == 6


# ---------- row 6 ----------


def test_blat_share() -> None:
    from gosplan.metrics.phenomena import phenomenon_blat

    cfg = _cfg()
    ledger = _ledger(
        [
            _rec(cfg, phase="produce", k_step=0, alloc=(0.0, 2.0), trade_volume=0.5),
            _rec(cfg, phase="produce", k_step=0, enterprise=1, alloc=(2.0, 0.0)),
            _rec(cfg, phase="produce", k_step=0, t_period=1, alloc=(9.0, 0.0), trade_volume=9.0),
        ]
    )
    out = phenomenon_blat(ledger, cfg)
    assert out["trade_volume_share"] == pytest.approx(0.125)
    assert out["n_matched_pairs"] == 1
    assert math.isnan(out["mean_surplus"])


# ---------- row 7 ----------


def test_hidden_reserves_and_reconciliation() -> None:
    from gosplan.metrics.phenomena import phenomenon_hidden_reserves

    cfg = _cfg()
    ledger = _ledger(
        [
            _rec(cfg, inv_output_post=1.5, report=1.0, target=2.0),
            _rec(cfg, t_period=3, inv_output_post=0.5, report=1.0, target=1.0),
        ]
    )
    out = phenomenon_hidden_reserves(ledger, cfg)
    assert out["hidden_reserves"] == pytest.approx((0.25 + 0.0) / 2)
    assert math.isnan(out["reconciliation_stat"])  # fewer than two rows with a next DELIVER


def test_ledger_test_fallback() -> None:
    from gosplan.metrics._fallback import ledger_test

    a = np.array([[0.0, 0.5], [0.25, 0.0], [0.0, 0.5]])
    received = np.array([[0.0, 0.5], [0.25, 0.0], [0.0, 1.0]])  # implied output 1, 1, 2
    honest = ledger_test(np.array([1.0, 1.0, 2.0]), received, a, np.ones(3))
    assert (honest.statistic, honest.p_value) == (0.0, 0.5)
    padded = ledger_test(np.array([1.5, 1.2, 2.9]), received, a, np.array([2.0, 1.0, 1.0]))
    np.testing.assert_allclose(padded.residuals, [1.0, 0.2, 0.9])
    assert padded.statistic > 0 and padded.p_value < 0.05 and padded.n_obs == 3


# ---------- agents (R13.8-9) ----------


def _obs(cfg, log_target=0.0, stock=1.0):
    from gosplan.env.env import obs_spec

    d = len(obs_spec(cfg))
    obs = np.zeros((cfg.supply.n_enterprises, d))
    obs[:, 2] = log_target
    obs[:, 5] = stock
    obs[:, 6] = 1.0
    return obs


def test_heuristic_agents_follow_r13() -> None:
    from gosplan.agents.heuristic import (
        WEITZMAN_MAX_CUT,
        Berliner,
        Kornai,
        TruthfulMyopic,
        Weitzman,
    )

    cfg = _cfg(quality_matters=True)
    rng = np.random.default_rng(0)
    obs = _obs(cfg)
    tm = TruthfulMyopic(cfg)
    e_tm = tm.act(obs, "produce", rng).effort
    assert np.all(tm.act(obs, "produce", rng).quality == 1.0)

    b = Berliner(cfg, safety_factor=0.1)
    np.testing.assert_allclose(b.act(obs, "produce", rng).effort, np.clip(1.1 * e_tm, 0, 1))
    np.testing.assert_array_equal(b.act(obs, "report", rng).report_ratio, 1.0)

    lam = cfg.incentive.ratchet_lambda
    w = Weitzman(cfg)
    np.testing.assert_allclose(
        w.act(obs, "produce", rng).effort, e_tm * (1 - WEITZMAN_MAX_CUT * lam / (1 + lam))
    )
    np.testing.assert_allclose(
        w.act(obs, "report", rng).report_ratio, tm.act(obs, "report", rng).report_ratio
    )

    k = Kornai(cfg, request_inflation=1.5)
    np.testing.assert_array_equal(k.act(obs, "report", rng).input_request, 1.5)
    np.testing.assert_allclose(k.act(obs, "produce", rng).effort, e_tm)
    for agent in (b, w, k):
        assert agent.reset() is None
        assert not np.any(agent.act(obs, "report", rng).trade_offer)


def test_phase1_heuristics_keep_quality_zero() -> None:
    from gosplan.agents.heuristic import TruthfulMyopic

    cfg = _cfg()
    assert not cfg.supply.quality_matters
    out = TruthfulMyopic(cfg).act(_obs(cfg), "produce", np.random.default_rng(0))
    assert np.all(out.quality == 0.0)
