"""The entropy anneal, with and without the R19 anneal horizon (spec/P2_REVISION.md R19)."""

from __future__ import annotations

import dataclasses

import pytest


def test_default_anneals_over_the_whole_run() -> None:
    from gosplan.agents.ppo.adapter import PPOConfig
    from gosplan.agents.ppo.train import entropy_coefficient

    ppo = PPOConfig()
    assert ppo.entropy_anneal_updates is None
    assert entropy_coefficient(0, 3000, ppo) == pytest.approx(0.01)
    assert entropy_coefficient(1499, 3000, ppo) == pytest.approx(0.01 - 0.009 * 1499 / 2999)
    assert entropy_coefficient(2999, 3000, ppo) == pytest.approx(0.001)
    assert entropy_coefficient(0, 1, ppo) == pytest.approx(0.01)


def test_horizon_anneals_then_holds() -> None:
    from gosplan.agents.ppo.adapter import PPOConfig
    from gosplan.agents.ppo.train import entropy_coefficient

    ppo = dataclasses.replace(PPOConfig(), entropy_anneal_updates=1000)
    one_m = PPOConfig()
    for u in (0, 250, 500, 999):  # identical to a 1,000-update run's schedule
        assert entropy_coefficient(u, 3000, ppo) == pytest.approx(
            entropy_coefficient(u, 1000, one_m)
        )
    for u in (1000, 2000, 2999):
        assert entropy_coefficient(u, 3000, ppo) == pytest.approx(0.001)


def test_unset_horizon_keeps_the_stored_ppo_record_unchanged() -> None:
    from gosplan.agents.ppo.adapter import PPOConfig

    record = dataclasses.asdict(PPOConfig())
    assert record["entropy_anneal_updates"] is None  # dropped from the stored record by _g2
