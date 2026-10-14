"""CT's lone-producer leg and the self-enforcement check of #106.

R18's test (i) compares each learned population with seat 0 playing the truthful-myopic producer.
#106 found that this producer's return is the autarky payoff, the return it earns when the other
seats do nothing, and that all-production is not self-enforcing under C0, because an enterprise
among producers gains by dropping quality. The CT, LC and ES records rest on both statements;
these tests make them checkable. Every seat is a heuristic, so no checkpoint is needed.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _c0(seed_index: int = 0):
    from gosplan.experiments.phase2_acceptance import arm_config

    return arm_config("C0", seed_index)


def test_lone_producer_return_is_the_committed_r_dev(monkeypatch) -> None:
    """Under CT's own protocol, the producer's return against inert seats is the R_dev CT
    committed for the LC 3M population of seed_env 1000, and the one ES committed."""
    from gosplan.experiments.coordination_trap import (
        CT_EPISODES,
        SEAT,
        committed_populations,
        heuristic_seat_returns,
    )

    monkeypatch.chdir(ROOT)
    cfg = _c0(0)
    alone = heuristic_seat_returns(cfg, "alone", CT_EPISODES)[SEAT]
    committed = committed_populations()
    seed_env = int(cfg.tech.seed_env)
    assert abs(committed["LC 3M"][seed_env]["R_dev"] - alone) < 1e-9
    assert abs(committed["ES 3M"][seed_env]["R_dev"] - alone) < 1e-9
    assert committed["LC 3M"][seed_env]["qbar"] < 1e-3


def test_zero_quality_suppliers_deliver_nothing_usable() -> None:
    """Producers at quality 0 leave seat 0 exactly where inert seats do (a delivery is credited
    `deliv * qbar`), while producers at quality 1 leave it better off."""
    from gosplan.experiments.coordination_trap import SEAT, heuristic_seat_returns

    cfg = _c0(0)
    alone = heuristic_seat_returns(cfg, "alone", 3)[SEAT]
    amid_q0 = heuristic_seat_returns(cfg, "amid_q0", 3)[SEAT]
    producers = heuristic_seat_returns(cfg, "producers", 3)[SEAT]
    assert abs(amid_q0 - alone) < 1e-9
    assert producers > alone


def test_dropping_quality_pays_among_producers() -> None:
    """All-production is not self-enforcing under C0: among truthful-myopic producers, seat 0
    gains by playing the same policy at quality 0."""
    from gosplan.experiments.coordination_trap import self_enforcement

    r = self_enforcement(_c0(0), n_episodes=3)
    assert r["quality_gain"] == r["R_producers_seat0_q0"] - r["R_producers"]
    assert r["quality_gain"] > 0.0


def test_unknown_condition_is_rejected() -> None:
    import pytest

    from gosplan.experiments.coordination_trap import heuristic_seat_returns

    with pytest.raises(ValueError, match="unknown condition"):
        heuristic_seat_returns(_c0(0), "everyone_shirks", 1)
