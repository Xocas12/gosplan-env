"""WO-027 oracle (PLAN section 6.2; P2 spec revision): result keys, bound properties, parity.

Authored by the LEAD with the oracle formulation (`gosplan.oracle.kantorovich.FORMULATION`).
FROZEN BY CONTRACT RULE 2 once landed. The brute-force parity case enumerates every combination of
per-period effort level and shipping fraction on a symmetric two-enterprise, two-good economy under
the oracle's own timing (formulation note F4), routing the non-consumer share to the other
enterprise, and checks that the LP welfare is at least the best enumerated welfare and within the
enumeration's grid error of it.
"""

from __future__ import annotations

import dataclasses
import itertools

import numpy as np
import pytest


def _two_by_two(sigma: float = 0.1):
    """Two enterprises, each the only producer of its own good, each needing the other's."""
    from gosplan.config import p1_default_config

    base = p1_default_config()
    supply = dataclasses.replace(
        base.supply,
        n_enterprises=2,
        n_sectors=2,
        sector_of=(0, 1),
        io_matrix=((0.0, 0.2), (0.2, 0.0)),
        final_demand_share=(0.5, 0.5),
        productivity=(1.0, 1.0),
        yield_sigma=(sigma, sigma),
        ces_alpha=(0.5, 0.5),
    )
    cfg = dataclasses.replace(base, supply=supply)
    cfg.validate()
    return cfg


def test_result_carries_every_required_key() -> None:
    """Every `ORACLE_RESULT_KEYS` entry is present, the status is optimal and the gap is tiny."""
    from gosplan.oracle.kantorovich import ORACLE_RESULT_KEYS, solve_oracle

    result = solve_oracle(_two_by_two(), 6, False, None)
    for key in ORACLE_RESULT_KEYS:
        assert key in result, key
    assert result["status"] == "optimal"
    assert result["optimality_gap"] < 1e-6
    assert result["welfare"] > 0


def test_clairvoyant_equals_expected_value_without_noise() -> None:
    """With zero yield noise the clairvoyant bound and the expected-value solution coincide."""
    from gosplan.oracle.kantorovich import solve_oracle

    cfg = _two_by_two(sigma=0.0)
    ev = solve_oracle(cfg, 6, False, None)
    cv = solve_oracle(cfg, 6, True, 3)
    assert abs(ev["welfare"] - cv["welfare"]) < 1e-6


def test_clairvoyant_requires_a_seed() -> None:
    from gosplan.oracle.kantorovich import solve_oracle

    with pytest.raises(ValueError):
        solve_oracle(_two_by_two(), 6, True, None)


def _simulate(cfg, plan, horizon: int) -> float:
    """Welfare of a fixed symmetric plan `((effort, ship_fraction), ...)` under the oracle's own
    timing (F4), routing the non-consumer share to the other enterprise's input stock."""
    from gosplan.env.state import initial_targets
    from gosplan.oracle.kantorovich import _ces

    sup = cfg.supply
    a = np.asarray(sup.io_matrix, dtype=float)
    phi = np.asarray(sup.final_demand_share, dtype=float)
    h, h_x = sup.holding_loss, sup.input_holding_loss
    t0 = np.asarray(initial_targets(cfg), dtype=float)
    x = a * t0[:, None]
    stock = np.zeros(2)
    welfare = []
    for t in range(horizon):
        effort, fraction = plan[t]
        ship = fraction * stock
        cons = phi * ship
        x = x + np.array([[0.0, (1 - phi[1]) * ship[1]], [(1 - phi[0]) * ship[0], 0.0]])
        coverage = np.min(np.where(a > 0, x / np.where(a > 0, a, 1.0), np.inf), axis=1)
        y = np.minimum(effort, coverage)
        x = (1 - h_x) * (x - a * y[:, None])
        stock = np.minimum((1 - h) * (stock - ship) + y, 3.0)
        if t >= 2:
            welfare.append(_ces(cons, cfg))
    return float(np.mean(welfare))


def test_lp_dominates_brute_force_and_matches_it_within_grid_error() -> None:
    """Brute-force parity at N = 2 over a 4-period horizon (WO-027 acceptance criterion)."""
    from gosplan.oracle.kantorovich import solve_oracle

    cfg = _two_by_two(sigma=0.0)
    horizon = 4
    choices = list(itertools.product([0.5, 1.0], [0.0, 1 / 3, 2 / 3, 1.0]))
    best = 0.0
    for plan in itertools.product(choices, repeat=horizon):
        best = max(best, _simulate(cfg, plan, horizon))
    lp = solve_oracle(cfg, horizon, False, None)["welfare"]
    assert lp >= best - 1e-9
    assert lp <= best * 1.02 + 1e-9
