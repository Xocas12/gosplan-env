"""Labelled study CT (spec/P2_REVISION.md R18): the pre-registered decision rules."""

from __future__ import annotations


def _row(r_pop, r_dev, w_pop, w_tm):
    return {"R_pop": r_pop, "R_dev": r_dev, "W_pop": w_pop, "W_tm": w_tm}


def test_trap_needs_both_conditions() -> None:
    from gosplan.experiments.coordination_trap import verdicts

    trap = [_row(0.0, -1.0 - 0.1 * i, 0.0, 1.0 + 0.1 * i) for i in range(10)]
    v = verdicts(trap)
    assert v["unilateral_unprofitable"] and v["all_production_pays"] and v["trap"]

    deviation_pays = [_row(0.0, 0.5 + 0.1 * i, 0.0, 1.0 + 0.1 * i) for i in range(10)]
    v = verdicts(deviation_pays)
    assert not v["unilateral_unprofitable"] and v["all_production_pays"] and not v["trap"]

    no_gain = [_row(0.0, -1.0 - 0.1 * i, 1.0, 0.5 - 0.1 * i) for i in range(10)]
    v = verdicts(no_gain)
    assert v["unilateral_unprofitable"] and not v["all_production_pays"] and not v["trap"]


def test_mixed_signs_are_not_significant() -> None:
    from gosplan.experiments.coordination_trap import verdicts

    rows = [_row(0.0, (-1) ** i * 1.0, 0.0, (-1) ** i * 1.0) for i in range(10)]
    v = verdicts(rows)
    assert not v["unilateral_unprofitable"] and not v["all_production_pays"] and not v["trap"]


def test_median_ci_is_deterministic() -> None:
    from gosplan.experiments.coordination_trap import median_ci

    x = [0.1, 0.4, -0.2, 0.3, 0.0]
    assert median_ci(x) == median_ci(x)
    m, lo, hi = median_ci(x)
    assert lo <= m <= hi
