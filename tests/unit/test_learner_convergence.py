"""Labelled study LC (spec/P2_REVISION.md R17): the pre-registered paired analysis."""

from __future__ import annotations

import pytest


def _rows(values, root=1000):
    return [{"seed_env": root + i, "exploitability_ratio": v} for i, v in enumerate(values)]


def test_paired_difference_and_decision_rules() -> None:
    from gosplan.experiments.learner_convergence import paired_exploitability

    base = _rows([0.9, 0.8, 0.7, 0.95, 0.85, 0.6, 0.99, 0.75, 0.8, 0.9])
    new = _rows([0.02, 0.01, 0.03, 0.04, 0.0, 0.01, 0.02, 0.03, 0.01, 0.02])
    out = paired_exploitability(base, new)
    assert out["seeds"] == list(range(1000, 1010))
    assert out["median_diff"] < 0 and out["median_diff_ci"][1] < 0
    assert out["falls"] and out["converged_new"]
    assert out["n_new_above_threshold"] == 0


def test_no_change_is_neither_a_fall_nor_convergence() -> None:
    from gosplan.experiments.learner_convergence import paired_exploitability

    base = _rows([0.9, 0.8, 0.7, 0.95, 0.85])
    out = paired_exploitability(base, base)
    assert out["median_diff"] == pytest.approx(0.0)
    assert not out["falls"] and not out["converged_new"]
    assert out["n_new_above_threshold"] == 5


def test_seeds_are_paired_on_seed_env_not_order() -> None:
    from gosplan.experiments.learner_convergence import paired_exploitability

    base = _rows([0.5, 0.6, 0.7])
    new = list(reversed(_rows([0.4, 0.5, 0.6])))
    out = paired_exploitability(base, new)
    assert out["diff"] == pytest.approx([-0.1, -0.1, -0.1])


def test_render_reports_both_rules() -> None:
    from gosplan.experiments.learner_convergence import paired_exploitability, render

    paired = paired_exploitability(_rows([0.9, 0.8]), _rows([0.5, 0.6]))
    res = {
        "prices": {"welfare_ratio": 0.03},
        "verdicts": {"row6_blat": {"trade_volume_share": [0.0, 0.0, 0.0]}},
    }
    offer = {"sell_share": 0.0, "buy_share": 1.0, "mean_offer": -0.9, "n_offers": 10}
    text = render(res, res, paired, {"1M": [offer], "3M": [offer]}, "# Gate G3\n\nbody")
    assert "Exploitability falls with budget" in text and "Converged at 3M" in text
    assert "## Gate G3" in text


def test_trajectory_table_takes_medians_at_eval_points() -> None:
    from gosplan.experiments.learner_convergence import trajectory_table

    def log(efforts):
        rows = [{"update": 0, "agent_steps_total": 1000, "entropy_coef": 0.01}]
        rows += [
            {
                "agent_steps_total": 250_000 * (i + 1),
                "entropy_coef": 0.01 - 0.001 * i,
                "eval_mean_effort": e,
                "eval_mean_return": -e,
            }
            for i, e in enumerate(efforts)
        ]
        return rows

    out = trajectory_table({"3M": [log([0.6, 0.1]), log([0.8, 0.3]), log([0.7, 0.2])]})
    assert out["3M"]["k_steps"] == [250, 500]
    assert out["3M"]["median_effort"] == [0.7, 0.2]
    assert out["3M"]["median_return"] == [-0.7, -0.2]
