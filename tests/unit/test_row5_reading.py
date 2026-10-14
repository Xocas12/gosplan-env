"""Row 5 (hoarding): what R14's two legs measure on C0, and how the records read them (#109).

Row 5 APPEARS under spec/P2_REVISION.md R14 when two lower CI bounds are above 0: the excess of
request inflation over `TruthfulMyopic`, and the excess of `corr(X_ij, 1 - next-period fill)` over
`TruthfulMyopic`. Run with heuristic agents only, these tests pin the properties that
runs/G3_record.md note H1 rests on:

- a uniform request multiple cancels in the allocation, so inflating at a common multiple moves
  (almost) no inputs;
- `Kornai(3.0)` in every seat passes the inflation leg and moves the correlation leg only by the
  allocation's `1e-6` regularisers, yet R14, which sets no minimum effect size, marks it APPEARS;
- an agent that never inflates a request can pass the correlation leg.

No learned population is loaded and no committed verdict is re-evaluated. The last test checks
that the records which cited row 5 as evidence of hoarding that propagates shortage no longer do.
It checks the withdrawn wording as text, so a re-citation in new words would pass it; review
catches that.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

N_EPISODES = 2
"""Measurement episodes per rollout. Enough for a defined correlation on C0; the statistics here
are differences under common random numbers, not estimates."""


def _c0(seed_index: int = 0):
    from gosplan.experiments.phase2_acceptance import arm_config

    return arm_config("C0", seed_index)


def _rollout(agent, cfg):
    """`agent` through G3's own measurement path, on the post-training measurement seed block."""
    from gosplan.experiments._g2 import MEASURE_SEED_OFFSET
    from gosplan.experiments.phase2_acceptance import _rollout as rollout

    rng = np.random.default_rng(0)
    seed = int(cfg.tech.seed_env) + MEASURE_SEED_OFFSET
    return rollout(lambda obs, phase: agent.act(obs, phase, rng), cfg, N_EPISODES, seed)


def _universal_kornai(seed_index: int):
    """`(cfg, TruthfulMyopic ledger, Kornai(3.0)-in-every-seat ledger)` on one C0 seed."""
    from gosplan.agents.heuristic import Kornai, TruthfulMyopic

    cfg = _c0(seed_index)
    return cfg, _rollout(TruthfulMyopic(cfg), cfg), _rollout(Kornai(cfg, 3.0), cfg)


@dataclasses.dataclass
class _UnderProducer:
    """`TruthfulMyopic` with its effort scaled by `scale`. Requests stay at need: it never inflates."""

    cfg: object
    scale: float

    def act(self, obs, phase, rng):
        from gosplan.agents.heuristic import TruthfulMyopic

        action = TruthfulMyopic(self.cfg).act(obs, phase, rng)
        if phase == "produce":
            action.effort = self.scale * action.effort
        return action


@pytest.fixture(scope="module")
def kornai_seed0():
    return _universal_kornai(0)


def test_uniform_request_multiple_is_allocatively_neutral() -> None:
    from gosplan.env.env import GosplanEnv
    from gosplan.env.planner import allocate, make_planner_view

    cfg = _c0()
    env = GosplanEnv(cfg)
    env.reset(int(cfg.tech.seed_env), cfg.tech.seed_policy)
    view = make_planner_view(env.state, cfg)
    sector = np.asarray(cfg.supply.sector_of)
    need = np.asarray(view.planner_io, dtype=float)[sector] * np.asarray(view.targets)[:, None]
    bound = cfg.tech.request_max_multiple
    assert bound > 1.0 and cfg.incentive.alloc_eta_request > 0.0

    def alloc(requests):
        return allocate(dataclasses.replace(view, requests=requests), cfg)

    at_need, at_bound = alloc(need), alloc(bound * need)
    assert at_need.sum() > 0.0
    # Everyone at need and everyone at the bound: equal up to the 1e-6 regularisers.
    assert np.abs(at_bound - at_need).max() <= 1e-6 * np.abs(at_need).max()

    # Inflation still pays individually: seat 0 alone at need among inflators loses about half
    # its share of every good it needs.
    lone = bound * need
    lone[0] = need[0]
    alone = alloc(lone)
    goods = need[0] > 0.0
    share_common = at_need[0, goods] / at_need[:, goods].sum(axis=0)
    share_alone = alone[0, goods] / alone[:, goods].sum(axis=0)
    assert goods.any()
    assert np.all(share_alone < 0.6 * share_common)


def test_universal_inflation_leaves_the_trajectory_and_the_correlation_unmoved(
    kornai_seed0,
) -> None:
    from gosplan.metrics.phenomena import phenomenon_hoarding

    cfg, base, run = kornai_seed0
    out = phenomenon_hoarding(run, base, cfg)

    assert out["request_inflation"] - out["request_inflation_baseline"] == pytest.approx(2.0)
    # The trajectory is TruthfulMyopic's: input stocks agree up to the regularisers' effect.
    dx = max(
        float(np.abs(np.subtract(r.inv_inputs, b.inv_inputs)).max())
        for r, b in zip(run.records, base.records, strict=True)
    )
    assert dx < 1e-4
    assert np.isfinite(out["corr_stock_shortfall_baseline"])
    assert abs(out["corr_stock_shortfall"] - out["corr_stock_shortfall_baseline"]) < 1e-4


def test_r14_marks_universal_inflation_as_appearing(kornai_seed0) -> None:
    """R14's correlation leg asks only for a lower bound above 0, so an excess of order 1e-7 passes."""
    from gosplan.experiments.phase2_acceptance import heldout_verdicts
    from gosplan.metrics.phenomena import phenomenon_hoarding

    rows = []
    for cfg, base, run in (kornai_seed0, _universal_kornai(1)):
        rows.append(
            {
                "storming": {"excess": 0.0},
                "hoarding": phenomenon_hoarding(run, base, cfg),
                "blat": {"trade_volume_share": 0.0},
                "hidden_reserves": {"hidden_reserves": 0.0},
                "quality": {"mean_quality": 1.0},
            }
        )
    row5 = heldout_verdicts({"C0": rows, "R7_NULL": [], "R3_QW": []})["row5_hoarding"]

    assert row5["corr_excess"][2] < 1e-5
    assert row5["appears"]


def test_an_agent_that_never_inflates_can_pass_the_correlation_leg(kornai_seed0) -> None:
    from gosplan.metrics.phenomena import phenomenon_hoarding

    cfg, base, _ = kornai_seed0
    out = phenomenon_hoarding(_rollout(_UnderProducer(cfg, 0.8), cfg), base, cfg)

    assert out["request_inflation"] == pytest.approx(out["request_inflation_baseline"])
    assert np.isfinite(out["corr_stock_shortfall_baseline"])
    assert out["corr_stock_shortfall"] - out["corr_stock_shortfall_baseline"] > 0.2


WITHDRAWN = {
    "README.md": ("rows 2 and 5 appear",),
    "runs/G3_record.md": ("The direction of row 5 is a result",),
    "runs/G3b_record.md": (
        "This agrees with row 5",
        "hoard rather than barter",
        "learners hoard",
    ),
    "runs/CT_record.md": ("the hoarding result (row 5)",),
}
"""Per record, the wording that cited row 5 as evidence of hoarding that propagates shortage."""


@pytest.mark.parametrize("path", sorted(WITHDRAWN))
def test_records_do_not_cite_row5_as_propagated_shortage(path: str) -> None:
    text = " ".join((REPO_ROOT / path).read_text(encoding="utf-8").split())
    for phrase in WITHDRAWN[path]:
        assert phrase not in text, f"{path} still says {phrase!r}"
    # Each record that read row 5 now points at the corrected reading.
    assert "H1" in text, f"{path} does not point at G3 note H1"
    if path == "runs/G3_record.md":
        assert "allocatively neutral" in text
