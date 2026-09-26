"""WO-029 parity (PLAN section 12.4): the JAX port agrees with the NumPy environment to 1e-5.

Authored by the LEAD. FROZEN BY CONTRACT RULE 2 once landed. 100 agent-steps driven by
`TruthfulMyopic` on the Phase-1 and Phase-2 defaults, plus seeded random actions (which trade, set
quality and inflate requests) on the Phase-2 default and on its setup-cost, sector-aggregation,
backloaded-delivery and smooth-notch branches. A negative control checks that the comparison can
fail: different episode seeds per backend must diverge.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

pytest.importorskip("jax")

TOL = 1e-5


def _with(cfg, **sections):
    for name, values in sections.items():
        cfg = dataclasses.replace(cfg, **{name: dataclasses.replace(getattr(cfg, name), **values)})
    cfg.validate()
    return cfg


@pytest.mark.parametrize("which", ["p1", "p2"])
def test_truthful_myopic_parity(which) -> None:
    from gosplan.config import p1_default_config, p2_default_config
    from gosplan.jax import parity_max_deviation

    cfg = p1_default_config() if which == "p1" else p2_default_config()
    assert parity_max_deviation(cfg, 100) <= TOL


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"supply": {"setup_cost": 0.05}},
        {"information": {"aggregation_level": "sector"}},
        {"supply": {"delivery_timing": "backloaded"}},
        {"incentive": {"notch_width": 0.05}},
    ],
)
def test_random_trading_parity(overrides) -> None:
    from gosplan.config import p2_default_config
    from gosplan.jax import parity_max_deviation

    cfg = _with(p2_default_config(), **overrides)
    assert parity_max_deviation(cfg, 100, agent="random_trade") <= TOL


def test_negative_control_diverges() -> None:
    from gosplan.agents.heuristic import TruthfulMyopic
    from gosplan.config import p2_default_config
    from gosplan.env.env import GosplanEnv
    from gosplan.jax.env import JaxGosplanEnv

    cfg = p2_default_config()
    ref, port = GosplanEnv(cfg, records=False), JaxGosplanEnv(cfg)
    obs_ref, _ = ref.reset(7, 0)
    obs_jax = port.reset(8, 0)
    agent = TruthfulMyopic(cfg)
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(20):
        action = agent.act(obs_ref, ref.phase(), rng)
        obs_ref, *_ = ref.step(action)
        obs_jax, *_ = port.step(action)
        worst = max(worst, float(np.abs(obs_ref - obs_jax).max()))
    assert worst > 1e-3
