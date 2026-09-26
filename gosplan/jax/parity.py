"""WO-029 parity: the JAX port against the NumPy reference under the same keyed draws.

`parity_max_deviation(cfg, steps)` drives `gosplan.env.env.GosplanEnv` and
`gosplan.jax.env.JaxGosplanEnv` side by side for `steps` agent-steps from the same seeds and returns
the maximum absolute deviation over every observation entry, every reward and the `done` flag
(a `done` mismatch is an infinite deviation). The acceptance criterion (PLAN section 12.4) is
`<= 1e-5` over 100 agent-steps driven by `TruthfulMyopic`; `agent="random_trade"` additionally
exercises the trade matcher, quality and requests with seeded random actions fed identically to both
backends.
"""

from __future__ import annotations

import numpy as np

from gosplan.config import EnvConfig


def _random_trade_action(cfg: EnvConfig, rng: np.random.Generator):
    from gosplan.env.state import EnterpriseAction

    n, j = cfg.supply.n_enterprises, cfg.supply.n_sectors
    return EnterpriseAction(
        effort=rng.uniform(0.0, 1.0, n),
        quality=rng.uniform(0.0, 1.0, n),
        invest=np.zeros(n),
        report_ratio=rng.uniform(0.0, cfg.tech.report_max_ratio, n),
        input_request=rng.uniform(0.0, cfg.tech.request_max_multiple, (n, j)),
        trade_offer=rng.uniform(-1.0, 1.0, (n, j)),
    )


def parity_max_deviation(
    cfg: EnvConfig, steps: int = 100, agent: str = "truthful", seed_env: int = 7
) -> float:
    """Max |NumPy - JAX| over observations, rewards and `done` for `steps` agent-steps."""
    from gosplan.agents.heuristic import TruthfulMyopic
    from gosplan.env.env import GosplanEnv
    from gosplan.jax.env import JaxGosplanEnv

    ref = GosplanEnv(cfg, records=False)
    port = JaxGosplanEnv(cfg)
    obs_ref, _ = ref.reset(seed_env, 0)
    obs_jax = port.reset(seed_env, 0)
    worst = float(np.max(np.abs(obs_ref - obs_jax)))
    tm = TruthfulMyopic(cfg)
    rng = np.random.default_rng(seed_env)
    for _ in range(steps):
        if agent == "truthful":
            action = tm.act(obs_ref, ref.phase(), rng)
        elif agent == "random_trade":
            action = _random_trade_action(cfg, rng)
        else:
            raise ValueError(f"parity_max_deviation: unknown agent {agent!r}")
        obs_ref, r_ref, d_ref, _ = ref.step(action)
        obs_jax, r_jax, d_jax = port.step(action)
        if d_ref != d_jax or obs_ref.shape != obs_jax.shape:
            return float("inf")
        worst = max(
            worst, float(np.max(np.abs(obs_ref - obs_jax))), float(np.max(np.abs(r_ref - r_jax)))
        )
    return worst
