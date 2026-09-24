"""Audit selection is keyed by the episode's seed, not the configuration's root seed.

Realises: PLAN sections 2.7.4 and 2.15 (every environment draw keyed from the episode's `seed_env`)
and the LEAD ruling AMBIGUITY-019 B. Authored by the LEAD after the WO-019 pilot found that keying
`select_audits` on `cfg.tech.seed_env` gave every training episode the same audit schedule, which
PPO learned to exploit.

FROZEN BY CONTRACT RULE 2 once landed. It asserts only the keying of the draw, nothing about any
agent's behaviour.
"""

from __future__ import annotations

import numpy as np


def _audit_schedule(cfg, seed_env: int, n_periods: int) -> np.ndarray:
    from gosplan.agents.heuristic import TruthfulMyopic
    from gosplan.env.env import GosplanEnv

    env = GosplanEnv(cfg)
    agent = TruthfulMyopic(cfg)
    rng = np.random.default_rng(0)
    obs, _ = env.reset(seed_env, 0)
    rows = []
    while len(rows) < n_periods:
        phase = env.phase()
        obs, _reward, done, _info = env.step(agent.act(obs, phase, rng))
        if phase == "report":
            rows.append(np.asarray(env.state.last_audited, dtype=bool).copy())
        if done:
            obs, _ = env.reset(seed_env, 0)
            break
    return np.array(rows)


def test_audit_schedule_differs_across_episode_seeds_and_repeats_within_one(p1_cfg) -> None:
    """Two episode seeds under one configuration draw different audit schedules; the same seed
    draws the same one. Assertion: over the periods both episodes survive, the schedules for
    `seed_env = 1` and `seed_env = 2` differ, and re-running `seed_env = 1` reproduces it exactly.
    """
    a = _audit_schedule(p1_cfg, 1, 40)
    b = _audit_schedule(p1_cfg, 2, 40)
    again = _audit_schedule(p1_cfg, 1, 40)
    n = min(len(a), len(b))
    assert n > 0
    assert not np.array_equal(a[:n], b[:n])
    assert np.array_equal(a, again)
