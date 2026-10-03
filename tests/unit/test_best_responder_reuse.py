"""A completed best responder is reused, not retrained, and equals the trained one exactly."""

from __future__ import annotations

import dataclasses

import numpy as np


def test_completed_best_responder_is_reused(tmp_path, monkeypatch) -> None:
    from gosplan.agents.ppo import train as train_mod
    from gosplan.agents.ppo.adapter import IPPO, PPOConfig
    from gosplan.config import p1_default_config
    from gosplan.experiments import exploitability as ex
    from gosplan.experiments.phase1_gate import GATE_SIZING

    cfg = p1_default_config()
    sizing = dataclasses.replace(
        GATE_SIZING, n_envs=4, rollout_steps=25, total_agent_steps=4 * 25 * 3, eval_episodes=1
    )
    pop = IPPO(cfg, PPOConfig())
    first = ex.train_best_responder(cfg, pop, 0, sizing, tmp_path)

    def no_training(*_a, **_k):
        raise AssertionError("a completed best responder must not be retrained")

    monkeypatch.setattr(train_mod, "make_env_batch", no_training)
    second = ex.train_best_responder(cfg, pop, 0, sizing, tmp_path)
    p1, p2 = tmp_path / "a.ckpt", tmp_path / "b.ckpt"
    first.save_checkpoint(p1)
    second.save_checkpoint(p2)
    with np.load(p1) as a, np.load(p2) as b:
        for k in a.files:
            assert np.array_equal(a[k], b[k]), k
