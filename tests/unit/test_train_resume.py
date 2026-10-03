"""Resumable training is exact: a killed and resumed run equals an uninterrupted one bit for bit.

Infrastructure for long labelled studies (spec/P2_REVISION.md R19 record): container restarts must
not change a single number, so this compares final parameters and every logged training row.
"""

from __future__ import annotations

import json

import numpy as np
import pytest


def _cfg():
    from gosplan.config import p1_default_config

    return p1_default_config()


def _train_cfg(root, resume_every):
    from gosplan.agents.ppo.adapter import PPOConfig
    from gosplan.agents.ppo.train import TrainConfig

    return TrainConfig(
        env_cfg=_cfg(),
        ppo_cfg=PPOConfig(),
        n_envs=2,
        rollout_steps=25,
        total_agent_steps=2 * 25 * 8,  # 8 updates
        eval_every_updates=3,
        eval_episodes=1,
        checkpoint_every_updates=8,
        run_root=root,
        resume_every_updates=resume_every,
    )


def _rows(run_dir):
    drop = ("wall_clock_s", "steps_per_second", "update_steps_per_second")
    rows = [json.loads(x) for x in (run_dir / "train_log.jsonl").read_text().splitlines() if x]
    return [{k: v for k, v in r.items() if k not in drop} for r in rows]


def _params(run_dir):
    path = sorted((run_dir / "checkpoints").glob("*.ckpt*"))[-1]
    with np.load(path, allow_pickle=False) as data:
        return {k: np.asarray(data[k]) for k in data.files}


def test_resumed_run_equals_uninterrupted_run(tmp_path, monkeypatch) -> None:
    from gosplan.agents.ppo import train as train_mod

    ref = train_mod.train(_train_cfg(tmp_path / "ref", resume_every=0))

    real_log = train_mod.log_update
    calls = {"n": 0}

    def crash_once(run_dir, row):
        real_log(run_dir, row)
        if row["update"] == 4 and calls["n"] == 0:
            calls["n"] += 1
            raise KeyboardInterrupt("simulated container restart")

    monkeypatch.setattr(train_mod, "log_update", crash_once)
    with pytest.raises(KeyboardInterrupt):
        train_mod.train(_train_cfg(tmp_path / "res", resume_every=2))
    run_dir = tmp_path / "res" / _cfg().hash()
    assert (run_dir / train_mod.RESUME_STATE).exists()
    monkeypatch.setattr(train_mod, "log_update", real_log)
    res = train_mod.train(_train_cfg(tmp_path / "res", resume_every=2))

    assert _rows(res) == _rows(ref)
    a, b = _params(ref), _params(res)
    assert a.keys() == b.keys()
    for k in a:
        assert np.array_equal(a[k], b[k]), k
    assert not (res / train_mod.RESUME_STATE).exists()  # cleared on completion


def test_a_snapshot_from_another_run_is_ignored(tmp_path) -> None:
    import dataclasses

    from gosplan.agents.ppo.train import _load_resume, _save_resume

    tc = _train_cfg(tmp_path, resume_every=2)
    from gosplan.agents.ppo.adapter import IPPO

    run_dir = tmp_path / "x"
    run_dir.mkdir()
    _save_resume(run_dir, tc, IPPO(_cfg(), tc.ppo_cfg), {"next_update": 2})
    assert _load_resume(run_dir, tc) is not None
    other = dataclasses.replace(tc, total_agent_steps=tc.total_agent_steps * 2)
    assert _load_resume(run_dir, other) is None
