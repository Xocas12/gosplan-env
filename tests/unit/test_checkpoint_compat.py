"""Checkpoint matching ignores schedule-only PPOConfig fields, and nothing else (R19)."""

from __future__ import annotations

import dataclasses

import pytest


def _agent(ppo):
    from gosplan.agents.ppo.adapter import IPPO
    from gosplan.config import p1_default_config

    return IPPO(p1_default_config(), ppo)


def test_anneal_horizon_does_not_block_loading(tmp_path) -> None:
    from gosplan.agents.ppo.adapter import PPOConfig

    path = tmp_path / "a.ckpt"
    _agent(dataclasses.replace(PPOConfig(), entropy_anneal_updates=1000)).save_checkpoint(path)
    _agent(PPOConfig()).load_checkpoint(path)  # R16 warm start of an R19 population
    path2 = tmp_path / "b.ckpt"
    _agent(PPOConfig()).save_checkpoint(path2)
    _agent(dataclasses.replace(PPOConfig(), entropy_anneal_updates=1000)).load_checkpoint(path2)


def test_pre_r19_checkpoint_format_still_loads(tmp_path) -> None:
    import json

    import numpy as np

    from gosplan.agents.ppo.adapter import PPOConfig

    path = tmp_path / "old.ckpt"
    _agent(PPOConfig()).save_checkpoint(path)
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: np.asarray(data[k]) for k in data.files}
    record = json.loads(str(arrays["meta_ppo_config"]))
    record.pop("entropy_anneal_updates")  # as written before the field existed
    arrays["meta_ppo_config"] = np.asarray(json.dumps(record, sort_keys=True))
    with open(path, "wb") as handle:
        np.savez(handle, **arrays)
    _agent(PPOConfig()).load_checkpoint(path)


def test_other_ppo_fields_still_block_loading(tmp_path) -> None:
    from gosplan.agents.ppo.adapter import PPOConfig

    path = tmp_path / "c.ckpt"
    _agent(PPOConfig()).save_checkpoint(path)
    with pytest.raises(ValueError, match="PPOConfig differs"):
        _agent(dataclasses.replace(PPOConfig(), learning_rate=1e-3)).load_checkpoint(path)
