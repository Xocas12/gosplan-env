"""WO-018 smoke test: 100 PPO updates at `N = 1` complete and leave a complete run record.

Realises: the WO-018 must-pass line of PLAN section 12.3, "smoke test (100 updates on N = 1)",
whose path the card names as `tests/unit/test_train_smoke.py` and assigns to the lead (only the
lead writes frozen tests, CONTRACT rule 2). Authored by the LEAD alongside AMBIGUITY-016. Module
under test: `gosplan/agents/ppo/train.py` (with the WO-017 adapter underneath).

FROZEN BY CONTRACT RULE 2 once landed. It asserts the *shape* of a run - it completes, writes the
manifest with every CONTRACT rule 10 field, logs every update, and leaves a checkpoint that loads -
and nothing about what the policy learned: a learned behaviour is a finding, never a test
expectation.
"""

from __future__ import annotations

import json

import pytest


def _single_enterprise_cfg():
    """`p1_default_config()` cut to one enterprise and one sector with no input-output (`a = 0`),
    the DP's setting (PLAN section 5) and WO-019's."""
    import dataclasses

    from gosplan.config import p1_default_config

    base = p1_default_config()
    supply = dataclasses.replace(
        base.supply,
        n_enterprises=1,
        n_sectors=1,
        sector_of=(0,),
        io_matrix=((0.0,),),
        final_demand_share=(1.0,),
        productivity=(1.0,),
        yield_sigma=(base.supply.yield_sigma[0],),
        ces_alpha=(1.0,),
    )
    cfg = dataclasses.replace(base, supply=supply)
    cfg.validate()
    return cfg


@pytest.mark.skeleton
def test_one_hundred_updates_at_n_equals_one_complete_with_a_full_manifest(
    tmp_path, implemented
) -> None:
    """100 updates at `N = 1` finish and write manifest, log and a loadable checkpoint.

    Assertion: `train(TrainConfig(...))` with `n_envs = 2`, `rollout_steps = 10`,
    `total_agent_steps = 2000` (exactly 100 updates) returns `run_root / env_cfg.hash()`; its
    `manifest.json` has every key of `MANIFEST_FIELDS` with `reference_ppo_version` a non-empty
    string; `train_log.jsonl` has one row per update (100); at least one checkpoint exists under
    `checkpoints/` and `IPPO.load_checkpoint` restores it into a fresh agent without error.
    """
    from gosplan.agents.ppo.adapter import IPPO, PPOConfig
    from gosplan.agents.ppo.train import TrainConfig, train
    from gosplan.metrics.ledger import MANIFEST_FIELDS

    implemented(train)
    cfg = _single_enterprise_cfg()
    train_cfg = TrainConfig(
        env_cfg=cfg,
        ppo_cfg=PPOConfig(),
        n_envs=2,
        rollout_steps=10,
        total_agent_steps=2000,
        eval_every_updates=50,
        eval_episodes=2,
        checkpoint_every_updates=50,
        run_root=tmp_path,
    )
    run_dir = train(train_cfg)
    assert run_dir == tmp_path / cfg.hash()

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    for key in MANIFEST_FIELDS:
        assert key in manifest, key
    assert isinstance(manifest["reference_ppo_version"], str)
    assert manifest["reference_ppo_version"].strip()

    rows = (run_dir / "train_log.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 100

    checkpoints = sorted((run_dir / "checkpoints").glob("*"))
    assert checkpoints
    fresh = IPPO(cfg, PPOConfig())
    fresh.load_checkpoint(checkpoints[-1])
