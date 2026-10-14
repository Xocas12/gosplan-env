"""The best responder trains for the budget its study asked for (spec/P2_REVISION.md R16; #108).

The audit runs in a `spawn` process pool, and a spawned worker re-imports the harness module, so a
budget that only lives in module state of the parent never reaches it. G3b, LC and ES set R16's
1M agent-steps that way, and all 30 of their best responders trained for R14's 500k. These tests
pin the two halves of the fix: each R16 study hands its budget to `phase2_acceptance.run`, and a
budget handed to the audit job is the one the best responder's manifest records, in a worker of
the real pool. The Phase-3 contrasts' audit (`contrasts._exploit_one`) takes its budget the same
way and is checked the same way.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

R16_BUDGET = 1_000_000


def _audit_without_training(task: tuple) -> dict:
    """Runs inside the spawned worker. `task` is `(harness, job)`, and the worker calls that
    harness's own audit function on `job`: `phase2_acceptance._run_exploit` or
    `contrasts._exploit_one`. Two stand-ins are installed in this worker only, so that the audit
    takes seconds instead of hours. The best responder skips its PPO updates and is saved by
    `_save_best_responder`, the same code that saves a trained one, from the sizing
    `exploitability.run` hands it. Seat returns are 0, which skips the evaluation rollouts. Neither
    stand-in reads or changes the sizing."""
    import time

    from gosplan.agents.ppo.adapter import IPPO
    from gosplan.experiments import contrasts
    from gosplan.experiments import exploitability as ex
    from gosplan.experiments import phase2_acceptance as pa

    def untrained_best_responder(cfg, pop_agent, seat, sizing, run_root, warm_start=None):
        agent = IPPO(cfg, pop_agent.ppo)
        ex._save_best_responder(agent, cfg, seat, sizing, run_root, time.perf_counter())
        return agent

    ex.train_best_responder = untrained_best_responder
    ex.seat_return = lambda *_args, **_kwargs: 0.0
    harness, job = task
    audit = {"phase2_acceptance": pa._run_exploit, "contrasts": contrasts._exploit_one}[harness]
    return audit(job)


def _train_config(manifest: dict) -> dict:
    """The `TrainConfig` record a best-responder manifest carries in its flags."""
    prefix = "train_config="
    (flag,) = [f for f in manifest["flags"] if f.startswith(prefix)]
    return json.loads(flag[len(prefix) :])


@pytest.mark.parametrize("harness", ["phase2_acceptance", "contrasts"])
def test_requested_budget_reaches_the_spawned_audit_worker(harness, tmp_path) -> None:
    from gosplan.agents.ppo.adapter import IPPO
    from gosplan.agents.ppo.train import checkpoint_path
    from gosplan.experiments import contrasts
    from gosplan.experiments import phase2_acceptance as pa
    from gosplan.experiments.phase1_gate import study_ppo_config
    from gosplan.metrics.ledger import write_manifest

    # One population run in the layout `exploitability.run` reads: a manifest and a checkpoint.
    cfg = pa.arm_config("C0", 0)
    audit_dir = tmp_path / "exploit_subset" / "C0"
    pop_dir = audit_dir / cfg.hash()
    (pop_dir / "checkpoints").mkdir(parents=True)
    IPPO(cfg, study_ppo_config()).save_checkpoint(checkpoint_path(pop_dir, 0))
    write_manifest(str(pop_dir), cfg, {})

    request = pa.br_sizing(R16_BUDGET)
    assert request.total_agent_steps == R16_BUDGET
    assert pa.BR_TOTAL_AGENT_STEPS != R16_BUDGET  # the module default the worker would fall back to

    # `_pool` sets these two with setdefault for its workers; put the test process back afterwards.
    saved_env = {k: os.environ.get(k) for k in ("XLA_FLAGS", "OMP_NUM_THREADS")}
    try:
        task = (harness, ("C0", audit_dir, tmp_path, request))
        if harness == "phase2_acceptance":
            (result,) = pa._pool(_audit_without_training, [task], 1)
        else:
            (result,) = contrasts._pool(_audit_without_training, [task])
    finally:
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    br_dir = tmp_path / "exploitability" / "runs" / "br_seat0" / cfg.hash()
    manifest = json.loads((br_dir / "manifest.json").read_text(encoding="utf-8"))
    train_cfg = _train_config(manifest)
    assert train_cfg["total_agent_steps"] == R16_BUDGET
    n_updates = R16_BUDGET // (request.n_envs * request.rollout_steps)
    assert train_cfg["n_updates"] == n_updates
    assert checkpoint_path(br_dir, n_updates - 1).exists()
    report = (tmp_path / "exploitability" / "C0_report.md").read_text(encoding="utf-8")
    assert f"total_agent_steps={R16_BUDGET}" in report
    assert len(result["per_seed"]) == 1


class _Stop(Exception):
    """Raised by the stand-in `run` once it has recorded its arguments."""


@pytest.mark.parametrize("study", ["G3b", "LC", "ES"])
def test_r16_studies_hand_their_budget_to_the_harness(study, monkeypatch) -> None:
    import inspect

    from gosplan.experiments import entropy_schedule, learner_convergence
    from gosplan.experiments import phase2_acceptance as pa

    # A driver that still set the budget on the module would leak it into later tests; undo it.
    monkeypatch.setattr(pa, "BR_TOTAL_AGENT_STEPS", pa.BR_TOTAL_AGENT_STEPS)
    signature = inspect.signature(pa.run)
    calls: list[dict] = []

    def recording_run(*args, **kwargs):
        calls.append(signature.bind(*args, **kwargs).arguments)
        raise _Stop

    monkeypatch.setattr(pa, "run", recording_run)
    if study == "G3b":
        monkeypatch.setattr(sys, "argv", ["phase2_acceptance", "--g3b"])
        driver = pa.main
    elif study == "LC":
        monkeypatch.setattr(sys, "argv", ["learner_convergence"])
        driver = learner_convergence.main
    else:
        monkeypatch.setattr(sys, "argv", ["entropy_schedule"])
        driver = entropy_schedule.main
    with pytest.raises(_Stop):
        driver()
    (arguments,) = calls
    assert arguments.get("br_steps") == R16_BUDGET, f"{study} asked for {arguments.get('br_steps')}"
