"""Shared machinery of the gate G2 experiments (WO-019 `dp_vs_ppo`, WO-020 `phase1_gate`).

Private to `gosplan/experiments/`. It holds only what both cards need and neither card owns: reading
the gate G1 record, deriving the per-seed and per-level configurations, the run sizing fixed by the
lead before any G2 run (AMBIGUITY-019), a spawn-context process pool over training runs that skips
runs already completed with the same sizing, and the post-training measurement of one run.

Nothing here chooses a parameter. The `a * pen` levels, the Phase-1 values and the criterion-2
threshold are read from `runs/G1_decision.md`; the budgets are the AMBIGUITY-019 ruling, recorded
before the first G2 run.
"""

from __future__ import annotations

import dataclasses
import json
import multiprocessing
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from gosplan.config import EnvConfig

G1_DECISION_PATH = Path("runs/G1_decision.md")
"""The gate G1 record (PLAN section 13)."""

MEASURE_SEED_OFFSET = 2 * 10**6
"""Episode-seed block of the post-training gate measurement: `seed_env + MEASURE_SEED_OFFSET + e`.
Disjoint from the training block and from the harness's periodic-evaluation block
(`train.EVAL_SEED_OFFSET = 10**6`), so the gate never measures on an episode a checkpoint was
selected or logged on."""

MAX_WORKERS = 4
"""Training runs in flight at once: one per core of the reference container."""


@dataclasses.dataclass(frozen=True)
class G1Record:
    """What gate G2 reads from `runs/G1_decision.md`."""

    ap_levels: tuple[float, ...]
    """The three `a * pen` levels, ascending."""

    phase1_ap: float
    """The `a * pen` of the Phase-1 configuration itself (the top level)."""

    dp_share: float
    """The DP's share of REPORT rows in [1.00, 1.02] at the Phase-1 configuration."""

    share_threshold: float
    """The criterion-2 notched threshold on the learned share (AMBIGUITY-011 resolution)."""


@dataclasses.dataclass(frozen=True)
class RunSizing:
    """The batch shape and budget of one class of G2 training run (AMBIGUITY-019)."""

    n_envs: int
    rollout_steps: int
    total_agent_steps: int
    eval_every_updates: int
    eval_episodes: int
    measure_episodes: int


def read_g1(path: Path = G1_DECISION_PATH) -> G1Record:
    """Parse the gate G1 record. Raises `FileNotFoundError` when it is absent - running a G2
    criterion before G1 is out of order - and `ValueError` when a required line is missing."""
    text = Path(path).read_text(encoding="utf-8")
    levels = [
        float(m.group(1))
        for m in re.finditer(r"^\|\s*([0-9.]+)\s*\|\s*[0-9.]+\s*\|\s*\w+\s*\|", text, re.M)
    ]
    share = re.search(r">= 0\.5 x ([0-9.]+) = ([0-9.]+)", text)
    phase1 = re.search(r"`a\*pen` = ([0-9.]+)", text)
    if len(levels) != 3 or share is None or phase1 is None:
        raise ValueError(f"{path}: cannot read the three a*pen levels and the criterion-2 line")
    return G1Record(
        ap_levels=tuple(sorted(levels)),
        phase1_ap=float(phase1.group(1)),
        dp_share=float(share.group(1)),
        share_threshold=float(share.group(2)),
    )


def at_ap_level(cfg: EnvConfig, ap: float) -> EnvConfig:
    """`cfg` at `a * pen = ap`, factorised as at G1: `audit_rate` held, `penalty_scale = ap / a`."""
    incentive = dataclasses.replace(
        cfg.incentive, penalty_scale=float(ap) / float(cfg.information.audit_rate)
    )
    out = dataclasses.replace(cfg, incentive=incentive)
    out.validate()
    return out


def with_overrides(cfg: EnvConfig, overrides: dict[str, dict[str, object]]) -> EnvConfig:
    """`cfg` with section-level field overrides, e.g. `phase1_gate.ARMS[arm]`."""
    sections = {
        name: dataclasses.replace(getattr(cfg, name), **fields)
        for name, fields in overrides.items()
    }
    out = dataclasses.replace(cfg, **sections)
    out.validate()
    return out


def recovery_config(cfg: EnvConfig) -> EnvConfig:
    """PLAN section 5's setting: one enterprise, one sector, no input-output (`a = 0`, `phi = 1`),
    every other parameter as in `cfg`. The same object goes to the DP and to the harness."""
    s = cfg.supply
    supply = dataclasses.replace(
        s,
        n_enterprises=1,
        n_sectors=1,
        sector_of=(0,),
        io_matrix=((0.0,),),
        final_demand_share=(1.0,),
        productivity=(s.productivity[s.sector_of[0]],),
        yield_sigma=(s.yield_sigma[s.sector_of[0]],),
        ces_alpha=(1.0,),
    )
    out = dataclasses.replace(cfg, supply=supply)
    out.validate()
    return out


def seeded(cfg: EnvConfig, seed_index: int, seed_env: int | None = None) -> EnvConfig:
    """`cfg` for training seed `seed_index`: `seed_env = root + seed_index` and
    `seed_policy = cfg.tech.seed_policy + seed_index`. Two arms built from the same root share
    `seed_env` per index, hence environment draws (common random numbers, PLAN sections 2.15, 4.3);
    the seeds are in the configuration hash, so every seed has its own `runs/<hash>/`."""
    root = int(cfg.tech.seed_env) if seed_env is None else int(seed_env)
    tech = dataclasses.replace(
        cfg.tech,
        seed_env=root + int(seed_index),
        seed_policy=int(cfg.tech.seed_policy) + int(seed_index),
    )
    return dataclasses.replace(cfg, tech=tech)


def run_many(jobs: list[tuple[EnvConfig, RunSizing, Path]]) -> list[dict[str, object]]:
    """Train and measure every `(cfg, sizing, run_root)` job, `MAX_WORKERS` at a time, in a
    `spawn` pool with single-threaded XLA per worker. A job whose run directory already holds a
    measurement for the same sizing is read back rather than re-trained, so an interrupted G2 run
    resumes where it stopped. Returns one summary (see `train_and_measure`) per job, in order."""
    os.environ.setdefault(
        "XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    )
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with ProcessPoolExecutor(
        max_workers=max(1, min(MAX_WORKERS, len(jobs))),
        mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
        return list(pool.map(train_and_measure, jobs))


def train_and_measure(job: tuple[EnvConfig, RunSizing, Path]) -> dict[str, object]:
    """Train one run with the WO-018 harness, then measure its final policy.

    The measurement re-loads the final checkpoint and runs `measure_episodes` evaluation episodes
    on the `MEASURE_SEED_OFFSET` block with the harness's deterministic evaluation policy. Returns
    a JSON-able summary: the config hash and seeds, the run directory, the evaluation metrics of
    `train.evaluate`, the learned share of measured REPORT rows in [1.00, 1.02], the per-seed
    bunching estimate and CI, the measured `rho_report` samples, the manifest flags, and the
    criterion-4 training-episode runaway fraction after the first 20% of updates. It is also
    written to `run_dir / "g2_measure.json"`.
    """
    from gosplan.agents.ppo.adapter import IPPO
    from gosplan.agents.ppo.train import TrainConfig, checkpoint_path, evaluate, train
    from gosplan.metrics.phenomena import (
        BUNCHING_EXCESS_HI,
        BUNCHING_EXCESS_LO,
        MEASUREMENT_FIRST_PERIOD,
        phenomenon_bunching,
    )

    cfg, sizing, run_root = job
    run_dir = Path(run_root) / cfg.hash()
    out_path = run_dir / "g2_measure.json"
    sizing_record = dataclasses.asdict(sizing)
    if out_path.exists():
        done = json.loads(out_path.read_text(encoding="utf-8"))
        if done.get("sizing") == sizing_record:
            return done

    n_updates = sizing.total_agent_steps // (sizing.n_envs * sizing.rollout_steps)
    train_cfg = TrainConfig(
        env_cfg=cfg,
        ppo_cfg=_ppo_config(),
        n_envs=sizing.n_envs,
        rollout_steps=sizing.rollout_steps,
        total_agent_steps=sizing.total_agent_steps,
        eval_every_updates=sizing.eval_every_updates,
        eval_episodes=sizing.eval_episodes,
        checkpoint_every_updates=n_updates,
        run_root=Path(run_root),
    )
    train(train_cfg)

    agent = IPPO(cfg, train_cfg.ppo_cfg)
    agent.load_checkpoint(checkpoint_path(run_dir, n_updates - 1))
    metrics, ledger = evaluate(
        agent, cfg, sizing.measure_episodes, int(cfg.tech.seed_env) + MEASURE_SEED_OFFSET
    )
    rho = np.array(
        [
            r.report_ratio
            for r in ledger.records
            if r.phase == "report" and r.t_period >= MEASUREMENT_FIRST_PERIOD
        ]
    )
    share = float(np.mean((rho >= BUNCHING_EXCESS_LO) & (rho <= BUNCHING_EXCESS_HI)))
    bunching = phenomenon_bunching(ledger, cfg)

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (run_dir / "train_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = {
        "config_hash": cfg.hash(),
        "seed_env": int(cfg.tech.seed_env),
        "seed_policy": int(cfg.tech.seed_policy),
        "run_dir": str(run_dir),
        "sizing": sizing_record,
        "n_updates": n_updates,
        "metrics": {k: float(v) for k, v in metrics.items()},
        "share_window": share,
        "bunching": {k: float(v) for k, v in bunching.items()},
        "rho": [float(x) for x in rho],
        "manifest_flags": [f for f in manifest["flags"] if "=" not in f],
        "measure_bound_binding": bool(metrics["frac_at_bound"] > 0.01),
        "runaway_frac_after_20pct": _runaway_after(rows, n_updates, 0.20),
        "wall_clock_s": float(rows[-1]["wall_clock_s"]) if rows else float("nan"),
    }
    out_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")
    return summary


def git_hash() -> str | None:
    """`git rev-parse HEAD` plus a `-dirty` marker, or `None` outside a work tree."""
    import subprocess

    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return head + ("-dirty" if dirty else "")


def _ppo_config():
    from gosplan.agents.ppo.adapter import PPOConfig

    return PPOConfig()


def _runaway_after(rows: list[dict[str, object]], n_updates: int, start_frac: float) -> float:
    """Episode-weighted share of training episodes with some `T > 3 T_0`, over updates at or after
    `start_frac * n_updates` (G2 criterion 4). NaN when no episode finished in that window."""
    first = int(np.ceil(start_frac * n_updates))
    episodes = blown = 0.0
    for row in rows:
        if int(row["update"]) < first:
            continue
        n = float(row["train_episodes_finished"])
        frac = row["train_episode_target_blowup_frac"]  # null (NaN) when no episode finished
        if n > 0 and frac is not None:
            episodes += n
            blown += n * float(frac)
    return blown / episodes if episodes else float("nan")
