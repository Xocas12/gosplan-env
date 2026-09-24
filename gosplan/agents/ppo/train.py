"""Training harness for the `IPPO` adapter (PLAN section 12.3, card WO-018).

Realises: the Phase-1 training path of PLAN sections 6.1 (the `IPPO` agent), 4.3 (common random
numbers), 4.4 (measurement window), 4.5 (the G2 criteria this harness produces runs for), 2.12
(geometric episodes) and 14 (compute budget), under CONTRACT rules 4 (reward terms), 6 (welfare
blindness), 9 (RNG) and 10 (manifest). Owning work order: **WO-018**; it drives
`gosplan/agents/ppo/adapter.py` (WO-017) and writes through `gosplan/metrics/ledger.py` (WO-011).

What the card requires, and therefore what this module contains: a vectorised environment batch,
checkpoints, periodic evaluation through the ledger, manifest writing, common random numbers by
`seed_env`, wall-clock logging, and the entropy anneal 0.01 -> 0.001. Must pass: a smoke test of
100 updates at `N = 1`.

Boundaries this harness does not cross:

  *No reward touching.* The environment returns the reward of CONTRACT rule 4, already multiplied
  by the analytic `reward_scale(cfg)`. The harness stores it, computes GAE from it and hands it to
  the reference PPO. It does not normalise it, clip it, shape it, or wrap the environment in
  anything that rescales it - "no running reward normalisation" is a property of the whole stack,
  not only of the adapter, so any vector-environment wrapper used here is checked for a reward
  normaliser too.

  *No true quantity into the learner.* `StepInfo` is returned by every `GosplanEnv.step` and is
  consumed here **only** to append `StepRecord`s to the ledger and to raise run flags. It never
  enters an observation, an advantage, a loss or a logged learning-curve quantity that feeds back
  into training (CONTRACT rule 6). `welfare_true` and `val_measured` may be plotted after the fact
  from the ledger; they may not be evaluation targets used to select a checkpoint.

  *No held-out phenomenon.* Periodic evaluation computes rows 1 (bunching) and 4 (padding) of PLAN
  section 4.1 only - the Phase-1 rows WO-016 implements. Rows 2, 5, 6 and 7 are held out: no plot,
  table or test of them is produced before the Phase-2 acceptance run (PLAN section 4.1).

  *No silent bound change.* If the ledger raises `BOUND_BINDING` (more than 1% of reports at
  `rho_max`), the flag rides in the manifest and every report of the run displays it. The bound is
  never widened to clear it (CONTRACT rule 8).
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from gosplan.agents.ppo.adapter import IPPO, PPOConfig
from gosplan.env.env import GosplanEnv
from gosplan.env.state import EnterpriseAction, initial_targets
from gosplan.metrics.ledger import Ledger, bound_binding, write_manifest

if TYPE_CHECKING:  # runtime homes: WO-003 (config), WO-009 (env), WO-011 (ledger); PLAN section 8
    from gosplan.config import EnvConfig

EVAL_SEED_OFFSET: int = 10**6
"""LEAD ruling AMBIGUITY-016 point 5: `eval_root = train_root + 10**6`, a block disjoint from the
training seeds."""

SEED_RULE_TRAIN: str = "seed_env(b, e) = seed_env + b + n_envs * e"
"""LEAD ruling AMBIGUITY-016 point 5: training copy `b`, episode `e` (recorded in `flags`)."""

SEED_RULE_EVAL: str = "seed_env(e) = seed_env + 10**6 + e"
"""LEAD ruling AMBIGUITY-016 point 5: evaluation episode `e` (recorded in `flags`)."""

ENTROPY_SCHEDULE: str = "linear in the update index, start -> end over n_updates"
"""The shape of the entropy anneal `entropy_coefficient` implements (recorded in `flags`)."""

EVAL_POLICY: str = "deterministic: squashed Gaussian mean, a = squash(mean)"
"""The evaluation policy of `evaluate` (recorded in `flags`)."""

FINAL_LEDGER_FILE: str = "final_eval_ledger.parquet"
"""File name, under the run directory, of the final evaluation's ledger (AMBIGUITY-016 point 4)."""

HYGIENE_TARGET_MULTIPLE: float = 3.0
"""G2 criterion 4 (PLAN section 12.4): a training episode fails hygiene if any target exceeds this
multiple of `T_0`; the per-update fraction is logged (AMBIGUITY-018 item 8)."""

TARGET_BLOWUP_MULTIPLE: float = 3.0
"""Hygiene criterion 4 of PLAN section 4.5: `T > 3 T_0`."""


@dataclass(frozen=True)
class TrainConfig:
    """Everything one training run needs beyond the environment and PPO configurations (WO-018).

    Frozen and hashable, and written verbatim into the run manifest (CONTRACT rule 10) so a
    learning curve can always be traced to the budget that produced it.

    Sizing fields carry **no defaults**. PLAN fixes the hyper-parameters that are TECH (`gamma`,
    `lambda_GAE`, the learning rate, the clip coefficient, the entropy endpoints - all in
    `PPOConfig`) but not the batch shape or the training budget; those follow the experiment card
    that commissions the run (WO-019's `N = 1` DP comparison, WO-020's `N = 20` gate) within the
    compute envelope of PLAN section 14. Inventing a default here would make an unrecorded
    experimental choice look like a specification (CONTRACT rule 3).
    """

    env_cfg: EnvConfig
    """The environment configuration under training. `env_cfg.hash()` names the run directory."""

    ppo_cfg: PPOConfig
    """Pinned learner hyper-parameters and the reference implementation's identity."""

    n_envs: int
    """Number of environment copies stepped in lockstep by `make_env_batch`. Each copy is a
    separate episode of the same configuration; the batch axis is the outer one, the `N`
    enterprises the inner one."""

    rollout_steps: int
    """Agent-steps collected per environment per update, before one PPO update. An agent-step is
    one PRODUCE or REPORT step, so a period is `cfg.incentive.steps_per_period + 1` of them (PLAN
    section 2.5); choosing a rollout that is not a whole number of periods is legal but means an
    update boundary can fall inside a period, which the card's smoke test does not forbid and the
    manifest records."""

    total_agent_steps: int
    """Training budget in agent-steps, summed over the batch. The number of updates is
    `total_agent_steps // (n_envs * rollout_steps)`, and it is that count the entropy anneal runs
    over."""

    eval_every_updates: int
    """Updates between periodic evaluations (`evaluate`)."""

    eval_episodes: int
    """Episodes per periodic evaluation. They use their own seed block so evaluation never consumes
    the training seed sequence."""

    checkpoint_every_updates: int
    """Updates between checkpoints written to `runs/<hash>/checkpoints/`."""

    run_root: Path = Path("runs")
    """Root of the run tree (PLAN section 8: `runs/` holds manifests and results, one directory per
    run hash). The run directory is `run_root / env_cfg.hash()`."""


def make_env_batch(cfg: EnvConfig, n_envs: int, seed_env: int) -> list[GosplanEnv]:
    """Build the vectorised environment batch for one training run (WO-018).

    Takes: `cfg`; `n_envs`, the number of copies; `seed_env`, the run's root environment seed.
    Returns: a list of `n_envs` constructed `GosplanEnv` objects, each already `reset` onto its own
    episode seed.

    **Common random numbers.** PLAN section 4.3 obtains CRN across arms by *sharing `seed_env`*:
    every draw is keyed `(seed_env, purpose, *indices)` (PLAN section 2.15), so two arms run at the
    same root seed see the same shock realisations wherever their trajectories agree. The harness
    must therefore (a) derive each episode's seed from `seed_env` by a deterministic function of the
    batch index and the episode counter alone - never from wall-clock time, process id, or the
    order in which episodes happen to finish - and (b) use the identical derivation in every arm.
    The derivation is not fixed by PLAN; WO-018 states it once, records it in the manifest, and does
    not vary it between arms. `seed_policy` is a separate stream and is never mixed into these keys
    (CONTRACT rule 9).

    Under geometric termination (PLAN section 2.12) episodes end at different steps, so the batch
    must reset members independently and keep counting - the harness never truncates a live episode
    to keep the batch aligned, because that would put an end-game into an environment designed not
    to have one (finding F4).

    Owning WO: **WO-018**.
    """
    envs = []
    for b in range(n_envs):
        env = GosplanEnv(cfg, records=False)
        env.reset(_train_episode_seed(seed_env, b, 0, n_envs), cfg.tech.seed_policy)
        envs.append(env)
    return envs


def entropy_coefficient(update: int, n_updates: int, ppo_cfg: PPOConfig) -> float:
    """The annealed entropy bonus for one update (WO-018: entropy anneal 0.01 -> 0.001).

    Takes: `update`, the zero-based update index; `n_updates`, the total number of updates in the
    run (`total_agent_steps // (n_envs * rollout_steps)`); `ppo_cfg`, holding the endpoints.
    Returns: the entropy coefficient for this update, `ppo_cfg.entropy_coef_start` at `update = 0`
    and `ppo_cfg.entropy_coef_end` at the last update.

    The card fixes the endpoints (0.01 -> 0.001) and not the shape; the schedule is linear in the
    update index unless WO-018 records otherwise, and whichever is used is written into the manifest
    (CONTRACT rule 10) so two runs can be compared. Edge cases: `n_updates <= 1` returns the start
    value; `update` beyond the last update clamps to the end value rather than extrapolating past
    it.

    Why anneal at all: early exploration must span the notch at `rho = 1` (the report head starts
    there, PLAN section 6.1), while a late policy must be sharp enough for the stationary `rho`
    distribution to be a policy property rather than an exploration artefact - G2 criterion 1
    compares that distribution to the DP's with a Wasserstein-1 tolerance of 0.03.

    Owning WO: **WO-018**.
    """
    start, end = float(ppo_cfg.entropy_coef_start), float(ppo_cfg.entropy_coef_end)
    if n_updates <= 1:
        return start
    frac = min(max(update, 0), n_updates - 1) / (n_updates - 1)
    return start + frac * (end - start)


def evaluate(
    agent: IPPO, cfg: EnvConfig, n_episodes: int, seed_env: int
) -> tuple[dict[str, float], Ledger]:
    """Run a periodic evaluation through the ledger (WO-018).

    Takes: `agent`, the adapter under training; `cfg`; `n_episodes`, evaluation episodes;
    `seed_env`, the root seed for the evaluation block, disjoint from the training block so
    evaluation never perturbs the training shock sequence. Returns: `(metrics, ledger)` - a mapping
    of scalar diagnostics and the `Ledger` holding one `StepRecord` per enterprise per agent-step
    of the evaluation (WO-011).

    Policy: the evaluation uses the deterministic policy - the squashed Gaussian mean rather than a
    sample - so successive evaluations differ only through the environment's own draws. The sampled
    policy is what training optimises; the deterministic one is what the gate criteria of PLAN
    section 4.5 are stated about, and the choice is recorded in the manifest.

    Metrics, restricted to what Phase 1 is allowed to look at:

        mean_return              mean episode return under the reward of CONTRACT rule 4
        mean_effort              mean `e_ik` over the measurement window
        fictitious_padding       `mean_i max(0, R_i - S_i) / T_i`   (PLAN section 4.1 row 4)
        b_hat, b_hat_se, hole    bunching excess mass, bootstrap SE, hole mass
                                 (PLAN section 4.1 row 1, settings of PLAN section 4.5, via
                                  `gosplan.metrics.phenomena`)
        frac_at_bound            fraction of reports at `rho_max` (CONTRACT rule 8, test T-B8)
        frac_target_over_3T0     hygiene criterion 4 of PLAN section 4.5

    Rows 2, 5, 6 and 7 of PLAN section 4.1 - storming, hoarding, blat, hidden reserves - are **held
    out** and are neither computed nor logged here in Phase 1. `welfare_true` and `val_measured`
    reach the ledger (that is their purpose) but never a training signal or a checkpoint-selection
    rule (CONTRACT rule 6).

    Measurement window: periods `t >= 2` of each episode, per PLAN section 4.4; there is no
    end-of-episode exclusion under geometric termination, and reports at `rho_max` are included in
    the histograms and flagged.

    Owning WO: **WO-018** (harness), **WO-016** (the estimators it calls).
    """
    from gosplan.metrics.phenomena import (
        MEASUREMENT_FIRST_PERIOD,
        phenomenon_bunching,
        phenomenon_padding,
    )

    env = GosplanEnv(cfg)
    ledger = Ledger()
    env.attach_ledger(ledger)
    returns = []
    for e in range(n_episodes):
        obs, _opening = env.reset(seed_env + e, cfg.tech.seed_policy)
        agent.reset()
        episode_return = np.zeros(cfg.supply.n_enterprises)
        done = False
        while not done:
            obs, reward, done, _info = env.step(_deterministic_action(agent, obs, env.phase()))
            episode_return += reward
        returns.append(float(episode_return.mean()))

    records = ledger.records
    effort = [
        r.effort for r in records if r.phase == "produce" and r.t_period >= MEASUREMENT_FIRST_PERIOD
    ]
    reports = [r for r in records if r.phase == "report"]
    t_0 = np.asarray(initial_targets(cfg), dtype=float)
    blown = {r.episode for r in records if r.target > TARGET_BLOWUP_MULTIPLE * t_0[r.enterprise]}
    bunching = phenomenon_bunching(ledger, cfg)
    padding = phenomenon_padding(ledger, cfg)
    metrics = {
        "mean_return": float(np.mean(returns)),
        "mean_effort": float(np.mean(effort)) if effort else float("nan"),
        "fictitious_padding": float(padding["padding"]),
        "b_hat": float(bunching["excess_mass"]),
        "b_hat_se": float(bunching["se"]),
        "hole": float(bunching["hole_mass"]),
        "frac_at_bound": sum(1 for r in reports if r.at_bound) / max(len(reports), 1),
        "frac_target_over_3T0": len(blown) / max(n_episodes, 1),
    }
    return metrics, ledger


def checkpoint_path(run_dir: Path, update: int) -> Path:
    """Where the checkpoint for one update goes (WO-018).

    Takes: `run_dir`, the run directory (`TrainConfig.run_root / env_cfg.hash()`); `update`, the
    update index. Returns: the path `run_dir / "checkpoints" / f"update_{update:06d}.ckpt"`,
    creating no directory itself - `train` owns directory creation, so a path helper is safe to call
    from a report script.

    Checkpoints are written by `IPPO.save_checkpoint`, which stores the configuration hash and the
    `PPOConfig` alongside the parameters; a checkpoint whose hash disagrees with the environment it
    is loaded against is an error, not a warning.

    Owning WO: **WO-018**.
    """
    return Path(run_dir) / "checkpoints" / f"update_{update:06d}.ckpt"


def log_update(run_dir: Path, row: dict[str, float]) -> None:
    """Append one update's progress row to the run's training log (WO-018).

    Takes: `run_dir`; `row`, a flat mapping of scalars for this update. Returns: `None`. Appends the
    row as one JSON object per line to `run_dir / "train_log.jsonl"`, flushed each call so a killed
    run keeps everything up to its last update.

    The card requires **wall-clock logging**, so each row carries at least: `update`,
    `agent_steps_total`, `wall_clock_s` (seconds since the run started), `steps_per_second`,
    `entropy_coef` (from `entropy_coefficient`), the PPO diagnostics the reference implementation
    exposes (policy loss, value loss, entropy, approximate KL, clip fraction) and the evaluation
    metrics on update indices where `evaluate` ran. Wall-clock is what PLAN section 14's budget is
    stated in and what tells a lead whether the `N = 20` gate run fits in its envelope.

    Nothing in this log may be a quantity CONTRACT rule 6 keeps from the learner *if* it is used to
    steer training; logging `welfare_true` for post-hoc plots is fine, selecting a checkpoint on it
    is not.

    Owning WO: **WO-018**.
    """
    line = json.dumps({key: _jsonable(value) for key, value in row.items()})
    with open(Path(run_dir) / "train_log.jsonl", "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()


def train_manifest_extra(
    train_cfg: TrainConfig, agent: IPPO, wall_clock_s: float, flags: tuple[str, ...]
) -> dict[str, object]:
    """Assemble the run-specific half of the manifest (CONTRACT rule 10).

    Takes: `train_cfg`; `agent`, for `IPPO.manifest_entry()`; `wall_clock_s`, total run time;
    `flags`, every flag the ledger raised (notably `BOUND_BINDING`). Returns: the `extra` mapping
    handed to `gosplan.metrics.ledger.write_manifest(run_dir, cfg, extra)`.

    It must carry, at minimum: both seeds (`seed_env`, `seed_policy` from `train_cfg.env_cfg.tech`)
    and the episode-seed derivation used for CRN; every `TrainConfig` field; the adapter's
    `manifest_entry()` (reference-PPO name and version, the `PPOConfig`, `reward_normalisation:
    false`); the entropy-anneal schedule; the estimator version used by the periodic evaluation; the
    wall clock; and the flags. `write_manifest` adds the configuration itself, its hash,
    `SPEC_VERSION` and the git hash. Fields that do not apply to a training run - the LLM model ids,
    the MIP solver version and its optimality gap - are written as `null`, never omitted, so a
    missing field is always a bug and never an ambiguity.

    Owning WO: **WO-018** (this assembly), **WO-011** (`write_manifest`).
    """
    from gosplan.metrics import phenomena, resolve_estimators

    backend = resolve_estimators()
    entry = agent.manifest_entry()
    ppo_cfg = train_cfg.ppo_cfg
    tech = train_cfg.env_cfg.tech
    train_record = {
        field.name: getattr(train_cfg, field.name)
        for field in dataclasses.fields(train_cfg)
        if field.name not in ("env_cfg", "ppo_cfg")
    }
    train_record["run_root"] = str(train_cfg.run_root)
    train_record["n_updates"] = _n_updates(train_cfg)
    train_record["agent_steps_per_period"] = train_cfg.env_cfg.incentive.steps_per_period + 1
    records = {
        "seed_env": tech.seed_env,
        "seed_policy": tech.seed_policy,
        "seed_rule_train": SEED_RULE_TRAIN,
        "seed_rule_eval": SEED_RULE_EVAL,
        "seed_policy_streams": "SeedSequence(seed_policy): init in IPPO; spawn key 1 for rollouts",
        "entropy_schedule": {
            "shape": ENTROPY_SCHEDULE,
            "start": ppo_cfg.entropy_coef_start,
            "end": ppo_cfg.entropy_coef_end,
        },
        "eval_policy": EVAL_POLICY,
        "train_envs_records": False,
        "vector_env_wrappers": "none (list of GosplanEnv stepped in lockstep)",
        "train_config": train_record,
        "ppo_manifest": entry,
        "wall_clock_s": float(wall_clock_s),
    }
    record_flags = [
        f"{key}={json.dumps(_jsonable(value), sort_keys=True)}" for key, value in records.items()
    ]
    return {
        "git_hash": _git_hash(),
        "reference_ppo_version": entry["reference_ppo_version"],
        "estimator_version": backend.version,
        "estimator_backend": backend.name,
        "llm_models": None,
        "solver": None,
        "solver_version": None,
        "solver_optimality_gap": None,
        "bunching_settings": {
            "bin_width": phenomena.BUNCHING_BIN_WIDTH,
            "window_lo": phenomena.BUNCHING_WINDOW_LO,
            "window_hi": phenomena.BUNCHING_WINDOW_HI,
            "excl_lo": phenomena.BUNCHING_EXCL_LO,
            "excl_hi": phenomena.BUNCHING_EXCL_HI,
            "degree": phenomena.BUNCHING_POLY_DEGREE,
            "excess_lo": phenomena.BUNCHING_EXCESS_LO,
            "excess_hi": phenomena.BUNCHING_EXCESS_HI,
            "hole_lo": phenomena.BUNCHING_HOLE_LO,
            "hole_hi": phenomena.BUNCHING_HOLE_HI,
            "measurement_first_period": phenomena.MEASUREMENT_FIRST_PERIOD,
            "exclude_episode_end": phenomena.MEASUREMENT_EXCLUDE_EPISODE_END,
            "include_at_bound": phenomena.MEASUREMENT_INCLUDE_AT_BOUND,
        },
        "flags": sorted(set(flags)) + record_flags,
    }


def train(train_cfg: TrainConfig) -> Path:
    """Run one training job end to end (WO-018).

    Takes: `train_cfg`. Returns: the run directory `train_cfg.run_root / env_cfg.hash()`, which by
    then holds `manifest.json`, `train_log.jsonl`, the `checkpoints/` tree and the parquet ledger of
    the final evaluation.

    The loop, in order:

      1. Validate `train_cfg.env_cfg`, create the run directory, build `IPPO(env_cfg, ppo_cfg)` and
         `make_env_batch(env_cfg, n_envs, env_cfg.tech.seed_env)`, and record the start time.
      2. For each of `n_updates = total_agent_steps // (n_envs * rollout_steps)` updates: collect
         `rollout_steps` agent-steps per environment through `agent.act`, storing observations,
         actions, rewards, dones and the `StepInfo` payloads; append `StepRecord`s to the training
         ledger; compute GAE with `ppo_cfg.gamma` and `ppo_cfg.lambda_gae`; run the reference PPO's
         update with `entropy_coefficient(update, n_updates, ppo_cfg)` and, if
         `ppo_cfg.normalise_advantages`, per-batch advantage normalisation - and with **no** reward
         normalisation anywhere in the stack (CONTRACT rule 4).
      3. Every `eval_every_updates` updates call `evaluate` and pass its metrics to `log_update`;
         every `checkpoint_every_updates` updates call `agent.save_checkpoint` at
         `checkpoint_path(run_dir, update)`.
      4. On exit write the final evaluation ledger to parquet and call
         `gosplan.metrics.ledger.write_manifest(run_dir, env_cfg, train_manifest_extra(...))`.

    Termination is the environment's: geometric with continuation `cfg.incentive.tenure`, hard cap
    `cfg.tech.max_periods` (PLAN section 2.12). The harness never imposes its own episode cut, never
    reveals periods remaining, and never bootstraps a truncated episode as if it had terminated -
    the distinction matters because the agent must not be able to infer the horizon (test T-B9).

    Binds: the WO-018 smoke test - 100 updates at `N = 1` complete, write a manifest carrying every
    CONTRACT rule 10 field, and leave a loadable checkpoint. Owning WO: **WO-018**.
    """
    cfg = train_cfg.env_cfg
    ppo_cfg = train_cfg.ppo_cfg
    cfg.validate()
    run_dir = Path(train_cfg.run_root) / cfg.hash()
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "train_log.jsonl").write_text("", encoding="utf-8")  # one row per update, fresh
    agent = IPPO(cfg, ppo_cfg)
    n_envs, horizon = train_cfg.n_envs, train_cfg.rollout_steps
    n = cfg.supply.n_enterprises
    rows = n_envs * n
    n_updates = _n_updates(train_cfg)
    root = int(cfg.tech.seed_env)
    rng = np.random.default_rng(np.random.SeedSequence(int(cfg.tech.seed_policy)).spawn(2)[1])
    start = time.perf_counter()
    envs = make_env_batch(cfg, n_envs, root)
    episode_index = [0] * n_envs
    # The opening observations: `make_env_batch` returns environments already reset onto episode
    # 0; resetting again onto the identical seed reproduces that state and returns its observation.
    obs = np.concatenate(
        [
            env.reset(_train_episode_seed(root, b, 0, n_envs), cfg.tech.seed_policy)[0]
            for b, env in enumerate(envs)
        ]
    ).astype(np.float32)
    running_return = np.zeros((n_envs, n))
    # G2 criterion 4 is stated over TRAINING episodes (AMBIGUITY-018 item 8): per episode, whether
    # any enterprise's target exceeded HYGIENE_TARGET_MULTIPLE x T_0, read from the env state.
    hygiene_bound = HYGIENE_TARGET_MULTIPLE * np.asarray(initial_targets(cfg), dtype=float)
    episode_blowup = np.zeros(n_envs, dtype=bool)
    run_flags: set[str] = set()
    masks = {phase: agent.element_mask(phase) for phase in ("produce", "report")}
    d, a_dim = agent.obs_dim, agent.action_dim

    buf_obs = np.zeros((horizon, rows, d), dtype=np.float32)
    buf_z = np.zeros((horizon, rows, a_dim), dtype=np.float32)
    buf_mask = np.zeros((horizon, rows, a_dim), dtype=np.float32)
    buf_logp = np.zeros((horizon, rows), dtype=np.float32)
    buf_value = np.zeros((horizon, rows), dtype=np.float32)
    buf_reward = np.zeros((horizon, rows), dtype=np.float32)
    buf_done = np.zeros((horizon, rows), dtype=np.float32)

    agent_steps = 0
    final_ledger: Ledger | None = None
    final_eval_update = -1
    for update in range(n_updates):
        t_update = time.perf_counter()
        ent_coef = entropy_coefficient(update, n_updates, ppo_cfg)
        finished_returns: list[float] = []
        finished_blowups: list[bool] = []
        for t in range(horizon):
            mask = np.repeat(np.stack([masks[env.phase()] for env in envs]), n, axis=0)
            eps = rng.standard_normal((rows, a_dim)).astype(np.float32)
            z, action, logp, value = agent.policy.sample(obs, eps, mask)
            action = np.asarray(action, dtype=float).reshape(n_envs, n, a_dim)
            buf_obs[t], buf_mask[t] = obs, mask
            buf_z[t], buf_logp[t], buf_value[t] = z, logp, value
            next_obs = []
            for b, env in enumerate(envs):
                o, reward, done, info = env.step(agent._to_action(action[b]))
                run_flags.update(info.flags)  # StepInfo: run flags only (CONTRACT rule 6)
                buf_reward[t, b * n : (b + 1) * n] = reward
                running_return[b] += reward
                episode_blowup[b] |= bool(np.any(env.state.target > hygiene_bound))
                if done:
                    buf_done[t, b * n : (b + 1) * n] = 1.0
                    finished_returns.append(float(running_return[b].mean()))
                    running_return[b] = 0.0
                    finished_blowups.append(bool(episode_blowup[b]))
                    episode_blowup[b] = False
                    episode_index[b] += 1
                    seed = _train_episode_seed(root, b, episode_index[b], n_envs)
                    o, _opening = env.reset(seed, cfg.tech.seed_policy)
                else:
                    buf_done[t, b * n : (b + 1) * n] = 0.0
                next_obs.append(o)
            obs = np.concatenate(next_obs).astype(np.float32)
        agent_steps += n_envs * horizon

        _mean, _log_std, next_value = agent.policy.forward(obs)
        advantages, returns = _gae(
            buf_reward,
            buf_value,
            buf_done,
            np.asarray(next_value),
            ppo_cfg.gamma,
            ppo_cfg.lambda_gae,
        )
        batch = {
            "obs": buf_obs.reshape(-1, d),
            "z": buf_z.reshape(-1, a_dim),
            "mask": buf_mask.reshape(-1, a_dim),
            "logp": buf_logp.reshape(-1),
            "advantages": advantages.reshape(-1),
            "returns": returns.reshape(-1),
            "values": buf_value.reshape(-1),
        }
        diagnostics = agent.policy.update(batch, ent_coef, rng)

        now = time.perf_counter()
        row: dict[str, float] = {
            "update": update,
            "agent_steps_total": agent_steps,
            "enterprise_steps_total": agent_steps * n,
            "wall_clock_s": now - start,
            "steps_per_second": agent_steps / (now - start),
            "update_steps_per_second": n_envs * horizon / (now - t_update),
            "entropy_coef": ent_coef,
            "train_episodes_finished": len(finished_returns),
            "train_episode_return_mean": (
                float(np.mean(finished_returns)) if finished_returns else float("nan")
            ),
            "train_episode_target_blowup_frac": (
                float(np.mean(finished_blowups)) if finished_blowups else float("nan")
            ),
            **diagnostics,
        }
        if (update + 1) % train_cfg.eval_every_updates == 0:
            metrics, final_ledger = evaluate(
                agent, cfg, train_cfg.eval_episodes, root + EVAL_SEED_OFFSET
            )
            final_eval_update = update
            row.update({f"eval_{key}": value for key, value in metrics.items()})
        if (update + 1) % train_cfg.checkpoint_every_updates == 0:
            agent.save_checkpoint(checkpoint_path(run_dir, update))
        log_update(run_dir, row)

    if final_eval_update != n_updates - 1 or final_ledger is None:
        _metrics, final_ledger = evaluate(
            agent, cfg, train_cfg.eval_episodes, root + EVAL_SEED_OFFSET
        )
    final_ledger.to_parquet(str(run_dir / FINAL_LEDGER_FILE))
    flags = set(run_flags) | set(final_ledger.flags)
    if bound_binding(final_ledger):
        flags.add("BOUND_BINDING")
    wall_clock_s = time.perf_counter() - start
    extra = train_manifest_extra(train_cfg, agent, wall_clock_s, tuple(sorted(flags)))
    write_manifest(str(run_dir), cfg, extra)
    return run_dir


# ---------- private helpers ----------


def _train_episode_seed(root: int, b: int, e: int, n_envs: int) -> int:
    """AMBIGUITY-016 point 5: training copy `b`, episode `e` -> `root + b + n_envs * e`."""
    return int(root) + int(b) + int(n_envs) * int(e)


def _n_updates(train_cfg: TrainConfig) -> int:
    return train_cfg.total_agent_steps // (train_cfg.n_envs * train_cfg.rollout_steps)


def _deterministic_action(agent: IPPO, obs: np.ndarray, phase: str) -> EnterpriseAction:
    """The evaluation policy: `squash(mean)` per head through the adapter's public surface, the
    dimensions the phase does not read set to zero (as `IPPO.act` does)."""
    mean, _log_std, _value = agent.forward(obs)
    mask = agent.element_mask(phase)
    squashed = np.zeros_like(np.asarray(mean, dtype=float))
    for name in agent.head_names:
        cols = agent._head_slices[name]
        squashed[:, cols] = agent.squash(mean[:, cols], name)
    return agent._to_action(squashed * mask)


def _gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    next_value: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """CleanRL's GAE. `dones[t]` is the geometric termination flag returned by step `t`: a
    terminal step does not bootstrap. The rollout boundary bootstraps from `next_value` (the
    episode continues into the next rollout; it is not cut)."""
    horizon = rewards.shape[0]
    advantages = np.zeros_like(rewards)
    last = np.zeros(rewards.shape[1], dtype=rewards.dtype)
    for t in reversed(range(horizon)):
        following = next_value if t == horizon - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * following * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        advantages[t] = last
    return advantages, advantages + values


def _jsonable(value: object) -> object:
    """Plain JSON values: numpy scalars to Python, non-finite floats to null, recursively."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None  # strict JSON has no NaN/inf; a missing diagnostic is written as null
    return value


def _git_hash() -> str | None:
    """`git rev-parse HEAD`, `-dirty` when the tree has uncommitted changes; `None` without git."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return head + ("-dirty" if dirty else "")


__all__ = [
    "TrainConfig",
    "checkpoint_path",
    "entropy_coefficient",
    "evaluate",
    "log_update",
    "make_env_batch",
    "train",
    "train_manifest_extra",
]
