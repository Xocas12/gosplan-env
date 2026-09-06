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

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from gosplan.agents.ppo.adapter import IPPO, PPOConfig

if TYPE_CHECKING:  # runtime homes: WO-003 (config), WO-009 (env), WO-011 (ledger); PLAN section 8
    from gosplan.config import EnvConfig
    from gosplan.env.env import GosplanEnv
    from gosplan.metrics.ledger import Ledger


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
    raise NotImplementedError("PLAN section 12.3 WO-018 - implemented in WO-018")


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
    raise NotImplementedError("PLAN section 12.3 WO-018 - implemented in WO-018")


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
    raise NotImplementedError("PLAN section 12.3 WO-018 - implemented in WO-018")


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
    raise NotImplementedError("PLAN section 12.3 WO-018 - implemented in WO-018")


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
    raise NotImplementedError("PLAN section 12.3 WO-018 - implemented in WO-018")


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
    raise NotImplementedError("CONTRACT rule 10 - implemented in WO-018")


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
    raise NotImplementedError("PLAN section 12.3 WO-018 - implemented in WO-018")


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
