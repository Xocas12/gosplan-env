"""Thin adapter over a pinned reference PPO: the `IPPO` agent of PLAN section 6.1.

Realises: PLAN section 6.1 (the `IPPO` row), PLAN section 2.3 (action bounds), PLAN section 2.4
(observation), PLAN section 2.5 (phases) and PLAN section 3 (the TECH hyper-parameters that live
with the adapter rather than in `EnvConfig`, because the environment never reads them). Owning work
order: **WO-017** (LEAD).

**gosplan does not implement PPO.** PLAN section 6.1 requires a *thin adapter around a reference
implementation* - a CleanRL-style continuous PPO for the NumPy path, a PureJaxRL/JaxMARL-style loop
for the JAX path - whose exact package and version the lead verifies when the work order is issued
and which the run manifest records (CONTRACT rule 10, `PPOConfig.reference_impl` and
`reference_version`). Everything in this module is glue: observation in, bounded action out,
hyper-parameters pinned, and the two rules below enforced at the boundary.

What the adapter must guarantee, and what the frozen tests check:

  *Heads only for the active dimensions.* One Gaussian head per name in `active_action_dims(cfg)`
  - `["effort", "report_ratio", "input_request"]` at `p1_default_config()`. Inactive dimensions get
  no parameters at all, which is why `active_action_dims` must be a pure function of the
  configuration and must not change within a run.

  *Tanh-squashed Gaussian.* Each head emits `(mean, log_std)` in pre-squash space; the action is
  `tanh`-squashed and affinely mapped onto that dimension's box from `action_spec(cfg)`. Bounds are
  therefore respected by construction, never by clipping after the fact, and the log-probability
  carries the usual `log(1 - tanh(z)**2)` Jacobian correction. Test `test_ppo_adapter.py`: sampled
  actions lie inside the bounds.

  *Report head initialised at `rho = 1`, std 0.05* (PLAN section 6.1). The initialisation is
  specified in *ratio* units, so it must be pushed through the squash: with box `[lo, hi]` and
  `a(z) = lo + (hi - lo) * (tanh(z) + 1) / 2`, the bias solves `a(z0) = 1`, i.e.
  `z0 = atanh(2 * (1 - lo) / (hi - lo) - 1)`, and the initial log-std is set so that the induced
  standard deviation in ratio units, `|da/dz|(z0) * sigma_z = (hi - lo) / 2 * (1 - tanh(z0)**2) *
  sigma_z`, equals `PPOConfig.report_head_init_std`. Why it matters: a policy initialised at
  `rho = 1` starts *on* the notch of PLAN section 2.8, so the discontinuity is inside the initial
  exploration support and bunching is neither seeded nor made unreachable.

  *Sector one-hot handled by `param_sharing`.* At `cfg.tech.param_sharing == "shared"` (Phase 1)
  there is one policy for all `N` enterprises and sector identity enters only through the one-hot
  block of the observation (`obs_spec` indices `12 + J : 12 + 2J`). `"per_sector"` builds one
  parameter set per sector, `"independent"` one per enterprise. The adapter chooses the structure;
  it never edits the observation.

  *Fixed analytic reward scale, and no reward normalisation.* `scale = reward_scale(cfg)` is
  computed once from the configuration (`1 / B_cfg(1.1)`, PLAN section 2.9.1) and the environment
  has already applied it to the reward the adapter receives. CONTRACT rule 4 forbids running reward
  normalisation outright: running statistics change the effective reward over training and, with
  heavy-tailed penalties, shrink the notch in normalised units - which would erase exactly the
  feature the study measures. Concretely, the wrapped object must carry **no** `RunningMeanStd` on
  rewards or returns, no `NormalizeReward` wrapper, and no return-scaling option left at its
  library default; reference implementations ship these on by default, so switching them off is a
  positive step the adapter takes and `tests/unit/test_ppo_adapter.py` verifies by inspecting the
  wrapped object. Observation normalisation is a separate question and is not what rule 4 forbids;
  whatever is chosen is recorded in the manifest.

  *Per-batch advantage normalisation is permitted* (CONTRACT rule 4, explicitly) and is on by
  default (`PPOConfig.normalise_advantages`).

  *`gamma = 0.99`, `lambda_GAE = 0.97`* (PLAN section 6.1). `gamma` is the technical discount and is
  distinct from the economic continuation probability `psi = cfg.incentive.tenure` (PLAN section
  2.12): the environment terminates geometrically at `psi`, the learner discounts at `gamma`, and
  the single-enterprise DP uses `psi * gamma` (PLAN section 5, `gosplan.agents.dp.DP_DISCOUNT`).
  The two must never be conflated - G2 criterion 1 compares the DP and PPO at the same effective
  discount.

  *Phase-aware masking.* At a PRODUCE step only `effort` (Phase 2: `quality`, `invest`) is read; at
  the REPORT step only `report_ratio` and `input_request` (Phase 2: `trade_offer`). Inactive heads
  are masked per step so their log-probabilities and entropies do not enter that step's loss. The
  environment ignores irrelevant dimensions anyway (PLAN section 2.3), so masking is about credit
  assignment, not legality.

  *`forward` takes `obs` only.* CONTRACT rule 6: no `State`, no `StepInfo`, no `PlannerView`, no
  `welfare_true`, no `val_measured`, no other enterprise's true quantities. Test T-B5 in
  `tests/behavioural/test_welfare_blindness.py` inspects the signature of `IPPO.forward`, so the
  argument list is part of the interface.

Binds: `tests/unit/test_ppo_adapter.py` (forward takes `obs` only - T-B5; no `RunningMeanStd` on
rewards, by inspection of the wrapped object; sampled actions within bounds) and, downstream, the
G2 criteria of PLAN section 4.5 via WO-019 and WO-020.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from gosplan.agents.base import Array, Phase

if TYPE_CHECKING:  # runtime homes: WO-003 (config), WO-009 (state); PLAN section 8
    from pathlib import Path

    import numpy as np

    from gosplan.config import EnvConfig
    from gosplan.env.state import EnterpriseAction


@dataclass(frozen=True)
class PPOConfig:
    """Pinned PPO hyper-parameters and the identity of the reference implementation (PLAN sec. 6.1).

    These are TECH-arm parameters (PLAN section 3): fixed across every treatment arm, never swept in
    a contrast. They live here rather than in `EnvConfig` because the environment never reads them
    (the spec's `TechConfig` docstring records this decision), and every field is written into the
    run manifest (CONTRACT rule 10) so a result can always be traced to the learner that produced
    it.

    Frozen and hashable, like the environment configs. Owning WO: **WO-017**.
    """

    reference_impl: str
    """Import path or package name of the reference PPO this adapter wraps, e.g. a CleanRL-style
    continuous PPO on the NumPy path or a PureJaxRL/JaxMARL-style loop on the JAX path (PLAN section
    6.1). **No default**: PLAN names families, not a package, and the lead verifies the choice when
    WO-017 is issued. Recorded in the manifest (CONTRACT rule 10)."""

    reference_version: str
    """Exact pinned version (release tag or commit) of `reference_impl`. **No default** for the same
    reason. "reference-PPO version" is a named field of the CONTRACT rule 10 manifest, so a run
    whose learner version is unknown is not a reportable run."""

    gamma: float = 0.99
    """Technical discount factor (PLAN section 6.1). Distinct from `cfg.incentive.tenure`, the
    economic continuation probability of PLAN section 2.12; the DP's continuation factor is the
    product `psi * gamma` (PLAN section 5)."""

    lambda_gae: float = 0.97
    """GAE trace-decay `lambda_GAE` (PLAN section 6.1)."""

    learning_rate: float = 3e-4
    """Adam step size (PLAN section 3, TECH row quoted in the spec's `TechConfig` docstring)."""

    clip_coef: float = 0.2
    """PPO surrogate clipping coefficient (same source)."""

    entropy_coef_start: float = 0.01
    """Entropy bonus at the start of training; annealed to `entropy_coef_end` over the run (PLAN
    section 12.3, WO-018: "entropy anneal 0.01 -> 0.001"). The schedule itself lives in
    `gosplan/agents/ppo/train.py`."""

    entropy_coef_end: float = 0.001
    """Entropy bonus at the end of training (same source)."""

    report_head_init_ratio: float = 1.0
    """Report-head initialisation in *ratio* units: the initial squashed mean of `report_ratio`
    (PLAN section 6.1, "report head initialised with mean at `rho = 1`"). Pushed through the tanh
    squash as shown in the module docstring. Starting on the notch is deliberate."""

    report_head_init_std: float = 0.05
    """Initial standard deviation of `report_ratio` in ratio units (PLAN section 6.1, "std 0.05"),
    converted to a pre-squash log-std through the local Jacobian of the squash."""

    normalise_advantages: bool = True
    """Per-batch advantage normalisation, explicitly permitted by CONTRACT rule 4. There is
    deliberately **no** corresponding reward-normalisation field: rule 4 forbids running reward
    normalisation outright, so it is not a configurable choice and cannot be switched on from a
    config file."""


class IPPO:
    """Independent-PPO agent: the `Agent`-protocol face of a pinned reference PPO (PLAN sec. 6.1).

    One shared policy acts for all `N` enterprises at `cfg.tech.param_sharing == "shared"` (Phase
    1), with sector identity supplied by the one-hot block of the observation; the enterprises are
    independent learners sharing parameters, which is what "IPPO" names. It implements the `Agent`
    protocol of `gosplan/agents/base.py` structurally: `act(obs, phase, rng)` and `reset()`.

    The class holds the wrapped reference policy, the configuration and the analytic reward scale,
    and it owns exactly four responsibilities: build heads for `active_action_dims(cfg)`, initialise
    the report head at `rho = 1` with std 0.05, squash and map samples onto the `action_spec(cfg)`
    boxes, and mask heads by phase. Optimisation belongs to the reference implementation, driven by
    `gosplan/agents/ppo/train.py` (WO-018).

    Everything the module docstring states about CONTRACT rules 4 and 6 is a property of this class
    and is checked by `tests/unit/test_ppo_adapter.py`. Owning WO: **WO-017** (LEAD).
    """

    cfg: EnvConfig
    """The environment configuration. Read for `active_action_dims`, `action_spec`, `obs_spec`,
    `cfg.tech.param_sharing`, `cfg.supply.n_enterprises` and `cfg.supply.n_sectors`."""

    ppo: PPOConfig
    """The pinned hyper-parameters and reference identity."""

    policy: object
    """The wrapped reference-PPO policy object. Typed `object` on purpose: the reference is pinned
    at WO-017 issue time and its type must not leak into this interface, so the JAX path (WO-029)
    can substitute its own without a signature change."""

    scale: float
    """`reward_scale(cfg)` (PLAN section 2.9.1), stored for logging and for the manifest. The
    environment has already applied it to every reward the adapter sees; the adapter never rescales
    a reward again (CONTRACT rule 4)."""

    head_names: tuple[str, ...]
    """`tuple(active_action_dims(cfg))`, fixed for the life of the run: the dimensions that have
    Gaussian heads, in the order of PLAN section 2.3."""

    def __init__(self, cfg: EnvConfig, ppo: PPOConfig) -> None:
        """Build the wrapped policy for one configuration.

        Takes: `cfg`, an already validated `EnvConfig`; `ppo`, the pinned `PPOConfig`. Returns:
        nothing. Constructs the reference PPO with an observation dimension of `len(obs_spec(cfg))`
        (`12 + 3J` in Phase 1) and one Gaussian head per name in `active_action_dims(cfg)`, sized
        by that dimension's shape in `action_spec(cfg)` (`(N,)`, or `(N, J)` for `input_request`);
        applies the parameter-sharing structure of `cfg.tech.param_sharing`; initialises the report
        head's bias and log-std so the initial squashed policy has mean `ppo.report_head_init_ratio`
        and standard deviation `ppo.report_head_init_std` in ratio units; stores
        `scale = reward_scale(cfg)`.

        It must also switch off whatever reward or return normalisation the reference implementation
        enables by default (CONTRACT rule 4) and leave the object in a state where a test can see
        that it did.

        Owning WO: **WO-017**.
        """
        raise NotImplementedError("PLAN section 6.1 - implemented in WO-017")

    def forward(self, obs: Array) -> tuple[Array, Array, Array]:
        """Policy and value forward pass - **`obs` and nothing else** (CONTRACT rule 6, T-B5).

        Takes: `obs` `(N, d)`, `d = len(obs_spec(cfg))`. Returns: `(mean, log_std, value)` - the
        pre-squash Gaussian mean and log standard deviation for every active head, concatenated in
        `head_names` order over the trailing axis, each `(N, sum_of_head_sizes)`, and the state
        value estimate `(N,)`.

        The argument list is part of the interface: test T-B5 in
        `tests/behavioural/test_welfare_blindness.py` inspects this signature and fails if anything
        else appears. No `State`, no `StepInfo`, no `PlannerView`, no `welfare_true`, no
        `val_measured`, no other enterprise's `y`, `S` or `X`, no periods remaining. `phase` is not
        an argument either - it reaches the policy through observation field 0 (`phase`) and field 1
        (`k_over_M`), and it is `act` that masks heads, not `forward`.

        Owning WO: **WO-017**.
        """
        raise NotImplementedError("PLAN section 6.1 - implemented in WO-017")

    def act(self, obs: Array, phase: Phase, rng: np.random.Generator) -> EnterpriseAction:
        """Sample one joint action from the current policy.

        Takes: `obs` `(N, d)`; `phase`, which selects the heads this step reads (`phase_mask`);
        `rng`, a generator on the `seed_policy` stream (CONTRACT rule 9) - the adapter must route
        the reference implementation's sampling through it rather than through a framework-global
        seed, so that policy randomness stays separate from `seed_env` and common random numbers
        across arms are unaffected by the policy. Returns: an `EnterpriseAction` whose active,
        phase-relevant dimensions are `tanh`-squashed samples mapped onto their `action_spec(cfg)`
        boxes and whose other dimensions are zero.

        Evaluation runs (WO-018, WO-019, WO-020) may want the deterministic policy - the squashed
        mean rather than a sample. That is a property of the caller's evaluation loop, not a second
        signature here; `train.py` documents how it obtains it.

        Owning WO: **WO-017**.
        """
        raise NotImplementedError("PLAN section 6.1 - implemented in WO-017")

    def reset(self) -> None:
        """Clear per-episode policy state.

        Takes: nothing. Returns: `None`. A feed-forward reference policy has none and this is a
        no-op; a recurrent one clears its hidden state here. It never touches parameters, the
        optimiser or the configuration.

        Owning WO: **WO-017**.
        """
        raise NotImplementedError("PLAN section 6.1 - implemented in WO-017")

    def evaluate_actions(self, obs: Array, action: EnterpriseAction) -> tuple[Array, Array, Array]:
        """Log-probabilities, entropies and values for a stored batch - the update's seam.

        Takes: `obs` `(B, N, d)` and `action`, the `EnterpriseAction` recorded when that batch was
        collected. Returns: `(log_prob, entropy, value)`, each summed or averaged over the active,
        phase-relevant heads per agent-step: `log_prob` `(B, N)`, `entropy` `(B, N)`, `value`
        `(B, N)`.

        Two requirements. The log-probability must include the tanh Jacobian correction
        `-sum log(1 - tanh(z)**2)` and the affine box mapping's constant, or the ratio in the PPO
        surrogate is wrong. And heads masked out by `phase_mask` for a step contribute neither
        log-probability nor entropy at that step, so the entropy bonus of the anneal cannot be
        collected from a head the environment did not read.

        This is the only place the adapter looks at actions as well as observations; it is a
        *training* call on stored data, not an agent input, so CONTRACT rule 6 is untouched - what
        rule 6 constrains is `forward`, and the actions here are the agent's own.

        Owning WO: **WO-017**.
        """
        raise NotImplementedError("PLAN section 6.1 - implemented in WO-017")

    def phase_mask(self, phase: Phase) -> dict[str, bool]:
        """Which heads the environment actually reads at this phase (PLAN sections 2.3, 2.5).

        Takes: `phase`. Returns: a mapping from every name in `head_names` to whether that
        dimension is read at this phase:

            "produce":  effort, quality, invest              -> True;  the rest -> False
            "report":   report_ratio, input_request, trade_offer -> True;  the rest -> False

        Used to mask log-probabilities and entropies per step (see `evaluate_actions`) and to skip
        pointless sampling in `act`. The environment ignores irrelevant dimensions on its own (PLAN
        section 2.3), so a masking bug costs credit-assignment quality, never legality.

        Owning WO: **WO-017**.
        """
        raise NotImplementedError("PLAN sections 2.3, 2.5 - implemented in WO-017")

    def squash(self, raw: Array, name: str) -> Array:
        """Map a pre-squash Gaussian sample onto one action dimension's box (PLAN section 2.3).

        Takes: `raw`, the pre-squash sample for head `name`, shaped as that head's slice; `name`,
        one of `head_names`. Returns: the action values for that dimension, inside its box
        `(lo, hi)` from `action_spec(cfg)`:

            a = lo + (hi - lo) * (tanh(raw) + 1) / 2

        so `effort`, `quality` and `invest` land in [0, 1], `report_ratio` in
        [0, `cfg.tech.report_max_ratio`], `input_request` in
        [0, `cfg.tech.request_max_multiple`] (a multiple of `need_ij`, which the environment
        rescales), and `trade_offer` in [-1, 1]. Bounds hold by construction, so no clipping is
        applied afterwards and none is needed.

        CONTRACT rule 8 applies at the far end of this map: the fraction of reports sitting at
        `rho_max` is logged and flags the run `BOUND_BINDING` above 1%. A saturating tanh is exactly
        how a policy gets there, so the flag is a real diagnostic of the learner and the bound is
        never widened to clear it.

        Owning WO: **WO-017**.
        """
        raise NotImplementedError("PLAN section 2.3 - implemented in WO-017")

    def manifest_entry(self) -> dict[str, object]:
        """The learner's contribution to the run manifest (CONTRACT rule 10).

        Takes: nothing. Returns: a JSON-serialisable mapping carrying at least
        `reference_impl` and `reference_version` (the "reference-PPO version" field of rule 10),
        every `PPOConfig` field, the resolved `head_names`, `cfg.tech.param_sharing`, the stored
        `scale`, and an explicit `reward_normalisation: false` entry recording that rule 4 was
        honoured. Fields that do not apply are written as `null`, never omitted, so a missing field
        is always a bug and never an ambiguity.

        The mapping is passed through `train.py` into `gosplan.metrics.ledger.write_manifest`.
        Owning WO: **WO-017**.
        """
        raise NotImplementedError("CONTRACT rule 10 - implemented in WO-017")

    def save_checkpoint(self, path: Path) -> None:
        """Write policy parameters and optimiser state to disk.

        Takes: `path`, the destination file (created by `train.py` under `runs/<hash>/`). Returns:
        `None`. Writes whatever the reference implementation needs to resume exactly, plus the
        `PPOConfig` and the configuration hash, so a checkpoint can never be reloaded against a
        different environment configuration without the mismatch being detectable.

        Owning WO: **WO-017**.
        """
        raise NotImplementedError("PLAN section 12.3 WO-018 - implemented in WO-017")

    def load_checkpoint(self, path: Path) -> None:
        """Restore policy parameters and optimiser state from disk.

        Takes: `path`. Returns: `None`. Raises `ValueError` when the checkpoint's configuration hash
        or `PPOConfig` disagrees with this adapter's - silently loading mismatched parameters would
        attribute one run's behaviour to another run's configuration.

        Owning WO: **WO-017**.
        """
        raise NotImplementedError("PLAN section 12.3 WO-018 - implemented in WO-017")


__all__ = ["IPPO", "PPOConfig"]
