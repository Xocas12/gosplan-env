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

THE PINNED REFERENCE (LEAD ruling AMBIGUITY-016 point 1). The reference is CleanRL's
`ppo_continuous_action` ALGORITHM (clipped surrogate, clipped value loss, GAE, per-minibatch
advantage normalisation, Adam with eps 1e-5, entropy bonus, global-norm gradient clip 0.5, 4 epochs
x 4 minibatches, separate 64-64 tanh MLP actor and critic, orthogonal init with gains sqrt(2) /
0.01 / 1.0), re-expressed in JAX in this module because no maintained package ships it as a
library: `reference_impl = REFERENCE_IMPL`, `reference_version = REFERENCE_VERSION`. The algorithm
lives in `_CleanRLContinuousPPO` below; GAE and the rollout live in `train.py` (WO-018), as in
CleanRL's own script. Deviations from CleanRL demanded by the card:

  * tanh-squashed heads mapped onto the `action_spec` boxes, with the log-density corrected by the
    Jacobian `log((hi - lo) / 2) + log(1 - tanh(z)**2)`, the latter in the stable form
    `2 * (log 2 - z - softplus(-2 z))`; CleanRL samples an unbounded Gaussian and clips;
  * state-independent log-std per action element (CleanRL's `actor_logstd`), initialised at 0
    except the report head, which gets the `z0` bias and log-std of the card formula above;
  * heads only for `active_action_dims(cfg)`;
  * phase masking of log-probabilities and entropies per agent-step;
  * NO reward or observation normalisation of any kind. CleanRL's script wraps its environments in
    `NormalizeObservation`, an observation clip at 10, `NormalizeReward` at `gamma` and a reward
    clip at 10; none of them is reproduced (CONTRACT rule 4; observation normalisation is off,
    and the manifest entry records `observation_normalisation: false`);
  * a constant learning rate: CleanRL's default linear learning-rate anneal is not applied, since
    the learning rate is a pinned TECH constant (recorded as `learning_rate_anneal: false`).

ENTROPY. A tanh-squashed Gaussian has no closed-form entropy. The entropy bonus uses the PRE-SQUASH
Gaussian entropy `0.5 + 0.5 log(2 pi) + log_std` per active, phase-relevant element (the CleanRL
quantity), not the entropy of the bounded action.

Binds: `tests/unit/test_ppo_adapter.py` (forward takes `obs` only - T-B5; no `RunningMeanStd` on
rewards, by inspection of the wrapped object; sampled actions within bounds) and, downstream, the
G2 criteria of PLAN section 4.5 via WO-019 and WO-020.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from gosplan.agents.base import Array, Phase
from gosplan.env.env import action_spec, active_action_dims
from gosplan.env.obs import obs_spec
from gosplan.env.reward import reward_scale
from gosplan.env.state import EnterpriseAction

if TYPE_CHECKING:  # runtime homes: WO-003 (config), WO-009 (state); PLAN section 8
    from pathlib import Path

    from gosplan.config import EnvConfig

# ---------- pinned reference (LEAD ruling AMBIGUITY-016, point 1) ----------

REFERENCE_IMPL: str = (
    "cleanrl ppo_continuous_action (algorithm), JAX re-expression in gosplan.agents.ppo"
)
"""The pinned reference PPO, verbatim from LEAD ruling AMBIGUITY-016 point 1."""

REFERENCE_VERSION: str = "cleanrl v1.0.0 algorithm; gosplan-ppo 1"
"""The pinned reference version, verbatim from LEAD ruling AMBIGUITY-016 point 1."""

# ---------- CleanRL `ppo_continuous_action` constants that are not `PPOConfig` fields ----------
# Transcribed from the reference algorithm (AMBIGUITY-016 point 1) and recorded in the manifest
# through `IPPO.manifest_entry()["reference_constants"]`.

HIDDEN_SIZES: tuple[int, int] = (64, 64)
"""Two tanh hidden layers of 64 units, separate actor and critic networks (CleanRL)."""

HIDDEN_INIT_GAIN: float = math.sqrt(2.0)
"""Orthogonal-init gain of the hidden layers (CleanRL `layer_init` default, std = sqrt(2))."""

ACTOR_OUT_INIT_GAIN: float = 0.01
"""Orthogonal-init gain of the actor-mean output layer (CleanRL)."""

CRITIC_OUT_INIT_GAIN: float = 1.0
"""Orthogonal-init gain of the critic output layer (CleanRL)."""

ADAM_EPS: float = 1e-5
"""Adam epsilon (CleanRL: `optim.Adam(..., eps=1e-5)`)."""

ADAM_BETAS: tuple[float, float] = (0.9, 0.999)
"""Adam betas (the torch defaults CleanRL inherits)."""

VF_COEF: float = 0.5
"""Value-loss coefficient (CleanRL `vf_coef`)."""

MAX_GRAD_NORM: float = 0.5
"""Global gradient-norm clip over actor and critic together (CleanRL `max_grad_norm`)."""

UPDATE_EPOCHS: int = 4
"""Epochs over each rollout batch (CleanRL `update_epochs`)."""

NUM_MINIBATCHES: int = 4
"""Minibatches per epoch (CleanRL `num_minibatches`)."""

CLIP_VALUE_LOSS: bool = True
"""Clipped value loss, clipped at `PPOConfig.clip_coef` (CleanRL `clip_vloss`)."""

ADV_NORM_EPS: float = 1e-8
"""Denominator guard of the per-minibatch advantage normalisation (CleanRL)."""

_LOG_2PI = math.log(2.0 * math.pi)
_LOG_2 = math.log(2.0)
_PRODUCE_DIMS = frozenset({"effort", "quality", "invest"})
_REPORT_DIMS = frozenset({"report_ratio", "input_request", "trade_offer"})
_ALL_DIMS = ("effort", "quality", "invest", "report_ratio", "input_request", "trade_offer")
_INVERSE_SQUASH_GUARD = 1e-12


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

    reference_impl: str = REFERENCE_IMPL
    """Import path or package name of the reference PPO this adapter wraps, e.g. a CleanRL-style
    continuous PPO on the NumPy path or a PureJaxRL/JaxMARL-style loop on the JAX path (PLAN section
    6.1). **No default**: PLAN names families, not a package, and the lead verifies the choice when
    WO-017 is issued. Recorded in the manifest (CONTRACT rule 10). LEAD ruling AMBIGUITY-016 point
    2: the default is the lead's pin, `REFERENCE_IMPL`, which satisfies "filled by the lead"."""

    reference_version: str = REFERENCE_VERSION
    """Exact pinned version (release tag or commit) of `reference_impl`. **No default** for the same
    reason. "reference-PPO version" is a named field of the CONTRACT rule 10 manifest, so a run
    whose learner version is unknown is not a reportable run. LEAD ruling AMBIGUITY-016 point 2:
    the default is the lead's pin, `REFERENCE_VERSION`."""

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
        _require_shared_parameters(cfg)
        self.cfg = cfg
        self.ppo = ppo
        # CONTRACT rule 4, switched off visibly: CleanRL's `ppo_continuous_action` wraps its
        # environments in observation and reward normalisers and a reward clip by default; none of
        # them exists here and these records say so (see `manifest_entry`).
        self.reward_normalisation = False
        self.return_scaling = False
        self.reward_clipping = False
        self.observation_normalisation = False
        self.scale = float(reward_scale(cfg))
        self.head_names = tuple(active_action_dims(cfg))
        self.report_head_init_ratio = float(ppo.report_head_init_ratio)
        self.report_head_init_std = float(ppo.report_head_init_std)
        self.obs_dim = len(obs_spec(cfg))
        spec = action_spec(cfg)
        self._head_slices: dict[str, slice] = {}
        self._head_shapes: dict[str, tuple[int, ...]] = {}
        lo, hi, offset = [], [], 0
        for name in self.head_names:
            shape, low, high = spec[name]
            size = int(np.prod(shape[1:], dtype=int))
            self._head_slices[name] = slice(offset, offset + size)
            self._head_shapes[name] = tuple(shape)
            lo += [float(low)] * size
            hi += [float(high)] * size
            offset += size
        self.action_dim = offset
        self._lo = np.asarray(lo, dtype=float)
        self._hi = np.asarray(hi, dtype=float)
        self._element_masks = {
            phase: np.concatenate(
                [
                    np.full(self._head_slices[name].stop - self._head_slices[name].start, flag)
                    for name, flag in self.phase_mask(phase).items()
                ]
            ).astype(np.float32)
            for phase in ("produce", "report")
        }
        self.report_head_z0, self.report_head_log_std = _report_head_init(
            spec["report_ratio"][1],
            spec["report_ratio"][2],
            self.report_head_init_ratio,
            self.report_head_init_std,
        )
        init_rng = np.random.default_rng(np.random.SeedSequence(int(cfg.tech.seed_policy)))
        params = _init_params(self.obs_dim, self.action_dim, init_rng)
        if "report_ratio" in self._head_slices:
            report = self._head_slices["report_ratio"]
            params["actor_b2"][report] = self.report_head_z0
            params["actor_log_std"][report] = self.report_head_log_std
        self.policy = _CleanRLContinuousPPO(params, self._lo, self._hi, ppo)

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
        mean, log_std, value = self.policy.forward(np.asarray(obs, dtype=np.float32))
        return np.asarray(mean), np.asarray(log_std), np.asarray(value)

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
        obs = np.asarray(obs, dtype=np.float32)
        eps = rng.standard_normal((obs.shape[0], self.action_dim)).astype(np.float32)
        mask = np.broadcast_to(self._element_masks[phase], eps.shape)
        _z, action, _logp, _value = self.policy.sample(obs, eps, mask)
        return self._to_action(np.asarray(action, dtype=float))

    def reset(self) -> None:
        """Clear per-episode policy state.

        Takes: nothing. Returns: `None`. A feed-forward reference policy has none and this is a
        no-op; a recurrent one clears its hidden state here. It never touches parameters, the
        optimiser or the configuration.

        Owning WO: **WO-017**.
        """
        return None

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
        obs = np.asarray(obs, dtype=np.float32)
        a = np.concatenate(
            [
                np.asarray(getattr(action, name), dtype=float).reshape(
                    *obs.shape[:-1], self._head_slices[name].stop - self._head_slices[name].start
                )
                for name in self.head_names
            ],
            axis=-1,
        )
        u = 2.0 * (a - self._lo) / (self._hi - self._lo) - 1.0
        guard = 1.0 - _INVERSE_SQUASH_GUARD
        z = np.arctanh(np.clip(u, -guard, guard)).astype(np.float32)
        report = _acting_phase_is_report(obs, self.cfg.incentive.steps_per_period)
        mask = np.where(
            report[..., None], self._element_masks["report"], self._element_masks["produce"]
        ).astype(np.float32)
        log_prob, entropy, value = self.policy.evaluate(obs, z, mask)
        return np.asarray(log_prob), np.asarray(entropy), np.asarray(value)

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
        if phase not in ("produce", "report"):
            raise ValueError(f"phase must be 'produce' or 'report' (got {phase!r})")
        read = _PRODUCE_DIMS if phase == "produce" else _REPORT_DIMS
        return {name: name in read for name in self.head_names}

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
        spec = action_spec(self.cfg)
        _shape, lo, hi = spec[name]
        return lo + (hi - lo) * (np.tanh(np.asarray(raw, dtype=float)) + 1.0) / 2.0

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
        entry: dict[str, object] = dict(dataclasses.asdict(self.ppo))
        entry.update(
            {
                "reference_impl": self.ppo.reference_impl,
                "reference_version": self.ppo.reference_version,
                "reference_ppo_version": (
                    f"{self.ppo.reference_impl} @ {self.ppo.reference_version}"
                ),
                "reference_constants": {
                    "hidden_sizes": list(HIDDEN_SIZES),
                    "activation": "tanh",
                    "separate_actor_critic": True,
                    "hidden_init_gain": HIDDEN_INIT_GAIN,
                    "actor_out_init_gain": ACTOR_OUT_INIT_GAIN,
                    "critic_out_init_gain": CRITIC_OUT_INIT_GAIN,
                    "optimizer": "adam",
                    "adam_eps": ADAM_EPS,
                    "adam_betas": list(ADAM_BETAS),
                    "learning_rate_anneal": False,
                    "vf_coef": VF_COEF,
                    "max_grad_norm": MAX_GRAD_NORM,
                    "update_epochs": UPDATE_EPOCHS,
                    "num_minibatches": NUM_MINIBATCHES,
                    "clip_value_loss": CLIP_VALUE_LOSS,
                    "target_kl": None,
                    "log_std": "state-independent parameter per action element",
                    "entropy": "pre-squash Gaussian entropy (no closed form after tanh)",
                },
                "framework": f"jax {jax.__version__}",
                "head_names": list(self.head_names),
                "action_dim": self.action_dim,
                "obs_dim": self.obs_dim,
                "param_sharing": self.cfg.tech.param_sharing,
                "scale": self.scale,
                "reward_normalisation": False,
                "return_scaling": False,
                "reward_clipping": False,
                "observation_normalisation": False,
                "report_head_z0": self.report_head_z0,
                "report_head_log_std": self.report_head_log_std,
                "init_seed": "numpy SeedSequence(cfg.tech.seed_policy)",
            }
        )
        return entry

    def save_checkpoint(self, path: Path) -> None:
        """Write policy parameters and optimiser state to disk.

        Takes: `path`, the destination file (created by `train.py` under `runs/<hash>/`). Returns:
        `None`. Writes whatever the reference implementation needs to resume exactly, plus the
        `PPOConfig` and the configuration hash, so a checkpoint can never be reloaded against a
        different environment configuration without the mismatch being detectable.

        Owning WO: **WO-017**.
        """
        arrays = self.policy.state_arrays()
        arrays["meta_config_hash"] = np.asarray(self.cfg.hash())
        arrays["meta_ppo_config"] = np.asarray(_ppo_json(self.ppo))
        arrays["meta_head_names"] = np.asarray(json.dumps(list(self.head_names)))
        with open(path, "wb") as handle:
            np.savez(handle, **arrays)

    def load_checkpoint(self, path: Path) -> None:
        """Restore policy parameters and optimiser state from disk.

        Takes: `path`. Returns: `None`. Raises `ValueError` when the checkpoint's configuration hash
        or `PPOConfig` disagrees with this adapter's - silently loading mismatched parameters would
        attribute one run's behaviour to another run's configuration.

        Owning WO: **WO-017**.
        """
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: np.asarray(data[key]) for key in data.files}
        if str(arrays.pop("meta_config_hash")) != self.cfg.hash():
            raise ValueError(f"checkpoint {path}: configuration hash differs from this adapter's")
        if str(arrays.pop("meta_ppo_config")) != _ppo_json(self.ppo):
            raise ValueError(f"checkpoint {path}: PPOConfig differs from this adapter's")
        if json.loads(str(arrays.pop("meta_head_names"))) != list(self.head_names):
            raise ValueError(f"checkpoint {path}: head_names differ from this adapter's")
        self.policy.load_state_arrays(arrays)

    # ---------- private helpers shared with the training harness (WO-018) ----------

    def element_mask(self, phase: Phase) -> Array:
        """`phase_mask(phase)` expanded to one 0/1 entry per action element, `(action_dim,)`."""
        return self._element_masks[phase]

    def _to_action(self, a: Array) -> EnterpriseAction:
        """Split squashed rows `(N, action_dim)` into an `EnterpriseAction`; inactive dims are 0."""
        a = np.asarray(a, dtype=float)
        spec = action_spec(self.cfg)
        fields = {}
        for name in _ALL_DIMS:
            if name in self._head_slices:
                fields[name] = a[:, self._head_slices[name]].reshape(self._head_shapes[name])
            else:
                fields[name] = np.zeros(spec[name][0])
        return EnterpriseAction(**fields)


# =================================================================================================
# The reference algorithm: CleanRL `ppo_continuous_action`, re-expressed in JAX (AMBIGUITY-016)
# =================================================================================================


def _require_shared_parameters(cfg: EnvConfig) -> None:
    """Phase 1 builds the `"shared"` structure only; the other two arrive with Phase 2."""
    if cfg.tech.param_sharing != "shared":
        raise NotImplementedError(
            f"param_sharing={cfg.tech.param_sharing!r}: only 'shared' is built in Phase 1; "
            "'per_sector' and 'independent' are Phase 2 (PLAN section 6.1, WO-017 note 4)"
        )


def _report_head_init(lo: float, hi: float, ratio: float, std: float) -> tuple[float, float]:
    """WO-017 note 3: bias `z0` with `a(z0) = ratio`, and the log-std giving `std` in ratio units.

    `z0 = atanh(2 * (ratio - lo) / (hi - lo) - 1)`;
    `log_std = log(std / ((hi - lo) / 2 * (1 - tanh(z0)**2)))`.
    """
    z0 = math.atanh(2.0 * (ratio - lo) / (hi - lo) - 1.0)
    jacobian = (hi - lo) / 2.0 * (1.0 - math.tanh(z0) ** 2)
    return z0, math.log(std / jacobian)


def _orthogonal(rng: np.random.Generator, n_in: int, n_out: int, gain: float) -> np.ndarray:
    """`torch.nn.init.orthogonal_` on an `(n_out, n_in)` weight, returned as `(n_in, n_out)`."""
    rows, cols = n_out, n_in
    flat = rng.standard_normal((rows, cols))
    if rows < cols:
        flat = flat.T
    q, r = np.linalg.qr(flat)
    q = q * np.sign(np.diag(r))
    if rows < cols:
        q = q.T
    return (gain * q).T.astype(np.float32)


def _init_params(obs_dim: int, action_dim: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """CleanRL `layer_init`: orthogonal weights, zero biases; zero log-std (std 1 pre-squash)."""
    h0, h1 = HIDDEN_SIZES
    params: dict[str, np.ndarray] = {}
    for net, out_dim, out_gain in (
        ("critic", 1, CRITIC_OUT_INIT_GAIN),
        ("actor", action_dim, ACTOR_OUT_INIT_GAIN),
    ):
        params[f"{net}_w0"] = _orthogonal(rng, obs_dim, h0, HIDDEN_INIT_GAIN)
        params[f"{net}_b0"] = np.zeros(h0, dtype=np.float32)
        params[f"{net}_w1"] = _orthogonal(rng, h0, h1, HIDDEN_INIT_GAIN)
        params[f"{net}_b1"] = np.zeros(h1, dtype=np.float32)
        params[f"{net}_w2"] = _orthogonal(rng, h1, out_dim, out_gain)
        params[f"{net}_b2"] = np.zeros(out_dim, dtype=np.float32)
    params["actor_log_std"] = np.zeros(action_dim, dtype=np.float32)
    return params


def _mlp(p: dict, net: str, x: jax.Array) -> jax.Array:
    h = jnp.tanh(x @ p[f"{net}_w0"] + p[f"{net}_b0"])
    h = jnp.tanh(h @ p[f"{net}_w1"] + p[f"{net}_b1"])
    return h @ p[f"{net}_w2"] + p[f"{net}_b2"]


def _net(p: dict, obs: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Pre-squash mean, state-independent log-std broadcast to the mean's shape, and value."""
    mean = _mlp(p, "actor", obs)
    log_std = jnp.broadcast_to(p["actor_log_std"], mean.shape)
    return mean, log_std, _mlp(p, "critic", obs)[..., 0]


def _log_abs_det(z: jax.Array, lo: jax.Array, hi: jax.Array) -> jax.Array:
    """`log |da/dz|` of `a = lo + (hi - lo) (tanh z + 1) / 2`, elementwise.

    `log(1 - tanh(z)**2) = 2 (log 2 - z - softplus(-2 z))`, which is finite for every `z`.
    """
    return jnp.log((hi - lo) / 2.0) + 2.0 * (_LOG_2 - z - jax.nn.softplus(-2.0 * z))


def _log_prob(mean, log_std, z, mask, lo, hi) -> jax.Array:
    """Squashed-Gaussian log-density of the boxed action, summed over the phase's elements."""
    gauss = -0.5 * jnp.square((z - mean) / jnp.exp(log_std)) - log_std - 0.5 * _LOG_2PI
    return jnp.sum((gauss - _log_abs_det(z, lo, hi)) * mask, axis=-1)


def _entropy(log_std, mask) -> jax.Array:
    """Pre-squash Gaussian entropy summed over the phase's elements (no closed form post-tanh)."""
    return jnp.sum((0.5 + 0.5 * _LOG_2PI + log_std) * mask, axis=-1)


@jax.jit
def _forward_jit(p, obs):
    return _net(p, obs)


@jax.jit
def _sample_jit(p, obs, eps, mask, lo, hi):
    mean, log_std, value = _net(p, obs)
    z = mean + jnp.exp(log_std) * eps
    a = lo + (hi - lo) * (jnp.tanh(z) + 1.0) / 2.0
    return z, a * mask, _log_prob(mean, log_std, z, mask, lo, hi), value


@jax.jit
def _evaluate_jit(p, obs, z, mask, lo, hi):
    mean, log_std, value = _net(p, obs)
    return _log_prob(mean, log_std, z, mask, lo, hi), _entropy(log_std, mask), value


def _ppo_loss(p, mb, ent_coef, lo, hi, *, clip_coef: float, normalise_advantages: bool):
    """CleanRL's loss: clipped surrogate, clipped value loss, entropy bonus."""
    obs, z, mask, old_logp, adv, ret, old_value = mb
    mean, log_std, value = _net(p, obs)
    new_logp = _log_prob(mean, log_std, z, mask, lo, hi)
    entropy = _entropy(log_std, mask)
    log_ratio = new_logp - old_logp
    ratio = jnp.exp(log_ratio)
    if normalise_advantages:  # per-minibatch, CONTRACT rule 4 permits it; torch std is ddof=1
        adv = (adv - adv.mean()) / (jnp.std(adv, ddof=1) + ADV_NORM_EPS)
    pg_loss = jnp.maximum(
        -adv * ratio, -adv * jnp.clip(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
    ).mean()
    v_unclipped = jnp.square(value - ret)
    if CLIP_VALUE_LOSS:
        v_clipped = old_value + jnp.clip(value - old_value, -clip_coef, clip_coef)
        v_loss = 0.5 * jnp.maximum(v_unclipped, jnp.square(v_clipped - ret)).mean()
    else:
        v_loss = 0.5 * v_unclipped.mean()
    entropy_loss = entropy.mean()
    loss = pg_loss - ent_coef * entropy_loss + VF_COEF * v_loss
    aux = {
        "policy_loss": pg_loss,
        "value_loss": v_loss,
        "entropy": entropy_loss,
        "old_approx_kl": (-log_ratio).mean(),
        "approx_kl": ((ratio - 1.0) - log_ratio).mean(),
        "clipfrac": (jnp.abs(ratio - 1.0) > clip_coef).mean(),
    }
    return loss, aux


def _minibatch_step(p, opt, data, idx, ent_coef, lo, hi, *, clip_coef, normalise_advantages, lr):
    """One CleanRL minibatch step: loss, global-norm clip 0.5, Adam (torch semantics)."""
    mb = tuple(x[idx] for x in data)
    grad_fn = jax.value_and_grad(
        partial(_ppo_loss, clip_coef=clip_coef, normalise_advantages=normalise_advantages),
        has_aux=True,
    )
    (_loss, aux), grads = grad_fn(p, mb, ent_coef, lo, hi)
    leaves = jax.tree_util.tree_leaves(grads)
    norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in leaves))
    grads = jax.tree_util.tree_map(
        lambda g: g * jnp.minimum(1.0, MAX_GRAD_NORM / (norm + 1e-6)), grads
    )
    m, v, t = opt
    b1, b2 = ADAM_BETAS
    t = t + 1.0
    m = jax.tree_util.tree_map(lambda m_, g: b1 * m_ + (1.0 - b1) * g, m, grads)
    v = jax.tree_util.tree_map(lambda v_, g: b2 * v_ + (1.0 - b2) * jnp.square(g), v, grads)
    step = lr / (1.0 - b1**t)
    root_bc2 = jnp.sqrt(1.0 - b2**t)
    p = jax.tree_util.tree_map(
        lambda p_, m_, v_: p_ - step * m_ / (jnp.sqrt(v_) / root_bc2 + ADAM_EPS), p, m, v
    )
    aux["grad_norm"] = norm
    return p, (m, v, t), aux


class _CleanRLContinuousPPO:
    """The wrapped reference: parameters, Adam state and the jitted CleanRL update.

    Holds NO reward or return normaliser, NO observation normaliser and NO reward clip - the four
    records below are the visible switches CONTRACT rule 4 asks for; CleanRL's own script applies
    `NormalizeObservation`, an observation clip, `NormalizeReward` and a reward clip through its
    environment wrappers, none of which is reproduced.
    """

    reward_normalisation = False
    return_scaling = False
    reward_clipping = False
    observation_normalisation = False

    def __init__(
        self, params: dict[str, np.ndarray], lo: np.ndarray, hi: np.ndarray, ppo: PPOConfig
    ) -> None:
        self.params = {k: jnp.asarray(v) for k, v in params.items()}
        zeros = {k: jnp.zeros_like(v) for k, v in self.params.items()}
        self.opt_state = (zeros, dict(zeros), jnp.asarray(0.0, dtype=jnp.float32))
        self.lo = jnp.asarray(lo, dtype=jnp.float32)
        self.hi = jnp.asarray(hi, dtype=jnp.float32)
        self._step = jax.jit(
            partial(
                _minibatch_step,
                clip_coef=float(ppo.clip_coef),
                normalise_advantages=bool(ppo.normalise_advantages),
                lr=float(ppo.learning_rate),
            )
        )

    def forward(self, obs: np.ndarray):
        return _forward_jit(self.params, obs)

    def sample(self, obs: np.ndarray, eps: np.ndarray, mask: np.ndarray):
        return _sample_jit(self.params, obs, eps, mask, self.lo, self.hi)

    def evaluate(self, obs: np.ndarray, z: np.ndarray, mask: np.ndarray):
        return _evaluate_jit(self.params, obs, z, mask, self.lo, self.hi)

    def update(
        self, batch: dict[str, np.ndarray], ent_coef: float, rng: np.random.Generator
    ) -> dict[str, float]:
        """CleanRL's update: `UPDATE_EPOCHS` epochs of `NUM_MINIBATCHES` shuffled minibatches.

        `batch` holds flat `(B, ...)` arrays `obs, z, mask, logp, advantages, returns, values`.
        Minibatch permutations come from `rng` (the `seed_policy` stream, CONTRACT rule 9).
        Returns CleanRL's logged diagnostics (last minibatch; `clipfrac` averaged over all).
        """
        keys = ("obs", "z", "mask", "logp", "advantages", "returns", "values")
        data = tuple(jnp.asarray(batch[k], dtype=jnp.float32) for k in keys)
        size = int(data[0].shape[0])
        mb_size = size // NUM_MINIBATCHES
        if mb_size < 2 or size % NUM_MINIBATCHES:
            # A ragged last minibatch of one row makes the (ddof=1) advantage std NaN; refuse
            # rather than choose a way of dropping or merging rows.
            raise ValueError(
                f"batch of {size} rows must be a multiple of {NUM_MINIBATCHES} minibatches of at "
                "least 2 rows"
            )
        ent = jnp.asarray(ent_coef, dtype=jnp.float32)
        clipfracs = []
        aux: dict = {}
        for _epoch in range(UPDATE_EPOCHS):
            order = rng.permutation(size)
            for start in range(0, size, mb_size):
                idx = jnp.asarray(order[start : start + mb_size])
                self.params, self.opt_state, aux = self._step(
                    self.params, self.opt_state, data, idx, ent, self.lo, self.hi
                )
                clipfracs.append(aux["clipfrac"])
        out = {key: float(value) for key, value in aux.items()}
        out["clipfrac"] = float(jnp.mean(jnp.stack(clipfracs)))
        values, returns = batch["values"], batch["returns"]
        var_y = float(np.var(returns))
        out["explained_variance"] = (
            float("nan") if var_y == 0.0 else float(1.0 - np.var(returns - values) / var_y)
        )
        return out

    def state_arrays(self) -> dict[str, np.ndarray]:
        m, v, t = self.opt_state
        arrays = {f"param/{k}": np.asarray(x) for k, x in self.params.items()}
        arrays.update({f"adam_m/{k}": np.asarray(x) for k, x in m.items()})
        arrays.update({f"adam_v/{k}": np.asarray(x) for k, x in v.items()})
        arrays["adam_step"] = np.asarray(t)
        return arrays

    def load_state_arrays(self, arrays: dict[str, np.ndarray]) -> None:
        params = {k: arrays[f"param/{k}"] for k in self.params}
        for key, value in params.items():
            if value.shape != self.params[key].shape:
                raise ValueError(f"checkpoint parameter {key}: shape {value.shape} differs")
        self.params = {k: jnp.asarray(v) for k, v in params.items()}
        m = {k: jnp.asarray(arrays[f"adam_m/{k}"]) for k in self.params}
        v = {k: jnp.asarray(arrays[f"adam_v/{k}"]) for k in self.params}
        self.opt_state = (m, v, jnp.asarray(arrays["adam_step"], dtype=jnp.float32))


def _ppo_json(ppo: PPOConfig) -> str:
    return json.dumps(dataclasses.asdict(ppo), sort_keys=True)


def _acting_phase_is_report(obs: np.ndarray, m: int) -> np.ndarray:
    """Which rows were acted on at a REPORT step, read from observation fields 0 and 1.

    `GosplanEnv` builds each observation from the step just executed, so the agent acts at the
    REPORT step exactly when the observation shows PRODUCE (field 0 = 0) at `k / M = (M - 1) / M`.
    With `M = 1` a fresh episode and a finished PRODUCE step look identical, so it is refused.
    """
    if m < 2:
        raise ValueError("evaluate_actions: phase is not recoverable from obs when M < 2")
    return (obs[..., 0] == 0.0) & np.isclose(obs[..., 1], (m - 1) / m)


__all__ = ["IPPO", "PPOConfig"]
