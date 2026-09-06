"""The PPO adapter: what it may see, what it may not normalise, and where its heads come from.

Realises: PLAN section 6.1 (agents; the IPPO adapter over a pinned reference PPO), PLAN section
2.9.1 (the analytic reward scale) and PLAN section 11 (test architecture; the adapter half of
**T-B5**). Owning work order: **WO-002** (frozen tests; LEAD). Binds the WO-017 must-pass line of
PLAN section 12.3, verbatim - "`tests/unit/test_ppo_adapter.py` (forward takes `obs` only - T-B5; no
`RunningMeanStd` on rewards, checked by inspection of the wrapped object; actions within bounds)".
Module under test: `gosplan/agents/ppo/adapter.py`.

CONTRACT RULE 6, verbatim: "`welfare_true` and `val_measured` are logged and never appear in any
observation, reward, or agent input. The PPO adapter's forward pass takes obs only."

CONTRACT RULE 4, verbatim on the point this module tests: "No running reward normalisation (running
statistics change the effective reward over training and, with heavy-tailed penalties, shrink the
notch in normalised units). Per-batch advantage normalisation inside PPO is permitted.
`scale = reward_scale(cfg)`, computed analytically from the configuration."

WO-017 card, verbatim: "Heads only for `active_action_dims`; tanh-squashed Gaussian; report head
bias initialised at `rho = 1`; sector one-hot in obs handled by `param_sharing`; fixed
`reward_scale`; no reward normalisation (rule 4); per-batch advantage normalisation on;
`gamma = 0.99`, `lambda_GAE = 0.97`; phase-aware masking of inactive dims per step."

FROZEN BY CONTRACT RULE 2. SKELETON: every test is `@pytest.mark.skeleton` and skipped until WO-017
lands; each docstring states the exact assertion, formula and tolerance.
"""

from __future__ import annotations

import pytest


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-017")
def test_forward_signature_takes_obs_and_nothing_else(p1_cfg) -> None:
    """T-B5, adapter half: `IPPO.forward` accepts exactly `(self, obs)`.

    Assertion: `inspect.signature(IPPO.forward)` has parameters `("self", "obs")` and no others -
    no `state`, no `info`, no `view`, no `phase`, no keyword with a default that could carry one -
    and the annotation of `obs` is the array alias. `phase` is deliberately not an argument: it
    reaches the policy through observation components 0 (`phase`) and 1 (`k_over_M`), and it is
    `act` that masks heads, not `forward`.

    First bullet of the WO-017 must-pass list, and the signature clause of CONTRACT rule 6. The
    behavioural half - sentinels in `welfare_true`, other enterprises' `y` and periods remaining,
    absent from every observation - is test T-B5 in
    `tests/behavioural/test_welfare_blindness.py`.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-017")
def test_no_running_reward_normalisation_on_the_wrapped_object(p1_cfg) -> None:
    """No `RunningMeanStd` (or equivalent) touches rewards or returns anywhere in the adapter.

    Assertion, by inspection of the constructed `IPPO` and of the reference-PPO object it wraps:
    no attribute of either - recursively over the wrapped object's own attributes - is a running
    reward or return normaliser (a `RunningMeanStd`-shaped object, a `VecNormalize` with
    `norm_reward=True`, a `return_rms`, a reward-scaling wrapper), and whatever the reference
    implementation enables by default has been switched off, visibly, at construction. Per-batch
    advantage normalisation (`PPOConfig.normalise_advantages = True`) is permitted and is expected
    to be present; there is deliberately no reward-normalisation field on `PPOConfig`, so it cannot
    be switched on from a config file.

    Second bullet of the WO-017 must-pass list, and CONTRACT rule 4: running statistics change the
    effective reward over training and, with heavy-tailed penalties, shrink the notch in normalised
    units - which would dissolve the very discontinuity the Phase-1 design measures.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-017")
def test_sampled_actions_lie_inside_the_action_spec_boxes(p1_cfg, rng_seed) -> None:
    """Every sampled action is inside its `action_spec(cfg)` box, at every phase.

    Assertion: over many samples from a freshly constructed and from a randomly perturbed policy,
    every active dimension of the returned `EnterpriseAction` lies within `[lo, hi]` from
    `action_spec(cfg)` - `effort` in [0, 1], `report_ratio` in [0, `report_max_ratio`],
    `input_request` in [0, `request_max_multiple`] - with the correct shapes `(N,)` and `(N, J)`,
    and no NaN. The tanh squash is what guarantees this: `squash(raw, name)` maps the real line
    onto the box, so the bound is never enforced by a post-hoc clip that would hide a policy pushing
    against it (CONTRACT rule 8: the fraction of reports at the bound is a logged result).

    Third bullet of the WO-017 must-pass list.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-017")
def test_heads_exist_only_for_the_active_action_dimensions(p1_cfg) -> None:
    """`head_names == tuple(active_action_dims(cfg))`, fixed for the life of the run.

    Assertion: at `p1_cfg` the adapter builds heads for exactly `("effort", "report_ratio",
    "input_request")`, in the order of PLAN section 2.3, sized by each dimension's shape in
    `action_spec(cfg)` (`(N,)`, or `(N, J)` for `input_request`); no head exists for `quality`,
    `invest` or `trade_offer`; and switching on a Phase-2 mechanism
    (`supply.quality_matters`, `information.horizontal_visibility > 0`) adds the corresponding head
    at construction. `head_names` does not change within a run, which is what lets the observation
    dimension `len(obs_spec(cfg)) = 12 + 3J` be fixed once at construction.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-017")
def test_report_head_is_initialised_on_the_notch(p1_cfg) -> None:
    """The report head starts with squashed mean `rho = 1` and standard deviation 0.05.

    Assertion: at construction, sampling `report_ratio` from the untrained policy gives a mean of
    `ppo.report_head_init_ratio = 1.0` to 0.01 and a standard deviation of
    `ppo.report_head_init_std = 0.05` to 0.01, in ratio units - the head's bias and log-std are
    pushed through the tanh squash and its local Jacobian to achieve that, rather than being set in
    pre-squash units and hoped for. Starting on the notch is deliberate (PLAN section 6.1): the
    question is what the policy does around `rho = 1`, and an initialisation far from it would make
    the answer a fact about exploration.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-017")
def test_phase_masking_disables_the_heads_the_step_does_not_read(p1_cfg, rng_seed) -> None:
    """`act` masks heads by phase; `forward` does not.

    Assertion: `phase_mask("produce")` marks `effort` active and `report_ratio` and `input_request`
    inactive, and `phase_mask("report")` the reverse; two `act` calls at the same observation and
    the same generator state differ only in the dimensions the phase reads; and `forward` returns
    the same `(mean, log_std, value)` whatever the phase, because the phase is not one of its
    arguments. The environment ignores inactive dimensions rather than rejecting them (PLAN section
    2.3) and never trusts the agent to have masked correctly - the mask is for the learner's
    gradients, not for the environment's safety.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-017")
def test_reward_scale_is_stored_analytically_and_never_reapplied(p1_cfg) -> None:
    """`IPPO.scale == reward_scale(cfg)`, computed from the configuration and applied once.

    Assertion: the adapter stores `scale = reward_scale(cfg)` at construction, equal to the
    environment's own value to 1e-12; the rewards the adapter consumes are the environment's,
    already scaled, and the adapter never multiplies them again nor divides them by a running
    statistic;
    and changing `cfg` changes `scale` deterministically, with no dependence on rollout data
    (PLAN section 2.9.1, finding F9).
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-017")
def test_parameter_sharing_uses_the_sector_one_hot(p1_cfg) -> None:
    """One shared policy acts for all `N` enterprises, identified only by the sector one-hot.

    Assertion: at `cfg.tech.param_sharing = "shared"` (Phase 1) a single parameter set produces
    actions for all `N` rows of `obs` in one forward pass; two enterprises whose observations are
    identical - including their sector one-hot block - receive the same distribution parameters, so
    the policy cannot condition on an enterprise index it is never given; and the observation
    dimension the adapter was built with equals `len(obs_spec(cfg))`.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-017")
def test_manifest_entry_pins_the_reference_implementation(p1_cfg) -> None:
    """`manifest_entry()` records the pinned learner, as CONTRACT rule 10 requires.

    Assertion: the mapping returned carries `reference_impl` and `reference_version` (both non-empty
    strings, with no default on `PPOConfig` - the lead pins them when WO-017 is issued) plus the
    TECH hyper-parameters `gamma = 0.99`, `lambda_gae = 0.97`, `learning_rate = 3e-4`,
    `clip_coef = 0.2`, `entropy_coef_start = 0.01`, `entropy_coef_end = 0.001` and
    `normalise_advantages = True`; and those values reach the manifest's `reference_ppo_version`
    field. A run whose learner version is unknown is not a reportable run.
    """
    assert False
