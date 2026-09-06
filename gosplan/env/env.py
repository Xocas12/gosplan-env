"""The `GosplanEnv` wrapper: reset/step, the action specs, and the StepInfo -> ledger hookup.

Realises: PLAN section 2.5 (period schedule, driven through `gosplan/env/step.py`), PLAN section 2.3
(action space and the active-dimension rule), PLAN section 2.4 (the observation the wrapper returns,
built by `gosplan/env/obs.py`), PLAN section 2.10 (the plan prices precomputed at construction) and
PLAN section 2.12 (the geometric termination flag). Owning work order: **WO-009** (Step function and
env wrapper) - a **LEAD**-owned unit (PLAN sections 12.3, 1.3 finding F14).

This is the `env.py` row of the PLAN section 8 layout: "reset/step wrapper, specs, info/ledger
hookup". It holds no economics. The schedule is `gosplan/env/step.py`; the arithmetic is
`production.py`, `planner.py`, `reporting.py`, `reward.py`, `prices.py` and `obs.py`. `action_spec`
and `active_action_dims` live here because they are WO-009-owned in `spec/spec.py` and because the
PPO adapter (WO-017) reads them from the environment rather than from the spec file.

Three invariants this wrapper is responsible for.

**StepInfo is not agent-facing (CONTRACT rule 6).** `StepInfo` carries the true quantities - each
enterprise's realised output `y`, its stock `S`, its input stocks `X`, the audit measurement, and
the period-level `val_measured`, `val_true` and `welfare` - so that the ledger can check the
conservation identity and the forensic estimators of PLAN section 7.3 have something to be run
against. It is written by the environment and read by `gosplan/metrics/ledger.py` and by lead-run
experiments only. No agent, no policy, no reward term and no observation may read it: `Agent.act`
takes `obs`, `phase` and an RNG and nothing else, the PPO adapter's forward pass takes `obs` only,
and the WO-010 forbidden list names "any agent reading `StepInfo`" explicitly. Test T-B5 plants
sentinel values in `welfare`, in other enterprises' `y`, and in periods-remaining, and asserts none
of them appears in any observation.

**Reward is per enterprise (N,).** `step` returns a vector, not a scalar: this is a multi-agent
environment with `N = cfg.supply.n_enterprises` enterprises acting simultaneously, one shared policy
in Phase 1 (`param_sharing = "shared"`, with the sector one-hot in the observation carrying
identity). The terms are exactly those of CONTRACT rule 4 - `-scale * c_ik` at a PRODUCE step,
`scale * (B(rho) - penalty + trade_surplus)` at the REPORT step - with `scale = reward_scale(cfg)`
fixed analytically per configuration and no running normalisation anywhere.

**Done is the geometric termination flag.** `done` is one episode-level `bool` decided by a single
global draw at stage TERMINATE (PLAN section 2.12, purpose `terminate`): after `min_periods` the
episode continues with probability `tenure`, capped at `max_periods`. It is not a per-enterprise
flag, there is no truncation signal distinct from it in Phase 1, and **no observation field carries
periods remaining** - the agent learns the episode ended only when it has (finding F4). Tenure is an
economic parameter (INC arm), distinct from the PPO discount `gamma` that lives with the adapter.

Binding tests (PLAN section 11, WO-009 must-pass list): `tests/unit/test_conservation.py` (T-U1, the
per-period per-good identity to 1e-9 over trajectories driven through `step`), `tests/golden/*`
(T-B7, agreement with `ref/ref_step.py` to 1e-9 on seeded trajectories with `Random` and
`TruthfulMyopic`), `tests/behavioural/test_termination.py` (T-B9, empirical continuation equals
`tenure` and no observation field correlates with periods remaining), and
`tests/unit/test_env_api.py`.

Cross-module bindings. `State`, `EnterpriseAction`, `StepInfo` and the literal alias `Phase` are the
runtime declarations of `gosplan/env/state.py` - `StepInfo` is declared there, beside the state
record, because both this module and `gosplan/env/step.py` bind it; `Ledger` and `StepRecord` are
those of `gosplan/metrics/ledger.py`, and `EnvConfig` that of `gosplan/config.py`. Each must stay
field-for-field and name-for-name identical to `spec/spec.py`, which is not importable as a package;
`tests/unit/test_spec_imports.py` enforces the surface and `tests/unit/test_env_api.py` the fields.
Those imports are type-only here so this skeleton imports cleanly before the modules that own them
land; the implementer promotes the ones it calls at runtime. `advance` is imported at runtime
because `step` calls it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gosplan.env.step import advance

if TYPE_CHECKING:  # type-only: see the cross-module bindings note in the module docstring
    from gosplan.config import EnvConfig
    from gosplan.env.state import EnterpriseAction, Phase, State, StepInfo
    from gosplan.metrics.ledger import Ledger

Array = np.ndarray
"""Alias for every numeric array in this module (PLAN section 10). The Phase-2 JAX port (WO-029)
substitutes its own array type behind the same name."""

__all__ = ["GosplanEnv", "action_spec", "active_action_dims", "advance"]


def action_spec(cfg: EnvConfig) -> dict[str, tuple[tuple[int, ...], float, float]]:
    """Return the shape and box bounds of every action dimension (PLAN section 2.3).

    Takes: `cfg`. Returns: a mapping `name -> (shape, lo, hi)` covering all six dimensions whether
    or not they are active, with `N = cfg.supply.n_enterprises` and `J = cfg.supply.n_sectors`:

        "effort"         ((N,),   0.0, 1.0)
        "quality"        ((N,),   0.0, 1.0)
        "invest"         ((N,),   0.0, 1.0)
        "report_ratio"   ((N,),   0.0, cfg.tech.report_max_ratio)
        "input_request"  ((N, J), 0.0, cfg.tech.request_max_multiple)
        "trade_offer"    ((N, J), -1.0, 1.0)

    The dimension set is fixed across phases; configuration flags decide which dimensions the
    environment reads, and dimensions the current phase does not read are ignored rather than
    rejected, so a Phase-1 policy and a Phase-2 policy share one action type.

    `input_request` is expressed as a multiple of need because the true bound of PLAN section 2.3 is
    `r_max * need_ij` and `need_ij` is state-dependent; the environment rescales and clips it
    against the current need when it reads the action (`process_reports`, WO-007).

    The report bound is a result, not a nuisance: CONTRACT rule 8 forbids widening or narrowing it
    to fix an outcome, the fraction of reports at the bound is logged, and above 1% the run manifest
    is flagged `BOUND_BINDING` (test T-B8).

    Binds: `tests/unit/test_env_api.py` (all six keys present with these shapes and bounds; bounds
    track `cfg`) and `tests/unit/test_ppo_adapter.py` (WO-017 builds heads inside these bounds).
    Owning WO: **WO-009**.
    """
    raise NotImplementedError("PLAN section 2.3 - implemented in WO-009")


def active_action_dims(cfg: EnvConfig) -> list[str]:
    """Return the names of the action dimensions this configuration actually reads.

    Takes: `cfg`. Returns: a subset of `action_spec(cfg).keys()`, in the order of PLAN section 2.3.
    At `p1_default_config()` this is `["effort", "report_ratio", "input_request"]`: `input_request`
    is active and logged although it is inert while `incentive.alloc_eta_request == 0` (finding F7);
    `quality`, `invest` and `trade_offer` become active only when their Phase-2 mechanisms are
    switched on (`supply.quality_matters`, non-zero investment/`supply.capital_dep`,
    `information.horizontal_visibility > 0`).

    The PPO adapter builds Gaussian heads only for these names (PLAN section 6.1, WO-017), which is
    why the answer must be a pure function of the configuration and must not change within a run.

    Binds: `tests/unit/test_env_api.py` (the Phase-1 list above; each Phase-2 toggle adds exactly
    its own dimension) and `tests/unit/test_ppo_adapter.py`. Owning WO: **WO-009**.
    """
    raise NotImplementedError("PLAN section 2.3 - implemented in WO-009")


class GosplanEnv:
    """The environment: the period schedule of PLAN section 2.5 as an explicit state machine.

    One plan period is:

        0. DELIVER      planner allocates from last period's claims (sections 2.7.2/2.7.3);
                        the consumer sink receives; X updated
        1. [P2] TRADE   bilateral matching (section 2.13)
        2. PRODUCE x M  agent chooses effort/quality/invest; yield realised; obs shows cumulative y
        3. REPORT       S <- (1-h)*S + y; agent observes S and y exactly; chooses report_ratio and
                        input_request (section 2.8)
        4. AUDIT        audit selection; measurement; penalty (section 2.8)
        5. REWARD       bonus - penalty (+ trade surplus) delivered; val and welfare logged (2.9)
        6. TARGET       ratchet + growth directive (section 2.7.1)
        7. TERMINATE?   geometric (section 2.12)

    Agents act `M + 1` times per period: `M` PRODUCE steps and one REPORT step. Action dimensions
    not relevant to the current phase are ignored rather than rejected, and the environment never
    trusts an agent to have masked them itself.

    The stages themselves live in `gosplan/env/step.py`; this class holds the episode bookkeeping,
    the precomputed per-run constants, the observation call and the ledger hookup. It is not a
    Gym/Gymnasium subclass: the reset/step signatures below are the frozen ones of `spec/spec.py`
    (`reset` takes both seeds explicitly and returns `(obs, info)`; `step` returns a `(N,)` reward
    and one episode-level `done`), and the PPO adapter of WO-017 wraps this surface rather than the
    reverse.

    Binds: `tests/golden/*` (T-B7), `tests/unit/test_conservation.py` (T-U1),
    `tests/unit/test_env_api.py`, and `tests/behavioural/test_termination.py` (T-B9). Owning WO:
    **WO-009** (LEAD).
    """

    cfg: EnvConfig
    """The validated configuration this environment was constructed for; never mutated after
    `__init__`, and written verbatim into the run manifest (CONTRACT rule 10)."""

    state: State
    """The true state (PLAN section 2.2), unset until `reset`. It holds quantities no agent and no
    planner rule may see (CONTRACT rules 5, 6): only `make_planner_view` reads it on the planner's
    behalf and only `gosplan/env/obs.py` reads it on an agent's behalf."""

    ledger: Ledger | None
    """The optional ledger attached by `attach_ledger`. When set, every `StepInfo` this environment
    produces is appended as one `StepRecord` per enterprise. `None` means the run is not being
    recorded; nothing else changes (CONTRACT rule 6 - the ledger never feeds back)."""

    def __init__(self, cfg: EnvConfig) -> None:
        """Construct the environment for one configuration.

        Takes: `cfg`, already validated. Returns: nothing. Stores the configuration, precomputes the
        quantities that never change within a run - plan prices from `initial_prices(cfg)`, the
        coverage weights `omega_j = a_{s(i)j} / sum_j a_{s(i)j}`, `reward_scale(cfg)`, the initial
        targets `T_0 = cfg.tech.initial_target_frac * A_{s(i)} * cap_i` and the target floor
        `T_min = cfg.tech.target_floor_frac * T_0` - and leaves `state` unset until `reset` and
        `ledger` at `None` until `attach_ledger`.

        Precomputing here rather than per step is what keeps the plan prices genuinely fixed in
        Phase 1 (`supply.price_lag = inf`, PLAN section 2.10) and keeps `reward_scale` analytic and
        constant, which CONTRACT rule 4 requires.

        Realises: PLAN sections 2.5, 2.10. Owning WO: **WO-009**.
        """
        raise NotImplementedError("PLAN section 2.5 - implemented in WO-009")

    def reset(self, seed_env: int, seed_policy: int) -> tuple[Array, StepInfo]:
        """Start a new episode.

        Takes: `seed_env`, the root seed for every environment draw (shared across arms to obtain
        common random numbers by construction, PLAN sections 2.15 and 4.3), and `seed_policy`, the
        separate policy stream, stored on the state and never used by any environment draw (CONTRACT
        rule 9). Returns: `(obs, info)` - the first observation `(N, d)` with `d = 12 + 3J` in
        Phase 1, and the opening `StepInfo`.

        Initial state (PLAN sections 2.1-2.2, 3): `target = T_0`, `capital = cap = 1`,
        `inv_output = 0`, `inv_inputs = 0`, `cum_output = cum_cost = quality_acc = 0`, all `last_*`
        fields zero except `last_fill = 1`, `request = 0`, `pending_invest = 0`, `t_period = 0`,
        `k_step = 0`, `phase = "produce"`, `plan_prices = initial_prices(cfg)`, `planner_io = a`,
        `consumer_delivery = 0`, `alive = True`.

        The opening `StepInfo` carries the initial `StepRecord`s and zeroed period-level metrics; it
        is for the ledger, not for the agent (CONTRACT rule 6).

        Owning WO: **WO-009**.
        """
        raise NotImplementedError("PLAN section 2.5 - implemented in WO-009")

    def step(self, action: EnterpriseAction) -> tuple[Array, Array, bool, StepInfo]:
        """Advance one agent-step through the schedule above.

        Takes: `action`, a joint `EnterpriseAction`; only the dimensions in
        `active_action_dims(cfg)` that the current phase reads are used. Returns:
        `(obs, reward, done, info)` -

            obs     (N, d), laid out exactly as `obs_spec(cfg)` (PLAN section 2.4), built by
                    `gosplan/env/obs.py`; it contains no true quantity of any other enterprise, no
                    welfare, no `val_measured`, no current-period audit selection and no
                    periods-remaining field
            reward  (N,), per enterprise, exactly the terms of CONTRACT rule 4 (see the module
                    docstring); `-scale * c_ik` at a PRODUCE step, `scale * (B(rho) - penalty +
                    trade_surplus)` at the REPORT step
            done    one episode-level `bool` - the geometric termination flag of PLAN section 2.12,
                    a single global draw with purpose `terminate`, not a per-enterprise flag
            info    a `StepInfo` of true quantities for the ledger only (CONTRACT rule 6)

        Delegates the schedule to `advance(state, action, cfg)` in `gosplan/env/step.py`: at a
        PRODUCE step that runs `produce_step` (preceded, at the period's first step, by DELIVER and
        TRADE); at the REPORT step it runs `process_reports`, `select_audits`, `audit_and_penalise`,
        the REWARD, `update_targets` and the termination draw, in that order. The next period opens
        with DELIVER, which consumes the `PlannerView` built from this period's reports. `step` then
        builds the observation from the returned state, appends the `StepInfo` to the ledger if one
        is attached, and returns.

        Nothing here may leak a true quantity into `obs` (CONTRACT rule 6) and every draw goes
        through `gosplan.rng.draw` (CONTRACT rule 9). Calling `step` after `done` is an error rather
        than an implicit auto-reset: the episode boundary must be visible to the harness that owns
        the seeds.

        Binds: T-B7 (golden parity with `ref/`), T-U1 (conservation), T-B9 (empirical continuation
        equals `tenure`; no observation field correlates with periods remaining). Owning WO:
        **WO-009** (LEAD).
        """
        raise NotImplementedError("PLAN section 2.5 - implemented in WO-009")

    def phase(self) -> Phase:
        """Return the phase the next call to `step` will execute.

        Takes: nothing. Returns: `"produce"` while `state.k_step < cfg.incentive.steps_per_period`,
        `"report"` at the period's last agent-step (PLAN section 2.5). Agents use it to mask
        inactive action dimensions; the environment never trusts an agent to have masked correctly
        and ignores whatever the inactive dimensions contain.

        Owning WO: **WO-009**.
        """
        raise NotImplementedError("PLAN section 2.5 - implemented in WO-009")

    def obs_spec(self) -> list[str]:
        """Return the ordered names of this environment's observation components (PLAN section 2.4).

        Takes: nothing beyond `self`. Returns: `obs_spec(self.cfg)` from `gosplan/env/obs.py`
        (WO-008) - a `list[str]` of length `12 + 3J` in Phase 1, in the canonical order of PLAN
        section 2.4, whose length equals `step`'s and `reset`'s observation width.

        An accessor, not a second definition: the layout has exactly one owner (WO-008) so a
        disagreement between the vector and its names is impossible. Never present in that list, in
        any phase: `welfare_true`, `val_measured`, any other enterprise's `y`, `S` or `X`, the audit
        selection for the current period, and periods remaining under geometric termination
        (CONTRACT rule 6, test T-B5).

        Binds: `tests/unit/test_env_api.py` - `len(env.obs_spec()) == env.reset(...)[0].shape[1]`.
        Owning WO: **WO-009**.
        """
        raise NotImplementedError("PLAN section 2.4 - implemented in WO-009")

    def action_spec(self) -> dict[str, tuple[tuple[int, ...], float, float]]:
        """Return this environment's action shapes and box bounds (PLAN section 2.3).

        Takes: nothing beyond `self`. Returns: `action_spec(self.cfg)`, the module-level function
        above - all six dimensions as `name -> (shape, lo, hi)`, whether or not they are active.

        An accessor, so the PPO adapter (WO-017) and the heuristic agents (WO-010) read bounds from
        the environment they are attached to rather than re-deriving them from a configuration they
        might not share. Owning WO: **WO-009**.
        """
        raise NotImplementedError("PLAN section 2.3 - implemented in WO-009")

    def active_action_dims(self) -> list[str]:
        """Return the action dimensions this environment actually reads (PLAN section 2.3).

        Takes: nothing beyond `self`. Returns: `active_action_dims(self.cfg)` - at
        `p1_default_config()`, `["effort", "report_ratio", "input_request"]`.

        An accessor. The answer is a pure function of the configuration and does not change within a
        run, which is what lets WO-017 build its policy heads once at construction. Owning WO:
        **WO-009**.
        """
        raise NotImplementedError("PLAN section 2.3 - implemented in WO-009")

    def attach_ledger(self, ledger: Ledger) -> None:
        """Attach a ledger so every subsequent step is recorded.

        Takes: `ledger`, a `Ledger` from `gosplan/metrics/ledger.py` (WO-011). Returns: `None`.
        Stores it on `self.ledger`; from the next `step` (and from the next `reset`) onward, each
        `StepInfo` is appended as one `StepRecord` per enterprise per agent-step.

        The hookup is one-directional and that is the whole point: the ledger consumes true
        quantities and returns nothing to the environment, so nothing it records can influence an
        observation, a reward or a planner rule (CONTRACT rule 6). Attaching or detaching a ledger
        must not change a single trajectory - `tests/unit/test_env_api.py` runs the same seeded
        episode with and without one and compares the observations and rewards exactly.

        Owning WO: **WO-009**; the ledger itself is **WO-011**.
        """
        raise NotImplementedError("PLAN section 4 - implemented in WO-009")

    def record_step(self, info: StepInfo) -> None:
        """Append one agent-step's `StepInfo` to the attached ledger, if any.

        Takes: `info`, the `StepInfo` produced by `advance` for the step just executed. Returns:
        `None`. A no-op when `self.ledger is None`; otherwise it appends `info.records` - one
        `StepRecord` per enterprise, in enterprise-index order - and lets the ledger maintain the
        run flag set, including the `BOUND_BINDING` flag raised when more than 1% of reports sit at
        `rho_max` (CONTRACT rule 8, test T-B8).

        This method is the **only** consumer of `StepInfo` inside the environment. `StepInfo`
        carries the true quantities of PLAN section 2.2 plus the period-level `val_measured`,
        `val_true` and `welfare` of PLAN section 2.9.3; CONTRACT rule 6 and the WO-010 forbidden
        list ("any agent reading `StepInfo`") make it unreachable from any agent-facing path. It is
        never used to build an observation, never used to compute a reward, and never fed back into
        a planner rule.

        Binds: `tests/unit/test_ledger.py` (one record per enterprise per agent-step; every rule-10
        field present) and T-B8. Owning WO: **WO-009**; the ledger itself is **WO-011**.
        """
        raise NotImplementedError("PLAN section 4 - implemented in WO-009")
