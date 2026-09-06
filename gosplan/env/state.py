"""Environment state: the struct-of-arrays record, the action record, and period bookkeeping.

Realises: PLAN sections 2.2 (state), 2.3 (actions), 2.5 (period schedule bookkeeping and the
per-step diagnostic payload), 2.11 (inventory), 2.12 (the `alive` flag) and 2.15 (the two seed
fields). Owning work order: **WO-009** (step function and env wrapper; LEAD).

Struct-of-arrays with leading dimension `N` throughout, so the Phase-2 JAX port (WO-029) is a
mechanical translation and its parity test is meaningful. Shapes are stated in a comment on every
field. Dimensions used below: `N = cfg.supply.n_enterprises`, `J = cfg.supply.n_sectors`,
`L = cfg.supply.invest_lag`, `M = cfg.incentive.steps_per_period`.

**Binding to the frozen interface.** `spec/spec.py` (PLAN section 10) declares `State`,
`EnterpriseAction` and `StepInfo`. `spec/` is a document directory, not an importable package, so
this module declares the runtime dataclasses instead of importing them, and they MUST stay
field-for-field identical to the frozen declarations: same field names, same order, same
annotations. `tests/unit/test_spec_imports.py` (WO-001) and `tests/unit/test_env_api.py` (WO-009)
enforce the agreement; any divergence is a spec change and needs a `spec/CHANGELOG.md` entry
(CONTRACT rule 1). The cross-module types this module names - `EnvConfig` and the literal alias
`Phase` from `gosplan/config.py` (WO-003), `StepRecord` from `gosplan/metrics/ledger.py` (WO-011) -
are imported under `TYPE_CHECKING`, so this module stays importable while its siblings are still
skeletons.

`State` is the *true* state. It holds quantities no agent and no planner rule may ever see
(CONTRACT rules 5 and 6): `inv_output`, `inv_inputs`, `cum_output`, `cum_cost` and the seeds.
Exactly two functions may read it on someone's behalf - `make_planner_view` in
`gosplan/env/planner.py` (the single `State -> planner` boundary) and the observation builder in
`gosplan/env/obs.py`. Everything else that touches a `State` is environment-internal.

**Inventory rules (PLAN section 2.11), which this record is the home of.**

  * `inv_output` (`S_i`) is the *only* sink for output that is not reported or not delivered. It
    accumulates whenever the claim is below the stock on hand (`R_i < S_i`), it drains by physical
    delivery (`shipped_i`, PLAN section 2.7.3) and by the holding loss, and nothing else may
    create or destroy own-good units. Hidden reserves (PLAN section 4.1 row 7) and hoarding (row
    5) are consequences of this single sink, never rules of their own (CONTRACT rule 7).
  * Holding loss `h = cfg.supply.holding_loss` (Phase 1: 0.02) is charged once per period at the
    REPORT step, on the stock carried *in*, before this period's output is added:
    `S_i <- (1 - h) * S_i + y_i` (PLAN section 2.8). The order matters and is tested. The update
    itself belongs to `process_reports` in `gosplan/env/reporting.py` (WO-007), not to this
    module; this module owns only the field it writes into.
  * Cap `S_max = cfg.tech.inventory_cap_mult * cap_i`, i.e. `3 * cap_i` at the Phase-1 defaults.
    Stock above the cap is lost, and the lost amount (the "cap overflow") is logged per enterprise
    per period so that it stays visible in the conservation identity instead of silently
    vanishing. The cap is applied by `process_reports` immediately after the line above (WO-007).
  * Input stocks `inv_inputs` (`X_ij`) carry **no** holding loss in Phase 1:
    `cfg.supply.input_holding_loss = 0.0`, so holding inputs has no direct carrying cost. That is
    a design decision, not an oversight - it is what leaves the hoarding phenomenon free to be
    held out and measured in Phase 2, where the locked value is 0.01 (PLAN sections 2.11, 4.2).
  * Test T-U1 (`tests/unit/test_conservation.py`, PLAN section 11) is the identity these rules
    have to satisfy, per good and to 1e-9:

        sum y + sum S_prev = sum inputs consumed / a + sum consumer + sum S_next
                             + holding loss + cap overflow

CONTRACT rule 9 applies to every module that mutates this record: all randomness comes from
`gosplan.rng.draw(seed_env, purpose, *indices)`, keyed from `State.seed_env`; `State.seed_policy`
is a separate stream that the environment itself never draws from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # type-only: see the binding note in the module docstring
    from gosplan.config import EnvConfig, Phase
    from gosplan.metrics.ledger import StepRecord

Array = np.ndarray
"""Alias for every numeric array in this module (PLAN section 10), identical to `spec.Array`.
The Phase-2 JAX port substitutes its own array type behind the same name, so no function may rely
on a numpy-only method in a signature."""

INITIAL_CAPACITY: float = 1.0
"""`cap_i = Kap_i = 1`, the Phase-1 capacity and capital of every enterprise (PLAN section 2.1).
It is a normalisation rather than a configuration field - `cfg.supply.productivity` is fixed at
1.0 against it, which is what makes the TECH row `T_0 = 0.6 * A * cap` read as `T_0 = 0.6` - so it
is named here instead of appearing as a bare literal in `initial_state`, `initial_targets` and in
the `S_max` cap of PLAN section 2.11. Capital moves only in Phase 2, when `cfg.supply.capital_dep`
and the investment action are switched on (WO-021 onward)."""


@dataclass
class State:
    """The full environment state (PLAN section 2.2), struct-of-arrays with leading dimension `N`.

    Mutable by design: `gosplan/env/step.py` (WO-009) threads one `State` through the period
    schedule of PLAN section 2.5. Field-for-field identical to `spec.State`; see the module
    docstring for the enforcement and for the inventory rules of PLAN section 2.11.
    """

    target: Array  # (N,) T_i, target in units of own good (section 2.1)
    capital: Array  # (N,) Kap_i; Phase 1 fixed at 1.0 (section 2.1)
    inv_output: Array  # (N,) S_i, own-good stock on hand; the only place unreported output goes
    inv_inputs: Array  # (N, J) X_ij, input stocks held by i (section 2.6)
    cum_output: Array  # (N,) true output accumulated so far this period (reset each period)
    cum_cost: Array  # (N,) effort cost accumulated so far this period (reset each period)
    quality_acc: Array  # (N,) accumulator for the period-average quality qbar_i; Phase 1 inert

    last_report_ratio: Array  # (N,) rho_i = R_i / T_i as reported at the last REPORT step
    last_report: Array  # (N,) R_i in units, retained because the ratchet moves T after REPORT
    last_audited: Array  # (N,) bool, audit selection at the last AUDIT step (section 2.7.4)
    last_penalty: Array  # (N,) penalty charged at the last AUDIT step (section 2.8)
    last_fill: Array  # (N,) fraction of own last claim actually shipped (section 2.7.3)
    request: Array  # (N, J) q_ij, input requests from the last REPORT step (section 2.3)
    pending_invest: Array  # (N, L) output diverted to capital, maturing after invest_lag periods

    t_period: int  # plan period index t, from 0
    k_step: int  # production step index k within the period, 0 .. M-1; M at the REPORT step
    phase: Phase  # "produce" or "report" (section 2.5)
    plan_prices: Array  # (J,) p_j, plan prices (section 2.10)
    planner_io: Array  # (J, J) the planner's possibly stale copy of `a` (sections 2.2, 2.6)
    consumer_delivery: Array  # (J,) consumer_j received by the final-demand sink this period
    alive: bool  # episode has not yet terminated (section 2.12)
    seed_env: int  # root environment seed; every draw is keyed from it (section 2.15)
    seed_policy: int  # root policy seed, kept separate from seed_env (CONTRACT rule 9)


@dataclass
class EnterpriseAction:
    """One joint action for all `N` enterprises (PLAN section 2.3).

    The dimension set is fixed across phases; configuration flags decide which dimensions the
    environment reads, and the PPO adapter builds heads only for `active_action_dims(cfg)` (PLAN
    section 6.1, WO-017). Inactive dimensions are ignored by the environment rather than rejected,
    so a Phase-1 policy and a Phase-2 policy share one action type. Dimensions irrelevant to the
    current phase are likewise ignored: `effort`, `quality` and `invest` are read only at PRODUCE
    steps, `report_ratio` and `input_request` only at the REPORT step.

    Bounds are those of `action_spec(cfg)` (WO-009). Field-for-field identical to
    `spec.EnterpriseAction`; declared here because `spec/` is not importable and because
    `gosplan/env/production.py` and `gosplan/env/reporting.py` both need the type. If WO-009's
    `step.py` or a later spec revision gives this record another home, the duplicate must be
    removed and the move recorded in `spec/CHANGELOG.md`.
    """

    effort: Array  # (N,) e_ik in [0, 1]; active in Phase 1; read at PRODUCE steps
    quality: Array  # (N,) q_ik in [0, 1]; Phase 2
    invest: Array  # (N,) v_ik in [0, 1], fraction of step output diverted to capital; Phase 2
    report_ratio: Array  # (N,) in [0, rho_max]; active in Phase 1; read only at the REPORT step
    input_request: Array  # (N, J) q_ij in [0, r_max * need_ij]; logged in P1, inert at eta_q=0
    trade_offer: Array  # (N, J) in [-1, 1]; positive = offer, negative = want; Phase 2


@dataclass
class StepInfo:
    """Per-agent-step diagnostic payload: true quantities for the ledger, never for agents.

    Carries the `StepRecord`s produced by one agent-step - one per enterprise, in enterprise-index
    order - plus the period-level scalars. CONTRACT rule 6 and the WO-009 card make the boundary
    explicit: `StepInfo` is written by the environment and read by `gosplan/metrics/ledger.py` and
    by lead-run experiments; no agent, no policy and no reward term may read it (the WO-010
    forbidden list is "any agent reading `StepInfo`"), which is why `GosplanEnv.step` returns it
    beside the observation rather than inside it.

    The three period-level metrics are filled at the point of the schedule that defines them (PLAN
    sections 2.5, 2.9.3) and are zero at every other agent-step: `val_measured` and `val_true` at
    the REPORT step, `welfare` after DELIVER, when `consumer` is known. Field-for-field identical
    to `spec.StepInfo`; declared here because `gosplan/env/step.py` and `gosplan/env/env.py` both
    bind it to this module. Owning WO: **WO-009**.
    """

    records: tuple[StepRecord, ...]  # one per enterprise, in enterprise-index order
    t_period: int
    k_step: int
    phase: Phase
    val_measured: float  # section 2.9.3; period-level, filled at the REPORT step
    val_true: float  # section 2.9.3; period-level, filled at the REPORT step
    welfare: float  # section 2.9.3; period-level, filled after DELIVER
    consumer: Array  # (J,) final-demand receipts this period
    flags: tuple[str, ...]  # run-level flags raised this step, e.g. "BOUND_BINDING" (rule 8)
    terminated: bool  # geometric termination fired this period (section 2.12)


def initial_targets(cfg: EnvConfig) -> Array:
    """Return the opening targets `T_0` of every enterprise (PLAN sections 2.7.1, 3).

    Takes: `cfg`. Returns: `T_0` `(N,)`, strictly positive.

    Formula (PLAN section 3, TECH row `initial_target_frac`):

        T_0_i = cfg.tech.initial_target_frac * A_{s(i)} * cap_i

    with `A_{s(i)} = cfg.supply.productivity[cfg.supply.sector_of[i]]` and `cap_i =
    INITIAL_CAPACITY`. At the Phase-1 defaults this is `0.6 * 1.0 * 1.0 = 0.6` for every `i`:
    feasible at an effort of about 0.6, which is the point of the 0.6.

    This function is the single definition of `T_0`, and four callers depend on that:
    `initial_state` seeds `State.target` with it; `GosplanEnv.__init__` caches it (PLAN section
    2.5); `gosplan/env/obs.py` divides by it for observation field 2, `log(T_i / T_0)` (PLAN
    section 2.4); and `update_targets` floors the ratchet at `T_min = cfg.tech.target_floor_frac *
    T_0` (PLAN section 2.7.1). `T_0` never changes within an episode - only `State.target` moves.

    Binds: `tests/unit/test_obs.py` (observation field 2 is 0 at reset), test T-U4 (the target
    floor is respected). Owning WO: **WO-009**.
    """
    raise NotImplementedError("PLAN sections 2.7.1, 3 - implemented in WO-009")


def initial_state(cfg: EnvConfig) -> State:
    """Build the opening state of an episode (PLAN sections 2.1, 2.2, 3).

    Takes: `cfg`, already validated. Returns: a fresh `State` with exactly these contents (PLAN
    section 2.2 and the `GosplanEnv.reset` contract of PLAN section 10):

        target             initial_targets(cfg)                              (N,)
        capital            INITIAL_CAPACITY everywhere                       (N,)
        inv_output         zeros                                             (N,)
        inv_inputs         zeros                                             (N, J)
        cum_output         zeros                                             (N,)
        cum_cost           zeros                                             (N,)
        quality_acc        zeros                                             (N,)
        last_report_ratio  zeros                                             (N,)
        last_report        zeros                                             (N,)
        last_audited       False                                             (N,) bool
        last_penalty       zeros                                             (N,)
        last_fill          ones - nothing has been claimed yet, and PLAN section 2.7.3 gives
                           `fill = 1` when the claim is zero, so 1 is the consistent opening value
        request            zeros                                             (N, J)
        pending_invest     zeros                                             (N, L)
        t_period           0
        k_step             0
        phase              "produce"
        plan_prices        initial_prices(cfg) from `gosplan/env/prices.py`  (J,)
        planner_io         `cfg.supply.io_matrix` as a float array - at `t = 0` the planner's copy
                           of `a` is exact; it goes stale only when `tech_drift_sigma > 0` moves
                           the true `a` in Phase 2 (PLAN section 2.6)        (J, J)
        consumer_delivery  zeros                                             (J,)
        alive              True
        seed_env           cfg.tech.seed_env
        seed_policy        cfg.tech.seed_policy

    Every array is freshly allocated and owned by this state: no view of a configuration tuple may
    escape into it, or a later in-place update would mutate the configuration.

    Seeds: the two fields default from `cfg.tech`, so `initial_state(cfg)` alone is a complete and
    reproducible state. `GosplanEnv.reset(seed_env, seed_policy)` overwrites them with its
    arguments, which is how common random numbers across arms are obtained at a fixed `cfg` (PLAN
    sections 2.15, 4.3).

    Binds: `tests/unit/test_env_api.py` (reset returns this state; shapes and dtypes as declared)
    and `tests/golden/*` (T-B7 - the opening state must match `ref/ref_step.py` to 1e-9). Owning
    WO: **WO-009**.
    """
    raise NotImplementedError("PLAN section 2.2 - implemented in WO-009")


def reset_period_accumulators(state: State) -> State:
    """Zero the per-period accumulators at the start of a plan period (PLAN sections 2.2, 2.5).

    Takes: `state` at a period boundary. Returns: the same state with `cum_output`, `cum_cost`,
    `quality_acc` and `consumer_delivery` set to zero and every other field untouched.

    Called once per period by the step machine, at the DELIVER boundary of PLAN section 2.5: after
    `process_reports` has folded `cum_output` into `inv_output`, and after the REPORT-step reward,
    audit, ratchet and termination draw have consumed the period's numbers, but before the first
    PRODUCE step of the new period. Calling it earlier would erase the cumulative output the
    REPORT step reports on; calling it later would charge one period's costs to the next.

    The `last_*` fields are deliberately **not** reset: they are the memory that observation
    fields 7-10 of PLAN section 2.4 expose and that the ratchet consumes, and they are overwritten
    in place at the next REPORT and AUDIT steps.

    Binds: test T-U1 (`tests/unit/test_conservation.py`) - the conservation identity is stated per
    period, so an accumulator not zeroed at exactly this boundary breaks it. Owning WO: **WO-009**.
    """
    raise NotImplementedError("PLAN sections 2.2, 2.5 - implemented in WO-009")


def advance_phase(state: State, cfg: EnvConfig) -> State:
    """Advance the period and step counters by one agent-step (PLAN section 2.5).

    Takes: `state` after an agent-step has been executed, and `cfg`. Returns: the same state with
    `t_period`, `k_step` and `phase` moved on. Pure bookkeeping: it must not touch a physical
    quantity and must not implement any part of the schedule's economics.

    Rule, with `M = cfg.incentive.steps_per_period` (PLAN section 2.5 - `M` PRODUCE steps then one
    REPORT step, so agents act `M + 1` times per period):

        after a PRODUCE step at k < M - 1    k_step -> k + 1,   phase stays "produce"
        after the PRODUCE step at k = M - 1  k_step -> M,       phase -> "report"
        after the REPORT step                t_period -> t + 1, k_step -> 0, phase -> "produce"

    `k_step == M` is therefore the REPORT step's own index, which is why the yield key of PLAN
    section 2.6 only ever sees `k` in `0 .. M - 1`. `GosplanEnv.phase()` reports the phase the next
    call to `step` will execute, i.e. `state.phase` after this function has run.

    Binds: `tests/unit/test_env_api.py` (an episode of `P` periods produces exactly `P * (M + 1)`
    agent-steps, and the phase sequence within a period is `M` times "produce" then "report") and
    `tests/golden/*` (T-B7). Owning WO: **WO-009**.
    """
    raise NotImplementedError("PLAN section 2.5 - implemented in WO-009")
