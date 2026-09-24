"""Period schedule of PLAN section 2.5 as an explicit state machine; horizon of section 2.12.

Realises: PLAN section 2.5 (typed period schedule), PLAN section 2.12 (geometric horizon and
termination), and the `env/step.py` row of the PLAN section 8 layout ("LEAD: period schedule
section 2.5, assembles modules"). Owning work order: **WO-009** (Step function and env wrapper).

**This unit is LEAD-owned.** PLAN sections 12.3 and 1.3 (finding F14) mark the step function, the
JAX port and the PPO adapter as the three units the lead writes itself rather than delegating: the
step function is where the information invariants of CONTRACT rules 5, 6 and 9 are either preserved
or quietly broken, and it is reviewed line by line against CONTRACT rule 7. An implementer session
that finds itself editing this file has taken the wrong card.

What this module is. The eight stages of one plan period, in order, as data (`PeriodStage`,
`PERIOD_SCHEDULE`) plus one function per stage. Each stage function is a thin assembler: it calls
into the module that owns the arithmetic and threads the `State` through. **No formula of PLAN
sections 2.6-2.11 is restated here** - production lives in `gosplan/env/production.py` (WO-005),
the planner rules and physical delivery in `gosplan/env/planner.py` (WO-006), reporting, audit,
penalty, bonus, reward, val and welfare in `gosplan/env/reporting.py` and `gosplan/env/reward.py`
(WO-007), the observation in `gosplan/env/obs.py` (WO-008), and the two Phase-2 sketches in
`gosplan/env/trade.py` (WO-024) and `gosplan/env/ministry.py` (WO-025). Duplicating a formula here
would create a second place for the golden files (T-B7) to disagree with `ref/ref_step.py`.

One period (PLAN section 2.5, verbatim ordering):

    0. DELIVER      planner allocates from last period's claims (2.7.2 / 2.7.3); the consumer sink
                    receives; X updated
    1. [P2] TRADE   bilateral matching (2.13)
    2. PRODUCE x M  agent chooses effort/quality/invest; yield realised; obs shows cumulative y
    3. REPORT       S <- (1 - h) * S + y; agent observes S and y exactly; chooses report_ratio and
                    input_request (2.8)
    4. AUDIT        audit selection; measurement; penalty (2.8)
    5. REWARD       bonus - penalty (+ trade surplus) delivered; val and welfare logged (2.9)
    6. TARGET       ratchet + growth directive (2.7.1)
    7. TERMINATE?   geometric (2.12)

Agent-steps versus stages. The agent acts `M + 1` times per period (`M = cfg.incentive.
steps_per_period`): `M` PRODUCE steps and one REPORT step. One call to `advance` therefore executes
a *contiguous slice* of the eight stages, not one stage: the first PRODUCE step of a period is
preceded by DELIVER and TRADE, and the REPORT step is followed by AUDIT, REWARD, TARGET and
TERMINATE with no further agent input. `stages_for_step` is that mapping and is the state machine's
transition function.

Horizon (PLAN section 2.12). Under `cfg.tech.horizon_mode = "geometric"` (Phase 1) the episode runs
`P_min = cfg.tech.min_periods` periods (Phase 1: 4) unconditionally, then continues with probability
`psi = cfg.incentive.tenure` (Phase 1: 0.9) at the end of each period, with a hard cap
`P_max = cfg.tech.max_periods` (Phase 1: 20). Expected length is about 10 periods, i.e. about 50
agent-steps. **The agent never observes periods remaining** - `obs_spec` (PLAN section 2.4) has no
such field and no field that proxies it - so there is no end-game, which is the point of finding F4.
`horizon_mode = "fixed"` exists to study the known-rotation end-game as a separate question and is
never the Phase-1 setting.

`psi` is *tenure*: an economic parameter (managerial rotation), classified in the INC arm of PLAN
section 3, and deliberately distinct from the technical PPO discount `gamma = 0.99`, which is a TECH
constant that lives with the adapter (WO-017) and which the environment never reads. They multiply
only inside the single-enterprise DP of PLAN section 5, whose Bellman operator discounts at
`psi * gamma`. Conflating the two is exactly the confound finding F2 records; the separation is what
lets a sweep over tenure be an economic treatment rather than an optimiser setting.

Constraints this module inherits. CONTRACT rule 9: every stochastic term goes through
`gosplan.rng.draw(seed_env, purpose, *indices)` - the termination draw uses purpose `terminate`, and
there is no `numpy.random` call anywhere under `gosplan/env/`. CONTRACT rule 6: `StepInfo` and the
`StepRecord`s inside it carry true quantities and are written for the ledger only; nothing on an
agent's path reads them. CONTRACT rule 7: not one line here implements bunching, padding, storming,
hoarding, shaving or trade - the stages call rules, and the phenomena are consequences of those
rules or they are not results at all.

Binding tests (PLAN section 11, WO-009 must-pass list): `tests/golden/*` (T-B7, agreement with
`ref/ref_step.py` to 1e-9 on seeded trajectories), `tests/unit/test_conservation.py` (T-U1, the
per-period, per-good conservation identity to 1e-9), `tests/unit/test_env_api.py`, and
`tests/behavioural/test_termination.py` (T-B9, empirical continuation equals `psi` and no
observation field correlates with periods remaining).

Cross-module bindings. `State`, `EnterpriseAction`, `StepInfo` and the literal alias `Phase` are the
runtime declarations of `gosplan/env/state.py`, `PlannerView` that of `gosplan/env/planner.py`, and
the configuration dataclasses those of `gosplan/config.py`. Each must stay field-for-field and
name-for-name identical to `spec/spec.py`, which is not importable as a package;
`tests/unit/test_env_api.py` enforces the match. Those imports are type-only here so this skeleton
imports cleanly before the modules that own them land; the implementer promotes the ones it calls at
runtime.
"""

from __future__ import annotations

import functools
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from gosplan.env.planner import (
    allocate,
    deliver,
    make_planner_view,
    select_audits,
    update_targets,
)
from gosplan.env.production import produce_step
from gosplan.env.reporting import audit_and_penalise, process_reports
from gosplan.env.reward import enterprise_reward, val_measured, val_true, welfare_true
from gosplan.env.state import StepInfo, advance_phase, reset_period_accumulators
from gosplan.rng import draw

if TYPE_CHECKING:  # type-only: see the cross-module bindings note in the module docstring
    from gosplan.config import EnvConfig
    from gosplan.env.planner import PlannerView
    from gosplan.env.state import EnterpriseAction, Phase, State

Array = np.ndarray
"""Alias for every numeric array in this module (PLAN section 10). The Phase-2 JAX port (WO-029)
substitutes its own array type behind the same name, so no signature here may rely on a
numpy-only method."""


class PeriodStage(Enum):
    """The eight stages of one plan period (PLAN section 2.5), in execution order.

    Member values are the stage's position in `PERIOD_SCHEDULE`, so `stage.value` orders the
    schedule and `PERIOD_SCHEDULE[stage.value] is stage`. The enum exists so the schedule is a typed
    object the tests and the ledger can name, rather than an ordering implied by the body of a step
    function: PLAN section 2.5 is titled "period schedule (typed)" for that reason, and finding F10
    records the typed schedule with a separate REPORT step as the fix for the original sequencing
    ambiguity.

    Owning WO: **WO-009**.
    """

    DELIVER = 0
    """Stage 0. Last period's claims become promises and then physical goods (PLAN sections 2.7.2,
    2.7.3). Opens the period; at `t = 0` there are no claims and it is a no-op."""

    TRADE = 1
    """Stage 1, **Phase 2**. Bilateral horizontal matching among visible counterparties (PLAN
    section 2.13). Inert in Phase 1, where `information.horizontal_visibility = 0`."""

    PRODUCE = 2
    """Stage 2, repeated `M = cfg.incentive.steps_per_period` times. The agent's effort (and, in
    Phase 2, quality and investment) becomes output through the coverage aggregator and the yield
    shock (PLAN section 2.6)."""

    REPORT = 3
    """Stage 3, once per period. The period's books close (`S <- (1 - h) * S + y`) and the
    enterprise's claim `R_i` is recorded (PLAN section 2.8). The agent's second and last acting
    phase."""

    AUDIT = 4
    """Stage 4. Audit selection (PLAN section 2.7.4), the noisy stock measurement and the penalty
    (PLAN section 2.8). Never observable to any agent before it happens."""

    REWARD = 5
    """Stage 5. The only quantity any learner receives is delivered (PLAN section 2.9.1, CONTRACT
    rule 4); `val_measured`, `val_true` and `welfare_true` are computed for the ledger alone
    (CONTRACT rule 6)."""

    TARGET = 6
    """Stage 6. The ratchet and the growth directive move next period's targets (PLAN section
    2.7.1). It runs *after* REWARD, so this period's bonus is judged against the period's own
    target."""

    TERMINATE = 7
    """Stage 7. The geometric continuation draw of PLAN section 2.12, purpose `terminate`. The
    outcome reaches the learner only as `done`; it is in no observation field."""


PERIOD_SCHEDULE: tuple[PeriodStage, ...] = (
    PeriodStage.DELIVER,
    PeriodStage.TRADE,
    PeriodStage.PRODUCE,
    PeriodStage.REPORT,
    PeriodStage.AUDIT,
    PeriodStage.REWARD,
    PeriodStage.TARGET,
    PeriodStage.TERMINATE,
)
"""The period schedule of PLAN section 2.5 as ordered data - the single source of stage order for
the environment, the reference implementation and the tests. PRODUCE appears once here and is
repeated `cfg.incentive.steps_per_period` times at execution; `stages_for_step` expresses that
repetition. Reordering this tuple changes the dynamics and therefore requires regenerated golden
files (WO-013)."""

AGENT_STAGES: tuple[PeriodStage, ...] = (PeriodStage.PRODUCE, PeriodStage.REPORT)
"""The two stages at which an agent acts, matching the `Phase` literal `("produce", "report")` of
PLAN section 2.5: `M` PRODUCE steps then one REPORT step, so `M + 1` actions per period. Every other
stage is environment-internal and takes no action. Action dimensions not relevant to the current
stage are ignored by the environment rather than rejected (PLAN section 2.3), and the environment
never trusts an agent to have masked them itself."""


def stages_for_step(state: State, cfg: EnvConfig) -> tuple[PeriodStage, ...]:
    """Return the contiguous slice of `PERIOD_SCHEDULE` that the next `advance` call will execute.

    Takes: `state`, whose `k_step`, `phase` and `t_period` fields carry the machine's position, and
    `cfg`. Returns: a tuple of `PeriodStage` in execution order, a contiguous slice of
    `PERIOD_SCHEDULE`. With `M = cfg.incentive.steps_per_period` the mapping is exactly:

        k_step == 0, phase "produce"   (DELIVER, TRADE, PRODUCE)
        0 < k_step < M, phase "produce"  (PRODUCE,)
        k_step == M, phase "report"    (REPORT, AUDIT, REWARD, TARGET, TERMINATE)

    DELIVER opens the period, so it runs at the head of the period's first PRODUCE step and consumes
    the claims recorded at the *previous* period's REPORT step (PLAN section 2.5; the `GosplanEnv.
    step` note "the next period opens with DELIVER"). TRADE sits between DELIVER and the first
    PRODUCE step because it reallocates the inputs DELIVER has just placed in `X`.

    This is the state machine's transition function: `advance` executes the returned stages in the
    returned order and nothing else, which is what makes the schedule auditable rather than implied
    by control flow.

    Binds: `tests/unit/test_env_api.py` - the three cases above; the union over one period of the
    returned tuples equals `PERIOD_SCHEDULE` with PRODUCE repeated `M` times, in order. Owning WO:
    **WO-009**.
    """
    m = cfg.incentive.steps_per_period
    k = state.k_step
    if state.phase == "report":
        if k != m:
            raise ValueError(f"stages_for_step: REPORT phase at k_step={k}, expected {m}")
        return PERIOD_SCHEDULE[PeriodStage.REPORT.value :]
    if k == 0:
        return PERIOD_SCHEDULE[: PeriodStage.PRODUCE.value + 1]
    if 0 < k < m:
        return (PeriodStage.PRODUCE,)
    raise ValueError(f"stages_for_step: PRODUCE phase at k_step={k}, outside 0..{m - 1}")


def stage_deliver(state: State, cfg: EnvConfig) -> tuple[State, Array, Array, Array, Array]:
    """Stage 0 DELIVER: last period's claims become promises and then physical goods.

    Takes: `state` at the head of period `t`, and `cfg`. Returns:
    `(state, alloc, deliv, fill, consumer)` - the updated state; `alloc` (N, J), the promised
    quantities; `deliv` (N, J), physical receipts; `fill` (N,), the per-seller fill ratio; and
    `consumer` (J,), the final-demand sink's receipts.

    Delegates, in this order and with no arithmetic of its own, to `gosplan/env/planner.py`
    (WO-006):

        view  = make_planner_view(state, cfg)   # the single State -> planner boundary, rule 5
        alloc = allocate(view, cfg)             # promises in claimed units, PLAN section 2.7.2
        state, deliv, fill, consumer = deliver(state, alloc, cfg)   # PLAN section 2.7.3

    At `t = 0` there is nothing to allocate: `last_report` is zero, so `avail_j` is zero, `alloc`
    and `deliv` are zero, `fill` is 1 by the `claimed_i == 0` convention, and the period opens with
    `X = 0`. That branch must exist explicitly rather than falling out of a division.

    May write: `inv_inputs` (buyers' receipts), `inv_output` (sellers ship `min(S_i, claimed_i)`),
    `last_fill`, `consumer_delivery`. May NOT write: `target` (TARGET owns it), `cum_output` and
    `cum_cost` (PRODUCE owns them), `last_report_ratio` / `last_report` / `request` (REPORT),
    `last_audited` / `last_penalty` (AUDIT), `alive` (TERMINATE). May NOT read any true quantity on
    the planner's behalf except through the `PlannerView` it just built (CONTRACT rule 5).

    Padding to shortage and shaving to hidden reserves are consequences of the four formulas in
    `deliver`, never rules of their own (CONTRACT rule 7): a claim above stock lowers `poolfill` for
    the whole good and every downstream buyer's receipt with it; a claim below stock leaves the
    difference sitting in `S`.

    Binds: T-U1 (the delivered and shipped quantities are terms of the per-period conservation
    identity), T-B3 (`tests/behavioural/test_shortage_propagation.py`), T-B7 (golden parity).
    Owning WO: **WO-009**; the arithmetic is **WO-006**.
    """
    view = make_planner_view(state, cfg)
    alloc = allocate(view, cfg)
    state, deliv, fill, consumer = deliver(state, alloc, cfg)
    state.last_fill = np.asarray(fill, dtype=float)
    state.consumer_delivery = np.asarray(consumer, dtype=float)
    return state, np.asarray(alloc), np.asarray(deliv), state.last_fill, state.consumer_delivery


def stage_trade(
    state: State, action: EnterpriseAction, cfg: EnvConfig, t: int
) -> tuple[State, Array]:
    """Stage 1 TRADE: bilateral horizontal matching after DELIVER (**Phase 2**).

    Takes: `state` after DELIVER; `action`, of which only `trade_offer` (N, J) is read; `cfg`; and
    the plan period `t`, which keys the visibility draw. Returns: `(state, surplus)` with `surplus`
    (N,) in reward units before `reward_scale`.

    Delegates to `match_trades(state, action.trade_offer, cfg, t)` in `gosplan/env/trade.py`
    (WO-024), whose signature is frozen now and whose behaviour is frozen only at the Phase-2 spec
    revision (PLAN section 0, finding F14).

    Phase-1 branch: `information.horizontal_visibility = 0.0`, so no counterparty is visible, no
    match exists, `state` is returned unchanged and `surplus` is an all-zero (N,) array. The branch
    is present and exercised even in Phase 1 - the reward term is in CONTRACT rule 4's list, and a
    stage that only appears in Phase 2 is a stage the golden files never covered.

    May write: `inv_inputs` only - trade moves input stocks between enterprises, and (with the
    transaction cost `tau`) destroys a little of what it moves. May NOT write: `inv_output`,
    `target`, any `last_*` field, `cum_output` or `cum_cost`. The surplus it returns is computed by
    the environment from the input stocks before and after (PLAN section 2.13); no side of a trade
    reports its own gain.

    Binds: T-U1 (traded units and the `tau` loss are terms of the conservation identity), T-B1
    (`TruthfulMyopic` executes no trade under the Phase-1 configuration). Owning WO: **WO-009**; the
    matching rule is **WO-024**.
    """
    if cfg.information.horizontal_visibility == 0.0:
        return state, np.zeros(cfg.supply.n_enterprises)
    from gosplan.env.trade import match_trades

    return match_trades(state, action.trade_offer, cfg, t)


def stage_produce(
    state: State, action: EnterpriseAction, cfg: EnvConfig
) -> tuple[State, Array, Array]:
    """Stage 2 PRODUCE: one production step `k` for all `N` enterprises.

    Takes: `state` at the start of step `k`; `action`, of which `effort` (and, in Phase 2, `quality`
    and `invest`) is read; `cfg`. Returns: `(state, y, c)` - the updated state, output realised this
    step (N,), and cost incurred this step (N,).

    Delegates to `produce_step(state, action, cfg)` in `gosplan/env/production.py` (WO-005), which
    owns every formula of PLAN section 2.6: intended output, the per-good need, the CES coverage
    aggregator, the lognormal yield shock drawn with purpose `yield` at key
    `(seed_env, "yield", t, k, i)`, the investment diversion, the input consumption capped at stock,
    and the effort cost.

    May write: `cum_output`, `cum_cost`, `inv_inputs` (consumed against `y_tilde`, capped at stock),
    `quality_acc`, `pending_invest`, `k_step`. May NOT write: `inv_output` - this period's output
    reaches `S` only at REPORT, after the holding loss has been applied to the stock carried in
    (PLAN section 2.8, order matters) - nor `target`, nor any `last_*` field. May NOT read reports,
    targets or rewards at all (the WO-005 forbidden list).

    Binds: T-U1 (per-period conservation), T-U7 (the coverage aggregator), T-B7 (golden parity).
    Owning WO: **WO-009**; the arithmetic is **WO-005**.
    """
    return produce_step(state, action, cfg)


def stage_report(state: State, action: EnterpriseAction, cfg: EnvConfig) -> State:
    """Stage 3 REPORT: close the period's books and record the claim.

    Takes: `state` at the REPORT step, after the period's `M` PRODUCE steps; `action`, of which
    `report_ratio` (N,) and `input_request` (N, J) are read; `cfg`. Returns: the updated state.

    Delegates to `process_reports(state, action, cfg)` in `gosplan/env/reporting.py` (WO-007), which
    owns PLAN section 2.8: the holding loss applied to carried stock *before* this period's output
    is added, the claim `R_i = clip(rho_i, 0, rho_max) * T_i`, the `S_max` cap with its logged
    overflow, the request clipped to `r_max * need_ij`, and the `at_bound` flag of CONTRACT rule 8.

    May write: `inv_output`, `last_report_ratio`, `last_report`, `request`, `phase`, `k_step`. May
    NOT write: `target` - the ratchet moves `T` two stages later, and this period's bonus is judged
    against the target the period was set (PLAN section 2.5, stages 3 then 6) - nor `last_audited`,
    `last_penalty` or `alive`.

    At this step the agent has already observed `S_i` and `y_i` exactly (Phase 1,
    `self_obs_noise = 0`), so the report is a choice made under full knowledge of the truth. Whether
    a claim exceeds, equals or falls short of stock is the agent's business and is measured, never
    prescribed (CONTRACT rule 7).

    Binds: T-U1, T-B7, T-B8 (the `BOUND_BINDING` flag when over 1% of reports sit at `rho_max` -
    bounds are results, and this one is never silently moved). Owning WO: **WO-009**; the arithmetic
    is **WO-007**.
    """
    return process_reports(state, action, cfg)


def stage_audit(
    state: State, view: PlannerView, cfg: EnvConfig, t: int
) -> tuple[State, Array, Array]:
    """Stage 4 AUDIT: select, measure and penalise.

    Takes: `state` after REPORT; `view`, the `PlannerView` rebuilt from this period's reports
    (`select_audits` is a planner rule and takes the view, never the state - CONTRACT rule 5);
    `cfg`; and the plan period `t`, which keys both draws. Returns: `(state, audited, penalty)` with
    `audited` (N,) bool and `penalty` (N,) zero wherever `audited` is False.

    Delegates to `select_audits(view, cfg, t)` in `gosplan/env/planner.py` (WO-006), whose Phase-1
    branch draws `Bernoulli(audit_rate)` at key `(seed_env, "audit", t, i)`, and then to
    `audit_and_penalise(state, audited, cfg, t)` in `gosplan/env/reporting.py` (WO-007), which draws
    the measurement noise at key `(seed_env, "auditnoise", t, i)` and applies the penalty of PLAN
    section 2.8 in ratio units (finding F9).

    May write: `last_audited`, `last_penalty`. May NOT write anything else - in particular the audit
    neither confiscates nor corrects stock: `inv_output` is untouched, which is what makes the
    comparison of a claim against *stock on hand* the mechanism behind hidden reserves rather than
    an accounting adjustment.

    The selection is never observable to any agent before it happens (PLAN section 2.4): the current
    period's audit draw appears in no observation, and only `last_audited` - the previous period's
    outcome - is in the observation vector at index 8.

    Binds: T-U8 (`positive_part` gives exactly 0 for any under-report; `audited = False` gives 0),
    T-B7 (golden parity). Owning WO: **WO-009**; the arithmetic is **WO-006** and **WO-007**.
    """
    audited = np.asarray(select_audits(view, cfg, t), dtype=bool)
    penalty = np.asarray(audit_and_penalise(state, audited, cfg, t), dtype=float)
    state.last_audited = audited
    state.last_penalty = penalty
    return state, audited, penalty


def stage_reward(
    state: State,
    cfg: EnvConfig,
    phase: Phase,
    cost: Array | None,
    penalty: Array | None,
    trade_surplus: Array | None,
) -> Array:
    """Stage 5 REWARD: deliver the only quantity any learner receives.

    Takes: `state`; `cfg`; the current `phase`; and the three period quantities - `cost` (N,) at a
    PRODUCE step, `penalty` (N,) and `trade_surplus` (N,) at the REPORT step - each `None` in the
    phase where it does not apply. The argument names and order match `enterprise_reward` in
    `spec/spec.py` exactly. Returns: `r` (N,).

    Delegates to `enterprise_reward(state, cfg, phase, cost, penalty, trade_surplus)` in
    `gosplan/env/reward.py` (WO-007), which owns the formula of PLAN section 2.9.1:

        PRODUCE step k:   r_ik = - scale * c_ik
        REPORT step:      r_i  =   scale * ( B(rho_i) - penalty_i + trade_surplus_i )
        scale             = reward_scale(cfg)      # analytic, per configuration, never running

    These are the only terms (CONTRACT rule 4): no per-step shaping, no auxiliary reward, no
    curiosity term, no potential-based term, and no running reward normalisation - running
    statistics change the effective reward over training and, with heavy-tailed penalties, shrink
    the notch in normalised units. Per-batch advantage normalisation inside PPO is permitted and
    belongs to WO-017, not here. Effort cost is a real cost paid when it is incurred, not shaping.

    The period-level metrics `val_measured`, `val_true` and `welfare_true` (PLAN section 2.9.3) are
    computed by `advance` alongside this stage and placed in `StepInfo` for the ledger. They are
    never added to, subtracted from or used to rescale the array returned here, and no agent reads
    them (CONTRACT rule 6, test T-B5).

    May write: nothing. This stage is a pure function of the state and the three quantities; the
    reward is returned, not stored on the state.

    Binds: T-B6 (the reward recomputed independently from the formula on random states must agree
    exactly), T-U2 (`reward_scale(cfg) * bonus(1.1, cfg) == 1`), T-B7. Owning WO: **WO-009**; the
    arithmetic is **WO-007**.
    """
    return np.asarray(enterprise_reward(state, cfg, phase, cost, penalty, trade_surplus))


def stage_target(state: State, view: PlannerView, cfg: EnvConfig) -> State:
    """Stage 6 TARGET: the ratchet and the growth directive move next period's targets.

    Takes: `state` after REWARD; `view`, the `PlannerView` this period's claims reached the planner
    through (CONTRACT rule 5 - the target rule sees claims, never stock or output); `cfg`. Returns:
    the updated state.

    Delegates to `update_targets(view, cfg)` in `gosplan/env/planner.py` (WO-006), which owns PLAN
    section 2.7.1: the fulfilment measure, the capped multiplicative step, the deadband, the growth
    directive `(1 + g)` and the floor at `target_floor_frac * T_0`.

    Order matters and is the reason this stage sits after REWARD: the bonus of stage 5 is judged
    against the target the period was set, and only then does the ratchet move the target the next
    period will be judged against. Swapping stages 5 and 6 would make every period's bonus depend on
    its own ratchet step, which is a different economy.

    May write: `target` only. May NOT write `last_report*` (REPORT recorded them and the ratchet
    needs them intact), `inv_output`, `inv_inputs`, or `alive`.

    Binds: T-U4 (fixed point at `rho = 1, g = 0`; step bounded by `c_up` / `c_dn`; floor respected;
    deadband inert outside `|rho - 1| <= delta`) and T-B2 (`Padder` at `g = 0` keeps `T` constant;
    at `g > 0` it grows at exactly `(1 + g)`). Owning WO: **WO-009**; the arithmetic is **WO-006**.
    """
    state.target = np.asarray(update_targets(view, cfg), dtype=float)
    return state


def stage_terminate(state: State, cfg: EnvConfig) -> tuple[State, bool]:
    """Stage 7 TERMINATE: the geometric continuation draw (PLAN section 2.12).

    Takes: `state` after TARGET, whose `t_period` keys the draw; `cfg`. Returns: `(state, done)`,
    where `done` is a single episode-level Python `bool` - termination is one global draw for the
    episode, not one per enterprise, so every enterprise's episode ends together.

    Rule (PLAN section 2.12, verbatim), under `cfg.tech.horizon_mode = "geometric"`:

        t + 1 <  min_periods   ->  continue unconditionally
        t + 1 >= max_periods   ->  terminate (hard cap)
        otherwise              ->  continue with probability psi = cfg.incentive.tenure,
                                   drawn Bernoulli(psi) at key (seed_env, "terminate", t)

    Under `horizon_mode = "fixed"` the episode ends exactly at `max_periods` and the draw is not
    taken; that mode exists to study the known-rotation end-game as a separate question and is never
    the Phase-1 setting.

    Phase-1 values are `min_periods = 4`, `tenure = 0.9`, `max_periods = 20`, so an episode lasts
    about 10 periods, i.e. about 50 agent-steps. `psi` is an economic parameter (managerial tenure,
    INC arm), distinct from the PPO discount `gamma = 0.99`, which the environment never reads; the
    two multiply only in the DP Bellman operator of PLAN section 5.

    The draw goes through `gosplan.rng.draw` with purpose `terminate` (CONTRACT rule 9), so it is
    order-independent and shared across arms whenever `seed_env` is.

    May write: `alive`, `t_period`, `k_step`, `phase` - the episode flag and the reset of the period
    counters. May NOT write any economic quantity.

    **The agent never observes periods remaining.** Nothing about this draw enters `obs_spec` (PLAN
    section 2.4), not as a countdown, not as `t / P_max`, and not as any quantity monotone in the
    period index; the outcome reaches the learner only as the `done` flag of `GosplanEnv.step`. That
    is what removes the end-game (finding F4).

    Binds: T-B9 in `tests/behavioural/test_termination.py` - the empirical continuation frequency
    equals `tenure` after `min_periods`, the cap at `max_periods` binds, and the regression
    coefficient of every observation field on periods remaining is about 0. Owning WO: **WO-009**.
    """
    completed = state.t_period + 1
    if completed >= cfg.tech.max_periods:
        done = True
    elif cfg.tech.horizon_mode == "fixed" or completed < cfg.tech.min_periods:
        done = False
    else:
        cont = draw(
            state.seed_env,
            "terminate",
            state.t_period,
            shape=(1,),
            dist="bernoulli",
            p=cfg.incentive.tenure,
        )
        done = not bool(cont[0])
    state.alive = not done
    return state, done


def advance(
    state: State, action: EnterpriseAction, cfg: EnvConfig, *, records: bool = True
) -> tuple[State, Array, bool, StepInfo]:
    """Execute one agent-step: the stages of `stages_for_step`, in order, and nothing else.

    Takes: `state`; `action`, a joint `EnterpriseAction` of which only the dimensions in
    `active_action_dims(cfg)` that the current stage reads are used - the rest are ignored rather
    than rejected (PLAN section 2.3); `cfg`. Returns: `(state, reward, done, info)` -

        state    the state after the executed stages, with `k_step`, `phase` and `t_period`
                 advanced to the machine's next position
        reward   (N,), per enterprise, from `stage_reward`; `-scale * c_ik` at a PRODUCE step and
                 `scale * (B(rho) - penalty + trade_surplus)` at the REPORT step
        done     the episode-level geometric termination flag from `stage_terminate`; False at every
                 PRODUCE step, since termination is decided once per period at stage 7
        info     a `StepInfo` carrying one `StepRecord` per enterprise plus the period-level
                 `val_measured`, `val_true`, `welfare` and `consumer` - true quantities, for the
                 ledger only (CONTRACT rule 6)

    This is the functional core `GosplanEnv.step` wraps: `advance` threads the state, and
    `gosplan/env/env.py` adds the observation (built by `gosplan/env/obs.py`), the ledger hookup and
    the episode bookkeeping. Keeping the two apart is what lets the golden harness drive the
    schedule without an env object and lets `ref/ref_step.py` be compared stage by stage.

    Sequencing to preserve exactly (PLAN section 2.5): at the period's first PRODUCE step, DELIVER
    then TRADE run before production; at the REPORT step, REPORT, AUDIT, REWARD, TARGET, TERMINATE
    run in that order with no further agent input. The `PlannerView` that AUDIT and TARGET consume
    is rebuilt from this period's reports after REPORT; the view DELIVER consumed at the head of the
    period was built from the previous period's.

    Constraints. Every stochastic term goes through `gosplan.rng.draw` (CONTRACT rule 9). Nothing
    here may leak a true quantity into the observation or the reward (CONTRACT rules 4, 6). No stage
    call may be reordered to make a phenomenon appear (CONTRACT rule 7).

    Binds: T-B7 (golden parity with `ref/ref_step.py` to 1e-9 on seeded trajectories), T-U1
    (conservation), `tests/unit/test_env_api.py`. Owning WO: **WO-009**.
    """
    n, j = cfg.supply.n_enterprises, cfg.supply.n_sectors
    stages = stages_for_step(state, cfg)
    t, k, phase = state.t_period, state.k_step, state.phase
    s_pre = np.array(state.inv_output, dtype=float)
    x_pre = np.array(state.inv_inputs, dtype=float)
    alloc = np.zeros((n, j))
    deliv = np.zeros((n, j))
    fill = np.zeros(n)
    shipped = np.zeros(n)
    output = np.zeros(n)
    cost = np.zeros(n)
    consumed = np.zeros((n, j))
    audited = np.zeros(n, dtype=bool)
    penalty = np.zeros(n)
    surplus = np.zeros(n)
    holding = np.zeros(n)
    overflow = np.zeros(n)
    reward = np.zeros(n)
    done = False
    view = None
    metrics = (0.0, 0.0, 0.0)
    judged_target = np.array(state.target, dtype=float)

    for stage in stages:
        if stage is PeriodStage.DELIVER:
            state = reset_period_accumulators(state)
            stock_before = np.array(state.inv_output, dtype=float)
            state, alloc, deliv, fill, _consumer = stage_deliver(state, cfg)
            shipped = stock_before - np.asarray(state.inv_output, dtype=float)
            s_pre = np.array(state.inv_output, dtype=float)
            x_pre = np.array(state.inv_inputs, dtype=float)
        elif stage is PeriodStage.TRADE:
            state, surplus = stage_trade(state, action, cfg, t)
            x_pre = np.array(state.inv_inputs, dtype=float)
        elif stage is PeriodStage.PRODUCE:
            state, output, cost = stage_produce(state, action, cfg)
            consumed = x_pre - np.asarray(state.inv_inputs, dtype=float)
            reward = stage_reward(state, cfg, "produce", cost, None, None)
        elif stage is PeriodStage.REPORT:
            output = np.array(state.cum_output, dtype=float)
            state = stage_report(state, action, cfg)
            # Bookkeeping for the T-U1 terms: the holding loss on the stock carried in, and the
            # cap overflow as the residual of the stock update (process_reports owns the rule).
            holding = cfg.supply.holding_loss * s_pre
            overflow = s_pre + output - holding - np.asarray(state.inv_output, dtype=float)
            view = make_planner_view(state, cfg)
        elif stage is PeriodStage.AUDIT:
            state, audited, penalty = stage_audit(state, view, cfg, t)
        elif stage is PeriodStage.REWARD:
            reward = stage_reward(state, cfg, "report", None, penalty, surplus)
            metrics = (
                val_measured(state, cfg),
                val_true(state, cfg),
                welfare_true(state.consumer_delivery, cfg),
            )
        elif stage is PeriodStage.TARGET:
            judged_target = np.array(state.target, dtype=float)
            state = stage_target(state, view, cfg)
        elif stage is PeriodStage.TERMINATE:
            state, done = stage_terminate(state, cfg)

    # `records=False` (training throughput, spec 1.1.1): the per-enterprise ledger rows are built
    # only at the step that runs DELIVER, whose `deliv` the observation needs; elsewhere none.
    build = records or PeriodStage.DELIVER in stages
    step_rows = (
        _step_records(
            state,
            action,
            cfg,
            t,
            k,
            phase,
            s_pre,
            output,
            cost,
            consumed,
            reward,
            audited,
            penalty,
            alloc,
            deliv,
            fill,
            shipped,
            holding,
            overflow,
            metrics,
            judged_target,
        )
        if build
        else ()
    )
    info = StepInfo(
        records=step_rows,
        t_period=t,
        k_step=k,
        phase=phase,
        val_measured=metrics[0],
        val_true=metrics[1],
        welfare=metrics[2],
        consumer=np.array(state.consumer_delivery, dtype=float),
        flags=(),
        terminated=done,
    )
    state = advance_phase(state, cfg)
    return state, reward, done, info


def run_period(
    state: State, actions: tuple[EnterpriseAction, ...], cfg: EnvConfig
) -> tuple[State, Array, bool, tuple[StepInfo, ...]]:
    """Run one complete plan period: `M` PRODUCE agent-steps then one REPORT agent-step.

    Takes: `state` at the head of a period (`k_step == 0`, `phase == "produce"`); `actions`, exactly
    `cfg.incentive.steps_per_period + 1` joint actions, the last of which is the REPORT step's;
    `cfg`. Returns: `(state, rewards, done, infos)` -

        state    the state at the head of the next period, or the terminal state when `done`
        rewards  (M + 1, N), one row per agent-step in execution order
        done     the geometric termination flag drawn once, at stage 7 of this period
        infos    (M + 1,) `StepInfo`s, one per agent-step, in execution order

    Implemented as `M + 1` calls to `advance` and nothing more: it exists because the two properties
    that bind this module are *per-period* statements, and a helper that runs exactly one period
    keeps them from being written as ad-hoc loops in each test. It raises if `len(actions)` is not
    `M + 1` or if `state` is not at the head of a period, rather than silently running a partial
    period.

    The per-period, per-good conservation identity of T-U1 is checked across exactly this span
    (PLAN section 11, to 1e-9):

        sum y + sum S_prev = sum (inputs consumed) / a + sum consumer + sum S_next
                             + holding loss + cap overflow

    Binds: `tests/unit/test_conservation.py` (T-U1), `tests/golden/*` (T-B7), and the Monte-Carlo
    sanity harness of WO-012. Owning WO: **WO-009**.
    """
    m = cfg.incentive.steps_per_period
    if len(actions) != m + 1:
        raise ValueError(f"run_period: expected {m + 1} actions, got {len(actions)}")
    if state.k_step != 0 or state.phase != "produce":
        raise ValueError("run_period: state is not at the head of a period")
    rewards, infos = [], []
    done = False
    for action in actions:
        state, reward, done, info = advance(state, action, cfg)
        rewards.append(reward)
        infos.append(info)
    return state, np.stack(rewards), done, tuple(infos)


@functools.lru_cache(maxsize=64)
def _run_hash(cfg: EnvConfig) -> str:
    """`cfg.hash()`, memoised on the frozen configuration (a pure function of its argument)."""
    return cfg.hash()


def _step_records(
    state,
    action,
    cfg,
    t,
    k,
    phase,
    s_pre,
    output,
    cost,
    consumed,
    reward,
    audited,
    penalty,
    alloc,
    deliv,
    fill,
    shipped,
    holding,
    overflow,
    metrics,
    judged_target,
):
    """One `StepRecord` per enterprise for the step just executed (ledger only, CONTRACT rule 6).

    `target` is the target the step was judged against (before the REPORT step's ratchet).
    DELIVER quantities (`alloc`, `deliv`, `fill`, `shipped`) sit on the row of the step at which
    DELIVER ran and are zero elsewhere; `consumer` is the period's sink receipt on every row; the
    period metrics are filled on the REPORT row. `coverage`, `audit_meas` and `penalty_arg` are
    computed inside their owning modules and not returned by them, so they are recorded as NaN.
    """
    from gosplan.metrics.ledger import StepRecord

    n = cfg.supply.n_enterprises
    sector = np.asarray(cfg.supply.sector_of, dtype=int)
    need = np.asarray(state.planner_io, dtype=float)[sector] * judged_target[:, None]
    produce = phase == "produce"
    rho_max = cfg.tech.report_max_ratio
    run_hash = _run_hash(cfg)
    consumer = tuple(float(v) for v in np.asarray(state.consumer_delivery, dtype=float))
    nan = float("nan")
    out = []
    for i in range(n):
        out.append(
            StepRecord(
                run_hash=run_hash,
                episode=0,
                t_period=int(t),
                k_step=int(k),
                phase=phase,
                enterprise=i,
                sector=int(sector[i]),
                target=float(judged_target[i]),
                capital=float(state.capital[i]),
                inv_output_pre=float(s_pre[i]),
                inv_output_post=float(state.inv_output[i]),
                inv_inputs=tuple(float(v) for v in state.inv_inputs[i]),
                cum_output=float(state.cum_output[i]),
                cum_cost=float(state.cum_cost[i]),
                quality_acc=float(state.quality_acc[i]),
                last_report_ratio=float(state.last_report_ratio[i]),
                last_penalty=float(state.last_penalty[i]),
                last_fill=float(state.last_fill[i]),
                request=tuple(float(v) for v in state.request[i]),
                need=tuple(float(v) for v in need[i]),
                effort=float(np.clip(action.effort[i], 0.0, 1.0)) if produce else 0.0,
                quality=float(action.quality[i]) if produce else 0.0,
                invest=float(action.invest[i]) if produce else 0.0,
                output=float(output[i]),
                cost=float(cost[i]),
                coverage=nan,
                reward=float(reward[i]),
                report=0.0 if produce else float(state.last_report[i]),
                report_ratio=0.0 if produce else float(state.last_report_ratio[i]),
                at_bound=(not produce) and bool(state.last_report_ratio[i] >= rho_max),
                audited=bool(audited[i]),
                audit_meas=nan,
                penalty_arg=nan,
                penalty=float(penalty[i]),
                fill=float(fill[i]),
                shipped=float(shipped[i]),
                alloc=tuple(float(v) for v in alloc[i]),
                deliv=tuple(float(v) for v in deliv[i]),
                input_consumed=tuple(float(v) for v in consumed[i]),
                holding_loss=float(holding[i]),
                cap_overflow=float(overflow[i]),
                trade_volume=0.0,
                consumer=consumer,
                val_measured=float(metrics[0]),
                val_true=float(metrics[1]),
                welfare=float(metrics[2]),
            )
        )
    return tuple(out)
