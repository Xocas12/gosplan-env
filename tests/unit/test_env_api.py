"""The environment API: the typed period schedule, reset, step and the action surface (PLAN 2.5).

Realises: PLAN sections 2.3 (actions and their bounds), 2.5 (the period schedule as a typed state
machine) and 2.12 (horizon), and PLAN section 11 (test architecture, unit/property category). Owning
work order: **WO-002** (frozen tests; LEAD). Binds the WO-009 must-pass line of PLAN section 12.3,
verbatim - "`tests/unit/test_env_api.py`". Modules under test: `gosplan/env/env.py`,
`gosplan/env/step.py`, `gosplan/env/state.py`.

Period schedule (PLAN section 2.5, verbatim), one plan period:

    0. DELIVER      planner allocates from last period's claims; the consumer sink receives
    1. [P2] TRADE   bilateral matching (section 2.13)
    2. PRODUCE x M  effort/quality/invest; yield realised
    3. REPORT       S <- (1-h)*S + y; the agent observes S and y exactly; reports and requests
    4. AUDIT        audit selection; measurement; penalty
    5. REWARD       bonus - penalty (+ trade surplus); val and welfare logged
    6. TARGET       ratchet + growth directive
    7. TERMINATE?   geometric

Agents act `M + 1` times per period: `M` PRODUCE steps and one REPORT step. Action dimensions not
relevant to the current phase are **ignored** by the environment, never rejected, and the
environment never trusts an agent to have masked them itself.

CONTRACT RULE 6: `StepInfo` carries true quantities for the ledger and for lead-run experiments; no
agent, policy or reward term may read it (the WO-010 forbidden list names "any agent reading
`StepInfo`"). CONTRACT RULE 8: `report_ratio` is bounded at `rho_max` and the fraction of reports at
the bound is a logged result.

FROZEN BY CONTRACT RULE 2. SKELETON: every test is `@pytest.mark.skeleton` and skipped until WO-009
lands; each docstring states the exact assertion, formula and tolerance.
"""

from __future__ import annotations

import pytest


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-009")
def test_reset_returns_the_first_observation_and_step_info(p1_cfg, rng_seed) -> None:
    """`reset(seed_env, seed_policy)` returns `(obs, StepInfo)` of the documented shapes.

    Assertion: `obs` is a float array of shape `(N, d)` with `d = len(obs_spec(cfg)) = 12 + 3J`, is
    finite everywhere, and equals `build_observation` on the initial state; the returned `StepInfo`
    carries one `StepRecord` per enterprise in enterprise-index order, `t_period == 0`,
    `k_step == 0`, `phase == "produce"`, `terminated is False`, and its `flags` tuple is empty.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-009")
def test_reset_initialises_the_state_as_plan_sections_2_1_and_2_2_specify(p1_cfg) -> None:
    """The opening state is the one PLAN sections 2.1-2.2 and 3 prescribe.

    Assertion, after `reset`: `target == T_0 = cfg.tech.initial_target_frac * A_{s(i)} * cap_i`;
    `capital == cap == 1`; `inv_output == 0`; `inv_inputs == 0`; every `last_*` field is zero except
    `last_fill`, which is 1; `request == 0`; `pending_invest == 0`; `t_period == 0`; `k_step == 0`;
    `phase == "produce"`; `plan_prices == initial_prices(cfg)`;
    `planner_io == cfg.supply.io_matrix`; `alive is True`; and `seed_env` / `seed_policy` are the
    values passed in, stored separately
    (CONTRACT rule 9).
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-009")
def test_step_returns_obs_reward_done_and_info(p1_cfg, rng_seed) -> None:
    """`step(action)` returns `(obs, reward, done, StepInfo)` with the documented shapes and types.

    Assertion: `obs` has shape `(N, 12 + 3J)`; `reward` has shape `(N,)` and equals
    `enterprise_reward` for the phase just executed (`-scale * c_ik` at a PRODUCE step,
    `scale * (B(rho) - penalty + surplus)` at the REPORT step); `done` is a single Python `bool` for
    the whole episode, because termination is one global geometric draw (PLAN section 2.12), not a
    per-enterprise flag; and `StepInfo` carries `N` `StepRecord`s plus the period scalars
    `val_measured`, `val_true`, `welfare` and `consumer`, which are filled at the REPORT step and
    after DELIVER respectively.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-009")
def test_phase_is_produce_for_m_steps_then_report(p1_cfg) -> None:
    """`phase()` returns `"produce"` while `k_step < M` and `"report"` at the period's last step.

    Assertion: over one period, `env.phase()` returns `"produce"` for the first
    `M = cfg.incentive.steps_per_period` calls and `"report"` for the `(M + 1)`-th, then returns to
    `"produce"` for the next period; the value it returns always describes the step the *next*
    `step` call will execute, and observation component 0 (`phase`) agrees with it at every step.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-009")
def test_an_agent_acts_m_plus_one_times_per_period(p1_cfg) -> None:
    """One period consumes exactly `M + 1` actions (PLAN sections 2.1, 2.5).

    Assertion: counting `step` calls between two consecutive increments of `state.t_period` gives
    exactly `M + 1` for every period of an episode, at `M = 4` (Phase 1) and at `M = 8` (the other
    point of the PLAN section 3 grid); the `M` PRODUCE steps carry `k_step = 0 .. M-1` and the
    REPORT step carries `k_step = M`.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-009")
def test_stages_for_step_maps_the_machine_position_to_the_schedule_slice(p1_cfg) -> None:
    """`stages_for_step(state, cfg)` returns exactly the documented contiguous slice.

    Assertion, the three cases verbatim from `gosplan/env/step.py`:

        k_step == 0, phase "produce"      -> (DELIVER, TRADE, PRODUCE)
        0 < k_step < M, phase "produce"   -> (PRODUCE,)
        k_step == M, phase "report"       -> (REPORT, AUDIT, REWARD, TARGET, TERMINATE)

    each a contiguous slice of `PERIOD_SCHEDULE` in execution order. DELIVER opens the period and
    consumes the claims recorded at the *previous* period's REPORT step; TRADE sits between DELIVER
    and the first PRODUCE step because it reallocates the inputs DELIVER has just placed in `X`.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-009")
def test_one_period_executes_the_whole_schedule_in_order(p1_cfg) -> None:
    """The stages executed across a period are `PERIOD_SCHEDULE` with PRODUCE repeated `M` times.

    Assertion: concatenating `stages_for_step` over the `M + 1` agent-steps of a period yields
    `(DELIVER, TRADE, PRODUCE, PRODUCE, ..., PRODUCE, REPORT, AUDIT, REWARD, TARGET, TERMINATE)` -
    every member of `PERIOD_SCHEDULE`, in `PeriodStage` value order, with PRODUCE appearing exactly
    `M` times and every other stage exactly once, and `advance` executing those stages and nothing
    else. TARGET runs after REWARD, so this period's bonus is judged against this period's target
    (PLAN section 2.5, steps 5 then 6).
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-009")
def test_action_spec_matches_the_plan_section_2_3_bounds(p1_cfg) -> None:
    """`action_spec(cfg)` gives every dimension its shape and box bounds.

    Assertion: the mapping is exactly, with `N = n_enterprises` and `J = n_sectors`,

        "effort"         ((N,),   0.0, 1.0)
        "quality"        ((N,),   0.0, 1.0)
        "invest"         ((N,),   0.0, 1.0)
        "report_ratio"   ((N,),   0.0, cfg.tech.report_max_ratio)
        "input_request"  ((N, J), 0.0, cfg.tech.request_max_multiple)
        "trade_offer"    ((N, J), -1.0, 1.0)

    All six dimensions are present whether or not the configuration reads them, so a Phase-1 policy
    and a Phase-2 policy share one action type. `input_request` is expressed as a multiple of need
    because the true bound of PLAN section 2.3 is `r_max * need_ij` and `need_ij` is state
    dependent; the environment rescales and clips against the current need when it reads the action.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-009")
def test_active_action_dims_at_the_phase_1_configuration(p1_cfg) -> None:
    """`active_action_dims(p1_default_config())` is `["effort", "report_ratio", "input_request"]`.

    Assertion: exactly those three names, in the order of PLAN section 2.3, and a subset of
    `action_spec(cfg).keys()`. `input_request` is active and logged although it is inert while
    `alloc_eta_request == 0` (finding F7). `quality`, `invest` and `trade_offer` become active only
    when their Phase-2 mechanisms are switched on (`supply.quality_matters`, non-zero
    `capital_dep`/investment, `information.horizontal_visibility > 0`), which the same test checks
    by flipping each toggle. The answer must be a pure function of the configuration and must not
    change within a run: the PPO adapter builds heads from it once (WO-017).
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-009")
def test_inactive_action_dimensions_are_ignored_not_rejected(p1_cfg) -> None:
    """Dimensions the phase does not read are ignored, and out-of-range values are clipped.

    Assertion: two `step` calls whose actions differ only in dimensions the current phase does not
    read - `report_ratio` and `input_request` at a PRODUCE step, `effort` at the REPORT step, and
    `quality`, `invest`, `trade_offer` at both in Phase 1 - produce identical observations, rewards
    and `StepInfo` records; no exception is raised for a value outside its box, which is instead
    clipped to the `action_spec` bounds (and, for `report_ratio` at `rho_max`, recorded as
    at-bound, CONTRACT rule 8).
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-009")
def test_episode_length_respects_the_horizon_bounds(p1_cfg, rng_seed) -> None:
    """Termination is geometric between `P_min` and `P_max` (PLAN section 2.12).

    Assertion: under `horizon_mode = "geometric"`, no episode ends before `cfg.tech.min_periods`
    periods have run, none exceeds `cfg.tech.max_periods`, `done` is False at every intermediate
    step and True exactly once at the end, and the draw goes through `gosplan.rng.draw` with purpose
    `terminate` so it is deterministic in `(seed_env, t)`. The statistical property - empirical
    continuation equal to `tenure`, and no observation field correlating with periods remaining - is
    test T-B9 in `tests/behavioural/test_termination.py`; here only the bounds and the mechanics are
    asserted.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-009")
def test_step_info_is_recorded_to_an_attached_ledger(p1_cfg) -> None:
    """`attach_ledger` / `record_step` write one `StepRecord` per enterprise per agent-step.

    Assertion: with a `Ledger` attached, an episode of `P` periods at `M + 1` agent-steps each
    appends exactly `N * P * (M + 1)` records, each carrying the run hash, the episode, `t_period`,
    `k_step`, `phase`, `enterprise` and `sector` that identify it; without a ledger the environment
    runs unchanged and records nothing. No agent-facing code path touches `StepInfo` (CONTRACT rule
    6, WO-010 forbidden list).
    """
    assert False
