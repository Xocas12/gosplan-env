"""T-B6 - the reward is exactly the formula of CONTRACT rule 4, recomputed independently.

Realises: PLAN section 11 (behavioural test T-B6) and CONTRACT rule 4, read against PLAN sections
2.9.1 (the enterprise reward), 2.8 (`bonus`, the audit penalty), 2.9.2 (the fulfilment measure
`rho` keys on), 2.9.3 (the logged-only aggregates) and finding F9 (why the scale is analytic).
Owning work order: **WO-002**; on the must-pass list of **WO-007** (`gosplan/env/reward.py`).

CONTRACT rule 4 (REWARD TERMS), verbatim in substance. The enterprise reward is exactly

    production step:   -scale * c_ik
    report step:        scale * (B(rho_i) - 1[audited_i] * Pen_i + trade_surplus_i)

with `scale = reward_scale(cfg)` computed analytically from the configuration. No per-step shaping,
no auxiliary reward, no curiosity term, no potential-based term, and **no running reward
normalisation** - running statistics change the effective reward over training and, with
heavy-tailed penalties, shrink the notch in normalised units. Per-batch advantage normalisation
inside PPO is permitted and is checked in `tests/unit/test_ppo_adapter.py` (WO-017), not here.

What T-B6 asserts (PLAN section 11, verbatim): *reward equals the five-term formula recomputed
independently on random states.* The five terms are `scale`, `c_ik`, `B(rho_i)`,
`1[audited_i] * Pen_i` and `trade_surplus_i`.

INDEPENDENT RECOMPUTATION IS THE POINT. The test implements `Lambda_w`, `B`, `Pen` and `scale`
**inside the test file**, transcribed from the PLAN formulas below, and compares against
`enterprise_reward`. It must not call `gosplan.env.reward.bonus` or `reward_scale` to build its
expectation: a bug shared between the environment and its own check cancels, and this test's only
job is to prevent that. (`tests/unit/test_reward.py` checks `bonus` and `reward_scale` against
their own properties - T-U2, T-U3 - which is the other half of the same argument.)

The formulas to transcribe (PLAN sections 2.8, 2.9.1, verbatim):

    Lambda_w(x) = 1[x >= 0]                      if w = 0        # strict >=, a true Heaviside
                = 1 / (1 + exp(-x / w))          if w > 0
    B(rho)      = beta * Lambda_w(rho - 1) + s * clip(rho - 1, 0, rho_cap - 1)
                                                                 # no clip at all when rho_cap = inf
    f_i         = max(0, R_i - S_hat_i) / T_i    if penalty_arg = "positive_part"
                = abs(R_i - S_hat_i) / T_i       if penalty_arg = "absolute"
    Pen_i       = pen * f_i                      if penalty_form = "proportional"
                = pen * 1[f_i > 0]               if penalty_form = "fixed"
    scale       = 1 / B_cfg(1.1)

"Random states" means states drawn from the `seed_policy` stream, never from `draw` (CONTRACT rule
9 reserves `draw` for environment randomness): targets, stocks, claims, audit flags, costs and
audit measurements are sampled over the ranges of `RANDOM_STATE_RANGES` so that both branches of
every conditional above are exercised - over- and under-reports, audited and unaudited, `rho` below
1, at 1 and above `rho_cap`.

Held-out phenomena (PLAN section 4.1): none of the quantities here is a phenomenon. `trade_surplus`
is asserted to be zero in Phase 1 as a *term of the formula*, not measured as blat (row 6).
"""

from __future__ import annotations

import pytest

SKIP_REASON = (
    "skeleton: T-B6 assertions are written by WO-002 (frozen tests); they bind WO-007 "
    "(enterprise_reward, bonus, reward_scale, audit_and_penalise)"
)
"""Reason attached to every `@pytest.mark.skip` below; the bodies arrive with WO-002."""

Overrides = dict[str, dict[str, object]]
"""Shape of a configuration override document: arm name -> field name -> value, applied on top of
`p1_default_config()`. Named so the parametrised signatures below fit on one line."""

REWARD_TERMS: tuple[str, ...] = ("scale", "effort_cost", "bonus", "audit_penalty", "trade_surplus")
"""The five terms of CONTRACT rule 4, named so a failure message can say which one disagreed."""

CONFIG_MATRIX: tuple[Overrides, ...] = (
    {"incentive": {"notch_width": 0.0, "overfulfilment_cap": 1.2}},
    {"incentive": {"notch_width": 0.25, "overfulfilment_cap": float("inf")}},
    {"incentive": {"notch_width": 0.25, "overfulfilment_cap": 1.2}},
    {"incentive": {"penalty_form": "fixed", "penalty_arg": "absolute"}},
    {"incentive": {"overfulfilment_slope": 0.0, "notch_height": 0.25}},
)
"""Configurations swept, as overrides on `p1_default_config()`. The first three are the named
schedules of PLAN section 2.8 - notched (`w = 0`, `rho_cap = 1.2`), the smooth counterfactual
(`w = 0.25`, `rho_cap = inf`) and kink-only (`w = 0.25`, `rho_cap = 1.2`) - so both branches of
`Lambda_w` and both the capped and uncapped branches of `B` are exercised. The fourth flips both
penalty switches; the fifth removes the slope, leaving the notch alone."""

N_RANDOM_STATES = 512
"""Random states drawn per configuration and phase. A WO-002 test-design constant: enough to visit
every branch of every conditional several times, small enough to stay a unit-speed test."""

RANDOM_STATE_SEED = 20021
"""Seed for the `numpy.random.Generator` that draws the random states. It belongs to the *policy*
stream in spirit (CONTRACT rule 9): no environment draw is taken here, and the constant is fixed so
the test is deterministic and a failure is reproducible from its message alone."""

RANDOM_STATE_RANGES: dict[str, tuple[float, float]] = {
    "target": (0.1, 5.0),
    "inv_output": (0.0, 6.0),
    "report_ratio": (0.0, 10.0),
    "effort": (0.0, 1.0),
    "audit_meas": (0.0, 6.0),
}
"""Ranges the random states are drawn over. `report_ratio` spans the whole action bound
`[0, rho_max]` of PLAN section 2.3, so reports below, at and far above target all appear;
`audit_meas` is drawn independently of `inv_output` so that over-claims and under-claims both
occur; `target` stays strictly positive because `rho` and the penalty are ratios in `T`."""

EXACT_TOL = 1e-12
"""Tolerance for the term-by-term agreement. The two computations perform the same arithmetic in
the same order up to associativity, so the difference is rounding, not method."""


@pytest.mark.skeleton
@pytest.mark.skip(reason=SKIP_REASON)
@pytest.mark.parametrize("overrides", CONFIG_MATRIX)
def test_produce_step_reward_equals_minus_scaled_cost(overrides: Overrides) -> None:
    """At a PRODUCE step the reward is `-scale * c_ik` and nothing else.

    For each configuration in `CONFIG_MATRIX` and each of `N_RANDOM_STATES` random states drawn
    with `RANDOM_STATE_SEED` over `RANDOM_STATE_RANGES`, call

        enterprise_reward(state, cfg, "produce", cost=c, penalty=None, trade_surplus=None)

    and assert `abs(r_i - (-scale_expected * c_i)) <= EXACT_TOL` for every `i`, where
    `scale_expected = 1 / B_expected(1.1)` is computed from the transcribed formula in this file.
    Assert additionally that the result does not depend on `state.last_report_ratio`,
    `state.last_audited` or `state.last_penalty`: re-draw those three fields and assert the reward
    is bitwise unchanged, so no report-step quantity leaks into a production step.

    Owning WO: **WO-002**; binds **WO-007**.
    """
    raise NotImplementedError("PLAN section 11 (T-B6) - implemented in WO-002")


@pytest.mark.skeleton
@pytest.mark.skip(reason=SKIP_REASON)
@pytest.mark.parametrize("overrides", CONFIG_MATRIX)
def test_report_step_reward_equals_five_term_formula(overrides: Overrides) -> None:
    """At the REPORT step the reward is `scale * (B(rho) - 1[audited] * Pen + trade_surplus)`.

    For each configuration and each random state, compute the expectation inside this test from the
    transcribed formulas - `Lambda_w`, `B`, `f`, `Pen`, `scale` - never by calling
    `gosplan.env.reward` or `gosplan.env.reporting`, then assert

        abs(enterprise_reward(state, cfg, "report", None, penalty, trade_surplus)
            - scale_expected * (B_expected(rho) - penalty + trade_surplus)) <= EXACT_TOL

    for every enterprise. Assert coverage of the branch structure over the sampled states, and fail
    with a message naming the disagreeing entry of `REWARD_TERMS` when a term is isolatable:

        * `rho < 1`, `rho == 1` exactly, `1 < rho < rho_cap`, `rho >= rho_cap`;
        * `audited` True and False, with `penalty == 0` whenever `audited` is False;
        * `w = 0` (the Heaviside is strict `>=`, so `rho = 1` scores the full notch) and `w > 0`;
        * `rho_cap = inf`, where the slope term is never clipped.

    Assert `trade_surplus` is `None` or exactly zero throughout Phase 1, so the fifth term is
    present in the formula and inert in the configuration (PLAN section 2.13).

    Owning WO: **WO-002**; binds **WO-007**.
    """
    raise NotImplementedError("PLAN section 11 (T-B6) - implemented in WO-002")


@pytest.mark.skeleton
@pytest.mark.skip(reason=SKIP_REASON)
def test_no_term_outside_the_contract_rule_4_list() -> None:
    """Nothing else is added: zero inputs give exactly zero reward, in both phases.

    Assert on states drawn as above, at `p1_default_config()`:

        enterprise_reward(state, cfg, "produce", cost=zeros, ...)          == 0.0 exactly
        enterprise_reward(state, cfg, "report", ..., penalty=zeros,
                          trade_surplus=zeros) with rho such that B(rho) == 0
                                                                          == 0.0 exactly

    (the second uses `beta = 0`, `s = 0`, where `B` is identically zero.) Exact equality with `0.0`,
    not a tolerance: any shaping term, potential-based term, curiosity bonus, survival bonus or
    constant offset shows up here as a non-zero number, and CONTRACT rule 4 admits none of them.
    Assert also that the reward is unchanged when `state.cum_output`, `state.inv_inputs` and the
    period-level `welfare`, `val_true` and `val_measured` are re-drawn - the reward reads none of
    them (CONTRACT rules 4 and 6).

    Owning WO: **WO-002**; binds **WO-007**.
    """
    raise NotImplementedError("PLAN section 11 (T-B6) - implemented in WO-002")


@pytest.mark.skeleton
@pytest.mark.skip(reason=SKIP_REASON)
def test_scale_is_analytic_and_reward_is_not_normalised() -> None:
    """`scale` comes from the configuration alone, and no running statistic touches the reward.

    Assert:

        * `reward_scale(cfg)` is a pure function of `cfg`: repeated calls return the identical
          float, and two `EnvConfig` values with equal `hash()` give equal scales;
        * calling `enterprise_reward` `N_RANDOM_STATES` times on the *same* state and inputs
          returns the identical value every time, and the value from the first call equals the
          value from the last - a running normaliser would drift, which is exactly what CONTRACT
          rule 4 forbids and why finding F9 made the scale analytic;
        * interleaving calls with extreme rewards (a heavy penalty state) between two identical
          calls does not change the second one's result - the shape of the check that catches a
          hidden `RunningMeanStd`;
        * `reward_scale(cfg) * B_expected(1.1) == 1` to `EXACT_TOL`, the analytic definition
          (T-U2's behavioural echo; the unit form lives in `tests/unit/test_reward.py`).

    The adapter-side half of rule 4 - that the wrapped reference PPO carries no reward
    normalisation and that per-batch advantage normalisation is on - is
    `tests/unit/test_ppo_adapter.py` (WO-017). Owning WO: **WO-002**; binds **WO-007**.
    """
    raise NotImplementedError("PLAN section 11 (T-B6) - implemented in WO-002")
