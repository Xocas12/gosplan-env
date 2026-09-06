"""T-B5 - welfare blindness: logged-only quantities never reach an agent.

Realises: PLAN section 11 (behavioural test T-B5) and CONTRACT rule 6, read against PLAN sections
2.4 (the enumerated observation and its "never in any observation" list), 2.9.3 (`welfare_true`,
`val_measured`, `val_true` - logged, never observed), 2.12 (geometric termination, so periods
remaining are unobservable and there is no end-game, finding F4) and 6.1 (the `Agent` protocol).
Owning work order: **WO-002**; on the must-pass lists of **WO-008** (`gosplan/env/obs.py`) and
**WO-017** (the PPO adapter).

CONTRACT rule 6 (WELFARE BLINDNESS): `welfare_true` and `val_measured` are logged and never appear
in any observation, reward or agent input; the PPO adapter's forward pass takes `obs` only.

What T-B5 asserts (PLAN section 11, verbatim): *sentinel `welfare`, other enterprises' `y`, and
periods-remaining never appear in any observation; PPO adapter forward signature takes `obs` only.*
PLAN section 2.4's list of what is never in any observation is the full target set: `welfare_true`,
`val_measured`, any other enterprise's `y`, `S` or `X`, the audit selection for the current period,
and periods remaining under geometric termination.

THE SENTINEL SWEEP. A single sentinel value can be missed two ways: it can coincide with a
legitimate observation entry (a false pass), or it can leak through a transformation - the
observation is full of ratios, so a leaked `S_j` divided by `T_i` no longer equals the planted
number. The sweep answers both:

    * every value of `SENTINELS` is planted in turn, each with a per-enterprise offset, and the
      assertion is repeated - a coincidence would have to hold for every sentinel and every offset;
    * the forbidden quantity is planted at `SENTINEL_SCALE_FACTORS` different magnitudes, and the
      assertion also covers the images of the sentinel under the transformations the layout could
      apply: `x`, `x / T_i`, `x * reward_scale(cfg)`, `log(x / T_0)` and `x / need_ij`. A leak
      through a ratio changes the value but not the *dependence*, which the second test below
      catches directly by varying the sentinel and asserting that no observation entry moves.

The dependence test is the load-bearing one: it plants two different sentinel values in the same
forbidden field, holds every other input and every draw fixed, and asserts the two observations are
bitwise identical. That catches an arbitrary leaked transformation, not just an echoed number.

Related but distinct: T-B9 (`test_termination.py`) asserts *statistically* that no observation
field correlates with periods remaining under geometric termination; this file asserts that no
observation entry *is* the remaining-period count or any of its images. Both are needed - finding
F4 is that the agent must never see the end of the episode coming.
"""

from __future__ import annotations

import pytest

SKIP_REASON = (
    "skeleton: T-B5 assertions are written by WO-002 (frozen tests); they bind WO-008 "
    "(gosplan/env/obs.py), WO-009 (StepInfo) and WO-017 (the PPO adapter's forward pass)"
)
"""Reason attached to every `@pytest.mark.skip` below; the bodies arrive with WO-002."""

SENTINELS: tuple[float, ...] = (-6.02214076e5, 1.61803398e6, 2.71828182e7)
"""Sentinel magnitudes planted, in turn, in every forbidden field. Three of them so a numerical
coincidence cannot produce a false pass, all far outside the plausible range of any plan quantity
(targets are of order 1, welfare of order 1)."""

SENTINEL_SCALE_FACTORS: tuple[float, ...] = (1.0, 1e-3, 1e3)
"""Magnitude sweep applied to each sentinel, so that a leak surviving a division by a target or a
multiplication by `reward_scale(cfg)` still lands outside the legitimate range for at least one
factor."""

SENTINEL_TOL = 1e-9
"""Tolerance for "no observation entry equals a sentinel image":
`abs(entry - image) <= SENTINEL_TOL` is a leak."""

FORBIDDEN_QUANTITIES: tuple[str, ...] = (
    "welfare_true",
    "val_measured",
    "val_true",
    "other_enterprise_output",
    "other_enterprise_stock",
    "other_enterprise_inputs",
    "current_period_audit_selection",
    "periods_remaining",
)
"""Everything PLAN section 2.4 says is never in any observation, plus `val_true`, which PLAN
section 2.9.3 logs on the same terms. The sweep plants a sentinel for each in turn: the first three
in the `StepInfo` / period scalars, the next three in other enterprises' `State` rows, the audit
selection in `State.last_audited` before the AUDIT step of the current period, and periods
remaining as the episode length the environment has already drawn."""

FORBIDDEN_OBS_NAME_FRAGMENTS: tuple[str, ...] = (
    "welfare",
    "val_",
    "remaining",
    "horizon",
    "terminate",
    "other_",
    "audit_selection",
)
"""Substrings that may not appear in any name returned by `obs_spec(cfg)` (PLAN section 2.4). A
name-level check is weaker than the sentinel sweep but it fails earlier and reads as documentation;
`last_audited` (index 8) is the previous period's outcome and is admissible, which is why the
fragment is `audit_selection` and not `audit`."""

OBS_DIM_FORMULA = "12 + 3 * n_sectors"
"""Phase-1 observation dimension (PLAN section 2.4). Asserted here as a second guard: a leak that
appended a field would change the dimension, and `tests/unit/test_obs.py` (WO-008) checks the
layout itself."""

AGENT_ACT_PARAMETERS: tuple[str, ...] = ("self", "obs", "phase", "rng")
"""The exact parameter list of `Agent.act` (PLAN section 6.1, `spec/spec.py`). No `State`, no
`StepInfo`, no `PlannerView` - "any agent reading `StepInfo`" is on the WO-010 forbidden list."""

PPO_FORWARD_PARAMETERS: tuple[str, ...] = ("self", "obs")
"""The exact parameter list of `IPPO.forward` (CONTRACT rule 6: "the PPO adapter's forward pass
takes obs only")."""

FORBIDDEN_AGENT_ANNOTATIONS: tuple[str, ...] = ("State", "StepInfo", "PlannerView", "Ledger")
"""Type names that may not appear in any annotation of any agent's public methods."""


@pytest.mark.skeleton
@pytest.mark.skip(reason=SKIP_REASON)
@pytest.mark.parametrize("sentinel", SENTINELS)
@pytest.mark.parametrize("quantity", FORBIDDEN_QUANTITIES)
def test_no_sentinel_reaches_any_observation(quantity: str, sentinel: float) -> None:
    """The sweep: no forbidden quantity, at any planted magnitude, appears in any observation.

    For each `quantity` of `FORBIDDEN_QUANTITIES` and each `sentinel`, and for each factor in
    `SENTINEL_SCALE_FACTORS`, build a `State` at `p1_default_config()` with that quantity set to
    `sentinel * factor` (offset per enterprise so an element-level leak is caught), then build the
    observation for every enterprise at both phases through `gosplan.env.obs`. Assert that for
    every entry of every observation row and every image `img` in

        {x, x / T_i, x * reward_scale(cfg), log(x / T_0), x / need_ij}     with x the planted value

    `abs(entry - img) > SENTINEL_TOL`. Assert also that the observation dimension is
    `OBS_DIM_FORMULA` and that the run completes with `self_obs_noise = 0`, so fields 4 and 5 are
    exact and no noise could mask a leaked value.

    A hit is a CONTRACT rule 6 violation, and the strictest case is `welfare_true`: no agent, no
    planner rule and no reward term may read it (PLAN section 2.9.3). Owning WO: **WO-002**; binds
    **WO-008**.
    """
    raise NotImplementedError("PLAN section 11 (T-B5) - implemented in WO-002")


@pytest.mark.skeleton
@pytest.mark.skip(reason=SKIP_REASON)
@pytest.mark.parametrize("quantity", FORBIDDEN_QUANTITIES)
def test_observation_does_not_depend_on_forbidden_quantity(quantity: str) -> None:
    """No observation entry *moves* when a forbidden quantity changes.

    Build two states that are identical in every respect - same `seed_env`, same `seed_policy`,
    same targets, stocks, inputs, period and phase - except that `quantity` holds `SENTINELS[0]` in
    one and `SENTINELS[1]` in the other. Build the observation for every enterprise at both phases
    from each state and assert the two arrays are **bitwise identical** (`np.array_equal` on the
    raw floats, not `allclose`).

    This is the assertion that survives an arbitrary leaked transformation: a value echoed through
    a ratio, a logarithm or a scaling would still change when the source changes, and only a
    genuine independence keeps the two observations identical. It is also why the environment's
    draws are keyed rather than streamed (PLAN section 2.15): with `draw` keyed on
    `(seed_env, purpose, indices)`, two runs that differ only in a forbidden field see identical
    randomness, so any difference in the observation is a leak and never noise.

    Owning WO: **WO-002**; binds **WO-008** and **WO-009**.
    """
    raise NotImplementedError("PLAN section 11 (T-B5) - implemented in WO-002")


@pytest.mark.skeleton
@pytest.mark.skip(reason=SKIP_REASON)
def test_obs_spec_names_exclude_forbidden_fields() -> None:
    """`obs_spec(cfg)` names nothing forbidden, and has the Phase-1 length.

    Assert that no name returned by `obs_spec(p1_default_config())` contains any fragment of
    `FORBIDDEN_OBS_NAME_FRAGMENTS` (case-insensitive), and that
    `len(obs_spec(cfg)) == 12 + 3 * cfg.supply.n_sectors` (`OBS_DIM_FORMULA`, PLAN section 2.4).
    Repeat over a small matrix of configurations that vary `n_sectors`, so the formula is checked
    rather than the single Phase-1 number.

    The layout itself - the exact ordered names - is `tests/unit/test_obs.py` (WO-008); this is the
    blindness half only. Owning WO: **WO-002**.
    """
    raise NotImplementedError("PLAN section 11 (T-B5) - implemented in WO-002")


@pytest.mark.skeleton
@pytest.mark.skip(reason=SKIP_REASON)
def test_periods_remaining_are_not_observable() -> None:
    """Nothing in an observation is the number of periods left, or an image of it.

    Under `horizon_mode = "geometric"` at `p1_default_config()`, roll out an episode and, at every
    agent-step, assert that no observation entry equals - to `SENTINEL_TOL` - any of

        periods_remaining, periods_remaining / max_periods, max_periods - t_period,
        t_period / max_periods, tenure ** periods_remaining

    where `periods_remaining = episode_length - t_period` is known to the test from the finished
    episode and to nobody else. Index 1 of the observation is `k / M`, the step *within* the
    period, which is admissible and must not be mistaken for a horizon signal; index 3 is the
    constant `g`.

    Finding F4: under geometric termination the agent never observes periods remaining, so there is
    no end-game to exploit; that is the property the whole horizon design rests on. The statistical
    counterpart - that no field *correlates* with periods remaining - is T-B9 in
    `test_termination.py`. Owning WO: **WO-002**; binds **WO-008** and **WO-009**.
    """
    raise NotImplementedError("PLAN section 11 (T-B5) - implemented in WO-002")


@pytest.mark.skeleton
@pytest.mark.skip(reason=SKIP_REASON)
def test_agent_and_ppo_signatures_take_obs_only() -> None:
    """Signature check: `act` takes `(obs, phase, rng)` and the PPO forward pass takes `obs`.

    Assert with `inspect.signature`:

        tuple(inspect.signature(IPPO.forward).parameters) == PPO_FORWARD_PARAMETERS
        tuple(inspect.signature(A.act).parameters) == AGENT_ACT_PARAMETERS

    for `IPPO` from `gosplan.agents.ppo.adapter` and for every agent class `A` in
    `gosplan.agents.heuristic` (`Random`, `TruthfulMyopic`, `Padder`, `DPGreedy`, and the Phase-2
    `Berliner`, `Weitzman`, `Kornai`), plus the `Agent` protocol itself. Assert that no annotation
    on any public method of those classes names anything in `FORBIDDEN_AGENT_ANNOTATIONS`, using
    the same `ast`-based rendering as T-B4 (these modules carry `from __future__ import
    annotations`, so runtime annotations are strings).

    CONTRACT rule 6 fixes the forward pass to `obs` alone, and the WO-010 forbidden list adds "any
    agent reading `StepInfo`". `StepInfo` carries `welfare`, `val_true` and `val_measured` (PLAN
    section 2.9.3), so an agent that accepted one would read exactly the three quantities the rule
    exists to keep away from it. Owning WO: **WO-002**; binds **WO-010** and **WO-017**.
    """
    raise NotImplementedError("PLAN section 11 (T-B5) - implemented in WO-002")
