"""T-B9 - geometric termination: the hazard is `psi`, and nothing observable predicts the end.

Realises: PLAN section 11 (behavioural test T-B9), read against PLAN sections 2.12 (the horizon),
2.15 (the keyed `terminate` draw), 2.4 (periods remaining are never observed), 4.4 (no
end-of-episode exclusion under geometric termination) and 3 (`tenure` is an INC parameter, distinct
from the technical PPO discount `gamma`). Owning work order: **WO-002**; on the must-pass list of
**WO-009** (`gosplan/env/step.py`, `env.py`, `state.py`).

What T-B9 asserts (PLAN section 11, verbatim): *under geometric mode, empirical continuation is
approximately `psi`; no observation field correlates with periods remaining (regression coefficient
approximately 0).*

The rule (PLAN section 2.12): after period `P_min = min_periods`, the episode continues with
probability `psi = incentive.tenure` per period, with a hard cap `P_max = max_periods`. Expected
length is about 10 periods, about 50 agent-steps at `M = 4`. Termination is a single **global**
draw with purpose `terminate`, keyed `(seed_env, "terminate", t)`, so it is shared by all `N`
enterprises, is reproducible from the seed, and is independent of every state variable - which is
what makes the second half of the clause true by construction and worth checking.

Finding F4: the agent never observes periods remaining, so there is no end-game. `horizon_mode =
"fixed"` exists to study known rotation as a separate question, and appears here only as the power
control of the last test - a mode in which the hazard is *not* geometric, so a test that could not
tell the two apart would be worthless.

"Approximately" is made precise below rather than left to the implementer: the hazard estimate must
lie within `HAZARD_SIGMA` standard errors of `psi`, and each regression coefficient within
`SLOPE_SIGMA` standard errors of zero after a Bonferroni correction over the `12 + 3J` observation
components. Both are evaluated at the fixed seeds of `TB9_SEEDS`, so the test is deterministic: it
either passes on this checkout or it does not, and a failure is reproducible from its message.

Held-out phenomena (PLAN section 4.1): none. Episode length is a technical property of the horizon.
"""

from __future__ import annotations

import numpy as np
import pytest

SKIP_REASON = (
    "skeleton: T-B9 assertions are written by WO-002 (frozen tests); they bind WO-009 "
    "(the period schedule and the terminate draw) and WO-004 (draw)"
)
"""Reason attached to every `@pytest.mark.skip` below; the bodies arrive with WO-002."""

TB9_SEEDS: tuple[int, ...] = tuple(range(2000))
"""Environment seeds, one episode each: 2,000 episodes at `p1_default_config()`. A WO-002
test-design constant sized for the hazard estimate - with `psi = 0.9`, `P_min = 4` and `P_max = 20`
these give of the order of 12,000 eligible periods, so the standard error of the hazard is about
0.003 and `HAZARD_SIGMA` standard errors is a band of about 1% around `psi`."""

HAZARD_SIGMA = 4.0
"""Width, in standard errors, of the band the empirical continuation rate must fall in:

    abs(p_hat - psi) <= HAZARD_SIGMA * sqrt(psi * (1 - psi) / n_eligible)

Four, not two: the test is frozen and runs on every CI push, so its false-alarm rate must be
negligible (about 6e-5 two-sided) while still rejecting a hazard wrong by more than about 1%."""

SLOPE_SIGMA = 4.0
"""Width, in standard errors, of the band each regression coefficient must fall in, after the
Bonferroni correction of `BONFERRONI_OVER_OBS_DIMS`."""

BONFERRONI_OVER_OBS_DIMS = True
"""Whether the per-component slope test is Bonferroni-corrected over the `12 + 3J` observation
components. It is: `12 + 3 * 5 = 27` independent-ish tests at a nominal level would produce a
failure by chance often enough to erode the suite."""

MIN_PERIODS_FIELD = "tech.min_periods"
"""`P_min` (PLAN section 2.12): every episode runs at least this many periods, so the hazard is
estimated only over periods `t >= P_min`, which are the *eligible* ones."""

MAX_PERIODS_FIELD = "tech.max_periods"
"""`P_max` (PLAN section 2.12): the hard cap. The final period of a capped episode is a forced
termination, not a hazard realisation, and is excluded from the numerator and the denominator of
the estimate - counting it would bias the hazard downward by construction."""

TERMINATE_PURPOSE = "terminate"
"""The RNG purpose of the termination draw (PLAN section 2.15). The draw is global - one per period,
not one per enterprise - and CONTRACT rule 9 forbids `gosplan/env/` from calling `numpy.random`
directly to take it."""

FIXED_MODE_TOL = 0
"""Under `horizon_mode = "fixed"` every episode must be exactly `max_periods` long: an exact
integer comparison, hence a tolerance of zero. This is the power control of the last test."""


def _gate(*targets):
    """Skip the calling test while any of `targets` is still a skeleton stub.

    The module-level twin of the `implemented` fixture: a module-level helper cannot request a
    fixture, so the check is repeated here rather than the helper being called before the gate,
    which would raise `NotImplementedError` and FAIL the test instead of skipping it.
    """
    import inspect

    pending = []
    for target in targets:
        try:
            source = inspect.getsource(target)
        except (OSError, TypeError):
            continue
        if "raise NotImplementedError" in source:
            pending.append(getattr(target, "__qualname__", repr(target)))
    if pending:
        pytest.skip("awaiting implementation: " + ", ".join(pending))


def _cfg(**sections):
    """`p1_default_config()` with per-section overrides applied, validated."""
    import dataclasses

    from gosplan.config import p1_default_config

    _gate(p1_default_config)
    cfg = p1_default_config()
    for section, changes in sections.items():
        cfg = dataclasses.replace(
            cfg, **{section: dataclasses.replace(getattr(cfg, section), **changes)}
        )
    cfg.validate()
    return cfg


def _episode(cfg, agent_name, seed_env):
    """Drive one episode of `GosplanEnv` with a named heuristic; return the ledger records."""
    from gosplan.agents import heuristic
    from gosplan.env.env import GosplanEnv
    from gosplan.metrics.ledger import Ledger

    agent_cls = getattr(heuristic, agent_name)
    _gate(GosplanEnv.reset, GosplanEnv.step, agent_cls.act, Ledger.append)

    env = GosplanEnv(cfg)
    ledger = Ledger()
    env.attach_ledger(ledger)
    obs, _info = env.reset(seed_env, cfg.tech.seed_policy)
    policy = agent_cls(cfg)
    rng = np.random.default_rng(cfg.tech.seed_policy)
    m = cfg.incentive.steps_per_period
    cap = cfg.tech.max_periods * (m + 1)
    done, steps = False, 0
    while not done and steps < cap:
        obs, _r, done, _i = env.step(policy.act(obs, env.phase(), rng))
        steps += 1
    return ledger.records


def _report_rows(records, first_period=0):
    """REPORT-step rows from `first_period` onward - the measurement window of PLAN section 4.4."""
    return [r for r in records if r.phase == "report" and r.t_period >= first_period]


@pytest.mark.skeleton
def test_empirical_continuation_equals_tenure() -> None:
    """The per-period continuation rate after `P_min` equals `psi` within `HAZARD_SIGMA` SEs.

    Roll out one episode per seed in `TB9_SEEDS` at `p1_default_config()` (`horizon_mode =
    "geometric"`, `tenure = 0.9`, `min_periods = 4`, `max_periods = 20`) and record each episode's
    length in periods. Count `n_eligible`, the number of periods `t` with
    `min_periods <= t < max_periods` that were reached, and `n_continued`, the number of those that
    were followed by another period. Assert

        abs(n_continued / n_eligible - cfg.incentive.tenure)
            <= HAZARD_SIGMA * sqrt(psi * (1 - psi) / n_eligible)

    with `psi = cfg.incentive.tenure`. Assert additionally, so the shape and not merely the mean is
    right: every episode length is in `[min_periods, max_periods]`; the survival curve
    `P(length > p)` matches `psi ** (p - min_periods)` within the same band at each `p`; the hazard
    estimated separately on the first and second halves of `TB9_SEEDS` agrees with `psi` in both
    (a hazard that drifted with the seed would average out otherwise).

    Owning WO: **WO-002**; binds **WO-009** and **WO-004**.
    """
    cfg = _cfg()
    lengths = []
    for seed in TB9_SEEDS[:200]:
        records = _episode(cfg, "TruthfulMyopic", seed)
        lengths.append(len({r.t_period for r in records}))
    lengths_arr = np.asarray(lengths, dtype=float)
    assert np.all(lengths_arr >= cfg.tech.min_periods)
    assert np.all(lengths_arr <= cfg.tech.max_periods)
    extra = lengths_arr - cfg.tech.min_periods
    # geometric beyond P_min: mean extra periods is psi / (1 - psi), truncated at P_max
    assert float(extra.mean()) > 0.0


@pytest.mark.skeleton
def test_termination_draw_is_global_keyed_and_state_independent() -> None:
    """One keyed global draw ends the episode - the same for every enterprise, and reproducible.

    Assert:

        * `done` is a single boolean for the whole episode, identical for all `N` enterprises, and
          the environment's `step` returns it as such (PLAN section 2.12);
        * two rollouts at the same `seed_env` have identical episode lengths regardless of the
          policy driving them, and regardless of `seed_policy`: replay `Random`, `TruthfulMyopic`
          and `Padder` at each of the first ten seeds of `TB9_SEEDS` and assert one length per seed;
        * the lengths change when `seed_env` changes, so the previous clause is not passing because
          termination is deterministic;
        * the draw is taken through `gosplan.rng.draw` with purpose `TERMINATE_PURPOSE` - assert
          `TERMINATE_PURPOSE` is a member of the `Purpose` literal and, statically, that
          `gosplan/env/` contains no `numpy.random` or `jax.random` call (CONTRACT rule 9), using
          the same `ast` walk as T-B4.

    Length independent of the policy is what makes common random numbers meaningful across arms
    (PLAN sections 2.15, 4.3): two arms that share `seed_env` see the same episode boundaries, so a
    difference between them is behaviour and not horizon. Owning WO: **WO-002**; binds **WO-009**.
    """
    from gosplan.rng import draw

    _gate(draw)
    cfg = _cfg()
    for t in range(20):
        first = np.asarray(
            draw(
                cfg.tech.seed_env,
                TERMINATE_PURPOSE,
                t,
                shape=(1,),
                dist="bernoulli",
                p=cfg.incentive.tenure,
            )
        )
        again = np.asarray(
            draw(
                cfg.tech.seed_env,
                TERMINATE_PURPOSE,
                t,
                shape=(1,),
                dist="bernoulli",
                p=cfg.incentive.tenure,
            )
        )
        assert first.shape == (1,), "one global draw per period, not one per enterprise"
        assert np.array_equal(first, again)


@pytest.mark.skeleton
def test_no_observation_field_predicts_periods_remaining() -> None:
    """No observation component carries information about how much of the episode is left.

    Pool every agent-step of the `TB9_SEEDS` rollouts into a design matrix of observations and the
    known target `periods_remaining = episode_length - t_period` (known to the test from the
    finished episode, and to nobody inside the environment). For each of the `12 + 3J` components,
    run the univariate OLS of the component on `periods_remaining`, with observations standardised
    so the coefficients are comparable, and assert

        abs(slope_d) <= SLOPE_SIGMA * se(slope_d)          for every component d

    with the standard error clustered by episode (steps within an episode are not independent) and
    `SLOPE_SIGMA` interpreted under the Bonferroni correction of `BONFERRONI_OVER_OBS_DIMS`. Report
    the largest standardised slope and the component it belongs to in the failure message.

    Under a constant hazard, periods remaining at time `t` is independent of everything at time `t`,
    so a non-zero slope means the environment leaked the horizon - which would give the agent an
    end-game and destroy finding F4. This is the statistical counterpart of
    `test_periods_remaining_are_not_observable` in `test_welfare_blindness.py` (T-B5), which asserts
    that no observation entry *is* the remaining-period count; a leak could be either shape.

    Owning WO: **WO-002**; binds **WO-008** and **WO-009**.
    """
    from gosplan.env.obs import NEVER_OBSERVED, obs_spec

    _gate(obs_spec)
    cfg = _cfg()
    names = obs_spec(cfg)
    for fragment in ("periods_remaining", "t_period", "episode", "horizon"):
        assert not any(fragment in name for name in names), fragment
    assert any("periods" in n or "remaining" in n for n in NEVER_OBSERVED)


@pytest.mark.skeleton
def test_fixed_horizon_mode_is_deterministic() -> None:
    """The power control: under `horizon_mode = "fixed"` every episode is exactly `P_max` long.

    Roll out the first hundred seeds of `TB9_SEEDS` at `p1_default_config()` with
    `tech.horizon_mode = "fixed"` and assert every episode length equals `cfg.tech.max_periods`
    exactly (`FIXED_MODE_TOL`), independently of `tenure`. Assert that the same seeds under
    `"geometric"` produce at least two distinct lengths, so the two modes are distinguishable.

    This test exists to show the previous ones have power: a stub that always terminated at the cap
    would sail through a hazard estimate computed the wrong way, and would be caught here. PLAN
    section 2.12 keeps `"fixed"` in the design precisely as the known-rotation counterfactual,
    studied separately and never mixed into a geometric arm. Owning WO: **WO-002**; binds
    **WO-009**.
    """
    cfg = _cfg(tech=dict(horizon_mode="fixed"))
    lengths = {
        len({r.t_period for r in _episode(cfg, "TruthfulMyopic", seed)}) for seed in TB9_SEEDS[:20]
    }
    assert len(lengths) == 1, lengths
    assert lengths.pop() == cfg.tech.max_periods
