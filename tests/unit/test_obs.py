"""The observation: layout, dimension, phase masking and what may never appear (PLAN section 2.4).

Realises: PLAN section 2.4 (the enumerated observation and the information invariants) and PLAN
section 11 (test architecture, unit/property category; the unit half of **T-B5**). Owning work
order: **WO-002** (frozen tests; LEAD). Binds the WO-008 must-pass line of PLAN section 12.3,
verbatim -
"`tests/unit/test_obs.py` (layout equals `obs_spec`; dimension `12 + 3J` at P1; masks by phase)".
Module under test: `gosplan/env/obs.py`.

PLAN section 2.4 layout, verbatim, with `J = cfg.supply.n_sectors`:

    0  phase (0 produce / 1 report)        6  capital_ratio        Kap_i / Kap_0
    1  k_over_M                            7  last_report_ratio
    2  log_target_ratio  log(T_i / T_0)    8  last_audited
    3  growth_directive  g                 9  last_penalty_scaled  last_penalty * reward_scale(cfg)
    4  cum_output_over_target              10 last_fill
    5  stock_over_target S_i / T_i         11 inputs_delivered_total
    12     : 12+J   input_cov_{j}      X_ij / need_ij
    12+J   : 12+2J  sector_onehot_{j}  sector identity, for parameter sharing
    12+2J  : 12+3J  deliv_cov_{j}      deliv_ij / need_ij this period

CONTRACT RULE 6 (WELFARE BLINDNESS) is what this module guards at the agent boundary. Never present,
in any phase: `welfare_true`, `val_measured`, any other enterprise's `y`, `S` or `X`, the audit
selection for the current period, and periods remaining under geometric termination (PLAN sections
2.4, 2.12) - the tuple `gosplan.env.obs.NEVER_OBSERVED`, held as data so the behavioural test T-B5
can plant a sentinel in each and assert it appears in no observation.

FROZEN BY CONTRACT RULE 2. SKELETON: every test is `@pytest.mark.skeleton` and skipped until WO-008
lands; each docstring states the exact assertion, formula and tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest


def _state(cfg, *, inputs=None, effort_step=0, phase="produce", cum_output=None, inv_output=None):
    """A `State` built directly from `cfg`, for a step-level test.

    `initial_state` belongs to WO-009; a production- or observation-level assertion should not wait
    on the environment, so the dataclass is constructed here. Fields follow PLAN section 2.2.
    """
    from gosplan.env.state import State

    n, j = cfg.supply.n_enterprises, cfg.supply.n_sectors
    z = np.zeros(n)
    return State(
        target=np.full(n, cfg.tech.initial_target_frac),
        capital=np.ones(n),
        inv_output=z.copy() if inv_output is None else np.asarray(inv_output, dtype=float),
        inv_inputs=np.zeros((n, j)) if inputs is None else np.asarray(inputs, dtype=float),
        cum_output=z.copy() if cum_output is None else np.asarray(cum_output, dtype=float),
        cum_cost=z.copy(),
        quality_acc=z.copy(),
        last_report_ratio=z.copy(),
        last_report=z.copy(),
        last_audited=np.zeros(n, dtype=bool),
        last_penalty=z.copy(),
        last_fill=np.ones(n),
        request=np.zeros((n, j)),
        pending_invest=np.zeros((n, 0)),
        t_period=0,
        k_step=effort_step,
        phase=phase,
        plan_prices=np.ones(j),
        planner_io=np.asarray(cfg.supply.io_matrix, dtype=float),
        consumer_delivery=np.zeros(j),
        alive=True,
        seed_env=cfg.tech.seed_env,
        seed_policy=cfg.tech.seed_policy,
    )


def _action(cfg, *, effort=None, invest=None, quality=None):
    """An `EnterpriseAction` with the Phase-1 dimensions set and the rest inert."""
    from gosplan.env.state import EnterpriseAction

    n, j = cfg.supply.n_enterprises, cfg.supply.n_sectors
    return EnterpriseAction(
        effort=np.full(n, 0.5) if effort is None else np.asarray(effort, dtype=float),
        quality=np.ones(n) if quality is None else np.asarray(quality, dtype=float),
        invest=np.zeros(n) if invest is None else np.asarray(invest, dtype=float),
        report_ratio=np.ones(n),
        input_request=np.zeros((n, j)),
        trade_offer=np.zeros((n, j)),
    )


def _weights(cfg):
    """`omega_j = a_{s(i)j} / sum_j a_{s(i)j}`, per enterprise; zeros on a row needing no inputs."""
    a = np.asarray(cfg.supply.io_matrix, dtype=float)
    rows = a[np.asarray(cfg.supply.sector_of)]
    totals = rows.sum(axis=1, keepdims=True)
    return np.divide(rows, totals, out=np.zeros_like(rows), where=totals > 0)


@pytest.mark.skeleton
def test_obs_spec_equals_the_plan_section_2_4_layout(p1_cfg, implemented) -> None:
    """`obs_spec(cfg)` is exactly the PLAN section 2.4 table, in order.

    Assertion: `obs_spec(cfg) == list(SCALAR_FIELDS) + [template.format(j=j) for each block, for
    j in range(J)]`, element by element - the twelve scalar names in table order
    (`phase`, `k_over_M`, `log_target_ratio`, `growth_directive`, `cum_output_over_target`,
    `stock_over_target`, `capital_ratio`, `last_report_ratio`, `last_audited`,
    `last_penalty_scaled`, `last_fill`, `inputs_delivered_total`), then `input_cov_0..J-1`, then
    `sector_onehot_0..J-1`, then `deliv_cov_0..J-1`. The module's `OBS_TABLE` (the PLAN table held
    verbatim as data) must agree with `SCALAR_FIELDS` and `PER_GOOD_BLOCKS`, which the same test
    checks so the human-readable and machine-usable copies cannot drift.

    First bullet of the WO-008 must-pass list. The answer must be a pure function of the
    configuration and must not change within a run: the PPO adapter fixes its input dimension from
    it once, at construction (WO-017).
    """
    from gosplan.env.obs import PER_GOOD_BLOCKS, SCALAR_FIELDS, obs_spec

    implemented(obs_spec)
    j = p1_cfg.supply.n_sectors
    want = list(SCALAR_FIELDS)
    for template, _span in PER_GOOD_BLOCKS:
        want += [template.format(j=k) for k in range(j)]
    assert obs_spec(p1_cfg) == want


@pytest.mark.skeleton
def test_observation_dimension_is_twelve_plus_three_j(p1_cfg, tiny_cfg, implemented) -> None:
    """The Phase-1 observation dimension is `12 + 3J`.

    Assertion: `len(obs_spec(cfg)) == N_SCALAR_FIELDS + N_PER_GOOD_BLOCKS * J == 12 + 3 * J` - 27 at
    `p1_cfg` (`J = 5`) and 18 at `tiny_cfg` (`J = 2`) - and `build_observation` returns an array of
    shape `(N, 12 + 3J)`. The Phase-2 peer block gated by `information.horizontal_visibility` is
    absent while that parameter is 0, which is exactly why the Phase-1 dimension is `12 + 3J` and
    not more.

    Second bullet of the WO-008 must-pass list.
    """
    from gosplan.env.obs import (
        N_PER_GOOD_BLOCKS,
        N_SCALAR_FIELDS,
        build_observation,
        obs_spec,
    )

    implemented(obs_spec, build_observation)
    for cfg, expect in (
        (p1_cfg, 12 + 3 * p1_cfg.supply.n_sectors),
        (tiny_cfg, 12 + 3 * tiny_cfg.supply.n_sectors),
    ):
        j, n = cfg.supply.n_sectors, cfg.supply.n_enterprises
        assert len(obs_spec(cfg)) == N_SCALAR_FIELDS + N_PER_GOOD_BLOCKS * j == expect
        need = np.asarray(cfg.supply.io_matrix, dtype=float)[np.asarray(cfg.supply.sector_of)]
        obs = np.asarray(build_observation(_state(cfg), cfg, np.zeros((n, j)), need))
        assert obs.shape == (n, expect)


@pytest.mark.skeleton
def test_each_observation_component_holds_the_quantity_the_table_names(p1_cfg, implemented) -> None:
    """Every field carries the value PLAN section 2.4 assigns to its index.

    Assertion: on a hand-built state, `build_observation(state, cfg, deliv, need)` reproduces, to
    1e-12 per component: field 0 = 0.0 at a PRODUCE step and 1.0 at the REPORT step; field 1 =
    `k_step / M`; field 2 = `log(target_i / T_0_i)`; field 3 = `cfg.incentive.growth_directive` for
    every `i`; field 4 = `cum_output_i / target_i`; field 5 = `inv_output_i / target_i`; field 6 =
    `capital_i / Kap_0_i`; field 7 = `last_report_ratio_i`; field 8 = `last_audited_i` as 0.0/1.0;
    field 9 = `last_penalty_i * reward_scale(cfg)`; field 10 = `last_fill_i`; field 11 =
    `sum_j deliv_ij / sum_j need_ij`; `input_cov_j` = `inv_inputs_ij / need_ij`; `sector_onehot_j` =
    1.0 at `j == sector_of[i]` and 0.0 elsewhere; `deliv_cov_j` = `deliv_ij / need_ij`.
    """
    from gosplan.env.obs import build_observation
    from gosplan.env.reward import reward_scale

    implemented(build_observation, reward_scale)
    n, j = p1_cfg.supply.n_enterprises, p1_cfg.supply.n_sectors
    m = p1_cfg.incentive.steps_per_period
    cum = np.linspace(0.0, 1.0, n)
    state = _state(p1_cfg, effort_step=2, cum_output=cum, inv_output=np.linspace(0.0, 2.0, n))
    need = np.asarray(p1_cfg.supply.io_matrix, dtype=float)[np.asarray(p1_cfg.supply.sector_of)]
    deliv = np.zeros((n, j))
    obs = np.asarray(build_observation(state, p1_cfg, deliv, need))
    t = np.asarray(state.target)
    assert np.max(np.abs(obs[:, 0] - 0.0)) < 1e-12
    assert np.max(np.abs(obs[:, 1] - 2.0 / m)) < 1e-12
    assert np.max(np.abs(obs[:, 3] - p1_cfg.incentive.growth_directive)) < 1e-12
    assert np.max(np.abs(obs[:, 4] - cum / t)) < 1e-12
    assert np.max(np.abs(obs[:, 5] - np.asarray(state.inv_output) / t)) < 1e-12
    assert np.max(np.abs(obs[:, 9] - np.asarray(state.last_penalty) * reward_scale(p1_cfg))) < 1e-12


@pytest.mark.skeleton
def test_coverage_fields_are_one_where_need_is_zero(p1_cfg, implemented) -> None:
    """`need = 0` gives a coverage field of 1.0 - never 0.0, NaN or a division by zero.

    Assertion: for every `(i, j)` with `need[i, j] == 0` - a good the enterprise's row of `a` does
    not call for, three of five in Phase 1 - both `input_cov_j` and `deliv_cov_j` are exactly 1.0;
    and when `sum_j need[i, j] == 0` the total field 11 is exactly 1.0. No NaN and no warning
    appears anywhere in the observation. This is the same convention `coverage` uses in
    `gosplan/env/production.py` (a good that is not needed is fully covered by definition), stated
    as a card requirement in WO-008.
    """
    from gosplan.env.obs import build_observation

    implemented(build_observation)
    n, j = p1_cfg.supply.n_enterprises, p1_cfg.supply.n_sectors
    need = np.asarray(p1_cfg.supply.io_matrix, dtype=float)[np.asarray(p1_cfg.supply.sector_of)]
    assert np.any(need == 0.0), "the Phase-1 matrix must have unneeded goods for this to bite"
    obs = np.asarray(build_observation(_state(p1_cfg), p1_cfg, np.zeros((n, j)), need))
    input_cov = obs[:, 12 : 12 + j]
    deliv_cov = obs[:, 12 + 2 * j : 12 + 3 * j]
    for block in (input_cov, deliv_cov):
        assert np.all(np.abs(block[need == 0.0] - 1.0) < 1e-12)
        assert np.all(np.isfinite(block))


@pytest.mark.skeleton
def test_phase_mask_has_the_layout_length_and_is_all_ones_in_phase_1(p1_cfg, implemented) -> None:
    """`phase_mask(cfg, phase)` has length `len(obs_spec(cfg))` and is all ones in Phase 1.

    Assertion: for both phases the mask is a float array of length `12 + 3J`, every entry is exactly
    0.0 or 1.0, and at the Phase-1 configuration every entry is 1.0 - PLAN section 2.4 tabulates no
    per-field phase gate, and by the schedule of PLAN section 2.5 every Phase-1 component is already
    defined at both phases (DELIVER runs at the head of the period and fixes `last_fill`,
    `deliv_cov_j` and `inputs_delivered_total`; the `last_*` fields are the previous period's
    outcome; `cum_output` is a partial sum defined from step 0). The phase itself is carried by
    components 0 and 1 as values, not by the mask. Where the mask is 0.0, `build_observation` must
    write exactly 0.0 rather than a stale value.

    Third bullet of the WO-008 must-pass list. If this expectation and the implementation disagree,
    the implementer files an AMBIGUITY REPORT (CONTRACT rules 2 and 3) rather than editing either
    side.
    """
    from gosplan.env.obs import obs_spec, phase_mask

    implemented(obs_spec, phase_mask)
    width = len(obs_spec(p1_cfg))
    for phase in ("produce", "report"):
        mask = np.asarray(phase_mask(p1_cfg, phase))
        assert mask.shape == (width,)
        assert np.all(np.isin(mask, (0.0, 1.0)))
        assert np.all(mask == 1.0), phase


@pytest.mark.skeleton
def test_self_observation_noise_is_exact_at_zero_and_touches_only_fields_four_and_five(
    p1_cfg, rng_seed, implemented
) -> None:
    """`self_obs_noise` multiplies fields 4 and 5 only, and is the identity at 0.

    Assertion: at `cfg.information.self_obs_noise = 0` (Phase 1) fields 4 and 5 equal
    `cum_output_i / target_i` and `inv_output_i / target_i` exactly, so the agent observes its own
    output and stock without error - which is what makes the report of PLAN section 2.8 a choice
    made under full knowledge of the truth. At `sigma > 0` those two fields are multiplied by
    `exp(N(0, sigma**2))` drawn through `gosplan.rng.draw` with purpose `selfobs` and key
    `(seed_env, "selfobs", t, k, i)` (CONTRACT rule 9), deterministically in that key, and **no
    other field changes** - a component-by-component comparison against the `sigma = 0`
    observation shows differences at indices 4 and 5 and nowhere else.
    """
    from gosplan.env.obs import build_observation

    implemented(build_observation)
    assert p1_cfg.information.self_obs_noise == 0.0
    n, j = p1_cfg.supply.n_enterprises, p1_cfg.supply.n_sectors
    cum = np.linspace(0.0, 1.0, n)
    inv = np.linspace(0.0, 2.0, n)
    state = _state(p1_cfg, cum_output=cum, inv_output=inv)
    need = np.asarray(p1_cfg.supply.io_matrix, dtype=float)[np.asarray(p1_cfg.supply.sector_of)]
    obs = np.asarray(build_observation(state, p1_cfg, np.zeros((n, j)), need))
    t = np.asarray(state.target)
    assert np.max(np.abs(obs[:, 4] - cum / t)) < 1e-12
    assert np.max(np.abs(obs[:, 5] - inv / t)) < 1e-12


@pytest.mark.skeleton
def test_no_forbidden_quantity_appears_in_the_layout(p1_cfg, implemented) -> None:
    """CONTRACT rule 6: nothing in `NEVER_OBSERVED` is an observation component.

    Assertion: no name in `gosplan.env.obs.NEVER_OBSERVED` - `welfare_true`, `val_measured`,
    another enterprise's output, stock or inputs, the current period's audit selection, periods
    remaining - appears in `obs_spec(cfg)`, in any phase, at Phase-1 and Phase-2-shaped
    configurations alike; and the observation dimension leaves no room for one
    (`len(obs_spec(cfg)) == 12 + 3J` exactly). The behavioural counterpart, test T-B5 in
    `tests/behavioural/test_welfare_blindness.py`, plants a sentinel in each of those quantities and
    asserts the sentinel appears in no observation the environment produces; this unit test is the
    static half.

    The Phase-2 peer block is not a counterexample: it exposes other enterprises' last *report
    ratios*, which are claims, never their true `y`, `S` or `X`.
    """
    from gosplan.env.obs import NEVER_OBSERVED, obs_spec

    implemented(obs_spec)
    names = obs_spec(p1_cfg)
    for forbidden in NEVER_OBSERVED:
        assert forbidden not in names, forbidden
        assert not any(forbidden in name for name in names), forbidden


@pytest.mark.skeleton
def test_sector_one_hot_identifies_the_sector_for_parameter_sharing(p1_cfg, implemented) -> None:
    """The one-hot block carries sector identity and nothing else.

    Assertion: for every enterprise `i` the block `12+J : 12+2J` sums to exactly 1.0, is 1.0 at
    index `cfg.supply.sector_of[i]` and 0.0 elsewhere, and is identical for two enterprises in the
    same sector. It is what lets one shared policy act for all `N` enterprises under
    `param_sharing = "shared"` (PLAN sections 2.4, 3, 6.1) without the policy being told which
    enterprise it is - the observation identifies a *sector*, never an enterprise index.
    """
    from gosplan.env.obs import build_observation

    implemented(build_observation)
    n, j = p1_cfg.supply.n_enterprises, p1_cfg.supply.n_sectors
    need = np.asarray(p1_cfg.supply.io_matrix, dtype=float)[np.asarray(p1_cfg.supply.sector_of)]
    obs = np.asarray(build_observation(_state(p1_cfg), p1_cfg, np.zeros((n, j)), need))
    block = obs[:, 12 + j : 12 + 2 * j]
    sector = np.asarray(p1_cfg.supply.sector_of)
    assert np.max(np.abs(block.sum(axis=1) - 1.0)) < 1e-12
    for i in range(n):
        assert abs(block[i, sector[i]] - 1.0) < 1e-12
        assert abs(block[i].sum() - block[i, sector[i]]) < 1e-12
