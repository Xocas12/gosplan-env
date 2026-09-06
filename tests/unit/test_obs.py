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

import pytest


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-008")
def test_obs_spec_equals_the_plan_section_2_4_layout(p1_cfg) -> None:
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
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-008")
def test_observation_dimension_is_twelve_plus_three_j(p1_cfg, tiny_cfg) -> None:
    """The Phase-1 observation dimension is `12 + 3J`.

    Assertion: `len(obs_spec(cfg)) == N_SCALAR_FIELDS + N_PER_GOOD_BLOCKS * J == 12 + 3 * J` - 27 at
    `p1_cfg` (`J = 5`) and 18 at `tiny_cfg` (`J = 2`) - and `build_observation` returns an array of
    shape `(N, 12 + 3J)`. The Phase-2 peer block gated by `information.horizontal_visibility` is
    absent while that parameter is 0, which is exactly why the Phase-1 dimension is `12 + 3J` and
    not more.

    Second bullet of the WO-008 must-pass list.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-008")
def test_each_observation_component_holds_the_quantity_the_table_names(p1_cfg) -> None:
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
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-008")
def test_coverage_fields_are_one_where_need_is_zero(p1_cfg) -> None:
    """`need = 0` gives a coverage field of 1.0 - never 0.0, NaN or a division by zero.

    Assertion: for every `(i, j)` with `need[i, j] == 0` - a good the enterprise's row of `a` does
    not call for, three of five in Phase 1 - both `input_cov_j` and `deliv_cov_j` are exactly 1.0;
    and when `sum_j need[i, j] == 0` the total field 11 is exactly 1.0. No NaN and no warning
    appears anywhere in the observation. This is the same convention `coverage` uses in
    `gosplan/env/production.py` (a good that is not needed is fully covered by definition), stated
    as a card requirement in WO-008.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-008")
def test_phase_mask_has_the_layout_length_and_is_all_ones_in_phase_1(p1_cfg) -> None:
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
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-008")
def test_self_observation_noise_is_exact_at_zero_and_touches_only_fields_four_and_five(
    p1_cfg, rng_seed
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
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-008")
def test_no_forbidden_quantity_appears_in_the_layout(p1_cfg) -> None:
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
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-008")
def test_sector_one_hot_identifies_the_sector_for_parameter_sharing(p1_cfg) -> None:
    """The one-hot block carries sector identity and nothing else.

    Assertion: for every enterprise `i` the block `12+J : 12+2J` sums to exactly 1.0, is 1.0 at
    index `cfg.supply.sector_of[i]` and 0.0 elsewhere, and is identical for two enterprises in the
    same sector. It is what lets one shared policy act for all `N` enterprises under
    `param_sharing = "shared"` (PLAN sections 2.4, 3, 6.1) without the policy being told which
    enterprise it is - the observation identifies a *sector*, never an enterprise index.
    """
    assert False
