"""Observation construction: the enumerated agent-facing vector and its information invariants.

Realises: PLAN section 2.4 (observation table, information invariants) together with the phase
structure of PLAN section 2.5 that the mask keys on. Owning work order: **WO-008** (Observation;
MID-fast).

THE TABLE BELOW IS THE INTERFACE. `OBS_TABLE`, `SCALAR_FIELDS` and `PER_GOOD_BLOCKS` are the PLAN
section 2.4 index table transcribed as data, not as documentation: `obs_spec` composes its answer
from them, `tests/unit/test_obs.py` compares the layout against them, and the PPO adapter (WO-017)
slices the observation by them. They are therefore real content in a skeleton module - changing a
name, a width or an order changes the agent interface, and after the v1 freeze it needs a
`spec/CHANGELOG.md` entry (CONTRACT rule 1). The Phase-1 dimension is `12 + 3 * J` with
`J = cfg.supply.n_sectors`.

CONTRACT RULE 6 (WELFARE BLINDNESS) is what this module exists to enforce at the agent boundary.
`NEVER_OBSERVED` below lists, as data, every quantity that may not appear in any observation in any
phase: `welfare_true`, `val_measured`, any other enterprise's `y`, `S` or `X`, the audit selection
for the current period, and periods remaining under geometric termination. Test T-B5 in
`tests/behavioural/test_welfare_blindness.py` feeds sentinel values into exactly those fields and
asserts that no sentinel appears anywhere in any observation, and it iterates `NEVER_OBSERVED` to do
it. Two of the entries are subtler than they look:

  * The AUDIT SELECTION for the current period is forbidden, not audits in general. An agent learns
    that it was audited only through `last_audited` and `last_penalty_scaled`, which are last
    period's outcome; it never sees this period's selection before it acts, which is what makes the
    audit an actual gamble (PLAN sections 2.4, 2.7.4).
  * PERIODS REMAINING is forbidden because the horizon is geometric (PLAN section 2.12, finding
    F4): there is no end-game to learn, and any field that correlates with periods remaining
    smuggles one back in. Test T-B9 regresses every observation field on periods remaining and
    requires a coefficient of approximately zero.

CONTRACT RULE 9 (RNG). The one stochastic term here is the self-observation noise of PLAN section
2.4: `information.self_obs_noise` multiplies FIELDS 4 AND 5 - `cum_output_over_target` and
`stock_over_target` - by `exp(N(0, sigma**2))`, drawn through `gosplan.rng.draw` with purpose
`selfobs` and key `(seed_env, "selfobs", t, k, i)`. Phase 1 sets it to 0.0, so those two fields are
exact and the agent reports under full knowledge of its own truth (PLAN section 2.8); the branch
must exist regardless, because `self_obs_noise` is a mechanism toggle of the shape study over
[0, 0.05]. No other field is ever noised, and no direct `numpy.random` or `jax.random` call may
appear anywhere in `gosplan/env/`.

Binding to the frozen interface. `obs_spec` carries the name, argument name and return type of
`spec.spec.obs_spec` exactly. `build_observation` and `phase_mask` are not in `spec/spec.py` v0;
they are declared here for the first time and the lead records them in `spec/CHANGELOG.md` at the v1
freeze (WO-013). `spec/spec.py` is not an importable package, so the runtime dataclasses live in the
`gosplan` package - `EnvConfig` and the arm configs in `gosplan/config.py` (WO-003), `State` in
`gosplan/env/state.py` (WO-009) - and each MUST stay field-for-field identical to its `spec/spec.py`
declaration, which `tests/unit/test_spec_imports.py` enforces. They are imported under
`TYPE_CHECKING` so this module stays importable while its siblings are skeletons.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - types only; see the binding note in the module docstring
    from gosplan.config import EnvConfig, Phase
    from gosplan.env.state import State

Array = np.ndarray
"""Alias for every numeric array in this module (PLAN section 10), mirroring `spec.spec.Array`. The
Phase-2 JAX port (WO-029) substitutes its own array type behind the same name, so no signature here
may depend on a numpy-only method."""


SCALAR_FIELDS: tuple[str, ...] = (
    "phase",
    "k_over_M",
    "log_target_ratio",
    "growth_directive",
    "cum_output_over_target",
    "stock_over_target",
    "capital_ratio",
    "last_report_ratio",
    "last_audited",
    "last_penalty_scaled",
    "last_fill",
    "inputs_delivered_total",
)
"""Indices 0-11 of the observation: the one-per-enterprise scalar fields of the PLAN section 2.4
table, in table order. Length `N_SCALAR_FIELDS`. This tuple is the layout, not a description of it -
`obs_spec` returns it followed by the expanded per-good blocks, and `tests/unit/test_obs.py`
compares against it element by element."""

PER_GOOD_BLOCKS: tuple[tuple[str, str], ...] = (
    ("input_cov_{j}", "12 : 12+J"),
    ("sector_onehot_{j}", "12+J : 12+2J"),
    ("deliv_cov_{j}", "12+2J : 12+3J"),
)
"""The three length-`J` blocks that follow the scalar fields, in table order: `(name_template,
index_span)`. Each template is formatted with `j` running over `0 .. J-1`, so at `J = 5` the first
block expands to `input_cov_0 .. input_cov_4` at indices 12-16. The spans are recorded as text
because they are expressions in `J`; the authoritative arithmetic is `N_SCALAR_FIELDS + b * J` for
block `b`, and `tests/unit/test_obs.py` checks the two agree."""

N_SCALAR_FIELDS: int = 12
"""Number of scalar observation fields (PLAN section 2.4, indices 0-11). Equal to
`len(SCALAR_FIELDS)`; declared as a named constant so no function body needs the literal 12."""

N_PER_GOOD_BLOCKS: int = 3
"""Number of length-`J` blocks after the scalar fields (PLAN section 2.4). Equal to
`len(PER_GOOD_BLOCKS)`; the Phase-1 observation dimension is
`N_SCALAR_FIELDS + N_PER_GOOD_BLOCKS * J = 12 + 3J`."""

OBS_TABLE: tuple[tuple[str, str, str], ...] = (
    ("0", "phase", "0 produce / 1 report"),
    ("1", "k_over_M", "step within the period, k / M"),
    ("2", "log_target_ratio", "log(T_i / T_0), own target"),
    ("3", "growth_directive", "g; constant per config, permits config-conditioned policies later"),
    ("4", "cum_output_over_target", "own true production so far this period / T_i"),
    ("5", "stock_over_target", "S_i / T_i, own inventory"),
    ("6", "capital_ratio", "Kap_i / Kap_0"),
    ("7", "last_report_ratio", "rho_i as reported at the last REPORT step"),
    ("8", "last_audited", "whether i was audited last period"),
    ("9", "last_penalty_scaled", "last_penalty * reward_scale(cfg)"),
    ("10", "last_fill", "fraction of own last claimed output actually delivered"),
    ("11", "inputs_delivered_total", "inputs delivered this period / need, totals over goods"),
    ("12 : 12+J", "input_cov_{j}", "X_ij / need_ij, input coverage per good"),
    ("12+J : 12+2J", "sector_onehot_{j}", "sector identity, for parameter sharing"),
    ("12+2J : 12+3J", "deliv_cov_{j}", "deliv_ij / need_ij, deliveries this period per good"),
    (
        "P2 (gated by horizontal_visibility)",
        "peer_report_ratio_{i}",
        "other enterprises' last report ratios in own sector; absent in Phase 1",
    ),
)
"""The PLAN section 2.4 index table verbatim, as `(index_span, field_name, note)`. The final row is
the Phase-2 block gated by `information.horizontal_visibility`: it is NOT part of the Phase-1 layout
and `obs_spec` omits it while `horizontal_visibility == 0`, which is why the Phase-1 dimension is
exactly `12 + 3J`. Kept beside the machine-usable tuples above so a reader can check one against the
other without opening PLAN.md."""

NEVER_OBSERVED: tuple[str, ...] = (
    "welfare_true",
    "val_measured",
    "other_enterprise_output",
    "other_enterprise_stock",
    "other_enterprise_inputs",
    "audit_selection_current_period",
    "periods_remaining",
)
"""Quantities that may not appear in ANY observation, in any phase (PLAN section 2.4; CONTRACT rule
6). Held as data so `tests/behavioural/test_welfare_blindness.py` (T-B5) can iterate it, planting a
sentinel in each and asserting the sentinel appears in no observation the environment produces. The
Phase-2 peer block of `OBS_TABLE` is not a counterexample to the `other_enterprise_*` entries: it
exposes other enterprises' *last report ratios*, which are claims, never their true `y`, `S` or
`X`."""


def obs_spec(cfg: EnvConfig) -> list[str]:
    """Return the ordered names of the observation vector's components (PLAN section 2.4).

    Takes: `cfg`. Returns: a `list[str]` of length `N_SCALAR_FIELDS + N_PER_GOOD_BLOCKS * J` -
    `12 + 3 * J` in Phase 1, with `J = cfg.supply.n_sectors` - built as `list(SCALAR_FIELDS)`
    followed by each block of `PER_GOOD_BLOCKS` expanded over `j = 0 .. J-1` using the block's name
    template, in the order of `OBS_TABLE`:

        0                "phase"                     0 produce / 1 report
        1                "k_over_M"                  step within the period
        2                "log_target_ratio"          log(T_i / T_0)
        3                "growth_directive"          g; constant per config
        4                "cum_output_over_target"    own true production so far this period; exact
                                                     when self_obs_noise = 0
        5                "stock_over_target"         S_i / T_i
        6                "capital_ratio"             Kap_i / Kap_0
        7                "last_report_ratio"
        8                "last_audited"
        9                "last_penalty_scaled"       last_penalty * reward_scale(cfg)
        10               "last_fill"                 fraction of own last claim actually delivered
        11               "inputs_delivered_total"    delivered this period / need, totals
        12      : 12+J   "input_cov_{j}"             X_ij / need_ij, per good
        12+J   : 12+2J   "sector_onehot_{j}"         sector identity, for parameter sharing
        12+2J  : 12+3J   "deliv_cov_{j}"             deliv_ij / need_ij this period, per good
        (P2, gated by `horizontal_visibility`) other enterprises' last report ratios in own sector

    The Phase-2 peer block is appended only when `cfg.information.horizontal_visibility > 0`; at the
    Phase-1 default of 0.0 the returned length is exactly `12 + 3J`. The answer is a pure function
    of the configuration and must not change within a run, because the PPO adapter fixes its input
    dimension from it once at construction (WO-017).

    NEVER PRESENT, IN ANY PHASE, is every entry of `NEVER_OBSERVED`: `welfare_true`,
    `val_measured`, any other enterprise's `y`, `S` or `X`, the audit selection for the current
    period, and periods remaining under geometric termination (PLAN sections 2.4, 2.12; CONTRACT
    rule 6).

    Binds: `tests/unit/test_obs.py` (the layout equals this list; the dimension is `12 + 3J`; phase
    masking) and test T-B5 in `tests/behavioural/test_welfare_blindness.py`, which feeds sentinel
    values into the forbidden fields and asserts they appear in no observation. Owning WO:
    **WO-008**.
    """
    raise NotImplementedError("PLAN section 2.4 - implemented in WO-008")


def build_observation(state: State, cfg: EnvConfig, deliv: Array, need: Array) -> Array:
    """Assemble the agent-facing observation for every enterprise (PLAN section 2.4).

    Takes: `state`, the true state at the step about to be acted on; `cfg`; `deliv` `(N, J)`, the
    physical receipts this period returned by `gosplan.env.planner.deliver`; and `need` `(N, J)`,
    the period's planned input need `need_ij = a_{s(i)j} * T_i` - the same quantity the allocation
    weights of PLAN section 2.7.2 key on. Both are passed in by the caller
    (`gosplan/env/step.py`, WO-009) rather than recomputed here, so the observation can never
    disagree with the delivery that produced it. Returns: `obs` `(N, d)` with
    `d = len(obs_spec(cfg))`, i.e. `12 + 3J` in Phase 1, in the exact order `obs_spec` declares.

    Component by component, per enterprise `i` with `M = cfg.incentive.steps_per_period`,
    `T_0 = cfg.tech.initial_target_frac * A_{s(i)} * cap_i` and `Kap_0` the initial capital:

        0  phase                    0.0 at a PRODUCE step, 1.0 at the REPORT step
        1  k_over_M                 state.k_step / M
        2  log_target_ratio         log(state.target[i] / T_0[i])
        3  growth_directive         cfg.incentive.growth_directive, the same value for every i
        4  cum_output_over_target   state.cum_output[i] / state.target[i]
        5  stock_over_target        state.inv_output[i] / state.target[i]
        6  capital_ratio            state.capital[i] / Kap_0[i]
        7  last_report_ratio        state.last_report_ratio[i]
        8  last_audited             state.last_audited[i] as 0.0 / 1.0
        9  last_penalty_scaled      state.last_penalty[i] * reward_scale(cfg)   # gosplan.env.reward
        10 last_fill                state.last_fill[i]
        11 inputs_delivered_total   sum_j deliv[i, j] / sum_j need[i, j]
        12   : 12+J  input_cov_j    state.inv_inputs[i, j] / need[i, j]
        12+J : 12+2J sector_onehot  1.0 at j = cfg.supply.sector_of[i], else 0.0
        12+2J: 12+3J deliv_cov_j    deliv[i, j] / need[i, j]

    `need = 0` GIVES A COVERAGE FIELD OF 1.0 (WO-008 card). Wherever `need[i, j] == 0` - a good the
    enterprise's row of `a` does not call for - fields `input_cov_j` and `deliv_cov_j` are 1.0, not
    a division by zero, not a NaN and not 0.0; a good that is not needed is fully covered by
    definition, exactly as `coverage` in `gosplan/env/production.py` treats it (PLAN section 2.6).
    The same rule applies to field 11 when `sum_j need[i, j] == 0`.

    SELF-OBSERVATION NOISE (PLAN section 2.4). When `cfg.information.self_obs_noise = sigma > 0`,
    FIELDS 4 AND 5 ONLY - `cum_output_over_target` and `stock_over_target` - are multiplied by
    `exp(N(0, sigma**2))`, drawn through `gosplan.rng.draw` with purpose `selfobs` and key
    `(seed_env, "selfobs", t, k, i)` (CONTRACT rule 9). No other field is noised, and the two share
    the layout of the draw so the branch is a single multiply. Phase 1 sets `sigma = 0.0`, which
    makes those fields exact - the agent observes `S_i` and `y_i` exactly at the REPORT step, so the
    report of PLAN section 2.8 is a choice made under full knowledge of the truth.

    CONTRACT RULE 6. Nothing outside the enumerated components may be written into `obs`. In
    particular no element of `NEVER_OBSERVED` may enter it by any route, including derived ones: no
    function of `consumer`, no function of another enterprise's `cum_output`, `inv_output` or
    `inv_inputs`, no field carrying the current period's audit selection, and no field that
    correlates with periods remaining under geometric termination. `state` is read here because
    building the observation is, with `make_planner_view`, one of only two sanctioned readers of the
    true state - and this one reads it strictly on the acting enterprise's own behalf, row by row.

    Binds: `tests/unit/test_obs.py` (layout equals `obs_spec(cfg)`; dimension `12 + 3J`; `need = 0`
    gives 1.0; the `self_obs_noise = 0` case is exact) and test T-B5 in
    `tests/behavioural/test_welfare_blindness.py`. Owning WO: **WO-008**.
    """
    raise NotImplementedError("PLAN section 2.4 - implemented in WO-008")


def phase_mask(cfg: EnvConfig, phase: Phase) -> Array:
    """Return the per-component observation mask for one phase of the period (PLAN sections 2.4,
    2.5).

    Takes: `cfg` and `phase`, one of `"produce"` or `"report"`. Returns: a float mask `(d,)` with
    `d = len(obs_spec(cfg))`, carrying 1.0 for every component the period schedule of PLAN section
    2.5 has already defined at that phase and 0.0 for every component it has not. It is a pure
    function of `(cfg, phase)` - it never touches a `State` - and `build_observation` applies it as
    its final operation, so a masked component is 0.0 rather than stale.

    Phase-1 consequence, stated so an implementer does not go looking for a rule that is not there:
    PLAN section 2.4 tabulates no per-field phase gate, and by the schedule of PLAN section 2.5
    every one of the `12 + 3J` Phase-1 components is already defined at both phases. DELIVER runs at
    the head of the period and fixes `last_fill`, `deliv_cov_j` and `inputs_delivered_total`; the
    `last_*` fields are the previous period's outcome; `cum_output` is a partial sum that is well
    defined from step 0. So THE PHASE-1 MASK IS ALL ONES AT BOTH PHASES, and the phase itself is
    carried by component 0 (`phase`) and component 1 (`k_over_M`) as values rather than by the mask.
    The helper exists so that the Phase-2 blocks that genuinely are gated - the peer block of
    `OBS_TABLE`, gated by `information.horizontal_visibility`, and anything the Phase-2 spec
    revision adds - have exactly one place to be switched off, instead of a conditional scattered
    through `build_observation`.

    A masked component is a component the agent must not condition on; it is never a component whose
    true value is hidden to shape behaviour. Masking is not an information mechanism: the
    information arms of PLAN section 3 are `report_lag`, `aggregation_level`, `channel_noise`,
    `self_obs_noise` and `horizontal_visibility`, and every one of them is applied where it is
    defined, not here.

    Note for the implementer (CONTRACT rules 2 and 3): `tests/unit/test_obs.py` is frozen and checks
    that `build_observation` "masks by phase". If the frozen test expects a convention other than
    the all-ones Phase-1 mask described above - for example a mask over action dimensions rather
    than observation components - file an AMBIGUITY REPORT against WO-008 and stop. Do not
    special-case the implementation to pass it, and do not invent a per-field gate that PLAN section
    2.4 does not state.

    Binds: `tests/unit/test_obs.py` (the mask has length `len(obs_spec(cfg))`; every entry is 0.0 or
    1.0; `build_observation` writes 0.0 in every masked position). Owning WO: **WO-008**.
    """
    raise NotImplementedError("PLAN section 2.4 - implemented in WO-008")
