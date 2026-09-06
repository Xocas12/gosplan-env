"""Planner rules: the planner view, targets, allocation, physical delivery and audit selection.

Realises: PLAN section 2.4 (`PlannerView` and the planner information invariant), PLAN sections
2.7.1 (target rule), 2.7.2 (allocation), 2.7.3 (physical delivery), 2.7.4 (audit selection) and
2.7.5 (aggregation, lag, channel noise), plus PLAN section 2.9.2 (fulfilment measure, which the
target rule keys on). Owning work order: **WO-006** (Planner; MID-strong).

CONTRACT RULE 5 (PLANNER BLINDNESS) IS LOAD-BEARING IN THIS MODULE and is the reason the planner
rules live in a file of their own. `make_planner_view` is the ONLY function in this module that may
take a `State`: it is the single `State -> planner` boundary in the codebase. Every other planner
rule here takes a `PlannerView` and nothing else. A planner function whose signature accepts a
`State` is a contract violation, and test T-B4 (`tests/behavioural/test_planner_blindness.py`)
enforces it statically by inspecting the signatures of every function defined in this module, as
well as dynamically by planting sentinel values in a `State` and asserting that none of them
reaches the returned `PlannerView`.

The one documented exception is `deliver`, which takes a `State` because it is *physical execution*
of an allocation already decided from the view: it makes no planner decision, and reads no claim
except through the `alloc` array handed to it. `spec/spec.py` records that T-B4 whitelists `deliver`
alongside `make_planner_view`, and that the lead may instead relocate `deliver` to
`gosplan/env/step.py` at the v1 freeze (WO-013); either way the whitelist must be recorded in
`spec/CHANGELOG.md`. Do not add a third `State`-taking function here.

THE THREE INFORMATION FILTERS OF PLAN SECTION 2.7.5 must all be implemented, in this order, inside
`make_planner_view`, even though the Phase-1 configuration makes every one of them the identity
map:

    aggregation   `information.aggregation_level`; Phase 1 "enterprise" (identity)
    lag           `information.report_lag`;        Phase 1 0 periods (identity)
    channel noise `information.channel_noise`;     Phase 1 0.0 log-sd (identity)

They are in the frozen signature, so they are implemented now and the *identity case* is what
`tests/unit/test_planner.py` checks (WO-006 card). Writing `if cfg.information.report_lag: raise
NotImplementedError`, or silently skipping a branch because Phase 1 does not exercise it, is a
work-order failure: Phase 2 turns each of them on without touching this file.

CONTRACT RULE 7 (NO HARD-CODED PATHOLOGY). Padding (a claim above stock) and shaving (a claim below
stock) are **consequences** of the four lines of PLAN section 2.7.3, never rules. A claim above
stock lowers `poolfill_j` for the whole good and therefore starves every downstream buyer of that
good; a claim below stock leaves the difference sitting in `S_i`. Neither outcome may be written
into a transition rule, given a branch of its own, detected, rewarded or penalised anywhere in this
module. The same applies to hoarding: at the locked Phase-2 value `alloc_eta_request = 0.7` the
allocation weight of PLAN section 2.7.2 starts to pay attention to requests, and that is a *rule
about weights* - it is never an instruction to inflate a request. The lead reviews every diff to
`gosplan/env/` against this rule, and `tests/behavioural/test_no_hardcoded_pathology.py` (T-B1)
checks it behaviourally.

CONTRACT RULE 9 (RNG). The two stochastic terms in this module - the channel noise of PLAN section
2.7.5 (purpose `channel`) and the audit selection of PLAN section 2.7.4 (purpose `audit`) - go
through `gosplan.rng.draw`. No direct `numpy.random` or `jax.random` call may appear anywhere in
`gosplan/env/`, and no module-level generator may be created.

Binding to the frozen interface. `spec/spec.py` is the frozen interface (CONTRACT rule 1); the
callables below carry its names, argument names, argument order and return types exactly.
`spec/spec.py` is not an importable package, so the runtime dataclasses live in the `gosplan`
package - `EnvConfig` and the arm configs in `gosplan/config.py` (WO-003), `State` and
`EnterpriseAction` in `gosplan/env/state.py` (WO-009), and `PlannerView` here, since PLAN section
12.3 assigns it to WO-006 and WO-006 writes only this file. Each of them MUST stay field-for-field
identical to its `spec/spec.py` declaration; `tests/unit/test_spec_imports.py` enforces that by
comparing field names, order and annotations. The cross-module types are imported under
`TYPE_CHECKING` so this module stays importable while its sibling modules are still skeletons.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - types only; see the binding note in the module docstring
    from gosplan.config import AggLevel, EnvConfig
    from gosplan.env.state import State

Array = np.ndarray
"""Alias for every numeric array in this module (PLAN section 10), mirroring `spec.spec.Array`. The
Phase-2 JAX port (WO-029) substitutes its own array type behind the same name, so no signature here
may depend on a numpy-only method."""


@dataclass(frozen=True)
class PlannerView:
    """Everything the planner is allowed to know (PLAN sections 2.4, 2.7; CONTRACT rule 5).

    Built exclusively by `make_planner_view(state, cfg)` - the single `State -> planner` boundary in
    the codebase. It contains **no true quantity**: no `y`, no `S`, no `X`, no welfare, and no audit
    selection other than the audits already performed. Its contents are aggregated at
    `aggregation_level`, delayed by `report_lag` and perturbed by `channel_noise` before they arrive
    here (PLAN section 2.7.5), so a planner rule cannot recover the truth by inverting anything.

    Frozen, so a view is an immutable snapshot of one period's information and never a mutable
    scratch buffer. `N = cfg.supply.n_enterprises`, `J = cfg.supply.n_sectors`.

    Field-for-field identical to `spec.spec.PlannerView` by construction: the field list, order and
    annotations below are the contract, and `tests/unit/test_spec_imports.py` compares them against
    the frozen interface. Declaring fields is content, not implementation - this record has no
    `__post_init__`, no computed property and no validation.

    Binds: test T-B4 in `tests/behavioural/test_planner_blindness.py` - a state is constructed with
    `S != R` and sentinel `y` values, no sentinel may reach this record, and no function in
    `gosplan/env/planner.py` other than `make_planner_view` (and the whitelisted `deliver`) may
    accept a `State`. Owning WO: **WO-006**.
    """

    claims: Array  # (N,) claimed_i = R_i as it reached the planner, lagged/noised/aggregated
    requests: Array  # (N, J) q_bj, input requests as they reached the planner
    audited: Array  # (N,) bool, who was audited this period; all False before the AUDIT step
    audit_meas: Array  # (N,) S_hat_i, the noisy audit measurement; zeros where not audited
    measured_quality: Array  # (N,) q_hat_i = 1 + mu * (qbar_i - 1) (section 2.9.2); Phase 1 ones
    targets: Array  # (N,) T_i, the planner's own targets - planner-side by construction
    planner_io: Array  # (J, J) the planner's estimate of `a`, possibly stale (section 2.6)
    downstream_shortfall: Array  # (N,) noisy buyer complaints, gated by shortfall_visibility
    aggregation_level: AggLevel  # the level the arrays above are meaningful at (section 2.7.5)
    plan_prices: Array  # (J,) plan prices, needed by the `net_output` measure (section 2.9.2)


def make_planner_view(state: State, cfg: EnvConfig) -> PlannerView:
    """Build the planner's view of the world - the ONLY `State -> planner` function (CONTRACT rule
    5).

    Takes: the true `state` and `cfg`. Returns: a `PlannerView` carrying reports, requests, audit
    results, measured quality, targets, the planner's I-O estimate and (from Phase 2) noisy
    downstream shortfall - and nothing else.

    Source of each field, all of them planner-side quantities already recorded in `State`:

        claims               `state.last_report` (R_i in units), after the three filters below
        requests             `state.request` (N, J), after the same aggregation branch
        audited              `state.last_audited`; all False until `select_audits` has run
        audit_meas           the audit measurement `S_hat` of PLAN section 2.8; zeros where not
                             audited, and zeros entirely until the AUDIT step has run
        measured_quality     q_hat_i = 1 + mu * (qbar_i - 1) (PLAN section 2.9.2), with
                             mu = cfg.information.quality_measurability and qbar from
                             `state.quality_acc`; Phase 1 ones, since `quality_matters` is False
        targets              `state.target` - the planner set them, so they are planner-side
        planner_io           `state.planner_io`, the possibly stale copy of `a` (PLAN section 2.6)
        downstream_shortfall Phase 2 only, scaled by `cfg.information.shortfall_visibility`; Phase 1
                             zeros, which is what makes `audit_mode = "targeted"` inert
        aggregation_level    `cfg.information.aggregation_level`, so a rule can tell which level the
                             arrays it received are meaningful at
        plan_prices          `state.plan_prices` (J,), needed by the `net_output` measure

    The three information filters of PLAN section 2.7.5 are applied here, in this order:

        aggregation   at `aggregation_level = "sector"` the planner sees only
                      `sum_{i in j} claimed_i` - the per-sector total, with per-enterprise identity
                      destroyed - and `allocate` then keys on planned need alone
        lag           the rules consume claims from `report_lag` periods ago; at `report_lag = L`
                      the view issued in period `t` carries the claims of period `t - L`, and
                      periods `t < L` fall back on the initial (zero) claims
        channel noise claimed_i <- claimed_i * exp(xi_i),  xi_i ~ N(0, sigma_ch**2),
                      key = (seed_env, "channel", t, i)   via `gosplan.rng.draw`

    ALL THREE BRANCHES MUST EXIST even though the Phase-1 configuration (`"enterprise"`, `0`, `0.0`)
    makes each of them the identity map; the identity case is exactly what
    `tests/unit/test_planner.py` checks (WO-006 card). The order matters: aggregate, then lag, then
    noise - the noise is what the reporting channel does to whatever actually travels down it.

    The record is built once per period, after the REPORT step. `audited` and `audit_meas` are all
    False / zero until `select_audits` and the audit measurement of PLAN section 2.8 have run, after
    which the view is reissued with them filled; a planner rule therefore never sees this period's
    audit selection before it happens (PLAN section 2.4).

    Constraints: no true quantity may cross this boundary - not `y`, not `S`, not `X`, not
    `cum_output`, not `cum_cost`, not welfare, not `val_true`. The audit measurement `S_hat` is the
    single exception the design makes, and it is deliberately noisy and gated by `audited`.

    Binds: test T-B4 in `tests/behavioural/test_planner_blindness.py` (sentinels planted in the
    state reach no field of the view; the static signature check over this module) and the identity
    cases in `tests/unit/test_planner.py`. Owning WO: **WO-006**.
    """
    raise NotImplementedError("PLAN section 2.4/2.7 - implemented in WO-006")


def update_targets(view: PlannerView, cfg: EnvConfig) -> Array:
    """Apply the ratchet and the growth directive to every target (PLAN section 2.7.1).

    Takes: `view` (claims and current targets as the planner knows them) and `cfg`. Returns: the new
    targets `(N,)`. Takes no `State`: CONTRACT rule 5.

    Formula (PLAN section 2.7.1, verbatim):

        m_i    = fulfilment_measure(view, cfg)             # section 2.9.2; Phase 1: m_i = R_i
        rho_i  = m_i / T_i
        step_i = clip(rho_i - 1, -c_dn, +c_up)
        step_i = 0                       if |rho_i - 1| <= delta   # deadband; Phase 1 delta = 0
        T_i   <- max(T_min, (1 + g) * T_i * (1 + lambda * step_i))

    with, from `cfg.incentive`, `lambda = ratchet_lambda`, `g = growth_directive`,
    `c_up = ratchet_cap_up`, `c_dn = ratchet_cap_dn`, `delta = ratchet_deadband`; `T_i` is
    `view.targets`; and `T_min = cfg.tech.target_floor_frac * T_0`, where
    `T_0 = cfg.tech.initial_target_frac * A_{s(i)} * cap_i` is the *initial* target of PLAN section
    3, not the current one. Note the sign convention on the clip: the downward cap enters as
    `-c_dn`, so both caps are supplied as positive numbers and `EnvConfig.validate` rejects negative
    ones.

    Under `report_lag > 0` the rule uses `m_i` from `report_lag` periods ago - the view has already
    applied the lag, so this function reads `view.claims` as given and applies no lag of its own.
    Under `aggregation_level = "sector"` the measure arrives at sector level and the rule is applied
    to whatever the view carries; no per-enterprise information may be reconstructed here.

    `g > 0` is the forcing term added for finding F1. With `g = 0` and reports at target the map has
    a fixed point - that is the property test T-B2 checks, and it is exactly why `g` must be a
    treatment variable rather than a constant.

    Binds: test T-U4 in `tests/unit/test_planner.py` - fixed point at `rho = 1` with `g = 0`; the
    step bounded by `c_up` / `c_dn`; the floor `T_min` respected; the deadband inert outside
    `|rho - 1| <= delta` - and test T-B2 in `tests/behavioural/test_fixed_point.py` - `Padder` at
    `g = 0` keeps `T` constant, and at `g > 0` `T` grows at exactly `(1 + g)`. Owning WO:
    **WO-006**.
    """
    raise NotImplementedError("PLAN section 2.7.1 - implemented in WO-006")


def fulfilment_measure(view: PlannerView, cfg: EnvConfig) -> Array:
    """Compute the fulfilment quantity the bonus and the ratchet key on (PLAN section 2.9.2).

    Takes: `view` and `cfg`. Returns: `m_i` `(N,)` in units of own good, selected by
    `cfg.incentive.objective_metric`:

        val               m_i = R_i
        net_output        m_i = R_i - sum_j p_j * alloc_ij / p_{s(i)}    # net of allocated inputs
                                                                        # valued at plan prices
        quality_weighted  m_i = R_i * q_hat_i,  q_hat_i = 1 + mu * (qbar_i - 1)

    where `R_i` is `view.claims`, `alloc` is the allocation the planner itself computed for the same
    view via `allocate(view, cfg)`, `p` is `view.plan_prices`, `s(i)` is `cfg.supply.sector_of`, and
    `q_hat` is `view.measured_quality` with `mu = cfg.information.quality_measurability`. Every
    input is planner-side, so this function never needs the true state (CONTRACT rule 5). Phase 1
    is `val`, so `m_i = R_i` exactly and the other two branches are exercised only by their unit
    tests.

    `welfare` is deliberately absent from `ObjectiveMetric` (PLAN section 2.9.2, finding F6): the
    planner cannot key on a quantity it does not observe, and CONTRACT rule 6 keeps `welfare_true`
    out of every decision path. Adding a `welfare` branch here would be a rule-6 violation even if
    no configuration selected it.

    Binds: `tests/unit/test_planner.py` and `tests/unit/test_reward.py` - the `val` branch is the
    identity on `view.claims`; `net_output` falls as allocated inputs rise; `quality_weighted`
    reduces to `val` at `mu = 0`. Owning WO: **WO-006**.
    """
    raise NotImplementedError("PLAN section 2.9.2 - implemented in WO-006")


def allocate(view: PlannerView, cfg: EnvConfig) -> Array:
    """Allocate claimed supply across buyers - promises, not goods (PLAN section 2.7.2).

    Takes: `view` (last period's claims, requests, targets and the planner's I-O estimate) and
    `cfg`. Returns: `alloc` `(N, J)`, the promised quantity of each good to each buyer, in *claimed*
    units. Takes no `State`: CONTRACT rule 5. The planner allocates what it *believes* exists, which
    is the whole reason a claim above stock can be promised away at all.

    Formula (PLAN section 2.7.2, verbatim):

        claimed_i = R_i                                       # lagged/noised/aggregated
        avail_j   = sum_{i: s(i)=j} (1 - phi_j) * claimed_i    # what the planner believes exists
        need_bj   = planner_io[s(b), j] * T_b                  # what the plan says buyer b needs
        w_bj      = (q_bj + 1e-6)**eta_q * (need_bj + 1e-6)**eta_n     # P1: eta_q = 0, eta_n = 1
        alloc_bj  = avail_j * w_bj / sum_b w_bj

    with `phi_j = cfg.supply.final_demand_share` (the fraction routed to the consumer sink and so
    not available to intermediate buyers), `eta_q = cfg.incentive.alloc_eta_request`,
    `eta_n = cfg.incentive.alloc_eta_need`, `q_bj = view.requests`, `T_b = view.targets`,
    `s(b) = cfg.supply.sector_of[b]` and `planner_io = view.planner_io`. The `1e-6` regularisers are
    part of the formula, not a numerical convenience: they keep the weights finite and the
    normalisation well defined when a need or a request is zero.

    At `eta_q = 0` the request term is exactly 1 and requests are ignored, which is the Phase-1
    design (finding F7): requests are logged but inert. At the locked Phase-2 value `eta_q = 0.7`
    requests start to pay - that is the hoarding mechanism of PLAN section 4.2, and it is a *rule
    about weights*, never an instruction to inflate a request (CONTRACT rule 7).

    Under `aggregation_level = "sector"` the weight collapses to the need term alone, because the
    planner no longer holds per-enterprise claims to key on; `avail_j` is then formed from the
    sector totals the view carries.

    Binds: `tests/unit/test_planner.py` - the allocation sums to `avail_j` per good; `eta_q = 0`
    makes the result exactly invariant to `view.requests`; the `1e-6` regularisers keep the weights
    finite when a need or a request is zero. Owning WO: **WO-006**.
    """
    raise NotImplementedError("PLAN section 2.7.2 - implemented in WO-006")


def deliver(state: State, alloc: Array, cfg: EnvConfig) -> tuple[State, Array, Array, Array]:
    """Turn promises into physical goods (PLAN section 2.7.3) - the padding-to-shortage channel.

    Takes: `state` (the true stocks), `alloc` `(N, J)` from `allocate`, and `cfg`. Returns:
    `(state, deliv, fill, consumer)` - the updated state, physical receipts `deliv` `(N, J)`, the
    per-seller fill ratio `fill` `(N,)`, and the consumer sink's receipts `consumer` `(J,)`.

    Formulas (PLAN section 2.7.3, verbatim):

        fill_i     = min(1, S_i / claimed_i)        (fill_i = 1 when claimed_i = 0)
        shipped_i  = min(S_i, claimed_i)
        poolfill_j = sum_{i in j} fill_i * claimed_i / sum_{i in j} claimed_i
        deliv_bj   = alloc_bj * poolfill_j                       # physical receipt
        X_bj      += deliv_bj * qbar_j                           # quality-routed; Phase 1 qbar = 1
        consumer_j = sum_{i in j} phi_j * shipped_i * qbar_i
        S_i       -= shipped_i

    with `claimed_i = state.last_report` (the claim the seller actually made, in units),
    `phi_j = cfg.supply.final_demand_share`, and `qbar` the period-average quality from
    `state.quality_acc` (Phase 1 ones, since `supply.quality_matters` is False). The degenerate case
    `sum_{i in j} claimed_i = 0` gives `poolfill_j = 1`, consistent with `claimed_i = 0` giving
    `fill_i = 1`. `poolfill_j` lies in [0, 1] by construction and must not be clipped into the
    interval: a value outside it is a bug to be found, not a bound to be applied.

    CONTRACT RULE 7. A claim above stock lowers `poolfill_j` for the whole good and therefore
    reduces every downstream buyer's receipt; a claim below stock leaves the difference sitting in
    `S_i`. Both are CONSEQUENCES of these four lines, never rules of their own: no branch in this
    function may test for, name, reward or penalise either case, and every `gosplan/env/` diff is
    reviewed against that rule.

    Interface note for the v1 freeze (WO-013): this signature takes a `State` and lives in
    `gosplan/env/planner.py`, while CONTRACT rule 5 and the T-B4 static check forbid any function in
    this module except `make_planner_view` from accepting a `State`. `deliver` is physical execution
    of an allocation already decided from the view - it makes no planner decision and reads no claim
    except through `alloc` and the seller's own recorded claim - so T-B4 whitelists it alongside
    `make_planner_view`. The lead records the whitelist, or relocates `deliver` to
    `gosplan/env/step.py`, in `spec/CHANGELOG.md` at the v1 freeze.

    Binds: `tests/unit/test_planner.py` (`poolfill` in [0, 1]; delivery conservation - what leaves
    `S` equals what reaches buyers plus the consumer sink; `claimed = 0` gives `fill = 1`), test
    T-U1, the per-period conservation identity to 1e-9, and test T-B3 in
    `tests/behavioural/test_shortage_propagation.py` (a `Padder` with `S = 0` produces `fill < 1`
    for every downstream buyer; `TruthfulMyopic` produces `fill = 1`). Owning WO: **WO-006**.
    """
    raise NotImplementedError("PLAN section 2.7.3 - implemented in WO-006")


def select_audits(view: PlannerView, cfg: EnvConfig, t: int) -> Array:
    """Choose which enterprises to audit this period (PLAN section 2.7.4).

    Takes: `view`, `cfg`, and the plan period `t` (which keys the draw). Returns: a boolean `(N,)`.
    Takes no `State`: CONTRACT rule 5 - the selection is made from planner-side information alone.

    Formulas (PLAN section 2.7.4, verbatim):

        random    (Phase 1)  audited_i ~ Bernoulli(a),  key = (seed_env, "audit", t, i)
        targeted  (Phase 2)  probability a * (1 + kappa_t * downstream_shortfall_i),
                             clipped to [0, 1], active only when shortfall_visibility > 0

    with `a = cfg.information.audit_rate` and the branch chosen by `cfg.information.audit_mode`.
    Both branches must exist. Phase 1 uses `random`; `targeted` is inert in Phase 1 because
    `view.downstream_shortfall` is all zeros while `shortfall_visibility = 0`, which makes the two
    branches agree there. `kappa_t` is the targeting-strength coefficient of PLAN section 2.7.4,
    fixed by WO-023 when the targeted mode is switched on and never chosen here.

    `downstream_shortfall_i` is the planner's noisy knowledge of buyers' complaints and is therefore
    an information quantity, which is why `audit_mode` sits in `InformationConfig` even though the
    audit rate also enters the reward through the penalty (dual classification, PLAN section 3;
    `audit_rate` is always reported separately and never folded into the C_OGAS contrast).

    Constraints: the draw goes through `gosplan.rng.draw` with purpose `audit` (CONTRACT rule 9), so
    the selection is deterministic in `(seed_env, t)` and independent of call order; and the
    selection is never observable to any agent before it happens - it reaches an agent only through
    the `last_audited` observation field of the *following* period (PLAN section 2.4).

    Binds: `tests/unit/test_planner.py` - the empirical audit frequency matches `audit_rate` over
    many periods, and the selection is deterministic in `(seed_env, t)`. Owning WO: **WO-006**.
    """
    raise NotImplementedError("PLAN section 2.7.4 - implemented in WO-006")
