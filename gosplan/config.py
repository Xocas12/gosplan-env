"""Environment configuration: the `EnvConfig` family, its validation, its hash and its loaders.

Realises: PLAN section 3 (parameter registry and ranges), with the field semantics of PLAN sections
2.1-2.15 and the manifest requirement of CONTRACT rule 10. Owning work order: **WO-003** (Config;
P1; MID-fast; depends on WO-001). Must pass `tests/unit/test_config.py`; completion command
`pytest tests/unit/test_config.py -q`.

**Mirror obligation.** `spec/spec.py` is the frozen interface (CONTRACT rule 1) but is not an
importable package, so this module declares the runtime dataclasses instead. They MUST stay
field-for-field identical to `spec/spec.py`: same class names, same field names, same order, same
types, same Phase-1 defaults, same `frozen=True`. `tests/unit/test_spec_imports.py` (WO-001) checks
that the public surface matches, and `tests/unit/test_config.py` (WO-003) checks the defaults
against `gosplan.params.REGISTRY`. If a field here and a field there disagree, `spec/spec.py` wins
and the fix is an AMBIGUITY REPORT plus a lead-signed `spec/CHANGELOG.md` entry - never a silent
edit on this side (CONTRACT rules 1, 3).

**Skeleton status.** The dataclass field declarations and their defaults are real content: they are
the Phase-1 configuration of PLAN section 3. Everything executable - `validate`, `hash`,
`load_config`, `p1_default_config` - raises `NotImplementedError` until WO-003 lands. The
dataclasses carry fields only: no `__post_init__`, no computed property, no derived constant.
Derived quantities named in PLAN section 2 (`T_0 = initial_target_frac * A * cap`,
`S_max = inventory_cap_mult * cap`, `T_min = target_floor_frac * T_0`, `omega_j`, `reward_scale`)
are computed by the modules that own their formulas, not here.

Value semantics. Every configuration is frozen and hashable, and every array-valued default is a
`tuple` (never a list or an ndarray), so a configuration can be a dictionary key, `hash()` is
stable, and the whole record is written verbatim into `runs/<hash>/manifest.json` (CONTRACT rule
10). The arm each field belongs to is recorded in its docstring and in `gosplan.params`; changing
an arm requires a CHANGELOG entry (CONTRACT rule 11).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import tomllib
from dataclasses import dataclass, field
from typing import Literal

SPEC_VERSION = "0.1.0"
"""Mirror of `spec.spec.SPEC_VERSION` - the provisional (v0) interface version.

WO-013 bumps it to `"1.0.0"` at gate G1 and every later change needs a `spec/CHANGELOG.md` entry
(CONTRACT rule 1). It is the default of `EnvConfig.spec_version` and is written into every run
manifest (CONTRACT rule 10), so a result can never be silently attributed to a different interface
version. Distinct from `gosplan.__version__`, which versions the distribution."""

ObjectiveMetric = Literal["val", "net_output", "quality_weighted"]
"""Fulfilment measure the bonus and the ratchet key on (PLAN section 2.9.2). `welfare` is
deliberately not an option (finding F6): the planner cannot key on a quantity it does not observe,
and CONTRACT rule 6 keeps `welfare_true` out of every decision path."""

PenaltyForm = Literal["proportional", "fixed"]
"""Shape of the audit penalty (PLAN section 2.8): `Pen = pen * f_i` (`proportional`) or
`pen * 1[f_i > 0]` (`fixed`)."""

PenaltyArg = Literal["positive_part", "absolute"]
"""Argument of the audit penalty (PLAN section 2.8): `f = max(0, R - S_hat) / T`
(`positive_part`, Phase 1, so an under-report incurs no penalty) or `f = |R - S_hat| / T`
(`absolute`, penalising both directions)."""

AuditMode = Literal["random", "targeted"]
"""Audit selection rule (PLAN section 2.7.4). Phase 1 `random`; `targeted` is a Phase-2 information
mechanism gated by `shortfall_visibility` (WO-023)."""

AggLevel = Literal["enterprise", "sector"]
"""Level at which the planner sees reports (PLAN sections 2.4, 2.7.5). At `sector` the planner sees
only the per-sector sum of claims and allocates by planned need alone."""

DeliveryTiming = Literal["uniform", "stochastic", "backloaded"]
"""Within-period arrival of delivered inputs (PLAN section 2.6). Phase 1 `uniform` (everything
arrives at step 0); the Phase-2 storming mechanism uses `stochastic` with `arrival_probs` (PLAN
section 4.2)."""

HorizonMode = Literal["geometric", "fixed"]
"""Episode termination (PLAN section 2.12). `geometric`: after `min_periods`, continue with
probability `tenure` per period up to `max_periods`, so the agent never observes periods remaining
and there is no end-game (finding F4). `fixed` exists to study known rotation separately."""

ParamSharing = Literal["shared", "per_sector", "independent"]
"""Policy-parameter sharing across enterprises (PLAN sections 3, 6.1). Phase 1 `shared`, with the
sector one-hot of the observation (PLAN section 2.4) carrying sector identity."""


@dataclass(frozen=True)
class SupplyConfig:
    """Production and supply structure (PLAN section 3, arm SUPPLY, plus the sizing constants PLAN
    section 10 groups here).

    Mirrors `spec.spec.SupplyConfig` field for field. Frozen and hashable: every array-valued
    default is a `tuple`, so `EnvConfig.hash()` is stable and a configuration can be a dictionary
    key. Field docstrings record the arm (CONTRACT rule 11) and the sweep range of PLAN section 3;
    the ranges themselves are data in `gosplan.params.REGISTRY` and are enforced by
    `EnvConfig.validate` (WO-003).
    """

    n_enterprises: int = 20
    """TECH. `N`, number of enterprises (PLAN section 2.1). Phase 1: 20, four per sector."""

    n_sectors: int = 5
    """TECH. `J`, number of sectors, one good per sector (PLAN section 2.1). Phase 1: 5."""

    sector_of: tuple[int, ...] = (0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4)
    """TECH. `s(i)`, sector of each enterprise; length `n_enterprises`, values in [0, n_sectors).
    Phase 1: four enterprises per sector, contiguous (PLAN section 2.1). Not tabulated in PLAN
    section 3."""

    io_matrix: tuple[tuple[float, ...], ...] = (
        (0.0, 0.2, 0.2, 0.0, 0.0),
        (0.0, 0.0, 0.2, 0.2, 0.0),
        (0.0, 0.0, 0.0, 0.2, 0.2),
        (0.2, 0.0, 0.0, 0.0, 0.2),
        (0.2, 0.2, 0.0, 0.0, 0.0),
    )
    """SUPPLY. `a[j][k]`: units of good `k` needed per unit of good `j`; rows = producing sector,
    columns = input good (PLAN section 2.10). Phase 1: every sector needs two inputs at 0.2 each,
    a 5-cycle with chords, so a shortage in any sector propagates to all. `validate` rejects any
    row with `sum_k a[j][k] >= 1` (no self-sustaining sector). Not tabulated in PLAN section 3."""

    final_demand_share: tuple[float, ...] = (0.5, 0.5, 0.5, 0.5, 0.5)
    """SUPPLY. `phi_j`, fraction of shipped good `j` routed to the consumer sink rather than to
    intermediate buyers (PLAN sections 2.7.3, 2.10). Phase 1: 0.5 for all `j`; range [0.3, 0.7]."""

    productivity: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0)
    """SUPPLY. `A_j`, sector productivity in `y_hat = (A_{s(i)} * cap_i / M) * e` (PLAN section
    2.6). Not tabulated in PLAN section 3; fixed at 1.0 as the normalisation that makes the TECH
    row `T_0 = 0.6 * A * cap` read as `T_0 = 0.6` at `cap = 1`."""

    yield_sigma: tuple[float, ...] = (0.05, 0.08, 0.10, 0.12, 0.15)
    """SUPPLY. `sigma_j`, log-sd of the per-step multiplicative yield shock, one per sector (PLAN
    sections 2.6, 3). Heteroskedastic by construction; the sweep multiplies the whole tuple by
    [0.5, 2]. Source status: unsourced."""

    input_complementarity: float = 8.0
    """SUPPLY. `theta` in the CES coverage aggregator of PLAN section 2.6; `theta = inf` reduces to
    Leontief `min`. Phase 1: 8 (near-Leontief); grid {2, 8, inf}. `validate` rejects
    `theta < 1`."""

    setup_cost: float = 0.0
    """SUPPLY. `F`, fixed cost charged once per step when `e > 0` (PLAN section 2.6). Phase 1
    toggle off (0.0); Phase-2 range [0, 0.2]."""

    irs_alpha: float = 0.0
    """SUPPLY. `alpha_irs` in the increasing-returns toggle
    `A_j(Kap) = A_j * (Kap / Kap_0)**alpha_irs` (PLAN section 2.6). Phase 1 off (0.0); Phase-2
    range [0, 0.3]."""

    capital_dep: float = 0.0
    """SUPPLY. Capital depreciation in `Kap_{t+1} = (1 - dep) * Kap_t + matured investment` (PLAN
    section 2.6). Phase 1 off (0.0); Phase-2 range [0.02, 0.1]."""

    invest_lag: int = 1
    """SUPPLY. Periods between diverting output to investment and its maturing into capital; sets
    the second dimension of `State.pending_invest` (PLAN sections 2.2, 2.6). The Phase-1 cell of
    the PLAN section 3 registry is "-" because investment is inert (`v == 0`); 1 is the minimum
    viable buffer depth. Phase-2 grid {1, 2, 3}; `validate` requires `invest_lag >= 1`."""

    delivery_timing: DeliveryTiming = "uniform"
    """SUPPLY. Within-period arrival of delivered inputs (PLAN section 2.6). Phase 1 `uniform`: all
    inputs are available at step 0. This is the storming mechanism, locked for Phase 2 in PLAN
    section 4.2."""

    arrival_probs: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
    """SUPPLY. `pi_arrival`, distribution of a delivered unit's arrival step over `0..M-1`, used
    only when `delivery_timing != "uniform"` (PLAN section 2.6). Locked Phase-2 value from PLAN
    section 4.2: uniform-random arrival, so no mechanical backloading and any effort-Gini excess is
    behavioural. `validate` requires `len(arrival_probs) == steps_per_period` summing to 1 whenever
    the timing is not `uniform`. Not tabulated in PLAN section 3."""

    holding_loss: float = 0.02
    """SUPPLY. `h`, per-period proportional loss on carried own-good stock, applied before this
    period's output is added: `S <- (1 - h) * S + y` (PLAN sections 2.8, 2.11). Phase 1: 0.02."""

    input_holding_loss: float = 0.0
    """SUPPLY. `h_X`, per-period loss on held input stocks (PLAN section 2.11). Phase 1: 0.0, so
    hoarding carries no direct carrying cost by design. Locked Phase-2 value 0.01 (PLAN section
    4.2)."""

    price_markup: float = 0.1
    """SUPPLY. `m` in the cost-plus price fixed point
    `p_j = (1 + m) * (kappa_labour + sum_k a_jk * p_k)` (PLAN section 2.10), solved once at
    `t = 0` by `initial_prices`. Phase 1: 0.1."""

    price_lag: float = float("inf")
    """SUPPLY. Periods between price recomputations (PLAN section 2.10). Phase 1: `inf`, i.e. plan
    prices are fixed after `t = 0`. Serialised as the string "inf" by `EnvConfig.hash` and decoded
    back by `load_config` (WO-003)."""

    tech_drift_sigma: float = 0.0
    """SUPPLY. `sigma_drift` in the per-period I-O drift `a <- a * exp(zeta)`,
    `zeta ~ N(0, sigma_drift**2)`, while `State.planner_io` stays fixed (PLAN section 2.6). Phase
    1: 0.0; range [0, 0.05]. Acknowledged weak proxy (PLAN section 3)."""

    ces_alpha: tuple[float, ...] = (0.2, 0.2, 0.2, 0.2, 0.2)
    """SUPPLY. `alpha_j`, consumer CES weights in the welfare index of PLAN section 2.9.3. Phase 1:
    `1 / J` for all `j`. Swept only in the price-sensitivity check, never in a treatment arm (PLAN
    sections 2.10, 7.5). Not tabulated in PLAN section 3."""

    ces_sigma: float = 0.8
    """SUPPLY. `sigma_c`, consumer CES elasticity of substitution (PLAN sections 2.9.3, 2.10).
    Phase 1: 0.8, complements-leaning; `validate` requires `ces_sigma > 0`. The `sigma_c -> 1`
    limit is Cobb-Douglas and is tested in `tests/unit/test_reward.py` (WO-007). Not tabulated in
    PLAN section 3."""

    trade_tau: float = 0.05
    """SUPPLY. `tau`, per-unit transaction cost on bilateral trade (PLAN section 2.13). Inert in
    Phase 1 (no trade); locked Phase-2 value 0.05 (PLAN section 4.2). `tau > 0` is what makes wash
    trades unprofitable."""

    quality_matters: bool = False
    """SUPPLY. Master toggle for the quality mechanism: quality routed through the input bundle,
    `X_ij` credited as `deliv * qbar_j` (PLAN section 2.6). Phase 1: False (`q == 1` everywhere).
    Turned on by WO-021. Not tabulated in PLAN section 3."""

    quality_cost: float = 0.0
    """SUPPLY. `kappa_q` in the effort cost `c = kappa * e**2 + F * 1[e > 0] + kappa_q * q * e`
    (PLAN section 2.6). Phase 1: 0.0. Not tabulated in PLAN section 3."""


@dataclass(frozen=True)
class IncentiveConfig:
    """Incentive structure (PLAN section 3, arm INC): the bonus schedule, the audit penalty, the
    ratchet, tenure and the allocation rule's responsiveness.

    Mirrors `spec.spec.IncentiveConfig` field for field. Frozen and hashable. The fields marked
    "provisional" are the daggered rows of the PLAN section 3 registry (`ratchet_lambda`,
    `growth_directive`, `overfulfilment_slope`, `penalty_scale`, `effort_cost`, and `audit_rate` on
    `InformationConfig`): placeholders until a human picks Phase-1 values from the interior of the
    DP regime map at gate G1 and records them in `runs/G1_decision.md` (PLAN sections 5, 13).
    `gosplan.params.provisional()` enumerates them.
    """

    objective_metric: ObjectiveMetric = "val"
    """INC. Which fulfilment measure the bonus and the ratchet key on (PLAN section 2.9.2). Phase
    1: `val`. Historical motivation: `val` is the measure the NNO reforms attacked."""

    ratchet_lambda: float = 0.5  # provisional: replaced at G1
    """INC. `lambda`, ratchet coefficient in the target rule of PLAN section 2.7.1. Range [0, 1];
    Weitzman-type models motivate the form, the empirical value is unsourced."""

    growth_directive: float = 0.02  # provisional: replaced at G1
    """INC. `g`, the exogenous growth directive multiplying the target every period (PLAN section
    2.7.1). The forcing term added for finding F1; it must be a treatment variable because at
    `g = 0` with reports at target the target rule has a fixed point (test T-B2). Range
    [0, 0.07]."""

    ratchet_cap_up: float = 0.3
    """INC. `c_up`, cap on the positive per-period ratchet step (PLAN section 2.7.1). Engineering
    value, fixed, added to bound the transient of finding F4. `validate` rejects a negative cap."""

    ratchet_cap_dn: float = 0.3
    """INC. `c_dn`, cap on the negative per-period ratchet step (PLAN section 2.7.1). Fixed;
    `validate` rejects a negative cap."""

    ratchet_deadband: float = 0.0
    """INC. `delta`: the ratchet step is zeroed when `|rho - 1| <= delta` (PLAN section 2.7.1).
    Phase 1: 0.0 (inert); mechanism toggle for the shape study, range [0, 0.03]."""

    notch_height: float = 1.0
    """INC. `beta`, height of the fulfilment notch in `B(rho)` (PLAN section 2.8). Phase 1: 1.0 -
    it is the normalising unit, so sweeping it changes the notch relative to `pen` and `kappa` and
    not the gradient magnitude (`reward_scale`, PLAN section 2.9.1). Range [0.25, 2]."""

    notch_width: float = 0.0
    """INC. `w`, logistic width of the notch: `Lambda_w(x) = 1[x >= 0]` at `w = 0`, else
    `1 / (1 + exp(-x / w))` (PLAN section 2.8). Phase 1: 0 (a true discontinuity). This is the
    counterfactual knob (smooth arm `w = 0.25`) and the manipulation-strength knob of the
    estimator-bias study (PLAN section 7.2). Grid {0, 0.02, 0.05, 0.10, 0.25}; `validate` rejects
    `w < 0`."""

    overfulfilment_slope: float = 0.5  # provisional: replaced at G1
    """INC. `s`, linear bonus slope above target: `s * clip(rho - 1, 0, rho_cap - 1)` (PLAN section
    2.8). Range [0, 2]; the historical anchor is the per-percentage-point bonus increment (lead to
    source, PLAN section 15)."""

    overfulfilment_cap: float = 1.2
    """INC. `rho_cap`, the ratio at which the overfulfilment bonus stops accruing (PLAN section
    2.8). Phase 1: 1.2. `inf` means no cap and hence no kink - the smooth counterfactual is
    (`notch_width` = 0.25, `overfulfilment_cap` = inf). Grid {1.1, 1.2, inf}; `validate` rejects
    `rho_cap < 1`."""

    penalty_form: PenaltyForm = "proportional"
    """INC. `Pen = pen * f` (`proportional`, Phase 1) or `pen * 1[f > 0]` (`fixed`) (PLAN section
    2.8)."""

    penalty_arg: PenaltyArg = "positive_part"
    """INC. `f = max(0, R - S_hat) / T` (`positive_part`, Phase 1) or `|R - S_hat| / T`
    (`absolute`) (PLAN section 2.8). Under `positive_part` any under-report incurs no penalty -
    test T-U8."""

    penalty_scale: float = 60.0  # provisional: replaced at G1
    """INC. `pen`, penalty scale in ratio units (PLAN sections 2.8, 2.9.1; finding F9). Range
    [5, 200]; unsourced, chosen from the regime map. `audit_rate * penalty_scale` is the compound
    quantity the G2 padding-elasticity criterion sweeps (PLAN section 4.5)."""

    effort_cost: float = 0.15  # provisional: replaced at G1
    """INC. `kappa` in `c_ik = kappa * e_ik**2 + ...` (PLAN section 2.6). A real cost paid when
    incurred, not shaping (CONTRACT rule 4). Range [0.05, 0.5]; unsourced, from the regime map."""

    soft_budget: float = 0.0
    """INC. Kornai soft-budget intensity; the Phase-2 definition is the bailout probability when
    `fill < 1` (PLAN section 3, WO-023). Phase 1: 0.0. Range [0, 1]."""

    steps_per_period: int = 4
    """INC. `M`, PRODUCE steps per plan period; agents act `M + 1` times per period (PLAN sections
    2.1, 2.5). Phase 1: 4; grid {4, 8}."""

    tenure: float = 0.9
    """INC. `psi`, per-period continuation probability under `horizon_mode = "geometric"` (PLAN
    section 2.12). An economic parameter (managerial rotation), deliberately distinct from the
    technical PPO discount `gamma`; it also enters the DP Bellman operator as `psi * gamma` (PLAN
    section 5). Range [0.7, 0.98]."""

    alloc_eta_request: float = 0.0
    """INC. `eta_q`, exponent on requests in the allocation weight
    `w_bj = (q_bj + 1e-6)**eta_q * (need_bj + 1e-6)**eta_n` (PLAN section 2.7.2). Phase 1: 0, so
    requests are logged but inert. Locked Phase-2 value 0.7 (PLAN section 4.2) - it is the hoarding
    mechanism. Range [0, 1]."""

    alloc_eta_need: float = 1.0
    """INC. `eta_n`, exponent on planned need in the allocation weight (PLAN section 2.7.2). Fixed
    at 1.0."""

    bonus_heterogeneity: float = 0.0
    """INC. Dispersion of sector-specific bonus schedules around `notch_height` (PLAN section 3;
    historical: sector-specific schedules). Phase 1: 0.0. Range [0, 0.5]."""


@dataclass(frozen=True)
class InformationConfig:
    """Information architecture (PLAN section 3, arm INFO): what the planner sees, how late, how
    noisily, how aggregated, and how often it audits.

    Mirrors `spec.spec.InformationConfig` field for field. Frozen and hashable. These are the
    parameters the C_OGAS and C_AUDIT contrasts move (PLAN section 4.3); `audit_rate` is
    dual-classified - it also enters the reward through the penalty - and is therefore always
    reported separately, never folded into C_OGAS.
    """

    report_lag: int = 0
    """INFO. Periods of delay before a report reaches the planner's rules (PLAN sections 2.7.1,
    2.7.5). Phase 1: 0; grid {0, 1, 2}. Qualitative source: annual reporting cycles."""

    aggregation_level: AggLevel = "enterprise"
    """INFO. Level at which the planner observes claims (PLAN section 2.7.5). At `sector` it sees
    only `sum_{i in j} claimed_i` and allocates by planned need alone. Phase 1: `enterprise`."""

    audit_rate: float = 0.10  # provisional: replaced at G1
    """INFO (dual: it also enters the reward through the penalty, so it is reported separately -
    PLAN section 4.3). `a`, per-enterprise per-period audit probability (PLAN section 2.7.4). Range
    [0.01, 0.30]; unsourced. `validate` requires it in [0, 1]."""

    audit_noise: float = 0.0
    """INFO. `sigma_aud`, log-sd of the audit measurement error `S_hat = S * exp(nu)`,
    `nu ~ N(0, sigma_aud**2)` (PLAN section 2.8). Phase 1: 0.0; range [0, 0.15]."""

    audit_mode: AuditMode = "random"
    """INFO. Audit selection rule (PLAN section 2.7.4). Phase 1: `random`. `targeted` raises the
    probability with the planner's noisy knowledge of downstream complaints and requires
    `shortfall_visibility > 0` (WO-023)."""

    channel_noise: float = 0.0
    """INFO. `sigma_ch`, log-sd of the reporting-channel distortion
    `claimed_i <- claimed_i * exp(xi)`, `xi ~ N(0, sigma_ch**2)` (PLAN section 2.7.5). Phase 1:
    0.0; range [0, 0.10]."""

    ministry_passthrough: float = 1.0
    """INFO. `pi` in the ministry forwarding rule of PLAN section 2.14; 1.0 is a fully transparent
    ministry. Phase 1: 1.0 (no ministry layer active). Range [0.5, 1.0]; qualitative source: OGAS
    resistance."""

    n_ministries: int = 1
    """INFO. `M_min`, number of ministries partitioning the enterprises (PLAN section 2.14). Not
    tabulated in PLAN section 3; 1 in Phase 1, where the layer is inert. Set by WO-025."""

    horizontal_visibility: float = 0.0
    """INFO. Fraction of the other `N - 1` enterprises visible as trade counterparties, and the
    gate on the horizontal observation block (PLAN sections 2.4, 2.13). Phase 1: 0.0. Locked
    Phase-2 value 1.0 for the blat phenomenon (PLAN section 4.2). Range [0, 1]."""

    quality_measurability: float = 0.0
    """INFO. `mu` in the measured quality factor `q_hat = 1 + mu * (qbar - 1)` used by the
    `quality_weighted` objective metric (PLAN section 2.9.2). Phase 1: 0.0; range [0, 1]."""

    shortfall_visibility: float = 0.0
    """INFO. How much of buyers' complaints the planner sees, gating the `targeted` audit mode
    (PLAN section 2.7.4). Phase 1: 0.0; range [0, 1]."""

    self_obs_noise: float = 0.0
    """INFO. Log-sd of the multiplicative noise `exp(N(0, s**2))` applied to the agent's own
    cumulative output and stock observation fields, drawn with purpose `selfobs` (PLAN section 2.4,
    WO-008). Phase 1: 0.0, so those fields are exact. Mechanism toggle for the shape study; range
    [0, 0.05]."""


@dataclass(frozen=True)
class TechConfig:
    """Technical and engineering constants (PLAN section 3, arm TECH): bounds, horizon, initial
    scaling, policy-parameter sharing and the two seed streams.

    Mirrors `spec.spec.TechConfig` field for field. Frozen and hashable. None of these is a
    treatment variable in any contrast; they are fixed across arms. The PPO hyper-parameters
    (`gamma = 0.99`, `lambda_GAE = 0.97`, lr 3e-4, clip 0.2, entropy 0.01 -> 0.001) are TECH too
    but live with the adapter (WO-017), not in `EnvConfig`, because the environment never reads
    them; `gosplan.params.REGISTRY` records them under the `ppo_` names.
    """

    horizon_mode: HorizonMode = "geometric"
    """TECH. Termination rule (PLAN section 2.12). Phase 1: `geometric`."""

    min_periods: int = 4
    """TECH. `P_min`: periods that always run before geometric termination can fire (PLAN section
    2.12). `validate` requires `min_periods <= max_periods`."""

    max_periods: int = 20
    """TECH. `P_max`: hard cap on periods per episode (PLAN section 2.12). With `tenure = 0.9` the
    expected length is about 10 periods, i.e. about 50 agent-steps."""

    report_max_ratio: float = 10.0
    """TECH. `rho_max`, upper bound on the report action (PLAN sections 2.3, 2.8). CONTRACT rule 8:
    bounds are results - the fraction of reports at the bound is logged, and above 1% the run
    manifest is flagged `BOUND_BINDING` (test T-B8). The bound is never silently moved to fix a
    result. `validate` requires `report_max_ratio > 1`."""

    request_max_multiple: float = 3.0
    """TECH. `r_max`: the input request is bounded by `r_max * need_ij` (PLAN section 2.3)."""

    target_floor_frac: float = 0.05
    """TECH. `T_min` as a fraction of `T_0`: the target rule floors at `max(T_min, ...)` (PLAN
    sections 2.7.1, 3)."""

    inventory_cap_mult: float = 3.0
    """TECH. `S_max = inventory_cap_mult * cap_i` (PLAN section 2.11). Output above the cap is lost
    and the overflow is logged, so it stays visible in the conservation identity (test T-U1)."""

    initial_target_frac: float = 0.6
    """TECH. `T_0 = initial_target_frac * A_{s(i)} * cap_i` (PLAN section 3): feasible at effort
    about 0.6."""

    param_sharing: ParamSharing = "shared"
    """TECH. Policy-parameter sharing (PLAN sections 3, 6.1). Phase 1: one shared policy with the
    sector one-hot in the observation."""

    seed_env: int = 0
    """TECH. Root seed for every environment draw (PLAN section 2.15). Sharing it across arms is
    what delivers common random numbers by construction (PLAN section 4.3). Not tabulated in PLAN
    section 3."""

    seed_policy: int = 0
    """TECH. Root seed for the policy/action stream, kept strictly separate from `seed_env` (PLAN
    section 2.15, CONTRACT rule 9). Not tabulated in PLAN section 3."""


@dataclass(frozen=True)
class EnvConfig:
    """The complete environment configuration: the four arms plus the spec version that produced
    it.

    Mirrors `spec.spec.EnvConfig` field for field. Frozen and hashable, so a configuration is a
    value: it can be a dictionary key, it hashes reproducibly, and it is written verbatim into
    every run manifest (CONTRACT rule 10). The arm split is not cosmetic - it is the design decision
    that makes the C_OGAS / C_INC contrasts of PLAN section 4.3 well defined, and changing a
    parameter's arm requires a CHANGELOG entry (CONTRACT rule 11).
    """

    supply: SupplyConfig = field(default_factory=SupplyConfig)
    """Production and supply structure (arm SUPPLY)."""

    incentive: IncentiveConfig = field(default_factory=IncentiveConfig)
    """Incentive structure (arm INC)."""

    information: InformationConfig = field(default_factory=InformationConfig)
    """Information architecture (arm INFO)."""

    tech: TechConfig = field(default_factory=TechConfig)
    """Technical constants (arm TECH)."""

    spec_version: str = SPEC_VERSION
    """The `SPEC_VERSION` this configuration was written against; recorded in the manifest so a
    result can never be silently attributed to a different interface version (CONTRACT rules 1,
    10)."""

    def validate(self) -> None:
        """Check every cross-field invariant this configuration must satisfy; raise on the first
        violation.

        Takes: nothing beyond `self`. Returns: `None` on success; raises `ValueError` with a
        message naming the offending field and the rule it broke. It never repairs, clips or
        defaults a bad value: a configuration that cannot be run is an error, not something to be
        silently fixed.

        Must reject at least (the WO-003 must-pass list, `tests/unit/test_config.py`):
          - `incentive.overfulfilment_cap < 1` (`rho_cap < 1` would invert the bonus kink);
          - `incentive.notch_width < 0` (`w < 0` has no logistic reading);
          - `supply.input_complementarity < 1` (`theta < 1` leaves the CES branch of PLAN section
            2.6 undefined; `theta = inf` is legal and selects the Leontief `min` branch);
          - any row of `supply.io_matrix` with `sum_k a[j][k] >= 1` - no self-sustaining sector;
          - negative `incentive.ratchet_cap_up` or `incentive.ratchet_cap_dn`, and negative scales
            generally (`penalty_scale`, `effort_cost`, `notch_height`, `overfulfilment_slope`,
            `holding_loss`, `input_holding_loss`, every sigma).

        Further structural checks implied by PLAN sections 2.1-2.12:
          - `len(supply.sector_of) == supply.n_enterprises`, every entry in `[0, n_sectors)`;
          - `io_matrix` square of side `n_sectors`, and `final_demand_share`, `productivity`,
            `yield_sigma`, `ces_alpha` each of length `n_sectors`;
          - `len(supply.arrival_probs) == incentive.steps_per_period` and summing to 1 whenever
            `supply.delivery_timing != "uniform"`;
          - probabilities inside [0, 1]: `information.audit_rate`, `incentive.tenure`,
            `information.ministry_passthrough`, `supply.final_demand_share`,
            `information.horizontal_visibility`, `information.quality_measurability`,
            `information.shortfall_visibility`, `incentive.soft_budget`;
          - `tech.min_periods <= tech.max_periods`; `tech.report_max_ratio > 1`;
            `supply.invest_lag >= 1`; `supply.ces_sigma > 0`.

        The numeric ranges of PLAN section 3 live in `gosplan.params.REGISTRY`; this method
        enforces feasibility, not the sweep ranges, so a deliberate out-of-sweep sensitivity run
        stays possible and is recorded in the manifest.

        Realises: PLAN section 3 (registry ranges) and the WO-003 card. Owning WO: **WO-003**.
        """
        supply = self.supply
        incentive = self.incentive
        information = self.information
        tech = self.tech

        # SUPPLY: structure and sizes (PLAN sections 2.1, 2.6, 2.10, 2.11).
        if len(supply.sector_of) != supply.n_enterprises:
            raise ValueError(
                "supply.sector_of: length must equal supply.n_enterprises "
                f"({len(supply.sector_of)} != {supply.n_enterprises})"
            )
        for index, sector in enumerate(supply.sector_of):
            if not 0 <= sector < supply.n_sectors:
                raise ValueError(
                    f"supply.sector_of[{index}]: sector must lie in [0, n_sectors) "
                    f"(got {sector}, n_sectors={supply.n_sectors})"
                )
        if len(supply.io_matrix) != supply.n_sectors or any(
            len(row) != supply.n_sectors for row in supply.io_matrix
        ):
            raise ValueError(
                f"supply.io_matrix: must be n_sectors x n_sectors (n_sectors={supply.n_sectors})"
            )
        for index, row in enumerate(supply.io_matrix):
            if sum(row) >= 1:
                raise ValueError(
                    f"supply.io_matrix[{index}]: every row must sum to < 1, so no sector is "
                    f"self-sustaining (got {sum(row)})"
                )
        for name in ("final_demand_share", "productivity", "yield_sigma", "ces_alpha"):
            vector = getattr(supply, name)
            if len(vector) != supply.n_sectors:
                raise ValueError(
                    f"supply.{name}: length must equal supply.n_sectors "
                    f"({len(vector)} != {supply.n_sectors})"
                )
        if supply.input_complementarity < 1:
            raise ValueError(
                "supply.input_complementarity: theta must be >= 1, with inf selecting the "
                f"Leontief branch (got {supply.input_complementarity})"
            )
        if supply.invest_lag < 1:
            raise ValueError(f"supply.invest_lag: must be >= 1 (got {supply.invest_lag})")
        if supply.ces_sigma <= 0:
            raise ValueError(f"supply.ces_sigma: must be > 0 (got {supply.ces_sigma})")
        if supply.delivery_timing != "uniform":
            if len(supply.arrival_probs) != incentive.steps_per_period:
                raise ValueError(
                    "supply.arrival_probs: length must equal incentive.steps_per_period when "
                    f"delivery_timing != 'uniform' "
                    f"({len(supply.arrival_probs)} != {incentive.steps_per_period})"
                )
            if abs(sum(supply.arrival_probs) - 1) > 1e-9:
                raise ValueError(
                    f"supply.arrival_probs: must sum to 1 (got {sum(supply.arrival_probs)})"
                )
        for index, share in enumerate(supply.final_demand_share):
            if not 0 <= share <= 1:
                raise ValueError(
                    f"supply.final_demand_share[{index}]: must lie in [0, 1] (got {share})"
                )
        for name in (
            "holding_loss",
            "input_holding_loss",
            "trade_tau",
            "price_markup",
            "tech_drift_sigma",
        ):
            value = getattr(supply, name)
            if value < 0:
                raise ValueError(f"supply.{name}: must be non-negative (got {value})")
        for index, sigma in enumerate(supply.yield_sigma):
            if sigma < 0:
                raise ValueError(f"supply.yield_sigma[{index}]: must be non-negative (got {sigma})")

        # INC: bonus schedule, ratchet and probabilities (PLAN sections 2.7.1, 2.8).
        if incentive.overfulfilment_cap < 1:
            raise ValueError(
                "incentive.overfulfilment_cap: must be >= 1, with inf meaning no cap "
                f"(got {incentive.overfulfilment_cap})"
            )
        if incentive.notch_width < 0:
            raise ValueError(
                f"incentive.notch_width: must be non-negative (got {incentive.notch_width})"
            )
        for name in ("ratchet_cap_up", "ratchet_cap_dn"):
            value = getattr(incentive, name)
            if value < 0:
                raise ValueError(f"incentive.{name}: must be non-negative (got {value})")
        for name in ("penalty_scale", "effort_cost", "notch_height", "overfulfilment_slope"):
            value = getattr(incentive, name)
            if value < 0:
                raise ValueError(f"incentive.{name}: must be non-negative (got {value})")
        if not 0 <= incentive.tenure <= 1:
            raise ValueError(f"incentive.tenure: must lie in [0, 1] (got {incentive.tenure})")
        if not 0 <= incentive.soft_budget <= 1:
            raise ValueError(
                f"incentive.soft_budget: must lie in [0, 1] (got {incentive.soft_budget})"
            )

        # INFO: probabilities and noise scales (PLAN sections 2.7.4, 2.7.5, 2.8).
        for name in (
            "audit_rate",
            "ministry_passthrough",
            "horizontal_visibility",
            "quality_measurability",
            "shortfall_visibility",
        ):
            value = getattr(information, name)
            if not 0 <= value <= 1:
                raise ValueError(f"information.{name}: must lie in [0, 1] (got {value})")
        for name in ("audit_noise", "channel_noise", "self_obs_noise"):
            value = getattr(information, name)
            if value < 0:
                raise ValueError(f"information.{name}: must be non-negative (got {value})")

        # TECH: horizon and bounds (PLAN sections 2.3, 2.12).
        if tech.min_periods > tech.max_periods:
            raise ValueError(
                "tech.min_periods: must be <= tech.max_periods "
                f"({tech.min_periods} > {tech.max_periods})"
            )
        if tech.report_max_ratio <= 1:
            raise ValueError(f"tech.report_max_ratio: must be > 1 (got {tech.report_max_ratio})")

    def hash(self) -> str:
        """Return the stable content hash of this configuration.

        Takes: nothing beyond `self`. Returns: the SHA-256 hex digest of the canonical JSON encoding, pinned byte-for-byte by ambiguity report #51 because
        `ref/gen_golden.config_hash` re-derives it independently and `tests/golden/` asserts the two
        digests are equal:

            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

        over the nested per-section object `{"supply": {...}, "incentive": {...},
        "information": {...}, "tech": {...}}`, with floats rendered by `repr`, `float("inf")` as the
        string "inf", tuples as JSON arrays, no trailing newline, encoded UTF-8 and hashed with
        `hashlib.sha256`, rendered lowercase hex. The five semantic bullets alone did not determine
        the bytes - `sort_keys=True` on its own still emits ", " and ": " separators - so two
        faithful implementations could disagree and fail golden parity with no card at fault. The digest names the run directory `runs/<hash>/` and
        appears in the manifest (CONTRACT rule 10).

        Binds: `tests/unit/test_config.py` - the hash is stable under field order, and two
        configurations differing in any single parameter hash differently. Owning WO: **WO-003**.
        """

        def canonical(value: object) -> object:
            if isinstance(value, float) and value == float("inf"):
                return "inf"
            if isinstance(value, tuple):
                return [canonical(item) for item in value]
            if isinstance(value, dict):
                return {key: canonical(item) for key, item in value.items()}
            return value

        payload = canonical(dataclasses.asdict(self))
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_config(path: str) -> EnvConfig:
    """Load an `EnvConfig` from a JSON or TOML file on disk.

    Takes: `path`, a filesystem path to a configuration document whose top-level keys are `supply`,
    `incentive`, `information`, `tech` and optionally `spec_version`; missing keys take the Phase-1
    defaults declared on the dataclasses above. Returns: a validated `EnvConfig` - the loader calls
    `EnvConfig.validate()` before returning, and decodes the string `"inf"` back to `float("inf")`
    (the inverse of `EnvConfig.hash`'s encoding).

    Raises `ValueError` on an unknown key: silently ignoring an unrecognised parameter would let a
    sweep run at defaults while its manifest claimed otherwise.

    Realises: PLAN section 3. Owning WO: **WO-003**.
    """
    if path.lower().endswith(".toml"):
        with open(path, "rb") as handle:
            document = tomllib.load(handle)
    elif path.lower().endswith(".json"):
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
    else:
        raise ValueError(f"load_config: unsupported configuration format for {path!r}")
    if not isinstance(document, dict):
        raise ValueError("load_config: the configuration document must be a mapping")

    sections: dict[str, type] = {
        "supply": SupplyConfig,
        "incentive": IncentiveConfig,
        "information": InformationConfig,
        "tech": TechConfig,
    }
    unknown = sorted(set(document) - set(sections) - {"spec_version"})
    if unknown:
        raise ValueError(f"load_config: unknown configuration key {unknown[0]!r}")

    def decode(value: object, default: object) -> object:
        """Inverse of `EnvConfig.hash`'s encoding: `"inf"` -> `float("inf")`, arrays -> tuples."""
        if isinstance(value, str) and value == "inf":
            return float("inf")
        if isinstance(default, tuple):
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"load_config: expected an array, got {type(value).__name__}")
            element_default = default[0] if default else None
            return tuple(decode(item, element_default) for item in value)
        if isinstance(value, (list, tuple)):
            raise ValueError("load_config: expected a scalar, got an array")
        return value

    kwargs: dict[str, object] = {}
    for section_name, section_cls in sections.items():
        if section_name not in document:
            continue
        raw = document[section_name]
        if not isinstance(raw, dict):
            raise ValueError(f"load_config: section {section_name!r} must be a mapping")
        defaults = {spec.name: spec.default for spec in dataclasses.fields(section_cls)}
        section_kwargs: dict[str, object] = {}
        for key, value in raw.items():
            if key not in defaults:
                raise ValueError(
                    f"load_config: unknown configuration key {key!r} in section {section_name!r}"
                )
            section_kwargs[key] = decode(value, defaults[key])
        kwargs[section_name] = section_cls(**section_kwargs)

    spec_version = document.get("spec_version", SPEC_VERSION)
    if not isinstance(spec_version, str):
        raise ValueError("load_config: spec_version must be a string")
    config = EnvConfig(spec_version=spec_version, **kwargs)
    config.validate()
    return config


def p1_default_config() -> EnvConfig:
    """Return the Phase-1 configuration: the "P1 value" column of the PLAN section 3 registry.

    Takes: nothing. Returns: the `EnvConfig` whose every field equals the Phase-1 default declared
    on the four config dataclasses above, i.e. `EnvConfig()`, validated.

    The daggered rows of PLAN section 3 - `ratchet_lambda`, `growth_directive`,
    `overfulfilment_slope`, `penalty_scale`, `effort_cost`, `audit_rate`, enumerated by
    `gosplan.params.provisional()` - are provisional: gate G1 replaces them with values the human
    picks from the interior of the DP regime map, recorded in `runs/G1_decision.md` (PLAN sections
    5, 13). Until then this function returns the placeholders, and no result computed at them is
    evidence about anything.

    Binds: `tests/unit/test_config.py` asserts field-by-field agreement between this function and
    the registry data in `gosplan/params.py` - `params.by_name(<field>).default` for every registry
    entry that names an `EnvConfig` field (all but the five `ppo_` rows, which belong to the
    adapter of WO-017). The two must never drift. Owning WO: **WO-003**.
    """
    config = EnvConfig()
    config.validate()
    return config
