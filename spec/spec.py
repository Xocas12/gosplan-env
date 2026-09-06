"""gosplan-env frozen interface - full expansion of PLAN section 10 (v0, provisional until G1).

Realises: PLAN sections 2.1-2.15 (environment specification), 3 (parameter registry), 4
(pre-registration and phenomena), 5 (single-enterprise DP), 6 (agents/oracle), 9 (CONTRACT), 10
(this skeleton) and 11 (test architecture). Owning work order: **WO-001** (Spec v0, CONTRACT,
registry; LEAD).

This file is the anchor every other module imports from. It carries type aliases, configuration
dataclasses with the Phase-1 defaults of PLAN section 3, the state/action/view record types, and the
signature plus complete docstring of every function in the system. **Every body raises
NotImplementedError**; the work order named in each docstring supplies the implementation.

FROZEN BY CONTRACT RULE 1. `spec/spec.py` is provisional (v0, SPEC_VERSION "0.1.0") until gate G1
and frozen (v1, "1.0.0") thereafter at WO-013. After v1 only the lead may change this file, and
only with a `spec/CHANGELOG.md` entry recording version, reason and affected work orders. No other
session edits it.

Other contract rules that constrain what may be written against this interface:
  rule 4  the enterprise reward has exactly the terms in `enterprise_reward` - nothing else;
  rule 5  planner rules take a `PlannerView` and nothing else (see `make_planner_view`);
  rule 6  `welfare_true` / `val_measured` are logged and never observed by any agent;
  rule 7  no transition rule or reward term implements a pathology directly;
  rule 9  all environment randomness goes through `draw` (PLAN section 2.15).

Conventions. Struct-of-arrays with leading dimension `N` throughout, so the Phase-2 JAX port
(WO-029) is a mechanical translation. `Array = np.ndarray`; shapes are stated in every comment and
docstring. Symbols follow the table in PLAN section 2.1; where a docstring writes plain-ASCII
pseudo-maths it is transcribed from the numbered PLAN formula it cites, so an implementer never
needs PLAN.md to know what to compute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Protocol

import numpy as np

Array = np.ndarray
"""Alias for every numeric array in the interface (PLAN section 10). The JAX port substitutes its
own array type behind the same name; no module may rely on numpy-only methods in a signature."""

SPEC_VERSION = "0.1.0"
"""Provisional spec version (PLAN section 10 header). Bumped to "1.0.0" by WO-013 at the v1 freeze;
every later change needs a `spec/CHANGELOG.md` entry (CONTRACT rule 1). Written into every run
manifest (CONTRACT rule 10)."""


# ---------- enums ----------

Arm = Literal["info", "inc", "supply", "tech"]
"""Parameter-arm classification (PLAN section 3): information architecture, incentive structure,
production/supply structure, technical. The assignment is a design decision recorded in
`gosplan/params.py`; changing one requires a CHANGELOG entry (CONTRACT rule 11)."""

Phase = Literal["produce", "report"]
"""Agent-step phase within a plan period (PLAN section 2.5). Each period is `steps_per_period`
PRODUCE steps followed by exactly one REPORT step, so agents act `M + 1` times per period."""

ObjectiveMetric = Literal["val", "net_output", "quality_weighted"]
"""Fulfilment measure the bonus and the ratchet key on (PLAN section 2.9.2). `welfare` is
deliberately not an option (finding F6): the planner cannot key on a quantity it does not
observe."""

PenaltyForm = Literal["proportional", "fixed"]
"""Shape of the audit penalty (PLAN section 2.8): `pen * f_i`, or `pen * 1[f_i > 0]`."""

PenaltyArg = Literal["positive_part", "absolute"]
"""Argument of the audit penalty (PLAN section 2.8): `max(0, R - S_hat) / T` (over-claiming only,
Phase 1) or `|R - S_hat| / T` (both directions)."""

AuditMode = Literal["random", "targeted"]
"""Audit selection rule (PLAN section 2.7.4). Phase 1 `random`; `targeted` is a Phase-2 information
mechanism gated by `shortfall_visibility`."""

AggLevel = Literal["enterprise", "sector"]
"""Level at which the planner sees reports (PLAN sections 2.7.5, 2.4). At `sector` the planner sees
only the per-sector sum of claims and allocates by planned need alone."""

DeliveryTiming = Literal["uniform", "stochastic", "backloaded"]
"""Within-period arrival of delivered inputs (PLAN section 2.6). Phase 1 `uniform` (everything
arrives at step 0). The Phase-2 storming mechanism uses `stochastic` with `arrival_probs` (PLAN
section 4.2)."""

HorizonMode = Literal["geometric", "fixed"]
"""Episode termination (PLAN section 2.12). `geometric`: after `min_periods`, continue with
probability `tenure` per period, hard cap `max_periods`, so the agent never observes periods
remaining and there is no end-game (finding F4). `fixed` exists to study known rotation
separately."""

ParamSharing = Literal["shared", "per_sector", "independent"]
"""Policy-parameter sharing across enterprises (PLAN section 3, TECH). Phase 1 `shared`, with the
sector one-hot in the observation (PLAN section 2.4) carrying sector identity."""

RegimeLabel = Literal["bunching", "pad_to_cap", "truthful_underfulfilment", "mixed"]
"""Regime label attached to a DP solution (PLAN section 5); the return type of `classify_regime`.
Named here rather than inline so `gosplan/agents/dp.py` and `gosplan/experiments/regime_map.py`
share one definition."""

Purpose = Literal[
    "yield",
    "audit",
    "auditnoise",
    "arrival",
    "channel",
    "drift",
    "terminate",
    "trade_visibility",
    "selfobs",
]
"""Enumerated RNG purposes (PLAN section 2.15, plus `selfobs` for the observation noise of WO-008).
Keying by purpose is what makes draws order-independent, so the NumPy and JAX implementations agree
by construction and common random numbers across arms hold whenever `seed_env` is shared."""

Dist = Literal["lognormal", "normal", "bernoulli", "categorical"]
"""Distributions `draw` must support (WO-004 must-pass list). `lognormal` is parameterised by
(mean_log, sigma); the yield shock of PLAN section 2.6 uses
mean_log = -sigma**2 / 2 so E[eps] = 1."""


# ---------- config (section 3) ----------


@dataclass(frozen=True)
class SupplyConfig:
    """Production and supply structure (PLAN section 3, arm SUPPLY, plus the sizing constants of the
    TECH rows that PLAN section 10 groups here).

    Frozen and hashable: every array-valued default is a `tuple`, never a list or ndarray, so
    `EnvConfig.hash()` is stable and configurations can be used as dictionary keys. Field
    docstrings record the arm each parameter belongs to (CONTRACT rule 11) and the sweep range of
    PLAN section 3. Validation of these fields is `EnvConfig.validate` (WO-003).
    """

    n_enterprises: int = 20
    """TECH. `N`, number of enterprises (PLAN section 2.1). Phase 1: 20, four per sector."""

    n_sectors: int = 5
    """TECH. `J`, number of sectors, one good per sector (PLAN section 2.1). Phase 1: 5."""

    sector_of: tuple[int, ...] = (0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4)
    """TECH. `s(i)`, sector of each enterprise; length `n_enterprises`, values in [0, n_sectors).
    Phase 1: four enterprises per sector, contiguous (PLAN section 2.1)."""

    io_matrix: tuple[tuple[float, ...], ...] = (
        (0.0, 0.2, 0.2, 0.0, 0.0),
        (0.0, 0.0, 0.2, 0.2, 0.0),
        (0.0, 0.0, 0.0, 0.2, 0.2),
        (0.2, 0.0, 0.0, 0.0, 0.2),
        (0.2, 0.2, 0.0, 0.0, 0.0),
    )
    """SUPPLY. `a[j][k]`: units of good `k` needed per unit of good `j`; rows = producing sector,
    columns = input good (PLAN section 2.10). Phase 1: every sector needs two inputs at 0.2 each;
    the graph is a 5-cycle with chords, so a shortage in any sector propagates to all. `validate`
    rejects any row with `sum_k a[j][k] >= 1` (WO-003)."""

    final_demand_share: tuple[float, ...] = (0.5, 0.5, 0.5, 0.5, 0.5)
    """SUPPLY. `phi_j`, fraction of shipped good `j` routed to the consumer sink rather than to
    intermediate buyers (PLAN sections 2.7.3, 2.10). Phase 1: 0.5 for all `j`; range [0.3, 0.7]."""

    productivity: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0)
    """SUPPLY. `A_j`, sector productivity in `y_hat = (A_{s(i)} * cap_i / M) * e` (PLAN section
    2.6). Not tabulated in PLAN section 3; fixed at 1.0 as the normalisation that makes the TECH
    row `T_0 = 0.6 * A * cap` read as `T_0 = 0.6` at `cap = 1` (see
    `TechConfig.initial_target_frac`)."""

    yield_sigma: tuple[float, ...] = (0.05, 0.08, 0.10, 0.12, 0.15)
    """SUPPLY. `sigma_j`, log-sd of the per-step multiplicative yield shock, one per sector
    (PLAN sections 2.6, 3). Heteroskedastic by construction; sweep multiplies the whole tuple by
    [0.5, 2]. Source status: unsourced."""

    input_complementarity: float = 8.0
    """SUPPLY. `theta` in the CES coverage aggregator of PLAN section 2.6; `theta = inf` reduces to
    Leontief `min`. Phase 1: 8 (near-Leontief); grid {2, 8, inf}. `validate` rejects `theta < 1`."""

    setup_cost: float = 0.0
    """SUPPLY. `F`, fixed cost charged once per step when `e > 0` (PLAN section 2.6). Phase 1 toggle
    off (0.0); Phase-2 range [0, 0.2]."""

    irs_alpha: float = 0.0
    """SUPPLY. `alpha_irs` in the increasing-returns toggle `A_j(Kap) = A_j * (Kap /
    Kap_0)**alpha_irs` (PLAN section 2.6). Phase 1 off (0.0); Phase-2 range [0, 0.3]."""

    capital_dep: float = 0.0
    """SUPPLY. Capital depreciation in `Kap_{t+1} = (1 - dep) * Kap_t + matured investment`
    (PLAN section 2.6). Phase 1 off (0.0); Phase-2 range [0.02, 0.1]."""

    invest_lag: int = 1
    """SUPPLY. Periods between diverting output to investment and its maturing into capital; sets
    the second dimension of `State.pending_invest` (PLAN sections 2.2, 2.6). Phase-1 value is "-"
    in the PLAN section 3 registry because investment is inert (`v == 0`); 1 is the minimum viable
    buffer depth. Phase-2 grid {1, 2, 3}."""

    delivery_timing: DeliveryTiming = "uniform"
    """SUPPLY. Within-period arrival of delivered inputs (PLAN section 2.6). Phase 1 `uniform`: all
    inputs are available at step 0. This is the storming mechanism, locked for Phase 2 in PLAN
    section 4.2."""

    arrival_probs: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
    """SUPPLY. `pi_arrival`, distribution of a delivered unit's arrival step over `0..M-1`, used
    only when `delivery_timing != "uniform"` (PLAN section 2.6). Locked Phase-2 value from PLAN
    section 4.2: uniform-random arrival, so no mechanical backloading and any effort-Gini excess is
    behavioural."""

    holding_loss: float = 0.02
    """SUPPLY. `h`, per-period proportional loss on carried own-good stock, applied before this
    period's output is added: `S <- (1 - h) * S + y` (PLAN sections 2.8, 2.11). Phase 1: 0.02."""

    input_holding_loss: float = 0.0
    """SUPPLY. `h_X`, per-period loss on held input stocks (PLAN section 2.11). Phase 1: 0.0, so
    hoarding carries no direct carrying cost by design. Locked Phase-2 value 0.01 (PLAN section
    4.2)."""

    price_markup: float = 0.1
    """SUPPLY. `m` in the cost-plus price fixed point `p_j = (1 + m) * (kappa_labour + sum_k a_jk
    p_k)` (PLAN section 2.10), solved once at `t = 0` by `initial_prices`. Phase 1: 0.1."""

    price_lag: float = float("inf")
    """SUPPLY. Periods between price recomputations (PLAN section 2.10). Phase 1: `inf`, i.e. plan
    prices are fixed after `t = 0`. Serialised as the string "inf" by `EnvConfig.hash` (WO-003)."""

    tech_drift_sigma: float = 0.0
    """SUPPLY. `sigma_drift` in the per-period I-O drift `a <- a * exp(zeta)`, `zeta ~ N(0, s**2)`,
    while `State.planner_io` stays fixed (PLAN section 2.6). Phase 1: 0.0; range [0, 0.05].
    Acknowledged weak proxy (PLAN section 3)."""

    ces_alpha: tuple[float, ...] = (0.2, 0.2, 0.2, 0.2, 0.2)
    """SUPPLY. `alpha_j`, consumer CES weights in the welfare index of PLAN section 2.9.3.
    Phase 1: `1 / J` for all `j`. Swept only in the price-sensitivity check, never in a treatment
    arm (PLAN sections 2.10, 7.5)."""

    ces_sigma: float = 0.8
    """SUPPLY. `sigma_c`, consumer CES elasticity of substitution (PLAN sections 2.9.3, 2.10).
    Phase 1: 0.8, complements-leaning. The `sigma_c -> 1` limit is Cobb-Douglas and is tested in
    `tests/unit/test_reward.py` (WO-007)."""

    trade_tau: float = 0.05
    """SUPPLY. `tau`, per-unit transaction cost on bilateral trade (PLAN section 2.13). Inert in
    Phase 1 (no trade); locked Phase-2 value 0.05 (PLAN section 4.2). `tau > 0` is what makes wash
    trades unprofitable."""

    quality_matters: bool = False
    """SUPPLY. Master toggle for the quality mechanism: quality routed through the input bundle,
    `X_ij` credited as `deliv * qbar_j` (PLAN section 2.6). Phase 1: False (`q == 1` everywhere).
    Turned on by WO-021."""

    quality_cost: float = 0.0
    """SUPPLY. `kappa_q` in the effort cost `c = kappa * e**2 + F * 1[e > 0] + kappa_q * q * e`
    (PLAN section 2.6). Phase 1: 0.0."""


@dataclass(frozen=True)
class IncentiveConfig:
    """Incentive structure (PLAN section 3, arm INC): the bonus schedule, the audit penalty, the
    ratchet, tenure and the allocation rule's responsiveness.

    Frozen and hashable. Values marked `# provisional: replaced at G1` are the daggered rows of the
    PLAN section 3 registry: they are placeholders until the human picks Phase-1 values from the DP
    regime map at gate G1 (PLAN sections 5, 13) and records them in `runs/G1_decision.md`.
    """

    objective_metric: ObjectiveMetric = "val"
    """INC. Which fulfilment measure the bonus and the ratchet key on (PLAN section 2.9.2).
    Phase 1: `val`. Historical motivation: `val` is the measure the NNO reforms attacked."""

    ratchet_lambda: float = 0.5  # provisional: replaced at G1
    """INC. `lambda`, ratchet coefficient in the target rule of PLAN section 2.7.1. Range [0, 1];
    Weitzman-type models motivate the form, the empirical value is unsourced."""

    growth_directive: float = 0.02  # provisional: replaced at G1
    """INC. `g`, the exogenous growth directive multiplying the target every period (PLAN section
    2.7.1). This is the forcing term added for finding F1; it must be a treatment variable because
    at `g = 0` with reports at target the target rule has a fixed point (test T-B2).
    Range [0, 0.07]."""

    ratchet_cap_up: float = 0.3
    """INC. `c_up`, cap on the positive per-period ratchet step (PLAN section 2.7.1). Engineering
    value, fixed, added to bound the transient of finding F4."""

    ratchet_cap_dn: float = 0.3
    """INC. `c_dn`, cap on the negative per-period ratchet step (PLAN section 2.7.1). Fixed."""

    ratchet_deadband: float = 0.0
    """INC. `delta`: the ratchet step is zeroed when `|rho - 1| <= delta` (PLAN section 2.7.1).
    Phase 1: 0.0 (inert); mechanism toggle for the shape study, range [0, 0.03]."""

    notch_height: float = 1.0
    """INC. `beta`, height of the fulfilment notch in `B(rho)` (PLAN section 2.8). Phase 1: 1.0 - it
    is the normalising unit, so sweeping it changes the notch relative to `pen` and `kappa` and not
    the gradient magnitude (`reward_scale`, PLAN section 2.9.1). Range [0.25, 2]."""

    notch_width: float = 0.0
    """INC. `w`, logistic width of the notch: `Lambda_w(x) = 1[x >= 0]` at `w = 0`, else
    `1 / (1 + exp(-x / w))` (PLAN section 2.8). Phase 1: 0 (a true discontinuity). This is the
    counterfactual knob (smooth arm `w = 0.25`) and the manipulation-strength knob of the
    estimator-bias study (PLAN section 7.2). Grid {0, 0.02, 0.05, 0.10, 0.25}."""

    overfulfilment_slope: float = 0.5  # provisional: replaced at G1
    """INC. `s`, linear bonus slope above target: `s * clip(rho - 1, 0, rho_cap - 1)`
    (PLAN section 2.8). Range [0, 2]; historical anchor is the per-percentage-point bonus increment
    (lead to source, PLAN section 15)."""

    overfulfilment_cap: float = 1.2
    """INC. `rho_cap`, ratio at which the overfulfilment bonus stops accruing (PLAN section 2.8).
    Phase 1: 1.2. `inf` means no cap and hence no kink - the smooth counterfactual is
    (`notch_width` = 0.25, `overfulfilment_cap` = inf). Grid {1.1, 1.2, inf}; `validate` rejects
    `rho_cap < 1` (WO-003)."""

    penalty_form: PenaltyForm = "proportional"
    """INC. `Pen = pen * f` (`proportional`, Phase 1) or `pen * 1[f > 0]` (`fixed`)
    (PLAN section 2.8)."""

    penalty_arg: PenaltyArg = "positive_part"
    """INC. `f = max(0, R - S_hat) / T` (`positive_part`, Phase 1) or `|R - S_hat| / T` (`absolute`)
    (PLAN section 2.8). Under `positive_part` any under-report incurs no penalty - test T-U8."""

    penalty_scale: float = 60.0  # provisional: replaced at G1
    """INC. `pen`, penalty scale in ratio units (PLAN sections 2.8, 2.9.1; finding F9). Range
    [5, 200]; unsourced, chosen from the regime map. `audit_rate * penalty_scale` is the compound
    quantity the G2 padding-elasticity criterion sweeps (PLAN section 4.5)."""

    effort_cost: float = 0.15  # provisional: replaced at G1
    """INC. `kappa` in `c_ik = kappa * e_ik**2 + ...` (PLAN section 2.6). A real cost paid when
    incurred, not shaping (CONTRACT rule 4). Range [0.05, 0.5]; unsourced, from the regime map."""

    soft_budget: float = 0.0
    """INC. Kornai soft-budget intensity; Phase-2 definition is the bailout probability when
    `fill < 1` (PLAN section 3, WO-023). Phase 1: 0.0. Range [0, 1]."""

    steps_per_period: int = 4
    """INC. `M`, PRODUCE steps per plan period; agents act `M + 1` times per period
    (PLAN sections 2.1, 2.5). Phase 1: 4; grid {4, 8}."""

    tenure: float = 0.9
    """INC. `psi`, per-period continuation probability under `horizon_mode = "geometric"`
    (PLAN section 2.12). This is an economic parameter (managerial rotation), deliberately distinct
    from the technical PPO discount `gamma`; it also enters the DP Bellman operator as `psi * gamma`
    (PLAN section 5). Range [0.7, 0.98]."""

    alloc_eta_request: float = 0.0
    """INC. `eta_q`, exponent on requests in the allocation weight
    `w_bj = (q_bj + 1e-6)**eta_q * (need_bj + 1e-6)**eta_n` (PLAN section 2.7.2). Phase 1: 0, so
    requests are logged but inert. Locked Phase-2 value 0.7 (PLAN section 4.2) - it is the hoarding
    mechanism. Range [0, 1]."""

    alloc_eta_need: float = 1.0
    """INC. `eta_n`, exponent on planned need in the allocation weight (PLAN section 2.7.2).
    Fixed at 1.0."""

    bonus_heterogeneity: float = 0.0
    """INC. Dispersion of sector-specific bonus schedules around `notch_height` (PLAN section 3;
    historical: sector-specific schedules). Phase 1: 0.0. Range [0, 0.5]."""


@dataclass(frozen=True)
class InformationConfig:
    """Information architecture (PLAN section 3, arm INFO): what the planner sees, how late, how
    noisily, how aggregated, and how often it audits.

    Frozen and hashable. These are the parameters the C_OGAS and C_AUDIT contrasts move
    (PLAN section 4.3); `audit_rate` is dual-classified (it also enters the reward through the
    penalty) and is therefore always reported separately, never folded into C_OGAS.
    """

    report_lag: int = 0
    """INFO. Periods of delay before a report reaches the planner's rules (PLAN sections 2.7.1,
    2.7.5). Phase 1: 0; grid {0, 1, 2}. Qualitative source: annual reporting cycles."""

    aggregation_level: AggLevel = "enterprise"
    """INFO. Level at which the planner observes claims (PLAN section 2.7.5). At `sector` it sees
    only `sum_{i in j} claimed_i` and allocates by planned need alone. Phase 1: `enterprise`."""

    audit_rate: float = 0.10  # provisional: replaced at G1
    """INFO (dual: also enters the reward through the penalty; reported separately, PLAN section
    4.3). `a`, per-enterprise per-period audit probability (PLAN section 2.7.4). Range [0.01,
    0.30]; unsourced."""

    audit_noise: float = 0.0
    """INFO. `sigma_aud`, log-sd of the audit measurement error `S_hat = S * exp(nu)`,
    `nu ~ N(0, sigma_aud**2)` (PLAN section 2.8). Phase 1: 0.0; range [0, 0.15]."""

    audit_mode: AuditMode = "random"
    """INFO. Audit selection rule (PLAN section 2.7.4). Phase 1: `random`. `targeted` raises the
    probability with the planner's noisy knowledge of downstream complaints and requires
    `shortfall_visibility > 0` (WO-023)."""

    channel_noise: float = 0.0
    """INFO. `sigma_ch`, log-sd of the reporting-channel distortion `claimed_i <- claimed_i *
    exp(xi)`, `xi ~ N(0, sigma_ch**2)` (PLAN section 2.7.5). Phase 1: 0.0; range [0, 0.10]."""

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
    """INFO. Log-sd of multiplicative noise `exp(N(0, s**2))` applied to the agent's own cumulative
    output and stock observation fields, drawn with purpose `selfobs` (PLAN section 2.4, WO-008).
    Phase 1: 0.0, so those fields are exact. Mechanism toggle for the shape study; range [0,
    0.05]."""


@dataclass(frozen=True)
class TechConfig:
    """Technical and engineering constants (PLAN section 3, arm TECH): bounds, horizon, initial
    scaling, policy-parameter sharing and the two seed streams.

    Frozen and hashable. None of these is a treatment variable in any contrast; they are fixed
    across arms. PPO hyper-parameters (`gamma = 0.99`, `lambda_GAE = 0.97`, lr 3e-4, clip 0.2,
    entropy 0.01 -> 0.001) are also TECH but live with the adapter (WO-017), not in `EnvConfig`,
    because the environment never reads them.
    """

    horizon_mode: HorizonMode = "geometric"
    """TECH. Termination rule (PLAN section 2.12). Phase 1: `geometric`."""

    min_periods: int = 4
    """TECH. `P_min`: periods that always run before geometric termination can fire
    (PLAN section 2.12)."""

    max_periods: int = 20
    """TECH. `P_max`: hard cap on periods per episode (PLAN section 2.12). With `tenure = 0.9`,
    expected length is about 10 periods, i.e. about 50 agent-steps."""

    report_max_ratio: float = 10.0
    """TECH. `rho_max`, upper bound on the report action (PLAN sections 2.3, 2.8). CONTRACT rule 8:
    bounds are results - the fraction of reports at the bound is logged, and above 1% the run
    manifest is flagged `BOUND_BINDING` (test T-B8). The bound is never silently moved to fix a
    result."""

    request_max_multiple: float = 3.0
    """TECH. `r_max`: the input request is bounded by `r_max * need_ij` (PLAN section 2.3)."""

    target_floor_frac: float = 0.05
    """TECH. `T_min` as a fraction of `T_0`: the target rule floors at `max(T_min, ...)`
    (PLAN sections 2.7.1, 3)."""

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
    what delivers common random numbers by construction (PLAN section 4.3)."""

    seed_policy: int = 0
    """TECH. Root seed for the policy/action stream, kept strictly separate from `seed_env`
    (PLAN section 2.15, CONTRACT rule 9)."""


@dataclass(frozen=True)
class EnvConfig:
    """The complete environment configuration: the four arms plus the spec version that produced it.

    Frozen and hashable, so a configuration is a value: it can be a dictionary key, it hashes
    reproducibly, and it is written verbatim into every run manifest (CONTRACT rule 10). The arm
    split is not cosmetic - it is the design decision that makes the C_OGAS / C_INC contrasts of
    PLAN section 4.3 well defined, and changing a parameter's arm requires a CHANGELOG entry
    (CONTRACT rule 11).
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

        Takes: nothing beyond `self`. Returns: `None` on success; raises `ValueError` with a message
        naming the offending field and the rule it broke.

        Must reject at least (WO-003 must-pass list, `tests/unit/test_config.py`):
          - `incentive.overfulfilment_cap < 1`;
          - `incentive.notch_width < 0`;
          - `supply.input_complementarity < 1`;
          - any row of `supply.io_matrix` with `sum_k a[j][k] >= 1` (no self-sustaining sector);
          - negative `ratchet_cap_up` or `ratchet_cap_dn`, and negative scales generally.

        Further structural checks implied by PLAN sections 2.1-2.12: `len(sector_of) ==
        n_enterprises` with every entry in `[0, n_sectors)`; `io_matrix`, `final_demand_share`,
        `productivity`, `yield_sigma`, `ces_alpha` all sized by `n_sectors`; `len(arrival_probs) ==
        steps_per_period` summing to 1 when `delivery_timing != "uniform"`; probabilities
        (`audit_rate`, `tenure`, `ministry_passthrough`, `final_demand_share`, `horizontal_
        visibility`, `quality_measurability`, `shortfall_visibility`, `soft_budget`) inside [0, 1];
        `min_periods <= max_periods`; `report_max_ratio > 1`; `invest_lag >= 1`; `ces_sigma > 0`.

        Realises: PLAN section 3 (registry ranges) and the WO-003 card. Owning WO: **WO-003**.
        """
        raise NotImplementedError("PLAN section 3 - implemented in WO-003")

    def hash(self) -> str:
        """Return the stable content hash of this configuration.

        Takes: nothing beyond `self`. Returns: the SHA-256 hex digest of the canonical JSON
        encoding of the configuration (WO-003 notes): keys sorted, so the digest is invariant to
        field order; floats formatted with `repr`; `float("inf")` serialised as the string `"inf"`;
        tuples encoded as JSON arrays. The digest names the run directory `runs/<hash>/` and
        appears in the manifest (CONTRACT rule 10).

        Binds: `tests/unit/test_config.py` - hash stable under field order, and two configurations
        that differ in any single parameter hash differently. Owning WO: **WO-003**.
        """
        raise NotImplementedError("PLAN section 3 - implemented in WO-003")


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
    raise NotImplementedError("PLAN section 3 - implemented in WO-003")


def p1_default_config() -> EnvConfig:
    """Return the Phase-1 configuration: the "P1 value" column of the PLAN section 3 registry.

    Takes: nothing. Returns: the `EnvConfig` whose every field equals the Phase-1 default declared
    on the four config dataclasses above, i.e. `EnvConfig()`, validated.

    The daggered rows of PLAN section 3 (`ratchet_lambda`, `growth_directive`,
    `overfulfilment_slope`, `penalty_scale`, `effort_cost`, `audit_rate`) are provisional: gate G1
    replaces them with values the human picks from the interior of the DP regime map, recorded in
    `runs/G1_decision.md` (PLAN sections 5, 13). Until then this function returns the placeholders.

    Binds: `tests/unit/test_config.py` asserts field-by-field agreement between this function and
    the registry data in `gosplan/params.py` - the two must never drift. Owning WO: **WO-003**.
    """
    raise NotImplementedError("PLAN section 3 - implemented in WO-003")


# ---------- state / actions / views (sections 2.2-2.4) ----------


@dataclass
class State:
    """The full environment state (PLAN section 2.2), struct-of-arrays with leading dimension `N`.

    Mutable by design: `gosplan/env/step.py` (WO-009) threads one `State` through the period
    schedule of PLAN section 2.5. It is the *true* state - it contains quantities no agent and no
    planner rule may see (CONTRACT rules 5, 6). Only `make_planner_view` may read it on the
    planner's behalf, and only observation construction (`gosplan/env/obs.py`) may read it on an
    agent's behalf.

    `J = cfg.supply.n_sectors`, `N = cfg.supply.n_enterprises`, `L = cfg.supply.invest_lag`.
    Owning WO: **WO-009** (`gosplan/env/state.py`).
    """

    target: Array  # (N,) T_i, target in units of own good (section 2.1)
    capital: Array  # (N,) Kap_i; Phase 1 fixed at 1.0 (section 2.1)
    inv_output: Array  # (N,) S_i, own-good stock on hand; the only place unreported output goes
    inv_inputs: Array  # (N, J) X_ij, input stocks held by i (section 2.6)
    cum_output: Array  # (N,) true output accumulated so far this period (reset each period)
    cum_cost: Array  # (N,) effort cost accumulated so far this period (reset each period)
    quality_acc: Array  # (N,) accumulator for the period-average quality qbar_i; Phase 1 inert

    last_report_ratio: Array  # (N,) rho_i = R_i / T_i as reported at the last REPORT step
    last_report: Array  # (N,) R_i in units, retained because the ratchet moves T after REPORT
    last_audited: Array  # (N,) bool, audit selection at the last AUDIT step (section 2.7.4)
    last_penalty: Array  # (N,) penalty charged at the last AUDIT step (section 2.8)
    last_fill: Array  # (N,) fraction of own last claim actually shipped (section 2.7.3)
    request: Array  # (N, J) q_ij, input requests from the last REPORT step (section 2.3)
    pending_invest: Array  # (N, L) output diverted to capital, maturing after invest_lag periods

    t_period: int  # plan period index t, from 0
    k_step: int  # production step index k within the period, 0 .. M-1; M at the REPORT step
    phase: Phase  # "produce" or "report" (section 2.5)
    plan_prices: Array  # (J,) p_j, plan prices (section 2.10)
    planner_io: Array  # (J, J) the planner's possibly stale copy of `a` (sections 2.2, 2.6)
    consumer_delivery: Array  # (J,) consumer_j received by the final-demand sink this period
    alive: bool  # episode has not yet terminated (section 2.12)
    seed_env: int  # root environment seed; every draw is keyed from it (section 2.15)
    seed_policy: int  # root policy seed, kept separate from seed_env (CONTRACT rule 9)


@dataclass
class EnterpriseAction:
    """One joint action for all `N` enterprises (PLAN section 2.3).

    The dimension set is fixed across phases; configuration flags decide which dimensions the
    environment reads, and the PPO adapter builds heads only for `active_action_dims(cfg)`
    (PLAN section 6.1, WO-017). Inactive dimensions are ignored by the environment rather than
    rejected, so a Phase-1 policy and a Phase-2 policy share one action type.

    Bounds are those of `action_spec(cfg)`. Owning WO: **WO-009**.
    """

    effort: Array  # (N,) e_ik in [0, 1]; active in Phase 1; read at PRODUCE steps
    quality: Array  # (N,) q_ik in [0, 1]; Phase 2
    invest: Array  # (N,) v_ik in [0, 1], fraction of step output diverted to capital; Phase 2
    report_ratio: Array  # (N,) in [0, rho_max]; active in Phase 1; read only at the REPORT step
    input_request: Array  # (N, J) q_ij in [0, r_max * need_ij]; logged in Phase 1, inert at eta_q=0
    trade_offer: Array  # (N, J) in [-1, 1]; positive = offer, negative = want; Phase 2


@dataclass(frozen=True)
class PlannerView:
    """Everything the planner is allowed to know (PLAN sections 2.4, 2.7; CONTRACT rule 5).

    Built exclusively by `make_planner_view(state, cfg)` - the single `State -> planner` boundary in
    the codebase. It contains **no true quantity**: no `y`, no `S`, no `X`, no welfare, no audit
    selection other than the audits already performed. Its contents are aggregated at
    `aggregation_level`, delayed by `report_lag` and perturbed by `channel_noise` before they arrive
    here, so a planner rule cannot recover the truth by inverting anything.

    Binds: test T-B4 (planner blindness) constructs a state with `S != R` and sentinel `y`, asserts
    no sentinel reaches this record, and statically asserts that no function in
    `gosplan/env/planner.py` accepts a `State` other than `make_planner_view`. Owning WO:
    **WO-006**.
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


@dataclass(frozen=True)
class MinistryView:
    """What one ministry sees about its own enterprises (PLAN section 2.14; **Phase-2 sketch**).

    The ministry sits between enterprises and the planner: it receives claims and forwards possibly
    smoothed and padded numbers upward. The rule-based version is `ministry_forward`; the LLM study
    of PLAN section 7.4 replaces that function with a model call that receives a text rendering of
    this same record and returns per-enterprise forwarded values plus a free-text justification.

    Interface only. Behaviour is deliberately under-specified and is frozen at the Phase-2 spec
    revision (PLAN section 0, finding F14). Owning WO: **WO-025** (rule-based), **WO-026** (LLM).
    """

    ministry_id: int  # index of this ministry in [0, n_ministries)
    enterprise_ids: Array  # (n_i,) indices of the enterprises this ministry covers
    claims: Array  # (n_i,) R_i as received from its own enterprises
    targets: Array  # (n_i,) T_i of those enterprises
    prev_forward: Array  # (n_i,) what this ministry forwarded last period (the smoothing anchor)
    passthrough: float  # pi = cfg.information.ministry_passthrough; 1.0 is transparent
    t_period: int  # plan period index, for logging and for the LLM prompt


def obs_spec(cfg: EnvConfig) -> list[str]:
    """Return the ordered names of the observation vector's components (PLAN section 2.4).

    Takes: `cfg`. Returns: a `list[str]` of length `12 + 3 * J` in Phase 1 - the canonical layout,
    in this exact order (`J = cfg.supply.n_sectors`):

        0                "phase"                     0 produce / 1 report
        1                "k_over_M"                  step within the period
        2                "log_target_ratio"          log(T_i / T_0)
        3                "growth_directive"          g; constant per config, permits config-
                                                     conditioned policies later
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

    Never present, in any phase: `welfare_true`, `val_measured`, any other enterprise's `y`, `S` or
    `X`, the audit selection for the current period, and periods remaining under geometric
    termination (PLAN sections 2.4, 2.12; CONTRACT rule 6).

    Binds: `tests/unit/test_obs.py` (layout equals this list; dimension `12 + 3J`; phase masking)
    and test T-B5 in `tests/behavioural/test_welfare_blindness.py`, which feeds sentinel values
    into the forbidden fields and asserts they appear in no observation. Owning WO: **WO-008**.
    """
    raise NotImplementedError("PLAN section 2.4 - implemented in WO-008")


def action_spec(cfg: EnvConfig) -> dict[str, tuple[tuple[int, ...], float, float]]:
    """Return the shape and box bounds of every action dimension (PLAN section 2.3).

    Takes: `cfg`. Returns: a mapping `name -> (shape, lo, hi)` covering all six dimensions whether
    or not they are active, with `N = cfg.supply.n_enterprises` and `J = cfg.supply.n_sectors`:

        "effort"         ((N,),   0.0, 1.0)
        "quality"        ((N,),   0.0, 1.0)
        "invest"         ((N,),   0.0, 1.0)
        "report_ratio"   ((N,),   0.0, cfg.tech.report_max_ratio)
        "input_request"  ((N, J), 0.0, cfg.tech.request_max_multiple)
        "trade_offer"    ((N, J), -1.0, 1.0)

    `input_request` is expressed as a multiple of need because the true bound of PLAN section 2.3
    is `r_max * need_ij` and `need_ij` is state-dependent; the environment rescales and clips it
    against the current need when it reads the action.

    The report bound is a result, not a nuisance: CONTRACT rule 8 forbids widening or narrowing it
    to fix an outcome, and the fraction of reports at the bound is logged and flags the run
    manifest `BOUND_BINDING` above 1% (test T-B8). Owning WO: **WO-009**.
    """
    raise NotImplementedError("PLAN section 2.3 - implemented in WO-009")


def active_action_dims(cfg: EnvConfig) -> list[str]:
    """Return the names of the action dimensions this configuration actually reads.

    Takes: `cfg`. Returns: a subset of `action_spec(cfg).keys()`, in the order of PLAN section 2.3.
    At `p1_default_config()` this is `["effort", "report_ratio", "input_request"]`: `input_request`
    is active and logged although it is inert while `alloc_eta_request == 0`; `quality`, `invest`
    and `trade_offer` become active only when their Phase-2 mechanisms are switched on
    (`supply.quality_matters`, non-zero `capital_dep`/investment, `information.
    horizontal_visibility > 0`).

    The PPO adapter builds Gaussian heads only for these names (PLAN section 6.1, WO-017), which is
    why the answer must be a pure function of the configuration and must not change within a run.
    Owning WO: **WO-009**.
    """
    raise NotImplementedError("PLAN section 2.3 - implemented in WO-009")


# ---------- rng (section 2.15) ----------


def draw(
    seed_env: int,
    purpose: Purpose,
    *indices: int,
    shape: tuple[int, ...],
    dist: Dist,
    **params: float,
) -> Array:
    """Key-based random draw: the single source of environment randomness (PLAN section 2.15).

    Takes: `seed_env`, the run's root environment seed; `purpose`, one of the values of `Purpose`;
    `*indices`, the integer coordinates of the draw (typically `t`, then `k`, then `i`); `shape`,
    the shape of the array to return, vectorising over the trailing index; `dist`, one of the
    values of `Dist`; `**params`, the distribution parameters -

        lognormal    mean_log, sigma      returns exp(N(mean_log, sigma**2))
        normal       mean, sigma
        bernoulli    p                    returns a boolean array
        categorical  probs                returns integer category indices

    Returns: an `Array` of the requested `shape`.

    Derivation: `numpy.random.SeedSequence([seed_env, crc32(purpose), *indices])` spawns an
    independent generator per key (WO-004 notes). Consequences that the rest of the design leans on:
    draws are **order-independent**, so the NumPy and JAX implementations agree by construction
    (WO-029 parity); common random numbers across arms hold whenever `seed_env` is shared
    (PLAN section 4.3); and the policy stream `seed_policy` is entirely separate.

    Constraints: CONTRACT rule 9 forbids any direct `numpy.random` or `jax.random` call inside
    `gosplan/env/`, and forbids module-level global generators anywhere.

    Interface note: PLAN section 10 types `purpose` and `dist` as `str`; they are narrowed here to
    the `Purpose` and `Dist` literals, which enumerate exactly the values PLAN section 2.15 and the
    WO-004 card name. Record the narrowing in `spec/CHANGELOG.md` at the v1 freeze.

    Binds: test T-U6 in `tests/unit/test_rng.py` - `draw` is deterministic in
    `(seed_env, purpose, indices)` and independent of call order, and each distribution has the
    stated moments. Owning WO: **WO-004**.
    """
    raise NotImplementedError("PLAN section 2.15 - implemented in WO-004")


# ---------- production (section 2.6) ----------


def coverage(X: Array, need: Array, weights: Array, theta: float) -> Array:
    """CES aggregator over per-good input coverage ratios (PLAN section 2.6).

    Takes: `X` (N, J), input stocks on hand; `need` (N, J), the input need of this step
    `need_ikj = a_{s(i)j} * y_hat_ik`; `weights` (N, J), `omega_j = a_{s(i)j} / sum_j a_{s(i)j}`;
    `theta`, the complementarity exponent. Returns: `H` (N,), the coverage multiplier in [0, 1].

    Formula (PLAN section 2.6, verbatim):

        H_ik = ( sum_j omega_j * min(1, X_ij / need_ikj)**(-theta) )**(-1/theta)
        H_ik = 1                        if enterprise i requires no inputs

    Edge cases the implementation must handle explicitly: `need_ikj == 0` contributes a coverage
    ratio of 1 (not a division by zero); a whole row of `a` equal to zero gives `H = 1`;
    `theta = inf` takes the `min` branch (`np.min` over the goods with positive need).

    Binds: test T-U7 in `tests/unit/test_production.py` - `theta = inf` equals `min`; `theta -> 1`
    equals the weighted harmonic mean; `H = 1` when no inputs are needed. Owning WO: **WO-005**.
    """
    raise NotImplementedError("PLAN section 2.6 - implemented in WO-005")


def produce_step(
    state: State, action: EnterpriseAction, cfg: EnvConfig
) -> tuple[State, Array, Array]:
    """Execute one PRODUCE step `k` for all enterprises (PLAN section 2.6).

    Takes: `state` at the start of step `k`; `action`, whose `effort` (and, in Phase 2, `quality`
    and `invest`) dimensions are read; `cfg`. Returns: `(state, y_ik, c_ik)` - the updated state,
    output realised this step `(N,)`, and cost incurred this step `(N,)`.

    Formulas (PLAN section 2.6, verbatim):

        y_hat_ik   = (A_{s(i)} * cap_i / M) * e_ik          # intended output at full coverage
        need_ikj   = a_{s(i)j} * y_hat_ik                   # for each good j with a_{s(i)j} > 0
        H_ik       = coverage(X, need, omega, theta)
        eps_ik     ~ LogNormal(-sigma_{s(i)}**2 / 2, sigma_{s(i)})    mean 1
                     key = (seed_env, "yield", t, k, i)
        y_tilde_ik = y_hat_ik * H_ik * eps_ik
        y_ik       = y_tilde_ik * (1 - v_ik)                # v = invest fraction; Phase 1 v == 0
        X_ij      -= min(X_ij, a_{s(i)j} * y_tilde_ik)      # inputs consumed, capped at stock
        c_ik       = kappa * e_ik**2 + F * 1[e_ik > 0] + kappa_q * q_ik * e_ik  # P1: F=kappa_q=0

    Note that inputs are consumed against `y_tilde` (output before the investment diversion), and
    that the consumption is capped at the stock on hand. `cum_output` and `cum_cost` accumulate
    `y_ik` and `c_ik`; `pending_invest` receives `y_tilde_ik * v_ik` in Phase 2.

    Phase-2 toggles that route through this function, all off in Phase 1: setup cost `F`;
    increasing returns `A_j(Kap) = A_j * (Kap / Kap_0)**alpha_irs`; capital accumulation `Kap_{t+1}
    = (1 - dep) * Kap_t + matured investment`; I-O drift `a <- a * exp(zeta)` with `planner_io`
    held fixed; quality routed through the bundle (`X_ij` credited as `deliv * qbar_j`);
    non-uniform `delivery_timing`, under which a delivered unit arrives at step `k ~
    Categorical(arrival_probs)` with purpose `arrival`.

    Constraints: this function may not reference reports, targets or rewards (WO-005 forbidden
    list), and every stochastic term goes through `draw` (CONTRACT rule 9).

    Binds: `tests/unit/test_production.py` (yield mean 1 to 1e-3 over 1e5 draws; the `v` diversion;
    the cost formula; inputs consumed equal `a * y_tilde` capped at stock; `H = 1` when the `a` row
    is zero) and test T-U1, the per-period conservation identity. Owning WO: **WO-005**.
    """
    raise NotImplementedError("PLAN section 2.6 - implemented in WO-005")


# ---------- planner (section 2.7) ----------


def make_planner_view(state: State, cfg: EnvConfig) -> PlannerView:
    """Build the planner's view of the world - the ONLY `State -> planner` function (CONTRACT rule
    5).

    Takes: the true `state` and `cfg`. Returns: a `PlannerView` containing reports, requests, audit
    results, measured quality, targets, the planner's I-O estimate and (from Phase 2) noisy
    downstream shortfall, and nothing else.

    The three information filters of PLAN section 2.7.5 are applied here, in this order:

        aggregation   at aggregation_level = "sector" the planner sees only sum_{i in j} claimed_i,
                      and allocation keys on planned need alone
        lag           the rules consume claims from `report_lag` periods ago (Phase 1: 0)
        channel noise claimed_i <- claimed_i * exp(xi_i), xi ~ N(0, sigma_ch**2),
                      key = (seed_env, "channel", t, i)

    All three branches must exist in the implementation even though the Phase-1 configuration makes
    each of them the identity; the identity case is what `tests/unit/test_planner.py` checks
    (WO-006 notes).

    The record is built once per period after the REPORT step. `audited` and `audit_meas` are all
    False / zero until `select_audits` and the audit measurement have run, after which the view is
    reissued with them filled.

    Constraints: no true quantity may cross this boundary - not `y`, not `S`, not `X`, not welfare.
    Binds: test T-B4 in `tests/behavioural/test_planner_blindness.py`, which plants sentinels in the
    state and asserts none reaches the view, plus the static check that no other function in
    `gosplan/env/planner.py` takes a `State`. Owning WO: **WO-006**.
    """
    raise NotImplementedError("PLAN section 2.4/2.7 - implemented in WO-006")


def update_targets(view: PlannerView, cfg: EnvConfig) -> Array:
    """Apply the ratchet and the growth directive to every target (PLAN section 2.7.1).

    Takes: `view` (claims and current targets as the planner knows them) and `cfg`. Returns: the new
    targets `(N,)`.

    Formula (PLAN section 2.7.1, verbatim):

        m_i    = fulfilment_measure(view, cfg)             # section 2.9.2; Phase 1: m_i = R_i
        rho_i  = m_i / T_i
        step_i = clip(rho_i - 1, -c_dn, +c_up)
        step_i = 0                       if |rho_i - 1| <= delta  # deadband; Phase 1 delta = 0
        T_i   <- max(T_min, (1 + g) * T_i * (1 + lambda * step_i))

    with `T_min = cfg.tech.target_floor_frac * T_0`. Under `report_lag > 0` the rule uses `m_i` from
    `report_lag` periods ago, which the view has already applied.

    `g > 0` is the forcing term added for finding F1. With `g = 0` and reports at target the map has
    a fixed point - that is the property test T-B2 checks, and it is exactly why `g` must be a
    treatment variable rather than a constant.

    Binds: test T-U4 in `tests/unit/test_planner.py` (fixed point at `rho = 1`, `g = 0`; the step is
    bounded by `c_up`/`c_dn`; the floor is respected; the deadband is inert outside `|rho - 1| <=
    delta`) and test T-B2 in `tests/behavioural/test_fixed_point.py` (`Padder` at `g = 0` keeps `T`
    constant; at `g > 0` it grows at exactly `(1 + g)`). Owning WO: **WO-006**.
    """
    raise NotImplementedError("PLAN section 2.7.1 - implemented in WO-006")


def fulfilment_measure(view: PlannerView, cfg: EnvConfig) -> Array:
    """Compute the fulfilment quantity the bonus and the ratchet key on (PLAN section 2.9.2).

    Takes: `view` and `cfg`. Returns: `m_i` `(N,)` in units of own good, by
    `cfg.incentive.objective_metric`:

        val               m_i = R_i
        net_output        m_i = R_i - sum_j p_j * alloc_ij / p_{s(i)}    # net of allocated inputs
                                                                        # valued at plan prices
        quality_weighted  m_i = R_i * q_hat_i,  q_hat_i = 1 + mu * (qbar_i - 1)

    where `alloc` is the allocation the planner itself computed for the same view via
    `allocate(view, cfg)`, `p` is `view.plan_prices`, and `q_hat` is `view.measured_quality` - every
    input is planner-side, so the function never needs the true state.

    `welfare` is deliberately absent from `ObjectiveMetric` (finding F6): the planner cannot key on
    a quantity it does not observe, and CONTRACT rule 6 keeps `welfare_true` out of every decision
    path.

    Binds: `tests/unit/test_planner.py` and `tests/unit/test_reward.py` (the `val` branch is the
    identity on claims; `net_output` falls as allocated inputs rise). Owning WO: **WO-006**.
    """
    raise NotImplementedError("PLAN section 2.9.2 - implemented in WO-006")


def allocate(view: PlannerView, cfg: EnvConfig) -> Array:
    """Allocate claimed supply across buyers - promises, not goods (PLAN section 2.7.2).

    Takes: `view` (last period's claims, requests, targets, the planner's I-O estimate) and `cfg`.
    Returns: `alloc` `(N, J)`, the promised quantity of each good to each buyer, in *claimed* units.

    Formula (PLAN section 2.7.2, verbatim):

        claimed_i = R_i                                       # lagged/noised/aggregated
        avail_j   = sum_{i: s(i)=j} (1 - phi_j) * claimed_i    # what the planner believes exists
        need_bj   = planner_io[s(b), j] * T_b                  # what the plan says buyer b needs
        w_bj      = (q_bj + 1e-6)**eta_q * (need_bj + 1e-6)**eta_n     # P1: eta_q = 0, eta_n = 1
        alloc_bj  = avail_j * w_bj / sum_b w_bj

    At `eta_q = 0` the request term is exactly 1 and requests are ignored, which is the Phase-1
    design (finding F7): requests are logged but inert. At the locked Phase-2 value `eta_q = 0.7`
    requests start to pay, which is the hoarding mechanism (PLAN section 4.2) - and the mechanism is
    a *rule about weights*, never an instruction to inflate a request (CONTRACT rule 7).

    Under `aggregation_level = "sector"` the weight collapses to the need term alone.

    Binds: `tests/unit/test_planner.py` - allocation sums to `avail_j` per good; `eta_q = 0` makes
    the result invariant to `requests`; the `1e-6` regularisers keep the weights finite when a need
    or a request is zero. Owning WO: **WO-006**.
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

    A claim above stock lowers `poolfill` for the whole good and therefore reduces every downstream
    buyer's receipt; a claim below stock leaves the difference sitting in `S`. Both are consequences
    of these four lines, not rules of their own - CONTRACT rule 7 forbids implementing either
    directly, and every `env/` diff is reviewed against that rule.

    Interface note for the v1 freeze (WO-013): this signature takes a `State` and lives in
    `gosplan/env/planner.py`, while CONTRACT rule 5 and the T-B4 static check forbid any function in
    that module except `make_planner_view` from accepting a `State`. `deliver` is physical execution
    of an allocation already decided from the view - it makes no planner decision and reads no
    claim except through `alloc` - so the T-B4 check whitelists it alongside `make_planner_view`.
    The lead should record the whitelist (or relocate `deliver` to `gosplan/env/step.py`) in
    `spec/CHANGELOG.md` at the v1 freeze.

    Binds: `tests/unit/test_planner.py` (`poolfill` in [0, 1]; delivery conservation; `claimed = 0`
    gives `fill = 1`), test T-U1 (per-period conservation, to 1e-9) and test T-B3 in
    `tests/behavioural/test_shortage_propagation.py`. Owning WO: **WO-006**.
    """
    raise NotImplementedError("PLAN section 2.7.3 - implemented in WO-006")


def select_audits(view: PlannerView, cfg: EnvConfig, t: int) -> Array:
    """Choose which enterprises to audit this period (PLAN section 2.7.4).

    Takes: `view`, `cfg`, and the plan period `t` (which keys the draw). Returns: a boolean `(N,)`.

    Formulas (PLAN section 2.7.4, verbatim):

        random    (Phase 1)  audited_i ~ Bernoulli(a),  key = (seed_env, "audit", t, i)
        targeted  (Phase 2)  probability a * (1 + kappa_t * downstream_shortfall_i),
                             clipped to [0, 1], active only when shortfall_visibility > 0

    `downstream_shortfall_i` is the planner's noisy knowledge of buyers' complaints and is therefore
    an information parameter, which is why `audit_mode` sits in `InformationConfig` even though the
    audit rate also enters the reward through the penalty (dual classification, PLAN section 3).

    Constraints: the draw goes through `draw` with purpose `audit` (CONTRACT rule 9); the selection
    is never observable to any agent before it happens (PLAN section 2.4).

    Binds: `tests/unit/test_planner.py` (empirical audit frequency matches `audit_rate`;
    determinism in `(seed_env, t)`). Owning WO: **WO-006**.
    """
    raise NotImplementedError("PLAN section 2.7.4 - implemented in WO-006")


# ---------- reporting / reward (sections 2.8-2.9) ----------


def process_reports(state: State, action: EnterpriseAction, cfg: EnvConfig) -> State:
    """Close the period's books and record the enterprise's claim (PLAN section 2.8).

    Takes: `state` at the REPORT step, `action` (whose `report_ratio` and `input_request` dimensions
    are read), and `cfg`. Returns: the updated state.

    Formulas (PLAN section 2.8, verbatim):

        S_i <- (1 - h) * S_i + y_i            # holding loss on carried stock, THEN this period's y
        R_i  = clip(rho_i_report, 0, rho_max) * T_i

    Order matters: the holding loss applies to the stock carried in, not to output just produced.
    Stock above `S_max = inventory_cap_mult * cap_i` is lost and the overflow is logged so it stays
    visible in the conservation identity (PLAN section 2.11). At this step the agent has already
    observed `S_i` and `y_i` exactly (Phase 1, `self_obs_noise = 0`), so the report is a choice made
    under full knowledge of the truth.

    The function also stores `last_report_ratio`, `last_report` (the claim in units, kept because
    the ratchet moves `T` later in the same period - PLAN section 2.5 steps 3 then 6) and
    `request`, clipped to `r_max * need_ij`, and it records whether the report sat at `rho_max`
    (CONTRACT rule 8).

    Binds: `tests/unit/test_reporting.py` (holding loss applied before `y` is added; the report
    clipped to `rho_max`) and test T-B8 (`BOUND_BINDING`). Owning WO: **WO-007**.
    """
    raise NotImplementedError("PLAN section 2.8 - implemented in WO-007")


def audit_and_penalise(state: State, audited: Array, cfg: EnvConfig, t: int) -> Array:
    """Measure audited stock and charge the penalty (PLAN section 2.8).

    Takes: `state` after `process_reports`, `audited` `(N,)` bool from `select_audits`, `cfg`, and
    the period `t`. Returns: `penalty` `(N,)`, zero wherever `audited` is False.

    Formulas (PLAN section 2.8, verbatim):

        S_hat_i   = S_i * exp(nu_i),  nu_i ~ N(0, sigma_aud**2)
                    key = (seed_env, "auditnoise", t, i)
        f_i       = max(0, R_i - S_hat_i) / T_i      if penalty_arg = positive_part   (Phase 1)
                  = |R_i - S_hat_i| / T_i            if penalty_arg = absolute
        Pen_i     = pen * f_i                        if penalty_form = proportional   (Phase 1)
                  = pen * 1[f_i > 0]                 if penalty_form = fixed
        penalty_i = 1[audited_i] * Pen_i

    The audit compares the claim to **stock on hand**, never to the period's production. That
    single choice is what makes accumulated stock protect against audits, and it is the reason
    `penalty_arg` and `h` are the mechanism parameters behind the hidden-reserves phenomenon (PLAN
    section 4.2). The penalty is expressed in ratio units - divided by `T_i` - per finding F9.

    Binds: test T-U8 in `tests/unit/test_reporting.py` - `positive_part` gives exactly 0 for any
    under-report while `absolute` does not; `audited = False` gives 0 regardless. Owning WO:
    **WO-007**.
    """
    raise NotImplementedError("PLAN section 2.8 - implemented in WO-007")


def bonus(rho: Array, cfg: EnvConfig) -> Array:
    """Bonus schedule on the fulfilment ratio, in ratio units (PLAN section 2.8).

    Takes: `rho` `(N,)`, the fulfilment ratio `m_i / T_i`, and `cfg`. Returns: `B(rho)` `(N,)`.

    Formula (PLAN section 2.8, verbatim):

        Lambda_w(x) = 1[x >= 0]                if w = 0        # strict >=, a true Heaviside
                    = 1 / (1 + exp(-x / w))    if w > 0
        B(rho)      = beta * Lambda_w(rho - 1) + s * clip(rho - 1, 0, rho_cap - 1)

    `rho_cap = inf` means no cap at all, hence no kink; the implementation must not clip in that
    branch (WO-007 notes).

    Named configurations (PLAN section 2.8):
        notched               w = 0,    rho_cap = 1.2
        smooth counterfactual w = 0.25, rho_cap = inf     # no discontinuity, no kink anywhere
        kink-only             w = 0.25, rho_cap = 1.2     # isolates kink bunching at the cap

    The smooth arm shares `beta` and `s` with the notched arm: it is equal-parameter, not
    equal-expected-value, and the DP of PLAN section 5 supplies the exact predicted distribution
    under both.

    Binds: test T-U3 in `tests/unit/test_reward.py` - `bonus` is monotone in `rho`; discontinuous at
    `rho = 1` iff `w = 0`; continuous with continuous derivative iff `w > 0` and `rho_cap = inf`.
    Owning WO: **WO-007**.
    """
    raise NotImplementedError("PLAN section 2.8 - implemented in WO-007")


def reward_scale(cfg: EnvConfig) -> float:
    """Analytic per-configuration reward scale (PLAN section 2.9.1; finding F9).

    Takes: `cfg`. Returns: `scale = 1 / B_cfg(rho_ref = 1.1)`, a single float computed from the
    configuration alone - never from running statistics.

    Why it exists: it makes the bonus at 110% fulfilment equal to 1 in every configuration, so a
    sweep over `beta` changes the economics (the notch relative to `pen` and `kappa`) and not the
    gradient magnitude. CONTRACT rule 4 forbids running reward normalisation outright, because
    running statistics change the effective reward over training and, with heavy-tailed penalties,
    shrink the notch in normalised units. Per-batch advantage normalisation inside PPO is permitted.

    Binds: test T-U2 in `tests/unit/test_reward.py` - `reward_scale(cfg) * bonus(1.1, cfg) == 1` to
    floating-point tolerance, for every configuration in the test matrix. Owning WO: **WO-007**.
    """
    raise NotImplementedError("PLAN section 2.9.1 - implemented in WO-007")


def enterprise_reward(
    state: State,
    cfg: EnvConfig,
    phase: Phase,
    cost: Optional[Array],
    penalty: Optional[Array],
    trade_surplus: Optional[Array],
) -> Array:
    """The only quantity any learner receives (PLAN section 2.9.1; CONTRACT rule 4).

    Takes: `state`, `cfg`, the current `phase`, and the three period quantities - `cost` `(N,)` at a
    PRODUCE step, `penalty` `(N,)` and `trade_surplus` `(N,)` at the REPORT step; each may be `None`
    in the phase where it does not apply. Returns: `r` `(N,)`.

    Formula (PLAN section 2.9.1, verbatim):

        PRODUCE step k:   r_ik = - scale * c_ik
        REPORT step:      r_i  =   scale * ( B(rho_i) - penalty_i + trade_surplus_i )
        scale             = reward_scale(cfg)             # analytic, per configuration

    with `trade_surplus == 0` throughout Phase 1, `penalty_i = 1[audited_i] * Pen_i` from
    `audit_and_penalise`, and `rho_i = fulfilment_measure(...) / T_i`.

    These are the only terms. CONTRACT rule 4 forbids per-step shaping, auxiliary rewards, curiosity
    terms, potential-based terms and running reward normalisation. Effort cost is a real cost paid
    when it is incurred, not shaping. Nothing in this function may read `welfare_true` or
    `val_measured` (CONTRACT rule 6).

    Binds: test T-B6 in `tests/behavioural/` - the reward is recomputed independently from the
    five-term formula on random states and must agree exactly. Owning WO: **WO-007**.
    """
    raise NotImplementedError("PLAN section 2.9.1 - implemented in WO-007")


def val_measured(state: State, cfg: EnvConfig) -> float:
    """The planner-side output aggregate, in plan prices (PLAN section 2.9.3).

    Takes: `state` at the end of a period, `cfg`. Returns: the scalar

        val_measured_t = sum_i p_{s(i)} * R_i * q_hat_i

    with `q_hat_i = 1 + mu * (qbar_i - 1)` (Phase 1: 1). This is what the planning system believes
    it produced.

    Logged only. CONTRACT rule 6: it never appears in any observation, reward or agent input; test
    T-B5 asserts this with sentinels. It is one half of `padding_index = val_measured / val_true`
    (PLAN section 2.9.4). Owning WO: **WO-007**.
    """
    raise NotImplementedError("PLAN section 2.9.3 - implemented in WO-007")


def val_true(state: State, cfg: EnvConfig) -> float:
    """The true output aggregate, in plan prices (PLAN section 2.9.3).

    Takes: `state` at the end of a period, `cfg`. Returns: the scalar

        val_true_t = sum_i p_{s(i)} * y_i * qbar_i

    where `y_i` is the period's true production and `qbar_i` its period-average quality
    (Phase 1: 1).

    Logged only, exactly as `val_measured` (CONTRACT rule 6). The ratio `val_measured / val_true` is
    the `padding_index` of PLAN section 2.9.4 and is at least 1 whenever output is fictitious.
    Owning WO: **WO-007**.
    """
    raise NotImplementedError("PLAN section 2.9.3 - implemented in WO-007")


def welfare_true(consumer: Array, cfg: EnvConfig) -> float:
    """Consumer welfare from this period's final deliveries (PLAN section 2.9.3).

    Takes: `consumer` `(J,)`, the final-demand sink's receipts from `deliver`, and `cfg`. Returns:
    the scalar CES index

        rho_ces   = (sigma_c - 1) / sigma_c
        welfare_t = ( sum_j alpha_j * consumer_j**rho_ces )**(1 / rho_ces)

    with `alpha_j = cfg.supply.ces_alpha` and `sigma_c = cfg.supply.ces_sigma`. The `sigma_c -> 1`
    limit is the Cobb-Douglas index `prod_j consumer_j**alpha_j` and must be implemented as an
    explicit branch (WO-007 must-pass list). `W = mean_t welfare_t` over the measurement window of
    PLAN section 4.4 (periods `t >= 2`).

    Logged only, and the strictest case of CONTRACT rule 6: no agent, no planner rule and no reward
    term may read it. It is the numerator of `welfare_ratio = W / W_oracle` (PLAN section 2.9.4),
    where `W_oracle` comes from PLAN section 6.2 (Phase 1 uses `W_truthful_max` as a clearly
    labelled placeholder). Owning WO: **WO-007**.
    """
    raise NotImplementedError("PLAN section 2.9.3 - implemented in WO-007")


def initial_prices(cfg: EnvConfig) -> Array:
    """Solve the cost-plus plan-price fixed point at `t = 0` (PLAN section 2.10).

    Takes: `cfg`. Returns: `p` `(J,)`, strictly positive.

    Formula (PLAN section 2.10, verbatim):

        p_j = (1 + m) * (kappa_labour + sum_k a_jk * p_k)

    solved as a fixed point over `j`, with `m = cfg.supply.price_markup`. `kappa_labour` is not a
    configuration parameter: `p` is homogeneous of degree 1 in it, and every use of `p` in this
    design is a ratio - `val_measured / val_true`, `p_j / p_{s(i)}` in the `net_output` measure - so
    the scale cancels. Fix it at 1.0 and record that as a normalisation.

    Convergence requires every row of `a` to satisfy `(1 + m) * sum_k a_jk < 1`, which is why
    `EnvConfig.validate` rejects `sum_k a_jk >= 1`. Phase 1 holds prices fixed thereafter
    (`price_lag = inf`); Phase 2 recomputes every `price_lag` periods using the planner's stale
    `planner_io`, never the true `a`.

    The price vector is not a modelling nuisance to be tuned: PLAN sections 2.9.4 and 7.5 require
    every headline table to be recomputed under three perturbed price vectors
    (`p_j * exp(u_j)`, `u ~ N(0, 0.3**2)`, fixed seeds), and a sign change in `specification_gap` is
    reported rather than suppressed.

    Binds: `tests/unit/test_prices.py` - the fixed point converges and every price is positive.
    Owning WO: **WO-007**.
    """
    raise NotImplementedError("PLAN section 2.10 - implemented in WO-007")


# ---------- env (section 2.5) ----------


@dataclass
class StepInfo:
    """Per-agent-step diagnostic payload: true quantities for the ledger, never for agents.

    Carries the `StepRecord`s produced by one agent-step (one per enterprise) plus the period-level
    scalars. CONTRACT rule 6 and the WO-009 card make the boundary explicit: `StepInfo` is written
    by the environment and read by `gosplan/metrics/ledger.py` and by lead-run experiments; no
    agent, no policy and no reward term may read it (WO-010 forbidden list: "any agent reading
    `StepInfo`").

    Owning WO: **WO-009**.
    """

    records: tuple[StepRecord, ...]  # one per enterprise, in enterprise-index order
    t_period: int
    k_step: int
    phase: Phase
    val_measured: float  # section 2.9.3; period-level, filled at the REPORT step
    val_true: float  # section 2.9.3; period-level, filled at the REPORT step
    welfare: float  # section 2.9.3; period-level, filled after DELIVER
    consumer: Array  # (J,) final-demand receipts this period
    flags: tuple[str, ...]  # run-level flags raised this step, e.g. "BOUND_BINDING" (rule 8)
    terminated: bool  # geometric termination fired this period (section 2.12)


class GosplanEnv:
    """The environment: the period schedule of PLAN section 2.5 as an explicit state machine.

    One plan period is:

        0. DELIVER      planner allocates from last period's claims (sections 2.7.2/2.7.3);
                        the consumer sink receives; X updated
        1. [P2] TRADE   bilateral matching (section 2.13)
        2. PRODUCE x M  agent chooses effort/quality/invest; yield realised; obs shows cumulative y
        3. REPORT       S <- (1-h)*S + y; agent observes S and y exactly; chooses report_ratio and
                        input_request (section 2.8)
        4. AUDIT        audit selection; measurement; penalty (section 2.8)
        5. REWARD       bonus - penalty (+ trade surplus) delivered; val and welfare logged (2.9)
        6. TARGET       ratchet + growth directive (section 2.7.1)
        7. TERMINATE?   geometric (section 2.12)

    Agents act `M + 1` times per period: `M` PRODUCE steps and one REPORT step. Action dimensions
    not relevant to the current phase are ignored rather than rejected.

    Binds: `tests/golden/*` (T-B7, agreement with `ref/ref_step.py` to 1e-9 on seeded trajectories),
    `tests/unit/test_conservation.py` (T-U1), `tests/unit/test_env_api.py`, and
    `tests/behavioural/test_termination.py` (T-B9). Owning WO: **WO-009** (LEAD).
    """

    cfg: EnvConfig
    state: State
    ledger: Optional[Ledger]

    def __init__(self, cfg: EnvConfig) -> None:
        """Construct the environment for one configuration.

        Takes: `cfg`, already validated. Returns: nothing. Stores the configuration, precomputes the
        quantities that never change within a run - plan prices from `initial_prices(cfg)`, the
        coverage weights `omega_j = a_{s(i)j} / sum_j a_{s(i)j}`, `reward_scale(cfg)`, the initial
        targets `T_0 = initial_target_frac * A_{s(i)} * cap_i` and `T_min` - and leaves `state`
        unset until `reset`.

        Realises: PLAN sections 2.5, 2.10. Owning WO: **WO-009**.
        """
        raise NotImplementedError("PLAN section 2.5 - implemented in WO-009")

    def reset(self, seed_env: int, seed_policy: int) -> tuple[Array, StepInfo]:
        """Start a new episode.

        Takes: `seed_env`, the root seed for every environment draw (shared across arms to obtain
        common random numbers, PLAN sections 2.15 and 4.3), and `seed_policy`, the separate policy
        stream. Returns: `(obs, info)` - the first observation `(N, d)` with `d = 12 + 3J` in
        Phase 1, and the opening `StepInfo`.

        Initial state (PLAN sections 2.1-2.2, 3): `target = T_0`, `capital = cap = 1`,
        `inv_output = 0`, `inv_inputs = 0`, all `last_*` fields zero, `last_fill = 1`,
        `t_period = 0`, `k_step = 0`, `phase = "produce"`, `plan_prices = initial_prices(cfg)`,
        `planner_io = a`, `alive = True`.

        Owning WO: **WO-009**.
        """
        raise NotImplementedError("PLAN section 2.5 - implemented in WO-009")

    def step(self, action: EnterpriseAction) -> tuple[Array, Array, bool, StepInfo]:
        """Advance one agent-step through the schedule above.

        Takes: `action`, a joint `EnterpriseAction`; only the dimensions in
        `active_action_dims(cfg)` and relevant to the current phase are read. Returns: `(obs,
        reward, done, info)` - `obs` `(N, d)`, `reward` `(N,)` from `enterprise_reward`, `done` for
        the whole episode (termination is a single global geometric draw with purpose `terminate`,
        PLAN section 2.12), and `StepInfo`.

        At a PRODUCE step this runs `produce_step` and returns `-scale * c_ik`. At the REPORT step
        it runs `process_reports`, `select_audits`, `audit_and_penalise`, the REWARD,
        `update_targets` and the termination draw, in that order; the next period opens with
        DELIVER, which consumes the `PlannerView` built from this period's reports.

        Nothing here may leak a true quantity into `obs` (CONTRACT rule 6), and every draw goes
        through `draw` (CONTRACT rule 9).

        Binds: T-B7 (golden parity with `ref/`), T-U1 (conservation), T-B9 (empirical continuation
        equals `tenure`; no observation field correlates with periods remaining). Owning WO:
        **WO-009** (LEAD).
        """
        raise NotImplementedError("PLAN section 2.5 - implemented in WO-009")

    def phase(self) -> Phase:
        """Return the phase the next call to `step` will execute.

        Takes: nothing. Returns: `"produce"` while `k_step < steps_per_period`, `"report"` at the
        period's last agent-step (PLAN section 2.5). Agents use it to mask inactive action
        dimensions; the environment never trusts an agent to have masked correctly.

        Owning WO: **WO-009**.
        """
        raise NotImplementedError("PLAN section 2.5 - implemented in WO-009")


# ---------- agents (section 6) ----------


class Agent(Protocol):
    """The interface every policy implements - heuristic, DP-derived, learned or LLM-driven.

    Implementations (PLAN section 6.1): `Random`, `TruthfulMyopic`, `Padder` (sanity only),
    `DPGreedy` in Phase 1 (WO-010, WO-014); `Berliner`, `Weitzman`, `Kornai`, `LLMMinistry` in
    Phase 2; `IPPO` as a thin adapter over a pinned reference PPO (WO-017).

    Two hard constraints. CONTRACT rule 6: `act` takes the observation and nothing else - no
    `State`, no `StepInfo`, no `PlannerView`; the PPO adapter's forward pass takes `obs` only, and
    test T-B5 checks the signature. CONTRACT rule 9: policy randomness draws from the `rng`
    argument, which belongs to the `seed_policy` stream and is separate from every environment
    draw.
    """

    def act(self, obs: Array, phase: Phase, rng: np.random.Generator) -> EnterpriseAction:
        """Choose a joint action.

        Takes: `obs` `(N, d)` laid out as `obs_spec(cfg)`; `phase`, so the agent can fill only the
        dimensions this step reads; `rng`, a generator from the `seed_policy` stream. Returns: an
        `EnterpriseAction` whose active dimensions lie inside the bounds of `action_spec(cfg)`.

        Reference behaviours (PLAN section 6.1, WO-010): `Random` is uniform over the active
        dimensions; `TruthfulMyopic` sets effort `clip(T / (A * cap), 0, 1)` per step so that
        `E[y] = T`, reports `rho = S / T` (truthful of stock) and requests exactly `need`;
        `Padder` reports `rho = 1` always at effort 0.3 and exists only to exercise the shortage
        channel in the Monte-Carlo sanity harness - it is never a baseline.

        Binds: T-B1 (no hard-coded pathology), T-B2 (fixed point), T-B3 (shortage propagation),
        T-B5 (welfare blindness). Owning WO: **WO-010**.
        """
        raise NotImplementedError("PLAN section 6.1 - implemented in WO-010")

    def reset(self) -> None:
        """Clear any per-episode internal state.

        Takes: nothing. Returns: `None`. Called once per episode before the first `act`. A stateless
        agent implements it as a no-op body that still raises until WO-010 supplies it.

        Owning WO: **WO-010**.
        """
        raise NotImplementedError("PLAN section 6.1 - implemented in WO-010")


# ---------- DP (section 5) ----------


@dataclass(frozen=True)
class DPGrid:
    """Discretisation of the single-enterprise dynamic program (PLAN section 5).

    Frozen and hashable so a solved `DPSolution` can be cached by `(EnvConfig.hash(), grid)`. The
    numbers below are PLAN section 5's grid, verbatim; they are the reference resolution for the
    G2 recovery criterion and for the regime map, and changing one changes what `b_hat_dp` means.
    """

    n_target: int = 80
    """Points on the target grid `T`, log-spaced on `[T_min, target_hi_mult * A * cap]`."""

    target_hi_mult: float = 4.0
    """Upper end of the target grid as a multiple of `A * cap`."""

    n_stock: int = 50
    """Points on the stock grid `S`, on `[0, S_max]` with `S_max = inventory_cap_mult * cap`."""

    effort_step: float = 0.05
    """Spacing of the period-effort action grid `e in {0, 0.05, ..., 1}`. Uniform effort across the
    `M` steps is optimal when nothing is observed within the period, at cost `M * kappa * e**2`."""

    rho_lo: float = 0.0
    """Lower end of the report action grid."""

    rho_hi: float = 3.0
    """Upper end of the report action grid; extended automatically when the optimum sits at the
    edge, and every edge hit is logged as a regime signal (PLAN section 5)."""

    rho_step: float = 0.02
    """Spacing of the report action grid."""

    gh_nodes: int = 9
    """Gauss-Hermite nodes used to take the yield expectation in log space."""

    value_tol: float = 1e-6
    """Value-iteration convergence tolerance (policy iteration is an acceptable alternative)."""

    sim_episodes: int = 200
    """Episodes simulated under the optimal policy to obtain the stationary report distribution."""

    sim_periods: int = 200
    """Periods per simulated episode for that distribution."""


@dataclass
class DPSolution:
    """The solved single-enterprise problem and everything Phase 1 reads off it (PLAN section 5).

    The DP is solved **before any multi-agent RL**, once per configuration, on a single enterprise
    with no input-output structure (`a = 0`, `phi = 1`), fixed capacity and geometric continuation
    `psi`. It is the ground truth for gate G2 criterion 1, the source of the `b_hat_dp` threshold
    for criterion 2, the exact no-manipulation counterfactual for the estimator-bias study (PLAN
    section 7.2), the way the rare-audit optimisation gap is sized (finding F9), and the policy the
    `DPGreedy` baseline replays inside the N-enterprise environment.

    Owning WO: **WO-014**; consumed by **WO-015** (regime map) and **WO-019** (DP vs PPO).
    """

    policy_effort: Array  # (n_target, n_stock) optimal period effort e*(T, S)
    policy_rho: Array  # (n_target, n_stock) optimal report ratio rho*(T, S)
    value: Array  # (n_target, n_stock) converged V(T, S)
    stationary_rho: Array  # samples of rho_report under the optimal policy (section 5)
    b_hat_dp: float  # DP-predicted bunching excess mass, estimator settings of section 4.5
    fictitious_padding: float  # mean max(0, R - S) / T under the stationary policy
    hidden_reserves: float  # mean max(0, S - R) / T after delivery
    mean_effort: float  # mean e under the stationary policy
    regime: RegimeLabel  # `classify_regime(self)`
    rho_edge_frac: float  # fraction of the stationary mass at the report grid edge
    n_iterations: int  # iterations to convergence
    converged: bool  # value iteration reached `grid.value_tol`
    grid: DPGrid  # the discretisation this solution was computed on
    config_hash: str  # EnvConfig.hash() of the configuration solved


def solve_single_enterprise(cfg: EnvConfig, grid: DPGrid) -> DPSolution:
    """Solve the single-enterprise dynamic program exactly (PLAN section 5).

    Takes: `cfg` and a `DPGrid`. Returns: a `DPSolution`.

    Setting: one enterprise, no input-output (`a = 0`, `phi = 1`), fixed capacity, geometric
    continuation `psi = cfg.incentive.tenure`.

    Period yield and transition (PLAN section 5, verbatim):

        y   = A * cap * e * epsbar,   epsbar ~ LogNormal(-sigmabar**2 / 2, sigmabar),
              sigmabar = sigma / sqrt(M)      # aggregation approximation, documented as such
        S'  = (1 - h) * S + y
        R   = rho * T
        audit against S' with rate a and noise sigma_aud
        S'' = S' - min(S', R)                  # delivery next period, phi = 1
        T'  = max(T_min, (1 + g) * T * (1 + lambda * clip(rho - 1, -c_dn, c_up)))

    Bellman operator (PLAN section 5, verbatim):

        V(T, S) = max_{e, rho} { E_eps[ B(rho) - M * kappa * e**2 - a * E_nu[Pen] ]
                                 + psi * gamma * E V(T', S'') }

    Expectations in `eps` and `nu` are taken by `grid.gh_nodes`-node Gauss-Hermite quadrature in log
    space; value iteration runs to `grid.value_tol`, or policy iteration. The action grid is
    vectorised. `B` and `Pen` are `bonus` and the penalty of `audit_and_penalise` - the DP must call
    the same functions the environment uses, never a re-derivation of them, so the two can never
    drift.

    Outputs: the optimal policy tables; the stationary distribution of `rho_report` obtained by
    simulating the policy `grid.sim_episodes` x `grid.sim_periods`; `b_hat_dp`; fictitious padding;
    hidden reserves; mean effort; and the regime label from `classify_regime`.

    Binds: `tests/unit/test_dp.py` (WO-014) - value iteration converges; with `a * pen -> inf` and
    `g = 0` the optimal policy reports truthfully; with `beta = 0` and `s = 0` optimal effort is 0;
    `DPGreedy` reproduces the DP policy inside the environment at `N = 1`. Forbidden in WO-014: any
    reinforcement learning. Owning WO: **WO-014**.
    """
    raise NotImplementedError("PLAN section 5 - implemented in WO-014")


def classify_regime(sol: DPSolution) -> RegimeLabel:
    """Label a solved DP by the behaviour it predicts (PLAN section 5).

    Takes: a `DPSolution` with its stationary report distribution filled in. Returns: one of

        "bunching"                  stationary mass of rho in [1.00, 1.005] exceeds 0.5 and the mass
                                    at the report grid edge is below 0.05
        "pad_to_cap"                mass at the grid edge exceeds 0.5
        "truthful_underfulfilment"  mean rho below 0.9 and fictitious padding below 0.01
        "mixed"                     otherwise

    The thresholds are PLAN section 5's, verbatim; they are the classifier the regime map colours by
    (WO-015) and the human reads at gate G1 when picking Phase-1 values from the interior of the
    bunching region. A grid-edge hit is a regime signal, not an artefact to be smoothed away.

    Binds: `tests/unit/test_dp.py` - the classifier on synthetic distributions with known labels.
    Owning WO: **WO-014**.
    """
    raise NotImplementedError("PLAN section 5 - implemented in WO-014")


# ---------- ledger / metrics (section 4) ----------


@dataclass
class StepRecord:
    """One row of the ledger: one enterprise, one agent-step (PLAN sections 2.2, 4; WO-011).

    Carries every quantity the phenomena of PLAN section 4.1 and the forensic estimators of PLAN
    section 7.3 need, including the true quantities that make the conservation identity checkable.
    It is written by the environment and read only by metrics and lead-run experiments - never by an
    agent (CONTRACT rule 6).

    Fields are grouped: identifiers, the PLAN section 2.2 state columns, then the WO-011 additions
    (`y, R, rho, S_pre, S_post, audited, S_hat, f, Pen, fill, deliv, consumer, val_measured,
    val_true, welfare, at_bound`), then the action and conservation columns.

    Owning WO: **WO-011**.
    """

    run_hash: str  # EnvConfig.hash() of the run that produced this row
    episode: int
    t_period: int
    k_step: int
    phase: Phase
    enterprise: int
    sector: int

    target: float  # T_i
    capital: float  # Kap_i
    inv_output_pre: float  # S_i before this step (S_pre)
    inv_output_post: float  # S_i after this step (S_post)
    inv_inputs: tuple[float, ...]  # (J,) X_ij after this step
    cum_output: float
    cum_cost: float
    quality_acc: float
    last_report_ratio: float
    last_penalty: float
    last_fill: float
    request: tuple[float, ...]  # (J,) q_ij
    need: tuple[float, ...]  # (J,) need_ij, so request inflation q/need is recoverable

    effort: float  # e_ik, per step - the storming Gini of section 4.1 row 2 needs it
    quality: float  # q_ik; Phase 1 inert
    invest: float  # v_ik; Phase 1 inert
    output: float  # y_ik at a PRODUCE step, y_i at the REPORT step
    cost: float  # c_ik
    coverage: float  # H_ik
    reward: float  # the reward actually delivered this step

    report: float  # R_i, the claim in units
    report_ratio: float  # rho_i
    at_bound: bool  # rho_i sat at rho_max (CONTRACT rule 8, test T-B8)
    audited: bool
    audit_meas: float  # S_hat_i
    penalty_arg: float  # f_i
    penalty: float  # Pen_i charged
    fill: float  # fill_i
    shipped: float  # shipped_i
    alloc: tuple[float, ...]  # (J,) alloc_ij promised to this enterprise
    deliv: tuple[float, ...]  # (J,) deliv_ij physically received
    input_consumed: tuple[float, ...]  # (J,) inputs consumed this step
    holding_loss: float  # stock lost to h this period
    cap_overflow: float  # stock lost to S_max this period
    trade_volume: float  # executed trade volume; Phase 2, 0 in Phase 1

    consumer: tuple[float, ...]  # (J,) period-level, repeated on each row for convenience
    val_measured: float  # period-level (section 2.9.3)
    val_true: float  # period-level (section 2.9.3)
    welfare: float  # period-level (section 2.9.3)


class Ledger:
    """Append-only store of `StepRecord`s for one run, plus the run's flags (WO-011).

    One record per enterprise per agent-step. The ledger is the input to every metric in PLAN
    section 4.1, to the reconciliation test of PLAN section 7.3, and to the run manifest. Nothing an
    agent can read touches it (CONTRACT rule 6).

    Owning WO: **WO-011**.
    """

    records: list[StepRecord]
    flags: set[str]

    def append(self, rec: StepRecord) -> None:
        """Append one record.

        Takes: `rec`. Returns: `None`. Also maintains the run's flag set: when the running fraction
        of reports with `at_bound` exceeds 1% the ledger raises `BOUND_BINDING`, which the manifest
        carries and every report of the run must display (CONTRACT rule 8, test T-B8). A bound is
        never silently moved to clear the flag.

        Owning WO: **WO-011**.
        """
        raise NotImplementedError("PLAN section 4 - implemented in WO-011")

    def to_parquet(self, path: str) -> None:
        """Write the ledger to a columnar file.

        Takes: `path`. Returns: `None`. One row per `StepRecord`, tuple-valued fields exploded to
        one column per good (`inv_inputs_0 ... inv_inputs_{J-1}` and so on) so the file is readable
        without this module. `pyarrow` is the writer; it is a runtime dependency of
        `gosplan/metrics/ledger.py` only and is deliberately not imported by this interface file.

        Binds: `tests/unit/test_ledger.py` - parquet round-trip preserves every column and dtype.
        Owning WO: **WO-011**.
        """
        raise NotImplementedError("PLAN section 4 - implemented in WO-011")


def write_manifest(run_dir: str, cfg: EnvConfig, extra: dict) -> None:
    """Write `runs/<hash>/manifest.json` for a run (CONTRACT rule 10).

    Takes: `run_dir`, the run directory (named by `cfg.hash()`); `cfg`; `extra`, a mapping of the
    run-specific entries listed below. Returns: `None`.

    The manifest carries, at minimum: the config hash, the full configuration, `SPEC_VERSION`, the
    git hash of the working tree, both seeds, the reference-PPO version, the estimator version, the
    LLM model ids and versions, the solver version and its optimality gap, and every flag raised
    (notably `BOUND_BINDING`). Fields that do not apply to a run are written as `null` rather than
    omitted, so a missing field is always a bug and never an ambiguity.

    Binds: `tests/unit/test_ledger.py` - every rule-10 field is present. Owning WO: **WO-011**.
    """
    raise NotImplementedError("PLAN section 4 - implemented in WO-011")


def phenomenon_bunching(ledger: Ledger, cfg: EnvConfig) -> dict[str, float]:
    """Excess mass of the report distribution just above target - PLAN section 4.1 row 1.

    Class: **pipeline check**, Phase 1. If it fails to appear the optimiser is broken; it is not
    evidence for Claim A.

    Takes: a `Ledger` and `cfg`. Returns: a mapping with at least `excess_mass`, `hole_mass`, `se`,
    `ci_lo`, `ci_hi`, `n_obs`, `at_bound_frac`.

    Pre-registered estimator settings (PLAN section 4.5, hard-coded as defaults and recorded in the
    manifest): bins of width 0.005 over `rho` in [0.6, 1.4]; excluded window [0.95, 1.02];
    polynomial of degree 7 fitted outside the window; excess mass `b_hat = (observed -
    counterfactual mass in [1.00, 1.02]) / mean counterfactual density in the window`; hole mass
    computed identically on [0.95, 1.00); standard error by bootstrap over seeds. Only periods `t
    >= 2` enter, per the measurement window of PLAN section 4.4, and reports at `rho_max` are
    included in the histogram and flagged.

    Implementation: call `forensics_core.bunching.estimate(x, window_lo, window_hi, bin_width,
    degree, excl_lo, excl_hi)`. The import lives inside a `try/except ImportError` that falls back
    to `gosplan.metrics._fallback`, whose signatures are identical (PLAN section 7.3); this
    interface file never imports `forensics_core`.

    Binds: `tests/unit/test_phenomena_p1.py` - a synthetic density with known excess mass is
    recovered within 5%, hole mass likewise, and the fallback and `forensics_core` signatures agree.
    Owning WO: **WO-016**.
    """
    raise NotImplementedError("PLAN section 4.1 - implemented in WO-016")


def phenomenon_padding(ledger: Ledger, cfg: EnvConfig) -> dict[str, float]:
    """Fictitious output - PLAN section 4.1 row 4.

    Class: **pipeline check**, Phase 1.

    Takes: a `Ledger` and `cfg`. Returns: a mapping with at least `padding` - `mean_i max(0, R_i -
    S_i) / T_i` over the measurement window - together with `padding_index = val_measured /
    val_true` (PLAN section 2.9.4) and the elasticity of padding with respect to `audit_rate *
    penalty_scale` across the levels gate G1 recorded.

    The elasticity is compared against the DP's prediction: gate G2 criterion 3 requires learned
    fictitious padding to be monotone decreasing in `a * pen` and within 0.03 of the DP's value at
    each of the three levels (PLAN section 4.5).

    Binds: `tests/unit/test_phenomena_p1.py`. Owning WO: **WO-016**.
    """
    raise NotImplementedError("PLAN section 4.1 - implemented in WO-016")


def phenomenon_hidden_reserves(ledger: Ledger, cfg: EnvConfig) -> dict[str, float]:
    """Shaving and hidden reserves - PLAN section 4.1 row 7. **Phase 2, held out.**

    Class: **emergence** (Claim A). Operationalisation: `mean_i max(0, S_i - R_i) / T_i` after
    delivery; it must vanish when `g = 0` and `penalty_arg = "absolute"`. The ledger-level part uses
    `forensics_core.reconciliation` (PLAN section 7.3).

    HELD OUT (PLAN section 4.1): no plot, table or test of this quantity may be produced before the
    Phase-2 acceptance run - not during Phase 1 and not while debugging its mechanism. The Monte
    Carlo sanity harness of WO-012 may assert only conservation and boundedness on the same
    mechanism, never a direction. Its mechanism parameters (`g`, `penalty_arg`, `h`) are locked now
    in PLAN section 4.2 and may be changed only in a new, separately pre-registered study.

    Takes: a `Ledger` and `cfg`. Returns: a mapping with at least `hidden_reserves`,
    `reconciliation_stat` and `p_value`. Owning WO: **WO-030**.
    """
    raise NotImplementedError("PLAN section 4.1 - implemented in WO-030")


def phenomenon_storming(
    ledger: Ledger, baseline_ledger: Ledger, cfg: EnvConfig
) -> dict[str, float]:
    """Effort deferral beyond input timing - PLAN section 4.1 row 2. **Phase 2, held out.**

    Class: **emergence** (Claim A). Operationalisation: `Gini_k(e_ik)` of *effort* across the steps
    of a period, minus the same Gini computed for the truthful-myopic heuristic under the same
    delivery-timing draws (common random numbers). A positive excess means agents defer effort
    beyond what input arrival forces - which is why the comparison ledger is a required argument
    and not an optional convenience.

    HELD OUT (PLAN section 4.1), on the same terms as `phenomenon_hidden_reserves`. Mechanism
    parameters locked in PLAN section 4.2: `delivery_timing = "stochastic"` with
    `arrival_probs = (0.25, 0.25, 0.25, 0.25)` - uniform-random arrival, so there is no mechanical
    backloading and any Gini excess is behavioural - and `yield_sigma` at x1.

    Takes: the run's `ledger`, the truthful-myopic `baseline_ledger` under identical draws, and
    `cfg`. Returns: a mapping with at least `gini`, `gini_baseline`, `excess`. Owning WO:
    **WO-030**.
    """
    raise NotImplementedError("PLAN section 4.1 - implemented in WO-030")


def phenomenon_hoarding(
    ledger: Ledger, baseline_ledger: Ledger, cfg: EnvConfig
) -> dict[str, float]:
    """Input hoarding and propagated shortage - PLAN section 4.1 row 5. **Phase 2, held out.**

    Class: **emergence** (Claim A). Operationalisation: request inflation `q_ij / need_ij`, and
    `corr(X_ij, 1 - fill_downstream)`; both are reported against the truthful-myopic baseline run
    under common random numbers. The cross-sectional part uses `forensics_core.dispersion`
    (PLAN section 7.3).

    HELD OUT (PLAN section 4.1). Mechanism parameters locked in PLAN section 4.2:
    `alloc_eta_request = 0.7`, `input_complementarity = 8`, `input_holding_loss = 0.01`. Note that
    test T-B3 asserts shortage *propagation* only; the direction of hoarding is asserted nowhere.

    Takes: `ledger`, `baseline_ledger`, `cfg`. Returns: a mapping with at least `request_inflation`,
    `request_inflation_baseline`, `corr_stock_shortfall`, `dispersion_stat`. Owning WO: **WO-030**.
    """
    raise NotImplementedError("PLAN section 4.1 - implemented in WO-030")


def phenomenon_blat(ledger: Ledger, cfg: EnvConfig) -> dict[str, float]:
    """Horizontal barter - PLAN section 4.1 row 6. **Phase 2, held out.**

    Class: **emergence** (Claim A). Operationalisation: executed trade volume divided by total
    intermediate allocation; it must exceed the truthful-myopic baseline, which never trades.

    HELD OUT (PLAN section 4.1). Mechanism parameters locked in PLAN section 4.2:
    `horizontal_visibility = 1.0`, `trade_tau = 0.05`.

    Takes: `ledger` and `cfg`. Returns: a mapping with at least `trade_volume_share`,
    `n_matched_pairs`, `mean_surplus`. Owning WO: **WO-030**.
    """
    raise NotImplementedError("PLAN section 4.1 - implemented in WO-030")


def phenomenon_quality(ledger: Ledger, cfg: EnvConfig) -> dict[str, float]:
    """Quality degradation - PLAN section 4.1 row 3. **Phase 2.**

    Class: **pipeline check** (not part of Claim A: it is a direct optimum of the reward under agent
    control, finding F8). Operationalisation: mean `qbar` under `objective_metric = "val"` against
    the same configuration at `objective_metric = "quality_weighted"` with
    `quality_measurability = 1`.

    Takes: `ledger` and `cfg`. Returns: a mapping with at least `mean_quality` and
    `mean_quality_weighted`. Owning WO: **WO-030**, with the mechanism itself from **WO-021**.
    """
    raise NotImplementedError("PLAN section 4.1 - implemented in WO-030")


# ---------- P2 sketches (signatures only; bodies frozen at P2 revision) ----------


def match_trades(state: State, offers: Array, cfg: EnvConfig, t: int) -> tuple[State, Array]:
    """Bilateral horizontal trade after DELIVER (PLAN section 2.13). **Phase-2 sketch.**

    Takes: `state`, `offers` `(N, J)` in [-1, 1] (positive offers up to that fraction of `X_ij`,
    negative wants up to that fraction of `need_ij`), `cfg`, and the period `t`. Returns:
    `(state, surplus)` with `surplus` `(N,)`.

    Sketch (PLAN section 2.13, to be frozen at the Phase-2 spec revision): visible counterparties
    are a random subset of size `horizontal_visibility * (N - 1)`, drawn with purpose
    `trade_visibility`; matching is greedy by largest complementary pair; execution is at
    plan-price parity with a transaction cost `tau` per unit. Surplus is computed by the
    environment and never self-reported:

        delta_h_i = p_{s(i)} * [ Yhat_i(X_after) - Yhat_i(X_before) ]

    where `Yhat_i` is next step's intended output at the agent's last effort. Wash trades are
    unprofitable because `tau > 0` and `delta_h` is the same function for both sides.

    The signature is frozen now so no type moves later (PLAN section 0, finding F14); the behaviour
    is deliberately under-specified until the Phase-2 revision. `trade_surplus` enters the reward
    only through the term already named in CONTRACT rule 4. Owning WO: **WO-024**.
    """
    raise NotImplementedError("PLAN section 2.13 - implemented in WO-024")


def ministry_forward(view: MinistryView, cfg: EnvConfig) -> Array:
    """Forward one ministry's enterprises' claims upward (PLAN section 2.14). **Phase-2 sketch.**

    Takes: a `MinistryView` and `cfg`. Returns: the forwarded values `(n_i,)`.

    Sketch (PLAN section 2.14, to be frozen at the Phase-2 spec revision):

        R_tilde_i = pi * R_i + (1 - pi) * [ Rbar_i_prev + kappa_m * max(0, T_i - R_i) ]

    with `pi = cfg.information.ministry_passthrough` (1 = transparent): the ministry smooths toward
    its own last aggregate and pads shortfalls. `kappa_m` is not yet a configuration field; it is
    fixed at the Phase-2 revision, at which point it joins `InformationConfig` with a CHANGELOG
    entry.

    The LLM adapter of PLAN section 7.4 replaces this function with a model call that receives the
    same `MinistryView` rendered as text and returns per-enterprise forwarded values plus a
    free-text justification, with strict-JSON parsing, one retry, and the documented fallback
    action `pi = 1`. Pinned model versions and every prompt and completion are logged (CONTRACT
    rule 10).

    Owning WO: **WO-025** (rule-based), **WO-026** (LLM adapter).
    """
    raise NotImplementedError("PLAN section 2.14 - implemented in WO-025")


def solve_oracle(cfg: EnvConfig, horizon: int, clairvoyant: bool, seed_env: Optional[int]) -> dict:
    """Full-information planning benchmark (PLAN section 6.2). **Phase-2 sketch.**

    Takes: `cfg`; `horizon`, the planning horizon in periods (PLAN section 6.2 uses 12);
    `clairvoyant`, selecting the per-seed bound that knows the realised noise rather than the
    non-anticipative solution; `seed_env`, required when `clairvoyant` is True and ignored
    otherwise. Returns: a mapping with at least `welfare`, `val`, `optimality_gap`, `solver`,
    `solver_version`, `status`.

    Sketch (PLAN section 6.2, frozen at the Phase-2 revision): a non-anticipative expected-value MIP
    over the full true state at mean yields, with the configured nonconvexities (setup costs become
    binaries, increasing returns a piecewise-linear approximation), solved once per configuration
    with an open-source solver (HiGHS or CBC through OR-Tools or Pyomo; the lead verifies
    availability). `W_oracle` in the headline metrics of PLAN section 2.9.4 is the expected-value
    MIP's welfare; the clairvoyant number is reported as an upper bound only, never as the
    denominator. The solver's optimality gap is recorded in the run manifest (CONTRACT rule 10).

    In Phase 1, `welfare_ratio` uses `W_truthful_max` as a placeholder denominator and every table
    that reports it says so (PLAN section 2.9.4). Owning WO: **WO-027**.
    """
    raise NotImplementedError("PLAN section 6.2 - implemented in WO-027")


__all__ = [
    # version and aliases
    "SPEC_VERSION",
    "Array",
    # enums
    "Arm",
    "Phase",
    "ObjectiveMetric",
    "PenaltyForm",
    "PenaltyArg",
    "AuditMode",
    "AggLevel",
    "DeliveryTiming",
    "HorizonMode",
    "ParamSharing",
    "RegimeLabel",
    "Purpose",
    "Dist",
    # config
    "SupplyConfig",
    "IncentiveConfig",
    "InformationConfig",
    "TechConfig",
    "EnvConfig",
    "load_config",
    "p1_default_config",
    # state, actions, views, specs
    "State",
    "EnterpriseAction",
    "PlannerView",
    "MinistryView",
    "obs_spec",
    "action_spec",
    "active_action_dims",
    # rng
    "draw",
    # production
    "coverage",
    "produce_step",
    # planner
    "make_planner_view",
    "update_targets",
    "fulfilment_measure",
    "allocate",
    "deliver",
    "select_audits",
    # reporting and reward
    "process_reports",
    "audit_and_penalise",
    "bonus",
    "reward_scale",
    "enterprise_reward",
    "val_measured",
    "val_true",
    "welfare_true",
    "initial_prices",
    # env
    "StepInfo",
    "GosplanEnv",
    # agents
    "Agent",
    # DP
    "DPGrid",
    "DPSolution",
    "solve_single_enterprise",
    "classify_regime",
    # ledger and metrics
    "StepRecord",
    "Ledger",
    "write_manifest",
    "phenomenon_bunching",
    "phenomenon_padding",
    "phenomenon_hidden_reserves",
    "phenomenon_storming",
    "phenomenon_hoarding",
    "phenomenon_blat",
    "phenomenon_quality",
    # P2 sketches
    "match_trades",
    "ministry_forward",
    "solve_oracle",
]
