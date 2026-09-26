"""Parameter registry - PLAN section 3 transcribed as data.

Realises: PLAN section 3 (parameter registry), with the dual-classification and reporting rule of
PLAN section 4.3 and the locked Phase-2 mechanism values of PLAN section 4.2. Owning work order:
**WO-001** (Spec v0, CONTRACT, registry; LEAD). Consumed by **WO-003** (`gosplan/config.py`), whose
`tests/unit/test_config.py` asserts that `p1_default_config()` agrees field by field with the data
below - the two must never drift.

**This module is data, not a stub.** `REGISTRY` carries one `ParamSpec` per parameter of the PLAN
section 3 table: its arm, its Phase-1 default, its sweep or prior range, its source status and
whether it is provisional. Only the three lookup helpers at the foot of the file execute anything,
and each is a single expression over the literal table.

CONTRACT rule 11, verbatim:

    11. PARAMETER ARMS. The INFO/INC/SUPPLY/TECH classification in gosplan/params.py is a
        design decision. Changing an arm assignment requires a CHANGELOG entry.

The arm split is what makes the C_OGAS / C_INC contrasts of PLAN section 4.3 well defined, so an
arm is never adjusted to make a contrast come out: moving one is a `spec/CHANGELOG.md` entry
(version, reason, affected work orders) signed off by the lead. `audit_rate` is the one
dual-classified row - INFO by architecture and also a reward input through the audit penalty of
PLAN section 2.8 - and PLAN section 4.3 requires it to be reported separately and never folded into
the C_OGAS information contrast; the comment on its entry repeats this.

Naming. `name` is the `EnvConfig` field name that holds the parameter (unique across
`SupplyConfig`, `IncentiveConfig`, `InformationConfig` and `TechConfig`), not the PLAN symbol, so a
test can map a registry row onto a configuration field directly; the PLAN symbol (`a`, `lambda`,
`rho_cap`, ...) appears in `notes`. Rows that PLAN section 3 writes as one line covering several
parameters (`ratchet_cap_up` / `ratchet_cap_dn`, `penalty_form` / `penalty_arg`, `setup_cost` /
`irs_alpha`, `capital_dep` / `invest_lag`, `holding_loss` / `input_holding_loss`, `price_markup` /
`price_lag`, the `N, J, rho_max, r_max, T_min, S_max` row, the `horizon_mode, P_min, P_max` row and
the PPO row) are split into one entry per parameter, each carrying the same `plan_section` and a
note naming the shared row.

What is deliberately **not** here. PLAN section 3 does not tabulate every `EnvConfig` field. These
fields exist in `gosplan/config.py` with defaults declared and justified in `spec/spec.py`, and are
absent from `REGISTRY` because inventing registry rows for them would put values in a design
document that the design document does not contain: `sector_of`, `io_matrix`, `productivity`,
`arrival_probs` (its Phase-2 locked value is in PLAN section 4.2), `ces_alpha`, `ces_sigma`,
`quality_matters`, `quality_cost`, `n_ministries`, `seed_env`, `seed_policy` and
`EnvConfig.spec_version`. The `p1_default_config()` cross-check of WO-003 therefore compares the
fields `REGISTRY` names and asserts the remainder against `spec/spec.py` instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Arm = Literal["info", "inc", "supply", "tech"]
"""Parameter-arm classification (PLAN section 3): `info` = information architecture, `inc` =
incentive structure, `supply` = production/supply structure, `tech` = technical. Mirrors the `Arm`
alias of `spec/spec.py`. Changing a parameter's arm requires a CHANGELOG entry (CONTRACT rule
11)."""

PhaseTag = Literal["P1", "P2"]
"""Which phase of PLAN section 3 gives the parameter its value. `P1` where the table's "P1 value"
column carries a value - the parameter is configured from Phase 1 onwards, whether or not that
value switches its mechanism on. `P2` where the table's Phase-1 cell is "-" (`invest_lag`,
`trade_tau`), i.e. PLAN section 3 states no Phase-1 value because the mechanism is inert in Phase
1; for those two the `default` field carries the Phase-2 locked value or the placeholder declared
in `spec/spec.py`, and `notes` says which. It is **not** a claim about when a code path is written:
that is the work-order DAG of PLAN section 12."""

SourceStatus = Literal[
    "historical",
    "qualitative",
    "unsourced",
    "design",
    "engineering",
    "mechanism-toggle",
]
"""Provenance of a parameter's value and range, taken from the "Source status" column of PLAN
section 3: `historical` (a documented schedule or series is the anchor, even where the lead has
still to source it, PLAN section 15); `qualitative` (the literature constrains the structure or the
direction, not the number); `unsourced` (no defensible range - the value is a prior, and G1 picks
it from the DP regime map); `design` (a modelling choice of PLAN section 2); `engineering` (a
bound, size or horizon fixed across all arms); `mechanism-toggle` (the row exists to switch a
mechanism on, and PLAN section 3 marks it a toggle or a Phase-2 mechanism).

`SOURCED` / `PARTIAL` / `UNSOURCED` in `docs/params_sources.md` is a different, finer vocabulary
answering a different question (has WO-000 sourced this yet); WO-000 has not been executed, so no
value here is a historical claim."""


@dataclass(frozen=True)
class Range:
    """A continuous sweep or prior range `[lo, hi]` from PLAN section 3.

    Fields only - a skeleton dataclass carries no methods (no membership test, no sampler). WO-015
    (regime map) and the Phase-3 Saltelli design of PLAN section 4.3 read `lo` and `hi` directly
    and must state these ranges as an assumption in the same table as any Sobol index they report.
    """

    lo: float
    """Lower end of the range, inclusive, exactly as PLAN section 3 writes it."""

    hi: float
    """Upper end of the range, inclusive, exactly as PLAN section 3 writes it."""


@dataclass(frozen=True)
class Choices:
    """A discrete sweep grid `{...}` from PLAN section 3.

    Fields only. `values` preserves the order in which PLAN section 3 lists the grid, so a sweep
    enumerates it reproducibly; the Phase-1 default is always one of the members.
    """

    values: tuple[object, ...]
    """The grid members in PLAN order, e.g. `(0, 1, 2)` for `report_lag`, `("random", "targeted")`
    for `audit_mode`, `(1.1, 1.2, float("inf"))` for `overfulfilment_cap`."""


Sweep = Range | Choices | None
"""What PLAN section 3's "Sweep / prior range" cell says: a `Range` for an interval, a `Choices` for
a `{...}` grid, and `None` where the cell reads "fixed" or "-" (the parameter is not swept). `None`
is a statement about the design, not a missing value: CONTRACT rule 8 forbids widening a bound to
fix a result, and a parameter with `sweep is None` is not a treatment variable in any contrast."""

Default = bool | int | float | str | tuple[float, ...]
"""Type of a Phase-1 default: a scalar, a string for the `Literal`-typed parameters, or a tuple for
the per-sector vectors (`yield_sigma`, `final_demand_share`) and for the annealed PPO entropy
coefficient. Infinity is `float("inf")` (`price_lag`; a member of the `overfulfilment_cap` and
`input_complementarity` grids) and is serialised as the string "inf" by `EnvConfig.hash`
(WO-003)."""


@dataclass(frozen=True)
class ParamSpec:
    """One row of the PLAN section 3 registry.

    Frozen and hashable so a registry row is a value: it can key a sweep dictionary and it is safe
    to copy into a run manifest (CONTRACT rule 10). Fields only - no `__post_init__`, no computed
    property, no validation. Range checking of an actual configuration is `EnvConfig.validate`
    (WO-003), which is where PLAN section 3's ranges are enforced.
    """

    name: str
    """The `EnvConfig` field this row configures, unique across the four sub-configurations (e.g.
    `"audit_rate"`, `"ratchet_lambda"`, `"yield_sigma"`). The five PPO rows are the exception: they
    are prefixed `ppo_` and name no `EnvConfig` field (see their notes)."""

    arm: Arm
    """The parameter's arm. A design decision; changing it requires a CHANGELOG entry (CONTRACT
    rule 11). `audit_rate` is INFO and dual (see its notes and PLAN section 4.3)."""

    phase: PhaseTag
    """`"P1"` when PLAN section 3 gives a Phase-1 value, `"P2"` when its Phase-1 cell is "-"."""

    default: Default
    """The Phase-1 value of PLAN section 3, transcribed exactly, or - for the two `phase == "P2"`
    rows - the value `spec/spec.py` declares, with the provenance stated in `notes`."""

    sweep: Sweep
    """The sweep or prior range of PLAN section 3: `Range`, `Choices`, or `None` when the cell
    reads "fixed" or "-"."""

    source: SourceStatus
    """Provenance of the value and the range; the PLAN section 3 cell is quoted in `notes` wherever
    it says more than the six-value vocabulary can carry.

    The authority for this field is `docs/params_sources.md` (WO-000), not PLAN section 3's own
    Source status column. Where the memo could not retrieve a source for a figure PLAN section 3
    calls historical, this field reads `unsourced` and `notes` records which memo item looked and
    what it found. A `source` of `historical` is a claim that a document was read; it is never
    written here on the strength of the parameter merely sounding historical."""

    provisional: bool
    """`True` for the daggered rows of PLAN section 3 - `audit_rate`, `ratchet_lambda`,
    `growth_directive`, `overfulfilment_slope`, `penalty_scale`, `effort_cost` - whose defaults are
    placeholders until a human picks Phase-1 values from the interior of the DP regime map at gate
    G1 and records them in `runs/G1_decision.md` before any training run (PLAN sections 5, 13).
    Nothing in the repository may treat a provisional value as evidence about anything."""

    notes: str
    """The symbol PLAN section 2 uses, the role of the parameter, any locked Phase-2 value (PLAN
    section 4.2), the PLAN section 3 source cell where it is richer than `source`, and the work
    order that reads it. This is the field an implementer reads."""

    plan_section: str
    """The PLAN sections this row is governed by, semicolon-separated, always including `3`."""


# ---------------------------------------------------------------------------------------------
# REGISTRY - PLAN section 3, in table order: INFO, INC, SUPPLY, TECH.
#
# `audit_rate` is DUAL-CLASSIFIED (PLAN section 3): it sits in the INFO arm and also enters the
# enterprise reward through the audit penalty of PLAN section 2.8. PLAN section 4.3 therefore
# requires it to be reported SEPARATELY - the C_AUDIT contrast moves it, and it is never folded
# into the C_OGAS information contrast, whose conclusion is about architecture alone.
# ---------------------------------------------------------------------------------------------

REGISTRY: tuple[ParamSpec, ...] = (
    # ---------------- INFO: information architecture ----------------
    ParamSpec(
        name="report_lag",
        arm="info",
        phase="P1",
        default=0,
        sweep=Choices((0, 1, 2)),
        source="qualitative",
        provisional=False,
        notes=(
            "Periods of delay before a report reaches the planner's rules. PLAN section 3 source "
            "cell: qualitative (annual reporting cycles). The lag branch exists in WO-006 even "
            "though the Phase-1 value 0 makes it the identity, and the identity case is tested."
        ),
        plan_section="3; 2.7.1; 2.7.5",
    ),
    ParamSpec(
        name="aggregation_level",
        arm="info",
        phase="P1",
        default="enterprise",
        sweep=Choices(("enterprise", "sector")),
        source="qualitative",
        provisional=False,
        notes=(
            "Level at which the planner observes claims. At `sector` it sees only the per-sector "
            "sum of claims and allocates on planned need alone. PLAN section 3 source cell: "
            "qualitative (2,000 vs millions). Read by WO-006."
        ),
        plan_section="3; 2.4; 2.7.5",
    ),
    ParamSpec(
        name="audit_rate",
        arm="info",
        phase="P1",
        default=0.10,
        sweep=Range(0.01, 0.30),
        source="unsourced",
        provisional=True,
        notes=(
            "`a`, per-enterprise per-period audit probability, purpose `audit`. DUAL-CLASSIFIED "
            "in PLAN section 3: INFO by architecture, and it also enters the reward through the "
            "penalty of PLAN section 2.8 - so PLAN section 4.3 reports it separately (contrast "
            "C_AUDIT: audit_rate x4, audit_noise -> 0) and never folds it into C_OGAS. "
            "`audit_rate * penalty_scale` is the compound the G2 padding-elasticity criterion "
            "sweeps (PLAN section 4.5). Provisional: replaced at G1."
        ),
        plan_section="3; 2.7.4; 4.3; 4.5",
    ),
    ParamSpec(
        name="audit_noise",
        arm="info",
        phase="P1",
        default=0.0,
        sweep=Range(0.0, 0.15),
        source="unsourced",
        provisional=False,
        notes=(
            "`sigma_aud`, log-sd of the audit measurement error `S_hat = S * exp(nu)`, "
            "`nu ~ N(0, sigma_aud**2)`, drawn with purpose `auditnoise`. Read by WO-007."
        ),
        plan_section="3; 2.8",
    ),
    ParamSpec(
        name="audit_mode",
        arm="info",
        phase="P1",
        default="random",
        sweep=Choices(("random", "targeted")),
        source="qualitative",
        provisional=False,
        notes=(
            "Audit selection rule. Phase 1 is `random` (WO-006). `targeted` raises the probability "
            "with the planner's noisy knowledge of downstream complaints and therefore requires "
            "`shortfall_visibility > 0`; it is a Phase-2 information mechanism (WO-023)."
        ),
        plan_section="3; 2.7.4",
    ),
    ParamSpec(
        name="channel_noise",
        arm="info",
        phase="P1",
        default=0.0,
        sweep=Range(0.0, 0.10),
        source="unsourced",
        provisional=False,
        notes=(
            "`sigma_ch`, log-sd of the reporting-channel distortion "
            "`claimed_i <- claimed_i * exp(xi)`, `xi ~ N(0, sigma_ch**2)`, purpose `channel`. One "
            "of the three information filters of PLAN section 2.7.5; the identity at the Phase-1 "
            "value, and the branch is still implemented and tested (WO-006)."
        ),
        plan_section="3; 2.7.5",
    ),
    ParamSpec(
        name="ministry_passthrough",
        arm="info",
        phase="P1",
        default=1.0,
        sweep=Range(0.5, 1.0),
        source="qualitative",
        provisional=False,
        notes=(
            "`pi` in the ministry forwarding rule; 1.0 is a fully transparent ministry, so the "
            "layer is inert in Phase 1. PLAN section 3 source cell: qualitative (OGAS "
            "resistance). Read by the rule-based ministry (WO-025) and the LLM ministry study "
            "(WO-026)."
        ),
        plan_section="3; 2.14; 7.4",
    ),
    ParamSpec(
        name="horizontal_visibility",
        arm="info",
        phase="P1",
        default=0.0,
        sweep=Range(0.0, 1.0),
        source="qualitative",
        provisional=False,
        notes=(
            "Fraction of the other `N - 1` enterprises visible as trade counterparties, and the "
            "gate on the horizontal block of the observation. Locked Phase-2 value 1.0 for the "
            "blat phenomenon, alongside `trade_tau = 0.05` (PLAN section 4.2; WO-024)."
        ),
        plan_section="3; 2.4; 2.13; 4.2",
    ),
    ParamSpec(
        name="quality_measurability",
        arm="info",
        phase="P1",
        default=0.0,
        sweep=Range(0.0, 1.0),
        source="qualitative",
        provisional=False,
        notes=(
            "`mu` in the measured quality factor `q_hat = 1 + mu * (qbar - 1)` used by the "
            "`quality_weighted` objective metric. Phenomenon 3 compares `objective_metric = val` "
            "against `quality_weighted` at `mu = 1` (PLAN section 4.1; WO-021)."
        ),
        plan_section="3; 2.9.2; 4.1",
    ),
    ParamSpec(
        name="shortfall_visibility",
        arm="info",
        phase="P1",
        default=0.0,
        sweep=Range(0.0, 1.0),
        source="qualitative",
        provisional=False,
        notes=(
            "How much of buyers' complaints the planner sees; it gates the `targeted` audit mode "
            "and fills `PlannerView.downstream_shortfall`. C_OGAS sets it to 1 (PLAN section "
            "4.3). Phase-2 mechanism (WO-023)."
        ),
        plan_section="3; 2.7.4; 4.3",
    ),
    ParamSpec(
        name="self_obs_noise",
        arm="info",
        phase="P1",
        default=0.0,
        sweep=Range(0.0, 0.05),
        source="mechanism-toggle",
        provisional=False,
        notes=(
            "Log-sd of the multiplicative noise `exp(N(0, s**2))` on the agent's own cumulative "
            "output and stock observation fields, drawn with purpose `selfobs`; those fields are "
            "exact at the Phase-1 value. PLAN section 3 source cell: mechanism toggle (shape "
            "study). Read by WO-008."
        ),
        plan_section="3; 2.4",
    ),
    # ---------------- INC: incentive structure ----------------
    ParamSpec(
        name="objective_metric",
        arm="inc",
        phase="P1",
        default="val",
        sweep=Choices(("val", "net_output", "quality_weighted")),
        source="historical",
        provisional=False,
        notes=(
            "Fulfilment measure the bonus and the ratchet key on. PLAN section 3 source cell: "
            "historical (val -> NNO reforms). `welfare` is deliberately not an option (finding "
            "F6): the planner cannot key on a quantity it does not observe (CONTRACT rule 6). "
            "C_INC moves it to `net_output` (PLAN section 4.3)."
        ),
        plan_section="3; 2.9.2; 4.3",
    ),
    ParamSpec(
        name="ratchet_lambda",
        arm="inc",
        phase="P1",
        default=0.5,
        sweep=Range(0.0, 1.0),
        source="unsourced",
        provisional=True,
        notes=(
            "`lambda`, ratchet coefficient in the target rule "
            "`T <- max(T_min, (1 + g) * T * (1 + lambda * step))`. PLAN section 3 source cell: "
            "Weitzman-type models motivate the form, the empirical value is unsourced. C_INC "
            "moves it to 0.1 (PLAN section 4.3). Provisional: replaced at G1."
        ),
        plan_section="3; 2.7.1; 4.3",
    ),
    ParamSpec(
        name="growth_directive",
        arm="inc",
        phase="P1",
        default=0.02,
        sweep=Range(0.0, 0.07),
        source="unsourced",
        provisional=True,
        notes=(
            "`g`, the exogenous growth directive multiplying the target every period; the forcing "
            "term added for finding F1. It must be a treatment variable: at `g = 0` with reports "
            "at target the target rule has a fixed point (test T-B2). PLAN section 3 source cell: "
            "five-year-plan annual growth targets (lead to source, PLAN section 15). Phenomenon 7 "
            "is checked at the Phase-1 value (PLAN section 4.2). Provisional: replaced at G1."
            "WO-000 memo item 2: the FORM is attested - Harrison 2007 describes planning from the achieved level, plus an increment to allow for growth - but no directive (as opposed to realised) growth rate was retrieved, so the magnitude is unsourced."
        ),
        plan_section="3; 2.7.1; 4.2",
    ),
    ParamSpec(
        name="ratchet_cap_up",
        arm="inc",
        phase="P1",
        default=0.3,
        sweep=None,
        source="engineering",
        provisional=False,
        notes=(
            "`c_up`, cap on the positive per-period ratchet step in "
            "`step = clip(rho - 1, -c_dn, +c_up)`. Shares one PLAN section 3 row with "
            "`ratchet_cap_dn` (values 0.3 / 0.3, sweep cell `fixed`, source cell engineering "
            "(F4)); added to bound the transient of finding F4. `validate` rejects a negative cap."
        ),
        plan_section="3; 2.7.1",
    ),
    ParamSpec(
        name="ratchet_cap_dn",
        arm="inc",
        phase="P1",
        default=0.3,
        sweep=None,
        source="engineering",
        provisional=False,
        notes=(
            "`c_dn`, cap on the negative per-period ratchet step; shares its PLAN section 3 row "
            "with `ratchet_cap_up` (0.3 / 0.3, fixed, engineering (F4)). `validate` rejects a "
            "negative cap (WO-003)."
        ),
        plan_section="3; 2.7.1",
    ),
    ParamSpec(
        name="ratchet_deadband",
        arm="inc",
        phase="P1",
        default=0.0,
        sweep=Range(0.0, 0.03),
        source="mechanism-toggle",
        provisional=False,
        notes=(
            "`delta`: the ratchet step is zeroed when `|rho - 1| <= delta`. Inert at the Phase-1 "
            "value; PLAN section 3 source cell: mechanism toggle (shape study). Test T-U4 checks "
            "that it stays inert outside the band."
        ),
        plan_section="3; 2.7.1",
    ),
    ParamSpec(
        name="notch_height",
        arm="inc",
        phase="P1",
        default=1.0,
        sweep=Range(0.25, 2.0),
        source="design",
        provisional=False,
        notes=(
            "`beta`, height of the fulfilment notch in `B(rho)`. PLAN section 3 source cell: "
            "normalising unit - sweeping it moves the notch relative to `pen` and `kappa`, not "
            "the gradient magnitude, which `reward_scale(cfg)` fixes analytically (PLAN section "
            "2.9.1)."
        ),
        plan_section="3; 2.8; 2.9.1",
    ),
    ParamSpec(
        name="notch_width",
        arm="inc",
        phase="P1",
        default=0.0,
        sweep=Choices((0.0, 0.02, 0.05, 0.10, 0.25)),
        source="design",
        provisional=False,
        notes=(
            "`w`, logistic width of the notch: `Lambda_w(x) = 1[x >= 0]` at `w = 0` (a strict "
            "Heaviside, a true discontinuity), else `1 / (1 + exp(-x / w))`. PLAN section 3 "
            "source cell: counterfactual knob - the smooth arm is `w = 0.25` with "
            "`overfulfilment_cap = inf` (C_INC, PLAN section 4.3), and `w` is also the "
            "manipulation-strength knob of the estimator-bias study (PLAN section 7.2)."
        ),
        plan_section="3; 2.8; 4.3; 7.2",
    ),
    ParamSpec(
        name="overfulfilment_slope",
        arm="inc",
        phase="P1",
        default=0.5,
        sweep=Range(0.0, 2.0),
        source="unsourced",
        provisional=True,
        notes=(
            "`s`, linear bonus slope above target: `s * clip(rho - 1, 0, rho_cap - 1)`. PLAN "
            "section 3 source cell: historical - per-percentage-point bonus increments (lead to "
            "source, PLAN section 15). Provisional: replaced at G1."
            "WO-000 memo item 1: not obtained. Berliner 1957 has no retrievable full text and the one secondary source carrying bonus percentages now returns HTTP 404, so no per-percentage-point increment is sourced."
        ),
        plan_section="3; 2.8",
    ),
    ParamSpec(
        name="overfulfilment_cap",
        arm="inc",
        phase="P1",
        default=1.2,
        sweep=Choices((1.1, 1.2, float("inf"))),
        source="unsourced",
        provisional=False,
        notes=(
            "`rho_cap`, the ratio at which the overfulfilment bonus stops accruing. `inf` means "
            "no cap and hence no kink, and is half of the smooth counterfactual "
            "(`notch_width = 0.25`, `overfulfilment_cap = inf`). `validate` rejects `rho_cap < 1` "
            "(WO-003). PLAN section 3 source cell: historical - capped overfulfilment bonuses "
            "(lead to source)."
            "WO-000 memo item 1: not obtained. No cap on bonus accrual was sourced; the grid {1.1, 1.2, inf} is a design sweep, not a historical finding."
        ),
        plan_section="3; 2.8; 4.3",
    ),
    ParamSpec(
        name="penalty_form",
        arm="inc",
        phase="P1",
        default="proportional",
        sweep=Choices(("proportional", "fixed")),
        source="design",
        provisional=False,
        notes=(
            "Shape of the audit penalty: `Pen = pen * f` (`proportional`) or `pen * 1[f > 0]` "
            "(`fixed`). Shares one PLAN section 3 row with `penalty_arg` (Phase-1 values "
            "proportional / positive_part, sweep cell `both`, source cell design)."
        ),
        plan_section="3; 2.8",
    ),
    ParamSpec(
        name="penalty_arg",
        arm="inc",
        phase="P1",
        default="positive_part",
        sweep=Choices(("positive_part", "absolute")),
        source="design",
        provisional=False,
        notes=(
            "Argument of the audit penalty: `f = max(0, R - S_hat) / T` (`positive_part`, so an "
            "under-report incurs no penalty - test T-U8) or `f = |R - S_hat| / T` (`absolute`). "
            "Phenomenon 7 must vanish at `g = 0` with `absolute` (PLAN section 4.1). Shares its "
            "PLAN section 3 row with `penalty_form`."
        ),
        plan_section="3; 2.8; 4.1",
    ),
    ParamSpec(
        name="penalty_scale",
        arm="inc",
        phase="P1",
        default=60.0,
        sweep=Range(5.0, 200.0),
        source="unsourced",
        provisional=True,
        notes=(
            "`pen`, penalty scale in ratio units (finding F9). PLAN section 3 source cell: "
            "unsourced; regime map. `audit_rate * penalty_scale` is the compound quantity the G2 "
            "padding-elasticity criterion sweeps at three levels recorded at G1 (PLAN sections "
            "4.5, 13). Provisional: replaced at G1."
        ),
        plan_section="3; 2.8; 2.9.1; 4.5",
    ),
    ParamSpec(
        name="effort_cost",
        arm="inc",
        phase="P1",
        default=0.15,
        sweep=Range(0.05, 0.5),
        source="unsourced",
        provisional=True,
        notes=(
            "`kappa` in the step cost "
            "`c_ik = kappa * e_ik**2 + F * 1[e_ik > 0] + kappa_q * q_ik * e_ik`. A real cost paid "
            "when it is incurred, never shaping (CONTRACT rule 4). PLAN section 3 source cell: "
            "unsourced; regime map. Provisional: replaced at G1."
        ),
        plan_section="3; 2.6; 2.9.1",
    ),
    ParamSpec(
        name="soft_budget",
        arm="inc",
        phase="P1",
        default=0.0,
        sweep=Range(0.0, 1.0),
        source="qualitative",
        provisional=False,
        notes=(
            "Soft-budget intensity. PLAN section 3 source cell: Kornai (P2 definition: bailout "
            "probability on `fill < 1`). Inert at the Phase-1 value; implemented by WO-023."
        ),
        plan_section="3",
    ),
    ParamSpec(
        name="steps_per_period",
        arm="inc",
        phase="P1",
        default=4,
        sweep=Choices((4, 8)),
        source="design",
        provisional=False,
        notes=(
            "`M`, PRODUCE steps per plan period; agents act `M + 1` times per period (M PRODUCE "
            "steps then exactly one REPORT step). It also sizes `arrival_probs` when "
            "`delivery_timing != uniform` and the within-period effort Gini of phenomenon 2."
        ),
        plan_section="3; 2.1; 2.5",
    ),
    ParamSpec(
        name="tenure",
        arm="inc",
        phase="P1",
        default=0.9,
        sweep=Range(0.7, 0.98),
        source="unsourced",
        provisional=False,
        notes=(
            "`psi`, per-period continuation probability under `horizon_mode = geometric`, drawn "
            "with purpose `terminate`. An economic parameter (managerial rotation; lead to "
            "source), deliberately distinct from the technical PPO discount `gamma`; it enters "
            "the DP Bellman operator as `psi * gamma` (PLAN section 5)."
            "WO-000 memo item 5: not obtained. No distribution of Soviet enterprise director tenure was retrieved. Note the memo's conversion caveat: a per-year survival hazard is not a per-plan-period hazard unless the plan period is one year."
        ),
        plan_section="3; 2.12; 5",
    ),
    ParamSpec(
        name="alloc_eta_request",
        arm="inc",
        phase="P1",
        default=0.0,
        sweep=Range(0.0, 1.0),
        source="design",
        provisional=False,
        notes=(
            "`eta_q`, exponent on requests in the allocation weight "
            "`w_bj = (q_bj + 1e-6)**eta_q * (need_bj + 1e-6)**eta_n`. At 0 the request term is "
            "exactly 1, so requests are logged but inert (finding F7). The PLAN section 3 P1 cell "
            "reads `0 (P1) -> 0.7 (P2, locked)`: 0.7 is the locked hoarding mechanism of PLAN "
            "section 4.2. That mechanism is a rule about weights, never an instruction to inflate "
            "a request (CONTRACT rule 7)."
        ),
        plan_section="3; 2.7.2; 4.2",
    ),
    ParamSpec(
        name="alloc_eta_need",
        arm="inc",
        phase="P1",
        default=1.0,
        sweep=None,
        source="design",
        provisional=False,
        notes=(
            "`eta_n`, exponent on planned need in the allocation weight; the PLAN section 3 sweep "
            "cell reads `fixed`."
        ),
        plan_section="3; 2.7.2",
    ),
    ParamSpec(
        name="bonus_heterogeneity",
        arm="inc",
        phase="P1",
        default=0.0,
        sweep=Range(0.0, 0.5),
        source="unsourced",
        provisional=False,
        notes=(
            "Dispersion of sector-specific bonus schedules around `notch_height`. PLAN section 3 "
            "source cell: historical (sector-specific schedules). Inert at the Phase-1 value."
            "WO-000 memo item 1: not obtained. Sector-specific bonus schedules are asserted by PLAN section 3 but no source for them was retrieved."
        ),
        plan_section="3; 2.8",
    ),
    # ---------------- SUPPLY: production and supply structure ----------------
    ParamSpec(
        name="input_complementarity",
        arm="supply",
        phase="P1",
        default=8.0,
        sweep=Choices((2.0, 8.0, float("inf"))),
        source="design",
        provisional=False,
        notes=(
            "`theta` in the CES coverage aggregator "
            "`H = (sum_j omega_j * min(1, X_j / need_j)**(-theta))**(-1/theta)`; `theta = inf` "
            "reduces to Leontief `min`. PLAN section 3 source cell: design; near-Leontief. "
            "`validate` rejects `theta < 1` (WO-003); locked at 8 for the hoarding phenomenon "
            "(PLAN section 4.2)."
        ),
        plan_section="3; 2.6; 4.2",
    ),
    ParamSpec(
        name="yield_sigma",
        arm="supply",
        phase="P1",
        default=(0.05, 0.08, 0.10, 0.12, 0.15),
        sweep=Range(0.5, 2.0),
        source="unsourced",
        provisional=False,
        notes=(
            "`sigma_j`, log-sd of the per-step multiplicative yield shock, one entry per sector, "
            "heteroskedastic by construction; the shock is `LogNormal(-sigma**2 / 2, sigma)` so "
            "`E[eps] = 1`. The PLAN section 3 sweep cell reads `x[0.5, 2]`: the range is a "
            "MULTIPLIER on the whole tuple, not a range on a scalar. Locked at x1 for phenomenon "
            "2 (PLAN section 4.2)."
        ),
        plan_section="3; 2.6; 4.2",
    ),
    ParamSpec(
        name="setup_cost",
        arm="supply",
        phase="P1",
        default=0.0,
        sweep=Range(0.0, 0.2),
        source="mechanism-toggle",
        provisional=False,
        notes=(
            "`F`, fixed cost charged once per step when `e > 0`. Shares one PLAN section 3 row "
            "with `irs_alpha` (Phase-1 values 0 / 0, ranges [0, 0.2] / [0, 0.3], source cell `P2 "
            "toggles`). Off in Phase 1."
        ),
        plan_section="3; 2.6",
    ),
    ParamSpec(
        name="irs_alpha",
        arm="supply",
        phase="P1",
        default=0.0,
        sweep=Range(0.0, 0.3),
        source="mechanism-toggle",
        provisional=False,
        notes=(
            "`alpha_irs` in the increasing-returns toggle "
            "`A_j(Kap) = A_j * (Kap / Kap_0)**alpha_irs`. Shares its PLAN section 3 row with "
            "`setup_cost`; off in Phase 1."
        ),
        plan_section="3; 2.6",
    ),
    ParamSpec(
        name="capital_dep",
        arm="supply",
        phase="P1",
        default=0.0,
        sweep=Range(0.02, 0.1),
        source="mechanism-toggle",
        provisional=False,
        notes=(
            "Capital depreciation in `Kap_{t+1} = (1 - dep) * Kap_t + matured investment`. Shares "
            "one PLAN section 3 row with `invest_lag` (Phase-1 values 0 / -, source cell `P2`). "
            "The range starts at 0.02 while the Phase-1 value is 0: 0 is off, and the range "
            "applies once the mechanism is on."
        ),
        plan_section="3; 2.6",
    ),
    ParamSpec(
        name="invest_lag",
        arm="supply",
        phase="P2",
        default=1,
        sweep=Choices((1, 2, 3)),
        source="mechanism-toggle",
        provisional=False,
        notes=(
            "Periods between diverting output to capital and its maturing; it sizes the second "
            "dimension of `State.pending_invest`. PLAN section 3 gives NO Phase-1 value (cell "
            "`-`) because investment is inert while `v == 0`; the default recorded here is the "
            "minimum viable buffer depth declared in `spec/spec.py`, not a PLAN value. `validate` "
            "requires `invest_lag >= 1`."
        ),
        plan_section="3; 2.2; 2.6",
    ),
    ParamSpec(
        name="delivery_timing",
        arm="supply",
        phase="P1",
        default="uniform",
        sweep=Choices(("uniform", "stochastic", "backloaded")),
        source="mechanism-toggle",
        provisional=False,
        notes=(
            "Within-period arrival of delivered inputs; `uniform` means everything is available "
            "at step 0. PLAN section 3 source cell: storming mechanism; locked section 4.2 - the "
            "locked Phase-2 setting is `stochastic` with "
            "`arrival_probs = (0.25, 0.25, 0.25, 0.25)`, uniform-random arrival, so there is no "
            "mechanical backloading and any effort-Gini excess is behavioural (WO-022)."
        ),
        plan_section="3; 2.6; 4.2",
    ),
    ParamSpec(
        name="final_demand_share",
        arm="supply",
        phase="P1",
        default=(0.5, 0.5, 0.5, 0.5, 0.5),
        sweep=Range(0.3, 0.7),
        source="design",
        provisional=False,
        notes=(
            "`phi_j`, the fraction of shipped good `j` routed to the consumer sink rather than to "
            "intermediate buyers. PLAN section 3 gives the scalar 0.5; `spec/spec.py` stores one "
            "entry per sector, so the Phase-1 default is that scalar repeated `n_sectors` times "
            "and the range applies per sector."
        ),
        plan_section="3; 2.7.3; 2.10",
    ),
    ParamSpec(
        name="holding_loss",
        arm="supply",
        phase="P1",
        default=0.02,
        sweep=Range(0.0, 0.05),
        source="design",
        provisional=False,
        notes=(
            "`h`, per-period proportional loss on carried own-good stock, applied before this "
            "period's output is added: `S <- (1 - h) * S + y`. Shares one PLAN section 3 row (and "
            "the range) with `input_holding_loss` (Phase-1 values 0.02 / 0). Held at the Phase-1 "
            "value for phenomenon 7 (PLAN section 4.2)."
        ),
        plan_section="3; 2.8; 2.11; 4.2",
    ),
    ParamSpec(
        name="input_holding_loss",
        arm="supply",
        phase="P1",
        default=0.0,
        sweep=Range(0.0, 0.05),
        source="design",
        provisional=False,
        notes=(
            "`h_X`, per-period loss on held input stocks. 0 in Phase 1 by design, so hoarding "
            "carries no direct carrying cost; the locked Phase-2 value is 0.01 for the hoarding "
            "phenomenon (PLAN section 4.2; WO-023). Shares its PLAN section 3 row with "
            "`holding_loss`."
        ),
        plan_section="3; 2.11; 4.2",
    ),
    ParamSpec(
        name="price_markup",
        arm="supply",
        phase="P1",
        default=0.1,
        sweep=None,
        source="qualitative",
        provisional=False,
        notes=(
            "`m` in the cost-plus price fixed point "
            "`p_j = (1 + m) * (kappa_labour + sum_k a_jk * p_k)`, solved once at `t = 0`. Shares "
            "one PLAN section 3 row with `price_lag` (values 0.1 / inf, sweep cell `-`, source "
            "cell cost-plus, historical qualitative). The price vector is a reporting choice, not "
            "a result: every headline table is repeated under the sensitivity check of PLAN "
            "section 7.5."
        ),
        plan_section="3; 2.10; 7.5",
    ),
    ParamSpec(
        name="price_lag",
        arm="supply",
        phase="P1",
        default=float("inf"),
        sweep=None,
        source="qualitative",
        provisional=False,
        notes=(
            "Periods between price recomputations; `inf` means plan prices are fixed after "
            '`t = 0`. Serialised as the string "inf" by `EnvConfig.hash` and decoded back by '
            "`load_config` (WO-003). Shares its PLAN section 3 row with `price_markup`."
        ),
        plan_section="3; 2.10",
    ),
    ParamSpec(
        name="tech_drift_sigma",
        arm="supply",
        phase="P1",
        default=0.0,
        sweep=Range(0.0, 0.05),
        source="unsourced",
        provisional=False,
        notes=(
            "`sigma_drift` in the per-period I-O drift `a <- a * exp(zeta)`, "
            "`zeta ~ N(0, sigma_drift**2)`, purpose `drift`, while `State.planner_io` stays fixed "
            "so the planner's copy goes stale. PLAN section 3 source cell: acknowledged weak "
            "proxy - it stands in for technological change and is reported as such."
        ),
        plan_section="3; 2.2; 2.6",
    ),
    ParamSpec(
        name="trade_tau",
        arm="supply",
        phase="P2",
        default=0.05,
        sweep=None,
        source="design",
        provisional=False,
        notes=(
            "`tau`, per-unit transaction cost on bilateral trade; `tau > 0` is what makes wash "
            "trades unprofitable. PLAN section 3 gives NO Phase-1 value (cell `-`) because there "
            "is no trade in Phase 1; 0.05 is the locked Phase-2 value for the blat phenomenon "
            "(PLAN section 4.2), alongside `horizontal_visibility = 1.0` (WO-024)."
        ),
        plan_section="3; 2.13; 4.2",
    ),
    # ---------------- TECH: technical and engineering constants ----------------
    ParamSpec(
        name="n_enterprises",
        arm="tech",
        phase="P1",
        default=20,
        sweep=None,
        source="engineering",
        provisional=False,
        notes=(
            "`N`. From the PLAN section 3 row `N, J, rho_max, r_max, T_min, S_max = 20, 5, 10, 3, "
            "0.05*T_0, 3*cap` (sweep cell `-`, engineering). Phase 1: 20 enterprises, four per "
            "sector, with `sector_of` contiguous."
        ),
        plan_section="3; 2.1",
    ),
    ParamSpec(
        name="n_sectors",
        arm="tech",
        phase="P1",
        default=5,
        sweep=None,
        source="engineering",
        provisional=False,
        notes=(
            "`J`, one good per sector; from the same PLAN section 3 row as `n_enterprises`. It "
            "sizes `io_matrix`, `final_demand_share`, `productivity`, `yield_sigma`, `ces_alpha` "
            "and the three per-good observation blocks of PLAN section 2.4."
        ),
        plan_section="3; 2.1; 2.4",
    ),
    ParamSpec(
        name="report_max_ratio",
        arm="tech",
        phase="P1",
        default=10.0,
        sweep=None,
        source="engineering",
        provisional=False,
        notes=(
            "`rho_max`, the upper bound on the report action; from the same PLAN section 3 row as "
            "`n_enterprises`. CONTRACT rule 8: bounds are results - the fraction of reports at "
            "the bound is logged, above 1% the run manifest is flagged BOUND_BINDING (test T-B8), "
            "and the bound is never silently widened or narrowed to fix a result."
        ),
        plan_section="3; 2.3; 2.8",
    ),
    ParamSpec(
        name="request_max_multiple",
        arm="tech",
        phase="P1",
        default=3.0,
        sweep=None,
        source="engineering",
        provisional=False,
        notes=(
            "`r_max`: the input request is bounded by `r_max * need_ij`, and the action is "
            "expressed as a multiple of need because `need_ij` is state-dependent. From the same "
            "PLAN section 3 row as `n_enterprises`."
        ),
        plan_section="3; 2.3",
    ),
    ParamSpec(
        name="target_floor_frac",
        arm="tech",
        phase="P1",
        default=0.05,
        sweep=None,
        source="engineering",
        provisional=False,
        notes=(
            "`T_min` as a fraction of `T_0`: PLAN section 3 writes the entry `0.05 * T_0`, so the "
            "stored parameter is the fraction and the target rule floors at "
            "`max(target_floor_frac * T_0, ...)`. From the same PLAN section 3 row as "
            "`n_enterprises`."
        ),
        plan_section="3; 2.7.1",
    ),
    ParamSpec(
        name="inventory_cap_mult",
        arm="tech",
        phase="P1",
        default=3.0,
        sweep=None,
        source="engineering",
        provisional=False,
        notes=(
            "`S_max = inventory_cap_mult * cap_i`; PLAN section 3 writes the entry `3 * cap`. "
            "Output above the cap is lost and the overflow is logged, so it stays visible in the "
            "conservation identity (test T-U1). From the same PLAN section 3 row as "
            "`n_enterprises`."
        ),
        plan_section="3; 2.11",
    ),
    ParamSpec(
        name="initial_target_frac",
        arm="tech",
        phase="P1",
        default=0.6,
        sweep=None,
        source="engineering",
        provisional=False,
        notes=(
            "`T_0 = initial_target_frac * A_{s(i)} * cap_i`; PLAN section 3 writes the row as "
            "`0.6 * A * cap (feasible at e about 0.6)`, i.e. the initial target is reachable at "
            "about 0.6 effort. With `productivity = 1` and `cap = 1` it reads `T_0 = 0.6`."
        ),
        plan_section="3; 2.7.1",
    ),
    ParamSpec(
        name="horizon_mode",
        arm="tech",
        phase="P1",
        default="geometric",
        sweep=None,
        source="engineering",
        provisional=False,
        notes=(
            "Termination rule; from the PLAN section 3 row "
            "`horizon_mode, P_min, P_max = geometric, 4, 20` (sweep cell `-`, engineering). "
            "`geometric` means the agent never observes periods remaining and there is no "
            "end-game (finding F4). The `HorizonMode` alias also admits `fixed`, which exists to "
            "study known rotation separately; PLAN section 3 lists no sweep over it."
        ),
        plan_section="3; 2.12",
    ),
    ParamSpec(
        name="min_periods",
        arm="tech",
        phase="P1",
        default=4,
        sweep=None,
        source="engineering",
        provisional=False,
        notes=(
            "`P_min`: periods that always run before geometric termination can fire. From the "
            "same PLAN section 3 row as `horizon_mode`. `validate` requires "
            "`min_periods <= max_periods`."
        ),
        plan_section="3; 2.12",
    ),
    ParamSpec(
        name="max_periods",
        arm="tech",
        phase="P1",
        default=20,
        sweep=None,
        source="engineering",
        provisional=False,
        notes=(
            "`P_max`: hard cap on periods per episode. From the same PLAN section 3 row as "
            "`horizon_mode`. With `tenure = 0.9` the expected episode is about 10 periods, i.e. "
            "about 50 agent-steps. The measurement window is periods `t >= 2` (PLAN section 4.4)."
        ),
        plan_section="3; 2.12; 4.4",
    ),
    ParamSpec(
        name="param_sharing",
        arm="tech",
        phase="P1",
        default="shared",
        sweep=Choices(("shared", "per_sector", "independent")),
        source="engineering",
        provisional=False,
        notes=(
            "Policy-parameter sharing across enterprises. PLAN section 3 records the Phase-1 "
            "value as `shared (sector one-hot)`: one policy, with sector identity carried by the "
            "one-hot block of the observation (PLAN sections 2.4, 6.1). Read by the PPO adapter "
            "(WO-017), not by the environment."
        ),
        plan_section="3; 6.1",
    ),
    ParamSpec(
        name="ppo_gamma",
        arm="tech",
        phase="P1",
        default=0.99,
        sweep=None,
        source="engineering",
        provisional=False,
        notes=(
            "PPO discount `gamma`, from the PLAN section 3 row "
            "`PPO: gamma, lambda_GAE, lr, clip, entropy = 0.99, 0.97, 3e-4, 0.2, 0.01 -> 0.001` "
            "(engineering; fixed across arms). NOT an `EnvConfig` field: the PPO "
            "hyper-parameters live with the adapter (WO-017) because the environment never reads "
            "them, which is why the registry name is prefixed `ppo_`. Distinct from `tenure`, the "
            "economic continuation probability."
        ),
        plan_section="3; 6.1; 12.3 (WO-017)",
    ),
    ParamSpec(
        name="ppo_lambda_gae",
        arm="tech",
        phase="P1",
        default=0.97,
        sweep=None,
        source="engineering",
        provisional=False,
        notes=(
            "GAE parameter `lambda_GAE`, from the PLAN section 3 PPO row; fixed across arms. Not "
            "an `EnvConfig` field (WO-017)."
        ),
        plan_section="3; 12.3 (WO-017)",
    ),
    ParamSpec(
        name="ppo_lr",
        arm="tech",
        phase="P1",
        default=3e-4,
        sweep=None,
        source="engineering",
        provisional=False,
        notes=(
            "Learning rate, from the PLAN section 3 PPO row; fixed across arms. Not an "
            "`EnvConfig` field (WO-017)."
        ),
        plan_section="3; 12.3 (WO-017)",
    ),
    ParamSpec(
        name="ppo_clip",
        arm="tech",
        phase="P1",
        default=0.2,
        sweep=None,
        source="engineering",
        provisional=False,
        notes=(
            "PPO clipping parameter, from the PLAN section 3 PPO row; fixed across arms. Not an "
            "`EnvConfig` field (WO-017)."
        ),
        plan_section="3; 12.3 (WO-017)",
    ),
    ParamSpec(
        name="ppo_entropy_coef",
        arm="tech",
        phase="P1",
        default=(0.01, 0.001),
        sweep=None,
        source="engineering",
        provisional=False,
        notes=(
            "Entropy coefficient, from the PLAN section 3 PPO row, which writes it as the anneal "
            "`0.01 -> 0.001`; the tuple is (start, end) and WO-018 anneals between them over "
            "training. Fixed across arms; not an `EnvConfig` field (WO-017). CONTRACT rule 4 "
            "still governs the reward itself: no shaping, no reward normalisation, per-batch "
            "advantage normalisation only."
        ),
        plan_section="3; 12.3 (WO-017, WO-018)",
    ),
)
"""The PLAN section 3 table as data: one `ParamSpec` per parameter, in table order (INFO, INC,
SUPPLY, TECH), with the compound PLAN rows split into one entry per parameter.

Binds: `tests/unit/test_config.py` (WO-003) - `p1_default_config()` agrees with this table field by
field for every entry that names an `EnvConfig` field; `tests/unit/test_spec_imports.py` (WO-001) -
the module imports and exposes this surface. Consumers: WO-015 (the regime-map LHS over these
ranges), WO-020 (Phase-1 gate arms), WO-032 and WO-033 (contrasts and the optional Sobol design,
which must state these ranges as an assumption in the same table as any index it reports)."""


_BY_NAME: dict[str, ParamSpec] = {spec.name: spec for spec in REGISTRY}
"""Private lookup index over `REGISTRY`, keyed by `ParamSpec.name`. Names are unique by
construction; the registry is literal data, so this dictionary is built once at import."""


def by_name(name: str) -> ParamSpec:
    """Return the registry row for one parameter.

    Takes: `name`, an `EnvConfig` field name as recorded in `ParamSpec.name` (for example
    `"audit_rate"`, `"yield_sigma"`, `"ppo_gamma"`). Returns: the single `ParamSpec` with that
    name; raises `KeyError` if the registry has no such row - a missing row means the parameter is
    not tabulated in PLAN section 3 (the module docstring lists the `EnvConfig` fields that section
    deliberately omits), and the response is an AMBIGUITY REPORT, not a row invented here
    (CONTRACT rule 3).

    Realises: PLAN section 3 (registry lookup). Binds: `tests/unit/test_config.py` (WO-003), which
    uses it to compare `p1_default_config()` against the registry defaults field by field, and
    `tests/unit/test_spec_imports.py` (WO-001).
    """
    return _BY_NAME[name]


def by_arm(arm: Arm) -> tuple[ParamSpec, ...]:
    """Return every registry row belonging to one arm, in PLAN section 3 table order.

    Takes: `arm`, one of `"info"`, `"inc"`, `"supply"`, `"tech"`. Returns: a tuple of `ParamSpec`
    (empty for no match, which cannot happen for the four legal arms).

    The arm split is the design decision that makes the contrasts of PLAN section 4.3 well defined
    - C_OGAS moves INFO parameters, C_INC moves INC parameters - so this function is how a contrast
    harness enumerates what it is allowed to move. `audit_rate` is returned under `"info"` and is
    dual: it also enters the reward, and PLAN section 4.3 requires it to be reported separately
    (contrast C_AUDIT) and never folded into C_OGAS. Changing any arm assignment requires a
    CHANGELOG entry (CONTRACT rule 11).

    Realises: PLAN sections 3 and 4.3. Binds: `tests/unit/test_config.py` (WO-003) and the contrast
    harness of WO-032.
    """
    return tuple(spec for spec in REGISTRY if spec.arm == arm)


def provisional() -> tuple[ParamSpec, ...]:
    """Return every row whose Phase-1 value is provisional, in PLAN section 3 table order.

    Takes: nothing. Returns: the tuple of `ParamSpec` with `provisional is True` - the daggered
    rows of PLAN section 3: `audit_rate`, `ratchet_lambda`, `growth_directive`,
    `overfulfilment_slope`, `penalty_scale`, `effort_cost`.

    These six defaults are placeholders. Gate G1 replaces them with values a human selects from the
    interior of the bunching region of the DP regime map (WO-014, WO-015) and records in
    `runs/G1_decision.md` before any training run, together with the three
    `audit_rate * penalty_scale` levels for G2 and the `b_hat_DP` thresholds (PLAN sections 5, 13).
    Until then no report may present a quantity computed at these values as evidence about
    anything; `docs/params_sources.md` (WO-000, not yet executed) records their sourcing status.

    Realises: PLAN sections 3 and 13 (gate G1). Binds: `tests/unit/test_config.py` (WO-003), which
    checks the provisional set against the registry, and the G1 artefact check of WO-015.
    """
    return tuple(spec for spec in REGISTRY if spec.provisional)
