"""Heuristic agents: the fixed-rule policies of the PLAN section 6.1 table.

Realises: PLAN section 6.1 (agent table), read against PLAN sections 2.3 (actions), 2.4
(observation), 2.6 (production), 2.8 (report and audit) and 5 (the DP whose policy `DPGreedy`
replays). Owning work orders: **WO-010** (`Random`, `TruthfulMyopic`, `Padder`, `DPGreedy`) and
**WO-030** (`Berliner`, `Weitzman`, `Kornai`, Phase 2).

Every class here implements the `Agent` protocol of `gosplan/agents/base.py` structurally: `act`
takes `(obs, phase, rng)` and nothing else, and `reset` takes nothing. None of them reads `State`,
`StepInfo` or `PlannerView` - "any agent reading `StepInfo`" is on the WO-010 forbidden list and is
CONTRACT rule 6. They hold an `EnvConfig` because the *configuration* is public (sector
productivity, the initial target, the action bounds); they never hold or receive the *state*.

Reading the observation. `obs` is `(N, d)`, `d = 12 + 3J` in Phase 1, laid out exactly as
`obs_spec(cfg)` (PLAN section 2.4):

    0                phase                     0 produce / 1 report
    1                k_over_M                  step within the period
    2                log_target_ratio          log(T_i / T_0)
    3                growth_directive          g
    4                cum_output_over_target    own true production so far this period / T_i
    5                stock_over_target         S_i / T_i
    6                capital_ratio             Kap_i / Kap_0
    7                last_report_ratio
    8                last_audited
    9                last_penalty_scaled       last_penalty * reward_scale(cfg)
    10               last_fill
    11               inputs_delivered_total    delivered this period / need, totals
    12      : 12+J   input_cov_{j}             X_ij / need_ij
    12+J   : 12+2J   sector_onehot_{j}
    12+2J  : 12+3J   deliv_cov_{j}             deliv_ij / need_ij this period

Two reconstructions every rule below uses, both exact and both from the observation and the
configuration alone (no true quantity crosses into an agent):

    T_i   = T_0 * exp(obs[i, 2]),      T_0 = cfg.tech.initial_target_frac * A_{s(i)} * cap_i
    S_i   = obs[i, 5] * T_i            (`stock_over_target` is exact when self_obs_noise = 0)

CONTRACT rule 7 in this module. Rule 7 forbids a *transition rule or reward term* from implementing
bunching, padding, storming, hoarding, shaving or trade directly. A heuristic agent is neither: it
is a fixed probe used to exercise a channel (`Padder`) or to supply a truthful reference line
(`TruthfulMyopic`). The rule the module must respect instead is the reporting one: no result about
an *emergent* phenomenon may ever be read off a heuristic agent, and the held-out phenomena of PLAN
section 4.1 (rows 2, 5, 6, 7) are not computed from any agent here during Phase 1 (WO-012 forbidden
list).

Binds: T-B1 `tests/behavioural/test_no_hardcoded_pathology.py`, T-B2 `test_fixed_point.py`, T-B3
`test_shortage_propagation.py`; `Random` and `TruthfulMyopic` also generate the golden trajectories
of `ref/gen_golden.py` (T-B7).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from gosplan.agents.base import Array, Phase

if TYPE_CHECKING:  # runtime homes: WO-003 (config) and WO-009 (state), PLAN section 8
    import numpy as np

    from gosplan.agents.dp import DPSolution
    from gosplan.config import EnvConfig
    from gosplan.env.state import EnterpriseAction

REQUEST_MULTIPLE_NEED = 1.0
"""The `input_request` value that means "request exactly `need_ij`" (PLAN sections 2.3, 6.1). The
action is expressed as a multiple of the state-dependent `need_ij` and bounded by
`cfg.tech.request_max_multiple` (`action_spec`), so requesting need itself is the multiple 1.0.
Used by `TruthfulMyopic` ("requests = need", PLAN section 6.1) and asserted by test T-B1."""

PADDER_EFFORT = 0.3
"""The constant per-PRODUCE-step effort of `Padder` (PLAN section 6.1, verbatim). Not a swept
parameter and not a configuration field: `Padder` is a fixed sanity probe, so changing this number
changes what the Monte-Carlo sanity harness of WO-012 exercises."""

PADDER_REPORT_RATIO = 1.0
"""The constant report ratio of `Padder` (PLAN section 6.1: "rho = 1 always"), i.e. it claims
exactly its target every period whatever its stock. This is what makes it the fixed point probe of
test T-B2 and the shortage-channel probe of test T-B3."""


@dataclass
class Random:
    """Uniform random policy over the active action dimensions (PLAN section 6.1).

    Rule, verbatim from the PLAN section 6.1 table: *uniform over active dimensions*. For each name
    in `active_action_dims(cfg)` draw independently and uniformly from that dimension's box in
    `action_spec(cfg)` - `effort` on [0, 1], `report_ratio` on [0, `cfg.tech.report_max_ratio`],
    `input_request` on [0, `cfg.tech.request_max_multiple`] per good - using the `rng` argument
    (`seed_policy` stream, CONTRACT rule 9). Dimensions the configuration does not activate are
    filled with zeros; the environment ignores them (PLAN section 2.3).

    Purpose: it is the stress agent. It drives the golden trajectories of `ref/gen_golden.py`
    together with `TruthfulMyopic` (T-B7), and the Monte-Carlo sanity harness of WO-012 runs 2,000
    episodes of it to check conservation to 1e-9, absence of NaN/inf, bounded `T` and `S`, and
    `fill` in [0, 1] under adversarial-but-legal actions. It is never a baseline for any claim.

    Fields: `cfg` only - the policy is stateless, so `reset` is a no-op.
    """

    cfg: EnvConfig
    """The run's configuration. Read for `action_spec(cfg)` bounds, `active_action_dims(cfg)` and
    the array shapes `N = cfg.supply.n_enterprises`, `J = cfg.supply.n_sectors`. Public data; no
    state is held."""

    def act(self, obs: Array, phase: Phase, rng: np.random.Generator) -> EnterpriseAction:
        """Draw one uniform joint action.

        Takes: `obs` `(N, d)` - unread by this policy, present because the `Agent` protocol fixes
        the signature; `phase`; `rng`. Returns: an `EnterpriseAction` whose active dimensions are
        i.i.d. uniform on their `action_spec(cfg)` boxes and whose inactive dimensions are zero.

        All randomness comes from `rng` (CONTRACT rule 9); no call to `numpy.random` and no call to
        `gosplan.rng.draw`, which belongs to the environment's `seed_env` stream. Owning WO:
        **WO-010**.
        """
        raise NotImplementedError("PLAN section 6.1 - implemented in WO-010")

    def reset(self) -> None:
        """No-op: the policy is stateless.

        Takes: nothing. Returns: `None`. Owning WO: **WO-010**.
        """
        raise NotImplementedError("PLAN section 6.1 - implemented in WO-010")


@dataclass
class TruthfulMyopic:
    """Meets the target in expectation and reports its stock truthfully (PLAN section 6.1).

    Rule, verbatim from the PLAN section 6.1 table and the WO-010 card: *effort so that
    `E[y] = T`; `rho = S/T` (truthful of stock); requests `= need`; never trades*. Concretely, per
    enterprise `i`:

        PRODUCE step k:  e_ik = clip(T_i / (A_{s(i)} * cap_i), 0, 1)
        REPORT step:     rho_i = S_i / T_i          # truthful of STOCK, not of this period's y
                         input_request_ij = REQUEST_MULTIPLE_NEED  (= need_ij)
                         trade_offer_ij = 0

    Why that effort: intended output at a PRODUCE step is `y_hat_ik = (A_{s(i)} * cap_i / M) * e_ik`
    (PLAN section 2.6) and the yield shock has mean 1, so uniform effort `e = T / (A * cap)` over
    the `M` steps gives `E[sum_k y_ik] = T` exactly, at full input coverage. In observation terms
    the same number is `clip(cfg.tech.initial_target_frac * exp(obs[i, 2]), 0, 1)`, because
    `T_0 = initial_target_frac * A_{s(i)} * cap_i` cancels the productivity and the capacity.

    Why the report is of stock and not of output: the audit compares the claim to stock on hand
    (PLAN section 2.8), so "truthful" is defined against `S_i` after this period's holding loss and
    output have been applied. `S_i / T_i` is observation field 5 directly, so the report needs no
    reconstruction at all; it is clipped to `[0, cfg.tech.report_max_ratio]` like any report.

    Status: this is the **reference line**, not a baseline for an emergent claim. Every held-out
    phenomenon of PLAN section 4.1 is defined as an excess *over* this agent under common random
    numbers (rows 2, 5, 6), so the agent must contain no behavioural slack of its own - it never
    over-produces, never banks, never inflates a request and never trades.

    Binds: T-B1 in `tests/behavioural/test_no_hardcoded_pathology.py` - under
    (`notch_width = 0.25`, `overfulfilment_cap = inf`, `audit_rate = 1`, `penalty_scale = 200`,
    `growth_directive = 0`) its reports equal stock to 1e-9, its `rho` histogram has no bin holding
    more than three times the mean of its two neighbours, its within-period effort Gini equals the
    one implied by yield noise alone under `uniform` delivery, it never trades and its requests
    equal need. Also T-B3 (`fill = 1` for all downstream buyers) and the golden files of T-B7.

    Fields: `cfg` only - stateless, so `reset` is a no-op.
    """

    cfg: EnvConfig
    """The run's configuration. Read for `cfg.supply.productivity`, `cfg.tech.initial_target_frac`,
    `cfg.tech.report_max_ratio`, `cfg.supply.sector_of` and the array shapes."""

    def act(self, obs: Array, phase: Phase, rng: np.random.Generator) -> EnterpriseAction:
        """Return the truthful-myopic action for the current phase.

        Takes: `obs` `(N, d)` - fields 2 (`log_target_ratio`) and 5 (`stock_over_target`) are the
        only ones read; `phase`, selecting effort (PRODUCE) or report and request (REPORT); `rng`,
        unused, since the rule is deterministic. Returns: an `EnterpriseAction` filled per the
        formulas in the class docstring, with `quality`, `invest` and `trade_offer` at zero.

        Determinism matters: T-B1 compares reports with stock to 1e-9, and the common-random-number
        comparisons of PLAN section 4.1 rows 2 and 5 subtract this agent's trajectory from a
        learned one under the same `seed_env`. Owning WO: **WO-010**.
        """
        raise NotImplementedError("PLAN section 6.1 - implemented in WO-010")

    def reset(self) -> None:
        """No-op: the policy is stateless.

        Takes: nothing. Returns: `None`. Owning WO: **WO-010**.
        """
        raise NotImplementedError("PLAN section 6.1 - implemented in WO-010")


@dataclass
class Padder:
    """Claims exactly the target every period, whatever it produced - **sanity probe only**.

    Rule, verbatim from the PLAN section 6.1 table: *`rho = 1` always; effort 0.3. Exists to
    exercise the shortage channel in MC sanity; never a baseline.* Concretely:

        PRODUCE step k:  e_ik = PADDER_EFFORT              (0.3, every step, every enterprise)
        REPORT step:     rho_i = PADDER_REPORT_RATIO       (1.0, whatever S_i is)

    At `initial_target_frac = 0.6` the effort 0.3 produces roughly half the target, so the claim is
    fictitious by construction and the promise the planner allocates from is not backed by stock.
    That is the point: PLAN section 2.7.3's delivery step then ships less than was promised and
    `fill < 1` propagates to every downstream buyer, which is the channel test T-B3 exercises.

    **Never a baseline.** No table, plot or claim in any phase may use `Padder` as a comparison
    point for padding (PLAN section 4.1 row 4) or for any emergent phenomenon: its padding is
    assumed, not learned, so it measures the plumbing and nothing else. Its uses are exactly two -
    test T-B2 (fixed point) and test T-B3 (shortage propagation) - plus the Monte-Carlo sanity
    harness of WO-012, which asserts only conservation, boundedness and the *existence* of
    downstream shortage, never a direction or a magnitude.

    This is not a CONTRACT rule 7 violation: rule 7 binds transition rules and reward terms, and
    nothing in `gosplan/env/` knows this class exists. It is a fixed input used to check that a
    channel is wired, in the same spirit as a sentinel value in T-B4.

    Binds: T-B2 in `tests/behavioural/test_fixed_point.py` - at `growth_directive = 0` the targets
    stay constant under this agent (the ratchet map's fixed point at `rho = 1`), and at
    `growth_directive = g > 0` they grow at exactly `(1 + g)` per period. T-B3 in
    `tests/behavioural/test_shortage_propagation.py` - with `S = 0` it produces `fill < 1` for all
    downstream buyers.

    Fields: `cfg` only - stateless, so `reset` is a no-op. The two constants are module-level and
    deliberately not fields: they define the probe, and a probe with tunable knobs is a baseline.

    Requests: PLAN section 6.1 fixes only effort and the report for this agent. `input_request` is
    active-but-inert in Phase 1 (`alloc_eta_request = 0` makes the request term exactly 1 in the
    allocation weight of PLAN section 2.7.2), so no Phase-1 result can depend on it; setting it to
    `REQUEST_MULTIPLE_NEED` keeps T-B3 attributable to the report channel alone. Should any test or
    Phase-2 configuration make the choice load-bearing, that is an AMBIGUITY REPORT (CONTRACT rule
    3), not an implementer's decision.
    """

    cfg: EnvConfig
    """The run's configuration. Read for the array shapes and the report bound only; the rule
    itself is configuration-independent by design."""

    def act(self, obs: Array, phase: Phase, rng: np.random.Generator) -> EnterpriseAction:
        """Return the constant padding action for the current phase.

        Takes: `obs` `(N, d)` - unread, the rule is a constant; `phase`; `rng`, unused. Returns: an
        `EnterpriseAction` with `effort = PADDER_EFFORT` at a PRODUCE step and
        `report_ratio = PADDER_REPORT_RATIO` at the REPORT step, requests at
        `REQUEST_MULTIPLE_NEED`, everything else zero.

        Owning WO: **WO-010**.
        """
        raise NotImplementedError("PLAN section 6.1 - implemented in WO-010")

    def reset(self) -> None:
        """No-op: the policy is stateless.

        Takes: nothing. Returns: `None`. Owning WO: **WO-010**.
        """
        raise NotImplementedError("PLAN section 6.1 - implemented in WO-010")


@dataclass
class DPGreedy:
    """Replays the single-enterprise DP policy inside the `N`-enterprise environment (PLAN sec. 5).

    Rule, verbatim from the PLAN section 6.1 table: *applies the single-enterprise DP policy in the
    N-enterprise environment*. Per enterprise `i`, independently and identically:

        PRODUCE step k:  e_ik = solution.policy_effort[idx_T(T_i), idx_S(S_i)]
        REPORT step:     rho_i = solution.policy_rho[idx_T(T_i), idx_S(S_i)]

    where `T_i` and `S_i` are reconstructed from the observation as in this module's header, and
    `idx_T` / `idx_S` locate the enterprise on the DP's own grids - `T` log-spaced on
    `[T_min, grid.target_hi_mult * A * cap]` with `grid.n_target` points, `S` linear on
    `[0, cfg.tech.inventory_cap_mult * cap]` with `grid.n_stock` points (PLAN section 5). The
    single DP effort per period is applied unchanged at each of the `M` PRODUCE steps: uniform
    effort across the steps is what the DP assumes and what the cost `M * kappa * e**2` prices.

    Lookup rule: nearest grid point. PLAN section 5 fixes the grids but not an interpolation
    scheme, and "applies the DP policy" is only well defined as the tabulated policy, so the
    implementation reads the nearest grid entry and does not smooth between entries. This is what
    makes the WO-014 must-pass item checkable - *`DPGreedy` reproduces the DP policy inside the env
    at `N = 1`* - since at a state that sits on a grid point the action must equal the table entry
    exactly. A different rule (bilinear interpolation, say) is a spec question, not an
    implementer's choice (CONTRACT rules 1 and 3).

    What it is and is not. The DP is solved with no input-output structure (`a = 0`, `phi = 1`) on
    one enterprise (PLAN section 5), so inside the `N`-enterprise environment this is a *heuristic*:
    the best reply of an enterprise that believes it faces no supply coupling and no other agents.
    It is the "what would a single-enterprise optimiser do" baseline for PLAN section 6.1 and a
    sanity check on the DP tables themselves. It is not an equilibrium and is never reported as one
    - that is what the exploitability harness of PLAN section 6.3 measures, in Phase 2.

    Requests: the DP has no input dimension, so the request rule is `REQUEST_MULTIPLE_NEED` (exactly
    need), the same neutral value `TruthfulMyopic` uses; inert in Phase 1 at
    `alloc_eta_request = 0`. Under a Phase-2 configuration with `alloc_eta_request > 0` the DP
    supplies no request policy at all, and using this agent there needs an AMBIGUITY REPORT
    (CONTRACT rule 3).

    Binds: `tests/unit/test_dp.py` (WO-014) - `DPGreedy` reproduces the DP policy inside the
    environment at `N = 1`. Owning WOs: **WO-010** (this class), **WO-014** (the solution it reads).
    """

    cfg: EnvConfig
    """The run's configuration. Must be the same configuration the `DPSolution` was solved for;
    `solution.config_hash` is `cfg.hash()` and the implementation checks the two agree before the
    first `act`, because a policy table read against a different target grid is silently wrong."""

    solution: DPSolution
    """The solved single-enterprise problem (`gosplan/agents/dp.py`, WO-014), supplied already
    solved: either straight from `solve_single_enterprise(cfg, grid)` or from the cache keyed by
    `(EnvConfig.hash(), DPGrid)` that the DP work order maintains. This field is the "loads a
    `DPSolution`" of the WO-010 card. Only `policy_effort`, `policy_rho`, `grid` and `config_hash`
    are read; the stationary diagnostics on the solution are for experiments, not for acting."""

    def act(self, obs: Array, phase: Phase, rng: np.random.Generator) -> EnterpriseAction:
        """Look the DP policy up at each enterprise's own `(T_i, S_i)` and return it.

        Takes: `obs` `(N, d)` - fields 2 and 5 only; `phase`, selecting the effort table (PRODUCE)
        or the report table (REPORT); `rng`, unused, since the DP policy is deterministic.
        Returns: an `EnterpriseAction` with `effort` or `report_ratio` from the tables, requests at
        `REQUEST_MULTIPLE_NEED`, everything else zero. The report is clipped to
        `[0, cfg.tech.report_max_ratio]`; if the DP grid was extended past that bound the clipping
        is itself a result and is logged (CONTRACT rule 8).

        Owning WO: **WO-010**.
        """
        raise NotImplementedError("PLAN section 6.1 - implemented in WO-010")

    def reset(self) -> None:
        """No-op: the lookup carries no episode state.

        Takes: nothing. Returns: `None`. The `DPSolution` is run-scoped and is deliberately not
        cleared here. Owning WO: **WO-010**.
        """
        raise NotImplementedError("PLAN section 6.1 - implemented in WO-010")


@dataclass
class Berliner:
    """Safety-factor rule: over-produce, report at target, bank the difference. **Phase-2 stub.**

    Rule, verbatim from the PLAN section 6.1 table: *safety-factor rule: aims 5-10% above target,
    reports at target, banks the rest*. Concretely, per enterprise `i`:

        PRODUCE step k:  e_ik = clip((1 + safety_factor) * T_i / (A_{s(i)} * cap_i), 0, 1)
        REPORT step:     rho_i = 1                       # reports at target, never above
                         the surplus stays in S_i        # "banks the rest"

    It is `TruthfulMyopic` with a safety margin and a ceiling on the claim, so the excess
    accumulates in own-good stock - the mechanism behind hidden reserves (PLAN section 4.1 row 7),
    which is a **held-out** phenomenon: nothing computed from this agent may be plotted, tabulated
    or tested before the Phase-2 acceptance run (PLAN section 4.1, WO-012 forbidden list).

    Status: interface and rule only. The class exists now so that no type moves later (PLAN section
    0, finding F14); the body is written by **WO-030** after the Phase-2 spec revision.
    """

    cfg: EnvConfig
    """The run's configuration."""

    safety_factor: float
    """Fractional over-production target: `e` aims at `(1 + safety_factor) * T_i`. PLAN section 6.1
    gives the range 0.05-0.10 and no point value, and this is not a PLAN section 3 registry row, so
    the field has **no default**: the caller states it and the run manifest records it (CONTRACT
    rule 10). Values outside [0.05, 0.10] are outside the rule as written."""

    def act(self, obs: Array, phase: Phase, rng: np.random.Generator) -> EnterpriseAction:
        """Return the safety-factor action for the current phase.

        Takes: `obs`, `phase`, `rng` as in the `Agent` protocol. Returns: an `EnterpriseAction` per
        the class docstring. Owning WO: **WO-030**.
        """
        raise NotImplementedError("PLAN section 6.1 - implemented in WO-030")

    def reset(self) -> None:
        """Clear per-episode state.

        Takes: nothing. Returns: `None`. Owning WO: **WO-030**.
        """
        raise NotImplementedError("PLAN section 6.1 - implemented in WO-030")


@dataclass
class Weitzman:
    """Ratchet-aware effort reduction as a function of `lambda`. **Phase-2 stub.**

    Rule, verbatim from the PLAN section 6.1 table: *ratchet-aware effort reduction as a function
    of `lambda`*. The agent anticipates that today's fulfilment raises tomorrow's target through
    the ratchet of PLAN section 2.7.1 (`T' = max(T_min, (1 + g) * T * (1 + lambda * clip(rho - 1,
    -c_dn, c_up)))`) and therefore holds effort below the myopic level that `TruthfulMyopic` uses,
    by an amount increasing in `cfg.incentive.ratchet_lambda` and in `cfg.incentive.tenure`.

    The exact functional form is **not fixed by PLAN section 6.1** and is deliberately left open
    until the Phase-2 spec revision (PLAN section 0, finding F14). WO-030 must either transcribe
    the form the revision states or file an AMBIGUITY REPORT (CONTRACT rule 3); inventing a
    plausible reduction curve here is exactly the failure mode rule 3 exists to prevent. Note also
    that the single-enterprise DP of PLAN section 5 already computes the optimal ratchet-aware
    policy exactly, so this agent is a readable caricature for baselines, never the source of a
    quantitative claim about ratchet effects.

    Status: interface and rule sketch only; body written by **WO-030**.
    """

    cfg: EnvConfig
    """The run's configuration. The rule reads `cfg.incentive.ratchet_lambda`,
    `cfg.incentive.growth_directive` and `cfg.incentive.tenure`."""

    def act(self, obs: Array, phase: Phase, rng: np.random.Generator) -> EnterpriseAction:
        """Return the ratchet-aware action for the current phase.

        Takes: `obs`, `phase`, `rng` as in the `Agent` protocol. Returns: an `EnterpriseAction`
        whose effort is the `TruthfulMyopic` level reduced as a function of
        `cfg.incentive.ratchet_lambda`, per the form frozen at the Phase-2 spec revision. Owning
        WO: **WO-030**.
        """
        raise NotImplementedError("PLAN section 6.1 - implemented in WO-030")

    def reset(self) -> None:
        """Clear per-episode state.

        Takes: nothing. Returns: `None`. Owning WO: **WO-030**.
        """
        raise NotImplementedError("PLAN section 6.1 - implemented in WO-030")


@dataclass
class Kornai:
    """Request inflation, anticipating a soft budget constraint. **Phase-2 stub.**

    Rule, verbatim from the PLAN section 6.1 table: *request inflation factor,
    bailout-anticipating*. Concretely, the agent asks for more input than the plan says it needs,

        input_request_ij = request_inflation * need_ij      # in multiples of need, PLAN sec. 2.3

    and, where `cfg.incentive.soft_budget > 0` makes a bailout likely when `fill < 1` (PLAN section
    3, WO-023), it does not adjust effort downward for the input shortfall it expects to be
    covered.

    Request inflation only pays once `cfg.incentive.alloc_eta_request > 0` (the allocation weight
    of PLAN section 2.7.2 is `(q + 1e-6)**eta_q * (need + 1e-6)**eta_n`, and `eta_q = 0` in Phase
    1), so this agent is inert in Phase 1 by construction. Hoarding is a **held-out** phenomenon
    (PLAN section 4.1 row 5): nothing computed from this agent is inspected before the Phase-2
    acceptance run, and its inflation factor is an assumption, never evidence that hoarding
    emerged.

    The bailout side of the rule is under-specified until the Phase-2 spec revision fixes what a
    bailout does (WO-023); until then WO-030 implements the request-inflation half only or files an
    AMBIGUITY REPORT (CONTRACT rule 3).

    Status: interface and rule sketch only; body written by **WO-030**.
    """

    cfg: EnvConfig
    """The run's configuration. The rule reads `cfg.incentive.soft_budget`,
    `cfg.incentive.alloc_eta_request` and `cfg.tech.request_max_multiple`."""

    request_inflation: float
    """Multiplier on `need_ij` in the request action, in multiples of need. PLAN section 6.1 names
    the factor but gives no value, and it is not a PLAN section 3 registry row, so the field has
    **no default**: the caller states it, the manifest records it (CONTRACT rule 10), and the
    environment clips it at `cfg.tech.request_max_multiple` (`action_spec`)."""

    def act(self, obs: Array, phase: Phase, rng: np.random.Generator) -> EnterpriseAction:
        """Return the request-inflating action for the current phase.

        Takes: `obs`, `phase`, `rng` as in the `Agent` protocol. Returns: an `EnterpriseAction`
        whose `input_request` is `request_inflation` per good, with effort and report per the rule
        frozen at the Phase-2 spec revision. Owning WO: **WO-030**.
        """
        raise NotImplementedError("PLAN section 6.1 - implemented in WO-030")

    def reset(self) -> None:
        """Clear per-episode state.

        Takes: nothing. Returns: `None`. Owning WO: **WO-030**.
        """
        raise NotImplementedError("PLAN section 6.1 - implemented in WO-030")


__all__ = [
    "PADDER_EFFORT",
    "PADDER_REPORT_RATIO",
    "REQUEST_MULTIPLE_NEED",
    "Berliner",
    "DPGreedy",
    "Kornai",
    "Padder",
    "Random",
    "TruthfulMyopic",
    "Weitzman",
]
