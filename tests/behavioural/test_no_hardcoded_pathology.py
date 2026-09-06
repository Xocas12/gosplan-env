"""T-B1 - no hard-coded pathology; the file CONTRACT rule 7 names by path.

Realises: PLAN section 11 (behavioural test T-B1) and CONTRACT rule 7, read against PLAN sections
2.6-2.8 (the transition rules the clauses probe), 4.1 (phenomena and their classes), 4.4
(measurement window) and 4.5 (the pre-registered histogram settings). Owning work order:
**WO-002** (the LEAD writes `ref/`, `tests/unit`, `tests/behavioural`, `tests/golden`). It is on
the must-pass list of **WO-010** (heuristic agents) and is the standing check every
`gosplan/env/` diff is measured against.

CONTRACT rule 7 (NO HARD-CODED PATHOLOGY): no transition rule or reward term may implement
bunching, padding, storming, hoarding, shaving or trade directly, and this file is the behavioural
check the rule names by path.

    *** PASSING THIS FILE IS NECESSARY, NOT SUFFICIENT. ***

    A green run here says only that one fixed, non-strategic policy - `TruthfulMyopic` at the
    configuration below - is not pushed into pathological behaviour by the environment itself. It
    cannot show that no rule anywhere encodes a pathology: a rule that fires only under a
    configuration, a state or a policy this file never visits passes it untouched. CONTRACT rule 7
    therefore also requires that **the LEAD reviews every `gosplan/env/` diff against rule 7**, by
    reading it, at gate G0 (PLAN section 13) and at every later change. A green T-B1 never
    substitutes for that review, and no session may cite this file as evidence that a rule is
    admissible.

The configuration (PLAN section 11, T-B1, verbatim): `w = 0.25`, `rho_cap = inf`, `a = 1`,
`pen = 200`, `g = 0`, everything else at `p1_default_config()` - so `N = 20`, `J = 5`, `M = 4`,
`delivery_timing = "uniform"`, `alloc_eta_request = 0`, `horizontal_visibility = 0`. The schedule
is the smooth counterfactual (no notch, no cap, hence no kink anywhere), audits are certain, the
penalty is punitive and the growth directive is off. Under that configuration a truthful policy
has no incentive the environment could dress up as a pathology: whatever this file sees is the
environment's own doing.

The agent is `TruthfulMyopic` from `gosplan.agents.heuristic` (PLAN section 6.1): effort
`clip(T_i / (A_{s(i)} * cap_i), 0, 1)` at every PRODUCE step, `report_ratio = S_i / T_i` at the
REPORT step, `input_request = need_ij` exactly (`REQUEST_MULTIPLE_NEED = 1.0`). It is the same
policy `ref/gen_golden.py` drives half the golden matrix with, so a divergence here and a golden
mismatch localise together.

The clauses of T-B1, at that configuration, over the measurement window of PLAN section 4.4
(periods `t >= 2`), one episode per seed in `TB1_SEEDS`:

    1. truthful report      `R_i == S_i` to `TRUTHFUL_TOL`, at every REPORT step
    2. no spike             no bin of the `rho_report` histogram holds more than `SPIKE_RATIO`
                            times the mean of its two neighbours
    3. effort Gini          `Gini_k(e_ik)` within a period equals the value implied by yield noise
                            alone under `uniform` delivery
    4. no trade             executed trade volume and the trade-surplus reward term are exactly 0
    5. requests equal need  `q_ij == need_ij` to `TRUTHFUL_TOL`

(PLAN section 11 states clauses 4 and 5 in one sentence; they are separate tests here because they
fail for different reasons and a single assertion would hide which channel broke.)

HELD-OUT PHENOMENA (PLAN section 4.1). Clause 3 computes a within-period effort Gini, which is the
*operationalisation* of row 2 (storming). It is admissible here, and only here, because it is an
**equality against the mechanically implied value**: no baseline is subtracted, no excess is
formed, no direction is asserted and no number is reported. Rows 2, 5, 6 and 7 are computed as
phenomena for the first time in the Phase-2 acceptance run (WO-030). Nothing in this file may
compute request inflation as a *statistic*, `corr(X_ij, 1 - fill_downstream)`, trade volume as a
*measurement*, or `max(0, S_i - R_i) / T_i`.
"""

from __future__ import annotations

import pytest

SKIP_REASON = (
    "skeleton: T-B1 assertions are written by WO-002 (frozen tests); the behaviour they bind is "
    "WO-010 (heuristic agents) on the environment of WO-005..WO-009"
)
"""Reason attached to every `@pytest.mark.skip` below. The bodies are supplied by WO-002; until
then each test raises `NotImplementedError` and is skipped, so a skeleton checkout is green."""

TB1_CONFIG_OVERRIDES: dict[str, dict[str, object]] = {
    "incentive": {
        "notch_width": 0.25,
        "overfulfilment_cap": float("inf"),
        "penalty_scale": 200.0,
        "growth_directive": 0.0,
    },
    "information": {"audit_rate": 1.0},
}
"""The T-B1 configuration of PLAN section 11, as overrides on `p1_default_config()`: `w = 0.25`,
`rho_cap = inf`, `pen = 200`, `g = 0`, `a = 1`. Every other field keeps its Phase-1 default, which
already gives `delivery_timing = "uniform"` and `horizontal_visibility = 0`."""

TB1_AGENT = "TruthfulMyopic"
"""The policy named by PLAN section 11's T-B1 clause: `gosplan.agents.heuristic.TruthfulMyopic`."""

TB1_SEEDS: tuple[int, ...] = tuple(range(30))
"""Environment seeds rolled out, one episode each. A WO-002 test-design constant, not a PLAN
number: PLAN section 11 fixes the configuration, the policy and the clauses, not the sample size.
Thirty geometric episodes at `N = 20` give of the order of 6,000 REPORT rows, enough for clause
2's histogram; changing it changes the precision of the clauses, never what they assert."""

MEASUREMENT_WINDOW_START_PERIOD = 2
"""Periods `t >= 2` only (PLAN section 4.4): the target-initialisation transient is burn-in. Under
geometric termination there is no end-of-episode exclusion."""

TRUTHFUL_TOL = 1e-9
"""Tolerance for clauses 1, 3 and 5, from PLAN section 11 ("reports equal stock to 1e-9")."""

SPIKE_RATIO = 3.0
"""Clause 2's factor: no histogram bin may exceed `SPIKE_RATIO` times the mean of its two
neighbours (PLAN section 11, verbatim)."""

HIST_BIN_WIDTH = 0.005
"""Histogram bin width for clause 2, taken from the pre-registered bunching settings of PLAN
section 4.5 so that this file and `phenomenon_bunching` (WO-016) bin identically."""

HIST_RANGE: tuple[float, float] = (0.6, 1.4)
"""Histogram support for clause 2, also from PLAN section 4.5. Reports outside it - including any
at `rho_max` - are counted and displayed alongside a failure (CONTRACT rule 8), never dropped."""


@pytest.mark.skeleton
@pytest.mark.skip(reason=SKIP_REASON)
def test_truthful_report_equals_stock() -> None:
    """Clause 1: a truthful policy's claim equals its stock, to 1e-9.

    Roll out `TB1_AGENT` at `p1_default_config()` overridden by `TB1_CONFIG_OVERRIDES`, one episode
    per seed in `TB1_SEEDS`, recording a `Ledger`. For every REPORT row with
    `t_period >= MEASUREMENT_WINDOW_START_PERIOD`, assert

        abs(row.report - row.inv_output_post) <= TRUTHFUL_TOL

    i.e. `R_i == S_i`: the policy reports `rho = S_i / T_i` and PLAN section 2.8 sets
    `R_i = clip(rho, 0, rho_max) * T_i`, so the two agree exactly unless the environment moved the
    claim. Assert additionally that no such row has `at_bound` True - a clipped report could
    satisfy the identity only by coincidence, and the clip would be doing the work.

    A failure means the environment rewrote a report: padding implemented in `gosplan/env/`, which
    is exactly what CONTRACT rule 7 forbids. Owning WO: **WO-002**; binds **WO-007**
    (`process_reports`) and **WO-010**.
    """
    raise NotImplementedError("PLAN section 11 (T-B1) - implemented in WO-002")


@pytest.mark.skeleton
@pytest.mark.skip(reason=SKIP_REASON)
def test_report_histogram_has_no_spike() -> None:
    """Clause 2: the truthful report histogram has no bin above 3x its neighbours' mean.

    Bin every windowed REPORT row's `report_ratio` with `HIST_BIN_WIDTH` over `HIST_RANGE` (the
    pre-registered binning of PLAN section 4.5). For every bin `b` strictly inside the support
    whose two neighbours `b - 1` and `b + 1` are both non-empty, assert

        count[b] <= SPIKE_RATIO * (count[b - 1] + count[b + 1]) / 2

    On failure report the offending bin's centre, its count, its neighbours' counts, and the number
    of reports that fell outside `HIST_RANGE` or at `rho_max`: those are counted and displayed
    (CONTRACT rule 8), never discarded.

    The restriction to bins with two non-empty neighbours is WO-002's reading of PLAN section 11's
    clause - an isolated bin in the tail of a smooth density carries no information about bunching,
    while a spike inside the support is exactly what a hard-coded notch would produce. An
    implementer who finds that reading under-determined for a configuration it must run files an
    AMBIGUITY REPORT (CONTRACT rule 3); it never re-bins to get a pass.

    Under `w = 0.25` and `rho_cap = inf` the bonus is smooth with a continuous derivative everywhere
    (PLAN section 2.8), so a truthful policy has nothing to bunch at: a spike means the environment
    manufactured one. Owning WO: **WO-002**.
    """
    raise NotImplementedError("PLAN section 11 (T-B1) - implemented in WO-002")


@pytest.mark.skeleton
@pytest.mark.skip(reason=SKIP_REASON)
def test_effort_gini_equals_yield_noise_implied_value() -> None:
    """Clause 3: within-period effort dispersion is exactly what the policy's own rule implies.

    For every enterprise and every windowed period, compute `Gini_k(e_ik)` over that period's `M`
    PRODUCE rows from the ledger's `effort` column, and compute `gini_implied` by evaluating the
    `TruthfulMyopic` rule outside the environment on the same recorded observations - effort
    `clip(T_i / (A_{s(i)} * cap_i), 0, 1)` at each step. Assert

        abs(gini_observed - gini_implied) <= TRUTHFUL_TOL

    for every (enterprise, period) pair. Under Phase-1 `uniform` delivery that rule reads only
    `T_i`, `A_{s(i)}` and `cap_i`, none of which moves inside a period, so the implied sequence is
    constant and `gini_implied` is 0; yield noise enters `y`, never `e`. Any excess is the
    environment deferring effort - mechanical backloading, i.e. storming implemented in
    `gosplan/env/` - and a CONTRACT rule 7 violation.

    HELD-OUT DISCIPLINE (PLAN section 4.1 row 2): this is an equality against a mechanically
    implied value. No baseline ledger is subtracted, no excess is formed, no direction is asserted
    and nothing is written to a report. The storming phenomenon itself is computed for the first
    time by `phenomenon_storming` in the Phase-2 acceptance run (WO-030). Owning WO: **WO-002**.
    """
    raise NotImplementedError("PLAN section 11 (T-B1) - implemented in WO-002")


@pytest.mark.skeleton
@pytest.mark.skip(reason=SKIP_REASON)
def test_no_trade_occurs() -> None:
    """Clause 4: nothing trades, and no trade term reaches the reward.

    Over every row of the same rollout assert `row.trade_volume == 0.0` exactly, and that the
    `trade_surplus` argument the environment passes to `enterprise_reward` is `None` or an all-zero
    array, so the REPORT-step reward is `scale * (B(rho) - 1[audited] * Pen)` exactly (CONTRACT
    rule 4). Assert also `"trade_offer" not in active_action_dims(cfg)` at this configuration:
    Phase 1 has `horizontal_visibility = 0`, and trade is a Phase-2 mechanism (PLAN section 2.13).

    Exact zero, not a tolerance: Phase 1 executes no trade at all, so any non-zero value is the
    environment acting against its own configuration.

    HELD-OUT DISCIPLINE (PLAN section 4.1 row 6): this asserts absence; it does not *measure* trade
    volume as the blat statistic, which is `phenomenon_blat` in WO-030. Owning WO: **WO-002**.
    """
    raise NotImplementedError("PLAN section 11 (T-B1) - implemented in WO-002")


@pytest.mark.skeleton
@pytest.mark.skip(reason=SKIP_REASON)
def test_requests_equal_need() -> None:
    """Clause 5: a truthful policy's input requests are exactly its need.

    For every windowed REPORT row and every good `j`, assert

        abs(row.request[j] - row.need[j]) <= TRUTHFUL_TOL

    `TruthfulMyopic` requests `REQUEST_MULTIPLE_NEED = 1.0` times `need_ij` (PLAN section 6.1), and
    PLAN section 2.3 bounds the request at `r_max * need_ij` with `r_max = 3`, so the clip is
    inactive here. A discrepancy means the environment rescaled, padded or floored a request:
    hoarding implemented in `gosplan/env/`, a CONTRACT rule 7 violation.

    HELD-OUT DISCIPLINE (PLAN section 4.1 row 5): this is an identity on one truthful policy's
    action. Request inflation `q_ij / need_ij` as a *statistic*, and any correlation between held
    inputs and downstream fill, belong to `phenomenon_hoarding` in the Phase-2 acceptance run
    (WO-030) and appear nowhere in this file. Owning WO: **WO-002**.
    """
    raise NotImplementedError("PLAN section 11 (T-B1) - implemented in WO-002")
