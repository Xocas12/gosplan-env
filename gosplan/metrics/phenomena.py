"""Pre-registered phenomena - one function per row of the PLAN section 4.1 table.

Realises: PLAN section 4.1 (the seven operationalisations and their estimators), PLAN section 4.2
(the mechanism parameters behind the held-out phenomena, locked now), PLAN section 4.4 (the
measurement window), PLAN section 4.5 (the pre-registered bunching estimator settings and the G2
acceptance criteria) and PLAN section 7.3 (the `forensics_core` coupling and its vendored
fallback). Owning work orders: **WO-016** (rows 1 and 4 only) and **WO-030** (rows 2, 3, 5, 6, 7,
first computed in the Phase-2 acceptance run).

These functions *measure*; they never act. Nothing here feeds an observation, a reward term or an
agent (CONTRACT rule 6), and nothing here is a transition rule (CONTRACT rule 7): a phenomenon is a
statistic computed from a finished `Ledger`, after the run, by the lead.

-------------------------------------------------------------------------------------------------
HELD OUT - PLAN sections 4.1 and 12.3
-------------------------------------------------------------------------------------------------
Rows **2 (storming)**, **5 (hoarding)**, **6 (blat)** and **7 (hidden reserves)** are the emergence
claims. They are held out: **no plot, table or test of these quantities may be produced before the
Phase-2 acceptance run** - not during Phase 1, and not while debugging their own mechanisms. The
Monte Carlo sanity harness of WO-012 may assert conservation and boundedness on the same mechanisms
and **never a direction**, and WO-012 and WO-016 are *forbidden* from implementing them
(WO-016: "Forbidden: implementing rows 2, 5, 6, 7"). Their mechanism parameters are locked now in
PLAN section 4.2 and are restated in each docstring below so they cannot drift; changing one
requires a new, separately pre-registered study. If a held-out phenomenon fails to appear, that is
reported as a failure, not tuned away.
-------------------------------------------------------------------------------------------------

Class of each row (PLAN section 4.1) matters as much as its formula. A **pipeline check** (rows 1,
3, 4) tells us the optimiser and the plumbing work; its appearance is not evidence for Claim A. An
**emergence** row (2, 5, 6, 7) is the claim itself.

Estimator backend. Rows 1, 5 and 7 call into `forensics_core` where it is installed and into
`gosplan/metrics/_fallback.py` otherwise; the choice is made once by
`gosplan.metrics.resolve_estimators()` and is recorded in the run manifest as `estimator_backend`
plus `estimator_version` (CONTRACT rule 10). This module never imports `forensics_core` itself.
"""

from __future__ import annotations

import numpy as np

from gosplan.config import EnvConfig
from gosplan.metrics.ledger import Ledger, StepRecord

# ---------- measurement window (PLAN section 4.4) ----------

MEASUREMENT_FIRST_PERIOD: int = 2
"""Periods `t >= 2` of each episode enter every statistic in this module (PLAN section 4.4). The
burn-in excludes the target-initialisation transient; rows with `t_period < 2` are dropped by the
reader, never by the ledger writer."""

MEASUREMENT_EXCLUDE_EPISODE_END: bool = False
"""PLAN section 4.4: under geometric termination there is **no** end-of-episode exclusion. The
agent never observes periods remaining (PLAN section 2.12, finding F4), so there is no end-game to
trim, and trimming one would silently discard the periods where the ratchet has moved furthest.
This constant exists so the decision is visible in code rather than implied by its absence."""

MEASUREMENT_INCLUDE_AT_BOUND: bool = True
"""PLAN section 4.4: reports at `rho_max` are **included** in histograms and **flagged**. They are
not winsorised, trimmed or dropped - CONTRACT rule 8 makes the bound a result, so every phenomenon
that reads reports also reports `at_bound_frac`, and a run above 1% carries `BOUND_BINDING`
(`gosplan.metrics.ledger.bound_binding`)."""

# ---------- pre-registered bunching estimator settings (PLAN section 4.5) ----------
# Pre-registration data, locked before any learning run. These are the defaults WO-016 hard-codes
# and `write_manifest` records as `bunching_settings` (CONTRACT rule 10). They are not tuning
# knobs: the sensitivity of `b_hat` to them is itself a reported result of the estimator-bias study
# of PLAN section 7.2, which sweeps a grid around these values without changing them here.

BUNCHING_BIN_WIDTH: float = 0.005
"""Histogram bin width on the report ratio `rho` (PLAN section 4.5)."""

BUNCHING_WINDOW_LO: float = 0.6
"""Lower edge of the fitted range, `rho in [0.6, 1.4]` (PLAN section 4.5). Passed as `window_lo`
to `bunching.estimate`."""

BUNCHING_WINDOW_HI: float = 1.4
"""Upper edge of the fitted range (PLAN section 4.5). Passed as `window_hi`."""

BUNCHING_EXCL_LO: float = 0.95
"""Lower edge of the excluded window `[0.95, 1.02]` - the region the counterfactual polynomial is
fitted *without* (PLAN section 4.5). Passed as `excl_lo`."""

BUNCHING_EXCL_HI: float = 1.02
"""Upper edge of the excluded window (PLAN section 4.5). Passed as `excl_hi`."""

BUNCHING_POLY_DEGREE: int = 9
"""Degree of the counterfactual polynomial fitted outside the excluded window (PLAN section 4.5).
Passed as `degree`."""

BUNCHING_EXCESS_LO: float = 1.00
"""Lower edge of the excess-mass window `[1.00, 1.02]` (PLAN section 4.5): the numerator of
`b_hat` is observed minus counterfactual mass on this interval."""

BUNCHING_EXCESS_HI: float = 1.02
"""Upper edge of the excess-mass window (PLAN section 4.5)."""

BUNCHING_HOLE_LO: float = 0.95
"""Lower edge of the hole window `[0.95, 1.00)` (PLAN section 4.5): the missing mass just below
target, computed identically to the excess mass and reported alongside it."""

BUNCHING_HOLE_HI: float = 1.00
"""Upper edge of the hole window, exclusive - `[0.95, 1.00)` (PLAN section 4.5)."""


# =================================================================================================
# PHASE 1 - rows 1 and 4 of the PLAN section 4.1 table. Both are PIPELINE CHECKS.
# Implemented by WO-016; bound by `tests/unit/test_phenomena_p1.py`.
# =================================================================================================


def phenomenon_bunching(ledger: Ledger, cfg: EnvConfig) -> dict[str, float]:
    """Row 1 - fulfilment bunching. Excess mass of the report distribution just above target.

    Class: **pipeline check**. Phase: **1**. If bunching fails to appear the optimiser is broken;
    its appearance is *not* evidence for Claim A (PLAN section 4.1).

    Takes: `ledger`, a finished run's ledger; `cfg`, the configuration that produced it. Returns: a
    mapping with at least `excess_mass`, `hole_mass`, `se`, `ci_lo`, `ci_hi`, `n_obs`,
    `at_bound_frac`.

    Operationalisation (PLAN section 4.1 row 1): excess mass of `rho_report` in `[1.00, 1.02]`
    against a polynomial counterfactual, with the hole mass on `[0.95, 1.00)` reported alongside it.

    Pre-registered estimator settings (PLAN section 4.5, the module constants above, hard-coded as
    defaults and recorded in the manifest as `bunching_settings`):
      * bins of width `BUNCHING_BIN_WIDTH` = 0.005 over `rho` in
        `[BUNCHING_WINDOW_LO, BUNCHING_WINDOW_HI]` = [0.6, 1.4];
      * excluded window `[BUNCHING_EXCL_LO, BUNCHING_EXCL_HI]` = [0.95, 1.02];
      * polynomial of degree `BUNCHING_POLY_DEGREE` = 9 fitted to the bins *outside* that window;
      * `b_hat = (observed - counterfactual mass in [BUNCHING_EXCESS_LO, BUNCHING_EXCESS_HI])
        / (mean counterfactual density in the window)`;
      * `hole_mass` computed identically on `[BUNCHING_HOLE_LO, BUNCHING_HOLE_HI)` = [0.95, 1.00);
      * `se`, `ci_lo`, `ci_hi` by **bootstrap over seeds** - the resampling unit is the seed, not
        the report, because reports within a seed are not independent.

    Measurement window (PLAN section 4.4): only rows with `phase == "report"` and
    `t_period >= MEASUREMENT_FIRST_PERIOD` enter the histogram; there is no end-of-episode
    exclusion; reports at `rho_max` are included and their share returned as `at_bound_frac`
    (CONTRACT rule 8 - see `gosplan.metrics.ledger.bound_binding`).

    Implementation (PLAN section 7.3): call
    `bunching.estimate(x, window_lo, window_hi, bin_width, degree, excl_lo, excl_hi)` on the
    backend returned by `gosplan.metrics.resolve_estimators()`, which prefers `forensics_core` and
    falls back to `gosplan/metrics/_fallback.py` with the identical signature. Record the backend
    and its version in the manifest. This module never imports `forensics_core` directly.

    Binds: `tests/unit/test_phenomena_p1.py` (WO-016) - a synthetic density with known excess mass
    is recovered within 5%, hole mass likewise, the bootstrap SE is produced, and the fallback and
    `forensics_core` signatures agree. Acceptance: G2 criterion 2 of PLAN section 4.5 - at the
    notched configuration `b_hat >= 0.5 * b_hat_DP` with a bootstrap CI excluding 0 in at least 90%
    of 30 seeds, and at the smooth counterfactual (`notch_width = 0.25`, `overfulfilment_cap = inf`)
    the CI covers 0 in at least 90% of seeds.

    Owning WO: **WO-016**.
    """
    # Grouping unit for the seed-level bootstrap: a ledger is one run (one `cfg`, one `seed_env`),
    # so it carries no seed column; the `episode` column is the grouping unit, per the WO-016 lead
    # direction for a single ledger. `x` is passed as a LIST of per-episode arrays so the
    # estimator resamples groups, never individual reports (AMBIGUITY-011 resolution, point 2).
    from gosplan.metrics import resolve_estimators

    rows = _measured_reports(ledger)
    by_episode: dict[int, list[float]] = {}
    for rec in rows:
        by_episode.setdefault(rec.episode, []).append(rec.report_ratio)
    x = [np.asarray(by_episode[ep], dtype=float) for ep in sorted(by_episode)]
    res = resolve_estimators().bunching_estimate(
        x,
        BUNCHING_WINDOW_LO,
        BUNCHING_WINDOW_HI,
        BUNCHING_BIN_WIDTH,
        BUNCHING_POLY_DEGREE,
        BUNCHING_EXCL_LO,
        BUNCHING_EXCL_HI,
    )
    # At-bound reports are in `x` (MEASUREMENT_INCLUDE_AT_BOUND); `at_bound_frac` is their share of
    # the same measured sample `n_obs` counts (CONTRACT rule 8).
    at_bound_frac = sum(1 for rec in rows if rec.at_bound) / len(rows)
    return {
        "excess_mass": float(res.excess_mass),
        "hole_mass": float(res.hole_mass),
        "se": float(res.se),
        "ci_lo": float(res.ci_lo),
        "ci_hi": float(res.ci_hi),
        "n_obs": int(res.n_obs),
        "at_bound_frac": float(at_bound_frac),
    }


def phenomenon_padding(ledger: Ledger, cfg: EnvConfig) -> dict[str, float]:
    """Row 4 - padding, i.e. fictitious output: claims above the stock that backs them.

    Class: **pipeline check**. Phase: **1**.

    Takes: `ledger` and `cfg`. Returns: a mapping with at least `padding`, `padding_index` and the
    elasticity of padding with respect to `audit_rate * penalty_scale`.

    Operationalisation (PLAN section 4.1 row 4):

        padding = mean_i max(0, R_i - S_i) / T_i

    over the measurement window of PLAN section 4.4, where `R_i` is the claim, `S_i` the own-good
    stock the audit compares against - `inv_output_post` on the report row: the period's output
    booked, before the next period's shipment (LEAD ruling AMBIGUITY-019 A) - and
    `T_i` the target. Reported together with

        padding_index = val_measured / val_true                       (PLAN section 2.9.4)

    which is at least 1 whenever output is fictitious, and which is read from the period-level
    columns of the ledger - both are logged quantities and neither is ever observed by an agent
    (CONTRACT rule 6).

    Elasticity: computed across the three `audit_rate * penalty_scale` levels the human recorded in
    `runs/G1_decision.md` at gate G1, and compared against the single-enterprise DP's prediction
    (PLAN section 5). G2 criterion 3 (PLAN section 4.5) requires learned fictitious padding to be
    **monotone decreasing** in `a * pen` and within **0.03** (ratio units) of the DP's value at each
    of the three levels.

    Binds: `tests/unit/test_phenomena_p1.py` (WO-016). Owning WO: **WO-016**.
    """
    # LEAD ruling AMBIGUITY-017: `padding_index` is the ratio of window means of the period-level
    # `val_measured` and `val_true` (one value per (episode, period)); the elasticity across the
    # three `a * pen` levels is computed by the caller (WO-019/WO-020) from three calls.
    rows = _measured_reports(ledger)
    # AMBIGUITY-019 A: `S_i` is the audited stock, `inv_output_post` on the REPORT row (the DP's
    # `S'`); `inv_output_pre` omits the period's own output and counts truthful reports as padding.
    ratios = [max(0.0, rec.report - rec.inv_output_post) / rec.target for rec in rows]
    periods = {(rec.episode, rec.t_period): (rec.val_measured, rec.val_true) for rec in rows}
    measured = np.mean([vm for vm, _ in periods.values()])
    true = np.mean([vt for _, vt in periods.values()])
    index = float(measured / true) if true != 0.0 else float("nan")
    return {"padding": float(np.mean(ratios)), "padding_index": index}


def _measured_reports(ledger: Ledger) -> list[StepRecord]:
    """REPORT rows inside the PLAN section 4.4 measurement window, at-bound rows included.

    `t_period >= MEASUREMENT_FIRST_PERIOD`; no end-of-episode exclusion
    (`MEASUREMENT_EXCLUDE_EPISODE_END`); at-bound reports kept (`MEASUREMENT_INCLUDE_AT_BOUND`).
    The window is applied here, by the reader, never by the ledger writer.
    """
    rows = [
        rec
        for rec in ledger.records
        if rec.phase == "report" and rec.t_period >= MEASUREMENT_FIRST_PERIOD
    ]
    if not rows:
        raise ValueError("no REPORT rows inside the PLAN section 4.4 measurement window")
    return rows


# =================================================================================================
# PHASE 2 - rows 2, 3, 5, 6 and 7 of the PLAN section 4.1 table.
#
#   *** HELD OUT: rows 2, 5, 6 and 7 (storming, hoarding, blat, hidden reserves). ***
#
# Per PLAN sections 4.1 and 12.3 these four must not be computed, plotted, tabulated or tested
# before the Phase-2 acceptance run (WO-031, gate G3) - not in Phase 1, and not while debugging
# their own mechanisms. WO-012 (MC sanity) and WO-016 (P1 metrics) are FORBIDDEN from implementing
# them; WO-012 may assert conservation and boundedness on the same mechanisms and never a
# direction. Their mechanism parameters are locked in PLAN section 4.2 and restated below.
# Row 3 (quality) is a pipeline check, not an emergence claim, and is not held out.
#
# Implemented by WO-030.
# =================================================================================================


def phenomenon_storming(
    ledger: Ledger, baseline_ledger: Ledger, cfg: EnvConfig
) -> dict[str, float]:
    """Row 2 - storming: effort deferred within the period beyond what input arrival forces.

    Class: **emergence** (Claim A). Phase: **2**. **HELD OUT.**

    *** HELD OUT (PLAN sections 4.1, 12.3). No plot, table or test of this quantity may exist
    before the Phase-2 acceptance run. WO-012 and WO-016 are forbidden from implementing it; the MC
    sanity harness may assert only conservation and boundedness on the delivery-timing mechanism,
    never a direction. If storming fails to appear, that is reported as a failure. ***

    Takes: `ledger`, the run's ledger; `baseline_ledger`, the truthful-myopic heuristic's ledger
    under **the same delivery-timing draws** (common random numbers - shared `seed_env`); `cfg`.
    Returns: a mapping with at least `gini`, `gini_baseline`, `excess`.

    Operationalisation (PLAN section 4.1 row 2):

        gini      = Gini_k(e_ik) of *effort* across the steps k = 0 .. M-1 within a period
        excess    = gini - gini_baseline

    A positive excess means agents defer effort beyond what input arrival forces, which is why the
    comparison ledger is a required argument and not an optional convenience: the baseline absorbs
    the mechanical concentration that stochastic arrival alone produces. Effort is read from the
    `effort` column of the PRODUCE rows, over the measurement window of PLAN section 4.4.

    Locked mechanism parameters (PLAN section 4.2, so they cannot drift):
        delivery_timing = "stochastic"
        arrival_probs   = (0.25, 0.25, 0.25, 0.25)   uniform-random arrival over steps 0..3, so
                                                     there is no mechanical backloading and any
                                                     Gini excess is behavioural
        yield_sigma     x1 (unchanged from the Phase-1 tuple)

    Owning WO: **WO-030**.
    """
    gini, n_zero = _mean_period_gini(ledger, cfg)
    gini_base, n_zero_base = _mean_period_gini(baseline_ledger, cfg)
    return {
        "gini": gini,
        "gini_baseline": gini_base,
        "excess": gini - gini_base,
        "n_zero_effort_periods": float(n_zero),
        "n_zero_effort_periods_baseline": float(n_zero_base),
    }


def phenomenon_quality(ledger: Ledger, cfg: EnvConfig) -> dict[str, float]:
    """Row 3 - quality degradation under a quantity-keyed objective.

    Class: **pipeline check**. Phase: **2**. Not held out, and deliberately not part of Claim A:
    with quality under direct agent control and absent from the measured objective, degradation is
    a direct optimum of the reward rather than an emergent workaround (PLAN section 4.1, finding
    F8).

    Takes: `ledger` and `cfg`. Returns: a mapping with at least `mean_quality` and
    `mean_quality_weighted`.

    Operationalisation (PLAN section 4.1 row 3): mean `qbar` (the period-average quality, from the
    `quality_acc` column) under `objective_metric = "val"`, against the same configuration run at
    `objective_metric = "quality_weighted"` with `quality_measurability = 1`. Measurement window as
    PLAN section 4.4.

    The mechanism itself - quality routed through the input bundle, the effort-cost term
    `kappa_q * q * e`, and the measured factor `q_hat = 1 + mu * (qbar - 1)` - is WO-021's
    (`supply.quality_matters`, `supply.quality_cost`, `information.quality_measurability`).

    Owning WO: **WO-030**, with the mechanism from **WO-021**.
    """
    m = cfg.incentive.steps_per_period
    mu = cfg.information.quality_measurability
    rows = _measured_reports(ledger)
    if cfg.supply.quality_matters:
        qbar = np.array([rec.quality_acc / m for rec in rows], dtype=float)
    else:
        qbar = np.ones(len(rows))
    return {
        "mean_quality": float(qbar.mean()),
        "mean_quality_weighted": float(np.mean(1.0 + mu * (qbar - 1.0))),
        "n_obs": float(qbar.size),
    }


def phenomenon_hoarding(
    ledger: Ledger, baseline_ledger: Ledger, cfg: EnvConfig
) -> dict[str, float]:
    """Row 5 - input hoarding and the shortage it propagates.

    Class: **emergence** (Claim A). Phase: **2**. **HELD OUT.**

    *** HELD OUT (PLAN sections 4.1, 12.3). No plot, table or test of this quantity before the
    Phase-2 acceptance run; WO-012 and WO-016 are forbidden from implementing it. Note that test
    T-B3 asserts shortage *propagation* only - the direction of hoarding is asserted nowhere, by
    design. ***

    Takes: `ledger`; `baseline_ledger`, the truthful-myopic run under common random numbers; `cfg`.
    Returns: a mapping with at least `request_inflation`, `request_inflation_baseline`,
    `corr_stock_shortfall`, `dispersion_stat`.

    Operationalisation (PLAN section 4.1 row 5):

        request_inflation    = mean over the window of q_ij / need_ij   (`request` / `need` columns)
        corr_stock_shortfall = corr(X_ij, 1 - fill_downstream)          (`inv_inputs`, `fill`)

    both reported against the truthful-myopic baseline under the same draws. The cross-sectional
    part uses `dispersion.cross_section(values, groups)` on the backend from
    `gosplan.metrics.resolve_estimators()` (PLAN section 7.3), with sectors as groups.

    Locked mechanism parameters (PLAN section 4.2):
        alloc_eta_request     eta_q = 0.7    requests start to pay in the allocation weight
        input_complementarity theta = 8      near-Leontief inputs
        input_holding_loss    h_X   = 0.01   holding inputs is no longer free

    The mechanism is a rule about allocation *weights* (PLAN section 2.7.2), never an instruction to
    inflate a request (CONTRACT rule 7).

    Owning WO: **WO-030**.
    """
    from gosplan.metrics import resolve_estimators

    run = _hoarding_stats(ledger, cfg)
    base = _hoarding_stats(baseline_ledger, cfg)
    backend = resolve_estimators()
    if run["pair_values"].size:
        disp = backend.dispersion_cross_section(run["pair_values"], run["pair_groups"])
        stat, p = float(disp.statistic), float(disp.p_value)
    else:
        stat, p = float("nan"), float("nan")
    return {
        "request_inflation": run["inflation"],
        "request_inflation_baseline": base["inflation"],
        "corr_stock_shortfall": run["corr"],
        "corr_stock_shortfall_baseline": base["corr"],
        "dispersion_stat": stat,
        "dispersion_p": p,
    }


def phenomenon_blat(ledger: Ledger, cfg: EnvConfig) -> dict[str, float]:
    """Row 6 - blat: horizontal barter around the allocation system.

    Class: **emergence** (Claim A). Phase: **2**. **HELD OUT.**

    *** HELD OUT (PLAN sections 4.1, 12.3). No plot, table or test of this quantity before the
    Phase-2 acceptance run; WO-012 and WO-016 are forbidden from implementing it. ***

    Takes: `ledger` and `cfg`. Returns: a mapping with at least `trade_volume_share`,
    `n_matched_pairs`, `mean_surplus`.

    Operationalisation (PLAN section 4.1 row 6): executed trade volume (the `trade_volume` column)
    divided by total intermediate allocation (the `alloc` columns), over the measurement window of
    PLAN section 4.4. It must exceed the truthful-myopic baseline, which never trades - so a
    non-zero share is the whole claim.

    Locked mechanism parameters (PLAN section 4.2):
        horizontal_visibility = 1.0    every counterparty is visible
        trade_tau             = 0.05   per-unit transaction cost; tau > 0 is what makes wash trades
                                       unprofitable (PLAN section 2.13)

    Surplus is computed by the environment from the effect of the swap on next step's intended
    output and is never self-reported (PLAN section 2.13); it enters the reward only through the
    `trade_surplus` term already named in CONTRACT rule 4.

    Owning WO: **WO-030**.
    """
    rows = [rec for rec in ledger.records if rec.t_period >= MEASUREMENT_FIRST_PERIOD]
    volume = float(sum(rec.trade_volume for rec in rows))
    alloc = float(sum(sum(rec.alloc) for rec in rows))
    if alloc > 0:
        share = volume / alloc
    else:
        share = 0.0 if volume == 0 else float("inf")
    return {
        "trade_volume_share": share,
        "trade_volume": volume,
        # Pairs are not in the ledger: this counts selling enterprise-periods (P2_REVISION R13.4).
        "n_matched_pairs": float(sum(1 for rec in rows if rec.trade_volume > 0)),
        # Not recoverable from the ledger; the surplus is visible only through the reward (R13.4).
        "mean_surplus": float("nan"),
    }


def phenomenon_hidden_reserves(ledger: Ledger, cfg: EnvConfig) -> dict[str, float]:
    """Row 7 - hidden reserves and shaving: stock held back below the claim.

    Class: **emergence** (Claim A). Phase: **2**. **HELD OUT.**

    *** HELD OUT (PLAN sections 4.1, 12.3). No plot, table or test of this quantity before the
    Phase-2 acceptance run - not during Phase 1 and not while debugging its mechanism. WO-012 and
    WO-016 are forbidden from implementing it; the MC sanity harness may assert only conservation
    and boundedness on the inventory mechanism, never a direction. ***

    Takes: `ledger` and `cfg`. Returns: a mapping with at least `hidden_reserves`,
    `reconciliation_stat` and `p_value`.

    Operationalisation (PLAN section 4.1 row 7):

        hidden_reserves = mean_i max(0, S_i - R_i) / T_i     after delivery

    over the measurement window of PLAN section 4.4, with `S_i` the post-delivery own-good stock
    (`inv_output_post` on the report row). It **must vanish** when `g = 0` and
    `penalty_arg = "absolute"` - the falsification condition is part of the pre-registration.

    The ledger-level part uses
    `reconciliation.ledger_test(reported_supply, received_inputs, io_matrix, prices)` on the backend
    from `gosplan.metrics.resolve_estimators()` (PLAN section 7.3): claimed supply is reconciled
    against buyers' received inputs through the I-O matrix at plan prices, and the statistic and its
    p-value are returned. Its power curve under known fictitious output is itself a reported result
    of PLAN section 7.2.

    Locked mechanism parameters (PLAN section 4.2), unchanged from Phase 1:
        growth_directive g   = 0.02   (provisional until the G1 decision replaces it)
        penalty_arg          = "positive_part"   under-reporting is never penalised
        holding_loss     h   = 0.02

    Audits compare the claim to stock on hand, not to production (PLAN section 2.8), which is why a
    reserve protects against them - a consequence of the rules, never a rule (CONTRACT rule 7).

    Owning WO: **WO-030**.
    """
    from gosplan.env.prices import initial_prices
    from gosplan.metrics import resolve_estimators

    rows = _measured_reports(ledger)
    hidden = np.array(
        [max(0.0, rec.inv_output_post - rec.report) / rec.target for rec in rows], dtype=float
    )
    deliv_next = _next_deliver_rows(ledger)
    a = np.asarray(cfg.supply.io_matrix, dtype=float)
    prices = np.asarray(initial_prices(cfg), dtype=float)
    keyed = [(rec, deliv_next.get((rec.episode, rec.t_period + 1, rec.enterprise))) for rec in rows]
    keyed = [(rec, nxt) for rec, nxt in keyed if nxt is not None]
    if keyed:
        result = resolve_estimators().reconciliation_ledger_test(
            np.array([rec.report for rec, _ in keyed], dtype=float),
            np.array([nxt.deliv for _, nxt in keyed], dtype=float),
            a[[rec.sector for rec, _ in keyed]],
            prices[[rec.sector for rec, _ in keyed]],
        )
        stat, p = float(result.statistic), float(result.p_value)
    else:
        stat, p = float("nan"), float("nan")
    return {
        "hidden_reserves": float(hidden.mean()),
        "reconciliation_stat": stat,
        "p_value": p,
        "n_obs": float(hidden.size),
    }


# ---------- WO-030 helpers (definitions: spec/P2_REVISION.md R13) ----------


def _gini(values: np.ndarray) -> float:
    """`sum_{k,l} |e_k - e_l| / (2 n^2 mean)`; the caller excludes all-zero vectors."""
    values = np.asarray(values, dtype=float)
    n = values.size
    return float(np.abs(values[:, None] - values[None, :]).sum() / (2.0 * n * n * values.mean()))


def _mean_period_gini(ledger: Ledger, cfg: EnvConfig) -> tuple[float, int]:
    """Mean within-period effort Gini over the window, and the count of zero-effort periods."""
    efforts: dict[tuple[int, int, int], list[float]] = {}
    for rec in ledger.records:
        if rec.phase == "produce" and rec.t_period >= MEASUREMENT_FIRST_PERIOD:
            efforts.setdefault((rec.episode, rec.t_period, rec.enterprise), []).append(rec.effort)
    ginis, n_zero = [], 0
    for values in efforts.values():
        arr = np.asarray(values, dtype=float)
        if arr.sum() <= 0.0:
            n_zero += 1
        else:
            ginis.append(_gini(arr))
    return (float(np.mean(ginis)) if ginis else float("nan")), n_zero


def _next_deliver_rows(ledger: Ledger) -> dict[tuple[int, int, int], StepRecord]:
    """The step-0 PRODUCE row (where DELIVER ran) keyed by (episode, t_period, enterprise)."""
    return {
        (rec.episode, rec.t_period, rec.enterprise): rec
        for rec in ledger.records
        if rec.phase == "produce" and rec.k_step == 0
    }


def _hoarding_stats(ledger: Ledger, cfg: EnvConfig) -> dict[str, object]:
    """Request inflation, the stock/shortfall correlation and the per-(i, j) inflation values."""
    rows = _measured_reports(ledger)
    deliver = _next_deliver_rows(ledger)
    sector_of = np.asarray(cfg.supply.sector_of, dtype=int)
    fillbar: dict[tuple[int, int, int], float] = {}
    by_period: dict[tuple[int, int], list[StepRecord]] = {}
    for rec in deliver.values():
        by_period.setdefault((rec.episode, rec.t_period), []).append(rec)
    for (episode, t), recs in by_period.items():
        for j in range(cfg.supply.n_sectors):
            sellers = [r for r in recs if sector_of[r.enterprise] == j]
            if not sellers:
                continue
            w = np.array([r.shipped for r in sellers], dtype=float)
            f = np.array([r.fill for r in sellers], dtype=float)
            fillbar[(episode, t, j)] = (
                float((w * f).sum() / w.sum()) if w.sum() > 0 else float(f.mean())
            )
    ratios, stock, shortfall = [], [], []
    per_pair: dict[tuple[int, int], list[float]] = {}
    for rec in rows:
        for j, (q, need) in enumerate(zip(rec.request, rec.need, strict=True)):
            if need <= 0:
                continue
            ratios.append(q / need)
            per_pair.setdefault((rec.enterprise, j), []).append(q / need)
            nxt = fillbar.get((rec.episode, rec.t_period + 1, j))
            if nxt is not None:
                stock.append(rec.inv_inputs[j])
                shortfall.append(1.0 - nxt)
    corr = float("nan")
    if len(stock) >= 2 and np.std(stock) > 0 and np.std(shortfall) > 0:
        corr = float(np.corrcoef(stock, shortfall)[0, 1])
    keys = sorted(per_pair)
    return {
        "inflation": float(np.mean(ratios)) if ratios else float("nan"),
        "corr": corr,
        "pair_values": np.array([np.mean(per_pair[k]) for k in keys], dtype=float),
        "pair_groups": np.array([int(sector_of[i]) for i, _ in keys], dtype=int),
    }
