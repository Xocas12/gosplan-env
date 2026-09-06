"""T-B3 - a fictitious claim propagates as a physical shortage to every downstream buyer.

Realises: PLAN section 11 (behavioural test T-B3), read against PLAN sections 2.7.2 (allocation),
2.7.3 (physical delivery - the padding-to-shortage channel), 2.10 (the I-O structure that carries
the shortage), 2.6 (the coverage aggregator the shortage arrives at) and 4.4 (measurement window).
Owning work order: **WO-002**; on the must-pass list of **WO-010** and binding **WO-006**
(`allocate`, `deliver`).

What T-B3 asserts (PLAN section 11, verbatim): *`Padder` with `S = 0` produces `fill < 1` for all
downstream buyers; `TruthfulMyopic` produces `fill = 1`.*

The channel, from PLAN section 2.7.3 verbatim - four lines, no rule of their own:

    fill_i     = min(1, S_i / claimed_i)                          (fill_i = 1 when claimed_i = 0)
    shipped_i  = min(S_i, claimed_i)
    poolfill_j = sum_{i in j} fill_i * claimed_i / sum_{i in j} claimed_i
    deliv_bj   = alloc_bj * poolfill_j

A claim above stock lowers `poolfill` for the whole good, so every buyer of that good receives less
than it was promised; a claim below stock leaves the difference sitting in `S`. Both are
*consequences* of those four lines. CONTRACT rule 7 forbids implementing either directly, and this
test is the behavioural evidence that the consequence is present without such a rule - the
structural evidence is the lead's diff review named in `test_no_hardcoded_pathology.py`.

Propagation, not just shortfall. The Phase-1 `io_matrix` (PLAN section 3) is a 5-cycle with chords:
every sector needs two inputs at 0.2 each, so a shortage in any one sector reaches every other
sector within two hops. The tests below therefore assert the shortage at the buyers of the padded
good, and then that the coverage `H` of downstream production falls below 1 for the affected
buyers - the shortage arriving as a physical constraint, not as a bookkeeping entry.

    *** THE DIRECTION OF HOARDING IS ASSERTED NOWHERE. ***

    PLAN section 4.1 row 5 (hoarding -> shortage) is an **emergence** claim and is HELD OUT until
    the Phase-2 acceptance run (WO-030). This file asserts *propagation* only: that a claim not
    backed by stock reduces what downstream buyers physically receive. It does not compute request
    inflation `q_ij / need_ij` as a statistic, does not compute `corr(X_ij, 1 - fill_downstream)`,
    does not compare any quantity against a truthful baseline as an *excess*, and asserts no
    direction for input stocks `X_ij`. `TruthfulMyopic` appears here as the `fill = 1` control that
    PLAN section 11 names, never as a baseline to difference against. Nothing in this file is a
    measurement of any held-out phenomenon, and no number it computes may be reported anywhere.

`Padder` is a probe, not a baseline (PLAN section 6.1): its padding is assumed, not learned, so it
measures the plumbing and nothing else.
"""

from __future__ import annotations

import pytest

SKIP_REASON = (
    "skeleton: T-B3 assertions are written by WO-002 (frozen tests); they bind WO-006 "
    "(allocate/deliver), WO-009 (the DELIVER step) and WO-010 (Padder, TruthfulMyopic)"
)
"""Reason attached to every `@pytest.mark.skip` below; the bodies arrive with WO-002."""

PADDING_AGENT = "Padder"
"""`gosplan.agents.heuristic.Padder`: `rho = 1` every period at `PADDER_EFFORT = 0.3`. At
`initial_target_frac = 0.6` that effort produces roughly half the target, so the claim is
fictitious by construction (PLAN section 6.1)."""

TRUTHFUL_AGENT = "TruthfulMyopic"
"""`gosplan.agents.heuristic.TruthfulMyopic`: claims exactly the stock on hand, which is the
`fill = 1` control of PLAN section 11's T-B3 clause."""

TB3_SEEDS: tuple[int, ...] = (0, 1, 2)
"""Environment seeds, one episode each per agent, under common random numbers: both agents are run
at the same `seed_env`, so the yield and audit draws are identical and the only difference between
the two rollouts is the report (PLAN sections 2.15, 4.3). A WO-002 test-design constant."""

MEASUREMENT_WINDOW_START_PERIOD = 2
"""Periods `t >= 2` (PLAN section 4.4) for the rollout tests. The constructed-state test of
`test_zero_stock_claim_gives_zero_fill` needs no window: it calls `deliver` directly."""

FILL_TOL = 1e-12
"""Tolerance on the `fill = 1` control and on the delivery identities. `fill` is a ratio of two
recorded quantities, so the comparison is exact up to floating-point rounding."""

SHORTAGE_MARGIN = 1e-9
"""Strictness margin for `fill < 1` and `deliv < alloc`: an assertion of the form
`fill_i <= 1.0 - SHORTAGE_MARGIN`, so a value that merely rounds to just under 1 does not count as
a shortage."""


@pytest.mark.skeleton
@pytest.mark.skip(reason=SKIP_REASON)
def test_zero_stock_claim_gives_zero_fill() -> None:
    """The literal T-B3 clause: with `S = 0` and a positive claim, `fill = 0` and nothing ships.

    Build a `State` at `p1_default_config()` with `inv_output` all zero, `last_report` set to the
    targets (a full claim `R_i = T_i > 0`) and `last_report_ratio` at 1.0, then call
    `make_planner_view`, `allocate` and `deliver`. Assert, from `deliver`'s return values:

        fill_i     == 0.0                       for every enterprise           (to FILL_TOL)
        shipped_i  == 0.0                       for every enterprise
        deliv_bj   == 0.0                       for every buyer b and good j
        consumer_j == 0.0                       for every good j
        alloc_bj    > 0.0                       for at least one (b, j)        - the promise existed

    so the promise was made and the goods were not there. The last line matters: a test in which
    nothing was allocated would pass the first four vacuously.

    This is the constructed-state form of the clause. It exercises PLAN section 2.7.3 directly, in
    one call, so a failure localises to `deliver` rather than to the schedule. Owning WO:
    **WO-002**; binds **WO-006**.
    """
    raise NotImplementedError("PLAN section 11 (T-B3) - implemented in WO-002")


@pytest.mark.skeleton
@pytest.mark.skip(reason=SKIP_REASON)
def test_padder_shortage_reaches_every_downstream_buyer() -> None:
    """A fictitious claim in a rollout lowers what every buyer of that good receives.

    Roll out `PADDING_AGENT` at `p1_default_config()`, one episode per seed in `TB3_SEEDS`. For
    every windowed period assert:

        fill_i          <= 1.0 - SHORTAGE_MARGIN     for every seller i
        poolfill_j      <= 1.0 - SHORTAGE_MARGIN     for every good j
        deliv_bj        <= alloc_bj - SHORTAGE_MARGIN
                                                     for every (b, j) with alloc_bj > 0
        coverage H_ik   <= 1.0 - SHORTAGE_MARGIN     for every buyer i that needs any input,
                                                     at some PRODUCE step of the next period

    The first three are PLAN section 2.7.3's arithmetic; the fourth is the propagation clause - the
    shortage arrives as a binding physical constraint on production through the CES aggregator of
    PLAN section 2.6, not merely as a smaller number in a ledger column. With the Phase-1 `a` (a
    5-cycle with chords, every row summing to 0.4) every sector buys from two others, so "every
    downstream buyer" is every enterprise.

    Do not assert a magnitude and do not compare the shortfall against another agent's: the size of
    the shortage is not a pre-registered quantity, and the hoarding direction is held out (PLAN
    section 4.1 row 5). Owning WO: **WO-002**; binds **WO-006** and **WO-009**.
    """
    raise NotImplementedError("PLAN section 11 (T-B3) - implemented in WO-002")


@pytest.mark.skeleton
@pytest.mark.skip(reason=SKIP_REASON)
def test_truthful_myopic_gives_full_fill() -> None:
    """The control: claims backed by stock ship in full, so `fill = 1` everywhere.

    Roll out `TRUTHFUL_AGENT` at `p1_default_config()` under the same `TB3_SEEDS` (common random
    numbers with the `Padder` rollout above). For every windowed period assert:

        fill_i     == 1.0                            for every seller i        (to FILL_TOL)
        poolfill_j == 1.0                            for every good j
        deliv_bj   == alloc_bj                       for every buyer and good  (to FILL_TOL)

    including the `claimed_i == 0` case, which PLAN section 2.7.3 defines as `fill_i = 1` rather
    than as a division by zero - assert that case explicitly if any period produces it.

    This is the control PLAN section 11 names, and it is what makes the previous test attributable:
    the two rollouts share every environment draw and differ only in the report, so a shortage in
    one and none in the other isolates the claim as its cause. It is *not* a baseline to difference
    a phenomenon against (PLAN section 4.1 row 5 is held out); no excess is formed here. Owning WO:
    **WO-002**; binds **WO-006** and **WO-010**.
    """
    raise NotImplementedError("PLAN section 11 (T-B3) - implemented in WO-002")
