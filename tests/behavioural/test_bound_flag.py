"""T-B8 - bounds are results: reports at `rho_max` raise `BOUND_BINDING`.

Realises: PLAN section 11 (behavioural test T-B8) and CONTRACT rule 8, read against PLAN sections
2.3 (the action bounds), 2.8 (`R_i = clip(rho, 0, rho_max) * T_i`), 4.4 (the measurement window -
reports at `rho_max` are included in histograms and flagged), 4.5 (the G2 hygiene criterion: the
flag absent in all Phase-1 gate runs) and 10 (`Ledger`, `write_manifest`). Owning work order:
**WO-002**; on the must-pass list of **WO-011** (`gosplan/metrics/ledger.py`).

CONTRACT rule 8 (BOUNDS ARE RESULTS): `report_ratio` is bounded at `rho_max = 10`. The fraction of
reports at the bound is logged; more than 1% flags the run manifest `BOUND_BINDING` and the result
is reported with the flag. **Never silently widen or narrow a bound to fix a result.**

What T-B8 asserts (PLAN section 11, verbatim): *forcing `rho = 10` in >1% of reports sets
`BOUND_BINDING`.* The clause has a strict inequality in it, so the test drives both sides of the
threshold: a fraction above `AT_BOUND_FLAG_THRESHOLD` must raise the flag, and one below must not.
A test that only checked the raising half would pass a ledger that flagged everything.

The mechanism under test, in three pieces:

    1. `process_reports` (WO-007) clips `rho` to `rho_max` and records whether the report sat at
       the bound, so `StepRecord.at_bound` is the environment's own observation, not the test's;
    2. `Ledger.append` (WO-011) maintains the running fraction and raises `BOUND_BINDING_FLAG` when
       it exceeds `AT_BOUND_FLAG_THRESHOLD`, counting REPORT rows only;
    3. `write_manifest` (WO-011) carries every raised flag into `runs/<hash>/manifest.json`
       (CONTRACT rule 10), which is where a reader of the result meets it.

`bound_binding(ledger)` answers the same question of a finished ledger; the two answers must agree,
and this file asserts that they do - an incremental flag that disagreed with the batch predicate
would let a run be reported without its flag.

The driving policy is a test-local one: it reports `rho_max` on a chosen, deterministic subset of
REPORT steps and truthfully otherwise. It is a fixture, not a baseline, and it lives in this file
rather than in `gosplan/agents/heuristic.py` because nothing in the production agent set should
exist to trip a flag. `Random` reaches high ratios but does not land exactly on the bound (see
`ref_random_policy` in `ref/gen_golden.py`), which is why the bound is driven deliberately here.

Held-out phenomena (PLAN section 4.1): none. This file measures a flag, not a behaviour.
"""

from __future__ import annotations

import pytest

SKIP_REASON = (
    "skeleton: T-B8 assertions are written by WO-002 (frozen tests); they bind WO-011 "
    "(Ledger.append, bound_binding, write_manifest) and WO-007 (process_reports)"
)
"""Reason attached to every `@pytest.mark.skip` below; the bodies arrive with WO-002."""

BOUND_FLAG = "BOUND_BINDING"
"""The flag string of CONTRACT rule 8; `gosplan.metrics.ledger.BOUND_BINDING_FLAG` must equal it.
Written here as a literal so the test still states the contract if the constant is renamed."""

AT_BOUND_THRESHOLD = 0.01
"""The 1% of CONTRACT rule 8; `gosplan.metrics.ledger.AT_BOUND_FLAG_THRESHOLD` must equal it. The
comparison is strict: *more than* 1% raises the flag."""

RHO_MAX = 10.0
"""`report_max_ratio` at `p1_default_config()` (PLAN section 2.3, TECH row of PLAN section 3). The
test asserts the configured bound equals this and that `action_spec` reports the same number: a
silently widened bound is the failure mode CONTRACT rule 8 exists to prevent."""

ABOVE_THRESHOLD_FRACTION = 0.05
"""Fraction of REPORT steps driven to the bound in the raising case - comfortably above
`AT_BOUND_THRESHOLD`, so the assertion does not turn on the rounding of one row."""

BELOW_THRESHOLD_FRACTION = 0.005
"""Fraction driven to the bound in the non-raising case - comfortably below the threshold."""

OVERSHOOT_RATIO = 25.0
"""A report ratio far above `RHO_MAX`, used to check that the action is clipped to the bound rather
than accepted, and that the clipped row still counts as `at_bound` (PLAN section 2.8)."""

TB8_SEEDS: tuple[int, ...] = (0, 1, 2)
"""Environment seeds, one episode each. A WO-002 test-design constant; the driving policy makes the
at-bound fraction deterministic, so the seeds vary only the environment's draws."""


@pytest.mark.skeleton
@pytest.mark.skip(reason=SKIP_REASON)
def test_reports_at_the_bound_are_recorded() -> None:
    """A report at `rho_max` is clipped, recorded as `at_bound`, and priced at the bound.

    Drive the test-local policy so that a known subset of REPORT steps reports `RHO_MAX` and a
    further subset reports `OVERSHOOT_RATIO`, at `p1_default_config()` over `TB8_SEEDS`, recording
    a `Ledger`. Assert, for those rows:

        row.report_ratio == RHO_MAX                      exactly, including the overshoot rows
        row.report       == RHO_MAX * row.target         to 1e-12
        row.at_bound     is True

    and for every other REPORT row, `row.at_bound is False`. Assert also that
    `action_spec(cfg)["report_ratio"] == ((N,), 0.0, cfg.tech.report_max_ratio)` and that
    `cfg.tech.report_max_ratio == RHO_MAX`.

    The overshoot rows are the substance of CONTRACT rule 8's "never silently widen": an action
    above the bound is clipped and *counted*, never honoured. Owning WO: **WO-002**; binds
    **WO-007** and **WO-011**.
    """
    raise NotImplementedError("PLAN section 11 (T-B8) - implemented in WO-002")


@pytest.mark.skeleton
@pytest.mark.skip(reason=SKIP_REASON)
def test_flag_is_raised_above_one_percent() -> None:
    """More than 1% of reports at the bound raises `BOUND_BINDING`, incrementally and in batch.

    Drive `ABOVE_THRESHOLD_FRACTION` of REPORT steps to `RHO_MAX` and assert:

        BOUND_FLAG in ledger.flags
        bound_binding(ledger) is True
        gosplan.metrics.ledger.BOUND_BINDING_FLAG == BOUND_FLAG
        gosplan.metrics.ledger.AT_BOUND_FLAG_THRESHOLD == AT_BOUND_THRESHOLD

    - the incremental flag from `Ledger.append` and the batch predicate `bound_binding` agreeing on
    the same ledger. Assert the fraction is computed over REPORT rows only: PRODUCE rows carry no
    report, and counting them would dilute the fraction by a factor of `M + 1` and hide a binding
    bound. Owning WO: **WO-002**; binds **WO-011**.
    """
    raise NotImplementedError("PLAN section 11 (T-B8) - implemented in WO-002")


@pytest.mark.skeleton
@pytest.mark.skip(reason=SKIP_REASON)
def test_flag_is_absent_below_one_percent() -> None:
    """At or below 1% the flag stays down - the threshold is strict, and it is not a hair trigger.

    Drive `BELOW_THRESHOLD_FRACTION` of REPORT steps to `RHO_MAX` and assert `BOUND_FLAG not in
    ledger.flags` and `bound_binding(ledger) is False`. Repeat with no at-bound reports at all
    (`TruthfulMyopic` at `p1_default_config()`) and assert the same, and that a ledger with no
    REPORT rows at all returns `False` rather than raising.

    Without this half, a ledger that raised the flag unconditionally would pass T-B8 - and the G2
    hygiene criterion of PLAN section 4.5 requires the flag to be *absent* in all Phase-1 gate runs,
    so a false positive would block the gate as loudly as a false negative. Owning WO: **WO-002**;
    binds **WO-011**.
    """
    raise NotImplementedError("PLAN section 11 (T-B8) - implemented in WO-002")


@pytest.mark.skeleton
@pytest.mark.skip(reason=SKIP_REASON)
def test_manifest_carries_the_flag() -> None:
    """The flag reaches `runs/<hash>/manifest.json`, where a reader of the result meets it.

    Write a manifest for the flag-raising run with `write_manifest(run_dir, cfg, extra)` into a
    `tmp_path` directory, then read the JSON back and assert `BOUND_FLAG` is in its `flags` field,
    that the file also carries the config hash, the full configuration, `SPEC_VERSION` and both
    seeds (CONTRACT rule 10), and that fields which do not apply to the run are present as `null`
    rather than omitted. Assert the non-flagged run's manifest has a `flags` field that exists and
    does not contain `BOUND_FLAG`.

    CONTRACT rule 8 requires the result to be *reported with the flag*: a flag that lives only in a
    ledger object nobody reads is not a report. Owning WO: **WO-002**; binds **WO-011**.
    """
    raise NotImplementedError("PLAN section 11 (T-B8) - implemented in WO-002")
