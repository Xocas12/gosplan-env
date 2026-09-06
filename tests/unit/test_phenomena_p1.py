"""Phase-1 metrics: the bunching estimator and fictitious padding (PLAN section 4.1 rows 1 and 4).

Realises: PLAN sections 4.1 (rows 1 and 4 only - the two Phase-1 pipeline checks), 4.4 (measurement
window), 4.5 (pre-registered estimator settings and the G2 criteria), 7.3 (the coupling to
`forensics_core` and its vendored fallback) and PLAN section 11 (test architecture, unit/property
category). Owning work order: **WO-002** (frozen tests; LEAD). Binds the WO-016 must-pass line of
PLAN section 12.3, verbatim - "`tests/unit/test_phenomena_p1.py` (estimator on synthetic densities
with known excess mass recovers it within 5%; hole mass; SE by bootstrap; fallback and
`forensics_core` signatures identical)". Modules under test: `gosplan/metrics/phenomena.py`,
`gosplan/metrics/_fallback.py`, `gosplan/metrics/__init__.py`.

Pre-registered estimator settings (PLAN section 4.5, verbatim, and the constants of
`gosplan/metrics/phenomena.py`): bins of width 0.005 on `rho` in [0.6, 1.4]; excluded window
[0.95, 1.02]; polynomial of degree 7 fitted outside the window; excess mass
`b_hat = (observed - counterfactual mass in [1.00, 1.02]) / mean counterfactual density in the
window`; hole mass computed identically on [0.95, 1.00); standard error by bootstrap **over seeds**.
Measurement window (PLAN section 4.4): periods `t >= 2`; no end-of-episode exclusion under geometric
termination; reports at `rho_max` included in the histogram and flagged.

HELD OUT. Rows 2 (storming), 5 (hoarding), 6 (blat) and 7 (hidden reserves) of PLAN section 4.1 are
Phase-2 emergence claims: WO-016 is forbidden from implementing them, and no test here computes,
plots or asserts any quantity belonging to them. Rows 1 and 4 are **pipeline checks**, not evidence
for Claim A: if bunching fails to appear the optimiser is broken.

FROZEN BY CONTRACT RULE 2. SKELETON: every test is `@pytest.mark.skeleton` and skipped until WO-016
lands; each docstring states the exact assertion, formula and tolerance.
"""

from __future__ import annotations

import pytest


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-016")
def test_estimator_recovers_a_known_excess_mass_within_five_percent() -> None:
    """The bunching estimator recovers a planted excess mass to within 5%.

    Assertion: draw a large synthetic sample of `rho` from a smooth density on [0.6, 1.4] (a broad
    normal or lognormal, no notch), then move a known fraction of the mass from just below 1 into
    [1.00, 1.02] so that the true excess mass `b_true` - in the units PLAN section 4.5 defines,
    (observed minus counterfactual mass in the excess window) divided by the mean counterfactual
    density in that window - is known by construction. `estimate(x, window_lo=0.6, window_hi=1.4,
    bin_width=0.005, degree=7, excl_lo=0.95, excl_hi=1.02).excess_mass` must satisfy
    `|b_hat - b_true| / b_true <= 0.05`, at several planted values spanning the range gate G2
    cares about.

    First bullet of the WO-016 must-pass list; tolerance 5% is PLAN section 12.3's, verbatim.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-016")
def test_estimator_recovers_a_known_hole_mass() -> None:
    """The hole below target is recovered on the same synthetic densities.

    Assertion: on the samples of the previous test, `hole_mass` - the same computation on the hole
    window [0.95, 1.00) - recovers the mass removed from just below target to within 5%, and is
    approximately the negative counterpart of `excess_mass` when the planted mass was moved rather
    than added (the two need not cancel exactly, because the windows differ in width). On a sample
    with no hole, `hole_mass` is indistinguishable from 0 given its bootstrap standard error.

    The excess and hole windows are the caller's pre-registration constants
    (`BUNCHING_EXCESS_LO/HI`, `BUNCHING_HOLE_LO/HI`); they are not arguments of the frozen
    `estimate` signature and must not be inferred from `excl_lo`/`excl_hi`.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-016")
def test_estimator_finds_no_excess_on_a_smooth_density() -> None:
    """On an unperturbed smooth density the confidence interval covers zero.

    Assertion: with no planted mass, `excess_mass` is within one bootstrap standard error of 0 and
    the interval `[ci_lo, ci_hi]` contains 0, on repeated draws (in at least 90% of them, matching
    the criterion the gate applies). This is the null the smooth counterfactual arm of gate G2
    criterion 2 is judged against - "at the smooth counterfactual, the CI for `b_hat` covers 0 in
    >= 90% of seeds" (PLAN section 4.5) - so an estimator biased away from 0 would manufacture a
    pass or a fail out of nothing.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-016")
def test_standard_error_is_a_bootstrap_over_seeds() -> None:
    """The resampling unit is the seed, never the individual report.

    Assertion: given a sample grouped by seed, the reported `se` matches the standard deviation of
    the per-bootstrap-replicate `excess_mass` values obtained by resampling *seeds* with
    replacement, to within Monte-Carlo error; resampling individual reports instead gives a
    materially smaller `se` on the same data, and the estimator must not do that. `ci_lo` and
    `ci_hi` bracket `excess_mass` and widen as the number of seeds falls.

    Third bullet of the WO-016 must-pass list. Seeds are the independent replicates of PLAN section
    4.3; treating reports as independent would understate every interval in the paper.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-016")
def test_preregistered_settings_are_the_defaults_and_are_recorded(p1_cfg) -> None:
    """`phenomenon_bunching` uses the PLAN section 4.5 constants and writes them to the manifest.

    Assertion: the constants in `gosplan/metrics/phenomena.py` equal the pre-registration exactly -
    `BUNCHING_BIN_WIDTH = 0.005`, `BUNCHING_WINDOW_LO = 0.6`, `BUNCHING_WINDOW_HI = 1.4`,
    `BUNCHING_EXCL_LO = 0.95`, `BUNCHING_EXCL_HI = 1.02`, `BUNCHING_POLY_DEGREE = 7`,
    `BUNCHING_EXCESS_LO = 1.00`, `BUNCHING_EXCESS_HI = 1.02`, `BUNCHING_HOLE_LO = 0.95`,
    `BUNCHING_HOLE_HI = 1.00` - `phenomenon_bunching` passes them to the estimator without
    modification, and the same values appear in the manifest's `bunching_settings` field (CONTRACT
    rule 10). Settings are pre-registered: they are defaults in code, not choices made after seeing
    a histogram.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-016")
def test_measurement_window_keeps_only_periods_from_two_onward(p1_cfg) -> None:
    """Only periods `t >= 2` enter, with no end-of-episode exclusion (PLAN section 4.4).

    Assertion: `phenomenon_bunching(ledger, cfg)` uses exactly the rows with
    `t_period >= MEASUREMENT_FIRST_PERIOD = 2` - a ledger whose early periods carry an extreme
    report distribution gives the same answer as one whose early periods are absent - and
    `MEASUREMENT_EXCLUDE_EPISODE_END is False`, because under geometric termination there is no
    end-game to exclude (PLAN section 2.12, finding F4). `n_obs` in the result equals the number of
    reports that survived the window.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-016")
def test_at_bound_reports_are_included_in_the_histogram_and_flagged(p1_cfg) -> None:
    """Reports at `rho_max` are counted, not dropped (PLAN section 4.4, CONTRACT rule 8).

    Assertion: `MEASUREMENT_INCLUDE_AT_BOUND is True`; a ledger containing at-bound reports yields
    an `at_bound_frac` equal to their fraction of the measured reports, to 1e-12; and those reports
    are present in the histogram inputs even though `rho_max = 10` lies outside the [0.6, 1.4]
    fitting window, so the reported `n_obs` and `at_bound_frac` describe the same sample. Dropping
    them would hide exactly the failure CONTRACT rule 8 exists to surface.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-016")
def test_fallback_and_forensics_core_expose_identical_signatures() -> None:
    """The two estimator backends are interchangeable, argument for argument (PLAN section 7.3).

    Assertion: `gosplan.metrics.resolve_estimators()` - the ONLY place in the repository that may
    import `forensics_core`, and only inside a `try/except ImportError` that falls back to
    `gosplan.metrics._fallback` - returns an `EstimatorBackend` whose `bunching_estimate` accepts
    exactly `(x, window_lo, window_hi, bin_width, degree, excl_lo, excl_hi)` in that order, whose
    `reconciliation_ledger_test` accepts `(reported_supply, received_inputs, io_matrix, prices)`,
    and whose `dispersion_cross_section` accepts `(values, groups)`; the parameter names and order
    are compared by introspection against the vendored fallback's, and must match. When
    `forensics_core` is importable, both backends run on the same synthetic sample and their
    `excess_mass` values agree to within the estimator's own tolerance; when it is not, the test
    asserts the fallback resolved and `name == FALLBACK_BACKEND`.

    Fourth bullet of the WO-016 must-pass list. `backend.name` and `backend.version` are written to
    the manifest (`estimator_backend`, `estimator_version`, CONTRACT rule 10), so a `b_hat` is
    always traceable to the code that produced it.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-016")
def test_phenomenon_bunching_returns_the_documented_keys(p1_cfg) -> None:
    """`phenomenon_bunching(ledger, cfg)` returns the mapping PLAN section 4.1 row 1 requires.

    Assertion: the returned mapping carries at least `excess_mass`, `hole_mass`, `se`, `ci_lo`,
    `ci_hi`, `n_obs` and `at_bound_frac`; every value is finite; `ci_lo <= excess_mass <= ci_hi`;
    `n_obs` is a positive integer; and `at_bound_frac` lies in [0, 1]. These are the fields gate G2
    criterion 2 reads - `b_hat >= 0.5 * b_hat_dp` with the bootstrap CI excluding 0 in >= 90% of 30
    seeds at the notched arm (PLAN section 4.5) - so a missing key is a blocked gate.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-016")
def test_phenomenon_padding_reports_padding_and_the_padding_index(p1_cfg) -> None:
    """`phenomenon_padding` computes row 4 of PLAN section 4.1 over the measurement window.

    Assertion: the returned mapping carries at least `padding` - `mean_i max(0, R_i - S_i) / T_i`
    over periods `t >= 2`, matching a direct recomputation from the ledger to 1e-12 - together with
    `padding_index = val_measured / val_true` (PLAN section 2.9.4), which is exactly 1 on a
    truthful ledger and above 1 whenever a claim exceeds true output. Both are pipeline checks, not
    evidence for Claim A.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-016")
def test_padding_elasticity_is_reported_against_the_audit_penalty_product(p1_cfg) -> None:
    """The elasticity of padding is measured in `audit_rate * penalty_scale`, the G2 quantity.

    Assertion: given ledgers from the three `a * pen` levels gate G1 recorded in
    `runs/G1_decision.md`, `phenomenon_padding` reports the elasticity of fictitious padding with
    respect to `audit_rate * penalty_scale` across those levels, computed from the three points and
    not from a fitted curve of its own; the sign convention is such that gate G2 criterion 3 -
    "learned fictitious padding is monotone decreasing in `a * pen` and within 0.03 of the DP's
    value at each of the three levels" (PLAN section 4.5) - can be evaluated directly from the
    returned mapping. The test asserts the arithmetic and the reported comparison against the DP's
    values, never that the learned direction is the predicted one: that is a result of the gate
    run, not an assumption of the metric.
    """
    assert False
