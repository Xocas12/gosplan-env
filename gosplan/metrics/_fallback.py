"""Vendored estimator surface - the `forensics_core` interface of PLAN section 7.3.

Realises: PLAN section 7.3 (the interface gosplan consumes from the sibling `forensic-stats`
programme, "to agree with the `forensics_core` owner at G1; vendored fallback in
`gosplan/metrics/_fallback.py` with identical signatures until then") and PLAN section 7.2 (the
estimator-bias study these estimators are the payback for). Owning work order: **WO-016**.

Why it exists. The repository must type-check, import and run its Phase-1 metrics before the
`forensics_core` owner has agreed the interface at gate G1 and before the package is installed
(`pyproject.toml` marks it as the `forensics` extra, installed from a sibling checkout, not from
PyPI). This module supplies the three functions with **exactly** the signatures PLAN section 7.3
lists, so `gosplan/metrics/phenomena.py` is written once against one surface:

    forensics_core.bunching.estimate(x, window_lo, window_hi, bin_width, degree, excl_lo, excl_hi)
    forensics_core.reconciliation.ledger_test(reported_supply, received_inputs, io_matrix, prices)
    forensics_core.dispersion.cross_section(values, groups)

`forensics_core` lives in three submodules; the fallback flattens them into this one module, and
`gosplan.metrics.resolve_estimators()` binds `bunching.estimate` -> `estimate`,
`reconciliation.ledger_test` -> `ledger_test`, `dispersion.cross_section` -> `cross_section`. The
argument lists never differ; only the namespace does.

**Scope. What is OUT (PLAN section 7.3, explicitly):**
  * **digit tests** - Benford-style and terminal-digit detectors. RL policies have no digit
    preferences, so a digit test here would measure the sampler, not the behaviour (finding F12);
  * **any claim of calibration of archival detectors** - the coupling reports bias, RMSE, CI
    coverage and power curves of these estimators *on simulated data with known ground truth*. It
    does not license a statement about their behaviour on Soviet archival series;
  * **any use of Phase 1 output for the archival anchor** - the republic-level anchor uses only the
    ministry-level aggregated series generated in Phase 2 with an active ministry layer.
What is IN: bunching-bias curves (PLAN section 7.2), reconciliation power curves, dispersion-test
behaviour under known cross-sectional distortion, and those Phase-2 ministry-level series.

Result-type status. The three **function signatures** are frozen by PLAN section 7.3 and must not
move. The **fields of the three result dataclasses** are the part of the interface still to be
agreed with the `forensics_core` owner at G1: the fields below are the minimum that
`gosplan/metrics/phenomena.py` reads (rows 1, 5 and 7 of the PLAN section 4.1 table), and a field
added by that agreement is additive. Once the real package is installed its result types are used
verbatim; the ones here exist so the fallback is a drop-in for them. `tests/unit/
test_phenomena_p1.py` (WO-016) asserts that the fallback and `forensics_core` signatures agree.

Version: `FALLBACK_ESTIMATOR_VERSION` is written into the run manifest as `estimator_version`
alongside `estimator_backend` whenever the fallback is the resolved backend (CONTRACT rule 10), so
no bunching number is ever reported without saying which code produced it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray
"""Alias for every numeric array in this surface (PLAN section 10), restated from `spec/spec.py`.
No function here may rely on numpy-only methods in a signature: the JAX port substitutes its own
array type behind the same name."""

FALLBACK_ESTIMATOR_VERSION: str = "0.1.0+fallback"
"""Version of this vendored surface, written to the manifest as `estimator_version` when
`gosplan.metrics.resolve_estimators()` resolves to the fallback (CONTRACT rule 10). It is bumped
whenever anything here changes - a signature, a result field or an estimator's arithmetic - because
a bunching number computed under one version is not comparable with one computed under another. It
is deliberately *not* equal to any `forensics_core` version, and the `+fallback` local segment makes
that visible at a glance in a manifest."""


@dataclass(frozen=True)
class BunchingResult:
    """Return type of `estimate` - the bunching estimator of PLAN sections 4.1 (row 1) and 4.5.

    Fields provisional until the G1 interface agreement (see the module docstring); these are the
    ones `gosplan.metrics.phenomena.phenomenon_bunching` reads. All masses are in the units PLAN
    section 4.5 defines: mass relative to the mean counterfactual density in the window, so
    `excess_mass` is directly comparable with the DP's `b_hat_DP` at gate G2.
    """

    excess_mass: float
    """`b_hat` = (observed - counterfactual mass in the excess window) / (mean counterfactual
    density in that window), PLAN section 4.5."""

    hole_mass: float
    """The same quantity computed on the hole window `[0.95, 1.00)` - the missing mass just below
    target, reported alongside the excess (PLAN section 4.1 row 1)."""

    se: float
    """Standard error of `excess_mass`, by bootstrap over seeds (PLAN section 4.5); the resampling
    unit is the seed, never the individual report."""

    ci_lo: float
    """Lower end of the bootstrap confidence interval for `excess_mass`. Gate G2 criterion 2 asks
    whether this interval excludes 0 (notched arm) or covers it (smooth arm)."""

    ci_hi: float
    """Upper end of the same interval."""

    n_obs: int
    """Number of observations entering the histogram, after the measurement window of PLAN section
    4.4 has been applied by the caller."""

    bin_edges: Array
    """(n_bins + 1,) histogram bin edges actually used, so the fit is reproducible from the result
    alone."""

    observed_counts: Array
    """(n_bins,) observed counts per bin."""

    counterfactual_counts: Array
    """(n_bins,) counts implied by the degree-`degree` polynomial fitted outside the excluded
    window - the counterfactual the excess and hole masses are measured against."""


@dataclass(frozen=True)
class ReconResult:
    """Return type of `ledger_test` - the reconciliation estimator of PLAN sections 4.1 (row 7) and
    7.3.

    Fields provisional until the G1 interface agreement; these are the ones
    `gosplan.metrics.phenomena.phenomenon_hidden_reserves` reads.
    """

    statistic: float
    """Test statistic for the hypothesis that claimed supply reconciles with buyers' received
    inputs through the I-O matrix at plan prices."""

    p_value: float
    """Its p-value under that hypothesis. The estimator's **power curve** as a function of known
    fictitious output is a reported result of PLAN section 7.2; its calibration on archival data is
    explicitly out of scope."""

    residuals: Array
    """(N,) per-enterprise reconciliation residual in plan-price units - claimed supply minus the
    supply implied by what buyers received."""

    n_obs: int
    """Number of enterprise-periods entering the test."""


@dataclass(frozen=True)
class DispersionResult:
    """Return type of `cross_section` - the dispersion estimator of PLAN sections 4.1 (row 5) and
    7.3.

    Fields provisional until the G1 interface agreement; these are the ones
    `gosplan.metrics.phenomena.phenomenon_hoarding` reads for the cross-sectional part.
    """

    statistic: float
    """Test statistic for cross-sectional distortion of `values` across `groups`."""

    p_value: float
    """Its p-value. Behaviour under *known* cross-sectional distortion is what PLAN section 7.3
    puts in scope; a calibration claim about archival series is not."""

    group_stats: Array
    """(n_groups,) the per-group summary the statistic aggregates, in the order the group labels
    first appear in `groups`."""

    n_groups: int
    """Number of distinct groups."""

    n_obs: int
    """Number of values entering the test."""


def estimate(
    x: Array,
    window_lo: float,
    window_hi: float,
    bin_width: float,
    degree: int,
    excl_lo: float,
    excl_hi: float,
) -> BunchingResult:
    """Bunching estimator - `forensics_core.bunching.estimate` (PLAN sections 7.3, 4.5).

    Signature frozen by PLAN section 7.3: the argument names and order above are the interface and
    must not move.

    Takes: `x`, the sample of report ratios `rho` (one entry per report inside the measurement
    window of PLAN section 4.4, at-bound reports included); `window_lo`, `window_hi`, the range the
    histogram covers (pre-registered [0.6, 1.4]); `bin_width` (0.005); `degree`, the degree of the
    counterfactual polynomial (7); `excl_lo`, `excl_hi`, the excluded window the polynomial is
    fitted *without* ([0.95, 1.02]). Returns: a `BunchingResult`.

    What it must compute (PLAN section 4.5, verbatim):
      1. bin `x` on `[window_lo, window_hi]` with bins of width `bin_width`;
      2. fit a polynomial of degree `degree` to the bin counts **outside** `[excl_lo, excl_hi]`;
      3. `excess_mass` = (observed - counterfactual mass on the excess window) / (mean
         counterfactual density on that window);
      4. `hole_mass` = the same computation on the hole window, which lies below target;
      5. `se`, `ci_lo`, `ci_hi` by bootstrap **over seeds**, the caller supplying `x` grouped so
         that the seed is the resampling unit.
    The excess and hole windows are the caller's pre-registration constants
    (`gosplan/metrics/phenomena.py`: `BUNCHING_EXCESS_LO/HI`, `BUNCHING_HOLE_LO/HI`); they are not
    arguments of this frozen signature and must not be inferred from `excl_lo`/`excl_hi`.

    Reference implementation notes for WO-016: `numpy.polynomial.polynomial.Polynomial.fit` on the
    retained bins; no `scipy` is required (`scipy` is a project dependency but is not imported at
    module scope in a skeleton file).

    Binds: `tests/unit/test_phenomena_p1.py` - on a synthetic density with known excess mass the
    estimator recovers it within 5%, recovers the hole mass likewise, produces a bootstrap SE, and
    has a signature identical to `forensics_core.bunching.estimate`. Owning WO: **WO-016**.
    """
    raise NotImplementedError("PLAN section 7.3 - implemented in WO-016")


def ledger_test(
    reported_supply: Array,
    received_inputs: Array,
    io_matrix: Array,
    prices: Array,
) -> ReconResult:
    """Reconciliation estimator - `forensics_core.reconciliation.ledger_test` (PLAN section 7.3).

    Signature frozen by PLAN section 7.3: argument names and order are the interface.

    Takes: `reported_supply` (N,), claimed supply per enterprise in units of its own good (the
    ledger's `report` column); `received_inputs` (N, J), what each enterprise physically received
    (the ledger's `deliv` columns); `io_matrix` (J, J), the technical coefficients `a[j][k]` used to
    convert receipts into the supply they imply; `prices` (J,), the plan prices that make the two
    sides commensurate. Returns: a `ReconResult`.

    What it must compute: the discrepancy between claimed supply of each good and the supply implied
    by buyers' receipts of it, valued at plan prices, as a per-enterprise residual, plus a test
    statistic and p-value for the hypothesis that the ledger reconciles. Under truthful reporting
    the residuals are the delivery mechanics of PLAN section 2.7.3 alone; fictitious output shows up
    as claimed supply that no buyer ever received.

    Used by PLAN section 4.1 row 7 (hidden reserves, **held out** until the Phase-2 acceptance run)
    and by the power curve of PLAN section 7.2, which is computed on simulated ledgers with known
    fictitious output. Calibration against archival series is out of scope (PLAN section 7.3).

    Binds: `tests/unit/test_phenomena_p1.py` - signature identity with
    `forensics_core.reconciliation.ledger_test`. No Phase-1 behavioural test binds it, because row 7
    is held out. Owning WO: **WO-016** (surface), **WO-030** (its first use).
    """
    raise NotImplementedError("PLAN section 7.3 - implemented in WO-016")


def cross_section(values: Array, groups: Array) -> DispersionResult:
    """Dispersion estimator - `forensics_core.dispersion.cross_section` (PLAN section 7.3).

    Signature frozen by PLAN section 7.3: argument names and order are the interface.

    Takes: `values` (n,), the cross-sectional quantity under test - for PLAN section 4.1 row 5 the
    request-inflation ratios `q_ij / need_ij`; `groups` (n,), an integer group label per value, the
    sector `s(i)` in that use. Returns: a `DispersionResult` carrying the statistic, its p-value,
    the per-group summaries and the counts.

    What it must compute: a test of whether dispersion of `values` across `groups` departs from the
    no-distortion null, with the per-group summaries reported so the direction is inspectable.
    Behaviour under **known** cross-sectional distortion is what PLAN section 7.3 puts in scope; no
    claim about archival detector calibration follows from it.

    Binds: `tests/unit/test_phenomena_p1.py` - signature identity with
    `forensics_core.dispersion.cross_section`. Its consumer (row 5, hoarding) is **held out** until
    the Phase-2 acceptance run. Owning WO: **WO-016** (surface), **WO-030** (its first use).
    """
    raise NotImplementedError("PLAN section 7.3 - implemented in WO-016")
