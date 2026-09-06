"""Metrics package - the ledger, the run manifest, and the pre-registered phenomena.

Realises: PLAN section 4 (pre-registration; the ledger of section 4 and the phenomena of section
4.1), PLAN section 7.3 (the `forensics_core` coupling and its vendored fallback) and CONTRACT rules
6, 8 and 10. Owning work orders: **WO-011** (ledger and manifest), **WO-016** (Phase-1 phenomena,
rows 1 and 4, and `_fallback.py`), **WO-030** (Phase-2 rows 2, 3, 5, 6, 7).

Contents:
  `gosplan.metrics.ledger`      `StepRecord`, `Ledger`, `write_manifest`, the `BOUND_BINDING`
                                helper and the manifest field list;
  `gosplan.metrics.phenomena`   one function per row of the PLAN section 4.1 table, plus the
                                pre-registered estimator settings of PLAN section 4.5 and the
                                measurement window of PLAN section 4.4;
  `gosplan.metrics._fallback`   the vendored `forensics_core` surface of PLAN section 7.3.

Rows **2, 5, 6 and 7** of PLAN section 4.1 (storming, hoarding, blat, hidden reserves) are **held
out**: they are re-exported here for interface completeness only, and no plot, table or test of
them may exist before the Phase-2 acceptance run (PLAN sections 4.1, 12.3). WO-012 and WO-016 are
forbidden from implementing them.

Nothing in this package is agent-facing. Its inputs are the true quantities the environment logs -
`welfare`, `val_true`, `val_measured` among them - and CONTRACT rule 6 keeps every one of them out
of observations, rewards and agent inputs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from gosplan.metrics._fallback import (
    FALLBACK_ESTIMATOR_VERSION,
    BunchingResult,
    DispersionResult,
    ReconResult,
)
from gosplan.metrics.ledger import (
    AT_BOUND_FLAG_THRESHOLD,
    BOUND_BINDING_FLAG,
    MANIFEST_FIELDS,
    Ledger,
    StepRecord,
    bound_binding,
    write_manifest,
)
from gosplan.metrics.phenomena import (
    phenomenon_blat,
    phenomenon_bunching,
    phenomenon_hidden_reserves,
    phenomenon_hoarding,
    phenomenon_padding,
    phenomenon_quality,
    phenomenon_storming,
)

FORENSICS_CORE_BACKEND: str = "forensics_core"
"""Value of `EstimatorBackend.name` when the sibling `forensics_core` package supplied the
estimators. Written to the manifest as `estimator_backend` (CONTRACT rule 10)."""

FALLBACK_BACKEND: str = "gosplan.metrics._fallback"
"""Value of `EstimatorBackend.name` when the vendored fallback supplied them (PLAN section 7.3).
Also written to the manifest, so a bunching number is always traceable to the code behind it."""


@dataclass(frozen=True)
class EstimatorBackend:
    """Which estimator implementation a run used, and how to call it (PLAN section 7.3).

    One immutable record, produced by `resolve_estimators()` and consumed by
    `gosplan/metrics/phenomena.py` and by `write_manifest`. It exists because the answer to "which
    code computed `b_hat`?" is a provenance fact that belongs in the manifest (CONTRACT rule 10),
    not an implementation detail of an import.

    `forensics_core` splits the three estimators across three submodules; the vendored fallback
    flattens them into one. The three callable fields below erase that difference: the argument
    lists are identical either way (PLAN section 7.3), only the namespace differs.
    """

    name: str
    """`FORENSICS_CORE_BACKEND` or `FALLBACK_BACKEND`; the manifest's `estimator_backend`."""

    version: str
    """`forensics_core.__version__`, or `FALLBACK_ESTIMATOR_VERSION` when the fallback was
    resolved; the manifest's `estimator_version`."""

    bunching_estimate: Callable[..., object]
    """`forensics_core.bunching.estimate` or `gosplan.metrics._fallback.estimate`. Called as
    `estimate(x, window_lo, window_hi, bin_width, degree, excl_lo, excl_hi)` and returns a
    `BunchingResult`-shaped object. Typed `Callable[..., object]` rather than with the fallback's
    own result class, because the two backends return structurally identical but distinct types."""

    reconciliation_ledger_test: Callable[..., object]
    """`forensics_core.reconciliation.ledger_test` or `gosplan.metrics._fallback.ledger_test`.
    Called as `ledger_test(reported_supply, received_inputs, io_matrix, prices)`; returns a
    `ReconResult`-shaped object."""

    dispersion_cross_section: Callable[..., object]
    """`forensics_core.dispersion.cross_section` or `gosplan.metrics._fallback.cross_section`.
    Called as `cross_section(values, groups)`; returns a `DispersionResult`-shaped object."""


def resolve_estimators() -> EstimatorBackend:
    """Resolve the estimator surface of PLAN section 7.3, preferring `forensics_core`.

    Takes: nothing. Returns: an `EstimatorBackend` whose three callables have exactly the signatures
    PLAN section 7.3 fixes, and whose `name` and `version` are the manifest's `estimator_backend`
    and `estimator_version` (CONTRACT rule 10).

    This is the **only** place in the repository that may import `forensics_core`, and it may do so
    only inside a `try/except ImportError` that falls back to the vendored surface. The resolution
    to implement is exactly:

        try:
            import forensics_core
            backend = EstimatorBackend(
                name=FORENSICS_CORE_BACKEND,
                version=forensics_core.__version__,
                bunching_estimate=forensics_core.bunching.estimate,
                reconciliation_ledger_test=forensics_core.reconciliation.ledger_test,
                dispersion_cross_section=forensics_core.dispersion.cross_section,
            )
        except ImportError:
            from gosplan.metrics import _fallback
            backend = EstimatorBackend(
                name=FALLBACK_BACKEND,
                version=FALLBACK_ESTIMATOR_VERSION,
                bunching_estimate=_fallback.estimate,
                reconciliation_ledger_test=_fallback.ledger_test,
                dispersion_cross_section=_fallback.cross_section,
            )
        return backend

    Consequences that the rest of the design leans on. The two surfaces are interchangeable by
    construction, so `gosplan/metrics/phenomena.py` is written once (PLAN section 7.3: "vendored
    fallback ... with identical signatures until then"). The fallback is not a silent substitute:
    every run records which backend it used, so a result computed before the G1 interface agreement
    can never be confused with one computed after it. The import is *not* at module scope, so
    importing `gosplan.metrics` never depends on a package installed from a sibling checkout.

    Binds: `tests/unit/test_phenomena_p1.py` (WO-016) - the fallback and `forensics_core` signatures
    agree, and the resolver returns the fallback with `FALLBACK_ESTIMATOR_VERSION` when
    `forensics_core` is absent. Owning WO: **WO-016**.
    """
    raise NotImplementedError("PLAN section 7.3 - implemented in WO-016")


__all__ = [
    "AT_BOUND_FLAG_THRESHOLD",
    "BOUND_BINDING_FLAG",
    "FALLBACK_BACKEND",
    "FALLBACK_ESTIMATOR_VERSION",
    "FORENSICS_CORE_BACKEND",
    "MANIFEST_FIELDS",
    "BunchingResult",
    "DispersionResult",
    "EstimatorBackend",
    "Ledger",
    "ReconResult",
    "StepRecord",
    "bound_binding",
    "phenomenon_blat",
    "phenomenon_bunching",
    "phenomenon_hidden_reserves",
    "phenomenon_hoarding",
    "phenomenon_padding",
    "phenomenon_quality",
    "phenomenon_storming",
    "resolve_estimators",
    "write_manifest",
]
