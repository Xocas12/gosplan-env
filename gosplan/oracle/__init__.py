"""Oracle package - the full-information planning benchmark of PLAN section 6.2.

Realises: PLAN section 6.2 (oracle) and PLAN section 2.9.4 (the headline metrics whose denominators
it supplies). Owning work order: **WO-027**. Phase status: **Phase 2** - nothing in Phase 1 imports
this package, where `welfare_ratio` uses `W_truthful_max` as a labelled placeholder denominator.

The package holds exactly one solver module, `gosplan.oracle.kantorovich`, and re-exports its
public surface so that `from gosplan.oracle import solve_oracle` and
`from gosplan.oracle.kantorovich import solve_oracle` name the same object. `solve_oracle` mirrors
the signature frozen in `spec/spec.py`, argument names included (CONTRACT rule 1).
"""

from __future__ import annotations

from gosplan.oracle.kantorovich import (
    ORACLE_HORIZON_PERIODS,
    ORACLE_RESULT_KEYS,
    SOLVER_CANDIDATES,
    solve_oracle,
)

__all__ = [
    "ORACLE_HORIZON_PERIODS",
    "ORACLE_RESULT_KEYS",
    "SOLVER_CANDIDATES",
    "solve_oracle",
]
