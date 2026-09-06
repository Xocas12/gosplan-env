"""Saltelli/Sobol sensitivity design (OPTIONAL) - PLAN sections 4.3, 12.5 (WO-033), 13 and 14.

Realises: the optional Saltelli paragraph of PLAN section 4.3. Owning work order: **WO-033**
(MID-strong, Phase 3, marked *optional* in PLAN section 12.5). Gate: **G4**, as supporting material
only - PLAN section 13 lists contrasts, estimator-bias curves, the LLM study and price sensitivity
as the G4 conditions; the Sobol design is not among them. **This experiment never blocks a gate and
never substitutes for a contrast**: total-order indices rank parameters, they do not identify the
`Delta_X` of PLAN section 4.3.

What it produces. A Saltelli design over the swept INFO+INC parameters at `N_BASE = 100`, giving
`N_BASE * (N_FACTORS + 2) = 1500` runs for first- and total-order indices, or
`N_BASE * (2 * N_FACTORS + 2) = 2800` runs when second-order indices are also estimated - the
"approximately 1,500-2,800 runs" of PLAN section 4.3, which is what pins `N_FACTORS` at 13. Reported
quantity: **total-order indices**.

    THE REPORTING RULE (PLAN section 4.3, verbatim in substance). Total-order indices are reported
    **with the PLAN section 3 sweep ranges stated as an assumption in the same table**. A Sobol
    index is a statement about variance over an assumed input distribution: change the ranges and
    the ranking changes. `RANGE_ASSUMPTION_NOTE` is the sentence that must appear in the table
    caption, and the full range table is printed beside the indices, not referenced elsewhere.

Inputs
    The post-G3 Phase-2 full configuration as the design centre; the swept INFO+INC parameters with
    their PLAN section 3 ranges (see the ambiguity below); the JAX path (WO-029), which PLAN section
    14 makes a hard requirement for this design; `gosplan.metrics` for the outcomes.

Outputs
    `runs/sobol/table.parquet`  total-order indices per parameter and outcome, with their bootstrap
                                intervals, the assumed range for every parameter, `N_BASE`, and the
                                sampler and its version
    `runs/sobol/report.md`      the index table with `RANGE_ASSUMPTION_NOTE` in the caption, the
                                design size actually run, and the convergence diagnostics
    `runs/<config-hash>/`       per-run directories with `manifest.json` (CONTRACT rule 10)

    PLAN section 12.5 names no artefact paths for WO-033; these follow the `runs/<experiment>/`
    convention of the Phase-1 cards.

Cost (PLAN section 14): 1,500-2,800 runs, **JAX only**, about 1-3 GPU-days. If the JAX port is not
available, this experiment is not run - it is not re-scoped onto the NumPy path.

OPEN - AMBIGUITY FOR THE WO-033 SESSION (CONTRACT rule 3; do not silently choose)
    PLAN section 4.3 says "the 13 swept INFO+INC parameters" but does not enumerate them, and the
    PLAN section 3 registry lists more than 13 INFO and INC rows with sweep ranges (before counting
    the categorical rows `aggregation_level`, `audit_mode`, `objective_metric`, `penalty_form`,
    `penalty_arg`, and the fixed rows `ratchet_cap_up`, `ratchet_cap_dn`, `alloc_eta_need`). The
    count 13 is confirmed by arithmetic - 100 * (13 + 2) = 1500 and 100 * (2 * 13 + 2) = 2800 - but
    the membership is not. File an AMBIGUITY REPORT and let the lead fix the list at issue time; the
    list actually used is recorded in the manifest and printed in the table beside the ranges.

Runtime bindings. `EnvConfig` is `gosplan.config.EnvConfig` (field-for-field identical to
`spec/spec.py`, enforced by a unit test). The Saltelli sampler (`SALib` or an equivalent) and `jax`
are imported inside the function that uses them, never at module scope; the sampler's name and
version go in the manifest.
"""

from __future__ import annotations

from pathlib import Path

from gosplan.config import EnvConfig

N_BASE = 100
"""`N_base` of the Saltelli design (PLAN section 4.3)."""

N_FACTORS = 13
"""The number of swept INFO+INC parameters in the design (PLAN section 4.3). The value is not a
choice: PLAN's "approximately 1,500-2,800 runs" at `N_base = 100` is exactly
`N_base * (d + 2) = 1500` and `N_base * (2d + 2) = 2800` at `d = 13`. *Which* 13 parameters is the
open ambiguity in the module docstring."""

RUNS_FIRST_AND_TOTAL_ORDER = 1500
"""Design size when only first- and total-order indices are estimated:
`N_BASE * (N_FACTORS + 2)` (PLAN section 4.3)."""

RUNS_WITH_SECOND_ORDER = 2800
"""Design size when second-order indices are also estimated:
`N_BASE * (2 * N_FACTORS + 2)` (PLAN section 4.3). The upper end of PLAN's run-count range."""

REPORTED_INDEX = "total_order"
"""The index PLAN section 4.3 asks for: total-order (`S_T`). First-order indices may be tabulated
alongside since the design produces them, but the reported quantity - the one any conclusion may
lean on - is the total order."""

RANGE_ASSUMPTION_NOTE = (
    "Total-order indices are conditional on the input distribution assumed by the design: each "
    "parameter is sampled over its PLAN section 3 sweep range, printed in this table. Indices are "
    "not transportable to other ranges, and a ranking obtained here is not a causal effect - the "
    "causal statements of this study are the named contrasts of PLAN section 4.3 with their CIs."
)
"""The assumption sentence PLAN section 4.3 requires in the same table as the indices. It is
declared here as data so the table caption and the report cite one text, and so a reviewer can check
that it was printed rather than paraphrased."""

OUTCOMES: tuple[str, ...] = ("welfare_ratio", "padding_index", "specification_gap")
"""The three headline outcomes of PLAN section 2.9.4, the same set the contrasts report, so indices
and Deltas are read on identical quantities."""

REQUIRES_JAX = True
"""PLAN section 14 marks this design "JAX only". If `gosplan.jax` (WO-029) is unavailable the
experiment does not run: it is optional, and re-scoping it onto the NumPy path would cost days of
CPU for a result no gate depends on."""

OPTIONAL = True
"""PLAN section 12.5 marks WO-033 optional, and PLAN section 13 does not list Sobol indices among
the gate G4 conditions. Recorded as data so a report generator can place this section as supporting
material rather than as a result."""

OUT_DIR = Path("runs/sobol")
"""Artefact directory, relative to the repository root; a WO-033 convention."""

TABLE_PATH = OUT_DIR / "table.parquet"
"""Total-order indices per parameter and outcome, with the assumed range for every parameter."""

REPORT_PATH = OUT_DIR / "report.md"
"""The index table with `RANGE_ASSUMPTION_NOTE` in its caption."""


def run(
    cfg: EnvConfig,
    factors: tuple[str, ...],
    ranges: dict[str, tuple[float, float]],
    out_dir: Path = OUT_DIR,
    n_base: int = N_BASE,
    second_order: bool = False,
    seed_env: int | None = None,
) -> dict[str, object]:
    """Run the Saltelli design and write the total-order index table.

    Takes: `cfg`, the design centre - the post-G3 Phase-2 full configuration, already validated;
    `factors`, the `N_FACTORS` swept INFO+INC parameter names the lead fixed (see the module
    docstring's ambiguity); `ranges`, each factor's PLAN section 3 sweep range, passed in explicitly
    so the assumption is data the caller supplies and the table prints, never a default hidden in
    this module; `out_dir`, where the table and report are written; `n_base`, `N_base` of the
    design; `second_order`, whether to estimate second-order indices as well
    (`RUNS_WITH_SECOND_ORDER` instead of `RUNS_FIRST_AND_TOTAL_ORDER` runs); `seed_env`, the root
    environment seed - `None` means `cfg.tech.seed_env`, shared across design points so the
    environment draws are common random numbers (PLAN section 2.15).

    Returns: a mapping with at least

        "factors"          tuple[str, ...], the parameters swept, in design order
        "ranges"           dict[str, tuple[float, float]], the assumed range per factor
        "n_runs"           int, design points actually run
        "total_order"      dict[str, dict[str, dict[str, float]]], outcome -> factor ->
                           {"index", "ci_lo", "ci_hi"}
        "first_order"      dict, the same shape, tabulated alongside but never reported alone
        "sampler"          str, and "sampler_version" str, recorded in the manifest
        "range_assumption" str, `RANGE_ASSUMPTION_NOTE`, carried into the table caption
        "artefacts"        dict[str, str], the paths written

    Procedure (PLAN section 4.3, optional paragraph):

      1. Check `len(factors) == N_FACTORS` and that every factor has a range in `ranges`; a mismatch
         is an error, not a silently truncated design.
      2. Generate the Saltelli sample over `ranges` at `n_base`, giving `RUNS_FIRST_AND_TOTAL_ORDER`
         or `RUNS_WITH_SECOND_ORDER` design points.
      3. Build one validated `EnvConfig` per design point by overriding the sampled fields, and
         evaluate the `OUTCOMES` on the JAX path (WO-029), writing `runs/<hash>/manifest.json` per
         run (CONTRACT rule 10).
      4. Estimate total-order indices (and first-order alongside) per outcome, with bootstrap
         intervals and the sampler's convergence diagnostics.
      5. Write `table.parquet` and `report.md`, printing `RANGE_ASSUMPTION_NOTE` in the caption and
         the full `ranges` table beside the indices.

    Nothing here is a treatment effect: an index ranks variance contributions under an assumed input
    distribution. Any causal sentence in the final report cites a named contrast from
    `gosplan.experiments.contrasts` with its CI (PLAN section 4.3).

    Binds: no gate condition - PLAN section 13 does not list Sobol indices among the G4 conditions.
    The design size is bound by `RUNS_FIRST_AND_TOTAL_ORDER` / `RUNS_WITH_SECOND_ORDER`.

    Realises: PLAN sections 4.3, 12.5 (WO-033), 14. Owning WO: **WO-033**.
    """
    raise NotImplementedError("PLAN section 4.3 (WO-033) - implemented in WO-033")


def main() -> int:
    """Entry point: run the optional Saltelli design and write the index table.

    Takes: nothing; the design centre is the post-G3 Phase-2 full configuration, the factor list and
    its ranges are the lead's record from the ambiguity resolution, and `n_base` is `N_BASE`. Any
    command-line surface and any JAX device setup is built inside this function.

    Returns: a process exit code - 0 when the design ran and `runs/sobol/report.md` was written, 1
    when it could not run, including the case where the JAX path of WO-029 is unavailable
    (`REQUIRES_JAX`). Because this experiment is `OPTIONAL`, a non-zero exit blocks nothing: PLAN
    section 13 does not list Sobol indices among the gate G4 conditions.

    Realises: PLAN sections 4.3, 12.5 (WO-033). Owning WO: **WO-033**.
    """
    raise NotImplementedError("PLAN section 4.3 (WO-033) - implemented in WO-033")


if __name__ == "__main__":
    raise SystemExit(main())
