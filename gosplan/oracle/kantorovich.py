"""Full-information planning benchmark - the oracle of PLAN section 6.2.

Realises: PLAN section 6.2 (non-anticipative expected-value MIP and the clairvoyant per-seed
bound), PLAN section 2.9.4 (`W_oracle` is the denominator of `welfare_ratio` and `val_oracle` the
denominator inside `specification_gap`), and CONTRACT rule 10 (the solver's version and optimality
gap are manifest fields). Owning work order: **WO-027** (LEAD formulates the MIP, MID-strong
implements it against the chosen solver; parity test against brute force at `N = 2`).

Phase status: **Phase-2 sketch.** The signature below is frozen now so that no type moves later
(PLAN section 0, finding F14); the formulation is frozen at the Phase-2 spec revision. In Phase 1
`welfare_ratio` uses `W_truthful_max` as a clearly labelled placeholder denominator and every table
that reports it says so (PLAN section 2.9.4) - nothing in Phase 1 calls this module.

Binding to `spec/spec.py`: `spec/spec.py` is not importable as a package, so this module *mirrors*
its `solve_oracle` signature rather than importing it. The two must stay identical, argument names
included; `tests/unit/test_spec_imports.py` (WO-001) enforces that the public surface of PLAN
section 10 is present and unchanged.

Dependencies: the MIP is built and solved through an open-source solver (HiGHS or CBC, via OR-Tools
or Pyomo - the lead verifies availability; `ortools` is the `solver` extra in `pyproject.toml`).
That import belongs inside `solve_oracle`, not at module scope, so the skeleton stays importable in
an environment with no solver installed.

CONTRACT: the oracle is a *reporting* benchmark. Its welfare is logged and compared against, and it
never enters an observation, a reward term or any agent input (rule 6); it implements no pathology
and prescribes no agent behaviour (rule 7).
"""

from __future__ import annotations

from typing import Optional

from gosplan.config import EnvConfig

ORACLE_HORIZON_PERIODS: int = 12
"""Planning horizon of the expected-value MIP, in plan periods (PLAN section 6.2). Passed as the
`horizon` argument by every caller in this repository; it is a pre-registered design constant, not
a tuning knob, and a run that uses a different horizon says so in its manifest."""

SOLVER_CANDIDATES: tuple[str, ...] = ("HiGHS", "CBC")
"""The open-source MIP solvers PLAN section 6.2 admits, in preference order. Reached through
OR-Tools or Pyomo; the lead verifies availability before WO-027 is issued. The solver actually used
and its version are returned in the result mapping and written to the manifest (CONTRACT
rule 10)."""

ORACLE_RESULT_KEYS: tuple[str, ...] = (
    "welfare",
    "val",
    "optimality_gap",
    "solver",
    "solver_version",
    "status",
)
"""Keys every `solve_oracle` result must carry (`spec/spec.py`, PLAN section 6.2):

    welfare         W_oracle - mean per-period CES welfare of the solution (PLAN section 2.9.3),
                    the denominator of `welfare_ratio` in PLAN section 2.9.4
    val             val_oracle - the plan-price output aggregate of the solution, the denominator
                    of the first term of `specification_gap` (PLAN section 2.9.4)
    optimality_gap  the solver's relative MIP gap at termination; a manifest field (CONTRACT
                    rule 10), never rounded away and never suppressed when the solve stops early
    solver          which member of `SOLVER_CANDIDATES` was used
    solver_version  its version string, verbatim from the solver
    status          the solver's termination status (optimal / feasible / time limit / infeasible)

A caller may add keys; it may never omit one of these. A run whose oracle terminated at a non-zero
gap reports the gap next to every number derived from it."""


def solve_oracle(
    cfg: EnvConfig,
    horizon: int,
    clairvoyant: bool,
    seed_env: Optional[int],  # spelling mirrors `spec/spec.py` verbatim (CONTRACT rule 1)
) -> dict:
    """Solve the full-information planning benchmark for one configuration (PLAN section 6.2).

    Takes:
      `cfg`, the configuration whose environment is being benchmarked - the MIP is built from its
        SUPPLY block (I-O matrix, productivities, capacities, final-demand shares, CES weights and
        elasticity) and from the nonconvexities it switches on;
      `horizon`, the planning horizon in periods; PLAN section 6.2 fixes it at 12, i.e.
        `ORACLE_HORIZON_PERIODS`, which is what every caller in this repository passes;
      `clairvoyant`, selecting the per-seed bound that knows the realised yield noise instead of the
        non-anticipative expected-value solution;
      `seed_env`, required when `clairvoyant` is True (it identifies the noise path the bound is
        allowed to see) and ignored otherwise.

    Returns: a mapping carrying at least the keys of `ORACLE_RESULT_KEYS` - `welfare`, `val`,
    `optimality_gap`, `solver`, `solver_version`, `status`.

    What must be built (PLAN section 6.2, to be frozen at the Phase-2 spec revision):

      * a **non-anticipative expected-value MIP** over the *full true state* - the planner-side
        information filters of PLAN section 2.7.5 do not apply to the oracle; it is the benchmark
        of what the technology could deliver, not of what the planning system could learn;
      * **mean yields**: the multiplicative yield shock of PLAN section 2.6 is replaced by its mean
        of 1, which is exactly what makes the solution non-anticipative;
      * a **12-period planning horizon** (`ORACLE_HORIZON_PERIODS`), with the period schedule,
        production function, input consumption, inventory dynamics (holding loss `h`, cap `S_max`)
        and final-demand routing `phi_j` of PLAN sections 2.5-2.11 as constraints;
      * the **nonconvexities as configured**: a positive `supply.setup_cost` becomes a binary per
        enterprise-step (`F` charged iff `e > 0`); a positive `supply.irs_alpha` becomes a
        piecewise-linear approximation of `A_j(Kap) = A_j * (Kap / Kap_0)**alpha_irs`, whose
        breakpoints are recorded with the result because they are part of the formulation;
      * the objective is welfare: the CES index of PLAN section 2.9.3 evaluated on the consumer
        sink's receipts and averaged over the measured periods of PLAN section 4.4 (`t >= 2`).
        Whether the CES objective is linearised or handed to the solver's nonlinear support is a
        formulation decision taken at the Phase-2 spec revision and recorded there;
      * solved **once per configuration** with an open-source solver - HiGHS or CBC through
        OR-Tools or Pyomo (`SOLVER_CANDIDATES`); the import lives inside this function so the
        skeleton imports without a solver present.

    The **clairvoyant** branch (`clairvoyant=True`) solves the same programme per seed with the
    realised yield draws of `seed_env` substituted for their means. It is reported **as an upper
    bound only**: `W_oracle` in PLAN section 2.9.4 is the expected-value MIP's welfare, and the
    clairvoyant number is never used as the denominator of `welfare_ratio` or `specification_gap`.
    A table that shows both says which is which on the same line.

    Manifest (CONTRACT rule 10): `solver`, `solver_version` and `optimality_gap` from the returned
    mapping are written into `runs/<hash>/manifest.json` by `gosplan.metrics.ledger.write_manifest`
    - see its `MANIFEST_FIELDS`. A non-zero gap is carried through to every derived headline number
    rather than being cleared by re-running with a looser tolerance.

    Acceptance criteria (WO-027): a parity test against brute-force enumeration at `N = 2` over a
    short horizon agrees to solver tolerance; the expected-value solution is weakly dominated by the
    clairvoyant bound on every seed; the returned mapping carries every key of
    `ORACLE_RESULT_KEYS`. No Phase-1 test binds this function - it is not called before the Phase-2
    acceptance run (PLAN section 13, gate G3, "oracle gap recorded").

    Owning WO: **WO-027**.
    """
    raise NotImplementedError("PLAN section 6.2 - implemented in WO-027")
