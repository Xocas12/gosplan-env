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
    import numpy as np
    from ortools.linear_solver import pywraplp

    from gosplan.env.prices import initial_prices
    from gosplan.env.state import initial_targets
    from gosplan.rng import draw

    if clairvoyant and seed_env is None:
        raise ValueError("solve_oracle: clairvoyant=True needs seed_env")
    sup, inc, tech = cfg.supply, cfg.incentive, cfg.tech
    n, j_n, h_n = sup.n_enterprises, sup.n_sectors, int(horizon)
    sector = np.asarray(sup.sector_of, dtype=int)
    a = np.asarray(sup.io_matrix, dtype=float)
    prod = np.asarray(sup.productivity, dtype=float)
    phi = np.asarray(sup.final_demand_share, dtype=float)
    cap, s_max = 1.0, float(tech.inventory_cap_mult) * 1.0
    h, h_x, m = float(sup.holding_loss), float(sup.input_holding_loss), inc.steps_per_period
    t0 = np.asarray(initial_targets(cfg), dtype=float)
    prices = np.asarray(initial_prices(cfg), dtype=float)

    # Period capacity A * cap * eps_bar: eps_bar = 1 (expected value) or the realised period mean
    # of the M per-step yield draws at the environment's own keys (clairvoyant bound).
    eps = np.ones((h_n, n))
    if clairvoyant:
        for t in range(h_n):
            for i in range(n):
                sigma = float(sup.yield_sigma[sector[i]])
                eps[t, i] = np.mean(
                    [
                        draw(
                            int(seed_env),
                            "yield",
                            t,
                            k,
                            i,
                            shape=(1,),
                            dist="lognormal",
                            mean_log=-(sigma**2) / 2.0,
                            sigma=sigma,
                        )[0]
                        for k in range(m)
                    ]
                )

    solver = pywraplp.Solver.CreateSolver("HIGHS")
    if solver is None:
        raise RuntimeError("solve_oracle: HiGHS is not available through OR-Tools")
    solver.SuppressOutput()
    import ortools

    solver_version = f"HiGHS via OR-Tools {ortools.__version__}"
    inf = solver.infinity()
    y = [
        [solver.NumVar(0, prod[sector[i]] * cap * eps[t, i], f"y{t}_{i}") for i in range(n)]
        for t in range(h_n)
    ]
    ship = [[solver.NumVar(0, inf, f"ship{t}_{i}") for i in range(n)] for t in range(h_n)]
    stock = [[solver.NumVar(0, s_max, f"S{t}_{i}") for i in range(n)] for t in range(h_n)]
    route = [
        [[solver.NumVar(0, inf, f"r{t}_{b}_{g}") for g in range(j_n)] for b in range(n)]
        for t in range(h_n)
    ]
    x_end = [
        [[solver.NumVar(0, inf, f"X{t}_{b}_{g}") for g in range(j_n)] for b in range(n)]
        for t in range(h_n)
    ]
    cons = [[solver.NumVar(0, inf, f"c{t}_{g}") for g in range(j_n)] for t in range(h_n)]
    w = [solver.NumVar(0, inf, f"w{t}") for t in range(h_n)]

    for t in range(h_n):
        for g in range(j_n):
            shipped = sum(ship[t][i] for i in range(n) if sector[i] == g)
            solver.Add(cons[t][g] == phi[g] * shipped)
            solver.Add(sum(route[t][b][g] for b in range(n)) == (1.0 - phi[g]) * shipped)
        for i in range(n):
            s_prev = stock[t - 1][i] if t > 0 else 0.0
            solver.Add(ship[t][i] <= s_prev)
            solver.Add(stock[t][i] <= (1.0 - h) * (s_prev - ship[t][i]) + y[t][i])
            for g in range(j_n):
                x_prev = x_end[t - 1][i][g] if t > 0 else a[sector[i], g] * t0[i]
                x_in = x_prev + route[t][i][g]
                use = a[sector[i], g] * y[t][i]
                if a[sector[i], g] > 0:
                    solver.Add(use <= x_in)  # Leontief coverage (formulation note F2)
                solver.Add(x_end[t][i][g] <= (1.0 - h_x) * (x_in - use))

    measured = [t for t in range(h_n) if t >= _MEASURE_FROM]
    solver.Maximize(sum(w[t] for t in measured) * (1.0 / max(len(measured), 1)))

    # Tangent planes of the homogeneous concave CES index through the origin: f(c) <= grad f(cbar).c
    # for every cbar. Start from a fixed spread of directions, then add the tangent at each period's
    # solution until the planned welfare matches the true CES value (cutting planes, F3).
    def add_cut(t: int, cbar: np.ndarray) -> None:
        grad = _ces_gradient(cbar, cfg)
        solver.Add(w[t] <= sum(float(grad[g]) * cons[t][g] for g in range(j_n)))

    rng = np.random.default_rng(0)
    seeds = [np.ones(j_n)] + [rng.dirichlet(np.ones(j_n)) * j_n for _ in range(_INITIAL_CUTS - 1)]
    for t in range(h_n):
        for cbar in seeds:
            add_cut(t, cbar)
    status = pywraplp.Solver.NOT_SOLVED
    for _ in range(_MAX_CUT_ROUNDS):
        status = solver.Solve()
        if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            break
        c_now = np.array([[cons[t][g].solution_value() for g in range(j_n)] for t in range(h_n)])
        w_now = np.array([w[t].solution_value() for t in range(h_n)])
        f_now = np.array([_ces(c_now[t], cfg) for t in range(h_n)])
        if np.all(
            w_now[measured] - f_now[measured] <= _CUT_TOL * np.maximum(f_now[measured], 1e-9)
        ):
            break
        for t in measured:
            if w_now[t] - f_now[t] > _CUT_TOL * max(f_now[t], 1e-9):
                add_cut(t, np.maximum(c_now[t], 1e-9))
    status_name = {
        pywraplp.Solver.OPTIMAL: "optimal",
        pywraplp.Solver.FEASIBLE: "feasible",
        pywraplp.Solver.INFEASIBLE: "infeasible",
        pywraplp.Solver.UNBOUNDED: "unbounded",
    }.get(status, "not solved")
    if status_name not in ("optimal", "feasible"):
        return {
            "welfare": float("nan"),
            "val": float("nan"),
            "optimality_gap": float("nan"),
            "solver": "HiGHS",
            "solver_version": solver_version,
            "status": status_name,
        }
    c_sol = np.array([[cons[t][g].solution_value() for g in range(j_n)] for t in range(h_n)])
    y_sol = np.array([[y[t][i].solution_value() for i in range(n)] for t in range(h_n)])
    welfare = float(np.mean([_ces(c_sol[t], cfg) for t in measured]))
    upper = float(solver.Objective().Value())
    val = float(np.mean([np.dot(prices[sector], y_sol[t]) for t in measured]))
    return {
        "welfare": welfare,
        "val": val,
        "optimality_gap": max(0.0, (upper - welfare) / welfare) if welfare > 0 else float("nan"),
        "solver": "HiGHS",
        "solver_version": solver_version,
        "status": status_name,
        "upper_bound": upper,
        "horizon": h_n,
        "clairvoyant": bool(clairvoyant),
        "formulation": FORMULATION,
    }


_MEASURE_FROM = 2
"""Welfare is averaged over periods `t >= 2` (PLAN section 4.4)."""

_INITIAL_CUTS = 40
"""Tangent directions seeded per period before cutting-plane refinement."""

_MAX_CUT_ROUNDS = 60
"""Cutting-plane rounds (each adds the tangent at the current solution where it is loose)."""

_CUT_TOL = 1e-7
"""Relative tolerance between the planned welfare and the true CES value of the solution."""

FORMULATION = (
    "F1 welfare-only objective: effort and setup costs are not in the CES welfare of PLAN 2.9.3, "
    "so the programme is an LP (no binaries; setup cost is irrelevant to welfare and irs is out of "
    "Phase-2 scope, P2 revision R1). F2 Leontief coverage replaces CES input coverage (theta = 8 "
    "is near-Leontief; Leontief is the conservative side). F3 the CES welfare index is concave and "
    "homogeneous of degree 1, so it is represented exactly in the limit by tangent planes through "
    "the origin, refined by cutting planes until planned and true welfare agree to 1e-7; "
    "optimality_gap is (LP bound - true welfare) / true welfare. F4 timing as the environment: "
    "output made in period t is stored, shipped at t+1's DELIVER, a share phi_j to consumers and "
    "the rest routed freely to buyers' input stocks; holding losses h (output, after shipping) and "
    "h_X (inputs) apply; stock capped at S_max with free disposal; opening inputs a * T_0, no "
    "opening stock. F5 expected value: period yield multiplier 1; clairvoyant: the realised mean "
    "of the environment's own per-step yield draws."
)
"""The formulation decisions of the Phase-2 spec revision for WO-027, recorded with every result."""


def _ces(c, cfg: EnvConfig) -> float:
    """The CES welfare index of PLAN section 2.9.3 (Cobb-Douglas branch at sigma_c = 1)."""
    import numpy as np

    alpha = np.asarray(cfg.supply.ces_alpha, dtype=float)
    sigma = float(cfg.supply.ces_sigma)
    c = np.maximum(np.asarray(c, dtype=float), 0.0)
    if abs(sigma - 1.0) < 1e-12:
        return float(np.prod(c**alpha))
    rho = (sigma - 1.0) / sigma
    if rho < 0 and np.any(c <= 0):
        return 0.0
    return float(np.sum(alpha * c**rho) ** (1.0 / rho))


def _ces_gradient(c, cfg: EnvConfig):
    """Gradient of the CES index at `c > 0` (homogeneous of degree 1, so f(c) = grad f(c) . c)."""
    import numpy as np

    alpha = np.asarray(cfg.supply.ces_alpha, dtype=float)
    sigma = float(cfg.supply.ces_sigma)
    c = np.maximum(np.asarray(c, dtype=float), 1e-9)
    if abs(sigma - 1.0) < 1e-12:
        return alpha * float(np.prod(c**alpha)) / c
    rho = (sigma - 1.0) / sigma
    total = float(np.sum(alpha * c**rho))
    return alpha * c ** (rho - 1.0) * total ** (1.0 / rho - 1.0)
