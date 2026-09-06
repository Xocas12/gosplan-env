"""Production: one PRODUCE step of the period schedule.

Realises: PLAN section 2.6 (intended output, the CES input-coverage aggregator, the log-normal
yield shock, the investment diversion, input consumption and the effort cost). Owning work order:
**WO-005** (Production; MID-strong, depends on WO-003 and WO-004).

Scope, stated as a prohibition (the WO-005 forbidden list). Nothing in this module may reference a
report, a target, a bonus, a penalty, an audit or a reward - not in code, not in a docstring, not
in an example. Production is physical: it turns effort and input stocks into output and cost, and
it knows nothing about what will later be claimed about that output. The separation is what makes
CONTRACT rule 7 checkable, because padding and shaving then have nowhere to live except in the
agent's policy.

Randomness: the yield shock is the only draw in this module, and it goes through
`gosplan.rng.draw` with purpose `"yield"` (CONTRACT rule 9). No `numpy.random` call may appear
anywhere under `gosplan/env/`, and no generator may be held at module level.

Dimensions: `N = cfg.supply.n_enterprises`, `J = cfg.supply.n_sectors`,
`M = cfg.incentive.steps_per_period`. `a = cfg.supply.io_matrix` is the *true* I-O matrix, indexed
`a[s(i)][j]` = units of good `j` per unit of good `s(i)`; the planner's possibly stale copy lives
in `State.planner_io` and is never read here.

Cross-module bindings. `State` and `EnterpriseAction` are the runtime dataclasses of
`gosplan/env/state.py` (WO-009) and `EnvConfig` that of `gosplan/config.py` (WO-003); each must
stay field-for-field identical to its `spec/spec.py` declaration, which is not importable as a
package. They are imported under `TYPE_CHECKING` so this module stays importable while its
siblings are still skeletons; the implementer promotes the ones it calls at runtime, and
`gosplan.rng.draw` is the one it will need.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # type-only: see the cross-module bindings note in the module docstring
    from gosplan.config import EnvConfig
    from gosplan.env.state import EnterpriseAction, State

Array = np.ndarray
"""Alias for every numeric array in this module (PLAN section 10), identical to `spec.Array`."""


def coverage(X: Array, need: Array, weights: Array, theta: float) -> Array:
    """CES aggregator over per-good input coverage ratios (PLAN section 2.6).

    Takes: `X` `(N, J)`, input stocks on hand at the start of this step; `need` `(N, J)`, the input
    need of this step, `need_ikj = a_{s(i)j} * y_hat_ik`; `weights` `(N, J)`, the coverage weights
    `omega_j = a_{s(i)j} / sum_j a_{s(i)j}` of the producing sector; `theta`, the complementarity
    exponent `cfg.supply.input_complementarity` (Phase 1: 8.0; grid {2, 8, inf}). Returns: `H`
    `(N,)`, the coverage multiplier, in [0, 1].

    Formula (PLAN section 2.6, verbatim):

        H_ik = ( sum_j omega_j * min(1, X_ij / need_ikj)**(-theta) )**(-1/theta)
        H_ik = 1                        if enterprise i requires no inputs

    Read it as a CES aggregate of the per-good coverage ratios `min(1, X_ij / need_ikj)`, each
    capped at 1 so that a surplus of one input can never compensate a shortage of another beyond
    what `theta` allows. Larger `theta` is more complementary; `theta = inf` is Leontief.

    Edge cases the implementation must handle explicitly, and that T-U7 checks:

      * `need_ikj == 0` contributes a coverage ratio of exactly 1, never a division by zero. This
        covers both a zero I-O coefficient and a zero intended output (`e_ik = 0`).
      * a whole row of `a` equal to zero - the enterprise requires no inputs - gives `H_ik = 1`.
        There is no such row at the Phase-1 `io_matrix`, but the branch must exist and be tested.
      * `theta = float("inf")` takes the `min` branch: `H_ik = min_j min(1, X_ij / need_ikj)` over
        the goods with positive need, computed with `np.min` rather than by evaluating the power.
      * `X_ij / need_ikj == 0` with `theta` finite sends one term to infinity and `H` to 0; the
        implementation must return 0 there rather than a NaN.

    `weights` is the caller's responsibility, not this function's: `GosplanEnv.__init__` (WO-009)
    precomputes `omega_j = a_{s(i)j} / sum_j a_{s(i)j}` once per run, because `a` is fixed within a
    run in Phase 1 (`cfg.supply.tech_drift_sigma = 0.0`). The weights of a row of `a` that is
    entirely zero are undefined; that row takes the `H = 1` branch above and its weights are never
    read. `weights` is expected to sum to 1 along `j` for every row with a positive sum of `a`.

    Binds: test T-U7 in `tests/unit/test_production.py` (PLAN section 11) - `theta = inf` equals
    `min`; `theta -> 1` equals the weighted harmonic mean `1 / sum_j (omega_j / r_ij)`; `H = 1`
    when no inputs are needed. Owning WO: **WO-005**.
    """
    raise NotImplementedError("PLAN section 2.6 - implemented in WO-005")


def produce_step(
    state: State, action: EnterpriseAction, cfg: EnvConfig
) -> tuple[State, Array, Array]:
    """Execute one PRODUCE step `k` for all enterprises (PLAN section 2.6).

    Takes: `state` at the start of step `k = state.k_step`, with `t = state.t_period`; `action`,
    whose `effort` dimension is read (and, in Phase 2, `quality` and `invest`); `cfg`. Returns:
    `(state, y_ik, c_ik)` - the updated state, the output realised this step `(N,)`, and the cost
    incurred this step `(N,)`.

    Formulas (PLAN section 2.6, verbatim; `e` is `action.effort` clipped to [0, 1], `v` is
    `action.invest`, `q` is `action.quality`):

        y_hat_ik   = (A_{s(i)} * cap_i / M) * e_ik          # intended output at full coverage
        need_ikj   = a_{s(i)j} * y_hat_ik                   # for each good j with a_{s(i)j} > 0
        H_ik       = coverage(X, need, omega, theta)
        eps_ik     ~ LogNormal(-sigma_{s(i)}**2 / 2, sigma_{s(i)})    mean 1
                     key = (seed_env, "yield", t, k, i)
        y_tilde_ik = y_hat_ik * H_ik * eps_ik
        y_ik       = y_tilde_ik * (1 - v_ik)                # v = invest fraction; Phase 1 v == 0
        X_ij      -= min(X_ij, a_{s(i)j} * y_tilde_ik)      # inputs consumed, capped at stock
        c_ik       = kappa * e_ik**2 + F * 1[e_ik > 0] + kappa_q * q_ik * e_ik  # P1: F=kappa_q=0

    Symbols: `A_j = cfg.supply.productivity[j]`; `cap_i = state.capital[i]` (Phase 1: 1.0);
    `M = cfg.incentive.steps_per_period`, so a full-effort period produces `A * cap` in total;
    `a = cfg.supply.io_matrix`; `omega_j = a_{s(i)j} / sum_j a_{s(i)j}`;
    `theta = cfg.supply.input_complementarity`; `sigma_j = cfg.supply.yield_sigma[j]`;
    `kappa = cfg.incentive.effort_cost`; `F = cfg.supply.setup_cost`;
    `kappa_q = cfg.supply.quality_cost`.

    Two details of the formulas above are load-bearing and are tested separately. First, inputs are
    consumed against `y_tilde` - output *before* the investment diversion - so diverting output to
    capital does not economise on inputs. Second, the consumption is capped at the stock on hand,
    so `X` can never go negative even when `H` and the yield shock disagree about how much was
    producible; the cap is also what keeps the conservation identity exact.

    State fields written: `inv_inputs` loses the consumed inputs; `cum_output` accumulates `y_ik`;
    `cum_cost` accumulates `c_ik`. `quality_acc` accumulates the Phase-2 period-average quality
    `qbar_i` and stays untouched in Phase 1, where `q == 1`; its accumulation rule is frozen at the
    Phase-2 spec revision (WO-021). `pending_invest` receives `y_tilde_ik * v_ik` in Phase 2; its
    column convention and its maturing rule are frozen at the Phase-2 spec revision (WO-021), and
    in Phase 1 `v == 0` leaves the buffer at zero. No other field of `State` may be written here -
    in particular not `inv_output`: own-good stock receives the period's accumulated output only at
    the close of the period, under PLAN section 2.8 (`gosplan/env/reporting.py`, WO-007).

    Randomness (CONTRACT rule 9, PLAN section 2.15). The yield shock is drawn through
    `gosplan.rng.draw` with purpose `"yield"` and indices `(t, k)`, vectorised over the trailing
    enterprise index via `shape=(N,)`, so the key of a single enterprise's draw is
    `(seed_env, "yield", t, k, i)` exactly as PLAN section 2.6 states. `sigma` is per *sector* and
    `draw` takes scalar distribution parameters, so a single vectorised `lognormal` call cannot
    carry `N` different sigmas: the draw must be issued once per sector, with that sector's
    `sigma_j` and `mean_log = -sigma_j**2 / 2`, at the same key and `shape=(N,)`, each enterprise
    then taking entry `i` of the array drawn for its own sector. That keeps the key, the mean of 1
    and the order-independence of T-U6 intact. The convention must agree with `ref/ref_step.py` to
    1e-9 (T-B7), so WO-002 fixes it in the reference and WO-005 follows the reference.

    Phase-2 toggles that route through this function, every one of them off at
    `p1_default_config()`:

        setup cost            F > 0 adds a fixed cost per step with positive effort
        increasing returns    A_j(Kap) = A_j * (Kap / Kap_0)**alpha_irs   (alpha_irs = 0 in P1)
        capital               Kap_{t+1} = (1 - dep) * Kap_t + matured investment, lag invest_lag
        technology drift      a <- a * exp(zeta), zeta ~ N(0, sigma_drift**2) per period, purpose
                              "drift", while `State.planner_io` stays fixed - which is what makes
                              the planner's I-O estimate go stale
        quality routing       X_ij credited as `deliv * qbar_j` when `quality_matters` is True

    Delivery timing (`cfg.supply.delivery_timing`, PLAN section 2.6) decides *when* within the
    period delivered inputs become available to this function, and the deliveries themselves are
    computed by `deliver` in `gosplan/env/planner.py`:

        uniform      Phase 1 - everything delivered this period is available at step k = 0, so
                     `X` is constant across the period apart from consumption
        stochastic   each delivered unit arrives at step k ~ Categorical(cfg.supply.arrival_probs)
                     drawn with purpose "arrival"; the locked Phase-2 value is uniform-random
                     arrival, so no backloading is mechanical and any effort-Gini excess is
                     behavioural (PLAN section 4.2)
        backloaded   arrival_probs concentrated on late steps

    Binds: `tests/unit/test_production.py` (the WO-005 must-pass list) - T-U7 on `coverage`; the
    yield shock has mean 1 to 1e-3 over 1e5 draws; the `v` diversion; the cost formula; inputs
    consumed equal `a * y_tilde` capped at stock; `H = 1` when the enterprise's row of `a` is zero
    - and test T-U1, the per-period conservation identity, to 1e-9 per good. Owning WO: **WO-005**.
    """
    raise NotImplementedError("PLAN section 2.6 - implemented in WO-005")
