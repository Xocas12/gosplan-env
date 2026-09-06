"""Plan prices: the cost-plus fixed point, and the price-vector sensitivity perturbation.

Realises: PLAN section 2.10 (plan prices and final demand), plus the standing robustness check of
PLAN sections 2.9.4 and 7.5. Owning work order: **WO-007** (reporting and reward; MID-strong),
with the sensitivity harness itself in `gosplan/experiments/price_sensitivity.py` (WO-036).

Prices are a *measurement* instrument in this design, not a market. Nothing an agent does moves
them, and no agent observes them: they enter only the planner-side aggregate `val_measured`, the
true aggregate `val_true` (PLAN section 2.9.3), the `net_output` fulfilment measure (PLAN section
2.9.2) and the Phase-2 trade surplus (PLAN section 2.13). Because every use is a ratio, the
overall scale of `p` cancels - which is why `kappa_labour` below is a normalisation rather than a
configuration parameter.

**Where the final-demand constants live.** They are configuration, never literals in this module:

    alpha_j   `cfg.supply.ces_alpha`          consumer CES weights; Phase 1 `1 / J` for all j
    sigma_c   `cfg.supply.ces_sigma`          consumer CES elasticity; Phase 1 0.8
    phi_j     `cfg.supply.final_demand_share` share of shipped good j routed to the consumer sink;
                                              Phase 1 0.5 for all j
    a_jk      `cfg.supply.io_matrix`          true I-O coefficients; the planner's possibly stale
                                              copy is `State.planner_io`
    m         `cfg.supply.price_markup`       cost-plus markup; Phase 1 0.1

The functions that consume them live elsewhere and are named here only so a reader knows this
module is not their home: the CES welfare index of PLAN section 2.9.3 is `welfare_true` in
`gosplan/env/reward.py`, and the consumer receipts it aggregates are produced by `deliver` in
`gosplan/env/planner.py` (PLAN section 2.7.3). `alpha_j` and `sigma_c` are SUPPLY parameters that
are swept only in the price-sensitivity check of PLAN section 7.5, never in a treatment arm.

Dimensions: `J = cfg.supply.n_sectors`. One good per sector, so a price vector is `(J,)`.

Cross-module bindings. `EnvConfig` is the runtime dataclass of `gosplan/config.py` (WO-003) and
must stay field-for-field identical to its `spec/spec.py` declaration, which is not importable as
a package. It is imported under `TYPE_CHECKING` so this module stays importable while its siblings
are still skeletons.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # type-only: see the cross-module bindings note in the module docstring
    from gosplan.config import EnvConfig

Array = np.ndarray
"""Alias for every numeric array in this module (PLAN section 10), identical to `spec.Array`."""

LABOUR_COST: float = 1.0
"""`kappa_labour`, the per-unit labour cost in the cost-plus fixed point of PLAN section 2.10.

It is deliberately not a configuration field. `p` is homogeneous of degree 1 in it, and every use
of `p` in this design is a ratio (`val_measured / val_true`, `p_j / p_{s(i)}` in the `net_output`
measure), so the scale cancels; PLAN section 2.10 fixes it at 1.0 and asks that the choice be
recorded as a normalisation. It is named here so that no function body carries it as a bare
literal, and so that a reader who changes it knows they are changing a normalisation and not an
economic assumption."""

PRICE_PERTURBATION_SIGMA: float = 0.3
"""Log-sd of the price perturbation `p_j * exp(u_j)`, `u_j ~ N(0, 0.3**2)` (PLAN sections 2.9.4,
7.5). Fixed by the pre-registration: it is the size of the price mis-specification every headline
table must survive, so it is not a tuning knob and is never varied to change a result."""

N_PRICE_PERTURBATIONS: int = 3
"""Number of perturbed price vectors every headline table is recomputed under (PLAN sections
2.9.4, 7.5): `padding_index`, `welfare_ratio` and `specification_gap` are reported at the baseline
prices and under these three. A sign change in `specification_gap` across them is reported, never
suppressed."""


def initial_prices(cfg: EnvConfig) -> Array:
    """Solve the cost-plus plan-price fixed point at `t = 0` (PLAN section 2.10).

    Takes: `cfg`. Returns: `p` `(J,)`, strictly positive.

    Formula (PLAN section 2.10, verbatim):

        p_j = (1 + m) * (kappa_labour + sum_k a_jk * p_k)

    solved as a fixed point over `j`, with `m = cfg.supply.price_markup` (Phase 1: 0.1),
    `a = cfg.supply.io_matrix` (rows = producing sector, columns = input good) and
    `kappa_labour = LABOUR_COST`. In matrix form the fixed point is
    `p = (1 + m) * (kappa_labour * 1 + a @ p)`, whose solution is
    `p = (1 + m) * kappa_labour * (I - (1 + m) * a)**-1 @ 1`; iterating the map from any positive
    starting vector converges to the same answer and is equally acceptable, since the test asks
    only for convergence and positivity.

    Convergence requires every row of `a` to satisfy `(1 + m) * sum_k a_jk < 1`, which is exactly
    why `EnvConfig.validate` rejects `sum_k a_jk >= 1` (WO-003). At the Phase-1 `io_matrix` every
    row sums to 0.4, so `(1 + m) * 0.4 = 0.44 < 1` and the spectral radius is comfortably inside
    the unit circle. A configuration that fails the condition must raise rather than return a
    diverged vector.

    Phase 1 holds prices fixed after `t = 0`: `cfg.supply.price_lag = float("inf")`, so
    `State.plan_prices` is written once by `initial_state` and never rewritten, and no agent can
    move a price. Phase 2 recomputes every `price_lag` periods from the planner's stale
    `State.planner_io` and never from the true `a` - see `recompute_prices`.

    The price vector is not a modelling nuisance to be tuned. PLAN sections 2.9.4 and 7.5 require
    every headline table to be recomputed under the perturbed vectors of `perturbed_price_vectors`,
    and a sign change in `specification_gap` is reported rather than suppressed.

    Binds: `tests/unit/test_prices.py` (the WO-007 must-pass list) - the fixed point converges and
    every price is strictly positive. Owning WO: **WO-007**.
    """
    raise NotImplementedError("PLAN section 2.10 - implemented in WO-007")


def recompute_prices(planner_io: Array, cfg: EnvConfig) -> Array:
    """Re-solve the cost-plus fixed point from the planner's I-O estimate (PLAN section 2.10).

    **Phase 2.** Inert in Phase 1, where `cfg.supply.price_lag = float("inf")` means prices are
    fixed after `t = 0` and this function is never called.

    Takes: `planner_io` `(J, J)`, the planner's possibly stale copy of `a` (`State.planner_io`),
    and `cfg`. Returns: `p` `(J,)`, strictly positive - the same fixed point as `initial_prices`
    with `planner_io` in place of `cfg.supply.io_matrix`:

        p_j = (1 + m) * (kappa_labour + sum_k planner_io[j][k] * p_k)

    Using the planner's estimate rather than the true `a` is the point of the function: once
    `cfg.supply.tech_drift_sigma > 0` moves the true coefficients while `State.planner_io` stays
    fixed (PLAN section 2.6), plan prices are computed from a picture of the economy that is out of
    date, and the gap between `val_measured` and `val_true` widens for a reason that has nothing to
    do with anybody's report. Reading `cfg.supply.io_matrix` here would destroy that channel.

    The *schedule* - recomputation every `cfg.supply.price_lag` periods - is applied by the step
    machine (`gosplan/env/step.py`, WO-009), not here; this function is the solver alone, so it
    stays a pure function of `(planner_io, cfg)`. The exact trigger predicate and the treatment of
    a non-integer `price_lag` are frozen at the Phase-2 spec revision (PLAN section 0).

    Binds: `tests/unit/test_prices.py` - agrees with `initial_prices` to 1e-12 when `planner_io`
    equals `cfg.supply.io_matrix`, which is the Phase-1 state at every `t`. Owning WO: **WO-007**
    (solver), Phase-2 activation with WO-022/WO-023.
    """
    raise NotImplementedError("PLAN section 2.10 - implemented in WO-007")


def perturbed_price_vectors(prices: Array, seeds: tuple[int, ...]) -> tuple[Array, ...]:
    """Return the perturbed price vectors of the standing robustness check (PLAN section 7.5).

    Takes: `prices` `(J,)`, the baseline plan prices from `initial_prices`; `seeds`, the fixed
    integer seeds of the perturbation, one per vector, of length `N_PRICE_PERTURBATIONS`. Returns:
    a tuple of `len(seeds)` arrays, each `(J,)` and strictly positive.

    Formula (PLAN sections 2.9.4, 7.5, verbatim):

        p_j^(r) = p_j * exp(u_j^(r)),   u^(r) ~ N(0, PRICE_PERTURBATION_SIGMA**2),  r = 1 .. 3

    one independent `u` per seed, each of length `J`. The seeds are fixed and passed in rather
    than defaulted, so that the same three vectors are reused across every arm, every seed of the
    learning run and every headline table: the check asks whether a *result* survives a price
    mis-specification, which it can only answer if the mis-specification itself is held constant.
    The chosen triple is recorded in the run manifest (CONTRACT rule 10) and never re-rolled to
    change a table; re-rolling it is the exact move CONTRACT rule 8's spirit forbids.

    Use: `gosplan/experiments/price_sensitivity.py` (WO-036) recomputes `padding_index`,
    `welfare_ratio` and `specification_gap` (PLAN section 2.9.4) under each returned vector and
    reports them beside the baseline. A sign change in `specification_gap` is reported, not
    suppressed (PLAN section 7.5). The perturbation is post-hoc: it re-values a ledger that has
    already been produced, so it never enters an episode, an observation or a reward, and
    `State.plan_prices` is not touched by it.

    Randomness - an open item for the v1 freeze (WO-013). CONTRACT rule 9 requires every draw made
    under `gosplan/env/` to go through `gosplan.rng.draw(seed_env, purpose, *indices)`, and the
    `Purpose` enumeration of PLAN section 2.15 has no value for a price perturbation. Two
    resolutions are admissible and the lead must record one in `spec/CHANGELOG.md`: (i) add a
    purpose (for example `pricepert`) to `Purpose` at the v1 freeze and draw
    `dist="normal", mean=0.0, sigma=PRICE_PERTURBATION_SIGMA` with `shape=(J,)` keyed by each seed;
    or (ii) move this helper to `gosplan/experiments/price_sensitivity.py`, which is outside
    `gosplan/env/` and therefore outside rule 9. Until that is recorded, the implementer must not
    pick one silently and must not call `numpy.random` here.

    Binds: `tests/unit/test_prices.py` - `len(result) == len(seeds)`; every vector is strictly
    positive; the result is deterministic in `(prices, seeds)` and independent of call order;
    the vectors differ from the baseline and from each other. Owning WO: **WO-007** (helper),
    **WO-036** (the tables it feeds).
    """
    raise NotImplementedError("PLAN sections 2.9.4, 7.5 - implemented in WO-007")
