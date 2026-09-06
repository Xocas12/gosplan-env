"""Plan prices: the cost-plus fixed point and its perturbations (PLAN section 2.10).

Realises: PLAN section 2.10 (prices and final demand), PLAN section 7.5 (the standing
price-sensitivity check) and PLAN section 11 (test architecture, unit/property category). Owning
work order: **WO-002** (frozen tests; LEAD). Binds the WO-007 must-pass line of PLAN section 12.3,
verbatim - "`tests/unit/test_prices.py` (cost-plus fixed point converges; positive)". Module under
test: `gosplan/env/prices.py`.

Formula (PLAN section 2.10, verbatim):

    p_j = (1 + m) * (kappa_labour + sum_k a_jk * p_k)

solved as a fixed point over `j` at `t = 0`, with `m = cfg.supply.price_markup` (0.1 in Phase 1).
`kappa_labour` is not a configuration parameter: `p` is homogeneous of degree 1 in it and every use
of `p` in the design is a ratio, so it is fixed at 1.0 as a recorded normalisation. Convergence
requires `(1 + m) * sum_k a_jk < 1` for every row, which is why `EnvConfig.validate` rejects
`sum_k a_jk >= 1` (WO-003).

Prices are not a modelling nuisance to be tuned: PLAN sections 2.9.4 and 7.5 require every headline
table to be recomputed under three perturbed price vectors (`p_j * exp(u_j)`, `u ~ N(0, 0.3**2)`,
fixed seeds), and a sign change in `specification_gap` is reported rather than suppressed.

FROZEN BY CONTRACT RULE 2. SKELETON: every test is `@pytest.mark.skeleton` and skipped until WO-007
lands; each docstring states the exact assertion, formula and tolerance.
"""

from __future__ import annotations

import pytest


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-007")
def test_initial_prices_solve_the_cost_plus_fixed_point(p1_cfg) -> None:
    """`initial_prices(cfg)` satisfies the fixed-point equation it solves.

    Assertion: the returned `p` `(J,)` satisfies
    `p_j == (1 + m) * (1.0 + sum_k a_jk * p_k)` for every `j` to 1e-12 (residual in the infinity
    norm), with `m = cfg.supply.price_markup` and `kappa_labour = 1.0` as the recorded
    normalisation. The solver converges for every I-O matrix that `EnvConfig.validate` accepts, and
    the same fixed point is reached from different starting vectors - it is a contraction while
    `(1 + m) * sum_k a_jk < 1`, so the answer is a property of the configuration and not of the
    iteration.

    First half of the WO-007 prices must-pass line.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-007")
def test_every_plan_price_is_strictly_positive(p1_cfg, tiny_cfg) -> None:
    """Prices are strictly positive and finite.

    Assertion: every entry of `initial_prices(cfg)` is `> 0` and finite, at `p1_cfg`, at `tiny_cfg`
    and at a matrix with a zero row (a sector needing no inputs, whose price is exactly
    `(1 + m) * kappa_labour`). Positivity matters downstream: `p` is a denominator in the
    `net_output` fulfilment measure (`p_j / p_{s(i)}`, PLAN section 2.9.2) and a weight in
    `val_measured` and `val_true`, so a zero or negative price would silently corrupt every
    headline metric of PLAN section 2.9.4.

    Second half of the WO-007 prices must-pass line.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-007")
def test_prices_rise_with_the_markup_and_with_input_intensity(p1_cfg) -> None:
    """The solution responds to `m` and to `a` in the direction the formula dictates.

    Assertion: raising `price_markup` raises every price; raising any `a_jk` (while keeping every
    row sum below the convergence bound) raises `p_j` weakly and never lowers it; and at `a = 0`
    every price equals `(1 + m) * kappa_labour` exactly. These are consequences of the fixed-point
    equation, not calibration targets - the test exists so that a sign error in the iteration is
    caught by a property rather than by a golden file.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-007")
def test_recompute_prices_agrees_with_initial_prices_on_the_true_matrix(p1_cfg) -> None:
    """`recompute_prices(planner_io, cfg)` equals `initial_prices(cfg)` when `planner_io == a`.

    Assertion: with `planner_io` equal to `cfg.supply.io_matrix` - which is the Phase-1 state at
    every `t`, since `tech_drift_sigma = 0` - the two agree to 1e-12. The distinction matters from
    Phase 2 on: recomputation uses the planner's possibly stale `planner_io`, never the true `a`
    (PLAN section 2.10), so the two diverge exactly when drift is on. Phase 1 holds prices fixed
    after `t = 0` (`price_lag = inf`), so `recompute_prices` is never called inside a Phase-1
    episode.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-007")
def test_perturbed_price_vectors_are_positive_and_deterministic(p1_cfg) -> None:
    """The price-sensitivity vectors of PLAN section 7.5 are reproducible.

    Assertion: `perturbed_price_vectors(prices, seeds)` returns exactly `len(seeds)` vectors; every
    vector is strictly positive (it is `p_j * exp(u_j)` with `u ~ N(0, 0.3**2)`, so positivity is
    structural); the result is deterministic in `(prices, seeds)` and independent of call order
    (the T-U6 property); and the vectors differ from the baseline and from each other. These are the
    three fixed-seed perturbations under which every headline table of PLAN section 2.9.4 is
    recomputed, and a sign change in `specification_gap` across them is a reported result, never a
    suppressed one.
    """
    assert False
