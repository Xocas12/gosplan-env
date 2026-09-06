"""Single-enterprise dynamic program: the analytical layer of PLAN section 5.

Realises: PLAN section 5 in full, plus the parts of PLAN sections 4.5 (estimator settings for
`b_hat`), 2.6 (period yield), 2.7.1 (target rule), 2.8 (bonus, audit, penalty), 2.11 (inventory)
and 2.12 (geometric continuation) that the DP reproduces exactly. Owning work order: **WO-014**;
consumed by **WO-015** (regime map), **WO-019** (DP vs PPO) and by `DPGreedy` in
`gosplan/agents/heuristic.py` (WO-010).

**No reinforcement learning in this module.** The WO-014 forbidden list is one line long - *any
reinforcement learning* - and it is the point of the whole layer: the DP is the independent ground
truth against which the learning stack is checked at gate G2, so it must share no code, no
sampling loop and no hyper-parameter with the learner. What it does share, and must share, are the
environment's own reward pieces: the Bellman operator calls `gosplan.env.reward.bonus` and the
penalty of `gosplan.env.reporting.audit_and_penalise`, never a re-derivation of them, so the DP and
the environment can never drift apart.

The setting (PLAN section 5). One enterprise, no input-output structure (`a = 0`, `phi = 1`, hence
coverage `H = 1` and every unit claimed is deliverable), fixed capacity `cap = 1`, geometric
continuation `psi = cfg.incentive.tenure`. The environment's multi-agent supply coupling is absent
by construction: that is what makes the problem a two-state dynamic program and what makes the
difference between DP and learned behaviour at `N = 20` a *multi-agent effect* rather than a
training artefact (PLAN section 4.5, criterion 2).

The five uses of a solved `DPSolution`, all named in PLAN section 5:

  1. **G2 criterion 1 ground truth.** At `N = 1`, PPO's mean fictitious padding must be within 0.02
     of the DP's, mean effort within 0.05, and the Wasserstein-1 distance between the two
     stationary `rho_report` distributions below 0.03, in at least 8 of 10 seeds, at each of the
     three `a * pen` levels chosen at gate G1 (WO-019).
  2. **The `b_hat_dp` threshold.** G2 criterion 2 requires the measured excess mass at the notched
     configuration to reach at least `0.5 * b_hat_dp`, with a bootstrap CI excluding 0, in at least
     90% of 30 seeds (WO-020).
  3. **The exact no-manipulation counterfactual** for the estimator-bias study of PLAN section 7.2:
     the DP supplies the true `rho` distribution for each `(notch_width, overfulfilment_cap)` cell,
     so the bias and coverage of the bunching estimator are measured against a known answer.
  4. **Sizing the rare-audit optimisation gap** (finding F9): the DP's value at the optimum bounds
     how much of a learned policy's shortfall is optimisation failure rather than equilibrium.
  5. **The `DPGreedy` baseline**, replayed inside the `N`-enterprise environment (PLAN section 6.1).

Cost: about one minute per configuration on one core at the grid below; the 500-point regime map of
WO-015 is about one CPU-hour on eight cores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from gosplan.agents.base import Array

if TYPE_CHECKING:  # runtime home of the configuration: gosplan/config.py (WO-003, PLAN section 8)
    from gosplan.config import EnvConfig

RegimeLabel = Literal["bunching", "pad_to_cap", "truthful_underfulfilment", "mixed"]
"""Regime label attached to a solved DP (PLAN section 5); the return type of `classify_regime`.
Declared here, in the module that produces it, so that `gosplan/experiments/regime_map.py` (WO-015)
imports one definition rather than restating the four labels. Identical to `spec/spec.py`'s alias;
the unit test that compares the runtime package against the frozen spec keeps them so."""

DP_DISCOUNT = 0.99
"""The technical discount `gamma` in the Bellman operator's `psi * gamma` continuation factor (PLAN
sections 5 and 6.1). It is 0.99, the same value the PPO adapter uses
(`gosplan/agents/ppo/adapter.py`, `PPOConfig.gamma`), because G2 criterion 1 compares the two
policies directly and a discount mismatch would compare two different problems. It is deliberately
**not** an `EnvConfig` field: the environment never reads a discount (the spec's `TechConfig`
docstring records this), and `psi = cfg.incentive.tenure` is the separate *economic* continuation
probability of PLAN section 2.12."""


@dataclass(frozen=True)
class DPGrid:
    """Discretisation of the single-enterprise dynamic program (PLAN section 5).

    Frozen and hashable, so a solved `DPSolution` can be cached by `(EnvConfig.hash(), grid)`. The
    values below are PLAN section 5's grid verbatim; they are the reference resolution for the G2
    recovery criterion and for the regime map, so changing one changes what `b_hat_dp` means and is
    recorded in the run manifest (CONTRACT rule 10) rather than adjusted quietly.

    Field-for-field identical to `spec/spec.py`'s `DPGrid`. Owning WO: **WO-014**.
    """

    n_target: int = 80
    """Points on the target grid `T`, log-spaced on `[T_min, target_hi_mult * A * cap]` with
    `T_min = cfg.tech.target_floor_frac * T_0` and `T_0 = cfg.tech.initial_target_frac * A * cap`
    (PLAN sections 5, 2.7.1, 3)."""

    target_hi_mult: float = 4.0
    """Upper end of the target grid as a multiple of `A * cap`. A stationary distribution that
    piles up at this edge means the ratchet is running away and the solution is reported with that
    fact, not re-gridded to hide it."""

    n_stock: int = 50
    """Points on the stock grid `S`, linearly spaced on `[0, S_max]` with
    `S_max = cfg.tech.inventory_cap_mult * cap` (PLAN section 2.11)."""

    effort_step: float = 0.05
    """Spacing of the period-effort action grid `e in {0, 0.05, ..., 1}` (21 points). One effort per
    *period*, not per step: nothing within the period is observed, so uniform effort across the `M`
    steps is optimal and its cost is `M * kappa * e**2` (PLAN sections 5, 2.6)."""

    rho_lo: float = 0.0
    """Lower end of the report action grid."""

    rho_hi: float = 3.0
    """Upper end of the report action grid. Extended automatically when the optimum sits at the
    edge (see `report_action_grid`), and every edge hit is logged as a regime signal rather than
    smoothed away (PLAN section 5)."""

    rho_step: float = 0.02
    """Spacing of the report action grid."""

    gh_nodes: int = 9
    """Gauss-Hermite nodes used to take the yield (and audit-noise) expectations in log space."""

    value_tol: float = 1e-6
    """Value-iteration convergence tolerance on the sup-norm change in `V`. Policy iteration is an
    acceptable alternative (PLAN section 5) provided it reports the same `converged` flag."""

    sim_episodes: int = 200
    """Episodes simulated under the optimal policy to obtain the stationary `rho_report`
    distribution (PLAN section 5: 200 episodes x 200 periods)."""

    sim_periods: int = 200
    """Periods per simulated episode for that distribution."""


@dataclass
class DPSolution:
    """The solved single-enterprise problem and everything Phase 1 reads off it (PLAN section 5).

    Field-for-field identical to `spec/spec.py`'s `DPSolution`. Produced by
    `solve_single_enterprise`; consumed by `classify_regime`, by `DPGreedy`
    (`gosplan/agents/heuristic.py`), by the regime map (WO-015) and by the DP-vs-PPO comparison
    (WO-019). Owning WO: **WO-014**.
    """

    policy_effort: Array
    """(n_target, n_stock) optimal period effort `e*(T, S)` on the effort grid."""

    policy_rho: Array
    """(n_target, n_stock) optimal report ratio `rho*(T, S)` on the (possibly extended) report
    grid."""

    value: Array
    """(n_target, n_stock) converged value function `V(T, S)`."""

    stationary_rho: Array
    """Samples of `rho_report` under the optimal policy, from `simulate_stationary_reports`
    (`sim_episodes * sim_periods` draws, burn-in per PLAN section 4.4)."""

    b_hat_dp: float
    """DP-predicted bunching excess mass, computed from `stationary_rho` with the pre-registered
    estimator settings of PLAN section 4.5 (see `dp_excess_mass`)."""

    fictitious_padding: float
    """`mean max(0, R - S) / T` under the stationary policy - PLAN section 4.1 row 4, the padding
    operationalisation, evaluated exactly rather than estimated."""

    hidden_reserves: float
    """`mean max(0, S - R) / T` after delivery - PLAN section 4.1 row 7. Computed here because the
    DP is a closed-form prediction, and **held out** as an empirical phenomenon until the Phase-2
    acceptance run: no Phase-1 table or plot reports it from a simulation."""

    mean_effort: float
    """Mean period effort under the stationary policy; G2 criterion 1 compares PPO's to this within
    0.05."""

    regime: RegimeLabel
    """`classify_regime(self)` - the label the regime map of WO-015 colours by."""

    rho_edge_frac: float
    """Fraction of the stationary mass at the report grid edge, from
    `report_grid_edge_fraction`.
    A regime signal in its own right: above 0.5 the label is `pad_to_cap`."""

    n_iterations: int
    """Iterations to convergence (value iteration sweeps, or policy-iteration rounds)."""

    converged: bool
    """True when the sup-norm change fell below `grid.value_tol` before the iteration cap. A
    solution with `converged = False` is reported with the flag and never silently used as ground
    truth at gate G2."""

    grid: DPGrid
    """The discretisation this solution was computed on, including any automatic extension of the
    report grid."""

    config_hash: str
    """`EnvConfig.hash()` of the configuration solved. `DPGreedy` checks it against its own `cfg`
    before acting, and the run manifest records it (CONTRACT rule 10)."""


def solve_single_enterprise(cfg: EnvConfig, grid: DPGrid) -> DPSolution:
    """Solve the single-enterprise dynamic program exactly (PLAN section 5).

    Takes: `cfg`, the configuration whose incentive parameters define the problem, and `grid`, the
    discretisation. Returns: a fully populated `DPSolution`.

    **Setting.** One enterprise; no input-output (`a = 0`, `phi = 1`), so coverage is 1 and every
    claimed unit can be delivered; fixed capacity `cap = 1`; geometric continuation
    `psi = cfg.incentive.tenure`; discount `DP_DISCOUNT`. Sector quantities are the enterprise's
    own: `A = cfg.supply.productivity[j]`, `sigma = cfg.supply.yield_sigma[j]`,
    `M = cfg.incentive.steps_per_period`, `kappa = cfg.incentive.effort_cost`,
    `h = cfg.supply.holding_loss`, `a_rate = cfg.information.audit_rate`,
    `sigma_aud = cfg.information.audit_noise`.

    **State grids.** `T`: `grid.n_target` (80) points, log-spaced on
    `[T_min, grid.target_hi_mult * A * cap]`, `T_min = cfg.tech.target_floor_frac * T_0`,
    `T_0 = cfg.tech.initial_target_frac * A * cap`. `S`: `grid.n_stock` (50) points, linear on
    `[0, cfg.tech.inventory_cap_mult * cap]`.

    **Action grids.** Period effort `e in {0, grid.effort_step, ..., 1}` (uniform across the `M`
    steps, cost `M * kappa * e**2`). Report `rho` on `report_action_grid(grid)`, i.e.
    `[grid.rho_lo, grid.rho_hi]` in steps of `grid.rho_step`, **extended automatically** when the
    argmax sits at the top edge (see that function), with every edge hit logged as a regime signal.
    The maximisation is vectorised over the full `(e, rho)` action grid for every state.

    **Period yield.** With nothing observed within the period, the `M` step shocks aggregate to one
    period shock, approximated as

        y       = A * cap * e * epsbar
        epsbar ~ LogNormal(-sigma_hat**2 / 2, sigma_hat),   sigma_hat = sigma / sqrt(M)

    which has mean 1. This aggregation is an approximation (the sum of lognormals is not lognormal)
    and PLAN section 5 requires it to be documented as one wherever a DP number is reported. The
    expectation over `epsbar`, and the expectation over the audit noise `nu ~ N(0, sigma_aud**2)`,
    are taken with `grid.gh_nodes` (9) node Gauss-Hermite quadrature **in log space** - see
    `gauss_hermite_lognormal_nodes`.

    **Transition** (PLAN section 5, verbatim):

        S'  = (1 - h) * S + y
        R   = rho * T
        audit against S' with rate `a_rate` and log-noise `sigma_aud`
        S'' = S' - min(S', R)                    # delivery next period, phi = 1
        T'  = max(T_min, (1 + g) * T * (1 + lambda * clip(rho - 1, -c_dn, c_up)))

    with `g = cfg.incentive.growth_directive`, `lambda = cfg.incentive.ratchet_lambda`,
    `c_up = cfg.incentive.ratchet_cap_up`, `c_dn = cfg.incentive.ratchet_cap_dn`, and the deadband
    of PLAN section 2.7.1 applied exactly as the environment applies it. `S''` is additionally
    capped at `S_max` with the overflow discarded (PLAN section 2.11), matching the environment.

    **Bellman operator** (PLAN section 5, verbatim):

        V(T, S) = max_{e, rho} { E_eps[ B(rho) - M * kappa * e**2 - a_rate * E_nu[Pen] ]
                                 + psi * gamma * E V(T', S'') }

    `B` is `gosplan.env.reward.bonus` and `Pen` is the penalty of
    `gosplan.env.reporting.audit_and_penalise` - the same functions the environment calls, never a
    re-derivation (PLAN section 5). Rewards enter unscaled here; `reward_scale(cfg)` is a
    monotone positive rescaling of the objective and changes no argmax, so applying it is optional
    and must be recorded either way. Continuation values off the grid are interpolated on the
    `(T, S)` grids; value iteration runs to `grid.value_tol` in the sup norm (policy iteration is an
    acceptable alternative), and `n_iterations` and `converged` record the outcome.

    **Outputs.** The two policy tables and `value`; `stationary_rho` from
    `simulate_stationary_reports`; `b_hat_dp` from `dp_excess_mass`; `rho_edge_frac` from
    `report_grid_edge_fraction`; `fictitious_padding = mean max(0, R - S) / T` and
    `hidden_reserves = mean max(0, S - R) / T` after delivery, both over the same simulated path;
    `mean_effort`; `regime = classify_regime(sol)`; `grid`; `config_hash = cfg.hash()`.

    **Forbidden (WO-014 card): any reinforcement learning.** No sampling-based policy improvement,
    no function approximation, no gradient step - this is the independent ground truth for the
    learning stack and must stay independent of it.

    Binds: `tests/unit/test_dp.py` (WO-014) - value iteration converges; with `a_rate * pen -> inf`
    and `g = 0` the optimal policy reports truthfully; with `notch_height = 0` and
    `overfulfilment_slope = 0` optimal effort is 0; `DPGreedy` reproduces this policy inside the
    environment at `N = 1`. Owning WO: **WO-014**.
    """
    raise NotImplementedError("PLAN section 5 - implemented in WO-014")


def classify_regime(sol: DPSolution) -> RegimeLabel:
    """Label a solved DP by the behaviour it predicts (PLAN section 5).

    Takes: `sol`, a `DPSolution` whose `stationary_rho`, `rho_edge_frac` and `fictitious_padding`
    are filled in. Returns: one `RegimeLabel`, by these thresholds, verbatim from PLAN section 5 and
    evaluated in this order:

        "pad_to_cap"                mass at the report grid edge > 0.5
        "bunching"                  stationary mass of rho in [1.00, 1.005] > 0.5
                                    AND mass at the report grid edge < 0.05
        "truthful_underfulfilment"  mean rho < 0.9 AND fictitious padding < 0.01
        "mixed"                     otherwise

    Mass at the edge is `sol.rho_edge_frac`; the bunching window `[1.00, 1.005]` is a closed
    interval on the stationary samples, and is deliberately narrower than the estimator window
    `[1.00, 1.02]` of PLAN section 4.5 - the classifier reads the exact distribution, the estimator
    reads a binned one.

    A grid-edge hit is a **regime signal, not an artefact**: `pad_to_cap` means the optimum wants to
    report beyond the grid, and the answer is to report that fact (and the automatic extension the
    solver logged), never to widen the grid until the label changes.

    Binds: `tests/unit/test_dp.py` - the classifier on synthetic distributions with known labels.
    Owning WO: **WO-014**.
    """
    raise NotImplementedError("PLAN section 5 - implemented in WO-014")


def gauss_hermite_lognormal_nodes(sigma: float, n_nodes: int) -> tuple[Array, Array]:
    """Quadrature nodes and weights for a mean-one lognormal expectation (PLAN section 5).

    Takes: `sigma`, the log-sd - `sigma_hat = sigma_j / sqrt(M)` for the period yield, or
    `cfg.information.audit_noise` for the audit measurement; `n_nodes`, the node count
    (`grid.gh_nodes`, 9). Returns: `(nodes, weights)`, each `(n_nodes,)`, such that for any `f`

        E[f(eps)] ~= sum_n weights[n] * f(nodes[n]),    eps ~ LogNormal(-sigma**2 / 2, sigma)

    Construction, in log space as PLAN section 5 requires: take the physicists' Gauss-Hermite rule
    `(x_n, w_n)` from `numpy.polynomial.hermite.hermgauss(n_nodes)`, which integrates against
    `exp(-x**2)`. For a normal variate with mean `m` and sd `s`,
    `E[g(z)] = sum_n (w_n / sqrt(pi)) * g(m + sqrt(2) * s * x_n)`. With `m = -sigma**2 / 2` this
    gives

        nodes[n]   = exp(-sigma**2 / 2 + sqrt(2) * sigma * x_n)
        weights[n] = w_n / sqrt(pi)

    so `sum weights == 1` to machine precision and `sum weights * nodes == 1` to quadrature
    accuracy - the mean-one property of the yield shock (PLAN section 2.6). At `sigma == 0` the rule
    degenerates to a single node at 1 with weight 1 and the implementation must return that rather
    than a numerically noisy nine-node collapse.

    `numpy` supplies the rule; nothing here may import `scipy` (skeleton import rule; `scipy` is a
    project dependency but is not needed for a Gauss-Hermite rule). Binds: `tests/unit/test_dp.py` -
    weights sum to 1 and the node-weighted mean is 1 to 1e-10 for the sigmas of
    `cfg.supply.yield_sigma`. Owning WO: **WO-014**.
    """
    raise NotImplementedError("PLAN section 5 - implemented in WO-014")


def report_action_grid(grid: DPGrid, rho_hi: float | None = None) -> Array:
    """Build the report action grid, with the automatic extension of PLAN section 5.

    Takes: `grid`; `rho_hi`, an override for the upper end, used when a previous solve found its
    optimum at the edge. Returns: the report action grid `(n_rho,)`,
    `[grid.rho_lo, rho_hi or grid.rho_hi]` inclusive in steps of `grid.rho_step` (Phase-1 default:
    0.00 to 3.00 in 0.02 steps, 151 points).

    Automatic extension (PLAN section 5). After a solve, if the argmax `rho` sits at the top edge
    for any state carrying stationary mass, the solver re-solves on a grid extended upward and
    **logs the edge hit as a regime signal**. The extension is recorded in the returned
    `DPSolution.grid` and in the run manifest, so a `pad_to_cap` regime is never mistaken for a
    grid that was simply too short. Two constraints on the implementation: the extension is
    monotone (the grid only grows) and it terminates, so a solution that keeps hitting the edge is
    reported as `pad_to_cap` with `rho_edge_frac` near 1 rather than extended indefinitely.

    Note that the environment's own bound is `cfg.tech.report_max_ratio` (10.0, CONTRACT rule 8).
    An extension beyond it says the unconstrained optimum lies outside what the environment allows,
    which is a result to report - the environment's bound is never widened to match the DP.

    Owning WO: **WO-014**.
    """
    raise NotImplementedError("PLAN section 5 - implemented in WO-014")


def report_grid_edge_fraction(stationary_rho: Array, rho_grid: Array) -> float:
    """Fraction of stationary report mass sitting at the report grid's top edge (PLAN section 5).

    Takes: `stationary_rho` `(n_samples,)`, the simulated stationary reports; `rho_grid`
    `(n_rho,)`, the grid those reports were chosen from. Returns: the fraction of samples equal to
    `rho_grid[-1]` (exactly - reports are chosen from the grid, so this is an equality test on grid
    values, not a tolerance band).

    This is the `rho_edge_frac` field of `DPSolution` and the quantity two of the four regime
    thresholds key on: `pad_to_cap` above 0.5, `bunching` requires below 0.05 (`classify_regime`).
    It is also the edge-hit log of PLAN section 5: the value is written to the run manifest for
    every solved configuration, so a regime map cell can always be traced back to whether its
    optimum was interior.

    Owning WO: **WO-014**.
    """
    raise NotImplementedError("PLAN section 5 - implemented in WO-014")


def simulate_stationary_reports(
    cfg: EnvConfig, grid: DPGrid, policy_effort: Array, policy_rho: Array, seed_env: int
) -> tuple[Array, Array, Array]:
    """Simulate the optimal policy to obtain its stationary distribution (PLAN section 5).

    Takes: `cfg`; `grid`; `policy_effort` and `policy_rho`, the solved `(n_target, n_stock)` tables;
    `seed_env`, the root environment seed for the simulation's draws. Returns:
    `(rho_report, padding, reserves)` - three `(n_samples,)` arrays over the retained periods, where
    `rho_report` is the report ratio actually chosen, `padding` is `max(0, R - S) / T` (PLAN section
    4.1 row 4) and `reserves` is `max(0, S - R) / T` after delivery (row 7). `DPSolution` stores
    the first array and the means of the other two.

    Design (PLAN section 5): `grid.sim_episodes` (200) episodes of `grid.sim_periods` (200) periods
    each, started from the initial state of the environment (`T = T_0`, `S = 0`, PLAN section 2.2)
    and stepped with exactly the transition of `solve_single_enterprise` - the same holding loss,
    the same delivery `S'' = S' - min(S', R)`, the same stock cap, the same ratchet. Actions are
    read off the tables by nearest-grid-point lookup, the same rule `DPGreedy` uses, so the two can
    be compared. Episodes here are long on purpose: this is a stationary distribution, not the
    geometric-horizon episode of PLAN section 2.12, and the continuation probability `psi` enters
    only through the Bellman operator.

    Burn-in: periods `t < 2` are discarded, matching the measurement window of PLAN section 4.4, so
    the target-initialisation transient never enters `b_hat_dp`.

    Randomness: every draw goes through `gosplan.rng.draw(seed_env, purpose, *indices, ...)` with
    the purposes of PLAN section 2.15 - `yield` for `epsbar`, `audit` for the audit selection,
    `auditnoise` for the measurement error - so the DP simulation is order-independent and shares
    common random numbers with the environment when `seed_env` is shared (CONTRACT rule 9). No
    direct `numpy.random` call and no module-level generator.

    Owning WO: **WO-014**.
    """
    raise NotImplementedError("PLAN section 5 - implemented in WO-014")


def dp_excess_mass(stationary_rho: Array) -> float:
    """The DP's predicted bunching excess mass `b_hat_dp` (PLAN sections 5, 4.5).

    Takes: `stationary_rho` `(n_samples,)`, the stationary report ratios from
    `simulate_stationary_reports`. Returns: `b_hat_dp`, the excess mass at the notch, a float.

    Estimator settings, verbatim from the pre-registration of PLAN section 4.5 and identical to the
    ones every measured run uses - that is the whole point of the number, since G2 criterion 2
    compares a measured `b_hat` against `0.5 * b_hat_dp`:

        bins of width 0.005 over rho in [0.6, 1.4]
        excluded window [0.95, 1.02]; polynomial of degree 7 fitted outside it
        b_hat = (observed - counterfactual mass in [1.00, 1.02])
                / mean counterfactual density in the window
        hole mass computed identically on [0.95, 1.00) and reported alongside

    The estimator itself is **not implemented here**. This function calls the single pre-registered
    entry point in `gosplan/metrics/phenomena.py` (WO-016), which is the one place in the repository
    where `forensics_core.bunching.estimate` is imported inside a `try/except ImportError` that
    falls back to `gosplan/metrics/_fallback.py` with an identical signature (PLAN section 7.3).
    This module must never import `forensics_core` itself: two import sites would make it possible
    for the DP prediction and the measured value to come from different estimators, which would
    void the comparison.

    Because the DP's distribution is exact, `b_hat_dp` carries no sampling CI of its own; the
    bootstrap SE of PLAN section 4.5 applies to measured runs, over seeds.

    Owning WO: **WO-014** (this wrapper), **WO-016** (the estimator it calls).
    """
    raise NotImplementedError("PLAN sections 5, 4.5 - implemented in WO-014")


__all__ = [
    "DP_DISCOUNT",
    "DPGrid",
    "DPSolution",
    "RegimeLabel",
    "classify_regime",
    "dp_excess_mass",
    "gauss_hermite_lognormal_nodes",
    "report_action_grid",
    "report_grid_edge_fraction",
    "simulate_stationary_reports",
    "solve_single_enterprise",
]
