"""The single-enterprise dynamic program and its regime classifier (PLAN section 5).

Realises: PLAN section 5 (the analytical layer - the single-enterprise DP solved before any
multi-agent RL) and PLAN section 11 (test architecture, unit/property category). Owning work order:
**WO-002** (frozen tests; LEAD). Binds the WO-014 must-pass line of PLAN section 12.3, verbatim -
"`tests/unit/test_dp.py` (value iteration converges; with `a*pen -> inf` and `g = 0` the policy
reports truthfully; with `beta = 0, s = 0` effort is 0; regime classifier on synthetic
distributions; `DPGreedy` reproduces the DP policy inside the env at `N = 1`)". Module under test:
`gosplan/agents/dp.py`.

Setting (PLAN section 5, verbatim): one enterprise, no input-output (`a = 0`, `phi = 1`), fixed
capacity, geometric continuation `psi = cfg.incentive.tenure`, with

    y   = A * cap * e * epsbar,  epsbar ~ LogNormal(-sigmabar**2 / 2, sigmabar),
          sigmabar = sigma / sqrt(M)          # aggregation approximation, documented as such
    S'  = (1 - h) * S + y
    R   = rho * T
    audit against S' with rate a and noise sigma_aud
    S'' = S' - min(S', R)
    T'  = max(T_min, (1 + g) * T * (1 + lambda * clip(rho - 1, -c_dn, c_up)))

    V(T, S) = max_{e, rho} { E_eps[ B(rho) - M * kappa * e**2 - a * E_nu[Pen] ]
                             + psi * gamma * E V(T', S'') }

FORBIDDEN IN WO-014: any reinforcement learning. The DP is the ground truth for gate G2 criterion 1,
the source of the `b_hat_dp` threshold for criterion 2, the exact no-manipulation counterfactual of
the estimator-bias study (PLAN section 7.2), and the policy `DPGreedy` replays inside the
`N`-enterprise environment.

FROZEN BY CONTRACT RULE 2. SKELETON: every test is `@pytest.mark.skeleton` and skipped until WO-014
lands; each docstring states the exact assertion, formula and tolerance.
"""

from __future__ import annotations

import pytest


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-014")
def test_value_iteration_converges(p1_cfg) -> None:
    """The Bellman iteration reaches `grid.value_tol` and reports that it did.

    Assertion: `solve_single_enterprise(cfg, grid)` returns a `DPSolution` with `converged is True`
    and `n_iterations` finite and below the solver's own cap, at the PLAN section 5 grid
    (`n_target = 80`, `n_stock = 50`, effort step 0.05, `rho` on [0, 3] at step 0.02, 9
    Gauss-Hermite nodes, `value_tol = 1e-6`); the sup-norm change in `V` between the last two
    sweeps is below `grid.value_tol`; `value`, `policy_effort` and `policy_rho` all have shape
    `(n_target, n_stock)` and are finite everywhere. Solving twice at the same `(cfg, grid)` gives
    identical tables, so a solution can be cached by `(EnvConfig.hash(), grid)`.

    First bullet of the WO-014 must-pass list. Policy iteration is an acceptable alternative
    provided the same convergence report is produced.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-014")
def test_policy_reports_truthfully_in_the_high_penalty_limit(p1_cfg) -> None:
    """With `audit_rate * penalty_scale -> inf` and `g = 0` the optimal report is truthful.

    Assertion: at `audit_rate = 1.0`, a very large `penalty_scale` (large enough that the expected
    penalty dominates the notch, e.g. `pen >= 1e4 * beta`) and `growth_directive = 0`, the optimal
    `rho*(T, S)` satisfies `rho* * T <= S'` at every grid point up to one report-grid step
    (`grid.rho_step = 0.02`), i.e. the claim never exceeds the stock available to back it; and the
    resulting `fictitious_padding = mean max(0, R - S) / T` is below 0.01. Reporting above stock is
    strictly dominated once the expected penalty exceeds the notch, and the DP must find that.

    Second bullet of the WO-014 must-pass list.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-014")
def test_optimal_effort_is_zero_without_a_bonus(p1_cfg) -> None:
    """With `notch_height = 0` and `overfulfilment_slope = 0` the optimal effort is 0 everywhere.

    Assertion: at `beta = 0` and `s = 0` the bonus is identically 0, so the period return is
    `-M * kappa * e**2 - a * E[Pen]` and effort is pure cost: `policy_effort` is exactly 0.0 at
    every grid point, `mean_effort == 0`, and the value function is non-positive everywhere. This is
    the sanity check that the cost term enters with the right sign and that the DP is not rewarding
    production through some other channel.

    Third bullet of the WO-014 must-pass list.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-014")
def test_regime_classifier_labels_a_bunching_distribution(p1_cfg) -> None:
    """`classify_regime` returns `"bunching"` on a synthetic distribution with known mass.

    Assertion, thresholds verbatim from PLAN section 5: a `DPSolution` whose `stationary_rho` puts
    more than 0.5 of its mass in `[1.00, 1.005]` and whose `rho_edge_frac` is below 0.05 classifies
    as `"bunching"`; moving the mass to 0.49 or the edge fraction to 0.06 takes it out of that
    label. The distributions are synthetic and constructed by the test, so the label is checked
    against a known answer rather than against whatever the solver happened to produce.

    Fourth bullet of the WO-014 must-pass list.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-014")
def test_regime_classifier_labels_a_pad_to_cap_distribution(p1_cfg) -> None:
    """`classify_regime` returns `"pad_to_cap"` when the grid edge holds most of the mass.

    Assertion: `rho_edge_frac > 0.5` gives `"pad_to_cap"`, whatever the mass near 1; at exactly 0.5
    it does not. A grid-edge hit is a regime signal, not an artefact to be smoothed away - the
    solver extends `rho_hi` when the optimum sits at the edge and logs every edge hit (PLAN section
    5), and this label is how the regime map of WO-015 colours that region.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-014")
def test_regime_classifier_labels_truthful_underfulfilment(p1_cfg) -> None:
    """`classify_regime` returns `"truthful_underfulfilment"` on low mean `rho` with no padding.

    Assertion: a solution with `mean(stationary_rho) < 0.9` and `fictitious_padding < 0.01`
    classifies as `"truthful_underfulfilment"`; raising the padding above 0.01 or the mean above 0.9
    removes the label. This is the region gate G1 must avoid when the human picks Phase-1 values:
    the interesting parameter points sit in the interior of the bunching region (PLAN sections 5,
    13).
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-014")
def test_regime_classifier_falls_back_to_mixed(p1_cfg) -> None:
    """Anything matching none of the three named regimes classifies as `"mixed"`.

    Assertion: a distribution with mass 0.3 near 1, an edge fraction of 0.1, mean `rho` of 1.05 and
    padding of 0.05 - matching no named branch - returns `"mixed"`; the classifier is total (every
    input returns one of the four `RegimeLabel` values) and deterministic. The branches are checked
    in the order PLAN section 5 lists them, so a distribution satisfying two conditions gets the
    earlier label.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-014")
def test_dp_reuses_the_environment_bonus_and_penalty_functions(p1_cfg) -> None:
    """The DP calls `bonus` and the penalty of `audit_and_penalise`, never a re-derivation.

    Assertion: the period return the solver evaluates at a grid point equals
    `bonus(rho, cfg) - M * kappa * e**2 - audit_rate * E_nu[Pen(rho, S', cfg)]` recomputed by the
    test from `gosplan.env.reward.bonus` and the penalty formula of `gosplan.env.reporting`, to
    1e-12; and a change to the bonus configuration (`notch_width`, `overfulfilment_cap`) changes the
    DP's return in exactly the way the environment's `bonus` changes. Stated as a requirement in
    PLAN section 5: "the DP must call the same functions the environment uses, never a
    re-derivation of them, so the two can never drift".
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-014")
def test_expectations_use_gauss_hermite_quadrature_in_log_space(p1_cfg) -> None:
    """The yield and audit-noise expectations are `grid.gh_nodes`-node Gauss-Hermite in log space.

    Assertion: `gauss_hermite_lognormal_nodes(sigma, n_nodes)` returns nodes and weights whose
    weights sum to 1 to 1e-12 and which integrate the lognormal exactly enough that
    `sum_k w_k * node_k == 1` to 1e-10 at `mean_log = -sigma**2 / 2` (the unit-mean shock of PLAN
    section 2.6) and `sum_k w_k * node_k**2 == exp(sigma**2)` to 1e-8; and the aggregation
    approximation `sigmabar = sigma / sqrt(M)` is the one the solver uses, documented as an
    approximation in the solution it returns.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-014")
def test_grid_edge_hits_are_measured_and_reported(p1_cfg) -> None:
    """`rho_edge_frac` counts the stationary mass sitting at the report-grid edge.

    Assertion: `report_grid_edge_fraction(stationary_rho, rho_grid)` equals the fraction of samples
    at the top grid point to 1e-12, is 0 when no sample sits there and 1 when all do; and a solution
    whose optimum sits at the edge triggers the automatic extension of `grid.rho_hi` described in
    PLAN section 5, with the edge hit still logged after the extension. An edge hit is a regime
    signal (the `"pad_to_cap"` label above), never something to be smoothed away.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-014")
def test_dpgreedy_reproduces_the_dp_policy_inside_the_environment_at_one_enterprise(
    p1_cfg, rng_seed
) -> None:
    """`DPGreedy` replays the solved policy exactly at `N = 1`, `a = 0`, `phi = 1`.

    Assertion: with a configuration matching the DP's setting - one enterprise, a zero I-O matrix,
    `final_demand_share = 1` - the actions `DPGreedy` emits inside `GosplanEnv` equal the DP tables
    looked up at the current `(T, S)` to within one grid step (`grid.effort_step = 0.05` for effort,
    `grid.rho_step = 0.02` for the report), at every step of several seeded episodes; and the
    realised mean effort and mean report ratio match `sol.mean_effort` and
    `mean(sol.stationary_rho)` to 0.02. This is what makes the DP a usable baseline inside the
    multi-agent environment and the reference the G2 criterion-1 comparison is made against.

    Fifth bullet of the WO-014 must-pass list.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-014")
def test_solution_records_the_grid_and_configuration_it_was_solved_for(p1_cfg) -> None:
    """A `DPSolution` carries `grid` and `config_hash`, so it can never be mis-attributed.

    Assertion: `sol.grid` equals the `DPGrid` passed in and `sol.config_hash == cfg.hash()`; two
    solutions for different configurations carry different hashes; and the summary statistics the
    Phase-1 gates read - `b_hat_dp`, `fictitious_padding`, `hidden_reserves`, `mean_effort`,
    `regime`, `rho_edge_frac` - are all finite and present. `b_hat_dp` must be computed with the
    pre-registered estimator settings of PLAN section 4.5, the same ones the learned runs use, or
    the G2 criterion-2 comparison `b_hat >= 0.5 * b_hat_dp` compares two different quantities.
    """
    assert False
