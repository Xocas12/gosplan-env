AMBIGUITY REPORT   WO-014   gosplan/agents/dp.py
Questions raised by the WO-014 session (one ruling each):

A. `tests/unit/test_dp.py` built `DPGrid` / `DPSolution` with names the frozen spec does not declare
   (`target_points`, `stock_points`, `report_step`, `report_hi`, `max_iterations`,
   `policy_report`, `excess_mass`), while the test's own docstrings use the spec names.
B. How is `E V(T', S'')` evaluated off the grid?
C. How far, and when, is the report grid extended when the optimum sits at its edge?
D. Is `S_max` applied to `S'` (as the environment does) or to `S''` (the dp.py docstring)?
E. Audit-noise quadrature: `N(0, sigma_aud**2)` as the environment, or mean-one lognormal nodes?
F. `dp_excess_mass` has no array-level estimator to call: `phenomenon_bunching` takes a ledger and
   the estimator is WO-016's, scheduled after G1 although G1 needs `b_hat_DP`.
G. Which seed and draw indices does the DP's policy simulation use?
J. Which sector supplies `A` and `sigma` when `J > 1`?

LEAD RESOLUTION (2026-09-24):
A. Lead edit of the test helpers to the declared names; `DPGrid` gains `max_iterations: int = 5000`
   (the test's value) in `spec/spec.py` and `gosplan/agents/dp.py` - an additive v0 change recorded
   in `spec/CHANGELOG.md`. Reaching the cap returns `converged = False`.
B. Bilinear interpolation in `(log T, S)`, clamped at the grid ends (no extrapolation): the target
   axis is log-spaced, consistent with AMBIGUITY-009. The returned policy tables themselves are
   never interpolated.
C. Extend `rho_hi` by its original span each time the stationary mass at the top grid point is
   positive (`rho_edge_frac > 0`), re-solving, until the edge carries no stationary mass or
   `rho_hi` reaches `cfg.tech.report_max_ratio` (the environment's bound), where it stops; the
   final `rho_edge_frac` is reported as the regime signal. Never narrowed.
D. `S'` is capped at `S_max` before the audit and delivery, exactly as `process_reports` does: the
   DP models the environment, and the environment is the ground truth G2 compares against.
E. `N(0, sigma_aud**2)` Gauss-Hermite nodes, matching `audit_and_penalise`. Inert in Phase 1
   (`sigma_aud = 0`).
F. `dp_excess_mass` calls `gosplan.metrics._fallback.estimate` directly with the section 4.5
   constants from `phenomena.py` and returns its `excess_mass` (point estimate). The estimator's
   point estimate is pulled forward from WO-016 because the regime map needs `b_hat_DP` before G1;
   its bootstrap and `resolve_estimators` stay open (AMBIGUITY-011) until the human decides.
G. `seed = cfg.tech.seed_env`; the per-period yield shock is `draw(seed, "yield", episode, t, ...)`
   and the audit draw `draw(seed, "audit", episode, t, ...)`: the DP simulation is its own
   stream, not a common-random-number twin of the environment (whose shocks are per step).
J. The single enterprise is enterprise 0: `A` and `sigma` of `cfg.supply.sector_of[0]`. Documented
   in the code as an approximation for `J > 1`; configurations built for G2 criterion 1 have `J = 1`.
