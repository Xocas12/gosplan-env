AMBIGUITY REPORT   WO-016 (early slice) / WO-014   gosplan/metrics/_fallback.py   (NEEDS HUMAN)
Question (one sentence):
How is the seed-level bootstrap of the pre-registered bunching estimator specified, and what is
done about the estimator's measured bias on a smooth density under the PLAN section 4.5 settings?

What the spec says / does not say (quote):
PLAN section 4.5 and the `estimate` docstring: "`se`, `ci_lo`, `ci_hi` by bootstrap over seeds, the
caller supplying `x` grouped so that the seed is the resampling unit". The frozen signature has no
grouping argument, and nothing gives the grouping format, the replicate count, the CI level or
method, or the bootstrap seed. The frozen tests pass a flat 1-D sample. Separately,
`test_estimator_finds_no_excess_on_a_smooth_density` requires the CI to cover 0 on a smooth N(1,
0.15) sample, but under the pre-registered settings (bins 0.005 on [0.6, 1.4], degree-7 fit outside
[0.95, 1.02], excess on [1.00, 1.02]) the NOISE-FREE N(1, 0.15) density already gives
`b_hat = +0.058` (counts-per-bin units), and the test's own sample gives 0.076 - a positive bias of
the counterfactual fit, not sampling noise. Planted-mass recovery is accurate (3.83 vs about 3.80).

Options considered:
Bootstrap: (A) `x` as a sequence of per-seed arrays, a 1-D array being one seed; (B) report-level
resampling of a 1-D array (forbidden by the docstring); (C) a `groups` argument (changes the frozen
section 7.3 signature). Bias: (i) keep the pre-registered settings and report the bias (PLAN section
7.2 treats estimator bias as a result), amending the smooth-null test; (ii) amend the
pre-registration (no learning run has happened yet, so section 4 is still amendable with a record)
- e.g. a lower polynomial degree or a wider fit range; (iii) difference against the DP's own
`b_hat_DP` on the matching smooth configuration.

Impact: every G2 criterion-2 interval; the smooth arm could fail criterion 2 through estimator bias
rather than through behaviour.

Tests blocked: the four estimator tests in tests/unit/test_phenomena_p1.py (currently skipping,
because `resolve_estimators` is left unimplemented on purpose).

LEAD INTERIM RESOLUTION (2026-09-24):
Implemented now, because the regime map needs `b_hat_DP` before G1: the point estimate only, with
the conventions the frozen tests pin - density = mean counterfactual COUNT PER BIN in the window,
hole mass = counterfactual - observed ("missing mass"), bins assigned by centre, edges from
`linspace`, excess and hole windows read from `phenomena.py`, `n_obs` = all histogram inputs.
`se`, `ci_lo`, `ci_hi` are NaN. `resolve_estimators` stays unimplemented, so the estimator tests
keep skipping and nothing downstream mistakes the slice for the finished estimator. The bootstrap
and the bias question are OPEN FOR THE HUMAN and must be decided before gate G2 (WO-016/WO-020);
they do not affect G1, which uses `b_hat_DP` as a point estimate only.
