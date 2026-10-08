# Estimator-bias study (WO-034, PLAN section 7.2)

Design: spec/P3_REVISION.md S4. Truth: (a) the DP's exact stationary distribution, (b) the arm's pooled simulation, each measured against its own w = 0.25 counterpart at the same cap. Reused Phase-1 gate arms: 30 seeds; other arms: 5 seeds (reduced for compute). Estimator: gosplan.metrics._fallback 0.1.0+fallback. No digit tests; no calibration claim about archival detectors.

## Pre-registered setting (excl[0.95,1.02]|deg9|bw0.005)

| w | cap | DP P(rho in [1.00,1.02]) | DP truth | sim truth | defined / seeds | sim bias | sim RMSE | sim median error | CI coverage |
|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 1.2 | 0.879 | undefined | 64.189 | 18 / 30 | 40497.882 | 97089.650 | 25.524 | 0.00 |
| 0.0 | inf | 0.879 | undefined | 72.461 | 3 / 5 | -41.530 | 48.069 | -52.385 | 0.00 |
| 0.02 | 1.2 | 0.000 | undefined | -3.405 | 5 / 5 | -0.486 | 0.530 | -0.592 | 0.20 |
| 0.02 | inf | 0.000 | undefined | -2.883 | 5 / 5 | -0.931 | 0.950 | -0.999 | 0.00 |
| 0.05 | 1.2 | 0.000 | undefined | -2.293 | 5 / 5 | -0.770 | 1.080 | -0.735 | 0.40 |
| 0.05 | inf | 0.000 | undefined | -1.710 | 5 / 5 | -1.292 | 1.568 | -1.594 | 0.00 |
| 0.1 | 1.2 | 0.000 | undefined | -0.643 | 5 / 5 | -1.383 | 1.682 | -1.409 | 0.20 |
| 0.1 | inf | 0.000 | undefined | -1.468 | 5 / 5 | -0.525 | 1.466 | -0.462 | 0.20 |
| 0.25 | 1.2 | 0.000 | undefined | 0.000 | 5 / 5 | 0.368 | 0.912 | 0.201 | 0.40 |
| 0.25 | inf | 0.000 | undefined | 0.000 | 30 / 30 | 0.078 | 1.692 | 0.078 | 0.33 |

Notes (read before the numbers):

- **DP truth is undefined in the estimator's units on every arm.** The DP's reports lie on its 0.02 report grid, so its stationary distribution is a set of point masses. Every smooth (w = 0.25) counterpart puts zero mass in the excess window [1.00, 1.02], and S4's truth divides by that counterpart's mean per-bin mass there. The DP column therefore reports the defined quantity, the DP's probability of a report in the window. The estimator is not scored against the DP (the estimator on a 0.02-grid point mass is degenerate at bin width 0.005 or 0.01 by construction).
- **Undefined estimates are counted, not dropped silently.** When a seed's measured mass falls almost entirely inside the excluded window, the polynomial counterfactual has no support and the estimate is infinite or its CI undefined (AMBIGUITY-022, as in the Phase-1 gate report). `defined / seeds` counts the seeds that enter bias, RMSE and median error. Near-zero but positive counterfactual support gives finite but huge estimates, which dominate the mean; the median error is shown beside it as a supplementary, robust summary (added when reporting, not pre-registered).
- These are the study's findings about the estimator under full bunching; nothing was re-tuned (PLAN section 4.5).

## Across settings (simulation; mean |bias| over arms, mean coverage)

| setting | mean abs bias | mean RMSE | mean CI coverage |
|---|---|---|---|
| excl[0.95,1.02]|deg5|bw0.005 | 8931.370 | 21398.816 | 0.10 |
| excl[0.95,1.02]|deg5|bw0.01 | 4473.265 | 10717.567 | 0.10 |
| excl[0.95,1.02]|deg7|bw0.005 | 5797.919 | 13891.497 | 0.13 |
| excl[0.95,1.02]|deg7|bw0.01 | 2910.992 | 6974.567 | 0.13 |
| excl[0.95,1.02]|deg9|bw0.005 (pre-registered) | 4054.525 | 9714.760 | 0.17 |
| excl[0.95,1.02]|deg9|bw0.01 | 2042.768 | 4894.526 | 0.17 |
| excl[0.97,1.02]|deg5|bw0.005 | 9848.567 | 23593.672 | 0.10 |
| excl[0.97,1.02]|deg5|bw0.01 | 4928.152 | 11806.100 | 0.10 |
| excl[0.97,1.02]|deg7|bw0.005 | 6623.540 | 15868.038 | 0.09 |
| excl[0.97,1.02]|deg7|bw0.01 | 3319.295 | 7952.037 | 0.09 |
| excl[0.97,1.02]|deg9|bw0.005 | 4790.926 | 11478.110 | 0.15 |
| excl[0.97,1.02]|deg9|bw0.01 | 2405.982 | 5764.251 | 0.16 |
| excl[0.93,1.03]|deg5|bw0.005 | 302.670 | 554.173 | 0.11 |
| excl[0.93,1.03]|deg5|bw0.01 | 151.913 | 278.157 | 0.11 |
| excl[0.93,1.03]|deg7|bw0.005 | 185.291 | 344.397 | 0.11 |
| excl[0.93,1.03]|deg7|bw0.01 | 93.403 | 173.598 | 0.11 |
| excl[0.93,1.03]|deg9|bw0.005 | 122.138 | 231.809 | 0.14 |
| excl[0.93,1.03]|deg9|bw0.01 | 61.959 | 117.554 | 0.14 |

## Reconciliation power curve

`ledger_test` at alpha = 0.05, claims inflated by x1.2 on the given share of enterprise-periods, 20 TruthfulMyopic episodes:

- share 0.0: rejection rate 0.05 (size)
- share 0.05: rejection rate 0.90
- share 0.1: rejection rate 1.00
- share 0.2: rejection rate 1.00
- share 0.3: rejection rate 1.00
- share 0.5: rejection rate 1.00

Figures: `bias_curves.png`, `power_curve.png`.
