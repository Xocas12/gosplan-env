# CT - self-enforcement and the lone producer's payoff (descriptive, post hoc)

Not pre-registered; no decision rule is attached (#106). Every seat here is a heuristic, so no checkpoint is needed. C0, seeds 0-9, CT's protocol: 50 episodes per condition on the measurement seed block, the measurement window, seat 0. Producer = `TruthfulMyopic` (quality 1). Inert = the all-zero action (quality 0).

Conditions:

- `R_alone`: seat 0 producer; the other seats inert (the all-zero action, quality 0).
- `R_amid_q0`: seat 0 producer; the other seats producers at quality 0.
- `R_producers`: every seat a producer (quality 1).
- `R_producers_seat0_q0`: every seat a producer; seat 0 at quality 0.

## R18 leg (i) against the inert-opponent payoff

`R_dev` is seat 0's return as a producer among each trained population, as committed by CT (LC 3M, G3b 1M) and ES. `|diff|` is `|R_dev - R_alone|`.

| seed_env | R_alone | R_amid_q0 | LC 3M R_dev | diff | ES 3M R_dev | diff | G3b 1M R_dev | diff |
|---|---|---|---|---|---|---|---|---|
| 1000 | -1.425313 | -1.425313 | -1.425313 | 0.0e+00 | -1.425313 | 0.0e+00 | -1.425313 | 0.0e+00 |
| 1001 | -1.447976 | -1.447976 | -1.447976 | 0.0e+00 | -1.447976 | 0.0e+00 | -1.447976 | 0.0e+00 |
| 1002 | -1.445657 | -1.445657 | -1.445657 | 0.0e+00 | -1.445657 | 0.0e+00 | -1.445657 | 0.0e+00 |
| 1003 | -1.448223 | -1.448223 | -1.448223 | 0.0e+00 | -1.448223 | 0.0e+00 | -1.448223 | 0.0e+00 |
| 1004 | -1.449843 | -1.449843 | -1.449843 | 0.0e+00 | -1.449843 | 0.0e+00 | -1.454481 | 4.6e-03 |
| 1005 | -1.447664 | -1.447664 | -1.447664 | 0.0e+00 | -1.447664 | 0.0e+00 | -1.447664 | 0.0e+00 |
| 1006 | -1.439691 | -1.439691 | -1.439691 | 0.0e+00 | -1.439691 | 0.0e+00 | -1.439691 | 0.0e+00 |
| 1007 | -1.453022 | -1.453022 | -1.453022 | 0.0e+00 | -1.453022 | 0.0e+00 | -1.453022 | 0.0e+00 |
| 1008 | -1.459152 | -1.459152 | -1.459152 | 0.0e+00 | -1.459152 | 0.0e+00 | -1.459152 | 2.2e-16 |
| 1009 | -1.469815 | -1.469815 | -1.469815 | 0.0e+00 | -1.469815 | 0.0e+00 | -1.469815 | 4.4e-16 |

## Mean quality of the trained populations

`qbar`: the acceptance harness's mean period-average quality (`quality.mean_quality` in each study's `result.json`) for the same populations.

| seed_env | LC 3M | ES 3M | G3b 1M |
|---|---|---|---|
| 1000 | 1.64e-15 | 3.26e-06 | 2.98e-01 |
| 1001 | 4.96e-06 | 8.47e-04 | 4.79e-01 |
| 1002 | 4.31e-06 | 7.78e-04 | 2.67e-01 |
| 1003 | 1.91e-05 | 1.99e-03 | 5.84e-01 |
| 1004 | 9.99e-12 | 8.75e-07 | 7.47e-01 |
| 1005 | 1.94e-09 | 1.31e-07 | 5.81e-02 |
| 1006 | 1.11e-05 | 3.12e-03 | 3.25e-01 |
| 1007 | 2.85e-15 | 4.08e-06 | 1.20e-01 |
| 1008 | 3.40e-05 | 1.86e-04 | 5.96e-01 |
| 1009 | 2.47e-08 | 4.35e-07 | 3.97e-01 |

## Does all-production hold by itself? Seat 0 drops quality among producers

| seed_env | R_producers | R_producers_seat0_q0 | quality_gain |
|---|---|---|---|
| 1000 | 1.210565 | 2.247307 | +1.0367 |
| 1001 | 1.191086 | 2.251202 | +1.0601 |
| 1002 | 1.154870 | 2.203874 | +1.0490 |
| 1003 | 1.202650 | 2.259949 | +1.0573 |
| 1004 | 1.198429 | 2.257341 | +1.0589 |
| 1005 | 1.217972 | 2.266195 | +1.0482 |
| 1006 | 1.096565 | 2.121135 | +1.0246 |
| 1007 | 1.173955 | 2.232665 | +1.0587 |
| 1008 | 1.215107 | 2.279637 | +1.0645 |
| 1009 | 1.187176 | 2.272165 | +1.0850 |

- quality_gain: min +1.0246, max +1.0850; positive on 10 of 10 seeds.
- LC 3M: R_dev equals R_alone (to 1e-9) on 10 of 10 seeds; qbar 1.64e-15 to 3.40e-05.
- ES 3M: R_dev equals R_alone (to 1e-9) on 10 of 10 seeds; qbar 1.31e-07 to 3.12e-03.
- G3b 1M: R_dev equals R_alone (to 1e-9) on 9 of 10 seeds; qbar 5.81e-02 to 7.47e-01.
