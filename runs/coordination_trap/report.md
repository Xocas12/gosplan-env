# Labelled study CT - coordination-trap test (spec/P2_REVISION.md R18)

Evaluation only (no training): LC's 3M populations (primary) and G3b's 1M populations (contrast), seeds 0-9, 50 episodes per condition on the measurement seed block, deterministic policies, seat 0. Producer = `TruthfulMyopic`. Pre-registered before any CT evaluation. CT re-evaluates no gate.

## 3M populations (primary)

| seed_env | R_pop (seat 0) | R_dev (seat 0 produces) | W_pop (mean seat) | W_tm (all produce) | seats better off, all produce | seat-0 effort amid learned | amid producers |
|---|---|---|---|---|---|---|---|
| 1000 | -0.003 | -1.425 | -0.003 | 1.276 | 1.00 | 0.019 | 0.082 |
| 1001 | 1.049 | -1.448 | 0.924 | 1.315 | 0.95 | 0.023 | 0.186 |
| 1002 | -0.736 | -1.446 | -0.628 | 1.306 | 1.00 | 0.039 | 0.142 |
| 1003 | 0.610 | -1.448 | 0.711 | 1.321 | 1.00 | 0.019 | 0.159 |
| 1004 | -0.002 | -1.450 | -0.002 | 1.324 | 1.00 | 0.012 | 0.047 |
| 1005 | 0.758 | -1.448 | 0.810 | 1.304 | 1.00 | 0.018 | 0.209 |
| 1006 | 0.274 | -1.440 | 0.381 | 1.260 | 1.00 | 0.025 | 0.147 |
| 1007 | -0.000 | -1.453 | -0.000 | 1.278 | 1.00 | 0.007 | 0.006 |
| 1008 | 0.262 | -1.459 | 0.430 | 1.290 | 1.00 | 0.017 | 0.137 |
| 1009 | 0.951 | -1.470 | 0.823 | 1.311 | 1.00 | 0.019 | 0.102 |

- (i) d1 = R_dev - R_pop, median -1.718 [-2.240, -1.438]: producing alone does NOT pay (R18: CI entirely below 0).
- (ii) d2 = W_tm - W_pop, median 0.869 [0.494, 1.303]: all-production pays more (R18: CI entirely above 0).
- **Coordination trap (both): YES.**

## 1M populations (contrast)

| seed_env | R_pop (seat 0) | R_dev (seat 0 produces) | W_pop (mean seat) | W_tm (all produce) | seats better off, all produce | seat-0 effort amid learned | amid producers |
|---|---|---|---|---|---|---|---|
| 1000 | -0.036 | -1.425 | -0.044 | 1.276 | 1.00 | 0.044 | 0.136 |
| 1001 | -39.590 | -1.448 | -39.255 | 1.315 | 1.00 | 0.201 | 0.515 |
| 1002 | -0.238 | -1.446 | -0.233 | 1.306 | 1.00 | 0.042 | 0.194 |
| 1003 | -1.245 | -1.448 | -1.375 | 1.321 | 1.00 | 0.361 | 0.400 |
| 1004 | -2.330 | -1.454 | -2.838 | 1.324 | 1.00 | 0.488 | 0.513 |
| 1005 | -0.009 | -1.448 | -0.009 | 1.304 | 1.00 | 0.027 | 0.156 |
| 1006 | -35.444 | -1.440 | -44.719 | 1.260 | 1.00 | 0.100 | 0.213 |
| 1007 | -35.641 | -1.453 | -50.873 | 1.278 | 1.00 | 0.074 | 0.159 |
| 1008 | -12.044 | -1.459 | -10.014 | 1.290 | 1.00 | 0.468 | 0.441 |
| 1009 | -0.545 | -1.470 | -1.790 | 1.311 | 1.00 | 0.060 | 0.170 |

- (i) d1 = R_dev - R_pop, median 0.336 [-1.208, 34.004]: producing alone is not shown to be unprofitable (R18: CI entirely below 0).
- (ii) d2 = W_tm - W_pop, median 3.632 [1.539, 40.570]: all-production pays more (R18: CI entirely above 0).
- **Coordination trap (both): NO.**

