# Gate G2 criteria 2-4 - Phase-1 gate (WO-020)

**LABELLED STUDY - criterion 1 NOT met** (`runs/dp_vs_ppo/report.md`, AMBIGUITY-021). By the owner's decision these criteria are run and reported, but they are not a G2 pass.

Git: `0b7a96ebde2b5727fdc1ff9854f8807f977a28e3`. Configuration: G1 record at `N = 20`, `a*pen` = 20.
Sizing (AMBIGUITY-019): `RunSizing(n_envs=8, rollout_steps=125, total_agent_steps=1000000, eval_every_updates=250, eval_episodes=10, measure_episodes=100)`; 10 seeds per extra level.
Learner (AMBIGUITY-021, owner's choice): report-head init std 0.05 in ratio units, per-period discount - the attempt-2 learner.
Estimator: `gosplan.metrics._fallback` 0.1.0+fallback, PLAN section 4.5 settings (degree 9).

## Criterion 2 - bunching present / absent

Notched condition: learned share in [1.00, 1.02] >= 0.440 and CI of
`b_hat` excluding 0. Smooth condition: CI of `b_hat` covers 0.

| arm | seed | b_hat | CI | hole | share | padding | effort | pass |
|---|---|---|---|---|---|---|---|---|
| notched | 0 | inf | [nan, nan] | -inf | 1.000 | 0.0000 | 0.644 | undefined |
| notched | 1 | 7126.810 | [5017.699, 11291.234] | 10.000 | 0.998 | 0.0000 | 0.644 | pass |
| notched | 2 | 278657.296 | [78780.525, 308513.337] | 10.000 | 1.000 | 0.0000 | 0.632 | pass |
| notched | 3 | inf | [nan, nan] | nan | 1.000 | 0.0000 | 0.639 | undefined |
| notched | 4 | -4.000 | [-4.000, -4.000] | 10.000 | 0.000 | 0.0031 | 0.712 | fail |
| notched | 5 | inf | [nan, nan] | nan | 1.000 | 0.0000 | 0.634 | undefined |
| notched | 6 | 112.401 | [105.590, 119.165] | 10.000 | 0.887 | 0.0000 | 0.633 | pass |
| notched | 7 | -1.751 | [-1.843, -1.632] | 8.750 | 0.109 | 0.0000 | 0.536 | fail |
| notched | 8 | 5.800 | [4.871, 6.676] | 8.558 | 0.340 | 0.0000 | 0.554 | fail |
| notched | 9 | inf | [nan, nan] | nan | 1.000 | 0.0000 | 0.641 | undefined |
| notched | 10 | inf | [nan, nan] | nan | 1.000 | 0.0000 | 0.634 | undefined |
| notched | 11 | 1.931 | [1.757, 2.093] | 10.000 | 0.284 | 0.0000 | 0.702 | fail |
| notched | 12 | inf | [nan, nan] | nan | 1.000 | 0.0000 | 0.637 | undefined |
| notched | 13 | inf | [nan, nan] | -inf | 1.000 | 0.0000 | 0.635 | undefined |
| notched | 14 | 28644.249 | [18654.461, 57249.783] | 10.000 | 0.999 | 0.0000 | 0.637 | pass |
| notched | 15 | 286021.884 | [78863.377, 313889.486] | 10.000 | 1.000 | 0.0000 | 0.648 | pass |
| notched | 16 | inf | [nan, nan] | -inf | 0.918 | 0.0000 | 0.620 | undefined |
| notched | 17 | 29771.849 | [17720.791, 80675.311] | -1824.973 | 0.948 | 0.0000 | 0.622 | pass |
| notched | 18 | 7384.620 | [5303.317, 12461.046] | 10.000 | 0.998 | 0.0000 | 0.647 | pass |
| notched | 19 | 86.136 | [74.574, 101.566] | -0.485 | 0.789 | 0.0000 | 0.584 | pass |
| notched | 20 | 92185.918 | [41892.091, 287877.760] | 10.000 | 1.000 | 0.0000 | 0.643 | pass |
| notched | 21 | inf | [nan, nan] | nan | 1.000 | 0.0000 | 0.626 | undefined |
| notched | 22 | 36.379 | [34.947, 37.674] | 10.000 | 0.731 | 0.0000 | 0.675 | pass |
| notched | 23 | 1.583 | [0.667, 2.472] | 10.000 | 0.260 | 0.0000 | 0.674 | fail |
| notched | 24 | 93.289 | [76.511, 123.083] | 10.000 | 0.867 | 0.0002 | 0.653 | pass |
| notched | 25 | inf | [nan, nan] | nan | 1.000 | 0.0000 | 0.632 | undefined |
| notched | 26 | inf | [nan, nan] | -inf | 0.823 | 0.0000 | 0.577 | undefined |
| notched | 27 | -3.132 | [-3.272, -3.003] | 10.000 | 0.054 | 0.0003 | 0.703 | fail |
| notched | 28 | -3.992 | [-3.997, -3.986] | 10.000 | 0.000 | 0.0003 | 0.654 | fail |
| notched | 29 | inf | [nan, nan] | nan | 1.000 | 0.0000 | 0.639 | undefined |
| smooth | 0 | 1.033 | [0.721, 1.344] | -2.178 | 0.074 | 0.0000 | 0.485 | fail |
| smooth | 1 | 1.129 | [0.521, 1.798] | -0.403 | 0.017 | 0.0000 | 0.582 | fail |
| smooth | 2 | -0.654 | [-1.226, -0.017] | 1.629 | 0.008 | 0.0001 | 0.555 | fail |
| smooth | 3 | -0.454 | [-0.962, 0.067] | 1.779 | 0.009 | 0.0001 | 0.585 | pass |
| smooth | 4 | -6.773 | [-9.270, -5.433] | 17.804 | 0.001 | 0.0000 | 0.648 | fail |
| smooth | 5 | 2.044 | [1.714, 2.351] | -4.762 | 0.126 | 0.0000 | 0.507 | fail |
| smooth | 6 | 1.288 | [0.138, 2.887] | -7.001 | 0.005 | 0.0000 | 0.646 | fail |
| smooth | 7 | -0.686 | [-1.201, -0.132] | 0.681 | 0.010 | 0.0000 | 0.591 | fail |
| smooth | 8 | 0.382 | [-0.453, 1.328] | -0.469 | 0.007 | 0.0000 | 0.624 | pass |
| smooth | 9 | -1.065 | [-1.524, -0.564] | 1.420 | 0.010 | 0.0002 | 0.594 | fail |
| smooth | 10 | 0.222 | [-0.393, 0.943] | -0.810 | 0.010 | 0.0000 | 0.556 | pass |
| smooth | 11 | 0.458 | [-0.101, 0.992] | -0.648 | 0.015 | 0.0000 | 0.550 | pass |
| smooth | 12 | -1.591 | [-1.766, -1.368] | 6.300 | 0.075 | 0.0000 | 0.441 | fail |
| smooth | 13 | -0.345 | [-0.905, 0.225] | 1.419 | 0.010 | 0.0000 | 0.544 | pass |
| smooth | 14 | 0.796 | [0.509, 1.085] | -2.564 | 0.066 | 0.0000 | 0.457 | fail |
| smooth | 15 | 1.697 | [1.421, 2.001] | 1.323 | 0.133 | 0.0000 | 0.471 | fail |
| smooth | 16 | -0.016 | [-0.576, 0.614] | -0.578 | 0.013 | 0.0000 | 0.601 | pass |
| smooth | 17 | 1.656 | [1.376, 1.939] | -4.117 | 0.111 | 0.0000 | 0.472 | fail |
| smooth | 18 | -0.923 | [-1.351, -0.466] | 2.112 | 0.009 | 0.0000 | 0.589 | fail |
| smooth | 19 | -0.629 | [-1.029, -0.228] | -0.005 | 0.015 | 0.0000 | 0.562 | fail |
| smooth | 20 | 0.533 | [0.260, 0.860] | 4.126 | 0.103 | 0.0000 | 0.427 | fail |
| smooth | 21 | -0.572 | [-1.068, -0.030] | 0.710 | 0.013 | 0.0001 | 0.609 | fail |
| smooth | 22 | 0.530 | [-0.010, 1.168] | -0.107 | 0.014 | 0.0001 | 0.589 | pass |
| smooth | 23 | -0.655 | [-1.221, -0.097] | -0.135 | 0.008 | 0.0001 | 0.598 | fail |
| smooth | 24 | 0.083 | [-0.109, 0.285] | 3.299 | 0.107 | 0.0000 | 0.481 | pass |
| smooth | 25 | 1.502 | [1.210, 1.780] | 0.500 | 0.092 | 0.0000 | 0.448 | fail |
| smooth | 26 | -0.660 | [-1.013, -0.301] | 1.011 | 0.021 | 0.0000 | 0.513 | fail |
| smooth | 27 | 0.052 | [-0.344, 0.512] | 2.740 | 0.021 | 0.0000 | 0.474 | pass |
| smooth | 28 | 3.872 | [1.293, 8.794] | -165.030 | 0.004 | 0.0001 | 0.644 | fail |
| smooth | 29 | 0.072 | [-0.426, 0.651] | 0.135 | 0.014 | 0.0000 | 0.606 | pass |

- notched: 37% of seeds pass (need >= 90%); 77% if an undefined CI counts as met
- smooth: 33% of seeds pass (need >= 90%); 33% if an undefined CI counts as met
- seeds with an undefined CI (all measured mass inside the excluded window, so the polynomial counterfactual has no support; AMBIGUITY-022): 12
- **criterion_2 (strict: undefined = not met): FAIL**; with undefined counted as met: FAIL

## Criterion 3 - padding elasticity in `a*pen`

| `a*pen` | learned padding (mean over seeds) | DP padding | deviation |
|---|---|---|---|
| 0.8 | 0.1083 | 0.0055 | 0.1029 |
| 4 | 0.0019 | 0.0016 | 0.0003 |
| 20 | 0.0001 | 0.0006 | 0.0004 |

- monotone decreasing: True; tolerance 0.03
- **criterion_3: FAIL**

## Criterion 4 - hygiene

- runs with `BOUND_BINDING`: 0 of 80
- max share of training episodes with `T > 3 T_0` after the first 20% of training: 0.1085 (need < 0.05)
- **criterion_4: FAIL**

## Price sensitivity

PLAN section 7.5's standing price check recomputes padding_index, welfare_ratio and specification_gap; the last two need the Phase-2 oracle (WO-026) and the check itself is WO-036 (Phase 3). No G2 criterion is price-weighted, so the table is deferred to WO-036, which must run it on this gate's ledgers (AMBIGUITY-019).

**All three criteria passed: False**. Flags raised: none.

If criterion 1 passed in `runs/dp_vs_ppo/report.md`, a criterion-2 failure is a
multi-agent effect and is a result, reported, never tuned away (PLAN section 4.5).

Total training wall clock: 23.9 h over 80 runs.
