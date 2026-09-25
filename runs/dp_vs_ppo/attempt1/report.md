# Gate G2 criterion 1 - DP-vs-PPO recovery (WO-019)

Git: `69e3f2d5a67a89754971a4adc6721e87a2f5f8e5`. Setting: PLAN section 5 (`N = 1`, `a = 0`, `phi = 1`), the G1
configuration at each recorded `a*pen` level. Sizing (AMBIGUITY-019): `RunSizing(n_envs=8, rollout_steps=125, total_agent_steps=2000000, eval_every_updates=200, eval_episodes=50, measure_episodes=400)`.
Tolerances (PLAN section 4.5): |padding| <= 0.02, |effort| <= 0.05,
W1 < 0.03; a level passes with >= 8 of 10 seeds.

## DP reference

| `a*pen` | regime | padding | effort | share in [1.00, 1.02] | converged | config hash |
|---|---|---|---|---|---|---|
| 0.8 | bunching | 0.0055 | 0.495 | 0.8838 | True | `1dcece8c41c9` |
| 4 | bunching | 0.0016 | 0.498 | 0.8838 | True | `bfac9e5495ba` |
| 20 | bunching | 0.0006 | 0.513 | 0.8794 | True | `5e3b553dd8ee` |

## Per seed

| `a*pen` | seed | padding PPO | effort PPO | W1 | share PPO | return PPO | pass |
|---|---|---|---|---|---|---|---|
| 0.8 | 0 | 0.3718 | 0.541 | 0.1921 | 0.0000 | 5.173 | no |
| 0.8 | 1 | 0.7089 | 0.412 | 0.3807 | 0.0000 | 3.100 | no |
| 0.8 | 2 | 0.2148 | 0.669 | 0.1874 | 0.0000 | 5.194 | no |
| 0.8 | 3 | 0.4316 | 0.531 | 0.1768 | 0.0000 | 5.331 | no |
| 0.8 | 4 | 0.7227 | 0.243 | 0.2055 | 0.0003 | 3.964 | no |
| 0.8 | 5 | 0.3109 | 0.515 | 0.1501 | 0.0716 | 5.757 | no |
| 0.8 | 6 | 0.3427 | 0.777 | 0.3298 | 0.0000 | 3.184 | no |
| 0.8 | 7 | 0.2043 | 0.631 | 0.1437 | 0.0003 | 6.176 | no |
| 0.8 | 8 | 0.2402 | 0.631 | 0.1580 | 0.0000 | 5.874 | no |
| 0.8 | 9 | 0.5998 | 0.385 | 0.1950 | 0.0000 | 4.629 | no |
| 4 | 0 | 0.0127 | 0.814 | 0.1434 | 0.0000 | 5.127 | no |
| 4 | 1 | 0.0177 | 0.811 | 0.1456 | 0.0000 | 5.110 | no |
| 4 | 2 | 0.0001 | 0.765 | 0.1282 | 1.0000 | 6.207 | no |
| 4 | 3 | 0.0006 | 0.803 | 0.1360 | 0.6194 | 5.668 | no |
| 4 | 4 | 0.0120 | 0.792 | 0.1376 | 0.2379 | 5.564 | no |
| 4 | 5 | 0.0107 | 0.782 | 0.1366 | 0.6271 | 5.570 | no |
| 4 | 6 | 0.0101 | 0.837 | 0.1488 | 0.0000 | 4.792 | no |
| 4 | 7 | 0.0099 | 0.805 | 0.1431 | 0.3988 | 5.392 | no |
| 4 | 8 | 0.0154 | 0.822 | 0.1470 | 0.0000 | 4.957 | no |
| 4 | 9 | 0.0120 | 0.809 | 0.1432 | 0.0018 | 5.282 | no |
| 20 | 0 | 0.0033 | 0.841 | 0.1466 | 0.1816 | 4.622 | no |
| 20 | 1 | 0.0086 | 0.825 | 0.1496 | 0.0674 | 3.876 | no |
| 20 | 2 | 0.0010 | 0.805 | 0.1418 | 0.5413 | 5.573 | no |
| 20 | 3 | 0.0015 | 0.913 | 0.1572 | 0.1047 | 3.719 | no |
| 20 | 4 | 0.0000 | 0.802 | 0.1378 | 0.6910 | 5.656 | no |
| 20 | 5 | 0.0229 | 0.953 | 0.1713 | 0.0000 | 0.473 | no |
| 20 | 6 | 0.0065 | 0.790 | 0.1448 | 0.5652 | 4.353 | no |
| 20 | 7 | 0.0056 | 0.841 | 0.1450 | 0.1411 | 4.142 | no |
| 20 | 8 | 0.0186 | 0.866 | 0.1584 | 0.0000 | 1.692 | no |
| 20 | 9 | 0.0000 | 0.738 | 0.1286 | 0.1637 | -1.856 | no |

## Result

- `a*pen` = 0.8: 0/10 seeds pass -> **FAIL**
- `a*pen` = 4: 0/10 seeds pass -> **FAIL**
- `a*pen` = 20: 0/10 seeds pass -> **FAIL**

**criterion_1_passed: False**

Flags raised: none.

Per PLAN section 4.5 a criterion-1 failure is a training-stack failure: it blocks
criteria 2-4 and the next step is a LEAD diagnosis work order, never a tolerance,
level or parameter change.

Total training wall clock: 7.3 h over 30 runs.
