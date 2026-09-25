# Gate G2 criterion 1 - DP-vs-PPO recovery (WO-019)

Git: `2e5c9a74dc38bf104a06c903bdb1714e3e951c41`. Setting: PLAN section 5 (`N = 1`, `a = 0`, `phi = 1`), the G1
configuration at each recorded `a*pen` level. Sizing (AMBIGUITY-019): `RunSizing(n_envs=8, rollout_steps=125, total_agent_steps=2000000, eval_every_updates=200, eval_episodes=50, measure_episodes=400)`.
Tolerances (PLAN section 4.5): |padding| <= 0.02, |effort| <= 0.05,
W1 < 0.03; a level passes with >= 8 of 10 seeds.

## DP reference

| `a*pen` | regime | padding | effort | share in [1.00, 1.02] | DP policy return in env (context) | converged | config hash |
|---|---|---|---|---|---|---|---|
| 0.8 | bunching | 0.0055 | 0.495 | 0.8838 | 7.125 | True | `1dcece8c41c9` |
| 4 | bunching | 0.0016 | 0.498 | 0.8838 | 6.426 | True | `bfac9e5495ba` |
| 20 | bunching | 0.0006 | 0.513 | 0.8794 | 5.262 | True | `5e3b553dd8ee` |

## Per seed

| `a*pen` | seed | padding PPO | effort PPO | W1 | share PPO | return PPO | pass |
|---|---|---|---|---|---|---|---|
| 0.8 | 0 | 0.0541 | 0.753 | 0.1404 | 0.0000 | 5.613 | no |
| 0.8 | 1 | 0.2672 | 0.611 | 0.1683 | 0.0000 | 5.810 | no |
| 0.8 | 2 | 0.2589 | 0.628 | 0.1864 | 0.0016 | 5.597 | no |
| 0.8 | 3 | 0.6677 | 0.374 | 0.2379 | 0.0000 | 3.984 | no |
| 0.8 | 4 | 0.7153 | 0.223 | 0.1566 | 0.0000 | 4.338 | no |
| 0.8 | 5 | 0.2519 | 0.570 | 0.1485 | 0.0053 | 6.024 | no |
| 0.8 | 6 | 0.3365 | 0.713 | 0.2793 | 0.0000 | 4.166 | no |
| 0.8 | 7 | 0.1911 | 0.638 | 0.1422 | 0.0310 | 6.240 | no |
| 0.8 | 8 | 0.1037 | 0.687 | 0.1371 | 0.5526 | 6.253 | no |
| 0.8 | 9 | 0.3372 | 0.531 | 0.1605 | 0.0008 | 5.810 | no |
| 4 | 0 | 0.0052 | 0.868 | 0.1453 | 0.2575 | 4.381 | no |
| 4 | 1 | 0.0254 | 0.829 | 0.1573 | 0.0000 | 4.405 | no |
| 4 | 2 | 0.0038 | 0.808 | 0.1393 | 0.5859 | 5.340 | no |
| 4 | 3 | 0.0128 | 0.835 | 0.1483 | 0.1314 | 4.863 | no |
| 4 | 4 | 0.0068 | 0.794 | 0.1349 | 0.6805 | 5.629 | no |
| 4 | 5 | 0.0011 | 0.790 | 0.1311 | 0.7971 | 5.670 | no |
| 4 | 6 | 0.0033 | 0.798 | 0.1370 | 0.4713 | 5.638 | no |
| 4 | 7 | 0.0242 | 0.825 | 0.1514 | 0.0053 | 4.556 | no |
| 4 | 8 | 0.0034 | 0.777 | 0.1337 | 0.7843 | 5.833 | no |
| 4 | 9 | 0.0019 | 0.785 | 0.1334 | 0.8990 | 5.966 | no |
| 20 | 0 | 0.0000 | 0.804 | 0.1377 | 0.5144 | 5.249 | no |
| 20 | 1 | 0.0098 | 0.823 | 0.1483 | 0.0040 | 4.046 | no |
| 20 | 2 | 0.0119 | 0.868 | 0.1587 | 0.0000 | 2.911 | no |
| 20 | 3 | 0.0000 | 0.985 | 0.1584 | 0.0000 | 3.039 | no |
| 20 | 4 | 0.0000 | 0.895 | 0.1397 | 0.7363 | 4.371 | no |
| 20 | 5 | 0.0020 | 0.860 | 0.1474 | 0.1324 | 4.728 | no |
| 20 | 6 | 0.0000 | 0.647 | 0.1335 | 0.9288 | 6.360 | no |
| 20 | 7 | 0.0002 | 0.838 | 0.1456 | 0.1033 | 4.976 | no |
| 20 | 8 | 0.0000 | 0.797 | 0.1288 | 0.1790 | -1.385 | no |
| 20 | 9 | 0.0140 | 0.826 | 0.1519 | 0.0722 | 3.338 | no |

## Result

- `a*pen` = 0.8: 0/10 seeds pass -> **FAIL**
- `a*pen` = 4: 0/10 seeds pass -> **FAIL**
- `a*pen` = 20: 0/10 seeds pass -> **FAIL**

**criterion_1_passed: False**

Flags raised: none.

Per PLAN section 4.5 a criterion-1 failure is a training-stack failure: it blocks
criteria 2-4 and the next step is a LEAD diagnosis work order, never a tolerance,
level or parameter change.

Total training wall clock: 9.2 h over 30 runs.
