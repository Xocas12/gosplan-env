# Gate G2 criterion 1 - DP-vs-PPO recovery (WO-019)

Git: `b64e721a000306e8180b9215a15a7d34702331e3`. Setting: PLAN section 5 (`N = 1`, `a = 0`, `phi = 1`), the G1
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
| 0.8 | 0 | 0.0000 | 0.427 | 0.3645 | 0.0106 | 3.732 | no |
| 0.8 | 1 | 0.0000 | 0.406 | 0.5500 | 0.0024 | 3.763 | no |
| 0.8 | 2 | 0.0000 | 0.408 | 0.4929 | 0.0116 | 3.671 | no |
| 0.8 | 3 | 0.0000 | 0.327 | 0.6189 | 0.0005 | 3.597 | no |
| 0.8 | 4 | 0.0000 | 0.000 | 0.8838 | 0.0000 | -0.000 | no |
| 0.8 | 5 | 0.0000 | 0.056 | 0.7843 | 0.0005 | 1.922 | no |
| 0.8 | 6 | 0.0938 | 0.301 | 0.5493 | 0.0000 | 3.944 | no |
| 0.8 | 7 | 0.0000 | 0.349 | 0.4255 | 0.0068 | 3.904 | no |
| 0.8 | 8 | 0.0000 | 0.353 | 0.7155 | 0.0003 | 3.568 | no |
| 0.8 | 9 | 0.0000 | 0.241 | 0.5385 | 0.0029 | 3.548 | no |
| 4 | 0 | 0.0000 | 0.428 | 0.5591 | 0.0024 | 3.251 | no |
| 4 | 1 | 0.0000 | 0.135 | 1.0136 | 0.0000 | 1.938 | no |
| 4 | 2 | 0.0000 | 0.316 | 0.6163 | 0.0000 | 3.231 | no |
| 4 | 3 | 0.0459 | 0.206 | 0.9315 | 0.0026 | 1.556 | no |
| 4 | 4 | 0.0000 | 0.110 | 0.8498 | 0.0000 | 2.150 | no |
| 4 | 5 | 0.0000 | 0.057 | 0.7927 | 0.0003 | 1.854 | no |
| 4 | 6 | 0.0000 | 0.228 | 0.5620 | 0.0021 | 3.856 | no |
| 4 | 7 | 0.0000 | 0.294 | 0.4284 | 0.0047 | 3.585 | no |
| 4 | 8 | 0.0000 | 0.280 | 0.5003 | 0.0076 | 3.320 | no |
| 4 | 9 | 0.0000 | 0.095 | 0.7107 | 0.0011 | 2.148 | no |
| 20 | 0 | 0.0000 | 0.457 | 0.3819 | 0.0077 | 2.996 | no |
| 20 | 1 | 0.0002 | 0.119 | 0.9440 | 0.0005 | 1.658 | no |
| 20 | 2 | 0.0000 | 0.392 | 0.8332 | 0.0008 | 2.507 | no |
| 20 | 3 | 0.0000 | 0.203 | 0.8537 | 0.0003 | 2.292 | no |
| 20 | 4 | 0.0000 | 0.000 | 0.8794 | 0.0000 | -0.000 | no |
| 20 | 5 | 0.0000 | 0.239 | 0.6647 | 0.0032 | 3.205 | no |
| 20 | 6 | 0.0000 | 0.197 | 0.7228 | 0.0011 | 2.503 | no |
| 20 | 7 | 0.0000 | 0.169 | 0.6332 | 0.0050 | 2.894 | no |
| 20 | 8 | 0.0000 | 0.363 | 0.7404 | 0.0008 | 3.501 | no |
| 20 | 9 | 0.0000 | 0.244 | 0.7013 | 0.0011 | 3.029 | no |

## Result

- `a*pen` = 0.8: 0/10 seeds pass -> **FAIL**
- `a*pen` = 4: 0/10 seeds pass -> **FAIL**
- `a*pen` = 20: 0/10 seeds pass -> **FAIL**

**criterion_1_passed: False**

Flags raised: none.

Per PLAN section 4.5 a criterion-1 failure is a training-stack failure: it blocks
criteria 2-4 and the next step is a LEAD diagnosis work order, never a tolerance,
level or parameter change.

Total training wall clock: 6.8 h over 30 runs.
