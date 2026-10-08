# Labelled study LC - learner convergence (spec/P2_REVISION.md R17)

C0, seeds 0-9, population budget 3M agent-steps against G3b's 1M; R16 audit unchanged (warm-started best responder, 1M steps, ratio floor 1). Pre-registered before any LC run. LC re-evaluates no gate; G3 and G3b stand as recorded.

## Primary: exploitability, seed-paired (3M - 1M)

| seed_env | 1M (G3b) | 3M (LC) | difference |
|---|---|---|---|
| 1000 | 0.125 | -0.005 | -0.130 |
| 1001 | 0.990 | -0.027 | -1.017 |
| 1002 | 0.341 | 0.429 | 0.088 |
| 1003 | 0.899 | -2.713 | -3.612 |
| 1004 | 0.972 | -0.001 | -0.973 |
| 1005 | 0.364 | -2.249 | -2.613 |
| 1006 | 0.929 | 0.074 | -0.855 |
| 1007 | -1.244 | 0.000 | 1.244 |
| 1008 | 0.983 | 0.125 | -0.859 |
| 1009 | 0.694 | -0.100 | -0.794 |

- Median exploitability: 1M 0.797, 3M -0.003.
- Median paired difference -0.857 [95% CI -1.793, -0.130].
- **Exploitability falls with budget (R17 rule: CI upper bound < 0): YES.**
- **Converged at 3M (R17 rule: max <= 0.05): NO** - max 0.429, 3 of 10 seeds above the threshold.

## Secondary (descriptive)

- welfare_ratio: 1M 0.0335, 3M 0.0000.
- Row-6 trade volume share (mean over seeds): 1M 0.000e+00, 3M 0.000e+00.
- Posted offers at REPORT, 1M (seeds 0-9, 5 episodes each): sell share 0.000, buy share 1.000, mean offer -0.953.
- Posted offers at REPORT, 3M (seeds 0-9, 5 episodes each): sell share 0.000, buy share 1.000, mean offer -1.000.

---

## Acceptance-harness output for the 3M populations

_The harness prints its gate-condition lines and a G3 verdict line. For LC they are descriptive only: LC is a labelled study and re-evaluates no gate._

## Gate G3 - Phase-2 acceptance (WO-031)

Pre-registration: `spec/P2_REVISION.md` R14. Limitations carried from Phase 1: L1 (PPO does not recover the single-enterprise DP's mixed under-reporting strategy) and L2 (the bunching estimator on degenerate and peaked distributions); see `runs/G2_record.md`.

**G3: NOT PASSED**

### Conditions

- heldout_evaluated: PASS (rows 2, 5, 6, 7 computed on the PLAN section 4.2 values; appearance is reported per row below)
- exploitability: FAIL
- oracle_gap_recorded: PASS
- jax_parity: PASS
- price_sensitivity: PASS (computed; a sign change is reported below, never suppressed)
- hygiene: PASS

### Held-out phenomena (C0, mean over seeds [95% seed-bootstrap CI])

- Row 2 storming: excess Gini 0.2197 [0.1738, 0.2800] - APPEARS
- Row 5 hoarding: request inflation excess 1.9819 [1.9518, 1.9981]; corr(X, shortfall) excess 1.0736 [1.0298, 1.1078] (0 seeds with an undefined correlation) - APPEARS
- Row 6 blat: trade volume share 0.000e+00 [0.000e+00, 0.000e+00] - FAILURE (does not appear)
- Row 7 hidden reserves: C0 0.0975 [0.0720, 0.1217]; R7_NULL nan [nan, nan] (vanishes if upper < 0.01) - NOT EVALUATED (the R7_NULL arm was not run)
- Row 3 quality (pipeline check): mean qbar C0 - R3_QW nan [nan, nan] - NOT EVALUATED (the R3_QW arm was not run)

### Oracle (WO-027)

- C0: W_oracle 1.9836, val_oracle 38.1873, status optimal, solver HiGHS (HiGHS via OR-Tools 9.15.6755), optimality gap 0.00e+00, horizon 40
- R7_NULL: W_oracle 1.9836, val_oracle 38.1873, status optimal, solver HiGHS (HiGHS via OR-Tools 9.15.6755), optimality gap 0.00e+00, horizon 40
- R3_QW: W_oracle 1.9836, val_oracle 38.1873, status optimal, solver HiGHS (HiGHS via OR-Tools 9.15.6755), optimality gap 0.00e+00, horizon 40
- Clairvoyant welfare, C0 seeds 0-4 (UPPER BOUND ONLY, never a denominator): 1.9791, 1.9797, 1.9812, 1.9816, 1.9815

### Headline metrics and price sensitivity (C0)

- welfare_ratio W / W_oracle: 0.0000
- specification_gap at base prices and under price seeds (11, 12, 13): 0.0055, 0.0055, 0.0055, 0.0055 - sign change: no

### Exploitability (WO-028)

- C0: max 0.4286, median -0.0028 over 10 seeds (threshold 0.05, provisional; finalised by the lead at G3 (PLAN section 6.3)) NON-CONVERGED

### JAX parity (WO-029)

- max |NumPy - JAX| over 100 agent-steps: p1 1.07e-14, p2 7.11e-14 (tolerance 1e-05)

### Hygiene

- BOUND_BINDING runs: 0
- max training-episode runaway fraction after 20%: 0.0000 (limit 0.05)

### Arms and seeds

- C0: 10 seeds, overrides none
- R7_NULL: 0 seeds, overrides {'incentive': {'growth_directive': 0.0, 'penalty_arg': 'absolute'}}
- R3_QW: 0 seeds, overrides {'incentive': {'objective_metric': 'quality_weighted'}, 'information': {'quality_measurability': 1.0}}

Sizing: `{'n_envs': 8, 'rollout_steps': 125, 'total_agent_steps': 3000000, 'eval_every_updates': 250, 'eval_episodes': 10, 'measure_episodes': 100}`; git `4c53d493b09f32bcc1ef7636a0f3a8edfcc2d5bc`.
