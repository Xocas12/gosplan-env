# Labelled study ES - budget versus exploration schedule (spec/P2_REVISION.md R19)

C0, seeds 0-9, 3M agent-steps with the entropy bonus annealed over the first 1M steps (the 1M runs' schedule) and held at 0.001 after; R16 audit unchanged. Pre-registered before any ES run. ES re-evaluates no gate.

## Primary: does the collapse reproduce under the matched schedule?

| seed | final mean effort |
|---|---|
| 0 | 0.012 |
| 1 | 0.014 |
| 2 | 0.022 |
| 3 | 0.022 |
| 4 | 0.011 |
| 5 | 0.019 |
| 6 | 0.025 |
| 7 | 0.024 |
| 8 | 0.019 |
| 9 | 0.010 |

- Median final effort 0.019 [95% CI 0.012, 0.023].
- **Collapse reproduced (R19 rule: CI upper bound < 0.05): YES.**

## Secondary (descriptive)

- welfare_ratio: ES 0.0000; LC 3M 0.0000.
- R16 exploitability, ES: median -0.008, max 0.173 (1 of 10 above 5%); LC 3M median -0.003.
- R18 coordination-trap test on the ES populations: (i) median d1 -1.785 [-2.056, -1.449]; (ii) median d2 1.111 [0.678, 1.295]; trap: YES.

### Training trajectories (median over seeds of the periodic evaluation)

**1M (G3b)** - k-steps [250, 500, 750, 1000]

- entropy coef: [0.0078, 0.0055, 0.0033, 0.001]
- median effort: [0.662, 0.615, 0.267, 0.089]
- median eval return: [-206.67, -68.71, -56.57, -4.59]

**3M, stretched schedule (LC)** - k-steps [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000]

- entropy coef: [0.0093, 0.0085, 0.0078, 0.007, 0.0063, 0.0055, 0.0048, 0.004, 0.0033, 0.0025, 0.0018, 0.001]
- median effort: [0.663, 0.603, 0.348, 0.116, 0.033, 0.023, 0.022, 0.022, 0.024, 0.025, 0.022, 0.02]
- median eval return: [-190.82, -42.93, -20.4, -5.93, -1.31, -0.01, -0.01, 0.49, 0.69, 0.69, 0.85, 0.71]

**3M, matched schedule (ES)** - k-steps [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000]

- entropy coef: [0.0078, 0.0055, 0.0033, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
- median effort: [0.662, 0.615, 0.267, 0.089, 0.043, 0.028, 0.021, 0.019, 0.022, 0.023, 0.023, 0.022]
- median eval return: [-206.67, -68.71, -56.57, -4.59, -0.8, -0.75, -0.01, -0.0, 0.03, 0.29, 0.36, 0.35]

---

## Acceptance-harness output for the ES populations

_The harness prints its gate-condition lines and a G3 verdict line. For ES they are descriptive only._

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

- Row 2 storming: excess Gini 0.2219 [0.1711, 0.2743] - APPEARS
- Row 5 hoarding: request inflation excess 1.7899 [1.6486, 1.9174]; corr(X, shortfall) excess 1.0329 [0.9270, 1.1169] (0 seeds with an undefined correlation) - APPEARS
- Row 6 blat: trade volume share 0.000e+00 [0.000e+00, 0.000e+00] - FAILURE (does not appear)
- Row 7 hidden reserves: C0 0.0842 [0.0622, 0.1108]; R7_NULL nan [nan, nan] (vanishes if upper < 0.01) - NOT EVALUATED (the R7_NULL arm was not run)
- Row 3 quality (pipeline check): mean qbar C0 - R3_QW nan [nan, nan] - NOT EVALUATED (the R3_QW arm was not run)

### Oracle (WO-027)

- C0: W_oracle 1.9836, val_oracle 38.1873, status optimal, solver HiGHS (HiGHS via OR-Tools 9.15.6755), optimality gap 0.00e+00, horizon 40
- R7_NULL: W_oracle 1.9836, val_oracle 38.1873, status optimal, solver HiGHS (HiGHS via OR-Tools 9.15.6755), optimality gap 0.00e+00, horizon 40
- R3_QW: W_oracle 1.9836, val_oracle 38.1873, status optimal, solver HiGHS (HiGHS via OR-Tools 9.15.6755), optimality gap 0.00e+00, horizon 40
- Clairvoyant welfare, C0 seeds 0-4 (UPPER BOUND ONLY, never a denominator): 1.9791, 1.9797, 1.9812, 1.9816, 1.9815

### Headline metrics and price sensitivity (C0)

- welfare_ratio W / W_oracle: 0.0000
- specification_gap at base prices and under price seeds (11, 12, 13): 0.0059, 0.0059, 0.0059, 0.0058 - sign change: no

### Exploitability (WO-028)

- C0: max 0.1726, median -0.0082 over 10 seeds (threshold 0.05, provisional; finalised by the lead at G3 (PLAN section 6.3)) NON-CONVERGED

### JAX parity (WO-029)

- max |NumPy - JAX| over 100 agent-steps: p1 1.07e-14, p2 7.11e-14 (tolerance 1e-05)

### Hygiene

- BOUND_BINDING runs: 0
- max training-episode runaway fraction after 20%: 0.0000 (limit 0.05)

### Arms and seeds

- C0: 10 seeds, overrides none
- R7_NULL: 0 seeds, overrides {'incentive': {'growth_directive': 0.0, 'penalty_arg': 'absolute'}}
- R3_QW: 0 seeds, overrides {'incentive': {'objective_metric': 'quality_weighted'}, 'information': {'quality_measurability': 1.0}}

Sizing: `{'n_envs': 8, 'rollout_steps': 125, 'total_agent_steps': 3000000, 'eval_every_updates': 250, 'eval_episodes': 10, 'measure_episodes': 100}`; git `451fd86718257e516baefc1046464b1f63718f53`.
