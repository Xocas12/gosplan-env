> **Labelled study G3b** (spec/P2_REVISION.md R15, R16; runs/G3_record.md). Spec 2.1.0 (trade
> offers posted at REPORT), C0 only, 30 seeds; its purpose is row 6 and the R16 exploitability
> audit. Rows 2, 5 and 7 stand as first evaluated in runs/phase2_acceptance/report.md; their
> values below are supplementary, and the rows needing the R7_NULL / R3_QW arms are not run.

# Gate G3 - Phase-2 acceptance (WO-031)

Pre-registration: `spec/P2_REVISION.md` R14. Limitations carried from Phase 1: L1 (PPO does not recover the single-enterprise DP's mixed under-reporting strategy) and L2 (the bunching estimator on degenerate and peaked distributions); see `runs/G2_record.md`.

**G3: NOT PASSED**

## Conditions

- heldout_evaluated: PASS (rows 2, 5, 6, 7 computed on the PLAN section 4.2 values; appearance is reported per row below)
- exploitability: FAIL
- oracle_gap_recorded: PASS
- jax_parity: PASS
- price_sensitivity: PASS (computed; a sign change is reported below, never suppressed)
- hygiene: PASS

## Held-out phenomena (C0, mean over seeds [95% seed-bootstrap CI])

- Row 2 storming: excess Gini 0.1180 [0.0863, 0.1501] - APPEARS
- Row 5 hoarding: request inflation excess 1.9608 [1.9402, 1.9779]; corr(X, shortfall) excess 0.5153 [0.3500, 0.6780] (0 seeds with an undefined correlation) - APPEARS
- Row 6 blat: trade volume share 0.000e+00 [0.000e+00, 0.000e+00] - FAILURE (does not appear)
- Row 7 hidden reserves: C0 0.1734 [0.0677, 0.3161]; R7_NULL nan [nan, nan] (vanishes if upper < 0.01) - FAILURE (present: True, vanishes under null: False)
- Row 3 quality (pipeline check): mean qbar C0 - R3_QW nan [nan, nan] - FAILURE

## Oracle (WO-027)

- C0: W_oracle 1.9836, val_oracle 38.1873, status optimal, solver HiGHS (HiGHS via OR-Tools 9.15.6755), optimality gap 0.00e+00, horizon 40
- R7_NULL: W_oracle 1.9836, val_oracle 38.1873, status optimal, solver HiGHS (HiGHS via OR-Tools 9.15.6755), optimality gap 0.00e+00, horizon 40
- R3_QW: W_oracle 1.9836, val_oracle 38.1873, status optimal, solver HiGHS (HiGHS via OR-Tools 9.15.6755), optimality gap 0.00e+00, horizon 40
- Clairvoyant welfare, C0 seeds 0-4 (UPPER BOUND ONLY, never a denominator): 1.9791, 1.9797, 1.9812, 1.9816, 1.9815

## Headline metrics and price sensitivity (C0)

- welfare_ratio W / W_oracle: 0.0335
- specification_gap at base prices and under price seeds (11, 12, 13): 0.0124, 0.0122, 0.0125, 0.0125 - sign change: no

## Exploitability (WO-028)

- C0: max 0.9905, median 0.7966 over 10 seeds (threshold 0.05, provisional; finalised by the lead at G3 (PLAN section 6.3)) NON-CONVERGED

## JAX parity (WO-029)

- max |NumPy - JAX| over 100 agent-steps: p1 1.07e-14, p2 7.11e-14 (tolerance 1e-05)

## Hygiene

- BOUND_BINDING runs: 0
- max training-episode runaway fraction after 20%: 0.0000 (limit 0.05)

## Arms and seeds

- C0: 30 seeds, overrides none
- R7_NULL: 0 seeds, overrides {'incentive': {'growth_directive': 0.0, 'penalty_arg': 'absolute'}}
- R3_QW: 0 seeds, overrides {'incentive': {'objective_metric': 'quality_weighted'}, 'information': {'quality_measurability': 1.0}}

Sizing: `{'n_envs': 8, 'rollout_steps': 125, 'total_agent_steps': 1000000, 'eval_every_updates': 250, 'eval_episodes': 10, 'measure_episodes': 100}`; git `347721bc2018a4126fe9b255cce51a1e060eafb2-dirty`.
