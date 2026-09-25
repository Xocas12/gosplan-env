# Monte-Carlo sanity report (WO-012, gate G0 artefact)

This report is an input to gate G0; it is not the gate. G0 is the lead's written
sign-off on this report together with a green frozen suite and the CONTRACT rule 7
diff review of `gosplan/env/` (PLAN section 13).

## Held-out prohibition this harness ran under

`HELD_OUT_PHENOMENON_ROWS = (2, 5, 6, 7)`. Restated verbatim from the
module docstring of `gosplan/experiments/mc_sanity.py`:

```
FORBIDDEN - HELD-OUT PHENOMENA (PLAN sections 4.1 and 12.3, WO-012 card)
    ***  THIS HARNESS MUST NEVER COMPUTE, TABULATE, PLOT OR ASSERT A DIRECTION FOR ANY QUANTITY ***
    ***  IN PLAN SECTION 4.1 ROWS 2, 5, 6 OR 7.                                                 ***

    Row 2 storming - the within-period effort Gini `Gini_k(e_ik)` and any excess over a baseline.
    Row 5 hoarding - request inflation `q_ij / need_ij`, input stocks `X_ij`, and any correlation
                     between held inputs and downstream fill.
    Row 6 blat     - executed trade volume, matched pairs or trade surplus.
    Row 7 hidden reserves - `max(0, S_i - R_i) / T_i`, and any reconciliation statistic on it.

    "Inspected" means: no plot, no table, no test, not in passing and not while debugging the
    mechanisms behind them (PLAN section 4.1). They are computed for the first time in the Phase-2
    acceptance run (WO-030, WO-031). The Monte-Carlo checks that touch the same mechanisms assert
    only conservation and boundedness, never a direction.

    The one directional assertion this harness does make - that `Padder` produces downstream
    shortage - is about `fill`, the delivery channel of PLAN section 2.7.3, and is the property test
    T-B3 already states. It is not row 5: no `X_ij`, no request inflation and no correlation with
    held stock may be computed to support it.
```

## Run summary

- overall: **ALL ASSERTIONS HELD**
- seed_env (shared by every agent and configuration; CRN): 0
- seed_policy: 0 (one generator per configuration x agent cell)
- perturbation generator: `numpy.random.default_rng(0)` (the harness seed)
- configurations: 21 (1 baseline + 20 SUPPLY perturbations)
- agents: Random, TruthfulMyopic, Padder; episodes per cell: 2000
- theta = inf (Leontief branch) exercised: yes
- max conservation error: 3.553e-15 (tolerance 1e-09) - HELD
- non-finite values: 0 - HELD
- T bound failures: 0; S bound failures: 0; fill bound failures: 0
- Padder minimum fill: 0.188839; shortage in every configuration: True
- mean wall clock per episode: 0.0619 s over 126000 episodes; harness total 9427.9 s
- flags raised: none (`BOUND_BINDING` is CONTRACT rule 8)
- git: 69e3f2d5a67a89754971a4adc6721e87a2f5f8e5-dirty

Conventions. Conservation is the per-period, per-good T-U1 identity in the form recorded
in `spec/CHANGELOG.md` 0.1.3 (ambiguity #64), which carries the input stocks explicitly:
`sum y + sum S_prev + sum X_prev = sum S_next + sum X_next + sum consumed + consumer +
sum holding + sum overflow`, with `S_prev`, `X_prev` the previous period's REPORT row
(the reset state in period 0). Only the residual is reported. The non-finite scan covers
every numeric ledger column; in `coverage`, `audit_meas`, `penalty_arg` NaN is
the placeholder ruled in AMBIGUITY-007 (their owning functions do not return them) and is
counted separately, while inf there still counts as non-finite. `T` bounds are
`[target_floor_frac * T_0, T_0 * ((1 + g) * (1 + lambda * c_up))**P_max]` per
enterprise; `S` bounds `[0, inventory_cap_mult * cap_i]` on both `inv_output_pre` and
`inv_output_post`; `fill` bounds `FILL_BOUNDS` on every row.

## Configuration 0: `22a662ee58e71543bda1f4671392908b876d44f94c64f0784a3a72670ca0b02e`

- run directory: `runs/22a662ee58e71543bda1f4671392908b876d44f94c64f0784a3a72670ca0b02e/` (manifest.json, ledger/<agent>/)
- kind: baseline (`p1_default_config()`)
- yield_sigma: (0.05, 0.08, 0.1, 0.12, 0.15)
- final_demand_share: 0.500000 (every sector)
- holding_loss: 0.020000
- input_complementarity: 8.0
- locked (PERTURBATION_EXCLUDED) at Phase-1 values: delivery_timing, arrival_probs, input_holding_loss, trade_tau, alloc_eta_request

### Configuration 0 x Random

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0612 s, max 0.1801 s
- conservation: max |residual| 2.665e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.326166, max 2.30688; min T/T_min 10.8722, max T/T_upper 0.132645; failures 0 - HELD
- S: min 0, max S/S_max 0.575425 (S_max 3); failures 0 - HELD
- fill: min 0.00155929, max 1; failures 0 - HELD
- flags: none

### Configuration 0 x TruthfulMyopic

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0620 s, max 0.1460 s
- conservation: max |residual| 1.776e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.576154, max 0.851059; min T/T_min 19.2051, max T/T_upper 0.0489356; failures 0 - HELD
- S: min 0, max S/S_max 0.300441 (S_max 3); failures 0 - HELD
- fill: min 1, max 1; failures 0 - HELD
- flags: none

### Configuration 0 x Padder

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0619 s, max 0.1579 s
- conservation: max |residual| 8.882e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.6, max 0.708528; min T/T_min 20, max T/T_upper 0.0407401; failures 0 - HELD
- S: min 0, max S/S_max 0.124424 (S_max 3); failures 0 - HELD
- fill: min 0.390689, max 1; failures 0 - HELD
- Padder shortage (T-B3 property, fill < 1 somewhere): True - HELD
- flags: none

## Configuration 1: `2e62e93e6513ad2cc79ad465fbee0a22022b4afe349df7a5222eaaa4e96f8d78`

- run directory: `runs/2e62e93e6513ad2cc79ad465fbee0a22022b4afe349df7a5222eaaa4e96f8d78/` (manifest.json, ledger/<agent>/)
- kind: SUPPLY perturbation
- yield_sigma: (0.072772, 0.116435, 0.145544, 0.174653, 0.218316) (x1.455443)
- final_demand_share: 0.407915 (every sector)
- holding_loss: 0.002049
- input_complementarity: 2.0
- locked (PERTURBATION_EXCLUDED) at Phase-1 values: delivery_timing, arrival_probs, input_holding_loss, trade_tau, alloc_eta_request

### Configuration 1 x Random

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0627 s, max 0.1600 s
- conservation: max |residual| 3.553e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.326166, max 2.30688; min T/T_min 10.8722, max T/T_upper 0.132645; failures 0 - HELD
- S: min 0, max S/S_max 0.61905 (S_max 3); failures 0 - HELD
- fill: min 0.00157675, max 1; failures 0 - HELD
- flags: none

### Configuration 1 x TruthfulMyopic

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0627 s, max 0.1488 s
- conservation: max |residual| 1.776e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.536625, max 0.941647; min T/T_min 17.8875, max T/T_upper 0.0541444; failures 0 - HELD
- S: min 0, max S/S_max 0.343581 (S_max 3); failures 0 - HELD
- fill: min 1, max 1; failures 0 - HELD
- flags: none

### Configuration 1 x Padder

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0620 s, max 0.1283 s
- conservation: max |residual| 8.882e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.6, max 0.708528; min T/T_min 20, max T/T_upper 0.0407401; failures 0 - HELD
- S: min 0, max S/S_max 0.137446 (S_max 3); failures 0 - HELD
- fill: min 0.372987, max 1; failures 0 - HELD
- Padder shortage (T-B3 property, fill < 1 somewhere): True - HELD
- flags: none

## Configuration 2: `c82a05d13de3b484e5a336af8a57415d45829ddd82fa44e413bbb63928812f8d`

- run directory: `runs/c82a05d13de3b484e5a336af8a57415d45829ddd82fa44e413bbb63928812f8d/` (manifest.json, ledger/<agent>/)
- kind: SUPPLY perturbation
- yield_sigma: (0.085995, 0.137592, 0.171991, 0.206389, 0.257986) (x1.719905)
- final_demand_share: 0.665102 (every sector)
- holding_loss: 0.030332
- input_complementarity: 2.0
- locked (PERTURBATION_EXCLUDED) at Phase-1 values: delivery_timing, arrival_probs, input_holding_loss, trade_tau, alloc_eta_request

### Configuration 2 x Random

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0627 s, max 0.1405 s
- conservation: max |residual| 1.776e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.326166, max 2.30688; min T/T_min 10.8722, max T/T_upper 0.132645; failures 0 - HELD
- S: min 0, max S/S_max 0.378699 (S_max 3); failures 0 - HELD
- fill: min 0.00158617, max 1; failures 0 - HELD
- flags: none

### Configuration 2 x TruthfulMyopic

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0622 s, max 0.1415 s
- conservation: max |residual| 8.882e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.239744, max 0.623819; min T/T_min 7.99145, max T/T_upper 0.0358694; failures 0 - HELD
- S: min 0, max S/S_max 0.206911 (S_max 3); failures 0 - HELD
- fill: min 1, max 1; failures 0 - HELD
- flags: none

### Configuration 2 x Padder

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0621 s, max 0.1697 s
- conservation: max |residual| 4.441e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.6, max 0.708528; min T/T_min 20, max T/T_upper 0.0407401; failures 0 - HELD
- S: min 0, max S/S_max 0.145617 (S_max 3); failures 0 - HELD
- fill: min 0.310834, max 1; failures 0 - HELD
- Padder shortage (T-B3 property, fill < 1 somewhere): True - HELD
- flags: none

## Configuration 3: `2dcdedd08d5fd6a0172ed1ca5463e2e3c4479531ab97fcaea70e2a86675aa9b9`

- run directory: `runs/2dcdedd08d5fd6a0172ed1ca5463e2e3c4479531ab97fcaea70e2a86675aa9b9/` (manifest.json, ledger/<agent>/)
- kind: SUPPLY perturbation
- yield_sigma: (0.079712, 0.12754, 0.159424, 0.191309, 0.239137) (x1.594245)
- final_demand_share: 0.517450 (every sector)
- holding_loss: 0.046754
- input_complementarity: 2.0
- locked (PERTURBATION_EXCLUDED) at Phase-1 values: delivery_timing, arrival_probs, input_holding_loss, trade_tau, alloc_eta_request

### Configuration 3 x Random

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0622 s, max 0.1288 s
- conservation: max |residual| 1.776e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.326166, max 2.30688; min T/T_min 10.8722, max T/T_upper 0.132645; failures 0 - HELD
- S: min 0, max S/S_max 0.582677 (S_max 3); failures 0 - HELD
- fill: min 0.00158176, max 1; failures 0 - HELD
- flags: none

### Configuration 3 x TruthfulMyopic

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0628 s, max 0.1299 s
- conservation: max |residual| 1.776e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.52479, max 0.952105; min T/T_min 17.493, max T/T_upper 0.0547457; failures 0 - HELD
- S: min 0, max S/S_max 0.35399 (S_max 3); failures 0 - HELD
- fill: min 1, max 1; failures 0 - HELD
- flags: none

### Configuration 3 x Padder

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0613 s, max 0.1366 s
- conservation: max |residual| 8.882e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.6, max 0.708528; min T/T_min 20, max T/T_upper 0.0407401; failures 0 - HELD
- S: min 0, max S/S_max 0.141677 (S_max 3); failures 0 - HELD
- fill: min 0.367739, max 1; failures 0 - HELD
- Padder shortage (T-B3 property, fill < 1 somewhere): True - HELD
- flags: none

## Configuration 4: `5623439f0cd8698ee1653d7760978eb095bccd33302058b5fe29e10abe229c3b`

- run directory: `runs/5623439f0cd8698ee1653d7760978eb095bccd33302058b5fe29e10abe229c3b/` (manifest.json, ledger/<agent>/)
- kind: SUPPLY perturbation
- yield_sigma: (0.025205, 0.040329, 0.050411, 0.060493, 0.075616) (x0.504108)
- final_demand_share: 0.642962 (every sector)
- holding_loss: 0.001679
- input_complementarity: inf
- locked (PERTURBATION_EXCLUDED) at Phase-1 values: delivery_timing, arrival_probs, input_holding_loss, trade_tau, alloc_eta_request

### Configuration 4 x Random

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0620 s, max 0.1271 s
- conservation: max |residual| 1.776e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.326166, max 2.30688; min T/T_min 10.8722, max T/T_upper 0.132645; failures 0 - HELD
- S: min 0, max S/S_max 0.396473 (S_max 3); failures 0 - HELD
- fill: min 0.00153855, max 1; failures 0 - HELD
- flags: none

### Configuration 4 x TruthfulMyopic

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0617 s, max 0.1554 s
- conservation: max |residual| 8.882e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.316617, max 0.617362; min T/T_min 10.5539, max T/T_upper 0.0354981; failures 0 - HELD
- S: min 0, max S/S_max 0.202933 (S_max 3); failures 0 - HELD
- fill: min 1, max 1; failures 0 - HELD
- flags: none

### Configuration 4 x Padder

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0617 s, max 0.1560 s
- conservation: max |residual| 4.441e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.6, max 0.708528; min T/T_min 20, max T/T_upper 0.0407401; failures 0 - HELD
- S: min 0, max S/S_max 0.111641 (S_max 3); failures 0 - HELD
- fill: min 0.410839, max 1; failures 0 - HELD
- Padder shortage (T-B3 property, fill < 1 somewhere): True - HELD
- flags: none

## Configuration 5: `db5cf2b95c48d050ecef935b7e4457efe378a39f54c0b77c28a055cfaa89975b`

- run directory: `runs/db5cf2b95c48d050ecef935b7e4457efe378a39f54c0b77c28a055cfaa89975b/` (manifest.json, ledger/<agent>/)
- kind: SUPPLY perturbation
- yield_sigma: (0.079724, 0.127559, 0.159448, 0.191338, 0.239172) (x1.594483)
- final_demand_share: 0.370262 (every sector)
- holding_loss: 0.043159
- input_complementarity: 2.0
- locked (PERTURBATION_EXCLUDED) at Phase-1 values: delivery_timing, arrival_probs, input_holding_loss, trade_tau, alloc_eta_request

### Configuration 5 x Random

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0623 s, max 0.2554 s
- conservation: max |residual| 3.553e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.326166, max 2.30688; min T/T_min 10.8722, max T/T_upper 0.132645; failures 0 - HELD
- S: min 0, max S/S_max 0.647231 (S_max 3); failures 0 - HELD
- fill: min 0.00158177, max 1; failures 0 - HELD
- flags: none

### Configuration 5 x TruthfulMyopic

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0615 s, max 0.1313 s
- conservation: max |residual| 3.553e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.52477, max 0.952108; min T/T_min 17.4923, max T/T_upper 0.0547459; failures 0 - HELD
- S: min 0, max S/S_max 0.354006 (S_max 3); failures 0 - HELD
- fill: min 1, max 1; failures 0 - HELD
- flags: none

### Configuration 5 x Padder

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0611 s, max 0.1217 s
- conservation: max |residual| 8.882e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.6, max 0.708528; min T/T_min 20, max T/T_upper 0.0407401; failures 0 - HELD
- S: min 0, max S/S_max 0.141684 (S_max 3); failures 0 - HELD
- fill: min 0.36773, max 1; failures 0 - HELD
- Padder shortage (T-B3 property, fill < 1 somewhere): True - HELD
- flags: none

## Configuration 6: `ef4bdc24363307cdbf971a27fff49f28a36b69d16551a82ca9516afbf7b94a80`

- run directory: `runs/ef4bdc24363307cdbf971a27fff49f28a36b69d16551a82ca9516afbf7b94a80/` (manifest.json, ledger/<agent>/)
- kind: SUPPLY perturbation
- yield_sigma: (0.047478, 0.075965, 0.094957, 0.113948, 0.142435) (x0.949568)
- final_demand_share: 0.469075 (every sector)
- holding_loss: 0.001416
- input_complementarity: 8.0
- locked (PERTURBATION_EXCLUDED) at Phase-1 values: delivery_timing, arrival_probs, input_holding_loss, trade_tau, alloc_eta_request

### Configuration 6 x Random

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0550 s, max 0.1163 s
- conservation: max |residual| 3.553e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.326166, max 2.30688; min T/T_min 10.8722, max T/T_upper 0.132645; failures 0 - HELD
- S: min 0, max S/S_max 0.59063 (S_max 3); failures 0 - HELD
- fill: min 0.00155726, max 1; failures 0 - HELD
- flags: none

### Configuration 6 x TruthfulMyopic

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0601 s, max 0.1510 s
- conservation: max |residual| 8.882e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.580591, max 0.841539; min T/T_min 19.353, max T/T_upper 0.0483882; failures 0 - HELD
- S: min 0, max S/S_max 0.295967 (S_max 3); failures 0 - HELD
- fill: min 1, max 1; failures 0 - HELD
- flags: none

### Configuration 6 x Padder

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0618 s, max 0.1316 s
- conservation: max |residual| 8.882e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.6, max 0.708528; min T/T_min 20, max T/T_upper 0.0407401; failures 0 - HELD
- S: min 0, max S/S_max 0.12306 (S_max 3); failures 0 - HELD
- fill: min 0.392696, max 1; failures 0 - HELD
- Padder shortage (T-B3 property, fill < 1 somewhere): True - HELD
- flags: none

## Configuration 7: `4700ccb9bb8e319243d12efdd086d1b0927b1b1e09e70a30c156be9eebd92b60`

- run directory: `runs/4700ccb9bb8e319243d12efdd086d1b0927b1b1e09e70a30c156be9eebd92b60/` (manifest.json, ledger/<agent>/)
- kind: SUPPLY perturbation
- yield_sigma: (0.034321, 0.054914, 0.068642, 0.082371, 0.102964) (x0.686425)
- final_demand_share: 0.568250 (every sector)
- holding_loss: 0.032359
- input_complementarity: 2.0
- locked (PERTURBATION_EXCLUDED) at Phase-1 values: delivery_timing, arrival_probs, input_holding_loss, trade_tau, alloc_eta_request

### Configuration 7 x Random

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0636 s, max 0.1644 s
- conservation: max |residual| 1.776e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.326166, max 2.30688; min T/T_min 10.8722, max T/T_upper 0.132645; failures 0 - HELD
- S: min 0, max S/S_max 0.556323 (S_max 3); failures 0 - HELD
- fill: min 0.00154638, max 1; failures 0 - HELD
- flags: none

### Configuration 7 x TruthfulMyopic

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0625 s, max 0.1294 s
- conservation: max |residual| 8.882e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.594595, max 0.797484; min T/T_min 19.8198, max T/T_upper 0.045855; failures 0 - HELD
- S: min 0, max S/S_max 0.271229 (S_max 3); failures 0 - HELD
- fill: min 1, max 1; failures 0 - HELD
- flags: none

### Configuration 7 x Padder

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0619 s, max 0.1363 s
- conservation: max |residual| 8.882e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.6, max 0.708528; min T/T_min 20, max T/T_upper 0.0407401; failures 0 - HELD
- S: min 0, max S/S_max 0.11618 (S_max 3); failures 0 - HELD
- fill: min 0.403322, max 1; failures 0 - HELD
- Padder shortage (T-B3 property, fill < 1 somewhere): True - HELD
- flags: none

## Configuration 8: `fe2103b7b62029e0d3cca499a17e53fe865f799c1106ed819fe15cc5f8480187`

- run directory: `runs/fe2103b7b62029e0d3cca499a17e53fe865f799c1106ed819fe15cc5f8480187/` (manifest.json, ledger/<agent>/)
- kind: SUPPLY perturbation
- yield_sigma: (0.053776, 0.086041, 0.107552, 0.129062, 0.161327) (x1.075516)
- final_demand_share: 0.698884 (every sector)
- holding_loss: 0.049042
- input_complementarity: 8.0
- locked (PERTURBATION_EXCLUDED) at Phase-1 values: delivery_timing, arrival_probs, input_holding_loss, trade_tau, alloc_eta_request

### Configuration 8 x Random

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0633 s, max 0.1621 s
- conservation: max |residual| 1.332e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.326166, max 2.30688; min T/T_min 10.8722, max T/T_upper 0.132645; failures 0 - HELD
- S: min 0, max S/S_max 0.340599 (S_max 3); failures 0 - HELD
- fill: min 0.00156229, max 1; failures 0 - HELD
- flags: none

### Configuration 8 x TruthfulMyopic

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0629 s, max 0.1693 s
- conservation: max |residual| 8.882e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.213427, max 0.620921; min T/T_min 7.11425, max T/T_upper 0.0357028; failures 0 - HELD
- S: min 0, max S/S_max 0.205126 (S_max 3); failures 0 - HELD
- fill: min 1, max 1; failures 0 - HELD
- flags: none

### Configuration 8 x Padder

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0618 s, max 0.1177 s
- conservation: max |residual| 4.441e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.6, max 0.708528; min T/T_min 20, max T/T_upper 0.0407401; failures 0 - HELD
- S: min 0, max S/S_max 0.120952 (S_max 3); failures 0 - HELD
- fill: min 0.188839, max 1; failures 0 - HELD
- Padder shortage (T-B3 property, fill < 1 somewhere): True - HELD
- flags: none

## Configuration 9: `c0544d897fe09f27d4974035c2f981d7bb18451706a688e623f2f59820a158b1`

- run directory: `runs/c0544d897fe09f27d4974035c2f981d7bb18451706a688e623f2f59820a158b1/` (manifest.json, ledger/<agent>/)
- kind: SUPPLY perturbation
- yield_sigma: (0.076416, 0.122265, 0.152831, 0.183398, 0.229247) (x1.528313)
- final_demand_share: 0.560184 (every sector)
- holding_loss: 0.034422
- input_complementarity: inf
- locked (PERTURBATION_EXCLUDED) at Phase-1 values: delivery_timing, arrival_probs, input_holding_loss, trade_tau, alloc_eta_request

### Configuration 9 x Random

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0621 s, max 0.1420 s
- conservation: max |residual| 2.665e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.326166, max 2.30688; min T/T_min 10.8722, max T/T_upper 0.132645; failures 0 - HELD
- S: min 0, max S/S_max 0.587594 (S_max 3); failures 0 - HELD
- fill: min 0.0015794, max 1; failures 0 - HELD
- flags: none

### Configuration 9 x TruthfulMyopic

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0617 s, max 0.1460 s
- conservation: max |residual| 8.882e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.530399, max 0.854884; min T/T_min 17.68, max T/T_upper 0.0491555; failures 0 - HELD
- S: min 0, max S/S_max 0.320761 (S_max 3); failures 0 - HELD
- fill: min 1, max 1; failures 0 - HELD
- flags: none

### Configuration 9 x Padder

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0610 s, max 0.1420 s
- conservation: max |residual| 8.882e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.6, max 0.708528; min T/T_min 20, max T/T_upper 0.0407401; failures 0 - HELD
- S: min 0, max S/S_max 0.139652 (S_max 3); failures 0 - HELD
- fill: min 0.370223, max 1; failures 0 - HELD
- Padder shortage (T-B3 property, fill < 1 somewhere): True - HELD
- flags: none

## Configuration 10: `186eabfabe784fe91653b3e1276265936e775bd0e3642187092c296fca84356e`

- run directory: `runs/186eabfabe784fe91653b3e1276265936e775bd0e3642187092c296fca84356e/` (manifest.json, ledger/<agent>/)
- kind: SUPPLY perturbation
- yield_sigma: (0.035132, 0.056212, 0.070264, 0.084317, 0.105397) (x0.702645)
- final_demand_share: 0.588595 (every sector)
- holding_loss: 0.026268
- input_complementarity: 8.0
- locked (PERTURBATION_EXCLUDED) at Phase-1 values: delivery_timing, arrival_probs, input_holding_loss, trade_tau, alloc_eta_request

### Configuration 10 x Random

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0628 s, max 0.1352 s
- conservation: max |residual| 1.776e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.326166, max 2.30688; min T/T_min 10.8722, max T/T_upper 0.132645; failures 0 - HELD
- S: min 0, max S/S_max 0.52773 (S_max 3); failures 0 - HELD
- fill: min 0.00154706, max 1; failures 0 - HELD
- flags: none

### Configuration 10 x TruthfulMyopic

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0630 s, max 0.4204 s
- conservation: max |residual| 8.882e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.594165, max 0.768602; min T/T_min 19.8055, max T/T_upper 0.0441943; failures 0 - HELD
- S: min 0, max S/S_max 0.257381 (S_max 3); failures 0 - HELD
- fill: min 1, max 1; failures 0 - HELD
- flags: none

### Configuration 10 x Padder

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0621 s, max 0.1645 s
- conservation: max |residual| 8.882e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.6, max 0.708528; min T/T_min 20, max T/T_upper 0.0407401; failures 0 - HELD
- S: min 0, max S/S_max 0.116593 (S_max 3); failures 0 - HELD
- fill: min 0.402659, max 1; failures 0 - HELD
- Padder shortage (T-B3 property, fill < 1 somewhere): True - HELD
- flags: none

## Configuration 11: `4ea5e22fccf3304e95843d1aacd04bdc87dcfd2cc4b1a070954eaab8f6796f90`

- run directory: `runs/4ea5e22fccf3304e95843d1aacd04bdc87dcfd2cc4b1a070954eaab8f6796f90/` (manifest.json, ledger/<agent>/)
- kind: SUPPLY perturbation
- yield_sigma: (0.048268, 0.077229, 0.096536, 0.115844, 0.144804) (x0.965363)
- final_demand_share: 0.494334 (every sector)
- holding_loss: 0.044474
- input_complementarity: 2.0
- locked (PERTURBATION_EXCLUDED) at Phase-1 values: delivery_timing, arrival_probs, input_holding_loss, trade_tau, alloc_eta_request

### Configuration 11 x Random

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0626 s, max 0.1446 s
- conservation: max |residual| 2.665e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.326166, max 2.30688; min T/T_min 10.8722, max T/T_upper 0.132645; failures 0 - HELD
- S: min 0, max S/S_max 0.559793 (S_max 3); failures 0 - HELD
- fill: min 0.0015579, max 1; failures 0 - HELD
- flags: none

### Configuration 11 x TruthfulMyopic

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0630 s, max 0.1466 s
- conservation: max |residual| 1.776e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.5792, max 0.84451; min T/T_min 19.3067, max T/T_upper 0.0485591; failures 0 - HELD
- S: min 0, max S/S_max 0.297362 (S_max 3); failures 0 - HELD
- fill: min 1, max 1; failures 0 - HELD
- flags: none

### Configuration 11 x Padder

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0619 s, max 0.1507 s
- conservation: max |residual| 4.441e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.6, max 0.708528; min T/T_min 20, max T/T_upper 0.0407401; failures 0 - HELD
- S: min 0, max S/S_max 0.123485 (S_max 3); failures 0 - HELD
- fill: min 0.392066, max 1; failures 0 - HELD
- Padder shortage (T-B3 property, fill < 1 somewhere): True - HELD
- flags: none

## Configuration 12: `31d01c6f3fcf3338b149aa1e745dc4d8bfa9359fd3d4726395fd1c4a7e236c50`

- run directory: `runs/31d01c6f3fcf3338b149aa1e745dc4d8bfa9359fd3d4726395fd1c4a7e236c50/` (manifest.json, ledger/<agent>/)
- kind: SUPPLY perturbation
- yield_sigma: (0.051835, 0.082935, 0.103669, 0.124403, 0.155504) (x1.036693)
- final_demand_share: 0.528612 (every sector)
- holding_loss: 0.016093
- input_complementarity: inf
- locked (PERTURBATION_EXCLUDED) at Phase-1 values: delivery_timing, arrival_probs, input_holding_loss, trade_tau, alloc_eta_request

### Configuration 12 x Random

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0612 s, max 0.1513 s
- conservation: max |residual| 1.776e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.326166, max 2.30688; min T/T_min 10.8722, max T/T_upper 0.132645; failures 0 - HELD
- S: min 0, max S/S_max 0.57917 (S_max 3); failures 0 - HELD
- fill: min 0.00156076, max 1; failures 0 - HELD
- flags: none

### Configuration 12 x TruthfulMyopic

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0614 s, max 0.1689 s
- conservation: max |residual| 1.776e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.572934, max 0.858048; min T/T_min 19.0978, max T/T_upper 0.0493375; failures 0 - HELD
- S: min 0, max S/S_max 0.303733 (S_max 3); failures 0 - HELD
- fill: min 1, max 1; failures 0 - HELD
- flags: none

### Configuration 12 x Padder

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0617 s, max 0.1915 s
- conservation: max |residual| 8.882e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.6, max 0.708528; min T/T_min 20, max T/T_upper 0.0407401; failures 0 - HELD
- S: min 0, max S/S_max 0.125426 (S_max 3); failures 0 - HELD
- fill: min 0.389235, max 1; failures 0 - HELD
- Padder shortage (T-B3 property, fill < 1 somewhere): True - HELD
- flags: none

## Configuration 13: `78232cf87ecfd13b925f6e595f4631218bf627e7055a711304f676d02479b40e`

- run directory: `runs/78232cf87ecfd13b925f6e595f4631218bf627e7055a711304f676d02479b40e/` (manifest.json, ledger/<agent>/)
- kind: SUPPLY perturbation
- yield_sigma: (0.069573, 0.111316, 0.139145, 0.166974, 0.208718) (x1.391450)
- final_demand_share: 0.435164 (every sector)
- holding_loss: 0.019581
- input_complementarity: 2.0
- locked (PERTURBATION_EXCLUDED) at Phase-1 values: delivery_timing, arrival_probs, input_holding_loss, trade_tau, alloc_eta_request

### Configuration 13 x Random

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0623 s, max 0.1508 s
- conservation: max |residual| 2.665e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.326166, max 2.30688; min T/T_min 10.8722, max T/T_upper 0.132645; failures 0 - HELD
- S: min 0, max S/S_max 0.591175 (S_max 3); failures 0 - HELD
- fill: min 0.00157439, max 1; failures 0 - HELD
- flags: none

### Configuration 13 x TruthfulMyopic

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0615 s, max 0.1162 s
- conservation: max |residual| 1.776e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.542117, max 0.928404; min T/T_min 18.0706, max T/T_upper 0.0533829; failures 0 - HELD
- S: min 0, max S/S_max 0.337213 (S_max 3); failures 0 - HELD
- fill: min 1, max 1; failures 0 - HELD
- flags: none

### Configuration 13 x Padder

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0608 s, max 0.1715 s
- conservation: max |residual| 8.882e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.6, max 0.708528; min T/T_min 20, max T/T_upper 0.0407401; failures 0 - HELD
- S: min 0, max S/S_max 0.135538 (S_max 3); failures 0 - HELD
- fill: min 0.375429, max 1; failures 0 - HELD
- Padder shortage (T-B3 property, fill < 1 somewhere): True - HELD
- flags: none

## Configuration 14: `ea9142de23611a5278182a8fd771b42b1d2451df2e6b74062344c457fc29c63e`

- run directory: `runs/ea9142de23611a5278182a8fd771b42b1d2451df2e6b74062344c457fc29c63e/` (manifest.json, ledger/<agent>/)
- kind: SUPPLY perturbation
- yield_sigma: (0.042037, 0.067259, 0.084074, 0.100888, 0.12611) (x0.840736)
- final_demand_share: 0.549275 (every sector)
- holding_loss: 0.004201
- input_complementarity: inf
- locked (PERTURBATION_EXCLUDED) at Phase-1 values: delivery_timing, arrival_probs, input_holding_loss, trade_tau, alloc_eta_request

### Configuration 14 x Random

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0611 s, max 0.1351 s
- conservation: max |residual| 2.665e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.326166, max 2.30688; min T/T_min 10.8722, max T/T_upper 0.132645; failures 0 - HELD
- S: min 0, max S/S_max 0.57852 (S_max 3); failures 0 - HELD
- fill: min 0.00155282, max 1; failures 0 - HELD
- flags: none

### Configuration 14 x TruthfulMyopic

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0602 s, max 0.1534 s
- conservation: max |residual| 8.882e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.587191, max 0.825207; min T/T_min 19.573, max T/T_upper 0.0474491; failures 0 - HELD
- S: min 0, max S/S_max 0.286508 (S_max 3); failures 0 - HELD
- fill: min 1, max 1; failures 0 - HELD
- flags: none

### Configuration 14 x Padder

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0607 s, max 0.1618 s
- conservation: max |residual| 4.441e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.6, max 0.708528; min T/T_min 20, max T/T_upper 0.0407401; failures 0 - HELD
- S: min 0, max S/S_max 0.120166 (S_max 3); failures 0 - HELD
- fill: min 0.397059, max 1; failures 0 - HELD
- Padder shortage (T-B3 property, fill < 1 somewhere): True - HELD
- flags: none

## Configuration 15: `c437eb664d5642a999e80c40dfcc6c4d53d77758265f1dc7277617df17124473`

- run directory: `runs/c437eb664d5642a999e80c40dfcc6c4d53d77758265f1dc7277617df17124473/` (manifest.json, ledger/<agent>/)
- kind: SUPPLY perturbation
- yield_sigma: (0.087448, 0.139917, 0.174897, 0.209876, 0.262345) (x1.748966)
- final_demand_share: 0.614839 (every sector)
- holding_loss: 0.011968
- input_complementarity: inf
- locked (PERTURBATION_EXCLUDED) at Phase-1 values: delivery_timing, arrival_probs, input_holding_loss, trade_tau, alloc_eta_request

### Configuration 15 x Random

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0619 s, max 0.1307 s
- conservation: max |residual| 1.776e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.326166, max 2.30688; min T/T_min 10.8722, max T/T_upper 0.132645; failures 0 - HELD
- S: min 0, max S/S_max 0.472229 (S_max 3); failures 0 - HELD
- fill: min 0.00158717, max 1; failures 0 - HELD
- flags: none

### Configuration 15 x TruthfulMyopic

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0606 s, max 0.1530 s
- conservation: max |residual| 1.776e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.480399, max 0.623986; min T/T_min 16.0133, max T/T_upper 0.035879; failures 0 - HELD
- S: min 0, max S/S_max 0.208669 (S_max 3); failures 0 - HELD
- fill: min 1, max 1; failures 0 - HELD
- flags: none

### Configuration 15 x Padder

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0611 s, max 0.1311 s
- conservation: max |residual| 4.441e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.6, max 0.708528; min T/T_min 20, max T/T_upper 0.0407401; failures 0 - HELD
- S: min 0, max S/S_max 0.146543 (S_max 3); failures 0 - HELD
- fill: min 0.361969, max 1; failures 0 - HELD
- Padder shortage (T-B3 property, fill < 1 somewhere): True - HELD
- flags: none

## Configuration 16: `3bb4c959eb9e55b7229fb2386f174251835eedcef170f9a880164b017eaa238d`

- run directory: `runs/3bb4c959eb9e55b7229fb2386f174251835eedcef170f9a880164b017eaa238d/` (manifest.json, ledger/<agent>/)
- kind: SUPPLY perturbation
- yield_sigma: (0.029393, 0.047028, 0.058785, 0.070542, 0.088178) (x0.587852)
- final_demand_share: 0.434447 (every sector)
- holding_loss: 0.007514
- input_complementarity: inf
- locked (PERTURBATION_EXCLUDED) at Phase-1 values: delivery_timing, arrival_probs, input_holding_loss, trade_tau, alloc_eta_request

### Configuration 16 x Random

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0619 s, max 0.1296 s
- conservation: max |residual| 2.665e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.326166, max 2.30688; min T/T_min 10.8722, max T/T_upper 0.132645; failures 0 - HELD
- S: min 0, max S/S_max 0.598216 (S_max 3); failures 0 - HELD
- fill: min 0.00154217, max 1; failures 0 - HELD
- flags: none

### Configuration 16 x TruthfulMyopic

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0624 s, max 0.1451 s
- conservation: max |residual| 1.776e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.597208, max 0.788418; min T/T_min 19.9069, max T/T_upper 0.0453338; failures 0 - HELD
- S: min 0, max S/S_max 0.265849 (S_max 3); failures 0 - HELD
- fill: min 1, max 1; failures 0 - HELD
- flags: none

### Configuration 16 x Padder

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0619 s, max 0.1587 s
- conservation: max |residual| 8.882e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.6, max 0.708528; min T/T_min 20, max T/T_upper 0.0407401; failures 0 - HELD
- S: min 0, max S/S_max 0.113703 (S_max 3); failures 0 - HELD
- fill: min 0.40737, max 1; failures 0 - HELD
- Padder shortage (T-B3 property, fill < 1 somewhere): True - HELD
- flags: none

## Configuration 17: `5d7c3cdfb5bc02ec899603de3d6edd43bd163c6c5cfc593f490ae01e67f96d72`

- run directory: `runs/5d7c3cdfb5bc02ec899603de3d6edd43bd163c6c5cfc593f490ae01e67f96d72/` (manifest.json, ledger/<agent>/)
- kind: SUPPLY perturbation
- yield_sigma: (0.058775, 0.094041, 0.117551, 0.141061, 0.176326) (x1.175509)
- final_demand_share: 0.618530 (every sector)
- holding_loss: 0.011532
- input_complementarity: inf
- locked (PERTURBATION_EXCLUDED) at Phase-1 values: delivery_timing, arrival_probs, input_holding_loss, trade_tau, alloc_eta_request

### Configuration 17 x Random

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0635 s, max 0.1547 s
- conservation: max |residual| 1.776e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.326166, max 2.30688; min T/T_min 10.8722, max T/T_upper 0.132645; failures 0 - HELD
- S: min 0, max S/S_max 0.435993 (S_max 3); failures 0 - HELD
- fill: min 0.0015662, max 1; failures 0 - HELD
- flags: none

### Configuration 17 x TruthfulMyopic

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0626 s, max 0.2274 s
- conservation: max |residual| 8.882e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.466563, max 0.621317; min T/T_min 15.5521, max T/T_upper 0.0357255; failures 0 - HELD
- S: min 0, max S/S_max 0.20537 (S_max 3); failures 0 - HELD
- fill: min 1, max 1; failures 0 - HELD
- flags: none

### Configuration 17 x Padder

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0621 s, max 0.1478 s
- conservation: max |residual| 8.882e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.6, max 0.708528; min T/T_min 20, max T/T_upper 0.0407401; failures 0 - HELD
- S: min 0, max S/S_max 0.12929 (S_max 3); failures 0 - HELD
- fill: min 0.383778, max 1; failures 0 - HELD
- Padder shortage (T-B3 property, fill < 1 somewhere): True - HELD
- flags: none

## Configuration 18: `f2e062d33b8529e8176755147524ccf0f80c27e13076d42efd6f1f94450703de`

- run directory: `runs/f2e062d33b8529e8176755147524ccf0f80c27e13076d42efd6f1f94450703de/` (manifest.json, ledger/<agent>/)
- kind: SUPPLY perturbation
- yield_sigma: (0.055341, 0.088546, 0.110683, 0.132819, 0.166024) (x1.106828)
- final_demand_share: 0.379405 (every sector)
- holding_loss: 0.004538
- input_complementarity: 2.0
- locked (PERTURBATION_EXCLUDED) at Phase-1 values: delivery_timing, arrival_probs, input_holding_loss, trade_tau, alloc_eta_request

### Configuration 18 x Random

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0637 s, max 0.1612 s
- conservation: max |residual| 3.553e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.326166, max 2.30688; min T/T_min 10.8722, max T/T_upper 0.132645; failures 0 - HELD
- S: min 0, max S/S_max 0.643394 (S_max 3); failures 0 - HELD
- fill: min 0.00156353, max 1; failures 0 - HELD
- flags: none

### Configuration 18 x TruthfulMyopic

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0634 s, max 0.1708 s
- conservation: max |residual| 1.776e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.566794, max 0.871555; min T/T_min 18.8931, max T/T_upper 0.0501141; failures 0 - HELD
- S: min 0, max S/S_max 0.310113 (S_max 3); failures 0 - HELD
- fill: min 1, max 1; failures 0 - HELD
- flags: none

### Configuration 18 x Padder

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0622 s, max 0.1442 s
- conservation: max |residual| 8.882e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.6, max 0.708528; min T/T_min 20, max T/T_upper 0.0407401; failures 0 - HELD
- S: min 0, max S/S_max 0.127364 (S_max 3); failures 0 - HELD
- fill: min 0.386469, max 1; failures 0 - HELD
- Padder shortage (T-B3 property, fill < 1 somewhere): True - HELD
- flags: none

## Configuration 19: `9e8e7b9fc3845494d4a3c8cefc152654d5f33b8febf92cac02ea2c906e238b90`

- run directory: `runs/9e8e7b9fc3845494d4a3c8cefc152654d5f33b8febf92cac02ea2c906e238b90/` (manifest.json, ledger/<agent>/)
- kind: SUPPLY perturbation
- yield_sigma: (0.068525, 0.10964, 0.13705, 0.16446, 0.205575) (x1.370499)
- final_demand_share: 0.419478 (every sector)
- holding_loss: 0.033600
- input_complementarity: inf
- locked (PERTURBATION_EXCLUDED) at Phase-1 values: delivery_timing, arrival_probs, input_holding_loss, trade_tau, alloc_eta_request

### Configuration 19 x Random

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0626 s, max 0.1358 s
- conservation: max |residual| 3.553e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.326166, max 2.30688; min T/T_min 10.8722, max T/T_upper 0.132645; failures 0 - HELD
- S: min 0, max S/S_max 0.587919 (S_max 3); failures 0 - HELD
- fill: min 0.00157361, max 1; failures 0 - HELD
- flags: none

### Configuration 19 x TruthfulMyopic

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0621 s, max 0.5317 s
- conservation: max |residual| 1.776e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.54392, max 0.924105; min T/T_min 18.1307, max T/T_upper 0.0531357; failures 0 - HELD
- S: min 0, max S/S_max 0.335151 (S_max 3); failures 0 - HELD
- fill: min 1, max 1; failures 0 - HELD
- flags: none

### Configuration 19 x Padder

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0613 s, max 0.1256 s
- conservation: max |residual| 8.882e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.6, max 0.708528; min T/T_min 20, max T/T_upper 0.0407401; failures 0 - HELD
- S: min 0, max S/S_max 0.134919 (S_max 3); failures 0 - HELD
- fill: min 0.376232, max 1; failures 0 - HELD
- Padder shortage (T-B3 property, fill < 1 somewhere): True - HELD
- flags: none

## Configuration 20: `16c9d0d929e1cd891a35b1297363417bec3aa88cbcbf1560150d3469cb229d41`

- run directory: `runs/16c9d0d929e1cd891a35b1297363417bec3aa88cbcbf1560150d3469cb229d41/` (manifest.json, ledger/<agent>/)
- kind: SUPPLY perturbation
- yield_sigma: (0.095658, 0.153054, 0.191317, 0.22958, 0.286975) (x1.913170)
- final_demand_share: 0.446044 (every sector)
- holding_loss: 0.005275
- input_complementarity: 2.0
- locked (PERTURBATION_EXCLUDED) at Phase-1 values: delivery_timing, arrival_probs, input_holding_loss, trade_tau, alloc_eta_request

### Configuration 20 x Random

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0626 s, max 0.1331 s
- conservation: max |residual| 2.665e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.326166, max 2.30688; min T/T_min 10.8722, max T/T_upper 0.132645; failures 0 - HELD
- S: min 0, max S/S_max 0.621886 (S_max 3); failures 0 - HELD
- fill: min 0.00159271, max 1; failures 0 - HELD
- flags: none

### Configuration 20 x TruthfulMyopic

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0627 s, max 0.3108 s
- conservation: max |residual| 1.776e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.498017, max 0.955594; min T/T_min 16.6006, max T/T_upper 0.0549463; failures 0 - HELD
- S: min 0, max S/S_max 0.375473 (S_max 3); failures 0 - HELD
- fill: min 1, max 1; failures 0 - HELD
- flags: none

### Configuration 20 x Padder

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0609 s, max 0.1463 s
- conservation: max |residual| 8.882e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.6, max 0.708528; min T/T_min 20, max T/T_upper 0.0407401; failures 0 - HELD
- S: min 0, max S/S_max 0.151885 (S_max 3); failures 0 - HELD
- fill: min 0.355935, max 1; failures 0 - HELD
- Padder shortage (T-B3 property, fill < 1 somewhere): True - HELD
- flags: none
