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
- Padder minimum fill: 0.190139; shortage in every configuration: True
- mean wall clock per episode: 0.0696 s over 126000 episodes; harness total 10798.7 s
- flags raised: none (`BOUND_BINDING` is CONTRACT rule 8)
- git: c883e5f30ccc8289d3abead4581df303f49ffe1a-dirty

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

## Configuration 0: `2c7ae80e89f21be62ffe7e91242f8460b45d8465019d423c0187b340cde5900b`

- run directory: `runs/2c7ae80e89f21be62ffe7e91242f8460b45d8465019d423c0187b340cde5900b/` (manifest.json, ledger/<agent>/)
- kind: baseline (`p1_default_config()`)
- yield_sigma: (0.05, 0.08, 0.1, 0.12, 0.15)
- final_demand_share: 0.500000 (every sector)
- holding_loss: 0.020000
- input_complementarity: 8.0
- locked (PERTURBATION_EXCLUDED) at Phase-1 values: delivery_timing, arrival_probs, input_holding_loss, trade_tau, alloc_eta_request

### Configuration 0 x Random

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0720 s, max 0.2215 s
- conservation: max |residual| 2.665e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.339022, max 2.15048; min T/T_min 11.3007, max T/T_upper 0.147375; failures 0 - HELD
- S: min 0, max S/S_max 0.576684 (S_max 3); failures 0 - HELD
- fill: min 0.00164359, max 1; failures 0 - HELD
- flags: none

### Configuration 0 x TruthfulMyopic

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0695 s, max 0.1549 s
- conservation: max |residual| 1.776e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.578121, max 0.835789; min T/T_min 19.2707, max T/T_upper 0.0572777; failures 0 - HELD
- S: min 0, max S/S_max 0.29688 (S_max 3); failures 0 - HELD
- fill: min 1, max 1; failures 0 - HELD
- flags: none

### Configuration 0 x Padder

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0700 s, max 0.1648 s
- conservation: max |residual| 8.882e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.6, max 0.702996; min T/T_min 20, max T/T_upper 0.0481772; failures 0 - HELD
- S: min 0, max S/S_max 0.124424 (S_max 3); failures 0 - HELD
- fill: min 0.393378, max 1; failures 0 - HELD
- Padder shortage (T-B3 property, fill < 1 somewhere): True - HELD
- flags: none

## Configuration 1: `e553488ea375e8a234f52455442746cbb580a37711e594503f6d00a2605fea47`

- run directory: `runs/e553488ea375e8a234f52455442746cbb580a37711e594503f6d00a2605fea47/` (manifest.json, ledger/<agent>/)
- kind: SUPPLY perturbation
- yield_sigma: (0.072772, 0.116435, 0.145544, 0.174653, 0.218316) (x1.455443)
- final_demand_share: 0.407915 (every sector)
- holding_loss: 0.002049
- input_complementarity: 2.0
- locked (PERTURBATION_EXCLUDED) at Phase-1 values: delivery_timing, arrival_probs, input_holding_loss, trade_tau, alloc_eta_request

### Configuration 1 x Random

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0704 s, max 0.1669 s
- conservation: max |residual| 3.553e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.339022, max 2.15048; min T/T_min 11.3007, max T/T_upper 0.147375; failures 0 - HELD
- S: min 0, max S/S_max 0.620476 (S_max 3); failures 0 - HELD
- fill: min 0.001662, max 1; failures 0 - HELD
- flags: none

### Configuration 1 x TruthfulMyopic

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0695 s, max 0.1714 s
- conservation: max |residual| 1.776e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.540711, max 0.919999; min T/T_min 18.0237, max T/T_upper 0.0630487; failures 0 - HELD
- S: min 0, max S/S_max 0.338553 (S_max 3); failures 0 - HELD
- fill: min 1, max 1; failures 0 - HELD
- flags: none

### Configuration 1 x Padder

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0750 s, max 0.3364 s
- conservation: max |residual| 8.882e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.6, max 0.702996; min T/T_min 20, max T/T_upper 0.0481772; failures 0 - HELD
- S: min 0, max S/S_max 0.137446 (S_max 3); failures 0 - HELD
- fill: min 0.375554, max 1; failures 0 - HELD
- Padder shortage (T-B3 property, fill < 1 somewhere): True - HELD
- flags: none

## Configuration 2: `7c00cea619c356f1ccf764e20f5f71b224cf10fbb266db6f81b9ed9246175df3`

- run directory: `runs/7c00cea619c356f1ccf764e20f5f71b224cf10fbb266db6f81b9ed9246175df3/` (manifest.json, ledger/<agent>/)
- kind: SUPPLY perturbation
- yield_sigma: (0.085995, 0.137592, 0.171991, 0.206389, 0.257986) (x1.719905)
- final_demand_share: 0.665102 (every sector)
- holding_loss: 0.030332
- input_complementarity: 2.0
- locked (PERTURBATION_EXCLUDED) at Phase-1 values: delivery_timing, arrival_probs, input_holding_loss, trade_tau, alloc_eta_request

### Configuration 2 x Random

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0721 s, max 0.1731 s
- conservation: max |residual| 1.776e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.339022, max 2.15048; min T/T_min 11.3007, max T/T_upper 0.147375; failures 0 - HELD
- S: min 0, max S/S_max 0.381838 (S_max 3); failures 0 - HELD
- fill: min 0.00167192, max 1; failures 0 - HELD
- flags: none

### Configuration 2 x TruthfulMyopic

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0716 s, max 0.1538 s
- conservation: max |residual| 8.882e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.251872, max 0.622574; min T/T_min 8.39572, max T/T_upper 0.0426657; failures 0 - HELD
- S: min 0, max S/S_max 0.206911 (S_max 3); failures 0 - HELD
- fill: min 1, max 1; failures 0 - HELD
- flags: none

### Configuration 2 x Padder

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0706 s, max 0.1636 s
- conservation: max |residual| 4.441e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.6, max 0.702996; min T/T_min 20, max T/T_upper 0.0481772; failures 0 - HELD
- S: min 0, max S/S_max 0.145617 (S_max 3); failures 0 - HELD
- fill: min 0.312973, max 1; failures 0 - HELD
- Padder shortage (T-B3 property, fill < 1 somewhere): True - HELD
- flags: none

## Configuration 3: `6fc631bbdffd6de35658a3f7754945900f3678c9bd8c8b49171795a13bcad0a2`

- run directory: `runs/6fc631bbdffd6de35658a3f7754945900f3678c9bd8c8b49171795a13bcad0a2/` (manifest.json, ledger/<agent>/)
- kind: SUPPLY perturbation
- yield_sigma: (0.079712, 0.12754, 0.159424, 0.191309, 0.239137) (x1.594245)
- final_demand_share: 0.517450 (every sector)
- holding_loss: 0.046754
- input_complementarity: 2.0
- locked (PERTURBATION_EXCLUDED) at Phase-1 values: delivery_timing, arrival_probs, input_holding_loss, trade_tau, alloc_eta_request

### Configuration 3 x Random

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0708 s, max 0.1695 s
- conservation: max |residual| 2.665e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.339022, max 2.15048; min T/T_min 11.3007, max T/T_upper 0.147375; failures 0 - HELD
- S: min 0, max S/S_max 0.583871 (S_max 3); failures 0 - HELD
- fill: min 0.00166728, max 1; failures 0 - HELD
- flags: none

### Configuration 3 x TruthfulMyopic

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0707 s, max 0.1476 s
- conservation: max |residual| 1.776e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.529485, max 0.9297; min T/T_min 17.6495, max T/T_upper 0.0637135; failures 0 - HELD
- S: min 0, max S/S_max 0.348707 (S_max 3); failures 0 - HELD
- fill: min 1, max 1; failures 0 - HELD
- flags: none

### Configuration 3 x Padder

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0702 s, max 0.1741 s
- conservation: max |residual| 8.882e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.6, max 0.702996; min T/T_min 20, max T/T_upper 0.0481772; failures 0 - HELD
- S: min 0, max S/S_max 0.141677 (S_max 3); failures 0 - HELD
- fill: min 0.37027, max 1; failures 0 - HELD
- Padder shortage (T-B3 property, fill < 1 somewhere): True - HELD
- flags: none

## Configuration 4: `6c4c435b7b32139a9aa689ba23c4ca6d4b2f6126eabf018c92616d99745df261`

- run directory: `runs/6c4c435b7b32139a9aa689ba23c4ca6d4b2f6126eabf018c92616d99745df261/` (manifest.json, ledger/<agent>/)
- kind: SUPPLY perturbation
- yield_sigma: (0.025205, 0.040329, 0.050411, 0.060493, 0.075616) (x0.504108)
- final_demand_share: 0.642962 (every sector)
- holding_loss: 0.001679
- input_complementarity: inf
- locked (PERTURBATION_EXCLUDED) at Phase-1 values: delivery_timing, arrival_probs, input_holding_loss, trade_tau, alloc_eta_request

### Configuration 4 x Random

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0707 s, max 0.1542 s
- conservation: max |residual| 1.776e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.339022, max 2.15048; min T/T_min 11.3007, max T/T_upper 0.147375; failures 0 - HELD
- S: min 0, max S/S_max 0.399979 (S_max 3); failures 0 - HELD
- fill: min 0.00162173, max 1; failures 0 - HELD
- flags: none

### Configuration 4 x TruthfulMyopic

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0712 s, max 0.1873 s
- conservation: max |residual| 8.882e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.321324, max 0.616488; min T/T_min 10.7108, max T/T_upper 0.0422487; failures 0 - HELD
- S: min 0, max S/S_max 0.202933 (S_max 3); failures 0 - HELD
- fill: min 1, max 1; failures 0 - HELD
- flags: none

### Configuration 4 x Padder

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0700 s, max 0.1710 s
- conservation: max |residual| 4.441e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.6, max 0.702996; min T/T_min 20, max T/T_upper 0.0481772; failures 0 - HELD
- S: min 0, max S/S_max 0.111641 (S_max 3); failures 0 - HELD
- fill: min 0.413667, max 1; failures 0 - HELD
- Padder shortage (T-B3 property, fill < 1 somewhere): True - HELD
- flags: none

## Configuration 5: `5c5b3cd02139630e7ae5e257a609a6fdcf0db7eb258ec27ccb671e25c7252515`

- run directory: `runs/5c5b3cd02139630e7ae5e257a609a6fdcf0db7eb258ec27ccb671e25c7252515/` (manifest.json, ledger/<agent>/)
- kind: SUPPLY perturbation
- yield_sigma: (0.079724, 0.127559, 0.159448, 0.191338, 0.239172) (x1.594483)
- final_demand_share: 0.370262 (every sector)
- holding_loss: 0.043159
- input_complementarity: 2.0
- locked (PERTURBATION_EXCLUDED) at Phase-1 values: delivery_timing, arrival_probs, input_holding_loss, trade_tau, alloc_eta_request

### Configuration 5 x Random

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0720 s, max 0.1747 s
- conservation: max |residual| 3.553e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.339022, max 2.15048; min T/T_min 11.3007, max T/T_upper 0.147375; failures 0 - HELD
- S: min 0, max S/S_max 0.64894 (S_max 3); failures 0 - HELD
- fill: min 0.00166729, max 1; failures 0 - HELD
- flags: none

### Configuration 5 x TruthfulMyopic

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0723 s, max 0.1729 s
- conservation: max |residual| 1.776e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.529466, max 0.929703; min T/T_min 17.6489, max T/T_upper 0.0637137; failures 0 - HELD
- S: min 0, max S/S_max 0.348722 (S_max 3); failures 0 - HELD
- fill: min 1, max 1; failures 0 - HELD
- flags: none

### Configuration 5 x Padder

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0717 s, max 0.2211 s
- conservation: max |residual| 8.882e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.6, max 0.702996; min T/T_min 20, max T/T_upper 0.0481772; failures 0 - HELD
- S: min 0, max S/S_max 0.141684 (S_max 3); failures 0 - HELD
- fill: min 0.370261, max 1; failures 0 - HELD
- Padder shortage (T-B3 property, fill < 1 somewhere): True - HELD
- flags: none

## Configuration 6: `42d2200d46df659f0b8551c2b58b288e18c0bcaa0abe925ba8a05479347ce74d`

- run directory: `runs/42d2200d46df659f0b8551c2b58b288e18c0bcaa0abe925ba8a05479347ce74d/` (manifest.json, ledger/<agent>/)
- kind: SUPPLY perturbation
- yield_sigma: (0.047478, 0.075965, 0.094957, 0.113948, 0.142435) (x0.949568)
- final_demand_share: 0.469075 (every sector)
- holding_loss: 0.001416
- input_complementarity: 8.0
- locked (PERTURBATION_EXCLUDED) at Phase-1 values: delivery_timing, arrival_probs, input_holding_loss, trade_tau, alloc_eta_request

### Configuration 6 x Random

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0729 s, max 0.2200 s
- conservation: max |residual| 2.665e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.339022, max 2.15048; min T/T_min 11.3007, max T/T_upper 0.147375; failures 0 - HELD
- S: min 0, max S/S_max 0.592161 (S_max 3); failures 0 - HELD
- fill: min 0.00164146, max 1; failures 0 - HELD
- flags: none

### Configuration 6 x TruthfulMyopic

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0721 s, max 0.1881 s
- conservation: max |residual| 8.882e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.582122, max 0.827271; min T/T_min 19.4041, max T/T_upper 0.0566939; failures 0 - HELD
- S: min 0, max S/S_max 0.292555 (S_max 3); failures 0 - HELD
- fill: min 1, max 1; failures 0 - HELD
- flags: none

### Configuration 6 x Padder

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0705 s, max 0.1664 s
- conservation: max |residual| 4.441e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.6, max 0.702996; min T/T_min 20, max T/T_upper 0.0481772; failures 0 - HELD
- S: min 0, max S/S_max 0.12306 (S_max 3); failures 0 - HELD
- fill: min 0.395399, max 1; failures 0 - HELD
- Padder shortage (T-B3 property, fill < 1 somewhere): True - HELD
- flags: none

## Configuration 7: `67f9140bbaf6f378a7acabef90c381cbbc2711cc43d5d628cb474fbde7cbdffc`

- run directory: `runs/67f9140bbaf6f378a7acabef90c381cbbc2711cc43d5d628cb474fbde7cbdffc/` (manifest.json, ledger/<agent>/)
- kind: SUPPLY perturbation
- yield_sigma: (0.034321, 0.054914, 0.068642, 0.082371, 0.102964) (x0.686425)
- final_demand_share: 0.568250 (every sector)
- holding_loss: 0.032359
- input_complementarity: 2.0
- locked (PERTURBATION_EXCLUDED) at Phase-1 values: delivery_timing, arrival_probs, input_holding_loss, trade_tau, alloc_eta_request

### Configuration 7 x Random

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0721 s, max 0.1804 s
- conservation: max |residual| 2.665e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.339022, max 2.15048; min T/T_min 11.3007, max T/T_upper 0.147375; failures 0 - HELD
- S: min 0, max S/S_max 0.557552 (S_max 3); failures 0 - HELD
- fill: min 0.00162998, max 1; failures 0 - HELD
- flags: none

### Configuration 7 x TruthfulMyopic

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0707 s, max 0.2036 s
- conservation: max |residual| 1.776e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.595031, max 0.787964; min T/T_min 19.8344, max T/T_upper 0.0540002; failures 0 - HELD
- S: min 0, max S/S_max 0.269665 (S_max 3); failures 0 - HELD
- fill: min 1, max 1; failures 0 - HELD
- flags: none

### Configuration 7 x Padder

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0702 s, max 0.2041 s
- conservation: max |residual| 4.441e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.6, max 0.702996; min T/T_min 20, max T/T_upper 0.0481772; failures 0 - HELD
- S: min 0, max S/S_max 0.11618 (S_max 3); failures 0 - HELD
- fill: min 0.406098, max 1; failures 0 - HELD
- Padder shortage (T-B3 property, fill < 1 somewhere): True - HELD
- flags: none

## Configuration 8: `ecec9b74f34da5973cd1fc313e88115de301f2ba3857ce5b74e53b67b82509ed`

- run directory: `runs/ecec9b74f34da5973cd1fc313e88115de301f2ba3857ce5b74e53b67b82509ed/` (manifest.json, ledger/<agent>/)
- kind: SUPPLY perturbation
- yield_sigma: (0.053776, 0.086041, 0.107552, 0.129062, 0.161327) (x1.075516)
- final_demand_share: 0.698884 (every sector)
- holding_loss: 0.049042
- input_complementarity: 8.0
- locked (PERTURBATION_EXCLUDED) at Phase-1 values: delivery_timing, arrival_probs, input_holding_loss, trade_tau, alloc_eta_request

### Configuration 8 x Random

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0700 s, max 0.1753 s
- conservation: max |residual| 1.332e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.339022, max 2.15048; min T/T_min 11.3007, max T/T_upper 0.147375; failures 0 - HELD
- S: min 0, max S/S_max 0.341905 (S_max 3); failures 0 - HELD
- fill: min 0.00164676, max 1; failures 0 - HELD
- flags: none

### Configuration 8 x TruthfulMyopic

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0701 s, max 0.1740 s
- conservation: max |residual| 4.441e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.227975, max 0.619843; min T/T_min 7.59917, max T/T_upper 0.0424786; failures 0 - HELD
- S: min 0, max S/S_max 0.205126 (S_max 3); failures 0 - HELD
- fill: min 1, max 1; failures 0 - HELD
- flags: none

### Configuration 8 x Padder

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0699 s, max 0.1930 s
- conservation: max |residual| 4.441e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.6, max 0.702996; min T/T_min 20, max T/T_upper 0.0481772; failures 0 - HELD
- S: min 0, max S/S_max 0.120952 (S_max 3); failures 0 - HELD
- fill: min 0.190139, max 1; failures 0 - HELD
- Padder shortage (T-B3 property, fill < 1 somewhere): True - HELD
- flags: none

## Configuration 9: `f87364165c34bbfd73aea1905ebba615ce67de32481bd70e5961b60fb80fe576`

- run directory: `runs/f87364165c34bbfd73aea1905ebba615ce67de32481bd70e5961b60fb80fe576/` (manifest.json, ledger/<agent>/)
- kind: SUPPLY perturbation
- yield_sigma: (0.076416, 0.122265, 0.152831, 0.183398, 0.229247) (x1.528313)
- final_demand_share: 0.560184 (every sector)
- holding_loss: 0.034422
- input_complementarity: inf
- locked (PERTURBATION_EXCLUDED) at Phase-1 values: delivery_timing, arrival_probs, input_holding_loss, trade_tau, alloc_eta_request

### Configuration 9 x Random

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0695 s, max 0.2194 s
- conservation: max |residual| 2.665e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.339022, max 2.15048; min T/T_min 11.3007, max T/T_upper 0.147375; failures 0 - HELD
- S: min 0, max S/S_max 0.588818 (S_max 3); failures 0 - HELD
- fill: min 0.00166479, max 1; failures 0 - HELD
- flags: none

### Configuration 9 x TruthfulMyopic

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0696 s, max 0.2099 s
- conservation: max |residual| 8.882e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.534806, max 0.83982; min T/T_min 17.8269, max T/T_upper 0.0575539; failures 0 - HELD
- S: min 0, max S/S_max 0.315108 (S_max 3); failures 0 - HELD
- fill: min 1, max 1; failures 0 - HELD
- flags: none

### Configuration 9 x Padder

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0682 s, max 0.1657 s
- conservation: max |residual| 8.882e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.6, max 0.702996; min T/T_min 20, max T/T_upper 0.0481772; failures 0 - HELD
- S: min 0, max S/S_max 0.139652 (S_max 3); failures 0 - HELD
- fill: min 0.372772, max 1; failures 0 - HELD
- Padder shortage (T-B3 property, fill < 1 somewhere): True - HELD
- flags: none

## Configuration 10: `a02c8243b61a2cae159111854d193c47b3501e84ac9fdf0d109d7e20d0d58499`

- run directory: `runs/a02c8243b61a2cae159111854d193c47b3501e84ac9fdf0d109d7e20d0d58499/` (manifest.json, ledger/<agent>/)
- kind: SUPPLY perturbation
- yield_sigma: (0.035132, 0.056212, 0.070264, 0.084317, 0.105397) (x0.702645)
- final_demand_share: 0.588595 (every sector)
- holding_loss: 0.026268
- input_complementarity: 8.0
- locked (PERTURBATION_EXCLUDED) at Phase-1 values: delivery_timing, arrival_probs, input_holding_loss, trade_tau, alloc_eta_request

### Configuration 10 x Random

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0707 s, max 0.1985 s
- conservation: max |residual| 2.665e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.339022, max 2.15048; min T/T_min 11.3007, max T/T_upper 0.147375; failures 0 - HELD
- S: min 0, max S/S_max 0.531426 (S_max 3); failures 0 - HELD
- fill: min 0.00163071, max 1; failures 0 - HELD
- flags: none

### Configuration 10 x TruthfulMyopic

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0697 s, max 0.1665 s
- conservation: max |residual| 8.882e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.594625, max 0.759452; min T/T_min 19.8208, max T/T_upper 0.0520462; failures 0 - HELD
- S: min 0, max S/S_max 0.254984 (S_max 3); failures 0 - HELD
- fill: min 1, max 1; failures 0 - HELD
- flags: none

### Configuration 10 x Padder

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0698 s, max 0.2124 s
- conservation: max |residual| 4.441e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.6, max 0.702996; min T/T_min 20, max T/T_upper 0.0481772; failures 0 - HELD
- S: min 0, max S/S_max 0.116593 (S_max 3); failures 0 - HELD
- fill: min 0.405431, max 1; failures 0 - HELD
- Padder shortage (T-B3 property, fill < 1 somewhere): True - HELD
- flags: none

## Configuration 11: `cdd315cd29dfcf07185a59a429ce199c1c14c951ef2780d7917b820d8bc363d4`

- run directory: `runs/cdd315cd29dfcf07185a59a429ce199c1c14c951ef2780d7917b820d8bc363d4/` (manifest.json, ledger/<agent>/)
- kind: SUPPLY perturbation
- yield_sigma: (0.048268, 0.077229, 0.096536, 0.115844, 0.144804) (x0.965363)
- final_demand_share: 0.494334 (every sector)
- holding_loss: 0.044474
- input_complementarity: 2.0
- locked (PERTURBATION_EXCLUDED) at Phase-1 values: delivery_timing, arrival_probs, input_holding_loss, trade_tau, alloc_eta_request

### Configuration 11 x Random

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0711 s, max 0.1999 s
- conservation: max |residual| 2.665e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.339022, max 2.15048; min T/T_min 11.3007, max T/T_upper 0.147375; failures 0 - HELD
- S: min 0, max S/S_max 0.560993 (S_max 3); failures 0 - HELD
- fill: min 0.00164213, max 1; failures 0 - HELD
- flags: none

### Configuration 11 x TruthfulMyopic

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0714 s, max 0.1705 s
- conservation: max |residual| 1.776e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.580998, max 0.829692; min T/T_min 19.3666, max T/T_upper 0.0568598; failures 0 - HELD
- S: min 0, max S/S_max 0.293903 (S_max 3); failures 0 - HELD
- fill: min 1, max 1; failures 0 - HELD
- flags: none

### Configuration 11 x Padder

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0701 s, max 0.1632 s
- conservation: max |residual| 8.882e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.6, max 0.702996; min T/T_min 20, max T/T_upper 0.0481772; failures 0 - HELD
- S: min 0, max S/S_max 0.123485 (S_max 3); failures 0 - HELD
- fill: min 0.394765, max 1; failures 0 - HELD
- Padder shortage (T-B3 property, fill < 1 somewhere): True - HELD
- flags: none

## Configuration 12: `79c56642525655c4f6af829836026b846a7046c179bc80561d30d96812b4ad12`

- run directory: `runs/79c56642525655c4f6af829836026b846a7046c179bc80561d30d96812b4ad12/` (manifest.json, ledger/<agent>/)
- kind: SUPPLY perturbation
- yield_sigma: (0.051835, 0.082935, 0.103669, 0.124403, 0.155504) (x1.036693)
- final_demand_share: 0.528612 (every sector)
- holding_loss: 0.016093
- input_complementarity: inf
- locked (PERTURBATION_EXCLUDED) at Phase-1 values: delivery_timing, arrival_probs, input_holding_loss, trade_tau, alloc_eta_request

### Configuration 12 x Random

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0708 s, max 0.2171 s
- conservation: max |residual| 2.665e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.339022, max 2.15048; min T/T_min 11.3007, max T/T_upper 0.147375; failures 0 - HELD
- S: min 0, max S/S_max 0.580439 (S_max 3); failures 0 - HELD
- fill: min 0.00164514, max 1; failures 0 - HELD
- flags: none

### Configuration 12 x TruthfulMyopic

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0705 s, max 0.1823 s
- conservation: max |residual| 8.882e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.575077, max 0.842295; min T/T_min 19.1692, max T/T_upper 0.0577235; failures 0 - HELD
- S: min 0, max S/S_max 0.300063 (S_max 3); failures 0 - HELD
- fill: min 1, max 1; failures 0 - HELD
- flags: none

### Configuration 12 x Padder

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0690 s, max 0.1659 s
- conservation: max |residual| 8.882e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.6, max 0.702996; min T/T_min 20, max T/T_upper 0.0481772; failures 0 - HELD
- S: min 0, max S/S_max 0.125426 (S_max 3); failures 0 - HELD
- fill: min 0.391914, max 1; failures 0 - HELD
- Padder shortage (T-B3 property, fill < 1 somewhere): True - HELD
- flags: none

## Configuration 13: `bc8ae23135d2acef205b4fd04e22a69c7651d1b7a356dc66baad7c512bbaacc0`

- run directory: `runs/bc8ae23135d2acef205b4fd04e22a69c7651d1b7a356dc66baad7c512bbaacc0/` (manifest.json, ledger/<agent>/)
- kind: SUPPLY perturbation
- yield_sigma: (0.069573, 0.111316, 0.139145, 0.166974, 0.208718) (x1.391450)
- final_demand_share: 0.435164 (every sector)
- holding_loss: 0.019581
- input_complementarity: 2.0
- locked (PERTURBATION_EXCLUDED) at Phase-1 values: delivery_timing, arrival_probs, input_holding_loss, trade_tau, alloc_eta_request

### Configuration 13 x Random

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0698 s, max 0.1621 s
- conservation: max |residual| 2.665e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.339022, max 2.15048; min T/T_min 11.3007, max T/T_upper 0.147375; failures 0 - HELD
- S: min 0, max S/S_max 0.592435 (S_max 3); failures 0 - HELD
- fill: min 0.00165951, max 1; failures 0 - HELD
- flags: none

### Configuration 13 x TruthfulMyopic

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0702 s, max 0.1813 s
- conservation: max |residual| 1.776e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.545915, max 0.907703; min T/T_min 18.1972, max T/T_upper 0.062206; failures 0 - HELD
- S: min 0, max S/S_max 0.332405 (S_max 3); failures 0 - HELD
- fill: min 1, max 1; failures 0 - HELD
- flags: none

### Configuration 13 x Padder

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0695 s, max 0.1824 s
- conservation: max |residual| 8.882e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.6, max 0.702996; min T/T_min 20, max T/T_upper 0.0481772; failures 0 - HELD
- S: min 0, max S/S_max 0.135538 (S_max 3); failures 0 - HELD
- fill: min 0.378013, max 1; failures 0 - HELD
- Padder shortage (T-B3 property, fill < 1 somewhere): True - HELD
- flags: none

## Configuration 14: `753fb1766430d7bcb03166446e5880a4c60e3b2ec71a84e992ad493a98afde44`

- run directory: `runs/753fb1766430d7bcb03166446e5880a4c60e3b2ec71a84e992ad493a98afde44/` (manifest.json, ledger/<agent>/)
- kind: SUPPLY perturbation
- yield_sigma: (0.042037, 0.067259, 0.084074, 0.100888, 0.12611) (x0.840736)
- final_demand_share: 0.549275 (every sector)
- holding_loss: 0.004201
- input_complementarity: inf
- locked (PERTURBATION_EXCLUDED) at Phase-1 values: delivery_timing, arrival_probs, input_holding_loss, trade_tau, alloc_eta_request

### Configuration 14 x Random

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0683 s, max 0.1488 s
- conservation: max |residual| 1.776e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.339022, max 2.15048; min T/T_min 11.3007, max T/T_upper 0.147375; failures 0 - HELD
- S: min 0, max S/S_max 0.579818 (S_max 3); failures 0 - HELD
- fill: min 0.00163677, max 1; failures 0 - HELD
- flags: none

### Configuration 14 x TruthfulMyopic

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0684 s, max 0.1831 s
- conservation: max |residual| 8.882e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.588295, max 0.812051; min T/T_min 19.6098, max T/T_upper 0.0556509; failures 0 - HELD
- S: min 0, max S/S_max 0.283408 (S_max 3); failures 0 - HELD
- fill: min 1, max 1; failures 0 - HELD
- flags: none

### Configuration 14 x Padder

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0676 s, max 0.1797 s
- conservation: max |residual| 4.441e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.6, max 0.702996; min T/T_min 20, max T/T_upper 0.0481772; failures 0 - HELD
- S: min 0, max S/S_max 0.120166 (S_max 3); failures 0 - HELD
- fill: min 0.399792, max 1; failures 0 - HELD
- Padder shortage (T-B3 property, fill < 1 somewhere): True - HELD
- flags: none

## Configuration 15: `024e35ed5d07cd2de04a64c08ac52f91c65d905ae9ee34a3c6c92fb8c3ae384d`

- run directory: `runs/024e35ed5d07cd2de04a64c08ac52f91c65d905ae9ee34a3c6c92fb8c3ae384d/` (manifest.json, ledger/<agent>/)
- kind: SUPPLY perturbation
- yield_sigma: (0.087448, 0.139917, 0.174897, 0.209876, 0.262345) (x1.748966)
- final_demand_share: 0.614839 (every sector)
- holding_loss: 0.011968
- input_complementarity: inf
- locked (PERTURBATION_EXCLUDED) at Phase-1 values: delivery_timing, arrival_probs, input_holding_loss, trade_tau, alloc_eta_request

### Configuration 15 x Random

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0681 s, max 0.1711 s
- conservation: max |residual| 1.776e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.339022, max 2.15048; min T/T_min 11.3007, max T/T_upper 0.147375; failures 0 - HELD
- S: min 0, max S/S_max 0.470843 (S_max 3); failures 0 - HELD
- fill: min 0.00167298, max 1; failures 0 - HELD
- flags: none

### Configuration 15 x TruthfulMyopic

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0684 s, max 0.2049 s
- conservation: max |residual| 8.882e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.484808, max 0.622731; min T/T_min 16.1603, max T/T_upper 0.0426765; failures 0 - HELD
- S: min 0, max S/S_max 0.208959 (S_max 3); failures 0 - HELD
- fill: min 1, max 1; failures 0 - HELD
- flags: none

### Configuration 15 x Padder

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0674 s, max 0.1609 s
- conservation: max |residual| 4.441e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.6, max 0.702996; min T/T_min 20, max T/T_upper 0.0481772; failures 0 - HELD
- S: min 0, max S/S_max 0.146543 (S_max 3); failures 0 - HELD
- fill: min 0.364461, max 1; failures 0 - HELD
- Padder shortage (T-B3 property, fill < 1 somewhere): True - HELD
- flags: none

## Configuration 16: `608f9a36f2d3c7e2f0e6e8535df2b87c65b17f29a88e742ca7f7ecf9d7b7fbd5`

- run directory: `runs/608f9a36f2d3c7e2f0e6e8535df2b87c65b17f29a88e742ca7f7ecf9d7b7fbd5/` (manifest.json, ledger/<agent>/)
- kind: SUPPLY perturbation
- yield_sigma: (0.029393, 0.047028, 0.058785, 0.070542, 0.088178) (x0.587852)
- final_demand_share: 0.434447 (every sector)
- holding_loss: 0.007514
- input_complementarity: inf
- locked (PERTURBATION_EXCLUDED) at Phase-1 values: delivery_timing, arrival_probs, input_holding_loss, trade_tau, alloc_eta_request

### Configuration 16 x Random

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0679 s, max 0.1396 s
- conservation: max |residual| 2.665e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.339022, max 2.15048; min T/T_min 11.3007, max T/T_upper 0.147375; failures 0 - HELD
- S: min 0, max S/S_max 0.609615 (S_max 3); failures 0 - HELD
- fill: min 0.00162555, max 1; failures 0 - HELD
- flags: none

### Configuration 16 x TruthfulMyopic

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0681 s, max 0.1536 s
- conservation: max |residual| 1.776e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.597494, max 0.777696; min T/T_min 19.9165, max T/T_upper 0.0532965; failures 0 - HELD
- S: min 0, max S/S_max 0.263118 (S_max 3); failures 0 - HELD
- fill: min 1, max 1; failures 0 - HELD
- flags: none

### Configuration 16 x Padder

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0670 s, max 0.1605 s
- conservation: max |residual| 8.882e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.6, max 0.702996; min T/T_min 20, max T/T_upper 0.0481772; failures 0 - HELD
- S: min 0, max S/S_max 0.113703 (S_max 3); failures 0 - HELD
- fill: min 0.410174, max 1; failures 0 - HELD
- Padder shortage (T-B3 property, fill < 1 somewhere): True - HELD
- flags: none

## Configuration 17: `04d7845532884c746d76324bfc1bd83399f590cabfacaee73ab7cbafc1e83f65`

- run directory: `runs/04d7845532884c746d76324bfc1bd83399f590cabfacaee73ab7cbafc1e83f65/` (manifest.json, ledger/<agent>/)
- kind: SUPPLY perturbation
- yield_sigma: (0.058775, 0.094041, 0.117551, 0.141061, 0.176326) (x1.175509)
- final_demand_share: 0.618530 (every sector)
- holding_loss: 0.011532
- input_complementarity: inf
- locked (PERTURBATION_EXCLUDED) at Phase-1 values: delivery_timing, arrival_probs, input_holding_loss, trade_tau, alloc_eta_request

### Configuration 17 x Random

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0673 s, max 0.1525 s
- conservation: max |residual| 1.776e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.339022, max 2.15048; min T/T_min 11.3007, max T/T_upper 0.147375; failures 0 - HELD
- S: min 0, max S/S_max 0.439482 (S_max 3); failures 0 - HELD
- fill: min 0.00165088, max 1; failures 0 - HELD
- flags: none

### Configuration 17 x TruthfulMyopic

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0668 s, max 0.1874 s
- conservation: max |residual| 8.882e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.470081, max 0.620216; min T/T_min 15.6694, max T/T_upper 0.0425041; failures 0 - HELD
- S: min 0, max S/S_max 0.20537 (S_max 3); failures 0 - HELD
- fill: min 1, max 1; failures 0 - HELD
- flags: none

### Configuration 17 x Padder

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0674 s, max 0.1790 s
- conservation: max |residual| 8.882e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.6, max 0.702996; min T/T_min 20, max T/T_upper 0.0481772; failures 0 - HELD
- S: min 0, max S/S_max 0.12929 (S_max 3); failures 0 - HELD
- fill: min 0.38642, max 1; failures 0 - HELD
- Padder shortage (T-B3 property, fill < 1 somewhere): True - HELD
- flags: none

## Configuration 18: `0aeaebe86e6b87180266be7ca60db260d5f5ebf11c1e77f172dc10b7579d2600`

- run directory: `runs/0aeaebe86e6b87180266be7ca60db260d5f5ebf11c1e77f172dc10b7579d2600/` (manifest.json, ledger/<agent>/)
- kind: SUPPLY perturbation
- yield_sigma: (0.055341, 0.088546, 0.110683, 0.132819, 0.166024) (x1.106828)
- final_demand_share: 0.379405 (every sector)
- holding_loss: 0.004538
- input_complementarity: 2.0
- locked (PERTURBATION_EXCLUDED) at Phase-1 values: delivery_timing, arrival_probs, input_holding_loss, trade_tau, alloc_eta_request

### Configuration 18 x Random

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0688 s, max 0.1759 s
- conservation: max |residual| 3.553e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.339022, max 2.15048; min T/T_min 11.3007, max T/T_upper 0.147375; failures 0 - HELD
- S: min 0, max S/S_max 0.644348 (S_max 3); failures 0 - HELD
- fill: min 0.00164806, max 1; failures 0 - HELD
- flags: none

### Configuration 18 x TruthfulMyopic

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0688 s, max 0.1835 s
- conservation: max |residual| 1.776e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.569273, max 0.854862; min T/T_min 18.9758, max T/T_upper 0.0585848; failures 0 - HELD
- S: min 0, max S/S_max 0.306229 (S_max 3); failures 0 - HELD
- fill: min 1, max 1; failures 0 - HELD
- flags: none

### Configuration 18 x Padder

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0672 s, max 0.1532 s
- conservation: max |residual| 8.882e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.6, max 0.702996; min T/T_min 20, max T/T_upper 0.0481772; failures 0 - HELD
- S: min 0, max S/S_max 0.127364 (S_max 3); failures 0 - HELD
- fill: min 0.389129, max 1; failures 0 - HELD
- Padder shortage (T-B3 property, fill < 1 somewhere): True - HELD
- flags: none

## Configuration 19: `c65ae1d6df7033a194eaadbbdf914a9a567634ef3e08d3ad63bec654121420e1`

- run directory: `runs/c65ae1d6df7033a194eaadbbdf914a9a567634ef3e08d3ad63bec654121420e1/` (manifest.json, ledger/<agent>/)
- kind: SUPPLY perturbation
- yield_sigma: (0.068525, 0.10964, 0.13705, 0.16446, 0.205575) (x1.370499)
- final_demand_share: 0.419478 (every sector)
- holding_loss: 0.033600
- input_complementarity: inf
- locked (PERTURBATION_EXCLUDED) at Phase-1 values: delivery_timing, arrival_probs, input_holding_loss, trade_tau, alloc_eta_request

### Configuration 19 x Random

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0676 s, max 0.1348 s
- conservation: max |residual| 2.665e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.339022, max 2.15048; min T/T_min 11.3007, max T/T_upper 0.147375; failures 0 - HELD
- S: min 0, max S/S_max 0.585724 (S_max 3); failures 0 - HELD
- fill: min 0.00165869, max 1; failures 0 - HELD
- flags: none

### Configuration 19 x TruthfulMyopic

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0665 s, max 0.3263 s
- conservation: max |residual| 1.776e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.547623, max 0.90371; min T/T_min 18.2541, max T/T_upper 0.0619324; failures 0 - HELD
- S: min 0, max S/S_max 0.330414 (S_max 3); failures 0 - HELD
- fill: min 1, max 1; failures 0 - HELD
- flags: none

### Configuration 19 x Padder

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0662 s, max 0.1701 s
- conservation: max |residual| 8.882e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.6, max 0.702996; min T/T_min 20, max T/T_upper 0.0481772; failures 0 - HELD
- S: min 0, max S/S_max 0.134919 (S_max 3); failures 0 - HELD
- fill: min 0.378821, max 1; failures 0 - HELD
- Padder shortage (T-B3 property, fill < 1 somewhere): True - HELD
- flags: none

## Configuration 20: `87115b16eb2a177459c4e08fb38090bfced183304a7867ad5af454026ddf3fb8`

- run directory: `runs/87115b16eb2a177459c4e08fb38090bfced183304a7867ad5af454026ddf3fb8/` (manifest.json, ledger/<agent>/)
- kind: SUPPLY perturbation
- yield_sigma: (0.095658, 0.153054, 0.191317, 0.22958, 0.286975) (x1.913170)
- final_demand_share: 0.446044 (every sector)
- holding_loss: 0.005275
- input_complementarity: 2.0
- locked (PERTURBATION_EXCLUDED) at Phase-1 values: delivery_timing, arrival_probs, input_holding_loss, trade_tau, alloc_eta_request

### Configuration 20 x Random

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0670 s, max 0.1447 s
- conservation: max |residual| 2.665e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.339022, max 2.15048; min T/T_min 11.3007, max T/T_upper 0.147375; failures 0 - HELD
- S: min 0, max S/S_max 0.623181 (S_max 3); failures 0 - HELD
- fill: min 0.00167881, max 1; failures 0 - HELD
- flags: none

### Configuration 20 x TruthfulMyopic

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0672 s, max 0.1567 s
- conservation: max |residual| 1.776e-15 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.504046, max 0.932924; min T/T_min 16.8015, max T/T_upper 0.0639344; failures 0 - HELD
- S: min 0, max S/S_max 0.369796 (S_max 3); failures 0 - HELD
- fill: min 1, max 1; failures 0 - HELD
- flags: none

### Configuration 20 x Padder

- episodes: 2000; periods: 18000; ledger parts: 20
- wall clock per episode: mean 0.0660 s, max 0.1374 s
- conservation: max |residual| 8.882e-16 - HELD
- non-finite: 0 (AMBIGUITY-007 NaN placeholders: 5400000) - HELD
- T: min 0.6, max 0.702996; min T/T_min 20, max T/T_upper 0.0481772; failures 0 - HELD
- S: min 0, max S/S_max 0.151885 (S_max 3); failures 0 - HELD
- fill: min 0.358385, max 1; failures 0 - HELD
- Padder shortage (T-B3 property, fill < 1 somewhere): True - HELD
- flags: none
