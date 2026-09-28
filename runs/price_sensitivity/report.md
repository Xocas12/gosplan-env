# Price-vector sensitivity (WO-036, PLAN sections 2.9.4, 7.5)

Perturbation `p_j exp(u_j)`, `u ~ N(0, 0.3**2)`, seeds (11, 12, 13) (keyed purpose `pricepert`; spec/P3_REVISION.md S6). Sign changes of `specification_gap` are listed first.

## Sign changes

- G3 / contrasts: C0: mean over 30 rows keeps its sign; 0 of 30 rows change sign
- contrasts: C_OGAS: mean over 15 rows keeps its sign; 0 of 15 rows change sign
- contrasts: C_AUDIT: mean over 15 rows keeps its sign; 0 of 15 rows change sign
- contrasts: C_INC: mean over 15 rows keeps its sign; 1 of 15 rows change sign
- contrasts: C_BOTH: mean over 15 rows keeps its sign; 0 of 15 rows change sign

## Headline metrics, mean over rows (base, then seeds 11, 12, 13)

### G3 / contrasts: C0 (recomputation)
- padding_index: 15.7508, 15.9019, 15.3936, 15.7489
- welfare_ratio: 0.0339, 0.0339, 0.0339, 0.0339
- specification_gap: 0.0105, 0.0105, 0.0106, 0.0105

### contrasts: C_OGAS (recomputation)
- padding_index: 3370.8822, 3707.1414, 2955.2849, 3514.6404
- welfare_ratio: 0.0065, 0.0065, 0.0065, 0.0065
- specification_gap: 0.0136, 0.0135, 0.0134, 0.0139

### contrasts: C_AUDIT (recomputation)
- padding_index: 13.1511, 13.5948, 12.4030, 13.1957
- welfare_ratio: 0.0402, 0.0402, 0.0402, 0.0402
- specification_gap: 0.0322, 0.0313, 0.0330, 0.0326

### contrasts: C_INC (rerun)
_recomputation only; the objective reads prices, a behavioural answer needs a rerun, which is not performed for compute reasons (spec/P3_REVISION.md S6)_
- padding_index: 39.4181, 38.8661, 39.7422, 39.2358
- welfare_ratio: 0.1012, 0.1012, 0.1012, 0.1012
- specification_gap: 0.0242, 0.0242, 0.0242, 0.0242

### contrasts: C_BOTH (rerun)
_recomputation only; the objective reads prices, a behavioural answer needs a rerun, which is not performed for compute reasons (spec/P3_REVISION.md S6)_
- padding_index: 63.2391, 61.2403, 66.9154, 64.8501
- welfare_ratio: 0.0334, 0.0334, 0.0334, 0.0334
- specification_gap: 0.0408, 0.0410, 0.0408, 0.0407

## Tables without a price-weighted metric

Phase-1 gate tables (runs/phase1_gate), the DP-vs-PPO tables and the estimator-bias tables carry no price-weighted headline metric (report ratios, padding in ratio units, excess mass), so a reprice leaves them unchanged; they are listed here so their omission is explicit.
