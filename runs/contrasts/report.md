# Contrasts (WO-032, PLAN section 4.3)

Design: spec/P3_REVISION.md S2. C0 = the 30 G3 runs; other arms 15 seeds each (reduced from PLAN's 30 for compute), common random numbers (`seed_env = 1000 + s`). IQM with a 95% percentile bootstrap over seeds (10,000 resamples). Limitations L1, L2 (runs/G2_record.md) and the G3 record apply.

Reporting rules (PLAN section 4.3):
1. **C_AUDIT is dual-classified and is never folded into C_OGAS.**
2. **No 'X% of the loss is informational' statement without a named contrast and a CI.**

Out of the PLAN section 3 sweep range (reported, not clamped): {'C_AUDIT': 0.4} (`audit_rate`, range [0.01, 0.30]).

## Outcomes per arm (IQM [95% CI])

| arm | seeds | welfare_ratio | padding_index | specification_gap | convergence |
|---|---|---|---|---|---|
| C0 | 30 | 0.0295 [0.0170, 0.0435] | 2.1848 [1.3187, 5.5575] | 0.0107 [0.0034, 0.0180] | max expl. -3.790 |
| C_OGAS | 15 | 0.0002 [0.0000, 0.0014] | 809.8061 [205.0264, 4824.3460] | 0.0118 [0.0083, 0.0167] | max expl. -10.014 |
| C_AUDIT | 15 | 0.0240 [0.0080, 0.0497] | 3.5385 [1.0798, 11.5524] | 0.0019 [-0.0091, 0.0104] | max expl. -5.353 |
| C_INC | 15 | 0.0974 [0.0492, 0.1501] | 2.8105 [0.9736, 8.9241] | 0.0241 [-0.0104, 0.0601] | max expl. -1.178 |
| C_BOTH | 15 | 0.0249 [0.0084, 0.0499] | 6.4780 [2.3216, 24.1774] | 0.0390 [0.0310, 0.0501] | max expl. -4.368 |

## Gap closed, seed-paired (Delta_X = welfare_ratio(C_X) - welfare_ratio(C0))

- C_OGAS: -0.0339 [-0.0503, -0.0163] over 15 paired seeds
- C_AUDIT: -0.0114 [-0.0336, 0.0201] over 15 paired seeds
- C_INC: 0.0676 [0.0208, 0.1106] over 15 paired seeds
- C_BOTH: -0.0083 [-0.0287, 0.0121] over 15 paired seeds
- Interaction I = Delta_BOTH - Delta_OGAS - Delta_INC: -0.0429 [-0.0811, -0.0018] over 15 paired seeds

## Oracle

- C0: W_oracle 1.9836, val_oracle 38.1873, HiGHS (HiGHS via OR-Tools 9.15.6755), gap 0.00e+00
- C_OGAS: W_oracle 1.9836, val_oracle 38.1873, HiGHS (HiGHS via OR-Tools 9.15.6755), gap 0.00e+00
- C_AUDIT: W_oracle 1.9836, val_oracle 38.1873, HiGHS (HiGHS via OR-Tools 9.15.6755), gap 0.00e+00
- C_INC: W_oracle 1.9836, val_oracle 38.1873, HiGHS (HiGHS via OR-Tools 9.15.6755), gap 0.00e+00
- C_BOTH: W_oracle 1.9836, val_oracle 38.1873, HiGHS (HiGHS via OR-Tools 9.15.6755), gap 0.00e+00

Flags: bailed_out:0, bailed_out:1, bailed_out:10, bailed_out:11, bailed_out:12, bailed_out:13, bailed_out:14, bailed_out:15, bailed_out:16, bailed_out:17, bailed_out:18, bailed_out:19, bailed_out:2, bailed_out:3, bailed_out:4, bailed_out:5, bailed_out:6, bailed_out:7, bailed_out:8, bailed_out:9. git `bde9abab56f347922a444c78008e7f8f9248d7c1`.
