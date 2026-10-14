# Reconciliation power against inflation size, and the null it rests on (#110)

Supplement to the S4 power curve in `report.md`: descriptive, added after the P3 run, and not pre-registered. Written by `python -m gosplan.experiments.estimator_bias --reconciliation`. Estimator: gosplan.metrics._fallback 0.1.1+fallback.

The call is S4's: per (period, good), the sellers' claimed intermediate supply `(1 - phi_j) sum R_i` against the buyers' receipts of good j at the next DELIVER, 20 TruthfulMyopic episodes per ledger at `seed_env + 5000 + rep`, alpha = 0.05. Which claims are inflated is drawn as in S4, from the replicate and the share only, so the rows for one ledger differ in the inflation factor alone.

## Null residuals (no claim inflated)

Relative residual = (claimed - received) / claimed, pooled over every (period, good) key of every episode. Shipment gap = the largest |received - (1 - phi_j) shipped_j| / received over the same keys. Size = share of episodes rejected with no claim inflated.

| ledger | settings | keys | mean | sd | max abs | shipment gap | size |
|---|---|---|---|---|---|---|---|
| Phase 1 (S4's pre-registered ledger) | no change | 1050 | 9.73e-18 | 3.66e-16 | 1.31e-15 | 1.31e-15 | 0.00 |
| Phase 1 + C0's channel noise | channel_noise = 0.05 | 1050 | 5.80e-18 | 5.05e-16 | 2.33e-15 | 2.33e-15 | 0.00 |
| Phase 1 + C0's report lag | report_lag = 1 | 1050 | 1.50e-17 | 3.78e-16 | 1.27e-15 | 1.27e-15 | 0.00 |
| Phase 1 + the locked stochastic delivery | delivery_timing = stochastic, arrival_probs = (0.25, 0.25, 0.25, 0.25) | 1050 | 3.03e-17 | 4.05e-16 | 1.31e-15 | 1.31e-15 | 0.00 |
| Phase 1 + the locked input holding loss | input_holding_loss = 0.01 | 1050 | 9.73e-18 | 3.66e-16 | 1.31e-15 | 1.31e-15 | 0.00 |
| Phase 1 + C0's ministry layer | ministry_passthrough = 0.75, n_ministries = 5, ministry_pad = 0.5 | 1050 | 8.25e-03 | 5.38e-03 | 2.90e-02 | 1.18e-15 | 1.00 |
| C0, seed 0 (the Phase-2 baseline) | every C0 setting | 990 | 1.61e-02 | 2.42e-02 | 1.21e-01 | 2.46e-15 | 1.00 |

## Rejection rate by inflated share and inflation factor

| ledger | factor | share 0.0 | share 0.05 | share 0.1 | share 0.2 | share 0.3 | share 0.5 |
|---|---|---|---|---|---|---|---|
| Phase 1 (S4's pre-registered ledger) | x1.2 | 0.00 | 0.90 | 1.00 | 1.00 | 1.00 | 1.00 |
| Phase 1 (S4's pre-registered ledger) | x1.01 | 0.00 | 0.90 | 1.00 | 1.00 | 1.00 | 1.00 |
| Phase 1 (S4's pre-registered ledger) | x1.0001 | 0.00 | 0.90 | 1.00 | 1.00 | 1.00 | 1.00 |
| Phase 1 (S4's pre-registered ledger) | x1.000001 | 0.00 | 0.90 | 1.00 | 1.00 | 1.00 | 1.00 |
| Phase 1 + C0's channel noise | x1.2 | 0.00 | 0.90 | 1.00 | 1.00 | 1.00 | 1.00 |
| Phase 1 + C0's channel noise | x1.01 | 0.00 | 0.90 | 1.00 | 1.00 | 1.00 | 1.00 |
| Phase 1 + C0's channel noise | x1.0001 | 0.00 | 0.90 | 1.00 | 1.00 | 1.00 | 1.00 |
| Phase 1 + C0's channel noise | x1.000001 | 0.00 | 0.90 | 1.00 | 1.00 | 1.00 | 1.00 |
| Phase 1 + C0's report lag | x1.2 | 0.00 | 0.90 | 1.00 | 1.00 | 1.00 | 1.00 |
| Phase 1 + C0's report lag | x1.01 | 0.00 | 0.90 | 1.00 | 1.00 | 1.00 | 1.00 |
| Phase 1 + C0's report lag | x1.0001 | 0.00 | 0.90 | 1.00 | 1.00 | 1.00 | 1.00 |
| Phase 1 + C0's report lag | x1.000001 | 0.00 | 0.90 | 1.00 | 1.00 | 1.00 | 1.00 |
| Phase 1 + the locked stochastic delivery | x1.2 | 0.00 | 0.90 | 1.00 | 1.00 | 1.00 | 1.00 |
| Phase 1 + the locked stochastic delivery | x1.01 | 0.00 | 0.90 | 1.00 | 1.00 | 1.00 | 1.00 |
| Phase 1 + the locked stochastic delivery | x1.0001 | 0.00 | 0.90 | 1.00 | 1.00 | 1.00 | 1.00 |
| Phase 1 + the locked stochastic delivery | x1.000001 | 0.00 | 0.90 | 1.00 | 1.00 | 1.00 | 1.00 |
| Phase 1 + the locked input holding loss | x1.2 | 0.00 | 0.90 | 1.00 | 1.00 | 1.00 | 1.00 |
| Phase 1 + the locked input holding loss | x1.01 | 0.00 | 0.90 | 1.00 | 1.00 | 1.00 | 1.00 |
| Phase 1 + the locked input holding loss | x1.0001 | 0.00 | 0.90 | 1.00 | 1.00 | 1.00 | 1.00 |
| Phase 1 + the locked input holding loss | x1.000001 | 0.00 | 0.90 | 1.00 | 1.00 | 1.00 | 1.00 |
| Phase 1 + C0's ministry layer | x1.2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Phase 1 + C0's ministry layer | x1.01 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Phase 1 + C0's ministry layer | x1.0001 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Phase 1 + C0's ministry layer | x1.000001 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| C0, seed 0 (the Phase-2 baseline) | x1.2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| C0, seed 0 (the Phase-2 baseline) | x1.01 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| C0, seed 0 (the Phase-2 baseline) | x1.0001 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| C0, seed 0 (the Phase-2 baseline) | x1.000001 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

## Reading

- Phase 1 (S4's pre-registered ledger): the truthful ledger reconciles to round-off (largest relative residual 1.3e-15), so there is no null noise; size 0.00. The curve is the same at every factor: rejection depends on how many (period, good) keys carry an inflated claim, not on how much the claim is inflated.
- Phase 1 + C0's channel noise: the truthful ledger reconciles to round-off (largest relative residual 2.3e-15), so there is no null noise; size 0.00. The curve is the same at every factor: rejection depends on how many (period, good) keys carry an inflated claim, not on how much the claim is inflated.
- Phase 1 + C0's report lag: the truthful ledger reconciles to round-off (largest relative residual 1.3e-15), so there is no null noise; size 0.00. The curve is the same at every factor: rejection depends on how many (period, good) keys carry an inflated claim, not on how much the claim is inflated.
- Phase 1 + the locked stochastic delivery: the truthful ledger reconciles to round-off (largest relative residual 1.3e-15), so there is no null noise; size 0.00. The curve is the same at every factor: rejection depends on how many (period, good) keys carry an inflated claim, not on how much the claim is inflated.
- Phase 1 + the locked input holding loss: the truthful ledger reconciles to round-off (largest relative residual 1.3e-15), so there is no null noise; size 0.00. The curve is the same at every factor: rejection depends on how many (period, good) keys carry an inflated claim, not on how much the claim is inflated.
- Phase 1 + C0's ministry layer: the truthful residuals are not zero (mean +8.25e-03 of the claimed supply, sd 5.38e-03; the most negative is -7.8e-16, which is round-off, so claims exceed receipts wherever the two differ), and the test rejects 1.00 of the truthful episodes. Every truthful episode is already rejected, so every curve is 1.00 and says nothing about the inflation.
- C0, seed 0 (the Phase-2 baseline): the truthful residuals are not zero (mean +1.61e-02 of the claimed supply, sd 2.42e-02; the most negative is -1.7e-15, which is round-off, so claims exceed receipts wherever the two differ), and the test rejects 1.00 of the truthful episodes. Every truthful episode is already rejected, so every curve is 1.00 and says nothing about the inflation.

On every ledger the buyers' receipts equal `(1 - phi_j)` times what the sellers shipped, to 2.5e-15, so a non-zero null residual is a gap between the claims and the shipments. Delivery timing moves when inputs arrive, not how much is delivered (`planner.deliver`, P2 revision R6). Under the ministry layer, delivery obligations are the ministry-forwarded claims, not the sellers' own (R10.3).

No ledger here has null residuals spread around zero: 5 reconcile to round-off and 2 have residuals of one sign, and the sizes are 0.00 and 1.00. None of them gives a power curve at the nominal size. A curve that measures detection of a given inflation needs a ledger whose claims and receipts differ by noise, and choosing that noise is a design decision this supplement does not make.
