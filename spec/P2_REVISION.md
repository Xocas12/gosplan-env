# Phase-2 spec revision (spec 2.0.0) - LEAD

Written after gate G2 (owner decision (a), `runs/G2_record.md`) and before any Phase-2 work order
is implemented. It freezes every behaviour PLAN section 2 leaves as a "P2 sketch", resolves the
four WO-006 questions left open, and states what Phase 2 does NOT cover. Work orders WO-021 to
WO-031 implement exactly this text. A gap found while implementing goes through an AMBIGUITY
REPORT, as in Phase 1 (CONTRACT rule 3).

Limitations carried from Phase 1 (cited by every Phase-2 report):
- L1 (criterion 1): the PPO learner does not recover the single-enterprise DP's mixed
  under-reporting strategy.
- L2 (criterion 2): the pre-registered bunching estimator is undefined on degenerate distributions
  and over-confident on sharply peaked ones (AMBIGUITY-011, -022).

The Phase-2 learner is the attempt-2 learner: per-period discount (AMBIGUITY-020) and report-head
initial std 0.05 (`phase1_gate.study_ppo_config()`).

---

## R1. Scope

**In Phase 2:**
- quality (2.6, 2.9.2)
- delivery timing (2.6)
- input holding loss `h_X` (2.11)
- report lag, downstream shortfall and targeted audits (2.7.4-2.7.5)
- soft budget
- trade (2.13)
- rule-based ministry (2.14)
- the trade peer observation block (2.4)
- the P2 default configuration (R11)
- setup cost `F`, which is already implemented

**Out of Phase 2.** `EnvConfig.validate()` rejects each of these with a message naming this
revision:
- `irs_alpha > 0`, `capital_dep > 0` (and so the invest action), `tech_drift_sigma > 0`,
  finite `price_lag` and `bonus_heterogeneity > 0`.
- `param_sharing != "shared"` is rejected by the PPO adapter rather than by `validate()`.
- Reason: the Phase-2 acceptance run (G3) needs none of them, and none enters a locked §4.2 value.
- They remain Phase-3 sweep toggles. Enabling one is a new revision with its own ambiguity round.
- Until then they cannot be switched on silently, which Phase 1 allowed (they were ignored).

## R2. Report lag (AMB-WO006-B)

- `State.claim_history`: `(N, 2)` float, the last two claims as they reached the planner.
  Column 0 is one period ago, column 1 two periods ago.
- It is initialised to `T_0` per enterprise, so a lagged planner starts from "on-plan" claims.
- At TARGET, after the target update, it shifts right and column 0 takes this period's forwarded
  claim (R10).
- With `report_lag = L > 0`, `make_planner_view` serves `claim_history[:, L-1]` as `claims` instead
  of the current forwarded claim.
- The same lagged claims drive the target rule (2.7.1) and allocation (2.7.2).
- The audit is unaffected: it always compares the current `R_i` with current stock.
- Aggregation and channel noise are applied after the lag selection, in that order.

## R3. Downstream shortfall (AMB-WO006-D)

- The planner's knowledge of buyers' complaints about seller `i`:
  `downstream_shortfall_i = shortfall_visibility * (1 - fill_i) * exp(xi_i)`.
- `fill_i` is the seller's own last delivery fill (2.7.3).
- `xi_i ~ N(0, channel_noise^2)` with key `(seed_env, "complaint", t, i)`, a new RNG purpose.
  At `channel_noise = 0` the draw is exactly 0.
- Under `aggregation_level = "sector"` it is replaced by its sector mean, as claims are.
- The quantity lies in `[0, shortfall_visibility * e^{3 sigma}]`. It is 0 when every claim was
  honoured.

## R4. Targeted audits (AMB-WO006-E)

- New field `InformationConfig.audit_target_gain` `kappa_t`: default 4.0, range [0, 10].
- With `audit_mode = "targeted"` and `shortfall_visibility > 0`:
  `p_i = clip(a * (1 + kappa_t * downstream_shortfall_i), 0, 1)`, drawn as before at key
  `(seed_env, "audit", t, i)`. The key is the episode seed (AMBIGUITY-019 B).
- With `shortfall_visibility = 0`, targeted audits reduce exactly to random audits.
- The shortfall used is the one from the most recent DELIVER.

## R5. Quality (AMB-WO006-F, WO-021)

- `qbar_i = quality_acc_i / M`: the period mean of the `quality` action over the `M` PRODUCE
  steps. This adopts the reference implementation's existing rule.
- `quality_acc` is reset at the start of each period.
- With `quality_matters = False`, `qbar_i = 1` as in Phase 1.
- Bundle routing (2.7.3): `X_bj += deliv_bj * qbar_j`, where `qbar_j` is the claim-weighted mean
  `qbar` of good `j`'s sellers in the delivering period.
- Consumer delivery: `consumer_j = sum_i phi_j shipped_i qbar_i`.
- Measured quality: `q_hat_i = 1 + mu (qbar_i - 1)`, already implemented.
- Cost: `kappa_q q e` per step, already implemented.

## R6. Delivery timing (WO-022)

- `delivery_timing = "uniform"`: all of a period's deliveries arrive at step 0 (Phase 1).
- `"stochastic"`: each buyer-good delivery `deliv_bj` arrives whole at step
  `k_bj ~ Categorical(arrival_probs)`, key `(seed_env, "arrival", t, b, j)`.
- `"backloaded"`: the same draw with `pi_k proportional to (k + 1)^2`, `k = 0..M-1`, ignoring
  `arrival_probs`.
- Undelivered quantities wait in `State.pending_deliv` `(N, J, M)` and are credited to `X` at the
  start of step `k`, before PRODUCE.
- Observation fields 11 and `12+2J:12+3J` report deliveries **received so far** this period.
- Allocation, fill and `poolfill` are computed at DELIVER as before: timing moves arrival, not
  quantity.

## R7. Input holding loss (WO-023)

At the REPORT step, alongside `S <- (1-h) S + y`: `X_ij <- (1 - h_X) X_ij`. The loss is logged in
the ledger.

## R8. Soft budget (WO-023)

- At AUDIT, an enterprise with `last_fill_i < 1` (it failed to ship its own last claim) is
  bailed out with probability `soft_budget`, key `(seed_env, "bailout", t, i)`, a new purpose.
- A bailout sets `penalty_i = 0` for that period and is logged (`bailed_out`).
- Nothing else changes, which is Kornai's soft budget: the loss is forgiven, not prevented.

## R9. Trade (2.13, WO-024)

The trade stage runs at step 0, after DELIVER, and uses the `trade_offer` action `o_ij` in
`[-1, 1]`.

1. **Visibility.** Enterprise `i` sees `K = round(horizontal_visibility * (N - 1))` others: the
   first `K` of a permutation of the other indices drawn at key `(seed_env, "trade_visibility", t,
   i)`. A pair `(i, b)` may trade if `b` is visible to `i` or `i` to `b`.
2. **Offers.** Supply is `sup_ij = max(0, o_ij) X_ij`. Demand is
   `dem_bj = max(0, -o_bj) need_bj`, where `need_bj = a_{s(b) j} T_b` (true `a`).
3. **Matching**, per good `j` in index order. The pair is seller `i`, buyer `b`, with `i != b`.
   - Repeatedly take the eligible pair with the largest `x = min(sup_ij, dem_bj) > 0`, breaking
     ties by lowest `(i, b)`.
   - Execute it: `X_ij -= x`, `X_bj += (1 - tau) x`, `sup_ij -= x`, `dem_bj -= x`.
   - Stop when no pair has `x > 0`.
   - `tau x` is lost: the transaction cost.
4. **Price parity.** No money moves; each side values the good at the same plan price, so
   payments cancel.
5. **Surplus.** `Yhat_i(X) = (A_{s(i)} cap_i / M) * e_i^last * H_i(X)`: next step's intended
   output at the agent's last effort and coverage `H` (2.6).
   - `trade_surplus_i = [Yhat_i(X_after) - Yhat_i(X_before)] / T_i`, in ratio units, so it is
     commensurate with the bonus. It is computed by the environment, never self-reported.
   - The enterprise's own price cancels in the ratio-unit form, so no price term appears.
   - It accumulates in `State.trade_surplus_acc` and is paid in the REPORT reward (2.9.1). The
     accumulator is reset each period.
   - With `tau > 0`, a wash trade lowers both parties' `H` and is never profitable.
6. **Peer observation block** (2.4), present only when `horizontal_visibility > 0`:
   - Width `G - 1`, where `G` is the largest sector size.
   - Contents: the `last_report_ratio` of the other members of the agent's own sector, in
     enterprise-index order, zero-padded.
   - It is appended after index `12 + 3J`.

## R10. Ministry (2.14, WO-025)

1. **Partition.** `n_ministries = n_m` ministries; enterprise `i` belongs to ministry
   `floor(i * n_m / N)` (contiguous blocks).
2. **Forwarding.** Between REPORT and the planner, each ministry forwards
   `Rtilde_i = pi R_i + (1 - pi) [Rbar_i^prev + kappa_m max(0, T_i - R_i)]`.
   - `pi = ministry_passthrough`.
   - `kappa_m = InformationConfig.ministry_pad`: a new field, default 0.5, range [0, 1].
   - `Rbar_i^prev` is the ministry's own previous forward for `i`, kept in
     `State.ministry_prev (N,)`. It is initialised to `T_0`.
3. **What sees what.**
   - The planner's claims (R2, then aggregation and noise) and the ratchet use `Rtilde`.
   - The enterprise's bonus and audit use its own `R`.
   - Delivery obligations (2.7.3) use `Rtilde`: the ministry has promised that output.
4. **Transparency.** `pi = 1` is exactly transparent.
5. **`MinistryPolicy.forward`.** It receives one ministry's `MinistryView` and returns the
   forwarded vector. The rule above is the rule-based policy. The LLM adapter (WO-026)
   implements the same protocol.

## R11. P2 default configuration

`p2_default_config()` is `p1_default_config()` (the G1 values) plus the following:

**Locked §4.2 values:**
- `delivery_timing = "stochastic"`, `arrival_probs = (0.25,)*4`
- `alloc_eta_request = 0.7`, `input_complementarity = 8`, `input_holding_loss = 0.01`
- `horizontal_visibility = 1.0`, `trade_tau = 0.05`
- `g`, `penalty_arg` and `h` as in Phase 1

**The Phase-2 baseline C0's information and incentive settings.** The LEAD chose these, recorded
here before any Phase-2 run and set at registry mid-range. C_OGAS later moves them to their
transparent values.
- `report_lag = 1`, `aggregation_level = "enterprise"`, `channel_noise = 0.05`
- `ministry_passthrough = 0.75`, `n_ministries = 5`, `ministry_pad = 0.5`
- `audit_mode = "targeted"`, `shortfall_visibility = 0.5`, `audit_target_gain = 4.0`
- `quality_matters = True`, `quality_cost = 0.05`, `quality_measurability = 0.5`
- `soft_budget = 0.25`

## R12. Tests and records

- Each work order adds unit tests for its mechanism.
- Conservation tests extend T-B3: trade conserves `X` up to `tau`, and pending deliveries conserve
  quantity.
- Held-out phenomena 2, 5, 6 and 7 get no directional test before WO-031 (§4.1).
- The reference implementation stays the Phase-1 oracle. Phase-2 branches are checked by unit
  tests and by hand-worked two-enterprise cases in the test docstrings.
