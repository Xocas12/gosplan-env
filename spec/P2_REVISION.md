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

## R13. Metrics and heuristic agents (WO-030) - the points WO-030 says the revision must state

Signatures stay as in `spec/spec.py`. No `StepRecord` column is added. Every statistic uses the
PLAN section 4.4 window (`t_period >= 2`).

1. **Gini (row 2).** Per (episode, enterprise, period), with `e_k` the effort on the `M` PRODUCE
   rows: `G = sum_{k,l} |e_k - e_l| / (2 M^2 mean(e))`.
   - A period with zero total effort has no defined Gini. It is excluded and counted in
     `n_zero_effort_periods`.
   - `gini` is the mean `G` over the included periods. `gini_baseline` is the same for the baseline
     ledger, and `excess = gini - gini_baseline`.
2. **Quality (row 3).** One call per ledger.
   - `mean_quality` = mean `qbar = quality_acc / M` over REPORT rows.
   - `mean_quality_weighted` = mean measured quality `1 + mu (qbar - 1)`.
   - WO-031 contrasts `mean_quality` between the `val` run and the `quality_weighted` run at
     `mu = 1`.
3. **Hoarding (row 5).**
   - `request_inflation`: the mean of `request_ij / need_ij` over REPORT rows and goods with
     `need_ij > 0`. `request` and `need` are in units.
   - `corr_stock_shortfall`: the Pearson correlation, over (episode, period, i, j) with
     `need_ij > 0`, of two quantities:
     - `X_ij` on the REPORT row, and
     - `1 - fillbar_j`, where `fillbar_j` is the shipped-weighted mean `fill` of good j's sellers
       on the next period's DELIVER row. Unweighted if nothing shipped. Pairs with no next period
       are dropped.
     It is NaN if either side is constant.
   - Each is also reported for the baseline ledger.
   - `dispersion_stat` and `dispersion_p`: `cross_section` on the per-(i, j) mean inflation, with
     the sector `s(i)` as groups.
4. **Blat (row 6).**
   - `trade_volume` is the quantity enterprise i sold in the period's trade stage, on the step-0
     row (WO-024 ledger convention).
   - `trade_volume_share` = sum of `trade_volume` / sum of `alloc` over window rows. It is 0 when
     `alloc` sums to 0 and trade is 0.
   - Pairs are not in the ledger. The key `n_matched_pairs` therefore counts selling
     enterprise-periods, and is documented as such.
   - `mean_surplus` is not recoverable from the ledger and is returned as NaN. The surplus enters
     the reward and is visible through it only.
5. **Hidden reserves (row 7).**
   - `hidden_reserves` = mean over REPORT rows of `max(0, inv_output_post - report) / target`.
   - The reconciliation part calls
     `ledger_test(report, deliv_next, io_rows, price_rows)`, over REPORT rows that have a next
     period. Here `io_rows` are the per-enterprise rows `a_{s(i), :}` `(n, J)`, and `price_rows`
     is `p_{s(i)}` `(n,)`.
   - This is how R13 reads the frozen argument names. `io_matrix` may be given as per-enterprise
     rows, because a `(J, J)` matrix alone cannot map an enterprise to its good.
6. **Fallback `ledger_test`.**
   - `implied_i = min over j with a_ij > 0 of received_ij / a_ij`: the output that the received
     inputs support under Leontief. It is `+inf`, and the row is dropped, when the row of `a` is
     zero.
   - `residual_i = price_i (reported_i - implied_i)`.
   - `statistic` = the t-statistic of the mean residual. `p_value` = one-sided, from the normal
     distribution: the claims exceed what receipts support.
   - Degenerate cases:
     - Zero variance with a zero mean gives `statistic = 0` and `p = 0.5`.
     - Zero variance with a non-zero mean gives `statistic = ±inf` and `p = 0` or `1`.
     - `n_obs < 2` gives NaN.
7. **Fallback `cross_section`.**
   - The Kruskal-Wallis H test of equal distributions across groups (`scipy.stats.kruskal`).
   - `group_stats` are the group means, in order of first appearance.
   - With fewer than 2 groups, or all values identical, `statistic = 0` and `p = 1`.
8. **Quality action of the fixed heuristics.** With `quality_matters = True`, every heuristic in
   `gosplan/agents/heuristic.py` except `Random` plays `quality = 1`, the non-degrading level. The
   truthful reference line must contain no slack. With `quality_matters = False` the action stays
   0 as in Phase 1, so the goldens are unchanged.
9. **Heuristic constants** (baselines, never evidence; PLAN section 6.1). `e_TM` below is
   `TruthfulMyopic`'s effort, `clip(initial_target_frac * exp(obs[:,2]), 0, 1)`.
   - `Berliner(safety_factor)`, with `safety_factor` in [0.05, 0.10] and no default:
     - effort `clip((1 + sf) * e_TM_unclipped, 0, 1)`
     - `rho = 1` always
     - requests = need, no trade
   - `Weitzman`:
     - effort `e_TM * (1 - WEITZMAN_MAX_CUT * lambda / (1 + lambda))`, with
       `WEITZMAN_MAX_CUT = 0.2` and `lambda = ratchet_lambda`; `tenure` is not used
     - report truthful of stock, as `TruthfulMyopic`
     - requests = need, no trade
   - `Kornai(request_inflation)`, with no default:
     - effort and report as `TruthfulMyopic`
     - `input_request = request_inflation`, in multiples of need
     - no trade
     - Its bailout anticipation is that it keeps full effort when inputs are short and keeps
       inflating whatever `soft_budget` is. `TruthfulMyopic` also keeps effort, so the two differ
       only in requests. This is stated, not hidden.

## R14. Phase-2 acceptance run (WO-031, gate G3) - pre-registration

Written before any Phase-2 learning run and before any rows 2, 5, 6 or 7 number exists. It settles
the points WO-031 says the revision must state. It also records a reduced design:
- PLAN section 14 budgets "about 8 configs x 30 seeds" and assumes the JAX port for training.
- Phase-1 runs took about 4 min each on this 4-core container (80 runs, about 5.5 h), and the
  Phase-2 environment is slower.
- So the acceptance set is the three configurations G3 actually needs. The other PLAN section 4.3
  contrasts are Phase-3 work (WO-032).

**Learner and sizing.**
- Learner: the attempt-2 learner (`phase1_gate.study_ppo_config()`), shared parameters.
- Sizing: `phase1_gate.GATE_SIZING` (1M agent-steps per run, 100 measurement episodes), and the
  Phase-1 training harness (WO-018).
- Seeds: `seed_env = 1000 + s`, shared across arms and with the baseline legs, which gives common
  random numbers (PLAN section 4.3).

**Arms.** No LLM ministry arm; PLAN section 13 puts the LLM study under G4.

| arm | configuration | seeds | used for |
|---|---|---|---|
| `C0` | `p2_default_config()` (R11) | 30 | rows 2, 5, 6, 7; oracle; welfare |
| `R7_NULL` | C0 with `growth_directive = 0`, `penalty_arg = "absolute"` | 10 | row 7 falsification |
| `R3_QW` | C0 with `objective_metric = "quality_weighted"`, `quality_measurability = 1` | 10 | row 3 |

**Baseline legs.** Every seed of every arm also gets a `TruthfulMyopic` measurement leg: the same
configuration, the same `seed_env`, and the same number of measurement episodes. It needs no
training.

**Pass rules**, fixed now. Each held-out row is a mean over seeds of the per-seed statistic, and its
95% interval is the percentile seed bootstrap (10,000 resamples, generator seed 0).
- **Row 2 (storming): APPEARS** if the lower CI bound of `excess` is > 0.
- **Row 5 (hoarding): APPEARS** if both of these hold:
  - the lower CI bound of `request_inflation - request_inflation_baseline` is > 0;
  - the lower CI bound of `corr_stock_shortfall - corr_stock_shortfall_baseline` is > 0, with a
    NaN baseline correlation counted as 0 and a seed whose own correlation is NaN dropped and
    counted.
- **Row 6 (blat): APPEARS** if the lower CI bound of `trade_volume_share` is > 0.
  `TruthfulMyopic` never trades.
- **Row 7 (hidden reserves): APPEARS** if both of these hold:
  - on `C0`, the lower CI bound of `hidden_reserves` is > 0;
  - on `R7_NULL`, it VANISHES: the upper CI bound is < 0.01 (1% of target).
  If the first holds and the second does not, row 7 is a reported failure of the falsification
  condition.
- **Row 3 (pipeline check):** the upper CI bound of `mean_quality(C0) - mean_quality(R3_QW)` is < 0.
  The difference is between two unpaired seed sets.
- A row that does not appear is a **reported failure** (PLAN section 4.2). Nothing is re-run and no
  value is changed.

**Exploitability** (WO-028):
- Audited on seeds 0-9 of `C0` and seeds 0-4 of each other arm.
- Best-responder sizing: `GATE_SIZING` with `total_agent_steps = 500_000`.
- Threshold 5%, provisional. The owner finalises it at G3, and an arm above it is labelled
  NON-CONVERGED.

**Oracle** (WO-027):
- `W_oracle` = `solve_oracle(C0-family config, horizon = 40, clairvoyant = False)`, with the solver,
  its version and its gap in the report.
- `welfare_ratio = W / W_oracle`, where `W` is the mean window welfare of the measurement episodes.
- The clairvoyant bound is computed per seed for seeds 0-4 and labelled "upper bound".

**Price sensitivity:**
- `specification_gap` and `welfare_ratio` are recomputed under `perturbed_price_vectors(p, (11, 12,
  13))`.
- Any sign change of `specification_gap` is reported.

**JAX parity** (WO-029):
- 100 agent-steps of `TruthfulMyopic` on `p1_default_config()` and `p2_default_config()`.
- The maximum absolute deviation over the ledger's numeric columns is reported against 1e-5.

**Hygiene:**
- `BOUND_BINDING` is counted per arm.
- Target runaways use the Phase-1 definition.

**C0 post-G3 values.** R11 stands as the Phase-3 C0 unless the owner's G3 decision changes it.
