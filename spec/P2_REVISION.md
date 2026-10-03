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

## R15. Trade offers are posted at REPORT (G3 diagnosis D1) - written, not yet applied

**Applied on 2026-09-28** (commit 51b001a, spec 2.1.0), after the in-flight Phase-3 runs had
finished. The text below is unchanged from the version written before it was applied.

This fixes the defect found at G3 (`runs/G3_record.md` D1). As implemented, the trade stage (R9)
reads the step-0 PRODUCE action's `trade_offer`. The PPO adapter (WO-017, PLAN section 2.3) emits
`trade_offer` only at REPORT, so learners could never trade.

**Rule.**
- At REPORT the environment stores the action's `trade_offer` in a new state field
  `State.trade_offer_posted (N, J)`, initialised to 0.
- The trade stage at step 0 of period `t + 1` executes those posted offers.
- At `t = 0` nothing has been posted, so there is no trade.
- Everything else in R9 is unchanged: visibility, matching, tau, surplus and the peer block.

**Application.**
- The rule is applied only after the Phase-3 runs launched from `3f40b6e` onwards have finished.
- It bumps `SPEC_VERSION`, so every configuration hash changes and no run under the old dynamics
  can be read back as if it were new.
- The golden parity test excludes the new field, as it does the other Phase-2 fields.
- The JAX port mirrors the rule, and the parity tests are re-run.

**Re-evaluation.** Row 6 is re-evaluated in a labelled study, **G3b**: C0 at 30 seeds, R14's pass
rule for row 6, run only if the owner approves. Rows 2, 5 and 7 are not re-evaluated.

## R16. Exploitability audit, revised (G3 diagnosis D2) - pre-registered before G3b

At G3 the best responder was trained from scratch for 500k steps. It never reached the population's
return, so the audit measured the best responder's weakness, not the population's exploitability.

**Rule.**
- **Warm start.** The best responder in seat 0 starts from the population's own checkpoint, then
  trains for the full gate budget (1M agent-steps, `GATE_SIZING`) against the frozen population in
  the other seats. Its starting point is at least the population's return, up to evaluation noise.
- **Ratio floor.**
  `exploitability = (R_BR - R_pop) / max(|R_pop|, 1.0)`.
  The floor is 1 reward unit, which is one notch bonus at `rho = 1.1` after `reward_scale`. It
  keeps the ratio defined when the population's return is near 0.
- **Threshold.** 5% (PLAN section 6.3), applied to the maximum over audited seeds. An arm above it
  is NON-CONVERGED.
- **Seeds.** G3b C0, seeds 0-9.

**Application.** Implemented with R15, and applied only after the in-flight Phase-3 runs finish.
Those runs' own exploitability numbers come from the old audit and are labelled as such.

## R17. Learner-convergence study LC (labelled) - pre-registered before any LC run

Written on 2026-09-29, after G3b (`runs/G3b_record.md`) and before any LC training.

**Question.** G3b found the C0 populations non-converged under R16 (median exploitability 0.80;
limitation L4). Does tripling the population's training budget lower that exploitability?

**Design.**
- C0 under spec 2.1.0, seeds 0-9 (`seed_env = 1000 + s`, the G3b seeds), learner
  `phase1_gate.study_ppo_config()`.
- Gate sizing with `total_agent_steps = 3,000,000`; every other sizing field, and every
  configuration field, is unchanged.
- Training, measurement and the R16 audit are `phase2_acceptance.run`, unchanged. The best
  responder keeps R16's 1M-step budget, so the population budget is the only thing that varies.
- The 1M leg is G3b's seeds 0-9, read back, not re-run.
- Artefacts: `runs/learner_convergence/`. Driver: `gosplan/experiments/learner_convergence.py`.

**Primary analysis.**
- Seed-paired difference `d_s = expl_3M(s) - expl_1M(s)`.
- Its median, with a 95% percentile bootstrap CI over seeds (10,000 resamples, generator seed 0).
- "Exploitability falls with budget" iff the CI's upper bound is below 0.
- "Converged at 3M" iff the maximum over the 10 seeds is at most R16's 5% threshold.

**Secondary, descriptive only (no test).**
- Welfare ratio.
- Row-6 trade volume share.
- The sell and buy shares of offers posted at REPORT, from a deterministic rollout of both
  budgets' populations (5 episodes per seed on the measurement seed block).

**What LC cannot do.**
- It re-evaluates no gate, and G3 and G3b stand as recorded.
- Its outcome is reported whichever way it falls.
- No further budget, learner or configuration change follows from it without a new
  pre-registration.

## R18. Coordination-trap test CT (labelled) - pre-registered before the CT evaluation

Written on 2026-09-30, after LC (`runs/LC_record.md`). LC read its 3M collapse, without testing it,
as a no-production coordination trap. CT tests that reading directly. It is **evaluation only**:
no training, and no population is changed.

**Disclosure.** Before this text was written, a 2-episode smoke run of the CT code was executed on
the 3M population for seed 0 (`seed_env` 1000), to check that it runs. It gave seat-0 returns of
-0.003 (learned) and -1.350 (producing alone), and mean per-seat returns of -0.005 (learned) and
1.265 (all producing). The rules below were not chosen with reference to those values. They are
the textbook definition of a coordination trap.

**Design.**
- Primary populations: LC's 3M C0 populations, seeds 0-9.
- Contrast: G3b's 1M C0 populations, seeds 0-9, under the same rules.
- Producer: `TruthfulMyopic`, the project's reference line. It sets effort to meet the target,
  reports stock truthfully, requests at need and never trades.
- Each population is evaluated under four conditions, with 50 episodes each, on the measurement
  seed block and in the measurement window of the exploitability audit. Policies are
  deterministic, and random numbers are common across conditions.
  1. All seats learned.
  2. Seat 0 producer, the others learned.
  3. All seats producers.
  4. Seat 0 learned, the others producers.
- Driver: `gosplan/experiments/coordination_trap.py`. Artefacts: `runs/coordination_trap/`.

**Tests.**
- (i) Producing alone does not pay: `d1 = R_dev - R_pop`, seat 0's return in condition 2 minus
  condition 1. It holds iff the 95% CI of the median of `d1` over seeds lies entirely below 0.
- (ii) Everyone producing pays more: `d2 = W_tm - W_pop`, the mean per-seat return in condition 3
  minus condition 1. It holds iff the 95% CI of the median of `d2` lies entirely above 0.
- "Coordination trap" iff both (i) and (ii) hold.
- CIs are percentile bootstrap over seeds, 10,000 resamples, generator seed 0.

**Secondary, descriptive only (no test).**
- The share of seats better off when all produce.
- Seat 0's learned effort in condition 4 against condition 1: does the learned policy produce when
  its inputs exist?

**What CT cannot do.**
- It re-evaluates no gate.
- It uses one producer policy. A trap against `TruthfulMyopic` does not show that no producing
  strategy pays.
- Its outcome is reported whichever way it falls.

## R19. Budget versus exploration schedule, study ES (labelled) - pre-registered before any ES run

Written on 2026-10-03, after the LC trajectory addendum (`runs/LC_record.md`). LC's budget
manipulation also stretched the entropy anneal, which runs over the whole run. At the 1M-step mark
a 3M run's entropy coefficient was 0.0070, against 0.0010 for a 1M run. ES separates the two.

**Design.**
- C0 under spec 2.1.0, seeds 0-9, the study learner, 3M agent-steps.
- The entropy bonus is annealed 0.01 -> 0.001 over the first 1,000 updates (1M agent-steps,
  exactly the 1M runs' schedule) and held at 0.001 afterwards. This is new `PPOConfig` field
  `entropy_anneal_updates = 1000`.
- Everything else is LC's, including the R16 audit. Its best responder keeps the unchanged study
  learner.
- Driver: `gosplan/experiments/entropy_schedule.py`. Artefacts: `runs/entropy_schedule/`.

**Supporting code change.** Checkpoint matching now ignores schedule-only `PPOConfig` fields
(`SCHEDULE_ONLY_PPO_FIELDS`).
- Pre-R19 checkpoints still load.
- An ES population can be warm-started by the R16 best responder.
- Every other `PPOConfig` field still has to match exactly.
- Stored run records omit the new field when it is unset, so earlier runs are still read back
  rather than retrained.

**Primary.**
- Statistic: the median over seeds of the final measured mean effort, with a 95% percentile
  bootstrap CI over seeds (10,000 resamples, generator seed 0).
- "The collapse is reproduced under the matched schedule" iff the CI's upper bound is below 0.05.
  For reference, LC's 3M median was 0.019, and the 1M runs' measured efforts were 0.09-0.2.
- A reproduced collapse means that more training, not the stretched schedule, drives it.
- A non-reproduced collapse means the stretched schedule is implicated.

**Secondary, descriptive only.**
- Welfare ratio.
- R16 exploitability against LC's 3M populations.
- The R18 coordination-trap test, with its rules unchanged, on the ES populations.
- The training trajectories of all three schedules side by side.

**What ES cannot do.**
- It re-evaluates no gate.
- One alternative schedule does not exhaust the learner's design space.
- Its outcome is reported whichever way it falls.
- The ES analysis code was smoke-tested on LC's existing data before this text was written. That
  run read LC's populations as a stand-in, and no ES population existed.

**R19 note (2026-10-03, infrastructure, no design change).** Two container reboots killed the
first ES batch at about 1,450 and 360 of 3,000 updates. The harness saved checkpoints only at the
end of a run, so each reboot restarted those runs from scratch.

The training harness now writes an exact resume snapshot every 50 updates
(`TrainConfig.resume_every_updates`, `_g2.RESUME_EVERY_UPDATES`). The snapshot holds the policy and
optimiser state, the generator state, the pickled environments and the loop counters.
`tests/unit/test_train_resume.py` shows that a killed and resumed run equals an uninterrupted one
bit for bit, in its final parameters and in every logged training row. No ES run had completed when
this was added. The design, the rules and every result are unaffected.

