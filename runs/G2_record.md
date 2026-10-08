# Gate G2 record - Phase 1

**Status: G2 NOT PASSED.** Criterion 1 was not met in three attempts. By the owner's decision
(AMBIGUITY-021), criteria 2-4 were run afterwards as a **separately labelled study**, not as the
gate. Phase 2 (WO-021 onwards) needs the owner's G2 decision on the record below.

Sources: `runs/dp_vs_ppo/report.md` (attempt 3), `runs/dp_vs_ppo/attempt1/`, `attempt2/`,
`runs/phase1_gate/report.md`, `workorders/AMBIGUITY-019.md` to `AMBIGUITY-022.md`.

## Criterion 1 - PPO recovers the single-enterprise DP (`N = 1`): NOT MET

| Attempt | Learner | Seeds passing (of 30) | Mean return at `a*pen` 0.8 / 4 / 20 | Main miss |
|---|---|---|---|---|
| 1 | discount 0.99 per agent-step, report-head std 0.05 | 0 | 4.84 / 5.37 / 3.23 | effort too high, W1 |
| 2 | discount 0.99 per plan period (AMBIGUITY-020) | 0 | 5.38 / 5.23 / 3.76 | effort ~0.81 vs 0.50, W1 ~0.15 |
| 3 | + wide report-head init (owner option A) | 0 | 3.17 / 2.69 / 2.46 | under-reports, effort ~0.24, W1 ~0.7 |
| DP policy replayed in the environment | - | - | 7.13 / 6.43 / 5.26 | - |

Two genuine defects were found and fixed on the way: the audit draw keyed by the configuration's
root seed rather than the episode's (AMBIGUITY-019 B), and the learner discounting per agent-step
where the spec's DP discounts per plan period (AMBIGUITY-020). The remaining gap
(AMBIGUITY-021) is one of exploration: the DP's optimum mixes about 12% deliberate zero reports,
which walk the target down through the ratchet, with truthful reports at the notch. A PPO learner
starting on the notch (std 0.05) never finds the zero reports. One starting wide (sigma_z = 1)
finds them but overshoots to 27-49% of reports. Neither reproduces the DP's policy within the
PLAN section 4.5 tolerances.

## Labelled study - criteria 2-4 (`N = 20`, attempt-2 learner, 80 runs)

**Criterion 2 - bunching present / absent at `a*pen` = 20: FAIL** (strict and lenient readings,
AMBIGUITY-022).
- Notched arm: 77% of measured reports lie in [1.00, 1.02] on average (range 0-100%; the
  threshold is 44%). 23 of 30 seeds clear the share threshold: 11 with a CI excluding 0, and
  12 where the CI is undefined because all their mass is inside the excluded window. That is 37%
  of seeds under the strict reading and 77% under the lenient one; 90% is needed.
- Smooth arm: 3.7% of reports in the window on average (max 13%), so there is no bunching. But
  the estimator's CI covers 0 in only 10 of 30 seeds (33%). The point estimates centre on zero
  (median `b_hat` 0.08) with narrow per-seed intervals of both signs. This fits the residual
  estimator bias on peaked distributions recorded in AMBIGUITY-011 (0.147 at sigma = 0.10).
- Reading: the notch produces a large, qualitatively unmistakable spike that the smooth schedule
  does not (77% vs 3.7% of reports at the notch). The pre-registered per-seed test does not
  certify it, in both arms, for reasons tied to the estimator: no counterfactual support in the
  notched arm, and tight biased intervals in the smooth arm.

**Criterion 3 - padding elasticity: FAIL.** Padding is monotone decreasing in `a*pen` (0.1083 /
0.0019 / 0.0001). It is within 0.0004 of the DP at `a*pen` = 4 and 20, but 0.103 above it at
0.8, where the learner pads and the DP does not (the same gap as criterion 1 at that level).

**Criterion 4 - hygiene: FAIL.** `BOUND_BINDING` is raised in 0 of 80 runs. Target runaways
(`T > 3 T_0` after the first 20% of training) occur in at most 0.1% of training episodes at the
Phase-1 configuration and 4.7% at `a*pen` = 4, but in up to 10.9% at `a*pen` = 0.8 (the limit is
5%).

**Price sensitivity:** deferred to WO-036 (AMBIGUITY-019 D).

## What the owner decides at G2

The PLAN (section 4.5) makes criterion 1 blocking. The evidence is that PPO does not recover the
DP's mixed under-reporting strategy under any of three learner settings, while at `N = 20` it
does produce heavy bunching at the notch and none under the smooth schedule. The options are:
- (a) Accept Phase 1 as a labelled result and proceed to Phase 2 with this learner, carrying the
  criterion-1 gap as a stated limitation.
- (b) Commission further training-stack work first. Candidates: a longer budget, entropy
  settings, or a learner that can represent mixed strategies. Each would be a new labelled study.
- (c) Revisit the pre-registered criterion-2 estimator for degenerate and peaked distributions,
  as a spec revision with its own record.
- (d) Stop at Phase 1.

## OWNER DECISION (2026-09-26): option (a)

The owner instructed the LEAD to continue building. Recorded as option (a): Phase 1 stands as a
labelled result. The criterion-1 gap (PPO does not recover the DP's mixed under-reporting
strategy) and the criterion-2 estimator behaviour are carried into Phase 2 as stated limitations,
and every Phase-2 report cites them. Phase 2 proceeds: the P2 spec revision (LEAD), then WO-021 to
WO-031, then G3. The attempt-2 learner (`phase1_gate.study_ppo_config()`) remains the working
learner.
