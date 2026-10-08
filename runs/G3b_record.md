# Labelled study G3b - record (LEAD)

**Status: G3b does not change G3.** G3 stays **NOT PASSED** as recorded in `runs/G3_record.md`.
G3b was approved at G3 (owner decision 2, delegated) to answer two questions. Row 6 now fails on
learned behaviour, and the revised audit finds the populations far from equilibrium. Written on
2026-09-28. The human sign-off is pending.

- Source: `runs/phase2_acceptance_g3b/report.md` and `result.json`.
- Spec 2.1.0: R15 (trade offers posted at REPORT) and R16 (the revised exploitability audit).
- Design: C0 only, 30 seeds, the same `seed_env = 1000 + s` and learner as G3. The R16 audit ran on
  seeds 0-9.
- The run started on 2026-09-28 at 07:51 UTC and ended at 21:08 UTC.
- Limitations L1 and L2 carry forward. L3 (no trade was possible) is lifted for G3b only.

## Question 1 - row 6 (blat), after the D1 fix: **FAILURE (does not appear)**

| Statistic | Value |
|---|---|
| trade volume share, mean over 30 seeds [95% CI] | 0 [0, 0] |
| matched trade pairs, all 30 seeds, 100 measurement episodes each | 0 |

This is a behavioural result, not a pipeline defect. It was checked three ways:

1. **The fix works.** The pre-run smoke test (tiny untrained policies under spec 2.1.0) matched
   166 and 335 trade pairs. Offers are posted at REPORT, stored, and executed by the next period's
   trade stage.
2. **The learned policies post only buy orders.** A deterministic rollout of the trained C0
   populations for seeds 0-9 recorded every posted `trade_offer` at REPORT.
   - Every offer was negative, meaning a want to buy (`gosplan/env/trade.py`: a negative offer is
     a want, a positive one an offer to sell).
   - Offers ranged from -1.000 to -0.358, with per-seed means between -0.90 and -0.99.
   - Not one of the offers was a sell offer. With no seller, the matching rule (R9) can execute
     nothing.
3. **This agrees with row 5.** The same populations push input requests to the upper bound
   (request inflation excess 1.96, as at G3). Every enterprise wants more of every input and none
   parts with any.

Row 6 is therefore evaluated, and it fails under R14's rule: the lower CI bound is not above 0.
In the terms of PLAN section 4.2: under C0 the learned enterprises hoard rather than barter, so
horizontal trade does not emerge.

## Question 2 - the R16 exploitability audit: **NON-CONVERGED**

| Seeds audited | Median ratio | Max ratio | Seeds above the 5% threshold |
|---|---|---|---|
| 10 | 0.797 | 0.990 | 9 of 10 |

- R16's design: the best responder starts from the population policy, trains for 1M steps, and the
  ratio is `(R_BR - R_pop) / max(|R_pop|, 1)`.
- Sorted per-seed ratios: -1.244, 0.125, 0.341, 0.364, 0.694, 0.899, 0.929, 0.972, 0.983, 0.990.
- The one negative seed (1007) is one where the population's own return is about -30 and the best
  responder fell further, to about -68.
- **Interpretation.** Unlike the pre-R16 audit (G3 D2), this audit is informative. Starting from
  the population's own policy, a single enterprise can improve its return substantially on 9 of
  10 seeds. The learned C0 populations are **not** approximate equilibria at the 5% threshold.
- Every Phase-2 and Phase-3 number from these populations therefore describes the learner's
  *outcome*, not an equilibrium of the institution. This strengthens L1 and is carried forward as
  **L4**.

## Supplementary (not re-evaluations; the G3 values stand)

- **Row 2:** excess storming Gini 0.118 [0.086, 0.150] (G3: 0.137).
- **Row 5:** request inflation excess 1.96, and corr(X, shortfall) excess 0.52 (G3: 1.96 and 0.55).
- **Row 7:** C0 hidden reserves 0.173 [0.068, 0.316] (G3: 0.222).
- **Welfare:** welfare_ratio is 0.0335 (G3: 0.034). specification_gap is 0.0124, with no sign
  change across price seeds 11, 12 and 13.
- **Not evaluated in G3b:** row 7's null arm and row 3 need the R7_NULL and R3_QW arms, which
  G3b does not run. The report's generic template prints "FAILURE" beside `nan` for these rows;
  read that as "not evaluated here".
- **Checks:** JAX parity passes (7.1e-14). Hygiene passes (runaway 0.0000, `BOUND_BINDING` 0). The
  oracle gap is 0.

## Consequences

- G3 remains NOT PASSED. G3b adds two findings:
  - Row 6 fails behaviourally, because learners hoard and never offer to sell.
  - Under the revised audit the C0 populations are non-converged (L4).
- The Phase-3 contrasts (`runs/P3_record.md`) were run before R15, carrying L3. G3b shows that
  even with trade possible, C0 populations do not trade. L3 is therefore unlikely to have changed
  the contrasts' C0 leg. This is an inference, not a re-run: the contrast arms were not re-trained
  under spec 2.1.0.
- The owner decides at G3/G4 whether a longer-trained or differently configured learner is
  commissioned. That would be a new labelled study, because the numbers here must not be tuned to
  pass.
