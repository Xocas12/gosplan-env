# Labelled study G3b - record (LEAD)

**Status: G3b does not change G3.** G3 stays **NOT PASSED** as recorded in `runs/G3_record.md`.
G3b was approved at G3 (owner decision 2, delegated) to answer two questions. Row 6 trades
nothing, but under the trade rule (R9) a sale never pays the seller, so row 6 is not evaluable
(corrected on 2026-10-14; see the correction at the end). The revised audit finds the populations
far from equilibrium. Written on 2026-09-28. The human sign-off is pending.

- Source: `runs/phase2_acceptance_g3b/report.md` and `result.json`.
- Spec 2.1.0: R15 (trade offers posted at REPORT) and R16 (the revised exploitability audit).
- Design: C0 only, 30 seeds, the same `seed_env = 1000 + s` and learner as G3. The R16 audit ran on
  seeds 0-9.
- The run started on 2026-09-28 at 07:51 UTC and ended at 21:08 UTC.
- Limitations L1 and L2 carry forward. L3 (no trade was possible) is lifted for G3b only.

## Question 1 - row 6 (blat), after the D1 fix: **NOT EVALUABLE under R9** (nothing trades)

| Statistic | Value |
|---|---|
| trade volume share, mean over 30 seeds [95% CI] | 0 [0, 0] |
| matched trade pairs, all 30 seeds, 100 measurement episodes each | 0 |

The zero is not a pipeline defect, and it is not evidence of learned hoarding either. It was
checked three ways:

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
3. **The trade rule predicts it.** Under R9 a sale gives the seller nothing back, so selling
   never raises the seller's own reward (see the correction at the end). A learner that never
   offers to sell is doing what the rule rewards, whatever it has learned about barter.

Row 6 is therefore **not evaluable under R9**. Applied mechanically, R14's rule gives "does not
appear" (the lower CI bound is not above 0), and the study report prints that verdict. But no
individually rewarded learner gains from selling under R9, so a zero is what the rule predicts
whatever the learners do. It cannot tell enterprises that hoard rather than barter from a market
that never pays a seller. Horizontal trade does not occur here; whether it would emerge under a
rule that pays the seller is untested.

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
  - Row 6 is not evaluable under R9: learners never offer to sell, and under R9 a sale never
    pays the seller (see the correction below).
  - Under the revised audit the C0 populations are non-converged (L4).
- The Phase-3 contrasts (`runs/P3_record.md`) were run before R15, carrying L3. G3b shows that
  even with trade possible, C0 populations do not trade. L3 is therefore unlikely to have changed
  the contrasts' C0 leg. This is an inference, not a re-run: the contrast arms were not re-trained
  under spec 2.1.0.
- The owner decides at G3/G4 whether a longer-trained or differently configured learner is
  commissioned. That would be a new labelled study, because the numbers here must not be tuned to
  pass.

## Correction (2026-10-14): row 6 is not evaluable under R9 (#107)

Question 1 first recorded row 6 as "FAILURE (does not appear)" and read it as "under C0 the learned
enterprises hoard rather than barter". Its check 3 read the buy-only offers as agreeing with row 5,
and the consequences said "Row 6 fails behaviourally, because learners hoard and never offer to
sell". Those readings claim more than the zero shows, and they are corrected above. No number
changes.

- **A sale gives the seller nothing back.** R9.3 executes `X_ij -= x` for the seller and
  `X_bj += (1 - tau) x` for the buyer. No good, money or credit returns to the seller, so R9.4's
  "no money moves; ... payments cancel" leaves no payment to cancel.
- **The seller's own reward cannot rise.** The only trade term in the reward is each enterprise's
  own `trade_surplus` (R9.5): the change in its own intended output through coverage. Coverage
  never falls as a stock grows.
  - A sale is never worth more than 0 to the seller. It is worth exactly 0 when the seller's stock
    of the sold good still covers its need after the sale.
  - It is worth less than 0 when the sale takes the stock below need, unless another input already
    holds the seller's output down: one it holds none of, or, with a Leontief aggregator
    (`theta = inf`), one whose coverage is no higher than the sold good's after the sale. Then it
    is worth exactly 0.
  - A buy never lowers the buyer's surplus.
  - The gain from moving an input to where coverage binds goes wholly to the buyer.
- **Nothing else pays the seller.** The planner never sees input stocks, cost depends on effort
  only, and PPO advantages are computed per agent, so parameter sharing does not pass the buyer's
  gain back. An offer is one number per good, so a seller cannot also buy that good. Matching is
  anonymous and no observation shows who traded with whom, so reciprocity cannot be learned
  either.
- **Checked** in `tests/unit/test_trade_seller_incentive.py`:
  - hand-worked cases: a sale that leaves the seller's stock above its need gains it exactly 0; a
    sale that takes the stock below need loses; and a sale below need gains exactly 0 when the
    seller holds none of another needed input, or, under `theta = inf`, when another needed
    input's coverage is lower than the sold good's after the sale;
  - 300 random states of `p2_default_config()`: wherever the sale executed (70 cases), the
    seller's own surplus never rose, and a buy never lowered the buyer's.

Selling is weakly dominated for every individually rewarded learner, so the buy-only, zero-volume
outcome is built into the trade rule. Row 6 shows that no one sold. It cannot show that the C0
institution suppresses barter, and so cannot bear on Claim A(iii). G3 stays NOT PASSED, and
question 2 is unaffected.

Evaluating row 6 needs a trade rule in which a sale pays the seller, for example a swap leg in which
the buyer ships back goods of equal plan value. That would be a spec revision and a new labelled
study, and it is the owner's decision. The correction is carried into `runs/LC_record.md`,
`runs/CT_record.md`, `runs/P3_record.md` and `README.md`.
