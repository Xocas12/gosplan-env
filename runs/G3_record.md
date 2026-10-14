# Gate G3 record - Phase 2

**Status: G3 NOT PASSED.** The LEAD signed this on 2026-09-27. The human sign-off is pending
(PLAN section 13: "Human + LEAD").

- Source: `runs/phase2_acceptance/report.md`, from the pre-registered design `spec/P2_REVISION.md`
  R14.
- The run started on 2026-09-26 at 10:05 UTC and ended on 2026-09-27 at 01:37 UTC, git `51132cd`.
- Arms: C0 (30 seeds), R7_NULL (10) and R3_QW (10), each trained at gate sizing with the attempt-2
  learner.
- Limitations carried forward: L1 and L2 (`runs/G2_record.md`).

## Conditions, as pre-registered

| G3 condition | Result |
|---|---|
| Held-out rows 2, 5, 6, 7 evaluated on the section 4.2 values | done (per-row outcomes below) |
| Exploitability below threshold on all arms | **FAIL**, and see D2: the audit is inconclusive |
| Oracle gap recorded | PASS (HiGHS via OR-Tools 9.15, gap 0, horizon 40) |
| JAX parity | PASS (max deviation 1.1e-14 for P1, 7.1e-14 for P2; tolerance 1e-5) |
| Price sensitivity | computed. `specification_gap` 0.0105 keeps its sign under all three vectors |
| Hygiene | **FAIL**: runaway share 0.154 against a limit of 0.05, in R7_NULL and R3_QW. C0's is 0.0. `BOUND_BINDING` 0 |

## Held-out rows (C0; mean over seeds, 95% seed-bootstrap CI)

| Row | Statistic | Outcome |
|---|---|---|
| 2 storming | excess within-period effort Gini 0.137 [0.095, 0.184] | **appears** |
| 5 hoarding | request inflation excess 1.96 [1.94, 1.98]; corr(X, shortfall) excess 0.55 [0.38, 0.72] | **appears** by R14's rule; not evidence of hoarding that propagates shortage (note H1) |
| 6 blat | trade volume share 0.0000 | **not evaluable**: defect D1 |
| 7 hidden reserves | C0 0.222 [0.092, 0.390]; R7_NULL 0.219 [0.055, 0.439] | **failure**: present, but does not vanish under the null |
| 3 quality (pipeline check) | mean qbar, C0 minus R3_QW: 0.152 [-0.074, 0.339] | **failure** |

**H1 (corrected 2026-10-14, #109).** Row 5 appears by R14's rule, and the numbers above are that
rule computed as pre-registered. They are not evidence of input hoarding that propagates shortage
(Claim A(ii)), and the records no longer cite them as such.
- **The inflation leg measures a free choice at its bound.** The learned requests sit at
  `request_max_multiple = 3.0` (per-seed mean inflation 2.81-3.00, mean 2.96). The magnitude is
  set by the action bound, and in the spirit of CONTRACT rule 8 it is reported that way, not as a
  free estimate. A request reaches the dynamics only through the allocation weight
  `(q + 1e-6)^0.7 * (need + 1e-6)^eta_n`, and no reward term charges for it, so raising one's own
  request raises one's own share at no direct cost. This leg is therefore closer to a pipeline
  check than to an emergence result. That is an analogy with bunching and padding, not the
  README's rule: those are direct optima of the reward, inflation pays through the allocation
  rule, and in the full game inflating could starve one's own suppliers, so dominance is not
  proven.
- **A uniform request multiple is allocatively neutral.** It cancels in the allocation's
  normalisation. With `allocate()` on the C0 planner view, everyone at 1x need and everyone at 3x
  need give allocations that agree to within the `1e-6` regularisers (relative difference below
  1e-8 at reset). A lone 1x requester among 3x requesters gets about half its share (0.062 instead
  of 0.125 of each good it needs), which is why inflating pays individually. At a common multiple
  near the bound it moves almost no inputs.
- **The correlation leg does not register request inflation.** `Kornai(3.0)` in every seat
  (`TruthfulMyopic` effort and reports, requests at 3x need) reproduces the `TruthfulMyopic`
  trajectory: input stocks differ by at most 2e-6, and the correlation excess is of order 1e-7,
  the effect of the allocation's regularisers. R14 sets no minimum effect size, so even this
  passes. On 30 C0 seeds at 10 measurement episodes each the excess is positive on every seed
  (7e-8 to 1.6e-7), and R14's rule marks that population APPEARS.
- **An agent that never inflates can pass the correlation leg.** `TruthfulMyopic` with its effort
  scaled by 0.8 and requests at need gives an excess of +0.41 to +0.54 on the same 30 seeds. Row 5
  does not appear for it only because its inflation excess is 0. Across the 29 C0 seeds with a
  defined correlation, the learned correlation has Spearman -0.68 with mean effort, and the
  collapsed 3M economies (effort about 0.02) give the largest excesses reported: 1.07 (LC) and
  1.03 (ES), against 0.55 here. The leg tracks effort and stock behaviour, not inflation.
- The note as first signed called the direction of row 5 a result. That reading is withdrawn. The
  G3b, CT, LC and ES records are corrected to match, and `tests/unit/test_row5_reading.py` pins
  the properties above on seeds 0 and 1.
- Showing propagation would need a different measurement under a labelled revision with its own
  pre-registration: for example, the share of each good's pool captured by over-requesting seats
  and the shortfall at seats that did not over-request, or a baseline matched on effort. That is
  the owner's decision.

**Welfare.**
- The learned populations reach `welfare_ratio = 0.034` (W 0.067 against `W_oracle` 1.98).
- The truthful-myopic baseline reaches W 1.06 (ratio about 0.53).
- The learned economy degrades quality: C0's mean qbar is 0.38. Deliveries are scaled by `qbar`,
  and production is near-Leontief (theta = 8), so the shortfall compounds into a collapse of
  consumer deliveries.
- This is a behavioural result of the pre-registered mechanisms, not a pipeline artefact. The
  oracle and the truthful baseline use the same measurement.

## LEAD diagnosis

**D1 - trade was structurally impossible (a pipeline defect).**
- The WO-017 PPO adapter treats `trade_offer` as a REPORT-phase action (`_REPORT_DIMS`, following
  PLAN section 2.3's action table).
- The WO-024 trade stage (spec/P2_REVISION.md R9) runs at step 0, after DELIVER, and reads the
  step-0 PRODUCE action.
- In that PRODUCE action the adapter masks `trade_offer` to 0, so no learner could ever post an
  offer. Row 6's zero therefore measures the plumbing, not the economy.
- No unit test caught it: the mechanism tests drive the environment with hand-built actions, and
  the P2 smoke used a heuristic agent that posts offers in every phase.
- **Proposed fix** (spec revision R15, not yet applied):
  - The trade stage uses the `trade_offer` posted at the most recent REPORT step, stored in state.
  - At `t = 0` there is no prior REPORT, so no trade happens.
  - This is consistent with PLAN section 2.3 and leaves the adapter untouched.
- The fix is held back until the Phase-3 runs now in flight finish. They were launched from the
  current code, and a code change under them would alter their evaluation dynamics and orphan the
  configuration-hash run directories they reuse.
- **Every Phase-2 and Phase-3 number from these runs describes an economy without horizontal
  trade.** This is carried forward as limitation **L3**.

**D2 - the exploitability audit is inconclusive, not passed.**
- In C0 and R3_QW every ratio is negative: the best-responder returns (-5 to -240) are far below
  the population's (about 0). The 500k-step best responder, trained from scratch, never learned to
  match the population, let alone exploit it.
- With `|R_pop|` near 0 the ratio is also ill-conditioned (C0's median is -130).
- "Below threshold" is therefore not evidence of convergence.
- R7_NULL's maximum of 0.70 (NON-CONVERGED) is the one case where a best responder exceeded the
  population.
- The threshold (provisional 5%) is the owner's to finalise at G3. The audit design (best-responder
  budget and initialisation, and a ratio floor on `|R_pop|`) needs a revision before the threshold
  means anything.

**D3.** Row 7's `reconciliation_stat` uses R13.5's per-enterprise form, which has no power
(spec/P3_REVISION.md S4 amendment). It is uninformative. Row 7's pass rule never used it.

**Findings that stand as reported failures** (PLAN section 4.2; nothing is re-run to make them
appear):
- Row 7's falsification condition fails: reserves persist under `g = 0` with the absolute penalty,
  where truthful reporting is optimal. This is consistent with L1, the learner not reaching the
  optimum.
- Row 3's pipeline check fails: quality does not rise under the quality-weighted objective.
- Hygiene fails in two arms.

## Owner decisions at G3

1. **Finalise the exploitability threshold.** In light of D2, the LEAD recommends first revising
   the audit design (best-responder warm start from the population policy, equal budget, and a
   floor on `|R_pop|`).
2. **D1.** The LEAD recommends applying R15 and running a labelled **G3b** study: C0 at 30 seeds
   (about 6 h), re-evaluating row 6 only. Rows 2, 5 and 7 stand as computed here, because their
   first evaluation is this one.
3. **Proceed.** The Phase-3 contrasts, already running under the current code, continue as a
   labelled study carrying L3. The LEAD recommends proceeding, because no contrast lever is a trade
   parameter.

## OWNER DECISION (2026-09-27): the LEAD's recommendations, delegated

The owner instructed: "Do what's recommended on your judgment". Recorded as:

1. **Exploitability.** The audit is redesigned first (spec/P2_REVISION.md R16), and the 5%
   threshold is then applied to the revised audit. The G3 audit above stands as inconclusive.
2. **D1.** R15 is applied after the in-flight Phase-3 runs finish. The labelled study **G3b** is
   approved: C0 at 30 seeds under spec 2.1.0, re-evaluating row 6. The R16 audit is run on G3b's
   populations.
3. **Phase 3** continues as a labelled study carrying L3 (no horizontal trade).

G3 stays **NOT PASSED** as recorded. G3b is a separate, labelled record and never rewrites this
one.
