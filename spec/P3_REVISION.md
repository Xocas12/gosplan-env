# Phase-3 revision - LEAD

Written on 2026-09-26, while the G3 acceptance run was still in its exploitability stage. No G3
number, and no rows 2, 5, 6 or 7 figure, had been read. This revision fixes the design of WO-032 to
WO-037 before any Phase-3 run. A gap found while implementing goes through an AMBIGUITY REPORT
(CONTRACT rule 3).

It resolves the points each Phase-3 card says "the revision must state". Where it reduces a PLAN
design, the reduction and its reason are stated here, and every report that uses it cites this
section.

**Compute.**
- A Phase-2 `N = 20` run at gate sizing (1M agent-steps) takes about 46 min on one core of this
  4-core container. A Phase-1 run takes about 16 min. The G3 run took about 11 h.
- PLAN section 14 assumed JAX training for Phase 3. WO-029 ported the step function, but there is
  no JAX trainer, so the PPO learner still steps the NumPy environment.
- The designs below are sized to fit in about one day of this container's compute.

Limitations carried into every Phase-3 report: L1 and L2 (runs/G2_record.md), and whatever the G3
record adds.

---

## S1. Post-G3 C0

`C0 = p2_default_config()` (spec/P2_REVISION.md R11), unchanged. R14 stated R11 stands unless the
owner's G3 decision changes it. If the owner later changes C0, the contrasts are re-run under that
change as a new labelled study.

## S2. Contrasts (WO-032, PLAN section 4.3)

**Arms.** `CONTRASTS` in `gosplan/experiments/contrasts.py`, verbatim from PLAN section 4.3.

**C_AUDIT range.** C_AUDIT's `audit_rate x4` gives 0.4, outside the PLAN section 3 sweep range
`[0.01, 0.30]`. This is reported, not clamped (WO-032).

**Seeds.**
- C0 reuses the 30 G3 runs. Its configuration and seeds are identical to G3's, so the runs share
  the same hashes and are read back rather than re-trained.
- Each other arm runs **15 seeds**: indices 0-14, `seed_env = 1000 + s`, sharing C0's draws
  (common random numbers).
- Reduced from PLAN's 30 for compute (about 11.5 h instead of about 23 h). The reduction is stated
  in the report.

**Per-seed outcomes.** Computed over the PLAN section 4.4 window of 100 measurement episodes:
- `padding_index` = mean `val_measured` / mean `val_true`
- `welfare_ratio` = mean `W` / `W_oracle`
- `specification_gap` = mean `val_measured` / `val_oracle` - `welfare_ratio`

`W_oracle` and `val_oracle` come from `solve_oracle(arm config, 40, False)` for each arm.

**Aggregation.**
- Per arm: the IQM over seeds, with a 95% bootstrap CI (stratified over seeds, single task,
  10,000 resamples).
  - The IQM itself is `rliable.metrics.aggregate_iqm`.
  - The bootstrap is a percentile bootstrap over seeds, generator seed 0.
  - Why not `rliable.library`: it cannot be imported in this environment, because `arch` 7.2 is
    incompatible with pandas 3.0 (`deprecate_kwarg`).
  - With one task, `rliable`'s stratified bootstrap *is* a bootstrap over seeds, so the two are
    the same procedure.
- `Delta_X` is computed on `welfare_ratio`, **paired by seed** under common random numbers:
  `d_s = wr(C_X, s) - wr(C0, s)` for `s = 0..14`.
  - Its point estimate is the IQM of `d_s`, with a percentile bootstrap CI over seed indices
    (10,000 resamples, generator seed 0).
- `I` is computed seed-paired as `d_BOTH,s - d_OGAS,s - d_INC,s`, with the same bootstrap.

**Convergence.**
- Exploitability is audited on seeds 0-2 of each new arm. C0's audit comes from G3.
- Best-responder sizing and the threshold are as in R14.
- An arm above the threshold carries the NON-CONVERGED label beside every number derived from it.

## S3. Sobol (WO-033): not run

PLAN marks it optional and "JAX only". About 1,500-2,800 runs at 46 min is out of reach without a
JAX trainer. `sobol.py` stays a stub, and the final report states it was not run.

## S4. Estimator bias (WO-034, PLAN section 7.2)

**Grid.** `w in {0, 0.02, 0.05, 0.10, 0.25}` x `rho_cap in {1.2, inf}`: 10 arms on the Phase-1
(G1-recorded) configuration.

**(a) DP truth.** The single-enterprise DP's exact stationary `rho` distribution per arm, from the
same solver G1 used. True excess mass is computed on that distribution with the pre-registered
windows.

**(b) Simulation.**
- `N = 20` PPO populations at Phase-1 gate sizing, using the attempt-2 learner.
- Reuse the Phase-1 gate's arms: `notched` (w = 0, cap 1.2; 30 seeds) and `smooth` (w = 0.25,
  cap inf; 30 seeds).
- The other 8 arms run **5 seeds** each, sharing the Phase-1 gate's `seed_env` root, so there are
  40 new runs (about 3 h).
- Truth for (b) is relative to the arm's own smooth counterpart (PLAN: "relative to its own
  w = 0.25 arm").

**Estimator grid.**
- Excluded window in `{[0.95, 1.02], [0.97, 1.02], [0.93, 1.03]}`
- Degree in `{5, 7, 9}`
- Bin width in `{0.005, 0.01}`
- That is 18 settings. The pre-registered setting is marked.
- Reported per (arm, setting): bias, RMSE and CI coverage.

**Reconciliation power curve.**
- Ledgers come from `TruthfulMyopic` on the Phase-1 configuration, 20 replicates.
- Fictitious output is injected by inflating the claims of a random fraction
  `f in {0, 0.05, 0.1, 0.2, 0.3, 0.5}` of enterprise-periods by 20%. The inflation is keyed by
  replicate seed.
- Power is the rejection rate of `ledger_test` at 5%. At `f = 0` this is the size.
- The rest of the reconciliation call follows R13.5-6.

Digit tests and calibration claims stay out of scope.

## S5. LLM ministry study (WO-035, PLAN section 7.4)

The harness is implemented. The run needs model access this container does not have: no API key
is configured.
- The report records the study as NOT RUN, with the exact command the owner runs once a key and a
  budget are provided.
- PLAN section 14 estimates about 4M tokens.
- Models: the owner chooses them. The Anthropic client adapter is the default.
- Prompts: the WO-026 framings and the manipulation-check texts are the LEAD's final versions.
- Payoff arms: `padding_dominated` sets `audit_rate = 1` and `penalty_scale = 1000`.
  `overfulfilment_optimal` sets `ratchet_lambda = 0` and `overfulfilment_slope = 2.0`.

## S6. Price sensitivity (WO-036, PLAN section 7.5)

**Seeds.** `PRICE_PERTURBATION_SEEDS = (11, 12, 13)`, the same vectors G3 used (R14), so that every
headline table in the project shares one triple. The WO-036 stub's `(0, 1, 2)` is replaced by this
decision.

**Tables covered.** The G3 C0 headline, and every contrast arm.

**Rerun label.** An arm whose `objective_metric` reads prices (`net_output`: C_INC, C_BOTH) gets
the label "recomputation only; the objective reads prices, a behavioural answer needs a rerun".
That rerun is not performed, for compute reasons, and the report says so.

## S7. Final report (WO-037) and G4

`runs/final_report/report.md` rolls up G1-G3, the contrasts, the estimator-bias study, the price
tables, and the not-run statements for Sobol and the LLM study. It cites every manifest hash.

G4 needs the human's sign-off. G4's "LLM study" condition cannot pass until the owner runs S5.
