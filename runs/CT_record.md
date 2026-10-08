# Labelled study CT - record (LEAD)

**Status: complete. CT is a labelled study, is evaluation only (no training) and re-evaluates no
gate.** Written on 2026-09-30; the human sign-off is pending.

- Pre-registration: `spec/P2_REVISION.md` R18, committed in 18864d0 before the CT evaluation. It
  discloses a 2-episode smoke run on one seed made to check the code.
- Driver: `gosplan/experiments/coordination_trap.py`. Report and data:
  `runs/coordination_trap/report.md`, `result.json`.
- Design:
  - Populations: LC's 3M C0 populations (primary) and G3b's 1M populations (contrast), seeds 0-9.
  - 50 episodes per condition, on the measurement seed block, with deterministic policies and
    common random numbers.
  - Producer: `TruthfulMyopic`, the project's reference line. It sets effort to meet the target,
    reports stock truthfully, requests at need and never trades.

## Pre-registered outcome (R18)

| Populations | (i) Producing alone does not pay (`d1 = R_dev - R_pop` < 0) | (ii) Everyone producing pays more (`d2 = W_tm - W_pop` > 0) | Coordination trap |
|---|---|---|---|
| **3M (primary)** | **yes**: median -1.718 [95% CI -2.240, -1.438] | **yes**: median +0.869 [0.494, 1.303] | **YES** |
| 1M (contrast) | not shown: median +0.336 [-1.208, 34.004] | yes: median +3.632 [1.539, 40.570] | NO |

**Reading.** The collapse LC found at 3M is a coordination trap in the textbook sense, against the
truthful-myopic producer:

- Every seed's lone producer loses. Seat 0's return when producing alone is between -1.47 and
  -1.43, against -0.74 to +1.05 when it plays the learned policy.
- When every seat produces, the mean seat return is 1.26-1.32. Every seat is better off on 9 of
  10 seeds, and 95% of seats are on the tenth.

The 1M populations are not a trap. Several are so far from equilibrium (seat-0 returns down to
-39.6) that producing alone would help. This is consistent with L4.

## Descriptive observations (not tested)

- **The lone producer's return barely depends on the budget.**
  - The producer in seat 0 earns almost the same return against the 1M and the 3M populations of
    each seed, for example -1.425 against both for seed 1000.
  - Against all-producers, the same seat is better off.
  - Both learned populations therefore leave a producer similarly short. This fits the input
    starvation that the hoarding result (row 5) and the buy-only trade offers (row 6, G3b) point
    to. CT did not measure the producer's input receipts, so this is a reading.
- **The learned policy responds only weakly to available inputs.** Seat 0's learned effort rises
  when the other seats are producers: at 3M the median rises from about 0.02 to about 0.14, with
  seed 1007 flat at 0.006. That is still far below what a producer supplies, so the trained policy
  has partly unlearned production. It is not only waiting for inputs.

## Limits

- One producer policy was used. A trap against `TruthfulMyopic` does not show that no producing
  strategy pays alone.
- CT reuses the 3M populations as trained. Why training reaches the trap is not tested.
- No gate is re-evaluated. G3 (NOT PASSED), G3b and LC stand as recorded, and so do limitations
  L1-L5.

## What this adds to the project's claims

- **This is the first tested mechanism result about the learned C0 economy.** With long enough
  training, learning enterprises under the C0 planner settle into a Pareto-dominated
  no-production state that no single enterprise can profitably leave.
- **It is not Claim A.** It is none of the four held-out phenomena.
- **It rests on the stated learner (L1, L4, L5).** Whether other learners reach the same trap is
  open. That question, and any follow-up, is the owner's decision and needs a new
  pre-registration.
