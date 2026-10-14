# Labelled study CT - record (LEAD)

**Status: complete. CT is a labelled study, is evaluation only (no training) and re-evaluates no
gate.** Written on 2026-09-30; the human sign-off is pending. The reading was revised on
2026-10-14 (#106): R18's two tests hold at 3M as recorded, but they do not identify a coordination
trap, and this record no longer calls the 3M state one. The numbers and the pre-registered
verdicts are unchanged.

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

`qbar` is the populations' mean period-average quality over seeds 0-9, from the acceptance
harness (`quality.mean_quality` in LC's and G3b's `result.json`). R18 names the conjunction of
(i) and (ii) "coordination trap", and the driver's report prints it under that name; the last
column gives R18's verdict, and the reading below says why this record does not use the name.

| Populations | qbar | (i) Producing alone does not pay (`d1 = R_dev - R_pop` < 0) | (ii) Everyone producing pays more (`d2 = W_tm - W_pop` > 0) | R18 verdict ("trap": both hold) |
|---|---|---|---|---|
| **3M (primary)** | about 0: at most 3.4e-5 on every seed | **yes**: median -1.718 [95% CI -2.240, -1.438] | **yes**: median +0.869 [0.494, 1.303] | **YES** |
| 1M (contrast) | 0.06-0.75 (median 0.36) | not shown: median +0.336 [-1.208, 34.004] | yes: median +3.632 [1.539, 40.570] | NO |

**What the two tests show at 3M**, against the truthful-myopic producer:

- Every seed's lone producer loses. Seat 0's return when producing alone is between -1.47 and
  -1.43, against -0.74 to +1.05 when it plays the learned policy.
- When every seat produces, the mean seat return is 1.26-1.32. Every seat is better off on 9 of
  10 seeds, and 95% of seats are on the tenth.

At 1M test (i) does not hold. Several populations are so far from equilibrium (seat-0 returns
down to -39.6) that producing alone would help. This is consistent with L4.

**Reading (revised 2026-10-14, #106): the 3M state is not shown to be a coordination trap.**
(i) shows that the no-production state is stable against the producer, and (ii) that
all-production Pareto-dominates it. Neither shows that all-production is itself an equilibrium,
which a coordination trap needs: mutual defection in a prisoner's dilemma passes both tests. Under
C0, all-production by the truthful-myopic producer is not one (second point below). Source for the
numbers below: `runs/coordination_trap/self_enforcement.md`
(`python -m gosplan.experiments.coordination_trap --self-enforcement`; heuristic agents only,
CT's protocol, descriptive and not pre-registered).

- **Test (i) compares the learned populations with the autarky payoff.** The 3M `R_dev` equals,
  to the last digit, seat 0's return as a producer when the other 19 seats play the inert
  all-zero action, on all 10 seeds. ES's `R_dev` is the same number on every seed. So `d1`
  compares `R_pop` with a constant of the C0 configuration, and a population that had learned
  only to do nothing (`R_pop = 0`) would pass it. What it does show is that the 3M populations
  deliver nothing a producer can use.
- **All-production is not self-enforcing against a quality cut.** C0 rewards `val`, so an
  enterprise's own quality enters its own reward only as a cost (`kappa_q * q * e`), while the
  benefit goes to its buyers (deliveries are credited `deliv * qbar`) and to consumers. Among
  truthful-myopic producers, seat 0 playing the same policy at quality 0 instead of 1 raises its
  return from 1.10-1.22 to 2.12-2.28, a gain of +1.02 to +1.08 on every seed.
- **The 3M populations play quality of about 0** (qbar at most 3.4e-5). With every seller at
  quality 0, every delivered input is worth 0 whatever effort anyone chooses: a producer among
  truthful-myopic producers who play quality 0 earns exactly the autarky payoff. Low effort can
  then be read as the best reply to receiving nothing, rather than as the cause of the collapse.

The data therefore fit a quality free-rider dilemma at least as well as a coordination trap, and
CT cannot tell the two apart. Telling them apart needs a test that all-production is
self-enforcing: seat 0's return in condition 4, which CT computed but did not record, and whether
a seat among producers gains by dropping quality or effort. That would be a new pre-registration
and is the owner's decision. The escape tests in #97 assume a coordination game; a free-rider
dilemma points instead towards incentive reform such as R3_QW.

## Descriptive observations (not tested)

- **The lone producer's return does not depend on the budget: it is the autarky payoff.**
  - The producer in seat 0 earns the same return against the 1M and the 3M populations of each
    seed, for example -1.425 against both for seed 1000. The 1M `R_dev` equals the autarky
    payoff (to rounding) on 9 of 10 seeds; on 1004 it is lower (-1.4545 against -1.4498).
  - The 1M against 3M contrast in test (i) is therefore a contrast in `R_pop` only.
  - Against all-producers, the same seat is better off.
  - At 3M, zero quality alone explains why the producer receives nothing usable. At 1M qbar was
    0.06-0.75 and `R_dev` is still the autarky payoff on 9 of 10 seeds, so quality alone does
    not explain the 1M case. There, the input starvation that the hoarding result (row 5) and
    the buy-only trade offers (row 6, G3b) point to remains a reading: CT did not measure the
    producer's input receipts, and rows 5 and 6 are themselves under review (#109, #107).
- **The learned policy responds only weakly to available inputs.** Seat 0's learned effort rises
  when the other seats are producers: at 3M the median rises from about 0.02 to about 0.14, with
  seed 1007 flat at 0.006. That is still far below what a producer supplies, so the trained policy
  has partly unlearned production. It is not only waiting for inputs.

## Limits

- One producer policy was used. Test (i) holding for `TruthfulMyopic` (the producer losing when
  it produces alone) does not show that no producing strategy pays alone.
- R18's two tests do not test whether all-production is self-enforcing (see the reading above).
- CT reuses the 3M populations as trained. Why training reaches the collapse is not tested.
- No gate is re-evaluated. G3 (NOT PASSED), G3b and LC stand as recorded, and so do limitations
  L1-L5.

## What this adds to the project's claims

- **What CT shows about the learned C0 economy.** With long enough training, learning
  enterprises under the C0 planner settle into a no-production, zero-quality state (qbar about
  0) that all-production by the truthful-myopic producer Pareto-dominates. A producer among them
  receives nothing usable and earns the autarky payoff.
- **What it does not show.**
  - That the state is a coordination trap rather than a quality free-rider dilemma (above).
  - That no single enterprise can profitably leave it. CT tried one producer policy, and the
    R16 audit of the same 3M populations found a warm-started best responder gaining more than
    5% on 3 of 10 seeds (1002 0.429, 1008 0.125, 1006 0.074; `runs/LC_record.md`). The
    committed best-responder artefacts are manifests only, so whether those gains involve
    producing cannot be checked. An earlier version of this record said the state was one "that
    no single enterprise can profitably leave"; that was unsupported, and in tension with the
    audit.
- **It is not Claim A.** It is none of the four held-out phenomena.
- **It rests on the stated learner (L1, L4, L5).** Whether other learners reach the same state is
  open. That question, and any follow-up, is the owner's decision and needs a new
  pre-registration.
