# Labelled study ES - record (LEAD)

**Status: complete. ES is a labelled study and re-evaluates no gate.** G3 (NOT PASSED), G3b, LC and
CT stand as recorded. Written on 2026-10-03; the human sign-off is pending. The reading of the
R18 test on the ES populations was revised on 2026-10-14 (#106), as CT's was; no number changed.

- Pre-registration: `spec/P2_REVISION.md` R19, committed in 3db46a9 before any ES run.
- Driver: `gosplan/experiments/entropy_schedule.py`. Report and data:
  `runs/entropy_schedule/report.md`, `comparison.json`, `result.json`.
- Design: C0, seeds 0-9, 3M agent-steps, as in LC. The difference from LC is the entropy bonus,
  which anneals 0.01 -> 0.001 over the first 1,000 updates (1M agent-steps, the 1M runs' exact
  schedule) and is then held at 0.001. Everything else is LC's, including the R16 audit.
- Run history:
  - Launched on 2026-10-03 at 09:58 UTC; finished at 20:58 UTC.
  - Two container reboots killed the first training batch at about 1,450 and 360 of 3,000
    updates. Those runs were retrained from scratch with identical configuration.
  - Exact resume snapshots were then added to the training harness (R19 note 1). Later restarts
    resumed from the last snapshot. `tests/unit/test_train_resume.py` shows a resumed run is
    bitwise equal to an uninterrupted one.
  - A restart interrupted the R16 audit with 5 of 10 best responders done. The completed ones
    were reused (R19 note 2, `tests/unit/test_best_responder_reuse.py`).
  - Neither change touches the design, the rules or any result.

## Pre-registered outcome (R19)

| Rule | Result |
|---|---|
| Collapse reproduced under the matched schedule (CI upper bound of the median final mean effort < 0.05) | **YES**: median 0.019 [95% CI 0.012, 0.023] |

Per-seed final mean effort is between 0.010 and 0.025, which is the range LC found (0.005-0.036).

**Reading (R19's own reading of the rule).** More training, not the stretched exploration
schedule, drives the collapse. LC's design caveat (`runs/LC_record.md`, addendum) is resolved for
this learner: with the 1M runs' exact schedule, a 3M-step run still collapses.

## Descriptive results (not tested)

**Trajectories.** Median over seeds of the periodic evaluation:

| agent-steps (k) | 250 | 500 | 750 | 1000 | 1250 | 1500 | 1750 | 2000 | 3000 |
|---|---|---|---|---|---|---|---|---|---|
| entropy coef, ES | 0.0078 | 0.0055 | 0.0033 | 0.0010 | 0.0010 | 0.0010 | 0.0010 | 0.0010 | 0.0010 |
| median effort, 1M runs (G3b) | 0.662 | 0.615 | 0.267 | 0.089 | | | | | |
| median effort, ES | 0.662 | 0.615 | 0.267 | 0.089 | 0.043 | 0.028 | 0.021 | 0.019 | 0.022 |
| median effort, LC (stretched) | 0.663 | 0.603 | 0.348 | 0.116 | 0.033 | 0.023 | 0.022 | 0.022 | 0.020 |
| median eval return, ES | -207 | -69 | -57 | -4.6 | -0.80 | -0.75 | -0.01 | 0.00 | 0.35 |

- **Through 1M steps ES reproduces the 1M runs exactly.** Same seeds, same configuration and the
  same schedule make it the same computation, so this is a consistency check rather than a
  finding.
- **After 1M steps the slide continues with the entropy coefficient fixed.** Effort halves again
  by 1.25M and reaches the collapse level of about 0.02 by 1.75M, where it stays. Each
  enterprise's own evaluation return keeps rising along the way, as in LC.
- **Both 3M schedules end in the same place.** LC's longer exploration delays nothing visible
  after 1.25M.

**Outcomes at 3M.**
- Welfare: `welfare_ratio` 0.0000, as in LC.
- Quality: mean qbar (`quality.mean_quality` in `result.json`) is 0.000-0.003 per seed (at most
  0.0031), against at most 3.4e-5 for LC and 0.06-0.75 for the 1M populations.
- Held-out rows in the harness output (descriptive only):
  - row 2 (storming) and row 5 (hoarding) appear: excess Gini 0.222 (LC 0.220), request-inflation
    excess 1.79 (LC 1.98);
  - row 6 (trade volume) stays at 0.
- R16 exploitability:
  - Median -0.008 and max 0.173, with 1 of 10 seeds above 5%. LC 3M had median -0.003 and 3 of
    10 above.
  - The paired difference from LC is median +0.003 [95% CI -0.296, +0.950], so no difference is
    shown.
  - By R17's convergence rule (max <= 5%) the ES populations are **not converged** either.

**R18 test on the ES populations (R18's rules unchanged; both tests hold, which R18 calls "trap:
YES"; qbar 0.000-0.003).**

| Test | Result |
|---|---|
| (i) Producing alone does not pay | `d1` median -1.785 [-2.056, -1.449] |
| (ii) Everyone producing pays more | `d2` median +1.111 [+0.678, +1.295] |

- A lone truthful-myopic producer in seat 0 earns between -1.47 and -1.43 on every seed. That is
  the autarky payoff: on every seed it equals, to the last digit, CT's `R_dev` against the LC
  populations and seat 0's return against 19 inert seats
  (`runs/coordination_trap/self_enforcement.md`).
- When everyone produces, **every seat is better off on all 10 seeds**.
- Seat 0's learned effort rises from about 0.02 to between 0.03 and 0.45 (median about 0.15)
  when the other seats are producers. This matches CT's finding that the learned policy has
  partly unlearned production.
- As for LC (`runs/CT_record.md`, revised 2026-10-14, #106), the two tests do not show a
  coordination trap. They do not test whether all-production is self-enforcing, and under C0 it
  is not: among truthful-myopic producers an enterprise gains by dropping quality.

## Limits

- One alternative schedule was tested. ES separates budget from *this* schedule; it does not
  exhaust the learner's exploration design.
- The learning rate is constant in every study, so no learning-rate schedule was tested.
- The study learner is the only learner (L1, L4). Whether other learners reach the same state is
  open.
- The R18 test uses one producer policy, `TruthfulMyopic`, as in CT.
- No gate is re-evaluated. Limitations L1-L5 stand.

## Consequences

- **L5 is sharpened, not changed.** The budget dependence of learned-economy outcomes is a
  training-length effect for this learner. It is not an artefact of the annealing schedule.
- CT's finding now holds under both 3M schedules. Given enough training, C0 learners settle into
  a no-production, zero-quality state that all-production by the truthful-myopic producer
  Pareto-dominates. Neither CT nor ES shows that this state is a coordination trap rather than a
  quality free-rider dilemma (#106).
- Any follow-up is the owner's decision and needs a new pre-registration (#97).
