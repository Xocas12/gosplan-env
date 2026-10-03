# Labelled study LC - record (LEAD)

**Status: complete. LC is a labelled study and re-evaluates no gate.** G3 (NOT PASSED) and G3b stand
as recorded. Written on 2026-09-29; the human sign-off is pending.

- Pre-registration: `spec/P2_REVISION.md` R17, committed in cfba026 before any LC run.
- Driver: `gosplan/experiments/learner_convergence.py`. Report: `runs/learner_convergence/report.md`.
- Design: C0, seeds 0-9, population budget 3M agent-steps against G3b's 1M. Everything else is
  unchanged, including the R16 audit.
- The run started at 09:15 UTC. A container reboot at about 15:25 UTC killed it with 8 of 10 runs
  complete.
  - It was relaunched at 15:30 UTC. The 8 completed runs were read back.
  - The 2 interrupted runs (seeds with hashes `3f0dee0b`, `cc06e97e`) were retrained from scratch
    with identical configuration.
  - It finished at 20:04 UTC.

## Pre-registered outcome (R17)

| Rule | Result |
|---|---|
| Exploitability falls with budget (CI upper bound of the median paired difference < 0) | **YES**: median difference -0.857 [95% CI -1.793, -0.130] |
| Converged at 3M (max over seeds <= 5%) | **NO**: max 0.429; 3 of 10 seeds above the threshold |

Median exploitability went from 0.797 (1M) to -0.003 (3M). Per-seed values are in the report.

## What the 3M populations are doing: the economy has collapsed to near-zero output

Read the primary result against this before citing it.

| C0, seeds 0-9 | Mean effort | Mean return | Fictitious padding | Welfare (mean W) | welfare_ratio |
|---|---|---|---|---|---|
| 1M (G3b) | 0.195 | -16.0 | 0.040 | 0.087 | 0.0335 |
| 3M (LC) | 0.019 | +0.27 | 0.000 | 0.000 on every seed | 0.0000 |

- With three times the training, the learned enterprises stop producing: mean effort 0.005-0.036
  on every seed. They also stop padding.
- Their returns rise from about -16 to about 0. The large losses the 1M populations carried are
  gone.
- Welfare is exactly 0 on all 10 seeds. The oracle and the truthful-myopic baseline, on the same
  configuration, reach W of 1.98 and 1.06, so the environment supports production. The collapse is
  in what the learners converge to.
- Every posted trade offer is still a buy order. The mean offer is -1.000, and none of the 3M
  populations' offers is a sell. Row 6 stays at zero.

**Reading of the primary result.** The fall in exploitability is real by R17's rule, but it is not
convergence to a functioning equilibrium.
- The 3M populations sit where returns are near 0 and, with the ratio floor of 1 reward unit, a
  best responder has little to gain.
- On two seeds (1003 and 1005) the warm-started best responder ended far *below* the population it
  started from (ratios -2.7 and -2.2). R16's audit therefore also measures the best-responder
  learner's own instability, which bounds how much the audit can certify.
- A plausible mechanism is a no-production coordination trap. Production needs other
  enterprises' outputs as inputs (near-Leontief, `theta = 8`), so when every supplier produces
  nothing, effort is wasted. **LC does not test this. It is an interpretation, not a finding.**

## Update (2026-09-30): the coordination-trap reading was tested

See `runs/CT_record.md`. R18 was pre-registered and run as an evaluation-only test. Against the
truthful-myopic producer, the 3M collapse **is** a coordination trap: producing alone does not pay,
and everyone producing pays more. The "interpretation, not a finding" caveat above therefore no
longer applies to that reading. What training does to reach the trap is still untested.

## Addendum (2026-10-01): training trajectories and a design caveat (descriptive, post hoc)

Not pre-registered and no test is attached. Source: `runs/learner_convergence/trajectories.md`,
from the runs' `train_log.jsonl` (`python -m gosplan.experiments.learner_convergence --trajectories`).

| agent-steps (k) | 250 | 500 | 750 | 1000 | 1250 | 1500 | 2000 | 3000 |
|---|---|---|---|---|---|---|---|---|
| median effort, 1M runs | 0.66 | 0.62 | 0.27 | 0.09 | | | | |
| median effort, 3M runs | 0.66 | 0.60 | 0.35 | 0.12 | 0.03 | 0.02 | 0.02 | 0.02 |
| median eval return, 3M runs | -191 | -43 | -20 | -6 | -1.3 | -0.01 | 0.49 | 0.71 |

- **The collapse is a steady slide, not a late event.**
  - Effort falls from about 0.66 to the trap level of about 0.02 by roughly 1.25-1.5M steps, then
    stays there.
  - Each enterprise's own evaluation return rises along the whole path. The learners improve
    their own payoff while welfare falls.
- **The 1M populations were mid-slide.** Their median effort at 1M steps was already 0.09. G3, G3b
  and the Phase-3 contrasts therefore describe an economy partway into the trap, which sharpens L5.
- **Design caveat.**
  - The entropy bonus anneals from 0.01 to 0.001 over the whole run, so LC's budget manipulation
    also stretches the exploration schedule. At the 1M-step mark a 3M run's coefficient is 0.0070,
    against 0.0010 for a 1M run.
  - The learning rate is constant, so the entropy schedule is the only part of the setup that
    scales with the budget.
  - Through the first 1M steps the two budgets' trajectories nearly coincide despite the different
    schedules. That suggests the schedule is not what drives the collapse, but LC cannot separate
    budget from schedule. R17's "everything else unchanged" holds for the configuration, not for
    the effective schedule.

## Update (2026-10-03): the design caveat was tested

See `runs/ES_record.md`. R19 was pre-registered and run: 3M-step populations with the 1M runs'
exact entropy schedule. They collapse just the same, with median effort 0.019 [95% CI 0.012,
0.023], and land in the same coordination trap. The collapse is a training-length effect for this
learner, not an artefact of the stretched schedule.

## Consequences

- **L4 stands.** The populations are non-converged at both budgets.
- **New limitation L5.** Learned-economy outcomes depend on the training budget. At 1M the C0
  economy is heavily degraded (welfare_ratio 0.034), and at 3M it has collapsed (0.000). Every
  learned-agent number in this repository comes from 1M-step populations: G3, G3b and the Phase-3
  contrasts. Each describes a point on a learning trajectory, not a limit, and none may be read as
  the institution's steady state.
- The Phase-3 contrasts are not re-run. Whether their deltas hold at 3M is unknown.
- Any follow-up is the owner's decision and needs a new pre-registration (R17: "no further budget,
  learner or configuration change follows from it without a new pre-registration"). Candidates:
  - a learner with a different exploration or population scheme;
  - contrasts at a fixed longer budget;
  - a direct test of the coordination-trap reading, for example a population seeded with producers.
