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
