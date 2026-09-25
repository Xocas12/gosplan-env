AMBIGUITY REPORT   WO-019 (gate G2 criterion 1)   LEAD diagnosis after attempt 2 - ESCALATED

## Status

Attempt 2 of criterion 1 (per-period discount, AMBIGUITY-020), `runs/dp_vs_ppo/report.md`:
**0 of 30 seeds pass** again. Seed means at `a*pen` = 0.8 / 4 / 20: padding 0.318 / 0.009 / 0.004
(DP 0.0055 / 0.0016 / 0.0006), effort 0.573 / 0.811 / 0.834 (DP 0.495 / 0.498 / 0.513), W1
0.176 / 0.141 / 0.145 (tolerance 0.03), return 5.38 / 5.23 / 3.76 (DP policy in the environment
7.13 / 6.43 / 5.26). Returns improved on attempt 1 (4.84 / 5.37 / 3.23) at two of three levels;
the behavioural gap did not close. This record is the diagnosis the failure
calls for (PLAN section 4.5), and it ends in a decision that is not the lead's to take alone.

## Diagnosis: PPO never discovers the DP's target walk-down

The DP's optimal policy places about 12% of its stationary reports at rho = 0 at all three levels
(`share < 0.01`: 0.116 / 0.116 / 0.121). A report far below 1 forfeits the period's bonus notch
but cuts the next target by the ratchet's full downward step; without it the target rises every
period (growth directive `g` plus the upward ratchet), and the enterprise has to exert more and more
to stay at the notch. PPO's measured reports never go below about 1.00 in any seed.

Direct test (scratch computation, not a gate artefact): the same DP solved with reports restricted
to `rho >= 0.98` (`DPGrid(rho_lo=0.98)`), everything else unchanged, replayed in the environment:

| `a*pen` | DP effort | constrained-DP effort | PPO effort (attempt 2) | DP return | constrained-DP return |
|---|---|---|---|---|---|
| 0.8 | 0.495 | 0.250 (pads 0.84) | 0.22-0.75 (pads 0.05-0.72) | 7.13 | 6.79 |
| 4 | 0.498 | 0.766 | 0.79-0.87 | 6.43 | 5.23 |
| 20 | 0.513 | 0.980 | (attempt 1: 0.74-0.95) | 5.26 | 0.99 |

Removing the low reports reproduces PPO's signature: effort overshoot at `a*pen` = 4 and 20, heavy
padding at 0.8, and a lower return. The constrained DP is too strict at 20 (PPO can and does
report mildly below 1), which is why its return there is below PPO's; the direction is the point.

Why PPO cannot find it: the report head starts at rho = 1 with std 0.05 (PLAN section 6.1 and the
WO-017 card, pinned), and the payoff in rho is non-convex - a small step below 1 loses the whole
notch for a tiny target cut, while the gain appears only at a large step down. Local
policy-gradient search around rho = 1 sees only the loss. This is an exploration limit of the
pinned learner configuration, not a defect in the environment, the DP or the discount.

## Decision required (owner + LEAD, gate G2)

Criterion 1 cannot pass with the pinned learner settings. Options:

A. Change the learner's exploration and re-run criterion 1 (attempt 3): e.g. a wider initial
   report-head std than the pinned 0.05. This changes a PLAN-pinned TECH value, so it needs the
   owner's sign-off; it is a training-stack change validated exactly where criterion 1 is meant to
   validate it (`N = 1`, before any `N = 20` run), and no economic parameter, tolerance, level or
   budget moves.
B. Record criterion 1 as failed with this diagnosis and run criteria 2-4 as a separately labelled
   study, knowing the learner does not find strategies that need a large under-report. PLAN
   section 4.5 says a criterion-1 failure blocks everything, so this also needs the owner.
C. Stop at G2: report the failure and the diagnosis, and end Phase 1 here.

LEAD recommendation: A, with the new value fixed before attempt 3 and recorded here.

## OWNER DECISION (2026-09-25): option A

The owner chose A. LEAD implementation, fixed before attempt 3 and not tuned on any result:
`PPOConfig.report_head_init_std` defaults to `None`, meaning the report head takes the same
pre-squash log-std as every other head, 0 (CleanRL's default, sigma_z = 1), with the bias solved so
the squashed initial mean is exactly `rho = 1` (Gauss-Hermite expectation, bisection). At
`rho_max = 10` that puts about 28% of the initial report draws below `rho = 0.1` (0% under the
pinned 0.05) and keeps the mean on the notch. No other hyper-parameter, economic parameter,
tolerance, level, seed or budget changes. Attempt 2 is preserved under `runs/dp_vs_ppo/attempt2/`;
attempt 3 re-runs criterion 1 in full and is reported whatever it shows.

## OWNER DECISION (2026-09-25): after attempt 3

- If attempt 3 passes criterion 1: continue straight to criteria 2-4 (WO-020) as the
  pre-registered gate.
- If attempt 3 fails: record criterion 1 as failed with its diagnosis and run criteria 2-4 as a
  SEPARATELY LABELLED STUDY ("criterion 1 not met"), not as a G2 pass. The WO-020 report carries
  the criterion-1 status read from `runs/dp_vs_ppo/report.md` in its header.

## Attempt 3 result (wide report-head initialisation): FAIL, 0 of 30

`runs/dp_vs_ppo/report.md`. Seed means at `a*pen` = 0.8 / 4 / 20: padding 0.009 / 0.005 / 0.000
(within tolerance in 28 of 30 seeds), effort 0.287 / 0.215 / 0.238 (DP ~0.50; 0 of 30 within
tolerance), W1 0.592 / 0.696 / 0.735 (worse than attempt 2's ~0.14-0.18), return 3.17 / 2.69 /
2.46 (DP policy 7.13 / 6.43 / 5.26; attempt 2 5.38 / 5.23 / 3.76). The learner now under-reports
far more often than the DP (27-49% of reports below rho = 0.1 against about 12%), several seeds
collapse to zero effort, and returns fall. Wider exploration moved the policy past the DP's mixed
strategy instead of to it: criterion 1 is not met under either initialisation. Per the owner's
decision, criteria 2-4 now run as a separately labelled study.
