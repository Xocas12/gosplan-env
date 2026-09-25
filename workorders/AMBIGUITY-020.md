AMBIGUITY REPORT   WO-019 (gate G2 criterion 1)   LEAD diagnosis of the criterion-1 failure

## The failure

`runs/dp_vs_ppo/attempt1/report.md` (git `69e3f2d`): PPO against the single-enterprise DP at
`a*pen` = 0.8, 4, 20, 10 seeds each, 2M agent-steps per run (AMBIGUITY-019 C). **0 of 30 seeds**
meet all three tolerances. PLAN section 4.5: a criterion-1 failure is a training-stack failure; it
blocks criteria 2-4, and the next step is this diagnosis, never a tolerance, level or economic
parameter change. No criterion-2-4 run has been made.

## Evidence

1. The DP is a valid optimum on the (audit-fixed, AMBIGUITY-019 B) environment. `DPGreedy`
   replaying the DP policy in the environment on the measurement seed block (400 episodes) earns
   7.13 / 6.43 / 5.26 at `a*pen` = 0.8 / 4 / 20; PPO's seed means are 4.84 / 5.37 / 3.23 and its
   best seeds 6.18 / 6.21 / 5.66. PPO is below the optimum, not above it.
2. The shortfall is systematic, not noise. At `a*pen` = 4 and 20 PPO pads correctly (padding
   within 0.02 of the DP in 19 of 20 seeds) but over-exerts (effort 0.74-0.95 against the DP's
   0.50-0.51) and reports just above target (rho ~ 1.02-1.03) rather than at 1.00. The DP places
   12% of its stationary reports at rho = 0: it periodically under-reports to walk its target
   down through the ratchet's downward step, a long-horizon strategy, and runs lower effort against
   the lower target. PPO never learns it; the W1 distance (0.13-0.17 at those levels) is dominated by that mass.
3. The two optimisers discount differently. The spec writes the DP's Bellman continuation as
   `psi * gamma` PER PLAN PERIOD (spec section 5, `DP_DISCOUNT = 0.99`). The harness applied
   `PPOConfig.gamma = 0.99` PER AGENT-STEP in GAE, and a period is `M + 1 = 5` agent-steps, so the
   learner discounted at 0.99^5 = 0.951 per period against the DP's 0.99 - five times the DP's
   per-period impatience (effective horizons of about 20 versus 100 periods before `psi`). A
   learner that impatient cannot value a target walked down over several periods.

## Ruling

`gamma` in PLAN sections 5 and 6.1 is ONE quantity, a per-plan-period technical discount, as the
spec's `psi * gamma` Bellman operator states. The learner applies it per agent-step as
`gamma ** (1 / (M + 1))` (`gosplan.agents.ppo.train.step_discount`, recorded in every manifest as
`gae_step_discount`). `PPOConfig.gamma` stays 0.99; `lambda_GAE` stays 0.97 per agent-step (it
trades bias against variance and does not change the objective). This is a correction of the
training stack to the objective the spec already states, not a change of any economic parameter,
tolerance, level or budget.

## Consequences

- Attempt 1 is kept as it stands under `runs/dp_vs_ppo/attempt1/` (report, manifests), labelled
  as run with the per-agent-step discount. It is not deleted or overwritten.
- Criterion 1 is re-run in full (attempt 2) with the same levels, seeds, tolerances and sizing.
  Its result is reported whatever it is; a second failure goes to a further LEAD diagnosis.
- A single-seed diagnostic pilot at `a*pen` = 20 on the corrected discount was run before the
  re-run to confirm the diagnosis moves the learner toward the DP; it is scratch output, not a
  gate artefact, and it does not select anything the re-run then uses.
