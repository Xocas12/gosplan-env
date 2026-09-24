# Gate G0 — scaffold and sanity: SIGN-OFF

Gate G0 (PLAN section 13): *full frozen suite green; MC sanity report clean; lead's diff review of
`env/` against CONTRACT rule 7.* Sign-off: LEAD. Date: 2026-09-24.

| Condition | Evidence | Status |
|---|---|---|
| Full frozen suite green | `uv run pytest -q`: 456 passed, 16 skipped (later cards: PPO, the rest of the Phase-1 metrics, estimator tests held by AMBIGUITY-011), 0 failed. CI `ci-ok` green on PR #84. Golden parity T-B7 exact on all 30 cells. | PASS |
| MC sanity report clean | `runs/mc_sanity/report.md`: **ALL ASSERTIONS HELD**. 21 configurations (baseline + 20 SUPPLY perturbations, `theta = inf` exercised) x 3 agents x 2,000 episodes = 126,000 episodes. Max conservation residual 3.6e-15 (tol 1e-9); 0 non-finite values; 0 target, stock or fill bound failures; Padder shortage (fill < 1) in every configuration, minimum fill 0.19; no `BOUND_BINDING` flag. Manifests `runs/<hash>/manifest.json` for all 21 configurations. | PASS, with the limitation below |
| Rule-7 diff review of `gosplan/env/` | `runs/G0_rule7_review.md`: no transition rule or reward term implements bunching, padding, storming, hoarding, shaving or trade. | PASS |

## Limitation recorded with the pass (AMBIGUITY-014 A)

Every episode of a cell reset to the same `seed_env` (the card's literal wording), so the
deterministic `TruthfulMyopic` and `Padder` contribute one trajectory per configuration; `Random`
contributes 2,000 distinct trajectories per configuration. The G0 properties are deterministic
per trajectory, so they were verified on 42,000 + 42 distinct trajectories. Later harnesses use
per-episode seeds (`root + e`, shared across agents) per the AMBIGUITY-014 ruling.

## Cost observed

0.070 s per episode at N = 20 (about 1.6 ms per agent-step, half of it ledger-record building);
the sweep took 3.0 h on one core. Relevant to the PLAN section 14 training budget (WO-018): the
per-step `StepRecord` construction is the first optimisation target.

## Decision

**G0 is signed off (LEAD).** The next card is WO-013 (spec v1 freeze).
