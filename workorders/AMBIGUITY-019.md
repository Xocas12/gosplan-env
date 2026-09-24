AMBIGUITY REPORT   WO-019 / WO-020 (gate G2)   raised by the LEAD
Two defects found by the WO-019 pilot, and the G2 run sizing, ruled before any G2 run.

## Evidence

A pilot PPO run on the WO-019 recovery setting (`N = 1`, `a*pen` = 20, 2M agent-steps, seed 0;
scratch output, not a G2 artefact) reached an evaluation return of 9.35 against 5.59 for
`DPGreedy` replaying the DP in the same environment, and 5.85 for the DP's own value at the
opening state. A correct DP cannot be beaten by a policy on the model it solves, so the
environment and the DP disagreed. The ledger showed the learned policy producing almost nothing,
reporting rho ~ 1.024 on near-zero stock, and never being audited except in plan period 13.

## A. Fictitious padding read the wrong stock (WO-016, `phenomenon_padding`)

`padding = mean max(0, R - S) / T` used `inv_output_pre` on the REPORT row: the stock BEFORE the
period's output is booked. The audit (`audit_and_penalise`) and the DP (`S'`) compare the report to
the stock AFTER booking, before the next period's shipment, which is `inv_output_post` on the REPORT
row. The old read counted every truthful report as padding of the period's own output (pilot at
200k steps: 0.131 old, 0.021 corrected).

RULING: `S_i` in PLAN section 4.1 row 4 is the audited stock, `inv_output_post` on the REPORT row.
Fixed in `gosplan/metrics/phenomena.py`. The frozen tests use `pre == post` and are unaffected.

## B. Audit selection keyed by the configuration's root seed (WO-006 / WO-009, and `ref/`)

`select_audits(view, cfg, t)` draws `Bernoulli(a)` at key `(cfg.tech.seed_env, "audit", t, i)`.
Every other environment draw is keyed by `state.seed_env`, the EPISODE's seed set by
`GosplanEnv.reset(seed_env, ...)`. With the root seed, every episode of a run shares one audit
schedule (at the pilot configuration: period 13 only, among the periods an episode usually
reaches), so padding is almost never checked and a learner exploits that. PLAN section 2.15 keys
every draw from the episode's `seed_env`; the reference implementation had the same slip, so the
golden files could not catch it.

RULING: the audit selection is keyed by the episode's seed. The frozen signature
`select_audits(view, cfg, t)` is kept; `step.stage_audit` hands it the configuration re-seeded to
`state.seed_env` (a no-op whenever the two agree), and `ref/ref_step.py` does the same. Golden files
regenerated (`uv run python -m ref.gen_golden`); T-B7 parity holds. New LEAD test
`tests/behavioural/test_audit_seeding.py` fails before the fix and passes after.

Rule-7 note (addendum to `runs/G0_rule7_review.md`): the change alters which periods are audited,
not the audit rule; it removes an exploitable regularity and adds no behavioural content. Clean.

Downstream: the DP, the regime map and the G1 decision are unaffected (the DP integrates the audit
analytically at rate `a`). `runs/mc_sanity/` was produced on the old keying and is re-run.

## C. G2 run sizing (`TrainConfig` sizing fields carry no defaults, WO-018)

PLAN section 14 gives about 2M environment steps per recovery run at `N = 1`, and about 5M per
Phase-1 run at `N = 20` "at ~3k steps/s NumPy, about 30 min". The measured throughput of this
environment at `N = 20` is about 1,000 environment steps/s per process (about 20,000 enterprise
decisions/s), so 5M environment steps would take about 83 min per run and about 28 h for the 80
runs of WO-020 on the 4-core reference container.

RULING (fixed before any G2 run):
- Recovery runs (WO-019): 8 environments x 125 agent-steps (25 plan periods) per update,
  2,000,000 agent-steps (2,000 updates), measured on 400 fresh episodes. PLAN's figure, unchanged.
- Gate runs (WO-020): 8 x 125 per update, 1,000,000 agent-steps (1,000 updates) = 20M enterprise
  decisions into the shared policy, ten times the samples of a recovery run; measured on 100 fresh
  episodes (about 1,500 measured REPORT rows per seed at `N = 20`). About 17 min per run.
- Criterion 3's two extra `a*pen` levels (0.8, 4.0): 10 seeds each, criterion 1's count. The
  Phase-1 level reuses the 30 notched runs.
If a budget proves insufficient, that is a finding and reported; a longer budget is a new,
labelled study (PLAN section 13), never a silent re-run.

## D. Price-sensitivity table at G2

The WO-020 card asks for PLAN section 7.5's price table. That check recomputes `padding_index`,
`welfare_ratio` and `specification_gap`; the last two need the Phase-2 oracle (WO-026), and the
check itself is WO-036 (Phase 3). No G2 criterion is price-weighted.

RULING: deferred to WO-036, which must run it on the G2 ledgers; the WO-020 report says so.
