AMBIGUITY REPORT   WO-002 / WO-007 / WO-010   ref/, gosplan/env/reporting.py, gosplan/agents/
Question (one sentence):
What do the golden `TruthfulMyopic` policy and the production agent play, what unit is
`input_request` in, and which rollout observation does a policy act on?

What the spec says / does not say (quote):
1. `ref/gen_golden.py::ref_truthful_myopic_policy` returned `effort = [1.0] * n`, while its own
   docstring, PLAN section 6.1 and the WO-010 card say `effort = clip(T_i / (A * cap), 0, 1)`.
2. T-B1 (`test_truthful_report_equals_stock`) requires the truthful report to equal the stock the
   REPORT step leaves, `(1 - h) * S + y` capped at `S_max`; the observation at the REPORT decision
   carries the stock carried in (field 5) and this period's output (field 4).
3. The WO-009 card (note 9), `action_spec` and T-B1 (`requests == need` with `REQUEST_MULTIPLE_NEED
   = 1.0`) treat `input_request` as a multiple of need; `process_reports` and `ref_report` clipped
   it as units.
4. `ref_rollout` fed the policy a fresh observation at each period head, but the frozen golden
   replay feeds the observation `GosplanEnv.step` returned - different after an auto-continued
   termination (AMBIGUITY-004).
5. Two frozen tests' code disagreed with their own docstrings: the T-B1 histogram clause skipped
   only bins with both neighbours empty (docstring: both neighbours non-empty), and
   `test_zero_stock_claim_gives_zero_fill` did a rollout although its docstring specifies a
   constructed state with `S = 0` - the rollout can never pair a zero stock with its own claim.

LEAD RESOLUTION (2026-09-24):
1. `ref_truthful_myopic_policy` plays `clip(initial_target_frac * exp(obs[2]), 0, 1)`, the
   docstring formula (configuration constants bound by `make_policy`).
2. The truthful report is `min((1 - h) * obs[5] + obs[4], S_max / T)` - the post-REPORT stock
   rebuilt from what the agent observes exactly (PLAN section 2.5: at REPORT the agent observes S
   and y) - in both the reference policy and `TruthfulMyopic`.
3. `input_request` is a multiple of need: `request = clip(q, 0, r_max) * need` in both
   `process_reports` and `ref_report`.
4. `ref_rollout` feeds the policy the observation of the previous record (the reset observation at
   the start), exactly as the golden replay does.
5. Lead edits of both tests to match their docstrings (`.github/FROZEN_TEST_EXEMPTION`).
Golden set regenerated. Full frozen suite: 436 passed, 35 skipped (later cards), 0 failed.
