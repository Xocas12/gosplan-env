AMBIGUITY REPORT   WO-009   gosplan/env/state.py, step.py, env.py
Question (one sentence):
After `GosplanEnv.step` returns, do `state.t_period`, `state.k_step` and `state.phase` describe the
step just executed or the next one?

What the spec says / does not say (quote):
`advance` ("the state after the executed stages, with `k_step`, `phase` and `t_period` advanced to
the machine's next position"), `advance_phase`, `GosplanEnv.phase()` and
`tests/unit/test_env_api.py::test_an_agent_acts_m_plus_one_times_per_period` (which counts `step`
calls until `env.state.t_period` changes and expects exactly `M + 1`) all require the NEXT
position. `ref/ref_step.py::_run_period` rendered the golden `state_digest` with the counters of
the step just EXECUTED, and `tests/golden/test_golden_parity.py` compares that digest with
`env.state` after each step. The golden observations (field 1 = `k / M` of the executed step) also
describe the executed step.

Options considered (A/B/...), and why the spec does not decide:
A. Counters advance eagerly (docstrings + test_env_api); observations are built at the executed
   position (golden obs unchanged); the reference renders its digest at the next position.
B. Counters lag (golden digests unchanged); test_env_api fails and `phase()` needs hidden state.
Two frozen tests contradict each other on the same attribute.

Impact if the wrong option is picked:
Either the golden digests or test_env_api are red.

Tests blocked:
tests/golden/test_golden_parity.py::test_state_digest_matches_reference (30 cells) or
tests/unit/test_env_api.py::test_an_agent_acts_m_plus_one_times_per_period.

LEAD RESOLUTION (2026-09-24):
A. `ref/ref_step.py` gains `_digest_at_next_position`, which renders the digest with only the
three counters moved to the next agent-step; nothing else in the rendering changes. Golden set
regenerated. `GosplanEnv.step` builds the observation from the state with the executed counters
(taken from `StepInfo`), so observations match the reference unchanged.

Also resolved while closing WO-009 (all golden-parity or frozen-test driven):
- `initial_state` opens `inv_inputs` at `a_{s(i)j} * T_0_i`, as ambiguity #62 / CHANGELOG 0.1.4
  already decided; the `initial_state` docstring still says zeros and is to be corrected at WO-013.
- `GosplanEnv.step` hands out a fresh `State` per step (the golden replay keeps references).
- Bit-for-bit parity: `prices._solve_cost_plus` iterates the fixed point exactly as
  `ref_initial_prices` (the `initial_prices` docstring allows iteration); `planner.allocate`
  accumulates `avail` and forms `alloc` in the reference's operation order; `ref_produce` computes
  `kappa * (e * e)` (Python `float ** 2` goes through `pow` and is not always correctly rounded).
- `StepInfo.welfare` is filled at the REPORT step, where the golden files carry it.
- `StepRecord`: the REPORT row's `target` and `need` are those the period was judged against (before
  the ratchet); the period's DELIVER quantities (`alloc`, `deliv`, `fill`, `shipped`) are carried on
  every row of the period; `coverage`, `audit_meas` and `penalty_arg` are NaN because their owning
  functions do not return them (a spec question for WO-013).
