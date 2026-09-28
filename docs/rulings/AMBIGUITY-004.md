AMBIGUITY REPORT   WO-009   gosplan/env/env.py
Question (one sentence):
What does `GosplanEnv.step` do when it is called after `done`, given that the frozen golden replay
keeps stepping past a termination?

What the spec says / does not say (quote):
The `GosplanEnv.step` docstring says "Calling `step` after `done` is an error rather than an
implicit auto-reset: the episode boundary must be visible to the harness that owns the seeds."
But `tests/golden/test_golden_parity.py::_replay` calls `env.step` exactly `n_steps` (30) times
without ever calling `reset`, and 20 of the 30 golden documents terminate at agent-step 26
(`done` true) and then carry three more agent-steps (period 9). Those steps come from
`ref/ref_step.py::ref_rollout`, which on termination rebuilds `ref_initial_state(cfg, seed_env,
seed_policy)`, clears the claim history and the `T_0` anchor, and keeps `t_period` counting from
the terminated period (its docstring says "the episode index appended to every subsequent draw
key"; the code instead continues `t_period`, which is what the files encode).

Options considered (A/B/…), and why the spec does not decide:
A. `step` after `done` starts a fresh episode exactly as `ref_rollout` does - fresh initial state
   under the same seeds, fresh claim history and `T_0`, `t_period` continuing - so T-B7 holds.
B. `step` after `done` raises, as the docstring says; T-B7 then fails on 20 of 30 cells until the
   LEAD changes `ref_rollout` / the golden replay and regenerates.
The docstring and the frozen test + oracle contradict each other.

Impact if the wrong option is picked:
Under B the golden suite is red on 20 cells. Under A a harness that ignores `done` silently rolls
into a new episode; every harness in this repository (MC sanity, training) checks `done` and calls
`reset`, so the continuation path is exercised only by the golden replay.

Tests blocked:
tests/golden/test_golden_parity.py (20 of 30 cells).

LEAD RESOLUTION (2026-09-24):
A. The frozen test and the frozen oracle win over the prose: a `step` after `done` continues into a
fresh episode exactly as `ref_rollout` does (`t_period` continues; everything else from
`initial_state`). Harnesses must still treat `done` as the episode boundary and call `reset`.
Recorded in spec/CHANGELOG.md; the `step` docstring is to be corrected at the WO-013 freeze.
