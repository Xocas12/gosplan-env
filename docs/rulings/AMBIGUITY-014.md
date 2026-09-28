AMBIGUITY REPORT   WO-012   gosplan/experiments/mc_sanity.py
Two findings from the full G0 sweep.

A. Episode seeding. The card asks for "ONE `seed_env` shared by all three agents ... recorded in
   every manifest"; draw keys are `(seed_env, purpose, t, k, i)` with no episode index. Read
   literally, every one of the 2,000 episodes of a cell resets to the same `seed_env`, so every
   episode sees the same yield, audit and termination draws. For the deterministic
   `TruthfulMyopic` and `Padder`, the 2,000 episodes are one trajectory repeated; only `Random`
   (whose policy generator advances across episodes) explores. Options: (A) literal, as run;
   (B) per-episode seeds derived from the root and shared across agents at the same episode
   index, which keeps common random numbers but needs a derivation rule and a manifest convention.
B. The NaN scan vs the AMBIGUITY-007 placeholders. `coverage`, `audit_meas` and `penalty_arg` are
   NaN on every row by ruling, while the frozen smoke test requires `n_nonfinite == 0`.

LEAD RESOLUTION (2026-09-24):
A. Accepted for G0 as run (option A), with the limitation stated in the G0 sign-off: the
   conservation, finiteness and bound properties are deterministic per trajectory and were checked
   on 2,000 distinct `Random` trajectories per configuration across 21 configurations (126,000
   episodes), plus one trajectory per configuration for each deterministic agent. For every later
   harness that samples environment randomness (training, DP-vs-PPO, the Phase-1 gate), episode
   `e` of a run uses `seed_env = root_seed + e`, identical across agents and arms at the same `e`
   (common random numbers), with the root seed and the rule recorded in the manifest `flags`. This
   is to be written into the spec at the WO-013 freeze as the seeding convention.
B. Confirmed: the three AMBIGUITY-007 placeholder columns are excluded from the NaN count (inf is
   still counted in them) and their NaN count is reported separately. Settled with the placeholder
   columns themselves at WO-013.
