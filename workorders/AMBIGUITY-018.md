AMBIGUITY REPORT   WO-017 / WO-018   gosplan/agents/ppo/
Nine choices the implementing session made to run, reported for the lead's ruling.

LEAD RESOLUTION (2026-09-24): all nine ACCEPTED as implemented, with one addition.
1. Extra run record (TrainConfig, `manifest_entry`, schedules, seed rules, wall clock) travels in
   the manifest `flags` as `key=<JSON>` strings, raised flags first (AMBIGUITY-013 C precedent).
2. One state-independent log-std per action element (CleanRL's `actor_logstd`).
3. Constant learning rate 3e-4 (the card pins it); recorded as `learning_rate_anneal: false`.
4. `ValueError` unless batch rows (n_envs x rollout_steps x N) divide into the 4 minibatches.
5. Policy seed streams: `SeedSequence(seed_policy)` for init, its second spawned child for
   rollout sampling and minibatch shuffles; no JAX PRNG key needed.
6. `evaluate_actions` infers the phase from observation fields 0-1 and raises when M < 2;
   training uses the recorded `env.phase()` masks.
7. The manifest `BOUND_BINDING` flag comes from the final evaluation ledger plus any
   `StepInfo.flags` raised in training; each periodic evaluation logs its at-bound fraction.
8. ADDITION: G2 criterion 4 is stated over TRAINING episodes, so the harness tracks, per training
   episode, whether any enterprise's target exceeded 3 x T_0 (read from `env.state`, no ledger
   rows needed) and logs the fraction per update; the evaluation-side diagnostic stays as well.
9. Harness conventions: zero-based update indices, `train_log.jsonl` reset at start, NaN as null,
   a final evaluation when the last update was not one, `final_eval_ledger.parquet`, the run
   directory keyed by `env_cfg.hash()` - experiment harnesses (WO-019/020) that train several
   seeds of one configuration pass a distinct `run_root` per seed; deterministic evaluation
   uses `squash(mean)`.
Also noted: after `reset` and after the first PRODUCE step, observation fields 0-1 coincide
((0, 0)), a consequence of the golden-pinned observation timing (AMBIGUITY-007); cumulative
output tells them apart, and uniform effort across a period is optimal in Phase 1 (PLAN section
5), so no information the policy needs is lost. Phase masking uses `env.phase()` and is exact.
