AMBIGUITY REPORT   WO-017 / WO-018   gosplan/agents/ppo/
Questions settled by the LEAD when issuing WO-017 and WO-018.

1. Which reference PPO? PLAN section 6.1 names families ("CleanRL-style continuous PPO for the
   NumPy path, PureJaxRL/JaxMARL-style for the JAX path"), not a package; the lead pins it at issue.
   No torch is in the dependency set; `jax`/`jaxlib` are (optional extra `jax`, installed in CI by
   `uv sync --all-extras`).
2. `PPOConfig.reference_impl` / `reference_version` have no defaults, but the frozen
   `tests/unit/test_ppo_adapter.py` constructs `PPOConfig()` with no arguments.
3. The WO-018 smoke test is named by the card but authored by no card.
4. `train`'s docstring appends every training `StepRecord` to a ledger, which halves throughput
   (spec 1.1.1) and is not read by anything the card names.
5. Seeding of training and evaluation episodes (AMBIGUITY-014 fixed the convention after G0).

LEAD RESOLUTION (2026-09-24):
1. The reference is CleanRL's `ppo_continuous_action` ALGORITHM (clipped surrogate, clipped value
   loss, GAE, per-minibatch advantage normalisation, Adam, entropy bonus, global-norm gradient
   clip 0.5, 4 epochs x 4 minibatches, 64-64 tanh MLP actor and critic, orthogonal init), re-expressed
   in JAX inside `gosplan/agents/ppo/` because no maintained package ships it as a library. The
   manifest records exactly that: `reference_impl = "cleanrl ppo_continuous_action (algorithm),
   JAX re-expression in gosplan.agents.ppo"`, `reference_version = "cleanrl v1.0.0 algorithm;
   gosplan-ppo 1"`. Deviations from CleanRL demanded by the card are listed in the adapter
   docstring: tanh-squashed heads with the Jacobian correction, per-head state-independent
   log-std, heads only for active dimensions, phase masking, NO reward/observation normalisation
   of any kind (rule 4; observation normalisation off, recorded).
2. Both fields default to the pinned strings above; "filled by the lead" is satisfied by the pin.
3. The lead authors `tests/unit/test_train_smoke.py` (100 updates at N = 1, full manifest, log,
   loadable checkpoint).
4. Training environments run with `records=False`; the ledger is attached only in `evaluate`,
   whose rows 1 and 4 are what the card reads. The final evaluation's ledger is the one written
   to parquet.
5. Training env copy `b` resets episode `e` with `seed_env = train_root + b + n_envs * e`;
   evaluation episode `e` uses `seed_env = eval_root + e` with `eval_root = train_root + 10**6`, a
   disjoint block; `train_root = cfg.tech.seed_env`. Identical across arms at the same indices
   (common random numbers). Recorded in the manifest `flags`.
