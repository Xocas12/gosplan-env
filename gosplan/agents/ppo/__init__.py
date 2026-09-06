"""PPO layer: the thin adapter over a pinned reference PPO, and its training harness.

Realises: PLAN section 6.1 (the `IPPO` row of the agent table) and the Phase-1 training path of
PLAN sections 4.5 (gate G2) and 14 (compute). Owning work orders: **WO-017** (adapter, LEAD) and
**WO-018** (training harness).

Two modules, deliberately separate:

  `adapter.py`  `PPOConfig` and `IPPO` - the `Agent`-protocol face of a *pinned reference PPO*
                implementation. gosplan does not write a PPO; PLAN section 6.1 requires a thin
                adapter around a reference (a CleanRL-style continuous PPO on the NumPy path, a
                PureJaxRL/JaxMARL-style loop on the JAX path), whose name and version the lead
                verifies at issue time and which the run manifest records (CONTRACT rule 10).
  `train.py`    the harness: a vectorised environment batch, checkpoints, periodic evaluation
                through the ledger, manifest writing, common random numbers by `seed_env`,
                wall-clock logging and the entropy anneal 0.01 -> 0.001.

Neither module is imported by `gosplan/agents/__init__.py`: the adapter pulls in the reference PPO
and its tensor framework once implemented, and the heuristic agents, the DP and the environment
must stay importable without either. Import them explicitly - `from gosplan.agents.ppo.adapter
import IPPO`.

CONTRACT rule 4 governs everything here: the reward the learner sees is exactly
`-scale * c_ik` at a PRODUCE step and `scale * (B(rho) - 1[audited] * Pen + trade_surplus)` at the
REPORT step, with `scale = reward_scale(cfg)` computed analytically from the configuration. No
shaping, no auxiliary head that feeds back into the reward, and **no running reward
normalisation** - see the adapter's class docstring for why the wrapper stack must be checked for
one.
"""

from __future__ import annotations
