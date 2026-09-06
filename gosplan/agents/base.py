"""Agent interface: the protocol every policy in the system implements (PLAN section 6.1).

Realises: PLAN sections 6.1 (the agent table), 2.3 (action dimensions), 2.4 (observation) and 2.5
(phase within a plan period), under CONTRACT rules 6 (welfare blindness) and 9 (RNG). Owning work
order: **WO-010** (heuristic agents; this module is its first deliverable), with `IPPO` supplied by
**WO-017** and `DPGreedy` by **WO-010**/**WO-014**.

Relation to the frozen interface. `spec/spec.py` is the frozen interface (CONTRACT rule 1) but is
not an importable package, so this module re-declares `Agent`, `Array` and `Phase` with signatures
identical to the spec's, argument names included. The runtime declarations here and the frozen
declarations in `spec/spec.py` must stay identical symbol-for-symbol; a unit test asserts the
correspondence, and any divergence is a spec change requiring a `spec/CHANGELOG.md` entry, never an
edit made in passing.

What an agent may see, and what it may never see. `act` receives the observation array of PLAN
section 2.4, the current `phase`, and a policy-stream generator. It receives **no** `State`, **no**
`StepInfo`, **no** `PlannerView` and **no** `EnvConfig`-derived true quantity computed from any of
those. This is CONTRACT rule 6 (welfare blindness): `welfare_true` and `val_measured` are logged and
never appear in an observation, a reward or any agent input; `StepInfo` is written by the
environment for `gosplan/metrics/ledger.py` and for lead-run experiments only, and "any agent
reading `StepInfo`" is on the WO-010 forbidden list. An agent that needs a configuration constant
(sector productivity, the initial target, the action bounds) holds an `EnvConfig` given to it at
construction time and reads the *configuration*, which is public, never the *state*, which is not.

Policy randomness. Every stochastic choice draws from the `rng` argument, which belongs to the
`seed_policy` stream and is kept strictly separate from `seed_env` (PLAN section 2.15, CONTRACT
rule 9). No agent may call `numpy.random` at module level or hold a module-level global generator,
and no agent calls `gosplan.rng.draw`: that key-based function is the environment's randomness, and
sharing it would break the common-random-numbers property of PLAN section 4.3.

Binds: T-B1 (`tests/behavioural/test_no_hardcoded_pathology.py`), T-B2 (`test_fixed_point.py`),
T-B3 (`test_shortage_propagation.py`) and T-B5 (`test_welfare_blindness.py`, which checks the
adapter's forward signature takes `obs` only).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np

if TYPE_CHECKING:  # runtime home of the action record: gosplan/env/state.py (WO-009, PLAN sec. 8)
    from gosplan.env.state import EnterpriseAction

Array = np.ndarray
"""Alias for every numeric array in the agent layer (PLAN section 10). The Phase-2 JAX port
(WO-029) substitutes its own array type behind the same name, so no agent may rely on a
numpy-only method in a signature. Identical to `spec/spec.py`'s alias by construction."""

Phase = Literal["produce", "report"]
"""Agent-step phase within a plan period (PLAN section 2.5). Each period is
`cfg.incentive.steps_per_period` (`M`) PRODUCE steps followed by exactly one REPORT step, so an
agent acts `M + 1` times per period. The environment ignores action dimensions that are not
relevant to the current phase rather than rejecting them (PLAN section 2.3), so an agent may fill
every dimension every step; masking is an efficiency and clarity choice, not a correctness one.
Mirrors `spec/spec.py`'s alias exactly; wherever else in the package the same alias is declared
(`gosplan/metrics/ledger.py` declares one for its `StepRecord`) the declarations must be identical,
and the unit test that compares the runtime package against the frozen spec is what keeps them
so."""


class Agent(Protocol):
    """The interface every policy implements - heuristic, DP-derived, learned or LLM-driven.

    Implementations (PLAN section 6.1): `Random`, `TruthfulMyopic`, `Padder` (sanity only) and
    `DPGreedy` in Phase 1 (`gosplan/agents/heuristic.py`, WO-010, WO-014); `Berliner`, `Weitzman`,
    `Kornai` and the LLM ministry study of PLAN section 7.4 in Phase 2 (WO-030, WO-026); `IPPO` as
    a thin adapter over a pinned reference PPO (`gosplan/agents/ppo/adapter.py`, WO-017).

    The protocol is structural: an implementation matches it by defining `act` and `reset` with
    these signatures and does not inherit from it. Argument names are part of the interface - the
    training harness and the experiment scripts call these methods by keyword in places.

    Two hard constraints, both checked by frozen tests. CONTRACT rule 6: `act` takes the
    observation and nothing else - no `State`, no `StepInfo`, no `PlannerView` - and the PPO
    adapter's forward pass takes `obs` only (test T-B5). CONTRACT rule 9: policy randomness comes
    from the `rng` argument on the `seed_policy` stream, never from the environment's key-based
    `draw` and never from a global generator.
    """

    def act(self, obs: Array, phase: Phase, rng: np.random.Generator) -> EnterpriseAction:
        """Choose one joint action for all `N` enterprises.

        Takes: `obs` `(N, d)` with `d = 12 + 3J` in Phase 1, laid out exactly as `obs_spec(cfg)`
        (PLAN section 2.4; the index table is reproduced in `gosplan/agents/heuristic.py`, which is
        the module that reads individual fields); `phase`, so the agent can fill only the
        dimensions this step reads; `rng`, a `numpy.random.Generator` from the `seed_policy`
        stream. Returns: an `EnterpriseAction` whose active dimensions lie inside the bounds of
        `action_spec(cfg)` - `effort`, `quality`, `invest` in [0, 1]; `report_ratio` in
        [0, `cfg.tech.report_max_ratio`]; `input_request` in [0, `cfg.tech.request_max_multiple`]
        expressed as a multiple of `need_ij`; `trade_offer` in [-1, 1].

        Dimensions the configuration does not activate (`active_action_dims(cfg)`) may be filled
        with zeros: the environment ignores them. Values outside the bounds are the agent's bug -
        the environment clips the report to `rho_max` and logs whether the bound bound (CONTRACT
        rule 8), and a bound is never widened to accommodate a policy.

        Reference behaviours are stated on each class in `gosplan/agents/heuristic.py`, transcribed
        from the PLAN section 6.1 table. Binds: T-B1 (no hard-coded pathology), T-B2 (fixed point),
        T-B3 (shortage propagation), T-B5 (welfare blindness). Owning WO: **WO-010**.
        """
        raise NotImplementedError("PLAN section 6.1 - implemented in WO-010")

    def reset(self) -> None:
        """Clear any per-episode internal state.

        Takes: nothing. Returns: `None`. Called once per episode, before the first `act` of that
        episode and after `GosplanEnv.reset`. A stateless agent (`Random`, `TruthfulMyopic`,
        `Padder`) implements it as a no-op; a stateful one (`Berliner` tracking its own banked
        stock estimate, `IPPO` carrying a recurrent state if the reference implementation has one)
        clears everything episode-scoped here. It must not touch anything run-scoped: the
        configuration, the loaded `DPSolution`, or the policy parameters.

        Owning WO: **WO-010**.
        """
        raise NotImplementedError("PLAN section 6.1 - implemented in WO-010")


__all__ = ["Agent", "Array", "Phase"]
