"""Agent layer: the `Agent` protocol, the heuristic policies and the single-enterprise DP.

Realises: PLAN section 6.1 (agents and baselines) and PLAN section 5 (the analytical layer whose
policy `DPGreedy` replays), read under CONTRACT rules 6 (welfare blindness - no agent reads
`StepInfo`, `State` or `PlannerView`) and 9 (policy randomness comes from the `seed_policy` stream).
Owning work orders: **WO-010** (`base`, `heuristic`), **WO-014** (`dp`), **WO-017** (`ppo.adapter`),
**WO-018** (`ppo.train`), **WO-026**/**WO-035** (`llm_ministry`).

The package re-exports the Phase-1 surface that every experiment script needs:

    Agent, Array, Phase                     gosplan.agents.base
    Random, TruthfulMyopic, Padder,         gosplan.agents.heuristic
    DPGreedy, Berliner, Weitzman, Kornai
    DPGrid, DPSolution, RegimeLabel,        gosplan.agents.dp
    solve_single_enterprise, classify_regime

Two members are deliberately **not** re-exported here and must be imported from their modules:

    gosplan.agents.ppo.adapter   `IPPO`, `PPOConfig` - pulls in the pinned reference PPO and its
                                 tensor framework, which the heuristics, the DP and the environment
                                 must never require in order to import.
    gosplan.agents.llm_ministry  the Phase-2 study of PLAN section 7.4, which is not part of the
                                 factorial and carries its own model-client dependency.

Ordering note for readers of the plan: the DP is solved **before** any multi-agent learning (PLAN
section 5), and its solution is the ground truth for gate G2 criterion 1. Nothing in this package
may make the DP depend on the learner.
"""

from __future__ import annotations

from gosplan.agents.base import Agent, Array, Phase
from gosplan.agents.dp import (
    DPGrid,
    DPSolution,
    RegimeLabel,
    classify_regime,
    solve_single_enterprise,
)
from gosplan.agents.heuristic import (
    Berliner,
    DPGreedy,
    Kornai,
    Padder,
    Random,
    TruthfulMyopic,
    Weitzman,
)

__all__ = [
    "Agent",
    "Array",
    "Berliner",
    "DPGreedy",
    "DPGrid",
    "DPSolution",
    "Kornai",
    "Padder",
    "Phase",
    "Random",
    "RegimeLabel",
    "TruthfulMyopic",
    "Weitzman",
    "classify_regime",
    "solve_single_enterprise",
]
