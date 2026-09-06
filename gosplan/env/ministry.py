"""The ministry layer between enterprises and the planner - **Phase-2 sketch** (PLAN section 2.14).

Realises: PLAN section 2.14 (ministry), the `MinistryView` record of PLAN section 10, the INFO
parameters `information.ministry_passthrough` and `information.n_ministries` of PLAN section 3, and
the adapter seam of PLAN section 7.4 (LLM ministry study). Owning work orders: **WO-025**
(rule-based ministry) and **WO-026** (LLM adapter).

**Scope tag: P2 sketch.** Per PLAN section 0 and finding F14 (freeze timing): the interface is in
`spec/spec.py` v0 so the type signatures never move, but the behaviour is deliberately
under-specified and is frozen only at the Phase-2 spec revision, after Phase 1 has run. For this
file:

  frozen now       the `MinistryView` field set, names and order; the name, argument names and
                   return type of `ministry_forward`; `information.ministry_passthrough` (range
                   [0.5, 1.0], Phase-1 value 1.0) and `information.n_ministries` (Phase-1 value 1)
                   as the INFO-arm parameters that govern the layer
  frozen later     the smoothing coefficient `kappa_m`, which is **not yet a configuration field**;
                   it is fixed at the Phase-2 spec revision, at which point it joins
                   `InformationConfig` with a `spec/CHANGELOG.md` entry (CONTRACT rules 1 and 11).
                   Also: how enterprises are partitioned across ministries beyond "a partition",
                   and how `prev_forward` is initialised in an episode's first period

Phase 1. `n_ministries = 1` and `ministry_passthrough = 1.0`, so the layer is exactly transparent:
every claim reaches the planner unchanged and this module is inert. `ministry_passthrough` below is
that identity, written as a function so the transparent case is a code path the Phase-1 golden files
exercise rather than a branch that first appears in Phase 2.

Where the layer sits. Enterprises report at the REPORT stage (PLAN section 2.5); the ministry
receives its own enterprises' claims and forwards possibly smoothed and padded numbers upward; the
planner then sees the forwarded numbers through the three filters of PLAN section 2.7.5
(aggregation, `report_lag`, `channel_noise`) when `make_planner_view` builds its view. The ministry
is therefore an information institution, which is why its parameters are in the INFO arm and why it
is one of the mechanisms the C_OGAS contrast of PLAN section 4.3 moves.

What it may see. A `MinistryView` carries claims, targets and its own last forwarded values for its
own enterprises, and nothing else: no `y`, no `S`, no `X`, no welfare, no audit selection. It is the
same discipline CONTRACT rule 5 imposes on the planner, applied one level down - an intermediary
that could see the truth would make the whole information architecture of PLAN section 2.4
decorative. `make_ministry_views` is the single `State -> ministry` boundary, exactly as
`make_planner_view` is the single `State -> planner` one.

CONTRACT rule 7 and this file. Rule 7 forbids any transition rule or reward term from implementing
an *enterprise* pathology - bunching, padding, storming, hoarding, shaving or trade - because those
are the phenomena the project claims to measure as emergent. The ministry is not an enterprise: it
is part of the institutional environment, like the ratchet and the audit rate, and its forwarding
rule is a stated, parameterised planner-side rule whose coefficients are treatment variables. The
distinction is load-bearing. Nothing here rewards an enterprise for anything, nothing here reads an
enterprise's true output, and the enterprise-level phenomena of PLAN section 4.1 are measured on the
ledger (WO-030), never implemented. The LLM study of PLAN section 7.4 exists precisely because the
interesting question about a ministry is behavioural - whether a model's forwarding *tracks the
payoff arm* (reasoning) or stays at the historical pattern regardless (retrieval) - and that
question is empty if the answer is written into the rule.

Cross-module bindings. `State` is the runtime dataclass of `gosplan/env/state.py` and `EnvConfig`
that of `gosplan/config.py`, each field-for-field identical to `spec/spec.py` (not importable as a
package); `tests/unit/test_env_api.py` enforces the match. `MinistryView` is declared here, in the
module of the layout row that owns the ministry (PLAN section 8), and must stay field-for-field
identical to `spec/spec.py`'s declaration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:  # type-only: see the cross-module bindings note in the module docstring
    from gosplan.config import EnvConfig
    from gosplan.env.state import State

Array = np.ndarray
"""Alias for every numeric array in this module (PLAN section 10). The Phase-2 JAX port (WO-029)
substitutes its own array type behind the same name."""

__all__ = [
    "MinistryPolicy",
    "MinistryView",
    "make_ministry_views",
    "ministry_forward",
    "ministry_passthrough",
]


@dataclass(frozen=True)
class MinistryView:
    """What one ministry sees about its own enterprises (PLAN section 2.14; **Phase-2 sketch**).

    The ministry sits between enterprises and the planner: it receives claims and forwards possibly
    smoothed and padded numbers upward. The rule-based version is `ministry_forward`; the LLM study
    of PLAN section 7.4 replaces that function with a model call that receives a text rendering of
    this same record and returns per-enterprise forwarded values plus a free-text justification.

    Contains no true quantity - no `y`, no `S`, no `X`, no welfare, no audit selection - so an
    intermediary cannot recover what the planner is not allowed to know either. Built only by
    `make_ministry_views`.

    Interface only. Behaviour is deliberately under-specified and is frozen at the Phase-2 spec
    revision (PLAN section 0, finding F14). This declaration must stay field-for-field identical to
    `spec/spec.py`; `tests/unit/test_env_api.py` enforces it. Owning WO: **WO-025** (rule-based),
    **WO-026** (LLM).
    """

    ministry_id: int  # index of this ministry in [0, n_ministries)
    enterprise_ids: Array  # (n_i,) indices of the enterprises this ministry covers
    claims: Array  # (n_i,) R_i as received from its own enterprises
    targets: Array  # (n_i,) T_i of those enterprises
    prev_forward: Array  # (n_i,) what this ministry forwarded last period (the smoothing anchor)
    passthrough: float  # pi = cfg.information.ministry_passthrough; 1.0 is transparent
    t_period: int  # plan period index, for logging and for the LLM prompt


def make_ministry_views(state: State, cfg: EnvConfig) -> tuple[MinistryView, ...]:
    """Build one `MinistryView` per ministry - the only `State -> ministry` function.

    Takes: the true `state`, after the REPORT stage of period `t` so `last_report` holds this
    period's claims, and `cfg`. Returns: a tuple of `cfg.information.n_ministries` views, in
    ministry-index order, whose `enterprise_ids` partition `range(cfg.supply.n_enterprises)` - every
    enterprise belongs to exactly one ministry, and the partition does not change within a run.

    Each view carries only `claims` (`state.last_report` restricted to that ministry's enterprises),
    `targets`, the ministry's own `prev_forward`, `passthrough` from
    `cfg.information.ministry_passthrough` and `t_period`. No true quantity crosses this boundary,
    for the reason given in the module docstring: an intermediary that could see `S` or `y` would
    make the information architecture of PLAN section 2.4 decorative.

    At `n_ministries = 1` (Phase 1) this returns a single view covering every enterprise, and with
    `passthrough = 1.0` the layer is the identity.

    Not part of the frozen interface: the partitioning rule beyond "a partition" is frozen at the
    Phase-2 spec revision. Owning WO: **WO-025**.
    """
    raise NotImplementedError("PLAN section 2.14 - implemented in WO-025")


def ministry_forward(view: MinistryView, cfg: EnvConfig) -> Array:
    """Forward one ministry's enterprises' claims upward (PLAN section 2.14). **Phase-2 sketch.**

    Takes: a `MinistryView` and `cfg`. Returns: the forwarded values `(n_i,)`, in the order of
    `view.enterprise_ids`, which the planner then sees in place of the raw claims.

    Rule (PLAN section 2.14, verbatim):

        R_tilde_i = pi * R_i + (1 - pi) * [ Rbar_i_prev + kappa_m * max(0, T_i - R_i) ]

    with `pi = cfg.information.ministry_passthrough` (1 = transparent), `R_i = view.claims`,
    `T_i = view.targets` and `Rbar_i_prev = view.prev_forward`. Read as two effects:

        smoothing   the `(1 - pi) * Rbar_i_prev` term pulls the forwarded number toward what this
                    ministry forwarded last period, so a series that reaches the planner through a
                    ministry is smoother than the underlying claims - which is what makes the layer
                    an information mechanism (INFO arm) and what the republic-level anchor of PLAN
                    section 7.3 needs from Phase 2
        padding     the `kappa_m * max(0, T_i - R_i)` term closes part of a reported shortfall, and
                    only a shortfall: it is exactly zero when `R_i >= T_i`, so the rule is
                    one-sided by construction

    `kappa_m` is **not yet a configuration field**. It is fixed at the Phase-2 spec revision, at
    which point it joins `InformationConfig` with a `spec/CHANGELOG.md` entry recording version,
    reason and affected work orders (CONTRACT rules 1 and 11). Until then no caller may assume a
    value for it; needing one before the revision is an AMBIGUITY REPORT (CONTRACT rule 3).

    At `pi = 1.0` (Phase 1) the rule reduces exactly to the identity `R_tilde_i = R_i` and neither
    `prev_forward` nor `kappa_m` is read - see `ministry_passthrough`, which is that branch. The
    identity branch must exist explicitly and be tested, exactly as the aggregation, lag and
    channel-noise branches of PLAN section 2.7.5 are (WO-006 notes).

    Boundaries. This function reads a `MinistryView` and nothing else - never a `State`, never a
    reward, never an observation. It writes nothing: the forwarded array is returned and the caller
    decides what the planner sees. Whatever it returns is a *claim as received by the planner* and
    is therefore subject to the same audit and penalty machinery as any other claim: the ministry
    changes what the planner is told, not what physically exists, and `deliver` (PLAN section 2.7.3)
    still ships from real stock.

    Binds: `tests/unit/test_ministry.py` (WO-025) - `pi = 1` is the identity on claims; the padding
    term is exactly zero when `R_i >= T_i`; the forwarded value lies between the claim and the
    smoothing anchor for `pi` in `[0.5, 1]`; ministry-level aggregates are smoother than the
    underlying claims. Owning WO: **WO-025** (rule-based), **WO-026** (LLM adapter).
    """
    raise NotImplementedError("PLAN section 2.14 - implemented in WO-025")


def ministry_passthrough(view: MinistryView) -> Array:
    """Forward every claim unchanged: the transparent ministry, `pi = 1`. **Phase-2 sketch.**

    Takes: a `MinistryView`. Returns: `view.claims` `(n_i,)`, unmodified - the `pi = 1` branch of
    `ministry_forward`, in which the smoothing anchor and the padding term drop out of the rule
    entirely.

    Two uses, and they are why this identity is a named function rather than an inline branch:

      Phase 1 / inert layer   `information.ministry_passthrough = 1.0` and `n_ministries = 1`, so
                              this is the whole ministry layer and the planner sees raw claims. The
                              Phase-1 golden files (T-B7) then cover the code path that Phase 2
                              generalises, instead of Phase 2 introducing an untested one.
      LLM fallback action     PLAN section 7.4 requires the LLM adapter to parse strict JSON with
                              exactly one retry and a **documented fallback action, `pi = 1`**. This
                              function is that fallback: when a model's response cannot be parsed
                              after the retry, the adapter calls it, logs the failure with the
                              prompt and the completion (CONTRACT rule 10), and the episode
                              continues with a transparent ministry rather than with an invented
                              number.

    Not part of the frozen interface: internal to `gosplan/env/ministry.py`. Owning WO: **WO-025**;
    the fallback path is **WO-026**.
    """
    raise NotImplementedError("PLAN section 2.14 - implemented in WO-025")


class MinistryPolicy(Protocol):
    """The adapter seam: anything that can forward a `MinistryView` (PLAN sections 2.14, 7.4).

    Two implementations are planned and they share this one surface, which is the whole point of
    naming it: the rule-based ministry of WO-025 wraps `ministry_forward`, and the LLM ministry of
    WO-026 wraps a model call. The environment holds a `MinistryPolicy` and does not know which it
    has, so the LLM study is a swap of one object, not a fork of the period schedule.

    **Documented here, not implemented here.** `gosplan/agents/llm_ministry.py` (PLAN section 8) is
    where the model call lives; this module holds the rule-based side and the seam. What the LLM
    adapter must do, per PLAN section 7.4, so that the seam is specified even though the adapter is
    not written:

      rendering      the same `MinistryView`, rendered as text; no field the rule-based ministry
                     cannot see is added to the prompt
      response       strict JSON: per-enterprise forwarded values plus a free-text justification,
                     parsed strictly, with exactly one retry
      fallback       on a second parse failure, the documented fallback action `pi = 1`, i.e.
                     `ministry_passthrough`; the failure, the prompt and the completion are logged
      logging        pinned model ids and versions, and every prompt and completion, recorded in the
                     run manifest (CONTRACT rule 10)
      design         2 framings (neutral / historical vocabulary) x 3 payoff arms (baseline;
                     padding-dominated, `a = 1` with `pen` large so forwarding padded numbers is
                     strictly worse for the ministry; overfulfilment-optimal, `lambda = 0` with `s`
                     large so smoothing away overfulfilment is strictly worse) x at least 2 models
                     x 20 episodes. The discriminating measure is whether ministry behaviour tracks
                     the payoff arm (reasoning) or stays at the historical pattern regardless
                     (retrieval); the framing contrast is secondary
      check          after each episode the model is asked, in a fresh context, to name the closest
                     real-world analogue of the game it played, and results are stratified by
                     whether it named Soviet planning (finding F13, contamination)

    The study is deliberately separate from the factorial of PLAN section 4.3 and its results never
    enter a contrast table.

    Owning WO: **WO-025** (rule-based implementation), **WO-026** (LLM adapter).
    """

    def forward(self, view: MinistryView, cfg: EnvConfig) -> Array:
        """Forward one ministry's claims upward.

        Takes: `view`, the ministry's own `MinistryView`, and `cfg`. Returns: the forwarded values
        `(n_i,)` in the order of `view.enterprise_ids`.

        Implementations receive a `MinistryView` and nothing else - no `State`, no reward, no
        observation - which is CONTRACT rule 5's discipline applied one level below the planner. The
        rule-based implementation delegates to `ministry_forward`; the LLM implementation issues the
        model call described in the class docstring and falls back to `ministry_passthrough` when
        the response cannot be parsed after one retry.

        Owning WO: **WO-025** (rule-based), **WO-026** (LLM).
        """
        raise NotImplementedError("PLAN section 2.14 - implemented in WO-025")
