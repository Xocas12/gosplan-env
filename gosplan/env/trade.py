"""Bilateral horizontal trade between enterprises - **Phase-2 sketch** (PLAN section 2.13).

Realises: PLAN section 2.13 (trade), with the offer dimension of PLAN section 2.3, the visibility
parameter of PLAN section 2.4 (`information.horizontal_visibility`), the transaction cost of PLAN
section 3 (`supply.trade_tau`) and the surplus term named in CONTRACT rule 4. Owning work order:
**WO-024** (trade matching): the lead writes the matching rule and the surplus definition into the
Phase-2 spec revision, and an implementer session then implements this module against it.

**Scope tag: P2 sketch.** PLAN section 0 defines the tag precisely, and finding F14 (freeze timing)
is why it exists: *the interface is in `spec/spec.py` v0 so the type signatures never move, but the
behaviour is deliberately under-specified and is frozen only at the Phase-2 spec revision, after
Phase 1 has run.* Concretely, for this file:

  frozen now       the name, argument names, argument order and return type of `match_trades`; the
                   `trade_offer` action dimension `(N, J)` in `[-1, 1]` and its bounds
                   (`action_spec`, PLAN section 2.3); the `trade_visibility` RNG purpose (PLAN
                   section 2.15); the configuration fields `supply.trade_tau` and
                   `information.horizontal_visibility` with their locked Phase-2 values (PLAN
                   section 4.2: `tau = 0.05`, `horizontal_visibility = 1.0` for the blat
                   phenomenon); the fact that `trade_surplus` enters the reward through exactly the
                   one term CONTRACT rule 4 already names, and through no other term
  frozen later     tie-breaking inside the greedy match, the treatment of partial fills, whether an
                   offer is per good or per counterparty pair, and every numeric constant not
                   already in the registry of PLAN section 3

Nothing downstream may assume more than the frozen half. Anything in the second column that an
implementer needs before the Phase-2 revision is an AMBIGUITY REPORT (CONTRACT rule 3), not a
judgement call.

Phase 1. `information.horizontal_visibility = 0.0`, so no counterparty is visible, no match exists,
`match_trades` returns the state unchanged with an all-zero surplus, and `trade_offer` is not in
`active_action_dims(cfg)`. The stage still runs (`stage_trade` in `gosplan/env/step.py`) so the
branch is exercised by the Phase-1 golden files rather than appearing for the first time in Phase 2.

CONTRACT rule 7 applies with full force here. Blat - informal horizontal exchange - is a held-out
phenomenon (PLAN section 4.1) whose mechanism parameters are locked now precisely so that the
mechanism cannot be tuned after seeing the result (PLAN section 4.2, finding F8). This module
defines a *market rule*: who can see whom, how offers are paired, at what price they clear, what
each unit costs to move, and what the resulting change in productive capacity is worth. It contains
no instruction to trade, no reward for trading, no term that makes trading attractive beyond the
change in physical input coverage it produces, and nothing that detects or rewards a "blat-like"
pattern. Whether horizontal exchange emerges, and how much, is a measurement (WO-030), not a rule.

Cross-module bindings. `State` and `EnterpriseAction` are the runtime dataclasses of
`gosplan/env/state.py` and `EnvConfig` that of `gosplan/config.py`, each field-for-field identical
to `spec/spec.py` (not importable as a package); `tests/unit/test_env_api.py` enforces the match.
The imports are type-only here so this skeleton imports cleanly before those modules land.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # type-only: see the cross-module bindings note in the module docstring
    from gosplan.config import EnvConfig
    from gosplan.env.state import State

Array = np.ndarray
"""Alias for every numeric array in this module (PLAN section 10). The Phase-2 JAX port (WO-029)
substitutes its own array type behind the same name."""

__all__ = ["match_trades", "trade_surplus", "visible_counterparties"]


def match_trades(state: State, offers: Array, cfg: EnvConfig, t: int) -> tuple[State, Array]:
    """Bilateral horizontal trade after DELIVER (PLAN section 2.13). **Phase-2 sketch.**

    Takes: `state`, immediately after the DELIVER stage of period `t`, so `inv_inputs` already holds
    this period's receipts; `offers` `(N, J)` in `[-1, 1]`, the `trade_offer` dimension of the joint
    action; `cfg`; and the plan period `t`, which keys the visibility draw. Returns:
    `(state, surplus)` with `surplus` `(N,)`.

    **Offer semantics** (PLAN section 2.13, verbatim). Each enterprise posts one number per good:

        offers_ij > 0    an offer to sell up to `offers_ij * X_ij` of good `j` out of its own
                         input stock - a fraction of what it holds, so an enterprise cannot offer
                         what it does not have
        offers_ij < 0    a want, to buy up to `|offers_ij| * need_ij` of good `j`, where `need_ij`
                         is its own input need for the coming step - a fraction of what it needs,
                         so a want is bounded by the technology, not by ambition
        offers_ij == 0   no position in good `j`

    The two sides are deliberately scaled by different quantities (stock for offers, need for
    wants), because that is what makes the same number mean "how much of what I have" on one side
    and "how much of what I lack" on the other.

    **Visibility subset.** An enterprise may transact only with counterparties it can see: a random
    subset of the other `N - 1` enterprises of size `horizontal_visibility * (N - 1)`, drawn with
    purpose `trade_visibility` at key `(seed_env, "trade_visibility", t, i)` through
    `gosplan.rng.draw` (CONTRACT rule 9 - no direct `numpy.random` under `gosplan/env/`). At
    `horizontal_visibility = 0` (Phase 1) the subset is empty for every `i` and the function returns
    immediately; at the locked Phase-2 value 1.0 (PLAN section 4.2) everyone sees everyone. The
    subset is redrawn per period, so visibility is an information parameter (INFO arm) and not a
    fixed network.

    **Matching.** Greedy by largest complementary pair: among all visible (seller, buyer, good)
    triples with a positive offer on one side and a want on the other, repeatedly execute the pair
    with the largest feasible quantity, decrement both sides, and continue until no complementary
    pair remains. Greedy is chosen for determinism and for being reproducible in the reference
    implementation, not for optimality - it is a market rule, not a welfare-maximising matcher, and
    the exact tie-breaking is frozen at the Phase-2 spec revision.

    **Execution price and transaction cost.** Every executed unit clears at plan-price parity: the
    quantities exchanged are equal in plan-price value, `p_{s} * Q_s = p_{b} * Q_b` using
    `state.plan_prices` (PLAN section 2.10), so trade never creates or destroys plan value by the
    price rule alone. A per-unit transaction cost `tau = cfg.supply.trade_tau` (Phase 1 inert;
    locked Phase-2 value 0.05, PLAN section 4.2) is charged on the executed quantity, which
    physically disappears - it is a real loss, and `tests/unit/test_conservation.py` (T-U1) accounts
    for it as an explicit term in the per-period, per-good identity rather than letting it vanish.

    **Surplus, computed by the environment.** For each side `i` of an executed trade:

        delta_h_i = p_{s(i)} * [ Yhat_i(X_after) - Yhat_i(X_before) ]

    where `Yhat_i` is next step's intended output at the agent's last effort - that is, the
    production function of PLAN section 2.6 evaluated through the coverage aggregator at the input
    stocks before and after the trade, holding effort at its most recent value. `X_before` and
    `X_after` are read from the true state by this function. **The surplus is never self-reported:**
    no side of a trade tells the environment what it gained, so a claim about a trade cannot be an
    action, and `trade_surplus` reaches the learner only through the single term CONTRACT rule 4
    already names in the REPORT-step reward.

    **Why wash trades are unprofitable.** A wash trade - shipping goods out and back, or trading
    with a partner purely to book a surplus - leaves `X_after = X_before` for both sides once the
    round trip closes, so `Yhat_i(X_after) = Yhat_i(X_before)` and `delta_h_i = 0` on the round
    trip, while `tau > 0` has destroyed real units of good on every executed leg. `delta_h` is the
    same function of the same true state for both sides, so there is no accounting asymmetry to
    exploit: one side cannot book a gain the other does not physically pay for. The only trades that
    pay are those that move an input from an enterprise whose coverage does not bind to one whose
    coverage does, and they pay by exactly the increase in physical output they make possible, less
    the cost of moving the goods.

    May write: `inv_inputs` (goods move, and `tau` destroys a fraction of what moves) and the
    executed volume that the ledger records as `trade_volume`. May NOT write `inv_output`, `target`,
    any `last_*` field, `cum_output` or `cum_cost`: trade reallocates inputs, it does not
    manufacture output, forgive a target or settle a claim.

    Binds: T-U1 (the traded quantities and the `tau` loss are terms of the conservation identity),
    T-B1 (under the Phase-1 configuration `TruthfulMyopic` executes no trade), and the Phase-2
    acceptance run of WO-031 through the blat metric of WO-030. Owning WO: **WO-024**.
    """
    raise NotImplementedError("PLAN section 2.13 - implemented in WO-024")


def visible_counterparties(state: State, cfg: EnvConfig, t: int) -> Array:
    """Draw each enterprise's visible counterparty set for period `t`. **Phase-2 sketch.**

    Takes: `state` (for `seed_env`); `cfg`; the plan period `t`. Returns: a boolean `(N, N)` mask
    whose `[i, j]` entry says whether `i` can transact with `j` this period, with a False diagonal.

    Each row is a random subset of the other `N - 1` enterprises of size
    `round(cfg.information.horizontal_visibility * (N - 1))`, drawn through `gosplan.rng.draw` with
    purpose `trade_visibility` at key `(seed_env, "trade_visibility", t, i)` (CONTRACT rule 9). The
    mask is symmetrised before matching - a trade needs both sides to see each other - and the
    symmetrisation rule is one of the details frozen at the Phase-2 spec revision.

    At `horizontal_visibility = 0.0` (Phase 1) the mask is all False and `match_trades` returns
    immediately; at the locked Phase-2 value 1.0 (PLAN section 4.2) it is all True off the diagonal.

    Not part of the frozen interface: this helper is internal to `gosplan/env/trade.py` and may be
    restructured at the Phase-2 spec revision. Only `match_trades` is signature-frozen (PLAN section
    0, finding F14). Owning WO: **WO-024**.
    """
    raise NotImplementedError("PLAN section 2.13 - implemented in WO-024")


def trade_surplus(
    state: State, inputs_before: Array, inputs_after: Array, effort: Array, cfg: EnvConfig
) -> Array:
    """Value the change in input stocks a trade produced. **Phase-2 sketch.**

    Takes: `state` (for `plan_prices` and `capital`); `inputs_before` `(N, J)` and `inputs_after`
    `(N, J)`, the true input stocks either side of the executed trades; `effort` `(N,)`, the agent's
    last effort, at which next step's intended output is evaluated; `cfg`. Returns: `surplus` `(N,)`
    in the same units as the bonus, before `reward_scale`.

    Formula (PLAN section 2.13, verbatim):

        delta_h_i = p_{s(i)} * [ Yhat_i(X_after) - Yhat_i(X_before) ]

    with `Yhat_i(X) = y_hat_i * coverage(X, need_i, omega_i, theta)` - the intended output of PLAN
    section 2.6 at the given effort, put through the CES coverage aggregator `coverage` in
    `gosplan/env/production.py` (WO-005) at the given stocks. Only the coverage term differs between
    the two evaluations, so `delta_h` is the plan-price value of the extra output the reallocated
    inputs make physically possible, and it is zero whenever coverage was not binding.

    Computed by the environment from the true state, never self-reported (PLAN section 2.13). It is
    the `trade_surplus` argument of `enterprise_reward`, the one term CONTRACT rule 4 admits for
    trade, and it may not be supplemented by any bonus for volume, for counterparty count or for
    reciprocity.

    Not part of the frozen interface: internal to `gosplan/env/trade.py`, restructurable at the
    Phase-2 spec revision; only `match_trades` is signature-frozen. Owning WO: **WO-024**.
    """
    raise NotImplementedError("PLAN section 2.13 - implemented in WO-024")
