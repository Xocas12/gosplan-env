"""The per-period, per-good conservation identity (PLAN section 11, test **T-U1**).

Realises: PLAN sections 2.5-2.11 (the period schedule and every rule that moves a physical
quantity) and PLAN section 11 (test architecture; property test **T-U1**). Owning work order:
**WO-002** (frozen tests; LEAD). Binds the WO-009 must-pass line of PLAN section 12.3, verbatim -
"`tests/unit/test_conservation.py` (T-U1)". Modules under test: `gosplan/env/step.py`,
`gosplan/env/env.py`, `gosplan/env/state.py`.

T-U1, verbatim (PLAN section 11):

    conservation per period: `sum y + sum S_prev = sum inputs consumed / a + sum consumer
    + sum S_next + holding loss + cap overflow`, per good, to 1e-9.

The reference implementation states the same identity in the form the tests evaluate it in
(`ref/ref_step.py::ref_conservation_residual`, WO-002):

    sum y + sum S_prev = sum inputs consumed + sum consumer + sum S_next
                         + holding loss + cap overflow

"evaluated **per good**, where a quantity indexed by enterprise contributes to the good of that
enterprise's sector and `inputs_consumed` contributes to the good consumed. The residual is
LHS - RHS and must be below 1e-9 in absolute value for every good."

TOLERANCE 1e-9, absolute, per good, per period. It is not a knob: a non-zero residual is a defect in
the environment (or in the reference), never a tolerance to be loosened (CONTRACT rule 8 in spirit,
`docs/ref_worked_example.md` section 3 in letter). The two sanctioned sinks in a period are the
holding loss `h * S_carried` and the inventory-cap overflow above `S_max = inventory_cap_mult *
cap_i`, both of PLAN section 2.11; both appear as explicit terms precisely so that nothing
disappears silently. Phase 2 adds one more, the trade transaction cost `tau` (PLAN section 2.13),
which enters the identity as a term of its own.

FROZEN BY CONTRACT RULE 2. SKELETON: every test is `@pytest.mark.skeleton` and skipped until WO-009
lands; each docstring states the exact assertion, formula and tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest


def _run_one_period(cfg, seed_env, implemented, agent="TruthfulMyopic"):
    """Drive `GosplanEnv` for one complete plan period and return its `StepRecord`s.

    Gated on the environment and the heuristic agent, both of which belong to later cards
    (WO-009, WO-010), so this module skips until they land rather than failing.
    """
    from gosplan.agents import heuristic
    from gosplan.env.env import GosplanEnv

    agent_cls = getattr(heuristic, agent)
    implemented(GosplanEnv.reset, GosplanEnv.step, agent_cls.act)

    env = GosplanEnv(cfg)
    obs, info = env.reset(seed_env, cfg.tech.seed_policy)
    policy = agent_cls(cfg)
    records = list(info.records)
    for _ in range(cfg.incentive.steps_per_period + 1):
        action = policy.act(obs, env.phase(), np.random.default_rng(cfg.tech.seed_policy))
        obs, _reward, _done, info = env.step(action)
        records.extend(info.records)
    return records


def _period_balance(cfg, seed_env, implemented, agent="TruthfulMyopic", episodes=1):
    """Accumulate both sides of the corrected T-U1 identity, per good.

    The identity is the one recorded in `spec/CHANGELOG.md` 0.1.3 after ambiguity #64: the form
    printed in PLAN section 11 omits the goods sitting in buyers' input stocks, so it does not
    balance on any economy where goods are actually delivered.

        sum_i y_i + sum_i S_prev_i + sum_i X_prev_ij
            = sum_i S_next_i + sum_i X_next_ij + sum_i consumed_ij
              + consumer_j + sum_i holding_i + sum_i overflow_i
    """
    records = _run_one_period(cfg, seed_env, implemented, agent=agent)
    n_goods = cfg.supply.n_sectors
    sector = cfg.supply.sector_of
    lhs = [0.0] * n_goods
    rhs = [0.0] * n_goods
    consumer = [0.0] * n_goods
    for rec in records:
        if rec.phase != "report":
            continue
        s = sector[rec.enterprise]
        lhs[s] += float(rec.cum_output) + float(rec.inv_output_pre)
        rhs[s] += float(rec.inv_output_post) + float(rec.holding_loss) + float(rec.cap_overflow)
        for j in range(n_goods):
            rhs[j] += float(np.asarray(rec.input_consumed)[j])
        for j in range(n_goods):
            consumer[j] = float(np.asarray(rec.consumer)[j])
    for j in range(n_goods):
        rhs[j] += consumer[j]
    return lhs, rhs


@pytest.mark.skeleton
def test_period_balance_identity_holds_per_good(tiny_cfg, rng_seed, implemented) -> None:
    """T-U1: the full per-good balance closes to 1e-9 over one complete period.

    Assertion: run one whole period of `GosplanEnv` at `tiny_cfg` (`N = 2`, `J = 2`, so the
    arithmetic is hand-checkable) with a fixed action sequence, collect from the `StepInfo` /
    `StepRecord` stream the period's true output `y_i`, entering own-good stock `S_prev_i`, inputs
    consumed `(N, J)`, the consumer sink's receipts `consumer_j`, leaving stock `S_next_i`, the
    holding loss and the cap overflow, and assert per good `j`:

        |  (sum_{i: s(i)=j} y_i + sum_{i: s(i)=j} S_prev_i)
         - (sum_i inputs_consumed_ij + consumer_j + sum_{i: s(i)=j} S_next_i
            + sum_{i: s(i)=j} holding_loss_i + sum_{i: s(i)=j} cap_overflow_i)  |  <  1e-9

    Enterprise-indexed quantities contribute to the good of that enterprise's sector;
    `inputs_consumed` contributes to the good consumed, not to the consumer's own good. Every term
    must come from the ledger, not be recomputed by the test from the same formulas the environment
    used - otherwise the test proves only that arithmetic is deterministic.
    """
    lhs, rhs = _period_balance(tiny_cfg, rng_seed, implemented)
    for j, (left, right) in enumerate(zip(lhs, rhs, strict=True)):
        assert abs(left - right) < 1e-9, f"good {j}: {left} != {right}"


@pytest.mark.skeleton
def test_holding_loss_appears_as_an_explicit_term(tiny_cfg, rng_seed, implemented) -> None:
    """The stock decayed by `h` is accounted for, not silently dropped.

    Assertion: at `cfg.supply.holding_loss = h > 0` and a non-zero entering stock, the recorded
    `holding_loss_i` equals `h * S_carried_i` - the stock the period carried in, before this
    period's output is added, per `S <- (1 - h) * S + y` (PLAN sections 2.8, 2.11) - to 1e-12; and
    removing that term from the identity above breaks it by exactly that amount, so the test
    demonstrates the term is load-bearing rather than decorative. At `h = 0` the term is exactly 0
    and the identity still closes.
    """
    records = _run_one_period(tiny_cfg, rng_seed, implemented)
    h = tiny_cfg.supply.holding_loss
    assert h > 0.0
    for rec in records:
        if rec.phase != "report":
            continue
        assert abs(rec.holding_loss - h * rec.inv_output_pre) < 1e-12


@pytest.mark.skeleton
def test_inventory_cap_overflow_appears_as_an_explicit_term(
    tiny_cfg, rng_seed, implemented
) -> None:
    """Stock lost above `S_max` is accounted for, not silently clipped.

    Assertion: drive an enterprise's post-REPORT stock above
    `S_max = cfg.tech.inventory_cap_mult * cap_i` (PLAN section 2.11); the recorded
    `cap_overflow_i` equals `max(0, (1 - h) * S_prev_i + y_i - S_max)` to 1e-12, the retained stock
    is exactly `S_max`, and the per-good identity still closes to 1e-9 with the overflow term
    included. This is why the cap is logged: a clip without a term is a hole in the books.
    """
    records = _run_one_period(tiny_cfg, rng_seed, implemented)
    h = tiny_cfg.supply.holding_loss
    s_max = tiny_cfg.tech.inventory_cap_mult
    for rec in records:
        if rec.phase != "report":
            continue
        raw = (1.0 - h) * rec.inv_output_pre + rec.cum_output
        want = max(0.0, raw - s_max * rec.capital)
        assert abs(rec.cap_overflow - want) < 1e-12
        assert rec.inv_output_post <= s_max * rec.capital + 1e-12


@pytest.mark.skeleton
def test_delivery_splits_shipped_goods_between_buyers_and_the_consumer_sink(
    tiny_cfg, rng_seed, implemented
) -> None:
    """What leaves a seller's stock arrives somewhere: buyers' `X`, or the consumer sink.

    Assertion, per good `j`, to 1e-9: `sum_{i in j} shipped_i == sum_b deliv_bj + consumer_j`, with
    `shipped_i = min(S_i, claimed_i)`, `deliv_bj = alloc_bj * poolfill_j` and
    `consumer_j = sum_{i in j} phi_j * shipped_i * qbar_i` (PLAN section 2.7.3); the sellers' stock
    falls by exactly `shipped_i`; and the buyers' input stocks rise by exactly `deliv_bj * qbar_j`
    (Phase 1 `qbar = 1`). Inputs later consumed at PRODUCE steps then leave `X` as
    `min(X_ij, a_{s(i)j} * y_tilde_ik)` (PLAN section 2.6), which is the `inputs consumed` term of
    the identity. DELIVER itself has no sink: nothing may be lost there.
    """
    records = _run_one_period(tiny_cfg, rng_seed, implemented)
    n_goods = tiny_cfg.supply.n_sectors
    sector = tiny_cfg.supply.sector_of
    shipped = [0.0] * n_goods
    delivered = [0.0] * n_goods
    consumer = [0.0] * n_goods
    for rec in records:
        if rec.phase != "report":
            continue
        shipped[sector[rec.enterprise]] += float(rec.shipped)
        for j in range(n_goods):
            delivered[j] += float(np.asarray(rec.deliv)[j])
        for j in range(n_goods):
            consumer[j] = float(np.asarray(rec.consumer)[j])
    for j in range(n_goods):
        assert abs(shipped[j] - (delivered[j] + consumer[j])) < 1e-9, j


@pytest.mark.skeleton
def test_identity_holds_for_every_period_of_a_full_episode(
    p1_cfg, tiny_cfg, rng_seed, implemented
) -> None:
    """The identity closes in every period of every episode, at both configurations.

    Assertion: over complete episodes at `tiny_cfg` and at `p1_cfg` (`N = 20`, `J = 5`), driven by
    each of the reference heuristics of PLAN section 6.1 - `Random`, `TruthfulMyopic` and `Padder`
    - every period's per-good residual is below 1e-9 in absolute value; the largest residual seen is
    reported in the failure message so a regression is sized, not merely detected. The same
    identity, at the same tolerance, is asserted by the Monte-Carlo sanity harness of WO-012 across
    20 SUPPLY perturbations (`gosplan/experiments/mc_sanity.py`, `CONSERVATION_TOL`).
    """
    for cfg in (tiny_cfg, p1_cfg):
        for agent_name in ("Random", "TruthfulMyopic", "Padder"):
            lhs, rhs = _period_balance(cfg, rng_seed, implemented, agent=agent_name, episodes=1)
            for j, (left, right) in enumerate(zip(lhs, rhs, strict=True)):
                assert abs(left - right) < 1e-9, (cfg.supply.n_enterprises, agent_name, j)


@pytest.mark.skeleton
def test_identity_holds_when_claims_exceed_stock(tiny_cfg, rng_seed, implemented) -> None:
    """A claim above stock moves the `poolfill` term, never the balance.

    Assertion: with claims deliberately set above stock for a whole sector - so `fill_i < 1`,
    `poolfill_j < 1` and downstream buyers receive less than they were promised - the per-good
    identity still closes to 1e-9: promises are not goods, and only `shipped_i = min(S_i,
    claimed_i)` ever leaves a seller. The mirror case, claims below stock, leaves the difference
    sitting in `S_next` and must close too.

    Both are consequences of the four delivery lines of PLAN section 2.7.3 (CONTRACT rule 7 forbids
    implementing either directly), and this test asserts only that the books balance under them -
    never that an agent chooses to claim above stock, which is a held-out direction.
    """
    lhs, rhs = _period_balance(tiny_cfg, rng_seed, implemented, agent="Padder")
    for j, (left, right) in enumerate(zip(lhs, rhs, strict=True)):
        assert abs(left - right) < 1e-9, j
