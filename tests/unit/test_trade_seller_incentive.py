"""What a sale is worth to the seller under the trade rule of spec/P2_REVISION.md R9.

Under R9 an executed sale lowers the seller's stock (`X_ij -= x`) and credits the buyer
(`X_bj += (1 - tau) x`). Nothing comes back to the seller: no good, no money, no credit. The only
trade term in the reward is each enterprise's own `trade_surplus`, the change in its own intended
output through the coverage aggregator, and coverage never falls as a stock grows. So a sale is
worth exactly 0 to a seller whose coverage of the sold good does not bind, and less than 0 to one
whose coverage does, while a buy never lowers the buyer's surplus.

The row-6 (blat) readings in `runs/G3b_record.md`, `runs/LC_record.md`, `runs/CT_record.md`,
`runs/P3_record.md` and `README.md` rest on this: row 6 is not evaluable under R9, because no
individually rewarded learner gains from selling. If a revision gives a sale a return leg, these
tests are expected to fail, and those readings have to be revisited with them.

These tests check the market rule's payoff to a seller, not the direction of the blat metric. R12's
ban on directional tests of the held-out rows applied before WO-031, which has run.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from gosplan.config import p1_default_config, p2_default_config


def _three():
    """Three enterprises, sectors (0, 0, 1); each sector needs both goods."""
    base = p1_default_config()
    supply = dataclasses.replace(
        base.supply,
        n_enterprises=3,
        n_sectors=2,
        sector_of=(0, 0, 1),
        io_matrix=((0.1, 0.2), (0.2, 0.1)),
        final_demand_share=(0.5, 0.5),
        productivity=(1.0, 1.0),
        yield_sigma=(0.05, 0.08),
        ces_alpha=(0.5, 0.5),
        trade_tau=0.05,
    )
    cfg = dataclasses.replace(base, supply=supply)
    cfg.validate()
    return cfg


def _sell_good_0(x0, seller_offer):
    """Enterprise 0 offers `seller_offer` of its good-0 stock; enterprise 2 wants 0.04 of good 0."""
    from gosplan.env.state import initial_state
    from gosplan.env.trade import execute_trades, trade_surplus

    cfg = _three()
    state = initial_state(cfg)
    offers = np.zeros((3, 2))
    offers[0, 0], offers[2, 0] = seller_offer, -1.0
    need = np.zeros((3, 2))
    need[2, 0] = 0.04
    x1, sold = execute_trades(x0, offers, need, np.ones((3, 3), dtype=bool), 0.05)
    surplus = trade_surplus(state, x0, x1, np.ones(3), cfg)
    return x1, sold, surplus


def test_a_seller_whose_coverage_does_not_bind_gains_nothing_from_a_sale() -> None:
    """R9.3-R9.5, hand-worked. At effort 1, `y_hat = 1 * 1 / 4 = 0.25`, so enterprise 0 (sector 0)
    needs (0.025, 0.05) and enterprise 2 (sector 1) needs (0.05, 0.025).

        X before   seller 0 (0.10, 0.02): good 0 covered (ratio 1), good 1 short (0.4)
                   buyer  2 (0.01, 0.10): good 0 short (0.2), good 1 covered
        trade      supply 0.5 * 0.10 = 0.05, demand 0.04, so x = 0.04 of good 0
        X after    seller 0 (0.06, 0.02): ratios still (1, 0.4), so its surplus is exactly 0
                   buyer  2 (0.048, 0.10): good 0 ratio 0.96, so its surplus is positive

    The buyer holds 0.075 of good 1 beyond its need, which is what the seller lacks, and none of it
    moves: the pair gains, and all of the gain is the buyer's.
    """
    x0 = np.array([[0.10, 0.02], [0.0, 0.0], [0.01, 0.10]])
    x1, sold, surplus = _sell_good_0(x0, 0.5)
    assert sold[0] == 0.04
    assert surplus[0] == 0.0
    assert surplus[2] > 0.0
    assert surplus.sum() > 0.0
    assert x1[0, 0] == 0.10 - 0.04
    assert x1[0, 1] == x0[0, 1]  # nothing comes back to the seller


def test_a_seller_whose_coverage_binds_loses_from_a_sale() -> None:
    """R9.3-R9.5, hand-worked, as above but with the seller holding 0.03 of good 0 (need 0.025).
    It offers 2/3 of it, so x = 0.02 and its good-0 stock falls to 0.01 (ratio 0.4): its own
    surplus is negative while the buyer's is positive."""
    x0 = np.array([[0.03, 0.02], [0.0, 0.0], [0.01, 0.10]])
    x1, sold, surplus = _sell_good_0(x0, 2.0 / 3.0)
    assert np.isclose(sold[0], 0.02)
    assert surplus[0] < 0.0
    assert surplus[2] > 0.0
    assert x1[0, 1] == x0[0, 1]


def test_selling_never_raises_and_buying_never_lowers_the_own_surplus() -> None:
    """R9 on random states of `p2_default_config()` (N = 20, J = 5, tau = 0.05, visibility 1.0).

    One enterprise's offer for one good is set to a sell, to 0 or to a buy, with every other offer
    fixed. Wherever the sale executes, the seller's own surplus is never above its surplus at 0 and
    no other holding of the seller rises; a buy never leaves the surplus below its value at 0. The
    draw is checked to contain executed sales, and sales whose loss is strict, so the check cannot
    pass vacuously.
    """
    from gosplan.env.state import initial_state
    from gosplan.env.trade import trade_stage

    cfg = p2_default_config()
    base = initial_state(cfg)
    n, n_goods = cfg.supply.n_enterprises, cfg.supply.n_sectors
    rng = np.random.default_rng(107)
    executed = strict_losses = 0
    for t in range(300):
        state = dataclasses.replace(
            base,
            inv_inputs=np.asarray(base.inv_inputs) * rng.uniform(0.0, 3.0, size=(n, n_goods)),
            capital=np.asarray(base.capital) * rng.uniform(0.5, 1.5, size=n),
            target=np.asarray(base.target) * rng.uniform(0.5, 1.5, size=n),
        )
        effort = rng.uniform(0.0, 1.0, size=n)
        offers = rng.uniform(-1.0, 1.0, size=(n, n_goods))
        i, j = int(rng.integers(n)), int(rng.integers(n_goods))
        sell, buy = rng.uniform(0.01, 1.0), -rng.uniform(0.01, 1.0)
        outcome = {}
        for name, own in (("sell", sell), ("none", 0.0), ("buy", buy)):
            posted = offers.copy()
            posted[i, j] = own
            after, surplus, _sold = trade_stage(state, posted, cfg, t, effort=effort)
            outcome[name] = (surplus[i], np.asarray(after.inv_inputs)[i])
        (s_sell, x_sell), (s_none, x_none), (s_buy, _x_buy) = (
            outcome["sell"],
            outcome["none"],
            outcome["buy"],
        )
        if x_sell[j] < x_none[j]:  # the sale executed
            executed += 1
            assert s_sell <= s_none
            strict_losses += int(s_sell < s_none)
            others = np.arange(n_goods) != j
            assert np.all(x_sell[others] <= x_none[others])
        assert s_buy >= s_none
    assert executed >= 30
    assert strict_losses >= 3
