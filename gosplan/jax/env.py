"""JAX port of the environment step (WO-029, LEAD; PLAN section 12.4).

A translation of `gosplan/env/` into `jax.numpy` at float64, stage for stage and formula for formula,
with the NumPy path as the reference. The period schedule, the stage slices per agent-step and the
episode bookkeeping mirror `gosplan/env/step.py` and `gosplan/env/env.py`; every numeric kernel below
names the NumPy function it translates.

Randomness (CONTRACT rule 9). No `jax.random` call appears here. Every environment draw is made on
the host by `gosplan.rng.draw` at the same key `(seed_env, purpose, *indices)` the NumPy path uses,
and handed to the device as an array. The one draw whose parameter depends on state - the targeted
audit probability - uses `gosplan.rng.uniforms`, the exact uniforms behind the Bernoulli draw, so
`audited = u < p` is evaluated on the device and agrees with NumPy by construction (PLAN section
2.15: agreement by keyed draws, not by call order).

Branches. Configuration branches (theta = inf, notch width 0, rho_cap = inf, sigma_c = 1, timing,
aggregation, ministry passthrough) are static Python conditions on the frozen configuration, as in
NumPy; data-dependent branches (the setup cost `F * 1[e > 0]`, coverage with zero ratios, `fill = 1`
at `claimed = 0`, the greedy trade matcher's stopping rule) are `jnp.where` and `lax.while_loop`.
No discontinuity is smoothed. Capital, IRS and investment are rejected by `EnvConfig.validate` in
Phase 2 (spec/P2_REVISION.md R1), so the port carries the `pending_invest` buffer but no
depreciation or IRS branch; porting them is part of enabling them.
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np

from gosplan.config import EnvConfig
from gosplan.env.state import EnterpriseAction

jax.config.update("jax_enable_x64", True)

_RHO_REF = 1.1


# ---------- static configuration helpers (host-side numpy, constant per configuration) ----------


def _static(cfg: EnvConfig) -> dict[str, np.ndarray]:
    sup = cfg.supply
    sector = np.asarray(sup.sector_of, dtype=int)
    n, j = sup.n_enterprises, sup.n_sectors
    a_rows = np.asarray(sup.io_matrix, dtype=float)[sector]
    totals = a_rows.sum(axis=1, keepdims=True)
    omega = np.divide(a_rows, totals, out=np.zeros_like(a_rows), where=totals > 0.0)
    onehot = np.zeros((n, j))
    onehot[np.arange(n), sector] = 1.0
    t_0 = cfg.tech.initial_target_frac * np.asarray(sup.productivity, dtype=float)[sector]
    return {
        "sector": sector,
        "a_rows": a_rows,
        "omega": omega,
        "onehot": onehot,
        "productivity": np.asarray(sup.productivity, dtype=float)[sector],
        "phi": np.asarray(sup.final_demand_share, dtype=float),
        "t_0": t_0,
        "counts": onehot.sum(axis=0),
    }


def _bincount(onehot, weights):
    """`np.bincount(sector, weights, minlength=J)` as a one-hot contraction."""
    return onehot.T @ weights


# ---------- kernels ----------


def coverage(x, need, weights, theta: float):
    """`gosplan.env.production.coverage`."""
    needed = need > 0.0
    ratio = jnp.minimum(1.0, jnp.where(needed, x / jnp.where(needed, need, 1.0), 1.0))
    requires = needed.any(axis=1)
    if math.isinf(theta):
        h = jnp.min(jnp.where(needed, ratio, jnp.inf), axis=1)
        return jnp.where(requires, h, 1.0)
    zero_ratio = (needed & (ratio == 0.0)).any(axis=1)
    safe = jnp.where(needed & (ratio > 0.0), ratio, 1.0)
    total = jnp.sum(weights * safe ** (-theta), axis=1)
    live = requires & ~zero_ratio
    h = jnp.where(live, total ** (-1.0 / theta), 1.0)
    return jnp.where(zero_ratio, 0.0, h)


def period_quality(quality_acc, cfg: EnvConfig):
    """`gosplan.env.production.period_quality`."""
    if not cfg.supply.quality_matters:
        return jnp.ones_like(quality_acc)
    return quality_acc / cfg.incentive.steps_per_period


def bonus(rho, cfg: EnvConfig):
    """`gosplan.env.reward.bonus`."""
    inc = cfg.incentive
    w = inc.notch_width
    x = rho - 1.0
    lam = (x >= 0.0).astype(float) if w == 0 else 1.0 / (1.0 + jnp.exp(-x / w))
    if math.isinf(inc.overfulfilment_cap) and w > 0:
        over = w * jnp.logaddexp(0.0, x / w)
    elif math.isinf(inc.overfulfilment_cap):
        over = jnp.maximum(x, 0.0)
    else:
        over = jnp.clip(x, 0.0, inc.overfulfilment_cap - 1.0)
    return inc.notch_height * lam + inc.overfulfilment_slope * over


def reward_scale(cfg: EnvConfig) -> float:
    return 1.0 / float(bonus(jnp.array([_RHO_REF]), cfg)[0])


def welfare_true(consumer, cfg: EnvConfig):
    """`gosplan.env.reward.welfare_true`."""
    alpha = jnp.asarray(cfg.supply.ces_alpha, dtype=float)
    sigma_c = cfg.supply.ces_sigma
    if sigma_c == 1.0:
        return jnp.prod(consumer**alpha)
    rho = (sigma_c - 1.0) / sigma_c
    return jnp.sum(alpha * consumer**rho) ** (1.0 / rho)


# ---------- state ----------


@dataclasses.dataclass
class JState:
    """Mirror of `gosplan.env.state.State`: device arrays plus host-side counters."""

    target: jax.Array
    capital: jax.Array
    inv_output: jax.Array
    inv_inputs: jax.Array
    cum_output: jax.Array
    cum_cost: jax.Array
    quality_acc: jax.Array
    last_report_ratio: jax.Array
    last_report: jax.Array
    last_audited: jax.Array
    last_penalty: jax.Array
    last_fill: jax.Array
    request: jax.Array
    pending_invest: jax.Array
    plan_prices: jax.Array
    planner_io: jax.Array
    consumer_delivery: jax.Array
    claim_history: jax.Array
    pending_deliv: jax.Array
    trade_surplus_acc: jax.Array
    ministry_prev: jax.Array
    t_period: int
    k_step: int
    phase: str
    alive: bool
    seed_env: int
    seed_policy: int


def initial_jstate(cfg: EnvConfig, seed_env: int, seed_policy: int, t_period: int = 0) -> JState:
    """`gosplan.env.state.initial_state`, on the device."""
    from gosplan.env.prices import initial_prices

    s = _static(cfg)
    n, j = cfg.supply.n_enterprises, cfg.supply.n_sectors
    target = jnp.asarray(s["t_0"])
    return JState(
        target=target,
        capital=jnp.ones(n),
        inv_output=jnp.zeros(n),
        inv_inputs=jnp.asarray(s["a_rows"]) * target[:, None],
        cum_output=jnp.zeros(n),
        cum_cost=jnp.zeros(n),
        quality_acc=jnp.zeros(n),
        last_report_ratio=jnp.zeros(n),
        last_report=jnp.zeros(n),
        last_audited=jnp.zeros(n, dtype=bool),
        last_penalty=jnp.zeros(n),
        last_fill=jnp.ones(n),
        request=jnp.zeros((n, j)),
        pending_invest=jnp.zeros((n, cfg.supply.invest_lag)),
        plan_prices=jnp.asarray(initial_prices(cfg), dtype=float),
        planner_io=jnp.asarray(cfg.supply.io_matrix, dtype=float),
        consumer_delivery=jnp.zeros(j),
        claim_history=jnp.repeat(target[:, None], 2, axis=1),
        pending_deliv=jnp.zeros((n, j, cfg.incentive.steps_per_period)),
        trade_surplus_acc=jnp.zeros(n),
        ministry_prev=target,
        t_period=int(t_period),
        k_step=0,
        phase="produce",
        alive=True,
        seed_env=int(seed_env),
        seed_policy=int(seed_policy),
    )


# ---------- host-side keyed draws ----------


def _channel(state: JState, cfg: EnvConfig):
    from gosplan.rng import draw

    n = cfg.supply.n_enterprises
    return jnp.asarray(
        draw(
            state.seed_env,
            "channel",
            state.t_period,
            shape=(n,),
            dist="normal",
            mean=0.0,
            sigma=cfg.information.channel_noise,
        )
    )


def _complaint(state: JState, cfg: EnvConfig):
    from gosplan.rng import draw

    n = cfg.supply.n_enterprises
    return jnp.asarray(
        draw(
            state.seed_env,
            "complaint",
            state.t_period,
            shape=(n,),
            dist="normal",
            mean=0.0,
            sigma=cfg.information.channel_noise,
        )
    )


# ---------- planner ----------


def _forwarded(state: JState, cfg: EnvConfig):
    if cfg.information.ministry_passthrough == 1.0:
        return state.last_report
    return state.ministry_prev


def _sector_mean(values, s):
    totals = _bincount(jnp.asarray(s["onehot"]), values)
    return (totals / s["counts"])[s["sector"]]


def planner_view(state: JState, cfg: EnvConfig) -> dict[str, jax.Array]:
    """`gosplan.env.planner.make_planner_view` (the fields the dynamics read)."""
    info = cfg.information
    s = _static(cfg)
    lag = info.report_lag
    claims = state.claim_history[:, lag - 1] if lag > 0 else _forwarded(state, cfg)
    if info.aggregation_level == "sector":
        claims = _sector_mean(claims, s)
    claims = claims * jnp.exp(_channel(state, cfg))
    qbar = period_quality(state.quality_acc, cfg)
    measured_quality = 1.0 + info.quality_measurability * (qbar - 1.0)
    if info.shortfall_visibility > 0:
        ds = info.shortfall_visibility * (1.0 - state.last_fill) * jnp.exp(_complaint(state, cfg))
        if info.aggregation_level == "sector":
            ds = _sector_mean(ds, s)
    else:
        ds = jnp.zeros(cfg.supply.n_enterprises)
    return {
        "claims": claims,
        "requests": state.request,
        "measured_quality": measured_quality,
        "targets": state.target,
        "planner_io": state.planner_io,
        "downstream_shortfall": ds,
        "plan_prices": state.plan_prices,
    }


def allocate(view, cfg: EnvConfig):
    """`gosplan.env.planner.allocate`."""
    inc = cfg.incentive
    s = _static(cfg)
    avail = _bincount(jnp.asarray(s["onehot"]), (1.0 - s["phi"][s["sector"]]) * view["claims"])
    need = view["planner_io"][s["sector"]] * view["targets"][:, None]
    w = (need + 1e-06) ** inc.alloc_eta_need
    if cfg.information.aggregation_level != "sector":
        w = (view["requests"] + 1e-06) ** inc.alloc_eta_request * w
    total = w.sum(axis=0)
    return jnp.where(total > 0.0, avail * w / jnp.where(total > 0.0, total, 1.0), 0.0)


def fulfilment_measure(view, cfg: EnvConfig):
    """`gosplan.env.planner.fulfilment_measure`."""
    metric = cfg.incentive.objective_metric
    claims = view["claims"]
    if metric == "val":
        return claims
    if metric == "net_output":
        s = _static(cfg)
        alloc = allocate(view, cfg)
        prices = view["plan_prices"]
        return claims - (alloc * prices[None, :]).sum(axis=1) / prices[s["sector"]]
    if metric == "quality_weighted":
        return claims * view["measured_quality"]
    raise ValueError(f"fulfilment_measure: unknown objective_metric {metric!r}")


def update_targets(view, cfg: EnvConfig):
    """`gosplan.env.planner.update_targets`."""
    inc = cfg.incentive
    s = _static(cfg)
    targets = view["targets"]
    rho = fulfilment_measure(view, cfg) / targets
    step = jnp.clip(rho - 1.0, -inc.ratchet_cap_dn, inc.ratchet_cap_up)
    band = inc.ratchet_deadband
    step = jnp.where((rho >= 1.0 - band) & (rho <= 1.0 + band), 0.0, step)
    t_min = cfg.tech.target_floor_frac * s["t_0"]
    grown = (1.0 + inc.growth_directive) * targets * (1.0 + inc.ratchet_lambda * step)
    return jnp.maximum(t_min, grown)


def arrival_steps(state: JState, cfg: EnvConfig) -> np.ndarray:
    """`gosplan.env.planner.arrival_steps` (the host-side keyed categorical draw itself)."""
    from gosplan.env.planner import arrival_steps as np_arrival

    return np_arrival(state.seed_env, state.t_period, cfg)


def deliver(state: JState, alloc, cfg: EnvConfig):
    """`gosplan.env.planner.deliver`, including the AMBIGUITY-023 conservation ruling."""
    s = _static(cfg)
    onehot = jnp.asarray(s["onehot"])
    j = cfg.supply.n_sectors
    phi = jnp.asarray(s["phi"])
    claimed = _forwarded(state, cfg)
    stock = state.inv_output
    qbar = period_quality(state.quality_acc, cfg)
    ratio = jnp.where(claimed > 0, stock / jnp.where(claimed > 0, claimed, 1.0), 1.0)
    fill = jnp.where(claimed > 0, jnp.minimum(1.0, ratio), 1.0)
    shipped = jnp.minimum(stock, claimed)
    pool_num = _bincount(onehot, fill * claimed)
    pool_den = _bincount(onehot, claimed)
    poolfill = jnp.where(pool_den > 0, pool_num / jnp.where(pool_den > 0, pool_den, 1.0), 1.0)
    deliv = alloc * poolfill[None, :]
    target = (1.0 - phi) * _bincount(onehot, shipped)
    current = deliv.sum(axis=0)
    if not np.allclose(np.asarray(current), np.asarray(target), rtol=1e-12, atol=1e-12):
        scale = jnp.where(current > 0, target / jnp.where(current > 0, current, 1.0), 0.0)
        deliv = deliv * scale[None, :]
    if cfg.supply.quality_matters:
        q_num = _bincount(onehot, claimed * qbar)
        q_plain = _bincount(onehot, qbar) / s["counts"]
        qbar_good = jnp.where(pool_den > 0, q_num / jnp.where(pool_den > 0, pool_den, 1.0), q_plain)
    else:
        qbar_good = jnp.ones(j)
    credit = deliv * qbar_good[None, :]
    pending = state.pending_deliv
    if cfg.supply.delivery_timing == "uniform":
        inv_inputs = state.inv_inputs + credit
    else:
        m = cfg.incentive.steps_per_period
        k_arr = jnp.asarray(arrival_steps(state, cfg))
        now = k_arr == 0
        inv_inputs = state.inv_inputs + jnp.where(now, credit, 0.0)
        later = jnp.where(now, 0.0, credit)
        pending = pending + jax.nn.one_hot(k_arr, m) * later[..., None]
    consumer = phi * _bincount(onehot, shipped * qbar)
    new = dataclasses.replace(
        state, inv_output=stock - shipped, inv_inputs=inv_inputs, pending_deliv=pending
    )
    return new, deliv, fill, consumer


# ---------- production ----------


def _yield_shocks(state: JState, cfg: EnvConfig):
    from gosplan.rng import draw

    sector = np.asarray(cfg.supply.sector_of, dtype=int)
    eps = np.empty(cfg.supply.n_enterprises)
    for i in range(eps.size):
        sigma = float(cfg.supply.yield_sigma[sector[i]])
        eps[i] = draw(
            state.seed_env,
            "yield",
            state.t_period,
            state.k_step,
            i,
            shape=(1,),
            dist="lognormal",
            mean_log=-(sigma**2) / 2.0,
            sigma=sigma,
        )[0]
    return jnp.asarray(eps)


def produce(state: JState, action, cfg: EnvConfig):
    """`gosplan.env.production.produce_step`."""
    sup = cfg.supply
    s = _static(cfg)
    m = cfg.incentive.steps_per_period
    a_rows = jnp.asarray(s["a_rows"])
    e = jnp.clip(action["effort"], 0.0, 1.0)
    q = jnp.clip(action["quality"], 0.0, 1.0) if sup.quality_matters else action["quality"]
    v = action["invest"]
    y_hat = s["productivity"] * state.capital / m * e
    need = a_rows * y_hat[:, None]
    h = coverage(state.inv_inputs, need, jnp.asarray(s["omega"]), float(sup.input_complementarity))
    y_tilde = y_hat * h * _yield_shocks(state, cfg)
    y = y_tilde * (1.0 - v)
    consumed = jnp.minimum(state.inv_inputs, a_rows * y_tilde[:, None])
    c = cfg.incentive.effort_cost * e**2 + sup.setup_cost * (e > 0.0) + sup.quality_cost * q * e
    pending = state.pending_invest
    if pending.ndim == 2 and pending.shape[1] > 0:
        pending = pending.at[:, -1].add(y_tilde * v)
    new = dataclasses.replace(
        state,
        inv_inputs=state.inv_inputs - consumed,
        cum_output=state.cum_output + y,
        cum_cost=state.cum_cost + c,
        quality_acc=state.quality_acc + q,
        pending_invest=pending,
    )
    return new, y, c


def credit_arrivals(state: JState):
    """`gosplan.env.production.credit_arrivals`."""
    k = state.k_step
    arriving = state.pending_deliv[:, :, k]
    return dataclasses.replace(
        state,
        inv_inputs=state.inv_inputs + arriving,
        pending_deliv=state.pending_deliv.at[:, :, k].set(0.0),
    )


# ---------- trade ----------


def _visible(state: JState, cfg: EnvConfig) -> np.ndarray:
    from gosplan.env.trade import visible_counterparties

    return visible_counterparties(state, cfg, state.t_period)


def execute_trades(x, offers, need, visible, tau: float):
    """`gosplan.env.trade.execute_trades`: per good, repeatedly execute the eligible pair with the
    largest `min(sup_i, dem_b)`, ties to the lowest `(i, b)` (row-major argmax), until none is
    positive - a `lax.while_loop`, since the number of matches is data-dependent."""
    n, n_goods = x.shape
    eligible = jnp.asarray((visible | visible.T) & ~np.eye(n, dtype=bool))
    sup_all = jnp.maximum(0.0, offers) * x
    dem_all = jnp.maximum(0.0, -offers) * need
    sold = jnp.zeros(n)
    for j in range(n_goods):

        def cond(carry):
            _x, sup, dem, _sold = carry
            pair = jnp.where(eligible, jnp.minimum(sup[:, None], dem[None, :]), 0.0)
            return pair.max() > 0.0

        def body(carry, j=j):
            x_, sup, dem, sold_ = carry
            pair = jnp.where(eligible, jnp.minimum(sup[:, None], dem[None, :]), 0.0)
            flat = jnp.argmax(pair)
            i, b = flat // n, flat % n
            q = pair[i, b]
            x_ = x_.at[i, j].set(jnp.maximum(0.0, x_[i, j] - q))
            x_ = x_.at[b, j].add((1.0 - tau) * q)
            return x_, sup.at[i].add(-q), dem.at[b].add(-q), sold_.at[i].add(q)

        x, _, _, sold = jax.lax.while_loop(cond, body, (x, sup_all[:, j], dem_all[:, j], sold))
    return x, sold


def trade_surplus(state: JState, x_before, x_after, effort, cfg: EnvConfig):
    """`gosplan.env.trade.trade_surplus`."""
    s = _static(cfg)
    a_rows = jnp.asarray(s["a_rows"])
    y_hat = s["productivity"] * state.capital / cfg.incentive.steps_per_period * effort
    need = a_rows * y_hat[:, None]
    theta = float(cfg.supply.input_complementarity)
    h_before = coverage(x_before, need, jnp.asarray(s["omega"]), theta)
    h_after = coverage(x_after, need, jnp.asarray(s["omega"]), theta)
    moved = jnp.any(x_after != x_before, axis=1)
    return jnp.where(moved, y_hat * h_after - y_hat * h_before, 0.0) / state.target


def trade(state: JState, action, cfg: EnvConfig):
    """`gosplan.env.step.stage_trade_volume` / `gosplan.env.trade.trade_stage`."""
    n = cfg.supply.n_enterprises
    if cfg.information.horizontal_visibility <= 0.0:
        return state, jnp.zeros(n)
    s = _static(cfg)
    need = jnp.asarray(s["a_rows"]) * state.target[:, None]
    x_before = state.inv_inputs
    x_after, sold = execute_trades(
        x_before, action["trade_offer"], need, _visible(state, cfg), float(cfg.supply.trade_tau)
    )
    surplus = trade_surplus(state, x_before, x_after, jnp.clip(action["effort"], 0.0, 1.0), cfg)
    return (
        dataclasses.replace(
            state, inv_inputs=x_after, trade_surplus_acc=state.trade_surplus_acc + surplus
        ),
        sold,
    )


# ---------- report, ministry, audit, reward, target, terminate ----------


def process_reports(state: JState, action, cfg: EnvConfig) -> JState:
    """`gosplan.env.reporting.process_reports`."""
    s = _static(cfg)
    stock = (1.0 - cfg.supply.holding_loss) * state.inv_output + state.cum_output
    stock = jnp.minimum(stock, cfg.tech.inventory_cap_mult * 1.0)
    ratio = jnp.clip(action["report_ratio"], 0.0, cfg.tech.report_max_ratio)
    need = state.planner_io[s["sector"]] * state.target[:, None]
    request = jnp.clip(action["input_request"], 0.0, cfg.tech.request_max_multiple) * need
    inv_inputs = state.inv_inputs
    if cfg.supply.input_holding_loss > 0.0:
        inv_inputs = (1.0 - cfg.supply.input_holding_loss) * inv_inputs
    return dataclasses.replace(
        state,
        inv_output=stock,
        inv_inputs=inv_inputs,
        last_report_ratio=ratio,
        last_report=ratio * state.target,
        request=request,
    )


def ministry(state: JState, cfg: EnvConfig) -> JState:
    """`gosplan.env.step.stage_ministry` with the rule-based `ministry_forward` (R10)."""
    pi = cfg.information.ministry_passthrough
    if pi == 1.0:
        return dataclasses.replace(state, ministry_prev=state.last_report)
    pad = cfg.information.ministry_pad * jnp.maximum(0.0, state.target - state.last_report)
    fwd = pi * state.last_report + (1.0 - pi) * (state.ministry_prev + pad)
    return dataclasses.replace(state, ministry_prev=fwd)


def audit(state: JState, view, cfg: EnvConfig):
    """`gosplan.env.planner.select_audits`, `gosplan.env.reporting.audit_and_penalise` and
    `gosplan.env.step.soft_budget_bailouts`."""
    from gosplan.rng import draw, uniforms

    info, inc = cfg.information, cfg.incentive
    n = cfg.supply.n_enterprises
    t = state.t_period
    if info.audit_mode == "targeted" and info.shortfall_visibility > 0:
        p = jnp.clip(
            info.audit_rate * (1.0 + info.audit_target_gain * view["downstream_shortfall"]),
            0.0,
            1.0,
        )
    else:
        p = jnp.full(n, info.audit_rate)
    audited = jnp.asarray(uniforms(state.seed_env, "audit", t, shape=(n,))) < p
    nu = jnp.asarray(
        draw(
            state.seed_env,
            "auditnoise",
            t,
            shape=(n,),
            dist="normal",
            mean=0.0,
            sigma=info.audit_noise,
        )
    )
    s_hat = state.inv_output * jnp.exp(nu)
    if inc.penalty_arg == "positive_part":
        f = jnp.maximum(0.0, state.last_report - s_hat) / state.target
    else:
        f = jnp.abs(state.last_report - s_hat) / state.target
    pen = inc.penalty_scale * (f if inc.penalty_form == "proportional" else (f > 0.0).astype(float))
    penalty = jnp.where(audited, pen, 0.0)
    if inc.soft_budget > 0.0:
        lucky = jnp.asarray(
            draw(state.seed_env, "bailout", t, shape=(n,), dist="bernoulli", p=inc.soft_budget)
        )
        penalty = jnp.where((state.last_fill < 1.0) & lucky, 0.0, penalty)
    return dataclasses.replace(state, last_audited=audited, last_penalty=penalty), penalty


def report_reward(state: JState, penalty, cfg: EnvConfig):
    """`gosplan.env.reward.enterprise_reward` at the REPORT step."""
    view = planner_view(state, cfg)
    view["claims"] = state.last_report
    rho = fulfilment_measure(view, cfg) / state.target
    return reward_scale(cfg) * (bonus(rho, cfg) - penalty + state.trade_surplus_acc)


def period_metrics(state: JState, cfg: EnvConfig):
    """`val_measured`, `val_true` and `welfare_true`."""
    s = _static(cfg)
    prices = state.plan_prices[s["sector"]]
    qbar = period_quality(state.quality_acc, cfg)
    q_hat = 1.0 + cfg.information.quality_measurability * (qbar - 1.0)
    return (
        jnp.sum(prices * state.last_report * q_hat),
        jnp.sum(prices * state.cum_output * qbar),
        welfare_true(state.consumer_delivery, cfg),
    )


def shift_claim_history(state: JState, cfg: EnvConfig) -> JState:
    """`gosplan.env.step.shift_claim_history`."""
    current = _forwarded(state, cfg)
    return dataclasses.replace(
        state, claim_history=jnp.stack([current, state.claim_history[:, 0]], axis=1)
    )


def terminate(state: JState, cfg: EnvConfig) -> bool:
    """`gosplan.env.step.stage_terminate`."""
    from gosplan.rng import draw

    completed = state.t_period + 1
    if completed >= cfg.tech.max_periods:
        return True
    if cfg.tech.horizon_mode == "fixed" or completed < cfg.tech.min_periods:
        return False
    cont = draw(
        state.seed_env,
        "terminate",
        state.t_period,
        shape=(1,),
        dist="bernoulli",
        p=cfg.incentive.tenure,
    )
    return not bool(cont[0])


# ---------- one agent-step (mirror of gosplan.env.step.advance) ----------


def advance(state: JState, action, cfg: EnvConfig):
    """One agent-step: the stage slice of `stages_for_step`, in the PLAN section 2.5 order.
    Returns `(state, reward, done, deliv)`, with `deliv` the DELIVER quantities when DELIVER ran."""
    n, m = cfg.supply.n_enterprises, cfg.incentive.steps_per_period
    reward = jnp.zeros(n)
    done = False
    deliv = None
    if state.phase == "produce":
        if state.k_step == 0:
            view = planner_view(state, cfg)
            alloc = allocate(view, cfg)
            state, deliv, fill, consumer = deliver(state, alloc, cfg)
            state = dataclasses.replace(
                state,
                last_fill=fill,
                cum_output=jnp.zeros(n),
                cum_cost=jnp.zeros(n),
                quality_acc=jnp.zeros(n),
                consumer_delivery=consumer,
            )
            state, _sold = trade(state, action, cfg)
        if cfg.supply.delivery_timing != "uniform" and state.k_step > 0:
            state = credit_arrivals(state)
        state, _y, cost = produce(state, action, cfg)
        reward = -reward_scale(cfg) * cost
    else:
        state = process_reports(state, action, cfg)
        state = ministry(state, cfg)
        view = planner_view(state, cfg)
        state, penalty = audit(state, view, cfg)
        reward = report_reward(state, penalty, cfg)
        state = dataclasses.replace(state, trade_surplus_acc=jnp.zeros(n))
        state = dataclasses.replace(state, target=update_targets(view, cfg))
        state = shift_claim_history(state, cfg)
        done = terminate(state, cfg)
        state = dataclasses.replace(state, alive=not done)
    # advance_phase
    if state.phase == "report":
        state = dataclasses.replace(state, t_period=state.t_period + 1, k_step=0, phase="produce")
    elif state.k_step < m - 1:
        state = dataclasses.replace(state, k_step=state.k_step + 1)
    else:
        state = dataclasses.replace(state, k_step=m, phase="report")
    return state, reward, done, deliv


# ---------- observation (mirror of gosplan.env.obs.build_observation) ----------


def build_observation(state: JState, cfg: EnvConfig, deliv, need):
    from gosplan.env.obs import N_SCALAR_FIELDS, _peer_index, obs_spec, peer_width
    from gosplan.rng import draw

    s = _static(cfg)
    n = cfg.supply.n_enterprises
    m = cfg.incentive.steps_per_period

    def cov(num, den):
        return jnp.where(den != 0.0, num / jnp.where(den != 0.0, den, 1.0), 1.0)

    cols = [
        jnp.full(n, 1.0 if state.phase == "report" else 0.0),
        jnp.full(n, state.k_step / m),
        jnp.log(state.target / s["t_0"]),
        jnp.full(n, cfg.incentive.growth_directive),
        state.cum_output / state.target,
        state.inv_output / state.target,
        state.capital / 1.0,
        state.last_report_ratio,
        state.last_audited.astype(float),
        state.last_penalty * reward_scale(cfg),
        state.last_fill,
        cov(deliv.sum(axis=1), need.sum(axis=1)),
    ]
    obs = jnp.concatenate(
        [
            jnp.stack(cols, axis=1),
            cov(state.inv_inputs, need),
            jnp.asarray(s["onehot"]),
            cov(deliv, need),
        ],
        axis=1,
    )
    width = peer_width(cfg)
    if width > 0:
        idx = _peer_index(cfg)
        peers = jnp.where(idx >= 0, state.last_report_ratio[np.maximum(idx, 0)], 0.0)
        obs = jnp.concatenate([obs, peers], axis=1)
    assert obs.shape[1] == len(obs_spec(cfg)) and N_SCALAR_FIELDS == 12
    sigma = cfg.information.self_obs_noise
    if sigma > 0.0:
        shock = jnp.asarray(
            draw(
                state.seed_env,
                "selfobs",
                state.t_period,
                state.k_step,
                shape=(n,),
                dist="normal",
                mean=0.0,
                sigma=sigma,
            )
        )
        obs = obs.at[:, 4:6].multiply(jnp.exp(shock)[:, None])
    return obs


class JaxGosplanEnv:
    """Mirror of `gosplan.env.env.GosplanEnv` (reset / step / phase) on the JAX kernels."""

    def __init__(self, cfg: EnvConfig) -> None:
        self.cfg = cfg
        n, j = cfg.supply.n_enterprises, cfg.supply.n_sectors
        self._deliv = jnp.zeros((n, j))
        self._arrival = None
        self._seeds = (int(cfg.tech.seed_env), int(cfg.tech.seed_policy))

    def reset(self, seed_env: int, seed_policy: int):
        self._seeds = (int(seed_env), int(seed_policy))
        self.state = initial_jstate(self.cfg, *self._seeds)
        self._deliv = jnp.zeros_like(self._deliv)
        self._arrival = None
        return np.asarray(self._observe(self.state))

    def phase(self) -> str:
        return self.state.phase

    def step(self, action: EnterpriseAction):
        if not self.state.alive:
            self.state = initial_jstate(self.cfg, *self._seeds, t_period=self.state.t_period)
            self._deliv = jnp.zeros_like(self._deliv)
            self._arrival = None
        t, k, phase = self.state.t_period, self.state.k_step, self.state.phase
        act = {
            f.name: jnp.asarray(getattr(action, f.name), dtype=float)
            for f in dataclasses.fields(action)
        }
        state, reward, done, deliv = advance(self.state, act, self.cfg)
        self.state = state
        if deliv is not None:
            self._deliv = deliv
            if self.cfg.supply.delivery_timing != "uniform":
                from gosplan.env.planner import arrival_steps as np_arrival

                self._arrival = jnp.asarray(np_arrival(state.seed_env, t, self.cfg))
        executed = dataclasses.replace(state, t_period=t, k_step=k, phase=phase)
        obs = self._observe(executed)
        return np.asarray(obs), np.asarray(reward), bool(done)

    def _observe(self, state: JState):
        s = _static(self.cfg)
        need = state.planner_io[s["sector"]] * state.target[:, None]
        deliv = self._deliv
        if self._arrival is not None:
            deliv = jnp.where(self._arrival <= state.k_step, deliv, 0.0)
        return build_observation(state, self.cfg, deliv, need)
