"""Reference dynamics - the slow, hand-checkable oracle for the period schedule.

Realises: PLAN sections 2.5 (period schedule), 2.6 (production), 2.7 (planner rules), 2.8
(reporting, audit, penalty, bonus), 2.9 (reward, logged metrics, scaling), 2.10 (prices and final
demand) and 2.11 (inventory). Three sections outside that range are unavoidably in scope because
the schedule and the golden files touch them: 2.12 (termination - schedule stage 7), 2.4 (the
observation the golden files record) and 2.15 (the keyed RNG this module must reproduce exactly).
Owning work order: **WO-002** (Reference dynamics and frozen tests; LEAD; PLAN section 12.3).

Why this file exists. PLAN section 11 puts four test categories in the repository, and the golden
category is defined as "implementation == reference to 1e-9 on seeded trajectories". This module is
that reference. `ref/gen_golden.py` rolls it out and writes `tests/golden/*.json`, so **no numeric
expectation anywhere in the frozen suite is hand-written** (review finding F14). The trust chain
therefore bottoms out here, and this module is validated by exactly two things and nothing else:

  1. the property tests of PLAN section 11 (T-U1 conservation, T-U3 bonus shape, T-U4 target rule,
     T-U5 permutation invariance, T-U6 RNG keying, T-U7 coverage aggregator, T-U8 penalty); and
  2. the hand-computed 2-enterprise, 2-sector worked example committed as
     `docs/ref_worked_example.md`, which is derived from the PLAN formulas *before* this code is
     run and is never edited to agree with it.

Hard rules for every implementer of this file (WO-002 card, PLAN section 12.3):

  * **`ref/` MUST NOT import anything from `gosplan/`.** That is the WO-002 "Forbidden" line, and
    it is the whole point: an oracle that shares code with the implementation cannot detect a bug
    in the shared code. Nothing here may import `gosplan.rng`, `gosplan.config`, `gosplan.env.*`,
    `gosplan.agents.*` or `gosplan.metrics.*`, directly or transitively. `spec/spec.py` is not
    importable as a package either; the binding to it is documentary, stated below.
  * **Plain Python loops and floats, not vectorised numpy.** Every quantity below is a `float`, a
    `list[float]` or a `list[list[float]]`, and every formula is written as an explicit loop over
    enterprises `i`, sectors/goods `j` and steps `k`. Readability and hand-checkability beat speed:
    this module is only ever run at `N <= 4` (WO-002 card), for 30 agent-steps at a time. If a
    formula here cannot be followed on paper, it is written wrong.
  * **numpy appears for one purpose only.** `ref_draw` must reproduce the *same* key-based
    construction as `gosplan/rng.py` (PLAN section 2.15) so that trajectories agree bit for bit:
    `numpy.random.SeedSequence([seed_env, zlib.crc32(purpose.encode()), *indices])`, spawning an
    independent `numpy.random.Generator` per key, then one call per distribution as documented on
    `ref_draw`. The construction is *described* here rather than imported, precisely so that a bug
    in `gosplan/rng.py` shows up as a golden mismatch. numpy is used nowhere else in this module -
    no array arithmetic, no broadcasting, no `np.ndarray` in any signature.
  * **CONTRACT rules bind here as everywhere.** Rule 4: the reward has exactly the terms in
    `ref_enterprise_reward`. Rule 5: `ref_make_planner_view` is the only function that turns a
    `RefState` into planner-visible data (with the same `ref_ship` whitelist note that
    `spec/spec.py` records for `deliver`). Rule 6: `ref_val_measured`, `ref_val_true` and
    `ref_welfare_true` are logged and never enter an observation, a reward or a policy input.
    Rule 7: nothing here implements bunching, padding, storming, hoarding, shaving or trade; those
    are consequences of the transition rules or they are not results. Rule 9: every stochastic term
    goes through `ref_draw`.

Binding to the frozen interface. This module deliberately does **not** import `spec/spec.py`; it
mirrors it. `RefState`, `RefAction` and `RefPlannerView` are field-for-field re-declarations of
`State`, `EnterpriseAction` and `PlannerView` in plain-Python types (`list[float]` where the spec
has `Array`), and `ref_draw`, `ref_coverage`, `ref_bonus`, `ref_reward_scale`, `ref_allocate`,
`ref_ship`, `ref_select_audits`, `ref_update_targets`, `ref_fulfilment_measure`,
`ref_val_measured`, `ref_val_true` and `ref_welfare_true` mirror the identically named spec
callables one for one. `tests/unit/test_spec_imports.py` (WO-001) and the golden parity test
together enforce that the mirror never drifts: a spec signature that moves without a matching move
here is a defect, and the golden files are regenerated under WO-013 with a `spec/CHANGELOG.md`
entry.

Configuration is passed as a plain nested mapping (`Config`), never as an `EnvConfig`, for the same
independence reason: constructing an `EnvConfig` would mean importing `gosplan.config`. The mapping
uses the document layout that `load_config` (PLAN section 3) accepts - top-level keys `supply`,
`incentive`, `information`, `tech` - so a golden configuration and a production configuration are
literally the same document read two ways.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

# ---------- type aliases (PLAN section 10 conventions, plain-Python restatement) ----------

Vec = list[float]
"""A per-enterprise quantity, length `N`. The plain-Python stand-in for a spec `Array` of shape
`(N,)`. `ref/` never uses `Array = np.ndarray`: the oracle is written in loops and floats."""

Mat = list[list[float]]
"""A per-enterprise, per-good quantity, shape `(N, J)`, as a list of `N` rows of `J` floats. The
plain-Python stand-in for a spec `Array` of shape `(N, J)`."""

Goods = list[float]
"""A per-good quantity, length `J = cfg["supply"]["n_sectors"]` (one good per sector, PLAN section
2.1). The plain-Python stand-in for a spec `Array` of shape `(J,)`."""

Draws = list[float] | list[bool] | list[int]
"""Return type of `ref_draw`: floats for `lognormal`/`normal`, booleans for `bernoulli`, integer
category indices for `categorical` (PLAN section 2.15)."""

Config = dict[str, dict[str, object]]
"""A configuration document in the layout `load_config` accepts (PLAN section 3): top-level keys
`"supply"`, `"incentive"`, `"information"`, `"tech"`, each mapping parameter name to value, plus an
optional `"spec_version"` handled by the caller. Values are plain JSON types - `float`, `int`,
`bool`, `str`, and `list`/`tuple` for the vector parameters (`sector_of`, `io_matrix`,
`final_demand_share`, `productivity`, `yield_sigma`, `ces_alpha`, `arrival_probs`) - with
`float("inf")` written as the string `"inf"` exactly as `EnvConfig.hash` encodes it. Every field
name and every Phase-1 default is the one declared on `SupplyConfig`, `IncentiveConfig`,
`InformationConfig` and `TechConfig` in `spec/spec.py`; this module reads them and never redefines
a default."""

Phase = Literal["produce", "report"]
"""Agent-step phase within a plan period (PLAN section 2.5); mirrors `spec.Phase`. Each period is
`M = cfg["incentive"]["steps_per_period"]` PRODUCE steps followed by exactly one REPORT step, so
agents act `M + 1` times per period."""

Purpose = Literal[
    "yield",
    "audit",
    "auditnoise",
    "arrival",
    "channel",
    "drift",
    "terminate",
    "trade_visibility",
    "selfobs",
]
"""Enumerated RNG purposes (PLAN section 2.15, plus `selfobs` for the observation noise of WO-008);
mirrors `spec.Purpose`. Keying by purpose is what makes draws order-independent (T-U6)."""

Dist = Literal["lognormal", "normal", "bernoulli", "categorical"]
"""Distributions `ref_draw` must support; mirrors `spec.Dist`. `lognormal` is parameterised by
`(mean_log, sigma)`; the yield shock of PLAN section 2.6 uses `mean_log = -sigma**2 / 2` so
`E[eps] = 1`."""

Policy = Callable[[Mat, Phase, np.random.Generator], "RefAction"]
"""What `ref_rollout` drives the environment with: a callable taking the observation `(N, d)` laid
out as PLAN section 2.4, the current phase, and a generator from the `seed_policy` stream, and
returning a `RefAction`. It mirrors `spec.Agent.act` (CONTRACT rule 6: the observation and nothing
else). `ref/gen_golden.py` supplies the two Phase-1 reference policies (`Random`,
`TruthfulMyopic`); `ref/` never imports `gosplan.agents`."""


# ---------- records (plain-Python mirrors of the spec dataclasses) ----------


@dataclass
class RefState:
    """The full environment state (PLAN section 2.2), field-for-field mirror of `spec.State`.

    Every array field of `spec.State` becomes a `list` here; the field names, their order and their
    meanings are identical, and a unit test asserts that the two field lists agree. Mutable by
    design: `ref_period` threads one `RefState` through the eight schedule stages of PLAN section
    2.5.

    This is the **true** state. It holds quantities no agent and no planner rule may see (CONTRACT
    rules 5 and 6): only `ref_make_planner_view` may read it on the planner's behalf, only
    `ref_observation` may read it on an agent's behalf, and `ref_ship` reads it as physical
    execution of an allocation already decided from the view.

    `N = cfg["supply"]["n_enterprises"]`, `J = cfg["supply"]["n_sectors"]`,
    `L = cfg["supply"]["invest_lag"]`.
    """

    target: Vec  # (N,) T_i, target in units of own good (PLAN section 2.1)
    capital: Vec  # (N,) Kap_i; Phase 1 fixed at 1.0 (PLAN section 2.1)
    inv_output: Vec  # (N,) S_i, own-good stock on hand; the only place unreported output goes
    inv_inputs: Mat  # (N, J) X_ij, input stocks held by i (PLAN section 2.6)
    cum_output: Vec  # (N,) true output accumulated so far this period (reset each period)
    cum_cost: Vec  # (N,) effort cost accumulated so far this period (reset each period)
    quality_acc: Vec  # (N,) accumulator for the period-average quality qbar_i; Phase 1 inert

    last_report_ratio: Vec  # (N,) rho_i = R_i / T_i as reported at the last REPORT step
    last_report: Vec  # (N,) R_i in units, retained because the ratchet moves T after REPORT
    last_audited: list[bool]  # (N,) audit selection at the last AUDIT step (section 2.7.4)
    last_penalty: Vec  # (N,) penalty charged at the last AUDIT step (section 2.8)
    last_fill: Vec  # (N,) fraction of own last claim actually shipped (section 2.7.3)
    request: Mat  # (N, J) q_ij, input requests from the last REPORT step (section 2.3)
    pending_invest: Mat  # (N, L) output diverted to capital, maturing after invest_lag periods

    t_period: int  # plan period index t, from 0
    k_step: int  # production step index k within the period, 0 .. M-1; M at the REPORT step
    phase: Phase  # "produce" or "report" (section 2.5)
    plan_prices: Goods  # (J,) p_j, plan prices (section 2.10)
    planner_io: Mat  # (J, J) the planner's possibly stale copy of `a` (sections 2.2, 2.6)
    consumer_delivery: Goods  # (J,) consumer_j received by the final-demand sink this period
    alive: bool  # episode has not yet terminated (section 2.12)
    seed_env: int  # root environment seed; every draw is keyed from it (section 2.15)
    seed_policy: int  # root policy seed, kept separate from seed_env (CONTRACT rule 9)


@dataclass
class RefAction:
    """One joint action for all `N` enterprises (PLAN section 2.3), mirror of
    `spec.EnterpriseAction`.

    The dimension set is fixed across phases; configuration flags decide which dimensions the
    environment reads, and dimensions irrelevant to the current phase are ignored rather than
    rejected (PLAN section 2.5). Bounds are those of `spec.action_spec(cfg)`: `effort`, `quality`
    and `invest` in [0, 1]; `report_ratio` in [0, `tech.report_max_ratio`]; `input_request` as a
    multiple of `need_ij` in [0, `tech.request_max_multiple`]; `trade_offer` in [-1, 1].

    Phase-1 active dimensions are `effort`, `report_ratio` and `input_request`
    (`spec.active_action_dims`); `quality`, `invest` and `trade_offer` are carried so that a
    Phase-1 and a Phase-2 action share one type.
    """

    effort: Vec  # (N,) e_ik in [0, 1]; active in Phase 1; read at PRODUCE steps
    quality: Vec  # (N,) q_ik in [0, 1]; Phase 2
    invest: Vec  # (N,) v_ik in [0, 1], fraction of step output diverted to capital; Phase 2
    report_ratio: Vec  # (N,) in [0, rho_max]; active in Phase 1; read only at the REPORT step
    input_request: Mat  # (N, J) q_ij in [0, r_max * need_ij]; logged in P1, inert at eta_q = 0
    trade_offer: Mat  # (N, J) in [-1, 1]; positive = offer, negative = want; Phase 2


@dataclass(frozen=True)
class RefPlannerView:
    """Everything the planner is allowed to know (PLAN sections 2.4, 2.7), mirror of
    `spec.PlannerView`.

    Built exclusively by `ref_make_planner_view` - the single `RefState -> planner` boundary in
    `ref/`, exactly as `make_planner_view` is in `gosplan/` (CONTRACT rule 5). It contains **no true
    quantity**: no `y`, no `S`, no `X`, no welfare, and no audit selection other than the audits
    already performed. Its contents are aggregated at `aggregation_level`, delayed by `report_lag`
    and perturbed by `channel_noise` before they arrive here, so no planner rule can recover the
    truth by inverting anything (test T-B4).
    """

    claims: Vec  # (N,) claimed_i = R_i as it reached the planner, lagged/noised/aggregated
    requests: Mat  # (N, J) q_bj, input requests as they reached the planner
    audited: list[bool]  # (N,) who was audited this period; all False before the AUDIT step
    audit_meas: Vec  # (N,) S_hat_i, the noisy audit measurement; zeros where not audited
    measured_quality: Vec  # (N,) q_hat_i = 1 + mu * (qbar_i - 1) (section 2.9.2); Phase 1 ones
    targets: Vec  # (N,) T_i, the planner's own targets - planner-side by construction
    planner_io: Mat  # (J, J) the planner's estimate of `a`, possibly stale (section 2.6)
    downstream_shortfall: Vec  # (N,) noisy buyer complaints, gated by shortfall_visibility
    aggregation_level: str  # "enterprise" or "sector" (section 2.7.5)
    plan_prices: Goods  # (J,) plan prices, needed by the `net_output` measure (section 2.9.2)


@dataclass
class RefStepRecord:
    """One agent-step of a reference rollout - the row `ref/gen_golden.py` writes to a golden file.

    One record per agent-step (not per enterprise): `obs` and `reward` carry the whole joint step.
    The three period-level logged scalars are filled at the REPORT step and are `None` at PRODUCE
    steps; they are recorded for parity only. CONTRACT rule 6: `val_measured`, `val_true` and
    `welfare` are logged and never appear in `obs`, in `reward`, or in any policy input - test T-B5
    plants sentinels in exactly these fields and asserts they reach no observation.
    """

    t_period: int  # plan period index t
    k_step: int  # step index within the period, 0 .. M-1 at PRODUCE, M at REPORT
    phase: Phase  # the phase this step executed (PLAN section 2.5)
    obs: Mat  # (N, d) observation returned after the step, laid out as PLAN section 2.4
    reward: Vec  # (N,) reward delivered by this step (PLAN section 2.9.1, CONTRACT rule 4)
    done: bool  # geometric termination fired at the end of this step (PLAN section 2.12)
    state_digest: str  # `ref_state_digest(state)` after the step - the golden parity anchor
    val_measured: float | None  # PLAN section 2.9.3; REPORT steps only; logged, never observed
    val_true: float | None  # PLAN section 2.9.3; REPORT steps only; logged, never observed
    welfare: float | None  # PLAN section 2.9.3; REPORT steps only; logged, never observed


# ---------- RNG mirror (PLAN section 2.15) ----------


def ref_draw(
    seed_env: int,
    purpose: Purpose,
    *indices: int,
    shape: tuple[int, ...],
    dist: Dist,
    **params: float,
) -> Draws:
    """Key-based random draw - the single source of randomness in `ref/` (PLAN section 2.15).

    Takes: `seed_env`, the run's root environment seed; `purpose`, one of the `Purpose` values;
    `*indices`, the integer coordinates of the draw (`t`, then `k`, then `i`, in that order);
    `shape`, the shape of the result, vectorising over the trailing index; `dist`; and `**params`,
    the distribution parameters -

        lognormal    mean_log, sigma      returns exp(N(mean_log, sigma**2))
        normal       mean, sigma
        bernoulli    p                    returns booleans
        categorical  probs                returns integer category indices

    Returns: a flat `list` of `shape[0]` values (this module only ever needs rank-1 draws; the
    production `draw` returns an `Array` of the requested `shape`, and the two must agree
    elementwise for every key).

    Construction, which MUST match `gosplan/rng.py` element for element (PLAN section 2.15, WO-004
    notes) - it is described here rather than imported, because an oracle that shares the
    implementation's RNG cannot detect a bug in it:

        key = numpy.random.SeedSequence([seed_env, zlib.crc32(purpose.encode()), *indices])
        gen = numpy.random.default_rng(key)
        lognormal    gen.lognormal(mean=mean_log, sigma=sigma, size=shape)
        normal       gen.normal(loc=mean, scale=sigma, size=shape)
        bernoulli    gen.random(size=shape) < p
        categorical  gen.choice(len(probs), size=shape, p=probs)

    Consequences the rest of the design leans on: draws are **order-independent** (nothing depends
    on how many draws were taken before), so this module and `gosplan/` agree by construction;
    common random numbers across arms hold whenever `seed_env` is shared (PLAN section 4.3); and
    the policy stream `seed_policy` is entirely separate (CONTRACT rule 9).

    Binds: T-U6 (`tests/unit/test_rng.py`) - deterministic in `(seed_env, purpose, indices)`,
    independent of call order, and each distribution has the stated moments; and every golden file,
    which is bit-for-bit reproducible only because this construction is keyed rather than streamed.
    """
    raise NotImplementedError("PLAN section 2.15 - implemented in WO-002")


# ---------- setup (PLAN sections 2.2, 2.10, 3) ----------


def ref_initial_prices(cfg: Config) -> Goods:
    """Solve the cost-plus plan-price fixed point at `t = 0` (PLAN section 2.10).

    Takes: `cfg`. Returns: `p`, length `J`, strictly positive.

    Formula (PLAN section 2.10, verbatim):

        p_j = (1 + m) * (kappa_labour + sum_k a_jk * p_k)

    solved as a fixed point over `j` by plain iteration from `p = kappa_labour * (1 + m)` until the
    largest coordinate change is below 1e-12, with `m = cfg["supply"]["price_markup"]`.
    `kappa_labour` is not a configuration parameter: `p` is homogeneous of degree 1 in it and every
    use of `p` in this design is a ratio (`val_measured / val_true`, `p_j / p_{s(i)}` in the
    `net_output` measure), so the scale cancels. It is fixed at 1.0 and that is recorded as a
    normalisation, identically to `spec.initial_prices`.

    Convergence requires `(1 + m) * sum_k a_jk < 1` for every row, which is why
    `EnvConfig.validate` rejects `sum_k a_jk >= 1` (WO-003). Phase 1 holds prices fixed thereafter
    (`price_lag = inf`).

    Binds: `tests/unit/test_prices.py` (fixed point converges; every price positive) and the golden
    files, whose first state digest contains these prices.
    """
    raise NotImplementedError("PLAN section 2.10 - implemented in WO-002")


def ref_initial_state(cfg: Config, seed_env: int, seed_policy: int) -> RefState:
    """Build the opening state of an episode (PLAN sections 2.2, 2.5, 3).

    Takes: `cfg`, and the two root seeds, which are stored on the state and key every later draw.
    Returns: a `RefState` identical field for field to what `GosplanEnv.reset` builds (spec
    `reset` docstring):

        target            T_0_i = cfg["tech"]["initial_target_frac"] * A_{s(i)} * cap_i
        capital           cap_i = 1.0 for all i (Phase 1, PLAN section 2.1)
        inv_output        0.0
        inv_inputs        0.0
        cum_output        0.0
        cum_cost          0.0
        quality_acc       0.0
        last_report_ratio 0.0
        last_report       0.0
        last_audited      False
        last_penalty      0.0
        last_fill         1.0            # nothing has been claimed yet, so nothing is unfilled
        request           0.0
        pending_invest    0.0
        t_period          0
        k_step            0
        phase             "produce"
        plan_prices       ref_initial_prices(cfg)
        planner_io        cfg["supply"]["io_matrix"]     # planner starts with the true `a`
        consumer_delivery 0.0
        alive             True

    `A_j` is `cfg["supply"]["productivity"]`, and `s(i)` is `cfg["supply"]["sector_of"][i]`. The
    initial targets are also the anchor for two later quantities: the target floor
    `T_min_i = cfg["tech"]["target_floor_frac"] * T_0_i` used by `ref_update_targets`, and the
    observation field `log(T_i / T_0_i)` (PLAN section 2.4 index 2), so callers keep `T_0` for the
    whole episode rather than recomputing it.

    Binds: the opening `state_digest` of every golden file, and `tests/unit/test_env_api.py`.
    """
    raise NotImplementedError("PLAN section 2.2 - implemented in WO-002")


def ref_state_digest(state: RefState) -> str:
    """Canonical content digest of a state - the golden files' parity anchor.

    Takes: `state`. Returns: a SHA-256 hex digest (`hashlib.sha256`, hex, lowercase) of a canonical
    text rendering of every field of `RefState`, in **declaration order**:

        - the field name, then "=", then the rendered value, then ";"
        - a float renders as `repr(float(x))`; an int as `repr(int(x))`; a bool as "true"/"false";
          a str verbatim
        - a list renders as its elements, comma-separated, inside "[" and "]", row-major for
          nested lists, so `inv_inputs` renders row by row
        - the whole string is UTF-8 encoded before hashing

    The rendering is specified in this docstring rather than left to the implementation because
    `tests/golden/` must rebuild the identical digest from the production `State`, whose fields are
    `Array`s: the test flattens each array row-major to Python floats and applies the same rules.
    Any change to the rendering invalidates every golden file and is therefore a regeneration under
    WO-013 with a `spec/CHANGELOG.md` entry (CONTRACT rule 1).

    Digesting rather than storing the whole state keeps a golden file small while still failing on
    any divergence; observations and rewards are stored in full, so a mismatch localises to either
    the observable interface or the hidden state.

    Binds: T-B7 (golden parity to 1e-9) - note that the digest is exact-equality, so the 1e-9
    tolerance of PLAN section 11 applies to the stored observations and rewards, and the digest is
    compared only after those pass (a digest mismatch with matching observations is a hidden-state
    divergence and is reported as such, never waived).
    """
    raise NotImplementedError("PLAN section 2.2 - implemented in WO-002")


# ---------- production helpers (PLAN section 2.6) ----------


def ref_coverage(x_row: Goods, need_row: Goods, weights: Goods, theta: float) -> float:
    """CES aggregator over one enterprise's per-good input coverage ratios (PLAN section 2.6).

    Takes: `x_row`, that enterprise's input stocks `X_ij` (length `J`); `need_row`, this step's
    need `need_ikj` (length `J`); `weights`, `omega_j = a_{s(i)j} / sum_j a_{s(i)j}` (length `J`);
    `theta`, the complementarity exponent. Returns: the scalar coverage multiplier `H_ik` in
    [0, 1].

    Formula (PLAN section 2.6, verbatim):

        H_ik = ( sum_j omega_j * min(1, X_ij / need_ikj)**(-theta) )**(-1/theta)
        H_ik = 1                        if enterprise i requires no inputs

    Edge cases that must be handled explicitly rather than by floating-point luck:
      - `need_ikj == 0` contributes a coverage ratio of exactly 1, never a division by zero;
      - a whole row of `a` equal to zero gives `H = 1` (the enterprise needs no inputs);
      - `theta = float("inf")` takes the `min` branch: the minimum coverage ratio over the goods
        with positive need;
      - a coverage ratio of exactly 0 with finite `theta` gives `H = 0` (the `-theta` power is
        infinite), which must not raise.

    Written per enterprise and per step, as a loop over `j` - the vectorised `spec.coverage` takes
    whole `(N, J)` arrays, and the difference in shape is deliberate: the two are compared through
    the golden files, not through a shared implementation.

    Binds: T-U7 (`tests/unit/test_production.py`) - `theta = inf` equals `min`; `theta -> 1` equals
    the weighted harmonic mean; `H = 1` when no inputs are needed. Edge cases E2, E4 and E5 of
    `docs/ref_worked_example.md`.
    """
    raise NotImplementedError("PLAN section 2.6 - implemented in WO-002")


def ref_input_need(cfg: Config, i: int, y_hat: float) -> Goods:
    """This step's input need for one enterprise (PLAN section 2.6).

    Takes: `cfg`; `i`, the enterprise index; `y_hat`, its intended output this step. Returns:
    `need_ikj` for every good `j`, length `J`.

    Formula (PLAN section 2.6, verbatim):

        need_ikj = a_{s(i)j} * y_hat_ik      for each input good j

    with `a` the **true** I-O matrix `cfg["supply"]["io_matrix"]` and `s(i)` the enterprise's
    sector. Goods with `a_{s(i)j} == 0` get a need of 0.0, which `ref_coverage` reads as full
    coverage. This is the production need of PLAN section 2.6 and is **not** the planned need
    `planner_io[s(b), j] * T_b` of PLAN section 2.7.2, which the planner uses in `ref_allocate`;
    the two coincide only by accident.

    Binds: `tests/unit/test_production.py` (inputs consumed equal `a * y_tilde` capped at stock)
    and T-U1.
    """
    raise NotImplementedError("PLAN section 2.6 - implemented in WO-002")


# ---------- planner helpers (PLAN section 2.7) ----------


def ref_make_planner_view(state: RefState, cfg: Config, claim_history: list[Vec]) -> RefPlannerView:
    """Build the planner's view - the ONLY `RefState -> planner` function here (CONTRACT rule 5).

    Takes: the true `state`; `cfg`; and `claim_history`, the claims of every completed period in
    chronological order (`claim_history[t]` is the `(N,)` vector of `R_i` reported at the end of
    period `t`), which is what makes `report_lag` implementable. Returns: a `RefPlannerView`
    carrying reports, requests, audit results, measured quality, targets, the planner's I-O
    estimate, the (Phase-2) noisy downstream shortfall, the aggregation level and the plan prices -
    and nothing else.

    The three information filters of PLAN section 2.7.5 are applied here, in this order:

        aggregation   at aggregation_level = "sector" the planner sees only
                      sum_{i in sector j} claimed_i, and `ref_allocate` keys on planned need alone
        lag           the rules consume claims from `report_lag` periods ago (Phase 1: 0); before
                      that many periods have elapsed the claims are zeros
        channel noise claimed_i <- claimed_i * exp(xi_i), xi ~ N(0, sigma_ch**2),
                      key = (seed_env, "channel", t, i), drawn through `ref_draw`

    All three branches must exist even though the Phase-1 configuration makes each of them the
    identity; the identity case is exactly what `tests/unit/test_planner.py` checks.

    The view is built once per period after the REPORT step. `audited` and `audit_meas` are all
    False / zero until `ref_select_audits` and `ref_audit_penalty` have run, after which the view
    is reissued with them filled (that reissue is `ref_audit`).

    **Interface note (open item for the v1 freeze, WO-013).** `spec.make_planner_view` takes
    `(state, cfg)` only, and `spec.State` carries no claim history, so the production
    implementation must carry it somewhere - inside `GosplanEnv`, or as a new state field. This
    module makes the history an explicit argument because an oracle should hide nothing. The
    difference is invisible in Phase 1 (`report_lag = 0`), and it must be resolved before
    `report_lag > 0` is ever run; until then it is a documented divergence, not a licence to pick
    one.

    Constraints: no true quantity may cross this boundary - not `y`, not `S`, not `X`, not welfare.
    Binds: T-B4 (`tests/behavioural/test_planner_blindness.py`), which plants sentinels in the
    state and asserts none reaches the view.
    """
    raise NotImplementedError("PLAN section 2.7.5 - implemented in WO-002")


def ref_allocate(view: RefPlannerView, cfg: Config) -> Mat:
    """Allocate claimed supply across buyers - promises, not goods (PLAN section 2.7.2).

    Takes: `view` (last period's claims, requests, targets, the planner's I-O estimate) and `cfg`.
    Returns: `alloc`, shape `(N, J)`, the promised quantity of each good to each buyer, in
    **claimed** units.

    Formula (PLAN section 2.7.2, verbatim):

        claimed_i = R_i                                       # lagged/noised/aggregated by the view
        avail_j   = sum_{i: s(i)=j} (1 - phi_j) * claimed_i    # what the planner believes exists
        need_bj   = planner_io[s(b), j] * T_b                  # what the plan says buyer b needs
        w_bj      = (q_bj + 1e-6)**eta_q * (need_bj + 1e-6)**eta_n
        alloc_bj  = avail_j * w_bj / sum_b w_bj

    with `phi_j = cfg["supply"]["final_demand_share"]`, `eta_q =
    cfg["incentive"]["alloc_eta_request"]` (Phase 1: 0) and `eta_n =
    cfg["incentive"]["alloc_eta_need"]` (1.0). At `eta_q = 0` the request term is exactly 1 and
    requests are ignored: Phase 1 logs them and they are inert (finding F7). Under
    `aggregation_level = "sector"` the weight collapses to the need term alone.

    The `1e-6` regularisers are part of the formula, not an implementation detail: they keep the
    weights finite when a need or a request is zero, and they must appear exactly as written or the
    golden files will not match.

    CONTRACT rule 7: this is a rule about weights. Nothing here, in Phase 1 or Phase 2, instructs an
    enterprise to inflate a request.

    Binds: `tests/unit/test_planner.py` - allocation sums to `avail_j` per good; `eta_q = 0` makes
    the result invariant to `requests`; rows 0.2-0.5 of `docs/ref_worked_example.md`.
    """
    raise NotImplementedError("PLAN section 2.7.2 - implemented in WO-002")


def ref_ship(
    state: RefState, alloc: Mat, claims: Vec, cfg: Config
) -> tuple[RefState, Mat, Vec, Goods]:
    """Turn promises into physical goods (PLAN section 2.7.3) - the padding-to-shortage channel.

    Takes: `state` (the true stocks); `alloc` `(N, J)` from `ref_allocate`; `claims` `(N,)`, the
    claims those promises were computed from; `cfg`. Returns: `(state, deliv, fill, consumer)` -
    the updated state, physical receipts `deliv` `(N, J)`, the per-seller fill ratio `fill` `(N,)`,
    and the consumer sink's receipts `consumer` `(J,)`.

    Formulas (PLAN section 2.7.3, verbatim):

        fill_i     = min(1, S_i / claimed_i)        (fill_i = 1 when claimed_i = 0)
        shipped_i  = min(S_i, claimed_i)
        poolfill_j = sum_{i in j} fill_i * claimed_i / sum_{i in j} claimed_i
        deliv_bj   = alloc_bj * poolfill_j                       # physical receipt
        X_bj      += deliv_bj * qbar_j                           # quality-routed; Phase 1 qbar = 1
        consumer_j = sum_{i in j} phi_j * shipped_i * qbar_i
        S_i       -= shipped_i

    Guards: `claimed_i = 0` gives `fill_i = 1` (edge case E1 of `docs/ref_worked_example.md`); a
    sector whose total claim is zero gives `poolfill_j = 1` by the same convention, so no
    downstream buyer is penalised for a good nobody claimed.

    A claim above stock lowers `poolfill` for the whole good and therefore reduces every downstream
    buyer's receipt; a claim below stock leaves the difference sitting in `S`. Both are
    consequences of these lines, never rules of their own - CONTRACT rule 7 forbids implementing
    either directly, and the direction of neither is asserted anywhere.

    Interface note mirroring `spec.deliver`: this function takes a `RefState` while CONTRACT rule 5
    reserves that for `ref_make_planner_view`. It makes no planner decision and reads no claim
    except through `alloc` and its explicit `claims` argument - it is physical execution - so the
    T-B4 static check whitelists it alongside the view builder, exactly as the spec records for
    `deliver`.

    Binds: `tests/unit/test_planner.py` (`poolfill` in [0, 1]; delivery conservation; `claimed = 0`
    gives `fill = 1`), T-U1 (per-period conservation to 1e-9) and T-B3
    (`tests/behavioural/test_shortage_propagation.py`).
    """
    raise NotImplementedError("PLAN section 2.7.3 - implemented in WO-002")


def ref_select_audits(view: RefPlannerView, cfg: Config, t: int) -> list[bool]:
    """Choose which enterprises to audit this period (PLAN section 2.7.4).

    Takes: `view`, `cfg`, and the plan period `t`, which keys the draw. Returns: a list of `N`
    booleans.

    Formulas (PLAN section 2.7.4, verbatim):

        random    (Phase 1)  audited_i ~ Bernoulli(a),  key = (seed_env, "audit", t, i)
        targeted  (Phase 2)  probability a * (1 + kappa_t * downstream_shortfall_i), clipped to
                             [0, 1], active only when shortfall_visibility > 0

    with `a = cfg["information"]["audit_rate"]`. The whole vector is drawn with one keyed call,
    `ref_draw(seed_env, "audit", t, shape=(N,), dist="bernoulli", p=a)`, so the selection is
    independent of how many other draws the period took (T-U6). `kappa_t` is not yet a
    configuration field; the `targeted` branch is a Phase-2 sketch and is not exercised by any
    Phase-1 golden file.

    The selection is never observable to any agent before it happens (PLAN section 2.4: the audit
    selection for the current period is never in an observation).

    Binds: `tests/unit/test_planner.py` (empirical audit frequency matches `audit_rate`;
    determinism in `(seed_env, t)`) and row 4.1 of `docs/ref_worked_example.md`, whose seed must
    yield one audited and one unaudited enterprise so that both branches of T-U8 are hand-checked.
    """
    raise NotImplementedError("PLAN section 2.7.4 - implemented in WO-002")


def ref_fulfilment_measure(view: RefPlannerView, cfg: Config) -> Vec:
    """The fulfilment quantity the bonus and the ratchet key on (PLAN section 2.9.2).

    Takes: `view` and `cfg`. Returns: `m_i` `(N,)` in units of own good, selected by
    `cfg["incentive"]["objective_metric"]`:

        val               m_i = R_i
        net_output        m_i = R_i - sum_j p_j * alloc_ij / p_{s(i)}    # net of allocated inputs
                                                                        # valued at plan prices
        quality_weighted  m_i = R_i * q_hat_i,  q_hat_i = 1 + mu * (qbar_i - 1)

    where `alloc` is `ref_allocate(view, cfg)` - the allocation the planner itself computed from
    the same view - `p` is `view.plan_prices` and `q_hat` is `view.measured_quality`. Every input
    is planner-side, so this function never needs the true state.

    `welfare` is deliberately not a member of `ObjectiveMetric` (finding F6): the planner cannot key
    on a quantity it does not observe, and CONTRACT rule 6 keeps `welfare_true` out of every
    decision path.

    Binds: `tests/unit/test_planner.py` and `tests/unit/test_reward.py` (the `val` branch is the
    identity on claims; `net_output` falls as allocated inputs rise).
    """
    raise NotImplementedError("PLAN section 2.9.2 - implemented in WO-002")


def ref_update_targets(view: RefPlannerView, cfg: Config, initial_targets: Vec) -> Vec:
    """Apply the ratchet and the growth directive to every target (PLAN section 2.7.1).

    Takes: `view` (claims and current targets as the planner knows them); `cfg`; and
    `initial_targets`, the episode's `T_0` vector from `ref_initial_state`, which fixes the floor.
    Returns: the new targets `(N,)`.

    Formula (PLAN section 2.7.1, verbatim):

        m_i    = ref_fulfilment_measure(view, cfg)        # section 2.9.2; Phase 1: m_i = R_i
        rho_i  = m_i / T_i
        step_i = clip(rho_i - 1, -c_dn, +c_up)
        step_i = 0                       if |rho_i - 1| <= delta   # deadband; Phase 1 delta = 0
        T_i   <- max(T_min_i, (1 + g) * T_i * (1 + lambda * step_i))

    with `T_min_i = cfg["tech"]["target_floor_frac"] * initial_targets[i]`, `c_up =
    cfg["incentive"]["ratchet_cap_up"]`, `c_dn = ...["ratchet_cap_dn"]`, `delta =
    ...["ratchet_deadband"]`, `g = ...["growth_directive"]` and `lambda = ...["ratchet_lambda"]`.
    Under `report_lag > 0` the rule uses the `m_i` the view has already lagged.

    Note the order of operations: the deadband zeroes the *capped* step, and the growth factor
    `(1 + g)` multiplies before the floor is applied, so at `rho = 1` and `g > 0` the target grows
    at exactly `(1 + g)`.

    `g > 0` is the forcing term added for finding F1. With `g = 0` and reports at target the map
    has a fixed point - that is what test T-B2 checks, and it is exactly why `g` must be a
    treatment variable rather than a constant.

    Binds: T-U4 (`tests/unit/test_planner.py`: fixed point at `rho = 1`, `g = 0`; step bounded by
    `c_up`/`c_dn`; floor respected; deadband inert outside `|rho - 1| <= delta`) and T-B2
    (`tests/behavioural/test_fixed_point.py`). Rows 6.1-6.7 of `docs/ref_worked_example.md`.
    """
    raise NotImplementedError("PLAN section 2.7.1 - implemented in WO-002")


# ---------- reporting, audit, bonus, reward (PLAN sections 2.8, 2.9) ----------


def ref_audit_penalty(state: RefState, audited: list[bool], cfg: Config, t: int) -> tuple[Vec, Vec]:
    """Measure audited stock and charge the penalty (PLAN section 2.8).

    Takes: `state` after `ref_report` has closed the books; `audited` `(N,)` from
    `ref_select_audits`; `cfg`; and the period `t`, which keys the measurement noise. Returns:
    `(penalty, audit_meas)` - the penalty `(N,)`, zero wherever `audited` is False, and the
    measurement `S_hat` `(N,)`, zero where not audited (the view carries zeros there).

    Formulas (PLAN section 2.8, verbatim):

        S_hat_i   = S_i * exp(nu_i),  nu_i ~ N(0, sigma_aud**2)
                    key = (seed_env, "auditnoise", t, i)
        f_i       = max(0, R_i - S_hat_i) / T_i      if penalty_arg = positive_part   (Phase 1)
                  = |R_i - S_hat_i| / T_i            if penalty_arg = absolute
        Pen_i     = pen * f_i                        if penalty_form = proportional   (Phase 1)
                  = pen * 1[f_i > 0]                 if penalty_form = fixed
        penalty_i = 1[audited_i] * Pen_i

    The audit compares the claim to **stock on hand**, never to the period's production. The
    penalty is expressed in ratio units - divided by `T_i` - per finding F9.

    Draw the whole `nu` vector with one keyed call even when only some enterprises are audited, so
    that the draw does not depend on the audit selection (T-U6, order independence).

    Binds: T-U8 (`tests/unit/test_reporting.py`) - `positive_part` gives exactly 0 for any
    under-report while `absolute` does not; `audited = False` gives 0 regardless. Edge cases
    E9-E11 of `docs/ref_worked_example.md`.
    """
    raise NotImplementedError("PLAN section 2.8 - implemented in WO-002")


def ref_bonus(rho: float, cfg: Config) -> float:
    """Bonus schedule on one fulfilment ratio, in ratio units (PLAN section 2.8).

    Takes: `rho`, a scalar fulfilment ratio `m_i / T_i`, and `cfg`. Returns: `B(rho)`.

    Formula (PLAN section 2.8, verbatim):

        Lambda_w(x) = 1[x >= 0]                if w = 0        # strict >=, a true Heaviside
                    = 1 / (1 + exp(-x / w))    if w > 0
        B(rho)      = beta * Lambda_w(rho - 1) + s * clip(rho - 1, 0, rho_cap - 1)

    with `beta = cfg["incentive"]["notch_height"]`, `w = ...["notch_width"]`, `s =
    ...["overfulfilment_slope"]` and `rho_cap = ...["overfulfilment_cap"]`. `rho_cap = inf` means
    no cap at all and hence no kink: that branch must not clip. The Heaviside is strict `>=`, so
    the notch is paid at exactly `rho = 1` (edge case E7).

    Named configurations (PLAN section 2.8), all three of which the golden set must exercise:
        notched               w = 0,    rho_cap = 1.2
        smooth counterfactual w = 0.25, rho_cap = inf     # no discontinuity, no kink anywhere
        kink-only             w = 0.25, rho_cap = 1.2     # isolates kink bunching at the cap

    Scalar rather than vectorised, unlike `spec.bonus`, because it is read on paper in
    `docs/ref_worked_example.md`; the caller loops over enterprises.

    Binds: T-U3 (`tests/unit/test_reward.py`) - monotone in `rho`; discontinuous at `rho = 1` iff
    `w = 0`; continuous with continuous derivative iff `w > 0` and `rho_cap = inf`. Edge cases E7
    and E8.
    """
    raise NotImplementedError("PLAN section 2.8 - implemented in WO-002")


def ref_reward_scale(cfg: Config) -> float:
    """Analytic per-configuration reward scale (PLAN section 2.9.1; finding F9).

    Takes: `cfg`. Returns: `scale = 1 / ref_bonus(1.1, cfg)`, a single float computed from the
    configuration alone - **never** from running statistics.

    It makes the bonus at 110% fulfilment equal to 1 in every configuration, so a sweep over `beta`
    changes the economics (the notch relative to `pen` and `kappa`) and not the gradient magnitude.
    CONTRACT rule 4 forbids running reward normalisation outright: running statistics change the
    effective reward over training and, with heavy-tailed penalties, shrink the notch in normalised
    units.

    Binds: T-U2 (`tests/unit/test_reward.py`) - `ref_reward_scale(cfg) * ref_bonus(1.1, cfg) == 1`
    to floating-point tolerance for every configuration in the golden matrix.
    """
    raise NotImplementedError("PLAN section 2.9.1 - implemented in WO-002")


def ref_enterprise_reward(
    state: RefState,
    cfg: Config,
    phase: Phase,
    cost: Vec | None,
    penalty: Vec | None,
    trade_surplus: Vec | None,
) -> Vec:
    """The only quantity any learner receives (PLAN section 2.9.1; CONTRACT rule 4).

    Takes: `state`; `cfg`; the current `phase`; and the three period quantities - `cost` `(N,)` at
    a PRODUCE step, `penalty` `(N,)` and `trade_surplus` `(N,)` at the REPORT step - each `None` in
    the phase where it does not apply. Returns: `r` `(N,)`.

    Formula (PLAN section 2.9.1, verbatim):

        PRODUCE step k:   r_ik = - scale * c_ik
        REPORT step:      r_i  =   scale * ( B(rho_i) - penalty_i + trade_surplus_i )
        scale             = ref_reward_scale(cfg)          # analytic, per configuration

    with `trade_surplus == 0` throughout Phase 1, `penalty_i = 1[audited_i] * Pen_i` from
    `ref_audit_penalty`, and `rho_i = m_i / T_i` for the `m_i` of `ref_fulfilment_measure` computed
    on the same period's view.

    These are the only terms. CONTRACT rule 4 forbids per-step shaping, auxiliary rewards, curiosity
    terms, potential-based terms and running reward normalisation. Effort cost is a real cost paid
    when it is incurred, not shaping. Nothing here may read `val_measured` or a welfare quantity
    (CONTRACT rule 6).

    Binds: T-B6 - the reward is recomputed independently from this five-term formula on random
    states and must agree exactly; and rows 2.15 and 5.6 of `docs/ref_worked_example.md`.
    """
    raise NotImplementedError("PLAN section 2.9.1 - implemented in WO-002")


def ref_val_measured(state: RefState, cfg: Config, measured_quality: Vec) -> float:
    """The planner-side output aggregate, in plan prices (PLAN section 2.9.3).

    Takes: `state` at the end of a period; `cfg`; `measured_quality` `(N,)`, `q_hat_i = 1 + mu *
    (qbar_i - 1)` from the planner view (Phase 1: ones). Returns: the scalar

        val_measured_t = sum_i p_{s(i)} * R_i * q_hat_i

    using `state.last_report` for `R_i` and `state.plan_prices` for `p`. This is what the planning
    system believes it produced.

    Logged only. CONTRACT rule 6: it never appears in any observation, reward or agent input; test
    T-B5 asserts this with sentinels. It is the numerator of `padding_index = val_measured /
    val_true` (PLAN section 2.9.4).
    """
    raise NotImplementedError("PLAN section 2.9.3 - implemented in WO-002")


def ref_val_true(state: RefState, cfg: Config, period_output: Vec) -> float:
    """The true output aggregate, in plan prices (PLAN section 2.9.3).

    Takes: `state` at the end of a period; `cfg`; `period_output` `(N,)`, the period's true
    production `y_i` (the sum of the period's `y_ik`, i.e. `cum_output` before it is reset).
    Returns: the scalar

        val_true_t = sum_i p_{s(i)} * y_i * qbar_i

    with `qbar_i` the period-average quality (Phase 1: 1).

    Logged only, exactly as `ref_val_measured` (CONTRACT rule 6). The ratio `val_measured /
    val_true` is the `padding_index` of PLAN section 2.9.4.
    """
    raise NotImplementedError("PLAN section 2.9.3 - implemented in WO-002")


def ref_welfare_true(consumer: Goods, cfg: Config) -> float:
    """Consumer welfare from this period's final deliveries (PLAN section 2.9.3).

    Takes: `consumer` `(J,)`, the final-demand sink's receipts from `ref_ship`, and `cfg`. Returns:
    the scalar CES index

        rho_ces   = (sigma_c - 1) / sigma_c
        welfare_t = ( sum_j alpha_j * consumer_j**rho_ces )**(1 / rho_ces)

    with `alpha_j = cfg["supply"]["ces_alpha"]` and `sigma_c = cfg["supply"]["ces_sigma"]`. The
    `sigma_c -> 1` limit is the Cobb-Douglas index `prod_j consumer_j**alpha_j` and must be an
    explicit branch, not a limit taken numerically. `consumer_j = 0` is legal and gives welfare 0
    for `sigma_c < 1`; the implementation must not raise on it.

    Logged only, and the strictest case of CONTRACT rule 6: no agent, no planner rule and no reward
    term may read it. `W = mean_t welfare_t` over the measurement window of PLAN section 4.4
    (periods `t >= 2`), and `welfare_ratio = W / W_oracle` (PLAN section 2.9.4).
    """
    raise NotImplementedError("PLAN section 2.9.3 - implemented in WO-002")


# ---------- observation (PLAN section 2.4) ----------


def ref_observation(state: RefState, cfg: Config, initial_targets: Vec, deliv: Mat) -> Mat:
    """Build the observation the golden files record (PLAN section 2.4).

    Takes: `state`; `cfg`; `initial_targets`, the episode's `T_0` (index 2 is a ratio to it);
    `deliv` `(N, J)`, this period's physical receipts from `ref_ship` (indices 11 and 12+2J:12+3J).
    Returns: `obs`, shape `(N, d)` with `d = 12 + 3 * J` in Phase 1, in exactly the order
    `spec.obs_spec(cfg)` returns:

        0                phase                     0 produce / 1 report
        1                k_over_M                  step within the period
        2                log_target_ratio          log(T_i / T_0_i)
        3                growth_directive          g; constant per config
        4                cum_output_over_target    own true production so far this period
        5                stock_over_target         S_i / T_i
        6                capital_ratio             Kap_i / Kap_0
        7                last_report_ratio
        8                last_audited              0.0 / 1.0
        9                last_penalty_scaled       last_penalty * ref_reward_scale(cfg)
        10               last_fill
        11               inputs_delivered_total    delivered this period / need, totals
        12      : 12+J   input_cov_{j}             X_ij / need_ij, per good
        12+J   : 12+2J   sector_onehot_{j}
        12+2J  : 12+3J   deliv_cov_{j}             deliv_ij / need_ij this period, per good

    Never present, in any phase: `welfare_true`, `val_measured`, any other enterprise's `y`, `S` or
    `X`, the audit selection for the current period, and periods remaining under geometric
    termination (PLAN sections 2.4, 2.12; CONTRACT rule 6).

    Phase-1 `self_obs_noise = 0`, so indices 4 and 5 are exact; when it is non-zero they are
    multiplied by `exp(N(0, s**2))` drawn with purpose `selfobs` (WO-008).

    **OPEN ITEM - resolve before any golden file is generated.** PLAN section 2.4 writes the
    denominator of indices 11, 12:12+J and 12+2J:12+3J as `need_ij` without saying which need it
    means, and the two candidates differ numerically:
      (a) the *planned* need of PLAN section 2.7.2, `planner_io[s(i), j] * T_i` - a per-period
          quantity, constant within the period; or
      (b) the *production* need of PLAN section 2.6, `a_{s(i)j} * y_hat_ik` - a per-step quantity
          that depends on the effort just chosen, and is zero at zero effort.
    This module picks neither. WO-002 files an AMBIGUITY REPORT (CONTRACT rule 3,
    `workorders/AMBIGUITY_TEMPLATE.md`) naming both options; the resolution is recorded in
    `spec/CHANGELOG.md` and mirrored by `gosplan/env/obs.py` (WO-008), and only then are the golden
    files generated. Whichever is chosen, `need_ij == 0` must give a coverage field of exactly 1.0
    (edge case E3), not a division by zero.

    Scope note: PLAN section 2.4 sits outside the section 2.5-2.11 range of the WO-002 card, but
    the golden schema of PLAN section 11 records per-step observations, so the oracle must build
    them - independently of `gosplan/env/obs.py`, like everything else here.

    Binds: `tests/unit/test_obs.py` (layout equals `obs_spec(cfg)`; dimension `12 + 3J`; phase
    masking), T-B5 (`tests/behavioural/test_welfare_blindness.py`) and T-B7 (every golden file
    stores this vector).
    """
    raise NotImplementedError("PLAN section 2.4 - implemented in WO-002")


# ---------- conservation (PLAN section 2.11, test T-U1) ----------


def ref_conservation_residual(
    y_period: Vec,
    stock_prev: Vec,
    inputs_consumed: Mat,
    consumer: Goods,
    stock_next: Vec,
    holding_loss: Vec,
    cap_overflow: Vec,
    cfg: Config,
) -> Goods:
    """Per-good residual of the period conservation identity (PLAN section 11, test T-U1).

    Takes, all for one completed period: `y_period` `(N,)`, true output; `stock_prev` `(N,)`, own
    stock entering the period; `inputs_consumed` `(N, J)`, inputs consumed at PRODUCE steps;
    `consumer` `(J,)`, the final-demand sink's receipts; `stock_next` `(N,)`, own stock leaving the
    period; `holding_loss` `(N,)` and `cap_overflow` `(N,)`, the two sinks of PLAN section 2.11;
    and `cfg`. Returns: the residual per good, length `J`.

    Identity (PLAN section 11, T-U1, verbatim):

        sum y + sum S_prev = sum inputs consumed + sum consumer + sum S_next
                             + holding loss + cap overflow

    evaluated **per good**, where a quantity indexed by enterprise contributes to the good of that
    enterprise's sector and `inputs_consumed` contributes to the good consumed. The residual is
    LHS - RHS and must be below 1e-9 in absolute value for every good.

    A non-zero residual is a defect in the reference, never a tolerance to be loosened
    (`docs/ref_worked_example.md` section 3). This function exists so that `ref_period` can assert
    the identity on every period it simulates, which is the cheapest possible guard against a
    silent bookkeeping error in the oracle - and therefore in every golden file.

    Binds: T-U1 (`tests/unit/test_conservation.py`), to 1e-9, per good, per period.
    """
    raise NotImplementedError("PLAN section 2.11 - implemented in WO-002")


# ---------- schedule stage 0: DELIVER (PLAN section 2.5) ----------


def ref_deliver(
    state: RefState, cfg: Config, claim_history: list[Vec]
) -> tuple[RefState, RefPlannerView, Mat, Vec, Goods]:
    """Schedule stage 0 - the planner allocates last period's claims and goods physically move.

    Takes: `state` at the top of a period; `cfg`; `claim_history`, every completed period's claims
    (see `ref_make_planner_view`). Returns: `(state, view, deliv, fill, consumer)`.

    Sequence (PLAN section 2.5 stage 0, composing sections 2.7.2 and 2.7.3):

        view    = ref_make_planner_view(state, cfg, claim_history)
        alloc   = ref_allocate(view, cfg)
        state, deliv, fill, consumer = ref_ship(state, alloc, view.claims, cfg)
        state.last_fill = fill
        state.consumer_delivery = consumer

    In period 0 there is nothing to allocate: `claim_history` is empty, the claims are zeros, so
    `avail_j = 0`, `alloc` is all zeros, `fill_i = 1` by the `claimed_i = 0` guard, and no good
    moves. That is the recorded initial condition of `docs/ref_worked_example.md`, not a special
    case in the code.

    Phase-1 note on `delivery_timing`: at `uniform` every delivered unit is available at PRODUCE
    step 0, so `deliv` is credited to `X` here in full. Under the Phase-2 `stochastic` timing a
    unit instead arrives at step `k ~ Categorical(arrival_probs)` drawn with purpose `arrival`
    (PLAN section 2.6); the branch belongs here and is not exercised by any Phase-1 golden file.

    Stage 1 of PLAN section 2.5 - TRADE, section 2.13 - is a Phase-2 sketch and is deliberately
    absent from this module: `trade_surplus == 0` throughout Phase 1 (PLAN section 2.9.1), and the
    matching rule is frozen only at the Phase-2 spec revision (finding F14). A `ref_trade` stage is
    added then, with its own golden regeneration.

    Binds: T-U1 (the goods that move here are two terms of the identity), T-B3
    (`tests/behavioural/test_shortage_propagation.py`) and rows 0.1-0.13 of
    `docs/ref_worked_example.md`.
    """
    raise NotImplementedError("PLAN section 2.5 - implemented in WO-002")


# ---------- schedule stage 2: PRODUCE (PLAN section 2.5) ----------


def ref_produce(state: RefState, action: RefAction, cfg: Config) -> tuple[RefState, Vec, Vec, Mat]:
    """Schedule stage 2 - execute one PRODUCE step `k` for every enterprise (PLAN section 2.6).

    Takes: `state` at the start of step `k = state.k_step`; `action`, whose `effort` (and, in Phase
    2, `quality` and `invest`) dimensions are read; `cfg`. Returns: `(state, y, c,
    inputs_consumed)` - the updated state, this step's output `(N,)`, this step's cost `(N,)`, and
    the inputs consumed `(N, J)` (kept for T-U1).

    Formulas (PLAN section 2.6, verbatim), looped over `i` and then over `j`:

        y_hat_ik   = (A_{s(i)} * cap_i / M) * e_ik          # intended output at full coverage
        need_ikj   = a_{s(i)j} * y_hat_ik                   # ref_input_need
        H_ik       = ref_coverage(X_i, need_ik, omega_i, theta)
        eps_ik     ~ LogNormal(-sigma_{s(i)}**2 / 2, sigma_{s(i)})    mean 1
                     key = (seed_env, "yield", t, k, i)
        y_tilde_ik = y_hat_ik * H_ik * eps_ik
        y_ik       = y_tilde_ik * (1 - v_ik)                # v = invest fraction; Phase 1 v == 0
        X_ij      -= min(X_ij, a_{s(i)j} * y_tilde_ik)      # inputs consumed, capped at stock
        c_ik       = kappa * e_ik**2 + F * 1[e_ik > 0] + kappa_q * q_ik * e_ik   # P1: F=kappa_q=0

    with `omega_j = a_{s(i)j} / sum_j a_{s(i)j}` (and `omega` irrelevant when the row is all
    zeros), `M = cfg["incentive"]["steps_per_period"]`, `kappa = ...["effort_cost"]`, `theta =
    cfg["supply"]["input_complementarity"]`. Inputs are consumed against `y_tilde` - output
    **before** the investment diversion - and the consumption is capped at the stock on hand.

    State updates: `cum_output += y_ik`, `cum_cost += c_ik`, `quality_acc += q_ik` (Phase 1 inert),
    `pending_invest` receives `y_tilde_ik * v_ik` in Phase 2, and `k_step` advances (the driver
    owns the advance, not this function - see `ref_period`).

    The `eps` vector is drawn with one keyed call per `(t, k)` over all `i`, so the draw does not
    depend on the loop order (T-U6).

    Phase-2 toggles routed through this stage, all off in Phase 1: setup cost `F`; increasing
    returns `A_j(Kap) = A_j * (Kap / Kap_0)**alpha_irs`; capital accumulation `Kap_{t+1} = (1 -
    dep) * Kap_t + matured investment`; I-O drift `a <- a * exp(zeta)` (purpose `drift`) with
    `planner_io` held fixed; quality routed through the input bundle.

    Constraints: this stage may not reference reports, targets or rewards (WO-005 forbidden list,
    mirrored here), and every stochastic term goes through `ref_draw` (CONTRACT rule 9).

    Binds: `tests/unit/test_production.py` (yield mean 1 to 1e-3 over 1e5 draws; the `v` diversion;
    the cost formula; inputs consumed equal `a * y_tilde` capped at stock; `H = 1` on a zero `a`
    row), T-U7, T-U1, and rows 2.1-2.15 of `docs/ref_worked_example.md`.
    """
    raise NotImplementedError("PLAN section 2.6 - implemented in WO-002")


# ---------- schedule stage 3: REPORT (PLAN section 2.5) ----------


def ref_report(state: RefState, action: RefAction, cfg: Config) -> tuple[RefState, Vec, Vec, Vec]:
    """Schedule stage 3 - close the period's books and record each claim (PLAN sections 2.8, 2.11).

    Takes: `state` at the REPORT step; `action`, whose `report_ratio` and `input_request`
    dimensions are read; `cfg`. Returns: `(state, holding_loss, cap_overflow, period_output)` -
    the updated state and the three per-enterprise quantities the conservation identity and the
    logged metrics need (`period_output` is `cum_output` before it is reset).

    Formulas (PLAN section 2.8, verbatim):

        S_i <- (1 - h) * S_i + y_i            # holding loss on carried stock, THEN this period's y
        R_i  = clip(rho_i_report, 0, rho_max) * T_i

    Order matters and is checked: the holding loss `h = cfg["supply"]["holding_loss"]` applies to
    the stock carried in, not to the output just produced, so `holding_loss_i = h * S_i_before`.
    Stock above `S_max = cfg["tech"]["inventory_cap_mult"] * cap_i` is then lost (PLAN section
    2.11) and that overflow is returned so it stays visible in the conservation identity (edge case
    E12); the cap is applied after the output is added.

    At this step the agent has already observed `S_i` and `y_i` exactly (Phase 1, `self_obs_noise =
    0`), so the report is a choice made under full knowledge of the truth. That is a property of
    the information structure, not an instruction: CONTRACT rule 7 forbids any rule that pushes the
    claim in either direction.

    The stage also stores `last_report_ratio`, `last_report` (the claim in units, retained because
    the ratchet moves `T` later in the same period - PLAN section 2.5 stages 3 then 6) and
    `request`, clipped to `r_max * need_ij` with `r_max = cfg["tech"]["request_max_multiple"]`, and
    it records whether the report sat at `rho_max` (CONTRACT rule 8, flagged not clamped away).

    Binds: `tests/unit/test_reporting.py` (holding loss applied before `y` is added; report clipped
    to `rho_max`), T-B8 (`BOUND_BINDING`), T-U1, and rows 3.1-3.8 of
    `docs/ref_worked_example.md`.
    """
    raise NotImplementedError("PLAN section 2.8 - implemented in WO-002")


# ---------- schedule stage 4: AUDIT (PLAN section 2.5) ----------


def ref_audit(
    state: RefState, view: RefPlannerView, cfg: Config
) -> tuple[RefState, RefPlannerView, Vec]:
    """Schedule stage 4 - select audits, measure stock, charge the penalty (PLAN section 2.8).

    Takes: `state` after `ref_report`; `view`, the planner view for this period (its `audited` and
    `audit_meas` fields are still all False / zero); `cfg`. Returns: `(state, view, penalty)` -
    the state with `last_audited` and `last_penalty` written, the view **reissued** with `audited`
    and `audit_meas` filled (`RefPlannerView` is frozen, so a new record is built), and the penalty
    `(N,)`.

    Sequence (PLAN section 2.5 stage 4, composing sections 2.7.4 and 2.8):

        audited            = ref_select_audits(view, cfg, state.t_period)
        penalty, audit_meas = ref_audit_penalty(state, audited, cfg, state.t_period)
        view               = same view with audited and audit_meas filled
        state.last_audited  = audited
        state.last_penalty  = penalty

    The order is fixed: the selection is drawn before the measurement, and both are keyed on
    `(seed_env, purpose, t, i)` rather than on call order, so re-running the stage alone reproduces
    it exactly (T-U6).

    Binds: T-U8, `tests/unit/test_planner.py` (audit frequency) and rows 4.1-4.6 of
    `docs/ref_worked_example.md`.
    """
    raise NotImplementedError("PLAN section 2.8 - implemented in WO-002")


# ---------- schedule stage 5: REWARD (PLAN section 2.5) ----------


def ref_reward(
    state: RefState,
    view: RefPlannerView,
    cfg: Config,
    penalty: Vec,
    period_output: Vec,
    consumer: Goods,
) -> tuple[Vec, float, float, float]:
    """Schedule stage 5 - deliver the REPORT-step reward and log the period metrics (section 2.9).

    Takes: `state` after `ref_audit`; `view`, the reissued planner view; `cfg`; `penalty` `(N,)`
    from stage 4; `period_output` `(N,)`, the period's true production from stage 3; `consumer`
    `(J,)`, the final-demand receipts from stage 0. Returns: `(reward, val_measured, val_true,
    welfare)`.

    Sequence (PLAN section 2.5 stage 5, composing sections 2.9.1-2.9.3):

        m            = ref_fulfilment_measure(view, cfg)
        rho_i        = m_i / T_i
        reward       = ref_enterprise_reward(state, cfg, "report", None, penalty, trade_surplus)
                       with trade_surplus all zeros in Phase 1
        val_measured = ref_val_measured(state, cfg, view.measured_quality)
        val_true     = ref_val_true(state, cfg, period_output)
        welfare      = ref_welfare_true(consumer, cfg)

    The reward is exactly `scale * (B(rho_i) - penalty_i + trade_surplus_i)` and nothing else
    (CONTRACT rule 4). The three logged scalars are computed **after** the reward and are never
    inputs to it (CONTRACT rule 6): they exist for the ledger, the headline metrics of PLAN section
    2.9.4 and the golden record, and test T-B5 plants sentinels in them to prove no observation
    carries them.

    Note that the fulfilment ratio the bonus keys on comes from the *planner view* - the claim as
    the planner received it - not from the true state, so lag, aggregation and channel noise apply
    to the bonus as well as to the ratchet.

    Binds: T-B6 (independent recomputation of the five-term formula), T-U2, T-U3 and rows 5.1-5.10
    of `docs/ref_worked_example.md`.
    """
    raise NotImplementedError("PLAN section 2.9 - implemented in WO-002")


# ---------- schedule stage 6: TARGET (PLAN section 2.5) ----------


def ref_target(
    state: RefState, view: RefPlannerView, cfg: Config, initial_targets: Vec
) -> RefState:
    """Schedule stage 6 - apply the ratchet and the growth directive (PLAN section 2.7.1).

    Takes: `state` after the reward has been delivered; `view`, the period's planner view; `cfg`;
    `initial_targets`, the episode's `T_0` (the floor anchor). Returns: the state with `target`
    replaced by `ref_update_targets(view, cfg, initial_targets)`.

    The stage runs **after** stage 5, which is why `state.last_report` is retained through the
    period: the bonus is paid against the target the period was run under, and only then is the
    target moved for the next period (PLAN section 2.5, stages 3, 5, 6 in that order). Running the
    ratchet before the reward would pay the bonus against a target the agent never saw, and the
    period ordering is part of the specification, not an implementation convenience.

    Binds: T-U4, T-B2 and rows 6.1-6.7 of `docs/ref_worked_example.md`.
    """
    raise NotImplementedError("PLAN section 2.7.1 - implemented in WO-002")


# ---------- schedule stage 7: TERMINATE (PLAN section 2.5) ----------


def ref_terminate(state: RefState, cfg: Config) -> bool:
    """Schedule stage 7 - decide whether the episode ends after this period (PLAN section 2.12).

    Takes: `state` after the target update, with `state.t_period` the period just completed; `cfg`.
    Returns: `True` when the episode terminates here.

    Rule (PLAN section 2.12, verbatim):

        horizon_mode = "geometric"  (Phase 1)
            periods 0 .. P_min - 1 always run; from then on the episode continues with probability
            psi = cfg["incentive"]["tenure"] per period, with a hard cap at P_max periods
        horizon_mode = "fixed"
            the episode ends after exactly P_max periods

    with `P_min = cfg["tech"]["min_periods"]`, `P_max = cfg["tech"]["max_periods"]`. The
    continuation is a **single global draw** per period, not one per enterprise:
    `ref_draw(seed_env, "terminate", t, shape=(1,), dist="bernoulli", p=psi)`.

    The agent never observes periods remaining, so there is no end-game (finding F4): no
    observation field of PLAN section 2.4 encodes `t` or the termination draw, and T-B9 regresses
    every observation field on periods-remaining and requires a coefficient of approximately zero.

    Binds: T-B9 (`tests/behavioural/test_termination.py`: empirical continuation equals `psi`) and
    rows 7.1-7.4 of `docs/ref_worked_example.md`.
    """
    raise NotImplementedError("PLAN section 2.12 - implemented in WO-002")


# ---------- drivers ----------


def ref_period(
    state: RefState,
    actions: list[RefAction],
    cfg: Config,
    initial_targets: Vec,
    claim_history: list[Vec],
) -> tuple[RefState, list[RefStepRecord]]:
    """Run one complete plan period through the eight stages of PLAN section 2.5.

    Takes: `state` at the top of period `t = state.t_period`; `actions`, exactly `M + 1` joint
    actions (`M = cfg["incentive"]["steps_per_period"]` PRODUCE actions then one REPORT action);
    `cfg`; `initial_targets`, the episode's `T_0`; `claim_history`, the claims of every completed
    period, which this driver appends to at the REPORT step. Returns: `(state, records)` with one
    `RefStepRecord` per agent-step - `M + 1` of them.

    Order, verbatim from PLAN section 2.5, with no stage reordered or fused:

        0. DELIVER      ref_deliver        - allocation and physical delivery; X updated
        1. [P2] TRADE   absent in Phase 1  - section 2.13 sketch; trade_surplus == 0
        2. PRODUCE x M  ref_produce        - one agent-step each; reward -scale * c_ik
        3. REPORT       ref_report         - S <- (1-h)S + y; claim recorded
        4. AUDIT        ref_audit          - selection, measurement, penalty
        5. REWARD       ref_reward         - scale * (B(rho) - penalty); val and welfare logged
        6. TARGET       ref_target         - ratchet and growth directive
        7. TERMINATE?   ref_terminate      - geometric draw; writes state.alive

    Bookkeeping this driver owns rather than any stage: advancing `k_step` (0 .. M-1 at PRODUCE,
    `M` at REPORT) and `phase`; resetting `cum_output`, `cum_cost` and `quality_acc` to zero after
    the REPORT step; appending the period's claims to `claim_history`; incrementing `t_period`;
    building one `RefStepRecord` per agent-step with the observation from `ref_observation`, the
    reward from the stage that produced it, and `ref_state_digest(state)` taken **after** the
    stage completes.

    Self-check: after stage 7 the driver evaluates `ref_conservation_residual` for the period and
    raises if any good's residual exceeds 1e-9. The oracle checks its own books on every period it
    simulates - a golden file generated from an unbalanced period would be confidently wrong and
    the whole frozen suite would agree with it forever (finding F14).

    Binds: T-U1 (per period, per good), T-B7 (the records are the golden rows) and section 4 of
    `docs/ref_worked_example.md`, which requires at least two periods so that the target written at
    stage 6 and the claim recorded at stage 3 are both carried across the period boundary.
    """
    raise NotImplementedError("PLAN section 2.5 - implemented in WO-002")


def ref_rollout(
    cfg: Config,
    seed_env: int,
    seed_policy: int,
    policy: Policy,
    n_steps: int,
) -> list[RefStepRecord]:
    """Roll the reference dynamics forward for a fixed number of agent-steps.

    Takes: `cfg`; `seed_env`, the root environment seed (every draw is keyed from it, PLAN section
    2.15); `seed_policy`, the separate policy seed (CONTRACT rule 9), used only to build the
    `numpy.random.Generator` handed to `policy`; `policy`, a `Policy` callable; `n_steps`, the
    number of agent-steps to run (`ref/gen_golden.py` uses 30, PLAN section 11). Returns: the list
    of `RefStepRecord`s, one per agent-step executed.

    Sequence: build the opening state with `ref_initial_state(cfg, seed_env, seed_policy)` and the
    opening observation with `ref_observation`, then repeatedly call `policy(obs, phase, rng)` and
    advance one agent-step, driving periods through `ref_period`. Episodes run `M + 1` agent-steps
    per period; when `ref_terminate` fires (or `max_periods` is reached) the episode ends and a
    fresh one starts from `ref_initial_state` with the **same** `seed_env` and `seed_policy`, with
    the episode index appended to every subsequent draw key, so that a 30-step rollout is
    well-defined whatever the episode length. The returned list is therefore exactly `n_steps` long
    whether or not termination fired inside it.

    Determinism is total: `(cfg, seed_env, seed_policy, policy, n_steps)` fixes every value in the
    returned list bit for bit, which is what makes a golden file meaningful. Nothing here reads a
    clock, a process id, a global generator or an environment variable (CONTRACT rule 9).

    The policy sees `obs` and nothing else (CONTRACT rule 6): no `RefState`, no `RefPlannerView`,
    no `RefStepRecord`. `ref/gen_golden.py` supplies the two Phase-1 policies; `ref/` never imports
    `gosplan.agents`.

    Binds: T-B7 - `tests/golden/` replays these records against `GosplanEnv` under the same seeds
    and requires agreement to 1e-9 on observations and rewards, and exact agreement on the state
    digest.
    """
    raise NotImplementedError("PLAN section 2.5 - implemented in WO-002")


__all__ = [
    "Config",
    "Dist",
    "Draws",
    "Goods",
    "Mat",
    "Phase",
    "Policy",
    "Purpose",
    "RefAction",
    "RefPlannerView",
    "RefState",
    "RefStepRecord",
    "Vec",
    "ref_allocate",
    "ref_audit",
    "ref_audit_penalty",
    "ref_bonus",
    "ref_conservation_residual",
    "ref_coverage",
    "ref_deliver",
    "ref_draw",
    "ref_enterprise_reward",
    "ref_fulfilment_measure",
    "ref_initial_prices",
    "ref_initial_state",
    "ref_input_need",
    "ref_make_planner_view",
    "ref_observation",
    "ref_period",
    "ref_produce",
    "ref_report",
    "ref_reward",
    "ref_reward_scale",
    "ref_rollout",
    "ref_select_audits",
    "ref_ship",
    "ref_state_digest",
    "ref_target",
    "ref_terminate",
    "ref_update_targets",
    "ref_val_measured",
    "ref_val_true",
    "ref_welfare_true",
]
