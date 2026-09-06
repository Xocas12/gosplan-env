"""Reward, scaling, logged aggregates and the three headline metrics.

Realises: PLAN section 2.8 (the bonus schedule `B(rho)` and its smoothing `Lambda_w`), PLAN section
2.9.1 (the enterprise reward and the analytic reward scale), PLAN section 2.9.3 (`val_measured`,
`val_true` and true consumer welfare - logged, never observed) and PLAN section 2.9.4 (the three
dimensionless headline metrics). Owning work order: **WO-007** (Reporting and reward; MID-strong).

--------------------------------------------------------------------------------------------------
CONTRACT RULE 4, REPRODUCED IN FULL (`CONTRACT.md`; the two formula lines are transcribed into the
plain-ASCII pseudo-maths convention of PLAN section 10, and nothing else is altered):

    4. REWARD TERMS. The enterprise reward is exactly:
       production step:  -scale * c_ik
       report step:       scale * (B(rho) - 1[audited] * Pen + trade_surplus)
       No per-step shaping, no auxiliary reward, no curiosity term, no potential-based term.
       No running reward normalisation (running statistics change the effective reward over
       training and, with heavy-tailed penalties, shrink the notch in normalised units).
       Per-batch advantage normalisation inside PPO is permitted. scale = reward_scale(cfg),
       computed analytically from the configuration.

Consequences that bind every session touching this file:

  * There is no fifth term. Effort cost is a real cost paid when it is incurred, not shaping;
    `trade_surplus` is identically zero throughout Phase 1 and becomes non-zero only when the
    Phase-2 trade mechanism of PLAN section 2.13 is switched on.
  * `reward_scale(cfg) = 1 / B_cfg(rho_ref = 1.1)` IS ANALYTIC FROM THE CONFIGURATION AND NEVER
    FROM RUNNING STATISTICS. It is a pure function of `cfg`, constant for the whole run, computable
    before a single step is taken, and written into the manifest with the configuration.
  * NO RUNNING REWARD NORMALISATION IS PERMITTED anywhere - not here, not in the PPO adapter, not
    in the training harness. A `RunningMeanStd` wrapper on rewards is a rule-4 violation, and
    `tests/unit/test_ppo_adapter.py` checks the wrapped object by inspection for exactly that
    (WO-017). The reason is substantive rather than stylistic: running statistics change the
    effective reward over the course of training, and with heavy-tailed penalties they shrink the
    notch in normalised units - which is the very quantity the Phase-1 experiments measure.
  * PER-BATCH ADVANTAGE NORMALISATION INSIDE PPO *IS* PERMITTED, and Phase 1 has it switched on
    (PLAN section 12.3, WO-017). Normalising advantages within a batch does not change the reward
    function; normalising rewards across time does.
--------------------------------------------------------------------------------------------------

CONTRACT RULE 6 (WELFARE BLINDNESS). `welfare_true` and `val_measured` are computed here because
they are the logged outputs of PLAN section 2.9.3, and they are logged and nothing else: they never
appear in any observation, in any reward term, in any agent input, or in any planner rule. The PPO
adapter's forward pass takes `obs` only. Test T-B5 in
`tests/behavioural/test_welfare_blindness.py` plants sentinel values in these quantities and asserts
they appear in no observation, and `gosplan/env/obs.py` names them on its "never in any observation"
list. `enterprise_reward` in particular may not read either of them, nor `consumer`, nor any other
true aggregate beyond the three per-period quantities in its own signature.

THE FULFILMENT MEASURES OF PLAN SECTION 2.9.2. The ratio `rho_i = m_i / T_i` that `bonus` is
evaluated at, and that the ratchet of PLAN section 2.7.1 keys on, comes from
`gosplan.env.planner.fulfilment_measure(view, cfg)`, selected by `cfg.incentive.objective_metric`:

    val               m_i = R_i                                                  (Phase 1)
    net_output        m_i = R_i - sum_j p_j * alloc_ij / p_{s(i)}   # net of allocated inputs,
                                                                   # valued at plan prices
    quality_weighted  m_i = R_i * q_hat_i,  q_hat_i = 1 + mu * (qbar_i - 1)

with `mu = cfg.information.quality_measurability`. `welfare` IS NOT AN OPTION (PLAN section 2.9.2,
finding F6): the planner cannot key on a quantity it does not observe, and adding such a branch
would violate CONTRACT rule 6 even if no configuration selected it. That is also why the measure is
computed on the planner's side, from a `PlannerView`, and never here.

CONTRACT RULE 9 (RNG). Nothing in this module is stochastic. If a future revision needs a draw it
goes through `gosplan.rng.draw`; no direct `numpy.random` or `jax.random` call may appear anywhere
in `gosplan/env/`.

Binding to the frozen interface. `spec/spec.py` is the frozen interface (CONTRACT rule 1) and the
callables below carry its names, argument names, argument order and return types exactly; the three
headline metrics of PLAN section 2.9.4 are not in `spec/spec.py` v0 and are declared here for the
first time, so the lead records them in `spec/CHANGELOG.md` at the v1 freeze (WO-013).
`spec/spec.py` is not an importable package, so the runtime dataclasses live in the `gosplan`
package - `EnvConfig` and the arm configs in `gosplan/config.py` (WO-003), `State` in
`gosplan/env/state.py` (WO-009) - and each MUST stay field-for-field identical to its `spec/spec.py`
declaration, which `tests/unit/test_spec_imports.py` enforces. They are imported under
`TYPE_CHECKING` so this module stays importable while its siblings are skeletons.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - types only; see the binding note in the module docstring
    from gosplan.config import EnvConfig, Phase
    from gosplan.env.state import State

Array = np.ndarray
"""Alias for every numeric array in this module (PLAN section 10), mirroring `spec.spec.Array`. The
Phase-2 JAX port (WO-029) substitutes its own array type behind the same name, so no signature here
may depend on a numpy-only method."""


def bonus(rho: Array, cfg: EnvConfig) -> Array:
    """Bonus schedule on the fulfilment ratio, in ratio units (PLAN section 2.8).

    Takes: `rho` `(N,)`, the fulfilment ratio `m_i / T_i` with `m_i` from
    `gosplan.env.planner.fulfilment_measure` (PLAN section 2.9.2), and `cfg`. Returns: `B(rho)`
    `(N,)`, in the same units as the penalty of PLAN section 2.8 - ratio units, per finding F9, so
    that `beta` and `pen` are commensurable.

    Formula (PLAN section 2.8, verbatim):

        Lambda_w(x) = 1[x >= 0]                if w = 0        # strict >=, a true Heaviside
                    = 1 / (1 + exp(-x / w))    if w > 0
        B(rho)      = beta * Lambda_w(rho - 1) + s * clip(rho - 1, 0, rho_cap - 1)

    with `beta = cfg.incentive.notch_height`, `w = cfg.incentive.notch_width`,
    `s = cfg.incentive.overfulfilment_slope` and `rho_cap = cfg.incentive.overfulfilment_cap`.

    The `Lambda_w` smoothing is the counterfactual knob of the whole Phase-1 design. At `w = 0` it
    is a true discontinuity - a strict `>=` Heaviside, so `Lambda_0(0) = 1` and an enterprise
    reporting exactly at target receives the full notch. At `w > 0` it is the logistic
    `1 / (1 + exp(-x / w))`, which is smooth everywhere and equals `1/2` at `rho = 1`; `w` is
    simultaneously the manipulation-strength knob of the estimator-bias study (PLAN section 7.2).
    The implementation must branch on `w == 0` exactly rather than letting `w` approach zero
    numerically, because `exp(-x / w)` overflows long before the limit is reached.

    `rho_cap = inf` means no cap at all, hence no kink: the implementation must NOT clip in that
    branch (WO-007 notes), since `clip(x, 0, inf)` is the correct limit but must be reached by a
    branch, not by feeding `inf` into a clip that a JAX port would have to trace.

    Named configurations (PLAN section 2.8), which are the arms of the Phase-1 experiments:

        notched                 w = 0,    rho_cap = 1.2     # a discontinuity at 1 and a kink at
                                                            # the cap; the Phase-1 baseline
        smooth counterfactual   w = 0.25, rho_cap = inf     # no discontinuity and no kink anywhere
        kink-only               w = 0.25, rho_cap = 1.2     # isolates kink bunching at the cap

    The smooth arm shares `beta` and `s` with the notched arm: it is *equal-parameter*, not
    equal-expected-value, and the single-enterprise DP of PLAN section 5 supplies the exact
    predicted distribution under both. Re-tuning `beta` or `s` to equalise expected bonus across
    arms would destroy the comparison the G2 criteria of PLAN section 4.5 rest on.

    Binds: test T-U3 in `tests/unit/test_reward.py` - `bonus` is monotone non-decreasing in `rho`;
    discontinuous at `rho = 1` if and only if `w = 0`; continuous with continuous derivative if and
    only if `w > 0` and `rho_cap = inf`. Owning WO: **WO-007**.
    """
    raise NotImplementedError("PLAN section 2.8 - implemented in WO-007")


def reward_scale(cfg: EnvConfig) -> float:
    """Analytic per-configuration reward scale (PLAN section 2.9.1; finding F9).

    Takes: `cfg`. Returns: `scale = 1 / B_cfg(rho_ref = 1.1)`, a single float computed from the
    configuration ALONE - a pure function of `cfg`, evaluated by calling `bonus` at the reference
    ratio `rho_ref = 1.1`, and NEVER from running statistics of any kind.

    Why it exists: it makes the bonus at 110% fulfilment equal to 1 in every configuration, so a
    sweep over `beta` changes the economics - the notch relative to `pen` and `kappa` - and not the
    gradient magnitude the optimiser sees. Without it, `notch_height` would confound an economic
    treatment with a learning-rate-like nuisance and the arms of PLAN section 4.3 would not be
    comparable.

    CONTRACT rule 4 forbids running reward normalisation outright, because running statistics change
    the effective reward over training and, with heavy-tailed penalties, shrink the notch in
    normalised units. Per-batch advantage normalisation inside PPO is permitted and is on in Phase 1
    (WO-017). This function is the only normalisation in the system; it is constant for the whole
    run and is recorded in the run manifest alongside the configuration hash (CONTRACT rule 10).

    Degenerate configurations: `B_cfg(1.1) = 0` (for instance `beta = 0` together with `s = 0`, the
    zero-incentive configuration the DP tests of WO-014 use) makes the scale undefined; the
    implementation raises rather than silently returning `inf`, and `EnvConfig.validate` is the
    place a configuration is rejected, not this function.

    Binds: test T-U2 in `tests/unit/test_reward.py` - `reward_scale(cfg) * bonus(1.1, cfg) == 1` to
    floating-point tolerance, for every configuration in the test matrix, including the notched,
    smooth-counterfactual and kink-only arms of PLAN section 2.8. Owning WO: **WO-007**.
    """
    raise NotImplementedError("PLAN section 2.9.1 - implemented in WO-007")


def enterprise_reward(
    state: State,
    cfg: EnvConfig,
    phase: Phase,
    cost: Optional[Array],
    penalty: Optional[Array],
    trade_surplus: Optional[Array],
) -> Array:
    """The only quantity any learner receives (PLAN section 2.9.1; CONTRACT rule 4).

    Takes: `state`; `cfg`; the current `phase`; and the three period quantities - `cost` `(N,)` at a
    PRODUCE step (from `gosplan.env.production.produce_step`), `penalty` `(N,)` and `trade_surplus`
    `(N,)` at the REPORT step (from `gosplan.env.reporting.audit_and_penalise` and, in Phase 2, from
    `gosplan.env.trade.match_trades`). Each may be `None` in the phase where it does not apply, and
    a `None` in the phase where it *does* apply is an error rather than an implied zero. Returns:
    `r` `(N,)`.

    Formula (PLAN section 2.9.1, verbatim):

        PRODUCE step k:   r_ik = - scale * c_ik
        REPORT step:      r_i  =   scale * ( B(rho_i) - penalty_i + trade_surplus_i )
        scale             = reward_scale(cfg)             # analytic, per configuration

    with `trade_surplus == 0` throughout Phase 1, `penalty_i = 1[audited_i] * Pen_i` already gated
    by `audit_and_penalise`, and `rho_i = m_i / T_i` where `m_i` is the fulfilment measure of PLAN
    section 2.9.2 computed by `gosplan.env.planner.fulfilment_measure` from the planner's view.

    THESE ARE THE ONLY TERMS. CONTRACT rule 4 forbids per-step shaping, auxiliary rewards, curiosity
    terms, potential-based terms and running reward normalisation. Concretely, none of the following
    may appear here or anywhere downstream: a bonus for truthful reporting, a penalty for stock
    held, a term in `fill`, a term in delivered inputs, an entropy bonus added to the environment
    reward (the PPO entropy coefficient is an optimiser setting, not a reward term), a
    potential-based shaping term even though it would leave the optimal policy invariant, or any
    per-step reward at a PRODUCE step other than `-scale * c_ik`. Effort cost is a real cost paid
    when it is incurred - it is in the formula, so it is not shaping.

    CONTRACT rule 6: nothing in this function may read `welfare_true`, `val_measured`, `val_true`,
    `consumer`, or any other enterprise's true quantities. The `state` argument is present for the
    targets and the recorded report that define `rho_i`, and for nothing else.

    CONTRACT rule 7: no term here may be conditioned on padding, shaving, storming, hoarding, blat
    or quality degradation, whether by name or by a predicate that happens to detect one.

    Binds: test T-B6 in `tests/behavioural/` - the reward is recomputed independently from the
    five-term formula on random states and must agree exactly - and the T-U2 / T-U3 scale and
    monotonicity tests through `bonus` and `reward_scale`. Owning WO: **WO-007**.
    """
    raise NotImplementedError("PLAN section 2.9.1 - implemented in WO-007")


def val_measured(state: State, cfg: EnvConfig) -> float:
    """The planner-side output aggregate, in plan prices (PLAN section 2.9.3).

    Takes: `state` at the end of a period, and `cfg`. Returns: the scalar

        val_measured_t = sum_i p_{s(i)} * R_i * q_hat_i

    with `p = state.plan_prices`, `s(i) = cfg.supply.sector_of`, `R_i = state.last_report` (the
    claim in units) and `q_hat_i = 1 + mu * (qbar_i - 1)` for
    `mu = cfg.information.quality_measurability` (Phase 1: `q_hat = 1`). This is what the planning
    system believes it produced - the number that would appear in the plan-fulfilment report - and
    it is exactly as fictitious as the claims that fed it.

    LOGGED ONLY. CONTRACT rule 6: it never appears in any observation, reward or agent input; test
    T-B5 asserts this with sentinels, and `gosplan/env/obs.py` names it on its "never in any
    observation" list. It is one half of `padding_index = val_measured / val_true` (PLAN section
    2.9.4) and it is the numerator of the first term of `specification_gap`. Owning WO: **WO-007**.
    """
    raise NotImplementedError("PLAN section 2.9.3 - implemented in WO-007")


def val_true(state: State, cfg: EnvConfig) -> float:
    """The true output aggregate, in plan prices (PLAN section 2.9.3).

    Takes: `state` at the end of a period, and `cfg`. Returns: the scalar

        val_true_t = sum_i p_{s(i)} * y_i * qbar_i

    where `y_i = state.cum_output` is the period's true production, `qbar_i = state.quality_acc` its
    period-average quality (Phase 1: 1), `p = state.plan_prices` and `s(i) = cfg.supply.sector_of`.
    The same price vector and the same index set as `val_measured`, so their ratio is a clean
    dimensionless measure of fictitious output.

    LOGGED ONLY, exactly as `val_measured` (CONTRACT rule 6). The ratio
    `val_measured / val_true` is the `padding_index` of PLAN section 2.9.4 and is at least 1
    whenever output is fictitious. Owning WO: **WO-007**.
    """
    raise NotImplementedError("PLAN section 2.9.3 - implemented in WO-007")


def welfare_true(consumer: Array, cfg: EnvConfig) -> float:
    """Consumer welfare from this period's final deliveries (PLAN section 2.9.3).

    Takes: `consumer` `(J,)`, the final-demand sink's receipts returned by
    `gosplan.env.planner.deliver`, and `cfg`. Returns: the scalar CES index

        rho_ces   = (sigma_c - 1) / sigma_c
        welfare_t = ( sum_j alpha_j * consumer_j**rho_ces )**(1 / rho_ces)

    with `alpha_j = cfg.supply.ces_alpha` and `sigma_c = cfg.supply.ces_sigma` (Phase 1: 0.8,
    complements-leaning, so `rho_ces < 0` and a zero in any good drives the index toward zero -
    which is how propagated shortage shows up in welfare at all).

    The `sigma_c -> 1` limit is the Cobb-Douglas index `prod_j consumer_j**alpha_j` and MUST be
    implemented as an explicit branch (WO-007 must-pass list): at `sigma_c = 1` the exponent
    `rho_ces` is 0 and the closed form is a `0/0` that no amount of floating point recovers. The
    branch is tested directly against the Cobb-Douglas product in `tests/unit/test_reward.py`.
    `W = mean_t welfare_t` over the measurement window of PLAN section 4.4 (periods `t >= 2`, which
    excludes the target-initialisation transient; under geometric termination there is no
    end-of-episode exclusion).

    LOGGED ONLY, and the strictest case of CONTRACT rule 6: no agent, no planner rule and no reward
    term may read it, and `welfare` is deliberately not an option in `ObjectiveMetric` for exactly
    that reason (PLAN section 2.9.2, finding F6). It is the numerator of
    `welfare_ratio = W / W_oracle` (PLAN section 2.9.4), where `W_oracle` comes from the
    expected-value MIP oracle of PLAN section 6.2; Phase 1 uses `W_truthful_max` as a clearly
    labelled placeholder denominator. Owning WO: **WO-007**.
    """
    raise NotImplementedError("PLAN section 2.9.3 - implemented in WO-007")


def padding_index(val_measured_window: Array, val_true_window: Array) -> float:
    """Headline metric 1 of PLAN section 2.9.4: how much of measured output is fictitious.

    Takes: `val_measured_window` and `val_true_window`, each a `(P,)` array of the per-period
    scalars returned by `val_measured` and `val_true` over the measurement window of PLAN section
    4.4 (periods `t >= 2` of each episode). Returns: the dimensionless scalar

        padding_index = val_measured / val_true

    aggregated as the RATIO OF WINDOW MEANS - `mean_t val_measured_t / mean_t val_true_t` - not the
    mean of per-period ratios, matching the order of aggregation PLAN section 2.9.3 states
    explicitly for `W = mean_t welfare_t` and therefore for `welfare_ratio`. Aggregating in the
    other order would weight low-output periods arbitrarily heavily.

    It is at least 1 whenever output is fictitious and equals 1 exactly under truthful reporting,
    which is what makes it a pipeline check rather than a Claim-A phenomenon (PLAN section 1.1:
    padding is a direct optimum of the reward under agent control, so its absence indicates a broken
    optimiser rather than an interesting result).

    Both inputs are CONTRACT rule 6 quantities: they are logged and never observed. This function is
    called by the metrics layer (`gosplan/metrics/`) and by lead-run experiments, never by an agent,
    a policy or a reward term.

    Standing robustness check (PLAN sections 2.9.4, 7.5): every headline table is recomputed under
    three perturbed price vectors, `p_j * exp(u_j)` with `u ~ N(0, 0.3**2)` at fixed seeds. That
    recomputation happens in `gosplan/experiments/price_sensitivity.py`, by re-evaluating
    `val_measured` and `val_true` under the perturbed prices and calling this function again - never
    by rescaling the index here.

    Binds: `tests/unit/test_reward.py` - the index is exactly 1 on a truthful trajectory, and
    exceeds 1 whenever any claim exceeds the corresponding true output. Owning WO: **WO-007**.
    """
    raise NotImplementedError("PLAN section 2.9.4 - implemented in WO-007")


def welfare_ratio(welfare_window: Array, welfare_oracle: float) -> float:
    """Headline metric 2 of PLAN section 2.9.4: realised welfare against the oracle's.

    Takes: `welfare_window`, a `(P,)` array of the per-period `welfare_true` scalars over the
    measurement window of PLAN section 4.4 (periods `t >= 2`); and `welfare_oracle`, the scalar
    `W_oracle`. Returns: the dimensionless scalar

        W             = mean_t welfare_t            # PLAN section 2.9.3
        welfare_ratio = W / W_oracle

    `W_oracle` is the welfare of the non-anticipative expected-value MIP oracle of PLAN section 6.2,
    solved once per configuration; the clairvoyant per-seed bound is reported alongside it as an
    upper bound only and is never substituted here. IN PHASE 1 THE DENOMINATOR IS `W_truthful_max`,
    a clearly labelled placeholder, and every Phase-1 table carrying this metric must say so - PLAN
    section 1.2 makes a Phase-1 artefact inadmissible as evidence for Claim B, and an unlabelled
    denominator is how that boundary gets crossed by accident.

    Both the numerator and the denominator are CONTRACT rule 6 quantities: logged, never observed,
    never in any reward term. `welfare_oracle <= 0` is a configuration or solver failure and must
    raise rather than produce a signed ratio.

    Binds: `tests/unit/test_reward.py` (the ratio is 1 when the trajectory equals the oracle's, and
    the `sigma_c -> 1` branch of `welfare_true` propagates through unchanged) and the Phase-1 gate
    reporting of PLAN section 4.5. Owning WO: **WO-007**.
    """
    raise NotImplementedError("PLAN section 2.9.4 - implemented in WO-007")


def specification_gap(
    val_measured_window: Array,
    val_oracle: float,
    welfare_window: Array,
    welfare_oracle: float,
) -> float:
    """Headline metric 3 of PLAN section 2.9.4: measured success minus real success.

    Takes: `val_measured_window`, a `(P,)` array of per-period `val_measured` scalars over the
    measurement window of PLAN section 4.4; `val_oracle`, the oracle plan's own measured output
    aggregate from PLAN section 6.2 (Phase 1: the same clearly labelled truthful-maximum
    placeholder that `welfare_ratio` uses); `welfare_window`, the `(P,)` per-period `welfare_true`
    scalars; and `welfare_oracle`, `W_oracle`. Returns: the dimensionless scalar

        specification_gap = val_measured / val_oracle - W / W_oracle

    with each ratio formed from window means, exactly as in `padding_index` and `welfare_ratio`, so
    the two terms are commensurable. It is 0 when the system is honest and efficient: the gap is the
    wedge between what the plan measures itself achieving and what consumers actually receive, and
    it is the single number this environment exists to make computable.

    A SIGN CHANGE UNDER THE PRICE PERTURBATION IS REPORTED, NEVER SUPPRESSED. PLAN sections 2.9.4
    and 7.5 require every headline table to be recomputed under three perturbed price vectors,
    `p_j * exp(u_j)` with `u ~ N(0, 0.3**2)` at fixed seeds; if the sign of this metric flips under
    any of them, the flip is part of the result. Selecting a price vector that preserves the sign,
    or reporting only the unperturbed value, is a reporting failure of the same kind CONTRACT rule 8
    forbids for bounds.

    Every input is a CONTRACT rule 6 quantity: logged, never observed by any agent, never in any
    reward term or planner rule. `val_oracle <= 0` or `welfare_oracle <= 0` is a solver or
    configuration failure and must raise.

    Binds: `tests/unit/test_reward.py` - the gap is 0 on a truthful, oracle-matching trajectory, and
    strictly positive when claims are padded while deliveries are unchanged - and the
    price-sensitivity table of PLAN section 7.5. Owning WO: **WO-007**.
    """
    raise NotImplementedError("PLAN section 2.9.4 - implemented in WO-007")
