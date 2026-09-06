"""Reporting and audit: closing the period's books, measuring stock, charging the penalty.

Realises: PLAN section 2.8 (REPORT and AUDIT steps 3 and 4 of the period schedule of PLAN section
2.5), together with the inventory rules of PLAN section 2.11 that the REPORT step applies. Owning
work order: **WO-007** (Reporting and reward; MID-strong).

Scope split within WO-007. PLAN section 2.8 also defines the bonus schedule `B(rho)` and its
smoothing `Lambda_w`; those live in `gosplan/env/reward.py` beside `reward_scale` and
`enterprise_reward`, because they are reward terms and CONTRACT rule 4 governs them as a set. This
module holds only the two functions that turn the agent's report action and the planner's audit
selection into recorded state and a penalty: `process_reports` and `audit_and_penalise`. The audit
*selection* itself is a planner rule and lives in `gosplan/env/planner.py`
(`select_audits`, PLAN section 2.7.4).

THE AUDIT COMPARES THE CLAIM TO STOCK ON HAND, NEVER TO THE PERIOD'S PRODUCTION. That single choice
is the substantive content of PLAN section 2.8: accumulated stock protects against audits, exactly
as it historically did, which is why `penalty_arg` and the holding loss `h` are the mechanism
parameters behind the held-out hidden-reserves phenomenon (PLAN sections 4.1 row 7 and 4.2). An
implementation that audits against `cum_output`, or against `y_i`, is wrong in a way no unit test
of the arithmetic would catch, so it is stated here as the first fact about the module.

CONTRACT RULE 7 (NO HARD-CODED PATHOLOGY). Shaving (a claim below stock) and padding (a claim above
stock) are consequences of these formulas plus the delivery rule of PLAN section 2.7.3 - never
rules. Nothing in this module may branch on, name, detect, reward or penalise either behaviour
beyond the symmetric penalty arithmetic below, whose asymmetry is a *configuration choice*
(`penalty_arg`) and not a special case for a named phenomenon.

CONTRACT RULE 8 (BOUNDS ARE RESULTS). `report_ratio` is bounded at `rho_max =
cfg.tech.report_max_ratio` (Phase 1: 10.0). `process_reports` records, per enterprise per period,
whether the report sat at the bound; `gosplan/metrics/ledger.py` (WO-011) aggregates that into the
fraction of reports at `rho_max`, and above 1% the run manifest is flagged `BOUND_BINDING` and the
result is reported *with the flag* (test T-B8). The bound is never silently widened or narrowed to
make a result look better, and reports at `rho_max` are included in the histograms of PLAN section
4.4, flagged rather than dropped.

CONTRACT RULE 9 (RNG). The one stochastic term here is the audit measurement error of PLAN section
2.8, drawn through `gosplan.rng.draw` with purpose `auditnoise`. No direct `numpy.random` or
`jax.random` call may appear anywhere in `gosplan/env/`.

Binding to the frozen interface. `spec/spec.py` is the frozen interface (CONTRACT rule 1) and the
two callables below carry its names, argument names, argument order and return types exactly.
`spec/spec.py` is not an importable package, so the runtime dataclasses live in the `gosplan`
package - `EnvConfig` and the arm configs in `gosplan/config.py` (WO-003), `State` and
`EnterpriseAction` in `gosplan/env/state.py` (WO-009) - and each MUST stay field-for-field identical
to its `spec/spec.py` declaration, which `tests/unit/test_spec_imports.py` enforces. They are
imported under `TYPE_CHECKING` so this module stays importable while its siblings are skeletons.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - types only; see the binding note in the module docstring
    from gosplan.config import EnvConfig
    from gosplan.env.state import EnterpriseAction, State

Array = np.ndarray
"""Alias for every numeric array in this module (PLAN section 10), mirroring `spec.spec.Array`. The
Phase-2 JAX port (WO-029) substitutes its own array type behind the same name, so no signature here
may depend on a numpy-only method."""


def process_reports(state: State, action: EnterpriseAction, cfg: EnvConfig) -> State:
    """Close the period's books and record the enterprise's claim (PLAN section 2.8).

    Takes: `state` at the REPORT step, `action` (whose `report_ratio` and `input_request` dimensions
    are read; every other dimension is ignored, not rejected), and `cfg`. Returns: the updated
    `state`.

    Formulas (PLAN section 2.8, verbatim):

        S_i <- (1 - h) * S_i + y_i            # holding loss on carried stock, THEN this period's y
        R_i  = clip(rho_i_report, 0, rho_max) * T_i

    with `h = cfg.supply.holding_loss` (Phase 1: 0.02), `y_i = state.cum_output` (the true output
    accumulated over this period's `M` PRODUCE steps), `rho_i_report = action.report_ratio[i]`,
    `rho_max = cfg.tech.report_max_ratio` (Phase 1: 10.0) and `T_i = state.target`.

    ORDER MATTERS AND IS THE POINT. The holding loss applies to the stock *carried in* from previous
    periods, not to output just produced: `(1 - h) * S_i` is evaluated first and `y_i` is added
    afterwards, undiscounted. Applying `h` to `S_i + y_i` would tax this period's production and
    change the economics of retaining output, so `tests/unit/test_reporting.py` checks the ordering
    directly with `h > 0`, `S > 0` and `y > 0`.

    Inventory cap (PLAN section 2.11): stock above `S_max = cfg.tech.inventory_cap_mult * cap_i` is
    lost, and the lost quantity - the *overflow* - is recorded rather than dropped silently, so it
    stays visible on the left-hand side of the per-period conservation identity of test T-U1.

    At this step the agent has already observed `S_i` and `y_i` exactly (Phase 1, where
    `information.self_obs_noise = 0`), so the report is a choice made under full knowledge of the
    truth. Nothing in this function may nudge, clip toward, or otherwise shape the agent's chosen
    ratio beyond the box clip to `[0, rho_max]` that `action_spec` already declares.

    The function also stores, on the returned state:

        last_report_ratio   the clipped ratio actually used
        last_report         R_i in units - retained because the ratchet moves `T` later in the same
                            period (PLAN section 2.5, step 3 REPORT then step 6 TARGET), so the
                            claim must survive the target update that follows it
        request             `action.input_request`, clipped to `r_max * need_ij` with
                            `r_max = cfg.tech.request_max_multiple` and `need_ij` the current
                            planned need; logged in Phase 1 and inert while `alloc_eta_request = 0`
        at_bound            per enterprise, whether the report sat at `rho_max` (CONTRACT rule 8)

    The `at_bound` record is the whole mechanism behind CONTRACT rule 8: `gosplan/metrics/ledger.py`
    turns it into the fraction of reports at the bound, and above 1% the manifest is flagged
    `BOUND_BINDING`.

    Binds: `tests/unit/test_reporting.py` - the holding loss is applied before `y` is added, and the
    report is clipped to `rho_max` - and test T-B8 in the behavioural suite, which forces `rho = 10`
    in more than 1% of reports and asserts the `BOUND_BINDING` flag appears. Owning WO: **WO-007**.
    """
    raise NotImplementedError("PLAN section 2.8 - implemented in WO-007")


def audit_and_penalise(state: State, audited: Array, cfg: EnvConfig, t: int) -> Array:
    """Measure audited stock and charge the penalty (PLAN section 2.8).

    Takes: `state` after `process_reports` has run (so `S_i` already carries the holding loss and
    this period's output, and `R_i` is recorded), `audited` `(N,)` bool from
    `gosplan.env.planner.select_audits`, `cfg`, and the plan period `t`, which keys the draw.
    Returns: `penalty` `(N,)`, zero wherever `audited` is False.

    Formulas (PLAN section 2.8, verbatim):

        S_hat_i   = S_i * exp(nu_i),  nu_i ~ N(0, sigma_aud**2)
                    key = (seed_env, "auditnoise", t, i)
        f_i       = max(0, R_i - S_hat_i) / T_i      if penalty_arg = positive_part   (Phase 1)
                  = |R_i - S_hat_i| / T_i            if penalty_arg = absolute
        Pen_i     = pen * f_i                        if penalty_form = proportional   (Phase 1)
                  = pen * 1[f_i > 0]                 if penalty_form = fixed
        penalty_i = 1[audited_i] * Pen_i

    with `sigma_aud = cfg.information.audit_noise` (Phase 1: 0.0, so the measurement is exact),
    `pen = cfg.incentive.penalty_scale`, `penalty_arg = cfg.incentive.penalty_arg` and
    `penalty_form = cfg.incentive.penalty_form`; `R_i = state.last_report`, `T_i = state.target`,
    and `S_i = state.inv_output` as it stands after `process_reports`.

    Four facts the implementation must get right, each of them load-bearing:

    1. THE MEASUREMENT NOISE IS MULTIPLICATIVE AND LOG-NORMAL. `nu_i` is normal with mean 0, and
       `S_hat_i = S_i * exp(nu_i)`; the noise is therefore biased upward in levels
       (`E[S_hat] = S * exp(sigma_aud**2 / 2)`) and that is deliberate and must not be
       "corrected". At `sigma_aud = 0` the branch must return `S_hat = S` exactly, not `S * exp(0)`
       computed through a wasted draw - but the draw must still be structured so that turning
       `sigma_aud` on changes nothing else (CONTRACT rule 9: purpose `auditnoise`,
       key `(seed_env, "auditnoise", t, i)`).
    2. THE AUDIT COMPARES `R_i` TO STOCK ON HAND, never to `state.cum_output` and never to the
       period's production. Accumulated stock is what protects against an audit; this is the
       mechanism behind the held-out hidden-reserves phenomenon (PLAN sections 4.1, 4.2).
    3. `penalty_arg` SETS THE DIRECTION. Under `positive_part` (Phase 1) an under-report - any
       `R_i <= S_hat_i` - incurs exactly zero penalty, so under-reporting is never punished by the
       audit; under `absolute` both directions are punished symmetrically. Test T-U8 checks exactly
       this contrast.
    4. THE PENALTY IS IN RATIO UNITS. Both `penalty_arg` branches divide by `T_i`, per finding F9,
       so `pen` is commensurable with the bonus height `beta` across configurations and
       `audit_rate * penalty_scale` is the compound quantity the G2 padding-elasticity criterion
       sweeps (PLAN section 4.5).

    The final gate `1[audited_i]` is applied last and is unconditional: an enterprise that was not
    audited receives penalty 0 regardless of how far its claim sits from its stock, and no
    information about `S_hat_i` may leak to an unaudited enterprise through any channel.

    This function computes a penalty; it does not apply it. The penalty enters the reward only
    through `gosplan.env.reward.enterprise_reward`, as the `- penalty_i` term of the five-term
    formula (CONTRACT rule 4).

    Binds: test T-U8 in `tests/unit/test_reporting.py` - `positive_part` gives exactly 0 for any
    under-report while `absolute` does not, and `audited = False` gives 0 regardless - plus the
    audit-against-stock check in the same file. Owning WO: **WO-007**.
    """
    raise NotImplementedError("PLAN section 2.8 - implemented in WO-007")
