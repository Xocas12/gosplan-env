"""The `gosplan.env` package: the environment of PLAN sections 2.2-2.15.

Realises: PLAN section 8 (repository layout), which gives this package one module per numbered
block of the environment specification. Owning work order: **WO-009** (step function and env
wrapper; LEAD).

Modules, and the PLAN section each one realises:

    state.py       2.2   state record, action record, initial state, period bookkeeping  (WO-009)
    production.py  2.6   PRODUCE step: coverage aggregator, yield, input consumption     (WO-005)
    planner.py     2.7   targets, allocation, physical delivery, audits, planner view    (WO-006)
    reporting.py   2.8   report processing, audit measurement, penalty, bonus            (WO-007)
    reward.py      2.9   enterprise reward, reward scale, val, true welfare              (WO-007)
    obs.py         2.4   the observation vector                                          (WO-008)
    prices.py      2.10  cost-plus plan prices; price-sensitivity perturbation           (WO-007)
    trade.py       2.13  bilateral trade matching                             (Phase 2,  WO-024)
    ministry.py    2.14  the ministry layer                                   (Phase 2,  WO-025)
    step.py        2.5   the period schedule as an explicit state machine                (WO-009)
    env.py         2.5   reset/step wrapper, specs, `StepInfo` and ledger hookup         (WO-009)

**This file deliberately contains no re-export.** `step.py` imports from `state.py`,
`production.py`, `planner.py`, `reporting.py`, `reward.py` and `obs.py`, and `env.py` imports from
`step.py`; a convenience re-export here would place `gosplan.env` on the import path of every one
of those modules and turn that chain into an import cycle. Import the module you need directly,
for example `from gosplan.env.production import produce_step`.

Contract rules that bind every module in this package:

    rule 5  planner rules take a `PlannerView` and nothing else; `make_planner_view` (and, by the
            whitelist recorded for the v1 freeze, `deliver`) are the only functions in
            `planner.py` that may accept a `State`;
    rule 6  `welfare_true` and `val_measured` are logged and never enter an observation, a reward
            or any agent input;
    rule 7  no transition rule and no reward term may implement bunching, padding, storming,
            hoarding, shaving or trade directly - every such behaviour must be an emergent
            consequence of the formulas of PLAN sections 2.6-2.11;
    rule 9  every environment draw goes through `gosplan.rng.draw(seed_env, purpose, *indices)`;
            no `numpy.random` or `jax.random` call may appear anywhere under `gosplan/env/`, and
            no module-level generator may exist.

The runtime records defined in this package (`State`, `EnterpriseAction` and `StepInfo` in
`state.py`, `PlannerView` in `planner.py`, `MinistryView` in `ministry.py`) must stay
field-for-field identical to the frozen declarations in `spec/spec.py` (PLAN section 10). `spec/`
is not an importable package, so the agreement is enforced by a unit test rather than by an
import; see the module docstring of `gosplan/env/state.py`.
"""

from __future__ import annotations
