"""The frozen interface exists and exposes the whole PLAN section 10 surface.

Realises: PLAN section 10 (`spec/spec.py` v0 - the interface skeleton) and PLAN section 11 (test
architecture, unit/property category). Owning work order: **WO-002** (frozen tests; LEAD). Binds:
the WO-001 must-pass line of PLAN section 12.3, verbatim - "`tests/unit/test_spec_imports.py`
(spec imports; every public symbol in section 10 present)".

THIS MODULE IS THE ONE EXECUTABLE TEST IN THE SKELETON. Every other module in `tests/unit` is a
placeholder whose tests are skipped until their work order lands (see `tests/conftest.py`). This one
runs now and must pass now: it is the only thing standing between a typo in the frozen interface and
twenty work orders written against it.

CONTRACT RULE 1 is what it guards. `spec/spec.py` is provisional (v0, `SPEC_VERSION` "0.1.0") until
gate G1 and frozen (v1, "1.0.0") thereafter at WO-013; after v1 only the lead may change it, and
only with a `spec/CHANGELOG.md` entry. So this module asserts the *presence and shape* of the
section 10 surface and never a version literal: a bump from "0.1.0" to "1.0.0" is a sanctioned
change, a missing symbol never is.

CONTRACT RULE 2 freezes this file along with the rest of `tests/unit`: implementers do not edit it.
If a symbol here disagrees with `spec/spec.py`, that is an AMBIGUITY REPORT against the owning work
order (CONTRACT rule 3), not an edit to either side.

Loading mechanism. `spec/` is a directory outside the `gosplan` package and is not installable, so
the module is loaded by path with `importlib.util.spec_from_file_location`. The path is derived from
this file's own location (`parents[2]` is the repository root), which keeps the test correct on
Windows and POSIX alike and independent of the working directory pytest was started from.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
"""Repository root, derived from this file's location: `tests/unit/test_spec_imports.py` ->
`tests/unit` -> `tests` -> the root. Never the process working directory, so the test is correct
however pytest was invoked."""

SPEC_PATH = REPO_ROOT / "spec" / "spec.py"
"""The frozen interface file of PLAN section 10. It is not part of the `gosplan` package and is not
installable, which is why this module loads it by path rather than importing it."""

SPEC_MODULE_NAME = "gosplan_spec_under_test"
"""Name the loaded module is registered under in `sys.modules`. Deliberately not `spec`: the
repository has a `spec/` directory with no `__init__.py`, and a real import could resolve to it as a
namespace package and mask a broken file with a silently empty module."""

PLAN_SECTION_10_SYMBOLS: tuple[str, ...] = (
    # version and aliases
    "SPEC_VERSION",
    "Array",
    # enums
    "Arm",
    "Phase",
    "ObjectiveMetric",
    "PenaltyForm",
    "PenaltyArg",
    "AuditMode",
    "AggLevel",
    "DeliveryTiming",
    "HorizonMode",
    "ParamSharing",
    # config (PLAN section 3)
    "SupplyConfig",
    "IncentiveConfig",
    "InformationConfig",
    "TechConfig",
    "EnvConfig",
    "load_config",
    "p1_default_config",
    # state, actions, views (PLAN sections 2.2-2.4)
    "State",
    "EnterpriseAction",
    "PlannerView",
    "MinistryView",
    "obs_spec",
    "action_spec",
    "active_action_dims",
    # rng (PLAN section 2.15)
    "draw",
    # production (PLAN section 2.6)
    "coverage",
    "produce_step",
    # planner (PLAN section 2.7)
    "make_planner_view",
    "update_targets",
    "fulfilment_measure",
    "allocate",
    "deliver",
    "select_audits",
    # reporting and reward (PLAN sections 2.8-2.9)
    "process_reports",
    "audit_and_penalise",
    "bonus",
    "reward_scale",
    "enterprise_reward",
    "val_measured",
    "val_true",
    "welfare_true",
    "initial_prices",
    # env (PLAN section 2.5)
    "StepInfo",
    "GosplanEnv",
    # agents (PLAN section 6)
    "Agent",
    # DP (PLAN section 5)
    "DPGrid",
    "DPSolution",
    "solve_single_enterprise",
    "classify_regime",
    # ledger and metrics (PLAN section 4)
    "StepRecord",
    "Ledger",
    "write_manifest",
    "phenomenon_bunching",
    "phenomenon_padding",
    "phenomenon_hidden_reserves",
    "phenomenon_storming",
    "phenomenon_hoarding",
    "phenomenon_blat",
    "phenomenon_quality",
    # Phase-2 sketches (signatures only)
    "match_trades",
    "ministry_forward",
    "solve_oracle",
)
"""Every public symbol the PLAN section 10 code block declares, in its order. This tuple is the
test's copy of section 10 and the reason the test can fail: a symbol dropped from `spec/spec.py`
must break WO-001's must-pass line rather than be discovered by whichever work order needed it
first. `spec/spec.py` legitimately exports *more* than this - `RegimeLabel`, `Purpose` and `Dist`
narrow the `str` parameters section 10 types loosely, a narrowing the spec header records for the
v1 CHANGELOG - so the assertions below are containment, never equality."""

SUPPLY_CONFIG_FIELDS: frozenset[str] = frozenset(
    {
        "n_enterprises",
        "n_sectors",
        "sector_of",
        "io_matrix",
        "final_demand_share",
        "productivity",
        "yield_sigma",
        "input_complementarity",
        "setup_cost",
        "irs_alpha",
        "capital_dep",
        "invest_lag",
        "delivery_timing",
        "arrival_probs",
        "holding_loss",
        "input_holding_loss",
        "price_markup",
        "price_lag",
        "tech_drift_sigma",
        "ces_alpha",
        "ces_sigma",
        "trade_tau",
        "quality_matters",
        "quality_cost",
    }
)
"""The SUPPLY-arm fields named in the PLAN section 10 `SupplyConfig` comment. Section 10 writes the
two sizing fields as the symbols `N` and `J` of PLAN section 2.1; the spec spells them
`n_enterprises` and `n_sectors`, which is the mapping asserted here. Held as a set because section
10 fixes the membership, not the declaration order."""

INCENTIVE_CONFIG_FIELDS: frozenset[str] = frozenset(
    {
        "objective_metric",
        "ratchet_lambda",
        "growth_directive",
        "ratchet_cap_up",
        "ratchet_cap_dn",
        "ratchet_deadband",
        "notch_height",
        "notch_width",
        "overfulfilment_slope",
        "overfulfilment_cap",
        "penalty_form",
        "penalty_arg",
        "penalty_scale",
        "effort_cost",
        "soft_budget",
        "steps_per_period",
        "tenure",
        "alloc_eta_request",
        "alloc_eta_need",
        "bonus_heterogeneity",
    }
)
"""The INC-arm fields named in the PLAN section 10 `IncentiveConfig` comment."""

INFORMATION_CONFIG_FIELDS: frozenset[str] = frozenset(
    {
        "report_lag",
        "aggregation_level",
        "audit_rate",
        "audit_noise",
        "audit_mode",
        "channel_noise",
        "ministry_passthrough",
        "n_ministries",
        "horizontal_visibility",
        "quality_measurability",
        "shortfall_visibility",
        "self_obs_noise",
    }
)
"""The INFO-arm fields named in the PLAN section 10 `InformationConfig` comment."""

TECH_CONFIG_FIELDS: frozenset[str] = frozenset(
    {
        "horizon_mode",
        "min_periods",
        "max_periods",
        "report_max_ratio",
        "request_max_multiple",
        "target_floor_frac",
        "inventory_cap_mult",
        "initial_target_frac",
        "param_sharing",
        "seed_env",
        "seed_policy",
    }
)
"""The TECH-arm fields named in the PLAN section 10 `TechConfig` comment. PPO hyper-parameters are
TECH too but deliberately live with the adapter (WO-017), not in `EnvConfig`, because the
environment never reads them."""

ENV_CONFIG_FIELDS: frozenset[str] = frozenset(
    {"supply", "incentive", "information", "tech", "spec_version"}
)
"""`EnvConfig`'s own fields: the four arms of PLAN section 3 plus the `spec_version` that produced
the configuration, which every run manifest carries (CONTRACT rules 1, 10)."""

CONFIG_FIELD_SETS: tuple[tuple[str, frozenset[str]], ...] = (
    ("SupplyConfig", SUPPLY_CONFIG_FIELDS),
    ("IncentiveConfig", INCENTIVE_CONFIG_FIELDS),
    ("InformationConfig", INFORMATION_CONFIG_FIELDS),
    ("TechConfig", TECH_CONFIG_FIELDS),
    ("EnvConfig", ENV_CONFIG_FIELDS),
)
"""The five configuration dataclasses of PLAN section 10 paired with their expected field sets, so
one parametrised assertion covers the whole PLAN section 3 registry surface."""


def _load_spec_module():
    """Load `spec/spec.py` as a module object, by path.

    Takes: nothing; the path is `SPEC_PATH`. Returns: the executed module. Raises `ImportError` if
    the loader cannot be built (a missing or unreadable file), and propagates whatever
    `spec/spec.py` itself raises at import time - which is the point: "spec imports" is the first
    half of WO-001's must-pass line.

    Mechanism: `importlib.util.spec_from_file_location(SPEC_MODULE_NAME, SPEC_PATH)`, then
    `module_from_spec`, then registration in `sys.modules` before `exec_module` (so that the
    module's own `from __future__ import annotations` and dataclass machinery resolve normally),
    then `exec_module`. Deliberately simple: no package tricks, no `sys.path` mutation, no
    reliance on the working directory.
    """
    loader = importlib.util.spec_from_file_location(SPEC_MODULE_NAME, SPEC_PATH)
    if loader is None or loader.loader is None:
        raise ImportError(f"could not build an import loader for {SPEC_PATH}")
    module = importlib.util.module_from_spec(loader)
    sys.modules[SPEC_MODULE_NAME] = module
    loader.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def spec_module():
    """The executed `spec/spec.py` module object.

    Takes: nothing. Returns: the module loaded by `_load_spec_module()`, once per test module.
    Every assertion below reads its attributes; none of them calls a function on it, because every
    body in the frozen interface raises `NotImplementedError` by design (PLAN section 10).
    """
    return _load_spec_module()


def test_spec_file_exists() -> None:
    """`spec/spec.py` is present at the path PLAN section 8 gives it.

    Asserts `SPEC_PATH.is_file()`. A missing interface file is the failure this suite must report
    first and most loudly: every work order in PLAN section 12.3 reads it, and the loader error of
    the next test would otherwise name an import problem rather than an absent file.
    """
    assert SPEC_PATH.is_file(), f"frozen interface missing: {SPEC_PATH}"


def test_spec_module_imports(spec_module) -> None:
    """The frozen interface imports cleanly - the first half of WO-001's must-pass line.

    Asserts the module object exists and that its `__file__` resolves to `SPEC_PATH`, i.e. the
    module under test is the repository's own `spec/spec.py` and not a same-named module picked up
    from `sys.path`. Importing it must have no side effect beyond defining names: PLAN section 10
    fixes its imports at `dataclasses`, `typing` and `numpy`, and it may not import `gosplan/`,
    `forensics_core`, or any solver or learner package.
    """
    assert spec_module is not None
    assert Path(spec_module.__file__).resolve() == SPEC_PATH


def test_spec_version_is_a_string(spec_module) -> None:
    """`SPEC_VERSION` is a non-empty dotted version string (PLAN section 10, CONTRACT rule 1).

    Asserts `isinstance(spec_module.SPEC_VERSION, str)`, that it is non-empty, and that it is
    dot-separated digits (v0 is "0.1.0"; WO-013 bumps it to "1.0.0" at the v1 freeze). The literal
    value is deliberately NOT asserted: bumping it is the sanctioned change of CONTRACT rule 1,
    recorded in `spec/CHANGELOG.md`, and a frozen test that pinned the literal would turn a
    sanctioned bump into a test failure. It is written into every run manifest (CONTRACT rule 10),
    which is why it must be a string rather than a tuple or a number.
    """
    version = spec_module.SPEC_VERSION
    assert isinstance(version, str)
    assert version
    parts = version.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)


def test_array_alias_is_the_numpy_array_type(spec_module) -> None:
    """`Array` is `numpy.ndarray` (PLAN section 10).

    Asserts `spec_module.Array is np.ndarray`. Every numeric signature in the interface is typed
    with this alias, and the Phase-2 JAX port (WO-029) substitutes its own array type behind the
    same name - which is only mechanical if the alias is exactly the array type and never a
    subclass, a `TypeAlias` wrapper or a string.
    """
    assert spec_module.Array is np.ndarray


def test_plan_section_10_symbols_are_present(spec_module) -> None:
    """Every public symbol of PLAN section 10 exists in the frozen interface.

    Asserts `hasattr(spec_module, name)` for every name in `PLAN_SECTION_10_SYMBOLS` - the second
    half of WO-001's must-pass line, "every public symbol in section 10 present" - and reports the
    complete set of missing names in one failure rather than stopping at the first, so a spec
    session sees the whole gap at once.
    """
    missing = [name for name in PLAN_SECTION_10_SYMBOLS if not hasattr(spec_module, name)]
    assert missing == [], f"PLAN section 10 symbols missing from spec/spec.py: {missing}"


def test_all_covers_every_plan_section_10_symbol(spec_module) -> None:
    """`spec.__all__` covers the PLAN section 10 surface.

    Asserts `spec_module.__all__` is a list or tuple of strings and that
    `set(PLAN_SECTION_10_SYMBOLS) <= set(spec_module.__all__)`. Containment, not equality: the
    interface may export more than section 10 lists - `RegimeLabel`, `Purpose` and `Dist` narrow
    parameters section 10 types as bare `str` - but it may never export less, because `__all__` is
    what a reader and a star-import take to be the interface.
    """
    exported = spec_module.__all__
    assert isinstance(exported, (list, tuple))
    assert all(isinstance(name, str) for name in exported)
    uncovered = sorted(set(PLAN_SECTION_10_SYMBOLS) - set(exported))
    assert uncovered == [], f"declared in PLAN section 10 but absent from __all__: {uncovered}"


def test_every_exported_name_resolves(spec_module) -> None:
    """Nothing in `__all__` is a dangling name.

    Asserts `hasattr(spec_module, name)` for every entry of `spec_module.__all__`. A name exported
    but never defined would break `from spec.spec import *` and, worse, would let a work order cite
    a symbol that does not exist; the check is the mirror image of the previous test.
    """
    dangling = [name for name in spec_module.__all__ if not hasattr(spec_module, name)]
    assert dangling == [], f"exported by __all__ but not defined: {dangling}"


@pytest.mark.parametrize(("class_name", "expected_fields"), CONFIG_FIELD_SETS)
def test_config_dataclass_fields_match_plan_section_10(
    spec_module, class_name: str, expected_fields: frozenset[str]
) -> None:
    """Each configuration dataclass carries exactly the fields PLAN section 10 names.

    Asserts, for each of `SupplyConfig`, `IncentiveConfig`, `InformationConfig`, `TechConfig` and
    `EnvConfig`: the attribute is a dataclass, and the set of its `dataclasses.fields` names equals
    the corresponding constant above - the field list of the PLAN section 10 comment, with section
    10's symbols `N` and `J` spelled `n_enterprises` and `n_sectors`. Equality, not containment: an
    extra configuration field is a new parameter, and PLAN section 3 plus CONTRACT rule 11 require
    every parameter to carry an arm assignment and a registry row in `gosplan/params.py`, so it may
    not appear here unannounced.

    Declaration order is not asserted; membership is. `EnvConfig.hash()` sorts keys before hashing
    (WO-003), so a reordering is invisible to every run directory and manifest.
    """
    cls = getattr(spec_module, class_name)
    assert dataclasses.is_dataclass(cls), f"{class_name} is not a dataclass"
    actual = {f.name for f in dataclasses.fields(cls)}
    assert actual == set(expected_fields)


def test_env_config_exposes_validate_and_hash(spec_module) -> None:
    """`EnvConfig` carries the two methods PLAN section 10 gives it.

    Asserts `EnvConfig.validate` and `EnvConfig.hash` exist and are callable. `validate` is the
    cross-field gate of PLAN section 3 (WO-003); `hash` is the SHA-256 of the canonical JSON
    encoding that names `runs/<hash>/` and appears in every manifest (CONTRACT rule 10). Neither is
    called here - both raise `NotImplementedError` until WO-003 lands.
    """
    cfg_cls = spec_module.EnvConfig
    assert callable(cfg_cls.validate)
    assert callable(cfg_cls.hash)


def test_gosplan_env_exposes_the_section_10_methods(spec_module) -> None:
    """`GosplanEnv` carries `__init__`, `reset`, `step` and `phase` (PLAN sections 2.5, 10).

    Asserts all four attributes exist and are callable. They are the whole agent-facing surface of
    the environment: `reset(seed_env, seed_policy) -> (obs, StepInfo)`,
    `step(action) -> (obs, reward, done, StepInfo)` and `phase() -> "produce" | "report"`. Nothing
    else may be added to it without a `spec/CHANGELOG.md` entry (CONTRACT rule 1).
    """
    env_cls = spec_module.GosplanEnv
    for name in ("__init__", "reset", "step", "phase"):
        assert callable(getattr(env_cls, name)), f"GosplanEnv.{name} missing or not callable"


def test_ledger_exposes_append_and_to_parquet(spec_module) -> None:
    """`Ledger` carries the two methods PLAN section 10 gives it (PLAN section 4).

    Asserts `Ledger.append` and `Ledger.to_parquet` exist and are callable. `append` also maintains
    the run's flag set - `BOUND_BINDING` above 1% of reports at `rho_max` (CONTRACT rule 8) - and
    `to_parquet` is the columnar dump `tests/unit/test_ledger.py` round-trips.
    """
    ledger_cls = spec_module.Ledger
    assert callable(ledger_cls.append)
    assert callable(ledger_cls.to_parquet)


def test_agent_protocol_exposes_act_and_reset(spec_module) -> None:
    """`Agent` is a Protocol with exactly the two methods of PLAN section 6.1.

    Asserts `Agent.act` and `Agent.reset` exist and are callable. `act(obs, phase, rng)` takes the
    observation, the phase and a `seed_policy`-stream generator and nothing else: CONTRACT rule 6
    keeps `State`, `StepInfo` and `PlannerView` out of every agent signature, and test T-B5 checks
    the PPO adapter's forward pass against that rule.
    """
    agent_cls = spec_module.Agent
    assert callable(agent_cls.act)
    assert callable(agent_cls.reset)
