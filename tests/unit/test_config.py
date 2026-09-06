"""Configuration validation, hashing and the Phase-1 registry (PLAN section 3).

Realises: PLAN section 3 (parameter registry, ranges and the P1 column) and PLAN section 11 (test
architecture, unit/property category). Owning work order: **WO-002** (frozen tests; LEAD). Binds the
WO-003 must-pass line of PLAN section 12.3, verbatim - "`tests/unit/test_config.py` (validation
rejects: `rho_cap < 1`, `w < 0`, `theta < 1`, `sum_k a_jk >= 1`, negative caps; hash stable under
field order; `p1_default_config()` matches `params.py`)". Module under test: `gosplan/config.py`.

FROZEN BY CONTRACT RULE 2. Implementers do not edit, skip or special-case these tests; a test that
looks wrong is an AMBIGUITY REPORT (CONTRACT rule 3).

SKELETON. Every test below carries `@pytest.mark.skeleton` and is skipped while `gosplan/config.py`
is a skeleton whose bodies raise `NotImplementedError`. Each docstring states the exact assertion
the eventual test must make, with the PLAN formula and tolerance it comes from; the skip lifts when
WO-003 lands, and the assertion itself is already frozen.

CONTRACT RULE 11 stands behind the registry test: the INFO/INC/SUPPLY/TECH classification in
`gosplan/params.py` is a design decision, and a parameter that drifts between the registry and the
dataclass defaults would silently change what a contrast of PLAN section 4.3 means.
"""

from __future__ import annotations

import pytest


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-003")
def test_validate_accepts_the_phase_1_default_config(p1_cfg) -> None:
    """`p1_default_config().validate()` returns `None` and raises nothing.

    Assertion: calling `validate()` on the Phase-1 configuration completes and returns `None`. This
    is the identity case that gives every other test in this module a baseline to perturb: each
    rejection test below copies this configuration, changes exactly one field, and asserts that the
    single change is what `validate` catches (PLAN section 3; `EnvConfig.validate` docstring, which
    specifies "raise `ValueError` with a message naming the offending field and the rule it broke").
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-003")
def test_validate_rejects_overfulfilment_cap_below_one(p1_cfg) -> None:
    """`validate` raises `ValueError` when `incentive.overfulfilment_cap < 1`.

    Assertion: with `overfulfilment_cap` set below 1.0 (for example 0.9), `EnvConfig.validate()`
    raises `ValueError` and the message names `overfulfilment_cap`. Reason (PLAN section 2.8):
    `rho_cap` is the ratio at which the overfulfilment bonus stops accruing, in
    `B(rho) = beta * Lambda_w(rho - 1) + s * clip(rho - 1, 0, rho_cap - 1)`; a cap below 1 would
    make `clip(., 0, rho_cap - 1)` clip to a negative upper bound and invert the schedule. The
    legitimate grid is {1.1, 1.2, inf}, and `inf` (no cap, hence no kink) must still validate.

    First bullet of the WO-003 must-pass list.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-003")
def test_validate_rejects_negative_notch_width(p1_cfg) -> None:
    """`validate` raises `ValueError` when `incentive.notch_width < 0`.

    Assertion: with `notch_width = -0.1`, `EnvConfig.validate()` raises `ValueError` naming
    `notch_width`. Reason (PLAN section 2.8): `w` is the logistic width of the notch,
    `Lambda_w(x) = 1[x >= 0]` at `w = 0` and `1 / (1 + exp(-x / w))` at `w > 0`; a negative width
    would invert the logistic and turn the bonus into a decreasing function of `rho`, breaking
    T-U3 monotonicity. `w = 0` is the Phase-1 value and must validate; so must the smooth arm's
    `w = 0.25`.

    Second bullet of the WO-003 must-pass list.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-003")
def test_validate_rejects_input_complementarity_below_one(p1_cfg) -> None:
    """`validate` raises `ValueError` when `supply.input_complementarity < 1`.

    Assertion: with `input_complementarity = 0.5`, `EnvConfig.validate()` raises `ValueError`
    naming `input_complementarity`. Reason (PLAN section 2.6): `theta` is the exponent of the CES
    coverage aggregator
    `H = (sum_j omega_j * min(1, X_ij / need_ij)**(-theta))**(-1/theta)`; the aggregator is defined
    for `theta >= 1`, where `theta = 1` is the weighted harmonic mean and `theta = inf` the
    Leontief `min` (T-U7). The legitimate grid is {2, 8, inf}, all of which must validate.

    Third bullet of the WO-003 must-pass list.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-003")
def test_validate_rejects_io_row_summing_to_one_or_more(p1_cfg) -> None:
    """`validate` raises `ValueError` when any row of `supply.io_matrix` has `sum_k a[j][k] >= 1`.

    Assertion: with one row of the I-O matrix scaled so that its entries sum to 1.0 (or more),
    `EnvConfig.validate()` raises `ValueError` naming `io_matrix` and the offending row index; the
    Phase-1 matrix, every row of which sums to 0.4, validates. Reason: no self-sustaining sector
    (PLAN section 2.10), and the cost-plus price fixed point
    `p_j = (1 + m) * (kappa_labour + sum_k a_jk * p_k)` converges only while
    `(1 + m) * sum_k a_jk < 1` (PLAN section 2.10, `initial_prices`).

    Fourth bullet of the WO-003 must-pass list.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-003")
def test_validate_rejects_negative_ratchet_caps(p1_cfg) -> None:
    """`validate` raises `ValueError` on a negative `ratchet_cap_up` or `ratchet_cap_dn`.

    Assertion: setting either cap to a negative value (for example -0.3) makes
    `EnvConfig.validate()` raise `ValueError` naming that field; both must be rejected, tested one
    at a time so the failure names the field. Reason (PLAN section 2.7.1): the caps bound the
    per-period ratchet step, `step_i = clip(rho_i - 1, -c_dn, +c_up)`; a negative bound makes the
    clip interval empty and the target rule ill-defined. The same test covers the general rule that
    scales are non-negative - `penalty_scale`, `effort_cost`, `notch_height`,
    `overfulfilment_slope`, every `yield_sigma` entry - each of which `validate` must also reject
    when negative.

    Fifth bullet of the WO-003 must-pass list ("negative caps").
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-003")
def test_validate_rejects_structurally_inconsistent_sizes(p1_cfg) -> None:
    """`validate` raises `ValueError` when a sized field disagrees with `n_enterprises`/`n_sectors`.

    Assertion, one perturbation at a time, each raising `ValueError` naming the offending field
    (`EnvConfig.validate` docstring, PLAN sections 2.1-2.12): `len(sector_of) != n_enterprises`; a
    `sector_of` entry outside `[0, n_sectors)`; `io_matrix` not `n_sectors x n_sectors`;
    `final_demand_share`, `productivity`, `yield_sigma` or `ces_alpha` not of length `n_sectors`;
    `len(arrival_probs) != steps_per_period` or its entries not summing to 1 while
    `delivery_timing != "uniform"`; a probability field outside [0, 1] (`audit_rate`, `tenure`,
    `ministry_passthrough`, `final_demand_share`, `horizontal_visibility`,
    `quality_measurability`, `shortfall_visibility`, `soft_budget`); `min_periods > max_periods`;
    `report_max_ratio <= 1`; `invest_lag < 1`; `ces_sigma <= 0`.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-003")
def test_hash_is_stable_under_field_order(p1_cfg) -> None:
    """`EnvConfig.hash()` is invariant to the order the fields were written in.

    Assertion: two `EnvConfig` values built with the same parameters but constructed with their
    keyword arguments in different orders (and with the four arm dataclasses themselves constructed
    in different orders) produce byte-identical `hash()` strings, and the hash is stable across
    processes - re-running the same construction yields the same digest. Mechanism (WO-003 notes):
    the digest is the SHA-256 hex digest of the canonical JSON encoding of the configuration, keys
    sorted, floats formatted with `repr`, tuples encoded as JSON arrays.

    Why it matters: the digest names the run directory `runs/<hash>/` and is written into every
    manifest (CONTRACT rule 10), so an order-sensitive hash would scatter one configuration's runs
    across directories and break every provenance claim in PLAN section 4.

    Sixth bullet of the WO-003 must-pass list.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-003")
def test_hash_differs_when_any_single_parameter_differs(p1_cfg) -> None:
    """Changing any one parameter changes `EnvConfig.hash()`.

    Assertion: for every field of every arm dataclass in turn, a configuration that differs from
    `p1_cfg` in exactly that field hashes differently, and the digests of all such perturbations
    are pairwise distinct. This is the other half of hash stability: a hash that collided across
    configurations would let two arms of a contrast (PLAN section 4.3) share a run directory and a
    manifest.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-003")
def test_hash_serialises_infinite_values_as_the_string_inf(p1_cfg) -> None:
    """`float("inf")` is encoded as the string `"inf"` in the hashed JSON (WO-003 notes).

    Assertion: a configuration with `supply.price_lag = float("inf")` (the Phase-1 default) and one
    with `incentive.overfulfilment_cap = float("inf")` (the smooth counterfactual of PLAN section
    2.8) both hash without raising, and the digest differs from the same configuration with a large
    finite value in that field. Plain `json.dumps` would emit the non-standard token `Infinity`,
    which is why the encoding is specified: the manifest must be readable by any JSON parser
    (CONTRACT rule 10).
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-003")
def test_p1_default_config_matches_the_params_registry() -> None:
    """`p1_default_config()` agrees field by field with `gosplan/params.py`.

    Assertion: for every `ParamSpec` in the registry, the value of the named field in
    `p1_default_config()` equals that spec's Phase-1 default, and every configuration field is
    covered by exactly one registry row - the two directions together, so neither a stale registry
    row nor an unregistered parameter can survive. Where a registry row records a range, the
    Phase-1 default lies inside it; where it records a choice set, the default is a member.

    Why it matters (CONTRACT rule 11, PLAN section 3): the registry carries each parameter's arm
    (INFO / INC / SUPPLY / TECH), phase, default, range and source, and the arm assignment is what
    makes the C_OGAS and C_INC contrasts of PLAN section 4.3 well defined. If the registry and the
    dataclass defaults drift apart, a run's manifest describes a configuration it did not use. The
    six daggered rows (`ratchet_lambda`, `growth_directive`, `overfulfilment_slope`,
    `penalty_scale`, `effort_cost`, `audit_rate`) are provisional until gate G1 records the chosen
    values in `runs/G1_decision.md`; this test asserts agreement between the two sources, never a
    particular provisional level.

    Seventh bullet of the WO-003 must-pass list.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-003")
def test_load_config_rejects_an_unknown_key(tmp_path) -> None:
    """`load_config` raises `ValueError` on a key no dataclass declares.

    Assertion: a JSON (and a TOML) document containing an unrecognised key - at the top level or
    inside one of `supply`, `incentive`, `information`, `tech` - makes `load_config(path)` raise
    `ValueError` naming the key. Reason (`load_config` docstring): silently ignoring an
    unrecognised parameter would let a sweep run at defaults while its manifest claimed otherwise,
    which is the quietest possible way to invalidate a whole arm of PLAN section 4.3.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-003")
def test_load_config_round_trips_missing_keys_and_the_inf_encoding(tmp_path) -> None:
    """`load_config` fills omitted keys with the Phase-1 defaults and decodes `"inf"`.

    Assertion: a document that omits a section, or a key inside a section, loads to a configuration
    whose omitted fields equal the Phase-1 defaults declared on the dataclasses; a document
    carrying the string `"inf"` for `price_lag` or `overfulfilment_cap` loads to `float("inf")`,
    the exact inverse of the encoding `EnvConfig.hash` uses; and the returned configuration has
    already been through `validate()`, so an invalid document raises rather than returning.
    """
    assert False
