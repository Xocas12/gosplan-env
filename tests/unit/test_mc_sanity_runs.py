"""Smoke test for the Monte-Carlo sanity harness (PLAN section 12.3, WO-012; gate G0).

Realises: PLAN section 12.3 (the WO-012 card), PLAN section 13 (gate G0, whose artefact is
`runs/mc_sanity/report.md`), PLAN section 14 (the wall-clock estimate the harness measures against
reality) and PLAN section 11 (test architecture, unit/property category). Owning work order:
**WO-002** (frozen tests; LEAD). Binds the WO-012 must-pass line of PLAN section 12.3, verbatim -
"`tests/unit/test_mc_sanity_runs.py` (smoke)". Module under test:
`gosplan/experiments/mc_sanity.py`.

WO-012 card, verbatim: "Runs 2,000 episodes each of `Random`, `TruthfulMyopic`, `Padder` at
`p1_default_config()` and at 20 random perturbations of SUPPLY parameters; asserts conservation to
1e-9, no NaN/inf, bounded `T` and `S`, `fill in [0,1]`, `Padder` produces downstream shortage, run
time per episode; writes `runs/mc_sanity/report.md`."

SMOKE, NOT A GATE. These tests run the harness at a small `n_episodes` and `n_perturbations` and
assert that it runs, returns the documented mapping and writes its report. The gate itself is G0,
run by the lead with the full 2,000-episode sweep (CONTRACT rule 13: tests are not experiments, and
nothing in `tests/acceptance/` is on any work order's must-pass list).

HELD OUT. WO-012 is forbidden from computing or plotting any quantity in PLAN section 4.1 rows 2
(storming), 5 (hoarding), 6 (blat) and 7 (hidden reserves), and no test here asks it to. The
`Padder` shortage assertion is the property of test T-B3 (shortage *propagation*), which is not a
held-out row; the direction of hoarding is asserted nowhere.

FROZEN BY CONTRACT RULE 2. SKELETON: every test is `@pytest.mark.skeleton` and skipped until WO-012
lands; each docstring states the exact assertion, formula and tolerance.
"""

from __future__ import annotations

import pytest


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-012")
def test_run_completes_and_returns_the_documented_mapping(p1_cfg, tmp_path) -> None:
    """`mc_sanity.run(...)` returns the mapping its docstring promises.

    Assertion: calling `run(cfg, out_dir=tmp_path, n_episodes=<small>, n_perturbations=<small>)`
    returns a mapping carrying at least `n_configs`, `config_hashes`, `n_episodes`,
    `max_conservation_error`, `n_nonfinite`, `target_bound_failures`, `stock_bound_failures`,
    `fill_bound_failures`, `padder_min_fill`, `padder_shortage`, `seconds_per_episode`, `flags`,
    `passed` and `report_path`; `n_configs == 1 + n_perturbations`; `len(config_hashes) ==
    n_configs` with distinct entries, each equal to the `EnvConfig.hash()` of the configuration it
    names; and every numeric value is finite.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-012")
def test_run_writes_the_gate_g0_report(p1_cfg, tmp_path) -> None:
    """The harness writes `report.md` where the gate expects it.

    Assertion: after the call, `Path(result["report_path"])` exists, sits under the `out_dir` given
    (defaulting to `runs/mc_sanity/report.md`, the artefact PLAN section 13 names for gate G0), and
    contains a section per configuration and agent with each assertion and its observed extreme,
    the flags raised, the per-episode wall clock, and the verbatim restatement of the held-out
    prohibition the harness ran under.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-012")
def test_run_reports_conservation_within_tolerance(p1_cfg, tmp_path) -> None:
    """`max_conservation_error` is below `CONSERVATION_TOL` on a clean run.

    Assertion: the reported maximum violation of the per-period, per-good identity of test T-U1 is
    below `mc_sanity.CONSERVATION_TOL` (1e-9), and it is reported as a number rather than as a
    boolean - the harness records the observed extreme for every assertion instead of stopping at
    the first failure, so a regression can be sized. A failure is reported, never repaired by
    widening the tolerance (CONTRACT rule 8 in spirit).
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-012")
def test_run_finds_no_non_finite_values_and_no_out_of_bounds_state(p1_cfg, tmp_path) -> None:
    """No NaN or inf anywhere, and `T`, `S` and `fill` stay inside their analytic bounds.

    Assertion: `n_nonfinite == 0` across every ledger column; `target_bound_failures == 0`, with
    `T_i` inside `[T_min, T_0 * ((1 + g) * (1 + lambda * c_up))**P_max]` - the floor of PLAN section
    2.7.1 and the largest value the capped ratchet can reach in `P_max` periods, both computed from
    the configuration and never hard-coded; `stock_bound_failures == 0`, with `S_i` inside
    `[0, inventory_cap_mult * cap_i]` (PLAN section 2.11); and `fill_bound_failures == 0`, with
    `fill_i` inside `mc_sanity.FILL_BOUNDS` = [0, 1].
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-012")
def test_run_exercises_the_padder_shortage_channel(p1_cfg, tmp_path) -> None:
    """Under `Padder` at least one enterprise sees `fill < 1`.

    Assertion: `padder_shortage is True` and `padder_min_fill < 1`, i.e. the delivery channel of
    PLAN section 2.7.3 propagated a shortage when claims exceeded stock. This is the property of
    test T-B3 (shortage propagation), which is a pipeline check and not one of the held-out rows;
    `Padder` exists only to exercise this channel and is never a baseline. Nothing here asserts a
    direction of *learned* behaviour.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-012")
def test_run_measures_wall_clock_per_episode(p1_cfg, tmp_path) -> None:
    """The harness times itself, so the PLAN section 14 compute estimate can be checked.

    Assertion: `seconds_per_episode` is a positive finite float and is written into the report. The
    estimate exists to be checked against reality before Phase 1 commits to it (PLAN section 14),
    which is why the number is an output of the harness rather than a note in a log.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-012")
def test_run_leaves_the_locked_mechanism_parameters_at_their_phase_1_values(
    p1_cfg, tmp_path
) -> None:
    """Perturbations never touch a parameter locked behind a held-out phenomenon.

    Assertion: every configuration in `config_hashes` agrees with the baseline on every name in
    `mc_sanity.PERTURBATION_EXCLUDED` - `delivery_timing`, `arrival_probs`, `input_holding_loss`,
    `trade_tau`, `alloc_eta_request` - while differing on the SUPPLY parameters the harness is
    allowed to move (`SUPPLY_PERTURBATION_RANGES` drawn uniformly on their ranges,
    `SUPPLY_PERTURBATION_CHOICES` uniformly over their grids, `theta = inf` included so the
    Leontief branch of `coverage` is exercised); and every perturbed configuration passes
    `EnvConfig.validate()`. Those five are the mechanism parameters locked in PLAN section 4.2
    behind rows 2, 5 and 6 of PLAN section 4.1, and moving one here would exercise a locked
    mechanism before its pre-registered study exists.
    """
    assert False
