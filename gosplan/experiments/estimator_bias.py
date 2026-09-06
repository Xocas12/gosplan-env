"""Estimator-bias study - PLAN sections 4.5, 5, 7.2, 7.3, 12.5 (WO-034), 13 (gate G4) and 14.

Realises: PLAN section 7.2, the load-bearing payback of the `forensics_core` coupling, with the
interface scoped in PLAN section 7.3. Owning work order: **WO-034** (MID-strong, Phase 3). Gate:
**G4** - the final report needs the estimator-bias curves (PLAN section 13).

What it answers. The bunching estimator of PLAN section 4.5 is used to *measure* manipulation. This
study asks what it recovers when the truth is known: how `b_hat` behaves as the manipulation gets
weaker (the notch is smoothed, `w` rises) and as the estimator's own settings move. The environment
is the instrument that supplies a known ground truth, which is exactly what archival data cannot.

Two independent sources of truth (PLAN section 7.2):
    (a) the DP's **exact** stationary `rho` distribution for the same parameters (PLAN section 5) -
        an analytic no-manipulation-error counterfactual, not a simulation;
    (b) the N-enterprise simulation at `N_SEEDS` seeds under common random numbers, whose true
        excess mass is known *relative to its own `w = 0.25` arm*.
Both are reported; they are not averaged into one "truth".

Design (PLAN section 7.2): `notch_width` w in {0, 0.02, 0.05, 0.10, 0.25} x `overfulfilment_cap`
rho_cap in {1.2, inf}, i.e. `N_ARMS = 10` arms, each at `N_SEEDS = 30` seeds with CRN, plus one DP
solve per arm.

Reported: **bias**, **RMSE** and **CI coverage** of `b_hat`, as functions of `w` and of the
estimator settings grid; and the **power curve** of `forensics_core.reconciliation.ledger_test` on
ledgers whose fictitious output is known by construction.

    OUT OF SCOPE (PLAN sections 7.2, 7.3; finding F12). **Digit tests.** Reinforcement-learning
    policies have no digit preferences, so a digit test run on this environment measures the
    floating-point tail of the action head, not manipulation. No digit statistic is computed,
    plotted or reported here, and no claim about archival digit tests may be supported from this
    study. Also out of scope (PLAN section 7.3): any claim of *calibration* of archival detectors,
    and any use of Phase-1 output for the archival anchor.

Inputs
    `gosplan.agents.dp.solve_single_enterprise` (WO-014) per arm, on the PLAN section 5 `DPGrid`
    defaults; `gosplan.agents.ppo.train` (WO-018) for the simulated arms; the ledgers of WO-011; the
    estimator interface of PLAN section 7.3.

Outputs
    `runs/estimator_bias/table.parquet`     one row per (arm, seed, estimator setting) with
                                            `b_hat`, its CI, the truth from (a) and from (b), and
                                            the derived bias / squared error / covered flag
    `runs/estimator_bias/report.md`         bias, RMSE and coverage tables against `w` and against
                                            the estimator settings, plus the reconciliation power
                                            curve and the out-of-scope note above
    `runs/estimator_bias/bias_curves.png`   bias and coverage against `w`, per estimator setting
    `runs/estimator_bias/power_curve.png`   reconciliation power against known fictitious share
    `runs/<config-hash>/`                   per-run directories with `manifest.json` (CONTRACT rule
                                            10, estimator version included)

    PLAN section 12.5 names no artefact paths for WO-034; these follow the `runs/<experiment>/`
    convention of the Phase-1 cards.

Cost (PLAN section 14): 10 arms x 30 seeds plus the DP solves, on JAX or NumPy - hours.

Estimator coupling (PLAN section 7.3). The estimator is called through

    forensics_core.bunching.estimate(x, window_lo, window_hi, bin_width, degree, excl_lo, excl_hi)
    forensics_core.reconciliation.ledger_test(reported_supply, received_inputs, io_matrix, prices)

and the import lives inside a `try/except ImportError` that falls back to
`gosplan.metrics._fallback`, whose signatures are identical. The interface is agreed with the
`forensics_core` owner at G1; the fallback is what makes this study runnable before that. Which
package answered, and at which version, is recorded in the manifest (CONTRACT rule 10), because a
bias curve is meaningless without the estimator version that produced it.

Runtime bindings. `EnvConfig` is `gosplan.config.EnvConfig` (field-for-field identical to
`spec/spec.py`, enforced by a unit test). `forensics_core`, plotting and parquet dependencies are
imported inside the functions that use them, never at module scope.
"""

from __future__ import annotations

from pathlib import Path

from gosplan.config import EnvConfig

NOTCH_WIDTH_GRID: tuple[float, ...] = (0.0, 0.02, 0.05, 0.10, 0.25)
"""`w` grid of PLAN section 7.2, identical to the PLAN section 3 registry grid for `notch_width`.
`w = 0` is a true discontinuity (the Phase-1 schedule) and `w = 0.25` is the smooth counterfactual;
the intermediate values are the manipulation-strength dial this study sweeps."""

OVERFULFILMENT_CAP_GRID: tuple[float, ...] = (1.2, float("inf"))
"""`rho_cap` grid of PLAN section 7.2. At 1.2 there is a kink at the cap as well as the notch; at
`inf` there is no cap and hence no kink, so the (w = 0.25, rho_cap = inf) cell is the fully smooth
arm of PLAN section 2.8."""

N_ARMS = 10
"""Arms in the design: `len(NOTCH_WIDTH_GRID) * len(OVERFULFILMENT_CAP_GRID)`, the "10" of the PLAN
section 14 cost line."""

N_SEEDS = 30
"""Seeds per arm (PLAN section 7.2, and the PLAN section 14 cost line), sharing `seed_env` so every
arm meets identical environment draws - common random numbers by construction (PLAN section
2.15)."""

TRUTH_SOURCES: tuple[str, ...] = ("dp_exact", "simulation_relative_to_smooth")
"""The two independent ground truths of PLAN section 7.2, reported side by side and never merged:
`dp_exact` is the DP's exact stationary `rho` distribution for the same parameters (PLAN section 5);
`simulation_relative_to_smooth` is the simulated arm's excess mass measured relative to its own
`w = 0.25` arm."""

REPORTED_STATISTICS: tuple[str, ...] = ("bias", "rmse", "ci_coverage")
"""What PLAN section 7.2 asks for, as functions of `w` and of the estimator settings: the bias of
`b_hat` against the known truth, its root mean squared error over seeds, and the empirical coverage
of its CI."""

NOMINAL_CI_COVERAGE = 0.95
"""The nominal level the empirical coverage is compared against: the estimator's bootstrap CI is a
95% interval (PLAN section 4.5). Coverage materially below this is the study's headline finding
about the estimator, not a defect to be corrected by re-tuning the estimator until it covers."""

ESTIMATOR_GRID_AXES: tuple[str, ...] = ("excluded_window", "degree", "bin_width")
"""The three axes of the estimator-settings grid of PLAN section 7.2, mapped onto the PLAN section
7.3 interface: "excluded window" is `(excl_lo, excl_hi)`, "polynomial degree" is `degree`, and
"bandwidth" is `bin_width` - the interface exposes no separate bandwidth argument. If the interface
agreed with the `forensics_core` owner at G1 gains one, this tuple gains a fourth axis and the
change is recorded in the manifest. The *extent* of each axis is fixed by the lead at issue time and
printed in the table; the pre-registered PLAN section 4.5 point must be a member of the grid and is
labelled as such in every figure."""

PREREGISTERED_SETTINGS_SOURCE = "gosplan.metrics.phenomena.phenomenon_bunching defaults (WO-016)"
"""Where the pre-registered PLAN section 4.5 estimator settings live. This module deliberately does
not restate the numbers: they have exactly one home, they are recorded in the manifest from there,
and a study of estimator settings must not be able to drift from the setting it is a study of."""

RECONCILIATION_POWER_AXIS = "known fictitious output share"
"""The x-axis of the reconciliation power curve (PLAN section 7.2): the fraction of reported supply
that is fictitious by construction in the ledger being tested. Because the environment logs `R_i`
and `S_i` separately, this quantity is known exactly, which is the whole reason the curve can be
drawn at all."""

DIGIT_TESTS_IN_SCOPE = False
"""Digit tests are **out of scope** (PLAN sections 7.2 and 7.3, finding F12): RL policies have no
digit preferences. Declared as data so the prohibition is greppable and so no report generator can
add a digit panel by default."""

CALIBRATION_CLAIMS_IN_SCOPE = False
"""PLAN section 7.3 puts any claim of *calibration* of archival detectors, and any use of Phase-1
output for the archival anchor, out of scope. This study reports bias, RMSE, coverage and power on a
known-truth simulator; it does not calibrate anything against archives."""

OUT_DIR = Path("runs/estimator_bias")
"""Artefact directory, relative to the repository root; a WO-034 convention."""

TABLE_PATH = OUT_DIR / "table.parquet"
"""One row per (arm, seed, estimator setting)."""

REPORT_PATH = OUT_DIR / "report.md"
"""Bias, RMSE and coverage tables, the reconciliation power curve, and the out-of-scope note."""

BIAS_FIG_PATH = OUT_DIR / "bias_curves.png"
"""Bias and coverage against `w`, one line per estimator setting."""

POWER_FIG_PATH = OUT_DIR / "power_curve.png"
"""Reconciliation power against `RECONCILIATION_POWER_AXIS`."""


def run(
    cfg: EnvConfig,
    estimator_grid: tuple[dict[str, object], ...],
    out_dir: Path = OUT_DIR,
    n_seeds: int = N_SEEDS,
    seed_env: int | None = None,
) -> dict[str, object]:
    """Run the estimator-bias study and write its curves.

    Takes: `cfg`, the base configuration the arms perturb - the post-G3 Phase-2 configuration, or
    the Phase-1 configuration when the study is run on Phase-1 dynamics - already validated;
    `estimator_grid`, the settings grid, each entry a mapping over `ESTIMATOR_GRID_AXES` and one of
    them the pre-registered PLAN section 4.5 point (supplied by the caller so the grid extent is
    data, not a hidden default); `out_dir`, where the artefacts are written; `n_seeds`, seeds per
    arm; `seed_env`, the root environment seed - `None` means `cfg.tech.seed_env`, shared across
    arms and seeds for common random numbers (PLAN section 2.15).

    Returns: a mapping with at least

        "arms"            tuple[dict[str, float], ...], the `N_ARMS` (w, rho_cap) cells
        "dp_truth"        dict, per arm: the DP's exact excess mass and its stationary distribution
                          summary (truth source `dp_exact`)
        "sim_truth"       dict, per arm: excess mass relative to that arm's `w = 0.25` counterpart
                          (truth source `simulation_relative_to_smooth`)
        "estimates"       tuple[dict[str, object], ...], one row per (arm, seed, setting) with
                          `b_hat`, `ci_lo`, `ci_hi`, `hole_mass`
        "bias"            dict, `REPORTED_STATISTICS[0]` against `w` and against settings
        "rmse"            dict, likewise
        "ci_coverage"     dict, empirical coverage against `NOMINAL_CI_COVERAGE`
        "power_curve"     dict, reconciliation rejection rate against
                          `RECONCILIATION_POWER_AXIS`, with the test's nominal size
        "estimator"       str, and "estimator_version" str - whether `forensics_core` or
                          `gosplan.metrics._fallback` answered, and at which version
        "artefacts"       dict[str, str], the paths written

    Procedure (PLAN sections 7.2, 7.3):

      1. Build the `N_ARMS` arms from `NOTCH_WIDTH_GRID` x `OVERFULFILMENT_CAP_GRID` as overrides on
         `cfg`, validating each.
      2. Per arm, solve the single-enterprise DP once (PLAN section 5) and record its exact `rho`
         distribution and excess mass - truth source `dp_exact`.
      3. Per arm, train and evaluate `n_seeds` N-enterprise runs under CRN, logging ledgers and
         writing `runs/<hash>/manifest.json` (CONTRACT rule 10). Truth source
         `simulation_relative_to_smooth` is each arm's excess mass measured against its own
         `w = 0.25` counterpart.
      4. For every (arm, seed, setting) in `estimator_grid`, call the bunching estimator through the
         PLAN section 7.3 interface - `forensics_core.bunching.estimate` inside a `try/except
         ImportError` falling back to `gosplan.metrics._fallback` - over the measurement window of
         PLAN section 4.4, and record `b_hat` with its CI.
      5. Compute `REPORTED_STATISTICS` against each truth source separately: bias, RMSE over seeds,
         and empirical CI coverage against `NOMINAL_CI_COVERAGE`.
      6. Run `forensics_core.reconciliation.ledger_test` on ledgers whose fictitious output share is
         known by construction, sweeping that share to obtain the power curve.
      7. Write `table.parquet`, `report.md`, `bias_curves.png` and `power_curve.png`, each stating
         the estimator package and version that produced it.

    No digit statistic is computed at any step (`DIGIT_TESTS_IN_SCOPE`), and no result here is
    presented as a calibration of an archival detector (`CALIBRATION_CLAIMS_IN_SCOPE`).

    Binds: gate G4 of PLAN section 13 - "estimator-bias curves" - and the design of PLAN section
    7.2.

    Realises: PLAN sections 4.5, 5, 7.2, 7.3, 12.5 (WO-034), 13, 14. Owning WO: **WO-034**.
    """
    raise NotImplementedError("PLAN section 7.2 (WO-034) - implemented in WO-034")


def main() -> int:
    """Entry point: run the ten arms and write the bias, coverage and power artefacts.

    Takes: nothing; the base configuration, the arm grids (`NOTCH_WIDTH_GRID`,
    `OVERFULFILMENT_CAP_GRID`), the seed count (`N_SEEDS`) and the estimator-settings grid fixed by
    the lead are assembled here. Any command-line surface is built inside this function.

    Returns: a process exit code - 0 when every arm ran and `runs/estimator_bias/report.md` was
    written, 1 otherwise. The exit code says nothing about the estimator's performance: poor
    coverage or large bias at small `w` is the study's finding, reported as such, and never a reason
    to re-tune the pre-registered settings of PLAN section 4.5 after the fact.

    Realises: PLAN sections 7.2, 12.5 (WO-034), 13. Owning WO: **WO-034**.
    """
    raise NotImplementedError("PLAN section 7.2 (WO-034) - implemented in WO-034")


if __name__ == "__main__":
    raise SystemExit(main())
