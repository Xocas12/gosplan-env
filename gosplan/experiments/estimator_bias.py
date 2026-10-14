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
    `runs/estimator_bias/reconciliation.md` the reconciliation supplement of #110: the null
                                            residuals, and the curve against the inflation factor
                                            on the Phase-1, single-mechanism and C0 ledgers (with
                                            `reconciliation.json`; `--reconciliation` alone)
    `runs/<config-hash>/`                  per-run directories with `manifest.json` (CONTRACT rule
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

import numpy as np

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
         known by construction, sweeping that share to obtain the power curve, and write the
         reconciliation supplement of #110 beside it (`reconciliation_supplement`).
      7. Write `table.parquet`, `report.md`, `bias_curves.png` and `power_curve.png`, each stating
         the estimator package and version that produced it.

    No digit statistic is computed at any step (`DIGIT_TESTS_IN_SCOPE`), and no result here is
    presented as a calibration of an archival detector (`CALIBRATION_CLAIMS_IN_SCOPE`).

    Binds: gate G4 of PLAN section 13 - "estimator-bias curves" - and the design of PLAN section
    7.2.

    Realises: PLAN sections 4.5, 5, 7.2, 7.3, 12.5 (WO-034), 13, 14. Owning WO: **WO-034**.
    """
    from gosplan.agents.dp import DPGrid, solve_single_enterprise
    from gosplan.experiments import _g2
    from gosplan.experiments.phase1_gate import GATE_SIZING, study_ppo_config

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arms = arm_grid()
    arm_cfgs = {key: _g2.with_overrides(cfg, overrides(*key)) for key in arms}

    # (a) DP truth: the exact stationary rho distribution per arm.
    dp_rho = {
        key: np.asarray(
            solve_single_enterprise(_g2.recovery_config(c), DPGrid()).stationary_rho, dtype=float
        )
        for key, c in arm_cfgs.items()
    }

    # (b) Simulation: reuse the Phase-1 gate's notched and smooth arms, train the rest.
    sizing = GATE_SIZING
    ppo = study_ppo_config()
    jobs = [(key, s) for key in arms for s in range(seeds_for(key, n_seeds))]
    job_cfgs = [_g2.seeded(arm_cfgs[key], s, seed_env) for key, s in jobs]
    roots = [run_root_for(key, out_dir) for key, _s in jobs]
    _g2.run_many([(c, sizing, r, ppo) for c, r in zip(job_cfgs, roots, strict=True)])
    rho_by_seed = _pool(
        measure_rho_seed, [(c, r, sizing, ppo) for c, r in zip(job_cfgs, roots, strict=True)]
    )
    sim: dict[tuple, list[list[np.ndarray]]] = {key: [] for key in arms}
    for (key, _s), episodes in zip(jobs, rho_by_seed, strict=True):
        sim[key].append([np.asarray(e, dtype=float) for e in episodes])

    rows, dp_truth, sim_truth = [], {}, {}
    for key in arms:
        smooth = (SMOOTH_W, key[1])
        pooled = np.concatenate([np.concatenate(ep) for ep in sim[key]])
        pooled_smooth = np.concatenate([np.concatenate(ep) for ep in sim[smooth]])
        for setting in estimator_grid:
            bw = float(setting["bin_width"])
            t_dp = true_excess_mass(dp_rho[key], dp_rho[smooth], bw)
            t_sim = true_excess_mass(pooled, pooled_smooth, bw)
            dp_truth[f"{key[0]}|{key[1]}|{_setting_id(setting)}"] = t_dp
            sim_truth[f"{key[0]}|{key[1]}|{_setting_id(setting)}"] = t_sim
            dp_est = _estimate(dp_rho[key], setting)
            rows.append(
                {
                    "w": key[0],
                    "rho_cap": key[1],
                    "source": "dp_exact",
                    "seed": -1,
                    "setting": _setting_id(setting),
                    "truth": t_dp,
                    **dp_est,
                }
            )
            for s, episodes in enumerate(sim[key]):
                est = _estimate(episodes, setting)
                rows.append(
                    {
                        "w": key[0],
                        "rho_cap": key[1],
                        "source": "simulation",
                        "seed": s,
                        "setting": _setting_id(setting),
                        "truth": t_sim,
                        **est,
                    }
                )

    stats = summarise(rows)
    power = reconciliation_power(cfg)
    reconciliation_supplement(out_dir)
    backend = _backend()
    result = {
        "arms": tuple({"w": k[0], "rho_cap": k[1]} for k in arms),
        "dp_truth": dp_truth,
        "sim_truth": sim_truth,
        "estimates": tuple(rows),
        "bias": stats["bias"],
        "rmse": stats["rmse"],
        "ci_coverage": stats["ci_coverage"],
        "median_error": stats["median_error"],
        "n_defined": stats["n_defined"],
        "n_total": stats["n_total"],
        "dp_window_mass": {
            f"{k[0]}|{k[1]}": float(np.mean((dp_rho[k] >= 1.0) & (dp_rho[k] <= 1.02))) for k in arms
        },
        "power_curve": power,
        "estimator": backend[0],
        "estimator_version": backend[1],
        "seeds": {f"{k[0]}|{k[1]}": seeds_for(k, n_seeds) for k in arms},
    }
    import pandas as pd

    pd.DataFrame(rows).to_parquet(out_dir / TABLE_PATH.name, index=False)
    _plots(rows, power, out_dir)
    (out_dir / REPORT_PATH.name).write_text(render_report(result, estimator_grid), encoding="utf-8")
    result["artefacts"] = {
        "report": str(out_dir / REPORT_PATH.name),
        "table": str(out_dir / TABLE_PATH.name),
        "bias_fig": str(out_dir / BIAS_FIG_PATH.name),
        "power_fig": str(out_dir / POWER_FIG_PATH.name),
        "reconciliation": str(out_dir / RECON_REPORT_PATH.name),
    }
    return result


def main() -> int:
    """Entry point: run the ten arms and write the bias, coverage and power artefacts.

    Takes: nothing; the base configuration, the arm grids (`NOTCH_WIDTH_GRID`,
    `OVERFULFILMENT_CAP_GRID`), the seed count (`N_SEEDS`) and the estimator-settings grid fixed by
    the lead are assembled here. Any command-line surface is built inside this function.

    Returns: a process exit code - 0 when every arm ran and `runs/estimator_bias/report.md` was
    written, 1 otherwise. The exit code says nothing about the estimator's performance: poor
    coverage or large bias at small `w` is the study's finding, reported as such, and never a reason
    to re-tune the pre-registered settings of PLAN section 4.5 after the fact.

    `--reconciliation` writes only the reconciliation supplement of #110
    (`runs/estimator_bias/reconciliation.md` and `.json`), which needs heuristic agents and no
    training run.

    Realises: PLAN sections 7.2, 12.5 (WO-034), 13. Owning WO: **WO-034**.
    """
    import sys

    from gosplan.config import p1_default_config

    if "--reconciliation" in sys.argv[1:]:
        sup = reconciliation_supplement()
        print(
            f"reconciliation supplement: {len(sup['ledgers'])} ledgers; {RECON_REPORT_PATH.as_posix()}"
        )
        return 0
    res = run(p1_default_config(), ESTIMATOR_GRID)
    print(f"estimator-bias: {len(res['estimates'])} estimates; report {res['artefacts']['report']}")
    return 0


# ---------- implementation (spec/P3_REVISION.md S4) ----------

SMOOTH_W = 0.25
"""The notch width of each arm's own smooth counterpart (PLAN section 7.2)."""

REUSED_ARMS: dict[tuple[float, float], str] = {
    (0.0, 1.2): "notched",
    (0.25, float("inf")): "smooth",
}
"""Arms identical to the Phase-1 gate's (`phase1_gate.ARMS`): their 30 runs are read back."""

P1_RUN_ROOT = Path("runs/phase1_gate/runs")
NEW_ARM_SEEDS = 5
"""Seeds per arm not run at the Phase-1 gate (P3 revision S4: reduced from 30 for compute)."""

ESTIMATOR_GRID: tuple[dict[str, object], ...] = tuple(
    {"excluded_window": win, "degree": deg, "bin_width": bw}
    for win in ((0.95, 1.02), (0.97, 1.02), (0.93, 1.03))
    for deg in (5, 7, 9)
    for bw in (0.005, 0.01)
)
"""The 18 settings of P3 revision S4; the pre-registered one is `PREREGISTERED`."""

PREREGISTERED = {"excluded_window": (0.95, 1.02), "degree": 9, "bin_width": 0.005}

POWER_SHARES: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
POWER_REPLICATES = 20
POWER_INFLATION = 1.2
POWER_ALPHA = 0.05

POWER_INFLATIONS: tuple[float, ...] = (1.2, 1.01, 1.0001, 1.000001)
"""Inflation factors of the reconciliation supplement (#110): S4's 1.2 and the three smaller
factors #110 tried on the same ledgers. Not pre-registered; the supplement says so."""

RECON_LEDGERS: tuple[tuple[str, str], ...] = (
    ("phase1", "Phase 1 (S4's pre-registered ledger)"),
    ("phase1+channel_noise", "Phase 1 + C0's channel noise"),
    ("phase1+report_lag", "Phase 1 + C0's report lag"),
    ("phase1+stochastic_delivery", "Phase 1 + the locked stochastic delivery"),
    ("phase1+input_holding_loss", "Phase 1 + the locked input holding loss"),
    ("phase1+ministry", "Phase 1 + C0's ministry layer"),
    ("C0", "C0, seed 0 (the Phase-2 baseline)"),
)
"""The ledgers of the reconciliation supplement (#110), as (name, label). Each `phase1+...` ledger
switches on one mechanism the Phase-2 baseline adds, at its `p2_default_config()` value, over the
Phase-1 configuration; `C0` is `phase2_acceptance.arm_config("C0", 0)`. Built by
`recon_ledger_config`."""

RECON_REPORT_PATH = OUT_DIR / "reconciliation.md"
"""The reconciliation supplement (#110): null residuals and the curve against inflation size."""

RECON_JSON_PATH = OUT_DIR / "reconciliation.json"
"""The numbers behind `RECON_REPORT_PATH`."""

RECONCILIATION_CAVEAT: tuple[str, ...] = (
    "**This curve holds only for a noiseless Phase-1 ledger (#110).** Under Phase-1 delivery the "
    "truthful per-(period, good) residuals are zero up to floating-point round-off, so the size "
    "says how `ledger_test` treats rounding noise, not how often it raises a false alarm on a "
    "ledger with noise in it. Up to estimator 0.1.0+fallback the test ran its t-test on that "
    "round-off; from 0.1.1+fallback it treats it as zero. Once any claim is inflated, its residual "
    "dwarfs the round-off, and rejection depends on how many (period, good) keys carry an "
    "inflated claim, not on how much: in `reconciliation.md`, x1.000001 gives the same curve as "
    "x1.2. That file also gives the null residuals and the same curves on C0, whose truthful "
    "ledgers do not reconcile.",
)
"""The qualification `render_report` prints under the S4 power curve (#110)."""


def arm_grid() -> tuple[tuple[float, float], ...]:
    return tuple((w, cap) for w in NOTCH_WIDTH_GRID for cap in OVERFULFILMENT_CAP_GRID)


def overrides(w: float, cap: float) -> dict[str, dict[str, object]]:
    return {"incentive": {"notch_width": w, "overfulfilment_cap": cap}}


def seeds_for(key: tuple[float, float], n_seeds: int) -> int:
    return n_seeds if key in REUSED_ARMS else min(NEW_ARM_SEEDS, n_seeds)


def run_root_for(key: tuple[float, float], out_dir: Path) -> Path:
    return P1_RUN_ROOT if key in REUSED_ARMS else Path(out_dir) / "runs"


def _setting_id(setting: dict[str, object]) -> str:
    lo, hi = setting["excluded_window"]
    return f"excl[{lo},{hi}]|deg{setting['degree']}|bw{setting['bin_width']}"


def true_excess_mass(x: np.ndarray, x_smooth: np.ndarray, bin_width: float) -> float:
    """Excess mass in [1.00, 1.02] of `x` over its smooth counterpart `x_smooth`, in the estimator's
    units: (P(E) - P_smooth(E)) / mean smooth probability per bin in E (P3 revision S4). NaN when the
    smooth counterpart has no mass there."""
    from gosplan.metrics.phenomena import (
        BUNCHING_EXCESS_HI,
        BUNCHING_EXCESS_LO,
        BUNCHING_WINDOW_HI,
        BUNCHING_WINDOW_LO,
    )

    n_bins = round((BUNCHING_WINDOW_HI - BUNCHING_WINDOW_LO) / bin_width)
    edges = np.linspace(BUNCHING_WINDOW_LO, BUNCHING_WINDOW_HI, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    excess = (centres >= BUNCHING_EXCESS_LO) & (centres <= BUNCHING_EXCESS_HI)
    p = np.histogram(x, bins=edges)[0] / max(len(x), 1)
    q = np.histogram(x_smooth, bins=edges)[0] / max(len(x_smooth), 1)
    per_bin = q[excess].mean()
    if not per_bin > 0:
        return float("nan")
    return float((p[excess].sum() - q[excess].sum()) / per_bin)


def _estimate(x, setting: dict[str, object]) -> dict[str, float]:
    from gosplan.metrics import resolve_estimators
    from gosplan.metrics.phenomena import BUNCHING_WINDOW_HI, BUNCHING_WINDOW_LO

    lo, hi = setting["excluded_window"]
    res = resolve_estimators().bunching_estimate(
        x,
        BUNCHING_WINDOW_LO,
        BUNCHING_WINDOW_HI,
        float(setting["bin_width"]),
        int(setting["degree"]),
        float(lo),
        float(hi),
    )
    return {
        "b_hat": float(res.excess_mass),
        "ci_lo": float(res.ci_lo),
        "ci_hi": float(res.ci_hi),
        "hole_mass": float(res.hole_mass),
    }


def summarise(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    """Bias, RMSE and CI coverage per (source, w, rho_cap, setting). Pure."""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(f"{r['source']}|{r['w']}|{r['rho_cap']}|{r['setting']}", []).append(r)
    bias, rmse, cov, med, n_def, n_tot = {}, {}, {}, {}, {}, {}
    for key, rs in groups.items():
        err = np.array([r["b_hat"] - r["truth"] for r in rs], dtype=float)
        n_tot[key] = int(err.size)
        err = err[np.isfinite(err)]
        # An undefined estimate (no counterfactual support, AMBIGUITY-022) or an undefined truth
        # drops out of bias and RMSE; `n_defined` says how many remain, so none vanish silently.
        n_def[key] = int(err.size)
        bias[key] = float(err.mean()) if err.size else float("nan")
        rmse[key] = float(np.sqrt((err**2).mean())) if err.size else float("nan")
        med[key] = float(np.median(err)) if err.size else float("nan")
        covered = [
            r["ci_lo"] <= r["truth"] <= r["ci_hi"]
            for r in rs
            if np.isfinite(r["truth"]) and np.isfinite(r["ci_lo"]) and np.isfinite(r["ci_hi"])
        ]
        cov[key] = float(np.mean(covered)) if covered else float("nan")
    return {
        "bias": bias,
        "rmse": rmse,
        "ci_coverage": cov,
        "median_error": med,
        "n_defined": n_def,
        "n_total": n_tot,
    }


def measure_rho_seed(job: tuple) -> list[list[float]]:
    """Per-episode measured report ratios of one trained run (100 evaluation episodes), cached as
    `run_dir / "rho_by_episode.json"`."""
    import json

    from gosplan.agents.ppo.adapter import IPPO
    from gosplan.agents.ppo.train import checkpoint_path, evaluate
    from gosplan.experiments._g2 import MEASURE_SEED_OFFSET
    from gosplan.metrics.phenomena import MEASUREMENT_FIRST_PERIOD

    cfg, run_root, sizing, ppo_cfg = job
    run_dir = Path(run_root) / cfg.hash()
    out_path = run_dir / "rho_by_episode.json"
    if out_path.exists():
        return json.loads(out_path.read_text(encoding="utf-8"))
    n_updates = sizing.total_agent_steps // (sizing.n_envs * sizing.rollout_steps)
    agent = IPPO(cfg, ppo_cfg)
    agent.load_checkpoint(checkpoint_path(run_dir, n_updates - 1))
    _m, ledger = evaluate(
        agent, cfg, sizing.measure_episodes, int(cfg.tech.seed_env) + MEASURE_SEED_OFFSET
    )
    by_ep: dict[int, list[float]] = {}
    for rec in ledger.records:
        if rec.phase == "report" and rec.t_period >= MEASUREMENT_FIRST_PERIOD:
            by_ep.setdefault(rec.episode, []).append(float(rec.report_ratio))
    out = [by_ep[e] for e in sorted(by_ep)]
    out_path.write_text(json.dumps(out), encoding="utf-8")
    return out


def reconciliation_power(
    cfg: EnvConfig,
    inflations: tuple[float, ...] = (POWER_INFLATION,),
    replicates: int = POWER_REPLICATES,
) -> dict[str, object]:
    """Rejection rate of `ledger_test` at `POWER_ALPHA` against the share of enterprise-periods whose
    claim is inflated by `POWER_INFLATION` (P3 revision S4).

    The call is per (period, good), the reading PLAN section 7.3 gives the test ("claimed supply of
    each good against the supply implied by buyers' receipts of it"): `reported_supply` is the
    sellers' claimed intermediate supply `(1 - phi_j) * sum R_i`, `received_inputs` is a one-hot row
    carrying the buyers' total receipts of good `j` at the next DELIVER, `io_matrix` the matching
    unit row and `prices` `p_j`. Ledgers: `TruthfulMyopic` on `cfg`, `replicates` episodes; the
    post-hoc inflation draw uses a local generator keyed by replicate (analysis outside
    `gosplan/env/`, CONTRACT rule 9 does not apply).

    `rejection_rate` is S4's curve at `POWER_INFLATION`. Two additions (#110) say what that curve
    can and cannot show. `by_inflation` repeats it for every factor in `inflations` on the same
    ledgers and the same draws, so two curves differ only in the factor. `null` describes the
    residuals with no claim inflated (`null_residuals`): when they are zero up to round-off, the
    size is a property of how the test treats rounding noise, and the curve measures how many
    (period, good) keys carry an inflated claim rather than how large the inflation is."""
    from gosplan.env.prices import initial_prices
    from gosplan.metrics import resolve_estimators

    test = resolve_estimators().reconciliation_ledger_test
    phi = np.asarray(cfg.supply.final_demand_share, dtype=float)
    prices = np.asarray(initial_prices(cfg), dtype=float)
    episodes = truthful_ledgers(cfg, replicates)
    curves = {
        str(f): _rejection_curve(episodes, phi, prices, f, test)
        for f in dict.fromkeys((POWER_INFLATION, *inflations))
    }
    return {
        "alpha": POWER_ALPHA,
        "inflation": POWER_INFLATION,
        "replicates": replicates,
        "form": "per (period, good)",
        "rejection_rate": curves[str(POWER_INFLATION)],
        "by_inflation": {str(f): curves[str(f)] for f in inflations},
        "null": null_residuals(episodes, phi, prices, test),
    }


def truthful_ledgers(cfg: EnvConfig, replicates: int = POWER_REPLICATES) -> list[tuple]:
    """S4's power ledgers: `replicates` `TruthfulMyopic` episodes on `cfg`, at
    `seed_env = cfg.tech.seed_env + 5000 + rep`. Per episode, `(reports, receipts, shipped)`: the
    REPORT rows, and per period the buyers' total receipts of each good and the sellers' total
    shipments of each good at that period's DELIVER (the `produce` rows with `k_step == 0`)."""
    from gosplan.agents.heuristic import TruthfulMyopic
    from gosplan.env.env import GosplanEnv
    from gosplan.metrics.ledger import Ledger

    j_n = cfg.supply.n_sectors
    episodes = []
    for rep in range(replicates):
        env = GosplanEnv(cfg)
        ledger = Ledger()
        env.attach_ledger(ledger)
        agent = TruthfulMyopic(cfg)
        rng = np.random.default_rng(rep)
        obs, _ = env.reset(int(cfg.tech.seed_env) + 5000 + rep, 0)
        done = False
        while not done:
            obs, _r, done, _i = env.step(agent.act(obs, env.phase(), rng))
        reports = [r for r in ledger.records if r.phase == "report"]
        receipts: dict[int, np.ndarray] = {}
        shipped: dict[int, np.ndarray] = {}
        for r in ledger.records:
            if r.phase == "produce" and r.k_step == 0:
                receipts[r.t_period] = receipts.get(r.t_period, np.zeros(j_n)) + np.asarray(r.deliv)
                shipped.setdefault(r.t_period, np.zeros(j_n))[r.sector] += r.shipped
        episodes.append((reports, receipts, shipped))
    return episodes


def _per_good_call(reports, receipts, phi, inflate, inflation: float):
    """The per-(period, good) arguments of S4's `ledger_test` call: `(keys, supply, received,
    unit)`, or `None` with fewer than two keys. `inflate` marks the REPORT rows whose claim is
    multiplied by `inflation`."""
    j_n = phi.shape[0]
    claimed: dict[tuple[int, int], float] = {}
    for rec, up in zip(reports, inflate, strict=True):
        if rec.t_period + 1 not in receipts:
            continue
        key = (rec.t_period, rec.sector)
        claimed[key] = claimed.get(key, 0.0) + rec.report * (inflation if up else 1.0)
    keys = sorted(claimed)
    if len(keys) < 2:
        return None
    supply = np.array([(1.0 - phi[g]) * claimed[(t, g)] for t, g in keys])
    received = np.zeros((len(keys), j_n))
    unit = np.zeros((len(keys), j_n))
    for row, (t, g) in enumerate(keys):
        received[row, g] = receipts[t + 1][g]
        unit[row, g] = 1.0
    return keys, supply, received, unit


def _rejection_curve(episodes, phi, prices, inflation: float, test) -> dict[str, float]:
    """Rejection rate at `POWER_ALPHA` per share in `POWER_SHARES`, claims inflated by
    `inflation`. The draw of which rows are inflated depends on the replicate and the share only,
    so curves at different factors use the same rows."""
    curve = {}
    for share in POWER_SHARES:
        rejections = []
        for rep, (reports, receipts, _shipped) in enumerate(episodes):
            gen = np.random.default_rng([rep, round(share * 1000)])
            inflate = gen.random(len(reports)) < share
            call = _per_good_call(reports, receipts, phi, inflate, inflation)
            if call is None:
                continue
            keys, supply, received, unit = call
            res = test(supply, received, unit, prices[[g for _, g in keys]])
            if np.isfinite(res.p_value):
                rejections.append(res.p_value < POWER_ALPHA)
        curve[str(share)] = float(np.mean(rejections)) if rejections else float("nan")
    return curve


def null_residuals(episodes, phi, prices, test) -> dict[str, object]:
    """The residuals of S4's per-(period, good) call with no claim inflated (#110).

    `relative` is the residual over the priced claimed supply, `(claimed - received) / claimed`,
    pooled over every key of every episode: `mean`, `sd`, `max_abs`, `min` and `n_keys`.
    `max_abs_shipment_gap` is the largest `|received - (1 - phi_j) shipped_j| / received` over the
    same keys, i.e. how far the buyers' receipts sit from what the sellers shipped; when it is
    round-off, any non-zero residual lies between the claims and the shipments. `statistic` and
    `p_value` are per replicate, and `size` is the share of replicates rejected at `POWER_ALPHA`."""
    rel, ship_gap, stats, pvals = [], [], [], []
    for reports, receipts, shipped in episodes:
        call = _per_good_call(reports, receipts, phi, np.zeros(len(reports), dtype=bool), 1.0)
        if call is None:
            continue
        keys, supply, received, unit = call
        res = test(supply, received, unit, prices[[g for _, g in keys]])
        rel.extend((np.asarray(res.residuals) / (prices[[g for _, g in keys]] * supply)).tolist())
        for t, g in keys:
            got, sent = receipts[t + 1][g], (1.0 - phi[g]) * shipped[t + 1][g]
            ship_gap.append(abs(got - sent) / got if got > 0 else abs(got - sent))
        stats.append(float(res.statistic))
        pvals.append(float(res.p_value))
    r = np.asarray(rel, dtype=float)
    finite = [p for p in pvals if np.isfinite(p)]
    return {
        "n_keys": int(r.size),
        "relative_mean": float(r.mean()) if r.size else float("nan"),
        "relative_sd": float(r.std(ddof=1)) if r.size > 1 else float("nan"),
        "relative_max_abs": float(np.abs(r).max()) if r.size else float("nan"),
        "relative_min": float(r.min()) if r.size else float("nan"),
        "max_abs_shipment_gap": float(max(ship_gap)) if ship_gap else float("nan"),
        "statistic": stats,
        "p_value": pvals,
        "size": float(np.mean([p < POWER_ALPHA for p in finite])) if finite else float("nan"),
    }


_RECON_MECHANISMS: dict[str, dict[str, tuple[str, ...]]] = {
    "phase1": {},
    "phase1+channel_noise": {"information": ("channel_noise",)},
    "phase1+report_lag": {"information": ("report_lag",)},
    "phase1+stochastic_delivery": {"supply": ("delivery_timing", "arrival_probs")},
    "phase1+input_holding_loss": {"supply": ("input_holding_loss",)},
    "phase1+ministry": {
        "information": ("ministry_passthrough", "n_ministries", "ministry_pad"),
    },
}
"""The `p2_default_config()` fields each `phase1+...` ledger of `RECON_LEDGERS` copies onto the
Phase-1 configuration. The values are read from `p2_default_config()`, never restated here."""


def recon_ledger_config(name: str) -> EnvConfig:
    """The configuration of one `RECON_LEDGERS` entry (see there)."""
    from gosplan.config import p1_default_config, p2_default_config
    from gosplan.experiments import _g2
    from gosplan.experiments.phase2_acceptance import arm_config

    if name == "C0":
        return arm_config("C0", 0)
    p2 = p2_default_config()
    overrides = {
        section: {f: getattr(getattr(p2, section), f) for f in names}
        for section, names in _RECON_MECHANISMS[name].items()
    }
    return _g2.with_overrides(p1_default_config(), overrides)


def _mechanism_values(name: str) -> str:
    """`field = value` for the fields a `RECON_LEDGERS` entry sets, for the report."""
    from gosplan.config import p2_default_config

    if name not in _RECON_MECHANISMS:
        return "every C0 setting"
    p2 = p2_default_config()
    parts = [
        f"{f} = {getattr(getattr(p2, section), f)}"
        for section, names in _RECON_MECHANISMS[name].items()
        for f in names
    ]
    return ", ".join(parts) if parts else "no change"


def reconciliation_supplement(out_dir: Path = OUT_DIR) -> dict[str, object]:
    """The reconciliation supplement of #110: for every `RECON_LEDGERS` entry, S4's per-(period,
    good) power curve at every factor in `POWER_INFLATIONS`, with the null residuals beside it.

    Takes: `out_dir`, where `reconciliation.json` and `reconciliation.md` are written. Returns:
    the JSON-able result, `{"estimator", "estimator_version", "alpha", "replicates", "shares",
    "inflations", "ledgers": {name: {"label", "settings", "by_inflation", "null"}}}`.

    Descriptive and not pre-registered. S4 fixed the Phase-1 ledger and the factor 1.2; the other
    ledgers and factors are here because the Phase-1 ledger reconciles to round-off under truthful
    play, so its curve cannot say how large an inflation the test detects (#110). Heuristic agents
    only, minutes on one core."""
    import json

    backend = _backend()
    ledgers = {}
    for name, label in RECON_LEDGERS:
        power = reconciliation_power(recon_ledger_config(name), POWER_INFLATIONS)
        ledgers[name] = {
            "label": label,
            "settings": _mechanism_values(name),
            "by_inflation": power["by_inflation"],
            "null": power["null"],
        }
    sup = {
        "estimator": backend[0],
        "estimator_version": backend[1],
        "alpha": POWER_ALPHA,
        "replicates": POWER_REPLICATES,
        "shares": list(POWER_SHARES),
        "inflations": list(POWER_INFLATIONS),
        "ledgers": ledgers,
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / RECON_JSON_PATH.name).write_text(
        json.dumps(sup, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (out_dir / RECON_REPORT_PATH.name).write_text(
        render_reconciliation(sup), encoding="utf-8", newline="\n"
    )
    return sup


def render_reconciliation(sup: dict) -> str:
    """`reconciliation.md` from the result of `reconciliation_supplement`. Pure."""
    shares = [str(s) for s in sup["shares"]]
    led = sup["ledgers"]
    lines = [
        "# Reconciliation power against inflation size, and the null it rests on (#110)",
        "",
        "Supplement to the S4 power curve in `report.md`: descriptive, added after the P3 run, and "
        "not pre-registered. Written by `python -m gosplan.experiments.estimator_bias "
        f"--reconciliation`. Estimator: {sup['estimator']} {sup['estimator_version']}.",
        "",
        "The call is S4's: per (period, good), the sellers' claimed intermediate supply "
        "`(1 - phi_j) sum R_i` against the buyers' receipts of good j at the next DELIVER, "
        f"{sup['replicates']} TruthfulMyopic episodes per ledger at `seed_env + 5000 + rep`, "
        f"alpha = {sup['alpha']}. Which claims are inflated is drawn as in S4, from the replicate "
        "and the share only, so the rows for one ledger differ in the inflation factor alone.",
        "",
        "## Null residuals (no claim inflated)",
        "",
        "Relative residual = (claimed - received) / claimed, pooled over every (period, good) "
        "key of every episode. Shipment gap = the largest |received - (1 - phi_j) shipped_j| / "
        "received over the same keys. Size = share of episodes rejected with no claim inflated.",
        "",
        "| ledger | settings | keys | mean | sd | max abs | shipment gap | size |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for d in led.values():
        n = d["null"]
        lines.append(
            f"| {d['label']} | {d['settings']} | {n['n_keys']} | {n['relative_mean']:.2e} | "
            f"{n['relative_sd']:.2e} | {n['relative_max_abs']:.2e} | "
            f"{n['max_abs_shipment_gap']:.2e} | {n['size']:.2f} |"
        )
    lines += [
        "",
        "## Rejection rate by inflated share and inflation factor",
        "",
        "| ledger | factor | " + " | ".join(f"share {s}" for s in shares) + " |",
        "|---|---|" + "---|" * len(shares),
    ]
    for d in led.values():
        for f, curve in d["by_inflation"].items():
            lines.append(
                f"| {d['label']} | x{f} | " + " | ".join(f"{curve[s]:.2f}" for s in shares) + " |"
            )
    from gosplan.metrics._fallback import RECON_ZERO_RTOL as tol

    lines += ["", "## Reading", ""]
    exact, one_signed, two_signed = [], [], []
    for d in led.values():
        curves = list(d["by_inflation"].values())
        n = d["null"]
        if n["size"] >= 1.0:
            factor = (
                " Every truthful episode is already rejected, so every curve is 1.00 and says "
                "nothing about the inflation."
            )
        elif all(c == curves[0] for c in curves[1:]):
            factor = (
                " The curve is the same at every factor: rejection depends on how many (period, "
                "good) keys carry an inflated claim, not on how much the claim is inflated."
            )
        else:
            factor = " The curve moves with the factor."
        if n["relative_max_abs"] <= tol:
            exact.append(d["label"])
            lines.append(
                f"- {d['label']}: the truthful ledger reconciles to round-off (largest relative "
                f"residual {n['relative_max_abs']:.1e}), so there is no null noise; size "
                f"{n['size']:.2f}." + factor
            )
            continue
        (one_signed if n["relative_min"] >= -tol else two_signed).append(d["label"])
        sign = (
            f"; the most negative is {n['relative_min']:+.1e}, which is round-off, so claims "
            "exceed receipts wherever the two differ"
            if n["relative_min"] >= -tol
            else f"; the smallest is {n['relative_min']:+.2e}"
        )
        lines.append(
            f"- {d['label']}: the truthful residuals are not zero (mean "
            f"{n['relative_mean']:+.2e} of the claimed supply, sd {n['relative_sd']:.2e}{sign}), "
            f"and the test rejects {n['size']:.2f} of the truthful episodes." + factor
        )
    gap = max(d["null"]["max_abs_shipment_gap"] for d in led.values())
    sizes = sorted({f"{d['null']['size']:.2f}" for d in led.values()})
    if gap <= tol:
        lines += [
            "",
            "On every ledger the buyers' receipts equal `(1 - phi_j)` times what the sellers "
            f"shipped, to {gap:.1e}, so a non-zero null residual is a gap between the claims and "
            "the shipments. Delivery timing moves when inputs arrive, not how much is delivered "
            "(`planner.deliver`, P2 revision R6). Under the ministry layer, delivery obligations "
            "are the ministry-forwarded claims, not the sellers' own (R10.3).",
        ]
    if not two_signed:
        lines += [
            "",
            "No ledger here has null residuals spread around zero: "
            f"{len(exact)} reconcile to round-off and {len(one_signed)} have residuals of one "
            f"sign, and the sizes are {' and '.join(sizes)}. None of them gives a power curve at "
            "the nominal size. A curve that measures detection of a given inflation needs a ledger "
            "whose claims and receipts differ by noise, and choosing that noise is a design "
            "decision this supplement does not make.",
        ]
    else:
        lines += ["", "Ledgers with null residuals of both signs: " + ", ".join(two_signed) + "."]
    lines.append("")
    return "\n".join(lines)


def _backend() -> tuple[str, str]:
    from gosplan.metrics import resolve_estimators

    b = resolve_estimators()
    return str(b.name), str(b.version)


def _pool(fn, jobs: list) -> list:
    from gosplan.experiments.contrasts import _pool as pool

    return pool(fn, jobs)


def _plots(rows, power, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pre = _setting_id(PREREGISTERED)
    fig, ax = plt.subplots(figsize=(6, 4))
    for cap, style in ((1.2, "-o"), (float("inf"), "--s")):
        for source in ("dp_exact", "simulation"):
            pts = {}
            for r in rows:
                if r["setting"] == pre and r["rho_cap"] == cap and r["source"] == source:
                    pts.setdefault(r["w"], []).append(r["b_hat"] - r["truth"])
            ws = sorted(pts)
            ax.plot(ws, [np.nanmean(pts[w]) for w in ws], style, label=f"{source}, cap {cap}")
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.set_xlabel("notch width w")
    ax.set_ylabel("bias of b_hat (pre-registered setting)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / BIAS_FIG_PATH.name, dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    shares = [float(k) for k in power["rejection_rate"]]
    ax.plot(shares, list(power["rejection_rate"].values()), "-o")
    ax.axhline(power["alpha"], color="grey", lw=0.8)
    ax.set_xlabel("share of enterprise-periods with inflated claims")
    ax.set_ylabel("rejection rate")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / POWER_FIG_PATH.name, dpi=120)
    plt.close(fig)


def render_report(res: dict, grid) -> str:
    pre = _setting_id(PREREGISTERED)
    lines = [
        "# Estimator-bias study (WO-034, PLAN section 7.2)",
        "",
        "Design: spec/P3_REVISION.md S4. Truth: (a) the DP's exact stationary distribution, (b) the "
        "arm's pooled simulation, each measured against its own w = 0.25 counterpart at the same "
        "cap. Reused Phase-1 gate arms: 30 seeds; other arms: 5 seeds (reduced for compute). "
        f"Estimator: {res['estimator']} {res['estimator_version']}. No digit tests; no calibration "
        "claim about archival detectors.",
        "",
        f"## Pre-registered setting ({pre})",
        "",
        "| w | cap | DP P(rho in [1.00,1.02]) | DP truth | sim truth | defined / seeds | "
        "sim bias | sim RMSE | sim median error | CI coverage |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for arm in res["arms"]:
        w, cap = arm["w"], arm["rho_cap"]
        k_sim = f"simulation|{w}|{cap}|{pre}"
        lines.append(
            f"| {w} | {cap} | {res.get('dp_window_mass', {}).get(f'{w}|{cap}', float('nan')):.3f} | "
            f"{_truth(res, 'dp', w, cap, pre)} | {_truth(res, 'sim', w, cap, pre)} | "
            f"{res.get('n_defined', {}).get(k_sim, '-')} / {res.get('n_total', {}).get(k_sim, '-')} | "
            f"{res['bias'].get(k_sim, float('nan')):.3f} | "
            f"{res['rmse'].get(k_sim, float('nan')):.3f} | "
            f"{res.get('median_error', {}).get(k_sim, float('nan')):.3f} | "
            f"{res['ci_coverage'].get(k_sim, float('nan')):.2f} |"
        )
    lines += [
        "",
        "Notes (read before the numbers):",
        "",
        "- **DP truth is undefined in the estimator's units on every arm.** The DP's reports lie on "
        "its 0.02 report grid, so its stationary distribution is a set of point masses. Every "
        "smooth (w = 0.25) counterpart puts zero mass in the excess window [1.00, 1.02], and S4's "
        "truth divides by that counterpart's mean per-bin mass there. The DP column therefore "
        "reports the defined quantity, the DP's probability of a report in the window. The "
        "estimator is not scored against the DP (the estimator on a 0.02-grid point mass is "
        "degenerate at bin width 0.005 or 0.01 by construction).",
        "- **Undefined estimates are counted, not dropped silently.** When a seed's measured mass "
        "falls almost entirely inside the excluded window, the polynomial counterfactual has no "
        "support and the estimate is infinite or its CI undefined (AMBIGUITY-022, as in the "
        "Phase-1 gate report). `defined / seeds` counts the seeds that enter bias, RMSE and median "
        "error. Near-zero but positive counterfactual support gives finite but huge estimates, "
        "which dominate the mean; the median error is shown beside it as a supplementary, "
        "robust summary (added when reporting, not pre-registered).",
        "- These are the study's findings about the estimator under full bunching; nothing was "
        "re-tuned (PLAN section 4.5).",
    ]
    lines += [
        "",
        "## Across settings (simulation; mean |bias| over arms, mean coverage)",
        "",
        "| setting | mean abs bias | mean RMSE | mean CI coverage |",
        "|---|---|---|---|",
    ]
    for setting in grid:
        sid = _setting_id(setting)
        keys = [k for k in res["bias"] if k.startswith("simulation|") and k.endswith(sid)]
        b = [abs(res["bias"][k]) for k in keys if np.isfinite(res["bias"][k])]
        r = [res["rmse"][k] for k in keys if np.isfinite(res["rmse"][k])]
        c = [res["ci_coverage"][k] for k in keys if np.isfinite(res["ci_coverage"][k])]
        mark = " (pre-registered)" if sid == pre else ""
        lines.append(
            f"| {sid}{mark} | {np.mean(b) if b else float('nan'):.3f} | "
            f"{np.mean(r) if r else float('nan'):.3f} | "
            f"{np.mean(c) if c else float('nan'):.2f} |"
        )
    p = res["power_curve"]
    lines += [
        "",
        "## Reconciliation power curve",
        "",
        f"`ledger_test` at alpha = {p['alpha']}, claims inflated by x{p['inflation']} on the "
        f"given share of enterprise-periods, {p['replicates']} TruthfulMyopic episodes:",
        "",
    ]
    for share, rate in p["rejection_rate"].items():
        lines.append(
            f"- share {share}: rejection rate {rate:.2f}"
            + (" (size)" if float(share) == 0.0 else "")
        )
    lines += ["", *RECONCILIATION_CAVEAT, "", "Figures: `bias_curves.png`, `power_curve.png`.", ""]
    return "\n".join(lines)


def _truth(res, which: str, w, cap, sid: str) -> str:
    d = res["dp_truth"] if which == "dp" else res["sim_truth"]
    v = d.get(f"{w}|{cap}|{sid}")
    if v is None:
        return "n/a"
    return "undefined" if not np.isfinite(v) else f"{v:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
