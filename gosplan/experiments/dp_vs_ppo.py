"""DP-vs-PPO recovery experiment - PLAN sections 4.5 (criterion 1), 5, 12.3 (WO-019) and 14.

Realises: gate G2 criterion 1 of PLAN section 4.5 and the WO-019 card of PLAN section 12.3. Owning
work order: **WO-019** (MID-strong, difficulty 3; depends on WO-014 the DP and WO-018 the training
harness). Gate: **G2 criterion 1** - the single-enterprise recovery check. PLAN section 4.5 is
explicit about its status: *failure of criterion 1 is a training-stack failure and blocks
everything*. It is not a result about planning, it is the check that the optimiser can find an
optimum the analytical layer already knows.

Setting (PLAN section 5, verbatim). One enterprise, no input-output structure, fixed capacity,
geometric continuation `psi = incentive.tenure`. The comparison configuration is therefore the
G1-recorded Phase-1 configuration with `supply.n_enterprises = 1`, an all-zero `io_matrix` and
`final_demand_share` all ones (`a = 0`, `phi = 1`) - a rewriting of the base configuration rather
than a fixed literal, because `J` comes from that configuration. The *same* `EnvConfig` object is
handed to `solve_single_enterprise` and to the PPO training harness, so the DP and the environment
cannot drift: the DP already calls the environment's own `bonus` and penalty functions.

Inputs
    `runs/G1_decision.md` - the human's gate G1 record, which fixes the Phase-1 values of the
    daggered PLAN section 3 rows, the three `a * pen` levels that span the bunching region, and the
    `b_hat_DP` thresholds. This experiment reads those levels; it never chooses them, and it never
    re-derives them from a training result.
    `gosplan.agents.dp.solve_single_enterprise` (WO-014) on the PLAN section 5 `DPGrid` defaults;
    `gosplan.agents.ppo.train` (WO-018) with the pinned reference PPO of WO-017.

Outputs
    `runs/dp_vs_ppo/report.md`      the gate G2 artefact of PLAN section 13: the comparison table,
                                    the Wasserstein-1 distances, and one pass/fail line per
                                    `a * pen` level with the seed count that met the tolerances
    `runs/dp_vs_ppo/table.parquet`  the same comparison, one row per (level, seed), for re-analysis.
                                    The report name is PLAN's verbatim; this filename is a WO-019
                                    convention
    `runs/<config-hash>/`           per-run directory with `manifest.json` (CONTRACT rule 10,
                                    including the reference-PPO version) and the ledger

Cost (PLAN section 14): 3 levels x 10 seeds at `N = 1`, about 2M environment steps - about 1
GPU-hour or about 8 CPU-hours.

Measurement. Periods `t >= 2` of each episode (PLAN section 4.4); under geometric termination there
is no end-of-episode exclusion, and reports at `rho_max` are included and flagged (CONTRACT rule 8).

Runtime bindings. `EnvConfig` is `gosplan.config.EnvConfig` (field-for-field identical to
`spec/spec.py`, enforced by a unit test). The Wasserstein-1 distance uses `scipy.stats`
(`scipy.stats.wasserstein_distance`) or an equivalent exact one-dimensional computation on the
sorted samples; the import lives inside the function that computes it, never at module scope.
"""

from __future__ import annotations

from pathlib import Path

from gosplan.config import EnvConfig

PADDING_TOL = 0.02
"""Gate G2 criterion 1, PLAN section 4.5: PPO's mean fictitious padding must be within 0.02 *ratio
units* of the DP's at each `a * pen` level. Fictitious padding is `mean_i max(0, R_i - S_i) / T_i`
(PLAN section 4.1 row 4), so the tolerance is in the same units as the quantity."""

EFFORT_TOL = 0.05
"""Gate G2 criterion 1: PPO's mean effort must be within 0.05 of the DP's. Effort is in [0, 1]
(PLAN section 2.3), so this is five percentage points of the action range."""

WASSERSTEIN_TOL = 0.03
"""Gate G2 criterion 1: the Wasserstein-1 distance between PPO's and the DP's stationary
`rho_report` distributions must be below 0.03. Distances are in ratio units, on distributions
restricted to the measurement window of PLAN section 4.4."""

SEEDS_PER_LEVEL = 10
"""Training seeds per `a * pen` level (PLAN section 4.5, and the PLAN section 14 cost line)."""

MIN_PASSING_SEEDS = 8
"""Seeds per level that must satisfy all three tolerances simultaneously for criterion 1 to pass:
">= 8 of 10 seeds per value" (PLAN section 4.5). A level with 7 passing seeds fails, and the failure
is reported - the criterion is never restated as a mean over seeds."""

N_AP_LEVELS = 3
"""The three `a * pen` levels of PLAN section 4.5. Their *values* are chosen by the human at gate G1
from the interior of the DP regime map, recorded in `runs/G1_decision.md`, and read from there by
this experiment. They are deliberately not constants in this module: a value chosen after training
would invert the pre-registration."""

MEASUREMENT_WINDOW_START_PERIOD = 2
"""First plan period that enters any measurement, `t >= 2` (PLAN section 4.4). The burn-in excludes
the target-initialisation transient."""

RECOVERY_N_ENTERPRISES = 1
"""`N = 1` for the recovery check (PLAN sections 4.5, 5): the DP solves a single enterprise, so the
comparison is against a single-enterprise environment with `a = 0` and `phi = 1`."""

G1_DECISION_PATH = Path("runs/G1_decision.md")
"""The gate G1 record this experiment reads its three `a * pen` levels, the Phase-1 parameter values
and the `b_hat_DP` thresholds from (PLAN section 13). Its absence is a hard error: running the
recovery check before G1 has been signed off is out of order."""

OUT_DIR = Path("runs/dp_vs_ppo")
"""Artefact directory, relative to the repository root (PLAN sections 12.3, 13)."""

REPORT_PATH = OUT_DIR / "report.md"
"""The gate G2 artefact named in PLAN section 13."""

TABLE_PATH = OUT_DIR / "table.parquet"
"""One row per (level, seed) comparison; a WO-019 convention, not a PLAN-named artefact."""

COMPARED_QUANTITIES: tuple[str, ...] = (
    "fictitious_padding",
    "mean_effort",
    "stationary_rho_wasserstein1",
)
"""The three quantities gate G2 criterion 1 compares, in the order PLAN section 4.5 states them,
each with its tolerance (`PADDING_TOL`, `EFFORT_TOL`, `WASSERSTEIN_TOL`). Nothing else is compared:
adding a fourth quantity after seeing a result would change the criterion after the fact."""


def run(
    cfg: EnvConfig,
    ap_levels: tuple[float, ...],
    out_dir: Path = OUT_DIR,
    seeds_per_level: int = SEEDS_PER_LEVEL,
    seed_env: int | None = None,
) -> dict[str, object]:
    """Run gate G2 criterion 1: PPO against the exact single-enterprise DP.

    Takes: `cfg`, the G1-recorded Phase-1 configuration, already validated - this function applies
    the recovery setting to it (`n_enterprises = 1`, all-zero `io_matrix`, `final_demand_share` all
    ones), so the caller passes the ordinary configuration; `ap_levels`, the `N_AP_LEVELS` values of
    `a * pen` read from `runs/G1_decision.md`; `out_dir`, where the report and table are written;
    `seeds_per_level`, training seeds per level; `seed_env`, the root environment seed - `None`
    means `cfg.tech.seed_env`. Seeds are `seed_env` offset per seed index and shared between the DP
    simulation and PPO evaluation, so the comparison runs under common random numbers (PLAN sections
    2.15, 4.3).

    Returns: a mapping with at least

        "ap_levels"          tuple[float, ...], the levels compared, as read from G1
        "per_seed"           tuple[dict[str, object], ...], one entry per (level, seed) with
                             `padding_ppo`, `padding_dp`, `effort_ppo`, `effort_dp`,
                             `wasserstein1`, and a bool per tolerance
        "passing_seeds"      dict[float, int], seeds meeting all three tolerances, per level
        "criterion_1_passed" bool, `passing_seeds[level] >= MIN_PASSING_SEEDS` at every level
        "flags"              tuple[str, ...], run-level flags, `BOUND_BINDING` included
        "artefacts"          dict[str, str], the paths written

    Procedure (PLAN sections 4.5 criterion 1, 5; WO-019 card):

      1. For each level in `ap_levels`, build the recovery configuration and solve
         `solve_single_enterprise(cfg_level, DPGrid())` once (WO-014). Read off the DP's
         `fictitious_padding`, `mean_effort` and `stationary_rho`.
      2. Train `seeds_per_level` PPO runs on the same configuration with the WO-018 harness at the
         WO-017 adapter's fixed hyper-parameters, logging every step to a `Ledger` and writing
         `runs/<hash>/manifest.json` (CONTRACT rule 10, reference-PPO version included).
      3. Evaluate each trained policy over the measurement window (`t >= 2`) and compute, per seed:
         mean fictitious padding, mean effort, and the Wasserstein-1 distance between the evaluated
         `rho_report` samples and the DP's `stationary_rho`.
      4. A seed passes when `|padding_ppo - padding_dp| <= PADDING_TOL` and
         `|effort_ppo - effort_dp| <= EFFORT_TOL` and `wasserstein1 < WASSERSTEIN_TOL`. A level
         passes when at least `MIN_PASSING_SEEDS` of `seeds_per_level` seeds pass. Criterion 1
         passes when every level passes.
      5. Write `table.parquet` (one row per level and seed) and `report.md` with the comparison
         table, the distances, and one pass/fail line per level stating the seed count.

    A tolerance is never widened and a level is never dropped to obtain a pass: a failure here is a
    training-stack failure that blocks everything downstream (PLAN section 4.5), and the next work
    order is a lead diagnosis, never a parameter change (PLAN section 13).

    Binds: gate G2 criterion 1 of PLAN section 4.5, in full - the three tolerances of
    `COMPARED_QUANTITIES` at `MIN_PASSING_SEEDS` of `SEEDS_PER_LEVEL` seeds, at each of the
    `N_AP_LEVELS` levels recorded at G1.

    Realises: PLAN sections 4.5, 5, 12.3 (WO-019), 13, 14. Owning WO: **WO-019**.
    """
    raise NotImplementedError("PLAN section 4.5 (WO-019) - implemented in WO-019")


def main() -> int:
    """Entry point: read the G1 record, run the recovery check, write the report.

    Takes: nothing. The configuration and the three `a * pen` levels come from
    `runs/G1_decision.md` (`G1_DECISION_PATH`); the seed count is `SEEDS_PER_LEVEL`. Any
    command-line surface and any parallelism over seeds is built inside this function.

    Returns: a process exit code - 0 when all three levels ran and `runs/dp_vs_ppo/report.md` was
    written, 1 when the run could not complete or `runs/G1_decision.md` is missing. The exit code
    does *not* encode the criterion: whether criterion 1 passed is the `criterion_1_passed` line of
    the report, and gate G2 is a written sign-off by the human and the lead on that report together
    with `runs/phase1_gate/report.md` (PLAN section 13).

    Realises: PLAN sections 4.5, 12.3 (WO-019), 13. Owning WO: **WO-019**.
    """
    raise NotImplementedError("PLAN section 4.5 (WO-019) - implemented in WO-019")


if __name__ == "__main__":
    raise SystemExit(main())
