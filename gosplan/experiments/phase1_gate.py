"""Phase-1 gate experiment - PLAN sections 4.4, 4.5 (criteria 2-4), 7.5, 12.3 (WO-020) and 14.

Realises: gate G2 criteria 2, 3 and 4 of PLAN section 4.5 and the WO-020 card of PLAN section 12.3.
Owning work order: **WO-020** (MID-strong, difficulty 3; depends on WO-016 metrics and WO-018
training harness). Gate: **G2 criteria 2-4** - bunching present at the notched schedule and absent
at the smooth counterfactual, the padding elasticity in `a * pen`, and the hygiene flags. Criterion
1 is the neighbouring experiment `gosplan.experiments.dp_vs_ppo` (WO-019).

    THE RESULT CLAUSE (PLAN section 4.5, verbatim in substance). Failure of criterion 1 is a
    training-stack failure and blocks everything. **Failure of criterion 2 with criterion 1 passing
    is a multi-agent effect and is a RESULT** - it is reported, with its CIs and its seed counts,
    and it is never tuned away. No parameter is moved, no arm is re-picked, no estimator setting is
    re-chosen in order to make criterion 2 pass. A gate that fails produces a written failure report
    whose successor is a lead diagnosis (PLAN section 13); parameter changes after G1 create a new,
    labelled study with its own pre-registration.

Arms (PLAN sections 2.8, 4.5). Two schedules, equal-parameter rather than equal-expected-value: the
notched arm and the smooth counterfactual share `beta` and `s` with the G1-recorded configuration
and differ only in the two fields of `ARMS`. The DP of PLAN section 5 supplies the exact predicted
distribution under both, which is where `b_hat_DP` comes from.

Inputs
    `runs/G1_decision.md` - the human's gate G1 record: the Phase-1 values of the daggered PLAN
    section 3 rows, the three `a * pen` levels, and the `b_hat_DP` thresholds. Read, never chosen
    here.
    `gosplan.metrics.phenomena.phenomenon_bunching` and `phenomenon_padding` (WO-016), whose
    defaults are the pre-registered estimator settings of PLAN section 4.5 - bins of width 0.005 on
    `rho` in [0.6, 1.4], excluded window [0.95, 1.02], polynomial degree 7, excess mass on
    [1.00, 1.02], hole mass on [0.95, 1.00), bootstrap SE over seeds. This module must NOT restate
    those numbers: they have exactly one home (WO-016) and are recorded in the manifest from there.
    `gosplan.agents.ppo.train` (WO-018) with the pinned reference PPO of WO-017.
    `gosplan.experiments.price_sensitivity` (WO-036) for the standing price-vector check the WO-020
    card requires on the headline table.

Outputs
    `runs/phase1_gate/report.md`      the gate G2 artefact of PLAN section 13, carrying one
                                      **pass/fail line per criterion** (2, 3, 4), the per-seed
                                      counts behind each, the price-sensitivity table, and every
                                      hygiene flag
    `runs/phase1_gate/table.parquet`  per-(arm, seed) estimates and CIs; a WO-020 convention
    `runs/<config-hash>/`             per-run directory with `manifest.json` (CONTRACT rule 10) and
                                      the ledger

Cost (PLAN section 14): 2 arms x 30 seeds at `N = 20`, about 5M environment steps at roughly 3k
steps/s in NumPy (about 30 minutes per run) - about 30 CPU-hours, about 4 hours on 8 cores.

Measurement. Periods `t >= 2` of every episode (PLAN section 4.4). Under geometric termination there
is no end-of-episode exclusion. Reports at `rho_max` are included in the histograms and flagged
(CONTRACT rule 8) - a bound is never silently moved to change a result.

Held out. Rows 2, 5, 6 and 7 of PLAN section 4.1 are not computed here, not plotted here and not
tested here. Phase 1 touches rows 1 (bunching) and 4 (padding) only; WO-030 computes the held-out
rows for the first time, in the Phase-2 acceptance run.

Runtime bindings. `EnvConfig` is `gosplan.config.EnvConfig` (field-for-field identical to
`spec/spec.py`, enforced by a unit test). Bootstrap and plotting dependencies are imported inside
the function that uses them, never at module scope.
"""

from __future__ import annotations

from pathlib import Path

from gosplan.config import EnvConfig

N_SEEDS = 30
"""Seeds per arm (PLAN sections 4.3, 4.5 and the PLAN section 14 cost line), run under common random
numbers: every arm shares `seed_env`, so the two schedules meet identical environment draws (PLAN
section 2.15) and the contrast is within-draw."""

ARMS: dict[str, dict[str, dict[str, object]]] = {
    "notched": {"incentive": {"notch_width": 0.0, "overfulfilment_cap": 1.2}},
    "smooth": {"incentive": {"notch_width": 0.25, "overfulfilment_cap": float("inf")}},
}
"""The two gate G2 criterion 2 arms, as overrides on the G1-recorded configuration (PLAN sections
2.8, 4.5). `notched` is the Phase-1 schedule: a true discontinuity at `rho = 1` plus a kink at the
cap. `smooth` is the counterfactual with no discontinuity and no kink anywhere - `w = 0.25` and
`rho_cap = inf`, which means no clip at all. The two arms share `notch_height` and
`overfulfilment_slope`: the comparison is equal-parameter by design."""

BHAT_DP_FRACTION = 0.5
"""Gate G2 criterion 2: at the notched arm, `b_hat >= 0.5 * b_hat_DP`, where `b_hat_DP` is the
DP-predicted excess mass **for the same parameters**, recorded at G1 in `runs/G1_decision.md`."""

SEED_PASS_FRACTION = 0.90
"""Gate G2 criterion 2: each of its two conditions must hold in at least 90% of the `N_SEEDS` seeds
- the notched arm's `b_hat >= BHAT_DP_FRACTION * b_hat_DP` with a bootstrap CI excluding 0, and the
smooth arm's `b_hat` CI covering 0."""

PADDING_DP_TOL = 0.03
"""Gate G2 criterion 3: at each of the three `a * pen` levels, learned fictitious padding must be
within 0.03 (ratio units) of the DP's, and the sequence across levels must be monotone
decreasing."""

PADDING_MONOTONE_IN = "audit_rate * penalty_scale"
"""The compound quantity criterion 3's monotonicity is stated in (PLAN sections 4.5, 4.1 row 4):
learned fictitious padding must fall as `a * pen` rises. `audit_rate` is dual-classified - INFO, and
also entering the reward through the penalty - which is why the compound, not either factor, is the
sweep variable and why the elasticity is always reported against it by name."""

BOUND_FLAG = "BOUND_BINDING"
"""Gate G2 criterion 4, first half: this flag must be absent from every run. It is raised by the
ledger when more than 1% of reports sit at `rho_max` (CONTRACT rule 8, test T-B8). If it is raised,
the result is reported carrying the flag; the bound is never widened or narrowed to clear it."""

TARGET_RUNAWAY_MULT = 3.0
"""Gate G2 criterion 4, second half: an episode counts as a target runaway when `T > 3 * T_0` is
reached during it, with `T_0 = tech.initial_target_frac * A_{s(i)} * cap_i` (PLAN section 3)."""

TARGET_RUNAWAY_MAX_FRAC = 0.05
"""Gate G2 criterion 4: target runaways must occur in fewer than 5% of training episodes counted
over the window that begins at `HYGIENE_WINDOW_START_FRAC` of training."""

HYGIENE_WINDOW_START_FRAC = 0.20
"""Gate G2 criterion 4: the runaway fraction is measured over training episodes *after the first
20% of training*, so the early transient of finding F4 is excluded by pre-registration rather than
by inspection."""

MEASUREMENT_WINDOW_START_PERIOD = 2
"""First plan period entering any measurement, `t >= 2` (PLAN section 4.4)."""

CRITERIA: tuple[str, ...] = ("criterion_2", "criterion_3", "criterion_4")
"""The criteria this experiment evaluates: bunching present/absent (2), padding elasticity (3),
hygiene (4). Criterion 1 belongs to `gosplan.experiments.dp_vs_ppo` (WO-019). The report carries one
pass/fail line per name in this tuple, and G2 passes only if all four criteria pass across the two
reports (PLAN section 4.5)."""

OUT_DIR = Path("runs/phase1_gate")
"""Artefact directory, relative to the repository root (PLAN sections 12.3, 13)."""

REPORT_PATH = OUT_DIR / "report.md"
"""The gate G2 artefact named in PLAN section 13."""

TABLE_PATH = OUT_DIR / "table.parquet"
"""Per-(arm, seed) estimates, CIs and flags; a WO-020 convention, not a PLAN-named artefact."""

G1_DECISION_PATH = Path("runs/G1_decision.md")
"""The gate G1 record supplying the Phase-1 values, the three `a * pen` levels and the `b_hat_DP`
thresholds (PLAN section 13). Its absence is a hard error: the thresholds must exist in writing
before any training run."""


def run(
    cfg: EnvConfig,
    b_hat_dp: float,
    ap_levels: tuple[float, ...],
    out_dir: Path = OUT_DIR,
    n_seeds: int = N_SEEDS,
    seed_env: int | None = None,
) -> dict[str, object]:
    """Run gate G2 criteria 2-4 and write the Phase-1 gate report.

    Takes: `cfg`, the G1-recorded Phase-1 configuration at `N = 20`, already validated; `b_hat_dp`,
    the DP-predicted excess mass for the same parameters, read from `runs/G1_decision.md`;
    `ap_levels`, the three `a * pen` levels from the same record; `out_dir`, where the report and
    table are written; `n_seeds`, seeds per arm; `seed_env`, the root environment seed - `None`
    means `cfg.tech.seed_env`. The same `seed_env` is used for both arms, so the notched and smooth
    runs share environment draws (common random numbers, PLAN sections 2.15 and 4.3).

    Returns: a mapping with at least

        "criterion_2"        dict with `notched_pass_frac`, `smooth_pass_frac`, `b_hat_dp`,
                             per-seed `b_hat`, `ci_lo`, `ci_hi`, `hole_mass`, and `passed`
        "criterion_3"        dict with per-level learned padding, the DP's padding, the deviations,
                             a `monotone` bool, and `passed`
        "criterion_4"        dict with `bound_binding_runs`, `runaway_frac` per run, and `passed`
        "price_sensitivity"  dict, the PLAN section 7.5 table on this experiment's headline metrics,
                             delegated to `gosplan.experiments.price_sensitivity`
        "flags"              tuple[str, ...], every run-level flag raised
        "passed"             bool, all three criteria passed
        "artefacts"          dict[str, str], the paths written

    Procedure (PLAN section 4.5, criteria 2-4; WO-020 card):

      1. Train `n_seeds` runs per arm in `ARMS` on the G1-recorded configuration with the WO-018
         harness, logging every step to a `Ledger` and writing `runs/<hash>/manifest.json` with the
         estimator version and the reference-PPO version (CONTRACT rule 10).
      2. Criterion 2. Per seed, evaluate over the measurement window and call
         `phenomenon_bunching` (WO-016) with its pre-registered PLAN section 4.5 defaults. The
         notched arm's condition is `b_hat >= BHAT_DP_FRACTION * b_hat_dp` with the bootstrap CI
         excluding 0; the smooth arm's condition is that the CI covers 0. Each must hold in at least
         `SEED_PASS_FRACTION` of seeds. Hole mass on [0.95, 1.00) is reported alongside, as PLAN
         section 4.1 row 1 requires.
      3. Criterion 3. Per level in `ap_levels`, evaluate `phenomenon_padding` (WO-016) and compare
         the learned fictitious padding with the DP's at the same level: monotone decreasing in
         `PADDING_MONOTONE_IN`, and within `PADDING_DP_TOL` at each level.
      4. Criterion 4. `BOUND_FLAG` must be absent from every run's manifest, and the fraction of
         training episodes reaching `T > TARGET_RUNAWAY_MULT * T_0` after the first
         `HYGIENE_WINDOW_START_FRAC` of training must be below `TARGET_RUNAWAY_MAX_FRAC`.
      5. Run the standing price-vector check of PLAN sections 2.9.4 and 7.5 on this experiment's
         headline table through `gosplan.experiments.price_sensitivity.run`, and include its table
         in the report. A sign change in `specification_gap` is reported, never suppressed.
      6. Write `table.parquet` and `report.md`, the latter carrying one pass/fail line per name in
         `CRITERIA`, the seed counts behind each, the estimator settings and versions actually used,
         and every flag raised.

    Nothing here is tuned to obtain a pass. If criterion 2 fails while criterion 1 passes in
    `runs/dp_vs_ppo/report.md`, that combination is a multi-agent effect and is reported as the
    result it is (PLAN section 4.5).

    Binds: gate G2 criteria 2, 3 and 4 of PLAN section 4.5, in full.

    Realises: PLAN sections 4.4, 4.5, 7.5, 12.3 (WO-020), 13, 14. Owning WO: **WO-020**.
    """
    raise NotImplementedError("PLAN section 4.5 (WO-020) - implemented in WO-020")


def main() -> int:
    """Entry point: read the G1 record, run criteria 2-4, write the Phase-1 gate report.

    Takes: nothing. The configuration, the `b_hat_DP` thresholds and the three `a * pen` levels come
    from `runs/G1_decision.md` (`G1_DECISION_PATH`); the seed count is `N_SEEDS` and the arms are
    `ARMS`. Any command-line surface and any parallelism over seeds is built inside this function.

    Returns: a process exit code - 0 when both arms and all three levels ran and
    `runs/phase1_gate/report.md` was written, 1 when the run could not complete or
    `runs/G1_decision.md` is missing. The exit code does *not* encode the criteria: pass and fail
    live on the report's per-criterion lines, and gate G2 is a written sign-off by the human and the
    lead on this report together with `runs/dp_vs_ppo/report.md` (PLAN section 13). A criterion-2
    failure is a result to be reported, not an error to be retried.

    Realises: PLAN sections 4.5, 12.3 (WO-020), 13. Owning WO: **WO-020**.
    """
    raise NotImplementedError("PLAN section 4.5 (WO-020) - implemented in WO-020")


if __name__ == "__main__":
    raise SystemExit(main())
