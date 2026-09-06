"""Contrast harness (Claim B) - PLAN sections 4.3, 7.1, 12.5 (WO-032), 13 (gate G4) and 14.

Realises: the contrast design of PLAN section 4.3, referenced by PLAN section 7.1. Owning work
order: **WO-032** (MID-strong, Phase 3). Gate: **G4** - the final report needs the contrasts with
their CIs, alongside the estimator-bias curves, the LLM study and the price sensitivity on every
headline table (PLAN section 13).

What it answers. Claim B is comparative: how much of the measured loss is closed by moving the
*information architecture* (C_OGAS), how much by moving the *incentive structure* (C_INC), and
whether the two interact. The reported quantities are

    Delta_X = welfare_ratio(C_X) - welfare_ratio(C0)          for each contrast X
    I       = Delta_BOTH - Delta_OGAS - Delta_INC             the interaction

and the OGAS conclusion is the sign and magnitude of `Delta_OGAS` relative to `Delta_INC` and `I`
(PLAN section 4.3).

    TWO REPORTING RULES (PLAN section 4.3), binding on every table, figure and sentence produced
    from this harness:

    1. **C_AUDIT is dual-classified and is NEVER folded into C_OGAS.** `audit_rate` is an
       information parameter that also enters the reward through the penalty (PLAN sections 3, 4.3),
       so an audit change is not a pure information change. C_AUDIT is always reported as its own
       row, with its own Delta. Adding it to C_OGAS - in a table, in a total, or in prose - is a
       violation of the pre-registration.
    2. **No "X% of the loss is informational" statement may be produced without a named contrast and
       a CI.** Every such number is `Delta_X` for a contrast named in `CONTRASTS`, reported as an
       IQM with its stratified bootstrap 95% interval. A percentage without a contrast name and an
       interval attached is not a result this design can produce.

Inputs
    The post-G3 Phase-2 full configuration as C0 (PLAN section 4.3), plus the overrides in
    `CONTRASTS`; `gosplan.agents.ppo.train` (WO-018, JAX path after WO-029) at `N_SEEDS` seeds under
    common random numbers; `gosplan.metrics` for the three outcomes; `rliable` (the `stats` extra)
    for the IQM and the stratified bootstrap.

Outputs
    `runs/contrasts/table.parquet`  one row per (contrast, seed, outcome), plus the aggregate rows
                                    carrying IQM and CI bounds
    `runs/contrasts/report.md`      the contrast table, the Deltas, the interaction `I`, and the two
                                    reporting rules restated verbatim above the table
    `runs/<config-hash>/`           per-run directories with `manifest.json` (CONTRACT rule 10)

    PLAN section 12.5 does not name artefact paths for WO-032; these follow the
    `runs/<experiment>/` convention of the Phase-1 cards.

Cost (PLAN section 14): 5 contrasts x 30 seeds, on the JAX path - hours.

Runtime bindings. `EnvConfig` is `gosplan.config.EnvConfig` (field-for-field identical to
`spec/spec.py`, enforced by a unit test). `rliable`, `pyarrow` and any plotting dependency are
imported inside the function that uses them, never at module scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gosplan.config import EnvConfig


@dataclass(frozen=True)
class Multiplier:
    """A relative override: multiply the C0 value of a field rather than replacing it.

    PLAN section 4.3 states one contrast relatively - C_AUDIT is `audit_rate x4` - and the rest
    absolutely. Wrapping the relative case in its own type keeps `CONTRASTS` unambiguous data: a
    bare number in an override dict is an absolute value, a `Multiplier` is a factor applied to the
    C0 configuration. The applier validates the resulting configuration, so a x4 that would leave
    `audit_rate` above 1.0 raises rather than silently clipping.
    """

    factor: float
    """The factor applied to the C0 value of the field this override is attached to."""


ContrastOverrides = dict[str, dict[str, object]]
"""Type of a contrast's override table: arm name (`"supply"`, `"incentive"`, `"information"`,
`"tech"`) mapped to field name mapped to either an absolute value or a `Multiplier`."""

CONTRASTS: dict[str, ContrastOverrides] = {
    "C0": {},
    "C_OGAS": {
        "information": {
            "report_lag": 0,
            "aggregation_level": "enterprise",
            "channel_noise": 0.0,
            "ministry_passthrough": 1.0,
            "shortfall_visibility": 1.0,
        },
    },
    "C_AUDIT": {
        "information": {
            "audit_rate": Multiplier(4.0),
            "audit_noise": 0.0,
        },
    },
    "C_INC": {
        "incentive": {
            "notch_width": 0.25,
            "overfulfilment_cap": float("inf"),
            "objective_metric": "net_output",
            "ratchet_lambda": 0.1,
        },
    },
    "C_BOTH": {
        "information": {
            "report_lag": 0,
            "aggregation_level": "enterprise",
            "channel_noise": 0.0,
            "ministry_passthrough": 1.0,
            "shortfall_visibility": 1.0,
        },
        "incentive": {
            "notch_width": 0.25,
            "overfulfilment_cap": float("inf"),
            "objective_metric": "net_output",
            "ratchet_lambda": 0.1,
        },
    },
}
"""The PLAN section 4.3 contrast table as data, verbatim.

    C0       the Phase-2 full configuration with the post-G3 values; no overrides
    C_OGAS   `report_lag -> 0`, `aggregation_level -> enterprise`, `channel_noise -> 0`,
             `ministry_passthrough -> 1`, `shortfall_visibility -> 1`; **audit unchanged**
    C_AUDIT  `audit_rate x4`, `audit_noise -> 0`; dual-classified, reported separately, never folded
             into C_OGAS
    C_INC    `notch_width -> 0.25`, `overfulfilment_cap -> inf`, `objective_metric -> net_output`,
             `ratchet_lambda -> 0.1`
    C_BOTH   C_OGAS + C_INC, written out in full rather than composed at import time, so the file
             states exactly what is run

C_OGAS deliberately leaves `audit_rate` and `audit_noise` at their C0 values: the audit change is
C_AUDIT's alone. `horizontal_visibility` is not part of C_OGAS either - it is the blat mechanism
locked at its PLAN section 4.2 value and is not an OGAS lever."""

DUAL_CLASSIFIED: tuple[str, ...] = ("C_AUDIT",)
"""Contrasts whose parameters are dual-classified (INFO and, through the penalty, also the reward).
Declared as data so a reviewer, a table builder and a reader can all check the same list: nothing
here may be summed into, averaged with, or presented as part of C_OGAS (PLAN sections 3, 4.3)."""

N_SEEDS = 30
"""Seeds per contrast (PLAN sections 4.3, 14), all sharing `seed_env`, so every contrast meets
identical environment draws - common random numbers by construction (PLAN section 2.15)."""

OUTCOMES: tuple[str, ...] = ("welfare_ratio", "padding_index", "specification_gap")
"""The three headline outcomes of PLAN sections 2.9.4 and 4.3, all dimensionless:

    padding_index      val_measured / val_true                       (>= 1 under fictitious output)
    welfare_ratio      W / W_oracle                                  (oracle from PLAN section 6.2)
    specification_gap  val_measured / val_oracle - W / W_oracle      (0 when honest and efficient)

Every table states which denominator `W_oracle` came from; in Phase 1 it is the clearly labelled
`W_truthful_max` placeholder (PLAN section 2.9.4)."""

DELTA_OUTCOME = "welfare_ratio"
"""The outcome the reported `Delta_X` and the interaction `I` are computed on (PLAN section 4.3:
`Delta_X = welfare_ratio(C_X) - welfare_ratio(C0)`). The other two outcomes are reported per
contrast in the same table but do not enter `I`."""

INTERACTION_FORMULA = "I = Delta_BOTH - Delta_OGAS - Delta_INC"
"""The interaction of PLAN section 4.3, stated as data so the report and the code cite the same
expression. `I > 0` means information and incentive changes are complements, `I < 0` substitutes;
the OGAS conclusion is the sign and magnitude of `Delta_OGAS` relative to `Delta_INC` and `I`."""

CI_LEVEL = 0.95
"""Confidence level of the stratified bootstrap intervals (PLAN section 4.3): 95%."""

AGGREGATOR = "IQM"
"""Point summary across seeds (PLAN section 4.3): the interquartile mean, with stratified bootstrap
CIs, as implemented by `rliable`. The number of bootstrap resamples and the `rliable` version are
recorded in the manifest (CONTRACT rule 10) rather than fixed here."""

OUT_DIR = Path("runs/contrasts")
"""Artefact directory, relative to the repository root; a WO-032 convention."""

TABLE_PATH = OUT_DIR / "table.parquet"
"""Per-(contrast, seed, outcome) rows plus the aggregate IQM/CI rows."""

REPORT_PATH = OUT_DIR / "report.md"
"""The contrast table, the Deltas, the interaction, and the two reporting rules restated above
it."""


def run(
    cfg: EnvConfig,
    out_dir: Path = OUT_DIR,
    contrasts: dict[str, ContrastOverrides] | None = None,
    n_seeds: int = N_SEEDS,
    seed_env: int | None = None,
) -> dict[str, object]:
    """Run every contrast at `n_seeds` seeds and write the contrast table.

    Takes: `cfg`, the C0 configuration - the Phase-2 full configuration with the post-G3 values,
    already validated; `out_dir`, where the table and report are written; `contrasts`, the override
    tables to run - `None` means `CONTRASTS`, and a caller narrowing the set must still report
    C_AUDIT separately; `n_seeds`, seeds per contrast; `seed_env`, the root environment seed -
    `None` means `cfg.tech.seed_env`. One `seed_env` is shared by every contrast and every seed
    index, which is what makes the design common-random-number paired (PLAN sections 2.15, 4.3).

    Returns: a mapping with at least

        "outcomes"      dict[str, dict[str, dict[str, float]]], contrast -> outcome ->
                        {"iqm", "ci_lo", "ci_hi"} at `CI_LEVEL`
        "per_seed"      tuple[dict[str, object], ...], one row per (contrast, seed, outcome)
        "deltas"        dict[str, dict[str, float]], contrast -> {"delta", "ci_lo", "ci_hi"} for
                        `DELTA_OUTCOME`, each `Delta_X = welfare_ratio(C_X) - welfare_ratio(C0)`
                        computed seed-paired before aggregation, never as a difference of IQMs of
                        unpaired samples
        "interaction"   dict[str, float], `I` with its CI, per `INTERACTION_FORMULA`
        "dual_classified" tuple[str, ...], `DUAL_CLASSIFIED`, carried into the report
        "flags"         tuple[str, ...], run-level flags, `BOUND_BINDING` included
        "artefacts"     dict[str, str], the paths written

    Procedure (PLAN sections 4.3, 7.1):

      1. Build one configuration per contrast by applying its override table to `cfg`, resolving
         `Multiplier` entries against the C0 value, and validating the result (a x4 that pushes a
         probability above 1 raises rather than clipping).
      2. Train and evaluate `n_seeds` runs per contrast under common random numbers, logging to a
         `Ledger` and writing `runs/<hash>/manifest.json` (CONTRACT rule 10, `rliable` and
         reference-PPO versions included).
      3. Compute the three `OUTCOMES` per (contrast, seed) over the measurement window of PLAN
         section 4.4, then the IQM with stratified bootstrap `CI_LEVEL` intervals via `rliable`.
      4. Compute `Delta_X` on `DELTA_OUTCOME` seed-paired, and the interaction `I` per
         `INTERACTION_FORMULA`, each with a bootstrap CI.
      5. Write `table.parquet` and `report.md`. The report restates the two reporting rules above
         the table: C_AUDIT is its own row and is never folded into C_OGAS, and no "X% of the loss
         is informational" sentence appears without a named contrast and a CI.

    Every Delta is a difference between two named contrasts, and every number in the report carries
    an interval. A contrast whose runs are flagged non-converged by
    `gosplan.experiments.exploitability` is reported with that label rather than as a result (PLAN
    section 6.3).

    Binds: gate G4 of PLAN section 13 - "contrasts with CIs" - and the design of PLAN section 4.3.

    Realises: PLAN sections 4.3, 7.1, 12.5 (WO-032), 13, 14. Owning WO: **WO-032**.
    """
    raise NotImplementedError("PLAN section 4.3 (WO-032) - implemented in WO-032")


def main() -> int:
    """Entry point: run the five contrasts on the post-G3 configuration and write the report.

    Takes: nothing; the C0 configuration is the Phase-2 full configuration recorded after gate G3,
    the contrasts are `CONTRASTS`, and the seed count is `N_SEEDS`. Any command-line surface and any
    JAX device setup is built inside this function.

    Returns: a process exit code - 0 when every contrast ran and `runs/contrasts/report.md` was
    written, 1 otherwise. The exit code encodes nothing about the conclusion: `Delta_OGAS`,
    `Delta_INC` and `I` are read off the report with their intervals, and gate G4 is the human's
    sign-off on the final report (PLAN section 13).

    Realises: PLAN sections 4.3, 12.5 (WO-032), 13. Owning WO: **WO-032**.
    """
    raise NotImplementedError("PLAN section 4.3 (WO-032) - implemented in WO-032")


if __name__ == "__main__":
    raise SystemExit(main())
