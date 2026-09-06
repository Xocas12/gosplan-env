"""Price-vector sensitivity - PLAN sections 2.9.4, 2.10, 7.5, 12.5 (WO-036), 13 (gate G4) and 14.

Realises: the standing robustness check of PLAN section 2.9.4, scoped as PLAN section 7.5. Owning
work order: **WO-036** (MID-fast, Phase 3). Gate: **G4** - PLAN section 13 requires "price
sensitivity on every headline table".

    THE STANDING RULE (PLAN sections 2.9.4, 7.5). Recompute all three headline metrics under
    `N_PRICE_VECTORS` perturbed price vectors, `p_j * exp(u_j)` with `u ~ N(0, 0.3**2)` at fixed
    seeds, **on every headline table** - the Phase-1 gate table, the contrast table, the
    estimator-bias summary, the Phase-2 acceptance table. Not a subset, and not only the tables
    whose result survives.

    **A sign change in `specification_gap` is reported, never suppressed** (PLAN sections 2.9.4,
    7.5). `specification_gap = val_measured / val_oracle - W / W_oracle` is zero when the system is
    honest and efficient, so its sign is the direction of the whole story: if a plausible reprice
    flips it, that fact is the result, and it appears in the same table as the headline number
    rather than in a footnote or an appendix.

Why prices matter here. All three headline metrics of PLAN section 2.9.4 are built on plan prices:

    padding_index      = val_measured / val_true
    welfare_ratio      = W / W_oracle
    specification_gap  = val_measured / val_oracle - W / W_oracle

with `val_measured = sum_i p_{s(i)} * R_i * q_hat_i` and `val_true = sum_i p_{s(i)} * y_i * qbar_i`
(PLAN section 2.9.3). Plan prices are a cost-plus fixed point solved once at `t = 0` (PLAN section
2.10) - a modelling choice, not a market outcome - so every price-weighted conclusion must be shown
to survive a different plausible weighting.

    RECOMPUTATION VERSUS RE-RUN (a distinction WO-036 must respect). For `objective_metric = "val"`
    - Phase 1, and C0 in the contrasts - prices enter only the logged aggregates, so a perturbed
    price vector is a pure **post-hoc recomputation** over an existing ledger: no retraining, and
    the environment is untouched. For `objective_metric = "net_output"` - the C_INC and C_BOTH arms
    of PLAN section 4.3 - plan prices enter the *fulfilment measure* (PLAN section 2.9.2) and hence
    the reward, so a perturbed price vector is a **different environment**: the check there is a
    re-run at the perturbed prices, not a recomputation, and it costs one extra run per seed. Every
    row of the output states which of the two it is; a recomputation is never labelled as a re-run
    or the reverse.

Inputs
    The headline table to check, and the ledgers behind it (`runs/<config-hash>/`, WO-011). The
    unperturbed plan prices come from `initial_prices(cfg)` (PLAN section 2.10).

Outputs
    `runs/price_sensitivity/table.parquet`  one row per (source table, row, price vector, metric):
                                            the baseline value, the perturbed value, the seed and
                                            the draw, and the recomputation-or-re-run label
    `runs/price_sensitivity/report.md`      the sensitivity block for every headline table, with the
                                            sign-change column for `specification_gap` first

    The same block is embedded beneath each headline table by the report generator (WO-037), which
    is what "on every headline table" means in practice. PLAN section 12.5 names no artefact paths
    for WO-036; these follow the `runs/<experiment>/` convention of the Phase-1 cards.

Cost (PLAN section 14): no separate line - the recomputation path is CPU minutes over existing
ledgers. The re-run path described above costs one extra run per affected arm and seed, and is
budgeted with the arm it audits.

OPEN - AMBIGUITY FOR THE WO-036 SESSION (CONTRACT rule 3; do not silently choose)
    PLAN section 2.10 also assigns the consumer CES parameters `ces_alpha` and `ces_sigma` to this
    check ("swept in the price-sensitivity check, not in the treatment arms"), but neither PLAN
    section 2.9.4 nor PLAN section 3 gives them a sweep grid, and PLAN section 7.5 specifies only
    the price-vector perturbation. The price-vector half of this module is fully determined; the CES
    half is not. File an AMBIGUITY REPORT and let the lead fix the grid; whatever is used is printed
    in the table beside the indices it moves.

Runtime bindings. `EnvConfig` is `gosplan.config.EnvConfig` (field-for-field identical to
`spec/spec.py`, enforced by a unit test). The perturbation draws are analysis-time, not environment
draws: CONTRACT rule 9 governs `gosplan/env/`, and `Purpose` (PLAN section 2.15) has no price
member, so a local `numpy.random.default_rng(seed)` seeded from `PRICE_PERTURBATION_SEEDS` is used
here - never a module-level global generator, and never an addition to the frozen `Purpose` enum.
"""

from __future__ import annotations

from pathlib import Path

from gosplan.config import EnvConfig

N_PRICE_VECTORS = 3
"""Perturbed price vectors per headline table (PLAN sections 2.9.4, 7.5): "three perturbed price
vectors"."""

PRICE_LOG_SIGMA = 0.3
"""Log-scale of the perturbation: `p_j * exp(u_j)` with `u ~ N(0, 0.3**2)` (PLAN section 2.9.4,
verbatim). The draw is per good `j`, so relative prices move, not just the overall level - a common
scale factor would cancel out of every ratio in PLAN section 2.9.4 and check nothing."""

PRICE_PERTURBATION_SEEDS: tuple[int, ...] = (0, 1, 2)
"""The "fixed seeds" of PLAN section 2.9.4. PLAN fixes that the seeds are *fixed and reused*, not
which integers they are; these three are this module's registered choice and they are recorded in
the manifest. The requirement they satisfy is that the same three perturbed price vectors are
applied to every headline table, so sensitivity is comparable across tables rather than being three
fresh draws each time."""

HEADLINE_METRICS: tuple[str, ...] = ("padding_index", "welfare_ratio", "specification_gap")
"""The three dimensionless headline metrics of PLAN section 2.9.4, all recomputed under every
perturbed vector. `welfare_ratio` moves only through its denominator when `W_oracle` is itself
price-weighted, and every table states which `W_oracle` it used - in Phase 1 the clearly labelled
`W_truthful_max` placeholder."""

SIGN_CHANGE_METRIC = "specification_gap"
"""The metric whose sign change PLAN sections 2.9.4 and 7.5 single out for reporting. The output
carries an explicit `sign_changed` column for it, and the report puts that column first: a sign
change is the headline of the check, never a suppressed footnote."""

RECOMPUTATION_LABELS: tuple[str, ...] = ("recomputation", "rerun")
"""The two ways a perturbed price vector can be applied, one of which labels every output row.
`recomputation` when prices enter only the logged aggregates (`objective_metric = "val"`); `rerun`
when they enter the fulfilment measure and hence the reward (`objective_metric = "net_output"`,
PLAN section 2.9.2), which makes the perturbed configuration a different environment."""

RERUN_TRIGGER_FIELD = "incentive.objective_metric"
"""The configuration field that decides between the two labels above. `net_output` (and any future
metric that reads `plan_prices`) forces the `rerun` path; `val` and `quality_weighted` do not read
prices in the fulfilment measure and take the `recomputation` path."""

OUT_DIR = Path("runs/price_sensitivity")
"""Artefact directory, relative to the repository root; a WO-036 convention."""

TABLE_PATH = OUT_DIR / "table.parquet"
"""One row per (source table, row, price vector, metric)."""

REPORT_PATH = OUT_DIR / "report.md"
"""The sensitivity block for every headline table, sign-change column first."""


def run(
    cfg: EnvConfig,
    ledger_dirs: tuple[Path, ...],
    table_name: str,
    out_dir: Path = OUT_DIR,
    seeds: tuple[int, ...] = PRICE_PERTURBATION_SEEDS,
) -> dict[str, object]:
    """Recompute one headline table's metrics under the three perturbed price vectors.

    Takes: `cfg`, the configuration behind the table, already validated - its
    `RERUN_TRIGGER_FIELD` decides which of `RECOMPUTATION_LABELS` applies; `ledger_dirs`, the
    `runs/<config-hash>/` directories holding the ledgers behind each row of the table (one per
    seed or per arm-seed); `table_name`, the headline table being checked, recorded in every output
    row so the report can place the block under the right table; `out_dir`, where the artefacts are
    written; `seeds`, the fixed perturbation seeds - `PRICE_PERTURBATION_SEEDS`, reused unchanged
    across every headline table so the same three price vectors are applied everywhere.

    Returns: a mapping with at least

        "table_name"     str, the table checked
        "label"          str, one of `RECOMPUTATION_LABELS`
        "base_prices"    tuple[float, ...], the unperturbed `initial_prices(cfg)` vector
        "price_vectors"  tuple[tuple[float, ...], ...], the `N_PRICE_VECTORS` perturbed vectors,
                         each with the seed that produced it
        "metrics"        tuple[dict[str, object], ...], one row per (row, price vector, metric) with
                         the baseline value, the perturbed value and the relative change
        "sign_changed"   dict[str, bool], per table row: whether `SIGN_CHANGE_METRIC` changed sign
                         under any of the three vectors
        "artefacts"      dict[str, str], the paths written

    Procedure (PLAN sections 2.9.4, 7.5):

      1. Solve the unperturbed plan prices with `initial_prices(cfg)` (PLAN section 2.10).
      2. For each seed in `seeds`, draw `u ~ N(0, PRICE_LOG_SIGMA**2)` per good with a local
         generator seeded by that integer, and form `p_j * exp(u_j)`. The draws are analysis-time
         and never touch the environment's keyed RNG (see the module docstring).
      3. Decide the label from `RERUN_TRIGGER_FIELD`. Under `recomputation`, recompute
         `val_measured`, `val_true` and the three `HEADLINE_METRICS` directly from the existing
         ledgers at the perturbed prices - no retraining, no new episodes. Under `rerun`, rebuild
         the configuration with the perturbed prices and re-run the affected arm at the same seeds
         and the same budget, because the perturbation changes the reward through the fulfilment
         measure of PLAN section 2.9.2.
      4. Record, per row and metric, the baseline value, the perturbed value, the relative change,
         and for `SIGN_CHANGE_METRIC` an explicit `sign_changed` flag.
      5. Write `table.parquet` and `report.md`, sign-change column first, with the label stated on
         every row.

    Nothing is suppressed and nothing is re-drawn: the seeds are fixed, a sign change is reported
    with the row it belongs to, and a table whose conclusion does not survive a plausible reprice is
    reported as such (PLAN section 7.5).

    Binds: gate G4 of PLAN section 13 - "price sensitivity on every headline table" - and the
    standing robustness check of PLAN section 2.9.4.

    Realises: PLAN sections 2.9.4, 2.10, 7.5, 12.5 (WO-036), 13. Owning WO: **WO-036**.
    """
    raise NotImplementedError("PLAN section 7.5 (WO-036) - implemented in WO-036")


def main() -> int:
    """Entry point: run the check over every headline table and write the sensitivity report.

    Takes: nothing; the headline tables and the ledgers behind them are discovered from the run
    directories of the experiments that produced them - `runs/phase1_gate/`, `runs/contrasts/`,
    `runs/estimator_bias/`, `runs/phase2_acceptance/` - and the perturbation is fixed by
    `N_PRICE_VECTORS`, `PRICE_LOG_SIGMA` and `PRICE_PERTURBATION_SEEDS`. Any command-line surface is
    built inside this function.

    Returns: a process exit code - 0 when every headline table found was checked and
    `runs/price_sensitivity/report.md` was written, 1 when a table could not be checked, which
    includes finding a headline table whose ledgers are missing: an unchecked headline table is a
    gate G4 failure (PLAN section 13), not a silent omission. A sign change in `SIGN_CHANGE_METRIC`
    is *not* an error condition - it is a reported result, and the exit code stays 0.

    Realises: PLAN sections 2.9.4, 7.5, 12.5 (WO-036), 13. Owning WO: **WO-036**.
    """
    raise NotImplementedError("PLAN section 7.5 (WO-036) - implemented in WO-036")


if __name__ == "__main__":
    raise SystemExit(main())
