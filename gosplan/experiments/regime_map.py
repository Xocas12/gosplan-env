"""DP regime map - PLAN sections 5, 12.3 (WO-015 card), 13 (gate G1) and 14.

Realises: the regime-map bullet of PLAN section 5 and the WO-015 card of PLAN section 12.3. Owning
work order: **WO-015** (MID-fast, difficulty 2; depends on WO-014, the single-enterprise DP). Gate:
**G1** - the human reads this map, picks the Phase-1 values of the daggered PLAN section 3 rows from
the *interior* of the bunching region, and records them, the three `a * pen` levels for gate G2 and
the `b_hat_DP` thresholds in `runs/G1_decision.md` **before any training run**.

Inputs
    A base `EnvConfig` (WO-015 runs `p1_default_config()`); `solve_single_enterprise` and
    `classify_regime` from `gosplan.agents.dp` (WO-014), on the `DPGrid` defaults of PLAN section 5
    - 80 log-spaced target points, 50 stock points, effort step 0.05, `rho` in [0, 3] step 0.02,
    9 Gauss-Hermite nodes, value tolerance 1e-6, and a stationary distribution from 200 episodes x
    200 periods.

Outputs (PLAN section 12.3, WO-015)
    `runs/regime_map/table.parquet`   one row per sampled point: the eight swept parameters, the
                                      `RegimeLabel` from `classify_regime`, `b_hat_dp`,
                                      `fictitious_padding`, `mean_effort`, `rho_edge_frac`,
                                      `n_iterations`, `converged`, and the `EnvConfig.hash()` of the
                                      configuration solved
    `runs/regime_map/regime.png`      heat map of the regime label over the swept space
    `runs/regime_map/bhat.png`        heat map of `b_hat_dp` over the same space
    `runs/regime_map/candidates.md`   the candidate list the WO-015 card requires: at least
                                      `MIN_BUNCHING_CANDIDATES` interior bunching-region points with
                                      their `b_hat_DP`. The three parquet/png names are PLAN's
                                      verbatim; this fourth filename is a WO-015 convention, and its
                                      content is what the human copies into `runs/G1_decision.md`.

Cost (PLAN section 14): 500 configurations at about 1 CPU-minute each - about 1 hour on 8 cores.

FORBIDDEN
    No reinforcement learning of any kind (the WO-014/WO-015 line of PLAN section 12.3): this map is
    analytical, and it is produced *before* any MARL run so that the Phase-1 values cannot be chosen
    to suit a training result.

    `DPSolution.hidden_reserves` must NOT be written to `table.parquet` and must NOT be plotted.
    Hidden reserves are PLAN section 4.1 row 7, held out until the Phase-2 acceptance run: the DP
    computes the field, this experiment simply never tabulates or plots it. The same prohibition
    covers rows 2, 5 and 6, none of which the single-enterprise DP can produce anyway.

OPEN - AMBIGUITIES FOR THE WO-015 SESSION (CONTRACT rule 3; do not silently choose)
    1. Factorisation of `a * pen`. PLAN section 5 sweeps the compound `a * pen`, while `EnvConfig`
       carries `information.audit_rate` and `incentive.penalty_scale` separately, and `audit_rate`
       is dual-classified (PLAN section 4.3). Two conventions are consistent with the text: hold
       `audit_rate` at the registry value and set `penalty_scale = a_pen / audit_rate`, or sample
       both marginals inside their PLAN section 3 ranges and record the realised product. The card
       does not decide. File an AMBIGUITY REPORT; whichever the lead fixes is recorded in the run
       manifest and printed in the table header.
    2. "Interior" of the bunching region. PLAN section 5 asks the human to pick from the interior
       but does not define a neighbourhood. Candidate definitions: every point within a fixed
       normalised parameter distance is also labelled `bunching`; or the k nearest neighbours in the
       Latin-hypercube design are. File an AMBIGUITY REPORT rather than choosing; the definition
       used is stated in `candidates.md` next to every candidate.

Runtime bindings. `EnvConfig` is `gosplan.config.EnvConfig` (field-for-field identical to
`spec/spec.py`, enforced by a unit test; `spec/` is not an importable package). `matplotlib` (the
`viz` extra) and the parquet writer (`pyarrow`) are imported inside the functions that draw and
write, never at module scope.
"""

from __future__ import annotations

from pathlib import Path

from gosplan.config import EnvConfig

N_LHS_POINTS = 500
"""Points in the Latin-hypercube design of PLAN section 5, and the "500 configs" of the PLAN section
14 cost line."""

LHS_RANGES: dict[str, tuple[float, float]] = {
    "notch_height": (0.25, 2.0),
    "overfulfilment_slope": (0.0, 2.0),
    "audit_times_penalty": (0.05, 60.0),
    "ratchet_lambda": (0.0, 1.0),
    "tenure": (0.7, 0.98),
    "growth_directive": (0.0, 0.07),
    "effort_cost": (0.05, 0.5),
}
"""The seven continuous dimensions of the PLAN section 5 sweep `(beta, s, a*pen, lambda, psi, g,
kappa)`, with the PLAN section 3 registry ranges verbatim: `notch_height` beta [0.25, 2];
`overfulfilment_slope` s [0, 2]; `ratchet_lambda` lambda [0, 1]; `tenure` psi [0.7, 0.98];
`growth_directive` g [0, 0.07]; `effort_cost` kappa [0.05, 0.5]. `audit_times_penalty` is the
compound `a * pen`, whose endpoints are the products of the registry ranges of `audit_rate`
[0.01, 0.30] and `penalty_scale` [5, 200]; how a sampled product is factorised into the two fields
is ambiguity 1 in the module docstring."""

LHS_LEVELS: dict[str, tuple[float, ...]] = {
    "notch_width": (0.0, 0.25),
}
"""The eighth, discrete dimension: `w` in {0, 0.25} (PLAN section 5). `w = 0` is the notched Phase-1
schedule, a true discontinuity; `w = 0.25` is the smooth counterfactual knob of PLAN section 2.8.
Sampling is stratified so both levels carry the same number of design points."""

MIN_BUNCHING_CANDIDATES = 5
"""Minimum number of interior bunching-region points the WO-015 card requires in `candidates.md`,
each reported with its `b_hat_DP`. Fewer than this is a reported failure of the map - the human then
has nothing to pick from at G1 - and never a reason to relax `classify_regime`."""

OUT_DIR = Path("runs/regime_map")
"""Artefact directory, relative to the repository root (PLAN section 12.3)."""

TABLE_PATH = OUT_DIR / "table.parquet"
"""One row per design point; the PLAN section 12.3 artefact."""

REGIME_FIG_PATH = OUT_DIR / "regime.png"
"""Heat map of the regime label; the PLAN section 12.3 artefact."""

BHAT_FIG_PATH = OUT_DIR / "bhat.png"
"""Heat map of `b_hat_dp`; the PLAN section 12.3 artefact."""

CANDIDATES_PATH = OUT_DIR / "candidates.md"
"""The candidate list of at least `MIN_BUNCHING_CANDIDATES` interior bunching-region points. The
filename is a WO-015 convention; the requirement is PLAN section 12.3's, and the content is what the
human transcribes into `runs/G1_decision.md` at gate G1."""

TABLE_COLUMNS: tuple[str, ...] = (
    "point_index",
    "notch_height",
    "overfulfilment_slope",
    "audit_times_penalty",
    "audit_rate",
    "penalty_scale",
    "ratchet_lambda",
    "tenure",
    "growth_directive",
    "effort_cost",
    "notch_width",
    "regime",
    "b_hat_dp",
    "fictitious_padding",
    "mean_effort",
    "rho_edge_frac",
    "n_iterations",
    "converged",
    "config_hash",
)
"""The exact columns of `table.parquet`, declared as data so the prohibition is checkable: this is
the whole table, and `hidden_reserves` (PLAN section 4.1 row 7) is deliberately absent. `audit_rate`
and `penalty_scale` are both recorded alongside their product so the factorisation convention of
ambiguity 1 is legible from the file itself."""


def run(
    base_cfg: EnvConfig,
    out_dir: Path = OUT_DIR,
    n_points: int = N_LHS_POINTS,
    seed: int | None = None,
) -> dict[str, object]:
    """Solve the DP over the Latin-hypercube design and write the regime map.

    Takes: `base_cfg`, the configuration every design point starts from (WO-015 runs
    `p1_default_config()`), already validated; `out_dir`, where the four artefacts are written;
    `n_points`, design points to sample; `seed`, the seed of the Latin-hypercube generator - `None`
    means `base_cfg.tech.seed_env`. The seed is recorded in the table header and in every run
    manifest, so the design is reproducible point for point.

    Returns: a mapping with at least

        "n_points"            int, design points solved
        "n_converged"         int, points whose value iteration reached `DPGrid.value_tol`
        "regime_counts"       dict[str, int], points per `RegimeLabel`
        "bhat_range"          tuple[float, float], min and max `b_hat_dp` over the design
        "edge_hit_points"     int, points with `rho_edge_frac` above the `classify_regime`
                              threshold, i.e. the report grid was hit and the grid was extended
        "candidates"          tuple[dict[str, object], ...], at least `MIN_BUNCHING_CANDIDATES`
                              interior bunching-region points, each with its parameters and
                              `b_hat_dp`
        "artefacts"           dict[str, str], the four paths written
        "ap_factorisation"    str, the convention used for ambiguity 1, as fixed by the lead

    Procedure (PLAN section 5, regime-map bullet; WO-015 card):

      1. Draw `n_points` Latin-hypercube points over the seven continuous dimensions of
         `LHS_RANGES`, stratified over the two levels of `LHS_LEVELS["notch_width"]`.
      2. Turn each point into an `EnvConfig` by overriding the corresponding fields of `base_cfg`
         (`incentive.notch_height`, `overfulfilment_slope`, `ratchet_lambda`, `tenure`,
         `growth_directive`, `effort_cost`, `notch_width`, plus `information.audit_rate` and
         `incentive.penalty_scale` under the factorisation convention of ambiguity 1), and validate
         it.
      3. Solve `solve_single_enterprise(cfg, DPGrid())` - PLAN section 5's grid, unchanged - and
         label it with `classify_regime`. The DP calls the environment's own `bonus` and penalty
         functions, so the map and the environment can never drift.
      4. Write `table.parquet` with exactly `TABLE_COLUMNS`, then `regime.png` (categorical colours
         over the regime label) and `bhat.png` (continuous scale over `b_hat_dp`). Grid-edge hits
         are a regime signal, not an artefact: they are plotted and reported, never smoothed away.
      5. Write `candidates.md` with at least `MIN_BUNCHING_CANDIDATES` points labelled `bunching`
         that are interior under the definition the lead fixed (ambiguity 2), each with all eight
         parameters, `b_hat_DP`, the regime labels of its neighbours, and its `config_hash`.

    Nothing in this function trains anything (no RL, WO-014/WO-015 forbidden list) and nothing
    writes or plots `DPSolution.hidden_reserves` (PLAN section 4.1 row 7, held out).

    Binds: the WO-015 smoke test (a small `n_points` produces the four artefacts and the mapping
    above). Gate: this map is the G1 artefact of PLAN section 13; the human's selection is recorded
    in `runs/G1_decision.md` before any training run.

    Realises: PLAN sections 5, 12.3 (WO-015), 13, 14. Owning WO: **WO-015**.
    """
    raise NotImplementedError("PLAN section 5 (WO-015) - implemented in WO-015")


def main() -> int:
    """Entry point: build the regime map at `p1_default_config()` and write the four artefacts.

    Takes: nothing; the WO-015 card fixes the base configuration (`p1_default_config()`), the design
    size (`N_LHS_POINTS`) and the DP grid (the PLAN section 5 defaults of `DPGrid`). Any
    command-line surface, and any process pool used to reach the PLAN section 14 figure of about an
    hour on 8 cores, is built inside this function.

    Returns: a process exit code - 0 when the design was solved and all four artefacts were written
    with at least `MIN_BUNCHING_CANDIDATES` interior candidates, 1 otherwise. The exit code never
    means "gate G1 passed": G1 is a human decision recorded in `runs/G1_decision.md`, taken after
    reading this map and before any training run (PLAN section 13).

    Realises: PLAN sections 5, 12.3 (WO-015), 13. Owning WO: **WO-015**.
    """
    raise NotImplementedError("PLAN section 5 (WO-015) - implemented in WO-015")


if __name__ == "__main__":
    raise SystemExit(main())
