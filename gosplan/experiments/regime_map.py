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

import dataclasses
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import get_args

import numpy as np

from gosplan.agents.dp import DPGrid, RegimeLabel, solve_single_enterprise
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


_MAX_WORKERS = 3
"""Process-pool size for the design solves: this machine has 4 cores and one is kept free. Private
because neither `run` nor `main` documents a worker-count parameter."""

_AP_FACTORISATION = "audit_rate held at base; penalty_scale = a_pen / audit_rate"
"""Ambiguity 1 as fixed by the lead (AMBIGUITY-012, resolution 1), verbatim."""

_INTERIOR_K = 5
"""Ambiguity 2 as fixed by the lead (AMBIGUITY-012, resolution 2): the neighbourhood size."""

_INTERIOR_DEFINITION = (
    "A `bunching` point is interior when its k = 5 nearest neighbours in the design - Euclidean "
    "distance over the seven continuous dimensions, each normalised to [0, 1] by its `LHS_RANGES` "
    "span, among points with the SAME `notch_width` level - are all labelled `bunching` "
    "(lead ruling AMBIGUITY-012, resolution 2)."
)
"""The "interior" definition printed beside every candidate in `candidates.md`."""


def _design(n_points: int, seed: int) -> list[dict[str, float]]:
    """The Latin-hypercube design: one LHS over `LHS_RANGES` per `notch_width` level, so both
    levels carry the same number of points; generated from `seed`, so it is reproducible."""
    from scipy.stats import qmc

    levels = LHS_LEVELS["notch_width"]
    if n_points <= 0 or n_points % len(levels):
        raise ValueError(
            f"n_points={n_points} cannot be stratified equally over notch_width levels {levels}"
        )
    per_level = n_points // len(levels)
    names = tuple(LHS_RANGES)
    lo = np.array([LHS_RANGES[k][0] for k in names])
    hi = np.array([LHS_RANGES[k][1] for k in names])
    rng = np.random.default_rng(seed)
    points: list[dict[str, float]] = []
    for w in levels:
        unit = qmc.LatinHypercube(d=len(names), rng=rng).random(per_level)
        for row in qmc.scale(unit, lo, hi):
            point = {k: float(v) for k, v in zip(names, row, strict=True)}
            point["notch_width"] = float(w)
            points.append(point)
    return points


def _point_config(base_cfg: EnvConfig, point: dict[str, float]) -> EnvConfig:
    """Override `base_cfg` with one design point and validate it. `a * pen` is factorised as the
    lead fixed it: `audit_rate` held at the base value, `penalty_scale = a_pen / audit_rate`."""
    audit_rate = float(base_cfg.information.audit_rate)
    incentive = dataclasses.replace(
        base_cfg.incentive,
        notch_height=point["notch_height"],
        overfulfilment_slope=point["overfulfilment_slope"],
        ratchet_lambda=point["ratchet_lambda"],
        tenure=point["tenure"],
        growth_directive=point["growth_directive"],
        effort_cost=point["effort_cost"],
        notch_width=point["notch_width"],
        penalty_scale=point["audit_times_penalty"] / audit_rate,
    )
    information = dataclasses.replace(base_cfg.information, audit_rate=audit_rate)
    cfg = dataclasses.replace(base_cfg, incentive=incentive, information=information)
    cfg.validate()
    return cfg


def _solve_point(job: tuple[int, dict[str, float], EnvConfig]) -> dict[str, object]:
    """Solve one design point on PLAN section 5's grid and return its table row. Only the
    `TABLE_COLUMNS` fields are read off the solution; `hidden_reserves` is never touched."""
    index, point, cfg = job
    sol = solve_single_enterprise(cfg, DPGrid())
    return {
        "point_index": index,
        "notch_height": point["notch_height"],
        "overfulfilment_slope": point["overfulfilment_slope"],
        "audit_times_penalty": point["audit_times_penalty"],
        "audit_rate": float(cfg.information.audit_rate),
        "penalty_scale": float(cfg.incentive.penalty_scale),
        "ratchet_lambda": point["ratchet_lambda"],
        "tenure": point["tenure"],
        "growth_directive": point["growth_directive"],
        "effort_cost": point["effort_cost"],
        "notch_width": point["notch_width"],
        "regime": str(sol.regime),
        "b_hat_dp": float(sol.b_hat_dp),
        "fictitious_padding": float(sol.fictitious_padding),
        "mean_effort": float(sol.mean_effort),
        "rho_edge_frac": float(sol.rho_edge_frac),
        "n_iterations": int(sol.n_iterations),
        "converged": bool(sol.converged),
        "config_hash": str(sol.config_hash),
        "_rho_hi": float(sol.grid.rho_hi),
    }


def _write_table(rows: list[dict[str, object]], path: Path, seed: int) -> None:
    """`table.parquet` with exactly `TABLE_COLUMNS`; the LHS seed and the factorisation convention
    go in the parquet header (schema metadata)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({col: [row[col] for row in rows] for col in TABLE_COLUMNS})
    table = table.replace_schema_metadata(
        {"lhs_seed": str(seed), "ap_factorisation": _AP_FACTORISATION}
    )
    pq.write_table(table, path)


def _interior_candidates(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Interior `bunching` points under the lead's definition (AMBIGUITY-012, resolution 2),
    deepest first: decreasing smallest normalised distance to a non-`bunching` point of the same
    `notch_width` level."""
    names = tuple(LHS_RANGES)
    lo = np.array([LHS_RANGES[k][0] for k in names])
    span = np.array([LHS_RANGES[k][1] - LHS_RANGES[k][0] for k in names])
    found: list[tuple[float, dict[str, object]]] = []
    for w in LHS_LEVELS["notch_width"]:
        level = [r for r in rows if r["notch_width"] == float(w)]
        if len(level) <= _INTERIOR_K:
            continue
        x = (np.array([[float(r[k]) for k in names] for r in level]) - lo) / span
        dist = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=-1))
        bunching = np.array([r["regime"] == "bunching" for r in level])
        for i, row in enumerate(level):
            if not bunching[i]:
                continue
            others = np.delete(np.arange(len(level)), i)
            nearest = others[np.argsort(dist[i, others], kind="stable")[:_INTERIOR_K]]
            if not bunching[nearest].all():
                continue
            depth = float(dist[i, ~bunching].min()) if (~bunching).any() else math.inf
            cand = {k: row[k] for k in TABLE_COLUMNS if k != "regime"}
            cand["neighbour_point_indices"] = tuple(int(level[j]["point_index"]) for j in nearest)
            cand["neighbour_regimes"] = tuple(str(level[j]["regime"]) for j in nearest)
            cand["depth"] = depth
            found.append((depth, cand))
    found.sort(key=lambda item: -item[0])
    return [cand for _, cand in found]


def _write_candidates(candidates: list[dict[str, object]], path: Path, seed: int) -> None:
    """`candidates.md`: each interior point with all eight parameters, its `b_hat_DP` (a
    non-finite value is printed as such, AMBIGUITY-011 addendum K), its neighbours' labels and its
    `config_hash`, with the interior definition stated beside every candidate."""
    params = (*LHS_RANGES, "notch_width")
    lines = [
        "# DP regime map - interior bunching-region candidates (WO-015, gate G1)",
        "",
        f"- LHS seed: {seed}",
        f"- `a * pen` factorisation: {_AP_FACTORISATION}",
        f"- interior candidates: {len(candidates)} (required: {MIN_BUNCHING_CANDIDATES})",
        "- order: deepest first (decreasing smallest normalised distance to a non-`bunching`"
        " point of the same `notch_width` level)",
        "",
    ]
    if len(candidates) < MIN_BUNCHING_CANDIDATES:
        lines += [
            f"**SHORTFALL: fewer than {MIN_BUNCHING_CANDIDATES} interior candidates. This is a"
            " reported failure of the map, not a reason to relax `classify_regime`.**",
            "",
        ]
    for rank, cand in enumerate(candidates, start=1):
        lines += [f"## Candidate {rank}: design point {cand['point_index']}", ""]
        lines += [f"- `{name}` = {float(cand[name])!r}" for name in params]
        lines += [
            f"- `audit_rate` = {float(cand['audit_rate'])!r}, `penalty_scale` ="
            f" {float(cand['penalty_scale'])!r}",
            f"- `b_hat_DP` = {float(cand['b_hat_dp'])!r}",
            "- neighbours (point index: regime): "
            + ", ".join(
                f"{i}: {lab}"
                for i, lab in zip(
                    cand["neighbour_point_indices"], cand["neighbour_regimes"], strict=True
                )
            ),
            f"- depth (normalised distance to nearest non-`bunching` point): {cand['depth']!r}",
            f"- `config_hash` = `{cand['config_hash']}`",
            f"- interior definition: {_INTERIOR_DEFINITION}",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _edge_hit_points(rows: list[dict[str, object]]) -> int:
    """Points whose report grid was hit: the grid was extended past `DPGrid().rho_hi`, or
    stationary mass remains at the top grid point. LEAD ruling AMBIGUITY-013 A - the most
    inclusive reading, since grid-edge hits are a result to report (CONTRACT rule 8)."""
    base_hi = DPGrid().rho_hi
    return sum(
        1 for row in rows if float(row["rho_edge_frac"]) > 0.0 or float(row["_rho_hi"]) > base_hi
    )


_CONTINUOUS_DIMS: tuple[str, ...] = tuple(LHS_RANGES)
"""The seven continuous design dimensions, in `LHS_RANGES` order (the plotted axes)."""


def _plot_maps(rows: list[dict[str, object]], regime_path: Path, bhat_path: Path) -> None:
    """`regime.png` and `bhat.png` (LEAD ruling AMBIGUITY-013 B).

    Each figure is a pairwise scatter matrix of the seven continuous dimensions (lower triangle),
    one block per `notch_width` level, every design point drawn at its own coordinates - no binning,
    so nothing is averaged away. `regime.png` colours points by regime label (categorical);
    `bhat.png` colours finite `b_hat_dp` on a continuous scale and draws non-finite values as black
    crosses with their own legend entry, never clipped. Points whose report grid was hit are
    outlined in red in both figures (CONTRACT rule 8)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels = sorted({float(row["notch_width"]) for row in rows})
    labels = list(get_args(RegimeLabel))
    palette = dict(zip(labels, ("#1b9e77", "#d95f02", "#7570b3", "#999999"), strict=True))
    base_hi = DPGrid().rho_hi
    dims = _CONTINUOUS_DIMS
    k = len(dims)

    def draw(path: Path, kind: str) -> None:
        fig, axes = plt.subplots(
            k - 1, (k - 1) * len(levels), figsize=(3.0 * (k - 1) * len(levels), 3.0 * (k - 1))
        )
        finite = [float(r["b_hat_dp"]) for r in rows if np.isfinite(float(r["b_hat_dp"]))]
        vmin, vmax = (min(finite), max(finite)) if finite else (0.0, 1.0)
        for li, level in enumerate(levels):
            sub = [r for r in rows if float(r["notch_width"]) == level]
            edge = np.array(
                [float(r["rho_edge_frac"]) > 0.0 or float(r["_rho_hi"]) > base_hi for r in sub],
                dtype=bool,
            )
            for yi in range(1, k):
                for xi in range(k - 1):
                    ax = axes[yi - 1, li * (k - 1) + xi]
                    if xi >= yi:
                        ax.axis("off")
                        continue
                    x = np.array([float(r[dims[xi]]) for r in sub])
                    y = np.array([float(r[dims[yi]]) for r in sub])
                    if kind == "regime":
                        colours = [palette[str(r["regime"])] for r in sub]
                        ax.scatter(x, y, c=colours, s=10)
                    else:
                        b = np.array([float(r["b_hat_dp"]) for r in sub])
                        ok = np.isfinite(b)
                        ax.scatter(x[ok], y[ok], c=b[ok], s=10, vmin=vmin, vmax=vmax)
                        ax.scatter(x[~ok], y[~ok], marker="x", c="black", s=12)
                    ax.scatter(
                        x[edge], y[edge], s=24, facecolors="none", edgecolors="red", linewidths=0.8
                    )
                    if yi == k - 1:
                        ax.set_xlabel(dims[xi], fontsize=7)
                    if xi == 0:
                        ax.set_ylabel(dims[yi], fontsize=7)
                    ax.tick_params(labelsize=6)
            axes[0, li * (k - 1)].set_title(f"notch_width = {level}", fontsize=9, loc="left")
        handles = []
        if kind == "regime":
            for label in labels:
                handles.append(plt.Line2D([], [], ls="", marker="o", color=palette[label]))
            names = list(labels)
        else:
            handles.append(plt.Line2D([], [], ls="", marker="x", color="black"))
            names = ["non-finite b_hat_DP (AMBIGUITY-011 K)"]
            sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin, vmax))
            fig.colorbar(sm, ax=axes, shrink=0.5, label="b_hat_DP")
        handles.append(plt.Line2D([], [], ls="", marker="o", mfc="none", mec="red"))
        names.append("report grid hit / extended")
        fig.legend(handles, names, loc="upper right", fontsize=9)
        fig.savefig(path, dpi=110)
        plt.close(fig)

    draw(regime_path, "regime")
    draw(bhat_path, "bhat")


def _git_hash() -> str | None:
    """`git rev-parse HEAD` plus a `-dirty` marker, or `None` outside a work tree."""
    import subprocess

    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return head + ("-dirty" if dirty else "")


def _write_manifests(
    rows: list[dict[str, object]], base_cfg: EnvConfig, out_dir: Path, seed: int
) -> None:
    """One CONTRACT rule 10 manifest for the whole map, `out_dir/manifest.json` (LEAD ruling
    AMBIGUITY-013 C). It is written for the BASE configuration; each point's own `config_hash` is
    in `table.parquet`. The design seed, the factorisation convention and the design size travel in
    `flags` (the only free-form manifest field); the solver fields name the PLAN section 5 DP and
    the `DPGrid` it ran on."""
    from gosplan.metrics.ledger import write_manifest

    write_manifest(
        str(out_dir),
        base_cfg,
        {
            "git_hash": _git_hash(),
            "solver": "single-enterprise DP, Howard policy iteration (PLAN section 5, WO-014)",
            "solver_version": repr(DPGrid()),
            "solver_optimality_gap": None,
            "flags": [
                f"lhs_seed={seed}",
                f"ap_factorisation={_AP_FACTORISATION}",
                f"n_points={len(rows)}",
                f"n_converged={sum(bool(r['converged']) for r in rows)}",
            ],
        },
    )


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
    seed = int(base_cfg.tech.seed_env) if seed is None else int(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    points = _design(n_points, seed)
    cfgs = [_point_config(base_cfg, point) for point in points]
    jobs = [(i, point, cfg) for i, (point, cfg) in enumerate(zip(points, cfgs, strict=True))]
    with ProcessPoolExecutor(max_workers=max(1, min(_MAX_WORKERS, len(jobs)))) as pool:
        rows = list(pool.map(_solve_point, jobs))

    paths = {
        "table": out_dir / TABLE_PATH.name,
        "regime": out_dir / REGIME_FIG_PATH.name,
        "bhat": out_dir / BHAT_FIG_PATH.name,
        "candidates": out_dir / CANDIDATES_PATH.name,
    }
    _write_table(rows, paths["table"], seed)
    _plot_maps(rows, paths["regime"], paths["bhat"])
    candidates = _interior_candidates(rows)
    _write_candidates(candidates, paths["candidates"], seed)
    _write_manifests(rows, base_cfg, out_dir, seed)

    bhat = np.array([float(row["b_hat_dp"]) for row in rows])
    counts = {label: 0 for label in get_args(RegimeLabel)}
    for row in rows:
        counts[str(row["regime"])] += 1
    return {
        "n_points": len(rows),
        "n_converged": sum(bool(row["converged"]) for row in rows),
        "regime_counts": counts,
        # Plain min / max: a non-finite `b_hat_dp` propagates and is shown as such (AMBIGUITY-011).
        "bhat_range": (float(np.min(bhat)), float(np.max(bhat))),
        "edge_hit_points": _edge_hit_points(rows),
        "candidates": tuple(candidates),
        "artefacts": {key: str(path) for key, path in paths.items()},
        "ap_factorisation": _AP_FACTORISATION,
    }


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
    from gosplan.config import p1_default_config

    out = run(p1_default_config())
    for key in ("n_points", "n_converged", "regime_counts", "bhat_range", "edge_hit_points"):
        print(f"{key}: {out[key]}")
    print(f"candidates: {len(out['candidates'])} (required: {MIN_BUNCHING_CANDIDATES})")
    for key, path in dict(out["artefacts"]).items():
        print(f"{key}: {path}")
    written = all(Path(path).exists() for path in dict(out["artefacts"]).values())
    solved = out["n_points"] == N_LHS_POINTS
    return 0 if solved and written and len(out["candidates"]) >= MIN_BUNCHING_CANDIDATES else 1


if __name__ == "__main__":
    raise SystemExit(main())
