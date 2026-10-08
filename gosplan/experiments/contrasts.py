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

import json
import os
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
    n_seeds: int = 15,
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
    from gosplan.experiments import _g2
    from gosplan.experiments.phase1_gate import GATE_SIZING, study_ppo_config
    from gosplan.oracle.kantorovich import solve_oracle

    out_dir = Path(out_dir)
    contrasts = CONTRASTS if contrasts is None else contrasts
    root = SEED_ROOT if seed_env is None else int(seed_env)
    sizing = GATE_SIZING
    ppo = study_ppo_config()
    arm_cfgs = {name: apply_overrides(cfg, table) for name, table in contrasts.items()}
    seeds = {name: (C0_SEEDS if name == "C0" else n_seeds) for name in contrasts}
    jobs = [(name, s) for name in contrasts for s in range(seeds[name])]

    def run_root(name: str) -> Path:
        return C0_RUN_ROOT if name == "C0" else out_dir / "runs"

    job_cfgs = [_g2.seeded(arm_cfgs[name], s, root) for name, s in jobs]
    summaries = _g2.run_many(
        [(c, sizing, run_root(name), ppo) for (name, _s), c in zip(jobs, job_cfgs, strict=True)]
    )
    measured = _pool(
        measure_contrast_seed,
        [(c, run_root(name), sizing, ppo) for (name, _s), c in zip(jobs, job_cfgs, strict=True)],
    )
    oracle = {
        name: solve_oracle(_g2.seeded(arm_cfgs[name], 0, root), ORACLE_HORIZON, False, None)
        for name in contrasts
    }

    per_seed: list[dict[str, object]] = []
    by_arm: dict[str, dict[int, dict[str, float]]] = {name: {} for name in contrasts}
    for (name, s), summ, meas in zip(jobs, summaries, measured, strict=True):
        o = oracle[name]
        outcomes = seed_outcomes(meas, o["welfare"], o["val"])
        by_arm[name][s] = outcomes
        for key, value in outcomes.items():
            per_seed.append(
                {
                    "contrast": name,
                    "seed_index": s,
                    "seed_env": int(root + s),
                    "config_hash": summ["config_hash"],
                    "outcome": key,
                    "value": value,
                }
            )

    table = aggregate(by_arm)
    flags = sorted({f for summ in summaries for f in summ["manifest_flags"]})
    exploit = _exploitability(out_dir, arm_cfgs, root, contrasts)
    result = {
        **table,
        "per_seed": tuple(per_seed),
        "dual_classified": DUAL_CLASSIFIED,
        "flags": tuple(flags),
        "oracle": {
            name: {
                k: o[k]
                for k in ("welfare", "val", "optimality_gap", "solver", "solver_version", "status")
            }
            for name, o in oracle.items()
        },
        "exploitability": exploit,
        "seeds": seeds,
        "audit_rate_out_of_range": _audit_out_of_range(arm_cfgs),
        "git": _g2.git_hash(),
    }
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_seed).to_parquet(out_dir / TABLE_PATH.name, index=False)
    (out_dir / "result.json").write_text(
        json.dumps(result, default=str, indent=1), encoding="utf-8"
    )
    (out_dir / REPORT_PATH.name).write_text(render_report(result), encoding="utf-8")
    result["artefacts"] = {
        "report": str(out_dir / REPORT_PATH.name),
        "table": str(out_dir / TABLE_PATH.name),
    }
    return result


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
    from gosplan.config import p2_default_config

    res = run(p2_default_config())
    for name, d in res["deltas"].items():
        print(f"{name}: delta {d['delta']:.4f} [{d['ci_lo']:.4f}, {d['ci_hi']:.4f}]")
    return 0


# ---------- implementation (spec/P3_REVISION.md S2) ----------

SEED_ROOT = 1000
"""`seed_env = SEED_ROOT + s`, the G3 root, so C0 reuses the G3 runs (P3 revision S2)."""

C0_SEEDS = 30
"""C0 is the 30 G3 runs, read back from `C0_RUN_ROOT` rather than re-trained."""

C0_RUN_ROOT = Path("runs/phase2_acceptance/runs")

N_NEW_ARM_SEEDS = 15
"""Seeds per non-C0 arm (P3 revision S2: reduced from PLAN's 30 for compute)."""

EXPLOIT_SEEDS = 3
ORACLE_HORIZON = 40
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
PRICE_SEEDS = (11, 12, 13)
PRICE_READING_METRICS = ("net_output",)


def apply_overrides(cfg: EnvConfig, table: ContrastOverrides) -> EnvConfig:
    """`cfg` with `table` applied; a `Multiplier` scales the C0 value. Validated."""
    import dataclasses

    sections = {}
    for section, fields in table.items():
        base = getattr(cfg, section)
        resolved = {
            name: (getattr(base, name) * v.factor if isinstance(v, Multiplier) else v)
            for name, v in fields.items()
        }
        sections[section] = dataclasses.replace(base, **resolved)
    out = dataclasses.replace(cfg, **sections)
    out.validate()
    return out


def _audit_out_of_range(arm_cfgs: dict[str, EnvConfig]) -> dict[str, float]:
    """Arms whose `audit_rate` leaves the PLAN section 3 range [0.01, 0.30] (reported, not
    clamped; WO-032)."""
    return {
        name: c.information.audit_rate
        for name, c in arm_cfgs.items()
        if not 0.01 <= c.information.audit_rate <= 0.30
    }


def measure_contrast_seed(job: tuple) -> dict[str, object]:
    """Window means of `val_measured`, `val_true` and welfare for one trained run, at base prices
    and re-priced under `PRICE_SEEDS`. Written to `run_dir / "contrast_measure.json"`."""
    import numpy as np

    from gosplan.agents.ppo.adapter import IPPO
    from gosplan.agents.ppo.train import checkpoint_path, evaluate
    from gosplan.env.prices import initial_prices, perturbed_price_vectors
    from gosplan.experiments._g2 import MEASURE_SEED_OFFSET
    from gosplan.metrics.phenomena import MEASUREMENT_FIRST_PERIOD

    cfg, run_root, sizing, ppo_cfg = job
    run_dir = Path(run_root) / cfg.hash()
    out_path = run_dir / "contrast_measure.json"
    if out_path.exists():
        return json.loads(out_path.read_text(encoding="utf-8"))
    n_updates = sizing.total_agent_steps // (sizing.n_envs * sizing.rollout_steps)
    agent = IPPO(cfg, ppo_cfg)
    agent.load_checkpoint(checkpoint_path(run_dir, n_updates - 1))
    _metrics, ledger = evaluate(
        agent, cfg, sizing.measure_episodes, int(cfg.tech.seed_env) + MEASURE_SEED_OFFSET
    )
    base = np.asarray(initial_prices(cfg), dtype=float)
    vectors = (base, *perturbed_price_vectors(base, PRICE_SEEDS))
    m = cfg.incentive.steps_per_period
    mu = cfg.information.quality_measurability
    periods: dict[tuple[int, int], list] = {}
    for rec in ledger.records:
        if rec.phase == "report" and rec.t_period >= MEASUREMENT_FIRST_PERIOD:
            periods.setdefault((rec.episode, rec.t_period), []).append(rec)
    welfare, vm, vt = [], [[] for _ in vectors], [[] for _ in vectors]
    for recs in periods.values():
        welfare.append(recs[0].welfare)
        for r, p in enumerate(vectors):
            meas = true = 0.0
            for rec in recs:
                qbar = rec.quality_acc / m if cfg.supply.quality_matters else 1.0
                meas += p[rec.sector] * rec.report * (1.0 + mu * (qbar - 1.0))
                true += p[rec.sector] * rec.cum_output * qbar
            vm[r].append(meas)
            vt[r].append(true)
    out = {
        "config_hash": cfg.hash(),
        "welfare_mean": float(np.mean(welfare)),
        "val_measured_mean": [float(np.mean(v)) for v in vm],
        "val_true_mean": [float(np.mean(v)) for v in vt],
        "n_periods": len(welfare),
        "bound_binding": bool("BOUND_BINDING" in ledger.flags),
    }
    out_path.write_text(json.dumps(out, sort_keys=True), encoding="utf-8")
    return out


def seed_outcomes(
    meas: dict, w_oracle: float, val_oracle: float, price: int = 0
) -> dict[str, float]:
    """The three `OUTCOMES` for one seed at price vector `price` (0 = base prices)."""
    wr = meas["welfare_mean"] / w_oracle
    return {
        "padding_index": meas["val_measured_mean"][price] / meas["val_true_mean"][price],
        "welfare_ratio": wr,
        "specification_gap": meas["val_measured_mean"][price] / val_oracle - wr,
    }


def iqm_ci(values) -> tuple[float, float, float]:
    """IQM (`rliable.metrics.aggregate_iqm`) with a percentile bootstrap CI over seeds."""
    import numpy as np
    from rliable.metrics import aggregate_iqm
    from scipy.stats import trim_mean

    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, x.size, size=(BOOTSTRAP_RESAMPLES, x.size))
    # `aggregate_iqm` is `scipy.stats.trim_mean(x, 0.25)`; vectorised over the resamples.
    boots = trim_mean(x[idx], 0.25, axis=1)
    alpha = (1.0 - CI_LEVEL) / 2.0
    return (
        float(aggregate_iqm(x[:, None])),
        float(np.quantile(boots, alpha)),
        float(np.quantile(boots, 1.0 - alpha)),
    )


def aggregate(by_arm: dict[str, dict[int, dict[str, float]]]) -> dict[str, object]:
    """Per-arm IQM/CI for every outcome, seed-paired `Delta_X` and `I` (P3 revision S2). Pure."""
    outcomes = {
        name: {
            o: dict(
                zip(("iqm", "ci_lo", "ci_hi"), iqm_ci([v[o] for v in seeds.values()]), strict=True)
            )
            for o in OUTCOMES
        }
        for name, seeds in by_arm.items()
    }
    c0 = by_arm.get("C0", {})
    paired: dict[str, dict[int, float]] = {}
    deltas = {}
    for name, seeds in by_arm.items():
        if name == "C0":
            continue
        common = sorted(set(seeds) & set(c0))
        paired[name] = {s: seeds[s][DELTA_OUTCOME] - c0[s][DELTA_OUTCOME] for s in common}
        d, lo, hi = iqm_ci(list(paired[name].values()))
        deltas[name] = {"delta": d, "ci_lo": lo, "ci_hi": hi, "n_pairs": len(common)}
    interaction = {"formula": INTERACTION_FORMULA}
    if all(k in paired for k in ("C_BOTH", "C_OGAS", "C_INC")):
        common = sorted(set(paired["C_BOTH"]) & set(paired["C_OGAS"]) & set(paired["C_INC"]))
        vals = [paired["C_BOTH"][s] - paired["C_OGAS"][s] - paired["C_INC"][s] for s in common]
        i, lo, hi = iqm_ci(vals)
        interaction.update({"I": i, "ci_lo": lo, "ci_hi": hi, "n_pairs": len(common)})
    return {"outcomes": outcomes, "deltas": deltas, "interaction": interaction}


def _pool(fn, jobs: list) -> list:
    import multiprocessing
    import os
    from concurrent.futures import ProcessPoolExecutor

    from gosplan.experiments._g2 import MAX_WORKERS

    os.environ.setdefault(
        "XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    )
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with ProcessPoolExecutor(
        max_workers=max(1, min(MAX_WORKERS, len(jobs))),
        mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
        return list(pool.map(fn, jobs))


def _exploit_one(args: tuple) -> dict[str, object]:
    from gosplan.experiments import exploitability
    from gosplan.experiments.phase2_acceptance import br_sizing

    name, audit_dir, out_dir = args
    res = exploitability.run(
        None, name, audit_dir, Path(out_dir) / "exploitability", sizing=br_sizing()
    )
    return {k: v for k, v in res.items() if k != "per_seed"} | {"per_seed": list(res["per_seed"])}


def _exploitability(out_dir: Path, arm_cfgs, root: int, contrasts) -> dict[str, object]:
    """Audit seeds 0..EXPLOIT_SEEDS-1 of every non-C0 arm; C0's audit is G3's (S2)."""
    from gosplan.experiments import _g2

    jobs = []
    for name in contrasts:
        if name == "C0":
            continue
        audit = Path(out_dir) / "exploit_subset" / name
        audit.mkdir(parents=True, exist_ok=True)
        for s in range(EXPLOIT_SEEDS):
            target = (Path(out_dir) / "runs" / _g2.seeded(arm_cfgs[name], s, root).hash()).resolve()
            link = audit / target.name
            if target.exists() and not link.exists():
                # Relative, so the committed run tree stays valid in any checkout and in an sdist.
                link.symlink_to(
                    os.path.relpath(target, link.parent.resolve()), target_is_directory=True
                )
        jobs.append((name, audit, out_dir))
    results = _pool(_exploit_one, jobs) if jobs else []
    out = {name: r for (name, *_), r in zip(jobs, results, strict=True)}
    g3 = Path("runs/phase2_acceptance/result.json")
    if g3.exists():
        out["C0"] = json.loads(g3.read_text(encoding="utf-8"))["exploitability"].get("C0")
    return out


def _fmt(t: dict) -> str:
    return f"{t['iqm']:.4f} [{t['ci_lo']:.4f}, {t['ci_hi']:.4f}]"


def render_report(res: dict) -> str:
    """The contrast report: the two reporting rules, the per-arm table, the Deltas and `I`."""
    ex = res.get("exploitability", {})

    def label(name: str) -> str:
        e = ex.get(name) or {}
        return f" **{e['label']}**" if e.get("label") else ""

    lines = [
        "# Contrasts (WO-032, PLAN section 4.3)",
        "",
        "Design: spec/P3_REVISION.md S2. C0 = the 30 G3 runs; other arms 15 seeds each (reduced "
        "from PLAN's 30 for compute), common random numbers (`seed_env = 1000 + s`). IQM with a "
        "95% percentile bootstrap over seeds (10,000 resamples). Limitations L1, L2 "
        "(runs/G2_record.md) and the G3 record apply.",
        "",
        "Reporting rules (PLAN section 4.3):",
        "1. **C_AUDIT is dual-classified and is never folded into C_OGAS.**",
        "2. **No 'X% of the loss is informational' statement without a named contrast and a CI.**",
        "",
    ]
    if res.get("audit_rate_out_of_range"):
        lines += [
            f"Out of the PLAN section 3 sweep range (reported, not clamped): "
            f"{res['audit_rate_out_of_range']} (`audit_rate`, range [0.01, 0.30]).",
            "",
        ]
    lines += [
        "## Outcomes per arm (IQM [95% CI])",
        "",
        "| arm | seeds | welfare_ratio | padding_index | specification_gap | convergence |",
        "|---|---|---|---|---|---|",
    ]
    for name, o in res["outcomes"].items():
        e = ex.get(name) or {}
        conv = (
            f"max expl. {e['max_exploitability']:.3f}{label(name)}"
            if "max_exploitability" in e
            else "not audited"
        )
        lines.append(
            f"| {name} | {res['seeds'][name]} | {_fmt(o['welfare_ratio'])} | "
            f"{_fmt(o['padding_index'])} | {_fmt(o['specification_gap'])} | {conv} |"
        )
    lines += [
        "",
        "## Gap closed, seed-paired (Delta_X = welfare_ratio(C_X) - welfare_ratio(C0))",
        "",
    ]
    for name, d in res["deltas"].items():
        lines.append(
            f"- {name}: {d['delta']:.4f} [{d['ci_lo']:.4f}, {d['ci_hi']:.4f}] over "
            f"{d['n_pairs']} paired seeds{label(name)}"
        )
    it = res["interaction"]
    if "I" in it:
        lines.append(
            f"- Interaction {it['formula']}: {it['I']:.4f} [{it['ci_lo']:.4f}, "
            f"{it['ci_hi']:.4f}] over {it['n_pairs']} paired seeds"
        )
    lines += ["", "## Oracle", ""]
    for name, o in res["oracle"].items():
        lines.append(
            f"- {name}: W_oracle {o['welfare']:.4f}, val_oracle {o['val']:.4f}, "
            f"{o['solver']} ({o['solver_version']}), gap {o['optimality_gap']:.2e}"
        )
    lines += ["", f"Flags: {', '.join(res['flags']) or 'none'}. git `{res['git']}`.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
