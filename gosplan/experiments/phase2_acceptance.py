"""Phase-2 acceptance experiment - gate G3 (PLAN sections 4.1, 4.2, 6.2, 6.3, 7.5, 13).

Realises WO-031 under the pre-registration of `spec/P2_REVISION.md` R14, which fixes the arms, the
seeds, the pass rule of every held-out row, the exploitability subset, the oracle horizon, the price
seeds and the parity check, all before any Phase-2 learning run. The harness composes existing
functions (WO-018 training, WO-027 oracle, WO-028 exploitability, WO-029 parity, WO-030 metrics) and
re-implements none of them.

HELD OUT. This module is the first and only place PLAN section 4.1 rows 2, 5, 6 and 7 are computed
on simulated data (WO-030). A row that does not appear is a REPORTED FAILURE (PLAN section 4.2):
nothing here is re-run, re-seeded or re-parameterised to make it appear.

The report carries one pass/fail line per G3 condition, then the numbers, as `phase1_gate` does.
"""

from __future__ import annotations

import dataclasses
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from gosplan.config import EnvConfig

OUT_DIR = Path("runs/phase2_acceptance")
"""Artefact directory (PLAN section 13 names `runs/phase2_acceptance/report.md` as the G3 record)."""

REPORT_PATH = OUT_DIR / "report.md"
TABLE_PATH = OUT_DIR / "table.parquet"

SEED_ROOT = 1000
"""`seed_env = SEED_ROOT + s` for seed index `s`, shared across arms and baseline legs (R14)."""

ARMS: dict[str, dict[str, dict[str, object]]] = {
    "C0": {},
    "R7_NULL": {"incentive": {"growth_directive": 0.0, "penalty_arg": "absolute"}},
    "R3_QW": {
        "incentive": {"objective_metric": "quality_weighted"},
        "information": {"quality_measurability": 1.0},
    },
}
"""The three R14 arms, as overrides of `p2_default_config()`."""

SEEDS: dict[str, int] = {"C0": 30, "R7_NULL": 10, "R3_QW": 10}
EXPLOIT_SEEDS: dict[str, int] = {"C0": 10, "R7_NULL": 5, "R3_QW": 5}
BR_TOTAL_AGENT_STEPS = 500_000
ORACLE_HORIZON = 40
CLAIRVOYANT_SEEDS = 5
PRICE_SEEDS = (11, 12, 13)
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
BOOTSTRAP_LEVEL = 0.95
VANISH_TOL = 0.01
PARITY_TOL = 1e-5
PARITY_STEPS = 100
TARGET_RUNAWAY_MAX_FRAC = 0.05

CONDITIONS = (
    "heldout_evaluated",
    "exploitability",
    "oracle_gap_recorded",
    "jax_parity",
    "price_sensitivity",
    "hygiene",
)
"""The G3 pass conditions of WO-031, one report line each."""


# ---------- configuration ----------


def arm_config(arm: str, seed_index: int) -> EnvConfig:
    """`p2_default_config()` with the arm's overrides, seeded for `seed_index`."""
    from gosplan.config import p2_default_config
    from gosplan.experiments import _g2

    cfg = _g2.with_overrides(p2_default_config(), ARMS[arm])
    return _g2.seeded(cfg, seed_index, SEED_ROOT)


def br_sizing():
    from gosplan.experiments.phase1_gate import GATE_SIZING

    return dataclasses.replace(GATE_SIZING, total_agent_steps=BR_TOTAL_AGENT_STEPS)


def exploitability_arms(
    out_dir: Path = OUT_DIR, exploit_seeds: dict[str, int] | None = None
) -> dict[str, Path]:
    """Per arm, a directory of symlinks to the R14 exploitability subset of its trained runs
    (seed indices `0 .. EXPLOIT_SEEDS[arm] - 1`), the input `exploitability.run` expects."""
    out: dict[str, Path] = {}
    for arm, n in (EXPLOIT_SEEDS if exploit_seeds is None else exploit_seeds).items():
        audit = Path(out_dir) / "exploit_subset" / arm
        audit.mkdir(parents=True, exist_ok=True)
        for s in range(n):
            target = (Path(out_dir) / "runs" / arm_config(arm, s).hash()).resolve()
            link = audit / target.name
            if target.exists() and not link.exists():
                link.symlink_to(target, target_is_directory=True)
        out[arm] = audit
    return out


# ---------- per-seed measurement ----------


def _rollout(policy, cfg: EnvConfig, n_episodes: int, seed_env: int):
    """Run `n_episodes` of `policy(obs, phase) -> EnterpriseAction` through a ledger."""
    from gosplan.env.env import GosplanEnv
    from gosplan.metrics.ledger import Ledger

    env = GosplanEnv(cfg)
    ledger = Ledger()
    env.attach_ledger(ledger)
    for e in range(n_episodes):
        obs, _ = env.reset(seed_env + e, cfg.tech.seed_policy)
        done = False
        while not done:
            obs, _reward, done, _info = env.step(policy(obs, env.phase()))
    return ledger


def _period_series(ledger, cfg: EnvConfig, price_vectors: tuple) -> dict[str, object]:
    """Window welfare and `val_measured` per period, plus `val_measured` re-priced under each
    perturbed price vector (and at the base prices, as a check against the logged value)."""
    from gosplan.metrics.phenomena import MEASUREMENT_FIRST_PERIOD

    m = cfg.incentive.steps_per_period
    mu = cfg.information.quality_measurability
    periods: dict[tuple[int, int], list] = {}
    for rec in ledger.records:
        if rec.phase == "report" and rec.t_period >= MEASUREMENT_FIRST_PERIOD:
            periods.setdefault((rec.episode, rec.t_period), []).append(rec)
    welfare, val_logged, repriced = [], [], [[] for _ in price_vectors]
    for recs in periods.values():
        welfare.append(recs[0].welfare)
        val_logged.append(recs[0].val_measured)
        for r, prices in enumerate(price_vectors):
            total = 0.0
            for rec in recs:
                qbar = rec.quality_acc / m if cfg.supply.quality_matters else 1.0
                total += prices[rec.sector] * rec.report * (1.0 + mu * (qbar - 1.0))
            repriced[r].append(total)
    return {
        "welfare": np.array(welfare, dtype=float),
        "val_logged": np.array(val_logged, dtype=float),
        "val_repriced": [np.array(v, dtype=float) for v in repriced],
    }


def measure_seed(job: tuple) -> dict[str, object]:
    """Measure one trained run and its `TruthfulMyopic` leg (same configuration, same draws).

    Takes `(arm, seed_index, run_root, sizing, ppo_cfg)`. Returns, and writes to
    `run_dir / "p2_measure.json"`, the per-seed statistics of rows 2, 3, 5, 6 and 7, the window
    welfare and `val_measured` (base and re-priced), and hygiene figures. Resumable: an existing
    measurement for the same sizing is read back.
    """
    from gosplan.agents.heuristic import TruthfulMyopic
    from gosplan.agents.ppo.adapter import IPPO
    from gosplan.agents.ppo.train import checkpoint_path, evaluate
    from gosplan.env.prices import initial_prices, perturbed_price_vectors
    from gosplan.experiments._g2 import MEASURE_SEED_OFFSET
    from gosplan.metrics.phenomena import (
        phenomenon_blat,
        phenomenon_hidden_reserves,
        phenomenon_hoarding,
        phenomenon_quality,
        phenomenon_storming,
    )

    arm, s, run_root, sizing, ppo_cfg = job
    cfg = arm_config(arm, s)
    run_dir = Path(run_root) / cfg.hash()
    out_path = run_dir / "p2_measure.json"
    sizing_record = dataclasses.asdict(sizing)
    if out_path.exists():
        done = json.loads(out_path.read_text(encoding="utf-8"))
        if done.get("sizing") == sizing_record:
            return done
    n_updates = sizing.total_agent_steps // (sizing.n_envs * sizing.rollout_steps)
    agent = IPPO(cfg, ppo_cfg)
    agent.load_checkpoint(checkpoint_path(run_dir, n_updates - 1))
    seed = int(cfg.tech.seed_env) + MEASURE_SEED_OFFSET
    metrics, ledger = evaluate(agent, cfg, sizing.measure_episodes, seed)
    rng = np.random.default_rng(0)
    tm = TruthfulMyopic(cfg)
    tm_ledger = _rollout(lambda o, ph: tm.act(o, ph, rng), cfg, sizing.measure_episodes, seed)

    base_prices = np.asarray(initial_prices(cfg), dtype=float)
    vectors = (base_prices, *perturbed_price_vectors(base_prices, PRICE_SEEDS))
    series = _period_series(ledger, cfg, vectors)
    tm_series = _period_series(tm_ledger, cfg, vectors[:1])

    def _floats(d):
        return {k: float(v) for k, v in d.items()}

    summary = {
        "arm": arm,
        "seed_index": int(s),
        "seed_env": int(cfg.tech.seed_env),
        "config_hash": cfg.hash(),
        "sizing": sizing_record,
        "storming": _floats(phenomenon_storming(ledger, tm_ledger, cfg)),
        "hoarding": _floats(phenomenon_hoarding(ledger, tm_ledger, cfg)),
        "blat": _floats(phenomenon_blat(ledger, cfg)),
        "blat_baseline": _floats(phenomenon_blat(tm_ledger, cfg)),
        "hidden_reserves": _floats(phenomenon_hidden_reserves(ledger, cfg)),
        "quality": _floats(phenomenon_quality(ledger, cfg)),
        "welfare_mean": float(series["welfare"].mean()),
        "welfare_mean_baseline": float(tm_series["welfare"].mean()),
        "val_measured_mean": float(series["val_logged"].mean()),
        "val_repriced_mean": [float(v.mean()) for v in series["val_repriced"]],
        "val_reprice_check": float(
            np.max(np.abs(series["val_repriced"][0] - series["val_logged"]))
        ),
        "metrics": _floats(metrics),
        "bound_binding": bool(ledger.flags and "BOUND_BINDING" in ledger.flags),
    }
    out_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")
    return summary


# ---------- aggregation (pure; unit-tested on synthetic numbers) ----------


def bootstrap_mean_ci(values, n_resamples: int = BOOTSTRAP_RESAMPLES) -> tuple[float, float, float]:
    """Mean over seeds and its percentile seed-bootstrap interval (R14). NaNs are dropped."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = x[rng.integers(0, x.size, size=(n_resamples, x.size))].mean(axis=1)
    alpha = (1.0 - BOOTSTRAP_LEVEL) / 2.0
    return float(x.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def diff_ci(a, b, n_resamples: int = BOOTSTRAP_RESAMPLES) -> tuple[float, float, float]:
    """Difference of means of two unpaired seed sets and its percentile bootstrap interval."""
    xa = np.asarray(a, dtype=float)
    xb = np.asarray(b, dtype=float)
    xa, xb = xa[np.isfinite(xa)], xb[np.isfinite(xb)]
    if xa.size == 0 or xb.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    da = xa[rng.integers(0, xa.size, size=(n_resamples, xa.size))].mean(axis=1)
    db = xb[rng.integers(0, xb.size, size=(n_resamples, xb.size))].mean(axis=1)
    d = da - db
    alpha = (1.0 - BOOTSTRAP_LEVEL) / 2.0
    return (
        float(xa.mean() - xb.mean()),
        float(np.quantile(d, alpha)),
        float(np.quantile(d, 1 - alpha)),
    )


def heldout_verdicts(per_arm: dict[str, list[dict]]) -> dict[str, dict[str, object]]:
    """Apply the R14 pass rules to per-seed summaries. Pure: no environment is touched."""
    c0 = per_arm["C0"]
    out: dict[str, dict[str, object]] = {}

    m, lo, hi = bootstrap_mean_ci([r["storming"]["excess"] for r in c0])
    out["row2_storming"] = {"excess": (m, lo, hi), "appears": bool(lo > 0)}

    infl = [
        r["hoarding"]["request_inflation"] - r["hoarding"]["request_inflation_baseline"] for r in c0
    ]
    corr_diff, n_nan = [], 0
    for r in c0:
        run_c = r["hoarding"]["corr_stock_shortfall"]
        base_c = r["hoarding"]["corr_stock_shortfall_baseline"]
        if not np.isfinite(run_c):
            n_nan += 1
            continue
        corr_diff.append(run_c - (base_c if np.isfinite(base_c) else 0.0))
    ci_infl = bootstrap_mean_ci(infl)
    ci_corr = bootstrap_mean_ci(corr_diff)
    out["row5_hoarding"] = {
        "inflation_excess": ci_infl,
        "corr_excess": ci_corr,
        "n_seeds_corr_nan": n_nan,
        "appears": bool(ci_infl[1] > 0 and ci_corr[1] > 0),
    }

    ci_share = bootstrap_mean_ci([r["blat"]["trade_volume_share"] for r in c0])
    out["row6_blat"] = {"trade_volume_share": ci_share, "appears": bool(ci_share[1] > 0)}

    ci_hr = bootstrap_mean_ci([r["hidden_reserves"]["hidden_reserves"] for r in c0])
    null = per_arm.get("R7_NULL", [])
    ci_null = bootstrap_mean_ci([r["hidden_reserves"]["hidden_reserves"] for r in null])
    present = bool(ci_hr[1] > 0)
    vanishes = bool(np.isfinite(ci_null[2]) and ci_null[2] < VANISH_TOL)
    out["row7_hidden_reserves"] = {
        "c0": ci_hr,
        "null": ci_null,
        "present": present,
        "vanishes_under_null": vanishes,
        "appears": bool(present and vanishes),
    }

    qw = per_arm.get("R3_QW", [])
    ci_q = diff_ci(
        [r["quality"]["mean_quality"] for r in c0], [r["quality"]["mean_quality"] for r in qw]
    )
    out["row3_quality"] = {"diff_c0_minus_qw": ci_q, "appears": bool(ci_q[2] < 0)}
    return out


def price_table(
    rows: list[dict], w_oracle: float, val_oracle_by_vector: list[float]
) -> dict[str, object]:
    """`specification_gap` and `welfare_ratio` (mean over seeds) under the base and each perturbed
    price vector; `sign_change` is True when any perturbed gap differs in sign from the base."""
    gaps = []
    for r_i, val_oracle in enumerate(val_oracle_by_vector):
        per_seed = [
            r["val_repriced_mean"][r_i] / val_oracle - r["welfare_mean"] / w_oracle for r in rows
        ]
        gaps.append(float(np.mean(per_seed)))
    ratio = float(np.mean([r["welfare_mean"] / w_oracle for r in rows]))
    signs = {float(np.sign(g)) for g in gaps}
    return {"specification_gap": gaps, "welfare_ratio": ratio, "sign_change": len(signs) > 1}


# ---------- orchestration ----------


def _run_exploit(args: tuple) -> dict[str, object]:
    from gosplan.experiments import exploitability

    arm, audit_dir, out_dir = args
    res = exploitability.run(
        None, arm, audit_dir, Path(out_dir) / "exploitability", sizing=br_sizing()
    )
    return {k: v for k, v in res.items() if k != "per_seed"} | {"per_seed": list(res["per_seed"])}


def _pool(fn, jobs: list, workers: int) -> list:
    os.environ.setdefault(
        "XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    )
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with ProcessPoolExecutor(
        max_workers=max(1, min(workers, len(jobs))),
        mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
        return list(pool.map(fn, jobs))


def jax_parity() -> dict[str, object]:
    """WO-029 parity on `p1_default_config()` and `p2_default_config()` (R14)."""
    try:
        from gosplan.jax import parity_max_deviation
    except ImportError as exc:
        return {"status": "not available", "reason": str(exc), "passed": False}
    from gosplan.config import p1_default_config, p2_default_config

    devs = {
        "p1": float(parity_max_deviation(p1_default_config(), PARITY_STEPS)),
        "p2": float(parity_max_deviation(p2_default_config(), PARITY_STEPS)),
    }
    return {
        "status": "run",
        "max_deviation": devs,
        "passed": all(d <= PARITY_TOL for d in devs.values()),
    }


def run(
    out_dir: Path = OUT_DIR,
    sizing=None,
    seeds: dict[str, int] | None = None,
    exploit_seeds: dict[str, int] | None = None,
) -> dict[str, object]:
    """Train, measure, audit and report the R14 acceptance set. `sizing`, `seeds` and
    `exploit_seeds` default to R14's values; other values are for a smoke run only."""
    from gosplan.experiments import _g2
    from gosplan.experiments.phase1_gate import GATE_SIZING, study_ppo_config
    from gosplan.oracle.kantorovich import solve_oracle

    sizing = GATE_SIZING if sizing is None else sizing
    seeds = SEEDS if seeds is None else seeds
    exploit_seeds = EXPLOIT_SEEDS if exploit_seeds is None else exploit_seeds
    out_dir = Path(out_dir)
    run_root = out_dir / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    ppo = study_ppo_config()

    jobs = [(arm, s) for arm in ARMS for s in range(seeds[arm])]
    train_summ = _g2.run_many([(arm_config(a, s), sizing, run_root, ppo) for a, s in jobs])
    measured = _pool(
        measure_seed, [(a, s, run_root, sizing, ppo) for a, s in jobs], _g2.MAX_WORKERS
    )
    per_arm: dict[str, list[dict]] = {arm: [] for arm in ARMS}
    for (arm, _s), summ, meas in zip(jobs, train_summ, measured, strict=True):
        meas["runaway_frac"] = summ["runaway_frac_after_20pct"]
        meas["manifest_flags"] = summ["manifest_flags"]
        per_arm[arm].append(meas)

    # Oracle per arm, plus clairvoyant upper bounds for C0.
    from gosplan.env.prices import initial_prices, perturbed_price_vectors

    oracle = {arm: solve_oracle(arm_config(arm, 0), ORACLE_HORIZON, False, None) for arm in ARMS}
    clair = [
        solve_oracle(
            arm_config("C0", s), ORACLE_HORIZON, True, int(arm_config("C0", s).tech.seed_env)
        )
        for s in range(min(CLAIRVOYANT_SEEDS, seeds["C0"]))
    ]
    base_p = np.asarray(initial_prices(arm_config("C0", 0)), dtype=float)
    vectors = (base_p, *perturbed_price_vectors(base_p, PRICE_SEEDS))
    sector = np.asarray(arm_config("C0", 0).supply.sector_of, dtype=int)
    val_oracle = [float(np.dot(p[sector], oracle["C0"]["output_mean"])) for p in vectors]
    prices = price_table(per_arm["C0"], oracle["C0"]["welfare"], val_oracle)

    # Exploitability on the R14 subset.
    subset_dirs = {arm: d for arm, d in exploitability_arms(out_dir, exploit_seeds).items()}
    exploit = dict(
        zip(
            subset_dirs,
            _pool(
                _run_exploit, [(a, d, out_dir) for a, d in subset_dirs.items()], len(subset_dirs)
            ),
            strict=True,
        )
    )

    verdicts = heldout_verdicts(per_arm)
    parity = jax_parity()
    all_rows = [r for rows in per_arm.values() for r in rows]
    bound_runs = [
        r["config_hash"]
        for r in all_rows
        if r["bound_binding"] or "BOUND_BINDING" in r["manifest_flags"]
    ]
    runaway = [r["runaway_frac"] for r in all_rows]
    conditions = {
        "heldout_evaluated": True,
        "exploitability": not any(e["non_converged"] for e in exploit.values()),
        "oracle_gap_recorded": all(np.isfinite(o["optimality_gap"]) for o in oracle.values()),
        "jax_parity": bool(parity["passed"]),
        "price_sensitivity": True,
        "hygiene": bool(not bound_runs and np.nanmax(runaway) < TARGET_RUNAWAY_MAX_FRAC),
    }
    result = {
        "conditions": conditions,
        "passed": all(conditions.values()),
        "verdicts": verdicts,
        "oracle": oracle,
        "clairvoyant": [c["welfare"] for c in clair],
        "prices": prices,
        "exploitability": exploit,
        "parity": parity,
        "bound_binding_runs": bound_runs,
        "max_runaway_frac": float(np.nanmax(runaway)),
        "per_arm": per_arm,
        "sizing": dataclasses.asdict(sizing),
        "git": _g2.git_hash(),
    }
    import pandas as pd

    flat = []
    for r in all_rows:
        flat.append(
            {
                "arm": r["arm"],
                "seed_index": r["seed_index"],
                "seed_env": r["seed_env"],
                "config_hash": r["config_hash"],
                **{f"storming_{k}": v for k, v in r["storming"].items()},
                **{f"hoarding_{k}": v for k, v in r["hoarding"].items()},
                **{f"blat_{k}": v for k, v in r["blat"].items()},
                **{f"hidden_{k}": v for k, v in r["hidden_reserves"].items()},
                **{f"quality_{k}": v for k, v in r["quality"].items()},
                "welfare_mean": r["welfare_mean"],
                "welfare_mean_baseline": r["welfare_mean_baseline"],
                "val_measured_mean": r["val_measured_mean"],
                "runaway_frac": r["runaway_frac"],
                "bound_binding": r["bound_binding"],
            }
        )
    pd.DataFrame(flat).to_parquet(out_dir / TABLE_PATH.name, index=False)
    (out_dir / "result.json").write_text(
        json.dumps(result, default=_jsonable, indent=1), encoding="utf-8"
    )
    (out_dir / REPORT_PATH.name).write_text(render_report(result), encoding="utf-8")
    return result


def _jsonable(value: object) -> object:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _ci(t) -> str:
    m, lo, hi = t
    return f"{m:.4f} [{lo:.4f}, {hi:.4f}]"


def render_report(result: dict) -> str:
    """The G3 report: one pass/fail line per condition, then the numbers (R14)."""
    c = result["conditions"]
    v = result["verdicts"]
    ok = {True: "PASS", False: "FAIL"}
    lines = [
        "# Gate G3 - Phase-2 acceptance (WO-031)",
        "",
        "Pre-registration: `spec/P2_REVISION.md` R14. Limitations carried from Phase 1: L1 (PPO does "
        "not recover the single-enterprise DP's mixed under-reporting strategy) and L2 (the "
        "bunching estimator on degenerate and peaked distributions); see `runs/G2_record.md`.",
        "",
        f"**G3: {'PASSED' if result['passed'] else 'NOT PASSED'}**",
        "",
        "## Conditions",
        "",
        f"- heldout_evaluated: {ok[c['heldout_evaluated']]} (rows 2, 5, 6, 7 computed on the "
        "PLAN section 4.2 values; appearance is reported per row below)",
        f"- exploitability: {ok[c['exploitability']]}",
        f"- oracle_gap_recorded: {ok[c['oracle_gap_recorded']]}",
        f"- jax_parity: {ok[c['jax_parity']]}",
        f"- price_sensitivity: {ok[c['price_sensitivity']]} (computed; a sign change is reported "
        "below, never suppressed)",
        f"- hygiene: {ok[c['hygiene']]}",
        "",
        "## Held-out phenomena (C0, mean over seeds [95% seed-bootstrap CI])",
        "",
        f"- Row 2 storming: excess Gini {_ci(v['row2_storming']['excess'])} - "
        f"{'APPEARS' if v['row2_storming']['appears'] else 'FAILURE (does not appear)'}",
        f"- Row 5 hoarding: request inflation excess {_ci(v['row5_hoarding']['inflation_excess'])}; "
        f"corr(X, shortfall) excess {_ci(v['row5_hoarding']['corr_excess'])} "
        f"({v['row5_hoarding']['n_seeds_corr_nan']} seeds with an undefined correlation) - "
        f"{'APPEARS' if v['row5_hoarding']['appears'] else 'FAILURE (does not appear)'}",
        f"- Row 6 blat: trade volume share {_ci(v['row6_blat']['trade_volume_share'])} - "
        f"{'APPEARS' if v['row6_blat']['appears'] else 'FAILURE (does not appear)'}",
        f"- Row 7 hidden reserves: C0 {_ci(v['row7_hidden_reserves']['c0'])}; R7_NULL "
        f"{_ci(v['row7_hidden_reserves']['null'])} (vanishes if upper < {VANISH_TOL}) - "
        f"{'APPEARS' if v['row7_hidden_reserves']['appears'] else 'FAILURE'}"
        + (
            ""
            if v["row7_hidden_reserves"]["appears"]
            else f" (present: {v['row7_hidden_reserves']['present']}, vanishes under null: "
            f"{v['row7_hidden_reserves']['vanishes_under_null']})"
        ),
        f"- Row 3 quality (pipeline check): mean qbar C0 - R3_QW "
        f"{_ci(v['row3_quality']['diff_c0_minus_qw'])} - "
        f"{'APPEARS' if v['row3_quality']['appears'] else 'FAILURE'}",
        "",
        "## Oracle (WO-027)",
        "",
    ]
    for arm, o in result["oracle"].items():
        lines.append(
            f"- {arm}: W_oracle {o['welfare']:.4f}, val_oracle {o['val']:.4f}, status "
            f"{o['status']}, solver {o['solver']} ({o['solver_version']}), optimality gap "
            f"{o['optimality_gap']:.2e}, horizon {o.get('horizon')}"
        )
    lines += [
        f"- Clairvoyant welfare, C0 seeds 0-{len(result['clairvoyant']) - 1} (UPPER BOUND ONLY, never "
        f"a denominator): {', '.join(f'{x:.4f}' for x in result['clairvoyant'])}",
        "",
        "## Headline metrics and price sensitivity (C0)",
        "",
        f"- welfare_ratio W / W_oracle: {result['prices']['welfare_ratio']:.4f}",
        "- specification_gap at base prices and under price seeds "
        f"{PRICE_SEEDS}: {', '.join(f'{g:.4f}' for g in result['prices']['specification_gap'])}"
        f" - sign change: {'YES (reported)' if result['prices']['sign_change'] else 'no'}",
        "",
        "## Exploitability (WO-028)",
        "",
    ]
    for arm, e in result["exploitability"].items():
        lines.append(
            f"- {arm}: max {e['max_exploitability']:.4f}, median {e['median_exploitability']:.4f} "
            f"over {len(e['per_seed'])} seeds (threshold {e['threshold']}, "
            f"{e['threshold_status']}) {e['label']}"
        )
    p = result["parity"]
    lines += [
        "",
        "## JAX parity (WO-029)",
        "",
        (
            f"- max |NumPy - JAX| over {PARITY_STEPS} agent-steps: "
            + ", ".join(f"{k} {d:.2e}" for k, d in p["max_deviation"].items())
            + f" (tolerance {PARITY_TOL})"
            if p["status"] == "run"
            else f"- NOT RUN: {p['reason']}"
        ),
        "",
        "## Hygiene",
        "",
        f"- BOUND_BINDING runs: {len(result['bound_binding_runs'])}",
        f"- max training-episode runaway fraction after 20%: {result['max_runaway_frac']:.4f} "
        f"(limit {TARGET_RUNAWAY_MAX_FRAC})",
        "",
        "## Arms and seeds",
        "",
    ]
    for arm, rows in result["per_arm"].items():
        lines.append(f"- {arm}: {len(rows)} seeds, overrides {ARMS[arm] or 'none'}")
    lines += ["", f"Sizing: `{result['sizing']}`; git `{result['git']}`.", ""]
    return "\n".join(lines)


def main() -> int:
    """Run the R14 acceptance set and write `runs/phase2_acceptance/report.md`."""
    import sys

    if "--smoke" in sys.argv:
        from gosplan.experiments.phase1_gate import GATE_SIZING

        out = Path(sys.argv[sys.argv.index("--smoke") + 1])
        tiny = dataclasses.replace(
            GATE_SIZING,
            total_agent_steps=2_000,
            rollout_steps=25,
            eval_every_updates=5,
            eval_episodes=1,
            measure_episodes=2,
        )
        global BR_TOTAL_AGENT_STEPS, ORACLE_HORIZON, CLAIRVOYANT_SEEDS
        BR_TOTAL_AGENT_STEPS, ORACLE_HORIZON, CLAIRVOYANT_SEEDS = 2_000, 6, 1
        res = run(out, tiny, {a: 2 for a in ARMS}, {a: 1 for a in ARMS})
    else:
        res = run()
    for name in CONDITIONS:
        print(f"{name}: {'PASS' if res['conditions'][name] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
