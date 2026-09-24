"""Monte-Carlo sanity harness - PLAN sections 12.3 (WO-012 card), 13 (gate G0) and 14.

Realises: the WO-012 card of PLAN section 12.3. Owning work order: **WO-012** (MID-strong,
difficulty 3; depends on WO-010 heuristics and WO-011 ledger). Gate: **G0** - the lead signs off on
a green frozen suite, a clean report from this harness, and a review of every `gosplan/env/` diff
against CONTRACT rule 7.

Inputs
    `p1_default_config()` (PLAN section 3) plus `N_SUPPLY_PERTURBATIONS` random perturbations of the
    SUPPLY arm, each parameter drawn inside its PLAN section 3 sweep range (see
    `SUPPLY_PERTURBATION_RANGES` and `SUPPLY_PERTURBATION_CHOICES`); the three Phase-1 heuristics of
    PLAN section 6.1 - `Random`, `TruthfulMyopic`, `Padder` - from `gosplan.agents.heuristic`;
    `gosplan.env.env.GosplanEnv`; `gosplan.metrics.ledger.Ledger` and `write_manifest`.

Outputs
    `runs/mc_sanity/report.md`   the artefact gate G0 signs off on (PLAN section 13). One section
                                 per configuration and agent: every assertion below with the
                                 observed extreme value, the wall clock per episode, the flags
                                 raised (CONTRACT rule 8 `BOUND_BINDING` among them), and the
                                 configuration hash that names the run directory.
    `runs/<config-hash>/`        per-configuration run directory holding `manifest.json` and the
                                 ledger, exactly as CONTRACT rule 10 requires of every run. This
                                 harness writes no manifest of its own; the report references the
                                 hashes.

Cost (PLAN section 14): 3 agents x 2,000 episodes x 21 configurations at about 50 agent-steps per
episode - CPU minutes.

FORBIDDEN - HELD-OUT PHENOMENA (PLAN sections 4.1 and 12.3, WO-012 card)
    ***  THIS HARNESS MUST NEVER COMPUTE, TABULATE, PLOT OR ASSERT A DIRECTION FOR ANY QUANTITY ***
    ***  IN PLAN SECTION 4.1 ROWS 2, 5, 6 OR 7.                                                 ***

    Row 2 storming - the within-period effort Gini `Gini_k(e_ik)` and any excess over a baseline.
    Row 5 hoarding - request inflation `q_ij / need_ij`, input stocks `X_ij`, and any correlation
                     between held inputs and downstream fill.
    Row 6 blat     - executed trade volume, matched pairs or trade surplus.
    Row 7 hidden reserves - `max(0, S_i - R_i) / T_i`, and any reconciliation statistic on it.

    "Inspected" means: no plot, no table, no test, not in passing and not while debugging the
    mechanisms behind them (PLAN section 4.1). They are computed for the first time in the Phase-2
    acceptance run (WO-030, WO-031). The Monte-Carlo checks that touch the same mechanisms assert
    only conservation and boundedness, never a direction.

    The one directional assertion this harness does make - that `Padder` produces downstream
    shortage - is about `fill`, the delivery channel of PLAN section 2.7.3, and is the property test
    T-B3 already states. It is not row 5: no `X_ij`, no request inflation and no correlation with
    held stock may be computed to support it.

Runtime bindings. `EnvConfig` is `gosplan.config.EnvConfig`, the runtime dataclass that must stay
field-for-field identical to `spec/spec.py` (`spec/` is a frozen interface document, not an
importable package; a unit test enforces the agreement). `pyarrow`, `pandas` and any plotting
dependency are imported inside the function that uses them, never at module scope.
"""

from __future__ import annotations

from pathlib import Path

from gosplan.config import EnvConfig

N_EPISODES = 2000
"""Episodes per (agent, configuration) cell (WO-012 card). The PLAN section 14 cost line is
3 agents x 2,000 episodes x 21 configurations."""

AGENTS = ("Random", "TruthfulMyopic", "Padder")
"""The three Phase-1 heuristics exercised here, by their `gosplan.agents.heuristic` class names
(PLAN section 6.1). `Padder` is a sanity agent only - it reports `rho = 1` at effort 0.3 to exercise
the shortage channel and is never a baseline. `DPGreedy` is deliberately absent: it needs a
`DPSolution`, which does not exist until WO-014, i.e. after gate G0."""

N_SUPPLY_PERTURBATIONS = 20
"""Random perturbations of the SUPPLY arm run alongside the baseline (WO-012 card)."""

N_CONFIGS = 21
"""Configurations in total: the baseline plus `N_SUPPLY_PERTURBATIONS`. This is the "21 configs" of
the PLAN section 14 cost line, and the harness checks the list it builds against it."""

CONSERVATION_TOL = 1e-9
"""Absolute tolerance on the per-period conservation identity of test T-U1 (WO-012 card). A
violation is a reported failure, never a widened tolerance."""

FILL_BOUNDS = (0.0, 1.0)
"""Inclusive bounds `fill_i` must satisfy every period (PLAN section 2.7.3). `claimed = 0` gives
`fill = 1` by the WO-006 convention, so the upper bound is attainable and is not an error."""

HELD_OUT_PHENOMENON_ROWS = (2, 5, 6, 7)
"""Rows of the PLAN section 4.1 table this harness is forbidden to touch: storming, hoarding, blat,
hidden reserves. Declared as data so the report can restate the prohibition it ran under and a
reviewer can grep for it."""

SUPPLY_PERTURBATION_RANGES: dict[str, tuple[float, float]] = {
    "yield_sigma_multiplier": (0.5, 2.0),
    "final_demand_share": (0.3, 0.7),
    "holding_loss": (0.0, 0.05),
}
"""Continuous SUPPLY parameters perturbed, with the sweep ranges of the PLAN section 3 registry
verbatim. `yield_sigma_multiplier` multiplies the whole `yield_sigma` tuple, which is how PLAN
section 3 states that row ("x[0.5, 2]"); `final_demand_share` and `holding_loss` are drawn once per
configuration and applied to every sector. Draws are uniform on the range, independent across
parameters, from a generator seeded by the harness seed and recorded in the report."""

SUPPLY_PERTURBATION_CHOICES: dict[str, tuple[float, ...]] = {
    "input_complementarity": (2.0, 8.0, float("inf")),
}
"""Discrete SUPPLY parameters perturbed, with the PLAN section 3 grid verbatim. `theta = inf` is the
Leontief `min` branch of `coverage` (PLAN section 2.6) and must be exercised here: it is the branch
the Phase-1 default of 8.0 never reaches."""

PERTURBATION_EXCLUDED: tuple[str, ...] = (
    "delivery_timing",
    "arrival_probs",
    "input_holding_loss",
    "trade_tau",
    "alloc_eta_request",
)
"""Parameters this harness must leave at their Phase-1 values. They are the mechanism parameters
locked in PLAN section 4.2 behind held-out phenomena 2 (storming), 5 (hoarding) and 6 (blat);
perturbing them here would exercise a locked mechanism before its pre-registered study exists. They
may be changed only in a new, separately pre-registered study."""

OUT_DIR = Path("runs/mc_sanity")
"""Artefact directory, relative to the repository root (PLAN sections 12.3, 13)."""

REPORT_PATH = OUT_DIR / "report.md"
"""The gate G0 artefact named in PLAN section 13."""


def run(
    cfg: EnvConfig,
    out_dir: Path = OUT_DIR,
    n_episodes: int = N_EPISODES,
    n_perturbations: int = N_SUPPLY_PERTURBATIONS,
    seed_env: int | None = None,
) -> dict[str, object]:
    """Run the Monte-Carlo sanity sweep and write the gate G0 report.

    Takes: `cfg`, the base configuration (WO-012 runs `p1_default_config()`), already validated;
    `out_dir`, where `report.md` is written; `n_episodes`, episodes per (agent, configuration) cell;
    `n_perturbations`, random SUPPLY perturbations to add to the baseline; `seed_env`, the root
    environment seed - `None` means `cfg.tech.seed_env`. One `seed_env` is shared by all three
    agents, so they meet common random numbers by construction (PLAN sections 2.15, 4.3), and it is
    recorded in every run manifest.

    Returns: a mapping with at least

        "n_configs"              int, equal to `1 + n_perturbations` (`N_CONFIGS` at defaults)
        "config_hashes"          tuple[str, ...], `EnvConfig.hash()` per configuration, in order
        "n_episodes"             int, episodes per cell actually run
        "max_conservation_error" float, the largest violation of the T-U1 identity seen anywhere
        "n_nonfinite"            int, NaN or inf values found across every ledger column
        "target_bound_failures"  int, periods with `T_i` outside its analytic bounds (below)
        "stock_bound_failures"   int, periods with `S_i` outside `[0, S_max]`
        "fill_bound_failures"    int, periods with `fill_i` outside `FILL_BOUNDS`
        "padder_min_fill"        float, the smallest `fill` observed under `Padder`
        "padder_shortage"        bool, whether some enterprise saw `fill < 1` under `Padder`
        "seconds_per_episode"    float, mean wall clock per episode
        "flags"                  tuple[str, ...], every run-level flag raised, `BOUND_BINDING`
                                 included (CONTRACT rule 8)
        "passed"                 bool, every assertion below held
        "report_path"            str, the file written

    Procedure (WO-012 card, PLAN section 12.3):

      1. Build the configuration list: the baseline, then `n_perturbations` draws in which every
         parameter of `SUPPLY_PERTURBATION_RANGES` is drawn uniformly on its range and every
         parameter of `SUPPLY_PERTURBATION_CHOICES` uniformly over its grid, independently, leaving
         every name in `PERTURBATION_EXCLUDED` at its Phase-1 value. Validate each configuration
         with `EnvConfig.validate()` and name it by `EnvConfig.hash()`.
      2. For each configuration x agent in `AGENTS`, run `n_episodes` episodes of `GosplanEnv`,
         logging one `StepRecord` per enterprise per agent-step to a `Ledger`, and write
         `runs/<hash>/` with `write_manifest` (CONTRACT rule 10).
      3. Assert the following, recording the observed extreme for each rather than stopping at the
         first failure:
           - the per-period conservation identity of test T-U1 holds to `CONSERVATION_TOL`;
           - no NaN and no inf appears in any ledger column;
           - `T_i` stays inside `[T_min, T_0 * ((1 + g) * (1 + lambda * c_up))**P_max]` - the floor
             of PLAN section 2.7.1 and the largest value the capped ratchet can reach in `P_max`
             periods; both endpoints are computed from the configuration, never hard-coded;
           - `S_i` stays inside `[0, inventory_cap_mult * cap_i]`, the cap of PLAN section 2.11;
           - `fill_i` stays inside `FILL_BOUNDS`;
           - under `Padder` at least one enterprise sees `fill < 1` in some period, i.e. the
             delivery channel of PLAN section 2.7.3 propagates a shortage (the property of test
             T-B3; NOT a held-out row - see the module docstring);
           - the wall clock per episode is measured, so the PLAN section 14 estimate can be checked
             against reality before Phase 1 commits to it.
      4. Write `report.md`: one section per configuration and agent, every assertion with its
         observed extreme, the flags raised, the per-episode wall clock, and a verbatim restatement
         of the held-out prohibition this harness ran under (`HELD_OUT_PHENOMENON_ROWS`).

    A failed assertion is reported, never repaired by widening a tolerance or moving a bound
    (CONTRACT rule 8). Nothing here feeds a true quantity to an agent (CONTRACT rule 6): the ledger
    is read by this function alone.

    Binds: `tests/unit/test_mc_sanity_runs.py` (smoke - the harness runs a small number of episodes
    and returns the mapping above). Gate: the report is the G0 artefact of PLAN section 13.

    Realises: PLAN sections 12.3 (WO-012), 13, 14. Owning WO: **WO-012**.
    """
    import dataclasses
    import math
    import shutil
    import subprocess
    import time

    import numpy as np
    import pandas as pd

    from gosplan.agents import heuristic
    from gosplan.env.env import GosplanEnv
    from gosplan.env.state import initial_targets
    from gosplan.metrics.ledger import BOUND_BINDING_FLAG, Ledger, write_manifest

    harness_start = time.perf_counter()
    out_dir = Path(out_dir)
    runs_root = out_dir.parent  # `runs/` at the default `OUT_DIR`: run dirs are `runs/<hash>/`
    root_seed = int(cfg.tech.seed_env if seed_env is None else seed_env)
    # The one seed_env of every configuration and every agent (CRN, PLAN sections 2.15, 4.3); it
    # is written into the configuration so each manifest records the seed actually used.
    base = dataclasses.replace(cfg, tech=dataclasses.replace(cfg.tech, seed_env=root_seed))
    base.validate()

    # AMBIGUITY-007 (LEAD): these three StepRecord columns are NaN by construction because their
    # owning functions do not return them. NaN there is a placeholder, not a defect; any inf there
    # is still counted as non-finite. Every other numeric column is scanned for NaN and inf.
    nan_placeholder_columns = ("coverage", "audit_meas", "penalty_arg")
    chunk_episodes = 100  # episodes per ledger part file; memory only, no semantic content

    # 1. Configuration list: baseline, then SUPPLY-only perturbations (PLAN section 3 ranges).
    perturb_rng = np.random.default_rng(root_seed)
    configs: list[EnvConfig] = [base]
    draws: list[dict[str, float]] = [{}]
    for _ in range(n_perturbations):
        drawn: dict[str, float] = {}
        for name, (lo, hi) in SUPPLY_PERTURBATION_RANGES.items():
            drawn[name] = float(perturb_rng.uniform(lo, hi))
        for name, grid in SUPPLY_PERTURBATION_CHOICES.items():
            drawn[name] = float(grid[int(perturb_rng.integers(len(grid)))])
        n_sec = base.supply.n_sectors
        supply = dataclasses.replace(
            base.supply,
            yield_sigma=tuple(s * drawn["yield_sigma_multiplier"] for s in base.supply.yield_sigma),
            final_demand_share=tuple(drawn["final_demand_share"] for _ in range(n_sec)),
            holding_loss=drawn["holding_loss"],
            input_complementarity=drawn["input_complementarity"],
        )
        pert = dataclasses.replace(base, supply=supply)
        pert.validate()
        for name in PERTURBATION_EXCLUDED:
            for section in ("supply", "incentive", "information", "tech"):
                if hasattr(getattr(base, section), name):
                    if getattr(getattr(pert, section), name) != getattr(
                        getattr(base, section), name
                    ):
                        raise RuntimeError(f"perturbation moved locked parameter {name}")
        configs.append(pert)
        draws.append(drawn)
    if n_perturbations == N_SUPPLY_PERTURBATIONS and len(configs) != N_CONFIGS:
        raise RuntimeError(f"built {len(configs)} configurations, expected {N_CONFIGS}")
    hashes = tuple(c.hash() for c in configs)

    try:
        git_hash = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        git_hash = git_hash + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        git_hash = None

    def check_chunk(df: pd.DataFrame, c: EnvConfig, starts: dict, cell: dict) -> None:
        """Fold one ledger part's assertions into `cell` (observed extremes, failure counts)."""
        n_goods = c.supply.n_sectors
        # No NaN / inf in any ledger column.
        for col in df.columns:
            if not (pd.api.types.is_float_dtype(df[col]) or pd.api.types.is_integer_dtype(df[col])):
                continue
            values = df[col].to_numpy(dtype=float)
            if col in nan_placeholder_columns:
                cell["placeholder_nan"] += int(np.isnan(values).sum())
                bad = int(np.isinf(values).sum())
            else:
                bad = int((~np.isfinite(values)).sum())
            if bad:
                cell["nonfinite_by_column"][col] = cell["nonfinite_by_column"].get(col, 0) + bad
                cell["n_nonfinite"] += bad
        # T inside [T_min, T_0 * ((1 + g) * (1 + lambda * c_up))**P_max], per enterprise.
        t0 = np.asarray(initial_targets(c), dtype=float)
        inc = c.incentive
        t_lo = c.tech.target_floor_frac * t0
        t_hi = (
            t0
            * ((1.0 + inc.growth_directive) * (1.0 + inc.ratchet_lambda * inc.ratchet_cap_up))
            ** c.tech.max_periods
        )
        ent = df["enterprise"].to_numpy()
        target = df["target"].to_numpy(dtype=float)
        cell["target_failures"] += int(((target < t_lo[ent]) | (target > t_hi[ent])).sum())
        cell["target_min_over_floor"] = min(
            cell["target_min_over_floor"], float(np.min(target / t_lo[ent]))
        )
        cell["target_max_over_cap"] = max(
            cell["target_max_over_cap"], float(np.max(target / t_hi[ent]))
        )
        cell["target_min"] = min(cell["target_min"], float(target.min()))
        cell["target_max"] = max(cell["target_max"], float(target.max()))
        # S inside [0, inventory_cap_mult * cap_i], before and after every step.
        s_max = c.tech.inventory_cap_mult * df["capital"].to_numpy(dtype=float)
        for col in ("inv_output_pre", "inv_output_post"):
            s = df[col].to_numpy(dtype=float)
            cell["stock_failures"] += int(((s < 0.0) | (s > s_max)).sum())
            cell["stock_min"] = min(cell["stock_min"], float(s.min()))
            cell["stock_max_over_cap"] = max(cell["stock_max_over_cap"], float(np.max(s / s_max)))
        cell["stock_cap"] = float(np.max(s_max))
        # fill inside FILL_BOUNDS.
        fill = df["fill"].to_numpy(dtype=float)
        lo, hi = FILL_BOUNDS
        cell["fill_failures"] += int(((fill < lo) | (fill > hi)).sum())
        cell["fill_min"] = min(cell["fill_min"], float(fill.min()))
        cell["fill_max"] = max(cell["fill_max"], float(fill.max()))
        # T-U1 per period and per good, in the CHANGELOG 0.1.3 form (ambiguity #64):
        #   sum y + sum S_prev + sum X_prev
        #     = sum S_next + sum X_next + sum consumed + consumer + sum holding + sum overflow
        # S_prev / X_prev are the previous period's REPORT-row values, or the reset state at t = 0.
        keys = ["episode", "t_period"]
        goods = list(range(n_goods))
        rep = df[df["phase"] == "report"]
        consumed = df.groupby(keys)[[f"input_consumed_{j}" for j in goods]].sum()
        x_next = rep.groupby(keys)[[f"inv_inputs_{j}" for j in goods]].sum()
        consumer = rep.groupby(keys)[[f"consumer_{j}" for j in goods]].first()
        per_good = {}
        for col in ("cum_output", "inv_output_post", "holding_loss", "cap_overflow"):
            table = rep.pivot_table(
                index=keys, columns="sector", values=col, aggfunc="sum", fill_value=0.0
            )
            per_good[col] = table.reindex(columns=goods, fill_value=0.0).fillna(0.0)
        index = x_next.index
        consumed = consumed.reindex(index).to_numpy(dtype=float)
        consumer = consumer.reindex(index).to_numpy(dtype=float)
        y = per_good["cum_output"].reindex(index).to_numpy(dtype=float)
        s_next = per_good["inv_output_post"].reindex(index).to_numpy(dtype=float)
        hold = per_good["holding_loss"].reindex(index).to_numpy(dtype=float)
        over = per_good["cap_overflow"].reindex(index).to_numpy(dtype=float)
        x_nx = x_next.to_numpy(dtype=float)
        episodes = index.get_level_values("episode").to_numpy()
        periods = index.get_level_values("t_period").to_numpy()
        s_prev = np.empty_like(s_next)
        x_prev = np.empty_like(x_nx)
        for row in range(len(index)):
            if periods[row] == 0:
                s_prev[row], x_prev[row] = starts[int(episodes[row])]
            else:
                if episodes[row - 1] != episodes[row] or periods[row - 1] != periods[row] - 1:
                    raise RuntimeError("ledger part is missing a period")
                s_prev[row], x_prev[row] = s_next[row - 1], x_nx[row - 1]
        lhs = y + s_prev + x_prev
        rhs = s_next + x_nx + consumed + consumer + hold + over
        residual = np.abs(lhs - rhs)
        if residual.size:
            cell["max_conservation_error"] = max(
                cell["max_conservation_error"], float(np.max(residual))
            )
        cell["n_periods"] += len(index)
        cell["report_rows"] += len(rep)

    all_cells: list[dict] = []
    for ci, c in enumerate(configs):
        run_dir = runs_root / hashes[ci]
        run_dir.mkdir(parents=True, exist_ok=True)
        config_flags: set[str] = set()
        for agent_name in AGENTS:
            agent = getattr(heuristic, agent_name)(c)
            env = GosplanEnv(c)
            ledger = Ledger()
            env.attach_ledger(ledger)
            policy_rng = np.random.default_rng(c.tech.seed_policy)  # one seed_policy stream/cell
            ledger_dir = run_dir / "ledger" / agent_name
            if ledger_dir.exists():
                shutil.rmtree(ledger_dir)
            ledger_dir.mkdir(parents=True)
            cell = {
                "config_index": ci,
                "config_hash": hashes[ci],
                "agent": agent_name,
                "n_episodes": 0,
                "episode_seconds": [],
                "n_nonfinite": 0,
                "nonfinite_by_column": {},
                "placeholder_nan": 0,
                "target_failures": 0,
                "target_min": math.inf,
                "target_max": -math.inf,
                "target_min_over_floor": math.inf,
                "target_max_over_cap": -math.inf,
                "stock_failures": 0,
                "stock_min": math.inf,
                "stock_max_over_cap": -math.inf,
                "stock_cap": 0.0,
                "fill_failures": 0,
                "fill_min": math.inf,
                "fill_max": -math.inf,
                "max_conservation_error": 0.0,
                "n_periods": 0,
                "report_rows": 0,
                "step_flags": set(),
                "ledger_parts": 0,
            }
            starts: dict[int, tuple] = {}
            sector = np.asarray(c.supply.sector_of, dtype=int)
            done_episodes = 0
            while done_episodes < n_episodes:
                batch = min(chunk_episodes, n_episodes - done_episodes)
                starts.clear()
                for _ in range(batch):
                    t_start = time.perf_counter()
                    obs, info = env.reset(root_seed, c.tech.seed_policy)
                    agent.reset()
                    s0 = np.bincount(
                        sector,
                        weights=np.asarray(env.state.inv_output, dtype=float),
                        minlength=c.supply.n_sectors,
                    )
                    x0 = np.asarray(env.state.inv_inputs, dtype=float).sum(axis=0)
                    done = False
                    while not done:
                        action = agent.act(obs, env.phase(), policy_rng)
                        obs, _reward, done, info = env.step(action)
                        cell["step_flags"].update(info.flags)
                    cell["episode_seconds"].append(time.perf_counter() - t_start)
                    starts[int(ledger.records[-1].episode)] = (s0, x0)
                    done_episodes += 1
                part = ledger_dir / f"part-{cell['ledger_parts']:05d}.parquet"
                ledger.to_parquet(str(part))
                ledger.records.clear()  # flushed to disk; `ledger.flags` keeps the running counts
                cell["ledger_parts"] += 1
                check_chunk(pd.read_parquet(part), c, starts, cell)
                cell["n_episodes"] = done_episodes
            cell["flags"] = tuple(sorted(set(ledger.flags) | cell["step_flags"]))
            config_flags.update(cell["flags"])
            all_cells.append(cell)
        write_manifest(
            str(run_dir),
            c,
            {
                "git_hash": git_hash,
                "reference_ppo_version": None,
                "estimator_version": None,
                "estimator_backend": None,
                "llm_models": None,
                "solver": None,
                "solver_version": None,
                "solver_optimality_gap": None,
                "bunching_settings": None,
                "flags": sorted(config_flags),
            },
        )

    # Aggregate.
    seconds = [s for cell in all_cells for s in cell["episode_seconds"]]
    padder_cells = [cell for cell in all_cells if cell["agent"] == "Padder"]
    padder_min_fill = min(cell["fill_min"] for cell in padder_cells)
    padder_shortage = all(cell["fill_min"] < 1.0 for cell in padder_cells)
    flags = tuple(sorted({f for cell in all_cells for f in cell["flags"]}))
    max_cons = max(cell["max_conservation_error"] for cell in all_cells)
    n_nonfinite = sum(cell["n_nonfinite"] for cell in all_cells)
    t_fail = sum(cell["target_failures"] for cell in all_cells)
    s_fail = sum(cell["stock_failures"] for cell in all_cells)
    f_fail = sum(cell["fill_failures"] for cell in all_cells)
    passed = (
        max_cons < CONSERVATION_TOL
        and n_nonfinite == 0
        and t_fail == 0
        and s_fail == 0
        and f_fail == 0
        and padder_shortage
    )
    leontief = any(math.isinf(c.supply.input_complementarity) for c in configs)
    total_seconds = time.perf_counter() - harness_start

    # 4. Report.
    def verdict(ok: bool) -> str:
        return "HELD" if ok else "FAILED"

    doc = __doc__ or ""
    prohibition = doc[doc.index("FORBIDDEN - HELD-OUT") : doc.index("Runtime bindings.")].rstrip()
    lines = [
        "# Monte-Carlo sanity report (WO-012, gate G0 artefact)",
        "",
        "This report is an input to gate G0; it is not the gate. G0 is the lead's written",
        "sign-off on this report together with a green frozen suite and the CONTRACT rule 7",
        "diff review of `gosplan/env/` (PLAN section 13).",
        "",
        "## Held-out prohibition this harness ran under",
        "",
        f"`HELD_OUT_PHENOMENON_ROWS = {HELD_OUT_PHENOMENON_ROWS}`. Restated verbatim from the",
        "module docstring of `gosplan/experiments/mc_sanity.py`:",
        "",
        "```",
        prohibition,
        "```",
        "",
        "## Run summary",
        "",
        f"- overall: **{'ALL ASSERTIONS HELD' if passed else 'ASSERTION FAILURE'}**",
        f"- seed_env (shared by every agent and configuration; CRN): {root_seed}",
        f"- seed_policy: {base.tech.seed_policy} (one generator per configuration x agent cell)",
        f"- perturbation generator: `numpy.random.default_rng({root_seed})` (the harness seed)",
        f"- configurations: {len(configs)} (1 baseline + {n_perturbations} SUPPLY perturbations)",
        f"- agents: {', '.join(AGENTS)}; episodes per cell: {n_episodes}",
        f"- theta = inf (Leontief branch) exercised: {'yes' if leontief else 'NO'}",
        f"- max conservation error: {max_cons:.3e} (tolerance {CONSERVATION_TOL:g}) - "
        f"{verdict(max_cons < CONSERVATION_TOL)}",
        f"- non-finite values: {n_nonfinite} - {verdict(n_nonfinite == 0)}",
        f"- T bound failures: {t_fail}; S bound failures: {s_fail}; fill bound failures: {f_fail}",
        f"- Padder minimum fill: {padder_min_fill:.6g}; shortage in every configuration: "
        f"{padder_shortage}",
        f"- mean wall clock per episode: {float(np.mean(seconds)):.4f} s over {len(seconds)} "
        f"episodes; harness total {total_seconds:.1f} s",
        f"- flags raised: {', '.join(flags) if flags else 'none'} "
        f"(`{BOUND_BINDING_FLAG}` is CONTRACT rule 8)",
        f"- git: {git_hash}",
        "",
        "Conventions. Conservation is the per-period, per-good T-U1 identity in the form recorded",
        "in `spec/CHANGELOG.md` 0.1.3 (ambiguity #64), which carries the input stocks explicitly:",
        "`sum y + sum S_prev + sum X_prev = sum S_next + sum X_next + sum consumed + consumer +",
        "sum holding + sum overflow`, with `S_prev`, `X_prev` the previous period's REPORT row",
        "(the reset state in period 0). Only the residual is reported. The non-finite scan covers",
        "every numeric ledger column; in `" + "`, `".join(nan_placeholder_columns) + "` NaN is",
        "the placeholder ruled in AMBIGUITY-007 (their owning functions do not return them) and is",
        "counted separately, while inf there still counts as non-finite. `T` bounds are",
        "`[target_floor_frac * T_0, T_0 * ((1 + g) * (1 + lambda * c_up))**P_max]` per",
        "enterprise; `S` bounds `[0, inventory_cap_mult * cap_i]` on both `inv_output_pre` and",
        "`inv_output_post`; `fill` bounds `FILL_BOUNDS` on every row.",
        "",
    ]
    for ci, c in enumerate(configs):
        s = c.supply
        lines += [
            f"## Configuration {ci}: `{hashes[ci]}`",
            "",
            f"- run directory: `{runs_root / hashes[ci]}/` (manifest.json, ledger/<agent>/)",
            "- kind: " + ("baseline (`p1_default_config()`)" if ci == 0 else "SUPPLY perturbation"),
            f"- yield_sigma: {tuple(round(v, 6) for v in s.yield_sigma)}"
            + (f" (x{draws[ci]['yield_sigma_multiplier']:.6f})" if ci else ""),
            f"- final_demand_share: {s.final_demand_share[0]:.6f} (every sector)",
            f"- holding_loss: {s.holding_loss:.6f}",
            f"- input_complementarity: {s.input_complementarity}",
            "- locked (PERTURBATION_EXCLUDED) at Phase-1 values: "
            + ", ".join(PERTURBATION_EXCLUDED),
            "",
        ]
        for cell in (x for x in all_cells if x["config_index"] == ci):
            ep_s = cell["episode_seconds"]
            nonfin = cell["nonfinite_by_column"]
            lines += [
                f"### Configuration {ci} x {cell['agent']}",
                "",
                f"- episodes: {cell['n_episodes']}; periods: {cell['n_periods']}; ledger parts: "
                f"{cell['ledger_parts']}",
                f"- wall clock per episode: mean {float(np.mean(ep_s)):.4f} s, max "
                f"{float(np.max(ep_s)):.4f} s",
                f"- conservation: max |residual| {cell['max_conservation_error']:.3e} - "
                f"{verdict(cell['max_conservation_error'] < CONSERVATION_TOL)}",
                f"- non-finite: {cell['n_nonfinite']}"
                + (f" {nonfin}" if nonfin else "")
                + f" (AMBIGUITY-007 NaN placeholders: {cell['placeholder_nan']}) - "
                f"{verdict(cell['n_nonfinite'] == 0)}",
                f"- T: min {cell['target_min']:.6g}, max {cell['target_max']:.6g}; min T/T_min "
                f"{cell['target_min_over_floor']:.6g}, max T/T_upper "
                f"{cell['target_max_over_cap']:.6g}; failures {cell['target_failures']} - "
                f"{verdict(cell['target_failures'] == 0)}",
                f"- S: min {cell['stock_min']:.6g}, max S/S_max {cell['stock_max_over_cap']:.6g} "
                f"(S_max {cell['stock_cap']:.6g}); failures {cell['stock_failures']} - "
                f"{verdict(cell['stock_failures'] == 0)}",
                f"- fill: min {cell['fill_min']:.6g}, max {cell['fill_max']:.6g}; failures "
                f"{cell['fill_failures']} - {verdict(cell['fill_failures'] == 0)}",
            ]
            if cell["agent"] == "Padder":
                lines.append(
                    f"- Padder shortage (T-B3 property, fill < 1 somewhere): "
                    f"{cell['fill_min'] < 1.0} - {verdict(cell['fill_min'] < 1.0)}"
                )
            lines += [
                f"- flags: {', '.join(cell['flags']) if cell['flags'] else 'none'}",
                "",
            ]
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "n_configs": len(configs),
        "config_hashes": hashes,
        "n_episodes": n_episodes,
        "max_conservation_error": max_cons,
        "n_nonfinite": n_nonfinite,
        "target_bound_failures": t_fail,
        "stock_bound_failures": s_fail,
        "fill_bound_failures": f_fail,
        "padder_min_fill": padder_min_fill,
        "padder_shortage": padder_shortage,
        "seconds_per_episode": float(np.mean(seconds)),
        "flags": flags,
        "passed": passed,
        "report_path": str(report_path),
    }


def main() -> int:
    """Entry point: run the sweep at `p1_default_config()` and write the report.

    Takes: nothing. The WO-012 card fixes the configuration (`p1_default_config()`), the agents
    (`AGENTS`), the episode count (`N_EPISODES`) and the perturbation count
    (`N_SUPPLY_PERTURBATIONS`); any command-line surface WO-012 adds is built inside this function,
    so the module's import surface stays `pathlib` plus `gosplan`.

    Returns: a process exit code - 0 when every configuration ran and `runs/mc_sanity/report.md` was
    written with every assertion holding, 1 when an assertion failed or the report could not be
    written. The exit code never means "gate G0 passed": G0 is the lead's written sign-off on this
    report together with a green frozen suite and the CONTRACT rule 7 diff review of `gosplan/env/`
    (PLAN section 13).

    Realises: PLAN sections 12.3 (WO-012), 13. Owning WO: **WO-012**.
    """
    from gosplan.config import p1_default_config

    try:
        result = run(p1_default_config())
    except OSError as exc:  # the report (or a run directory) could not be written
        print(f"mc_sanity: could not write artefacts: {exc}")
        return 1
    for key, value in result.items():
        print(f"{key}: {value}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
