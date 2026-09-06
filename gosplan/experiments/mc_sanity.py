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
    raise NotImplementedError("PLAN section 12.3 (WO-012) - implemented in WO-012")


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
    raise NotImplementedError("PLAN section 12.3 (WO-012) - implemented in WO-012")


if __name__ == "__main__":
    raise SystemExit(main())
