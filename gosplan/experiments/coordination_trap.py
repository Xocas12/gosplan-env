"""Labelled study CT - is the LC collapse a coordination trap? (spec/P2_REVISION.md R18).

LC found the C0 economy collapsing to near-zero output at 3M agent-steps (`runs/LC_record.md`) and
offered, untested, a reading: a no-production coordination trap, in which production needs other
enterprises' outputs as inputs, so nobody gains by producing alone. CT tests the two parts of that
reading on the trained populations, by evaluation only (no training):

- (i) **Producing alone does not pay.** Seat 0 switches to the truthful-myopic producer
  (`TruthfulMyopic`: effort to meet the target, truthful report, requests at need, no trade) while
  the other seats keep the learned policy. `d1 = R_dev - R_pop` on seat 0's return.
- (ii) **Everyone producing pays more.** Every seat plays `TruthfulMyopic`. `d2` is the mean
  per-seat return under all-producers minus that under the learned population.

Rules (R18, fixed before any CT evaluation): the median over seeds of each difference, with a 95%
percentile bootstrap CI over seeds (10,000 resamples, generator seed 0). (i) holds iff the CI of
`d1` lies below 0; (ii) holds iff the CI of `d2` lies above 0; "coordination trap" iff both. The
primary populations are LC's 3M seeds 0-9; G3b's 1M seeds 0-9 are evaluated with the same rules as
a contrast. Secondary, descriptive: seat 0's learned effort when the other seats are producers
against when they are the learned population (does the learned policy produce when inputs exist?).

Every evaluation uses the deterministic policies, the measurement seed block and the measurement
window of the exploitability audit (common random numbers across conditions). CT re-evaluates no
gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CT_DIR = Path("runs/coordination_trap")
POPULATIONS = {
    "3M": Path("runs/learner_convergence/runs"),
    "1M": Path("runs/phase2_acceptance_g3b/runs"),
}
PRIMARY = "3M"
CT_SEEDS = 10
CT_EPISODES = 50
SEAT = 0
BOOTSTRAP_RESAMPLES = 10_000


def median_ci(values, n_resamples: int = BOOTSTRAP_RESAMPLES) -> tuple[float, float, float]:
    """Median over seeds and its 95% percentile bootstrap CI (generator seed 0). Pure."""
    x = np.asarray(values, dtype=float)
    if not x.size:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(0)
    boot = np.median(x[rng.integers(0, x.size, size=(n_resamples, x.size))], axis=1)
    return float(np.median(x)), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def verdicts(rows: list[dict]) -> dict[str, object]:
    """R18's two tests on per-seed rows holding `R_pop`, `R_dev`, `W_pop`, `W_tm`. Pure."""
    d1 = [r["R_dev"] - r["R_pop"] for r in rows]
    d2 = [r["W_tm"] - r["W_pop"] for r in rows]
    c1, c2 = median_ci(d1), median_ci(d2)
    unilateral_unprofitable = bool(np.isfinite(c1[2]) and c1[2] < 0.0)
    all_production_pays = bool(np.isfinite(c2[1]) and c2[1] > 0.0)
    return {
        "d1": d1,
        "d2": d2,
        "d1_median_ci": c1,
        "d2_median_ci": c2,
        "unilateral_unprofitable": unilateral_unprofitable,
        "all_production_pays": all_production_pays,
        "trap": unilateral_unprofitable and all_production_pays,
    }


def _mixed_episode_stats(cfg, pop_agent, producer_seats: np.ndarray, n_episodes: int) -> dict:
    """Per-seat mean return over the measurement window, and seat 0's mean PRODUCE effort, with
    `producer_seats` (bool `(N,)`) playing `TruthfulMyopic` and the rest the learned policy."""
    from gosplan.agents.heuristic import TruthfulMyopic
    from gosplan.env.env import GosplanEnv
    from gosplan.experiments._g2 import MEASURE_SEED_OFFSET
    from gosplan.experiments.exploitability import (
        MEASUREMENT_WINDOW_START_PERIOD,
        _deterministic_rows,
    )

    tm = TruthfulMyopic(cfg)
    rng = np.random.default_rng(0)  # unused by TruthfulMyopic, which is deterministic
    env = GosplanEnv(cfg, records=False)
    n = cfg.supply.n_enterprises
    totals = np.zeros(n)
    efforts = []
    for e in range(n_episodes):
        obs, _ = env.reset(int(cfg.tech.seed_env) + MEASURE_SEED_OFFSET + e, cfg.tech.seed_policy)
        done = False
        while not done:
            phase = env.phase()
            action = pop_agent._to_action(_deterministic_rows(pop_agent, obs, phase))
            if producer_seats.any():
                prod = tm.act(obs, phase, rng)
                for name in (
                    "effort",
                    "quality",
                    "invest",
                    "report_ratio",
                    "input_request",
                    "trade_offer",
                ):
                    mine = np.array(getattr(action, name), dtype=float, copy=True)
                    mine[producer_seats] = np.asarray(getattr(prod, name), dtype=float)[
                        producer_seats
                    ]
                    setattr(action, name, mine)
            obs, reward, done, info = env.step(action)
            if info.t_period >= MEASUREMENT_WINDOW_START_PERIOD:
                totals += np.asarray(reward, dtype=float)
                if str(phase).lower().endswith("produce"):
                    efforts.append(float(np.asarray(action.effort)[SEAT]))
    return {"per_seat": (totals / n_episodes).tolist(), "seat0_effort": float(np.mean(efforts))}


def evaluate_population(job: tuple) -> dict[str, object]:
    """All four R18 conditions for one trained population. `job = (cfg, run_dir, ppo_cfg)`."""
    from gosplan.agents.ppo.adapter import IPPO

    cfg, run_dir, ppo_cfg = job
    ckpt = sorted((Path(run_dir) / "checkpoints").glob("*.ckpt*"))[-1]
    pop = IPPO(cfg, ppo_cfg)
    pop.load_checkpoint(ckpt)
    n = cfg.supply.n_enterprises
    none = np.zeros(n, dtype=bool)
    seat0 = none.copy()
    seat0[SEAT] = True
    others = ~seat0
    every = np.ones(n, dtype=bool)
    learned = _mixed_episode_stats(cfg, pop, none, CT_EPISODES)
    deviant = _mixed_episode_stats(cfg, pop, seat0, CT_EPISODES)
    producers = _mixed_episode_stats(cfg, pop, every, CT_EPISODES)
    amid = _mixed_episode_stats(cfg, pop, others, CT_EPISODES)
    return {
        "seed_env": int(cfg.tech.seed_env),
        "R_pop": learned["per_seat"][SEAT],
        "R_dev": deviant["per_seat"][SEAT],
        "W_pop": float(np.mean(learned["per_seat"])),
        "W_tm": float(np.mean(producers["per_seat"])),
        "share_seats_better_all_tm": float(
            np.mean(np.asarray(producers["per_seat"]) > np.asarray(learned["per_seat"]))
        ),
        "effort_learned_amid_learned": learned["seat0_effort"],
        "effort_learned_amid_producers": amid["seat0_effort"],
    }


def render(results: dict[str, list[dict]], tests: dict[str, dict]) -> str:
    def f(v):
        return f"{v:.3f}"

    def ci(t):
        return f"{f(t[0])} [{f(t[1])}, {f(t[2])}]"

    lines = [
        "# Labelled study CT - coordination-trap test (spec/P2_REVISION.md R18)",
        "",
        "Evaluation only (no training): LC's 3M populations (primary) and G3b's 1M populations "
        f"(contrast), seeds 0-9, {CT_EPISODES} episodes per condition on the measurement seed "
        "block, deterministic policies, seat 0. Producer = `TruthfulMyopic`. Pre-registered "
        "before any CT evaluation. CT re-evaluates no gate.",
        "",
    ]
    for budget in (PRIMARY, *[b for b in POPULATIONS if b != PRIMARY]):
        t = tests[budget]
        label = "primary" if budget == PRIMARY else "contrast"
        lines += [
            f"## {budget} populations ({label})",
            "",
            "| seed_env | R_pop (seat 0) | R_dev (seat 0 produces) | W_pop (mean seat) | "
            "W_tm (all produce) | seats better off, all produce | seat-0 effort amid learned | "
            "amid producers |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in results[budget]:
            lines.append(
                f"| {r['seed_env']} | {f(r['R_pop'])} | {f(r['R_dev'])} | {f(r['W_pop'])} | "
                f"{f(r['W_tm'])} | {r['share_seats_better_all_tm']:.2f} | "
                f"{f(r['effort_learned_amid_learned'])} | "
                f"{f(r['effort_learned_amid_producers'])} |"
            )
        lines += [
            "",
            f"- (i) d1 = R_dev - R_pop, median {ci(t['d1_median_ci'])}: producing alone "
            f"{'does NOT pay' if t['unilateral_unprofitable'] else 'is not shown to be unprofitable'}"
            " (R18: CI entirely below 0).",
            f"- (ii) d2 = W_tm - W_pop, median {ci(t['d2_median_ci'])}: all-production "
            f"{'pays more' if t['all_production_pays'] else 'is not shown to pay more'}"
            " (R18: CI entirely above 0).",
            f"- **Coordination trap (both): {'YES' if t['trap'] else 'NO'}.**",
            "",
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    from gosplan.experiments import phase2_acceptance as pa
    from gosplan.experiments.phase1_gate import study_ppo_config

    ppo = study_ppo_config()
    cfgs = [pa.arm_config("C0", s) for s in range(CT_SEEDS)]
    results, tests = {}, {}
    for budget, root in POPULATIONS.items():
        results[budget] = pa._pool(
            evaluate_population, [(c, root / c.hash(), ppo) for c in cfgs], 4
        )
        tests[budget] = verdicts(results[budget])
    CT_DIR.mkdir(parents=True, exist_ok=True)
    (CT_DIR / "result.json").write_text(
        json.dumps({"results": results, "tests": tests}, indent=1), encoding="utf-8"
    )
    (CT_DIR / "report.md").write_text(render(results, tests), encoding="utf-8")
    print(f"CT: 3M trap={tests['3M']['trap']} 1M trap={tests['1M']['trap']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
