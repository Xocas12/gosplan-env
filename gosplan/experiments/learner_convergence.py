"""Labelled study LC - learner convergence (spec/P2_REVISION.md R17).

G3b found the C0 populations non-converged under the revised audit (R16): a best responder warm
started from the population gains on 9 of 10 seeds (limitation L4). LC asks one pre-registered
question: **does tripling the population's training budget lower that exploitability?**

Design (R17, fixed before any LC run):
- C0 under spec 2.1.0, seeds 0-9 (`seed_env = 1000 + s`, the G3b seeds), learner
  `phase1_gate.study_ppo_config()`, gate sizing with `total_agent_steps = 3,000,000`. Nothing else
  changes. Training, measurement and the R16 audit are the acceptance harness's own
  (`phase2_acceptance.run`), unchanged.
- The 1M leg is G3b's seeds 0-9, read back from `runs/phase2_acceptance_g3b/result.json`.
- Primary analysis: the seed-paired difference `d_s = expl_3M(s) - expl_1M(s)`; its median with a
  95% percentile bootstrap CI over seeds (10,000 resamples, generator seed 0). "Exploitability
  falls" iff the CI's upper bound is below 0. "Converged at 3M" iff the maximum over seeds is at
  most the R16 threshold (5%).
- Secondary, descriptive only: welfare ratio, row-6 trade share, and the share of sell offers
  posted at REPORT (a deterministic rollout of both budgets' populations).

LC never re-evaluates a gate and never changes G3 or G3b; its outcome is reported whichever way it
falls, and no further budget or learner change follows from it without a new pre-registration.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

LC_DIR = Path("runs/learner_convergence")
G3B_RESULT = Path("runs/phase2_acceptance_g3b/result.json")
G3B_RUN_ROOT = Path("runs/phase2_acceptance_g3b/runs")
LC_BUDGET = 3_000_000
BASE_BUDGET = 1_000_000
LC_SEEDS = 10
THRESHOLD = 0.05
BOOTSTRAP_RESAMPLES = 10_000
OFFER_EPISODES = 5


def paired_exploitability(
    base: list[dict], new: list[dict], n_resamples: int = BOOTSTRAP_RESAMPLES
) -> dict[str, object]:
    """R17's primary analysis. `base` and `new` are exploitability `per_seed` rows (each with
    `seed_env` and `exploitability_ratio`); seeds are paired on `seed_env`. Pure."""
    b = {int(r["seed_env"]): float(r["exploitability_ratio"]) for r in base}
    n = {int(r["seed_env"]): float(r["exploitability_ratio"]) for r in new}
    seeds = sorted(set(b) & set(n))
    x0 = np.array([b[s] for s in seeds])
    x1 = np.array([n[s] for s in seeds])
    d = x1 - x0
    rng = np.random.default_rng(0)
    if len(seeds):
        idx = rng.integers(0, len(seeds), size=(n_resamples, len(seeds)))
        boot = np.median(d[idx], axis=1)
        ci = (float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)))
    else:
        ci = (float("nan"), float("nan"))
    return {
        "seeds": seeds,
        "base": x0.tolist(),
        "new": x1.tolist(),
        "diff": d.tolist(),
        "median_base": float(np.median(x0)) if len(seeds) else float("nan"),
        "median_new": float(np.median(x1)) if len(seeds) else float("nan"),
        "median_diff": float(np.median(d)) if len(seeds) else float("nan"),
        "median_diff_ci": ci,
        "max_new": float(np.max(x1)) if len(seeds) else float("nan"),
        "n_new_above_threshold": int(np.sum(x1 > THRESHOLD)),
        "falls": bool(len(seeds) and ci[1] < 0.0),
        "converged_new": bool(len(seeds) and np.max(x1) <= THRESHOLD),
    }


def offer_profile(job: tuple) -> dict[str, float]:
    """Posted `trade_offer`s at REPORT of one trained population, deterministic policy,
    `OFFER_EPISODES` episodes on the measurement seed block: sell share (offers > 0), buy share
    (offers < 0) and mean offer. `job = (cfg, run_dir, ppo_cfg)`."""
    from gosplan.agents.ppo.adapter import IPPO
    from gosplan.agents.ppo.train import _deterministic_action
    from gosplan.env.env import GosplanEnv
    from gosplan.experiments._g2 import MEASURE_SEED_OFFSET

    cfg, run_dir, ppo_cfg = job
    ckpt = sorted((Path(run_dir) / "checkpoints").glob("*.ckpt*"))[-1]
    agent = IPPO(cfg, ppo_cfg)
    agent.load_checkpoint(ckpt)
    env = GosplanEnv(cfg)
    offers = []
    for e in range(OFFER_EPISODES):
        obs, _ = env.reset(int(cfg.tech.seed_env) + MEASURE_SEED_OFFSET + e, cfg.tech.seed_policy)
        agent.reset()
        done = False
        while not done:
            phase = env.phase()
            action = _deterministic_action(agent, obs, phase)
            if str(phase).lower().endswith("report"):
                offers.append(np.asarray(action.trade_offer, dtype=float))
            obs, _r, done, _i = env.step(action)
    x = np.concatenate([o.ravel() for o in offers]) if offers else np.array([])
    return {
        "sell_share": float(np.mean(x > 0)) if x.size else float("nan"),
        "buy_share": float(np.mean(x < 0)) if x.size else float("nan"),
        "mean_offer": float(np.mean(x)) if x.size else float("nan"),
        "n_offers": int(x.size),
    }


def welfare_ratio(result: dict) -> float:
    return float(result["prices"]["welfare_ratio"])


def render(lc: dict, g3b: dict, paired: dict, offers: dict, harness_report: str) -> str:
    def fmt(v):
        return f"{v:.3f}"

    lines = [
        "# Labelled study LC - learner convergence (spec/P2_REVISION.md R17)",
        "",
        "C0, seeds 0-9, population budget 3M agent-steps against G3b's 1M; R16 audit unchanged "
        "(warm-started best responder, 1M steps, ratio floor 1). Pre-registered before any LC run. "
        "LC re-evaluates no gate; G3 and G3b stand as recorded.",
        "",
        "## Primary: exploitability, seed-paired (3M - 1M)",
        "",
        "| seed_env | 1M (G3b) | 3M (LC) | difference |",
        "|---|---|---|---|",
    ]
    for s, a, b, d in zip(
        paired["seeds"], paired["base"], paired["new"], paired["diff"], strict=True
    ):
        lines.append(f"| {s} | {fmt(a)} | {fmt(b)} | {fmt(d)} |")
    lo, hi = paired["median_diff_ci"]
    lines += [
        "",
        f"- Median exploitability: 1M {fmt(paired['median_base'])}, 3M {fmt(paired['median_new'])}.",
        f"- Median paired difference {fmt(paired['median_diff'])} [95% CI {fmt(lo)}, {fmt(hi)}].",
        f"- **Exploitability falls with budget (R17 rule: CI upper bound < 0): "
        f"{'YES' if paired['falls'] else 'NO'}.**",
        f"- **Converged at 3M (R17 rule: max <= {THRESHOLD}): "
        f"{'YES' if paired['converged_new'] else 'NO'}** - max {fmt(paired['max_new'])}, "
        f"{paired['n_new_above_threshold']} of {len(paired['seeds'])} seeds above the threshold.",
        "",
        "## Secondary (descriptive)",
        "",
        f"- welfare_ratio: 1M {welfare_ratio(g3b):.4f}, 3M {welfare_ratio(lc):.4f}.",
        f"- Row-6 trade volume share (mean over seeds): 1M "
        f"{g3b['verdicts']['row6_blat']['trade_volume_share'][0]:.3e}, 3M "
        f"{lc['verdicts']['row6_blat']['trade_volume_share'][0]:.3e}.",
    ]
    for budget in ("1M", "3M"):
        rows = offers[budget]
        lines.append(
            f"- Posted offers at REPORT, {budget} (seeds 0-9, {OFFER_EPISODES} episodes each): "
            f"sell share {np.mean([r['sell_share'] for r in rows]):.3f}, buy share "
            f"{np.mean([r['buy_share'] for r in rows]):.3f}, mean offer "
            f"{np.mean([r['mean_offer'] for r in rows]):.3f}."
        )
    lines += [
        "",
        "---",
        "",
        "## Acceptance-harness output for the 3M populations",
        "",
        "_The harness prints its gate-condition lines and a G3 verdict line. For LC they are "
        "descriptive only: LC is a labelled study and re-evaluates no gate._",
        "",
    ]
    lines += [("#" + ln) if ln.startswith("#") else ln for ln in harness_report.splitlines()]
    return "\n".join(lines) + "\n"


def main() -> int:
    from gosplan.experiments import phase2_acceptance as pa
    from gosplan.experiments.phase1_gate import GATE_SIZING, study_ppo_config

    pa.BR_TOTAL_AGENT_STEPS = BASE_BUDGET  # R16, unchanged
    sizing = dataclasses.replace(GATE_SIZING, total_agent_steps=LC_BUDGET)
    lc = pa.run(LC_DIR, sizing, {"C0": LC_SEEDS, "R7_NULL": 0, "R3_QW": 0}, {"C0": LC_SEEDS})
    lc = json.loads((LC_DIR / "result.json").read_text(encoding="utf-8"))
    g3b = json.loads(G3B_RESULT.read_text(encoding="utf-8"))
    paired = paired_exploitability(
        g3b["exploitability"]["C0"]["per_seed"], lc["exploitability"]["C0"]["per_seed"]
    )
    ppo = study_ppo_config()
    cfgs = [pa.arm_config("C0", s) for s in range(LC_SEEDS)]
    offers = {
        "1M": pa._pool(offer_profile, [(c, G3B_RUN_ROOT / c.hash(), ppo) for c in cfgs], 4),
        "3M": pa._pool(offer_profile, [(c, LC_DIR / "runs" / c.hash(), ppo) for c in cfgs], 4),
    }
    harness = (LC_DIR / "report.md").read_text(encoding="utf-8")
    (LC_DIR / "harness_report.md").write_text(harness, encoding="utf-8")
    (LC_DIR / "report.md").write_text(render(lc, g3b, paired, offers, harness), encoding="utf-8")
    (LC_DIR / "comparison.json").write_text(
        json.dumps({"paired": paired, "offers": offers}, indent=1), encoding="utf-8"
    )
    print(f"LC: falls={paired['falls']} converged_3M={paired['converged_new']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
