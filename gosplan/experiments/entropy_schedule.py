"""Labelled study ES - training budget versus exploration schedule (spec/P2_REVISION.md R19).

LC tripled the population budget (1M -> 3M agent-steps) and found the C0 economy collapsing into a
coordination trap (`runs/LC_record.md`, `runs/CT_record.md`). Because the entropy bonus anneals
over the whole run, LC's 3M runs also explored for longer: at the 1M-step mark their coefficient
was 0.0070, against 0.0010 for a 1M run. ES separates the two.

Design (R19, fixed before any ES run): C0 under spec 2.1.0, seeds 0-9, the study learner, 3M
agent-steps, with the entropy bonus annealed 0.01 -> 0.001 over the first 1,000 updates (1M
agent-steps, exactly the 1M runs' schedule) and held at 0.001 afterwards. Everything else is LC's,
including the R16 audit. Training, measurement and the audit are `phase2_acceptance.run`.

Primary (R19): collapse under the matched schedule. The statistic is the median over seeds of the
final measured mean effort, with a 95% percentile bootstrap CI over seeds (10,000 resamples,
generator seed 0); "the collapse is reproduced" iff the CI's upper bound is below 0.05 (LC's 3M
median was 0.019; the 1M runs' was 0.09-0.2). Secondary, descriptive: welfare ratio, the R16
exploitability, the R18 coordination-trap test on the ES populations, and the training
trajectories of all three schedules side by side.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

ES_DIR = Path("runs/entropy_schedule")
ES_BUDGET = 3_000_000
ANNEAL_UPDATES = 1_000
ES_SEEDS = 10
COLLAPSE_EFFORT = 0.05
LC_RESULT = Path("runs/learner_convergence/result.json")


def primary(efforts: list[float]) -> dict[str, object]:
    """R19's primary rule on the per-seed final measured mean effort. Pure."""
    from gosplan.experiments.coordination_trap import median_ci

    m, lo, hi = median_ci(efforts)
    return {
        "efforts": [float(e) for e in efforts],
        "median_effort_ci": (m, lo, hi),
        "collapse_reproduced": bool(np.isfinite(hi) and hi < COLLAPSE_EFFORT),
    }


def es_ppo_config():
    from gosplan.experiments.phase1_gate import study_ppo_config

    return dataclasses.replace(study_ppo_config(), entropy_anneal_updates=ANNEAL_UPDATES)


def render(prim, es, lc, paired, trap, traj, harness_report: str) -> str:
    def f(v):
        return f"{v:.3f}"

    m, lo, hi = prim["median_effort_ci"]
    lines = [
        "# Labelled study ES - budget versus exploration schedule (spec/P2_REVISION.md R19)",
        "",
        "C0, seeds 0-9, 3M agent-steps with the entropy bonus annealed over the first 1M steps "
        "(the 1M runs' schedule) and held at 0.001 after; R16 audit unchanged. Pre-registered "
        "before any ES run. ES re-evaluates no gate.",
        "",
        "## Primary: does the collapse reproduce under the matched schedule?",
        "",
        "| seed | final mean effort |",
        "|---|---|",
    ]
    lines += [f"| {s} | {f(e)} |" for s, e in enumerate(prim["efforts"])]
    lines += [
        "",
        f"- Median final effort {f(m)} [95% CI {f(lo)}, {f(hi)}].",
        f"- **Collapse reproduced (R19 rule: CI upper bound < {COLLAPSE_EFFORT}): "
        f"{'YES' if prim['collapse_reproduced'] else 'NO'}.**",
        "",
        "## Secondary (descriptive)",
        "",
        f"- welfare_ratio: ES {float(es['prices']['welfare_ratio']):.4f}; LC 3M "
        f"{float(lc['prices']['welfare_ratio']):.4f}.",
        f"- R16 exploitability, ES: median {f(paired['median_new'])}, max {f(paired['max_new'])} "
        f"({paired['n_new_above_threshold']} of {len(paired['seeds'])} above 5%); LC 3M median "
        f"{f(paired['median_base'])}.",
        f"- R18 coordination-trap test on the ES populations: (i) median d1 "
        f"{f(trap['d1_median_ci'][0])} [{f(trap['d1_median_ci'][1])}, "
        f"{f(trap['d1_median_ci'][2])}]; (ii) median d2 {f(trap['d2_median_ci'][0])} "
        f"[{f(trap['d2_median_ci'][1])}, {f(trap['d2_median_ci'][2])}]; trap: "
        f"{'YES' if trap['trap'] else 'NO'}.",
        "",
        "### Training trajectories (median over seeds of the periodic evaluation)",
        "",
    ]
    for name, t in traj.items():
        lines += [
            f"**{name}** - k-steps {t['k_steps']}",
            "",
            f"- entropy coef: {[round(c, 4) for c in t['entropy_coef']]}",
            f"- median effort: {[round(e, 3) for e in t['median_effort']]}",
            f"- median eval return: {[round(r, 2) for r in t['median_return']]}",
            "",
        ]
    lines += [
        "---",
        "",
        "## Acceptance-harness output for the ES populations",
        "",
        "_The harness prints its gate-condition lines and a G3 verdict line. For ES they are "
        "descriptive only._",
        "",
    ]
    lines += [("#" + ln) if ln.startswith("#") else ln for ln in harness_report.splitlines()]
    return "\n".join(lines) + "\n"


def _logs(root: Path) -> list[list[dict]]:
    from gosplan.experiments import phase2_acceptance as pa

    out = []
    for s in range(ES_SEEDS):
        path = root / pa.arm_config("C0", s).hash() / "train_log.jsonl"
        out.append(
            [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        )
    return out


def main() -> int:
    from gosplan.experiments import coordination_trap as ct
    from gosplan.experiments import learner_convergence as lcm
    from gosplan.experiments import phase2_acceptance as pa
    from gosplan.experiments.phase1_gate import GATE_SIZING

    pa.BR_TOTAL_AGENT_STEPS = lcm.BASE_BUDGET  # R16, unchanged
    sizing = dataclasses.replace(GATE_SIZING, total_agent_steps=ES_BUDGET)
    ppo = es_ppo_config()
    pa.run(
        ES_DIR, sizing, {"C0": ES_SEEDS, "R7_NULL": 0, "R3_QW": 0}, {"C0": ES_SEEDS}, ppo_cfg=ppo
    )
    es = json.loads((ES_DIR / "result.json").read_text(encoding="utf-8"))
    lc = json.loads(LC_RESULT.read_text(encoding="utf-8"))
    rows = sorted(es["per_arm"]["C0"], key=lambda r: r["seed_index"])
    prim = primary([r["metrics"]["mean_effort"] for r in rows])
    paired = lcm.paired_exploitability(
        lc["exploitability"]["C0"]["per_seed"], es["exploitability"]["C0"]["per_seed"]
    )
    cfgs = [pa.arm_config("C0", s) for s in range(ES_SEEDS)]
    trap_rows = pa._pool(
        ct.evaluate_population, [(c, ES_DIR / "runs" / c.hash(), ppo) for c in cfgs], 4
    )
    trap = ct.verdicts(trap_rows)
    traj = lcm.trajectory_table(
        {
            "1M (G3b)": _logs(lcm.G3B_RUN_ROOT),
            "3M, stretched schedule (LC)": _logs(lcm.LC_DIR / "runs"),
            "3M, matched schedule (ES)": _logs(ES_DIR / "runs"),
        }
    )
    harness = (ES_DIR / "report.md").read_text(encoding="utf-8")
    (ES_DIR / "harness_report.md").write_text(harness, encoding="utf-8")
    (ES_DIR / "report.md").write_text(
        render(prim, es, lc, paired, trap, traj, harness), encoding="utf-8"
    )
    (ES_DIR / "comparison.json").write_text(
        json.dumps(
            {
                "primary": prim,
                "paired_vs_lc": paired,
                "trap": trap,
                "trap_rows": trap_rows,
                "trajectories": traj,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"ES: collapse_reproduced={prim['collapse_reproduced']} trap={trap['trap']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
