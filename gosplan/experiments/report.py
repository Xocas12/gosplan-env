"""Final report (WO-037; spec/P3_REVISION.md S7) - the G4 artefact of PLAN section 13.

This module presents; it computes nothing. It collects the gate records and every study's own
report, rolls up every run manifest under `runs/` (CONTRACT rule 10), and writes
`runs/final_report/report.md` with a G4 checklist that states, for each pass condition, whether its
artefact exists - a missing or not-run study is listed as such, never omitted. Held-out and
non-converged labels are carried through verbatim because the study reports are embedded unedited.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT_DIR = Path("runs/final_report")
REPORT_PATH = OUT_DIR / "report.md"
MANIFEST_TABLE = OUT_DIR / "manifests.md"

SECTIONS: tuple[tuple[str, str], ...] = (
    ("Gate G1 - regime map and Phase-1 values", "runs/G1_decision.md"),
    ("Gate G2 - record", "runs/G2_record.md"),
    ("Phase 1 - DP vs PPO (G2 criterion 1)", "runs/dp_vs_ppo/report.md"),
    ("Phase 1 - labelled study, G2 criteria 2-4", "runs/phase1_gate/report.md"),
    ("Gate G3 - record", "runs/G3_record.md"),
    ("Phase 2 - acceptance run (G3)", "runs/phase2_acceptance/report.md"),
    ("Phase 3 - contrasts (WO-032)", "runs/contrasts/report.md"),
    ("Phase 3 - estimator-bias study (WO-034)", "runs/estimator_bias/report.md"),
    ("Phase 3 - price-vector sensitivity (WO-036)", "runs/price_sensitivity/report.md"),
    ("Phase 3 - LLM ministry study (WO-035)", "runs/llm_study/report.md"),
)
"""Every report the final report embeds, in reading order."""

G4_CONDITIONS: tuple[tuple[str, str, str], ...] = (
    ("contrasts with CIs", "runs/contrasts/report.md", ""),
    ("estimator-bias curves", "runs/estimator_bias/report.md", ""),
    ("LLM study", "runs/llm_study/report.md", "NOT RUN"),
    ("price sensitivity on every headline table", "runs/price_sensitivity/report.md", ""),
)
"""PLAN section 13's G4 pass conditions: (name, artefact, text that marks it as not run)."""

SOBOL_STATEMENT = (
    "The optional Saltelli/Sobol design (WO-033) was not run: PLAN marks it optional and JAX-only, "
    "and at about 46 min per Phase-2 run its 1,500-2,800 runs are out of reach without a JAX "
    "trainer (spec/P3_REVISION.md S3)."
)


def g4_checklist(root: Path = Path(".")) -> list[dict[str, object]]:
    """Per G4 condition: whether its artefact exists and whether it records a completed study."""
    out = []
    for name, rel, not_run_marker in G4_CONDITIONS:
        path = root / rel
        exists = path.exists()
        text = path.read_text(encoding="utf-8") if exists else ""
        not_run = bool(not_run_marker) and not_run_marker in text.splitlines()[0] if text else False
        out.append(
            {
                "condition": name,
                "artefact": rel,
                "exists": exists,
                "complete": bool(exists and not not_run),
            }
        )
    return out


def manifest_rollup(root: Path = Path(".")) -> list[dict[str, object]]:
    """One row per `manifest.json` under `runs/`: path, config hash, spec version, git hash, flags."""
    rows = []
    for path in sorted((root / "runs").rglob("manifest.json")):
        try:
            m = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rows.append(
            {
                "path": str(path.parent.relative_to(root)),
                "config_hash": str(m.get("config_hash", ""))[:12],
                "spec_version": m.get("spec_version", ""),
                "git_hash": str(m.get("git_hash", ""))[:12],
                "flags": ", ".join(f for f in m.get("flags", []) if "=" not in f) or "-",
            }
        )
    return rows


def render(root: Path = Path(".")) -> tuple[str, str]:
    """The final report and the manifest table, as markdown."""
    checklist = g4_checklist(root)
    passed = all(c["complete"] for c in checklist)
    lines = [
        "# gosplan-env - final report (WO-037, gate G4)",
        "",
        "Every section below is the producing study's own report, embedded unedited, so its labels "
        "(held-out, NON-CONVERGED, NOT RUN, reduced designs) survive verbatim. Pre-registrations: "
        "PLAN.md sections 4 and 7, spec/P2_REVISION.md (R1-R14), spec/P3_REVISION.md (S1-S7).",
        "",
        f"## G4 checklist - {'all conditions have a completed artefact' if passed else 'NOT MET'}",
        "",
    ]
    for c in checklist:
        status = "complete" if c["complete"] else ("NOT RUN" if c["exists"] else "MISSING")
        lines.append(f"- {c['condition']}: {status} (`{c['artefact']}`)")
    lines += [
        "",
        f"- Sobol (optional): {SOBOL_STATEMENT}",
        "",
        "G4 is signed off by the human (PLAN section 13); this checklist does not sign it.",
        "",
    ]
    for title, rel in SECTIONS:
        path = root / rel
        lines += ["---", "", f"## {title}", "", f"_Source: `{rel}`_", ""]
        if path.exists():
            body = path.read_text(encoding="utf-8").strip().splitlines()
            # Demote the embedded report's headings by two levels so the outline stays readable.
            lines += [("##" + ln) if ln.startswith("#") else ln for ln in body]
        else:
            lines.append("**MISSING** - this artefact does not exist.")
        lines.append("")
    rows = manifest_rollup(root)
    lines += [
        "---",
        "",
        "## Manifest roll-up (CONTRACT rule 10)",
        "",
        f"{len(rows)} run manifests; the full table is `{MANIFEST_TABLE}`.",
        "",
    ]
    table = ["| run | config hash | spec | git | flags |", "|---|---|---|---|---|"]
    table += [
        f"| {r['path']} | {r['config_hash']} | {r['spec_version']} | {r['git_hash']} | "
        f"{r['flags']} |"
        for r in rows
    ]
    return "\n".join(lines) + "\n", "\n".join(table) + "\n"


def main() -> int:
    report, manifests = render()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    MANIFEST_TABLE.write_text(manifests, encoding="utf-8")
    for c in g4_checklist():
        print(f"{c['condition']}: {'complete' if c['complete'] else 'not complete'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
