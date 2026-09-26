"""WO-037 final report: G4 checklist states and the manifest roll-up (spec/P3_REVISION.md S7).

Authored by the LEAD. FROZEN BY CONTRACT RULE 2 once landed. Uses a temporary directory only.
"""

from __future__ import annotations

import json


def test_checklist_marks_missing_not_run_and_complete(tmp_path) -> None:
    from gosplan.experiments.report import g4_checklist, manifest_rollup, render

    (tmp_path / "runs/contrasts").mkdir(parents=True)
    (tmp_path / "runs/contrasts/report.md").write_text("# Contrasts\n")
    (tmp_path / "runs/llm_study").mkdir(parents=True)
    (tmp_path / "runs/llm_study/report.md").write_text("# LLM ministry study (WO-035) - NOT RUN\n")
    run = tmp_path / "runs/x/abc"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(
        json.dumps({"config_hash": "abc", "spec_version": "2.0.0", "git_hash": "g", "flags": []})
    )
    status = {c["condition"]: c for c in g4_checklist(tmp_path)}
    assert status["contrasts with CIs"]["complete"]
    assert status["LLM study"]["exists"] and not status["LLM study"]["complete"]
    assert not status["estimator-bias curves"]["exists"]
    assert len(manifest_rollup(tmp_path)) == 1
    report, table = render(tmp_path)
    assert "NOT MET" in report and "MISSING" in report and "NOT RUN" in report
    assert "abc" in table
