"""WO-035 LLM ministry study: classification, measures and guards (spec/P3_REVISION.md S5).

Authored by the LEAD. FROZEN BY CONTRACT RULE 2 once landed. No model is called and no environment
is run: the measures are checked on synthetic decision rows.
"""

from __future__ import annotations

import dataclasses

import pytest


def test_manipulation_check_classification() -> None:
    from gosplan.experiments.llm_study import names_soviet_planning

    assert names_soviet_planning("It resembles Soviet central planning under Gosplan.")
    assert names_soviet_planning("A command economy.")
    assert not names_soviet_planning("A franchise network reporting to head office.")


def _rows(arm, episode, ratio):
    return [
        {
            "model": "m",
            "framing": "neutral",
            "arm": arm,
            "episode": episode,
            "claim": 1.0,
            "target": 1.0,
            "forwarded": ratio,
            "prev_forward": 1.0,
        }
        for _ in range(5)
    ]


def test_tracking_detects_arm_response() -> None:
    from gosplan.experiments.llm_study import tracking_measures

    decisions = []
    for e in range(10):
        decisions += _rows("baseline", e, 1.10)
        decisions += _rows("padding_dominated", e, 1.00)
    checks = [
        {"model": "m", "framing": "neutral", "arm": a, "episode": e, "named_soviet": e < 5}
        for a in ("baseline", "padding_dominated")
        for e in range(10)
    ]
    out = tracking_measures(decisions, checks)
    p = out["tracking"]["m|neutral"]["padding_response"]
    assert p["diff"] == pytest.approx(-0.10)
    assert p["ci_hi"] < 0
    assert out["manipulation_check"]["share_named_soviet"]["m|neutral"] == pytest.approx(0.5)
    assert "m|neutral|named=True" in out["manipulation_check"]["stratified"]


def test_transparent_ministry_is_rejected_and_not_run_report() -> None:
    from gosplan.config import p2_default_config
    from gosplan.experiments.llm_study import arm_config, not_run_report

    cfg = p2_default_config()
    assert arm_config(cfg, "padding_dominated").information.audit_rate == 1.0
    open_cfg = dataclasses.replace(
        cfg, information=dataclasses.replace(cfg.information, ministry_passthrough=1.0)
    )
    with pytest.raises(ValueError):
        arm_config(open_cfg, "baseline")
    assert "NOT RUN" in not_run_report(("a", "b"))
