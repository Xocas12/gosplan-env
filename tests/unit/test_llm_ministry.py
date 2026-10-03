"""WO-026 LLM ministry adapter (PLAN section 7.4; spec/P2_REVISION.md R10.5).

Authored by the LEAD. FROZEN BY CONTRACT RULE 2 once landed. No real model is called: a scripted
client returns canned completions, so these tests check the adapter (rendering, strict parsing,
retry, fallback, logging) and nothing about any model.
"""

from __future__ import annotations

import json

import numpy as np
import pytest


class _Scripted:
    """A client that replays completions in order and records every prompt it was sent."""

    def __init__(self, *completions):
        self.completions = list(completions)
        self.prompts: list[str] = []

    def complete(self, prompt, temperature=None):
        self.prompts.append(prompt)
        return self.completions.pop(0)


def _view(claims=(1.0, 0.9), targets=(1.0, 1.0), prev=(1.0, 1.0)):
    from gosplan.env.ministry import MinistryView

    return MinistryView(
        ministry_id=2,
        enterprise_ids=np.array([4, 5]),
        claims=np.asarray(claims, dtype=float),
        targets=np.asarray(targets, dtype=float),
        prev_forward=np.asarray(prev, dtype=float),
        passthrough=0.75,
        t_period=7,
    )


def _ministry(tmp_path, client, framing="neutral"):
    from gosplan.agents.llm_ministry import LLMMinistry, LLMMinistryConfig

    cfg = LLMMinistryConfig(
        model_id="test/model",
        model_version="2026-01-01",
        framing=framing,
        payoff_arm="baseline",
        log_dir=tmp_path,
    )
    return LLMMinistry(cfg=cfg, client=client)


def _log(tmp_path):
    lines = (tmp_path / "llm_exchanges.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


@pytest.mark.parametrize(
    "text",
    [
        'Sure! {"forwarded": [1, 2], "justification": "x"}',
        '{"forwarded": [1, 2]}',
        '{"forwarded": [1, 2], "justification": "x", "extra": 1}',
        '{"forwarded": [1], "justification": "x"}',
        '{"forwarded": [1, -2], "justification": "x"}',
        '{"forwarded": [1, NaN], "justification": "x"}',
        '{"forwarded": [1, true], "justification": "x"}',
        '{"forwarded": [1, "2"], "justification": "x"}',
        '{"forwarded": [1, 2], "justification": 3}',
        "[1, 2]",
    ],
)
def test_parse_is_strict(text) -> None:
    from gosplan.agents.llm_ministry import parse_forward_response

    with pytest.raises(ValueError):
        parse_forward_response(text, 2)


def test_parse_accepts_a_clean_reply() -> None:
    from gosplan.agents.llm_ministry import parse_forward_response

    values, why = parse_forward_response(' {"forwarded": [1, 0.5], "justification": "ok"}\n', 2)
    np.testing.assert_array_equal(values, [1.0, 0.5])
    assert why == "ok"


def test_rendering_is_deterministic_and_framings_share_the_numbers() -> None:
    from gosplan.agents.llm_ministry import render_ministry_view

    view = _view(claims=(1.234567, 0.5))
    neutral = render_ministry_view(view, "neutral")
    assert neutral == render_ministry_view(_view(claims=(1.234567, 0.5)), "neutral")
    historical = render_ministry_view(view, "historical")
    assert neutral != historical
    for text in (neutral, historical):
        assert "1.234567" in text and "0.5" in text
        assert '"period": 7' in text
        assert "passthrough" not in text and "0.75" not in text
    assert "Gosplan" in historical and "Gosplan" not in neutral


def test_valid_reply_is_returned_and_logged(tmp_path) -> None:
    client = _Scripted('{"forwarded": [1.1, 0.95], "justification": "because"}')
    ministry = _ministry(tmp_path, client)
    out = ministry.forward(_view(), None)
    np.testing.assert_array_equal(out, [1.1, 0.95])
    assert ministry.n_fallbacks == 0
    (record,) = _log(tmp_path)
    assert record["attempt"] == 0 and record["fallback"] is False
    assert record["model_id"] == "test/model" and record["model_version"] == "2026-01-01"
    assert record["completion"].startswith('{"forwarded"')
    assert record["prompt"] == client.prompts[0]


def test_one_retry_with_the_identical_prompt(tmp_path) -> None:
    client = _Scripted("not json", '{"forwarded": [1, 1], "justification": "retry"}')
    ministry = _ministry(tmp_path, client)
    np.testing.assert_array_equal(ministry.forward(_view(), None), [1.0, 1.0])
    assert client.prompts[0] == client.prompts[1]
    records = _log(tmp_path)
    assert [r["attempt"] for r in records] == [0, 1]
    assert "parse_error" in records[0] and records[1]["fallback"] is False


def test_fallback_forwards_claims_unchanged_and_is_counted(tmp_path) -> None:
    client = _Scripted("nope", "still nope")
    ministry = _ministry(tmp_path, client)
    out = ministry.forward(_view(claims=(0.8, 1.3), prev=(2.0, 2.0)), None)
    np.testing.assert_array_equal(out, [0.8, 1.3])
    assert ministry.n_fallbacks == 1
    records = _log(tmp_path)
    assert len(records) == 3
    assert records[-1]["fallback"] is True and records[-1]["forwarded"] == [0.8, 1.3]


def test_reset_advances_the_episode_in_the_log(tmp_path) -> None:
    client = _Scripted(*['{"forwarded": [1, 1], "justification": ""}'] * 2)
    ministry = _ministry(tmp_path, client)
    ministry.forward(_view(), None)
    ministry.reset()
    ministry.forward(_view(), None)
    assert [r["episode"] for r in _log(tmp_path)] == [0, 1]


def test_manipulation_check_is_logged_and_names_no_answer(tmp_path) -> None:
    from gosplan.agents.llm_ministry import (
        FRAMINGS,
        manipulation_check_prompt,
        run_manipulation_check,
    )

    for framing in FRAMINGS:
        text = manipulation_check_prompt(framing).lower()
        for word in ("soviet", "gosplan", "ussr", "communis"):
            assert word not in text
    client = _Scripted(("A franchise network.", {"output_tokens": 5}))
    cfg = _ministry(tmp_path, client).cfg
    assert run_manipulation_check(client, cfg, 3) == "A franchise network."
    (record,) = _log(tmp_path)
    assert record["kind"] == "manipulation_check" and record["episode"] == 3
    assert record["usage"] == {"output_tokens": 5}
