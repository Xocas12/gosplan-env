"""Labelled study ES (spec/P2_REVISION.md R19): the pre-registered primary rule."""

from __future__ import annotations


def test_collapse_rule() -> None:
    from gosplan.experiments.entropy_schedule import primary

    assert primary([0.02, 0.018, 0.025, 0.01, 0.03, 0.02, 0.015, 0.022, 0.019, 0.021])[
        "collapse_reproduced"
    ]
    assert not primary([0.2, 0.15, 0.3, 0.1, 0.25, 0.2, 0.18, 0.22, 0.19, 0.21])[
        "collapse_reproduced"
    ]
    # One clear producer among collapsed seeds still bounds the median below the threshold.
    assert primary([0.02] * 9 + [0.6])["collapse_reproduced"]


def test_es_learner_differs_only_in_the_anneal_horizon() -> None:
    import dataclasses

    from gosplan.experiments.entropy_schedule import ANNEAL_UPDATES, es_ppo_config
    from gosplan.experiments.phase1_gate import study_ppo_config

    a, b = dataclasses.asdict(es_ppo_config()), dataclasses.asdict(study_ppo_config())
    assert a.pop("entropy_anneal_updates") == ANNEAL_UPDATES
    assert b.pop("entropy_anneal_updates") is None
    assert a == b
