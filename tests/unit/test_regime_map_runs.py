"""WO-015 smoke test: the regime map runs end to end on a tiny design (PLAN sections 5, 12.3).

Realises: the WO-015 must-pass line "smoke test" of PLAN section 12.3, whose path the card names as
`tests/unit/test_regime_map_runs.py` and assigns to the lead (only the lead writes frozen tests,
CONTRACT rule 2). Authored by the LEAD alongside AMBIGUITY-012. Module under test:
`gosplan/experiments/regime_map.py`.

FROZEN BY CONTRACT RULE 2 once landed. It asserts the *shape* of a run - the four artefacts exist
and the documented return mapping is present and internally consistent - and nothing about which
regime any point falls in: a regime label is a finding, never a test expectation.
"""

from __future__ import annotations

import pytest


@pytest.mark.skeleton
def test_tiny_design_writes_the_four_artefacts_and_the_documented_mapping(
    tmp_path, implemented
) -> None:
    """Two design points (one per `notch_width` level) at the PLAN section 5 grid.

    Assertion: `run(p1_default_config(), out_dir=tmp_path, n_points=2)` returns a mapping holding
    every key its docstring lists; `n_points == 2`; the regime counts sum to 2 and use only the four
    `RegimeLabel` values; `0 <= n_converged <= 2`; the four artefact paths exist on disk under
    `tmp_path`; and `ap_factorisation` is a non-empty string. With two points the candidate list may
    be shorter than `MIN_BUNCHING_CANDIDATES` - that is a reported shortfall of the map, and this
    test does not assert its length.
    """
    from pathlib import Path

    from gosplan.config import p1_default_config
    from gosplan.experiments.regime_map import run

    implemented(run)
    out = run(p1_default_config(), out_dir=tmp_path, n_points=2)
    for key in (
        "n_points",
        "n_converged",
        "regime_counts",
        "bhat_range",
        "edge_hit_points",
        "candidates",
        "artefacts",
        "ap_factorisation",
    ):
        assert key in out, key
    assert out["n_points"] == 2
    counts = out["regime_counts"]
    assert sum(counts.values()) == 2
    assert set(counts) <= {"bunching", "pad_to_cap", "truthful_underfulfilment", "mixed"}
    assert 0 <= out["n_converged"] <= 2
    artefacts = out["artefacts"]
    assert len(artefacts) == 4
    for path in artefacts.values():
        assert Path(path).exists(), path
        assert Path(path).resolve().is_relative_to(Path(tmp_path).resolve()), path
    assert isinstance(out["ap_factorisation"], str) and out["ap_factorisation"]
