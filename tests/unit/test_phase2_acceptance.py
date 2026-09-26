"""WO-031 harness: the R14 pass rules applied to SYNTHETIC per-seed summaries.

Authored by the LEAD. FROZEN BY CONTRACT RULE 2 once landed. No environment is run and no agent is
trained: rows 2, 5, 6 and 7 are held out until the acceptance run itself (PLAN section 4.1), so the
numbers below are invented and test only the aggregation and the pass rules of spec/P2_REVISION.md
R14.
"""

from __future__ import annotations

import math

import numpy as np


def _seed(storm=0.1, infl=1.5, infl_b=1.0, corr=0.3, corr_b=0.0, share=0.02, hidden=0.05, q=0.6):
    return {
        "storming": {"excess": storm},
        "hoarding": {
            "request_inflation": infl,
            "request_inflation_baseline": infl_b,
            "corr_stock_shortfall": corr,
            "corr_stock_shortfall_baseline": corr_b,
        },
        "blat": {"trade_volume_share": share},
        "hidden_reserves": {"hidden_reserves": hidden},
        "quality": {"mean_quality": q},
    }


def _jitter(values, scale=0.01, seed=0):
    rng = np.random.default_rng(seed)
    return values + scale * rng.standard_normal(len(values))


def test_all_rows_appear_on_clear_synthetic_evidence() -> None:
    from gosplan.experiments.phase2_acceptance import heldout_verdicts

    c0 = [_seed(storm=s) for s in _jitter(np.full(30, 0.1))]
    null = [_seed(hidden=0.0) for _ in range(10)]
    qw = [_seed(q=0.95) for _ in range(10)]
    v = heldout_verdicts({"C0": c0, "R7_NULL": null, "R3_QW": qw})
    for row in (
        "row2_storming",
        "row5_hoarding",
        "row6_blat",
        "row7_hidden_reserves",
        "row3_quality",
    ):
        assert v[row]["appears"], row


def test_absent_rows_are_failures_not_passes() -> None:
    from gosplan.experiments.phase2_acceptance import heldout_verdicts

    c0 = [
        _seed(storm=s, infl=1.0, corr=0.0, share=0.0, hidden=h)
        for s, h in zip(_jitter(np.zeros(30)), np.full(30, 0.05), strict=True)
    ]
    null = [_seed(hidden=0.05) for _ in range(10)]  # does not vanish
    v = heldout_verdicts({"C0": c0, "R7_NULL": null, "R3_QW": [_seed() for _ in range(10)]})
    assert not v["row2_storming"]["appears"]
    assert not v["row5_hoarding"]["appears"]
    assert not v["row6_blat"]["appears"]
    assert v["row7_hidden_reserves"]["present"]
    assert not v["row7_hidden_reserves"]["vanishes_under_null"]
    assert not v["row7_hidden_reserves"]["appears"]
    assert not v["row3_quality"]["appears"]


def test_nan_correlations_follow_r14() -> None:
    from gosplan.experiments.phase2_acceptance import heldout_verdicts

    c0 = [_seed(corr=0.3, corr_b=float("nan")) for _ in range(20)] + [
        _seed(corr=float("nan")) for _ in range(5)
    ]
    v = heldout_verdicts({"C0": c0, "R7_NULL": [], "R3_QW": []})
    assert v["row5_hoarding"]["n_seeds_corr_nan"] == 5
    assert abs(v["row5_hoarding"]["corr_excess"][0] - 0.3) < 1e-12  # NaN baseline counted as 0
    assert math.isnan(v["row7_hidden_reserves"]["null"][0])
    assert not v["row7_hidden_reserves"]["appears"]


def test_bootstrap_is_deterministic_and_brackets_the_mean() -> None:
    from gosplan.experiments.phase2_acceptance import bootstrap_mean_ci

    x = _jitter(np.full(30, 1.0), scale=0.1)
    a = bootstrap_mean_ci(x)
    assert a == bootstrap_mean_ci(x)
    assert a[1] < a[0] < a[2]


def test_price_table_reports_sign_change() -> None:
    from gosplan.experiments.phase2_acceptance import price_table

    rows = [{"welfare_mean": 1.0, "val_repriced_mean": [1.1, 0.9, 1.2, 1.0]}]
    out = price_table(rows, w_oracle=1.0, val_oracle_by_vector=[1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(out["specification_gap"], [0.1, -0.1, 0.2, 0.0], atol=1e-12)
    assert out["sign_change"] and out["welfare_ratio"] == 1.0
