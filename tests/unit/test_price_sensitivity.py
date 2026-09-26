"""WO-036 price sensitivity: row metrics and sign-change flag (spec/P3_REVISION.md S6).

Authored by the LEAD. FROZEN BY CONTRACT RULE 2 once landed. Synthetic inputs only.
"""

from __future__ import annotations

import pytest


def test_row_metrics_and_sign_change() -> None:
    from gosplan.experiments.price_sensitivity import (
        PRICE_PERTURBATION_SEEDS,
        _sign_change,
        row_metrics,
    )

    meas = {"welfare_mean": 0.5, "val_measured_mean": [1.2, 0.8], "val_true_mean": [1.0, 1.0]}
    base = row_metrics(meas, 0, w_oracle=1.0, val_oracle=2.0)
    moved = row_metrics(meas, 1, w_oracle=1.0, val_oracle=2.0)
    assert base == pytest.approx(
        {"padding_index": 1.2, "welfare_ratio": 0.5, "specification_gap": 0.1}
    )
    assert moved["specification_gap"] == pytest.approx(-0.1)
    assert _sign_change([base["specification_gap"], moved["specification_gap"]])
    assert not _sign_change([0.1, 0.2, float("nan")])
    assert PRICE_PERTURBATION_SEEDS == (11, 12, 13)
