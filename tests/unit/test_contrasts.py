"""WO-032 contrasts harness: overrides and seed-paired aggregation (spec/P3_REVISION.md S2).

Authored by the LEAD. FROZEN BY CONTRACT RULE 2 once landed. Synthetic numbers only; no training.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_overrides_apply_multiplier_and_validate() -> None:
    from gosplan.config import p2_default_config
    from gosplan.experiments.contrasts import CONTRASTS, _audit_out_of_range, apply_overrides

    c0 = p2_default_config()
    arms = {name: apply_overrides(c0, t) for name, t in CONTRASTS.items()}
    assert arms["C0"] == c0
    assert arms["C_AUDIT"].information.audit_rate == pytest.approx(4 * c0.information.audit_rate)
    assert arms["C_AUDIT"].information.audit_noise == 0.0
    assert arms["C_OGAS"].information.audit_rate == c0.information.audit_rate
    assert arms["C_BOTH"].incentive.objective_metric == "net_output"
    assert arms["C_BOTH"].information.ministry_passthrough == 1.0
    assert set(_audit_out_of_range(arms)) == {"C_AUDIT"}


def test_paired_deltas_and_interaction() -> None:
    from gosplan.experiments.contrasts import aggregate

    rng = np.random.default_rng(0)
    base = rng.uniform(0.5, 0.7, 15)

    def arm(shift):
        return {
            s: {"welfare_ratio": base[s] + shift, "padding_index": 1.0, "specification_gap": 0.0}
            for s in range(15)
        }

    res = aggregate(
        {
            "C0": arm(0.0),
            "C_OGAS": arm(0.1),
            "C_INC": arm(0.05),
            "C_BOTH": arm(0.2),
            "C_AUDIT": arm(0.0),
        }
    )
    assert res["deltas"]["C_OGAS"]["delta"] == pytest.approx(0.1)
    assert res["deltas"]["C_OGAS"]["ci_lo"] == pytest.approx(0.1)  # paired: no seed noise left
    assert res["deltas"]["C_AUDIT"]["delta"] == pytest.approx(0.0)
    assert res["interaction"]["I"] == pytest.approx(0.05)
    assert res["deltas"]["C_BOTH"]["n_pairs"] == 15
    o = res["outcomes"]["C0"]["welfare_ratio"]
    assert o["ci_lo"] <= o["iqm"] <= o["ci_hi"]


def test_seed_outcomes() -> None:
    from gosplan.experiments.contrasts import seed_outcomes

    meas = {"welfare_mean": 0.5, "val_measured_mean": [1.2, 1.0], "val_true_mean": [1.0, 1.0]}
    out = seed_outcomes(meas, w_oracle=1.0, val_oracle=2.0)
    assert out == pytest.approx(
        {"padding_index": 1.2, "welfare_ratio": 0.5, "specification_gap": 0.1}
    )
