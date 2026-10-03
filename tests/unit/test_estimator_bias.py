"""WO-034 estimator-bias study: truth definition and summary statistics (P3 revision S4).

Authored by the LEAD. FROZEN BY CONTRACT RULE 2 once landed. Synthetic samples only.
"""

from __future__ import annotations

import math

import numpy as np
import pytest


def test_true_excess_mass_known_values() -> None:
    from gosplan.experiments.estimator_bias import true_excess_mass

    rng = np.random.default_rng(0)
    smooth = rng.uniform(0.6, 1.4, 400_000)  # flat: 1/160 of the mass per 0.005 bin
    assert true_excess_mass(smooth, smooth, 0.005) == pytest.approx(0.0)
    # Move 5% of the mass onto the notch: excess = 0.05 / (1/160) = 8 bins, up to sampling noise.
    notched = smooth.copy()
    notched[:20_000] = 1.005
    assert true_excess_mass(notched, smooth, 0.005) == pytest.approx(8.0 - 0.05 * 5, rel=0.03)
    assert math.isnan(true_excess_mass(notched, np.full(10, 0.7), 0.005))


def test_summary_statistics() -> None:
    from gosplan.experiments.estimator_bias import summarise

    rows = [
        {
            "source": "simulation",
            "w": 0.0,
            "rho_cap": 1.2,
            "setting": "s",
            "truth": 1.0,
            "b_hat": b,
            "ci_lo": b - 0.5,
            "ci_hi": b + 0.5,
        }
        for b in (0.8, 1.2, 1.4, 2.0)
    ]
    out = summarise(rows)
    key = "simulation|0.0|1.2|s"
    assert out["bias"][key] == pytest.approx(0.35)
    assert out["rmse"][key] == pytest.approx(math.sqrt((0.04 + 0.04 + 0.16 + 1.0) / 4))
    assert out["ci_coverage"][key] == pytest.approx(0.75)


def test_undefined_estimates_are_counted() -> None:
    from gosplan.experiments.estimator_bias import summarise

    rows = [
        {
            "source": "simulation",
            "w": 0.0,
            "rho_cap": 1.2,
            "setting": "s",
            "truth": 1.0,
            "b_hat": b,
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
        }
        for b in (float("inf"), 2.0, 3.0, 100.0)
    ]
    out = summarise(rows)
    key = "simulation|0.0|1.2|s"
    assert out["n_total"][key] == 4
    assert out["n_defined"][key] == 3
    assert out["median_error"][key] == pytest.approx(2.0)
    assert math.isnan(out["ci_coverage"][key])


def test_grid_and_reuse() -> None:
    from gosplan.experiments.estimator_bias import (
        ESTIMATOR_GRID,
        PREREGISTERED,
        REUSED_ARMS,
        arm_grid,
        seeds_for,
    )
    from gosplan.experiments.phase1_gate import ARMS

    assert len(arm_grid()) == 10 and len(ESTIMATOR_GRID) == 18
    assert PREREGISTERED in ESTIMATOR_GRID
    for key, name in REUSED_ARMS.items():
        inc = ARMS[name]["incentive"]
        assert (inc["notch_width"], inc["overfulfilment_cap"]) == key
        assert seeds_for(key, 30) == 30
    assert seeds_for((0.05, 1.2), 30) == 5
