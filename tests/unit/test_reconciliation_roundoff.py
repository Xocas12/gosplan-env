"""#110: the reconciliation power curve and the null it is measured on.

On a ledger that reconciles exactly in arithmetic, `ledger_test` ran its t-test on floating-point
round-off, so the size of the S4 power curve was rounding noise and the curve was the same for any
inflation factor. These tests pin R13.6's degenerate case under round-off, what the curve does on
the Phase-1 and C0 ledgers, the committed supplement, and the qualification in the records.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
EPS = float(np.finfo(float).eps)

# Per-good claimed supplies of the size the S4 ledgers carry.
CLAIMED = np.array([1.2060, 1.1592, 1.1986, 1.1592, 1.1362, 0.4903, 0.5726, 0.6880, 1.2612])


def _per_good(claimed: np.ndarray, received: np.ndarray):
    """S4's call shape: one good, unit `io` rows, prices 1."""
    n = claimed.size
    return claimed, received.reshape(n, 1), np.ones((n, 1)), np.ones(n)


def test_110_one_signed_roundoff_takes_the_degenerate_case() -> None:
    from gosplan.metrics._fallback import ledger_test

    # Receipts a few ulps below the claims, all on one side: as a sample, a large t-statistic.
    received = CLAIMED * (1.0 - EPS * np.array([2, 3, 2, 4, 3, 2, 3, 2, 4]))
    res = ledger_test(*_per_good(CLAIMED, received))
    assert np.all(res.residuals > 0.0) and np.max(np.abs(res.residuals)) < 1e-14
    assert (res.statistic, res.p_value) == (0.0, 0.5)
    assert res.n_obs == CLAIMED.size


@pytest.mark.parametrize("factor", [1.2, 1.000001])
def test_110_two_signed_roundoff_is_degenerate_and_a_real_gap_is_not(factor: float) -> None:
    from gosplan.metrics._fallback import ledger_test

    received = CLAIMED * (1.0 + EPS * np.array([3, -2, 1, -4, 2, 2, -1, 3, -3]))
    res = ledger_test(*_per_good(CLAIMED, received))
    assert np.count_nonzero(res.residuals) >= 5 and np.max(np.abs(res.residuals)) < 1e-14
    assert (res.statistic, res.p_value) == (0.0, 0.5)
    # One inflated key among the same round-off: the tolerance must not swallow it, and the
    # statistic is sqrt(k (n - 1) / (n - k)) = 1 for k = 1, whatever the factor.
    claimed = CLAIMED.copy()
    claimed[3] *= factor
    res = ledger_test(*_per_good(claimed, received))
    assert res.statistic == pytest.approx(1.0, rel=1e-6)
    assert res.p_value == pytest.approx(0.5 * math.erfc(res.statistic / math.sqrt(2.0)))


def test_110_fallback_version_moves_with_the_arithmetic() -> None:
    from gosplan.metrics._fallback import FALLBACK_ESTIMATOR_VERSION

    # Every manifest committed before #110 says 0.1.0+fallback (CONTRACT rule 10).
    assert FALLBACK_ESTIMATOR_VERSION != "0.1.0+fallback"
    assert FALLBACK_ESTIMATOR_VERSION.endswith("+fallback")


@pytest.fixture(scope="module")
def phase1_power() -> dict:
    from gosplan.config import p1_default_config
    from gosplan.experiments.estimator_bias import reconciliation_power

    return reconciliation_power(p1_default_config(), (1.2, 1.000001))


def test_110_phase1_null_is_roundoff_and_never_rejected(phase1_power: dict) -> None:
    null = phase1_power["null"]
    assert 0.0 < null["relative_max_abs"] < 1e-14
    assert len(null["statistic"]) == 20
    assert set(null["statistic"]) == {0.0} and set(null["p_value"]) == {0.5}
    assert null["size"] == 0.0 == phase1_power["rejection_rate"]["0.0"]


def test_110_phase1_curve_does_not_depend_on_the_inflation_factor(phase1_power: dict) -> None:
    by = phase1_power["by_inflation"]
    assert by["1.2"] == by["1.000001"] == phase1_power["rejection_rate"]
    # S4's recorded power is unchanged; only the size moves, from 0.05 to 0.
    assert by["1.2"]["0.0"] == 0.0
    assert by["1.2"]["0.05"] == pytest.approx(0.90)
    assert all(by["1.2"][s] == 1.0 for s in ("0.1", "0.2", "0.3", "0.5"))


def test_110_c0_null_is_one_signed_and_every_ledger_is_rejected() -> None:
    from gosplan.experiments.estimator_bias import recon_ledger_config, reconciliation_power

    power = reconciliation_power(recon_ledger_config("C0"), (1.2, 1.000001))
    null = power["null"]
    assert null["relative_max_abs"] > 1e-3 and null["relative_min"] >= -1e-9
    assert null["max_abs_shipment_gap"] < 1e-12  # receipts track shipments; the gap is the claims
    assert null["size"] == 1.0
    assert all(set(c.values()) == {1.0} for c in power["by_inflation"].values())


def test_110_committed_supplement(phase1_power: dict) -> None:
    from gosplan.experiments.estimator_bias import (
        POWER_INFLATIONS,
        RECON_LEDGERS,
        render_reconciliation,
    )
    from gosplan.metrics._fallback import FALLBACK_ESTIMATOR_VERSION, RECON_ZERO_RTOL

    sup = json.loads((ROOT / "runs/estimator_bias/reconciliation.json").read_text("utf-8"))
    md = (ROOT / "runs/estimator_bias/reconciliation.md").read_text("utf-8")
    assert md == render_reconciliation(sup)
    assert sup["estimator_version"] == FALLBACK_ESTIMATOR_VERSION
    assert list(sup["ledgers"]) == [name for name, _ in RECON_LEDGERS]
    # The pre-registered ledger in the committed file is the one recomputed here.
    phase1 = sup["ledgers"]["phase1"]
    assert phase1["null"]["relative_max_abs"] == phase1_power["null"]["relative_max_abs"]
    for f in ("1.2", "1.000001"):
        assert phase1["by_inflation"][f] == phase1_power["by_inflation"][f]
    # The curve is reported against the inflation factor on ledgers whose null is not zero.
    noisy = [d for d in sup["ledgers"].values() if d["null"]["relative_max_abs"] > RECON_ZERO_RTOL]
    assert noisy
    assert all(list(d["by_inflation"]) == [str(f) for f in POWER_INFLATIONS] for d in noisy)
    assert all(math.isfinite(d["null"]["size"]) for d in noisy)


@pytest.mark.parametrize(
    "rel",
    [
        "runs/P3_record.md",
        "spec/P3_REVISION.md",
        "runs/estimator_bias/report.md",
        "runs/final_report/report.md",
    ],
)
def test_110_records_qualify_the_power_figure(rel: str) -> None:
    text = re.sub(r"\s+", " ", (ROOT / rel).read_text("utf-8"))
    hits = [m.start() for m in re.finditer(r"size 0\.05|rejection rate 0\.05 \(size\)", text)]
    assert hits, f"{rel} no longer gives the figure; this check is then obsolete"
    for i in hits:
        # The qualification sits right after the figure, or the figure sits inside it.
        window = text[max(0, i - 200) : i + 1200]
        assert "noiseless Phase-1 ledger" in window and "#110" in window, (rel, text[i : i + 80])
