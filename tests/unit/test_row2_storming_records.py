"""Row 2 (storming) as computed is not evidence for storming, and everything that reports it says so.

Issue #105. The pre-registered statistic (spec/P2_REVISION.md R13.1) subtracts the truthful-myopic
agent's within-period effort Gini from the learned one under common random numbers. That agent's
effort is a function of its target alone, which moves once a period, so its Gini is 0 and the
excess is the learned Gini. The Gini is also symmetric in step order. These tests pin both facts,
in the committed measurements and in the environment as built, and check that the records, the
final report, the README status table and both docstrings (the implementation's and the spec's)
say what row 2 can and cannot show, and that the claim that under uniform arrival any effort-Gini
excess is behavioural survives nowhere in the code, the spec or the gate modules.

Nothing here computes a held-out row for a learned agent: the learned values are read back from the
committed `p2_measure.json` files, and the environment is driven only by `TruthfulMyopic`.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]

NOT_EVIDENCE = "row 2 as computed is not evidence for storming"

ROW2_RECORDS: dict[str, str] = {
    "phase2_acceptance": "runs/G3_record.md",
    "phase2_acceptance_g3b": "runs/G3b_record.md",
    "learner_convergence": "runs/LC_record.md",
    "entropy_schedule": "runs/ES_record.md",
}
"""Every study whose committed run tree holds row-2 measurements, and the record that reports it."""


def _norm(text: str) -> str:
    """Lower case, `#` dropped, whitespace collapsed: markdown wrapping and the final report's heading
    demotion do not change what a text says."""
    return " ".join(text.replace("#", " ").split()).lower()


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


@pytest.mark.parametrize("study", sorted(ROW2_RECORDS))
def test_committed_row2_baseline_subtracts_nothing_and_the_record_says_so(study: str) -> None:
    files = sorted((ROOT / "runs" / study / "runs").glob("*/p2_measure.json"))
    assert files, f"no committed row-2 measurements under runs/{study}/runs"
    for path in files:
        storming = json.loads(path.read_text(encoding="utf-8"))["storming"]
        assert storming["gini_baseline"] == 0.0, path
        assert storming["excess"] == storming["gini"], path
    assert NOT_EVIDENCE in _norm(_read(ROW2_RECORDS[study])), ROW2_RECORDS[study]


def test_readme_status_and_final_report_carry_the_row2_correction() -> None:
    studies = {p.parts[-4] for p in (ROOT / "runs").glob("*/runs/*/p2_measure.json")}
    assert studies == set(ROW2_RECORDS), "a study with row-2 output needs its record listed here"

    g3_row = next(ln for ln in _read("README.md").splitlines() if ln.startswith("| G3 |"))
    assert NOT_EVIDENCE in _norm(g3_row)

    report = _norm(_read("runs/final_report/report.md"))
    assert NOT_EVIDENCE in report
    for rel in ROW2_RECORDS.values():
        assert _norm(_read(rel)) in report, f"{rel} is not embedded as it stands: regenerate"


def test_storming_docstrings_describe_the_baseline_as_built() -> None:
    from gosplan.agents.heuristic import TruthfulMyopic
    from gosplan.experiments.phase2_acceptance import _rollout, arm_config
    from gosplan.metrics import phenomena

    cfg = arm_config("C0", 0)
    assert cfg.supply.delivery_timing == "stochastic"
    tm = TruthfulMyopic(cfg)
    rng = np.random.default_rng(0)
    ledger = _rollout(lambda o, ph: tm.act(o, ph, rng), cfg, 1, int(cfg.tech.seed_env))
    effort: dict[tuple[int, int, int], set[float]] = {}
    coverage: dict[tuple[int, int, int], set[float]] = {}
    for rec in ledger.records:
        if rec.phase == "produce":
            key = (rec.episode, rec.t_period, rec.enterprise)
            effort.setdefault(key, set()).add(rec.effort)
            coverage.setdefault(key, set()).add(rec.coverage)
    assert effort
    # Inputs on hand move within a period; the baseline's effort does not.
    assert any(len(v) > 1 for v in coverage.values())
    assert all(len(v) == 1 for v in effort.values())

    # No time direction: front-loaded and back-loaded effort score the same.
    gini = phenomena._gini
    assert gini(np.array([1.0, 0, 0, 0])) == gini(np.array([0, 0, 0, 1.0])) == 0.75
    assert gini(np.array([1.0, 1, 0, 0])) == gini(np.array([0, 0, 1, 1.0])) == 0.5

    implementation = phenomena.phenomenon_storming.__doc__ or ""
    for source in (implementation, _spec_docstring("phenomenon_storming")):
        doc = _norm(source)
        assert "absorbs the mechanical concentration" not in doc
        assert "any gini excess is behavioural" not in doc
        assert NOT_EVIDENCE in doc

    # The rationale for the locked arrival setting repeated the claim in the spec, its config
    # mirror, the parameter registry, `produce` and the G3 gate. No copy of it may survive.
    scanned = {p.relative_to(ROOT).as_posix(): p for p in _claim_scan_files()}
    missing = CLAIM_SITES - set(scanned)
    assert not missing, f"the scan no longer reads {sorted(missing)}"
    for rel, path in sorted(scanned.items()):
        text = _norm(path.read_text(encoding="utf-8"))
        for claim in STALE_CLAIMS:
            assert claim not in text, f"{rel} still says {claim!r}"

    from gosplan.params import REGISTRY, by_name

    assert "uniform-random arrival" in _norm(by_name("delivery_timing").notes)
    for spec in REGISTRY:  # the notes as Python joins them, so a split literal cannot hide it
        for claim in STALE_CLAIMS:
            assert claim not in _norm(spec.notes), f"params registry {spec.name!r} says {claim!r}"


STALE_CLAIMS: tuple[str, ...] = (
    "excess is behavioural",
    "excess is behavioral",
    "absorbs the mechanical concentration",
)
"""The claim #105 refutes, in every wording the repository used for it. `spec/CHANGELOG.md` quotes
it as the reason for 2.1.1, so the changelog is not scanned."""

CLAIM_SITES: frozenset[str] = frozenset(
    {
        "spec/spec.py",
        "gosplan/config.py",
        "gosplan/params.py",
        "gosplan/metrics/phenomena.py",
        "gosplan/env/production.py",
        "tests/acceptance/gate_g3_phase2.py",
    }
)
"""Every file that stated the claim before 2.1.1. The scan must read each of them, so a glob that
stops matching fails loudly instead of passing on nothing."""


def _claim_scan_files() -> list[Path]:
    """The code, the spec, the gate modules, the README and the gate and study records."""
    return [
        *ROOT.glob("gosplan/**/*.py"),
        ROOT / "spec" / "spec.py",
        *ROOT.glob("tests/acceptance/**/*.py"),
        ROOT / "README.md",
        *ROOT.glob("runs/*.md"),
    ]


def _spec_docstring(name: str) -> str:
    """A function's docstring in `spec/spec.py`, read without importing the file."""
    tree = ast.parse(_read("spec/spec.py"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_docstring(node) or ""
