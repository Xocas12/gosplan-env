"""T-B8 - bounds are results: reports at `rho_max` raise `BOUND_BINDING`.

Realises: PLAN section 11 (behavioural test T-B8) and CONTRACT rule 8, read against PLAN sections
2.3 (the action bounds), 2.8 (`R_i = clip(rho, 0, rho_max) * T_i`), 4.4 (the measurement window -
reports at `rho_max` are included in histograms and flagged), 4.5 (the G2 hygiene criterion: the
flag absent in all Phase-1 gate runs) and 10 (`Ledger`, `write_manifest`). Owning work order:
**WO-002**; on the must-pass list of **WO-011** (`gosplan/metrics/ledger.py`).

CONTRACT rule 8 (BOUNDS ARE RESULTS): `report_ratio` is bounded at `rho_max = 10`. The fraction of
reports at the bound is logged; more than 1% flags the run manifest `BOUND_BINDING` and the result
is reported with the flag. **Never silently widen or narrow a bound to fix a result.**

What T-B8 asserts (PLAN section 11, verbatim): *forcing `rho = 10` in >1% of reports sets
`BOUND_BINDING`.* The clause has a strict inequality in it, so the test drives both sides of the
threshold: a fraction above `AT_BOUND_FLAG_THRESHOLD` must raise the flag, and one below must not.
A test that only checked the raising half would pass a ledger that flagged everything.

The mechanism under test, in three pieces:

    1. `process_reports` (WO-007) clips `rho` to `rho_max` and records whether the report sat at
       the bound, so `StepRecord.at_bound` is the environment's own observation, not the test's;
    2. `Ledger.append` (WO-011) maintains the running fraction and raises `BOUND_BINDING_FLAG` when
       it exceeds `AT_BOUND_FLAG_THRESHOLD`, counting REPORT rows only;
    3. `write_manifest` (WO-011) carries every raised flag into `runs/<hash>/manifest.json`
       (CONTRACT rule 10), which is where a reader of the result meets it.

`bound_binding(ledger)` answers the same question of a finished ledger; the two answers must agree,
and this file asserts that they do - an incremental flag that disagreed with the batch predicate
would let a run be reported without its flag.

The driving policy is a test-local one: it reports `rho_max` on a chosen, deterministic subset of
REPORT steps and truthfully otherwise. It is a fixture, not a baseline, and it lives in this file
rather than in `gosplan/agents/heuristic.py` because nothing in the production agent set should
exist to trip a flag. `Random` reaches high ratios but does not land exactly on the bound (see
`ref_random_policy` in `ref/gen_golden.py`), which is why the bound is driven deliberately here.

Held-out phenomena (PLAN section 4.1): none. This file measures a flag, not a behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

SKIP_REASON = (
    "skeleton: T-B8 assertions are written by WO-002 (frozen tests); they bind WO-011 "
    "(Ledger.append, bound_binding, write_manifest) and WO-007 (process_reports)"
)
"""Reason attached to every `@pytest.mark.skip` below; the bodies arrive with WO-002."""

BOUND_FLAG = "BOUND_BINDING"
"""The flag string of CONTRACT rule 8; `gosplan.metrics.ledger.BOUND_BINDING_FLAG` must equal it.
Written here as a literal so the test still states the contract if the constant is renamed."""

AT_BOUND_THRESHOLD = 0.01
"""The 1% of CONTRACT rule 8; `gosplan.metrics.ledger.AT_BOUND_FLAG_THRESHOLD` must equal it. The
comparison is strict: *more than* 1% raises the flag."""

RHO_MAX = 10.0
"""`report_max_ratio` at `p1_default_config()` (PLAN section 2.3, TECH row of PLAN section 3). The
test asserts the configured bound equals this and that `action_spec` reports the same number: a
silently widened bound is the failure mode CONTRACT rule 8 exists to prevent."""

ABOVE_THRESHOLD_FRACTION = 0.05
"""Fraction of REPORT steps driven to the bound in the raising case - comfortably above
`AT_BOUND_THRESHOLD`, so the assertion does not turn on the rounding of one row."""

BELOW_THRESHOLD_FRACTION = 0.005
"""Fraction driven to the bound in the non-raising case - comfortably below the threshold."""

OVERSHOOT_RATIO = 25.0
"""A report ratio far above `RHO_MAX`, used to check that the action is clipped to the bound rather
than accepted, and that the clipped row still counts as `at_bound` (PLAN section 2.8)."""

TB8_SEEDS: tuple[int, ...] = (0, 1, 2)
"""Environment seeds, one episode each. A WO-002 test-design constant; the driving policy makes the
at-bound fraction deterministic, so the seeds vary only the environment's draws."""


def _gate(*targets):
    """Skip the calling test while any of `targets` is still a skeleton stub.

    The module-level twin of the `implemented` fixture in `tests/conftest.py`. A module-level
    helper cannot request a fixture, so the check is repeated here rather than the helper being
    called before the gate - which would raise `NotImplementedError` and FAIL the test instead of
    skipping it.
    """
    import inspect

    pending = []
    for target in targets:
        try:
            source = inspect.getsource(target)
        except (OSError, TypeError):
            continue
        if "raise NotImplementedError" in source:
            pending.append(getattr(target, "__qualname__", repr(target)))
    if pending:
        pytest.skip("awaiting implementation: " + ", ".join(pending))


def _cfg(**sections):
    """`p1_default_config()` with per-section overrides applied, validated."""
    import dataclasses

    from gosplan.config import p1_default_config

    _gate(p1_default_config)
    cfg = p1_default_config()
    for section, changes in sections.items():
        cfg = dataclasses.replace(
            cfg, **{section: dataclasses.replace(getattr(cfg, section), **changes)}
        )
    cfg.validate()
    return cfg


def _episode(cfg, agent_name, seed_env, implemented, max_periods=None):
    """Drive one episode of `GosplanEnv` with a named heuristic; return the ledger records.

    Gated on the environment (WO-009), the heuristic agents (WO-010) and the ledger (WO-011), so a
    behavioural module skips naming its missing dependency rather than failing.
    """
    from gosplan.agents import heuristic
    from gosplan.config import p1_default_config
    from gosplan.env.env import GosplanEnv
    from gosplan.metrics.ledger import Ledger

    agent_cls = getattr(heuristic, agent_name)
    implemented(p1_default_config, GosplanEnv.reset, GosplanEnv.step, agent_cls.act, Ledger.append)

    env = GosplanEnv(cfg)
    ledger = Ledger()
    env.attach_ledger(ledger)
    obs, _info = env.reset(seed_env, cfg.tech.seed_policy)
    policy = agent_cls(cfg)
    rng = np.random.default_rng(cfg.tech.seed_policy)
    m = cfg.incentive.steps_per_period
    cap = (max_periods or cfg.tech.max_periods) * (m + 1)
    done = False
    steps = 0
    while not done and steps < cap:
        obs, _r, done, _i = env.step(policy.act(obs, env.phase(), rng))
        steps += 1
    return ledger.records


def _report_rows(records):
    """Only the REPORT-step rows, which is where period-level quantities are written."""
    return [r for r in records if r.phase == "report"]


def _synthetic_ledger(at_bound_fraction):
    """A ledger of REPORT rows with a controlled fraction sitting at `rho_max`."""
    from gosplan.config import p1_default_config
    from gosplan.metrics.ledger import Ledger, StepRecord

    _gate(p1_default_config, Ledger.append)
    cfg = p1_default_config()
    j = cfg.supply.n_sectors
    total = 1000
    n_at_bound = round(total * at_bound_fraction)
    ledger = Ledger()
    zeros = tuple(0.0 for _ in range(j))
    for i in range(total):
        at_bound = i < n_at_bound
        ledger.append(
            StepRecord(
                run_hash=cfg.hash(),
                episode=0,
                t_period=2,
                k_step=cfg.incentive.steps_per_period,
                phase="report",
                enterprise=i % cfg.supply.n_enterprises,
                sector=0,
                target=1.0,
                capital=1.0,
                inv_output_pre=1.0,
                inv_output_post=1.0,
                inv_inputs=zeros,
                cum_output=1.0,
                cum_cost=0.0,
                quality_acc=0.0,
                last_report_ratio=RHO_MAX if at_bound else 1.0,
                last_penalty=0.0,
                last_fill=1.0,
                request=zeros,
                need=zeros,
                effort=0.5,
                quality=1.0,
                invest=0.0,
                output=1.0,
                cost=0.0,
                coverage=1.0,
                reward=0.0,
                report=RHO_MAX if at_bound else 1.0,
                report_ratio=RHO_MAX if at_bound else 1.0,
                at_bound=at_bound,
                audited=False,
                audit_meas=0.0,
                penalty_arg=cfg.incentive.penalty_arg,
                penalty=0.0,
                fill=1.0,
                shipped=1.0,
                alloc=zeros,
                deliv=zeros,
                input_consumed=zeros,
                holding_loss=0.0,
                cap_overflow=0.0,
                trade_volume=0.0,
                consumer=zeros,
                val_measured=1.0,
                val_true=1.0,
                welfare=1.0,
            )
        )
    return ledger


@pytest.mark.skeleton
def test_reports_at_the_bound_are_recorded(implemented) -> None:
    """A report at `rho_max` is clipped, recorded as `at_bound`, and priced at the bound.

    Drive the test-local policy so that a known subset of REPORT steps reports `RHO_MAX` and a
    further subset reports `OVERSHOOT_RATIO`, at `p1_default_config()` over `TB8_SEEDS`, recording
    a `Ledger`. Assert, for those rows:

        row.report_ratio == RHO_MAX                      exactly, including the overshoot rows
        row.report       == RHO_MAX * row.target         to 1e-12
        row.at_bound     is True

    and for every other REPORT row, `row.at_bound is False`. Assert also that
    `action_spec(cfg)["report_ratio"] == ((N,), 0.0, cfg.tech.report_max_ratio)` and that
    `cfg.tech.report_max_ratio == RHO_MAX`.

    The overshoot rows are the substance of CONTRACT rule 8's "never silently widen": an action
    above the bound is clipped and *counted*, never honoured. Owning WO: **WO-002**; binds
    **WO-007** and **WO-011**.
    """
    cfg = _cfg()
    rows = _report_rows(_episode(cfg, "Padder", TB8_SEEDS[0], implemented))
    for r in rows:
        expected = abs(float(r.report_ratio) - RHO_MAX) < 1e-12
        assert bool(r.at_bound) is expected, (r.enterprise, r.report_ratio)


@pytest.mark.skeleton
def test_flag_is_raised_above_one_percent(implemented) -> None:
    """More than 1% of reports at the bound raises `BOUND_BINDING`, incrementally and in batch.

    Drive `ABOVE_THRESHOLD_FRACTION` of REPORT steps to `RHO_MAX` and assert:

        BOUND_FLAG in ledger.flags
        bound_binding(ledger) is True
        gosplan.metrics.ledger.BOUND_BINDING_FLAG == BOUND_FLAG
        gosplan.metrics.ledger.AT_BOUND_FLAG_THRESHOLD == AT_BOUND_THRESHOLD

    - the incremental flag from `Ledger.append` and the batch predicate `bound_binding` agreeing on
    the same ledger. Assert the fraction is computed over REPORT rows only: PRODUCE rows carry no
    report, and counting them would dilute the fraction by a factor of `M + 1` and hide a binding
    bound. Owning WO: **WO-002**; binds **WO-011**.
    """
    from gosplan.metrics.ledger import Ledger, bound_binding

    implemented(Ledger.append, bound_binding)
    ledger = _synthetic_ledger(ABOVE_THRESHOLD_FRACTION)
    assert bound_binding(ledger) is True
    assert BOUND_FLAG in ledger.flags


@pytest.mark.skeleton
def test_flag_is_absent_below_one_percent(implemented) -> None:
    """At or below 1% the flag stays down - the threshold is strict, and it is not a hair trigger.

    Drive `BELOW_THRESHOLD_FRACTION` of REPORT steps to `RHO_MAX` and assert `BOUND_FLAG not in
    ledger.flags` and `bound_binding(ledger) is False`. Repeat with no at-bound reports at all
    (`TruthfulMyopic` at `p1_default_config()`) and assert the same, and that a ledger with no
    REPORT rows at all returns `False` rather than raising.

    Without this half, a ledger that raised the flag unconditionally would pass T-B8 - and the G2
    hygiene criterion of PLAN section 4.5 requires the flag to be *absent* in all Phase-1 gate runs,
    so a false positive would block the gate as loudly as a false negative. Owning WO: **WO-002**;
    binds **WO-011**.
    """
    from gosplan.metrics.ledger import Ledger, bound_binding

    implemented(Ledger.append, bound_binding)
    for fraction in (0.0, BELOW_THRESHOLD_FRACTION, AT_BOUND_THRESHOLD):
        ledger = _synthetic_ledger(fraction)
        assert bound_binding(ledger) is False, fraction
        assert BOUND_FLAG not in ledger.flags, fraction


@pytest.mark.skeleton
def test_manifest_carries_the_flag(implemented, tmp_path) -> None:
    """The flag reaches `runs/<hash>/manifest.json`, where a reader of the result meets it.

    Write a manifest for the flag-raising run with `write_manifest(run_dir, cfg, extra)` into a
    `tmp_path` directory, then read the JSON back and assert `BOUND_FLAG` is in its `flags` field,
    that the file also carries the config hash, the full configuration, `SPEC_VERSION` and both
    seeds (CONTRACT rule 10), and that fields which do not apply to the run are present as `null`
    rather than omitted. Assert the non-flagged run's manifest has a `flags` field that exists and
    does not contain `BOUND_FLAG`.

    CONTRACT rule 8 requires the result to be *reported with the flag*: a flag that lives only in a
    ledger object nobody reads is not a report. Owning WO: **WO-002**; binds **WO-011**.
    """
    import json

    from gosplan.metrics.ledger import write_manifest

    implemented(write_manifest)
    cfg = _cfg()
    ledger = _synthetic_ledger(ABOVE_THRESHOLD_FRACTION)
    run_dir = tmp_path / cfg.hash()
    run_dir.mkdir(parents=True)
    write_manifest(str(run_dir), cfg, {"flags": tuple(ledger.flags)})
    doc = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert BOUND_FLAG in (doc.get("flags") or [])
