"""The ledger and the run manifest: round-trip, completeness and the bound flag (PLAN section 4).

Realises: PLAN sections 2.2 and 4 (the `StepRecord` columns and the ledger they form), CONTRACT rule
10 (the manifest) and CONTRACT rule 8 (bounds are results), plus PLAN section 11 (test architecture;
the unit half of **T-B8**). Owning work order: **WO-002** (frozen tests; LEAD). Binds the WO-011
must-pass line of PLAN section 12.3, verbatim - "`tests/unit/test_ledger.py` (round-trip parquet;
manifest has every rule-10 field; `BOUND_BINDING` logic, T-B8)". Module under test:
`gosplan/metrics/ledger.py`.

CONTRACT RULE 10, verbatim: "Every run writes `runs/<hash>/manifest.json`: config hash, spec
version, git hash, seeds, reference-PPO version, estimator version, LLM model ids and versions,
solver version and optimality gap, flags."

CONTRACT RULE 8, verbatim: "`report_ratio` is bounded at `rho_max = 10`. The fraction of reports at
the bound is logged; > 1% flags the run manifest `BOUND_BINDING` and the result is reported with the
flag. Never silently widen or narrow a bound to fix a result."

CONTRACT RULE 6: the ledger holds true quantities and is read by metrics and by lead-run experiments
only; nothing an agent can read touches it.

FROZEN BY CONTRACT RULE 2. SKELETON: every test is `@pytest.mark.skeleton` and skipped until WO-011
lands; each docstring states the exact assertion, formula and tolerance.
"""

from __future__ import annotations

import pytest


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-011")
def test_parquet_round_trip_preserves_every_column_and_dtype(p1_cfg, tmp_path) -> None:
    """`Ledger.to_parquet` writes a file that reads back identical.

    Assertion: build a ledger of `StepRecord`s covering every field (identifiers, the PLAN section
    2.2 state columns, the WO-011 additions `y, R, rho, S_pre, S_post, audited, S_hat, f, Pen, fill,
    deliv, consumer, val_measured, val_true, welfare, at_bound`, and the action and conservation
    columns), write it with `to_parquet(path)`, read it back, and assert: one row per `StepRecord`
    in append order; every column present; every dtype preserved (float columns as float, `audited`
    and `at_bound` as boolean, `phase` as string, the index columns as integers); and every float
    value equal to the original bit for bit, so a metric computed from the file equals the metric
    computed from memory.

    First bullet of the WO-011 must-pass list. `pyarrow` is the writer and is a runtime dependency
    of `gosplan/metrics/ledger.py` alone; `spec/spec.py` deliberately does not import it.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-011")
def test_parquet_explodes_tuple_valued_fields_one_column_per_good(p1_cfg, tmp_path) -> None:
    """Per-good tuples become `name_0 .. name_{J-1}` columns, readable without this module.

    Assertion: the written file carries `inv_inputs_0 .. inv_inputs_{J-1}` and the same expansion
    for `request`, `need`, `alloc`, `deliv`, `input_consumed` and `consumer`; the column count
    matches the scalar fields plus `J` per tuple-valued field; and reading the file with a plain
    parquet reader (no import of `gosplan`) reconstructs the per-good arrays in good order. The
    ledger has to be readable by a later analysis that does not have this codebase - which is what
    makes a result reproducible rather than merely repeatable.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-011")
def test_append_stores_one_record_per_enterprise_per_agent_step(p1_cfg) -> None:
    """The ledger is append-only and its length is `N * periods * (M + 1)`.

    Assertion: appending the `StepInfo` stream of a complete episode leaves
    `len(ledger.records) == N * P * (M + 1)`, in append order, with no record mutated after the
    fact; identifiers (`run_hash`, `episode`, `t_period`, `k_step`, `phase`, `enterprise`,
    `sector`) are unique per row and `run_hash` equals `cfg.hash()` for every row, so a file can
    never mix two configurations.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-011")
def test_bound_binding_is_raised_above_one_percent_of_reports_at_the_bound(p1_cfg) -> None:
    """T-B8, unit half: more than 1% of reports at `rho_max` raises `BOUND_BINDING`.

    Assertion: appending records in which the fraction of REPORT-step rows with `at_bound is True`
    exceeds `AT_BOUND_FLAG_THRESHOLD = 0.01` puts `BOUND_BINDING_FLAG = "BOUND_BINDING"` into
    `ledger.flags`, and the flag reaches the manifest's `flags` field; the fraction is computed over
    reports, not over all agent-steps, so the `M` PRODUCE rows of each period do not dilute it. The
    behavioural counterpart, test T-B8, forces `rho = 10` in more than 1% of reports through a live
    environment and asserts the same flag.

    Third bullet of the WO-011 must-pass list. CONTRACT rule 8: the flag is reported with the
    result; the bound is never moved to clear it.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-011")
def test_bound_binding_is_absent_below_the_threshold(p1_cfg) -> None:
    """At or below 1% at-bound reports the flag is not raised.

    Assertion: a ledger whose at-bound fraction is exactly at and just below
    `AT_BOUND_FLAG_THRESHOLD` leaves `flags` without `BOUND_BINDING`, and a ledger with no at-bound
    report at all leaves `flags` empty; `bound_binding(ledger)` agrees with the flag set in both
    directions. Gate G2 criterion 4 (hygiene) requires the flag absent in all runs (PLAN section
    4.5), so a false positive here would fail a gate for no reason and a false negative would pass
    one it should not.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-011")
def test_manifest_carries_every_contract_rule_10_field(p1_cfg, tmp_path) -> None:
    """`write_manifest` writes every key of `MANIFEST_FIELDS`, in order.

    Assertion: `runs/<hash>/manifest.json` parses as JSON and its keys are exactly
    `MANIFEST_FIELDS` - `config_hash`, `config`, `spec_version`, `git_hash`, `seed_env`,
    `seed_policy`, `reference_ppo_version`, `estimator_version`, `estimator_backend`, `llm_models`,
    `solver`, `solver_version`, `solver_optimality_gap`, `bunching_settings`, `flags` - with
    `config_hash == cfg.hash()`, `config` the canonical JSON encoding of the configuration,
    `spec_version == SPEC_VERSION`, the two seeds as written, `bunching_settings` the pre-registered
    PLAN section 4.5 constants from `gosplan/metrics/phenomena.py`, and `flags` a list carrying
    every flag the run raised.

    Second bullet of the WO-011 must-pass list, and CONTRACT rule 10 verbatim.
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-011")
def test_manifest_writes_null_for_fields_that_do_not_apply(p1_cfg, tmp_path) -> None:
    """A field that does not apply to a run is `null`, never omitted.

    Assertion: for a Phase-1 run with no LLM ministry and no oracle solve, the manifest still
    carries `llm_models`, `solver`, `solver_version` and `solver_optimality_gap` as JSON `null`
    rather than dropping the keys; the key set is identical across runs of different kinds. A
    missing field is then always a bug and never an ambiguity - which is the reason the convention
    exists (`MANIFEST_FIELDS` docstring, CONTRACT rule 10).
    """
    assert False


@pytest.mark.skeleton
@pytest.mark.skip(reason="skeleton: implemented in WO-011")
def test_manifest_names_the_run_directory_by_the_configuration_hash(p1_cfg, tmp_path) -> None:
    """The run directory is `runs/<cfg.hash()>/` and the manifest agrees with it.

    Assertion: `write_manifest(run_dir, cfg, extra)` writes into a directory whose name equals
    `cfg.hash()`, and the `config_hash` field inside the file equals that same digest, so a
    directory can never be attributed to a configuration it was not produced from; writing twice for
    the same configuration is idempotent in the key set and overwrites rather than appending.
    """
    assert False
