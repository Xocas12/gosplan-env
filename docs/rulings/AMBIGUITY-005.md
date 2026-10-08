AMBIGUITY REPORT   WO-011   gosplan/metrics/ledger.py
Question (one sentence):
Does `Ledger.append` maintain the `BOUND_BINDING` flag, and what does `to_parquet` do with the
string the frozen fixture puts in the float `penalty_arg` column?

What the spec says / does not say (quote):
The `Ledger.append` docstring (after ambiguity #54) says "`append` maintains no flag", but
`tests/unit/test_ledger.py` and `tests/behavioural/test_bound_flag.py` assert
`BOUND_BINDING_FLAG in ledger.flags` after appends alone. Separately, the frozen `_record` fixture
in `tests/unit/test_ledger.py` sets `penalty_arg=cfg.incentive.penalty_arg` (the enum string
"positive_part") although `StepRecord.penalty_arg: float  # f_i`, and `to_parquet` must keep
declared dtypes, so the parquet round-trip tests crash.

Options considered (A/B/…), and why the spec does not decide:
Flag: A. `append` keeps the flag in step with the `bound_binding` predicate incrementally;
B. `flags` becomes a derived property; C. the tests are stale after #54. Fixture: A. declared
dtypes, fix the fixture; B. infer dtypes from values.

Impact if the wrong option is picked:
CONTRACT rule 8 hygiene; stable parquet schemas.

Tests blocked:
test_ledger.py (3), test_bound_flag.py (2).

LEAD RESOLUTION (2026-09-24):
Flag: A - the frozen tests win; `append` sets or clears `BOUND_BINDING` from running counts of
report rows and at-bound report rows, using the same threshold as `bound_binding`. The #54
"maintains no flag" wording is to be corrected at the WO-013 freeze. Fixture: the test is wrong -
a lead edit sets the fixture's `penalty_arg` to `0.0` (`.github/FROZEN_TEST_EXEMPTION` entry, to be
removed once merged). The manifest's embedded `config` is the four-section object, so that its
canonical digest equals `EnvConfig.hash()` (see the WO-003 follow-up commit).
