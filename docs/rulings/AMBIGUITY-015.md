AMBIGUITY REPORT   WO-016   gosplan/metrics/__init__.py
Question: `resolve_estimators` is documented to return an `EstimatorBackend` dataclass, while the
frozen `tests/unit/test_phenomena_p1.py` uses its result as a mapping (`name in resolved`,
`resolved["bunching_estimate"]`).
LEAD RESOLUTION (2026-09-24): both hold - `EstimatorBackend` stays an immutable dataclass and gains
read-only `__getitem__` / `__contains__` over its field names (additive, no field moved).
