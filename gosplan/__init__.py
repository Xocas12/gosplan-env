"""gosplan-env: a plan-fulfilment environment with a rule-based planner and learning enterprises.

Realises: PLAN sections 1 (claims and scope), 2 (environment specification), 3 (parameter
registry) and 8 (repository layout). Owning work order: **WO-001** (Spec v0, CONTRACT, registry;
LEAD).

What this package is. A rule-based planner sets targets, allocates promised inputs, audits and
penalises; enterprises choose effort and what to report; the pre-registered phenomena of PLAN
section 4 are measured from the ledger. The frozen interface is `spec/spec.py` (CONTRACT rule 1):
every module in this package defines the same names with the same signatures, and a unit test
enforces that the runtime dataclasses here stay field-for-field identical to the ones declared
there.

**Skeleton status.** This is a skeleton repository. Every function and method body in `gosplan`
raises `NotImplementedError("<PLAN section> - implemented in WO-###")`; the work order named in the
message supplies the behaviour (PLAN section 12). The only real content a skeleton module carries
is: module and symbol docstrings, type aliases, enums, `Protocol` definitions, dataclass field
declarations with the Phase-1 defaults of PLAN section 3, and the registry data of
`gosplan.params`. Consequently the frozen test suites (PLAN section 11) collect and fail at
`NotImplementedError` until each work order lands, which is the expected state.

Import policy. This module imports **no submodule**, so `import gosplan` costs nothing and no
import cycle can form through the package root. Import what you need explicitly, for example
`from gosplan import params`, `from gosplan.config import EnvConfig` or `from gosplan.rng import
draw`.
"""

from __future__ import annotations

__version__ = "0.1.0"
"""Distribution version of the `gosplan-env` package, kept equal to `[project].version` in
`pyproject.toml`.

This is **not** the interface version: `gosplan.config.SPEC_VERSION` mirrors `spec/spec.py`'s
`SPEC_VERSION` (v0 "0.1.0", frozen to "1.0.0" by WO-013 at gate G1, CONTRACT rule 1) and is what a
run manifest records alongside the config hash (CONTRACT rule 10). The two happen to coincide at
the skeleton and are free to diverge afterwards.
"""
