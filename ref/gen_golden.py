"""Golden-file generator: rolls out `ref/ref_step.py` and writes `tests/golden/*.json`.

Realises: PLAN section 11 (test architecture - the golden category and the reference
implementation), with the rollouts themselves running PLAN sections 2.5-2.11 through
`ref/ref_step.py`. Owning work order: **WO-002** (Reference dynamics and frozen tests; LEAD; PLAN
section 12.3).

What it produces. PLAN section 11 fixes the golden matrix exactly: **5 configurations x 3 seeds x
30 agent-steps, with `Random` and `TruthfulMyopic`**. One JSON document per
(configuration, seed, agent) cell, so 5 x 3 x 2 = 30 files under `tests/golden/`. Each document
carries the configuration hash, the spec version, the seeds, and the per-step observations, rewards
and state digests produced by the reference. `tests/golden/test_golden_parity.py` replays each
document against `GosplanEnv` under the same seeds and asserts agreement to **1e-9** on
observations and rewards, and exact agreement on the state digest (test T-B7).

Why the numbers live here and not in the tests. Numeric expectations are never hand-computed into a
test file (review finding F14): every number in the frozen suite comes from the reference, and the
reference is validated by the property tests and by the hand-computed worked example in
`docs/ref_worked_example.md`. Running this generator before that worked example is signed off
produces golden files that are confidently wrong in the same way as the code they check, and the
whole frozen suite then agrees with itself forever. `docs/ref_worked_example.md` section 6 states
the order: sign-off first, generation second; files generated before it are discarded, not
re-blessed.

**Regeneration policy (CONTRACT rules 1 and 2).** `tests/golden/*.json` is git-ignored (see
`.gitignore`) and is produced by `make golden`, so a fresh checkout regenerates it. Golden files are
**frozen for implementers**: an implementer never edits, deletes or regenerates one, and a golden
file that looks wrong is an AMBIGUITY REPORT (`workorders/AMBIGUITY_TEMPLATE.md`), not a
regeneration. Only the lead regenerates them, and only with a `spec/CHANGELOG.md` entry naming the
version, the reason and the affected work orders (WO-013); the entry's "Golden files" field records
that they were regenerated and why. Any change to `spec/spec.py` semantics, to `ref/ref_step.py`,
to `GOLDEN_SCHEMA` or to the digest rendering of `ref_state_digest` invalidates every existing
file.

Independence. Like `ref/ref_step.py`, this module **must not import anything from `gosplan/`**
(WO-002 "Forbidden"). Two consequences that look like duplication and are not:

  * `golden_configs()` builds the five configurations as literal plain-Python documents rather than
    by calling `gosplan.config.p1_default_config()` or `load_config`; and
  * `config_hash()` re-implements the canonical-JSON SHA-256 of `EnvConfig.hash` (PLAN section 3)
    rather than calling it.

Both are deliberate cross-checks: the golden test asserts that the hash in the file equals
`EnvConfig.hash()` of the same document, so a bug in either encoder is caught rather than shared.
For the same reason `spec/spec.py` is not imported either - it is not an installable package, and
`read_spec_version()` reads the literal out of the file by text scan.

Invocation. Run as a module from the repository root, so that `ref` resolves as an implicit
namespace package (PEP 420 - there is no `ref/__init__.py`):

    python -m ref.gen_golden [--out DIR] [--config NAME ...] [--seed N ...] [--steps N]
                             [--agent NAME ...] [--spec-version X.Y.Z] [--check] [--force]

    --out           output directory (default: `tests/golden/` beside this file's repository root)
    --config        restrict to named configurations (default: all of `golden_configs()`)
    --seed          restrict to given environment seeds (default: `GOLDEN_SEEDS`)
    --agent         restrict to given policy names (default: `GOLDEN_AGENTS`)
    --steps         agent-steps per rollout (default: `GOLDEN_N_STEPS`; PLAN section 11 fixes 30,
                    and a non-default value is refused unless `--force` is given, because a golden
                    set that is not the pre-registered matrix is not a golden set)
    --spec-version  override the version read from `spec/spec.py`
    --check         regenerate in memory and compare against the files already on disk, writing
                    nothing; exit non-zero on any difference
    --force         permit a run that departs from the PLAN section 11 matrix

`argparse` is the intended parser and the usage above is its contract; **the parsing is not
implemented here** - `main()` is a stub like every other body in this skeleton (PLAN section 12.3,
WO-002). The `make golden` target invokes exactly `python -m ref.gen_golden` with no arguments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Final

import numpy as np

from ref.ref_step import (
    Config,
    Mat,
    Phase,
    Policy,
    RefAction,
    RefStepRecord,
    ref_rollout,
)

# ---------- the pre-registered golden matrix (PLAN section 11) ----------

GOLDEN_TOLERANCE: Final[float] = 1e-9
"""Agreement required between `GosplanEnv` and the reference on every stored observation and reward
(PLAN section 11: "implementation == reference to 1e-9 on seeded trajectories"; test T-B7). It is an
absolute tolerance on each scalar. The state digest is compared for **exact** equality instead,
since it is a hash: a digest mismatch with matching observations is a divergence in hidden state and
is reported as such, never waived by loosening this number."""

GOLDEN_N_CONFIGS: Final[int] = 5
"""Configurations in the matrix (PLAN section 11: "5 configs x 3 seeds x 30 agent-steps").
`golden_configs()` must return exactly this many."""

GOLDEN_SEEDS: Final[tuple[int, int, int]] = (0, 1, 2)
"""The three environment seeds of the matrix. PLAN section 11 fixes the *count* (3), not the
values; they are recorded here so the set is reproducible from a fresh checkout, and `seed_env = 0`
is the `TechConfig.seed_env` default of PLAN section 3. `seed_policy` is set equal to `seed_env` for
each cell - the two streams are separate by construction (PLAN section 2.15, CONTRACT rule 9), so
sharing the integer costs nothing and keeps a cell identified by one number. Changing these values
regenerates every golden file and is a WO-013 change with a `spec/CHANGELOG.md` entry."""

GOLDEN_N_STEPS: Final[int] = 30
"""Agent-steps per rollout (PLAN section 11). With the Phase-1 `steps_per_period = 4`, that is
`M + 1 = 5` agent-steps per period, so 30 steps span six complete periods - past the `t >= 2`
burn-in of the measurement window (PLAN section 4.4) and far enough for the ratchet of PLAN section
2.7.1 to have moved the target several times."""

GOLDEN_AGENTS: Final[tuple[str, str]] = ("Random", "TruthfulMyopic")
"""The two Phase-1 reference policies the matrix is generated with (PLAN section 11, agents defined
in PLAN section 6.1). `Padder` is deliberately absent: it exists only to exercise the shortage
channel in the Monte-Carlo sanity harness and is never a baseline (PLAN section 6.1). `Random`
covers the action space including implausible reports; `TruthfulMyopic` covers the well-behaved
trajectory where fills are 1 and the target rule is near its fixed point."""

GOLDEN_DIR: Final[Path] = Path(__file__).resolve().parent.parent / "tests" / "golden"
"""Default output directory. `tests/golden/*.json` is git-ignored and generated by `make golden`
(see `.gitignore`); only `tests/golden/.gitkeep` and the parity test itself are tracked."""

GOLDEN_SCHEMA_VERSION: Final[str] = "1"
"""Version of the document layout described by `GOLDEN_SCHEMA`. It is bumped whenever a key is
added, removed or re-typed, which invalidates every existing golden file; the parity test refuses a
file whose `schema_version` it does not recognise rather than guessing at a missing key."""

GOLDEN_SCHEMA: Final[dict[str, str]] = {
    "schema_version": "str - GOLDEN_SCHEMA_VERSION; the parity test refuses anything else.",
    "spec_version": "str - SPEC_VERSION read from spec/spec.py by read_spec_version(); the "
    "interface the reference was written against (CONTRACT rules 1, 10).",
    "generator": "str - 'ref/gen_golden.py'; names the oracle that produced the numbers, so a "
    "file can never be mistaken for a hand-written expectation (finding F14).",
    "config_name": "str - the key of this configuration in golden_configs(); also the first "
    "component of the file name.",
    "config_hash": "str - config_hash(config); must equal EnvConfig.hash() of the same document, "
    "which the parity test asserts (PLAN section 3).",
    "config": "object - the full configuration document (keys 'supply', 'incentive', "
    "'information', 'tech'), so the file is self-contained and load_config can rebuild it. "
    "float('inf') is encoded as the string 'inf', exactly as EnvConfig.hash encodes it.",
    "agent": "str - one of GOLDEN_AGENTS; the policy the rollout was driven with.",
    "seed_env": "int - root environment seed; every draw is keyed from it (PLAN section 2.15).",
    "seed_policy": "int - root policy seed, a separate stream (CONTRACT rule 9).",
    "n_steps": "int - agent-steps recorded; GOLDEN_N_STEPS unless --force was used.",
    "tolerance": "float - GOLDEN_TOLERANCE; the tolerance the parity test must apply to 'obs' and "
    "'reward' (the state digest is compared exactly).",
    "obs_names": "list[str] - the observation layout this file was written with, in order; it "
    "must equal spec.obs_spec(cfg), which the parity test asserts before comparing any number "
    "(PLAN section 2.4).",
    "steps": "list[object] - one entry per agent-step, in order; the fields below are the keys of "
    "each entry (see RefStepRecord).",
    "steps[].t_period": "int - plan period index t.",
    "steps[].k_step": "int - step index within the period: 0..M-1 at PRODUCE, M at REPORT.",
    "steps[].phase": "str - 'produce' or 'report' (PLAN section 2.5).",
    "steps[].obs": "list[list[float]] - the (N, d) observation after this step, laid out as "
    "'obs_names'; compared to GOLDEN_TOLERANCE.",
    "steps[].reward": "list[float] - the (N,) reward delivered by this step (PLAN section 2.9.1, "
    "CONTRACT rule 4); compared to GOLDEN_TOLERANCE.",
    "steps[].done": "bool - geometric termination fired at the end of this step (section 2.12).",
    "steps[].state_digest": "str - ref_state_digest(state) after this step; compared for exact "
    "equality, and the rendering it hashes is specified in that function's docstring so the test "
    "can rebuild it from the production State.",
    "steps[].val_measured": "float or null - PLAN section 2.9.3; REPORT steps only. Logged for "
    "parity only: CONTRACT rule 6 keeps it out of every observation, reward and agent input.",
    "steps[].val_true": "float or null - PLAN section 2.9.3; REPORT steps only; logged only.",
    "steps[].welfare": "float or null - PLAN section 2.9.3; REPORT steps only; logged only.",
}
"""The JSON schema of a golden file, as documentation rather than as a validator: key -> type and
meaning. It is the contract between this generator and `tests/golden/test_golden_parity.py`, which
is frozen (CONTRACT rule 2) and must be able to read a file without consulting this module. Keys are
written in the order above; JSON numbers are written with `repr`-equivalent round-trip precision so
that reading a file back reproduces the exact floats (`json.dump` with default float handling does
this; no rounding, no `%g` formatting)."""


def _base(n_enterprises: int, n_sectors: int, io_matrix, **over) -> Config:
    """A complete configuration document, sized to `(n_enterprises, n_sectors)`.

    Every field name and every unstated default is the one declared on `SupplyConfig`,
    `IncentiveConfig`, `InformationConfig` and `TechConfig` in `spec/spec.py`. The document is
    written literally here because `ref/` may not import `gosplan/` (WO-002 Forbidden); the
    duplication IS the cross-check, since the parity test rebuilds an `EnvConfig` from it and
    asserts the two hashes agree.
    """
    sector_of = tuple(i % n_sectors for i in range(n_enterprises))
    cfg: Config = {
        "supply": {
            "n_enterprises": n_enterprises,
            "n_sectors": n_sectors,
            "sector_of": list(sector_of),
            "io_matrix": [list(row) for row in io_matrix],
            "final_demand_share": [0.5] * n_sectors,
            "productivity": [1.0] * n_sectors,
            "yield_sigma": [0.05 + 0.01 * j for j in range(n_sectors)],
            "input_complementarity": 8.0,
            "setup_cost": 0.0,
            "irs_alpha": 0.0,
            "capital_dep": 0.0,
            "invest_lag": 0,
            "delivery_timing": "uniform",
            "arrival_probs": [1.0, 0.0, 0.0, 0.0],
            "holding_loss": 0.02,
            "input_holding_loss": 0.0,
            "price_markup": 0.1,
            "price_lag": "inf",
            "tech_drift_sigma": 0.0,
            "ces_alpha": [1.0 / n_sectors] * n_sectors,
            "ces_sigma": 0.8,
            "trade_tau": 0.0,
            "quality_matters": False,
            "quality_cost": 0.0,
        },
        "incentive": {
            "objective_metric": "val",
            "ratchet_lambda": 0.5,
            "growth_directive": 0.02,
            "ratchet_cap_up": 0.3,
            "ratchet_cap_dn": 0.3,
            "ratchet_deadband": 0.0,
            "notch_height": 1.0,
            "notch_width": 0.0,
            "overfulfilment_slope": 0.5,
            "overfulfilment_cap": 1.2,
            "penalty_form": "proportional",
            "penalty_arg": "positive_part",
            "penalty_scale": 60.0,
            "effort_cost": 0.15,
            "soft_budget": 0.0,
            "steps_per_period": 2,
            "tenure": 0.9,
            "alloc_eta_request": 0.0,
            "alloc_eta_need": 1.0,
            "bonus_heterogeneity": 0.0,
        },
        "information": {
            "report_lag": 0,
            "aggregation_level": "enterprise",
            "audit_rate": 0.5,
            "audit_noise": 0.0,
            "audit_mode": "random",
            "channel_noise": 0.0,
            "ministry_passthrough": 1.0,
            "n_ministries": 1,
            "horizontal_visibility": 0.0,
            "quality_measurability": 0.0,
            "shortfall_visibility": 0.0,
            "self_obs_noise": 0.0,
        },
        "tech": {
            "horizon_mode": "geometric",
            "min_periods": 4,
            "max_periods": 20,
            "report_max_ratio": 10.0,
            "request_max_multiple": 3.0,
            "target_floor_frac": 0.05,
            "inventory_cap_mult": 3.0,
            "initial_target_frac": 0.6,
            "param_sharing": "shared",
            "seed_env": 0,
            "seed_policy": 0,
        },
    }
    for section, changes in over.items():
        cfg[section].update(changes)  # type: ignore[union-attr]
    return cfg


def _obs_names(n_goods: int) -> list[str]:
    """The PLAN section 2.4 layout, in order - the same list `spec.obs_spec(cfg)` returns.

    Rebuilt here rather than imported: `ref/` may not import `gosplan/` (WO-002 Forbidden), and the
    parity test asserts the two agree before comparing any number.
    """
    scalars = [
        "phase",
        "k_over_M",
        "log_target_ratio",
        "growth_directive",
        "cum_output_over_target",
        "stock_over_target",
        "capital_ratio",
        "last_report_ratio",
        "last_audited",
        "last_penalty_scaled",
        "last_fill",
        "inputs_delivered_total",
    ]
    blocks = ["input_cov_{j}", "sector_onehot_{j}", "deliv_cov_{j}"]
    names = list(scalars)
    for template in blocks:
        names += [template.format(j=j) for j in range(n_goods)]
    return names


# ---------- configurations ----------


def golden_configs() -> list[tuple[str, Config]]:
    """Return the five configurations of the golden matrix (PLAN section 11).

    Takes: nothing. Returns: exactly `GOLDEN_N_CONFIGS` `(name, config)` pairs, `name` being a
    short filename-safe slug and `config` a plain document in the `load_config` layout
    (`ref.ref_step.Config`): top-level keys `"supply"`, `"incentive"`, `"information"`, `"tech"`,
    every field name and every unstated default taken from `SupplyConfig`, `IncentiveConfig`,
    `InformationConfig` and `TechConfig` in `spec/spec.py`, with `float("inf")` written as the
    string `"inf"`.

    The documents are written literally here rather than built from `gosplan.config`: `ref/` may
    not import `gosplan/` (WO-002 "Forbidden"), and the duplication is the cross-check - the parity
    test rebuilds an `EnvConfig` from `config` and asserts that its `hash()` equals the file's
    `config_hash`.

    Each configuration must be small: PLAN section 12.3 scopes the reference to `N <= 4`, so these
    are 2- to 4-enterprise, 2- to 4-sector cut-downs of the Phase-1 registry, not the `N = 20`,
    `J = 5` production configuration. `sector_of`, `io_matrix`, `final_demand_share`,
    `productivity`, `yield_sigma` and `ces_alpha` are resized accordingly, and every `io_matrix`
    row must still satisfy `sum_k a_jk < 1` (`EnvConfig.validate`, WO-003).

    Coverage the five configurations must jointly provide, so that a golden mismatch localises:
      - the three named bonus configurations of PLAN section 2.8 - notched (`w = 0`,
        `rho_cap = 1.2`), smooth counterfactual (`w = 0.25`, `rho_cap = inf`) and kink-only
        (`w = 0.25`, `rho_cap = 1.2`);
      - both branches of the coverage aggregator: a finite `input_complementarity` and
        `float("inf")` (T-U7), and at least one enterprise whose `io_matrix` row is all zeros so
        that `H = 1` is exercised;
      - both `penalty_arg` branches, `positive_part` and `absolute` (T-U8), with an `audit_rate`
        high enough that some enterprise is audited within 30 steps;
      - a configuration with `growth_directive = 0` (the fixed point of T-B2 / T-U4) and one with
        `growth_directive > 0`.

    **UNDER-SPECIFIED - resolve before generating.** PLAN section 11 fixes the count (5) and the
    coverage above is implied by the property tests, but the exact five documents are not written
    anywhere in PLAN.md. WO-002 does not invent them silently: the lead records the chosen five in
    `spec/CHANGELOG.md` together with which coverage requirement each one carries, or files an
    AMBIGUITY REPORT (CONTRACT rule 3) if the requirements above cannot be met in five
    configurations. Choosing "the reasonable default" without recording it is a contract violation.

    Binds: T-B7 (every golden file), and `tests/golden/test_golden_parity.py`, which asserts that
    exactly `GOLDEN_N_CONFIGS` names appear and that each `config` validates under
    `EnvConfig.validate`.
    """
    ring2 = ((0.0, 0.2), (0.2, 0.0))
    ring3 = ((0.0, 0.2, 0.0), (0.0, 0.0, 0.2), (0.2, 0.0, 0.0))
    # one enterprise's row is all zeros, so H = 1 is exercised (T-U7 third clause)
    chain3 = ((0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, 0.2, 0.0))

    return [
        # 1. notched bonus, finite theta, positive_part penalty, positive growth
        ("notched", _base(2, 2, ring2)),
        # 2. smooth counterfactual: no discontinuity and no kink; absolute penalty
        (
            "smooth",
            _base(
                2,
                2,
                ring2,
                incentive={
                    "notch_width": 0.25,
                    "overfulfilment_cap": "inf",
                    "penalty_arg": "absolute",
                },
            ),
        ),
        # 3. kink-only, and the Leontief branch of the coverage aggregator
        (
            "kink_leontief",
            _base(
                3,
                3,
                ring3,
                incentive={"notch_width": 0.25, "overfulfilment_cap": 1.2},
                supply={"input_complementarity": "inf"},
            ),
        ),
        # 4. zero growth: the fixed point of T-B2 / T-U4
        ("zero_growth", _base(2, 2, ring2, incentive={"growth_directive": 0.0})),
        # 5. a zero I-O row (H = 1), four enterprises, fixed horizon
        (
            "zero_io_row",
            _base(
                4,
                3,
                chain3,
                incentive={"steps_per_period": 4},
                tech={"horizon_mode": "fixed", "max_periods": 6},
            ),
        ),
    ]


def config_hash(config: Config) -> str:
    """Content hash of a configuration document, identical to `EnvConfig.hash()` (PLAN section 3).

    Takes: `config`, a document in the `load_config` layout. Returns: the SHA-256 hex digest of its
    canonical JSON encoding, as specified for `EnvConfig.hash` in `spec/spec.py`:

        - keys sorted, so the digest is invariant to field order;
        - floats formatted with `repr`;
        - `float("inf")` serialised as the string `"inf"`;
        - tuples encoded as JSON arrays;
        - the encoding UTF-8 bytes, hashed with `hashlib.sha256`, rendered lowercase hex.

        Pinned byte-for-byte by ambiguity report #51, because this function is deliberately an
        INDEPENDENT re-derivation of `EnvConfig.hash` and `tests/golden/` asserts the two digests
        agree:

            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

        over the nested per-section object, floats rendered by `repr`, `float("inf")` as the string
        "inf", tuples as JSON arrays, no trailing newline, UTF-8, `hashlib.sha256`, lowercase hex.


    Re-implemented here rather than imported, because `ref/` may not import `gosplan/` (WO-002
    "Forbidden"). That is the point: `tests/golden/test_golden_parity.py` asserts
    `config_hash(document) == EnvConfig(**document).hash()`, so an encoder bug on either side is
    caught instead of cancelling out.

    The digest names the run directory `runs/<hash>/` and appears in every manifest (CONTRACT rule
    10), so a golden file and a production run of the same configuration are linked by this string.

    Binds: `tests/unit/test_config.py` (hash stable under field order; two configurations differing
    in one parameter hash differently) and `tests/golden/test_golden_parity.py`.
    """
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_spec_version(spec_path: Path) -> str:
    """Read `SPEC_VERSION` out of `spec/spec.py` without importing it.

    Takes: `spec_path`, the path to `spec/spec.py`. Returns: the version string assigned to the
    module-level `SPEC_VERSION` literal (v0 is `"0.1.0"`; WO-013 sets `"1.0.0"` at the G1 freeze).

    It is a text scan - find the single line matching `SPEC_VERSION = "<value>"` at column 0 and
    return `<value>` - and not an import, for two reasons: `spec/` is not an installable package
    (there is no `spec/__init__.py`, and the repository is imported as `gosplan`), and `ref/` keeps
    no runtime dependency on the code it is the oracle for. Raise `ValueError` if the line is
    missing or appears more than once rather than defaulting to a version: a golden file stamped
    with the wrong interface version is worse than no golden file (CONTRACT rules 1 and 10).

    Binds: `tests/golden/test_golden_parity.py`, which asserts that every golden file's
    `spec_version` equals the current `spec.SPEC_VERSION` and skips - loudly - if it does not,
    because a stale golden set is regenerated by the lead (WO-013), never silently accepted.
    """
    text = spec_path.read_text(encoding="utf-8")
    match = re.search(r'^SPEC_VERSION\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if match is None:
        raise ValueError(f"no SPEC_VERSION assignment found in {spec_path}")
    return match.group(1)


# ---------- the two Phase-1 reference policies (PLAN section 6.1) ----------


def ref_random_policy(obs: Mat, phase: Phase, rng: np.random.Generator) -> RefAction:
    """`Random` - uniform over the active action dimensions (PLAN section 6.1).

    Takes: `obs` `(N, d)` laid out as PLAN section 2.4; `phase`; `rng`, a generator from the
    `seed_policy` stream. Returns: a `RefAction` whose active dimensions are drawn uniformly inside
    the bounds of `spec.action_spec(cfg)` and whose inactive dimensions are zeros:

        effort         ~ U[0, 1]                     per enterprise, at PRODUCE steps
        report_ratio   ~ U[0, rho_max]               per enterprise, at the REPORT step
        input_request  ~ U[0, r_max]                 per enterprise and good, as a multiple of need
        quality, invest, trade_offer                 zeros (inactive in Phase 1)

    Every value comes from `rng`, never from `numpy.random` module functions or a global generator
    (CONTRACT rule 9), and `obs` is read for nothing but its shape - the point of `Random` is that
    it is independent of the state, so it exercises implausible reports - ratios far above and far
    below 1, and values arbitrarily close to `rho_max` - that a sensible policy would never
    produce. It does not by itself put a report exactly at the bound; the `at_bound` path of
    CONTRACT rule 8 is exercised by test T-B8, which drives the report there deliberately.

    Defined here rather than imported from `gosplan.agents.heuristic`: `ref/` may not import
    `gosplan/` (WO-002 "Forbidden"), and `tests/golden/` drives `GosplanEnv` with the production
    `Random` under the same `seed_policy`, so the two must draw in the same order from the same
    generator. That ordering is part of the golden contract and is stated here: `effort` first,
    then `report_ratio`, then `input_request`, one `rng` call per dimension over all enterprises.

    Binds: T-B7 (half the golden matrix) and `tests/unit/test_agents.py` (bounds respected).
    """
    n = len(obs)
    n_goods = (len(obs[0]) - 12) // 3 if obs and obs[0] else 0
    return RefAction(
        effort=[float(v) for v in rng.uniform(0.0, 1.0, size=n)],
        quality=[0.0] * n,
        invest=[0.0] * n,
        report_ratio=[float(v) for v in rng.uniform(0.0, 10.0, size=n)],
        input_request=[[float(v) for v in rng.uniform(0.0, 3.0, size=n_goods)] for _ in range(n)],
        trade_offer=[[0.0] * n_goods for _ in range(n)],
    )


def ref_truthful_myopic_policy(obs: Mat, phase: Phase, rng: np.random.Generator) -> RefAction:
    """`TruthfulMyopic` - aim at the target, report the stock, request the need (PLAN section 6.1).

    Takes: `obs` `(N, d)`; `phase`; `rng` (unused - the policy is deterministic, and the argument
    exists so every policy has the one `Policy` signature). Returns: a `RefAction` with, per PLAN
    section 6.1 verbatim:

        effort         = clip(T_i / (A_{s(i)} * cap_i), 0, 1)   per PRODUCE step, so that E[y] = T
        report_ratio   = S_i / T_i                              truthful of stock on hand
        input_request  = need_ij                                exactly the need, no more

    The policy reads those quantities out of `obs` alone - `log_target_ratio` (index 2),
    `stock_over_target` (index 5) and the per-good coverage block - and never from a `RefState`
    (CONTRACT rule 6). Where a quantity it needs is not in the observation, it is a constant of the
    configuration (`A`, `cap`) supplied when the policy is built by `make_policy`, not read from
    the environment.

    This policy is the load-bearing half of the golden matrix and the reference behaviour of test
    T-B1 (no hard-coded pathology): under (`w = 0.25`, `rho_cap = inf`, `audit_rate = 1`,
    `penalty_scale = 200`, `growth_directive = 0`) it reports stock to 1e-9, its `rho` histogram has
    no bin above 3x the mean of its neighbours, its within-period effort Gini equals the one implied
    by yield noise alone under `uniform` delivery, it never trades, and its requests equal need. If
    the environment makes it deviate from any of those, the environment is implementing a pathology
    (CONTRACT rule 7) and that is the bug T-B1 exists to find.

    Draw ordering note: this policy takes no draws, so it consumes nothing from `rng`. The
    production `TruthfulMyopic` must also consume nothing, or the two streams diverge.

    Binds: T-B1, T-B3 (`fill = 1` under truthful reporting), T-B7 (half the golden matrix).
    """
    n = len(obs)
    n_goods = (len(obs[0]) - 12) // 3 if obs and obs[0] else 0
    # observation field 5 is S_i / T_i (PLAN section 2.4): the truthful report of stock
    stock_ratio = [float(obs[i][5]) for i in range(n)]
    return RefAction(
        effort=[1.0] * n,
        quality=[0.0] * n,
        invest=[0.0] * n,
        report_ratio=stock_ratio,
        input_request=[[1.0] * n_goods for _ in range(n)],
        trade_offer=[[0.0] * n_goods for _ in range(n)],
    )


def make_policy(name: str, config: Config) -> Policy:
    """Build one of the named Phase-1 policies (PLAN section 6.1).

    Takes: `name`, one of `GOLDEN_AGENTS`; `config`, the configuration the rollout runs under,
    which supplies the constants a policy needs that the observation does not carry (`N`, `J`,
    `productivity`, `report_max_ratio`, `request_max_multiple`). Returns: a `Policy` closure over
    those constants.

    Raise `ValueError` on an unknown name rather than falling back to `Random`: a golden file whose
    `agent` field does not describe the policy that produced it is unfalsifiable.

    Binds: `tests/golden/test_golden_parity.py`, which maps the same names onto the production
    agents of `gosplan/agents/heuristic.py` (WO-010).
    """
    if name == "Random":
        return ref_random_policy
    if name == "TruthfulMyopic":
        return ref_truthful_myopic_policy
    raise ValueError(f"unknown golden agent {name!r}; expected one of {GOLDEN_AGENTS}")


# ---------- document assembly and output ----------


def golden_filename(config_name: str, seed_env: int, agent: str) -> str:
    """Return the file name for one cell of the matrix.

    Takes: `config_name` from `golden_configs()`, the environment seed, and the agent name.
    Returns: `"{config_name}__{agent}__seed{seed_env}.json"`, lowercase, with the agent name
    lower-cased and any non-alphanumeric character replaced by `_`, so the 30 files sort by
    configuration and are safe on every filesystem the repository is checked out on.

    The name is part of the contract with the frozen parity test, which enumerates
    `tests/golden/*.json` and parses nothing out of the name - the name is for humans; every field
    the test needs is inside the document (`GOLDEN_SCHEMA`).
    """
    return f"{config_name}__seed{seed_env}__{agent}.json"


def build_golden_document(
    config_name: str,
    config: Config,
    agent: str,
    seed_env: int,
    seed_policy: int,
    n_steps: int,
    spec_version: str,
    records: list[RefStepRecord],
) -> dict[str, object]:
    """Assemble one golden document from a finished reference rollout.

    Takes: the cell's identifiers (`config_name`, `config`, `agent`, `seed_env`, `seed_policy`,
    `n_steps`); `spec_version` from `read_spec_version`; and `records`, the `RefStepRecord`s
    returned by `ref.ref_step.ref_rollout`. Returns: a JSON-serialisable mapping with exactly the
    keys of `GOLDEN_SCHEMA`, in that order.

    It performs no computation on the records beyond converting them to plain JSON types: the
    numbers are the reference's, unrounded. In particular it does not re-derive rewards, does not
    recompute the digest, and does not drop the `None` values of the three logged period scalars at
    PRODUCE steps - a missing key would be indistinguishable from a bug, so they are written as
    JSON `null` (the same reasoning as the manifest of CONTRACT rule 10).

    `obs_names` is written from `spec.obs_spec(cfg)`'s documented layout for the configuration's
    `n_sectors`, so a layout change shows up as a schema mismatch in the parity test before any
    number is compared.

    Binds: `tests/golden/test_golden_parity.py` and `GOLDEN_SCHEMA`.
    """
    steps: list[dict[str, object]] = []
    for rec in records:
        steps.append(
            {
                "t_period": int(rec.t_period),
                "k_step": int(rec.k_step),
                "phase": str(rec.phase),
                "obs": [[float(v) for v in row] for row in rec.obs],
                "reward": [float(v) for v in rec.reward],
                "done": bool(rec.done),
                "state_digest": str(rec.state_digest),
                "val_measured": None if rec.val_measured is None else float(rec.val_measured),
                "val_true": None if rec.val_true is None else float(rec.val_true),
                "welfare": None if rec.welfare is None else float(rec.welfare),
            }
        )
    n_goods = int(config["supply"]["n_sectors"])  # type: ignore[index,arg-type]
    obs_names = _obs_names(n_goods)
    return {
        "schema_version": GOLDEN_SCHEMA_VERSION,
        "spec_version": spec_version,
        "generator": "ref/gen_golden.py",
        "config_name": config_name,
        "config_hash": config_hash(config),
        "config": config,
        "agent": agent,
        "seed_env": int(seed_env),
        "seed_policy": int(seed_policy),
        "n_steps": int(n_steps),
        "tolerance": GOLDEN_TOLERANCE,
        "obs_names": obs_names,
        "steps": steps,
    }


def write_golden_file(document: dict[str, object], path: Path) -> None:
    """Write one golden document to disk.

    Takes: `document` from `build_golden_document`; `path`, the destination file. Returns: `None`.

    Written with `json.dump(..., indent=2, sort_keys=False, allow_nan=False)` plus a trailing
    newline: `indent=2` so a diff between two generations is readable, `sort_keys=False` so the key
    order of `GOLDEN_SCHEMA` survives, and `allow_nan=False` so a NaN or an infinity anywhere in a
    trajectory fails loudly here instead of becoming the non-standard `NaN` token that a strict JSON
    reader would then reject. Floats are written with Python's default round-trip repr, so reading
    the file back reproduces the exact double - the 1e-9 tolerance of `GOLDEN_TOLERANCE` is for the
    implementation under test, not for the file format.

    Creates the parent directory if it does not exist. Overwrites unconditionally: regeneration is
    a lead action under WO-013 with a `spec/CHANGELOG.md` entry (CONTRACT rule 1), and the guard
    against an accidental regeneration is that policy plus `--check`, not a read-only file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, indent=1, sort_keys=False)
        handle.write("\n")


def generate_all(
    out_dir: Path,
    n_steps: int,
    spec_version: str,
    config_names: list[str] | None,
    seeds: list[int] | None,
    agents: list[str] | None,
) -> list[Path]:
    """Generate every cell of the matrix and write the files.

    Takes: `out_dir`, the destination directory (`GOLDEN_DIR` by default); `n_steps`
    (`GOLDEN_N_STEPS`); `spec_version` from `read_spec_version`; and three optional filters -
    `config_names`, `seeds`, `agents` - each `None` meaning "all of them". Returns: the list of
    paths written, in generation order.

    For every (configuration, seed, agent) triple it calls
    `ref_rollout(config, seed_env, seed_policy, make_policy(agent, config), n_steps)`, wraps the
    records with `build_golden_document`, and writes with `write_golden_file`. The full matrix is
    `GOLDEN_N_CONFIGS * len(GOLDEN_SEEDS) * len(GOLDEN_AGENTS) = 30` files (PLAN section 11).
    `seed_policy` is set equal to `seed_env` for each cell (see `GOLDEN_SEEDS`).

    Determinism is the whole product: nothing here may depend on iteration order of a set, on a
    dictionary built from a set, on the wall clock, on the process id or on an environment variable.
    Two runs of this function on the same checkout must produce byte-identical files - that is what
    makes `--check` meaningful and what lets a reviewer confirm a regeneration was mechanical.

    Binds: the whole `tests/golden/` suite (T-B7).
    """
    written: list[Path] = []
    for name, config in golden_configs():
        if config_names is not None and name not in config_names:
            continue
        for seed_env in GOLDEN_SEEDS:
            if seeds is not None and seed_env not in seeds:
                continue
            for agent in GOLDEN_AGENTS:
                if agents is not None and agent not in agents:
                    continue
                records = ref_rollout(
                    config, seed_env, seed_env, make_policy(agent, config), n_steps
                )
                document = build_golden_document(
                    name, config, agent, seed_env, seed_env, n_steps, spec_version, records
                )
                path = out_dir / golden_filename(name, seed_env, agent)
                write_golden_file(document, path)
                written.append(path)
    return written


def check_all(
    out_dir: Path,
    n_steps: int,
    spec_version: str,
) -> list[str]:
    """Regenerate the matrix in memory and report differences against the files on disk.

    Takes: the same `out_dir`, `n_steps` and `spec_version` as `generate_all`. Returns: a list of
    human-readable difference descriptions - empty when every file on disk is exactly what this
    generator would produce now. Writes nothing.

    This is what `--check` runs. It answers one question: are the golden files on disk still the
    output of the current `ref/ref_step.py`? A non-empty result means either the reference changed
    (regeneration is due, under WO-013 with a `spec/CHANGELOG.md` entry) or a file was edited by
    hand (a CONTRACT rule 2 violation - golden files are frozen and are never hand-edited). The
    report says which files differ and in which key; it does not repair anything.

    Missing files and extra files both count as differences, so a deleted cell or a stray file from
    an aborted `--config` run is caught.
    """
    stale: list[Path] = []
    for name, config in golden_configs():
        for seed_env in GOLDEN_SEEDS:
            for agent in GOLDEN_AGENTS:
                path = out_dir / golden_filename(name, seed_env, agent)
                if not path.exists():
                    stale.append(path)
                    continue
                existing = json.loads(path.read_text(encoding="utf-8"))
                records = ref_rollout(
                    config, seed_env, seed_env, make_policy(agent, config), n_steps
                )
                fresh = build_golden_document(
                    name, config, agent, seed_env, seed_env, n_steps, spec_version, records
                )
                if existing != fresh:
                    stale.append(path)
    return stale


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point - `make golden` runs `python -m ref.gen_golden`.

    Takes: `argv`, the argument list (`None` means `sys.argv[1:]`). Returns: a process exit code -
    `0` on success, non-zero when `--check` finds a difference or when a precondition fails.

    Intended behaviour, in order:
      1. parse the arguments documented in the module docstring with `argparse` (the flags, their
         defaults and `--force`'s meaning are specified there; **the parsing is not implemented in
         this skeleton**);
      2. refuse a run that departs from the PLAN section 11 matrix - a `--steps` other than
         `GOLDEN_N_STEPS`, or a filtered set of configurations, seeds or agents - unless `--force`
         is given, and say so on stderr naming PLAN section 11;
      3. read the spec version with `read_spec_version` unless `--spec-version` overrides it;
      4. dispatch to `check_all` when `--check` is given, otherwise to `generate_all`;
      5. print one line per file written or per difference found, and return the exit code.

    It prints a reminder on every successful generation: golden files are frozen for implementers
    (CONTRACT rule 2), `tests/golden/*.json` is git-ignored, and a regeneration needs a
    `spec/CHANGELOG.md` entry (CONTRACT rule 1, WO-013). It does **not** run the test suite, commit
    anything, or touch `spec/CHANGELOG.md` itself: those are lead decisions, and a generator that
    quietly writes the changelog entry for you defeats the rule.
    """
    parser = argparse.ArgumentParser(
        prog="python -m ref.gen_golden",
        description="Generate or check the golden matrix from the reference dynamics.",
    )
    parser.add_argument("--out-dir", type=Path, default=GOLDEN_DIR)
    parser.add_argument("--n-steps", type=int, default=GOLDEN_N_STEPS)
    parser.add_argument("--config", action="append", dest="config_names")
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--agent", action="append", dest="agents")
    parser.add_argument("--check", action="store_true", help="report stale files, write nothing")
    args = parser.parse_args(argv)

    spec_version = read_spec_version(Path(__file__).resolve().parent.parent / "spec" / "spec.py")
    if args.check:
        stale = check_all(args.out_dir, args.n_steps, spec_version)
        for path in stale:
            print(f"stale or missing: {path}")
        print(f"gen_golden: {len(stale)} stale or missing file(s)")
        return 1 if stale else 0

    written = generate_all(
        args.out_dir, args.n_steps, spec_version, args.config_names, args.seeds, args.agents
    )
    for path in written:
        print(f"wrote {path}")
    print(f"gen_golden: {len(written)} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
