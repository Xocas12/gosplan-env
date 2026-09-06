"""Run ledger and run manifest - one row per enterprise per agent-step, plus CONTRACT rule 10.

Realises: PLAN section 2.2 (the state columns every row carries), PLAN section 4 (the ledger is the
single input to every operationalisation of PLAN section 4.1 and to the reconciliation estimator of
PLAN section 7.3), PLAN section 4.4 (the measurement window is applied by the readers of this
ledger, never by the writer - the ledger stores everything), CONTRACT rule 8 (bounds are results:
the `at_bound` column and the `BOUND_BINDING` flag) and CONTRACT rule 10 (the manifest).
Owning work order: **WO-011** (`tests/unit/test_ledger.py`; parquet round-trip, every rule-10
manifest field present, `BOUND_BINDING` logic - test T-B8).

Direction of information flow. The environment writes `StepInfo` (PLAN section 2.5, WO-009), whose
`StepRecord`s land here; metrics and lead-run experiments read them. **No agent, no policy and no
reward term may read a ledger** (CONTRACT rule 6): the rows carry `welfare`, `val_true` and
`val_measured`, which are logged and never observed. The WO-010 forbidden list states the same
boundary from the other side ("any agent reading `StepInfo`").

Binding to `spec/spec.py`: `spec/spec.py` is not importable as a package, so `StepRecord`, `Ledger`
and `write_manifest` are *defined* here and must stay field-for-field and signature-for-signature
identical to the frozen interface (CONTRACT rule 1). `tests/unit/test_spec_imports.py` (WO-001)
enforces the surface; `tests/unit/test_ledger.py` enforces the behaviour.

Dependencies. `pyarrow` is the parquet writer and `pandas` the frame type of `to_dataframe`; both
are runtime dependencies of *this module only* and are imported **inside** the methods that need
them (WO-011), never at module scope, so the skeleton imports in an environment that has neither.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from gosplan.config import EnvConfig

Phase = Literal["produce", "report"]
"""Agent-step phase within a plan period (PLAN section 2.5), restated from `spec/spec.py`.

It is restated rather than imported so that `gosplan/metrics/` imports nothing from `gosplan/env/`:
the ledger sits strictly downstream of the environment and must not create a cycle between the two
packages. The runtime definition an environment module uses (`gosplan/env/state.py`, WO-009) and
this one must be the same two literals; a unit test compares them."""

BOUND_BINDING_FLAG: str = "BOUND_BINDING"
"""The flag name of CONTRACT rule 8, raised on the run and carried into the manifest's `flags`
field. Spelled once here so no caller can invent a variant spelling."""

AT_BOUND_FLAG_THRESHOLD: float = 0.01
"""CONTRACT rule 8: the fraction of reports sitting at `rho_max` above which the run manifest is
flagged `BOUND_BINDING` - 1%. The threshold and the bound `cfg.tech.report_max_ratio` are both
results, never dials: a run that trips this flag is reported *with* the flag, and neither number is
moved to clear it. Bound to test T-B8 (`tests/behavioural/`) and to the G2 hygiene criterion of
PLAN section 4.5 ("`BOUND_BINDING` flag absent in all runs")."""

MANIFEST_FIELDS: tuple[str, ...] = (
    "config_hash",
    "config",
    "spec_version",
    "git_hash",
    "seed_env",
    "seed_policy",
    "reference_ppo_version",
    "estimator_version",
    "estimator_backend",
    "llm_models",
    "solver",
    "solver_version",
    "solver_optimality_gap",
    "bunching_settings",
    "flags",
)
"""Every key `runs/<hash>/manifest.json` carries, in this order (CONTRACT rule 10):

    config_hash            `cfg.hash()` - the SHA-256 of the canonical JSON encoding, which also
                           names the run directory (PLAN section 3, WO-003)
    config                 the full configuration as canonical JSON, so a result can be replayed
                           without the code that produced it
    spec_version           `SPEC_VERSION` of the interface the run was written against, so a result
                           is never silently attributed to another version (CONTRACT rules 1, 10)
    git_hash               git hash of the working tree, with a dirty marker when it is not clean
    seed_env               root environment seed (PLAN section 2.15) - shared across arms, which is
                           what makes common random numbers hold by construction
    seed_policy            root policy seed, kept strictly separate (CONTRACT rule 9)
    reference_ppo_version  pinned version of the reference PPO the adapter wraps (WO-017)
    estimator_version      version of the estimator surface used for PLAN section 4.1 row 1 -
                           `forensics_core.__version__` or `FALLBACK_ESTIMATOR_VERSION`
    estimator_backend      which of the two was resolved, from `gosplan.metrics.resolve_estimators`
                           (PLAN section 7.3); the companion to `estimator_version`, so a bunching
                           number can always be traced to the code that produced it
    llm_models             LLM model ids and versions used, pinned (PLAN section 7.4); `null`
                           outside the LLM ministry study
    solver                 MIP solver used by the oracle (PLAN section 6.2)
    solver_version         its version string, verbatim
    solver_optimality_gap  the oracle's relative MIP gap at termination
    bunching_settings      the pre-registered estimator settings of PLAN section 4.5 actually used,
                           from the constants in `gosplan/metrics/phenomena.py` (WO-016 notes:
                           "settings hard-coded as defaults and recorded in the manifest")
    flags                  every flag raised by the run, notably `BOUND_BINDING`

A field that does not apply to a run is written as `null`, never omitted, so a missing key is
always a bug and never an ambiguity. `tests/unit/test_ledger.py` asserts every one of these keys is
present."""


@dataclass
class StepRecord:
    """One row of the ledger: one enterprise, one agent-step (PLAN sections 2.2, 4; WO-011).

    Field-for-field identical to `StepRecord` in `spec/spec.py`; a unit test enforces the identity.
    Fields are grouped: identifiers, the PLAN section 2.2 state columns, then the WO-011 additions
    (`y, R, rho, S_pre, S_post, audited, S_hat, f, Pen, fill, deliv, consumer, val_measured,
    val_true, welfare, at_bound`), then the action and conservation columns.

    It carries the true quantities that make the per-period conservation identity of test T-U1
    checkable and the reconciliation estimator of PLAN section 7.3 computable - which is exactly why
    no agent may read it (CONTRACT rule 6).

    `J = cfg.supply.n_sectors`; tuple-valued fields have length `J`. Owning WO: **WO-011**.
    """

    run_hash: str  # EnvConfig.hash() of the run that produced this row
    episode: int
    t_period: int
    k_step: int
    phase: Phase
    enterprise: int
    sector: int

    target: float  # T_i
    capital: float  # Kap_i
    inv_output_pre: float  # S_i before this step (S_pre)
    inv_output_post: float  # S_i after this step (S_post)
    inv_inputs: tuple[float, ...]  # (J,) X_ij after this step
    cum_output: float
    cum_cost: float
    quality_acc: float
    last_report_ratio: float
    last_penalty: float
    last_fill: float
    request: tuple[float, ...]  # (J,) q_ij
    need: tuple[float, ...]  # (J,) need_ij, so request inflation q/need is recoverable

    effort: float  # e_ik, per step - the storming Gini of section 4.1 row 2 needs it
    quality: float  # q_ik; Phase 1 inert
    invest: float  # v_ik; Phase 1 inert
    output: float  # y_ik at a PRODUCE step, y_i at the REPORT step
    cost: float  # c_ik
    coverage: float  # H_ik
    reward: float  # the reward actually delivered this step

    report: float  # R_i, the claim in units
    report_ratio: float  # rho_i
    at_bound: bool  # rho_i sat at rho_max (CONTRACT rule 8, test T-B8)
    audited: bool
    audit_meas: float  # S_hat_i
    penalty_arg: float  # f_i
    penalty: float  # Pen_i charged
    fill: float  # fill_i
    shipped: float  # shipped_i
    alloc: tuple[float, ...]  # (J,) alloc_ij promised to this enterprise
    deliv: tuple[float, ...]  # (J,) deliv_ij physically received
    input_consumed: tuple[float, ...]  # (J,) inputs consumed this step
    holding_loss: float  # stock lost to h this period
    cap_overflow: float  # stock lost to S_max this period
    trade_volume: float  # executed trade volume; Phase 2, 0 in Phase 1

    consumer: tuple[float, ...]  # (J,) period-level, repeated on each row for convenience
    val_measured: float  # period-level (section 2.9.3)
    val_true: float  # period-level (section 2.9.3)
    welfare: float  # period-level (section 2.9.3)


class Ledger:
    """Append-only store of `StepRecord`s for one run, plus the run's flags (WO-011).

    One record per enterprise per agent-step, in the order the environment produced them, so the
    period schedule of PLAN section 2.5 is recoverable from the `(episode, t_period, k_step, phase)`
    columns alone. The ledger is the input to every metric of PLAN section 4.1, to the
    reconciliation test of PLAN section 7.3 and to the run manifest. Nothing an agent can read
    touches it (CONTRACT rule 6).

    Attributes, both of which are the frozen surface of `spec/spec.py`:
      `records`  every appended row, in append order;
      `flags`    the run's flag set, e.g. `BOUND_BINDING` (CONTRACT rule 8).

    The ledger stores rows unfiltered: the measurement window of PLAN section 4.4 (`t >= 2`, no
    end-of-episode exclusion under geometric termination, at-bound reports included and flagged) is
    applied by the readers in `gosplan/metrics/phenomena.py`, never by the writer. Construction
    (both attributes start empty) is WO-011's and is not part of the frozen surface.

    Owning WO: **WO-011**.
    """

    records: list[StepRecord]
    flags: set[str]

    def append(self, rec: StepRecord) -> None:
        """Append one record and maintain the run's flag set.

        Takes: `rec`, one row. Returns: `None`; `self.records` grows by one.

        Flag maintenance (CONTRACT rule 8): when the running fraction of REPORT-step rows whose
        `at_bound` is True exceeds `AT_BOUND_FLAG_THRESHOLD` (1%), `BOUND_BINDING_FLAG` is added to
        `self.flags`. The fraction is over report rows only - PRODUCE rows carry no report and must
        not dilute the denominator. Once raised the flag stays raised: it records that the bound bit
        during the run, and a bound is never silently moved to clear it.

        Binds: `tests/unit/test_ledger.py` and test T-B8 (`tests/behavioural/`) - forcing
        `rho = rho_max` in more than 1% of reports sets `BOUND_BINDING`, and at or below 1% it does
        not. Owning WO: **WO-011**.
        """
        raise NotImplementedError("PLAN section 4 - implemented in WO-011")

    def to_parquet(self, path: str) -> None:
        """Write the ledger to a columnar file.

        Takes: `path`, the destination file. Returns: `None`.

        One row per `StepRecord`, in append order, with the tuple-valued fields exploded to one
        column per good - `inv_inputs_0 ... inv_inputs_{J-1}`, and likewise `request_*`, `need_*`,
        `alloc_*`, `deliv_*`, `input_consumed_*`, `consumer_*` - so the file is readable without
        this module. Scalar columns keep their declared dtypes; `phase` is written as its literal
        string, `at_bound` and `audited` as booleans.

        `pyarrow` is the writer and is imported inside this method (PLAN section 10 keeps it out of
        the interface, and this skeleton keeps it out of module scope so the import graph stays
        numpy-only).

        Binds: `tests/unit/test_ledger.py` - a parquet round-trip preserves every column and dtype.
        Owning WO: **WO-011**.
        """
        raise NotImplementedError("PLAN section 4 - implemented in WO-011")

    def to_dataframe(self) -> object:
        """Return the ledger as an in-memory frame, with the same columns as `to_parquet`.

        Takes: nothing beyond `self`. Returns: a `pandas.DataFrame` - one row per `StepRecord`, in
        append order, tuple-valued fields exploded to one column per good exactly as `to_parquet`
        writes them, so `to_dataframe()` and `read_parquet(to_parquet(...))` agree column for
        column and dtype for dtype.

        The return type is annotated `object` because `pandas` is a runtime dependency of this
        module only and is imported inside the method; the skeleton must import with numpy alone
        (PLAN section 10). This is the entry point every phenomenon in
        `gosplan/metrics/phenomena.py` uses, and the point at which the measurement window of PLAN
        section 4.4 is applied by the caller - the frame itself is unfiltered.

        Binds: `tests/unit/test_ledger.py` - the frame's columns equal the parquet file's, and a
        round-trip through either preserves the records. Owning WO: **WO-011**.
        """
        raise NotImplementedError("PLAN section 4 - implemented in WO-011")


def bound_binding(ledger: Ledger) -> bool:
    """Report whether a run trips the `BOUND_BINDING` flag of CONTRACT rule 8.

    Takes: `ledger`, a run's ledger. Returns: `True` iff

        (number of REPORT-step rows with `at_bound` True) / (number of REPORT-step rows)
            > AT_BOUND_FLAG_THRESHOLD

    i.e. more than 1% of reports sat at `rho_max = cfg.tech.report_max_ratio`. PRODUCE rows are not
    counted in either term. A ledger with no report rows returns `False`.

    This is the same predicate `Ledger.append` maintains incrementally; it exists so an experiment
    or a test can ask the question of a finished ledger without replaying the appends, and so the
    two answers can be compared. The flag string to raise is `BOUND_BINDING_FLAG`, and it is
    written into the manifest's `flags` field (CONTRACT rule 10).

    CONTRACT rule 8: the bound is a *result*. A run that returns `True` here is reported with the
    flag displayed next to its numbers; the bound is never widened or narrowed to make the flag go
    away, and the G2 hygiene criterion of PLAN section 4.5 requires the flag to be absent from all
    Phase-1 gate runs.

    Binds: test T-B8 (`tests/behavioural/`) and `tests/unit/test_ledger.py`. Owning WO: **WO-011**.
    """
    raise NotImplementedError("PLAN section 4 - implemented in WO-011")


def write_manifest(run_dir: str, cfg: EnvConfig, extra: dict) -> None:
    """Write `runs/<hash>/manifest.json` for a run (CONTRACT rule 10).

    Takes: `run_dir`, the run directory (named by `cfg.hash()`); `cfg`, the configuration the run
    used; `extra`, a mapping supplying the run-specific values of `MANIFEST_FIELDS` that the
    configuration alone does not determine - `git_hash`, `reference_ppo_version`,
    `estimator_version`, `estimator_backend`, `llm_models`, `solver`, `solver_version`,
    `solver_optimality_gap`, `bunching_settings` and `flags`. Returns: `None`; the file is written
    as JSON with sorted keys.

    The manifest carries exactly the keys of `MANIFEST_FIELDS`, whose docstring defines each one:
    config hash and the full configuration; `SPEC_VERSION`; the git hash of the working tree; both
    seeds (`seed_env`, `seed_policy`); the reference-PPO version; the estimator version and which
    backend supplied it (`forensics_core` or the vendored fallback, from
    `gosplan.metrics.resolve_estimators`); the LLM model ids and versions; the solver, its version
    and its optimality gap (PLAN section 6.2); the pre-registered bunching settings of PLAN section
    4.5 actually used; and every flag raised, notably `BOUND_BINDING` (CONTRACT rule 8).

    A field that does not apply to a run - no LLM, no oracle solve - is written as `null` rather
    than omitted, so a missing field is always a bug and never an ambiguity. An unknown key in
    `extra` is an error rather than a silent extra column: a manifest that quietly accepted
    `estimatorversion` would claim provenance it does not have.

    Binds: `tests/unit/test_ledger.py` - every rule-10 field is present, and `null` appears where a
    field does not apply. Owning WO: **WO-011**.
    """
    raise NotImplementedError("PLAN section 4 - implemented in WO-011")
