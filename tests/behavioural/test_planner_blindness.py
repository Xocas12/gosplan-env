"""T-B4 - planner blindness: no true quantity reaches a planner rule.

Realises: PLAN section 11 (behavioural test T-B4) and CONTRACT rule 5, read against PLAN sections
2.4 (the planner information invariant), 2.2 (the true state), 2.7.1-2.7.5 (the planner rules and
the three information filters) and 10 (`PlannerView`'s declared fields). Owning work order:
**WO-002**; on the must-pass list of **WO-006** (`gosplan/env/planner.py`).

CONTRACT rule 5 (PLANNER BLINDNESS): planner rules take a `PlannerView` and nothing else;
`PlannerView` is built by `make_planner_view()` and contains no true quantity; **any planner
function whose signature accepts `State` is a violation**.

T-B4 has two halves (PLAN section 11, verbatim): *construct a state with `S != R` and `y` sentinel
values; `PlannerView` contains none of the sentinels; static check that no function in `planner.py`
accepts `State` except `make_planner_view`.*

    1. DYNAMIC. Plant sentinel magnitudes in the fields no planner rule may see - `inv_output`
       (`S`), `inv_inputs` (`X`), `cum_output` (`y`) - with `S != R` so that a view that leaked
       stock could not be mistaken for one that carried the claim, then build the view and assert
       no sentinel survives anywhere in it, under a sweep of sentinel values.
    2. STATIC. Parse `gosplan/env/planner.py` and assert that no function in it takes a `State`,
       except the two names in `STATE_ARGUMENT_WHITELIST`.

Why the whitelist has two entries. `make_planner_view` is the single `State -> planner` boundary.
`deliver(state, alloc, cfg)` also lives in `gosplan/env/planner.py` and also takes a `State`, but
it makes no planner decision: it is the physical execution of an allocation already decided from
the view, and it reads no claim except through `alloc` (see `deliver`'s interface note in
`spec/spec.py`). The spec records that T-B4 whitelists it, and that the lead either records the
whitelist or relocates `deliver` to `gosplan/env/step.py` in `spec/CHANGELOG.md` at the v1 freeze
(WO-013). The whitelist is asserted to contain exactly those two names, so a third can only appear
by editing this frozen test - which CONTRACT rule 2 forbids an implementer from doing.

Implementing the static check (WO-002). Read `gosplan/env/planner.py` as text and `ast.parse` it;
walk every `ast.FunctionDef` and `ast.AsyncFunctionDef`, including methods, and inspect each
argument's annotation, rendering it with `ast.unparse`. A function is a violation when any
annotation's *terminal name* is `State` - so `State`, `spec.State`, `Optional[State]`,
`list[State]`, `"State"` as a string annotation and `State | None` are all caught. Prefer `ast`
over `inspect`: this module carries `from __future__ import annotations`, which makes every
runtime annotation a string, so `inspect.signature` alone would compare text and
`typing.get_type_hints` would have to import and resolve the module. The `ast` route also needs no
import, so it still runs while every body raises `NotImplementedError` - which is the point of a
structural check. If an `inspect`-based cross-check is added, resolve annotations with
`typing.get_type_hints(func, include_extras=True)` and compare against the `State` class object,
never against the string.

Scope: the check is on `gosplan/env/planner.py`, the module CONTRACT rule 5 and the WO-006 card
name. Planner rules that later move elsewhere move the check with them, which is a spec change
under CONTRACT rule 1, not a test edit.
"""

from __future__ import annotations

import pytest

SKIP_REASON = (
    "skeleton: T-B4 assertions are written by WO-002 (frozen tests); they bind WO-006 "
    "(make_planner_view and the planner rules of gosplan/env/planner.py)"
)
"""Reason attached to every `@pytest.mark.skip` below; the bodies arrive with WO-002."""

PLANNER_MODULE_PATH = "gosplan/env/planner.py"
"""The module CONTRACT rule 5 and the WO-006 card name; the static check parses this file."""

STATE_ARGUMENT_WHITELIST: tuple[str, ...] = ("make_planner_view", "deliver")
"""The only functions in `PLANNER_MODULE_PATH` that may take a `State`. `make_planner_view` is the
single `State -> planner` boundary of CONTRACT rule 5; `deliver` is physical execution of an
already-decided allocation and is whitelisted by the interface note in `spec/spec.py`, to be
recorded in `spec/CHANGELOG.md` at the v1 freeze (WO-013). The tests assert this tuple's exact
contents, so widening it is a deliberate edit of a frozen test (CONTRACT rule 2)."""

STATE_TYPE_NAME = "State"
"""The terminal annotation name the static check looks for, however it is qualified or wrapped."""

SENTINEL_VALUES: tuple[float, ...] = (-8.123456789e5, 3.141592653e6, 9.876543210e7)
"""Sentinel magnitudes planted in the true-only fields of the state. Three of them, swept: a single
value could coincide with a legitimate view entry by accident, while a leak would have to coincide
for all three. They are far outside every plausible plan quantity (targets are of order 1) and are
exactly representable enough to survive an equality comparison at `SENTINEL_TOL`."""

SENTINEL_TOL = 1e-9
"""Tolerance for "no view entry equals a sentinel": `abs(entry - sentinel) <= SENTINEL_TOL`."""

TRUE_ONLY_STATE_FIELDS: tuple[str, ...] = (
    "inv_output",
    "inv_inputs",
    "cum_output",
    "cum_cost",
    "quality_acc",
)
"""The `State` fields (PLAN section 2.2) that carry true quantities no planner rule may see: own
stock `S`, input stocks `X`, this period's true production and cost, and the quality accumulator.
The sentinel sweep writes into all of them."""

PLANNER_VIEW_FIELDS: tuple[str, ...] = (
    "claims",
    "requests",
    "audited",
    "audit_meas",
    "measured_quality",
    "targets",
    "planner_io",
    "downstream_shortfall",
    "aggregation_level",
    "plan_prices",
)
"""`PlannerView`'s complete declared field set (PLAN section 10 / `spec/spec.py`). The field test
asserts the record carries exactly these, so a new field cannot appear without an edit to the
frozen spec and to this frozen test - the two places CONTRACT rules 1 and 2 protect."""


@pytest.mark.skeleton
@pytest.mark.skip(reason=SKIP_REASON)
@pytest.mark.parametrize("sentinel", SENTINEL_VALUES)
def test_planner_view_contains_no_sentinel(sentinel: float) -> None:
    """No sentinel planted in a true-only state field reaches the `PlannerView`.

    Build a `State` at `p1_default_config()` in which every field of `TRUE_ONLY_STATE_FIELDS` is
    filled with `sentinel` (varying it per enterprise by a small integer offset, so a view that
    leaked one element rather than the array is caught too), and set the report fields so that
    `S != R`: `last_report` at the targets, `last_report_ratio` at 1.0, with the sentinel stock
    left in `inv_output`. Call `make_planner_view(state, cfg)` and assert, for every numeric field
    of `PLANNER_VIEW_FIELDS` and every element of it,

        abs(element - sentinel_variant) > SENTINEL_TOL     for every planted variant

    Assert in addition that `view.claims` equals the claims that were actually reported (so the
    test is not passing because the view is empty) and that `view.claims != state.inv_output`
    element-wise - the `S != R` construction of PLAN section 11.

    A sentinel appearing anywhere in the view is a CONTRACT rule 5 violation: the planner would be
    keying on a quantity it does not observe, which is finding F6 and the reason `welfare` is not
    an `ObjectiveMetric`. Owning WO: **WO-002**; binds **WO-006**.
    """
    raise NotImplementedError("PLAN section 11 (T-B4) - implemented in WO-002")


@pytest.mark.skeleton
@pytest.mark.skip(reason=SKIP_REASON)
def test_planner_view_fields_are_exactly_the_declared_set() -> None:
    """`PlannerView` carries exactly the ten planner-side fields and no more.

    Assert that the field names of the `PlannerView` dataclass, in declaration order, equal
    `PLANNER_VIEW_FIELDS`, and that the record is frozen (`dataclasses.fields` on a frozen
    dataclass; assignment raises). Assert that none of `TRUE_ONLY_STATE_FIELDS` appears among them
    and that no field name contains `welfare` or `val_` (CONTRACT rule 6).

    This is the structural half of the dynamic check: the sentinel sweep shows that today's values
    do not leak, while this shows that no *place* to leak them was added. A new planner-side field
    is a spec change under CONTRACT rule 1 with a `spec/CHANGELOG.md` entry, and it must arrive
    with an update to this frozen test by the lead (CONTRACT rule 2). Owning WO: **WO-002**.
    """
    raise NotImplementedError("PLAN section 11 (T-B4) - implemented in WO-002")


@pytest.mark.skeleton
@pytest.mark.skip(reason=SKIP_REASON)
def test_no_planner_function_accepts_state() -> None:
    """Static check: no function in `gosplan/env/planner.py` takes a `State`, bar the whitelist.

    `ast.parse` the source of `PLANNER_MODULE_PATH` (resolved relative to the repository root, not
    to the current working directory) and walk every `ast.FunctionDef` and `ast.AsyncFunctionDef`,
    methods included. For each, render every argument annotation with `ast.unparse` - positional,
    keyword-only, `*args`, `**kwargs` - and extract the terminal names, so that `State`,
    `spec.State`, `Optional[State]`, `list[State]`, `State | None` and the string annotation
    `"State"` are all recognised. Assert

        {f.name for f in functions if STATE_TYPE_NAME in terminal_names(f)}
            == set(STATE_ARGUMENT_WHITELIST)

    as set equality, not containment: it fails both when a new function takes a `State` and when a
    whitelisted one stops doing so or is renamed, either of which means the boundary moved without
    the lead noticing. Assert separately that `STATE_ARGUMENT_WHITELIST` is exactly
    `("make_planner_view", "deliver")`, so widening the whitelist requires editing this frozen test
    (CONTRACT rule 2), and report every offending function name in the failure message.

    The check reads source and imports nothing, so it holds while every body still raises
    `NotImplementedError`; that is deliberate, since CONTRACT rule 5 is a claim about signatures,
    not about behaviour. Owning WO: **WO-002**; binds **WO-006**.
    """
    raise NotImplementedError("PLAN section 11 (T-B4) - implemented in WO-002")


@pytest.mark.skeleton
@pytest.mark.skip(reason=SKIP_REASON)
def test_planner_rules_take_only_a_planner_view() -> None:
    """The planner rules' signatures name `PlannerView`, `EnvConfig` and scalars - nothing else.

    For `update_targets`, `fulfilment_measure`, `allocate` and `select_audits` in
    `PLANNER_MODULE_PATH`, assert from the same `ast` walk that each argument's annotation is one
    of `PlannerView`, `EnvConfig` or a builtin scalar (`int`, `float`, `bool`, `str`), and that
    none names `State`, `StepInfo`, `Ledger` or `StepRecord`. Assert that `make_planner_view` is
    the only function in the module whose *first* argument is annotated `State`, with `deliver`'s
    whitelisted `State` argument reported alongside it in the failure message.

    CONTRACT rule 5 says planner rules take a `PlannerView` "and nothing else"; the previous test
    catches the `State` route and this one catches the others - a rule that reached the ledger, a
    `StepInfo` or another module's record would be just as blind a violation. Owning WO:
    **WO-002**; binds **WO-006**.
    """
    raise NotImplementedError("PLAN section 11 (T-B4) - implemented in WO-002")
