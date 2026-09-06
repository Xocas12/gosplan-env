#!/usr/bin/env python3
"""Static enforcement of `CONTRACT.md` - the rules a linter can actually see.

WHY THIS FILE EXISTS. The whole delegation model of this repository assumes that an implementer
session cannot quietly violate the contract: cards are handed out one at a time, the implementer
reads only its whitelist, and nobody reviews every line of every diff. `CONTRACT.md` is therefore
load-bearing, and a contract nobody mechanises is a contract nobody keeps. This script mechanises
the subset of the thirteen rules that is decidable from source text, and it is deliberately the
most important piece of CI in the repository. `scripts/README.md` states, rule by rule, what is
checked here and what is *not* - rules 3, 7 (in general), 8, 10, 11 and 12 need a human.

WHAT THIS FILE IS NOT. It is CI infrastructure, not environment code. The skeleton discipline that
makes every body under `gosplan/` raise `NotImplementedError` does not apply to it: it is fully
implemented, it implements no environment dynamics, and it is not part of the frozen surface of
CONTRACT rule 2.

DEPENDENCIES. Standard library only - `argparse`, `ast`, `dataclasses`, `os`, `pathlib`, `re`,
`subprocess`, `sys`, `tomllib`. CI must be able to run this on a bare Python 3.12 before the
project's own dependencies resolve, so a third-party import here would defeat the purpose.
`tomllib` is standard library from Python 3.11, and this project requires 3.12, so parsing
`pyproject.toml` properly rather than scraping it costs nothing.

USAGE.

    python scripts/contract_guard.py                        # every whole-tree check
    python scripts/contract_guard.py --all                  # identical; states the scope
    python scripts/contract_guard.py --base origin/main     # ... plus the diff-sensitive checks
    python scripts/contract_guard.py --base origin/main --format github

Exit status is 0 when clean and 1 when any violation is reported.

STRUCTURE. Every rule is one `check_*(root, changed) -> list[Violation]` function, independently
callable and independently unit-testable (`tests/unit/test_contract_guard.py` builds a synthetic
tree per rule). `changed` is the set of repository-relative paths in the diff against `--base`, or
`None` when no diff was requested or none could be computed; the two diff-sensitive checks
(CONTRACT rules 1 and 2) return no violations when it is `None`, and every whole-tree check ignores
it by design - a violation introduced by an earlier commit must not be waved through by a pull
request that happens not to touch it.

THE STRING/COMMENT DISTINCTION IS THE POINT. Several of these checks look for names that the
docstrings under `gosplan/` legitimately and repeatedly mention, because those docstrings quote the
rule that forbids them: `gosplan/env/reward.py` names `RunningMeanStd` as FORBIDDEN,
`gosplan/env/obs.py` holds `"welfare_true"` in its `NEVER_OBSERVED` tuple, and
`gosplan/env/planner.py` explains at length that no `numpy.random` call may appear under
`gosplan/env/`. Flagging those would make the guard useless within a day. Every content check
therefore walks the AST and inspects only `ast.Name`, `ast.Attribute`, `ast.keyword`, `ast.alias`
and (where noted) `ast.arg` nodes: a string literal is an `ast.Constant` and a comment is not in
the tree at all, so both are exempt without a single special case.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

# =================================================================================================
# Violation record and output formats
# =================================================================================================


@dataclass(frozen=True)
class Violation:
    """One contract violation, anchored at a file and a line.

    `rule` is the CONTRACT rule number as a string ("1", "2", "4", "5", "6", "7", "9", "13") or one
    of the two pseudo-rules this guard also enforces: `HELD-OUT` (the rule-7 corollary of PLAN
    sections 4.1 and 4.2 - the four held-out phenomena) and `PLAN-LEAK` (PLAN.md is git-ignored and
    no tracked file embeds a local absolute path).

    `severity` is `error` everywhere except CONTRACT rule 2, which a script cannot decide on its
    own and which therefore reports `review-required`. Both exit non-zero; the distinction is
    printed so a reviewer can tell a mechanical failure from a request for a human decision.
    """

    rule: str
    path: str
    line: int
    message: str
    fix: str
    severity: str = "error"


RULE_ORDER: tuple[str, ...] = ("1", "2", "4", "5", "6", "7", "9", "13", "HELD-OUT", "PLAN-LEAK")
"""Report order: the numbered rules in `CONTRACT.md` order, then the two pseudo-rules."""


def _sort_key(v: Violation) -> tuple[int, str, int]:
    """Deterministic report order, so CI output diffs cleanly between runs."""
    index = RULE_ORDER.index(v.rule) if v.rule in RULE_ORDER else len(RULE_ORDER)
    return (index, v.path, v.line)


def format_text(v: Violation) -> str:
    """Render one violation for a human: the finding, then the one-line fix."""
    tag = "" if v.severity == "error" else f" [{v.severity}]"
    return f"{v.path}:{v.line}: [rule {v.rule}]{tag} {v.message}\n    fix: {v.fix}"


def format_github(v: Violation) -> str:
    """Render one violation as a GitHub Actions workflow-command annotation.

    `::error file=PATH,line=N,title=Rule K::MSG`. Newlines and `%` are percent-escaped, or the
    annotation is truncated at the first line break.
    """
    body = v.message if v.severity == "error" else f"[{v.severity}] {v.message}"
    body = f"{body} -- fix: {v.fix}"
    body = body.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    return f"::error file={v.path},line={max(v.line, 1)},title=Rule {v.rule}::{body}"


def resolve_format(requested: str | None) -> str:
    """Pick the output format: the flag if given, else `github` under GitHub Actions, else text."""
    if requested is not None:
        return requested
    if os.environ.get("GITHUB_ACTIONS", "").strip().lower() == "true":
        return "github"
    return "text"


def _notice(message: str) -> None:
    """Print a non-fatal notice to stderr.

    Notices are how this guard degrades gracefully: a missing `git`, a shallow clone or an
    unresolvable base ref makes a check unrunnable, not failed. They never affect the exit status.
    """
    print(f"contract_guard: note: {message}", file=sys.stderr)


# =================================================================================================
# Small helpers: paths, git, AST
# =================================================================================================

_SKIP_DIRS = frozenset(
    {".git", "__pycache__", ".venv", "venv", ".pytest_cache", ".ruff_cache", "node_modules"}
)


def _rel(root: Path, path: Path) -> str:
    """Repository-relative POSIX path, so violations read the same on Windows and on Linux."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _python_files(root: Path, subdir: str) -> list[Path]:
    """Every `*.py` under `root/subdir`, sorted, with cache and virtualenv directories dropped."""
    base = root / subdir
    if not base.is_dir():
        return []
    return sorted(p for p in base.rglob("*.py") if not _SKIP_DIRS & set(p.parts))


def _read(path: Path) -> str | None:
    """Read a text file, returning `None` for anything that is not decodable UTF-8 text."""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _parse(path: Path) -> ast.Module | None:
    """Parse one Python file, or return `None`.

    A syntax error is reported by ruff and by pytest collection, both of which run alongside this
    guard in CI; re-reporting it here as a contract violation would be noise, so the file is
    skipped with a notice instead.
    """
    text = _read(path)
    if text is None:
        _notice(f"{path} is not readable as UTF-8 text; skipped")
        return None
    try:
        return ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        _notice(f"{path} does not parse ({exc}); skipped - ruff and pytest will report it")
        return None


def _git(root: Path, *args: str) -> tuple[int, str]:
    """Run a read-only git command in `root`. Returns `(returncode, stdout)`; 127 means no git.

    This guard never mutates repository state: every invocation below is `rev-parse`, `ls-files`,
    `diff --name-only` or `show`.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return 127, ""
    return proc.returncode, proc.stdout


def git_available(root: Path) -> bool:
    """True when `git` runs and `root` is itself the top level of a work tree."""
    code, out = _git(root, "rev-parse", "--show-toplevel")
    if code != 0 or not out.strip():
        return False
    try:
        top = os.path.normcase(str(Path(out.strip()).resolve()))
        here = os.path.normcase(str(root.resolve()))
    except OSError:
        return False
    return top == here


def tracked_files(root: Path) -> list[str] | None:
    """Repository-relative paths of every tracked file, or `None` when git is unavailable."""
    if not git_available(root):
        return None
    code, out = _git(root, "ls-files")
    if code != 0:
        return None
    return [line.strip() for line in out.splitlines() if line.strip()]


def changed_files(root: Path, base: str) -> set[str] | None:
    """`git diff --name-only BASE...HEAD` as a set, or `None` when it cannot be computed.

    Degrades in three steps, because a CI checkout is often shallow and a fork's base ref is often
    absent: the three-dot (merge-base) diff first, then the two-dot diff, then a notice and `None`.
    A `None` here disables the diff-sensitive checks; it never fails the run.
    """
    if not git_available(root):
        _notice("git is unavailable or this is not a work tree; diff-sensitive checks skipped")
        return None
    shallow_code, shallow_out = _git(root, "rev-parse", "--is-shallow-repository")
    if shallow_code == 0 and shallow_out.strip() == "true":
        _notice(
            "this is a shallow clone; if the diff below looks empty or wrong, fetch with "
            "fetch-depth: 0 so the merge base with the base ref exists"
        )
    for spec in (f"{base}...HEAD", f"{base}..HEAD"):
        code, out = _git(root, "diff", "--name-only", spec)
        if code == 0:
            return {line.strip().replace("\\", "/") for line in out.splitlines() if line.strip()}
    _notice(f"cannot diff against {base!r}; diff-sensitive checks skipped")
    return None


def _dotted(node: ast.AST) -> str | None:
    """Render an attribute chain rooted at a plain name, e.g. `np.random.default_rng`."""
    parts: list[str] = []
    cur: ast.AST = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return None


def _annotation_names(node: ast.AST | None) -> set[str]:
    """Every terminal identifier appearing in a type annotation.

    `State`, `spec.State`, `Optional[State]`, `list[State]`, `State | None` and the string
    annotation `"State"` all yield `State`. String annotations are re-parsed rather than pattern
    matched, because `from __future__ import annotations` is on in every module under `gosplan/`
    and would otherwise hide the whole check behind quotation marks.
    """
    names: set[str] = set()
    if node is None:
        return names
    stack: list[ast.AST] = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, ast.Name):
            names.add(cur.id)
        elif isinstance(cur, ast.Attribute):
            names.add(cur.attr)
        elif isinstance(cur, ast.Constant) and isinstance(cur.value, str):
            names |= _string_annotation_names(cur.value)
            continue
        stack.extend(ast.iter_child_nodes(cur))
    return names


def _string_annotation_names(text: str) -> set[str]:
    """Terminal identifiers of a string annotation; a word scan if it will not parse."""
    try:
        parsed = ast.parse(text, mode="eval")
    except SyntaxError:
        return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text))
    return _annotation_names(parsed.body)


def _signature_names(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Terminal identifiers of every annotation in a signature, return annotation included."""
    a = fn.args
    slots = [*a.posonlyargs, *a.args, *a.kwonlyargs]
    if a.vararg is not None:
        slots.append(a.vararg)
    if a.kwarg is not None:
        slots.append(a.kwarg)
    names: set[str] = set()
    for slot in slots:
        names |= _annotation_names(slot.annotation)
    names |= _annotation_names(fn.returns)
    return names


def _executable_names(tree: ast.AST, *, include_params: bool = False) -> list[tuple[str, int]]:
    """`(identifier, line)` for every identifier that is executable code, never text.

    Collected: `ast.Name` (a read or a write of a name), `ast.Attribute` (its terminal attribute),
    `ast.keyword` (a keyword argument's name at a call site) and `ast.alias` (an imported or
    aliased name). With `include_params`, also `ast.arg` - a parameter *named* after a forbidden
    quantity is exactly the violation CONTRACT rule 6 describes.

    NOT collected, deliberately: `ast.Constant`. A docstring, a string literal and a comment are
    not executable code, and every module under `gosplan/` names the forbidden constructs in prose
    precisely because the contract forbids them.
    """
    out: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.append((node.id, node.lineno))
        elif isinstance(node, ast.Attribute):
            out.append((node.attr, node.lineno))
        elif isinstance(node, ast.keyword) and node.arg is not None:
            out.append((node.arg, getattr(node, "lineno", 0)))
        elif isinstance(node, ast.alias):
            out.append((node.asname or node.name.split(".")[-1], getattr(node, "lineno", 0)))
        elif include_params and isinstance(node, ast.arg):
            out.append((node.arg, node.lineno))
    return out


def _called_names(tree: ast.AST) -> list[tuple[str, int]]:
    """`(callee, line)` for every call whose callee is a plain or a dotted name."""
    out: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            out.append((node.func.id, node.lineno))
        elif isinstance(node.func, ast.Attribute):
            out.append((node.func.attr, node.lineno))
    return out


# =================================================================================================
# Rule 1 - FROZEN SPEC
# =================================================================================================

SPEC_PATH = "spec/spec.py"
SPEC_CHANGELOG_PATH = "spec/CHANGELOG.md"
_VERSION_HEADING_RE = re.compile(r"^#{1,6}\s*\[?v?(\d+\.\d+(?:\.\d+)?)\]?", re.MULTILINE)
_MAIN_REFS: tuple[str, ...] = ("main", "origin/main")


def _changelog_versions(text: str) -> set[str]:
    """Version numbers of every `## <version>` heading in a changelog body."""
    return set(_VERSION_HEADING_RE.findall(text))


def _changelog_on_main(root: Path) -> str | None:
    """`spec/CHANGELOG.md` as it stands on `main`, falling back to `origin/main`, else `None`.

    THERE IS DELIBERATELY NO FALLBACK TO `HEAD`. The baseline exists to answer "is this version
    heading new?", and `HEAD` is the branch being judged: taking it as the baseline makes the
    difference unconditionally empty and turns a correct spec revision into a guaranteed false
    rule-1 violation on any checkout where neither `main` nor `origin/main` resolves - a detached
    HEAD, a single-branch clone under another name, a fork whose default branch is not `main`.
    Returning `None` instead lets `check_spec_freeze` skip that half with a notice, which is what
    this guard's degradation contract promises everywhere else.
    """
    for ref in _MAIN_REFS:
        code, out = _git(root, "show", f"{ref}:{SPEC_CHANGELOG_PATH}")
        if code == 0:
            return out
    return None


def check_spec_freeze(root: Path, changed: set[str] | None) -> list[Violation]:
    """CONTRACT rule 1 (FROZEN SPEC): a spec change needs a new `spec/CHANGELOG.md` entry.

    Rule 1, verbatim: "spec/spec.py is provisional (v0) until gate G1 and frozen (v1) thereafter.
    After v1, only the lead may change it, and only with a spec/CHANGELOG.md entry (version,
    reason, affected work orders). No other session edits spec/spec.py."

    Mechanised as: if `spec/spec.py` is in the diff against the base ref, then
    `spec/CHANGELOG.md` must be in the same diff, and it must carry a version heading that is not
    already present on `main`. Editing the changelog without bumping the version - appending prose
    to an existing entry - is the failure mode the second half catches.

    "Only the lead may change it" is not decidable from source; that half of rule 1 is enforced by
    review and branch protection, and `scripts/README.md` says so.
    """
    if changed is None or SPEC_PATH not in changed:
        return []
    fix_add = (
        f"add a {SPEC_CHANGELOG_PATH} entry in the format that file's 'Required entry format' "
        "section fixes: version, reason, change, affected work orders, golden files, suite, "
        "approver."
    )
    if SPEC_CHANGELOG_PATH not in changed:
        return [
            Violation(
                rule="1",
                path=SPEC_PATH,
                line=1,
                message=(
                    "CONTRACT rule 1 (FROZEN SPEC): spec/spec.py changed in this diff but "
                    f"{SPEC_CHANGELOG_PATH} did not. Rule 1: the spec may change 'only with a "
                    "spec/CHANGELOG.md entry (version, reason, affected work orders)'."
                ),
                fix=fix_add,
            )
        ]
    text = _read(root / SPEC_CHANGELOG_PATH)
    if text is None:
        return [
            Violation(
                rule="1",
                path=SPEC_CHANGELOG_PATH,
                line=1,
                message=(
                    "CONTRACT rule 1 (FROZEN SPEC): spec/CHANGELOG.md is in the diff but cannot "
                    "be read as UTF-8 text, so the required entry cannot be verified."
                ),
                fix=fix_add,
            )
        ]
    baseline = _changelog_on_main(root)
    if baseline is None:
        _notice(
            f"cannot read {SPEC_CHANGELOG_PATH} from main or origin/main; the rule-1 "
            "new-version-heading half is skipped (the entry-present half still ran)"
        )
        return []
    if _changelog_versions(text) - _changelog_versions(baseline):
        return []
    heading = _VERSION_HEADING_RE.search(text)
    line = text[: heading.start()].count("\n") + 1 if heading else 1
    return [
        Violation(
            rule="1",
            path=SPEC_CHANGELOG_PATH,
            line=line,
            message=(
                "CONTRACT rule 1 (FROZEN SPEC): spec/spec.py changed and spec/CHANGELOG.md was "
                "edited, but it carries no version heading that is not already on main. Rule 1 "
                "requires an entry with a version, so that a spec revision is citable by number."
            ),
            fix=(
                f"add a new '## <version> - <YYYY-MM-DD>' section to {SPEC_CHANGELOG_PATH} and "
                "bump SPEC_VERSION in spec/spec.py to match."
            ),
        )
    ]


# =================================================================================================
# Rule 2 - FROZEN TESTS
# =================================================================================================

FROZEN_TEST_DIRS: tuple[str, ...] = ("tests/unit/", "tests/behavioural/", "tests/golden/")
FROZEN_TEST_NON_FROZEN_PATHS: frozenset[str] = frozenset({"tests/unit/test_contract_guard.py"})
"""Files that LIVE inside a frozen test directory but are not part of the frozen surface.

CONTRACT rule 2 freezes `tests/unit`, `tests/behavioural` and `tests/golden` because they hold the
frozen expectations the environment implementation is measured against (PLAN section 11). This
guard's own test suite sits in `tests/unit/` only because that is where pytest looks for it: it
makes no claim about environment behaviour, it is on no work order's must-pass list, and it is
edited by whoever edits `scripts/contract_guard.py` - in the same pull request, or the guard and
its test drift apart.

The carve-out is STRUCTURAL rather than an entry in `.github/FROZEN_TEST_EXEMPTION`, because that
file is for temporary, reviewable suppressions the lead removes once a change has landed; a
standing entry there would be a permanently disabled rule, and the very pull request that adds this
guard would otherwise fail the guard. Every other path under the three directories stays frozen,
and this set stays this short: adding a path to it is a decision to unfreeze that file for good."""
FROZEN_TEST_EXEMPTION_PATH = ".github/FROZEN_TEST_EXEMPTION"
FROZEN_TEST_ENV_VAR = "CONTRACT_ALLOW_FROZEN_TESTS"


def _frozen_test_exemptions(root: Path) -> set[str]:
    """Paths named in `.github/FROZEN_TEST_EXEMPTION`; blank lines and `#` comments ignored."""
    text = _read(root / FROZEN_TEST_EXEMPTION_PATH)
    if text is None:
        return set()
    out: set[str] = set()
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip().replace("\\", "/")
        if line:
            out.add(line.lstrip("./"))
    return out


def _is_exempt(path: str, exemptions: set[str]) -> bool:
    """A path is exempt when it is named exactly, or lies under a named directory prefix."""
    if path in exemptions:
        return True
    return any(entry.endswith("/") and path.startswith(entry) for entry in exemptions)


def check_frozen_tests(root: Path, changed: set[str] | None) -> list[Violation]:
    """CONTRACT rule 2 (FROZEN TESTS): the three frozen test directories are read-only.

    Rule 2, verbatim: "tests/unit, tests/behavioural and tests/golden are read-only for
    implementers. If a test looks wrong, file an AMBIGUITY REPORT; do not edit it, do not skip it,
    do not special-case the implementation to pass it."

    A SCRIPT CANNOT SEE WHO OPENED THE PULL REQUEST, so it cannot apply the rule's real condition
    ("read-only for implementers" - the lead may edit them). It is mechanised instead as a
    violation of severity `review-required` on every frozen-test file in the diff except the paths
    in `FROZEN_TEST_NON_FROZEN_PATHS`, which are not part of the frozen surface at all. The
    severity is suppressible two further ways, both of which leave a trace a reviewer can see:

      * `.github/FROZEN_TEST_EXEMPTION` names the paths, one per line, `#` comments allowed, a
        trailing `/` making an entry a directory prefix. The file is committed on the branch, so
        the exemption is itself reviewable in the diff; the lead adds it when the lead is the one
        editing the frozen suite, and removes it in the same pull request or the next one.
      * `CONTRACT_ALLOW_FROZEN_TESTS=1` in the environment, for a lead running the guard locally.
        A workflow that sets this unconditionally has disabled rule 2; do not.

    THE EXEMPTION FILE IS FOR TEMPORARY SUPPRESSIONS, not for standing ones: a permanent entry on
    a frozen directory is a disabled rule. CI infrastructure that merely happens to live under
    `tests/unit/` - this guard's own `tests/unit/test_contract_guard.py` - is therefore carved out
    structurally in `FROZEN_TEST_NON_FROZEN_PATHS` instead, so that editing the guard and its test
    together needs no suppression at all.
    """
    if changed is None:
        return []
    if os.environ.get(FROZEN_TEST_ENV_VAR, "").strip() == "1":
        _notice(
            f"{FROZEN_TEST_ENV_VAR}=1 is set; CONTRACT rule 2 (FROZEN TESTS) is suppressed for "
            "this run"
        )
        return []
    exemptions = _frozen_test_exemptions(root)
    out: list[Violation] = []
    for path in sorted(changed):
        if not path.startswith(FROZEN_TEST_DIRS) or path in FROZEN_TEST_NON_FROZEN_PATHS:
            continue
        if _is_exempt(path, exemptions):
            continue
        out.append(
            Violation(
                rule="2",
                path=path,
                line=1,
                message=(
                    "CONTRACT rule 2 (FROZEN TESTS): this file is in tests/unit, "
                    "tests/behavioural or tests/golden, which are read-only for implementers - "
                    "'do not edit it, do not skip it, do not special-case the implementation to "
                    "pass it'. A script cannot see whether the lead opened this pull request, so "
                    "the change is flagged for review rather than judged."
                ),
                fix=(
                    "if a test looks wrong, file an AMBIGUITY REPORT "
                    "(workorders/AMBIGUITY_TEMPLATE.md) instead of editing it; if the lead is "
                    f"making this change, list the path in {FROZEN_TEST_EXEMPTION_PATH} on this "
                    f"branch, or run with {FROZEN_TEST_ENV_VAR}=1 locally."
                ),
                severity="review-required",
            )
        )
    return out


# =================================================================================================
# Rule 4 - REWARD TERMS
# =================================================================================================

REWARD_SCOPE_FILE = "gosplan/env/reward.py"
REWARD_SCOPE_DIR = "gosplan/agents/ppo"
REWARD_DENYLIST: frozenset[str] = frozenset(
    {
        "RunningMeanStd",
        "NormalizeReward",
        "VecNormalize",
        "reward_norm",
        "running_reward",
        "curiosity",
        "intrinsic_reward",
        "potential_based",
        "shaping",
        "bonus_shaping",
        "reward_scale_running",
    }
)
"""Shaping and running-normalisation constructs CONTRACT rule 4 forbids outright. Matched as whole
identifiers and never as substrings: `reward_normalisation: false` in a PPO manifest entry is a
*record* that rule 4 was honoured, and `reward_scale` is the analytic per-configuration constant
that rule 4 mandates."""


def _reward_scope_files(root: Path) -> list[Path]:
    """`gosplan/env/reward.py` plus every module under `gosplan/agents/ppo/`."""
    files: list[Path] = []
    reward = root / REWARD_SCOPE_FILE
    if reward.is_file():
        files.append(reward)
    files.extend(_python_files(root, REWARD_SCOPE_DIR))
    return files


def check_reward_terms(root: Path, changed: set[str] | None) -> list[Violation]:
    """CONTRACT rule 4 (REWARD TERMS): no shaping, no auxiliary reward, no running normalisation.

    Rule 4, verbatim on the point this check mechanises: "No per-step shaping, no auxiliary reward,
    no curiosity term, no potential-based term. No running reward normalisation (running statistics
    change the effective reward over training and, with heavy-tailed penalties, shrink the notch in
    normalised units). Per-batch advantage normalisation inside PPO is permitted."

    Scope: `gosplan/env/reward.py` and everything under `gosplan/agents/ppo/` - where the reward is
    computed and where it is consumed. Any executable use of a name in `REWARD_DENYLIST` is a
    violation.

    THE DOCSTRINGS UNDER `gosplan/` NAME THESE CONSTRUCTS ON PURPOSE, to say they are forbidden:
    `gosplan/env/reward.py` reproduces rule 4 in full and states that "a `RunningMeanStd` wrapper on
    rewards is a rule-4 violation", and `gosplan/agents/ppo/adapter.py` requires "no
    `RunningMeanStd` on rewards or returns, no `NormalizeReward` wrapper". Only `ast.Name`,
    `ast.Attribute`, `ast.keyword` and `ast.alias` nodes are inspected, so prose is exempt and
    executable code is not.

    Not decidable here, and stated in `scripts/README.md`: a shaping term written under an innocent
    name, an extra addend in `enterprise_reward`, or a denylisted wrapper imported under an alias.
    Those are what `tests/unit/test_ppo_adapter.py` (inspection of the wrapped object) and test
    T-B6 (the reward recomputed independently from the formula) are for.
    """
    out: list[Violation] = []
    for path in _reward_scope_files(root):
        tree = _parse(path)
        if tree is None:
            continue
        rel = _rel(root, path)
        for name, line in _executable_names(tree):
            if name not in REWARD_DENYLIST:
                continue
            out.append(
                Violation(
                    rule="4",
                    path=rel,
                    line=line,
                    message=(
                        f"CONTRACT rule 4 (REWARD TERMS): executable use of {name!r} in {rel}. "
                        "The enterprise reward is exactly '-scale * c_ik' at a production step "
                        "and 'scale * (B(rho) - 1[audited]*Pen + trade_surplus)' at the report "
                        "step: no per-step shaping, no auxiliary reward, no curiosity term, no "
                        "potential-based term, no running reward normalisation."
                    ),
                    fix=(
                        "delete the term or the wrapper; scale rewards only with the analytic "
                        "reward_scale(cfg) = 1 / B_cfg(1.1). Per-batch advantage normalisation "
                        "inside PPO is permitted and is the only normalisation rule 4 allows."
                    ),
                )
            )
    return out


# =================================================================================================
# Rule 5 - PLANNER BLINDNESS
# =================================================================================================

PLANNER_PATH = "gosplan/env/planner.py"
PLANNER_BLINDNESS_TEST_PATH = "tests/behavioural/test_planner_blindness.py"
STATE_TYPE_NAME = "State"
PLANNER_STATE_WHITELIST: tuple[str, ...] = ("make_planner_view", "deliver")
"""The only functions in `gosplan/env/planner.py` that may take a `State`.

CONTRACT rule 5 names ONE: "Any planner function whose signature accepts State is a violation",
with `make_planner_view` the single `State -> planner` boundary. `deliver` is the documented second
entry: it is the physical execution of an allocation already decided from the view, it makes no
planner decision, and the interface note in `spec/spec.py` records that the frozen behavioural test
T-B4 whitelists it - to be recorded in `spec/CHANGELOG.md`, or relocated to `gosplan/env/step.py`,
at the v1 freeze (WO-013).

So the guard and the frozen test can never drift apart, `check_planner_blindness` also reads
`STATE_ARGUMENT_WHITELIST` out of `tests/behavioural/test_planner_blindness.py` and reports a
rule-5 violation if the two disagree. Widening the whitelist therefore takes an edit to a frozen
test (CONTRACT rule 2, which this guard also flags) *and* an edit here."""


def _module_literal(tree: ast.Module, name: str) -> tuple[str, ...] | None:
    """Value of a module-level assignment to `name`, when it is a literal tuple/list of strings."""
    for node in tree.body:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        if not any(isinstance(t, ast.Name) and t.id == name for t in targets):
            continue
        if node.value is None:
            continue
        try:
            literal = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            return None
        if isinstance(literal, (tuple, list)) and all(isinstance(x, str) for x in literal):
            return tuple(literal)
    return None


def check_planner_blindness(root: Path, changed: set[str] | None) -> list[Violation]:
    """CONTRACT rule 5 (PLANNER BLINDNESS): planner rules take a `PlannerView` and nothing else.

    Rule 5, verbatim: "Planner rules take a PlannerView and nothing else. PlannerView is built by
    make_planner_view() and contains no true quantity. Any planner function whose signature accepts
    State is a violation."

    Mechanised as: `ast.parse` `gosplan/env/planner.py`, walk every `def` and `async def` - module
    level, nested and methods alike - and extract the terminal identifiers of every parameter
    annotation and of the return annotation. A function whose signature mentions `State` and whose
    name is not in `PLANNER_STATE_WHITELIST` is a violation. `State`, `spec.State`,
    `Optional[State]`, `list[State]`, `State | None` and the string annotation `"State"` are all
    caught, which is why the check is on annotation *terminal names* rather than on the
    `from ..state import State` import line: the import is inert until a signature uses it, and
    aliasing it would defeat a line-based check.

    The check also asserts that the whitelist here matches `STATE_ARGUMENT_WHITELIST` in the frozen
    test `tests/behavioural/test_planner_blindness.py`, so the static guard and test T-B4 cannot
    drift apart silently.

    Not decidable here: whether `make_planner_view` actually leaks a true quantity into the view it
    returns. That is T-B4's dynamic half - sentinels planted in a `State`, asserted absent from
    every field of the returned `PlannerView`.
    """
    path = root / PLANNER_PATH
    if not path.is_file():
        return []
    tree = _parse(path)
    if tree is None:
        return []
    out: list[Violation] = []
    allowed = ", ".join(PLANNER_STATE_WHITELIST)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if STATE_TYPE_NAME not in _signature_names(node):
            continue
        if node.name in PLANNER_STATE_WHITELIST:
            continue
        out.append(
            Violation(
                rule="5",
                path=PLANNER_PATH,
                line=node.lineno,
                message=(
                    f"CONTRACT rule 5 (PLANNER BLINDNESS): {node.name}() in {PLANNER_PATH} has a "
                    f"signature naming {STATE_TYPE_NAME}. 'Planner rules take a PlannerView and "
                    "nothing else ... Any planner function whose signature accepts State is a "
                    f"violation.' The only functions here that may take a State are: {allowed}."
                ),
                fix=(
                    "take a PlannerView built by make_planner_view(state, cfg) instead, and read "
                    "the quantity from the view; if the rule genuinely needs a true quantity, "
                    "that is an AMBIGUITY REPORT, not a signature change."
                ),
            )
        )
    test_path = root / PLANNER_BLINDNESS_TEST_PATH
    test_tree = _parse(test_path) if test_path.is_file() else None
    if test_tree is not None:
        declared = _module_literal(test_tree, "STATE_ARGUMENT_WHITELIST")
        if declared is not None and declared != PLANNER_STATE_WHITELIST:
            out.append(
                Violation(
                    rule="5",
                    path=PLANNER_BLINDNESS_TEST_PATH,
                    line=1,
                    message=(
                        "CONTRACT rule 5 (PLANNER BLINDNESS): the State whitelist in the frozen "
                        f"test T-B4 is {declared} but scripts/contract_guard.py enforces "
                        f"{PLANNER_STATE_WHITELIST}. The static guard and the frozen test must "
                        "name the same boundary, or one of them is not enforcing rule 5."
                    ),
                    fix=(
                        "the boundary moves only by a lead decision recorded in "
                        "spec/CHANGELOG.md; make the two lists agree in the same pull request as "
                        "that entry."
                    ),
                )
            )
    return out


# =================================================================================================
# Rule 6 - WELFARE BLINDNESS
# =================================================================================================

OBS_PATH = "gosplan/env/obs.py"
GOSPLAN_DIR = "gosplan"
WELFARE_NAMES: frozenset[str] = frozenset({"welfare_true", "val_measured", "val_true", "welfare"})
AGENT_INPUT_FUNCTIONS: frozenset[str] = frozenset(
    {"act", "forward", "observe", "build_observation"}
)


def _welfare_whole_file_scope(root: Path) -> list[Path]:
    """The files rule 6 scans WHOLE: the observation builder, and the reward scope.

    Rule 6 names three destinations - "any observation, reward, or agent input". The observation
    and the agent input had a scope each; the reward clause had none, so the most literal possible
    violation, `return -cfg.scale * state.welfare_true` in `gosplan/env/reward.py`, used to pass
    clean. The reward scope is the same one CONTRACT rule 4 uses (`REWARD_SCOPE_FILE` plus
    `REWARD_SCOPE_DIR`): where the reward is computed, and where it is consumed.
    """
    files: list[Path] = []
    obs = root / OBS_PATH
    if obs.is_file():
        files.append(obs)
    files.extend(_reward_scope_files(root))
    return files


def check_welfare_blindness(root: Path, changed: set[str] | None) -> list[Violation]:
    """CONTRACT rule 6 (WELFARE BLINDNESS): logged quantities reach no observation, reward or agent.

    Rule 6, verbatim: "welfare_true and val_measured are logged and never appear in any
    observation, reward, or agent input. The PPO adapter's forward pass takes obs only."

    Rule 6 names three destinations, and each gets a scope:

      * OBSERVATION - every executable reference to `welfare_true`, `val_measured`, `val_true` or
        `welfare` anywhere in `gosplan/env/obs.py`; the observation builder must not be able to see
        them at all;
      * REWARD - every such reference anywhere in `gosplan/env/reward.py` or under
        `gosplan/agents/ppo/`, the same scope CONTRACT rule 4 uses. Scanning these whole rather
        than function by function is what catches the literal violation
        `return -cfg.scale * state.welfare_true`, which reaches a learner through the reward
        channel without touching an observation or an agent entry point; and
      * AGENT INPUT - every such reference inside a function named `act`, `forward`, `observe` or
        `build_observation` anywhere else under `gosplan/`.

    `gosplan/env/obs.py` holds `"welfare_true"` and `"val_measured"` as STRING LITERALS in its
    `NEVER_OBSERVED` tuple, so that test T-B5 can iterate the list and plant a sentinel in each,
    and `gosplan/env/reward.py` quotes rule 6 in prose for the same reason it quotes rule 4. That
    is the rule being documented, not broken, and it must not fire: only `ast.Name`,
    `ast.Attribute`, `ast.keyword`, `ast.alias` and `ast.arg` nodes are inspected here.

    Not decidable here: a true quantity smuggled into an observation or a reward term under another
    name, or a derived one - "no function of `consumer`, no function of another enterprise's
    `cum_output`". That is test T-B5's sentinel sweep, which is dynamic by necessity, and test
    T-B6, which recomputes the reward independently from the formula.
    """
    out: list[Violation] = []
    fix = (
        "log the quantity through the ledger instead (gosplan/metrics/ledger.py); it may enter no "
        "observation, no reward term, no planner rule and no agent input."
    )
    whole_file = _welfare_whole_file_scope(root)
    scanned_whole = {_rel(root, path) for path in whole_file}
    for path in whole_file:
        tree = _parse(path)
        if tree is None:
            continue
        rel = _rel(root, path)
        for name, line in _executable_names(tree, include_params=True):
            if name not in WELFARE_NAMES:
                continue
            out.append(
                Violation(
                    rule="6",
                    path=rel,
                    line=line,
                    message=(
                        f"CONTRACT rule 6 (WELFARE BLINDNESS): executable reference to {name!r} "
                        f"in {rel}. 'welfare_true and val_measured are logged and never appear in "
                        "any observation, reward, or agent input.'"
                    ),
                    fix=fix,
                )
            )
    for path in _python_files(root, GOSPLAN_DIR):
        rel = _rel(root, path)
        if rel in scanned_whole:
            continue  # already covered, whole-file, above
        tree = _parse(path)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name not in AGENT_INPUT_FUNCTIONS:
                continue
            for name, line in _executable_names(node, include_params=True):
                if name not in WELFARE_NAMES:
                    continue
                out.append(
                    Violation(
                        rule="6",
                        path=rel,
                        line=line,
                        message=(
                            f"CONTRACT rule 6 (WELFARE BLINDNESS): executable reference to "
                            f"{name!r} inside {node.name}() in {rel}. That function is an agent "
                            "entry point; the PPO adapter's forward pass takes obs only."
                        ),
                        fix=fix,
                    )
                )
    return out


# =================================================================================================
# Rule 7 - NO HARD-CODED PATHOLOGY (narrow, high-precision)
# =================================================================================================

ENV_DIR = "gosplan/env"
PATHOLOGY_NAMES: frozenset[str] = frozenset(
    {"bunch", "padding", "pad_amount", "storm", "hoard", "shave", "blat", "reserve_target"}
)
RULE_7_NOTE = (
    "CONTRACT rule 7 (NO HARD-CODED PATHOLOGY) IS NOT DECIDABLE STATICALLY. The check in this "
    "guard is narrow and high-precision by design: it catches a transition rule that NAMES the "
    "pathology it is supposed to let emerge, and nothing else. PASSING IT IS NECESSARY, NOT "
    "SUFFICIENT - rule 7's own wording is 'passing it is necessary, not sufficient - the lead "
    "reviews every env/ diff against this rule'. Behavioural test T-B1 "
    "(tests/behavioural/test_no_hardcoded_pathology.py) is the other half, and it is not "
    "sufficient either."
)


def _pathology_sites(tree: ast.AST) -> list[tuple[str, int, str]]:
    """`(identifier, line, how)` for every name assigned to or branched on in a module."""
    out: list[tuple[str, int, str]] = []

    def emit(node: ast.AST, how: str) -> None:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name):
                out.append((sub.id, sub.lineno, how))
            elif isinstance(sub, ast.Attribute):
                out.append((sub.attr, sub.lineno, how))

    for node in ast.walk(tree):
        targets: list[ast.AST] = []
        tests: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            targets = [node.target]
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            targets = [node.target]
        elif isinstance(node, ast.withitem):
            targets = [node.optional_vars] if node.optional_vars is not None else []
        elif isinstance(node, (ast.If, ast.While, ast.IfExp)):
            tests = [node.test]
        elif isinstance(node, ast.comprehension):
            tests = list(node.ifs)
        elif isinstance(node, ast.Match):
            tests = [node.subject]
        for target in targets:
            emit(target, "assigned to")
        for test in tests:
            emit(test, "branched on")
    return out


def check_hardcoded_pathology(root: Path, changed: set[str] | None) -> list[Violation]:
    """CONTRACT rule 7 (NO HARD-CODED PATHOLOGY) - the narrow, decidable corner of it.

    Rule 7, verbatim: "No transition rule or reward term may implement bunching, padding, storming,
    hoarding, shaving or trade directly. tests/behavioural/test_no_hardcoded_pathology.py checks
    this behaviourally with heuristic agents; passing it is necessary, not sufficient - the lead
    reviews every env/ diff against this rule."

    THIS ONE CANNOT BE DECIDED STATICALLY, AND THIS CHECK DOES NOT PRETEND TO. It flags exactly one
    thing: executable code under `gosplan/env/` that ASSIGNS TO or BRANCHES ON an identifier
    literally named `bunch`, `padding`, `pad_amount`, `storm`, `hoard`, `shave`, `blat` or
    `reserve_target` - a transition rule naming the pathology it is meant to let emerge. A rule
    that computes the same thing under a neutral name passes this check and violates rule 7 all the
    same; see `RULE_7_NOTE`, which the guard prints on every run.

    The docstrings under `gosplan/env/` discuss padding, shaving, storming, hoarding and blat
    constantly - `deliver` is even described as "the padding-to-shortage channel" - because those
    modules explain why such outcomes must be consequences of the four lines of PLAN section 2.7.3
    and never rules. Prose is exempt; assignments and branch conditions are not.
    """
    out: list[Violation] = []
    for path in _python_files(root, ENV_DIR):
        tree = _parse(path)
        if tree is None:
            continue
        rel = _rel(root, path)
        for name, line, how in _pathology_sites(tree):
            if name not in PATHOLOGY_NAMES:
                continue
            out.append(
                Violation(
                    rule="7",
                    path=rel,
                    line=line,
                    message=(
                        f"CONTRACT rule 7 (NO HARD-CODED PATHOLOGY): {name!r} is {how} in {rel}. "
                        "'No transition rule or reward term may implement bunching, padding, "
                        "storming, hoarding, shaving or trade directly.' A transition rule that "
                        "names the pathology it is supposed to let emerge is that pathology being "
                        "written in rather than measured."
                    ),
                    fix=(
                        "delete the branch or the variable and let the outcome fall out of the "
                        "PLAN section 2.7 formulas; if the mechanism genuinely needs it, that is "
                        "an AMBIGUITY REPORT to the lead, who reviews every gosplan/env/ diff "
                        "against rule 7 by hand."
                    ),
                )
            )
    return out


# =================================================================================================
# Rule 9 - RNG
# =================================================================================================

RNG_DOTTED_PREFIXES: tuple[str, ...] = ("numpy.random", "np.random", "jax.random")
RNG_TERMINAL_NAMES: frozenset[str] = frozenset({"default_rng"})
RNG_IMPORT_MODULES: frozenset[str] = frozenset(
    {"random", "numpy.random", "jax.random", "numpy.random.mtrand"}
)


def check_rng(root: Path, changed: set[str] | None) -> list[Violation]:
    """CONTRACT rule 9 (RNG): environment randomness goes through `gosplan.rng.draw`.

    Rule 9, verbatim: "All environment randomness goes through rng.draw(seed_env, purpose,
    *indices). No direct calls to numpy.random / jax.random in gosplan/env/. seed_policy is
    separate."

    Under `gosplan/env/`, an executable reference to `numpy.random`, `np.random`, `jax.random`, the
    stdlib `random` module or `default_rng` is a violation - as an attribute chain, as an import,
    or as a bare name. Key-based drawing is what makes a trajectory reproducible from
    `(seed_env, purpose, indices)` alone and independent of call order (test T-U6), so one
    stream-based generator anywhere in the environment silently destroys the golden files and the
    common-random-numbers design that every baseline comparison rests on.

    A type annotation naming `np.random.Generator` under `gosplan/env/` trips this check too, and
    that is intended: environment code holds no `Generator` at all, module-level generators are
    forbidden outright (WO-004), and the policy-side stream lives behind `seed_policy` in the agent
    layer, which this check does not scan.

    Every module under `gosplan/env/` states in prose that no `numpy.random` call may appear there;
    the AST walk never sees a docstring, so those statements are exempt.
    """
    out: list[Violation] = []
    fix = (
        "draw through gosplan.rng.draw(seed_env, purpose, *indices) with a purpose string, so the "
        "draw is deterministic in (seed_env, purpose, indices) and independent of call order."
    )
    for path in _python_files(root, ENV_DIR):
        tree = _parse(path)
        if tree is None:
            continue
        rel = _rel(root, path)
        seen: set[tuple[int, str]] = set()
        found: list[tuple[int, str]] = []

        def note(line: int, what: str, seen: set = seen, found: list = found) -> None:
            if (line, what) not in seen:
                seen.add((line, what))
                found.append((line, what))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in RNG_IMPORT_MODULES:
                        note(node.lineno, f"'import {alias.name}'")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module in RNG_IMPORT_MODULES:
                    note(node.lineno, f"'from {module} import ...'")
                elif module in {"numpy", "jax"}:
                    for alias in node.names:
                        if alias.name == "random":
                            note(node.lineno, f"'from {module} import random'")
                for alias in node.names:
                    if alias.name in RNG_TERMINAL_NAMES:
                        note(node.lineno, f"'{alias.name}'")
            elif isinstance(node, ast.Attribute):
                dotted = _dotted(node)
                if dotted is None:
                    if node.attr in RNG_TERMINAL_NAMES:
                        note(node.lineno, f"'{node.attr}'")
                    continue
                if any(dotted.startswith(prefix) for prefix in RNG_DOTTED_PREFIXES):
                    note(node.lineno, f"'{dotted}'")
                elif dotted.split(".")[0] == "random":
                    note(node.lineno, f"'{dotted}' (the stdlib random module)")
                elif node.attr in RNG_TERMINAL_NAMES:
                    note(node.lineno, f"'{dotted}'")
            elif isinstance(node, ast.Name) and node.id in RNG_TERMINAL_NAMES:
                note(node.lineno, f"'{node.id}'")

        for line, what in found:
            out.append(
                Violation(
                    rule="9",
                    path=rel,
                    line=line,
                    message=(
                        f"CONTRACT rule 9 (RNG): executable use of {what} in {rel}. 'All "
                        "environment randomness goes through rng.draw(seed_env, purpose, "
                        "*indices). No direct calls to numpy.random / jax.random in gosplan/env/.'"
                    ),
                    fix=fix,
                )
            )
    return out


# =================================================================================================
# Rule 13 - TESTS ARE NOT EXPERIMENTS
# =================================================================================================

ACCEPTANCE_DIR = "tests/acceptance"
PYTEST_COLLECTED_RE: tuple[re.Pattern[str], ...] = (
    re.compile(r"^test_.+\.py$"),
    re.compile(r"^.+_test\.py$"),
)
PYPROJECT_PATH = "pyproject.toml"


def _declared_testpaths(text: str) -> list[str] | None:
    """`[tool.pytest.ini_options] testpaths` as a list of strings, or `None` when absent.

    PARSED, NOT SCRAPED. A line-anchored regex over the raw text sees only `[` when the array is
    written across several lines - which is valid TOML and what most formatters produce - so the
    check silently passed on exactly the multi-line form a reformatting pull request would
    introduce. `tomllib` is standard library from Python 3.11, so this costs no dependency.

    Raises `tomllib.TOMLDecodeError` on an unparseable file; the caller turns that into a notice,
    because a broken `pyproject.toml` is ruff's and pytest's finding to report, not a contract
    violation.
    """
    data = tomllib.loads(text)
    ini = data.get("tool", {}).get("pytest", {}).get("ini_options", {})
    if not isinstance(ini, dict):
        return None
    declared = ini.get("testpaths")
    if declared is None:
        return None
    if isinstance(declared, str):  # pytest accepts a whitespace-separated string too
        return declared.split()
    if isinstance(declared, list):
        return [str(entry) for entry in declared]
    return [str(declared)]


def check_acceptance_isolation(root: Path, changed: set[str] | None) -> list[Violation]:
    """CONTRACT rule 13 (TESTS ARE NOT EXPERIMENTS): `tests/acceptance/` is never collected.

    Rule 13, verbatim: "tests/acceptance/ holds lead-run experiments (gates). Nothing there is a
    unit test, nothing there is on any work order's must-pass list, and no implementer session runs
    it."

    Two mechanical halves:

      * no file under `tests/acceptance/` matches pytest's default collection patterns
        (`test_*.py`, `*_test.py`), so a gate cannot be collected even by a bare `pytest` run from
        the repository root under a different ini file; and
      * `pyproject.toml`'s `testpaths` does not name `tests/acceptance`. The file is PARSED with
        `tomllib`, not scraped with a regex: a line-anchored pattern reads only `[` when the array
        spans several lines, so the multi-line form - valid TOML, and what most formatters produce
        - used to pass silently rather than skip with a notice. An unparseable `pyproject.toml` is
        a notice, not a violation; ruff, uv and pytest all report it themselves.

    A gate is an experiment costing GPU-hours whose result is a research finding, not a red bar. If
    CI ever collects one, the gate's outcome becomes a merge blocker and the incentive to tune it
    away appears - which is the precise failure rule 13 exists to prevent.
    """
    out: list[Violation] = []
    base = root / ACCEPTANCE_DIR
    if base.is_dir():
        for path in sorted(base.rglob("*.py")):
            if _SKIP_DIRS & set(path.parts):
                continue
            if not any(pattern.match(path.name) for pattern in PYTEST_COLLECTED_RE):
                continue
            out.append(
                Violation(
                    rule="13",
                    path=_rel(root, path),
                    line=1,
                    message=(
                        "CONTRACT rule 13 (TESTS ARE NOT EXPERIMENTS): this file under "
                        f"{ACCEPTANCE_DIR}/ matches pytest's default collection patterns "
                        "(test_*.py / *_test.py), so a bare pytest run would collect a lead-run "
                        "gate experiment as a test. 'Nothing there is a unit test, nothing there "
                        "is on any work order's must-pass list, and no implementer session runs "
                        "it.'"
                    ),
                    fix=(
                        "rename it to the gate convention already used there - "
                        "gate_<id>_<name>.py - so pytest cannot collect it under any ini file."
                    ),
                )
            )
    pyproject = _read(root / PYPROJECT_PATH)
    if pyproject is None:
        _notice("pyproject.toml is missing or unreadable; the rule-13 testpaths half is skipped")
        return out
    try:
        declared = _declared_testpaths(pyproject)
    except tomllib.TOMLDecodeError as exc:
        _notice(
            f"pyproject.toml does not parse as TOML ({exc}); the rule-13 testpaths half is "
            "skipped - ruff, uv and pytest will report it"
        )
        return out
    if declared is None:
        _notice("pyproject.toml declares no testpaths; the rule-13 testpaths half is skipped")
        return out
    if any(ACCEPTANCE_DIR in entry.replace("\\", "/") for entry in declared):
        marker = pyproject.find("testpaths")
        line = pyproject[:marker].count("\n") + 1 if marker >= 0 else 1
        out.append(
            Violation(
                rule="13",
                path=PYPROJECT_PATH,
                line=line,
                message=(
                    "CONTRACT rule 13 (TESTS ARE NOT EXPERIMENTS): pytest's testpaths names "
                    f"{ACCEPTANCE_DIR}. The gate experiments G0-G4 are never collected, by CI or "
                    "by an implementer session."
                ),
                fix=(
                    'set testpaths = ["tests/unit", "tests/behavioural", "tests/golden"] - the '
                    "three frozen categories of PLAN section 11 and nothing else."
                ),
            )
        )
    return out


# =================================================================================================
# HELD-OUT PHENOMENA - a CONTRACT rule 7 corollary, from PLAN sections 4.1 and 4.2
# =================================================================================================

HELD_OUT_FUNCTIONS: frozenset[str] = frozenset(
    {
        "phenomenon_storming",
        "phenomenon_hoarding",
        "phenomenon_blat",
        "phenomenon_hidden_reserves",
    }
)
"""Rows 2 (storming), 5 (hoarding), 6 (blat) and 7 (hidden reserves) of the PLAN section 4.1 table -
the four EMERGENCE claims. Rows 1, 3 and 4 are pipeline checks and are not held out."""

PHENOMENA_PATH = "gosplan/metrics/phenomena.py"
PHASE1_PHENOMENA: tuple[str, ...] = ("phenomenon_bunching", "phenomenon_padding")
"""The Phase-1 section of `gosplan/metrics/phenomena.py`: rows 1 and 4, owned by WO-016. The
Phase-1 region scanned below is module-level code plus these two function bodies; the Phase-2
functions may of course reference each other, since WO-030 is where they are computed."""

PHASE1_EXPERIMENT_MODULES: tuple[str, ...] = (
    "gosplan/experiments/mc_sanity.py",
    "gosplan/experiments/regime_map.py",
    "gosplan/experiments/dp_vs_ppo.py",
    "gosplan/experiments/phase1_gate.py",
    "gosplan/experiments/price_sensitivity.py",
)
"""Every experiment module that runs before the Phase-2 acceptance run: WO-012 (MC sanity), WO-015
(regime map), WO-019 (DP vs PPO), WO-020 (the Phase-1 gate), and the standing price-sensitivity
recomputation that PLAN section 7.5 applies to every headline table, Phase-1 tables included. The
list is explicit rather than inferred, because "which experiments are Phase 1" is a design fact
from PLAN section 12.3, not something a script should guess."""

HELD_OUT_COMPUTED_BY = "WO-030"


def _phase1_region_nodes(tree: ast.Module) -> list[ast.AST]:
    """Module-level statements plus the bodies of the Phase-1 phenomenon functions."""
    nodes: list[ast.AST] = []
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if stmt.name in PHASE1_PHENOMENA:
                nodes.append(stmt)
            continue
        if isinstance(stmt, ast.ClassDef):
            continue
        nodes.append(stmt)
    return nodes


def check_held_out_phenomena(root: Path, changed: set[str] | None) -> list[Violation]:
    """HELD-OUT PHENOMENA (PLAN sections 4.1 and 4.2; enforced as a CONTRACT rule 7 corollary).

    Rows 2 (storming), 5 (hoarding), 6 (blat) and 7 (hidden reserves) of the PLAN section 4.1 table
    are the emergence claims, and they are HELD OUT: no plot, table or test of these quantities may
    be produced before the Phase-2 acceptance run - not during Phase 1, and not while debugging
    their own mechanisms. The Monte Carlo sanity harness may assert conservation and boundedness on
    the same mechanisms and never a direction. Their mechanism parameters are locked now in PLAN
    section 4.2 so they cannot drift, and if a held-out phenomenon fails to appear, that is
    reported as a failure rather than tuned away.

    Mechanised as: a call to `phenomenon_storming`, `phenomenon_hoarding`, `phenomenon_blat` or
    `phenomenon_hidden_reserves` is a violation when it appears in
    `gosplan/experiments/mc_sanity.py` or any other Phase-1 experiment module
    (`PHASE1_EXPERIMENT_MODULES`), or in the Phase-1 section of `gosplan/metrics/phenomena.py` -
    module-level code plus the bodies of `phenomenon_bunching` and `phenomenon_padding`. The one
    place they may be computed is WO-030, in the Phase-2 acceptance run.

    This is the check most likely to be tripped by good intentions: it is genuinely tempting to
    plot hoarding while debugging the allocation weights. Looking is the violation. Not decidable
    here: computing the same statistic inline under another name, which the WO-012 and WO-016 cards
    forbid in words and which the lead checks in review.
    """
    out: list[Violation] = []
    fix = (
        f"delete the call. These four rows are computed for the first time in "
        f"{HELD_OUT_COMPUTED_BY}, in the Phase-2 acceptance run, and nowhere else; assert "
        "conservation and boundedness on the mechanism instead, never a direction."
    )
    targets: list[tuple[str, list[ast.AST]]] = []
    for rel in PHASE1_EXPERIMENT_MODULES:
        path = root / rel
        if not path.is_file():
            continue
        tree = _parse(path)
        if tree is not None:
            targets.append((rel, [tree]))
    phenomena = root / PHENOMENA_PATH
    if phenomena.is_file():
        tree = _parse(phenomena)
        if tree is not None:
            targets.append((PHENOMENA_PATH, _phase1_region_nodes(tree)))
    for rel, nodes in targets:
        for node in nodes:
            for name, line in _called_names(node):
                if name not in HELD_OUT_FUNCTIONS:
                    continue
                out.append(
                    Violation(
                        rule="HELD-OUT",
                        path=rel,
                        line=line,
                        message=(
                            "HELD-OUT PHENOMENA (PLAN sections 4.1, 4.2; a CONTRACT rule 7 "
                            f"corollary): {rel} calls {name}(). Rows 2 (storming), 5 (hoarding), "
                            "6 (blat) and 7 (hidden reserves) are the emergence claims and are "
                            "held out: no plot, table or test of them before the Phase-2 "
                            f"acceptance run. {HELD_OUT_COMPUTED_BY} is the only place they may "
                            "be computed."
                        ),
                        fix=fix,
                    )
                )
    return out


# =================================================================================================
# PLAN LEAK - PLAN.md stays untracked, and no tracked file embeds a local absolute path
# =================================================================================================

PLAN_PATH = "PLAN.md"

# The needles are assembled from fragments so that this file, which is itself tracked, does not
# match its own check. Do not "simplify" them back into single literals.
_WIN_USERS = "C:" + r"[\\/]" + "Users"
_LINUX_HOME = "/" + "home" + "/"
_MAC_USERS = "/" + "Users" + "/"
ABSOLUTE_PATH_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(_WIN_USERS), "a Windows user directory"),
    (re.compile(_LINUX_HOME), "a Linux home directory"),
    (re.compile(_MAC_USERS), "a macOS user directory"),
)

_SCANNABLE_SUFFIXES: frozenset[str] = frozenset(
    {
        "",
        ".cfg",
        ".ini",
        ".json",
        ".lock",
        ".md",
        ".mk",
        ".py",
        ".sh",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
)
_MAX_SCAN_BYTES = 2_000_000


def _scan_candidates(root: Path) -> list[str]:
    """Repository-relative paths to scan for absolute paths.

    Tracked files when git is available - that is the population the rule is about, since an
    untracked scratch file leaks nothing. When git is unavailable the tree is walked instead, so
    the check still means something on a synthetic tree and in a source tarball.
    """
    tracked = tracked_files(root)
    if tracked is not None:
        return tracked
    return [
        _rel(root, path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not _SKIP_DIRS & set(path.parts)
    ]


def check_plan_leak(root: Path, changed: set[str] | None) -> list[Violation]:
    """PLAN.md is never committed, and no tracked file embeds a local absolute path.

    `PLAN.md` is the build instruction for this repository. It is supplied out of band, it is not
    the output of any work order, it is deliberately git-ignored, and the repository is public: the
    guard asserts that `git ls-files --error-unmatch PLAN.md` FAILS. Without git - a source
    tarball, a synthetic tree - it falls back to asserting that `.gitignore` names it, which is the
    same property one step removed.

    The second half is a hygiene rule the public repository depends on just as much: a tracked file
    that embeds a local absolute path leaks the author's machine layout, breaks on every other
    checkout, and is the usual way a path to the out-of-band `PLAN.md` escapes into a committed
    artefact. Three needles are searched for - a Windows user directory, a Linux home directory and
    a macOS user directory - assembled from fragments in the source above so that this guard does
    not flag itself.
    """
    out: list[Violation] = []
    plan = root / PLAN_PATH
    if git_available(root):
        code, _ = _git(root, "ls-files", "--error-unmatch", PLAN_PATH)
        if code == 0:
            out.append(
                Violation(
                    rule="PLAN-LEAK",
                    path=PLAN_PATH,
                    line=1,
                    message=(
                        "PLAN.md is tracked by git. It is the out-of-band build instruction for "
                        "this repository, it is not the output of any work order, and this "
                        "repository is public: it is git-ignored deliberately and must never be "
                        "committed or un-ignored."
                    ),
                    fix=(
                        "git rm --cached PLAN.md, confirm the PLAN.md entry in .gitignore is "
                        "intact, and rewrite the history of any branch that already carries it."
                    ),
                )
            )
    elif plan.is_file():
        ignore = _read(root / ".gitignore") or ""
        if not any(line.strip() == PLAN_PATH for line in ignore.splitlines()):
            out.append(
                Violation(
                    rule="PLAN-LEAK",
                    path=".gitignore",
                    line=1,
                    message=(
                        "PLAN.md is present in the tree but .gitignore does not name it (git was "
                        "unavailable, so tracking could not be checked directly). PLAN.md is "
                        "supplied out of band and must never be committed."
                    ),
                    fix="add a line reading exactly 'PLAN.md' to .gitignore.",
                )
            )
    else:
        _notice("git is unavailable and PLAN.md is absent; the tracking half of PLAN-LEAK is inert")
    for rel in _scan_candidates(root):
        path = root / rel
        if not path.is_file() or path.suffix.lower() not in _SCANNABLE_SUFFIXES:
            continue
        try:
            if path.stat().st_size > _MAX_SCAN_BYTES:
                continue
        except OSError:
            continue
        text = _read(path)
        if text is None:
            continue
        for pattern, label in ABSOLUTE_PATH_PATTERNS:
            match = pattern.search(text)
            if match is None:
                continue
            line = text[: match.start()].count("\n") + 1
            out.append(
                Violation(
                    rule="PLAN-LEAK",
                    path=rel,
                    line=line,
                    message=(
                        f"{rel} embeds what looks like {label} - a local absolute path. Tracked "
                        "files in a public repository must be machine-independent; an absolute "
                        "path leaks the author's layout, breaks on every other checkout, and is "
                        "how a path to the out-of-band PLAN.md escapes into a committed artefact."
                    ),
                    fix=(
                        "use a repository-relative path, or resolve it at run time from "
                        "Path(__file__) or an environment variable."
                    ),
                )
            )
            break
    return out


# =================================================================================================
# Driver
# =================================================================================================

CHECKS: tuple[tuple[str, object], ...] = (
    ("rule 1 - FROZEN SPEC", check_spec_freeze),
    ("rule 2 - FROZEN TESTS", check_frozen_tests),
    ("rule 4 - REWARD TERMS", check_reward_terms),
    ("rule 5 - PLANNER BLINDNESS", check_planner_blindness),
    ("rule 6 - WELFARE BLINDNESS", check_welfare_blindness),
    ("rule 7 - NO HARD-CODED PATHOLOGY (narrow)", check_hardcoded_pathology),
    ("rule 9 - RNG", check_rng),
    ("rule 13 - TESTS ARE NOT EXPERIMENTS", check_acceptance_isolation),
    ("HELD-OUT PHENOMENA (rule 7 corollary)", check_held_out_phenomena),
    ("PLAN LEAK", check_plan_leak),
)
"""Every check, in report order. Each entry is `(label, check_*(root, changed) -> [Violation])`."""


def collect(root: Path, changed: set[str] | None) -> list[Violation]:
    """Run every check and return the violations, sorted into a stable report order."""
    found: list[Violation] = []
    for _label, check in CHECKS:
        found.extend(check(root, changed))  # type: ignore[operator]
    return sorted(found, key=_sort_key)


def run_all(root: Path, changed: set[str] | None, fmt: str = "text") -> int:
    """Run every check, print the report in `fmt`, and return the process exit status.

    `changed` is the set of repository-relative paths in the diff against the base ref, or `None`
    when no diff was requested or none could be computed. The two diff-sensitive checks (CONTRACT
    rules 1 and 2) return nothing when it is `None`; every other check runs over the whole tree
    regardless, so a violation introduced by an earlier commit is not waved through by a pull
    request that does not touch it.

    Returns 0 when clean and 1 when anything was reported, including the `review-required` severity
    that CONTRACT rule 2 produces.
    """
    violations = collect(root, changed)
    if fmt == "github":
        for v in violations:
            print(format_github(v))
        print(f"::notice title=CONTRACT rule 7::{RULE_7_NOTE}")
    else:
        for v in violations:
            print(format_text(v))
        if violations:
            print()
        print(RULE_7_NOTE)
    if changed is None:
        _notice(
            "no diff was supplied (--base), so CONTRACT rules 1 and 2 were not evaluated: they "
            "are properties of a change, not of a tree"
        )
    if violations:
        errors = sum(1 for v in violations if v.severity == "error")
        review = len(violations) - errors
        summary = f"contract_guard: {len(violations)} violation(s): {errors} error"
        if review:
            summary += f", {review} review-required"
        print(summary, file=sys.stderr)
        return 1
    print("contract_guard: clean", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The command-line interface described in this module's docstring."""
    parser = argparse.ArgumentParser(
        prog="contract_guard.py",
        description=(
            "Static enforcement of CONTRACT.md. Exit 0 when clean, 1 on any violation. See "
            "scripts/README.md for what is checked, what is not, and why."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "run every check over the whole tree. This is the default scope; the flag exists so a "
            "CI workflow can state it explicitly."
        ),
    )
    parser.add_argument(
        "--base",
        metavar="REF",
        default=None,
        help=(
            "additionally run the diff-sensitive checks (CONTRACT rules 1 and 2) against "
            "'git diff --name-only REF...HEAD'. Skipped with a notice when git is absent, the "
            "clone is shallow or the ref cannot be resolved."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("text", "github"),
        default=None,
        help=(
            "output format; defaults to 'github' when the GITHUB_ACTIONS environment variable is "
            "'true', and to 'text' otherwise."
        ),
    )
    parser.add_argument(
        "--root",
        metavar="PATH",
        default=None,
        help="repository root to check; defaults to the parent directory of scripts/.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run every check, and return the exit status."""
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parent.parent
    changed = changed_files(root, args.base) if args.base else None
    return run_all(root, changed, resolve_format(args.format))


if __name__ == "__main__":
    sys.exit(main())
