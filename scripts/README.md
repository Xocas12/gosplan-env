# `scripts/` — repository tooling

One script lives here.

| Script | What it is |
|---|---|
| `contract_guard.py` | Static enforcement of `CONTRACT.md`. Standard library only. Exit 0 clean, 1 on any violation. |

---

## `contract_guard.py`

### Why it exists

This repository is built by handing out one work-order card at a time. An implementer reads only
the files on its card's whitelist, writes only the files it names, and nobody reviews every line of
every diff. `CONTRACT.md` is what makes that safe — and a contract nobody mechanises is a contract
nobody keeps. `contract_guard.py` mechanises the subset of the thirteen rules that is decidable
from source text, so that a violation is a red build rather than a quiet fact discovered three
gates later.

It is CI infrastructure, not environment code. The skeleton discipline that makes every body under
`gosplan/` raise `NotImplementedError` does not apply to it: it is fully implemented, it implements
no environment dynamics, and it is **not** part of the frozen surface CONTRACT rule 2 protects.
Neither is its test, `tests/unit/test_contract_guard.py`, which the guard carves out by name (see
*The rule-2 exemption*, below).

### Running it

```
python scripts/contract_guard.py                       # every whole-tree check
python scripts/contract_guard.py --all                 # identical; states the scope explicitly
python scripts/contract_guard.py --base origin/main    # ... plus the diff-sensitive checks
python scripts/contract_guard.py --base origin/main --format github
python scripts/contract_guard.py --root /path/to/checkout
```

No dependencies: standard library only, Python 3.12, no `uv sync` needed first. That is deliberate
— CI must be able to run the guard before the project's own dependency resolution, and a
third-party import here would defeat the point.

* **`--all`** runs every whole-tree check. This is the default scope; the flag exists so a workflow
  can say so out loud.
* **`--base REF`** additionally runs the two diff-sensitive checks (CONTRACT rules 1 and 2) against
  `git diff --name-only REF...HEAD`. Without it they are skipped with a printed notice: they are
  properties of a *change*, not of a tree. The whole-tree checks always run over the whole tree
  even when `--base` is given, so a violation introduced by an earlier commit is never waved
  through by a pull request that happens not to touch it.
* **`--format {text,github}`** picks the output. It defaults to `github` when the `GITHUB_ACTIONS`
  environment variable is `true`, and to `text` otherwise. In `github` format each finding is a
  workflow-command annotation, `::error file=PATH,line=N,title=Rule K::MSG`, which GitHub renders
  inline on the diff.
* **`--root PATH`** points the guard at a checkout other than the one containing the script.

**Exit status** is 0 when clean and 1 when anything is reported, including the `review-required`
severity that rule 2 produces. Every violation carries a path, a line, the rule number, a message
quoting the rule, and a one-line *fix*.

**Degrading gracefully.** A missing `git`, a shallow clone, an unresolvable base ref or an
unparseable file makes a check *unrunnable*, not *failed*: the guard prints a notice to stderr and
carries on. Notices never change the exit status. If CI reports a shallow clone, check out with
`fetch-depth: 0` so the merge base with the base ref exists.

### What it checks, and which rule each check mechanises

| Check | Rule | Scope | Fires on |
|---|---|---|---|
| `check_spec_freeze` | **1** FROZEN SPEC | the diff | `spec/spec.py` changed without `spec/CHANGELOG.md` in the same diff, or with no version heading that is not already on `main` |
| `check_frozen_tests` | **2** FROZEN TESTS | the diff | any file under `tests/unit`, `tests/behavioural`, `tests/golden` in the diff, except the paths in `FROZEN_TEST_NON_FROZEN_PATHS` — severity `review-required`, suppressible (below) |
| `check_reward_terms` | **4** REWARD TERMS | `gosplan/env/reward.py`, `gosplan/agents/ppo/` | executable use of `RunningMeanStd`, `NormalizeReward`, `VecNormalize`, `reward_norm`, `running_reward`, `curiosity`, `intrinsic_reward`, `potential_based`, `shaping`, `bonus_shaping`, `reward_scale_running` |
| `check_planner_blindness` | **5** PLANNER BLINDNESS | `gosplan/env/planner.py` | any function whose signature names `State`, except `make_planner_view` and the documented `deliver`; also a disagreement between this whitelist and test T-B4's `STATE_ARGUMENT_WHITELIST` |
| `check_welfare_blindness` | **6** WELFARE BLINDNESS | whole-file: `gosplan/env/obs.py`, `gosplan/env/reward.py`, `gosplan/agents/ppo/`; elsewhere under `gosplan/`: `act` / `forward` / `observe` / `build_observation` | executable reference to `welfare_true`, `val_measured`, `val_true`, `welfare` |
| `check_hardcoded_pathology` | **7** NO HARD-CODED PATHOLOGY | `gosplan/env/` | executable code that *assigns to* or *branches on* an identifier literally named `bunch`, `padding`, `pad_amount`, `storm`, `hoard`, `shave`, `blat`, `reserve_target` |
| `check_rng` | **9** RNG | `gosplan/env/` | executable use of `numpy.random`, `np.random`, `jax.random`, the stdlib `random` module, or `default_rng` |
| `check_acceptance_isolation` | **13** TESTS ARE NOT EXPERIMENTS | `tests/acceptance/`, `pyproject.toml` | a file matching pytest's collection patterns (`test_*.py`, `*_test.py`) under `tests/acceptance/`; `testpaths` naming `tests/acceptance` |
| `check_held_out_phenomena` | **7 corollary** (PLAN §§4.1, 4.2) | `gosplan/experiments/mc_sanity.py` and the other Phase-1 experiment modules; the Phase-1 section of `gosplan/metrics/phenomena.py` | a call to `phenomenon_storming`, `phenomenon_hoarding`, `phenomenon_blat` or `phenomenon_hidden_reserves` |
| `check_plan_leak` | — | the tracked file list | `PLAN.md` tracked by git; any tracked file embedding a local absolute path (a Windows user directory, a Linux home directory, a macOS user directory) |

Each check is one function with the signature `check_*(root: Path, changed: set[str] | None) ->
list[Violation]`, callable and testable on its own; `run_all(root, changed, fmt) -> int` runs the
lot and returns the exit status.

#### Notes on the individual checks

**Rule 1.** The second half — "a version heading that is not already on `main`" — exists because
appending prose to an existing entry satisfies the letter of "a `spec/CHANGELOG.md` entry" while
leaving the revision uncitable. The guard compares the working-tree changelog against `main`, then
`origin/main`, and skips that half with a notice if neither can be read. There is deliberately **no
fallback to `HEAD`**: `HEAD` is the branch being judged, so using it as the baseline would make the
set difference unconditionally empty and turn a perfectly correct spec revision into a guaranteed
false rule-1 violation on any checkout where neither `main` nor `origin/main` resolves — a detached
HEAD, a single-branch clone under another name, a fork whose default branch is not `main`. On
GitHub, `actions/checkout` with `fetch-depth: 0` gives the job `origin/main`, which is why
`ci.yml`'s `contract` job sets it.

**Rule 4.** Names are matched as *whole identifiers*, never as substrings: `reward_normalisation:
false` in a PPO manifest entry is a **record that rule 4 was honoured**, and `reward_scale` is the
analytic per-configuration constant rule 4 *mandates*. Per-batch advantage normalisation inside PPO
is permitted and is switched on in Phase 1; running reward normalisation is not permitted anywhere.

**Rule 5.** The check is on the *terminal names* of annotations, so `State`, `spec.State`,
`Optional[State]`, `list[State]`, `State | None` and the string annotation `"State"` are all
caught — string annotations are re-parsed, because `from __future__ import annotations` is on in
every module under `gosplan/` and would otherwise hide the whole check behind quotation marks.

The whitelist has two entries. CONTRACT rule 5 names one, `make_planner_view`, the single
`State -> planner` boundary in the codebase. `deliver` is the documented second: it is the physical
execution of an allocation already decided from the view, it makes no planner decision, and the
interface note in `spec/spec.py` records that test T-B4 whitelists it — to be recorded in
`spec/CHANGELOG.md`, or relocated to `gosplan/env/step.py`, at the v1 freeze (WO-013). The guard
also reads `STATE_ARGUMENT_WHITELIST` out of `tests/behavioural/test_planner_blindness.py` and
fails if the two lists disagree, so the static guard and the frozen test cannot drift apart.
Widening the boundary therefore takes an edit to a frozen test *and* an edit to the guard *and* a
changelog entry — which is the intended amount of friction.

**Rule 6.** Rule 6 names three destinations — "any observation, **reward**, or agent input" — so
the guard gives each one a scope. `gosplan/env/obs.py` is scanned whole, and so are
`gosplan/env/reward.py` and everything under `gosplan/agents/ppo/` — the same reward scope rule 4
uses, where the reward is computed and where it is consumed. Everywhere else under `gosplan/`, only
the four agent entry points (`act`, `forward`, `observe`, `build_observation`) are scanned. The
reward scope is whole-file rather than function-scoped because the most literal violation the rule
describes, `return -cfg.scale * state.welfare_true` in `enterprise_reward`, reaches a learner
through the reward channel without touching an observation or an entry point, and
`enterprise_reward` is not one of the four names.

**Rule 7.** See *What it cannot check* below. The guard prints, on every run, a note that passing
this check is **necessary, not sufficient**.

**Rule 13.** The `testpaths` half **parses** `pyproject.toml` with `tomllib` rather than scraping
it with a regex. A line-anchored pattern captures only `[` when the array is written across several
lines — valid TOML, and what most formatters produce — so the check used to pass *silently* on
exactly the multi-line form a reformatting pull request would introduce, rather than skipping with
its notice. `tomllib` is standard library from Python 3.11, so this does not break the
standard-library-only rule. An unparseable `pyproject.toml` is a notice, not a violation: ruff, uv
and pytest all report it themselves, and re-reporting it here as a contract breach would be noise.

The other half of rule 13 — no file under `tests/acceptance/` matching pytest's collection patterns
— is also checked in CI, by the `test` job's *CONTRACT rule 13* step. That step passes `tests/`
explicitly to `pytest --collect-only` so that the command-line path overrides `testpaths` and
collection actually walks `tests/acceptance/`; without a path argument the directory is unreachable
by construction and the step can never fire.

**Held-out phenomena.** Rows 2 (storming), 5 (hoarding), 6 (blat) and 7 (hidden reserves) of the
PLAN §4.1 table are the emergence claims, and they are held out: no plot, table or test of these
quantities may be produced before the Phase-2 acceptance run — not during Phase 1, and not while
debugging their own mechanisms. The Monte Carlo sanity harness may assert conservation and
boundedness on the same mechanisms and **never a direction**. WO-012 and WO-016 are forbidden from
implementing them; **WO-030 is the only place they may be computed**, in the Phase-2 acceptance
run. Their mechanism parameters are locked in PLAN §4.2 so they cannot drift, and if a held-out
phenomenon fails to appear, that is reported as a failure rather than tuned away. This is the check
most likely to be tripped by good intentions — it is genuinely tempting to plot hoarding while
debugging the allocation weights. Looking is the violation.

The Phase-1 experiment modules are listed explicitly in `PHASE1_EXPERIMENT_MODULES`
(`mc_sanity`, `regime_map`, `dp_vs_ppo`, `phase1_gate`, `price_sensitivity`), because "which
experiments run before the Phase-2 acceptance run" is a design fact from PLAN §12.3 rather than
something a script should guess. Add to that list when a new Phase-1 experiment lands.

**PLAN leak.** `PLAN.md` is the out-of-band build instruction for this repository. It is not the
output of any work order, it is deliberately git-ignored, and this repository is public: the guard
asserts that `git ls-files --error-unmatch PLAN.md` **fails**, and without git it falls back to
asserting that `.gitignore` names it. The second half — no local absolute path in any tracked file
— is the usual way a path to the out-of-band plan escapes into a committed artefact, and it also
keeps every committed file machine-independent. The three needles are assembled from string
fragments in the source so the guard does not flag itself; do not "simplify" them back into single
literals.

### Why the checks walk the AST rather than grepping

Several of these checks look for names that the docstrings under `gosplan/` mention constantly and
legitimately, because those docstrings quote the rule that forbids them:

* `gosplan/env/reward.py` reproduces CONTRACT rule 4 in full and states that "a `RunningMeanStd`
  wrapper on rewards is a rule-4 violation";
* `gosplan/env/obs.py` holds `"welfare_true"` and `"val_measured"` as **string literals** in its
  `NEVER_OBSERVED` tuple, so that test T-B5 can iterate the list and plant a sentinel in each;
* every module under `gosplan/env/` says in prose that no `numpy.random` call may appear there;
* `deliver` is described in its own docstring as "the padding-to-shortage channel".

A grep-based guard would report all of that and be switched off within a day. Every content check
therefore `ast.parse`s the file and inspects only `ast.Name`, `ast.Attribute`, `ast.keyword`,
`ast.alias` and (for rule 6) `ast.arg` nodes. A string literal is an `ast.Constant` and a comment
is not in the tree at all, so both are exempt without a single special case. That distinction is
the point of the design, and `tests/unit/test_contract_guard.py` asserts it directly.

### The rule-2 exemption

CONTRACT rule 2 makes `tests/unit`, `tests/behavioural` and `tests/golden` read-only **for
implementers**; the lead may edit them. A script cannot see who opened a pull request, so it cannot
apply that condition. The check therefore reports every frozen-test file in the diff at severity
`review-required` — with one structural carve-out, and two suppressions for genuine lead edits.

**The carve-out: `FROZEN_TEST_NON_FROZEN_PATHS`.** One file is not part of the frozen surface at
all, and is skipped by name rather than suppressed:

```
FROZEN_TEST_NON_FROZEN_PATHS: frozenset[str] = frozenset({"tests/unit/test_contract_guard.py"})
```

Rule 2 freezes those three directories because they hold the frozen expectations the environment
implementation is measured against (PLAN §11). This guard's own test suite sits in `tests/unit/`
only because that is where pytest looks for it: it makes no claim about environment behaviour, it
is on no work order's must-pass list, and it is edited by whoever edits `contract_guard.py` — in
the same pull request, or the guard and its test drift apart. It is a **permanent** resident of the
directory, and the exemption file below is for **temporary** suppressions, so a standing entry
there would be a permanently disabled rule; the carve-out is structural for that reason, and
because otherwise the very pull request that adds this guard fails it. Every other path under the
three directories stays frozen, and the set stays this short: adding a path to it is a decision to
unfreeze that file for good, and it belongs in a `spec/CHANGELOG.md`-style written decision, not in
a passing thought.

The two suppressions, both of which leave a trace:

1. **`.github/FROZEN_TEST_EXEMPTION`** — a committed text file naming the exempt paths, one per
   line. `#` starts a comment; blank lines are ignored; an entry ending in `/` is a directory
   prefix. Because the file is committed on the branch, the exemption shows up in the diff and is
   reviewable alongside the change it covers. Example:

   ```
   # WO-013: lead is regenerating the golden matrix at the v1 spec freeze.
   tests/golden/
   tests/unit/test_spec_imports.py
   ```

   Remove entries once the change has landed. A permanent exemption on a frozen directory is a
   disabled rule — which is exactly why `tests/unit/test_contract_guard.py` is carved out above
   instead of listed here.

2. **`CONTRACT_ALLOW_FROZEN_TESTS=1`** in the environment, for a lead running the guard locally:

   ```
   CONTRACT_ALLOW_FROZEN_TESTS=1 python scripts/contract_guard.py --base origin/main
   ```

   A workflow that sets this unconditionally has disabled rule 2. Do not.

If you are an implementer and a frozen test looks wrong: **do not edit it, do not skip it, and do
not special-case the implementation to pass it.** File an AMBIGUITY REPORT
(`workorders/AMBIGUITY_TEMPLATE.md`) and end the session — CONTRACT rule 3.

### What this script cannot check

Everything below needs a human, and none of it is weakened by the guard passing.

| Not checked | Why not | Who checks it |
|---|---|---|
| **Rule 3, STOP AND REPORT** | "When the spec, the work order and the whitelisted files do not determine a choice, emit an AMBIGUITY REPORT." Whether a choice was determined is a judgement about *meaning*; fluent invention looks exactly like competent implementation from the outside, which is the entire reason the rule exists. No static check can see a decision that should have been a question. | The lead, reading the diff against the card |
| **Rule 4, in general** | A shaping term written under an innocent name, an extra addend in `enterprise_reward`, or a denylisted wrapper imported under an alias all pass the denylist. | `tests/unit/test_ppo_adapter.py` (inspection of the wrapped object) and test T-B6 (the reward recomputed independently from the five-term formula) |
| **Rule 5, dynamically** | Whether `make_planner_view` actually *leaks* a true quantity into the view it returns. The guard checks signatures; leakage is behaviour. | Test T-B4's sentinel half |
| **Rule 6, in general** | A true quantity smuggled into an observation under another name or as a derived function — "no function of `consumer`, no function of another enterprise's `cum_output`". | Test T-B5's sentinel sweep |
| **Rule 7, in general** | **The important one.** Rule 7 forbids any transition rule or reward term that *implements* bunching, padding, storming, hoarding, shaving or trade. That is a claim about what the code means, not about what it is called. The guard's check is narrow and high-precision by design: it catches a rule that names the pathology it is supposed to let emerge, and nothing else. A rule that computes the same thing under a neutral name passes cleanly and violates rule 7 exactly as much. **Passing is necessary, not sufficient** — rule 7's own wording: *"passing it is necessary, not sufficient — the lead reviews every `env/` diff against this rule."* The guard prints that note on every run. | Behavioural test T-B1 (also not sufficient), and the lead, by hand, on **every** `gosplan/env/` diff |
| **Rule 8, BOUNDS ARE RESULTS** | Whether a bound was silently widened or narrowed to fix a result is visible only against the pre-registration, and the `BOUND_BINDING` flag is a run-time property of a ledger. | Test T-B8; the run manifest; the lead |
| **Rule 10, MANIFEST** | Whether every run actually wrote a manifest with every required field is a property of runs, not of source. | `tests/unit/test_ledger.py`; the gate artefacts |
| **Rule 11, PARAMETER ARMS** | The INFO/INC/SUPPLY/TECH classification in `gosplan/params.py` is a design decision; whether a reassignment was justified is not decidable from the diff. The guard does not even try. | A `spec/CHANGELOG.md` entry and lead sign-off |
| **Rule 12, WORK ORDERS** | Whether an implementer read only the whitelisted files, wrote only the named files, ran the completion command verbatim and reported in the card's format. A script sees the files that changed, not the files that were read, and reading is half the rule. | The lead, comparing the diff and the session report against the card |
| **Rule 1, "only the lead"** | Authorship is pull-request metadata, not source. | Review and branch protection |

### Testing and extending it

`tests/unit/test_contract_guard.py` builds a tiny synthetic tree per rule under `tmp_path` — one
that violates it and a clean twin that does not — and calls each check directly. Two tests carry
more weight than the rest: one asserts that a rule-4 denylist word appearing **only in a
docstring** does not fire, and one asserts that the real repository tree passes `run_all` cleanly,
so that a violation appearing later is unambiguously the fault of the change that introduced it.

```
python -m pytest tests/unit/test_contract_guard.py -q
```

To add a check: write one `check_*(root, changed) -> list[Violation]` function, register it in
`CHECKS` in `CONTRACT.md` order, add its rule id to `RULE_ORDER`, and add a violating tree and a
clean tree to the test module. Keep every message quoting the rule it enforces and every `fix`
to one actionable line — the reader is an implementer who has not seen this conversation, does not
have `PLAN.md`, and needs to know what to do next without asking.
