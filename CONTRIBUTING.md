# Contributing to gosplan-env

This is the operating manual for a session - human or model - that has been handed one issue in
this repository and has to finish it correctly.

Read it end to end before writing a line. It is short on encouragement and long on prohibitions,
and that is deliberate: the value of this repository is that its shape was fixed in writing before
anything was implemented, so that the filling-in cannot quietly redefine the question. Most of what
follows exists to stop a fluent, well-meaning session from doing exactly that.

The binding text is [`CONTRACT.md`](CONTRACT.md). Where this document and the contract disagree,
the contract wins.

---

## 1. What this repository is

`gosplan-env` is a multi-agent reinforcement-learning environment in which learning **enterprises**
face a fixed **rule-based planner**. The planner sets targets, allocates inputs from reported
claims, audits a random sample of reports, and ratchets next period's targets on the reported
fulfilment ratio. Enterprises choose effort and what to report. Nothing else is scripted.

It supports two claims, stated, tested and reported **separately** (PLAN §1.1):

- **Claim A - emergence.** Under a fixed rule-based planner, enterprises rewarded only through the
  five-term reward of PLAN §2.9 produce (i) hidden reserves and report shaving, (ii) input hoarding
  and propagated shortage, (iii) horizontal barter, and (iv) storming *in excess of what input
  timing forces* - none of which is written into any transition rule or reward term (CONTRACT
  rule 7). Bunching, padding and quality degradation are **not** part of Claim A: they are direct
  optima of the reward under agent control, and serve as pipeline checks - if they fail to appear,
  the optimiser is broken.
- **Claim B - counterfactual.** For a pre-registered baseline configuration, the welfare gap to a
  full-information oracle closed by an "OGAS" information contrast, an incentive-reform contrast,
  and both together (with their interaction), estimated with common random numbers and bootstrap
  confidence intervals. Claim B is a set of **named contrasts** (PLAN §4.3), not a variance
  decomposition.

Not claimed: necessity (that these rules are the only way to produce these behaviours), causality
about the historical USSR, or transfer of any absolute number to archival data.

### Nothing has been run. No result exists.

Every function and method body in `gosplan/`, `spec/`, `ref/` and `tests/` is exactly

```python
raise NotImplementedError("<PLAN section> - implemented in WO-###")
```

**with three exceptions, which are real CI infrastructure and are expected to pass:**
`scripts/contract_guard.py`, its test `tests/unit/test_contract_guard.py` and `tests/conftest.py`'s `pytest_configure` (a hook that must keep a real body,
because one that raised would abort collection of the whole frozen suite). A failure in any
of the three is a genuine failure of the guard, never expected skeleton behaviour.

There are no trained agents, no runs, no figures, no estimates and no findings. `runs/` is empty.
Gates G0-G4 are all unsigned. Four phenomena - storming excess, hoarding to shortage, blat, hidden
reserves - are **held out**: no plot, table or test of them is produced before the Phase-2
acceptance run (PLAN §4.2).

Consequently **every number visible in this repository today is either a provisional Phase-1
parameter default from PLAN §3 (daggered rows are replaced by a human at gate G1) or a template
placeholder reading `TBD`.** No number here may be cited as a result, an estimate, or a historical
fact - not in a commit message, not in a pull request, not in a docstring, not anywhere.

A green test suite today means the *shape* is right. It is never evidence that anything works.

---

## 2. The contract

All 13 rules, with what each one means for a session doing an ordinary card. The **binding text is
[`CONTRACT.md`](CONTRACT.md)**; this table is a reading aid and is not a substitute for it.

| # | Rule | What it means for you |
|---|---|---|
| 1 | **Frozen spec.** `spec/spec.py` is provisional (v0) until gate G1 and frozen (v1) after. Post-v1 changes: lead only, with a `spec/CHANGELOG.md` entry (version, reason, affected work orders). | You do not edit `spec/spec.py`. If a signature there is wrong or missing, that is an ambiguity report, not a patch. |
| 2 | **Frozen tests.** `tests/unit`, `tests/behavioural`, `tests/golden` are read-only for implementers. | You never edit, never skip, never xfail, never route around and never special-case the implementation to pass a test. A test that looks wrong is an ambiguity report. Golden `.json` files are generated from `ref/`, never hand-edited. The one carve-out is `tests/unit/test_contract_guard.py`, which tests the CI guard rather than the environment - see section 6. |
| 3 | **Stop and report.** When the spec, the work order and the whitelisted files do not determine a choice, file an ambiguity report and end the session. Choosing "the reasonable default" is a violation. | See section 5 below. This is the rule that decides whether your session was valid. |
| 4 | **Reward terms.** The enterprise reward is exactly: production step `-scale * c_ik`; report step `scale * (B(rho) - 1[audited] * Pen + trade_surplus)`. No shaping, no auxiliary reward, no curiosity, no potential-based term, no running reward normalisation. Per-batch advantage normalisation inside PPO is permitted. `scale = reward_scale(cfg)`, computed analytically from the configuration. | Do not add a term "to help learning". Running normalisation is banned specifically because running statistics change the effective reward over training and, with heavy-tailed penalties, shrink the notch in normalised units - and the notch is the object of study. |
| 5 | **Planner blindness.** Planner rules take a `PlannerView` and nothing else. `PlannerView` is built by `make_planner_view()` and contains no true quantity. Any planner function whose signature accepts `State` is a violation. | In `gosplan/env/planner.py`, only `make_planner_view` sees `State`. If a planner rule seems to need a true quantity, either the rule is wrong or the view is wrong; either way it is an ambiguity report. |
| 6 | **Welfare blindness.** `welfare_true` and `val_measured` are logged and never appear in any observation, reward or agent input. The PPO adapter's forward pass takes `obs` only. | Never thread a welfare quantity into `obs.py`, into a reward term, or into an agent. Those quantities exist for the ledger. |
| 7 | **No hard-coded pathology.** No transition rule or reward term may implement bunching, padding, storming, hoarding, shaving or trade directly. `tests/behavioural/test_no_hardcoded_pathology.py` checks this with heuristic agents; passing it is necessary, not sufficient - the lead reviews every `env/` diff against this rule. | The whole point of Claim A is that these behaviours *emerge*. Storming is a consequence of `delivery_timing` plus the agent's effort choice, never a rule that concentrates effort late in a period. Write the mechanism; never write the outcome. |
| 8 | **Bounds are results.** `report_ratio` is bounded at `rho_max = 10`. The fraction of reports at the bound is logged; above 1% flags the run manifest `BOUND_BINDING` and the result is reported with the flag. | Never silently widen or narrow a bound to fix a result. A binding bound is a finding, and it is reported as one. |
| 9 | **RNG.** All environment randomness goes through `rng.draw(seed_env, purpose, *indices)`. No direct `numpy.random` / `jax.random` in `gosplan/env/`. `seed_policy` is separate. | No module-level generator anywhere. Every stochastic term names its purpose and its indices, so that a draw is reproducible from the manifest alone. |
| 10 | **Manifest.** Every run writes `runs/<hash>/manifest.json`: config hash, spec version, git hash, seeds, reference-PPO version, estimator version, LLM model ids and versions, solver version and optimality gap, flags. | A result with no manifest is not a result. If your card produces a run artefact, it produces a manifest. |
| 11 | **Parameter arms.** The INFO / INC / SUPPLY / TECH classification in `gosplan/params.py` is a design decision; changing an arm assignment requires a CHANGELOG entry. | Do not re-classify a parameter because it "obviously belongs" somewhere else. |
| 12 | **Work orders.** An implementer reads only the whitelisted files, writes only the files the card names, runs the completion command verbatim, and reports in the card's format. | This is the loop in section 4. "Only" is literal in both directions. |
| 13 | **Tests are not experiments.** `tests/acceptance/` holds lead-run experiments (gates). Nothing there is a unit test, nothing there is on any work order's must-pass list, and no implementer session runs it. | It is excluded from `testpaths` in `pyproject.toml`, so `pytest -q` never collects it. `make gate` refuses on purpose. Do not run it, do not add to it, do not "check" against it. |

Violations invalidate the session's output. That is not a figure of speech: a card whose diff
breaks a rule is reverted, not amended.

---

## 3. The work-order model

**One issue = one card = one branch = one PR.**

- **The card** is the file `workorders/WO-###.md`. It is authoritative. Its fields are fixed by
  PLAN §12.1 and reproduced in `workorders/TEMPLATE.md`: title, phase, owner tier, difficulty,
  *Depends on*, *Read first (whitelist, exhaustive)*, *Write only*, *Must pass (verbatim test
  ids)*, *Implementation notes*, *Forbidden*, *Completion command*, *Report format*.
- **The issue** is a transcription of the card, opened with the `work-order` template. Where issue
  and card disagree, the card wins and the issue is corrected.
- **The branch** is `wo/<nnn>-<slug>` - lower case, hyphenated, the number zero-padded to three
  digits. Examples: `wo/005-production`, `wo/013-spec-v1-freeze`, `wo/020-phase1-gate`.
- **The PR title** is `WO-005: production step` - the card number, a colon, a short lower-case
  description. One card per PR; never two, never half of one.
- **The PR body** is the four-part report (section 4, step 8).
  `.github/pull_request_template.md` is pre-filled with it.

There are 38 cards, `WO-000` to `WO-037`. `WO-000` .. `WO-020` are Phase-1 cards and are issuable,
subject to their dependencies. `WO-021` .. `WO-037` are **placeholders**: they are issued only
after the spec revision that follows their gate, and an implementer handed one in its present form
stops and files an ambiguity report rather than filling the gaps.

---

## 4. The loop

Follow these steps literally, in order.

1. **Pick the issue.** Take one issue labelled `work-order` that is not labelled `blocked` and not
   labelled `owner:LEAD`. Assign yourself. Do not pick a second one.

2. **Confirm every dependency is closed.** Read the card's *Depends on* line. Every work order it
   names must be a **closed** issue, and every gate it names must be **signed off in writing**. If
   one is open, stop: apply `blocked`, say which dependency is missing, and pick nothing else. A
   card built on an unfinished dependency produces a diff that has to be thrown away.

3. **Check `PLAN.md` is present.** It must exist at the repository root. It is deliberately
   git-ignored and supplied out of band, so a fresh clone does not have it. **Without it the
   section references in the cards cannot be resolved, and the correct action is to stop rather
   than to reconstruct the missing sections from the code.** Never un-ignore it, never commit it,
   never copy long prose from it into a file that will be published.

4. **Read ONLY the whitelist.** The card's *Read first* list is exhaustive (CONTRACT rule 12). Open
   those files and no others - no grepping the tree for context, no "just checking" a neighbouring
   module, no reading the tests belonging to a different card. If a file you need is not on the
   list, that is an ambiguity report naming the card, not a licence to open it.

5. **Implement only the "Write only" files.** Exactly the paths the card names are created, edited
   or deleted. Nothing else - not a typo fix in a neighbouring docstring, not a stray import
   cleanup, not a `.gitignore` line. Transcribe the formulas in *Implementation notes*; do not
   re-derive them. Respect *Forbidden* as a hard list. Lines are 100 characters or fewer.

6. **Run the completion command verbatim.** Copy-paste it from the card, in the order it gives,
   with no added flags, no narrowed test selection, and no rerun of a subset until it is green.
   Keep the full output, including failures.

7. **Run the contract guard.**

   ```
   uv run python scripts/contract_guard.py --all
   ```

   It is a static check of the mechanical parts of the contract - forbidden imports and direct RNG
   calls under `gosplan/env/`, planner functions typed against `State`, welfare quantities reaching
   an observation or a reward, reward terms outside rule 4. It must be clean before you open the
   PR. If it flags your diff, **fix the diff**; the guard is never edited, silenced, or given an
   exception in order to make a card pass. If you believe the guard itself is wrong, that is an
   ambiguity report.

8. **Open the PR and post the four-part report as the PR body.** Exactly four parts, in this order
   (PLAN §12.1):

   1. **Tests passed / failed, with output.** Paste it unedited, failures included.
   2. **Ambiguity reports filed.** Link each, or write "none".
   3. **Files changed.** Paste `git diff --name-only`, beside the card's *Write only* list. The two
      lists must be identical.
   4. **Anything you were tempted to invent and did not.** Never leave this empty. If there was
      genuinely nothing, write "nothing" and one sentence saying why the card was fully determined.
      Otherwise list every point where the written material did not decide something and you
      stopped or narrowed instead - the epsilon you wanted to add to a denominator, the clip you
      wanted to apply, the default you wanted for a Phase-2 toggle, the test you thought was wrong.
      This is the most useful section in the PR: it is where the next ambiguity report comes from,
      and it is how the lead audits whether rule 3 held.

9. **Wait for CI.** Branch protection on `main` requires one check, **`ci-ok`**
   (`.github/workflows/ci.yml`). It aggregates four jobs - `lint`, `test`, `contract` and
   `build` - and succeeds only when every one of them has succeeded, so it is red whenever any
   of them is red or was skipped. **A red `ci-ok` is not done.** Fix the diff. Never disable a
   job, never add `continue-on-error`, never merge past a red check, and never edit the workflow
   to make a card pass - that is the same act as editing the guard (step 7).

10. **Stop.** Do not start the next card in the same session.

### The six standing conditions

Every work-order issue closes on these six, plus whatever its own card adds on top of them.
They are the default *Definition of done* of `.github/ISSUE_TEMPLATE/work-order.yml`, and the
wording below is that field verbatim:

```
- [ ] Every file in "Write only" is implemented; no other file in the diff.
- [ ] Completion command run verbatim; output pasted in the PR body.
- [ ] `uv run ruff check . && uv run ruff format --check .` is clean.
- [ ] `uv run python scripts/contract_guard.py --all` is clean.
- [ ] CI `ci-ok` is green on the PR.
- [ ] Four-part report posted as the PR body (tests, ambiguity reports, files changed,
      what I was tempted to invent and did not).
```

A card may add conditions. It may not drop one of these six.

---

## 5. THE STOP RULE (CONTRACT rule 3)

> **When the spec, the work order and the whitelisted files do not determine a choice, you STOP.**

Not "pick the sensible option and flag it in the PR". Not "implement it behind a flag". Not "leave
a `TODO`". You open an issue with the **ambiguity-report** template, label it `ambiguity`, link it
to the work-order issue, apply `blocked` to that issue, and **end the session**.

Choosing the reasonable default is a violation of the contract, and the session's output is invalid
even if every test passes.

**Why the rule is this harsh.** A capable model asked to fill a gap will fill it fluently, and the
result will look exactly like the rest of the code: correctly typed, well named, plausibly
motivated, green. There is no downstream check that catches it, because the invented rule *is* the
specification as far as every later card is concerned. In a repository whose entire claim is "these
behaviours were not written in", one invented rule in the wrong place - a clip, a floor, a
tie-break, a default toggle - is not a small defect. It is the difference between a result and a
tautology. The skeleton exists so that gaps surface as issues rather than as code.

Filing an ambiguity report is **correct behaviour and is never treated as failure.** Two are
already committed as worked examples: `workorders/AMBIGUITY-001.md` and
`workorders/AMBIGUITY-002.md`. The template is `workorders/AMBIGUITY_TEMPLATE.md`, and its five
fields are exactly the fields of the issue form: Question (one sentence); What the spec says / does
not say (quote); Options considered (A/B/...), and why the spec does not decide; Impact if the
wrong option is picked; Tests blocked.

**Numbering and naming.** One report, one number, one question. The **issue** is titled
`AMB-###: <one-line question>` - which is exactly the default title the `ambiguity-report`
form gives you - and the **card** the lead commits for it is `workorders/AMBIGUITY-###.md`
with the *same* number, zero-padded to three digits. `AMB-001` pairs with
`workorders/AMBIGUITY-001.md` and `AMB-002` with `workorders/AMBIGUITY-002.md`, so the next
report is `AMB-003` and `workorders/AMBIGUITY-003.md`. Take the next unused number across
both. Do not invent a third prefix, and do not renumber an existing report.

### Worked example: a good ambiguity report versus a bad silent guess

The card for `gosplan/env/production.py` says the coverage aggregator must handle a zero input
requirement "explicitly rather than by an epsilon", and PLAN §2.6 gives the formula without
restating, for the infinite-complementarity branch, what happens to a good that is not needed
at all.

**The bad silent guess** - what a fluent session does:

```python
# a good that isn't needed shouldn't constrain production, so treat it as fully covered
ratios = np.where(need > 0, np.minimum(1.0, X / np.maximum(need, 1e-12)), 1.0)
H = ratios.min(axis=1)
```

Three inventions in four lines: the `1e-12` floor the card explicitly forbids, the decision that
unneeded goods contribute `1.0`, and the decision that the minimum ranges over all goods rather
than only over goods with positive need. Each is defensible. None is written down. The PR is green,
the reviewer sees a tidy vectorised expression, and the environment now has a rule nobody chose.

**The good ambiguity report** - same session, one hour earlier:

> **Question (one sentence):** When `theta` is infinite, does the minimum branch of the coverage
> aggregator range over all goods or only over goods with strictly positive need?
>
> **What the spec says / does not say (quote):** PLAN §2.6 gives
> `H_ik = (sum_j omega_j * min(1, X_ij/need_ikj)^(-theta))^(-1/theta)` and states that a good with
> `need_ikj == 0` contributes a coverage ratio of 1. It does not restate that convention where it
> introduces the `theta = inf` branch, and the card says only that the branch "takes the min", with
> no domain.
>
> **Options considered (A/B/...), and why the spec does not decide:** **A.** Minimum over all
> goods, with unneeded goods contributing 1 - the convention of the finite branch, so the two agree
> as `theta` grows. **B.** Minimum over goods with positive need only - identical whenever any good
> is needed, but leaves `H` undefined for an enterprise whose row of the input matrix is entirely
> zero, a case PLAN §2.6 separately fixes at `H = 1`. The spec does not decide because the ratio
> convention is stated once for the general formula, while the `inf` branch is introduced later as
> a special case without repeating it.
>
> **Impact if the wrong option is picked:** None on any Phase-1 configuration, where `theta` is
> finite. Under B the aggregator raises on an all-zero input row instead of returning 1, which
> breaks the third case of T-U7 and would surface later as a crash rather than as a wrong number.
>
> **Tests blocked:** `tests/unit/test_production.py` T-U7 (the `theta = inf` equals `min` case).

That report takes ten minutes, is answered in one sentence by whoever wrote PLAN §2.6, and leaves a
permanent record of a decision that was *made* rather than assumed.

**Rule of thumb.** If you are writing a comment that begins "assume", "presumably", "for now",
"reasonable to", or "since the spec doesn't say" - you have already found your ambiguity report.
Stop and file it.

---

## 6. What is frozen, and by whom

| Surface | Rule | Who may change it | How |
|---|---|---|---|
| `spec/spec.py` | 1 | **LEAD only**, and only after gate G1 has frozen it at v1 | A `spec/CHANGELOG.md` entry with version, reason and affected work orders. No entry, no change. |
| `tests/unit/`, `tests/behavioural/`, `tests/golden/` | 2 | **Nobody, as an implementer.** Read-only. | A test that looks wrong is an **ambiguity report**. Never edit it. Never skip it. Never xfail it. Never special-case the implementation to pass it. |
| `tests/golden/*.json` | 2 | Regenerated by the **LEAD** from the unmodified `ref/gen_golden.py`, and only when the dynamics deliberately changed | Run `uv run python -m ref.gen_golden --check` first: it regenerates in memory, writes nothing, and exits non-zero on any difference. A parity failure that is not an intended dynamics change is a bug in `gosplan/env/`, and it *blocks* the freeze. |
| `CONTRACT.md` | - | **LEAD.** It is PLAN §9 reproduced verbatim. | Not a work-order deliverable. |
| `workorders/` | 12 | **LEAD.** | A card is what authorises a session to touch a file at all; editing one silently changes what some future session may do. Cards are amended in the open, before they are picked up. |
| `ref/` | - | **LEAD** (WO-002). It is the test oracle behind the golden matrix. | `ref/` never imports from `gosplan/`. |
| `tests/acceptance/` | 13 | **LEAD.** Gate experiments, not tests. | Never collected by CI, never on a must-pass list, never run by an implementer session. |
| `tests/unit/test_contract_guard.py` | - | Whoever edits `scripts/contract_guard.py` | **The one carve-out from rule 2.** It is the test for the CI guard, not for the environment, so it is not part of the frozen surface rule 2 protects (`scripts/README.md`, *The rule-2 exemption*). The guard skips this one path structurally (`FROZEN_TEST_NON_FROZEN_PATHS` in `scripts/contract_guard.py`), so editing it produces no rule-2 finding and needs no `.github/FROZEN_TEST_EXEMPTION` entry — a standing entry there would be a permanently disabled rule (`scripts/README.md`, *The rule-2 exemption*). `.github/FROZEN_TEST_EXEMPTION` remains the mechanism for temporary LEAD edits to genuinely frozen tests. Never set `CONTRACT_ALLOW_FROZEN_TESTS=1` in a workflow: that disables rule 2 wholesale. |

`.github/CODEOWNERS` names these paths for the same reason, and repeats the `test_contract_guard.py`
carve-out as a comment. An implementer PR that touches a frozen path is rejected without review.

The one thing to internalise: **if a frozen test looks wrong, it is an ambiguity report.** Not an
edit, not a skip, not an `xfail`, not a narrowed `-k` selection, and above all not a branch in the
implementation that exists only to satisfy it. The frozen suites are the reason a later result can
be believed at all.

---

## 7. Gates

A gate is a **written sign-off on named artefacts**, not a vibe and not a feeling that things are
going well. There are five (PLAN §13):

| Gate | After | Pass condition | Artefacts | Sign-off |
|---|---|---|---|---|
| **G0** | WO-012 | Full frozen suite green; MC sanity report clean; lead's diff review of `gosplan/env/` against CONTRACT rule 7 | `runs/mc_sanity/report.md` | LEAD |
| **G1** | WO-015 | Regime map produced; **a human** selects the Phase-1 provisional values from the interior of the bunching region; three `a*pen` levels and the `b_hat_DP` thresholds recorded **before any training** | `runs/G1_decision.md`, `spec` v1.0.0 | Human |
| **G2** | WO-020 | PLAN §4.5 criteria 1-4 | `runs/dp_vs_ppo/report.md`, `runs/phase1_gate/report.md` | Human + LEAD |
| **G3** | WO-031 | Held-out phenomena 2, 5, 6, 7 evaluated on the PLAN §4.2 values (pass or reported failure); exploitability below threshold on all arms used; oracle gap recorded; JAX parity | `runs/phase2_acceptance/report.md` | Human + LEAD |
| **G4** | WO-037 | Contrasts with CIs; estimator-bias curves; LLM study; price sensitivity on every headline table | final report | Human |

**A gate that fails produces a written failure report.** The next work order is then a **LEAD
DIAGNOSIS** - never a parameter change made in order to pass the gate. If a gate fails and someone
proposes moving `penalty_scale`, `audit_rate`, `ratchet_lambda`, a bound, a seed set or the
measurement window so that it passes, that proposal is precisely the failure mode this rule exists
to prevent, and the answer is no.

**Parameter changes after G1 create a new, labelled study** with its own pre-registration. They do
not amend the existing one. A result obtained under different parameters is a different result and
is reported as such.

Related, and for the same reason: bounds are results (CONTRACT rule 8). If more than 1% of reports
sit at `rho_max = 10`, the run manifest is flagged `BOUND_BINDING` and the result is reported
**with the flag**. The bound is not moved.

Gates are run by the lead, deliberately. `make gate` exists and refuses on purpose;
`tests/acceptance/` is excluded from `testpaths` so no ordinary test run can trigger one.

---

## 8. Local setup

Python 3.12, [uv](https://docs.astral.sh/uv/), GNU make.

```
uv sync --all-extras --dev                        # .venv with the pinned dependency set
make setup                                        # uv sync, plus the pre-commit hook
make test                                         # the frozen suites only (see below)
make lint                                         # ruff check . && ruff format --check .
make format                                       # ruff check --fix . && ruff format .
make spec-check                                   # spec/spec.py imports; PLAN §10 symbols present
make golden                                       # LEAD ONLY: regenerate tests/golden/*.json
uv run python scripts/contract_guard.py --all     # the mechanical contract checks
```

- **`make test`** honours `testpaths` in `pyproject.toml`: `tests/unit`, `tests/behavioural`,
  `tests/golden`. It never collects `tests/acceptance/` (CONTRACT rule 13).
- **`make golden`** regenerates the golden matrix from `ref/gen_golden.py`. It is a **LEAD** action
  and requires a `spec/CHANGELOG.md` entry. As an implementer you run at most
  `uv run python -m ref.gen_golden --check`, which writes nothing and exits non-zero on any
  difference. Golden `.json` files are git-ignored, so a fresh checkout regenerates them once
  before the parity tests can run.
- **`uv run python scripts/contract_guard.py --all`** must be clean before every PR. Fix the diff,
  never the guard.
- **Lint**: `ruff`, `line-length = 100`, `target-version = "py312"`.
- **CI**: every push and pull request runs `.github/workflows/ci.yml`, whose jobs `lint`,
  `test`, `contract` and `build` run the four commands above. They are aggregated by **`ci-ok`**,
  the single required check for branch protection on `main`: it succeeds only when all four
  have succeeded. Green locally is not green on the PR - check `ci-ok` before you call a card
  done.

### `PLAN.md` must be present locally

`PLAN.md` ("gosplan-env - Build Plan v1.0") is the authoritative design document. Every module
docstring, work-order card and test cites the section it realises, and the cards are written to be
read next to it. It is **git-ignored and supplied out of band**, so a fresh clone does not contain
it.

**Without `PLAN.md` at the repository root, the section references in the cards cannot be resolved,
and no work order can be executed.** The correct action in that case is to stop and ask for the
file - never to reconstruct the missing sections from the code, and never to proceed on the parts
of a card that look self-contained.

Never un-ignore it, never commit it, never vendor it, and never copy long prose blocks from it into
code, docs or an issue. Cite section numbers.

---

## 9. Labels and the roadmap

### Label taxonomy

**Kind** - exactly one per issue:

| Label | Meaning |
|---|---|
| `work-order` | One card from `workorders/`. The normal unit of work. |
| `gate` | A gate sign-off, G0-G4. Lead and human only. |
| `ambiguity` | An ambiguity report under CONTRACT rule 3. Filing one is correct behaviour, never a failure. |
| `infrastructure` | Repository plumbing that is not a card: CI, templates, tooling, this file. Never touches environment dynamics. |

**State:**

| Label | Meaning |
|---|---|
| `blocked` | Cannot start or continue: an open dependency, an unsigned gate, or an unanswered ambiguity report. Never pick up a `blocked` issue. |

**Phase** (PLAN §1.2 fixes what each phase can and cannot establish):

| Label | Meaning |
|---|---|
| `phase:P1` | Buildable now. Establishes that the training stack recovers the exactly-solved single-enterprise optimum, that the 20-enterprise system bunches under a notch and not under a smooth bonus, and that padding responds to expected penalty as the DP predicts. Establishes **nothing** about Claim A or Claim B. |
| `phase:P2` | Issued only after gate G2 and the Phase-2 spec revision. Claim A phenomena, equilibrium verification, oracle, JAX parity. Cannot establish Claim B. |
| `phase:P3` | Issued only after gate G3. Claim B contrasts, estimator-bias study, LLM study. |

**Gate the card feeds:**

| Label | Fires after |
|---|---|
| `gate:G0` | WO-012 |
| `gate:G1` | WO-015 |
| `gate:G2` | WO-020 |
| `gate:G3` | WO-031 |
| `gate:G4` | WO-037 |

**Owner tier** - see section 10:

`owner:LEAD`, `owner:MID-strong`, `owner:MID-fast`

**Difficulty**, copied from the card header:

`difficulty:1`, `difficulty:2`, `difficulty:3`, `difficulty:4`, `difficulty:5`

The scale is 1-5 and there is no other difficulty label. A `phase:P2` or `phase:P3`
placeholder - WO-021 .. WO-037 - carries **no** `difficulty:` label at all: PLAN §12.4 and
§12.5 fix the owner tier only, and the lead sets the 1-5 number when the card is actually
issued. Do not invent a stand-in number and do not create a new label for the undecided
case; an absent label is the correct encoding of "the lead has not set it yet".

**Special handling:**

| Label | Meaning |
|---|---|
| `held-out` | The card touches a **held-out phenomenon** - PLAN §4.1 rows 2, 5, 6, 7: storming excess, hoarding to shortage, blat, hidden reserves. Their mechanism parameters were locked before any learning run (PLAN §4.2), and they are computed, plotted or tabulated for the **first time** in the Phase-2 acceptance run. Extra care: do not measure one early, do not "just check" one, do not add a diagnostic that reports one as a side effect. |
| `needs-human` | Requires a human decision, not a model one. The canonical case is G1, where a human picks the Phase-1 parameter values from the interior of the bunching region of the regime map. Also: resolving an ambiguity report that turns on an economic judgement, and every gate sign-off marked "Human" in section 7. |

### How to read the roadmap

The roadmap is the work-order DAG of PLAN §12.2, and it is visible in four places:

1. **The cards themselves**, `workorders/WO-000.md` .. `workorders/WO-037.md`. Each card's header
   line gives phase, owner tier and difficulty; its *Depends on* line gives its incoming edges.
   That is the authoritative graph, and it wins over any label and over any summary.
2. **[`ROADMAP.md`](ROADMAP.md)**, which renders that graph as prose: the build order, the gates,
   and why the order is what it is. It is a map, not a specification - where it summarises
   `PLAN.md` the plan wins, and where it summarises `CONTRACT.md` the contract wins.
3. **The gate milestones**, which partition the 38 cards. GitHub matches a milestone by exact
   string, so these five names are used verbatim everywhere they appear:

   | Milestone | Cards |
   |---|---|
   | `G0 Scaffold and sanity` | WO-000 .. WO-012 |
   | `G1 Regime map and freeze` | WO-013 .. WO-015 |
   | `G2 Phase 1 gate` | WO-016 .. WO-020 |
   | `G3 Phase 2 acceptance` | WO-021 .. WO-031 |
   | `G4 Final report` | WO-032 .. WO-037 |

   The same five strings head the tables of [`ROADMAP.md`](ROADMAP.md) sections 3 and 4 and
   fill the `Gate` dropdown of `.github/ISSUE_TEMPLATE/gate.yml`. Do not paraphrase one: an
   issue filed against "G2 Phase-1 gate" matches no milestone at all.
4. **The label filters.** To find work that can actually be started now:
   `is:issue is:open label:work-order -label:blocked -label:owner:LEAD`, then narrow with
   `label:gate:G0` or `label:difficulty:2` as appropriate.

   That filter returns Phase-1 cards only, and it does so **by design, not by accident.**
   Every `phase:P2` and `phase:P3` card - all seventeen, WO-021 .. WO-037 - carries
   `blocked`, because both of the things it waits on are missing: the gate it sits behind
   is unsigned (G2 for the P2 block, G3 for the P3 block), and the LEAD's Phase-2 spec
   revision, which is what rewrites those cards' implementation notes into something an
   implementer could execute, has not happened. That is exactly the `blocked` definition
   above: an open dependency and an unsigned gate. The label is the mechanism that keeps a
   placeholder out of this filter, and it is removed card by card - by the lead, at issue
   time - only once the gate is signed and the revision has landed (section 3).

The build order inside the gates, in one line: WO-000 (parameter sourcing), WO-001 (spec v0,
contract, registry) and WO-002 (reference dynamics and frozen tests) come first; the environment
modules, the step function, the heuristics, the ledger and the Monte-Carlo sanity harness reach
**G0**; the spec freeze, the DP and the regime map reach **G1**; metrics, the PPO adapter, the
training harness and the two Phase-1 experiments reach **G2**; the Phase-2 mechanisms, the oracle,
exploitability and the JAX port reach **G3**; the contrasts, the studies and the report reach
**G4**.

An issue with no gate label, no phase label and no owner label is not ready to be picked up.

---

## 10. Owner tiers

Every card names one owner tier (PLAN §12.1). `MID-fast` and `MID-strong` are **tiers, not models**:
the lead maps a tier to a specific current model at issue time and records the mapping in the run
manifest (CONTRACT rule 10), so a result can always be traced to what produced it.

| Tier | What it means |
|---|---|
| **LEAD** | The frontier model, plus the human wherever a gate says so. LEAD cards are the ones where a wrong invention is unrecoverable: the spec (WO-001, WO-013), the reference dynamics and the frozen test suite (WO-002), the step function and env wrapper (WO-009), the PPO adapter (WO-017), the Phase-2 spec revision, the JAX port, and every gate. The lead also runs the gates and writes the failure reports. |
| **MID-strong** | An implementer card with real formula content, several interacting edge cases, or a harness that has to be right for a gate to mean anything - production, the planner, reporting and reward, the Monte-Carlo sanity harness, the DP, the Phase-1 metrics and experiments. Typically difficulty 3-4. |
| **MID-fast** | A tightly bounded implementer card: one file, a determinate transcription, a short must-pass list - config, RNG, observation, heuristic agents, the ledger, the regime map. Typically difficulty 1-2. |

**An implementer session does not pick up a `LEAD` card.** If you were handed one, or the only
unblocked issue you can find is labelled `owner:LEAD`, stop and say so. This is not a judgement
about capability: LEAD cards change what every other card is allowed to do, and they carry
obligations - a `spec/CHANGELOG.md` entry, a golden-regeneration decision, a rule-7 diff review of
`gosplan/env/`, a gate sign-off - that only the lead can discharge.

The converse matters less often but still holds: a LEAD session working a MID card still reads only
the whitelist and writes only the named files. The tier decides who may hold a card, never how the
card is executed.

---

## Summary card

- One issue, one card, one branch `wo/<nnn>-<slug>`, one PR titled `WO-###: title`.
- Read only the whitelist. Write only the named files. Run the completion command verbatim.
- `uv run python scripts/contract_guard.py --all` clean, ruff clean, CI `ci-ok` green, no
  frozen file touched.
- Four-part report as the PR body; part 4 is never empty.
- **When it is not written down, you stop and file an ambiguity report.** That is the job.
