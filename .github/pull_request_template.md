<!--
  One issue = one card = one branch = one PR.
  Branch:    wo/<nnn>-<slug>        e.g. wo/005-production
  PR title:  WO-005: production step

  This PR body IS the four-part report the card asks for (PLAN §12.1). Do not summarise it
  elsewhere and link to it; write it here, in full, below.
-->

## Card

Closes #<issue>
Card: `workorders/WO-###.md`
Owner tier: `LEAD` / `MID-strong` / `MID-fast`

## Loop checklist

- [ ] **Dependencies closed.** Every issue on the card's "Depends on" list is closed, and every
      gate it names is signed off in writing.
- [ ] **`PLAN.md` present at the repository root.** Its section references resolved; nothing
      was reconstructed from the code. (It is git-ignored and supplied out of band; without
      it the correct action is to stop, not to reconstruct - CONTRIBUTING.md section 4, step 3.)
- [ ] **Whitelist respected.** I opened only the files on the card's "Read first" list. No other
      file in the repository was read.
- [ ] **"Write only" respected.** The diff touches exactly the paths the card names, and nothing
      else. (`git diff --name-only` output pasted below matches that list line for line.)
- [ ] **Completion command run verbatim**, in the order the card gives, with no added flags and no
      narrowed selection. Output pasted below.
- [ ] **`uv run python scripts/contract_guard.py --all` is clean.** Output pasted below.
- [ ] **Ruff clean**: `uv run ruff check .` and `uv run ruff format --check .` both pass.
- [ ] **No frozen file touched.** Nothing under `spec/`, `tests/unit/`, `tests/behavioural/`,
      `tests/golden/`, `tests/acceptance/`, `ref/`, `CONTRACT.md` or `workorders/` is in this
      diff.
      *(If it is: this is a LEAD PR, and the `spec/CHANGELOG.md` entry is in the diff. Tick the
      next box instead.)*
      The full list is CONTRIBUTING.md section 6 and `.github/CODEOWNERS`; the single
      carve-out is `tests/unit/test_contract_guard.py`, the CI guard's own test, which is
      not part of the rule-2 frozen surface.
- [ ] **LEAD exception used**: a frozen surface is touched, I am the lead, and a complete
      `spec/CHANGELOG.md` entry (version, reason, affected work orders, approver) is part of this
      diff (CONTRACT rule 1).
- [ ] **No test was edited, skipped, xfailed or special-cased** to get to green (CONTRACT rule 2).
- [ ] **`tests/acceptance/` was not run** (CONTRACT rule 13).
- [ ] **CI `ci-ok` is green on this PR.** It is the single required check for branch protection
      on `main` and aggregates `lint`, `test`, `contract` and `build` (`.github/workflows/ci.yml`).
- [ ] **No new reward term, no shaping, no reward normalisation** (CONTRACT rule 4); no planner
      function takes `State` (rule 5); `welfare_true` / `val_measured` reach no observation,
      reward or agent input (rule 6); no pathology is written into a rule (rule 7); all randomness
      under `gosplan/env/` goes through `rng.draw` (rule 9).

## Report

### 1. Tests passed / failed, with output

<!-- Paste the completion command and its FULL output, unedited, including any failure. A card is
     not done because the code looks right; it is done because this block says so. -->

```
$ <completion command line 1>
...
```

```
$ uv run python scripts/contract_guard.py --all
...
```

### 2. Ambiguity reports filed

<!-- Link each one, or write "none". If you filed one, this PR should normally be a draft or not
     exist at all: under CONTRACT rule 3 the session ENDS at the report. -->

none

### 3. Files changed

<!-- Paste `git diff --name-only` and put the card's "Write only" list beside it. These two lists
     must be identical. -->

```
$ git diff --name-only main...HEAD
```

Card's "Write only":

```
```

### 4. Anything I was tempted to invent and did not

<!-- REQUIRED. This section must not be left empty. If there was genuinely nothing, write the word
     "nothing" and a sentence saying why the card was fully determined.

     Otherwise list every point where you noticed the written material did not decide something
     and you stopped or narrowed instead: an epsilon you wanted to add to a denominator, a clip
     you wanted to apply, a default you wanted to pick for a Phase-2 toggle, a test you thought
     was wrong. This section is the single most useful thing in the PR for the reviewer, because
     it is where the next ambiguity report comes from. -->

## Notes for the reviewer

<!-- Optional. Anything that helps the lead's rule-7 diff review of `gosplan/env/`: which lines
     could be misread as encoding a pathology, and why they do not. -->
