AMBIGUITY REPORT   WO-002 / WO-013   tests/acceptance/
Question (one sentence):
Which work order is authorised to author `tests/acceptance/` (the five gate harnesses and its
README), given that no card's "Write only" list names them?

What the spec says / does not say (quote):
PLAN §11 assigns the Acceptance category to "LEAD" and PLAN §13 makes the lead run the gates, but
PLAN §12 issues no card for the directory. WO-002 says "tests/acceptance/ gets nothing from this
card" and lists "Writing any test into tests/acceptance/" under Forbidden; every other card only
forbids *running* them. CONTRACT rule 12 says an implementer "writes only the files it names",
and CONTRACT rule 13 says "tests/acceptance/ holds lead-run experiments (gates). Nothing there is
a unit test, nothing there is on any work order's must-pass list, and no implementer session runs
it." Neither rule says who writes them.

Options considered (A/B/…), and why the spec does not decide:
A. Add tests/acceptance/* to a LEAD card's "Write only" — WO-013 (spec v1 freeze) is the natural
   home, since it already re-runs the full suite at the freeze. Keeps every file inside the
   whitelist system, at the cost of attaching G0-era harnesses to a post-G0 card.
B. State that tests/acceptance/ is authored by the LEAD *outside* the whitelist system, exempt
   from CONTRACT rule 12, because it is not part of the frozen test surface of rule 2 and is never
   on a must-pass list. Matches rule 13's framing ("experiments, not tests") but creates the only
   directory in the repo that no card owns.
The spec does not decide because §11 names an owner (LEAD) while §12 — the only place ownership
becomes an authorisation — has no corresponding card.

Impact if the wrong option is picked:
Under A the gate harnesses become editable by whoever holds WO-013, which is a LEAD card, so the
blast radius is small. Under B a directory exists that no card authorises, which weakens rule 12
as a general invariant ("every file has an owning card") and makes the next audit re-raise this.
Neither option changes any dynamics, metric or result.

Tests blocked:
None. The five harnesses are excluded from `testpaths`, so nothing in CI depends on the answer.
Skeleton status: the files exist, written by the scaffolding pass, pending this decision.
