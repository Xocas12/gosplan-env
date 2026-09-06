"""Gate G0 - Monte-Carlo sanity and the rule-7 diff review. **An experiment, not a test.**

Realises: PLAN section 13 (gate G0), read against PLAN sections 12.3 (WO-012, the harness this
gate drives), 4.1 (the held-out phenomena the harness may not touch), 11 (the frozen suite that
must be green) and CONTRACT rules 7, 8, 10 and 13. Run by: the **LEAD**, deliberately, after
WO-012. Implemented by no work order: the experiment it drives is `gosplan.experiments.mc_sanity`
(**WO-012**), and this module is the lead's entry point around it.

CONTRACT rule 13: nothing here is a unit test, nothing here is on any work order's must-pass list,
and no implementer session runs it. The file is named `gate_*.py` so pytest never collects it, and
`tests/acceptance/` is excluded from `testpaths` in `pyproject.toml`.

PASS CONDITION (PLAN section 13, verbatim): *full frozen suite green; MC sanity report clean;
lead's diff review of `env/` against rule 7.* Three conditions, and the third is a human reading
code - it cannot be automated and this module must not pretend otherwise:

  1. **Frozen suite green.** `tests/unit`, `tests/behavioural` and `tests/golden` all pass, with
     the golden files generated (`make golden`), so T-B7 is actually exercised rather than skipped.
     Record the pytest summary line and the golden file count in the report.
  2. **MC sanity report clean.** `gosplan.experiments.mc_sanity.run` at `p1_default_config()` plus
     `N_SUPPLY_PERTURBATIONS` SUPPLY perturbations, 2,000 episodes each of `Random`,
     `TruthfulMyopic` and `Padder`: conservation to 1e-9, no NaN or inf, bounded `T` and `S`,
     `fill` in [0, 1], `Padder` produces downstream shortage, and the wall clock per episode
     recorded. Every `BOUND_BINDING` flag raised (CONTRACT rule 8) is displayed, never cleared.
  3. **Rule-7 diff review.** The lead reads every `gosplan/env/` diff since the last review against
     CONTRACT rule 7 and records the commit range reviewed, the files read and the verdict.
     `tests/behavioural/test_no_hardcoded_pathology.py` (T-B1) passing is **necessary, not
     sufficient**, and the review is what the rule requires in addition.

ARTEFACTS (PLAN section 13): `runs/mc_sanity/report.md`, plus a `runs/<config-hash>/` directory per
configuration carrying `manifest.json` and the ledger (CONTRACT rule 10). This gate adds its own
sign-off record naming all three conditions, the commit range reviewed and the reviewer.

SIGN-OFF: **LEAD**.

HELD OUT (PLAN sections 4.1, 4.2). The Monte-Carlo harness asserts conservation and boundedness on
the mechanisms behind rows 2, 5, 6 and 7 and **never a direction**; it may not compute, tabulate or
plot the within-period effort Gini, request inflation, input-stock correlations, trade volume or
`max(0, S - R) / T`. That prohibition binds this gate too: G0's report may not contain any of them.
The four are computed for the first time at gate G3.

IF THIS GATE FAILS (PLAN section 13): write the failure report - which condition failed, on which
configuration and seed, with the manifest - and stop. The next work order is a **lead diagnosis**,
never a parameter change to make the gate pass.
"""

from __future__ import annotations


def main() -> int:
    """Run gate G0 and write its sign-off record.

    Takes: nothing; G0's inputs are fixed by the WO-012 card (`p1_default_config()`, `AGENTS`,
    `N_EPISODES`, `N_SUPPLY_PERTURBATIONS`) and by PLAN section 13. Returns: a process exit code -
    0 when the experiment completed and its artefacts were written, non-zero when it could not
    complete. **The exit code never encodes the gate's verdict**: pass and fail are lines in
    `runs/mc_sanity/report.md` and in the sign-off record, signed by the LEAD.

    Intended sequence, for the lead who implements this harness at the time of the run:
      1. confirm the golden files exist (`make golden`) and run the frozen suite, capturing its
         summary; a skipped T-B7 is not a green suite;
      2. call `gosplan.experiments.mc_sanity.run(p1_default_config())` and collect its return value
         and `runs/mc_sanity/report.md`;
      3. collect the run directories it wrote and check each `manifest.json` against CONTRACT rule
         10, including every flag raised;
      4. record the rule-7 diff review: commit range, files read, reviewer, verdict - entered by
         the lead, never inferred from a green test;
      5. write the sign-off record under `runs/` and return.

    Owning WO: none - this is a lead-run gate harness (CONTRACT rule 13); the experiment it drives
    is implemented in **WO-012**.
    """
    raise NotImplementedError("PLAN section 13 (gate G0) - lead-run; experiment in WO-012")


if __name__ == "__main__":
    raise SystemExit(main())
