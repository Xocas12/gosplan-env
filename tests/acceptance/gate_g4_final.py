"""Gate G4 - the final report: contrasts, estimator bias, LLM study, price sensitivity.

**An experiment, not a test.**

Realises: PLAN section 13 (gate G4), read against PLAN sections 4.3 (the contrast design and its
reporting rules), 7.1-7.5 (the four studies), 12.5 (WO-032..WO-037, the harnesses this gate drives),
2.9.4 (the headline metrics) and CONTRACT rules 10 and 13. Run by: the **LEAD**; signed off by the
**Human**. Implemented by no work order: the harnesses it drives are
`gosplan.experiments.contrasts` (**WO-032**), `sobol` (**WO-033**, optional),
`estimator_bias` (**WO-034**), `llm_study` (**WO-035**), `price_sensitivity` (**WO-036**) and the
report roll-up of **WO-037**.

CONTRACT rule 13: not a test, not on any must-pass list, never run by an implementer session.

PASS CONDITION (PLAN section 13, verbatim): *contrasts with CIs; estimator-bias curves; LLM study;
price sensitivity on every headline table.* Four deliverables:

  1. **Contrasts with CIs** (PLAN section 4.3). C0, C_OGAS, C_AUDIT, C_INC and C_BOTH at 30 seeds
     with common random numbers (shared `seed_env`), reported as IQM with stratified bootstrap 95%
     intervals over `welfare_ratio`, `padding_index` and `specification_gap`; the gaps closed
     `Delta_X` per contrast and the interaction `I = Delta_BOTH - Delta_OGAS - Delta_INC`.

         *** TWO REPORTING RULES, BINDING ON EVERY TABLE, FIGURE AND SENTENCE. ***
         (a) **C_AUDIT is dual-classified and is NEVER folded into C_OGAS.** `audit_rate` is an
             information parameter that also enters the reward through the penalty, so an audit
             change is not a pure information change. C_AUDIT is always its own row with its own
             Delta; adding it to C_OGAS - in a table, in a total, or in prose - violates the
             pre-registration.
         (b) **No "X% of the loss is informational" statement without a named contrast and a CI.**
             Every such number is `Delta_X` for a named contrast, reported as an IQM with its
             interval. A percentage without a contrast name and an interval is not a result this
             design can produce.

  2. **Estimator-bias curves** (PLAN section 7.2): the bunching estimator's bias, coverage and
     power against the DP's exact no-manipulation counterfactual across the manipulation-strength
     knob `notch_width` and the cap grid. Poor coverage at small `w` is the study's *finding*,
     reported as such, never a reason to re-tune the estimator.
  3. **LLM ministry study** (PLAN section 7.4), kept separate from the factorial: framings x payoff
     arms x pinned model ids, with every prompt and completion logged and the model versions in the
     manifest (CONTRACT rule 10).
  4. **Price sensitivity on every headline table** (PLAN section 7.5): each table recomputed under
     three perturbed price vectors (`p_j * exp(u_j)`, `u ~ N(0, 0.3**2)`, fixed seeds). A sign
     change in `specification_gap` is **reported, not suppressed** - the price vector is not a
     modelling nuisance to be tuned.

  Optional (PLAN section 4.3): the Saltelli/Sobol design over the 13 swept INFO+INC parameters
  (WO-033), whose table must state the PLAN section 3 sweep ranges as an assumption in the same
  table. Optional means a non-zero exit blocks nothing.

ARTEFACTS (PLAN section 13): the final report, assembled by WO-037 from
`runs/contrasts/`, `runs/estimator_bias/`, `runs/llm_study/`, `runs/price_sensitivity/` and
(optionally) `runs/sobol/`, each carrying the manifest of CONTRACT rule 10 - config hash, spec
version, git hash, seeds, reference-PPO version, estimator version and backend, LLM model ids and
versions, solver version and optimality gap, and every flag raised, `BOUND_BINDING` included.

SIGN-OFF: **Human**.

SCOPE OF THE CLAIM. Phase 3 answers Claim B - how much of the measured loss is closed by moving the
information architecture versus the incentive structure, and whether the two interact. The OGAS
conclusion is the sign and magnitude of `Delta_OGAS` relative to `Delta_INC` and `I`, each with its
interval. The report states what this design can and cannot establish (PLAN section 1.2); a
conclusion wider than the contrasts run is not licensed by any number in it.

IF THIS GATE FAILS (PLAN section 13): a written failure report and a **lead diagnosis**. Parameter
changes after G1 create a new, labelled study - which at this stage means a fresh pre-registration,
not an amendment to this report.
"""

from __future__ import annotations


def main() -> int:
    """Run the Phase-3 studies and assemble the final report for gate G4.

    Takes: nothing; C0 is the post-G3 Phase-2 full configuration, the contrasts are the overrides
    of PLAN section 4.3, the seed count is 30 with common random numbers, and the prompts and model
    ids for the LLM study come from the lead's WO-035 record. Returns: a process exit code - 0 when
    every required study completed and the report was assembled, non-zero when one could not run.
    **The exit code encodes nothing about the conclusion**: `Delta_OGAS`, `Delta_INC` and `I` are
    read off the report with their intervals, and the gate is the human's sign-off.

    Intended sequence, for the lead who implements this harness at the time of the run:
      1. confirm gate G3 was signed off and that no arm carried into Phase 3 is labelled
         `NON-CONVERGED` (PLAN section 6.3) - a non-equilibrium arm's results are not
         interpretable, so a contrast built on one is not either;
      2. run `gosplan.experiments.contrasts.run` at 30 seeds under common random numbers and
         collect the IQMs, the intervals, the per-contrast `Delta_X` and the interaction `I`,
         keeping C_AUDIT as its own row;
      3. run `gosplan.experiments.estimator_bias.run` and collect the bias, coverage and power
         curves;
      4. run `gosplan.experiments.llm_study.run` with the pinned model ids, logging every prompt
         and completion;
      5. run `gosplan.experiments.price_sensitivity.run` over **every** headline table produced
         above, and carry any sign change in `specification_gap` into the report verbatim;
      6. optionally run `gosplan.experiments.sobol.run`; a failure here blocks nothing;
      7. hand the artefacts to the WO-037 roll-up and assemble the sign-off record: one line per
         deliverable, the two reporting rules restated, the manifest roll-up, and the scope
         statement of PLAN section 1.2.

    Owning WO: none - this is a lead-run gate harness (CONTRACT rule 13); the experiments it drives
    are implemented in **WO-032**..**WO-037**.
    """
    raise NotImplementedError("PLAN section 13 (gate G4) - lead-run; experiments in WO-032..WO-037")


if __name__ == "__main__":
    raise SystemExit(main())
