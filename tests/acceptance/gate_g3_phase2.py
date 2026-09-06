"""Gate G3 - the Phase-2 acceptance run and the held-out phenomena. **An experiment, not a test.**

Realises: PLAN section 13 (gate G3), read against PLAN sections 4.1 (the phenomena and their
classes), 4.2 (the mechanism parameters locked before Phase 2), 6.2 (the oracle), 6.3
(exploitability), 12.4 (WO-021..WO-031, the mechanisms and the acceptance harness this gate drives)
and CONTRACT rules 10 and 13. Run by: the **LEAD**; signed off by **Human + LEAD**. Implemented by
no work order: the harness it drives is **WO-031**, on the metrics of **WO-030**, the
exploitability audit of **WO-028** (`gosplan.experiments.exploitability`), the oracle of **WO-027**
and the JAX port of **WO-029**.

CONTRACT rule 13: not a test, not on any must-pass list, never run by an implementer session.

PASS CONDITION (PLAN section 13, verbatim): *held-out phenomena 2, 5, 6, 7 evaluated on the PLAN
section 4.2 values (pass or reported failure); exploitability below threshold on all arms used;
oracle gap recorded; JAX parity.* Four conditions:

  1. **The four held-out phenomena, evaluated for the first time**, at the values locked in PLAN
     section 4.2 - and locked *before* Phase 2 began, precisely so this evaluation cannot be
     tuned:

         row 2 storming          `delivery_timing = "stochastic"`,
                                 `arrival_probs = (0.25, 0.25, 0.25, 0.25)` (uniform-random
                                 arrival, so no mechanical backloading and any Gini excess is
                                 behavioural), `yield_sigma` x1
         row 5 hoarding          `alloc_eta_request = 0.7`, `input_complementarity = 8`,
                                 `input_holding_loss = 0.01`
         row 6 blat              `horizontal_visibility = 1.0`, `trade_tau = 0.05`
         row 7 hidden reserves   as Phase 1: `g` at its G1 value, `penalty_arg = "positive_part"`,
                                 `holding_loss = 0.02`

     Each is measured against the truthful-myopic baseline under common random numbers where its
     operationalisation says so (rows 2, 5, 6), by `gosplan.metrics.phenomena` (WO-030).
     **"Pass or reported failure"** is the pass condition: a phenomenon that fails to appear is
     written up as a failure, and PLAN section 4.2 is explicit that its parameters may then be
     changed only in a new, labelled study with its own pre-registration - never inside this one.
  2. **Exploitability below threshold on all arms used** (PLAN section 6.3): per arm and seed,
     `(R_BR - R_pop) / abs(R_pop)` under a fresh best-responder with the same PPO configuration and
     budget. The threshold is provisional at `EXPLOITABILITY_THRESHOLD = 0.05` and is finalised by
     the lead at this gate. An arm above it is labelled `NON-CONVERGED` and its *results are not
     interpretable*: it is reported with the label, never quietly dropped, re-trained until it
     passes, or re-labelled.
  3. **Oracle gap recorded** (PLAN section 6.2): the expected-value MIP's welfare is `W_oracle`,
     the denominator of `welfare_ratio`; the clairvoyant number is an upper bound only and is never
     the denominator; the solver, its version and its optimality gap go in the manifest (CONTRACT
     rule 10). Phase 1's `W_truthful_max` placeholder is retired here, and every table that used it
     said so.
  4. **JAX parity** (WO-029): 100 agent-steps with `TruthfulMyopic` identical to the NumPy path to
     1e-5 under the key-based RNG of PLAN section 2.15.

ARTEFACTS (PLAN section 13): `runs/phase2_acceptance/report.md`, plus
`runs/exploitability/report.md`, the oracle's solve record and the JAX parity record, each with the
`runs/<config-hash>/manifest.json` of CONTRACT rule 10.

SIGN-OFF: **Human + LEAD**.

THE HOLD ENDS HERE, AND ONLY HERE (PLAN section 4.1). Rows 2, 5, 6 and 7 have been computed nowhere
before this run - not in Phase 1, not while debugging their mechanisms, not "in passing". The
Monte-Carlo harness of WO-012 asserted only conservation and boundedness on the same mechanisms,
and `tests/behavioural/test_shortage_propagation.py` asserted propagation without a direction. That
discipline is what makes this evaluation a test of Claim A rather than a description of a system
already tuned to produce it.

IF THIS GATE FAILS (PLAN section 13): a written failure report, then a **lead diagnosis**. A
held-out phenomenon that does not appear is a *result* of this study, reported as a failure;
changing its PLAN section 4.2 mechanism parameters starts a new, labelled study instead.
"""

from __future__ import annotations


def main() -> int:
    """Run the Phase-2 acceptance experiment and collect gate G3's four conditions.

    Takes: nothing; the Phase-2 full configuration, the locked mechanism values of PLAN section 4.2
    and the arm list come from the WO-031 issue record and the Phase-2 spec revision. Returns: a
    process exit code - 0 when the acceptance run completed and its artefacts were written,
    non-zero when it could not complete. **The exit code never encodes the verdict**: pass, failure
    and the `NON-CONVERGED` labels are lines in the reports, signed off by the human and the lead.

    Intended sequence, for the lead who implements this harness at the time of the run:
      1. confirm the Phase-2 spec revision is in force and `spec/CHANGELOG.md` records it
         (CONTRACT rule 1), and that every mechanism value matches PLAN section 4.2 exactly -
         a mismatch invalidates the hold, whichever direction it moves the result;
      2. run the WO-031 acceptance harness over the arms, writing `runs/phase2_acceptance/`;
      3. compute rows 2, 5, 6 and 7 with `gosplan.metrics.phenomena` (WO-030) against their
         truthful-myopic baselines under common random numbers - the first and only computation of
         these four;
      4. audit every arm used with `gosplan.experiments.exploitability.run` and record the
         threshold in force, labelling arms above it `NON-CONVERGED`;
      5. record the oracle's welfare, solver, version and optimality gap (WO-027), and the JAX
         parity result (WO-029);
      6. assemble the sign-off record: one line per condition, the per-phenomenon pass-or-failure
         statement, the arm labels, the oracle gap, the parity number and every config hash.

    Owning WO: none - this is a lead-run gate harness (CONTRACT rule 13); the experiments it drives
    are implemented in **WO-027**, **WO-028**, **WO-029**, **WO-030** and **WO-031**.
    """
    raise NotImplementedError("PLAN section 13 (gate G3) - lead-run; experiment in WO-031")


if __name__ == "__main__":
    raise SystemExit(main())
