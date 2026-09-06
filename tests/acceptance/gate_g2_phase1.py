"""Gate G2 - the Phase-1 acceptance criteria. **An experiment, not a test.**

Realises: PLAN section 13 (gate G2), read against PLAN sections 4.5 (criteria 1-4 and the
pre-registered estimator settings), 4.4 (the measurement window), 5 (the DP that supplies the
ground truth), 12.3 (WO-019 and WO-020, the experiments this gate drives), 7.5 (the price-vector
sensitivity every headline table carries) and CONTRACT rules 8, 10 and 13. Run by: the **LEAD**;
signed off by **Human + LEAD**. Implemented by no work order: the experiments it drives are
`gosplan.experiments.dp_vs_ppo` (**WO-019**) and `gosplan.experiments.phase1_gate` (**WO-020**).

CONTRACT rule 13: not a test, not on any must-pass list, never run by an implementer session.

PASS CONDITION (PLAN section 13): *PLAN section 4.5 criteria 1-4*, all four:

  1. **DP recovery** (single enterprise, `N = 1`, `a = 0`). At the three `a·pen` values recorded in
     `runs/G1_decision.md`, PPO's mean fictitious padding is within 0.02 (ratio units) of the DP's,
     mean effort within 0.05, and the Wasserstein-1 distance between PPO's and the DP's stationary
     `rho_report` distributions is below 0.03 - in at least 8 of 10 seeds per value. (WO-019.)
  2. **Bunching present / absent** (`N = 20`). At the notched configuration, `b_hat >= 0.5 *
     b_hat_DP` with a bootstrap CI excluding 0, in at least 90% of 30 seeds; at the smooth
     counterfactual, the CI for `b_hat` covers 0 in at least 90% of seeds. (WO-020.)
  3. **Padding elasticity.** Across the three `a·pen` values, learned fictitious padding is
     monotone decreasing and within 0.03 of the DP's at each. (WO-020.)
  4. **Hygiene.** `BOUND_BINDING` absent in all runs (CONTRACT rule 8); `T > 3 * T_0` reached in
     fewer than 5% of training episodes after 20% of training. (WO-020.)

    *** HOW TO READ A FAILURE (PLAN section 4.5, and it is the point of the gate). ***

    Failure of criterion 1 is a **training-stack failure and blocks everything**: the optimiser
    could not find an optimum the analytical layer already knows, so nothing downstream is
    interpretable.

    Failure of criterion 2 with criterion 1 passing is a **multi-agent effect and is a RESULT**. It
    is reported - with its CIs and its per-seed counts - and it is **never tuned away**. No
    parameter is moved, no arm re-picked, no estimator setting re-chosen to make it pass. PLAN
    section 13: a failed gate produces a written failure report and the next work order is a lead
    diagnosis; and because G1 has already happened, any parameter change now creates a **new,
    labelled study** with its own pre-registration.

INPUTS: `runs/G1_decision.md` (`G1_DECISION_PATH`) - the human's pre-registration, fixing the
Phase-1 daggered values, the three `a·pen` levels and the `b_hat_DP` thresholds. This gate reads
them; it never chooses them and never re-derives them from a run. The estimator settings of PLAN
section 4.5 have exactly one home, `gosplan/metrics/phenomena.py` (WO-016), and are recorded in the
manifest from there.

ARTEFACTS (PLAN section 13): `runs/dp_vs_ppo/report.md` and `runs/phase1_gate/report.md`, each with
one pass/fail line per criterion, the per-seed counts behind it, the price-sensitivity table (PLAN
section 7.5) and every hygiene flag; plus a `runs/<config-hash>/` directory per run with
`manifest.json` (CONTRACT rule 10, including the reference-PPO version and the estimator backend).

SIGN-OFF: **Human + LEAD**.

HELD OUT (PLAN section 4.1): rows 2, 5, 6 and 7 are not computed, plotted or mentioned here. G2
concerns bunching (row 1) and padding (row 4), both **pipeline checks** - if they fail to appear
the optimiser is broken, and neither is evidence for Claim A.
"""

from __future__ import annotations


def main() -> int:
    """Run gate G2's four criteria and collect their reports.

    Takes: nothing; the configuration, the three `a·pen` levels and the `b_hat_DP` thresholds come
    from `runs/G1_decision.md`, the seed counts from `SEEDS_PER_LEVEL` (WO-019) and `N_SEEDS`
    (WO-020). Returns: a process exit code - 0 when both experiments completed and their reports
    were written, non-zero when a run could not complete or the G1 record is missing. **The exit
    code never encodes the criteria**: pass and fail are the per-criterion lines in the two
    reports, signed off by the human and the lead.

    Intended sequence, for the lead who implements this harness at the time of the run:
      1. read `runs/G1_decision.md`; abort if it is absent - without the pre-registration there is
         no gate, only a sweep;
      2. call `gosplan.experiments.dp_vs_ppo.run(cfg, ap_levels)` for criterion 1 and collect
         `runs/dp_vs_ppo/report.md`;
      3. if criterion 1 failed, stop and write the failure report: it blocks everything downstream,
         and running criteria 2-4 against a broken training stack would produce numbers that look
         like results;
      4. otherwise call `gosplan.experiments.phase1_gate.run(cfg, b_hat_dp, ap_levels)` for
         criteria 2-4 and collect `runs/phase1_gate/report.md`;
      5. assemble the sign-off record: one pass/fail line per criterion with its seed counts, the
         `BOUND_BINDING` and target-runaway hygiene flags, the price-sensitivity table, and the
         config hashes of every run;
      6. where criterion 2 failed with criterion 1 passing, state in the record that this is a
         result and name the diagnosis work order the lead will write - never a parameter to move.

    Owning WO: none - this is a lead-run gate harness (CONTRACT rule 13); the experiments it drives
    are implemented in **WO-019** and **WO-020**.
    """
    raise NotImplementedError("PLAN section 13 (gate G2) - lead-run; experiments in WO-019/WO-020")


if __name__ == "__main__":
    raise SystemExit(main())
