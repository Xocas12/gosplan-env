"""Gate G1 - the DP regime map and the human's pre-registration. **An experiment, not a test.**

Realises: PLAN section 13 (gate G1), read against PLAN sections 5 (the single-enterprise DP and its
regime classifier), 12.3 (WO-015, the experiment this gate drives, and WO-013, the v1 freeze), 3
(the daggered rows this gate fixes), 4.5 (the G2 criteria whose levels are recorded here) and
CONTRACT rules 1, 11 and 13. Run by: the **LEAD** for the map; the decision is the **human's**.
Implemented by no work order: the experiment it drives is `gosplan.experiments.regime_map`
(**WO-015**), on the DP of **WO-014**.

CONTRACT rule 13: not a test, not on any must-pass list, never run by an implementer session.

PASS CONDITION (PLAN section 13, verbatim): *regime map produced; human selects P1 daggered values
from the interior of the bunching region; three `a·pen` levels and `b̂_DP` thresholds recorded
**before** any training.*

    *** THE HUMAN DECIDES FIRST, AND WRITES IT DOWN BEFORE ANY TRAINING RUN EXISTS. ***

    The order is the pre-registration, and it is the whole content of this gate:

      1. `gosplan.experiments.regime_map.run(p1_default_config())` solves the DP over the Latin
         hypercube of PLAN section 5 and writes `runs/regime_map/{table.parquet, regime.png,
         bhat.png, candidates.md}` with at least `MIN_BUNCHING_CANDIDATES` interior bunching-region
         candidates and their `b_hat_DP`;
      2. **the human reads the map** and selects the Phase-1 values of the six daggered PLAN
         section 3 rows - `ratchet_lambda`, `growth_directive`, `overfulfilment_slope`,
         `penalty_scale`, `effort_cost`, `audit_rate` - from the *interior* of the bunching region,
         not from its boundary, so that a small parameter error does not change the regime;
      3. **the human writes `runs/G1_decision.md`**: those six values, the three `a·pen` levels
         that span the bunching region for G2 criterion 1, the `b_hat_DP` thresholds for criterion
         2, and the reasoning - referencing the candidate rows the values came from;
      4. only then does `spec/spec.py` become v1.0.0 with a `spec/CHANGELOG.md` entry (WO-013,
         CONTRACT rule 1), and only then may WO-018 train anything.

    Nothing in the training stack chooses these numbers. `gosplan/experiments/dp_vs_ppo.py` and
    `gosplan/experiments/phase1_gate.py` *read* `runs/G1_decision.md` (`G1_DECISION_PATH`) and
    never re-derive its values from a training result. Values picked after seeing a training result
    are values chosen to produce it, and no later reporting repairs that.

ARTEFACTS (PLAN section 13): `runs/G1_decision.md` and `spec` v1.0.0, on top of WO-015's four
regime-map files. The decision document is the pre-registration; the spec bump is what freezes the
interface it was written against.

SIGN-OFF: **Human**.

FORBIDDEN: no reinforcement learning anywhere in this gate (the WO-014/WO-015 line of PLAN section
12.3). The map is analytical and is produced *before* any MARL run, precisely so that the Phase-1
values cannot be chosen to suit a training result. `DPSolution.hidden_reserves` is computed by the
DP but is PLAN section 4.1 row 7: it is neither tabulated nor plotted here, and it appears in no
G1 artefact.

IF THIS GATE FAILS (PLAN section 13): if the map yields fewer than `MIN_BUNCHING_CANDIDATES`
interior candidates, or no interior bunching region at all, that is a written failure report and a
**lead diagnosis** - widening a sweep range or relaxing the classifier thresholds to manufacture a
bunching region is the specific move the rule forbids. Note also that after this gate, any
parameter change creates a **new, labelled study** with its own pre-registration.
"""

from __future__ import annotations


def main() -> int:
    """Produce the regime map and record the human's G1 decision.

    Takes: nothing; the base configuration is `p1_default_config()`, the design size is
    `N_LHS_POINTS` and the discretisation is the PLAN section 5 `DPGrid` defaults, all fixed by the
    WO-015 card. Returns: a process exit code - 0 when the map and its candidate list were written,
    non-zero when the design could not be solved. **The exit code never encodes the decision**: the
    decision is the human's and lives in `runs/G1_decision.md`.

    Intended sequence, for the lead who implements this harness at the time of the run:
      1. call `gosplan.experiments.regime_map.run(p1_default_config())` and collect the four
         artefacts and the candidate rows;
      2. check that at least `MIN_BUNCHING_CANDIDATES` candidates are interior - away from the
         classifier's boundaries and away from the `rho` grid edge, since a grid-edge hit is a
         regime signal (`pad_to_cap`), not an artefact to smooth away;
      3. print the candidate table and **stop**, leaving the selection to the human;
      4. verify afterwards that `runs/G1_decision.md` exists and carries all six daggered values,
         the three `a·pen` levels and the `b_hat_DP` thresholds, and that its values match a
         candidate row - a decision that names values absent from the map is not a selection *from*
         the map;
      5. confirm that no training artefact exists yet under `runs/`, so the ordering of the
         pre-registration is verifiable after the fact and not merely asserted.

    Owning WO: none - this is a lead-run gate harness (CONTRACT rule 13); the experiment it drives
    is implemented in **WO-015** on the DP of **WO-014**.
    """
    raise NotImplementedError("PLAN section 13 (gate G1) - lead-run; experiment in WO-015")


if __name__ == "__main__":
    raise SystemExit(main())
