# Phase-3 record - LEAD

**Status:** the Phase-3 studies are complete as a *labelled study*. The one exception is the LLM
ministry study, which is **NOT RUN** because no model access exists in this container. G4
(PLAN section 13) needs the human's sign-off, and its "LLM study" condition cannot pass until the
owner runs S5. Written on 2026-09-28.

- Design: `spec/P3_REVISION.md` S1-S7, including both S4 amendments.
- The embedded reports: `runs/final_report/report.md`.

## Limitations carried into every Phase-3 number

- **L1 and L2** (`runs/G2_record.md`, `spec/P2_REVISION.md`): the PPO learner does not recover
  the DP's mixed under-reporting strategy, and the pre-registered bunching estimator is undefined on
  degenerate distributions and over-confident on sharply peaked ones.
- **L3** (`runs/G3_record.md`, D1): all Phase-3 runs used spec 2.0.x, where no learner could post
  a trade offer. Every contrast therefore describes an economy **without horizontal trade**. No
  contrast lever is a trade parameter (owner decision 3 at G3).
- **Exploitability (D2).** The contrasts' convergence column comes from the pre-R16 audit, and a
  from-scratch best responder never matched the population. The column is therefore
  **inconclusive, not a pass**. R16 revises the audit, and it is applied to the G3b populations.
- **Reduced designs, for compute:**
  - the contrast arms at 15 seeds (PLAN: 30);
  - the estimator-bias arms at 5 seeds, apart from the two 30-seed arms;
  - Sobol not run (S3);
  - the price-reading arms (C_INC, C_BOTH) recomputed only, not rerun (S6).

## Contrasts (WO-032): `runs/contrasts/report.md`

Seed-paired deltas are on `welfare_ratio`: IQM with a 95% bootstrap CI over 15 seeds.

| contrast | Delta | reading |
|---|---|---|
| C_INC | +0.068 [0.021, 0.111] | an incentive change raises welfare |
| C_OGAS | -0.034 [-0.050, -0.016] | transparency *lowers* welfare |
| C_AUDIT | -0.011 [-0.034, 0.020] | CI covers 0 |
| C_BOTH | -0.008 [-0.029, 0.012] | CI covers 0 |
| I (interaction) | -0.043 [-0.081, -0.002] | negative: the two levers substitute |

- **C_OGAS collapse is behavioural, not an environment defect.**
  - The truthful-myopic sanity run gives W of 1.09 under C0, 1.07 under C_OGAS and 1.02 under
    C_INC. The environment under transparency is therefore sound.
  - The learned populations' padding under C_OGAS explodes: IQM padding_index about 810, against
    2.2 under C0.
- Levels are low in every arm: C0's welfare_ratio is 0.030, where the truthful baseline reaches
  about 0.53. All deltas are movements within a heavily degraded learned economy (G3 welfare
  note).
- Reporting rules held: C_AUDIT is dual-classified, and no "X% informational" statement is made.

## Estimator bias (WO-034): `runs/estimator_bias/report.md`

- **Simulation truth.**
  - On the smooth and partially notched arms (w >= 0.02), the pre-registered estimator's bias is
    small in absolute terms (|bias| < 1.4 units).
  - Its CI coverage is poor: 0.00 to 0.40, against a nominal 0.95.
- **Full bunching (w = 0).** The estimator breaks down.
  - 12 of 30 notched seeds, and 2 of 5 at cap inf, have no counterfactual support (AMBIGUITY-022).
  - Near-zero support gives finite but enormous estimates. The mean bias is 4e4, and the median
    error (+25.5; supplementary, labelled) is the readable summary.
- **DP truth is undefined in S4's units on every arm.** The DP's reports are point masses on its
  0.02 grid, and every smooth counterpart has zero mass in the excess window. The report shows the
  DP's window probability instead: 0.879 at w = 0, and 0 elsewhere.
  - This is recorded, not re-defined: a replacement truth definition would be a post-hoc design
    change.
- **Reconciliation power.** The per-(period, good) form (S4 amendment) has size 0.05 and power
  0.90 at a 5% inflated share, and 1.00 from 10% up.
- **S4 amendment:** the "reused" 30-seed arms were retrained under spec 2.0.x, because the config
  hashes changed. The G2 artefacts were left untouched.

## Price sensitivity (WO-036): `runs/price_sensitivity/report.md`

- Across price seeds 11, 12 and 13, `specification_gap` keeps its sign in every table's mean.
  One C_INC row of 15 flips.
- welfare_ratio is price-invariant to 4 decimals in every arm. padding_index moves by about 3-10%.
- For C_INC and C_BOTH (price-reading objective) this is a recomputation, not a behavioural rerun.

## LLM ministry study (WO-035): `runs/llm_study/report.md`

**NOT RUN.** The harness, the dominance check and the not-run report are implemented and
unit-tested. The owner's command is in the report: install the SDK, add a credential, then
`uv run python -m gosplan.experiments.llm_study claude-opus-5 claude-sonnet-5`.

## Next (owner decisions at G3, delegated)

1. Apply R15 (trade offers posted at REPORT; spec 2.1.0) and R16 (the revised exploitability
   audit).
2. Run the labelled **G3b** study: C0 at 30 seeds under spec 2.1.0. It re-evaluates row 6 and runs
   the R16 audit. The result goes in `runs/G3b_record.md`. G3 stays NOT PASSED as recorded.
3. G4 sign-off belongs to the human.
