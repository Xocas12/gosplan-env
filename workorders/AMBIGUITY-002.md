AMBIGUITY REPORT   WO-031 / WO-037   gosplan/experiments/
Question (one sentence):
What are the module names for the P2 acceptance harness and the P3 report generator, given that
PLAN §8's tree does not list them but WO-031 and WO-037 must write something?

What the spec says / does not say (quote):
PLAN §8 draws ten modules under `gosplan/experiments/`: mc_sanity, regime_map, dp_vs_ppo,
phase1_gate, exploitability, contrasts, sobol, estimator_bias, llm_study, price_sensitivity.
PLAN §12.4 gives WO-031 as "P2 acceptance experiment (G3) — MID-strong writes the harness; LEAD
runs" and §12.5 gives WO-037 as "report generation: figures, tables, manifest roll-up", neither
with a filename. The scaffolded cards use `gosplan/experiments/phase2_acceptance.py` and
`gosplan/experiments/report.py`, both flagged "Provisional; the revision fixes the final list and
may rename the module".

Options considered (A/B/…), and why the spec does not decide:
A. Add the two names to PLAN §8's tree now and create the stubs, so §8 stays the authoritative
   tree and every deliverable has a drawn home.
B. Add a note under §8's tree stating that P2/P3 harness module names are fixed at the P2 spec
   revision and are deliberately not drawn, leaving the two files uncreated until then.
The spec does not decide because §8 is presented as the authoritative layout while §12.4/§12.5
are explicitly "summaries" whose cards are "issued after the P2 spec revision".

Impact if the wrong option is picked:
Cosmetic and reversible either way: a renamed harness module costs one card edit. Choosing A now
freezes two names before the revision that §12.4 says fixes them; choosing B leaves §8 unable to
account for two named deliverables until G2 is passed.

Tests blocked:
None. Both modules are P2/P3 and no P1 test imports them. Skeleton status: neither file exists;
`gosplan/experiments/` holds exactly the ten modules PLAN §8 draws.
