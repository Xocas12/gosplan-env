AMBIGUITY REPORT   WO-015   gosplan/experiments/regime_map.py
The two open items the WO-015 card itself lists, plus the missing smoke test.

1. Factorisation of `a * pen`. PLAN section 5 sweeps the compound; `EnvConfig` carries
   `information.audit_rate` and `incentive.penalty_scale` separately.
2. "Interior" of the bunching region: PLAN section 5 asks the human to pick from the interior but
   defines no neighbourhood.
3. The smoke test the card names (`tests/unit/test_regime_map_runs.py`) is not on WO-002's write
   list and does not exist; only the lead writes frozen tests.

LEAD RESOLUTION (2026-09-24):
1. Hold `audit_rate` at the base configuration's value and set
   `penalty_scale = audit_times_penalty / audit_rate`. The sampled dimension is the compound PLAN
   section 5 names; holding the audit frequency fixed keeps the single-enterprise DP's expected
   penalty a function of the compound alone (it enters the Bellman operator only as `a * E[Pen]`).
   `run(...)["ap_factorisation"]` = "audit_rate held at base; penalty_scale = a_pen / audit_rate".
   The table records `audit_rate`, `penalty_scale` and their product.
2. A `bunching` point is interior when its k = 5 nearest neighbours in the design - Euclidean
   distance over the seven continuous dimensions, each normalised to [0, 1] by its `LHS_RANGES`
   span, among points with the SAME `notch_width` level - are all labelled `bunching`. k-nearest
   neighbours rather than a fixed radius because a Latin-hypercube design has no fixed spacing.
   The definition is printed in `candidates.md` beside every candidate, with the neighbours'
   labels. Candidates are listed in decreasing order of the smallest distance to a non-bunching
   point (deepest first); a non-finite `b_hat_DP` (AMBIGUITY-011 addendum K) is shown as such,
   never replaced.
3. The lead authors `tests/unit/test_regime_map_runs.py`: a two-point design writes the four
   artefacts and the documented return mapping (shape only, no regime expectation).
