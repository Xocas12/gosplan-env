AMBIGUITY REPORT   WO-015   gosplan/experiments/regime_map.py
Three choices the WO-015 session found undetermined after AMBIGUITY-012.

A. Which `rho_edge_frac` threshold defines `edge_hit_points` ("above the `classify_regime`
   threshold, i.e. the report grid was hit and the grid was extended"; `classify_regime` has two
   edge thresholds, 0.05 and 0.5, and a grid can be extended yet end with no edge mass).
B. What 2-D projection of the 8-D design the two "heat maps" show (PLAN section 5 fixes none).
C. Where the CONTRACT rule 10 manifest goes, and which field carries the LHS seed and the
   factorisation convention (`write_manifest` accepts only `MANIFEST_FIELDS`).

LEAD RESOLUTION (2026-09-24):
A. A point is an edge hit when its report grid was extended past `DPGrid().rho_hi` OR any
   stationary mass remains at the top grid point - the most inclusive reading, because grid-edge
   hits are a result to report (CONTRACT rule 8), never a count to minimise.
B. Pairwise scatter matrices (lower triangle) of the seven continuous dimensions, one block per
   `notch_width` level, every design point at its own coordinates - no binning, so nothing is
   averaged away. `regime.png`: categorical colours by label. `bhat.png`: finite `b_hat_DP` on a
   continuous scale; non-finite values (AMBIGUITY-011 K) as black crosses with a legend entry,
   never clipped. Edge-hit points outlined in red in both.
C. One manifest for the whole map at `<out_dir>/manifest.json`, written for the base configuration
   (each point's own `config_hash` is a table column). `flags` carries `lhs_seed=...`,
   `ap_factorisation=...`, `n_points=...`, `n_converged=...`; `solver` names the PLAN section 5 DP
   and `solver_version` the `DPGrid` it ran on; `git_hash` is `git rev-parse HEAD` with `-dirty`.
Also noted (information, not blocking): with `audit_rate` held at 0.10, `penalty_scale` spans
[0.5, 600], wider than the PLAN section 3 registry range [5, 200]; `validate()` accepts it and the
compound `a * pen` stays inside its registry range [0.05, 60], which is what the map samples.
