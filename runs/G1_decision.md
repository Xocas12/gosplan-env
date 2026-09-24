# Gate G1 — Phase-1 parameter decision

**Decision-maker.** The owner (Xocas12) DELEGATED this human decision to the lead (the Claude
session driving this branch) on 2026-09-24, in writing: "Pick for me". It was taken AFTER the
spec v1.0.0 freeze (WO-013) and the regime map (WO-015), and BEFORE ANY TRAINING RUN — no training
artefact exists under `runs/`. The owner may overrule it; doing so after training starts would
create a new, labelled study (PLAN section 13).

## Source

`runs/regime_map/candidates.md` (500-point Latin-hypercube DP regime map, seed 0, all converged, 0
grid-edge hits; 71 interior bunching candidates under the AMBIGUITY-012 interior rule).

## Choice: design point 49

Interior (all 5 nearest same-`notch_width` neighbours are `bunching`; depth 0.63) and, of the 71
candidates, the one CLOSEST to the provisional Phase-1 economy the plan was designed around
(normalised distance 0.26). The deepest candidates sit at extremes (`ratchet_lambda` ~ 1,
`tenure` ~ 0.97) that lengthen effective horizons and make PPO recovery (G2 criterion 1) harder
without making the bunching region any more certain.

| Daggered parameter (PLAN section 3) | Provisional | **G1 value** |
|---|---|---|
| `ratchet_lambda` | 0.5 | **0.53** |
| `growth_directive` | 0.02 | **0.021** |
| `overfulfilment_slope` | 0.5 | **0.331** |
| `effort_cost` | 0.15 | **0.193** |
| `audit_rate` | 0.10 | **0.10** (held: AMBIGUITY-012 factorisation) |
| `penalty_scale` | 60.0 | **200.0** (so `a*pen` = 20.0, the top G1 level below) |

Non-daggered parameters stay at their registry defaults (`notch_height` 1.0, `tenure` 0.9,
`notch_width` 0 in the notched arm). Point 49 itself had `notch_height` 0.928 and `tenure` 0.895;
the configuration actually adopted — defaults plus the six values above — was re-solved and is
`bunching` at every level below.

## Three `a*pen` levels (G2 criteria 1 and 3)

A 1-D DP scan along `a*pen` at point 49 (24 log-spaced values over the registry range [0.05, 60])
found `bunching` for every `a*pen` >= 0.43 and a sharp transition to zero-effort padding below it.
The three levels sit inside that region, log-spaced and away from its lower edge:

| `a*pen` | `penalty_scale` (a = 0.10) | DP regime (adopted config) | DP fictitious padding | DP mean effort | DP share of reports in [1.00, 1.02] | `b_hat_DP` | config hash (prefix) |
|---|---|---|---|---|---|---|---|
| 0.8 | 8.0 | bunching | 0.0055 | 0.495 | 0.8838 | inf | `f5e3dc5569a6` |
| 4.0 | 40.0 | bunching | 0.0016 | 0.498 | 0.8838 | inf | `7417c575a340` |
| 20.0 | 200.0 | bunching | 0.0006 | 0.513 | 0.8794 | inf | `22a662ee58e7` |

DP padding is monotone decreasing in `a*pen`, the pattern G2 criterion 3 tests. The smooth arm
(`notch_width` 0.25) at the same three levels is `mixed` with no stationary mass at rho = 1.

## `b_hat_DP` thresholds (G2 criterion 2)

`b_hat_DP` is non-finite at all three levels: the DP places its stationary report mass on grid
points, so the polynomial counterfactual has no support (AMBIGUITY-011 addendum K). Under the
AMBIGUITY-011 resolution, the notched-arm threshold is therefore the DP's excess-window share:
**learned share of REPORT rows in [1.00, 1.02] >= 0.5 x 0.8794 = 0.440** at the Phase-1
configuration (`a*pen` = 20), with the seed-level CI of `b_hat` excluding 0. The smooth-arm half of
criterion 2 is unchanged (CI for `b_hat` covers 0 in >= 90% of seeds), evaluated with the degree-9
estimator adopted in spec 1.0.1.

## Recorded before any training

Folded into `p1_default_config()` / `gosplan/params.py` / `spec/spec.py` by the LEAD with a
`spec/CHANGELOG.md` entry (1.1.0).
