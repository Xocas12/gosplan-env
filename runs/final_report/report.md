# gosplan-env - final report (WO-037, gate G4)

Every section below is the producing study's own report, embedded unedited, so its labels (held-out, NON-CONVERGED, NOT RUN, reduced designs) survive verbatim. Pre-registrations: PLAN.md sections 4 and 7, spec/P2_REVISION.md (R1-R14), spec/P3_REVISION.md (S1-S7).

## G4 checklist - NOT MET

- contrasts with CIs: complete (`runs/contrasts/report.md`)
- estimator-bias curves: complete (`runs/estimator_bias/report.md`)
- LLM study: NOT RUN (`runs/llm_study/report.md`)
- price sensitivity on every headline table: complete (`runs/price_sensitivity/report.md`)

- Sobol (optional): The optional Saltelli/Sobol design (WO-033) was not run: PLAN marks it optional and JAX-only, and at about 46 min per Phase-2 run its 1,500-2,800 runs are out of reach without a JAX trainer (spec/P3_REVISION.md S3).

G4 is signed off by the human (PLAN section 13); this checklist does not sign it.

---

## Gate G1 - regime map and Phase-1 values

_Source: `runs/G1_decision.md`_

### Gate G1 — Phase-1 parameter decision

**Decision-maker.** The owner (Xocas12) DELEGATED this human decision to the lead (the Claude
session driving this branch) on 2026-09-24, in writing: "Pick for me". It was taken AFTER the
spec v1.0.0 freeze (WO-013) and the regime map (WO-015), and BEFORE ANY TRAINING RUN — no training
artefact exists under `runs/`. The owner may overrule it; doing so after training starts would
create a new, labelled study (PLAN section 13).

#### Source

`runs/regime_map/candidates.md` (500-point Latin-hypercube DP regime map, seed 0, all converged, 0
grid-edge hits; 71 interior bunching candidates under the AMBIGUITY-012 interior rule).

#### Choice: design point 49

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

#### Three `a*pen` levels (G2 criteria 1 and 3)

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

#### `b_hat_DP` thresholds (G2 criterion 2)

`b_hat_DP` is non-finite at all three levels: the DP places its stationary report mass on grid
points, so the polynomial counterfactual has no support (AMBIGUITY-011 addendum K). Under the
AMBIGUITY-011 resolution, the notched-arm threshold is therefore the DP's excess-window share:
**learned share of REPORT rows in [1.00, 1.02] >= 0.5 x 0.8794 = 0.440** at the Phase-1
configuration (`a*pen` = 20), with the seed-level CI of `b_hat` excluding 0. The smooth-arm half of
criterion 2 is unchanged (CI for `b_hat` covers 0 in >= 90% of seeds), evaluated with the degree-9
estimator adopted in spec 1.0.1.

#### Recorded before any training

Folded into `p1_default_config()` / `gosplan/params.py` / `spec/spec.py` by the LEAD with a
`spec/CHANGELOG.md` entry (1.1.0).

---

## Gate G2 - record

_Source: `runs/G2_record.md`_

### Gate G2 record - Phase 1

**Status: G2 NOT PASSED.** Criterion 1 was not met in three attempts. By the owner's decision
(AMBIGUITY-021), criteria 2-4 were run afterwards as a **separately labelled study**, not as the
gate. Phase 2 (WO-021 onwards) needs the owner's G2 decision on the record below.

Sources: `runs/dp_vs_ppo/report.md` (attempt 3), `runs/dp_vs_ppo/attempt1/`, `attempt2/`,
`runs/phase1_gate/report.md`, `workorders/AMBIGUITY-019.md` to `AMBIGUITY-022.md`.

#### Criterion 1 - PPO recovers the single-enterprise DP (`N = 1`): NOT MET

| Attempt | Learner | Seeds passing (of 30) | Mean return at `a*pen` 0.8 / 4 / 20 | Main miss |
|---|---|---|---|---|
| 1 | discount 0.99 per agent-step, report-head std 0.05 | 0 | 4.84 / 5.37 / 3.23 | effort too high, W1 |
| 2 | discount 0.99 per plan period (AMBIGUITY-020) | 0 | 5.38 / 5.23 / 3.76 | effort ~0.81 vs 0.50, W1 ~0.15 |
| 3 | + wide report-head init (owner option A) | 0 | 3.17 / 2.69 / 2.46 | under-reports, effort ~0.24, W1 ~0.7 |
| DP policy replayed in the environment | - | - | 7.13 / 6.43 / 5.26 | - |

Two genuine defects were found and fixed on the way: the audit draw keyed by the configuration's
root seed rather than the episode's (AMBIGUITY-019 B), and the learner discounting per agent-step
where the spec's DP discounts per plan period (AMBIGUITY-020). The remaining gap
(AMBIGUITY-021) is one of exploration: the DP's optimum mixes about 12% deliberate zero reports,
which walk the target down through the ratchet, with truthful reports at the notch. A PPO learner
starting on the notch (std 0.05) never finds the zero reports. One starting wide (sigma_z = 1)
finds them but overshoots to 27-49% of reports. Neither reproduces the DP's policy within the
PLAN section 4.5 tolerances.

#### Labelled study - criteria 2-4 (`N = 20`, attempt-2 learner, 80 runs)

**Criterion 2 - bunching present / absent at `a*pen` = 20: FAIL** (strict and lenient readings,
AMBIGUITY-022).
- Notched arm: 77% of measured reports lie in [1.00, 1.02] on average (range 0-100%; the
  threshold is 44%). 23 of 30 seeds clear the share threshold: 11 with a CI excluding 0, and
  12 where the CI is undefined because all their mass is inside the excluded window. That is 37%
  of seeds under the strict reading and 77% under the lenient one; 90% is needed.
- Smooth arm: 3.7% of reports in the window on average (max 13%), so there is no bunching. But
  the estimator's CI covers 0 in only 10 of 30 seeds (33%). The point estimates centre on zero
  (median `b_hat` 0.08) with narrow per-seed intervals of both signs. This fits the residual
  estimator bias on peaked distributions recorded in AMBIGUITY-011 (0.147 at sigma = 0.10).
- Reading: the notch produces a large, qualitatively unmistakable spike that the smooth schedule
  does not (77% vs 3.7% of reports at the notch). The pre-registered per-seed test does not
  certify it, in both arms, for reasons tied to the estimator: no counterfactual support in the
  notched arm, and tight biased intervals in the smooth arm.

**Criterion 3 - padding elasticity: FAIL.** Padding is monotone decreasing in `a*pen` (0.1083 /
0.0019 / 0.0001). It is within 0.0004 of the DP at `a*pen` = 4 and 20, but 0.103 above it at
0.8, where the learner pads and the DP does not (the same gap as criterion 1 at that level).

**Criterion 4 - hygiene: FAIL.** `BOUND_BINDING` is raised in 0 of 80 runs. Target runaways
(`T > 3 T_0` after the first 20% of training) occur in at most 0.1% of training episodes at the
Phase-1 configuration and 4.7% at `a*pen` = 4, but in up to 10.9% at `a*pen` = 0.8 (the limit is
5%).

**Price sensitivity:** deferred to WO-036 (AMBIGUITY-019 D).

#### What the owner decides at G2

The PLAN (section 4.5) makes criterion 1 blocking. The evidence is that PPO does not recover the
DP's mixed under-reporting strategy under any of three learner settings, while at `N = 20` it
does produce heavy bunching at the notch and none under the smooth schedule. The options are:
- (a) Accept Phase 1 as a labelled result and proceed to Phase 2 with this learner, carrying the
  criterion-1 gap as a stated limitation.
- (b) Commission further training-stack work first. Candidates: a longer budget, entropy
  settings, or a learner that can represent mixed strategies. Each would be a new labelled study.
- (c) Revisit the pre-registered criterion-2 estimator for degenerate and peaked distributions,
  as a spec revision with its own record.
- (d) Stop at Phase 1.

#### OWNER DECISION (2026-09-26): option (a)

The owner instructed the LEAD to continue building. Recorded as option (a): Phase 1 stands as a
labelled result. The criterion-1 gap (PPO does not recover the DP's mixed under-reporting
strategy) and the criterion-2 estimator behaviour are carried into Phase 2 as stated limitations,
and every Phase-2 report cites them. Phase 2 proceeds: the P2 spec revision (LEAD), then WO-021 to
WO-031, then G3. The attempt-2 learner (`phase1_gate.study_ppo_config()`) remains the working
learner.

---

## Phase 1 - DP vs PPO (G2 criterion 1)

_Source: `runs/dp_vs_ppo/report.md`_

### Gate G2 criterion 1 - DP-vs-PPO recovery (WO-019)

Git: `b64e721a000306e8180b9215a15a7d34702331e3`. Setting: PLAN section 5 (`N = 1`, `a = 0`, `phi = 1`), the G1
configuration at each recorded `a*pen` level. Sizing (AMBIGUITY-019): `RunSizing(n_envs=8, rollout_steps=125, total_agent_steps=2000000, eval_every_updates=200, eval_episodes=50, measure_episodes=400)`.
Tolerances (PLAN section 4.5): |padding| <= 0.02, |effort| <= 0.05,
W1 < 0.03; a level passes with >= 8 of 10 seeds.

#### DP reference

| `a*pen` | regime | padding | effort | share in [1.00, 1.02] | DP policy return in env (context) | converged | config hash |
|---|---|---|---|---|---|---|---|
| 0.8 | bunching | 0.0055 | 0.495 | 0.8838 | 7.125 | True | `1dcece8c41c9` |
| 4 | bunching | 0.0016 | 0.498 | 0.8838 | 6.426 | True | `bfac9e5495ba` |
| 20 | bunching | 0.0006 | 0.513 | 0.8794 | 5.262 | True | `5e3b553dd8ee` |

#### Per seed

| `a*pen` | seed | padding PPO | effort PPO | W1 | share PPO | return PPO | pass |
|---|---|---|---|---|---|---|---|
| 0.8 | 0 | 0.0000 | 0.427 | 0.3645 | 0.0106 | 3.732 | no |
| 0.8 | 1 | 0.0000 | 0.406 | 0.5500 | 0.0024 | 3.763 | no |
| 0.8 | 2 | 0.0000 | 0.408 | 0.4929 | 0.0116 | 3.671 | no |
| 0.8 | 3 | 0.0000 | 0.327 | 0.6189 | 0.0005 | 3.597 | no |
| 0.8 | 4 | 0.0000 | 0.000 | 0.8838 | 0.0000 | -0.000 | no |
| 0.8 | 5 | 0.0000 | 0.056 | 0.7843 | 0.0005 | 1.922 | no |
| 0.8 | 6 | 0.0938 | 0.301 | 0.5493 | 0.0000 | 3.944 | no |
| 0.8 | 7 | 0.0000 | 0.349 | 0.4255 | 0.0068 | 3.904 | no |
| 0.8 | 8 | 0.0000 | 0.353 | 0.7155 | 0.0003 | 3.568 | no |
| 0.8 | 9 | 0.0000 | 0.241 | 0.5385 | 0.0029 | 3.548 | no |
| 4 | 0 | 0.0000 | 0.428 | 0.5591 | 0.0024 | 3.251 | no |
| 4 | 1 | 0.0000 | 0.135 | 1.0136 | 0.0000 | 1.938 | no |
| 4 | 2 | 0.0000 | 0.316 | 0.6163 | 0.0000 | 3.231 | no |
| 4 | 3 | 0.0459 | 0.206 | 0.9315 | 0.0026 | 1.556 | no |
| 4 | 4 | 0.0000 | 0.110 | 0.8498 | 0.0000 | 2.150 | no |
| 4 | 5 | 0.0000 | 0.057 | 0.7927 | 0.0003 | 1.854 | no |
| 4 | 6 | 0.0000 | 0.228 | 0.5620 | 0.0021 | 3.856 | no |
| 4 | 7 | 0.0000 | 0.294 | 0.4284 | 0.0047 | 3.585 | no |
| 4 | 8 | 0.0000 | 0.280 | 0.5003 | 0.0076 | 3.320 | no |
| 4 | 9 | 0.0000 | 0.095 | 0.7107 | 0.0011 | 2.148 | no |
| 20 | 0 | 0.0000 | 0.457 | 0.3819 | 0.0077 | 2.996 | no |
| 20 | 1 | 0.0002 | 0.119 | 0.9440 | 0.0005 | 1.658 | no |
| 20 | 2 | 0.0000 | 0.392 | 0.8332 | 0.0008 | 2.507 | no |
| 20 | 3 | 0.0000 | 0.203 | 0.8537 | 0.0003 | 2.292 | no |
| 20 | 4 | 0.0000 | 0.000 | 0.8794 | 0.0000 | -0.000 | no |
| 20 | 5 | 0.0000 | 0.239 | 0.6647 | 0.0032 | 3.205 | no |
| 20 | 6 | 0.0000 | 0.197 | 0.7228 | 0.0011 | 2.503 | no |
| 20 | 7 | 0.0000 | 0.169 | 0.6332 | 0.0050 | 2.894 | no |
| 20 | 8 | 0.0000 | 0.363 | 0.7404 | 0.0008 | 3.501 | no |
| 20 | 9 | 0.0000 | 0.244 | 0.7013 | 0.0011 | 3.029 | no |

#### Result

- `a*pen` = 0.8: 0/10 seeds pass -> **FAIL**
- `a*pen` = 4: 0/10 seeds pass -> **FAIL**
- `a*pen` = 20: 0/10 seeds pass -> **FAIL**

**criterion_1_passed: False**

Flags raised: none.

Per PLAN section 4.5 a criterion-1 failure is a training-stack failure: it blocks
criteria 2-4 and the next step is a LEAD diagnosis work order, never a tolerance,
level or parameter change.

Total training wall clock: 6.8 h over 30 runs.

---

## Phase 1 - labelled study, G2 criteria 2-4

_Source: `runs/phase1_gate/report.md`_

### Gate G2 criteria 2-4 - Phase-1 gate (WO-020)

**LABELLED STUDY - criterion 1 NOT met** (`runs/dp_vs_ppo/report.md`, AMBIGUITY-021). By the owner's decision these criteria are run and reported, but they are not a G2 pass.

Git: `0b7a96ebde2b5727fdc1ff9854f8807f977a28e3`. Configuration: G1 record at `N = 20`, `a*pen` = 20.
Sizing (AMBIGUITY-019): `RunSizing(n_envs=8, rollout_steps=125, total_agent_steps=1000000, eval_every_updates=250, eval_episodes=10, measure_episodes=100)`; 10 seeds per extra level.
Learner (AMBIGUITY-021, owner's choice): report-head init std 0.05 in ratio units, per-period discount - the attempt-2 learner.
Estimator: `gosplan.metrics._fallback` 0.1.0+fallback, PLAN section 4.5 settings (degree 9).

#### Criterion 2 - bunching present / absent

Notched condition: learned share in [1.00, 1.02] >= 0.440 and CI of
`b_hat` excluding 0. Smooth condition: CI of `b_hat` covers 0.

| arm | seed | b_hat | CI | hole | share | padding | effort | pass |
|---|---|---|---|---|---|---|---|---|
| notched | 0 | inf | [nan, nan] | -inf | 1.000 | 0.0000 | 0.644 | undefined |
| notched | 1 | 7126.810 | [5017.699, 11291.234] | 10.000 | 0.998 | 0.0000 | 0.644 | pass |
| notched | 2 | 278657.296 | [78780.525, 308513.337] | 10.000 | 1.000 | 0.0000 | 0.632 | pass |
| notched | 3 | inf | [nan, nan] | nan | 1.000 | 0.0000 | 0.639 | undefined |
| notched | 4 | -4.000 | [-4.000, -4.000] | 10.000 | 0.000 | 0.0031 | 0.712 | fail |
| notched | 5 | inf | [nan, nan] | nan | 1.000 | 0.0000 | 0.634 | undefined |
| notched | 6 | 112.401 | [105.590, 119.165] | 10.000 | 0.887 | 0.0000 | 0.633 | pass |
| notched | 7 | -1.751 | [-1.843, -1.632] | 8.750 | 0.109 | 0.0000 | 0.536 | fail |
| notched | 8 | 5.800 | [4.871, 6.676] | 8.558 | 0.340 | 0.0000 | 0.554 | fail |
| notched | 9 | inf | [nan, nan] | nan | 1.000 | 0.0000 | 0.641 | undefined |
| notched | 10 | inf | [nan, nan] | nan | 1.000 | 0.0000 | 0.634 | undefined |
| notched | 11 | 1.931 | [1.757, 2.093] | 10.000 | 0.284 | 0.0000 | 0.702 | fail |
| notched | 12 | inf | [nan, nan] | nan | 1.000 | 0.0000 | 0.637 | undefined |
| notched | 13 | inf | [nan, nan] | -inf | 1.000 | 0.0000 | 0.635 | undefined |
| notched | 14 | 28644.249 | [18654.461, 57249.783] | 10.000 | 0.999 | 0.0000 | 0.637 | pass |
| notched | 15 | 286021.884 | [78863.377, 313889.486] | 10.000 | 1.000 | 0.0000 | 0.648 | pass |
| notched | 16 | inf | [nan, nan] | -inf | 0.918 | 0.0000 | 0.620 | undefined |
| notched | 17 | 29771.849 | [17720.791, 80675.311] | -1824.973 | 0.948 | 0.0000 | 0.622 | pass |
| notched | 18 | 7384.620 | [5303.317, 12461.046] | 10.000 | 0.998 | 0.0000 | 0.647 | pass |
| notched | 19 | 86.136 | [74.574, 101.566] | -0.485 | 0.789 | 0.0000 | 0.584 | pass |
| notched | 20 | 92185.918 | [41892.091, 287877.760] | 10.000 | 1.000 | 0.0000 | 0.643 | pass |
| notched | 21 | inf | [nan, nan] | nan | 1.000 | 0.0000 | 0.626 | undefined |
| notched | 22 | 36.379 | [34.947, 37.674] | 10.000 | 0.731 | 0.0000 | 0.675 | pass |
| notched | 23 | 1.583 | [0.667, 2.472] | 10.000 | 0.260 | 0.0000 | 0.674 | fail |
| notched | 24 | 93.289 | [76.511, 123.083] | 10.000 | 0.867 | 0.0002 | 0.653 | pass |
| notched | 25 | inf | [nan, nan] | nan | 1.000 | 0.0000 | 0.632 | undefined |
| notched | 26 | inf | [nan, nan] | -inf | 0.823 | 0.0000 | 0.577 | undefined |
| notched | 27 | -3.132 | [-3.272, -3.003] | 10.000 | 0.054 | 0.0003 | 0.703 | fail |
| notched | 28 | -3.992 | [-3.997, -3.986] | 10.000 | 0.000 | 0.0003 | 0.654 | fail |
| notched | 29 | inf | [nan, nan] | nan | 1.000 | 0.0000 | 0.639 | undefined |
| smooth | 0 | 1.033 | [0.721, 1.344] | -2.178 | 0.074 | 0.0000 | 0.485 | fail |
| smooth | 1 | 1.129 | [0.521, 1.798] | -0.403 | 0.017 | 0.0000 | 0.582 | fail |
| smooth | 2 | -0.654 | [-1.226, -0.017] | 1.629 | 0.008 | 0.0001 | 0.555 | fail |
| smooth | 3 | -0.454 | [-0.962, 0.067] | 1.779 | 0.009 | 0.0001 | 0.585 | pass |
| smooth | 4 | -6.773 | [-9.270, -5.433] | 17.804 | 0.001 | 0.0000 | 0.648 | fail |
| smooth | 5 | 2.044 | [1.714, 2.351] | -4.762 | 0.126 | 0.0000 | 0.507 | fail |
| smooth | 6 | 1.288 | [0.138, 2.887] | -7.001 | 0.005 | 0.0000 | 0.646 | fail |
| smooth | 7 | -0.686 | [-1.201, -0.132] | 0.681 | 0.010 | 0.0000 | 0.591 | fail |
| smooth | 8 | 0.382 | [-0.453, 1.328] | -0.469 | 0.007 | 0.0000 | 0.624 | pass |
| smooth | 9 | -1.065 | [-1.524, -0.564] | 1.420 | 0.010 | 0.0002 | 0.594 | fail |
| smooth | 10 | 0.222 | [-0.393, 0.943] | -0.810 | 0.010 | 0.0000 | 0.556 | pass |
| smooth | 11 | 0.458 | [-0.101, 0.992] | -0.648 | 0.015 | 0.0000 | 0.550 | pass |
| smooth | 12 | -1.591 | [-1.766, -1.368] | 6.300 | 0.075 | 0.0000 | 0.441 | fail |
| smooth | 13 | -0.345 | [-0.905, 0.225] | 1.419 | 0.010 | 0.0000 | 0.544 | pass |
| smooth | 14 | 0.796 | [0.509, 1.085] | -2.564 | 0.066 | 0.0000 | 0.457 | fail |
| smooth | 15 | 1.697 | [1.421, 2.001] | 1.323 | 0.133 | 0.0000 | 0.471 | fail |
| smooth | 16 | -0.016 | [-0.576, 0.614] | -0.578 | 0.013 | 0.0000 | 0.601 | pass |
| smooth | 17 | 1.656 | [1.376, 1.939] | -4.117 | 0.111 | 0.0000 | 0.472 | fail |
| smooth | 18 | -0.923 | [-1.351, -0.466] | 2.112 | 0.009 | 0.0000 | 0.589 | fail |
| smooth | 19 | -0.629 | [-1.029, -0.228] | -0.005 | 0.015 | 0.0000 | 0.562 | fail |
| smooth | 20 | 0.533 | [0.260, 0.860] | 4.126 | 0.103 | 0.0000 | 0.427 | fail |
| smooth | 21 | -0.572 | [-1.068, -0.030] | 0.710 | 0.013 | 0.0001 | 0.609 | fail |
| smooth | 22 | 0.530 | [-0.010, 1.168] | -0.107 | 0.014 | 0.0001 | 0.589 | pass |
| smooth | 23 | -0.655 | [-1.221, -0.097] | -0.135 | 0.008 | 0.0001 | 0.598 | fail |
| smooth | 24 | 0.083 | [-0.109, 0.285] | 3.299 | 0.107 | 0.0000 | 0.481 | pass |
| smooth | 25 | 1.502 | [1.210, 1.780] | 0.500 | 0.092 | 0.0000 | 0.448 | fail |
| smooth | 26 | -0.660 | [-1.013, -0.301] | 1.011 | 0.021 | 0.0000 | 0.513 | fail |
| smooth | 27 | 0.052 | [-0.344, 0.512] | 2.740 | 0.021 | 0.0000 | 0.474 | pass |
| smooth | 28 | 3.872 | [1.293, 8.794] | -165.030 | 0.004 | 0.0001 | 0.644 | fail |
| smooth | 29 | 0.072 | [-0.426, 0.651] | 0.135 | 0.014 | 0.0000 | 0.606 | pass |

- notched: 37% of seeds pass (need >= 90%); 77% if an undefined CI counts as met
- smooth: 33% of seeds pass (need >= 90%); 33% if an undefined CI counts as met
- seeds with an undefined CI (all measured mass inside the excluded window, so the polynomial counterfactual has no support; AMBIGUITY-022): 12
- **criterion_2 (strict: undefined = not met): FAIL**; with undefined counted as met: FAIL

#### Criterion 3 - padding elasticity in `a*pen`

| `a*pen` | learned padding (mean over seeds) | DP padding | deviation |
|---|---|---|---|
| 0.8 | 0.1083 | 0.0055 | 0.1029 |
| 4 | 0.0019 | 0.0016 | 0.0003 |
| 20 | 0.0001 | 0.0006 | 0.0004 |

- monotone decreasing: True; tolerance 0.03
- **criterion_3: FAIL**

#### Criterion 4 - hygiene

- runs with `BOUND_BINDING`: 0 of 80
- max share of training episodes with `T > 3 T_0` after the first 20% of training: 0.1085 (need < 0.05)
- **criterion_4: FAIL**

#### Price sensitivity

PLAN section 7.5's standing price check recomputes padding_index, welfare_ratio and specification_gap; the last two need the Phase-2 oracle (WO-026) and the check itself is WO-036 (Phase 3). No G2 criterion is price-weighted, so the table is deferred to WO-036, which must run it on this gate's ledgers (AMBIGUITY-019).

**All three criteria passed: False**. Flags raised: none.

If criterion 1 passed in `runs/dp_vs_ppo/report.md`, a criterion-2 failure is a
multi-agent effect and is a result, reported, never tuned away (PLAN section 4.5).

Total training wall clock: 23.9 h over 80 runs.

---

## Gate G3 - record

_Source: `runs/G3_record.md`_

### Gate G3 record - Phase 2

**Status: G3 NOT PASSED.** The LEAD signed this on 2026-09-27. The human sign-off is pending
(PLAN section 13: "Human + LEAD").

- Source: `runs/phase2_acceptance/report.md`, from the pre-registered design `spec/P2_REVISION.md`
  R14.
- The run started on 2026-09-26 at 10:05 UTC and ended on 2026-09-27 at 01:37 UTC, git `51132cd`.
- Arms: C0 (30 seeds), R7_NULL (10) and R3_QW (10), each trained at gate sizing with the attempt-2
  learner.
- Limitations carried forward: L1 and L2 (`runs/G2_record.md`).

#### Conditions, as pre-registered

| G3 condition | Result |
|---|---|
| Held-out rows 2, 5, 6, 7 evaluated on the section 4.2 values | done (per-row outcomes below) |
| Exploitability below threshold on all arms | **FAIL**, and see D2: the audit is inconclusive |
| Oracle gap recorded | PASS (HiGHS via OR-Tools 9.15, gap 0, horizon 40) |
| JAX parity | PASS (max deviation 1.1e-14 for P1, 7.1e-14 for P2; tolerance 1e-5) |
| Price sensitivity | computed. `specification_gap` 0.0105 keeps its sign under all three vectors |
| Hygiene | **FAIL**: runaway share 0.154 against a limit of 0.05, in R7_NULL and R3_QW. C0's is 0.0. `BOUND_BINDING` 0 |

#### Held-out rows (C0; mean over seeds, 95% seed-bootstrap CI)

| Row | Statistic | Outcome |
|---|---|---|
| 2 storming | excess within-period effort Gini 0.137 [0.095, 0.184] | **appears** |
| 5 hoarding | request inflation excess 1.96 [1.94, 1.98]; corr(X, shortfall) excess 0.55 [0.38, 0.72] | **appears**; see note H1 |
| 6 blat | trade volume share 0.0000 | **not evaluable**: defect D1 |
| 7 hidden reserves | C0 0.222 [0.092, 0.390]; R7_NULL 0.219 [0.055, 0.439] | **failure**: present, but does not vanish under the null |
| 3 quality (pipeline check) | mean qbar, C0 minus R3_QW: 0.152 [-0.074, 0.339] | **failure** |

**H1.** The learned requests sit at `request_max_multiple = 3.0` (mean inflation 2.96). The
direction of row 5 is a result. Its magnitude is set by the action bound, and in the spirit of
CONTRACT rule 8 it is reported that way, not as a free estimate.

**Welfare.**
- The learned populations reach `welfare_ratio = 0.034` (W 0.067 against `W_oracle` 1.98).
- The truthful-myopic baseline reaches W 1.06 (ratio about 0.53).
- The learned economy degrades quality: C0's mean qbar is 0.38. Deliveries are scaled by `qbar`,
  and production is near-Leontief (theta = 8), so the shortfall compounds into a collapse of
  consumer deliveries.
- This is a behavioural result of the pre-registered mechanisms, not a pipeline artefact. The
  oracle and the truthful baseline use the same measurement.

#### LEAD diagnosis

**D1 - trade was structurally impossible (a pipeline defect).**
- The WO-017 PPO adapter treats `trade_offer` as a REPORT-phase action (`_REPORT_DIMS`, following
  PLAN section 2.3's action table).
- The WO-024 trade stage (spec/P2_REVISION.md R9) runs at step 0, after DELIVER, and reads the
  step-0 PRODUCE action.
- In that PRODUCE action the adapter masks `trade_offer` to 0, so no learner could ever post an
  offer. Row 6's zero therefore measures the plumbing, not the economy.
- No unit test caught it: the mechanism tests drive the environment with hand-built actions, and
  the P2 smoke used a heuristic agent that posts offers in every phase.
- **Proposed fix** (spec revision R15, not yet applied):
  - The trade stage uses the `trade_offer` posted at the most recent REPORT step, stored in state.
  - At `t = 0` there is no prior REPORT, so no trade happens.
  - This is consistent with PLAN section 2.3 and leaves the adapter untouched.
- The fix is held back until the Phase-3 runs now in flight finish. They were launched from the
  current code, and a code change under them would alter their evaluation dynamics and orphan the
  configuration-hash run directories they reuse.
- **Every Phase-2 and Phase-3 number from these runs describes an economy without horizontal
  trade.** This is carried forward as limitation **L3**.

**D2 - the exploitability audit is inconclusive, not passed.**
- In C0 and R3_QW every ratio is negative: the best-responder returns (-5 to -240) are far below
  the population's (about 0). The 500k-step best responder, trained from scratch, never learned to
  match the population, let alone exploit it.
- With `|R_pop|` near 0 the ratio is also ill-conditioned (C0's median is -130).
- "Below threshold" is therefore not evidence of convergence.
- R7_NULL's maximum of 0.70 (NON-CONVERGED) is the one case where a best responder exceeded the
  population.
- The threshold (provisional 5%) is the owner's to finalise at G3. The audit design (best-responder
  budget and initialisation, and a ratio floor on `|R_pop|`) needs a revision before the threshold
  means anything.

**D3.** Row 7's `reconciliation_stat` uses R13.5's per-enterprise form, which has no power
(spec/P3_REVISION.md S4 amendment). It is uninformative. Row 7's pass rule never used it.

**Findings that stand as reported failures** (PLAN section 4.2; nothing is re-run to make them
appear):
- Row 7's falsification condition fails: reserves persist under `g = 0` with the absolute penalty,
  where truthful reporting is optimal. This is consistent with L1, the learner not reaching the
  optimum.
- Row 3's pipeline check fails: quality does not rise under the quality-weighted objective.
- Hygiene fails in two arms.

#### Owner decisions at G3

1. **Finalise the exploitability threshold.** In light of D2, the LEAD recommends first revising
   the audit design (best-responder warm start from the population policy, equal budget, and a
   floor on `|R_pop|`).
2. **D1.** The LEAD recommends applying R15 and running a labelled **G3b** study: C0 at 30 seeds
   (about 6 h), re-evaluating row 6 only. Rows 2, 5 and 7 stand as computed here, because their
   first evaluation is this one.
3. **Proceed.** The Phase-3 contrasts, already running under the current code, continue as a
   labelled study carrying L3. The LEAD recommends proceeding, because no contrast lever is a trade
   parameter.

#### OWNER DECISION (2026-09-27): the LEAD's recommendations, delegated

The owner instructed: "Do what's recommended on your judgment". Recorded as:

1. **Exploitability.** The audit is redesigned first (spec/P2_REVISION.md R16), and the 5%
   threshold is then applied to the revised audit. The G3 audit above stands as inconclusive.
2. **D1.** R15 is applied after the in-flight Phase-3 runs finish. The labelled study **G3b** is
   approved: C0 at 30 seeds under spec 2.1.0, re-evaluating row 6. The R16 audit is run on G3b's
   populations.
3. **Phase 3** continues as a labelled study carrying L3 (no horizontal trade).

G3 stays **NOT PASSED** as recorded. G3b is a separate, labelled record and never rewrites this
one.

---

## Phase 2 - acceptance run (G3)

_Source: `runs/phase2_acceptance/report.md`_

### Gate G3 - Phase-2 acceptance (WO-031)

Pre-registration: `spec/P2_REVISION.md` R14. Limitations carried from Phase 1: L1 (PPO does not recover the single-enterprise DP's mixed under-reporting strategy) and L2 (the bunching estimator on degenerate and peaked distributions); see `runs/G2_record.md`.

**G3: NOT PASSED**

#### Conditions

- heldout_evaluated: PASS (rows 2, 5, 6, 7 computed on the PLAN section 4.2 values; appearance is reported per row below)
- exploitability: FAIL
- oracle_gap_recorded: PASS
- jax_parity: PASS
- price_sensitivity: PASS (computed; a sign change is reported below, never suppressed)
- hygiene: FAIL

#### Held-out phenomena (C0, mean over seeds [95% seed-bootstrap CI])

- Row 2 storming: excess Gini 0.1374 [0.0945, 0.1844] - APPEARS
- Row 5 hoarding: request inflation excess 1.9631 [1.9445, 1.9785]; corr(X, shortfall) excess 0.5526 [0.3788, 0.7178] (1 seeds with an undefined correlation) - APPEARS
- Row 6 blat: trade volume share 0.0000 [0.0000, 0.0000] - FAILURE (does not appear)
- Row 7 hidden reserves: C0 0.2219 [0.0923, 0.3895]; R7_NULL 0.2190 [0.0552, 0.4392] (vanishes if upper < 0.01) - FAILURE (present: True, vanishes under null: False)
- Row 3 quality (pipeline check): mean qbar C0 - R3_QW 0.1523 [-0.0741, 0.3394] - FAILURE

#### Oracle (WO-027)

- C0: W_oracle 1.9836, val_oracle 38.1873, status optimal, solver HiGHS (HiGHS via OR-Tools 9.15.6755), optimality gap 0.00e+00, horizon 40
- R7_NULL: W_oracle 1.9836, val_oracle 38.1873, status optimal, solver HiGHS (HiGHS via OR-Tools 9.15.6755), optimality gap 0.00e+00, horizon 40
- R3_QW: W_oracle 1.9836, val_oracle 38.1873, status optimal, solver HiGHS (HiGHS via OR-Tools 9.15.6755), optimality gap 0.00e+00, horizon 40
- Clairvoyant welfare, C0 seeds 0-4 (UPPER BOUND ONLY, never a denominator): 1.9791, 1.9797, 1.9812, 1.9816, 1.9815

#### Headline metrics and price sensitivity (C0)

- welfare_ratio W / W_oracle: 0.0339
- specification_gap at base prices and under price seeds (11, 12, 13): 0.0105, 0.0105, 0.0106, 0.0105 - sign change: no

#### Exploitability (WO-028)

- C0: max -3.7899, median -129.6335 over 10 seeds (threshold 0.05, provisional; finalised by the lead at G3 (PLAN section 6.3)) 
- R7_NULL: max 0.7043, median -1.2312 over 5 seeds (threshold 0.05, provisional; finalised by the lead at G3 (PLAN section 6.3)) NON-CONVERGED
- R3_QW: max -48.5668, median -16136.8353 over 5 seeds (threshold 0.05, provisional; finalised by the lead at G3 (PLAN section 6.3)) 

#### JAX parity (WO-029)

- max |NumPy - JAX| over 100 agent-steps: p1 1.07e-14, p2 7.11e-14 (tolerance 1e-05)

#### Hygiene

- BOUND_BINDING runs: 0
- max training-episode runaway fraction after 20%: 0.1538 (limit 0.05)

#### Arms and seeds

- C0: 30 seeds, overrides none
- R7_NULL: 10 seeds, overrides {'incentive': {'growth_directive': 0.0, 'penalty_arg': 'absolute'}}
- R3_QW: 10 seeds, overrides {'incentive': {'objective_metric': 'quality_weighted'}, 'information': {'quality_measurability': 1.0}}

Sizing: `{'n_envs': 8, 'rollout_steps': 125, 'total_agent_steps': 1000000, 'eval_every_updates': 250, 'eval_episodes': 10, 'measure_episodes': 100}`; git `51132cda560cf2069df2a0395147d5cc7a61b580`.

---

## Labelled study G3b - record

_Source: `runs/G3b_record.md`_

### Labelled study G3b - record (LEAD)

**Status: G3b does not change G3.** G3 stays **NOT PASSED** as recorded in `runs/G3_record.md`.
G3b was approved at G3 (owner decision 2, delegated) to answer two questions. Row 6 now fails on
learned behaviour, and the revised audit finds the populations far from equilibrium. Written on
2026-09-28. The human sign-off is pending.

- Source: `runs/phase2_acceptance_g3b/report.md` and `result.json`.
- Spec 2.1.0: R15 (trade offers posted at REPORT) and R16 (the revised exploitability audit).
- Design: C0 only, 30 seeds, the same `seed_env = 1000 + s` and learner as G3. The R16 audit ran on
  seeds 0-9.
- The run started on 2026-09-28 at 07:51 UTC and ended at 21:08 UTC.
- Limitations L1 and L2 carry forward. L3 (no trade was possible) is lifted for G3b only.

#### Question 1 - row 6 (blat), after the D1 fix: **FAILURE (does not appear)**

| Statistic | Value |
|---|---|
| trade volume share, mean over 30 seeds [95% CI] | 0 [0, 0] |
| matched trade pairs, all 30 seeds, 100 measurement episodes each | 0 |

This is a behavioural result, not a pipeline defect. It was checked three ways:

1. **The fix works.** The pre-run smoke test (tiny untrained policies under spec 2.1.0) matched
   166 and 335 trade pairs. Offers are posted at REPORT, stored, and executed by the next period's
   trade stage.
2. **The learned policies post only buy orders.** A deterministic rollout of the trained C0
   populations for seeds 0-9 recorded every posted `trade_offer` at REPORT.
   - Every offer was negative, meaning a want to buy (`gosplan/env/trade.py`: a negative offer is
     a want, a positive one an offer to sell).
   - Offers ranged from -1.000 to -0.358, with per-seed means between -0.90 and -0.99.
   - Not one of the offers was a sell offer. With no seller, the matching rule (R9) can execute
     nothing.
3. **This agrees with row 5.** The same populations push input requests to the upper bound
   (request inflation excess 1.96, as at G3). Every enterprise wants more of every input and none
   parts with any.

Row 6 is therefore evaluated, and it fails under R14's rule: the lower CI bound is not above 0.
In the terms of PLAN section 4.2: under C0 the learned enterprises hoard rather than barter, so
horizontal trade does not emerge.

#### Question 2 - the R16 exploitability audit: **NON-CONVERGED**

| Seeds audited | Median ratio | Max ratio | Seeds above the 5% threshold |
|---|---|---|---|
| 10 | 0.797 | 0.990 | 9 of 10 |

- R16's design: the best responder starts from the population policy, trains for 1M steps, and the
  ratio is `(R_BR - R_pop) / max(|R_pop|, 1)`.
- Sorted per-seed ratios: -1.244, 0.125, 0.341, 0.364, 0.694, 0.899, 0.929, 0.972, 0.983, 0.990.
- The one negative seed (1007) is one where the population's own return is about -30 and the best
  responder fell further, to about -68.
- **Interpretation.** Unlike the pre-R16 audit (G3 D2), this audit is informative. Starting from
  the population's own policy, a single enterprise can improve its return substantially on 9 of
  10 seeds. The learned C0 populations are **not** approximate equilibria at the 5% threshold.
- Every Phase-2 and Phase-3 number from these populations therefore describes the learner's
  *outcome*, not an equilibrium of the institution. This strengthens L1 and is carried forward as
  **L4**.

#### Supplementary (not re-evaluations; the G3 values stand)

- **Row 2:** excess storming Gini 0.118 [0.086, 0.150] (G3: 0.137).
- **Row 5:** request inflation excess 1.96, and corr(X, shortfall) excess 0.52 (G3: 1.96 and 0.55).
- **Row 7:** C0 hidden reserves 0.173 [0.068, 0.316] (G3: 0.222).
- **Welfare:** welfare_ratio is 0.0335 (G3: 0.034). specification_gap is 0.0124, with no sign
  change across price seeds 11, 12 and 13.
- **Not evaluated in G3b:** row 7's null arm and row 3 need the R7_NULL and R3_QW arms, which
  G3b does not run. The report's generic template prints "FAILURE" beside `nan` for these rows;
  read that as "not evaluated here".
- **Checks:** JAX parity passes (7.1e-14). Hygiene passes (runaway 0.0000, `BOUND_BINDING` 0). The
  oracle gap is 0.

#### Consequences

- G3 remains NOT PASSED. G3b adds two findings:
  - Row 6 fails behaviourally, because learners hoard and never offer to sell.
  - Under the revised audit the C0 populations are non-converged (L4).
- The Phase-3 contrasts (`runs/P3_record.md`) were run before R15, carrying L3. G3b shows that
  even with trade possible, C0 populations do not trade. L3 is therefore unlikely to have changed
  the contrasts' C0 leg. This is an inference, not a re-run: the contrast arms were not re-trained
  under spec 2.1.0.
- The owner decides at G3/G4 whether a longer-trained or differently configured learner is
  commissioned. That would be a new labelled study, because the numbers here must not be tuned to
  pass.

---

## Labelled study G3b - acceptance re-run (R15, R16)

_Source: `runs/phase2_acceptance_g3b/report.md`_

> **Labelled study G3b** (spec/P2_REVISION.md R15, R16; runs/G3_record.md). Spec 2.1.0 (trade
> offers posted at REPORT), C0 only, 30 seeds; its purpose is row 6 and the R16 exploitability
> audit. Rows 2, 5 and 7 stand as first evaluated in runs/phase2_acceptance/report.md; their
> values below are supplementary, and the rows needing the R7_NULL / R3_QW arms are not run.

### Gate G3 - Phase-2 acceptance (WO-031)

Pre-registration: `spec/P2_REVISION.md` R14. Limitations carried from Phase 1: L1 (PPO does not recover the single-enterprise DP's mixed under-reporting strategy) and L2 (the bunching estimator on degenerate and peaked distributions); see `runs/G2_record.md`.

**G3: NOT PASSED**

#### Conditions

- heldout_evaluated: PASS (rows 2, 5, 6, 7 computed on the PLAN section 4.2 values; appearance is reported per row below)
- exploitability: FAIL
- oracle_gap_recorded: PASS
- jax_parity: PASS
- price_sensitivity: PASS (computed; a sign change is reported below, never suppressed)
- hygiene: PASS

#### Held-out phenomena (C0, mean over seeds [95% seed-bootstrap CI])

- Row 2 storming: excess Gini 0.1180 [0.0863, 0.1501] - APPEARS
- Row 5 hoarding: request inflation excess 1.9608 [1.9402, 1.9779]; corr(X, shortfall) excess 0.5153 [0.3500, 0.6780] (0 seeds with an undefined correlation) - APPEARS
- Row 6 blat: trade volume share 0.000e+00 [0.000e+00, 0.000e+00] - FAILURE (does not appear)
- Row 7 hidden reserves: C0 0.1734 [0.0677, 0.3161]; R7_NULL nan [nan, nan] (vanishes if upper < 0.01) - FAILURE (present: True, vanishes under null: False)
- Row 3 quality (pipeline check): mean qbar C0 - R3_QW nan [nan, nan] - FAILURE

#### Oracle (WO-027)

- C0: W_oracle 1.9836, val_oracle 38.1873, status optimal, solver HiGHS (HiGHS via OR-Tools 9.15.6755), optimality gap 0.00e+00, horizon 40
- R7_NULL: W_oracle 1.9836, val_oracle 38.1873, status optimal, solver HiGHS (HiGHS via OR-Tools 9.15.6755), optimality gap 0.00e+00, horizon 40
- R3_QW: W_oracle 1.9836, val_oracle 38.1873, status optimal, solver HiGHS (HiGHS via OR-Tools 9.15.6755), optimality gap 0.00e+00, horizon 40
- Clairvoyant welfare, C0 seeds 0-4 (UPPER BOUND ONLY, never a denominator): 1.9791, 1.9797, 1.9812, 1.9816, 1.9815

#### Headline metrics and price sensitivity (C0)

- welfare_ratio W / W_oracle: 0.0335
- specification_gap at base prices and under price seeds (11, 12, 13): 0.0124, 0.0122, 0.0125, 0.0125 - sign change: no

#### Exploitability (WO-028)

- C0: max 0.9905, median 0.7966 over 10 seeds (threshold 0.05, provisional; finalised by the lead at G3 (PLAN section 6.3)) NON-CONVERGED

#### JAX parity (WO-029)

- max |NumPy - JAX| over 100 agent-steps: p1 1.07e-14, p2 7.11e-14 (tolerance 1e-05)

#### Hygiene

- BOUND_BINDING runs: 0
- max training-episode runaway fraction after 20%: 0.0000 (limit 0.05)

#### Arms and seeds

- C0: 30 seeds, overrides none
- R7_NULL: 0 seeds, overrides {'incentive': {'growth_directive': 0.0, 'penalty_arg': 'absolute'}}
- R3_QW: 0 seeds, overrides {'incentive': {'objective_metric': 'quality_weighted'}, 'information': {'quality_measurability': 1.0}}

Sizing: `{'n_envs': 8, 'rollout_steps': 125, 'total_agent_steps': 1000000, 'eval_every_updates': 250, 'eval_episodes': 10, 'measure_episodes': 100}`; git `347721bc2018a4126fe9b255cce51a1e060eafb2-dirty`.

---

## Labelled study LC - record

_Source: `runs/LC_record.md`_

### Labelled study LC - record (LEAD)

**Status: complete. LC is a labelled study and re-evaluates no gate.** G3 (NOT PASSED) and G3b stand
as recorded. Written on 2026-09-29; the human sign-off is pending.

- Pre-registration: `spec/P2_REVISION.md` R17, committed in cfba026 before any LC run.
- Driver: `gosplan/experiments/learner_convergence.py`. Report: `runs/learner_convergence/report.md`.
- Design: C0, seeds 0-9, population budget 3M agent-steps against G3b's 1M. Everything else is
  unchanged, including the R16 audit.
- The run started at 09:15 UTC. A container reboot at about 15:25 UTC killed it with 8 of 10 runs
  complete.
  - It was relaunched at 15:30 UTC. The 8 completed runs were read back.
  - The 2 interrupted runs (seeds with hashes `3f0dee0b`, `cc06e97e`) were retrained from scratch
    with identical configuration.
  - It finished at 20:04 UTC.

#### Pre-registered outcome (R17)

| Rule | Result |
|---|---|
| Exploitability falls with budget (CI upper bound of the median paired difference < 0) | **YES**: median difference -0.857 [95% CI -1.793, -0.130] |
| Converged at 3M (max over seeds <= 5%) | **NO**: max 0.429; 3 of 10 seeds above the threshold |

Median exploitability went from 0.797 (1M) to -0.003 (3M). Per-seed values are in the report.

#### What the 3M populations are doing: the economy has collapsed to near-zero output

Read the primary result against this before citing it.

| C0, seeds 0-9 | Mean effort | Mean return | Fictitious padding | Welfare (mean W) | welfare_ratio |
|---|---|---|---|---|---|
| 1M (G3b) | 0.195 | -16.0 | 0.040 | 0.087 | 0.0335 |
| 3M (LC) | 0.019 | +0.27 | 0.000 | 0.000 on every seed | 0.0000 |

- With three times the training, the learned enterprises stop producing: mean effort 0.005-0.036
  on every seed. They also stop padding.
- Their returns rise from about -16 to about 0. The large losses the 1M populations carried are
  gone.
- Welfare is exactly 0 on all 10 seeds. The oracle and the truthful-myopic baseline, on the same
  configuration, reach W of 1.98 and 1.06, so the environment supports production. The collapse is
  in what the learners converge to.
- Every posted trade offer is still a buy order. The mean offer is -1.000, and none of the 3M
  populations' offers is a sell. Row 6 stays at zero.

**Reading of the primary result.** The fall in exploitability is real by R17's rule, but it is not
convergence to a functioning equilibrium.
- The 3M populations sit where returns are near 0 and, with the ratio floor of 1 reward unit, a
  best responder has little to gain.
- On two seeds (1003 and 1005) the warm-started best responder ended far *below* the population it
  started from (ratios -2.7 and -2.2). R16's audit therefore also measures the best-responder
  learner's own instability, which bounds how much the audit can certify.
- A plausible mechanism is a no-production coordination trap. Production needs other
  enterprises' outputs as inputs (near-Leontief, `theta = 8`), so when every supplier produces
  nothing, effort is wasted. **LC does not test this. It is an interpretation, not a finding.**

#### Update (2026-09-30): the coordination-trap reading was tested

See `runs/CT_record.md`. R18 was pre-registered and run as an evaluation-only test. Against the
truthful-myopic producer, the 3M collapse **is** a coordination trap: producing alone does not pay,
and everyone producing pays more. The "interpretation, not a finding" caveat above therefore no
longer applies to that reading. What training does to reach the trap is still untested.

#### Addendum (2026-10-01): training trajectories and a design caveat (descriptive, post hoc)

Not pre-registered and no test is attached. Source: `runs/learner_convergence/trajectories.md`,
from the runs' `train_log.jsonl` (`python -m gosplan.experiments.learner_convergence --trajectories`).

| agent-steps (k) | 250 | 500 | 750 | 1000 | 1250 | 1500 | 2000 | 3000 |
|---|---|---|---|---|---|---|---|---|
| median effort, 1M runs | 0.66 | 0.62 | 0.27 | 0.09 | | | | |
| median effort, 3M runs | 0.66 | 0.60 | 0.35 | 0.12 | 0.03 | 0.02 | 0.02 | 0.02 |
| median eval return, 3M runs | -191 | -43 | -20 | -6 | -1.3 | -0.01 | 0.49 | 0.71 |

- **The collapse is a steady slide, not a late event.**
  - Effort falls from about 0.66 to the trap level of about 0.02 by roughly 1.25-1.5M steps, then
    stays there.
  - Each enterprise's own evaluation return rises along the whole path. The learners improve
    their own payoff while welfare falls.
- **The 1M populations were mid-slide.** Their median effort at 1M steps was already 0.09. G3, G3b
  and the Phase-3 contrasts therefore describe an economy partway into the trap, which sharpens L5.
- **Design caveat.**
  - The entropy bonus anneals from 0.01 to 0.001 over the whole run, so LC's budget manipulation
    also stretches the exploration schedule. At the 1M-step mark a 3M run's coefficient is 0.0070,
    against 0.0010 for a 1M run.
  - The learning rate is constant, so the entropy schedule is the only part of the setup that
    scales with the budget.
  - Through the first 1M steps the two budgets' trajectories nearly coincide despite the different
    schedules. That suggests the schedule is not what drives the collapse, but LC cannot separate
    budget from schedule. R17's "everything else unchanged" holds for the configuration, not for
    the effective schedule.

#### Update (2026-10-03): the design caveat was tested

See `runs/ES_record.md`. R19 was pre-registered and run: 3M-step populations with the 1M runs'
exact entropy schedule. They collapse just the same, with median effort 0.019 [95% CI 0.012,
0.023], and land in the same coordination trap. The collapse is a training-length effect for this
learner, not an artefact of the stretched schedule.

#### Consequences

- **L4 stands.** The populations are non-converged at both budgets.
- **New limitation L5.** Learned-economy outcomes depend on the training budget. At 1M the C0
  economy is heavily degraded (welfare_ratio 0.034), and at 3M it has collapsed (0.000). Every
  learned-agent number in this repository comes from 1M-step populations: G3, G3b and the Phase-3
  contrasts. Each describes a point on a learning trajectory, not a limit, and none may be read as
  the institution's steady state.
- The Phase-3 contrasts are not re-run. Whether their deltas hold at 3M is unknown.
- Any follow-up is the owner's decision and needs a new pre-registration (R17: "no further budget,
  learner or configuration change follows from it without a new pre-registration"). Candidates:
  - a learner with a different exploration or population scheme;
  - contrasts at a fixed longer budget;
  - a direct test of the coordination-trap reading, for example a population seeded with producers.

---

## Labelled study LC - learner convergence (R17)

_Source: `runs/learner_convergence/report.md`_

### Labelled study LC - learner convergence (spec/P2_REVISION.md R17)

C0, seeds 0-9, population budget 3M agent-steps against G3b's 1M; R16 audit unchanged (warm-started best responder, 1M steps, ratio floor 1). Pre-registered before any LC run. LC re-evaluates no gate; G3 and G3b stand as recorded.

#### Primary: exploitability, seed-paired (3M - 1M)

| seed_env | 1M (G3b) | 3M (LC) | difference |
|---|---|---|---|
| 1000 | 0.125 | -0.005 | -0.130 |
| 1001 | 0.990 | -0.027 | -1.017 |
| 1002 | 0.341 | 0.429 | 0.088 |
| 1003 | 0.899 | -2.713 | -3.612 |
| 1004 | 0.972 | -0.001 | -0.973 |
| 1005 | 0.364 | -2.249 | -2.613 |
| 1006 | 0.929 | 0.074 | -0.855 |
| 1007 | -1.244 | 0.000 | 1.244 |
| 1008 | 0.983 | 0.125 | -0.859 |
| 1009 | 0.694 | -0.100 | -0.794 |

- Median exploitability: 1M 0.797, 3M -0.003.
- Median paired difference -0.857 [95% CI -1.793, -0.130].
- **Exploitability falls with budget (R17 rule: CI upper bound < 0): YES.**
- **Converged at 3M (R17 rule: max <= 0.05): NO** - max 0.429, 3 of 10 seeds above the threshold.

#### Secondary (descriptive)

- welfare_ratio: 1M 0.0335, 3M 0.0000.
- Row-6 trade volume share (mean over seeds): 1M 0.000e+00, 3M 0.000e+00.
- Posted offers at REPORT, 1M (seeds 0-9, 5 episodes each): sell share 0.000, buy share 1.000, mean offer -0.953.
- Posted offers at REPORT, 3M (seeds 0-9, 5 episodes each): sell share 0.000, buy share 1.000, mean offer -1.000.

---

#### Acceptance-harness output for the 3M populations

_The harness prints its gate-condition lines and a G3 verdict line. For LC they are descriptive only: LC is a labelled study and re-evaluates no gate._

#### Gate G3 - Phase-2 acceptance (WO-031)

Pre-registration: `spec/P2_REVISION.md` R14. Limitations carried from Phase 1: L1 (PPO does not recover the single-enterprise DP's mixed under-reporting strategy) and L2 (the bunching estimator on degenerate and peaked distributions); see `runs/G2_record.md`.

**G3: NOT PASSED**

##### Conditions

- heldout_evaluated: PASS (rows 2, 5, 6, 7 computed on the PLAN section 4.2 values; appearance is reported per row below)
- exploitability: FAIL
- oracle_gap_recorded: PASS
- jax_parity: PASS
- price_sensitivity: PASS (computed; a sign change is reported below, never suppressed)
- hygiene: PASS

##### Held-out phenomena (C0, mean over seeds [95% seed-bootstrap CI])

- Row 2 storming: excess Gini 0.2197 [0.1738, 0.2800] - APPEARS
- Row 5 hoarding: request inflation excess 1.9819 [1.9518, 1.9981]; corr(X, shortfall) excess 1.0736 [1.0298, 1.1078] (0 seeds with an undefined correlation) - APPEARS
- Row 6 blat: trade volume share 0.000e+00 [0.000e+00, 0.000e+00] - FAILURE (does not appear)
- Row 7 hidden reserves: C0 0.0975 [0.0720, 0.1217]; R7_NULL nan [nan, nan] (vanishes if upper < 0.01) - NOT EVALUATED (the R7_NULL arm was not run)
- Row 3 quality (pipeline check): mean qbar C0 - R3_QW nan [nan, nan] - NOT EVALUATED (the R3_QW arm was not run)

##### Oracle (WO-027)

- C0: W_oracle 1.9836, val_oracle 38.1873, status optimal, solver HiGHS (HiGHS via OR-Tools 9.15.6755), optimality gap 0.00e+00, horizon 40
- R7_NULL: W_oracle 1.9836, val_oracle 38.1873, status optimal, solver HiGHS (HiGHS via OR-Tools 9.15.6755), optimality gap 0.00e+00, horizon 40
- R3_QW: W_oracle 1.9836, val_oracle 38.1873, status optimal, solver HiGHS (HiGHS via OR-Tools 9.15.6755), optimality gap 0.00e+00, horizon 40
- Clairvoyant welfare, C0 seeds 0-4 (UPPER BOUND ONLY, never a denominator): 1.9791, 1.9797, 1.9812, 1.9816, 1.9815

##### Headline metrics and price sensitivity (C0)

- welfare_ratio W / W_oracle: 0.0000
- specification_gap at base prices and under price seeds (11, 12, 13): 0.0055, 0.0055, 0.0055, 0.0055 - sign change: no

##### Exploitability (WO-028)

- C0: max 0.4286, median -0.0028 over 10 seeds (threshold 0.05, provisional; finalised by the lead at G3 (PLAN section 6.3)) NON-CONVERGED

##### JAX parity (WO-029)

- max |NumPy - JAX| over 100 agent-steps: p1 1.07e-14, p2 7.11e-14 (tolerance 1e-05)

##### Hygiene

- BOUND_BINDING runs: 0
- max training-episode runaway fraction after 20%: 0.0000 (limit 0.05)

##### Arms and seeds

- C0: 10 seeds, overrides none
- R7_NULL: 0 seeds, overrides {'incentive': {'growth_directive': 0.0, 'penalty_arg': 'absolute'}}
- R3_QW: 0 seeds, overrides {'incentive': {'objective_metric': 'quality_weighted'}, 'information': {'quality_measurability': 1.0}}

Sizing: `{'n_envs': 8, 'rollout_steps': 125, 'total_agent_steps': 3000000, 'eval_every_updates': 250, 'eval_episodes': 10, 'measure_episodes': 100}`; git `4c53d493b09f32bcc1ef7636a0f3a8edfcc2d5bc`.

---

## Labelled study LC - training trajectories (descriptive)

_Source: `runs/learner_convergence/trajectories.md`_

### LC - training trajectories (descriptive, post hoc)

Median over seeds 0-9 of the periodic evaluation (deterministic policy) during training; C0, the G3b 1M runs and the LC 3M runs. Not pre-registered; no test is attached.

#### 1M

| agent-steps (k) | entropy coef | median effort | median eval return |
|---|---|---|---|
| 250 | 0.0078 | 0.662 | -206.67 |
| 500 | 0.0055 | 0.615 | -68.71 |
| 750 | 0.0033 | 0.267 | -56.57 |
| 1000 | 0.0010 | 0.089 | -4.59 |

#### 3M

| agent-steps (k) | entropy coef | median effort | median eval return |
|---|---|---|---|
| 250 | 0.0093 | 0.663 | -190.82 |
| 500 | 0.0085 | 0.603 | -42.93 |
| 750 | 0.0078 | 0.348 | -20.40 |
| 1000 | 0.0070 | 0.116 | -5.93 |
| 1250 | 0.0063 | 0.033 | -1.31 |
| 1500 | 0.0055 | 0.023 | -0.01 |
| 1750 | 0.0048 | 0.022 | -0.01 |
| 2000 | 0.0040 | 0.022 | 0.49 |
| 2250 | 0.0033 | 0.024 | 0.69 |
| 2500 | 0.0025 | 0.025 | 0.69 |
| 2750 | 0.0018 | 0.022 | 0.85 |
| 3000 | 0.0010 | 0.020 | 0.71 |

---

## Labelled study CT - record

_Source: `runs/CT_record.md`_

### Labelled study CT - record (LEAD)

**Status: complete. CT is a labelled study, is evaluation only (no training) and re-evaluates no
gate.** Written on 2026-09-30; the human sign-off is pending.

- Pre-registration: `spec/P2_REVISION.md` R18, committed in 18864d0 before the CT evaluation. It
  discloses a 2-episode smoke run on one seed made to check the code.
- Driver: `gosplan/experiments/coordination_trap.py`. Report and data:
  `runs/coordination_trap/report.md`, `result.json`.
- Design:
  - Populations: LC's 3M C0 populations (primary) and G3b's 1M populations (contrast), seeds 0-9.
  - 50 episodes per condition, on the measurement seed block, with deterministic policies and
    common random numbers.
  - Producer: `TruthfulMyopic`, the project's reference line. It sets effort to meet the target,
    reports stock truthfully, requests at need and never trades.

#### Pre-registered outcome (R18)

| Populations | (i) Producing alone does not pay (`d1 = R_dev - R_pop` < 0) | (ii) Everyone producing pays more (`d2 = W_tm - W_pop` > 0) | Coordination trap |
|---|---|---|---|
| **3M (primary)** | **yes**: median -1.718 [95% CI -2.240, -1.438] | **yes**: median +0.869 [0.494, 1.303] | **YES** |
| 1M (contrast) | not shown: median +0.336 [-1.208, 34.004] | yes: median +3.632 [1.539, 40.570] | NO |

**Reading.** The collapse LC found at 3M is a coordination trap in the textbook sense, against the
truthful-myopic producer:

- Every seed's lone producer loses. Seat 0's return when producing alone is between -1.47 and
  -1.43, against -0.74 to +1.05 when it plays the learned policy.
- When every seat produces, the mean seat return is 1.26-1.32. Every seat is better off on 9 of
  10 seeds, and 95% of seats are on the tenth.

The 1M populations are not a trap. Several are so far from equilibrium (seat-0 returns down to
-39.6) that producing alone would help. This is consistent with L4.

#### Descriptive observations (not tested)

- **The lone producer's return barely depends on the budget.**
  - The producer in seat 0 earns almost the same return against the 1M and the 3M populations of
    each seed, for example -1.425 against both for seed 1000.
  - Against all-producers, the same seat is better off.
  - Both learned populations therefore leave a producer similarly short. This fits the input
    starvation that the hoarding result (row 5) and the buy-only trade offers (row 6, G3b) point
    to. CT did not measure the producer's input receipts, so this is a reading.
- **The learned policy responds only weakly to available inputs.** Seat 0's learned effort rises
  when the other seats are producers: at 3M the median rises from about 0.02 to about 0.14, with
  seed 1007 flat at 0.006. That is still far below what a producer supplies, so the trained policy
  has partly unlearned production. It is not only waiting for inputs.

#### Limits

- One producer policy was used. A trap against `TruthfulMyopic` does not show that no producing
  strategy pays alone.
- CT reuses the 3M populations as trained. Why training reaches the trap is not tested.
- No gate is re-evaluated. G3 (NOT PASSED), G3b and LC stand as recorded, and so do limitations
  L1-L5.

#### What this adds to the project's claims

- **This is the first tested mechanism result about the learned C0 economy.** With long enough
  training, learning enterprises under the C0 planner settle into a Pareto-dominated
  no-production state that no single enterprise can profitably leave.
- **It is not Claim A.** It is none of the four held-out phenomena.
- **It rests on the stated learner (L1, L4, L5).** Whether other learners reach the same trap is
  open. That question, and any follow-up, is the owner's decision and needs a new
  pre-registration.

---

## Labelled study CT - coordination-trap test (R18)

_Source: `runs/coordination_trap/report.md`_

### Labelled study CT - coordination-trap test (spec/P2_REVISION.md R18)

Evaluation only (no training): LC's 3M populations (primary) and G3b's 1M populations (contrast), seeds 0-9, 50 episodes per condition on the measurement seed block, deterministic policies, seat 0. Producer = `TruthfulMyopic`. Pre-registered before any CT evaluation. CT re-evaluates no gate.

#### 3M populations (primary)

| seed_env | R_pop (seat 0) | R_dev (seat 0 produces) | W_pop (mean seat) | W_tm (all produce) | seats better off, all produce | seat-0 effort amid learned | amid producers |
|---|---|---|---|---|---|---|---|
| 1000 | -0.003 | -1.425 | -0.003 | 1.276 | 1.00 | 0.019 | 0.082 |
| 1001 | 1.049 | -1.448 | 0.924 | 1.315 | 0.95 | 0.023 | 0.186 |
| 1002 | -0.736 | -1.446 | -0.628 | 1.306 | 1.00 | 0.039 | 0.142 |
| 1003 | 0.610 | -1.448 | 0.711 | 1.321 | 1.00 | 0.019 | 0.159 |
| 1004 | -0.002 | -1.450 | -0.002 | 1.324 | 1.00 | 0.012 | 0.047 |
| 1005 | 0.758 | -1.448 | 0.810 | 1.304 | 1.00 | 0.018 | 0.209 |
| 1006 | 0.274 | -1.440 | 0.381 | 1.260 | 1.00 | 0.025 | 0.147 |
| 1007 | -0.000 | -1.453 | -0.000 | 1.278 | 1.00 | 0.007 | 0.006 |
| 1008 | 0.262 | -1.459 | 0.430 | 1.290 | 1.00 | 0.017 | 0.137 |
| 1009 | 0.951 | -1.470 | 0.823 | 1.311 | 1.00 | 0.019 | 0.102 |

- (i) d1 = R_dev - R_pop, median -1.718 [-2.240, -1.438]: producing alone does NOT pay (R18: CI entirely below 0).
- (ii) d2 = W_tm - W_pop, median 0.869 [0.494, 1.303]: all-production pays more (R18: CI entirely above 0).
- **Coordination trap (both): YES.**

#### 1M populations (contrast)

| seed_env | R_pop (seat 0) | R_dev (seat 0 produces) | W_pop (mean seat) | W_tm (all produce) | seats better off, all produce | seat-0 effort amid learned | amid producers |
|---|---|---|---|---|---|---|---|
| 1000 | -0.036 | -1.425 | -0.044 | 1.276 | 1.00 | 0.044 | 0.136 |
| 1001 | -39.590 | -1.448 | -39.255 | 1.315 | 1.00 | 0.201 | 0.515 |
| 1002 | -0.238 | -1.446 | -0.233 | 1.306 | 1.00 | 0.042 | 0.194 |
| 1003 | -1.245 | -1.448 | -1.375 | 1.321 | 1.00 | 0.361 | 0.400 |
| 1004 | -2.330 | -1.454 | -2.838 | 1.324 | 1.00 | 0.488 | 0.513 |
| 1005 | -0.009 | -1.448 | -0.009 | 1.304 | 1.00 | 0.027 | 0.156 |
| 1006 | -35.444 | -1.440 | -44.719 | 1.260 | 1.00 | 0.100 | 0.213 |
| 1007 | -35.641 | -1.453 | -50.873 | 1.278 | 1.00 | 0.074 | 0.159 |
| 1008 | -12.044 | -1.459 | -10.014 | 1.290 | 1.00 | 0.468 | 0.441 |
| 1009 | -0.545 | -1.470 | -1.790 | 1.311 | 1.00 | 0.060 | 0.170 |

- (i) d1 = R_dev - R_pop, median 0.336 [-1.208, 34.004]: producing alone is not shown to be unprofitable (R18: CI entirely below 0).
- (ii) d2 = W_tm - W_pop, median 3.632 [1.539, 40.570]: all-production pays more (R18: CI entirely above 0).
- **Coordination trap (both): NO.**

---

## Labelled study ES - record

_Source: `runs/ES_record.md`_

### Labelled study ES - record (LEAD)

**Status: complete. ES is a labelled study and re-evaluates no gate.** G3 (NOT PASSED), G3b, LC and
CT stand as recorded. Written on 2026-10-03; the human sign-off is pending.

- Pre-registration: `spec/P2_REVISION.md` R19, committed in 3db46a9 before any ES run.
- Driver: `gosplan/experiments/entropy_schedule.py`. Report and data:
  `runs/entropy_schedule/report.md`, `comparison.json`, `result.json`.
- Design: C0, seeds 0-9, 3M agent-steps, as in LC. The difference from LC is the entropy bonus,
  which anneals 0.01 -> 0.001 over the first 1,000 updates (1M agent-steps, the 1M runs' exact
  schedule) and is then held at 0.001. Everything else is LC's, including the R16 audit.
- Run history:
  - Launched on 2026-10-03 at 09:58 UTC; finished at 20:58 UTC.
  - Two container reboots killed the first training batch at about 1,450 and 360 of 3,000
    updates. Those runs were retrained from scratch with identical configuration.
  - Exact resume snapshots were then added to the training harness (R19 note 1). Later restarts
    resumed from the last snapshot. `tests/unit/test_train_resume.py` shows a resumed run is
    bitwise equal to an uninterrupted one.
  - A restart interrupted the R16 audit with 5 of 10 best responders done. The completed ones
    were reused (R19 note 2, `tests/unit/test_best_responder_reuse.py`).
  - Neither change touches the design, the rules or any result.

#### Pre-registered outcome (R19)

| Rule | Result |
|---|---|
| Collapse reproduced under the matched schedule (CI upper bound of the median final mean effort < 0.05) | **YES**: median 0.019 [95% CI 0.012, 0.023] |

Per-seed final mean effort is between 0.010 and 0.025, which is the range LC found (0.005-0.036).

**Reading (R19's own reading of the rule).** More training, not the stretched exploration
schedule, drives the collapse. LC's design caveat (`runs/LC_record.md`, addendum) is resolved for
this learner: with the 1M runs' exact schedule, a 3M-step run still collapses.

#### Descriptive results (not tested)

**Trajectories.** Median over seeds of the periodic evaluation:

| agent-steps (k) | 250 | 500 | 750 | 1000 | 1250 | 1500 | 1750 | 2000 | 3000 |
|---|---|---|---|---|---|---|---|---|---|
| entropy coef, ES | 0.0078 | 0.0055 | 0.0033 | 0.0010 | 0.0010 | 0.0010 | 0.0010 | 0.0010 | 0.0010 |
| median effort, 1M runs (G3b) | 0.662 | 0.615 | 0.267 | 0.089 | | | | | |
| median effort, ES | 0.662 | 0.615 | 0.267 | 0.089 | 0.043 | 0.028 | 0.021 | 0.019 | 0.022 |
| median effort, LC (stretched) | 0.663 | 0.603 | 0.348 | 0.116 | 0.033 | 0.023 | 0.022 | 0.022 | 0.020 |
| median eval return, ES | -207 | -69 | -57 | -4.6 | -0.80 | -0.75 | -0.01 | 0.00 | 0.35 |

- **Through 1M steps ES reproduces the 1M runs exactly.** Same seeds, same configuration and the
  same schedule make it the same computation, so this is a consistency check rather than a
  finding.
- **After 1M steps the slide continues with the entropy coefficient fixed.** Effort halves again
  by 1.25M and reaches the trap level of about 0.02 by 1.75M, where it stays. Each enterprise's
  own evaluation return keeps rising along the way, as in LC.
- **Both 3M schedules end in the same place.** LC's longer exploration delays nothing visible
  after 1.25M.

**Outcomes at 3M.**
- Welfare: `welfare_ratio` 0.0000, as in LC.
- Held-out rows in the harness output (descriptive only):
  - row 2 (storming) and row 5 (hoarding) appear: excess Gini 0.222 (LC 0.220), request-inflation
    excess 1.79 (LC 1.98);
  - row 6 (trade volume) stays at 0.
- R16 exploitability:
  - Median -0.008 and max 0.173, with 1 of 10 seeds above 5%. LC 3M had median -0.003 and 3 of
    10 above.
  - The paired difference from LC is median +0.003 [95% CI -0.296, +0.950], so no difference is
    shown.
  - By R17's convergence rule (max <= 5%) the ES populations are **not converged** either.

**R18 coordination-trap test on the ES populations (R18's rules unchanged; trap: YES).**

| Test | Result |
|---|---|
| (i) Producing alone does not pay | `d1` median -1.785 [-2.056, -1.449] |
| (ii) Everyone producing pays more | `d2` median +1.111 [+0.678, +1.295] |

- A lone truthful-myopic producer in seat 0 earns between -1.47 and -1.43 on every seed. The same
  narrow band appeared against the LC and G3b populations (CT).
- When everyone produces, **every seat is better off on all 10 seeds**.
- Seat 0's learned effort rises from about 0.02 to between 0.03 and 0.45 (median about 0.15)
  when the other seats are producers. This matches CT's finding that the learned policy has
  partly unlearned production.

#### Limits

- One alternative schedule was tested. ES separates budget from *this* schedule; it does not
  exhaust the learner's exploration design.
- The learning rate is constant in every study, so no learning-rate schedule was tested.
- The study learner is the only learner (L1, L4). Whether other learners reach the same trap is
  open.
- The trap test uses one producer policy, `TruthfulMyopic`, as in CT.
- No gate is re-evaluated. Limitations L1-L5 stand.

#### Consequences

- **L5 is sharpened, not changed.** The budget dependence of learned-economy outcomes is a
  training-length effect for this learner. It is not an artefact of the annealing schedule.
- The project's mechanism result now holds under both 3M schedules. Given enough training, C0
  learners settle into a Pareto-dominated no-production coordination trap.
- Any follow-up is the owner's decision and needs a new pre-registration (#97).

---

## Labelled study ES - budget versus exploration schedule (R19)

_Source: `runs/entropy_schedule/report.md`_

### Labelled study ES - budget versus exploration schedule (spec/P2_REVISION.md R19)

C0, seeds 0-9, 3M agent-steps with the entropy bonus annealed over the first 1M steps (the 1M runs' schedule) and held at 0.001 after; R16 audit unchanged. Pre-registered before any ES run. ES re-evaluates no gate.

#### Primary: does the collapse reproduce under the matched schedule?

| seed | final mean effort |
|---|---|
| 0 | 0.012 |
| 1 | 0.014 |
| 2 | 0.022 |
| 3 | 0.022 |
| 4 | 0.011 |
| 5 | 0.019 |
| 6 | 0.025 |
| 7 | 0.024 |
| 8 | 0.019 |
| 9 | 0.010 |

- Median final effort 0.019 [95% CI 0.012, 0.023].
- **Collapse reproduced (R19 rule: CI upper bound < 0.05): YES.**

#### Secondary (descriptive)

- welfare_ratio: ES 0.0000; LC 3M 0.0000.
- R16 exploitability, ES: median -0.008, max 0.173 (1 of 10 above 5%); LC 3M median -0.003.
- R18 coordination-trap test on the ES populations: (i) median d1 -1.785 [-2.056, -1.449]; (ii) median d2 1.111 [0.678, 1.295]; trap: YES.

##### Training trajectories (median over seeds of the periodic evaluation)

**1M (G3b)** - k-steps [250, 500, 750, 1000]

- entropy coef: [0.0078, 0.0055, 0.0033, 0.001]
- median effort: [0.662, 0.615, 0.267, 0.089]
- median eval return: [-206.67, -68.71, -56.57, -4.59]

**3M, stretched schedule (LC)** - k-steps [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000]

- entropy coef: [0.0093, 0.0085, 0.0078, 0.007, 0.0063, 0.0055, 0.0048, 0.004, 0.0033, 0.0025, 0.0018, 0.001]
- median effort: [0.663, 0.603, 0.348, 0.116, 0.033, 0.023, 0.022, 0.022, 0.024, 0.025, 0.022, 0.02]
- median eval return: [-190.82, -42.93, -20.4, -5.93, -1.31, -0.01, -0.01, 0.49, 0.69, 0.69, 0.85, 0.71]

**3M, matched schedule (ES)** - k-steps [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000]

- entropy coef: [0.0078, 0.0055, 0.0033, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
- median effort: [0.662, 0.615, 0.267, 0.089, 0.043, 0.028, 0.021, 0.019, 0.022, 0.023, 0.023, 0.022]
- median eval return: [-206.67, -68.71, -56.57, -4.59, -0.8, -0.75, -0.01, -0.0, 0.03, 0.29, 0.36, 0.35]

---

#### Acceptance-harness output for the ES populations

_The harness prints its gate-condition lines and a G3 verdict line. For ES they are descriptive only._

#### Gate G3 - Phase-2 acceptance (WO-031)

Pre-registration: `spec/P2_REVISION.md` R14. Limitations carried from Phase 1: L1 (PPO does not recover the single-enterprise DP's mixed under-reporting strategy) and L2 (the bunching estimator on degenerate and peaked distributions); see `runs/G2_record.md`.

**G3: NOT PASSED**

##### Conditions

- heldout_evaluated: PASS (rows 2, 5, 6, 7 computed on the PLAN section 4.2 values; appearance is reported per row below)
- exploitability: FAIL
- oracle_gap_recorded: PASS
- jax_parity: PASS
- price_sensitivity: PASS (computed; a sign change is reported below, never suppressed)
- hygiene: PASS

##### Held-out phenomena (C0, mean over seeds [95% seed-bootstrap CI])

- Row 2 storming: excess Gini 0.2219 [0.1711, 0.2743] - APPEARS
- Row 5 hoarding: request inflation excess 1.7899 [1.6486, 1.9174]; corr(X, shortfall) excess 1.0329 [0.9270, 1.1169] (0 seeds with an undefined correlation) - APPEARS
- Row 6 blat: trade volume share 0.000e+00 [0.000e+00, 0.000e+00] - FAILURE (does not appear)
- Row 7 hidden reserves: C0 0.0842 [0.0622, 0.1108]; R7_NULL nan [nan, nan] (vanishes if upper < 0.01) - NOT EVALUATED (the R7_NULL arm was not run)
- Row 3 quality (pipeline check): mean qbar C0 - R3_QW nan [nan, nan] - NOT EVALUATED (the R3_QW arm was not run)

##### Oracle (WO-027)

- C0: W_oracle 1.9836, val_oracle 38.1873, status optimal, solver HiGHS (HiGHS via OR-Tools 9.15.6755), optimality gap 0.00e+00, horizon 40
- R7_NULL: W_oracle 1.9836, val_oracle 38.1873, status optimal, solver HiGHS (HiGHS via OR-Tools 9.15.6755), optimality gap 0.00e+00, horizon 40
- R3_QW: W_oracle 1.9836, val_oracle 38.1873, status optimal, solver HiGHS (HiGHS via OR-Tools 9.15.6755), optimality gap 0.00e+00, horizon 40
- Clairvoyant welfare, C0 seeds 0-4 (UPPER BOUND ONLY, never a denominator): 1.9791, 1.9797, 1.9812, 1.9816, 1.9815

##### Headline metrics and price sensitivity (C0)

- welfare_ratio W / W_oracle: 0.0000
- specification_gap at base prices and under price seeds (11, 12, 13): 0.0059, 0.0059, 0.0059, 0.0058 - sign change: no

##### Exploitability (WO-028)

- C0: max 0.1726, median -0.0082 over 10 seeds (threshold 0.05, provisional; finalised by the lead at G3 (PLAN section 6.3)) NON-CONVERGED

##### JAX parity (WO-029)

- max |NumPy - JAX| over 100 agent-steps: p1 1.07e-14, p2 7.11e-14 (tolerance 1e-05)

##### Hygiene

- BOUND_BINDING runs: 0
- max training-episode runaway fraction after 20%: 0.0000 (limit 0.05)

##### Arms and seeds

- C0: 10 seeds, overrides none
- R7_NULL: 0 seeds, overrides {'incentive': {'growth_directive': 0.0, 'penalty_arg': 'absolute'}}
- R3_QW: 0 seeds, overrides {'incentive': {'objective_metric': 'quality_weighted'}, 'information': {'quality_measurability': 1.0}}

Sizing: `{'n_envs': 8, 'rollout_steps': 125, 'total_agent_steps': 3000000, 'eval_every_updates': 250, 'eval_episodes': 10, 'measure_episodes': 100}`; git `451fd86718257e516baefc1046464b1f63718f53`.

---

## Phase 3 - contrasts (WO-032)

_Source: `runs/contrasts/report.md`_

### Contrasts (WO-032, PLAN section 4.3)

Design: spec/P3_REVISION.md S2. C0 = the 30 G3 runs; other arms 15 seeds each (reduced from PLAN's 30 for compute), common random numbers (`seed_env = 1000 + s`). IQM with a 95% percentile bootstrap over seeds (10,000 resamples). Limitations L1, L2 (runs/G2_record.md) and the G3 record apply.

Reporting rules (PLAN section 4.3):
1. **C_AUDIT is dual-classified and is never folded into C_OGAS.**
2. **No 'X% of the loss is informational' statement without a named contrast and a CI.**

Out of the PLAN section 3 sweep range (reported, not clamped): {'C_AUDIT': 0.4} (`audit_rate`, range [0.01, 0.30]).

#### Outcomes per arm (IQM [95% CI])

| arm | seeds | welfare_ratio | padding_index | specification_gap | convergence |
|---|---|---|---|---|---|
| C0 | 30 | 0.0295 [0.0170, 0.0435] | 2.1848 [1.3187, 5.5575] | 0.0107 [0.0034, 0.0180] | max expl. -3.790 |
| C_OGAS | 15 | 0.0002 [0.0000, 0.0014] | 809.8061 [205.0264, 4824.3460] | 0.0118 [0.0083, 0.0167] | max expl. -10.014 |
| C_AUDIT | 15 | 0.0240 [0.0080, 0.0497] | 3.5385 [1.0798, 11.5524] | 0.0019 [-0.0091, 0.0104] | max expl. -5.353 |
| C_INC | 15 | 0.0974 [0.0492, 0.1501] | 2.8105 [0.9736, 8.9241] | 0.0241 [-0.0104, 0.0601] | max expl. -1.178 |
| C_BOTH | 15 | 0.0249 [0.0084, 0.0499] | 6.4780 [2.3216, 24.1774] | 0.0390 [0.0310, 0.0501] | max expl. -4.368 |

#### Gap closed, seed-paired (Delta_X = welfare_ratio(C_X) - welfare_ratio(C0))

- C_OGAS: -0.0339 [-0.0503, -0.0163] over 15 paired seeds
- C_AUDIT: -0.0114 [-0.0336, 0.0201] over 15 paired seeds
- C_INC: 0.0676 [0.0208, 0.1106] over 15 paired seeds
- C_BOTH: -0.0083 [-0.0287, 0.0121] over 15 paired seeds
- Interaction I = Delta_BOTH - Delta_OGAS - Delta_INC: -0.0429 [-0.0811, -0.0018] over 15 paired seeds

#### Oracle

- C0: W_oracle 1.9836, val_oracle 38.1873, HiGHS (HiGHS via OR-Tools 9.15.6755), gap 0.00e+00
- C_OGAS: W_oracle 1.9836, val_oracle 38.1873, HiGHS (HiGHS via OR-Tools 9.15.6755), gap 0.00e+00
- C_AUDIT: W_oracle 1.9836, val_oracle 38.1873, HiGHS (HiGHS via OR-Tools 9.15.6755), gap 0.00e+00
- C_INC: W_oracle 1.9836, val_oracle 38.1873, HiGHS (HiGHS via OR-Tools 9.15.6755), gap 0.00e+00
- C_BOTH: W_oracle 1.9836, val_oracle 38.1873, HiGHS (HiGHS via OR-Tools 9.15.6755), gap 0.00e+00

Flags: bailed_out:0, bailed_out:1, bailed_out:10, bailed_out:11, bailed_out:12, bailed_out:13, bailed_out:14, bailed_out:15, bailed_out:16, bailed_out:17, bailed_out:18, bailed_out:19, bailed_out:2, bailed_out:3, bailed_out:4, bailed_out:5, bailed_out:6, bailed_out:7, bailed_out:8, bailed_out:9. git `bde9abab56f347922a444c78008e7f8f9248d7c1`.

---

## Phase 3 - estimator-bias study (WO-034)

_Source: `runs/estimator_bias/report.md`_

### Estimator-bias study (WO-034, PLAN section 7.2)

Design: spec/P3_REVISION.md S4. Truth: (a) the DP's exact stationary distribution, (b) the arm's pooled simulation, each measured against its own w = 0.25 counterpart at the same cap. Reused Phase-1 gate arms: 30 seeds; other arms: 5 seeds (reduced for compute). Estimator: gosplan.metrics._fallback 0.1.0+fallback. No digit tests; no calibration claim about archival detectors.

#### Pre-registered setting (excl[0.95,1.02]|deg9|bw0.005)

| w | cap | DP P(rho in [1.00,1.02]) | DP truth | sim truth | defined / seeds | sim bias | sim RMSE | sim median error | CI coverage |
|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 1.2 | 0.879 | undefined | 64.189 | 18 / 30 | 40497.882 | 97089.650 | 25.524 | 0.00 |
| 0.0 | inf | 0.879 | undefined | 72.461 | 3 / 5 | -41.530 | 48.069 | -52.385 | 0.00 |
| 0.02 | 1.2 | 0.000 | undefined | -3.405 | 5 / 5 | -0.486 | 0.530 | -0.592 | 0.20 |
| 0.02 | inf | 0.000 | undefined | -2.883 | 5 / 5 | -0.931 | 0.950 | -0.999 | 0.00 |
| 0.05 | 1.2 | 0.000 | undefined | -2.293 | 5 / 5 | -0.770 | 1.080 | -0.735 | 0.40 |
| 0.05 | inf | 0.000 | undefined | -1.710 | 5 / 5 | -1.292 | 1.568 | -1.594 | 0.00 |
| 0.1 | 1.2 | 0.000 | undefined | -0.643 | 5 / 5 | -1.383 | 1.682 | -1.409 | 0.20 |
| 0.1 | inf | 0.000 | undefined | -1.468 | 5 / 5 | -0.525 | 1.466 | -0.462 | 0.20 |
| 0.25 | 1.2 | 0.000 | undefined | 0.000 | 5 / 5 | 0.368 | 0.912 | 0.201 | 0.40 |
| 0.25 | inf | 0.000 | undefined | 0.000 | 30 / 30 | 0.078 | 1.692 | 0.078 | 0.33 |

Notes (read before the numbers):

- **DP truth is undefined in the estimator's units on every arm.** The DP's reports lie on its 0.02 report grid, so its stationary distribution is a set of point masses. Every smooth (w = 0.25) counterpart puts zero mass in the excess window [1.00, 1.02], and S4's truth divides by that counterpart's mean per-bin mass there. The DP column therefore reports the defined quantity, the DP's probability of a report in the window. The estimator is not scored against the DP (the estimator on a 0.02-grid point mass is degenerate at bin width 0.005 or 0.01 by construction).
- **Undefined estimates are counted, not dropped silently.** When a seed's measured mass falls almost entirely inside the excluded window, the polynomial counterfactual has no support and the estimate is infinite or its CI undefined (AMBIGUITY-022, as in the Phase-1 gate report). `defined / seeds` counts the seeds that enter bias, RMSE and median error. Near-zero but positive counterfactual support gives finite but huge estimates, which dominate the mean; the median error is shown beside it as a supplementary, robust summary (added when reporting, not pre-registered).
- These are the study's findings about the estimator under full bunching; nothing was re-tuned (PLAN section 4.5).

#### Across settings (simulation; mean |bias| over arms, mean coverage)

| setting | mean abs bias | mean RMSE | mean CI coverage |
|---|---|---|---|
| excl[0.95,1.02]|deg5|bw0.005 | 8931.370 | 21398.816 | 0.10 |
| excl[0.95,1.02]|deg5|bw0.01 | 4473.265 | 10717.567 | 0.10 |
| excl[0.95,1.02]|deg7|bw0.005 | 5797.919 | 13891.497 | 0.13 |
| excl[0.95,1.02]|deg7|bw0.01 | 2910.992 | 6974.567 | 0.13 |
| excl[0.95,1.02]|deg9|bw0.005 (pre-registered) | 4054.525 | 9714.760 | 0.17 |
| excl[0.95,1.02]|deg9|bw0.01 | 2042.768 | 4894.526 | 0.17 |
| excl[0.97,1.02]|deg5|bw0.005 | 9848.567 | 23593.672 | 0.10 |
| excl[0.97,1.02]|deg5|bw0.01 | 4928.152 | 11806.100 | 0.10 |
| excl[0.97,1.02]|deg7|bw0.005 | 6623.540 | 15868.038 | 0.09 |
| excl[0.97,1.02]|deg7|bw0.01 | 3319.295 | 7952.037 | 0.09 |
| excl[0.97,1.02]|deg9|bw0.005 | 4790.926 | 11478.110 | 0.15 |
| excl[0.97,1.02]|deg9|bw0.01 | 2405.982 | 5764.251 | 0.16 |
| excl[0.93,1.03]|deg5|bw0.005 | 302.670 | 554.173 | 0.11 |
| excl[0.93,1.03]|deg5|bw0.01 | 151.913 | 278.157 | 0.11 |
| excl[0.93,1.03]|deg7|bw0.005 | 185.291 | 344.397 | 0.11 |
| excl[0.93,1.03]|deg7|bw0.01 | 93.403 | 173.598 | 0.11 |
| excl[0.93,1.03]|deg9|bw0.005 | 122.138 | 231.809 | 0.14 |
| excl[0.93,1.03]|deg9|bw0.01 | 61.959 | 117.554 | 0.14 |

#### Reconciliation power curve

`ledger_test` at alpha = 0.05, claims inflated by x1.2 on the given share of enterprise-periods, 20 TruthfulMyopic episodes:

- share 0.0: rejection rate 0.05 (size)
- share 0.05: rejection rate 0.90
- share 0.1: rejection rate 1.00
- share 0.2: rejection rate 1.00
- share 0.3: rejection rate 1.00
- share 0.5: rejection rate 1.00

Figures: `bias_curves.png`, `power_curve.png`.

---

## Phase 3 - price-vector sensitivity (WO-036)

_Source: `runs/price_sensitivity/report.md`_

### Price-vector sensitivity (WO-036, PLAN sections 2.9.4, 7.5)

Perturbation `p_j exp(u_j)`, `u ~ N(0, 0.3**2)`, seeds (11, 12, 13) (keyed purpose `pricepert`; spec/P3_REVISION.md S6). Sign changes of `specification_gap` are listed first.

#### Sign changes

- G3 / contrasts: C0: mean over 30 rows keeps its sign; 0 of 30 rows change sign
- contrasts: C_OGAS: mean over 15 rows keeps its sign; 0 of 15 rows change sign
- contrasts: C_AUDIT: mean over 15 rows keeps its sign; 0 of 15 rows change sign
- contrasts: C_INC: mean over 15 rows keeps its sign; 1 of 15 rows change sign
- contrasts: C_BOTH: mean over 15 rows keeps its sign; 0 of 15 rows change sign

#### Headline metrics, mean over rows (base, then seeds 11, 12, 13)

##### G3 / contrasts: C0 (recomputation)
- padding_index: 15.7508, 15.9019, 15.3936, 15.7489
- welfare_ratio: 0.0339, 0.0339, 0.0339, 0.0339
- specification_gap: 0.0105, 0.0105, 0.0106, 0.0105

##### contrasts: C_OGAS (recomputation)
- padding_index: 3370.8822, 3707.1414, 2955.2849, 3514.6404
- welfare_ratio: 0.0065, 0.0065, 0.0065, 0.0065
- specification_gap: 0.0136, 0.0135, 0.0134, 0.0139

##### contrasts: C_AUDIT (recomputation)
- padding_index: 13.1511, 13.5948, 12.4030, 13.1957
- welfare_ratio: 0.0402, 0.0402, 0.0402, 0.0402
- specification_gap: 0.0322, 0.0313, 0.0330, 0.0326

##### contrasts: C_INC (rerun)
_recomputation only; the objective reads prices, a behavioural answer needs a rerun, which is not performed for compute reasons (spec/P3_REVISION.md S6)_
- padding_index: 39.4181, 38.8661, 39.7422, 39.2358
- welfare_ratio: 0.1012, 0.1012, 0.1012, 0.1012
- specification_gap: 0.0242, 0.0242, 0.0242, 0.0242

##### contrasts: C_BOTH (rerun)
_recomputation only; the objective reads prices, a behavioural answer needs a rerun, which is not performed for compute reasons (spec/P3_REVISION.md S6)_
- padding_index: 63.2391, 61.2403, 66.9154, 64.8501
- welfare_ratio: 0.0334, 0.0334, 0.0334, 0.0334
- specification_gap: 0.0408, 0.0410, 0.0408, 0.0407

#### Tables without a price-weighted metric

Phase-1 gate tables (runs/phase1_gate), the DP-vs-PPO tables and the estimator-bias tables carry no price-weighted headline metric (report ratios, padding in ratio units, excess mass), so a reprice leaves them unchanged; they are listed here so their omission is explicit.

---

## Phase 3 - LLM ministry study (WO-035)

_Source: `runs/llm_study/report.md`_

### LLM ministry study (WO-035) - NOT RUN

This container has no Anthropic SDK and no credentials (spec/P3_REVISION.md S5). The harness is implemented and unit-tested with a scripted client. To run it:

```
uv pip install anthropic
export ANTHROPIC_API_KEY=...   # or `ant auth login`
uv run python -m gosplan.experiments.llm_study claude-opus-5 claude-sonnet-5
```

Estimated cost (S5.4): about 50 ministry decisions per episode x 20 episodes x 2 framings x 3 arms x 2 models, about 16M tokens, plus the dominance check (no model calls). G4's 'LLM study' condition cannot pass until this is run.

---

## Phase 3 - record

_Source: `runs/P3_record.md`_

### Phase-3 record - LEAD

**Status:** the Phase-3 studies are complete as a *labelled study*. The one exception is the LLM
ministry study, which is **NOT RUN** because no model access exists in this container. G4
(PLAN section 13) needs the human's sign-off, and its "LLM study" condition cannot pass until the
owner runs S5. Written on 2026-09-28.

- Design: `spec/P3_REVISION.md` S1-S7, including both S4 amendments.
- The embedded reports: `runs/final_report/report.md`.

#### Limitations carried into every Phase-3 number

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

#### Contrasts (WO-032): `runs/contrasts/report.md`

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

#### Estimator bias (WO-034): `runs/estimator_bias/report.md`

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

#### Price sensitivity (WO-036): `runs/price_sensitivity/report.md`

- Across price seeds 11, 12 and 13, `specification_gap` keeps its sign in every table's mean.
  One C_INC row of 15 flips.
- welfare_ratio is price-invariant to 4 decimals in every arm. padding_index moves by about 3-10%.
- For C_INC and C_BOTH (price-reading objective) this is a recomputation, not a behavioural rerun.

#### LLM ministry study (WO-035): `runs/llm_study/report.md`

**NOT RUN.** The harness, the dominance check and the not-run report are implemented and
unit-tested. The owner's command is in the report: install the SDK, add a credential, then
`uv run python -m gosplan.experiments.llm_study claude-opus-5 claude-sonnet-5`.

#### Update (2026-09-28): G3b complete

See `runs/G3b_record.md`. With trade possible (spec 2.1.0), C0 populations post only buy offers,
so row 6 still does not appear. That makes it unlikely, though not shown by a re-run, that L3
changed the contrasts' C0 leg. The revised audit (R16) finds the C0 populations non-converged
(median exploitability 0.80). This is limitation **L4**, and it applies to every contrast above.

#### Update (2026-09-29): LC complete

See `runs/LC_record.md`. At three times the training budget, the C0 populations stop producing
(welfare 0 on every seed). Every contrast above comes from 1M-step populations, so it describes a
point on a learning trajectory, not a steady state. This is limitation **L5**. The contrasts are not
re-run at 3M.

#### Next (owner decisions at G3, delegated) - as written before G3b

1. Apply R15 (trade offers posted at REPORT; spec 2.1.0) and R16 (the revised exploitability
   audit).
2. Run the labelled **G3b** study: C0 at 30 seeds under spec 2.1.0. It re-evaluates row 6 and runs
   the R16 audit. The result goes in `runs/G3b_record.md`. G3 stays NOT PASSED as recorded.
3. G4 sign-off belongs to the human.

---

## Manifest roll-up (CONTRACT rule 10)

535 run manifests; the full table is `runs/final_report/manifests.md`.

