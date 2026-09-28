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

| w | cap | DP truth | DP-sample bias | sim truth | sim bias | sim RMSE | CI coverage |
|---|---|---|---|---|---|---|---|
| 0.0 | 1.2 | nan | nan | 64.189 | 40497.882 | 97089.650 | 0.00 |
| 0.0 | inf | nan | nan | 72.461 | -41.530 | 48.069 | 0.00 |
| 0.02 | 1.2 | nan | nan | -3.405 | -0.486 | 0.530 | 0.20 |
| 0.02 | inf | nan | nan | -2.883 | -0.931 | 0.950 | 0.00 |
| 0.05 | 1.2 | nan | nan | -2.293 | -0.770 | 1.080 | 0.40 |
| 0.05 | inf | nan | nan | -1.710 | -1.292 | 1.568 | 0.00 |
| 0.1 | 1.2 | nan | nan | -0.643 | -1.383 | 1.682 | 0.20 |
| 0.1 | inf | nan | nan | -1.468 | -0.525 | 1.466 | 0.20 |
| 0.25 | 1.2 | nan | nan | 0.000 | 0.368 | 0.912 | 0.40 |
| 0.25 | inf | nan | nan | 0.000 | 0.078 | 1.692 | 0.33 |

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

## Manifest roll-up (CONTRACT rule 10)

455 run manifests; the full table is `runs/final_report/manifests.md`.

