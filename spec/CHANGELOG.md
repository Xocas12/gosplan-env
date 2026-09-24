# spec/spec.py - CHANGELOG

This file exists because of **CONTRACT rule 1 (FROZEN SPEC)**:

> `spec/spec.py` is provisional (v0) until gate G1 and frozen (v1) thereafter. After v1, only the
> lead may change it, and only with a `spec/CHANGELOG.md` entry (version, reason, affected work
> orders). No other session edits `spec/spec.py`.

`spec/spec.py` is the single source of truth for every type, signature, enum and `Protocol` in the
repository. Implementers code against it and never edit it; if it is wrong or silent, the response
is an AMBIGUITY REPORT (CONTRACT rule 3), not a local fix.

## Versioning policy

| Version range | State | Who may change it | Requirement |
|---|---|---|---|
| `0.y.z` | **provisional (v0)** - the interface skeleton of PLAN §10, before gate G1 | LEAD only | an entry here; no downstream re-run obligation while the suite is still red |
| `1.0.0` | **frozen (v1)** - set by WO-013 at gate G1 (`SPEC_VERSION = "1.0.0"`) | LEAD only | full entry below, and the freeze recorded as a G1 artefact (PLAN §13) |
| `> 1.0.0` | post-freeze amendment | LEAD only | full entry below, signed off before any dependent work order is reissued |

Semantics after the freeze: **patch** = docstring, comment or type-alias clarification with no
signature change; **minor** = additive (a new symbol, a new optional field with a default) that
cannot change existing behaviour; **major** = any change to an existing signature, field, enum
member or documented semantics. A major bump invalidates every golden file generated under the
previous version: they are regenerated from `ref/` and the full frozen suite is re-run before the
entry is signed.

The parameter-arm classification in `gosplan/params.py` is a design decision; changing an arm
assignment also requires an entry here (**CONTRACT rule 11**), even when no signature moves.

## Required entry format

Every entry after v1 carries all of these fields. An entry missing any of them is not a valid
change, and the change is reverted rather than documented after the fact.

```
## <version> - <YYYY-MM-DD>

**Reason.** Why the spec had to move. Cite the PLAN section and, where the change originates in an
ambiguity report or a gate finding, the WO-### / gate that raised it.

**Change.** Exactly what moved: symbols added, removed or re-signed, with before -> after.

**Affected work orders.** WO-### … - every card whose whitelist, "write only" list or must-pass
tests are touched, including cards already completed that must be reissued.

**Golden files.** Regenerated / not regenerated, and why.

**Suite.** Result of the full frozen suite after the change (unit, behavioural, golden).

**Approver.** LEAD, plus Human where the change crosses a gate or alters a pre-registered quantity
(PLAN §4).
```

---

## Unreleased

No unreleased changes.

New entries are added here, in the format above, and are promoted to a numbered section when the
version is bumped in `spec/spec.py`.

---

## 0.1.0 - 2026-09-05

**Reason.** Initial skeleton of the repository. Establishes the frozen shape - types, signatures,
contract text, parameter registry and work-order cards - before any behaviour exists, so that the
implementation cannot quietly redefine the question (PLAN §0, §12).

**Change.** First version. `spec/spec.py` v0 written as the full expansion of PLAN §10: enums,
config dataclasses with the Phase-1 field defaults of PLAN §3, state / action / view types, RNG,
production, planner, reporting, reward, environment, agent, DP, ledger and metrics interfaces, plus
the P2 sketch signatures whose behaviour is deliberately under-specified until the Phase-2 spec
revision. Every body is `raise NotImplementedError(...)`. Alongside it: `CONTRACT.md` (PLAN §9,
verbatim), `gosplan/params.py` (PLAN §3 as data, with `arm`, `phase`, `default`, `range`, `source`),
the work-order cards and templates in `workorders/`, the packaging and lint configuration, and the
prose documents listed in `README.md`.

**Affected work orders.** All of them - this version is the baseline every card is written against.
Directly: WO-001 (spec v0, CONTRACT, registry) and WO-002 (reference dynamics and frozen tests).

**Golden files.** None generated. `ref/ref_step.py` is a skeleton; golden files are first produced
by WO-002 once the reference dynamics exist.

**Suite.** Not run to green. The frozen suites collect and fail at `NotImplementedError`, which is
the expected state of a skeleton (PLAN §11; `README.md`).

**Approver.** LEAD. Not a frozen version: `spec/spec.py` stays provisional until WO-013 bumps it to
`1.0.0` at gate G1.

## 0.1.1 - 2026-09-07

**Change.** Four under-determined points pinned, each raised as an ambiguity report against a card
that could not be executed without the answer. No signature, field, default or range changed; every
edit is to a docstring that is the specification of a behaviour not yet implemented.

- **#51 - canonical JSON bytes for `EnvConfig.hash`.** The five semantic bullets (keys sorted,
  floats by `repr`, `inf` as `"inf"`, tuples as arrays, sha256 lowercase hex) did not determine the
  bytes, yet `ref/gen_golden.config_hash` re-derives the digest independently and
  `tests/golden/test_golden_parity.py` asserts the two are equal. Pinned to
  `json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)` over the nested
  per-section object, no trailing newline, UTF-8. Recorded in `spec/spec.py`, `gosplan/config.py`
  and `ref/gen_golden.py` so each side is pinned inside its own whitelist.
- **#52 - the generator call for `bernoulli` and `categorical`.** These were pinned only in
  `ref/ref_step.py`, which is not on WO-004's exhaustive whitelist, so the card was not executable
  under CONTRACT rule 12. The exact calls now appear in `gosplan/rng.py`'s own docstring.
  `gen.random(size=shape) < p` and `gen.binomial(1, p, shape)` agree in distribution but consume
  the generator differently, so the substitution would have surfaced at WO-009 as an unexplained
  golden-parity break.
- **#53 - the `selfobs` key.** `gosplan/env/obs.py` put the enterprise index `i` in the key while
  `gosplan/rng.py` states twice that the trailing index is vectorised through `shape`. Resolved in
  favour of the convention: the key is `(seed_env, "selfobs", t, k)` with `shape=(N,)`. Phase 1
  sets `self_obs_noise = 0.0`, so the two readings are numerically identical for all of Phase 1 and
  the divergence would first have appeared in a Phase-2 information arm, after Phase-1 results were
  already recorded.
- **#54 - `BOUND_BINDING`.** Specified twice: as a monotone latch on a running fraction in
  `Ledger.append`, and as a predicate on the finished ledger in `bound_binding`. These are not
  equivalent - one at-bound report row followed by 99 clean ones latches the flag while the
  predicate returns `False` - so a gate G2 hygiene outcome could turn on row order. Resolved to the
  predicate: `append` maintains no flag, and `BOUND_BINDING` is evaluated once when the manifest is
  written.

**Affected work orders.** WO-002 (writes the frozen tests for all four), WO-003 (`hash`),
WO-004 (`draw`), WO-008 (`build_observation`), WO-011 (ledger and manifest).

**Golden files.** None yet; `ref/ref_step.py` is still a skeleton. All four decisions land before
the first golden file is generated, which is the point of resolving them now.

**Suite.** 93 passed, 233 skipped - unchanged. Every edit is to a docstring.

**Approver.** LEAD. `spec/spec.py` remains provisional until WO-013 bumps it to `1.0.0` at gate G1.

## 0.1.2 - 2026-09-07

**Change.** The denominator of the input-coverage observation fields is pinned. Docstrings only; no
signature, field, default or range changed.

- **#60 (AMB-007) - which `need_ij` do observation fields 11, `12:12+J` and `12+2J:12+3J` divide
  by?** PLAN section 2.4 names `need_ij` without saying whether it is the planned need of section
  2.7.2 or the per-step production need of section 2.6. Resolved to the **period need at target**,

      need_ij = a_{s(i)j} * T_i

  constant within a period. The production need `a_{s(i)j} * y_hat_ik` was rejected on two
  structural grounds, not on taste: it is undefined at the REPORT step, where no effort is chosen,
  leaving `2 + 2J` fields without a value once every `M + 1` steps; and it is `0/0` at zero effort,
  where the documented `need_ij == 0` convention would report FULL input coverage to an enterprise
  holding no inputs at all, inverting the field's meaning exactly where it matters most.

  The report was filed by WO-002 at the instruction of its own `ref_observation` docstring, which
  forbids generating any golden file until this is settled.

- **Which I-O matrix.** The denominator uses the enterprise's TRUE row `a_{s(i)j}`, not the
  planner's possibly stale `planner_io[s(i), j]` that PLAN section 2.7.2 uses for allocation. The
  two coincide in Phase 1 and diverge in Phase 2 once technology drifts. The true row is correct for
  an agent-facing field: an enterprise knows its own production function, while `planner_io` is a
  planner-side belief, and putting it in an observation would leak the planner's estimate into a
  policy input. `gosplan/env/obs.py` already stated it this way; `ref/ref_step.py` now agrees, so
  the oracle and the implementation cannot drift on it.

**Affected work orders.** WO-002 (generates the golden files), WO-008 (`build_observation`),
WO-009 (supplies `need` to the observation builder).

**Golden files.** Still none. This decision is a precondition for the first one.

**Suite.** 93 passed, 233 skipped - unchanged.

**Approver.** LEAD. `spec/spec.py` remains provisional until WO-013 bumps it to `1.0.0` at gate G1.

## 0.1.3 - 2026-09-07

**Change.** The T-U1 conservation identity is corrected, and `ref_conservation_residual` gains the
two arguments the correct identity needs. Found by the WO-002 hand-checked worked example.

- **#64 (AMB-009) - the stated T-U1 identity does not balance.** PLAN section 11 states

      sum y + sum S_prev = sum inputs consumed + sum consumer + sum S_next
                           + holding loss + cap overflow

  which omits the goods sitting in buyers' input stocks: a unit delivered into `X` has left the
  seller's `S` but has not been consumed, so it appears on neither side and the residual is exactly
  the period's change in `X`. Corrected to carry the input stocks explicitly, per good `j`:

      sum_i y_i + sum_i S_prev_i + sum_i X_prev_ij
          = sum_i S_next_i + sum_i X_next_ij + sum_i consumed_ij
            + consumer_j + sum_i holding_i + sum_i overflow_i

  Substituting `X_next = X_prev + deliv - consumed` reduces this to
  `y + S_prev = S_next + deliv + consumer + holding + overflow`, and `deliv_j + consumer_j` is
  exactly `sum_i shipped_i` for that good, which is why it balances. Verified: residual
  `[0.0, -1.11e-16]` and `[0.0, 0.0]` over two periods of the worked example.

- **Signature.** `ref_conservation_residual` now takes `inputs_prev` and `inputs_next` alongside
  `inputs_consumed`. It is a `ref/`-only function - it mirrors no `spec/spec.py` callable - so no
  frozen interface moves. `_run_period` snapshots `X` at the top of the period to supply it.

**Why this mattered more than it looks.** `_run_period` asserts the identity every period, so under
the stated form the reference could not run any economy in which goods are actually delivered. It
passed only in two degenerate cases: an economy with no I-O links, and the cold-start deadlock of
#62 where every quantity is zero - **zeros conserve**. Those are exactly the two situations
reachable before this worked example existed, which is why the defect survived step 2.

**Affected work orders.** WO-002 (writes T-U1 in `tests/unit/test_conservation.py`), WO-009 (the
production step function must satisfy the same identity).

**Golden files.** Still none, and still blocked on #62.

**Suite.** 93 passed, 233 skipped - unchanged.

**Approver.** LEAD. `spec/spec.py` remains provisional until WO-013 bumps it to `1.0.0` at gate G1.

## 0.1.4 - 2026-09-07

**Change.** Three open ambiguity reports resolved by the human at the gate they were raised for.
One is a behavioural change to the reference dynamics; two are ownership and naming decisions.

- **#62 (AMB-008) - the Phase-1 economy had no cold start.** With `inv_inputs = 0` at reset and
  every Phase-1 sector requiring inputs, the dynamics had a fixed point at zero: zero coverage gives
  zero output, which gives nothing to claim, which gives `avail_j = 0`, which leaves `X` at zero
  next period. Resolved by endowing one period's input need at the initial target,

      X_ij = a_{s(i)j} * T_0_i

  at reset. Chosen over an exogenous first delivery (same effect, fix in the schedule rather than
  the state) and over a coverage floor `H_min > 0`, which was rejected because it would weaken the
  CES complementarity of PLAN section 2.6 - the mechanism behind held-out phenomenon 5. The
  endowment is derived from parameters that already exist rather than a new constant, and it is the
  smallest quantity that lets a truthful enterprise reach its opening target.

  Verified: a 30-step rollout that previously reported `val_true = 0.000000` in every period now
  reports 1.698735 in period 0.

- **#48 (AMB-001) - `tests/acceptance/` was authored by no work order.** PLAN section 11 assigns the
  Acceptance category to LEAD but section 12 issued no card, so CONTRACT rule 12's invariant that
  every file has an owning card did not hold. The five gate harnesses and their README are added to
  **WO-013**'s Write-only list: that card is LEAD and already re-runs the full suite at the freeze.
  Rule 13 is untouched - the directory stays out of `testpaths` and off every must-pass list.

- **#49 (AMB-002) - module names for the P2 and P3 harnesses.** `gosplan/experiments/`
  `phase2_acceptance.py` (WO-031) and `report.py` (WO-037) are confirmed now rather than deferred to
  the P2 spec revision. PLAN section 8's tree omitted both; these are additions to it. A later
  rename costs one card edit, so deferring bought nothing.

**Affected work orders.** WO-002 (reference dynamics, and the golden matrix this unblocks),
WO-009 (`reset` must build the same opening state), WO-013 (gains `tests/acceptance/`),
WO-031 and WO-037 (names fixed).

**Golden files.** Still none - but #62 was the blocker, so WO-002 step 5 can now proceed.

**Approver.** Human, 2026-09-07, on the four decisions put to them at this point in the build.

## 0.1.5 - 2026-09-07

**Change.** WO-002 step 5: `ref/gen_golden.py` implemented and the golden matrix generated. The
five configurations are recorded here because `golden_configs`' own docstring requires it - PLAN
section 11 fixes the count at five and the property tests imply the coverage, but the five documents
are written nowhere in PLAN.md, and that docstring states that choosing them without recording the
choice is a CONTRACT rule 3 violation.

| name | size | what it carries that no other configuration does |
|---|---|---|
| `notched` | N=2, J=2 | the **notched** bonus (`w = 0`, `rho_cap = 1.2`), finite `theta`, `penalty_arg = positive_part`, `g > 0` |
| `smooth` | N=2, J=2 | the **smooth counterfactual** (`w = 0.25`, `rho_cap = inf`) - no discontinuity and no kink anywhere - and the `absolute` penalty branch (T-U8) |
| `kink_leontief` | N=3, J=3 | the **kink-only** arm (`w = 0.25`, `rho_cap = 1.2`) and `input_complementarity = inf`, the Leontief `min` branch of the coverage aggregator (T-U7) |
| `zero_growth` | N=2, J=2 | `growth_directive = 0`, the fixed point of the target rule (T-U4, T-B2) |
| `zero_io_row` | N=4, J=3 | an enterprise whose I-O row is all zeros, so `H = 1` is exercised (T-U7); `M = 4`; the `fixed` horizon mode |

Every configuration keeps `audit_rate = 0.5`, so an audit fires within 30 agent-steps and the
penalty path is exercised in every cell. Every I-O row satisfies `sum_k a_jk < 1`, which
`EnvConfig.validate` requires and which the cost-plus price fixed point needs to converge.

**Matrix.** 5 configurations x 3 seeds x 2 agents = **30 files**, 30 agent-steps each, written to
`tests/golden/` (git-ignored; regenerated by `make golden`).

**Verified.**
- `python -m ref.gen_golden --check` reports **0 stale or missing files** on a regeneration, so the
  matrix is reproducible from `(config, seed, agent)` alone.
- The reference's own per-period conservation self-check raised on none of the 30 rollouts, which
  is the first time it has been exercised on a live economy - before #62 was resolved every
  configuration was deadlocked at zero, where the identity holds trivially.
- `tests/golden/test_golden_parity.py`: **31 of 151 now pass** - the matrix completeness check and
  the per-file schema check. The remaining 120 gate on `load_config` (WO-003) and will activate
  when it lands, with no edit to any frozen file.

**Affected work orders.** WO-002 (complete), WO-003 (`load_config` activates 120 parity assertions),
WO-009 (`GosplanEnv` is what the parity test replays against), WO-013 (regenerates the matrix at the
spec v1 freeze).

**Approver.** LEAD.

## 0.1.6 - 2026-09-24

**Reason.** PLAN section 5 / WO-014, AMBIGUITY-010 item A. The frozen `tests/unit/test_dp.py`
checks `sol.n_iterations <= grid.max_iterations`, and "non-convergence is a result" (WO-014 card)
needs a declared iteration cap at which the solver stops and returns `converged = False`; the v0
`DPGrid` declared none.

**Change.** `DPGrid`: added `max_iterations: int = 5000` after `value_tol` (additive, defaulted;
no existing field, signature or default moved). Mirrored field for field in
`gosplan/agents/dp.py`. `SPEC_VERSION` stays "0.1.0" until the WO-013 freeze, as for 0.1.1-0.1.5.

**Affected work orders.** WO-014 (implements the cap), WO-015 (uses `DPGrid()` defaults), WO-019
(DP-vs-PPO reads `DPSolution.converged`). The `tests/unit/test_dp.py` helpers were aligned with the
declared names by a lead edit (AMBIGUITY-010 A).

**Golden files.** Not regenerated: the DP is not part of the golden dynamics.

**Suite.** Full frozen suite 455 passed, 16 skipped, 0 failed.

**Approver.** LEAD.

## 1.0.0 - 2026-09-24

**Reason.** WO-013, the v1 freeze at gate G1 (PLAN sections 10, 12.3, 13), after gate G0 was signed
off (`runs/G0_signoff.md`: full frozen suite green; MC sanity report clean over 126,000 episodes;
LEAD rule-7 review clean). Each amendment below names the G0 finding or ambiguity report behind it.

**Change.**
- `SPEC_VERSION`: "0.1.0" -> "1.0.0" in `spec/spec.py` and the mirror in `gosplan/config.py`.
- Recorded as owed by the v0 file: `draw`'s `purpose` / `dist` are narrowed from `str` (PLAN
  section 10) to the `Purpose` and `Dist` literals, which enumerate exactly the values of PLAN
  section 2.15 and the WO-004 card; `Purpose` carries `selfobs` beyond section 2.15's list, for the
  self-observation noise of WO-008 (PLAN section 2.4).
- `GosplanEnv.reset` docstring: `inv_inputs = a_{s(i)j} * T_0_i` (ambiguity #62, CHANGELOG 0.1.4;
  confirmed by golden parity in G0).
- `EnterpriseAction.input_request` comment and `process_reports` docstring: the action is a
  multiple of need in `[0, r_max]`, rescaled by need when read (AMBIGUITY-008; G0 T-B1).
- `GosplanEnv.step` docstring: counters advance to the next agent-step, the observation describes
  the executed step (AMBIGUITY-007), and a step after `done` auto-continues (AMBIGUITY-004).
- Seeding convention for every harness after G0 (AMBIGUITY-014, a G0 finding): episode `e` uses
  `seed_env = root + e`, shared across agents and arms at the same `e`; root and rule in the
  manifest `flags`. No signature change.
- No signature, field, enum member or default moved at this version; the only field added since
  0.1.0 is `DPGrid.max_iterations` (0.1.6).

Folded in from "Unreleased" (LEAD rulings on this branch, none a `spec/spec.py` signature change):

**Reason.** Defect in the reference golden generator found while checking WO-003:
`ref/gen_golden.py::_base` wrote `"invest_lag": 0`, contradicting its own docstring ("every
unstated default is the one declared on `SupplyConfig`", which is 1) and the `invest_lag >= 1`
rule of `EnvConfig.validate` (spec/spec.py, WO-003). With golden files present, every golden
parity test failed in `validate()` before comparing anything.

**Change.** `ref/gen_golden.py`: `invest_lag` 0 -> 1 in the base golden configuration. No change
to `spec/spec.py`; `SPEC_VERSION` unchanged. The only dynamics effect is the width of
`pending_invest`, which is inert in Phase 1 (`v == 0`).

**Affected work orders.** WO-002 (ref), WO-009 (golden parity must-pass).

**Golden files.** Regenerated locally (git-ignored; not committed).

**Suite.** `tests/golden`: 31 passed (config hash and obs layout parity), 120 skipped awaiting
WO-009/WO-010.

**Approver.** LEAD.

Further unreleased LEAD rulings on this branch (no `spec/spec.py` change; docstrings to be corrected
at the WO-013 freeze): AMBIGUITY-003 (yield shock keyed per enterprise, as `ref_produce`),
AMBIGUITY-004 (`GosplanEnv.step` after `done` continues as `ref_rollout` does), AMBIGUITY-005
(`Ledger.append` keeps `BOUND_BINDING` in step; lead edit of the `tests/unit/test_ledger.py` fixture
under `.github/FROZEN_TEST_EXEMPTION`), and the WO-003 follow-up (`EnvConfig.hash` covers the
four-section object only, per ambiguity #51). AMBIGUITY-006 (smooth-arm slope term smoothed by a softplus
at the notch width in both `gosplan/env/reward.py` and `ref/ref_step.py`; golden set regenerated;
lead edit of T-U3's threshold in `tests/unit/test_reward.py` under `.github/FROZEN_TEST_EXEMPTION`),
AMBIGUITY-007 (state counters advance eagerly; `ref_step` renders the digest at the next position)
and AMBIGUITY-008 (golden truthful policy, post-REPORT truthful report, `input_request` as a
multiple of need, rollout observation, and two lead test edits). Golden set regenerated after each.
AMBIGUITY-009 (`DPGreedy` grid lookup) and AMBIGUITY-010 (WO-014 DP rulings; the one `spec/spec.py`
change is entry 0.1.6 below).

**Affected work orders.** WO-003 (mirror constant), WO-004 (narrowing recorded), WO-008
(`selfobs`), WO-009 (docstrings), WO-012 (seeding convention going forward), WO-013 (this entry;
gate harnesses G0 and G1 implemented in `tests/acceptance/`, G2-G4 remain stubs until their
experiments exist), WO-016 onward (seeding convention). No completed card needs reissuing: no
behaviour changed.

**Golden files.** Regenerated: `ref.gen_golden --check` reported all 30 stale solely because each
document embeds `spec_version`; a key-by-key comparison of the regenerated set against the previous
one showed `spec_version` as the only differing key (dynamics unchanged).

**Suite.** Full frozen suite after the freeze: 456 passed, 16 skipped (later cards), 0 failed - identical to before the freeze; no skip introduced.

**Approver.** LEAD. Crosses gate G1: the human's G1 decision (`runs/G1_decision.md`) follows this
freeze and does not alter it; the daggered PLAN section 3 values stay at their provisional
placeholders until then.

## 1.0.1 - 2026-09-24

**Reason.** PLAN section 4.5 pre-registration, amended BEFORE ANY LEARNING RUN under the owner's
delegation (AMBIGUITY-011 resolution): the degree-7 counterfactual is biased on smooth report
distributions peaked near 1 (e.g. +0.058 on N(1, 0.15), +0.43 on N(1, 0.10)), enough to fail the
frozen smooth-null test and to decide G2's smooth arm by estimator bias rather than behaviour.

**Change.** Patch (docstring only in `spec/spec.py`): `phenomenon_bunching`'s docstring now reads
"polynomial of degree 9". `gosplan/metrics/phenomena.py`: `BUNCHING_POLY_DEGREE` 7 -> 9.
`gosplan/metrics/_fallback.py`: seed-level percentile bootstrap (1,000 replicates, 95%, seed 0).
`gosplan/metrics/__init__.py`: `resolve_estimators` implemented; `EstimatorBackend` gains
read-only mapping access (`__getitem__`, `__contains__`), which the frozen WO-016 tests use
(AMBIGUITY-015). G2 criterion 2's notched-arm threshold amended for non-finite `b_hat_DP`
(AMBIGUITY-011 point 3). No signature moved.

**Affected work orders.** WO-014 (`dp_excess_mass` now uses degree 9), WO-016 (estimator), WO-020
(criterion 2 threshold). Lead edit of `tests/unit/test_phenomena_p1.py` (degree assertion and
docstrings), recorded in `.github/FROZEN_TEST_EXEMPTION`.

**Golden files.** Not regenerated: the estimator is not part of the golden dynamics.

**Suite.** Full frozen suite after the change: 461 passed, 11 skipped (later cards), 0 failed.

**Approver.** LEAD, under the owner's explicit delegation of the AMBIGUITY-011 decisions; crosses a
pre-registered quantity of PLAN section 4, amended before any learning run.

## 1.1.0 - 2026-09-24

**Reason.** Gate G1 (PLAN section 13): the six daggered PLAN section 3 values, declared provisional
placeholders "replaced at G1", are replaced by the G1 decision (`runs/G1_decision.md`; the owner
delegated the choice to the LEAD). Folding them into the defaults is the lead decision WO-013
note 6 reserves for G1.

**Change.** Defaults of `IncentiveConfig` / `InformationConfig` in `spec/spec.py`, mirrored in
`gosplan/config.py` and `gosplan/params.py` (`provisional` -> False): `ratchet_lambda` 0.5 -> 0.53,
`growth_directive` 0.02 -> 0.021, `overfulfilment_slope` 0.5 -> 0.331, `effort_cost` 0.15 ->
0.193, `penalty_scale` 60.0 -> 200.0, `audit_rate` 0.10 (unchanged, now final). No field, type or
signature moved; minor because replacing these placeholders at G1 is their documented semantics.

**Affected work orders.** WO-016 onward (every Phase-1 experiment runs at `p1_default_config()`),
WO-019 (the three `a*pen` levels 0.8 / 4 / 20), WO-020 (criterion-2 threshold 0.440).

**Golden files.** Regenerated for the embedded `spec_version` only; the golden configurations are
written literally in `ref/gen_golden.py` and do not read these defaults.

**Suite.** Full frozen suite at the new defaults: 461 passed, 11 skipped (later cards), 0 failed.

**Approver.** LEAD under the owner's written delegation of the G1 decision (Human role).

## 1.1.1 - 2026-09-24

**Reason.** PLAN section 14 budgets the Phase-1 gate at about 3k env steps per second; the
environment ran at about 700 per second at `N = 20`, half of it building per-enterprise ledger
rows that training never reads (G0 cost observation, `runs/G0_signoff.md`).

**Change.** Additive, keyword-only, defaulted: `GosplanEnv.__init__(cfg, *, records: bool = True)`
in `spec/spec.py` and `gosplan/env/env.py`; `gosplan.env.step.advance(..., *, records=True)`.
With `records=False` and no ledger attached, `StepRecord`s are built only at the DELIVER step
(the observation needs that step's deliveries); with the default, or whenever a ledger is
attached, behaviour is unchanged. `GosplanEnv.step` also copies the state field-wise instead of
`copy.deepcopy` (same semantics). Measured: 1.36 -> 0.74 ms per agent-step at `N = 20`.

**Affected work orders.** WO-009 (env), WO-018 (training uses `records=False`).

**Golden files.** Not regenerated: dynamics unchanged (T-B7 exact).

**Suite.** Full frozen suite: 461 passed, 11 skipped, 0 failed; golden parity exact.

**Approver.** LEAD.

