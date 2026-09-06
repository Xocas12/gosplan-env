# Parameter sourcing memo - WO-000

> **TEMPLATE - UNFILLED. WO-000 HAS NOT BEEN EXECUTED.**
> Every Status cell below reads `UNSOURCED - WO-000 not executed`, every Source cell reads
> `TBD (WO-000)`, and every research item below is a question, not an answer. Nothing in this file
> is a historical claim. Do not cite it, and do not treat any provisional value in it as evidence
> about anything that happened.

**Work order.** WO-000 Parameter sourcing memo · P1 · LEAD · Difficulty 3 (PLAN §12.3).
Research items: PLAN §15. Output: this file, with, per provisional (†) parameter, **either** a
sourced range **or** an explicit "unsourced; prior = …". No code. WO-000 runs before any
implementation.

**What this memo is for.** The Phase-1 defaults in PLAN §3 marked with a dagger (†) are
placeholders. They are replaced at gate **G1** by values the human selects from the interior of the
bunching region of the DP regime map, recorded in `runs/G1_decision.md` **before any training run**
(PLAN §13). This memo does not select those values. It establishes, for each of them, what the
historical and theoretical literature can and cannot support - so that the G1 choice is made against
a stated prior and a stated sweep range, and so that no downstream table silently upgrades an
engineering guess into a historical fact.

**Scope note.** A sourced range constrains the *sweep*, never the conclusion. Sensitivity to
everything that stays unsourced is handled by the regime map (PLAN §5, WO-015) and, in Phase 3,
optionally by the Saltelli / total-order Sobol design over the swept INFO+INC parameters, whose
ranges are stated as an assumption in the same table as the indices (PLAN §4.3).

## How to fill this in

1. **Status vocabulary.** Exactly one of:
   - `SOURCED` - a range supported by cited work, with the range and the citation both given;
   - `PARTIAL` - the literature constrains one side of the range, or the qualitative structure only;
   - `UNSOURCED` - no defensible range; an explicit prior is stated instead.
2. **Citation format.** Author, year, work, and the page / table / series carrying the number. For
   archival series also record the **level** (enterprise / ministry / republic), the coverage years,
   and what the series actually measures.
3. **Never a point value.** A filled cell carries a range or an explicit prior. A single number with
   a citation attached is still a violation of the closing rule below unless the source itself gives
   a range and the range is reproduced.
4. **Units.** Every range is stated in this environment's units (PLAN §2.6-§2.9) with the conversion
   shown, not in the source's units. A schedule expressed as a fraction of base pay is converted
   against `β`, the normalising unit (PLAN §3).
5. **Arms.** A parameter's INFO / INC / SUPPLY / TECH arm is a design decision; this memo does not
   change one. Re-classification requires a `spec/CHANGELOG.md` entry (CONTRACT rule 11).
6. **Dual classification.** `audit_rate` sits in the INFO arm *and* enters the reward. It is
   reported separately and is never folded into the OGAS information contrast (PLAN §4.3).

## Provisional (†) parameters

One row per parameter marked † in PLAN §3. Provisional values and sweep ranges are copied from that
table; the Status and Source columns are this memo's deliverable.

| Parameter | Arm | Provisional value | Sweep range | Status | Source |
|---|---|---|---|---|---|
| `audit_rate` (a) | INFO (dual: also enters the reward; reported separately) | 0.10† | [0.01, 0.30] | UNSOURCED - WO-000 not executed | TBD (WO-000) |
| `ratchet_lambda` (λ) | INC | 0.5† | [0, 1] | UNSOURCED - WO-000 not executed | TBD (WO-000) |
| `growth_directive` (g) | INC | 0.02† | [0, 0.07] | UNSOURCED - WO-000 not executed | TBD (WO-000) |
| `overfulfilment_slope` (s) | INC | 0.5† | [0, 2] | UNSOURCED - WO-000 not executed | TBD (WO-000) |
| `penalty_scale` (pen) | INC | 60† | [5, 200] | UNSOURCED - WO-000 not executed | TBD (WO-000) |
| `effort_cost` (κ) | INC | 0.15† | [0.05, 0.5] | UNSOURCED - WO-000 not executed | TBD (WO-000) |

Non-† parameters that the research items below also touch (`notch_height` β, `overfulfilment_cap`
ρ_cap, `ratchet_deadband` δ, `penalty_form`, `tenure` ψ, and the reporting level of any archival
series) keep their PLAN §3 values unless an item produces a range for them; changing one of those is
a PLAN revision, not a memo edit.

---

## Research items

Each item has the same shape. `Sources consulted`, `Finding` and `Conclusion` are the cells WO-000
writes; `Conclusion` must take one of the two permitted forms given in the closing rule at the end
of this file.

### Item 1 - Historical bonus schedules

- **Question (PLAN §15.1).** Notch size relative to base pay, per-percentage-point increments, caps
  (Berliner 1957; Nove; the later reform literature).
- **Parameters.** `notch_height` β, `overfulfilment_slope` s†, `overfulfilment_cap` ρ_cap.
- **What would count as a source.** A published bonus schedule giving the payment at exactly 100%
  fulfilment relative to base pay, the increment per percentage point above it, and the ceiling -
  with the branch and the years it applied to. A range across schedules, not a representative one.
- **Conversion to model units.** β is the normalising unit; s and ρ_cap are expressed relative to it
  in the bonus function of PLAN §2.8. State the conversion explicitly.
- **Sources consulted.** TBD (WO-000)
- **Finding.** TBD (WO-000)
- **Conclusion.** TBD (WO-000)

### Item 2 - Growth directives

- **Question (PLAN §15.2).** Annual plan growth targets by period of Soviet history.
- **Parameters.** `growth_directive` g†.
- **What would count as a source.** Directive annual growth rates for enterprise output targets, by
  plan period and branch, with the spread across branches - not an aggregate outturn, and not a
  realised growth rate.
- **Why it matters.** g is the forcing term of the target rule (PLAN §2.7.1, review finding F1) and
  a treatment variable: with `g = 0` and reports at target the rule has a fixed point (test T-B2).
  Its value changes what the environment *is*, not merely its calibration.
- **Sources consulted.** TBD (WO-000)
- **Finding.** TBD (WO-000)
- **Conclusion.** TBD (WO-000)

### Item 3 - Ratchet

- **Question (PLAN §15.3).** Theoretical (Weitzman 1980) and any empirical estimates of target
  responsiveness; and whether tolerance bands existed.
- **Parameters.** `ratchet_lambda` λ†, `ratchet_deadband` δ.
- **What would count as a source.** An estimate of how next period's target responded to this
  period's measured fulfilment, at enterprise level, with its estimation sample; or documentary
  evidence of an explicit tolerance band around 100% fulfilment.
- **Note.** Theory that constrains only the *sign* and the qualitative mechanism is `PARTIAL`, not
  `SOURCED`: it does not deliver a range for λ.
- **Sources consulted.** TBD (WO-000)
- **Finding.** TBD (WO-000)
- **Conclusion.** TBD (WO-000)

### Item 4 - Inspection frequency and sanction structure

- **Question (PLAN §15.4).** Inspection / audit frequency and the sanction structure for *pripiski*;
  whatever can be said about `a`, `pen`, `penalty_form`.
- **Parameters.** `audit_rate` a†, `penalty_scale` pen†, `penalty_form`.
- **What would count as a source.** Documented inspection frequency per enterprise-year, and the
  sanction schedule actually applied on detection - enough to say whether the sanction scaled with
  the size of the discrepancy or was fixed on detection (`penalty_form`), and what the claim was
  compared against.
- **Note.** In this environment the audit compares the claim to **stock on hand**, not to the
  period's production (PLAN §2.8). If the historical comparison base differs, record that as a
  modelling assumption here; do not adjust a parameter to compensate for it.
- **Sources consulted.** TBD (WO-000)
- **Finding.** TBD (WO-000)
- **Conclusion.** TBD (WO-000)

### Item 5 - Managerial tenure and rotation

- **Question (PLAN §15.5).** Managerial tenure and rotation.
- **Parameters.** `tenure` ψ.
- **What would count as a source.** A distribution of enterprise-director tenure lengths, by period,
  convertible into a per-period continuation probability at this environment's period length. State
  the conversion.
- **Note.** ψ is an economic parameter in the INC arm and is distinct from the technical PPO
  discount γ (PLAN §2.12). A source about discounting is not a source about ψ.
- **Sources consulted.** TBD (WO-000)
- **Finding.** TBD (WO-000)
- **Conclusion.** TBD (WO-000)

### Item 6 - Archival fulfilment distributions, and the level of the Soviet series

- **Question (PLAN §15.6).** Enterprise-level fulfilment distributions in archival work (e.g.
  Harrison's Gosplan-archive studies) - giving the target shape for phenomenon 1 - and the level
  (enterprise / ministry / republic) of the `forensic-stats` Soviet series, which decides what
  PLAN §7.3 can promise.
- **Parameters.** None directly. This item fixes the qualitative target shape for phenomenon 1
  (fulfilment bunching, PLAN §4.1) and the scope of the estimator coupling.
- **What would count as a source.** A published histogram or tabulation of the enterprise fulfilment
  ratio with its bin definition, sample and years; plus a documented statement of the aggregation
  level of each Soviet series available to the coupled repository.
- **Note.** This item may *narrow* PLAN §7.3. If the available series are ministry-level only, say
  so here: the coupling is scoped to estimator robustness and must not promise an enterprise-level
  reconciliation it cannot deliver.
- **Sources consulted.** TBD (WO-000)
- **Finding.** TBD (WO-000)
- **Conclusion.** TBD (WO-000)

### Item 7 - Uzbek cotton affair

- **Question (PLAN §15.7).** Magnitude, level, and mechanism (ministry-level padding vs
  enterprise-level) - and hence whether it is a Phase-2 ministry-layer anchor at all.
- **Parameters.** None directly; decides whether the P2 ministry layer (PLAN §2.14) has an anchor.
- **What would count as a source.** An account that separates the level at which the discrepancy was
  generated from the level at which it was recorded, with a magnitude and a period.
- **Permitted negative answer.** "Not usable as an anchor" is a valid, and possibly the correct,
  conclusion for this item. Record it as such rather than weakening the criterion to keep the anchor.
- **Sources consulted.** TBD (WO-000)
- **Finding.** TBD (WO-000)
- **Conclusion.** TBD (WO-000)

### Item 8 - Tooling: availability, versions, prices

- **Question (PLAN §15.8).** Current availability and versions of: reference PPO implementations,
  JAX MARL scaffolds, `rliable`, open-source MIP solvers; and the current models behind the
  `MID-fast` / `MID-strong` tiers and their prices.
- **Parameters.** None. This item pins the environment of execution, not the economics.
- **What would count as a source.** Per dependency: the exact version to pin, its release date, and
  the interface this repository depends on. Per tier: the model ids the lead maps `MID-fast` /
  `MID-strong` to at issue time, and the price as of a stated date.
- **Where the answer goes.** Pinned versions, model ids, estimator version and solver version are
  recorded in every run manifest (CONTRACT rule 10); the tier mapping is recorded in the manifest at
  work-order issue time (PLAN §12.1). This memo records the decision and its date; the manifest
  records the fact of use.
- **Note.** Availability and prices move. Any figure here carries an as-of date, is a planning input
  only (PLAN §14), and is never reported as a result.
- **Sources consulted.** TBD (WO-000)
- **Finding.** TBD (WO-000)
- **Conclusion.** TBD (WO-000)

---

## Closing rule (PLAN §15)

> Each item ends with either a sourced range or the sentence "unsourced; prior = […]; sensitivity
> handled by regime map / Sobol". No item may end with a point value presented as historical fact.

Applied to this memo:

- The `Conclusion` line of every item above takes exactly one of those two forms. There is no third
  form. "No source found, so the PLAN default stands" is not a conclusion - it is the second form
  with the prior left out, and the prior must be written down.
- The prior in the second form is a distribution or an interval over the sweep range of PLAN §3, not
  a number.
- A sourced range does not become a point value downstream: the G1 selection (PLAN §13) picks the
  operating point from the regime map, and a sourced range only constrains the region it is picked
  from.
- No table, plot or sentence anywhere in this repository may present a parameter value as a
  historical fact on the strength of this memo.
