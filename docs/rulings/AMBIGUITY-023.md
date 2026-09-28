AMBIGUITY NOTES   WO-021 to WO-025 (Phase-2 mechanisms)   raised by the implementer

Where `spec/P2_REVISION.md` (spec 2.0.0) is silent, the simplest reading was chosen and the work
went on. Each item gives the question, the choice and the reason. The LEAD may overrule any of
them. At the Phase-1 values of every toggle, none of them changes a number: golden parity holds.

1. **Which `claim_history` column R2 serves at DELIVER.**
   Choice: `claim_history[:, L-1]` wherever `make_planner_view` runs, as R2 writes it.
   Why: R2 is explicit, and with a `(N, 2)` history and the shift at TARGET, no other index serves
   `L = 2`. Consequence: at the REPORT-step view (target rule, audit), `L = 1` gives the claim of
   period `t-1`. At DELIVER of `t+1` the shift has already happened, so `L = 1` gives the claim of
   period `t`, the same claim `L = 0` gives. Allocation is therefore lagged one period less than the
   ratchet. R2's "the same lagged claims drive ... allocation" holds only if "lag" counts back from
   the view's own period. If the LEAD wants allocation lagged by `L` as well, `claim_history` needs
   a third column or a pre-shift snapshot. Both are spec changes.

2. **Delivery obligations before the first forward, and planner claims that differ from
   obligations.**
   Choice: obligations are `last_report` at `pi = 1` and `ministry_prev` otherwise. At `t = 0`
   under `pi < 1` that is the opening value `T_0`. Stock is 0 then, so `fill = 0` in period 0,
   which feeds R3 complaints and R8 bailouts in period 0. That period is outside the §4.4 window.
   Why: R10 initialises `ministry_prev` to `T_0` as "the ministry's previous forward". Treating it
   as an outstanding promise avoids inventing a separate "no forward yet" state.
   Flag for the LEAD: the rules also move goods when the planner's claims differ from the sellers'
   obligations, which happens under lag, under channel noise, and at `t = 0` under `L > 0, pi = 1`.
   There `alloc` is built from the planner's claims while `poolfill` comes from obligations, so
   buyers receive `alloc * poolfill`, which differs from what sellers ship. This was already true
   of the Phase-1 code under `channel_noise > 0`. R6 ("fill and poolfill computed at DELIVER as
   before") keeps it. Per-good T-U1 conservation therefore holds only when claims equal
   obligations. It is not a defect of this change, but it is a real feature of the C0 configuration.

3. **How the R3, R6 and R8 keys are built.**
   Choice: the trailing index is vectorised through `shape`, as `gosplan/rng.py` states for every
   draw (ambiguity #53):
   - `complaint`: `draw(seed, "complaint", t, shape=(N,))`
   - `bailout`: `draw(seed, "bailout", t, shape=(N,))`
   - `arrival`: `draw(seed, "arrival", t, b, shape=(J,))`
   `trade_visibility` keeps `i` in the key (`(t, i)`), with the permutation of the other `N - 1`
   indices taken as the argsort of `N - 1` standard-normal draws. `draw` has no permutation
   distribution.
   Why: this matches the repository convention, and the audit and channel draws already work this
   way. It also avoids 100 per-cell SeedSequences per period.

4. **Which period's quality is routed at DELIVER.**
   Choice: the previous period's. DELIVER now runs before `reset_period_accumulators` in the same
   agent-step, so `qbar` at DELIVER is `quality_acc / M` of the period whose claims are being
   delivered.
   Why: the reset "at the start of each period" (R5) otherwise always precedes DELIVER. `qbar`
   would then be 0 at every DELIVER, and quality routing would delete every delivery. The reference
   implementation resets before DELIVER, but it is only the Phase-1 oracle (R12).

5. **Claim-weighted `qbar_j` with no claim in the pool.**
   Choice: the plain sector mean of `qbar`.
   Why: the weighted mean is 0/0 there. The plain mean is what the reference implementation uses.

6. **Out-of-box quality actions.**
   Choice: with `quality_matters`, `q` is clipped to `[0, 1]`, its `action_spec` box, before both
   the cost term and the accumulator, exactly as effort is clipped.
   Why: an unclipped `q > 1` would give `qbar > 1` and create input units from nothing. With
   `quality_matters = False` (Phase 1) `q` is untouched.

7. **When step-0 arrivals land, and what "received so far" means.**
   Choice:
   - Step-0 arrivals are credited at DELIVER, before TRADE, as under `uniform`.
   - A step-`k` arrival (`k >= 1`) is credited at the head of step `k`, before PRODUCE.
   - Observation fields 11 and `12+2J:12+3J` count deliveries with `k_bj <= k`, where `k` is the
     step just executed. By the REPORT step, everything has arrived.
   Why: this is R6 applied literally, with the observation describing the step just executed, as
   the Phase-1 observation does.

8. **"The agent's last effort" in the trade surplus (R9.5).**
   Choice: the effort of the step-0 action that carries the trade offers. `match_trades` gained a
   keyword-only `effort` argument. With trade on, omitting it raises rather than guessing.
   Why: no `State` field stores a previous effort, and adding one is a spec change. The step-0
   effort is the latest effort the agent has chosen, and it is the effort of the "next step" that
   `Yhat` is defined on.

9. **`round` in `K = round(hv (N-1))`.**
   Choice: Python's `round` (half to even).
   Why: it is the literal reading. At `N = 20` no locked or registry `hv` value lands on a half.

10. **Logging `h_X` and bailouts.**
    Choice: no `StepRecord` column was added (LEAD: "do not add StepRecord columns"). Bailouts are
    reported as `StepInfo.flags` entries `"bailed_out:<i>"`, which the ledger does not store.
    The `h_X` loss can be read from the ledger: the REPORT row's `inv_inputs` is exactly
    `(1 - h_X)` times the last PRODUCE row's.
    `trade_volume` follows the LEAD convention: the quantity sold (pre-`tau`, summed over goods),
    on the step-0 row, and 0 elsewhere.
    Why: R7 and R8 say "logged" but name no column. If the LEAD wants a `bailed_out` column, that
    is a spec change.

11. **Bonus and ministry details.**
    Choice:
    - The bonus is computed from a planner view whose claims are replaced by the enterprise's own
      `R` (R10.3: "the enterprise's bonus ... use its own R"), as in the reference implementation.
    - `val_measured` keeps using the own `R`.
    - Ministry views carry the target the period was set (before the ratchet).
    - At `pi == 1` the planner reads `last_report` directly, the transparent branch (R10.4).
    - The bailout draw applies to every enterprise with `last_fill < 1`, audited or not. Only an
      audited one has a penalty to forgive.
    Why: these are the literal readings of R8 and R10. Note for WO-026: a non-rule ministry policy
    run under `ministry_passthrough = 1` is bypassed by that transparent branch.

---

## LEAD rulings (2026-09-26)

- **Item 1: accepted.** The lag counts back from the view's own period. R2's "same lagged claims"
  holds in that sense, and allocation is lagged one period less than the ratchet. A third history
  column is not added in Phase 2.
- **Item 2: overruled in part.** Goods must be conserved. `deliver` keeps the planner's allocation as
  each buyer's share, and rescales deliveries per good so buyers receive exactly
  `(1 - phi_j) * shipped_j`.
  - When claims equal obligations, the old formula runs bit for bit, so Phase 1 and the goldens are
    unchanged.
  - A good with shipments but no allocation is still lost, as before. That happens only when no
    buyer needs it.
  - The opening obligation `ministry_prev = T_0` is accepted. Its `fill = 0` in period 0 lies
    outside the t >= 2 window.
  - Test: `test_deliveries_conserve_goods_when_claims_differ_from_obligations`.
- **Items 3-11: accepted** as written.
