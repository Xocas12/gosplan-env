# Reference dynamics - hand-checked worked example (WO-002)

> **FILLED. WO-002 step 3 executed 2026-09-07.** Every numeric cell below was derived from the PLAN
> formulas by an independent re-derivation, then compared against `ref/ref_step.py`. Two defects were
> found and are recorded in section 6: ambiguity **#62** (cold-start deadlock) and ambiguity **#64**
> (the T-U1 identity does not balance). #64 was found *here* and is fixed in this change.

**Work order.** WO-002 Reference dynamics and frozen tests · P1 · LEAD · Difficulty 5 (PLAN §12.3).
**Validate `ref` by hand on a 2-enterprise, 2-sector case - this document is that validation,
committed.**

## Why this document is load-bearing

`ref/ref_step.py` is the **oracle**: `tests/golden` asserts that the production implementation equals
the reference to 1e-9 on seeded trajectories (PLAN §11, T-B7), and `ref/gen_golden.py` generates
those files from it. So no numeric expectation anywhere in the suite is hand-written - which means
**nothing checks the reference except the property tests and this hand computation** (finding F14).

If this document is skipped, or done by running the code and transcribing its output, every golden
file in the repository is confidently wrong in exactly the same way, and the whole frozen suite will
agree with itself forever.

### How this check was actually performed, and its limits

The method the template demands is "derive each quantity by hand from the PLAN formula, before
running `ref/ref_step.py`". What was done, precisely:

1. An **independent re-derivation** was written directly from the PLAN §2.5-§2.11 formulas. It
   imports nothing from `ref/` except `ref_draw`, and only to obtain the stochastic draws - which
   the template explicitly permits as *given inputs*, since they cannot be hand-derived.
2. Every downstream quantity was computed by that re-derivation.
3. `ref/ref_step.py` was then run on the same configuration, and the two compared.
4. The residual is recorded in section 6. **A mismatch was found, and it was fixed in the identity,
   not in the numbers** - see #64.

**Stated limit, so no later reader over-trusts this document.** The re-derivation is independent of
`ref_step.py`'s *code*, but it was written by the same author. It therefore catches transcription
slips, sign errors, ordering errors and formula misreadings - it caught #64 - but it would **not**
catch a misunderstanding of PLAN shared by both. A second party re-deriving section 2 from PLAN
alone remains worth doing, and is the one check this document cannot self-administer.

---

## 1. Configuration for the hand check

`N = 2`, `J = 2`, one enterprise per sector, each sector requiring the other sector's good, so
coverage, delivery and shortage propagation are live in both directions.

| Item | Symbol | PLAN § | Value used | Note |
|---|---|---|---|---|
| Enterprises | `N` | §2.1 | 2 | `s(0) = 0`, `s(1) = 1` (0-indexed, as the code is) |
| Sectors | `J` | §2.1 | 2 | one good per sector |
| Production steps per period | `M` | §2.5 | **2** | the P1 default is 4; 2 is used here so the arithmetic stays followable, as the template permits |
| I-O matrix | `a_{jk}` | §2.10 | `[[0.0, 0.2], [0.2, 0.0]]` | each sector needs 0.2 of the other's good |
| Sector productivity | `A_j` | §2.6 | `[1.0, 1.0]` | |
| Capacity / capital | `cap_i`, `Kap_i` | §2.1 | `[1.0, 1.0]` | P1 holds these fixed |
| Yield noise | `σ_j` | §2.6 | `[0.05, 0.08]` | heteroskedastic by sector |
| Input complementarity | `θ` | §2.6 | 8.0 | E4 repeats the check at `θ = ∞` |
| Final-demand share | `φ_j` | §2.10 | `[0.5, 0.5]` | |
| CES weights / elasticity | `α_j`, `σ_c` | §2.9.3 | `[0.5, 0.5]`, 0.8 | welfare logged only (rule 6) |
| Plan prices | `p_j` | §2.10 | **`[1.4102564102563109, 1.4102564102563109]`** | see below |
| Initial targets | `T_i` | §2.2 | `[0.6, 0.6]` | `0.6 · A · cap` |
| Initial own-good stock | `S_i` | §2.11 | `[0.30, 0.10]` | non-zero, as the template requires |
| Initial input stocks | `X_ij` | §2.11 | `[[0.0, 0.02], [0.03, 0.0]]` | chosen so `H < 1` at k=0 and k=1 for i=0 |
| Initial claims | `claimed_i` | §2.7.2 | `[0.25, 0.15]` | consumed by the first DELIVER |
| Bonus schedule | `β, w, s, ρ_cap` | §2.8 | `1.0, 0.0, 0.5, 1.2` | the **notched** configuration |
| Audit | `a`, `σ_aud` | §2.8 | `0.5, 0.10` | `a = 0.5` chosen so one enterprise is audited and one is not (T-U8); the P1 value is 0.10 |
| Penalty | `pen`, form, arg | §2.8 | `60.0`, proportional, positive_part | E10 repeats under `absolute` |
| Effort cost | `κ` | §2.6 | 0.15 | |
| Target rule | `λ, g, c_up, c_dn, δ` | §2.7.1 | `0.5, 0.02, 0.3, 0.3, 0.0` | `T_min = 0.05 · T_0 = 0.03` |
| Allocation | `η_q`, `η_n` | §2.7.2 | `0.0, 1.0` | at `η_q = 0` requests are inert - shown in §2 step 0 |
| Holding loss / cap | `h`, `S_max` | §2.11 | `0.02`, `3.0` | |
| Report bound | `ρ_max` | §2.3 | 10.0 | |
| Horizon | `ψ, P_min, P_max` | §2.12 | `0.9, 4, 20` | |
| Seed | `seed_env` | §2.15 | **2** | chosen so the audit draw gives one audited, one not |

**Plan prices.** `p_j = (1 + m)(κ_labour + Σ_k a_jk p_k)` with `m = 0.1`, `κ_labour = 1.0`. The matrix
is symmetric here, so `p_0 = p_1 = p` and the fixed point has a closed form:

```
p = (1 + m)·κ_labour / (1 − (1 + m)·0.2) = 1.1 / 0.78 = 1.4102564102564104
```

The iteration converges to `1.4102564102563109`, agreeing with the closed form to 1.0e-13 - the
iteration's own 1e-12 stopping rule, as expected.

**Reward scale.** `B(1.1) = β·1 + s·min(0.1, ρ_cap−1) = 1 + 0.5·0.1 = 1.05`, so
`scale = 1/1.05 = 0.9523809523809523` (T-U2).

### Given draws (`seed_env = 2`), recorded verbatim as inputs

These are the only quantities not hand-derived (PLAN §2.15; keyed, never order-dependent).

| Draw | Key | Value |
|---|---|---|
| `ε[t=0,k=0,i=0]` | `(2,"yield",0,0,0)` | 0.8980091184369003 |
| `ε[t=0,k=0,i=1]` | `(2,"yield",0,0,1)` | 1.0591209875663259 |
| `ε[t=0,k=1,i=0]` | `(2,"yield",0,1,0)` | 1.0866929108975631 |
| `ε[t=0,k=1,i=1]` | `(2,"yield",0,1,1)` | 1.1052027413839804 |
| `audited[t=0]` | `(2,"audit",0)` | `[True, False]` |
| `ν[t=0]` | `(2,"auditnoise",0)` | `[0.0780942747544219, −0.02391539911683151]` |
| `continue[t=0]` | `(2,"terminate",0)` | `[True]` |
| `ε[t=1,k=0,i=0]` | `(2,"yield",1,0,0)` | 1.0242791666459867 |
| `ε[t=1,k=0,i=1]` | `(2,"yield",1,0,1)` | 1.1175690544101837 |
| `ε[t=1,k=1,i=0]` | `(2,"yield",1,1,0)` | 0.9340088681842174 |
| `ε[t=1,k=1,i=1]` | `(2,"yield",1,1,1)` | 0.8933890897793164 |
| `audited[t=1]` | `(2,"audit",1)` | `[True, False]` |

Actions are fixed rather than drawn: effort `[0.8, 0.5]` at `k=0` and `[0.6, 0.7]` at `k=1`; report
ratios `[1.05, 0.90]`. Period 1 uses effort `[0.7, 0.6]`, `[0.5, 0.8]` and reports `[1.00, 1.10]`.

---

## 2. The period schedule (PLAN §2.5) - period 0

### Step 0 - DELIVER (§2.7.2, §2.7.3)

`avail_j = Σ_{i∈j} (1−φ_j)·claimed_i` → `[0.125, 0.075]`.

Allocation weights with `η_q = 0` are `(need_bj + 1e-6)`, so **requests are inert** - `w` does not
contain `q` at all, which is the property the template asks to be shown. `need_bj = planner_io[s(b)][j]·T_b`:

| | good 0 | good 1 |
|---|---|---|
| `need_0j` | 0.0 | 0.12 |
| `need_1j` | 0.12 | 0.0 |
| `alloc_0j` | 1.0416493058449026e-06 | 0.0749993750104165 |
| `alloc_1j` | 0.12499895835069415 | 6.249895835069416e-07 |

Physical delivery:

| Quantity | i = 0 | i = 1 |
|---|---|---|
| `fill_i = min(1, S_i/claimed_i)` | 1.0 | 0.6666666666666667 |
| `shipped_i = min(S_i, claimed_i)` | 0.25 | 0.10 |
| `poolfill_j` | 1.0 | 0.6666666666666667 |
| `deliv_i0` | 1.0416493058449026e-06 | 0.12499895835069415 |
| `deliv_i1` | 0.04999958334027767 | 4.166597223379611e-07 |
| `S_i` after | 0.04999999999999999 | 0.0 |

`consumer_j = Σ_{i∈j} φ_j·shipped_i` → `[0.125, 0.05]`.

Enterprise 1 under-delivers (`fill = 2/3`) because it claimed 0.15 against a stock of 0.10. That
lowers `poolfill` for **good 1**, so enterprise 0 - the only buyer of good 1 - receives 0.04999958
instead of 0.07499938. **Shortage propagates, as a consequence of these lines and not as a rule**
(rule 7).

### Step 1 - TRADE (§2.13, P2)

Absent in Phase 1. `trade_surplus ≡ 0` throughout (§2.9.1).

### Step 2 - PRODUCE, `k = 0 … M−1` (§2.6)

`ŷ_ik = (A_{s(i)}·cap_i/M)·e_ik`; `need_ikj = a_{s(i)j}·ŷ_ik`; `H` is the CES aggregator;
`ỹ = ŷ·H·ε`; inputs consumed `= min(X, a·ỹ)`; `c = κ·e²`.

| k | i | `ŷ` | `H` | `ε` | `y` | `c` |
|---|---|---|---|---|---|---|
| 0 | 0 | 0.4 | 0.8749947917534707 | 0.8980091184369003 | 0.3143013206317654 | 0.096 |
| 0 | 1 | 0.25 | 1.0 | 1.0591209875663259 | 0.26478024689158147 | 0.0375 |
| 1 | 0 | 0.3 | 0.11898865356540986 | 1.0866929108975631 | 0.038791237892033086 | 0.054 |
| 1 | 1 | 0.35 | 1.0 | 1.1052027413839804 | 0.38682095948439316 | 0.0735 |

Two things to read here. Enterprise 1 has `H = 1` at both steps: it holds 0.03 + 0.12499896 of good
0 against a need of `0.2 · 0.25 = 0.05`, so it is never input-constrained. Enterprise 0 is: at `k=1`
it has almost exhausted good 1, and `H` collapses to 0.119, cutting output to 0.0388 despite a
favourable shock. That is the input-coverage channel doing its work.

`cum_y = [0.3530925585237985, 0.6516012063759746]`, `cum_c = [0.15000000000000002, 0.11099999999999999]`.

### Step 3 - REPORT (§2.8)

`S ← (1−h)·S + y`, holding loss on the **carried** stock only, then the `S_max` cap.

| Quantity | i = 0 | i = 1 |
|---|---|---|
| `holding_loss = h·S_before` | 0.0009999999999999998 | 0.0 |
| `cap_overflow` | 0.0 | 0.0 |
| `S` after | 0.4020925585237985 | 0.6516012063759746 |
| `ρ_report` (clipped to `ρ_max`) | 1.05 | 0.90 |
| `R = ρ·T` | 0.63 | 0.54 |

### Step 4 - AUDIT (§2.8)

`Ŝ = S·exp(ν)`; `f = max(0, R − Ŝ)/T` (positive_part); `Pen = pen·f`; gated by `1[audited]`.

| Quantity | i = 0 (audited) | i = 1 (not audited) |
|---|---|---|
| `Ŝ` | 0.4347523601427719 | 0.0 (not measured) |
| `f` | 0.32541273309538016 | 0.0 |
| `penalty` | 19.52476398572281 | 0.0 |

Enterprise 1 is not audited, so its penalty is 0 **regardless of its report** - the second branch of
T-U8. Enterprise 0 claimed 0.63 against a measured stock of 0.4348, so `f = 0.1952/0.6 = 0.3254`.

### Step 5 - REWARD (§2.9)

`r_i = scale·(B(ρ_i) − penalty_i + 0)`.

| Quantity | i = 0 | i = 1 |
|---|---|---|
| `ρ` | 1.05 | 0.9000000000000001 |
| `B(ρ)` | 1.025 | 0.0 |
| `reward` | −17.618822843545534 | 0.0 |

`B(0.9) = 0` exactly: below target the Heaviside is 0 and the overfulfilment term is 0. Enterprise 1
receives precisely nothing at the REPORT step - the notch is the whole story at `ρ < 1`.

Logged (rule 6 - never observed, never in a reward):

```
val_measured = 1.6499999999998838
val_true     = 1.416875822294452
welfare      = 0.07701408984357115
```

`val_measured > val_true`: `padding_index = 1.1645`. Enterprise 0 claimed 0.63 having produced
0.3531. **This is padding, and it is an outcome of the chosen report, not of any rule** (rule 7).

### Step 6 - TARGET (§2.7.1)

`step = clip(ρ−1, −c_dn, c_up)`; deadband `δ = 0` inert; `T ← max(T_min, (1+g)·T·(1+λ·step))`.

| | i = 0 | i = 1 |
|---|---|---|
| `ρ − 1` | +0.05 | −0.10 |
| `step` (capped at ±0.3) | +0.05 | −0.10 |
| `T'` | 0.6273 | 0.5814 |

Check by hand: `1.02 · 0.6 · (1 + 0.5·0.05) = 1.02·0.6·1.025 = 0.6273` ✓ and
`1.02 · 0.6 · (1 − 0.05) = 1.02·0.6·0.95 = 0.5814` ✓. The floor `T_min = 0.03` does not bind.

**T-U4 fixed point.** At `ρ = 1` and `g = 0` the map gives `T ← max(T_min, 1·T·1) = T`. At `ρ = 1`
and `g = 0.02` it gives exactly `1.02·T`, which is test T-B2.

### Step 7 - TERMINATE (§2.12)

Period 0 completed is `t + 1 = 1 < P_min = 4`, so the episode continues **without a draw** being
consulted. `done = False`. The `(2,"terminate",0)` draw is recorded above but not used at this
period - keying means recording it costs nothing and it stays reproducible.

---

## 3. Period conservation ledger (T-U1)

**This is where the check earned its keep.** The identity as PLAN §11 states it,

```
Σ y + Σ S_prev = Σ inputs consumed + Σ consumer + Σ S_next + holding loss + cap overflow
```

**does not balance**. The independent re-derivation and `ref_step.py` agreed on residuals of

```
[−0.005320241275194926, −0.019999583340277738]
```

and the agreement is what localised the fault to the **identity**, not to either implementation. The
omission: a unit delivered into a buyer's `X` has left the seller's `S` but has not been consumed,
so it appears on neither side, and the residual is exactly the period's change in `X`.

For good 0, using **shipped** instead:

```
LHS  y + S_prev                 = 0.653092558524
RHS  S_next + holding + shipped = 0.653092558524
     residual                   = 0.000e+00
consumer + delivered            = 0.250000000000   (== shipped)
```

The corrected identity, per good `j`, carries the input stocks explicitly:

```
Σ_i y_i + Σ_i S_prev_i + Σ_i X_prev_ij
    = Σ_i S_next_i + Σ_i X_next_ij + Σ_i consumed_ij
      + consumer_j + Σ_i holding_i + Σ_i overflow_i
```

Substituting `X_next = X_prev + deliv − consumed` reduces it to
`y + S_prev = S_next + deliv + consumer + holding + overflow`, and `deliv_j + consumer_j` is exactly
`Σ_i shipped_i` for that good - which is why it balances.

Residual under the corrected identity:

| Period | good 0 | good 1 |
|---|---|---|
| 0 | 0.0 | −1.1102230246251565e-16 |
| 1 | 0.0 | 0.0 |

Both are far below the 1e-9 tolerance; the `1.1e-16` is one unit in the last place of a double.

Filed as ambiguity **#64** and fixed in `ref_conservation_residual` in this change, recorded in
`spec/CHANGELOG.md` 0.1.3.

---

## 4. Two periods, not one

A second period is required so that the target written at stage 6 and the claim recorded at stage 3
are both carried across the boundary.

Period 1 opens with `T = [0.6273, 0.5814]` from period 0's ratchet and `claims = [0.63, 0.54]` from
period 0's report - which is where the padding bites:

| Quantity | i = 0 | i = 1 |
|---|---|---|
| `fill` | 0.6382421563869818 | 1.0 |
| `poolfill` | 0.6382421563869818 | 1.0 |
| `cum_y` | 0.5919999253721497 | 0.6926263522347818 |
| `R` | 0.6273 | 0.6395400000000001 |
| `audited` | True | False |
| `penalty` | 12.281924643531683 | 0.0 |
| `reward` | −10.74469013669684 | 1.0 |
| `T'` | 0.639846 | 0.6226794 |

```
val_measured = 1.7865692307691048
val_true     = 1.8116524427788783
welfare      = 0.2323539168312209
```

Two carries are visible. Enterprise 0's period-0 claim of 0.63 exceeded the 0.4021 it actually held,
so its `fill` in period 1 is 0.638 and **good 0's `poolfill` falls to 0.638 for every buyer** - the
padding-to-shortage channel, one period later. And enterprise 1, which reported 1.10 against a
target the ratchet had *lowered* to 0.5814, is rewarded `scale·B(1.1) = 0.95238·1.05 = 1.0` exactly:
the reward-scale normalisation of T-U2, visible as a round number.

---

## 5. Edge cases (one hand computation each)

| # | Case | PLAN § | Expected | Got |
|---|---|---|---|---|
| E1 | `fill` when `claimed = 0` | §2.7.3 | 1.0 | 1.0 |
| E2 | coverage, whole `a` row zero | §2.6 | 1.0 | 1.0 |
| E3 | observation coverage field at `need = 0` | §2.4 | 1.0 | 1.0 (AMB-007) |
| E4 | coverage at `θ = ∞` equals `min` | §2.6 | 0.1 | 0.09999999999999999 |
| E5 | coverage, `X = 0` with `need > 0`, finite `θ` | §2.6 | 0.0, must not raise | 0.0 |
| E7 | `B(ρ)` at exactly `ρ = 1` (strict `≥`) | §2.8 | 1.0 | 1.0 |
| E8 | `B(ρ)` at `ρ = 1.5`, capped at `ρ_cap = 1.2` | §2.8 | 1.1 | 1.1 |
| E9 | penalty `positive_part` on an under-report | §2.8 | 0.0 | 0.0 |
| E10 | penalty `absolute` on the same under-report | §2.8 | 0.5 | 0.5000000000000001 |
| E12 | overflow above `S_max = 3` from a stock of 4 | §2.11 | 1.0 | 1.0 |

E9 and E10 are the two branches of T-U8 on the same numbers: a claim of 0.5 against a measured 0.8
with `T = 0.6` gives `f = 0` under `positive_part` and `f = 0.5` under `absolute`. Under-reporting is
free in Phase 1 - which is precisely the mechanism behind the held-out hidden-reserves phenomenon,
and is why `penalty_arg` is a treatment parameter (§4.2).

---

## 6. Sign-off

**Residual between the independent re-derivation and `ref/ref_step.py`: exact agreement** on every
quantity in sections 2-5, at full double precision. Spot values, re-derivation vs reference:

| Quantity | Re-derivation | `ref_step.py` |
|---|---|---|
| `reward` (P0) | −17.618822843545534, 0.0 | −17.618822844, 0.0 |
| `val_measured` (P0) | 1.6499999999998838 | 1.650000000 |
| `val_true` (P0) | 1.416875822294452 | 1.416875822 |
| `welfare` (P0) | 0.07701408984357115 | 0.077014090 |
| `T'` (P0) | 0.6273, 0.5814 | 0.6273, 0.5814 |

**Defects found by this exercise, both filed rather than papered over:**

- **#64** - the T-U1 conservation identity does not balance. Found here, fixed here.
- **#62** - the Phase-1 economy is deadlocked at zero from a cold start. Found in step 2, and
  **resolved** (`spec/CHANGELOG.md` 0.1.4): reset now endows one period's input need at target,
  `X_ij = a_{s(i)j} * T_0_i`. A 30-step rollout that reported `val_true = 0.000000` in every period
  now reports 1.698735 in period 0. This worked example is unaffected - it sets its opening stocks
  by hand and never went through `reset`.

**Status.** The reference is validated for the dynamics exercised above. It is **not** yet validated
for: `report_lag > 0`, `aggregation_level = "sector"`, `channel_noise > 0`, `self_obs_noise > 0`,
`objective_metric` other than `val`, the `fixed` horizon mode, or any Phase-2 toggle. Those branches
exist in the code, are unexercised here, and should not be trusted until a case covers them.

**Approver.** LEAD, 2026-09-07. Not a substitute for the second-party re-derivation noted at the top.
