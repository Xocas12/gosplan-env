# Reference dynamics - hand-checked worked example (WO-002)

> **TEMPLATE - UNFILLED. WO-002 HAS NOT BEEN EXECUTED.**
> Every numeric cell below reads `TBD (WO-002)`. No number in this file has been computed, and none
> may be invented to fill a gap: a cell that cannot be derived from the formulas and the recorded
> draws is an AMBIGUITY REPORT (CONTRACT rule 3), not a guess.

**Work order.** WO-002 Reference dynamics and frozen tests · P1 · LEAD · Difficulty 5 (PLAN §12.3).
Writes `ref/ref_step.py` (PLAN §2.5-§2.11, pure Python, loops, `N ≤ 4`), `ref/gen_golden.py`, and all
of `tests/unit`, `tests/behavioural`, `tests/golden`. **Validate `ref` by hand on a 2-enterprise,
2-sector case - this document is that validation, committed.** Forbidden: any use of `gosplan/` from
`ref/`.

## Why this document is load-bearing

`ref/ref_step.py` is the **oracle**: `tests/golden` asserts that the production implementation equals
the reference to 1e-9 on seeded trajectories (PLAN §11, T-B7), and `ref/gen_golden.py` generates
those files from it (5 configs × 3 seeds × 30 agent-steps, with `Random` and `TruthfulMyopic`). So no
numeric expectation anywhere in the suite is hand-written - which means **nothing checks the
reference except the property tests and this hand computation** (PLAN §11, review finding F14).

This document is therefore the bottom of the trust chain. If it is skipped, or done by running the
code and transcribing its output, every golden file in the repository is confidently wrong in exactly
the same way, and the whole frozen suite will agree with itself forever.

**Method, non-negotiable.**

1. Derive each quantity **by hand from the PLAN formula**, before running `ref/ref_step.py`.
2. The stochastic draws (yield `ε`, audit selection, audit noise `ν`) cannot be hand-derived. Take
   them from the reference's own key-based RNG (PLAN §2.15: keyed on `(seed_env, purpose, indices)`,
   never on call order - T-U6), record them verbatim in the `draw` rows below as **given inputs**,
   and hand-compute everything downstream of them.
3. Run the reference, compare, and record the residual.
4. A mismatch is a defect report against `ref/ref_step.py` or against this hand computation. It is
   **never** fixed by editing the numbers here to agree with the code, and no golden file is
   generated until the residual is clean.

---

## 1. Configuration for the hand check

The case is deliberately the smallest one that exercises every branch: `N = 2` enterprises,
`J = 2` sectors, one enterprise per sector, each sector requiring the other sector's good as an
input (so coverage, delivery and shortage propagation are all live in both directions).

| Item | Symbol | PLAN § | Value used | Note |
|---|---|---|---|---|
| Enterprises | `N` | §2.1 | 2 | fixed by this exercise |
| Sectors | `J` | §2.1 | 2 | fixed by this exercise; `s(1) = 1`, `s(2) = 2` |
| Production steps per period | `M` | §2.1, §2.5 | TBD (WO-002) | state which value is used; the P1 default is in PLAN §3 |
| I-O matrix | `a_{jk}` | §2.10 | TBD (WO-002) | 2×2; both off-diagonal entries non-zero, so each sector depends on the other |
| Sector productivity | `A_j` | §2.6 | TBD (WO-002) | |
| Capacity / capital | `cap_i`, `Kap_i` | §2.1 | TBD (WO-002) | P1 holds these fixed |
| Yield noise | `σ_j` | §2.6 | TBD (WO-002) | heteroskedastic by sector |
| Input complementarity | `θ` | §2.6 | TBD (WO-002) | run the check at the configured value **and** at `θ = ∞` (T-U7) |
| Final-demand share | `φ_j` | §2.10 | TBD (WO-002) | |
| CES consumer weights / elasticity | `α_j`, `σ_c` | §2.9.3, §2.10 | TBD (WO-002) | welfare is logged only (CONTRACT rule 6) |
| Plan prices | `p_j` | §2.10 | TBD (WO-002) | cost-plus fixed point solved at `t = 0` |
| Initial targets | `T_i` | §2.2 | TBD (WO-002) | |
| Initial own-good stock | `S_i` | §2.11 | TBD (WO-002) | non-zero for at least one enterprise |
| Initial input stocks | `X_ij` | §2.11 | TBD (WO-002) | chosen so `H < 1` for at least one enterprise-step |
| Initial claims (period 1 DELIVER) | `claimed_i` | §2.7.2 | TBD (WO-002) | the initial condition consumed by the first DELIVER |
| Bonus schedule | `β`, `w`, `s`, `ρ_cap` | §2.8 | TBD (WO-002) | run at the notched configuration; repeat the REWARD table at `w > 0` for T-U3 |
| Audit | `a`, `σ_aud` | §2.8 | TBD (WO-002) | must yield one audited and one non-audited enterprise (T-U8) |
| Penalty | `pen`, `penalty_form`, `penalty_arg` | §2.8 | TBD (WO-002) | repeat the AUDIT table under `absolute` for T-U8 |
| Effort cost | `κ` | §2.6 | TBD (WO-002) | |
| Target rule | `λ`, `g`, `c_up`, `c_dn`, `δ`, `T_min` | §2.7.1 | TBD (WO-002) | run once at the configured `g` and once at `g = 0` for the fixed point (T-U4) |
| Allocation | `η_q`, `η_n` | §2.7.2 | TBD (WO-002) | at the P1 value of `η_q` the requests are inert; show that they are |
| Holding loss / stock cap | `h`, `S_max` | §2.11 | TBD (WO-002) | |
| Report bound | `ρ_max` | §2.3 | TBD (WO-002) | one edge case below sits at the bound |
| Termination | `horizon_mode`, `ψ`, `P_min`, `P_max` | §2.12 | TBD (WO-002) | |
| Environment seed | `seed_env` | §2.15 | TBD (WO-002) | recorded so the check is reproducible |

**Actions.** The hand check is driven by a **fixed, hand-written action sequence**, not by a learned
policy: effort per enterprise per step, and one `report_ratio` and `input_request` per enterprise per
period. Record the full sequence here.

| Action | Symbol | PLAN § | Enterprise 1 | Enterprise 2 |
|---|---|---|---|---|
| Effort, per step | `e_ik` | §2.3, §2.6 | TBD (WO-002) | TBD (WO-002) |
| Report ratio | `ρ_i^report` | §2.3, §2.8 | TBD (WO-002) | TBD (WO-002) |
| Input request | `q_ij` | §2.3, §2.7.2 | TBD (WO-002) | TBD (WO-002) |

**Branch coverage requirement.** The chosen actions must make one enterprise's claim exceed its stock
on hand and the other's fall below it, in the same period. Both branches of PLAN §2.7.3 (`fill < 1`
with the resulting `poolfill` reduction downstream, and stock left behind in `S`) are transition
rules that must be exercised. Exercising a branch is not asserting a direction: this document states
no expectation about which behaviour an agent adopts, and CONTRACT rule 7 forbids any rule that
implements one.

---

## 2. The period schedule (PLAN §2.5)

One table per step of the schedule. Each row is a quantity the lead computes by hand; the last column
names the invariant or test id that the row pins. Every value cell is `TBD (WO-002)`.

```
period t:
  0. DELIVER      1. [P2] TRADE      2. PRODUCE ×M      3. REPORT
  4. AUDIT        5. REWARD          6. TARGET          7. TERMINATE?
```

### Step 0 - DELIVER (PLAN §2.7.2, §2.7.3)

Allocates from **last period's** claims; in period 1 that is the recorded initial condition.

| # | Quantity | Formula / PLAN § | Enterprise 1 | Enterprise 2 | Pins |
|---|---|---|---|---|---|
| 0.1 | Claim entering the planner | `claimed_i` (§2.7.2), lagged / noised as configured | TBD (WO-002) | TBD (WO-002) | planner sees claims, never stock - T-B4, CONTRACT rule 5 |
| 0.2 | Believed availability per good | `avail_j = Σ_{i∈j} (1 − φ_j)·claimed_i` (§2.7.2) | TBD (WO-002) | TBD (WO-002) | allocation sums to `avail_j` (`test_planner`) |
| 0.3 | Planned need per buyer-good | `need_bj = planner_io[s(b), j] · T_b` (§2.7.2) | TBD (WO-002) | TBD (WO-002) | planner uses its own I-O copy, not `a` |
| 0.4 | Allocation weight | `w_bj = (q_bj + 1e-6)^η_q · (need_bj + 1e-6)^η_n` (§2.7.2) | TBD (WO-002) | TBD (WO-002) | requests inert at the P1 `η_q` (`test_planner`) |
| 0.5 | Promise, in claimed units | `alloc_bj = avail_j · w_bj / Σ_b w_bj` (§2.7.2) | TBD (WO-002) | TBD (WO-002) | Σ over buyers = `avail_j` |
| 0.6 | Fill of own claim | `fill_i = min(1, S_i / claimed_i)`; `= 1` if `claimed_i = 0` (§2.7.3) | TBD (WO-002) | TBD (WO-002) | guard at `claimed_i = 0` (`test_planner`) |
| 0.7 | Physically shipped | `shipped_i = min(S_i, claimed_i)` (§2.7.3) | TBD (WO-002) | TBD (WO-002) | T-U1 |
| 0.8 | Pool fill per good | `poolfill_j = Σ fill_i·claimed_i / Σ claimed_i` (§2.7.3) | TBD (WO-002) | TBD (WO-002) | `poolfill ∈ [0,1]` (`test_planner`) |
| 0.9 | Physical receipt | `deliv_bj = alloc_bj · poolfill_j` (§2.7.3) | TBD (WO-002) | TBD (WO-002) | delivery conservation, T-U1 |
| 0.10 | Input stock after receipt | `X_bj += deliv_bj · q̄_j` (§2.7.3) | TBD (WO-002) | TBD (WO-002) | T-U1 |
| 0.11 | Consumer sink | `consumer_j = Σ_{i∈j} φ_j · shipped_i · q̄_i` (§2.7.3) | TBD (WO-002) | TBD (WO-002) | T-U1; feeds welfare at step 5 |
| 0.12 | Own stock after shipment | `S_i −= shipped_i` (§2.7.3) | TBD (WO-002) | TBD (WO-002) | T-U1 |
| 0.13 | Observed fill field | `last_fill` (§2.4 index 10) | TBD (WO-002) | TBD (WO-002) | own fill only; no other enterprise's quantity - T-B5 |

### Step 1 - TRADE (PLAN §2.13, P2)

| # | Quantity | Formula / PLAN § | Value | Pins |
|---|---|---|---|---|
| 1.1 | Bilateral matching | §2.13 | not exercised | P2 sketch; behaviour frozen only at the P2 spec revision. `trade_surplus ≡ 0` in P1 (§2.9.1) |

### Step 2 - PRODUCE, repeated for `k = 0 … M−1` (PLAN §2.6)

**Reproduce this table once per production step `k`.** Step index: `k = TBD (WO-002)`.

| # | Quantity | Formula / PLAN § | Enterprise 1 | Enterprise 2 | Pins |
|---|---|---|---|---|---|
| 2.1 | Effort (action) | `e_ik ∈ [0,1]` (§2.3) | TBD (WO-002) | TBD (WO-002) | given by the action sequence |
| 2.2 | Intended output | `ŷ_ik = (A_{s(i)} · cap_i / M) · e_ik` (§2.6) | TBD (WO-002) | TBD (WO-002) | |
| 2.3 | Input need | `need_ikj = a_{s(i)j} · ŷ_ik` (§2.6) | TBD (WO-002) | TBD (WO-002) | `need = 0` guard (T-U7) |
| 2.4 | Coverage weights | `ω_j = a_{s(i)j} / Σ_j a_{s(i)j}` (§2.6) | TBD (WO-002) | TBD (WO-002) | Σ ω = 1 |
| 2.5 | Coverage ratios | `min(1, X_ij / need_ikj)` (§2.6) | TBD (WO-002) | TBD (WO-002) | clipped at 1 |
| 2.6 | CES coverage aggregate | `H_ik = (Σ_j ω_j · min(1, X_ij/need_ikj)^(−θ))^(−1/θ)`; `H = 1` if no inputs needed (§2.6) | TBD (WO-002) | TBD (WO-002) | **T-U7**: `θ = ∞` equals `min`; `θ → 1` equals the weighted harmonic mean; `H = 1` on a zero `a` row |
| 2.7 | Yield draw (**given**) | `ε_ik ~ LogNormal(−σ²/2, σ)`, key `(seed_env, "yield", t, k, i)` (§2.6) | TBD (WO-002) | TBD (WO-002) | **T-U6**: deterministic in key, independent of call order; mean 1 |
| 2.8 | Realised output before diversion | `ỹ_ik = ŷ_ik · H_ik · ε_ik` (§2.6) | TBD (WO-002) | TBD (WO-002) | |
| 2.9 | Investment diversion | `v_ik` (P1: `v ≡ 0`) (§2.6) | TBD (WO-002) | TBD (WO-002) | P2 toggle; show it is inert |
| 2.10 | Output to stock | `y_ik = ỹ_ik · (1 − v_ik)` (§2.6) | TBD (WO-002) | TBD (WO-002) | T-U1 |
| 2.11 | Inputs consumed | `min(X_ij, a_{s(i)j} · ỹ_ik)`, subtracted from `X_ij` (§2.6) | TBD (WO-002) | TBD (WO-002) | **T-U1**; consumption capped at stock |
| 2.12 | Input stock after | `X_ij` (§2.11) | TBD (WO-002) | TBD (WO-002) | T-U1 |
| 2.13 | Step cost | `c_ik = κ·e_ik² + F·1[e_ik > 0] + κ_q·q_ik·e_ik` (P1: `F = κ_q = 0`) (§2.6) | TBD (WO-002) | TBD (WO-002) | |
| 2.14 | Cumulative output | `cum_output` (§2.2), observed as `cum_output / T_i` (§2.4 index 4) | TBD (WO-002) | TBD (WO-002) | own quantity only - T-B5 |
| 2.15 | Production-step reward | `r_ik = − scale · c_ik` (§2.9.1) | TBD (WO-002) | TBD (WO-002) | **CONTRACT rule 4**: this is the entire production-step reward; T-B6 |

### Step 3 - REPORT (PLAN §2.8)

| # | Quantity | Formula / PLAN § | Enterprise 1 | Enterprise 2 | Pins |
|---|---|---|---|---|---|
| 3.1 | Stock before carry | `S_i` entering the step (§2.11) | TBD (WO-002) | TBD (WO-002) | |
| 3.2 | Holding loss on carried stock | `(1 − h) · S_i`, **then** add `y_i` (§2.8) | TBD (WO-002) | TBD (WO-002) | order matters (`test_reporting`); the loss is a T-U1 term |
| 3.3 | Stock after carry | `S_i ← (1 − h)·S_i + y_i` (§2.8) | TBD (WO-002) | TBD (WO-002) | T-U1 |
| 3.4 | Stock-cap overflow | excess above `S_max` is lost and logged (§2.11) | TBD (WO-002) | TBD (WO-002) | T-U1 (`cap overflow` term) |
| 3.5 | Report ratio (action) | `ρ_i^report ∈ [0, ρ_max]` (§2.3) | TBD (WO-002) | TBD (WO-002) | agent has observed `S_i` and `y_i` exactly at P1 |
| 3.6 | Reported quantity | `R_i = clip(ρ_i^report, 0, ρ_max) · T_i` (§2.8) | TBD (WO-002) | TBD (WO-002) | bound is a result, never widened - CONTRACT rule 8, T-B8 |
| 3.7 | Input request | `q_ij ∈ [0, r_max · need]` (§2.3) | TBD (WO-002) | TBD (WO-002) | logged; inert at the P1 `η_q` |
| 3.8 | At-bound flag | fraction of reports at `ρ_max` (§2.3, CONTRACT rule 8) | TBD (WO-002) | TBD (WO-002) | `BOUND_BINDING` logic - T-B8 |

### Step 4 - AUDIT (PLAN §2.8)

The audit compares the claim to **stock on hand**, not to the period's production.

| # | Quantity | Formula / PLAN § | Enterprise 1 | Enterprise 2 | Pins |
|---|---|---|---|---|---|
| 4.1 | Audit selection (**given**) | `audited_i ~ Bernoulli(a)`, key `(seed_env, "audit", t, i)` (§2.7.4) | TBD (WO-002) | TBD (WO-002) | T-U6; one audited and one not |
| 4.2 | Audit noise draw (**given**) | `ν_i ~ N(0, σ_aud²)`, key `(seed_env, "auditnoise", t, i)` (§2.8) | TBD (WO-002) | TBD (WO-002) | T-U6 |
| 4.3 | Measured stock | `Ŝ_i = S_i · exp(ν_i)` (§2.8) | TBD (WO-002) | TBD (WO-002) | measurement, not truth |
| 4.4 | Discrepancy in ratio units | `f_i = max(0, R_i − Ŝ_i)/T_i` (`positive_part`) or `|R_i − Ŝ_i|/T_i` (`absolute`) (§2.8) | TBD (WO-002) | TBD (WO-002) | **T-U8**: `positive_part` gives 0 for any under-report, `absolute` does not |
| 4.5 | Penalty magnitude | `Pen_i = pen · f_i` (proportional) or `pen · 1[f_i > 0]` (fixed) (§2.8) | TBD (WO-002) | TBD (WO-002) | T-U8 |
| 4.6 | Penalty applied | `penalty_i = 1[audited_i] · Pen_i` (§2.8) | TBD (WO-002) | TBD (WO-002) | **T-U8**: `audited = False` gives 0 |

### Step 5 - REWARD (PLAN §2.9)

| # | Quantity | Formula / PLAN § | Enterprise 1 | Enterprise 2 | Pins |
|---|---|---|---|---|---|
| 5.1 | Fulfilment measure | `m_i` per `objective_metric` (§2.9.2) | TBD (WO-002) | TBD (WO-002) | `welfare` is not an option (§2.9.2) |
| 5.2 | Fulfilment ratio | `ρ_i = m_i / T_i` (§2.7.1) | TBD (WO-002) | TBD (WO-002) | |
| 5.3 | Notch factor | `Λ_w(x) = 1[x ≥ 0]` if `w = 0`, else `1/(1 + exp(−x/w))` (§2.8) | TBD (WO-002) | TBD (WO-002) | **T-U3**: discontinuous at 1 iff `w = 0`; strict `≥` Heaviside |
| 5.4 | Bonus | `B(ρ) = β·Λ_w(ρ − 1) + s·clip(ρ − 1, 0, ρ_cap − 1)` (§2.8) | TBD (WO-002) | TBD (WO-002) | **T-U3**: monotone in ρ; `ρ_cap = ∞` means no clip and no kink |
| 5.5 | Reward scale | `scale = 1 / B_cfg(1.1)`, analytic from the config (§2.9.1) | TBD (WO-002) | TBD (WO-002) | **T-U2**: `reward_scale(cfg) · B_cfg(1.1) == 1`; never from running statistics (CONTRACT rule 4) |
| 5.6 | Report-step reward | `r_i = scale · [B(ρ_i) − penalty_i + trade_surplus_i]`, `trade_surplus ≡ 0` in P1 (§2.9.1) | TBD (WO-002) | TBD (WO-002) | **CONTRACT rule 4** - these are the only terms; T-B6 recomputes them independently |
| 5.7 | Planner-side value (logged) | `val_measured = Σ_i p_{s(i)}·R_i·q̂_i` (§2.9.3) | TBD (WO-002) | TBD (WO-002) | **CONTRACT rule 6**: logged, never observed |
| 5.8 | True value (logged) | `val_true = Σ_i p_{s(i)}·y_i·q̄_i` (§2.9.3) | TBD (WO-002) | TBD (WO-002) | CONTRACT rule 6 |
| 5.9 | Consumer welfare (logged) | `welfare_t = (Σ_j α_j · consumer_j^((σ_c−1)/σ_c))^(σ_c/(σ_c−1))` (§2.9.3) | TBD (WO-002) | TBD (WO-002) | CONTRACT rule 6; T-B5 sentinel test |
| 5.10 | Headline metrics (logged) | `padding_index`, `welfare_ratio`, `specification_gap` (§2.9.4) | TBD (WO-002) | TBD (WO-002) | dimensionless; recomputed under perturbed price vectors (§2.9.4, §7.5) |

### Step 6 - TARGET (PLAN §2.7.1)

| # | Quantity | Formula / PLAN § | Enterprise 1 | Enterprise 2 | Pins |
|---|---|---|---|---|---|
| 6.1 | Measure entering the rule | `m_i` from `report_lag` periods ago (P1 lag is the identity) (§2.7.1) | TBD (WO-002) | TBD (WO-002) | lag branch present and exercised as identity |
| 6.2 | Raw step | `ρ_i − 1` (§2.7.1) | TBD (WO-002) | TBD (WO-002) | |
| 6.3 | Capped step | `step_i = clip(ρ_i − 1, −c_dn, +c_up)` (§2.7.1) | TBD (WO-002) | TBD (WO-002) | **T-U4**: step bounded by the caps |
| 6.4 | Deadband | `step_i = 0` if `|ρ_i − 1| ≤ δ` (§2.7.1) | TBD (WO-002) | TBD (WO-002) | **T-U4**: inert outside `|ρ−1| ≤ δ` |
| 6.5 | New target | `T_i ← max(T_min, (1 + g)·T_i·(1 + λ·step_i))` (§2.7.1) | TBD (WO-002) | TBD (WO-002) | **T-U4**: floor respected |
| 6.6 | Fixed point, `g = 0` | rerun 6.1-6.5 with `g = 0` and `ρ_i = 1` (§2.7.1) | TBD (WO-002) | TBD (WO-002) | **T-U4**: `T` unchanged - the property T-B2 rests on |
| 6.7 | Growth at `g > 0`, `ρ = 1` | `T` grows at exactly `(1 + g)` (§2.7.1) | TBD (WO-002) | TBD (WO-002) | T-B2 |

### Step 7 - TERMINATE (PLAN §2.12)

| # | Quantity | Formula / PLAN § | Value | Pins |
|---|---|---|---|---|
| 7.1 | Continuation draw (**given**) | geometric mode: continue with probability `ψ` after `P_min` (§2.12) | TBD (WO-002) | T-U6 |
| 7.2 | Alive flag | `alive` after the draw (§2.2) | TBD (WO-002) | **T-B9**: empirical continuation ≈ ψ |
| 7.3 | Hard cap | episode ends at `P_max` (§2.12) | TBD (WO-002) | T-B9 |
| 7.4 | Periods remaining in the observation | absent by construction (§2.4) | not present | **T-B5 / T-B9**: no observation field correlates with periods remaining |

---

## 3. Period conservation ledger (T-U1)

Computed per good, per period, after step 7. The identity is stated in PLAN §11 as:

> `Σ y + Σ S_prev = Σ inputs consumed/ a + Σ consumer + Σ S_next + holding loss + cap overflow`,
> per good, to 1e-9.

| Term | Source row above | Good 1 | Good 2 |
|---|---|---|---|
| `Σ y` (period output) | 2.10 | TBD (WO-002) | TBD (WO-002) |
| `Σ S_prev` (stock entering) | 0.12 / 3.1 | TBD (WO-002) | TBD (WO-002) |
| `Σ` inputs consumed | 2.11 | TBD (WO-002) | TBD (WO-002) |
| `Σ consumer` | 0.11 | TBD (WO-002) | TBD (WO-002) |
| `Σ S_next` (stock leaving) | 3.3 | TBD (WO-002) | TBD (WO-002) |
| holding loss | 3.2 | TBD (WO-002) | TBD (WO-002) |
| cap overflow | 3.4 | TBD (WO-002) | TBD (WO-002) |
| **residual** | LHS − RHS | TBD (WO-002) | TBD (WO-002) |

The residual must be below 1e-9. A non-zero residual is a defect in the reference, not a tolerance to
be loosened.

---

## 4. Two periods, not one

The check runs at least **two** periods. Period 1 exercises steps 0-7 with the recorded initial
condition. Period 2 is what makes the ratchet observable: the target written at step 6 of period 1 is
the target every ratio in period 2 is measured against, and the claim reported at step 3 of period 1
is the claim DELIVER consumes at step 0 of period 2. A single-period check cannot pin T-U4 or
distinguish a target rule that is applied from one that is computed and discarded.

| Period | What it establishes | Value |
|---|---|---|
| 1 | Full schedule; conservation; reward terms; audit branches | TBD (WO-002) |
| 2 | Target carried forward and applied; claim carried forward into DELIVER; conservation across the boundary | TBD (WO-002) |

---

## 5. Edge cases (one hand computation each)

Each row is computed by hand from the same configuration with the single stated change.

| # | Edge case | PLAN § | Expected structural behaviour | Value | Pins |
|---|---|---|---|---|---|
| E1 | `claimed_i = 0` | §2.7.3 | `fill_i = 1` by the guard | TBD (WO-002) | `test_planner` |
| E2 | Zero `a` row (no inputs needed) | §2.6 | `H = 1` | TBD (WO-002) | T-U7 |
| E3 | `need = 0` in the observation | §2.4 | coverage field is 1.0 | TBD (WO-002) | `test_obs` |
| E4 | `θ = ∞` | §2.6 | aggregator equals `min` over coverage ratios | TBD (WO-002) | T-U7 |
| E5 | `θ → 1` | §2.6 | aggregator equals the weighted harmonic mean | TBD (WO-002) | T-U7 |
| E6 | Report at the bound `ρ_max` | §2.3, CONTRACT rule 8 | clipped at the bound and flagged, never silently re-bounded | TBD (WO-002) | T-B8 |
| E7 | `w = 0` at exactly `ρ = 1` | §2.8 | strict `≥` Heaviside - the bonus is paid at exactly 1 | TBD (WO-002) | T-U3 |
| E8 | `ρ_cap = ∞` | §2.8 | no clip and no kink anywhere | TBD (WO-002) | T-U3 |
| E9 | Under-report under `positive_part` | §2.8 | `f = 0` | TBD (WO-002) | T-U8 |
| E10 | Same under-report under `absolute` | §2.8 | `f > 0` | TBD (WO-002) | T-U8 |
| E11 | Not audited, large discrepancy | §2.8 | `penalty = 0` | TBD (WO-002) | T-U8 |
| E12 | Stock driven above `S_max` | §2.11 | excess lost and logged | TBD (WO-002) | T-U1 |
| E13 | Enterprise relabelling within a sector | §11 | all outputs permute identically | TBD (WO-002) | T-U5 |
| E14 | Same key, different call order | §2.15 | identical draw | TBD (WO-002) | T-U6 |

---

## 6. Sign-off

| Field | Value |
|---|---|
| Computed by | TBD (WO-002) |
| Date | TBD (WO-002) |
| `ref/ref_step.py` revision checked | TBD (WO-002) |
| Largest residual, hand vs reference | TBD (WO-002) |
| Discrepancies found and their resolution | TBD (WO-002) |
| Golden files generated after sign-off | TBD (WO-002) |

`ref/gen_golden.py` is run **only after** this sign-off is complete and the residual is clean. Golden
files generated before it are discarded, not re-blessed.
