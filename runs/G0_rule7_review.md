# Gate G0 — LEAD rule-7 diff review of `gosplan/env/`

CONTRACT rule 7: *no transition rule or reward term may implement bunching, padding, storming,
hoarding, shaving or trade directly.* Passing `contract_guard` and T-B1 is necessary, not
sufficient; this is the lead's line-by-line review of every `env/` change on this branch
(`git diff 5bea9fc -- gosplan/env/`: 9 files, +966 / −53).

Reviewer: LEAD (session acting as lead for the owner). Date: 2026-09-24.

## Method

Every conditional, clip, `min`/`max`, `where` and `maximum`/`minimum` added under `gosplan/env/`
was enumerated and traced to the PLAN formula it transcribes. The question asked of each line:
does this line *decide an agent-facing outcome* (how much to report, when to exert effort, how much
input to hold, whether to trade), or does it *execute a written rule* whose consequences an agent
may exploit?

## Findings, per module

| Module | What was added | Rule 7 verdict |
|---|---|---|
| `production.py` | CES / Leontief coverage (§2.6) with explicit `need = 0` branches; yield shock via `draw`; consumption `min(X, a·ỹ)`; cost `κe² + F·1[e>0] + κ_q·q·e`; effort clipped to its action box `[0, 1]` | Executes §2.6. Effort is taken as given per step; nothing concentrates it within a period — storming can only arise from `delivery_timing` (inert, `uniform`, in Phase 1) plus the agent's own choice. |
| `planner.py` | `make_planner_view` (claims · exp(channel noise), sector mean, audit_meas zeros); ratchet with caps, deadband, growth, floor (§2.7.1); allocation weights `(q+1e-6)^η_q (need+1e-6)^η_n` (§2.7.2); delivery `fill = min(1, S/claimed)`, `shipped = min(S, claimed)`, `poolfill`, `deliv = alloc·poolfill` (§2.7.3); Bernoulli audits (§2.7.4) | Executes §2.7. Padding → shortage and shaving → reserves are *consequences* of the four delivery lines, not rules. `η_q = 0` in Phase 1, so requests are inert. |
| `reporting.py` | `S ← min((1−h)S + y, S_max)`; `R = clip(ρ, 0, ρ_max)·T`; request = `clip(q, 0, r_max)·need` (AMBIGUITY-008); audit `Ŝ = S·exp(ν)`, `f = max(0, R−Ŝ)/T` or `|R−Ŝ|/T`, `Pen = pen·f` or `pen·1[f>0]` (§2.8) | Executes §2.8. The report is the agent's; the env only clips to the action box and the pre-registered bound (rule 8, logged). |
| `reward.py` | `B(ρ) = β·Λ_w(ρ−1) + s·over(ρ)` with the AMBIGUITY-006 softplus in the smooth arm only; `reward_scale = 1/B(1.1)`; reward exactly `−scale·c` / `scale·(B − Pen + surplus)`; val / welfare metrics (ledger only) | Exactly the rule-4 terms. The bonus is the *incentive being studied*, not a pathology rule; the smooth-arm change removes a kink rather than adding a pull toward ρ = 1. |
| `obs.py` | §2.4 layout; `need = 0 → 1.0`; `selfobs` noise | No welfare, no other enterprise's quantities, no periods remaining (T-B5, T-B9 pass). |
| `prices.py` | Cost-plus fixed point by iteration (§2.10) | Fixed plan prices; no behavioural content. |
| `state.py`, `step.py`, `env.py` | Opening state (X = a·T₀ per ambiguity #62); the §2.5 schedule as data; stage assemblers; geometric horizon; eager counters; auto-continue after `done` (AMBIGUITY-004); ledger records | Pure schedule and bookkeeping. No stage is reordered, fused or conditioned on an outcome. |

## Behavioural evidence (necessary, not sufficient)

- T-B1 `test_no_hardcoded_pathology.py`: truthful reports equal stock to 1e-9; no histogram spike
  (under the docstring's both-neighbours rule, AMBIGUITY-008); effort Gini equals the
  yield-noise-implied value; no trade; requests equal need — all pass.
- T-B3 shortage propagation, T-B4 planner blindness, T-B5 welfare blindness, T-B9 termination:
  all pass. Full frozen suite: 456 passed, 16 skipped (later cards), 0 failed.

## Verdict

No line under `gosplan/env/` implements bunching, padding, storming, hoarding, shaving or trade.
**Rule 7: clean** (LEAD). This review is one of the three G0 conditions; the other two are the
full frozen suite (green) and `runs/mc_sanity/report.md` (pending the sweep).
