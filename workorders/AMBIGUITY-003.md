AMBIGUITY REPORT   WO-005   gosplan/env/production.py
Question (one sentence):
Is the PLAN §2.6 yield shock of enterprise i drawn as a scalar at the per-enterprise key
`(seed_env, "yield", t, k, i)`, or as entry i of one `(N,)` vector drawn at `(seed_env, "yield", t, k)`?

What the spec says / does not say (quote):
The WO-005 card says "Draw one `(N,)` vector per step from the single key `(seed_env, "yield", t,
k)`", and the `produce_step` docstring says the draw is "vectorised over the trailing enterprise
index via `shape=(N,)`". But `ref/ref_step.py::ref_produce` (the frozen oracle behind the golden
files) draws `ref_draw(seed_env, "yield", t, k, i, shape=(1,), ...)[0]` per enterprise, and
`docs/ref_worked_example.md` lists `eps[t=0,k=0,i=0]` under key `(2, "yield", 0, 0, 0)` =
0.8980091184369003, which only the per-enterprise key reproduces (the `(N,)` convention gives
1.066878834783168). The `test_rng.py` docstring's claim that an `(n,)` draw equals `n` scalar draws
at the trailing index cannot hold under the `SeedSequence([seed_env, crc32(purpose), *indices])`
construction, so the vectorised wording is internally unsatisfiable as an identity.

Options considered (A/B/…), and why the spec does not decide:
A. Per-enterprise scalar key `(t, k, i)` with the enterprise's own sector sigma - matches
   `ref/ref_step.py`, the worked example and therefore golden parity T-B7.
B. One `(N,)` vector per sector at key `(t, k)`, entry i - matches the card and docstring text.
The written sources contradict each other and no unit test pins either (the production tests use
sigma = 0 or compare runs at the same seed).

Additional sub-questions raised by the same session:
- Which column of `pending_invest` receives `y_tilde * v`, and what if it has zero columns?
- Which Phase-2 production toggles (irs, capital accumulation, drift, arrival timing) belong here?

Impact if the wrong option is picked:
Every realised output differs; T-B7 (golden parity, 1e-9) fails on every cell under B.

Tests blocked:
tests/unit/test_production.py (the four `produce_step` tests); downstream T-B7 and T-U1 (WO-009).

LEAD RESOLUTION (2026-09-24):
A. The yield shock is `draw(seed_env, "yield", t, k, i, shape=(1,), dist="lognormal",
mean_log=-sigma**2/2, sigma=sigma)[0]` per enterprise, exactly as `ref_produce`. The frozen oracle
and the hand-checked worked example are authoritative over the card prose; the card's vectorised
wording is superseded. `pending_invest`: follow `ref_produce` - skip the write when the buffer has
zero columns, otherwise add `y_tilde * v` to the last column. Phase-2 toggles: implement exactly
the terms `ref_produce` implements (cost `kappa*e**2 + F*1[e>0] + kappa_q*q*e`, `quality_acc += q`,
consumption capped at stock, `y = y_tilde*(1-v)`) and nothing else; the rest is frozen at the
Phase-2 spec revision (WO-021).
