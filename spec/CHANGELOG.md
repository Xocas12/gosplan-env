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
