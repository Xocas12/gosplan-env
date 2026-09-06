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
