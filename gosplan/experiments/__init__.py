"""Lead-run experiments and studies - PLAN sections 7, 12.3-12.5, 13 and 14.

Realises: the `gosplan/experiments/` column of the repository layout in PLAN section 8, i.e. the
harnesses behind gates G0-G4. Owning work order: **WO-012** creates the package; every module names
its own card (WO-012, WO-015, WO-019, WO-020, WO-028, WO-032-WO-036).

Experiments are not tests. CONTRACT rule 13 separates them: `tests/unit`, `tests/behavioural` and
`tests/golden` are the frozen suites an implementer must pass, while the gates of PLAN section 13
are run deliberately by the lead, write their artefacts under `runs/` with the manifest of CONTRACT
rule 10, and sit on no work order's must-pass list. `make gate` refuses to run them on purpose.

Uniform module surface. Every module here exposes exactly two callables plus the pre-registered
design constants its card fixes:

    run(...)  -> dict[str, object]   executes the experiment, writes its artefacts, returns a
                                     summary mapping of the quantities the report is built from
    main()    -> int                 entry point; returns a process exit code

An exit code of 0 means the experiment ran and wrote its artefacts. It never means "the gate
passed". A gate is a written sign-off by the human and/or the lead on named artefacts (PLAN section
13); a gate that fails produces a written failure report whose successor is a lead diagnosis, never
a parameter change made in order to pass, and parameter changes after G1 create a new, labelled
study with its own pre-registration.

| Module | WO | Gate | Artefacts under `runs/` | Cost (PLAN section 14) |
|---|---|---|---|---|
| `mc_sanity` | WO-012 | G0 | `mc_sanity/report.md` | CPU minutes |
| `regime_map` | WO-015 | G1 | `regime_map/{table.parquet, regime.png, bhat.png}` | ~1 h, 8 cores |
| `dp_vs_ppo` | WO-019 | G2 (1) | `dp_vs_ppo/report.md` | ~1 GPU-h or ~8 CPU-h |
| `phase1_gate` | WO-020 | G2 (2-4) | `phase1_gate/report.md` | ~30 CPU-h, 4 h on 8 cores |
| `exploitability` | WO-028 | G3 | `exploitability/report.md` | doubles the runs it audits |
| `contrasts` | WO-032 | G4 | `contrasts/{table.parquet, report.md}` | hours (JAX) |
| `sobol` | WO-033 | G4, optional | `sobol/{table.parquet, report.md}` | ~1-3 GPU-days |
| `estimator_bias` | WO-034 | G4 | `estimator_bias/{table.parquet, report.md}` | hours |
| `llm_study` | WO-035 | G4 | `llm_study/{report.md, transcripts/}` | ~4M tokens, tens of dollars |
| `price_sensitivity` | WO-036 | G4 | `price_sensitivity/{table.parquet, report.md}` | minutes |

Phases follow the gates: G0-G2 are Phase 1 (WO-012 to WO-020), G3 is Phase 2 (WO-028), G4 is Phase
3 (WO-032 to WO-036). Every module's own docstring carries its full artefact list, its inputs, and
the PLAN section 14 cost line in full; `price_sensitivity` has no PLAN section 14 line of its own
because it recomputes over ledgers that already exist, except where a re-run is required (see that
module).

Held-out phenomena. PLAN section 4.1 rows 2 (storming), 5 (hoarding), 6 (blat) and 7 (hidden
reserves) are held out: no plot, table or test of them may be produced before the Phase-2 acceptance
run (WO-030 computes them for the first time, WO-031 runs it). `mc_sanity` and `regime_map` carry
an explicit prohibition to that effect; every other Phase-1 module here touches only rows 1 and 4.

Import discipline for this package (PLAN section 10, CONTRACT rule 9). Runtime configuration types
come from `gosplan.config`, whose dataclasses must stay field-for-field identical to `spec/spec.py`
(`spec/` is a frozen interface document, not an importable package; a unit test enforces the
agreement). Heavy or optional dependencies - `matplotlib` (`viz`), `pyarrow`/`pandas`, `rliable`
(`stats`), `jax`, `ortools` (`solver`) and any LLM client - are imported inside the function that
needs them, never at module scope, so importing this package stays cheap and dependency-free.
`forensics_core` is imported only inside a `try/except ImportError` that falls back to
`gosplan.metrics._fallback` with identical signatures (PLAN section 7.3).
"""

from __future__ import annotations

__all__ = [
    "contrasts",
    "dp_vs_ppo",
    "estimator_bias",
    "exploitability",
    "llm_study",
    "mc_sanity",
    "phase1_gate",
    "price_sensitivity",
    "regime_map",
    "sobol",
]
