"""LLM ministry study - PLAN sections 2.14, 7.4, 12.5 (WO-035), 13 (gate G4) and 14.

Realises: the study design of PLAN section 7.4, run against the ministry layer of PLAN section 2.14
through the adapter of WO-026. Owning work order: **WO-035** (MID-strong, Phase 3; **the lead writes
both framing prompts and the manipulation-check prompt** - this module never contains prompt text,
it loads the lead's files). Gate: **G4** - the final report needs the LLM study (PLAN section 13).

    Separate from the factorial. PLAN section 7.4 is explicit: this study stands beside the
    contrasts of PLAN section 4.3, it is not a cell in them. Nothing here feeds a `Delta_X`, and no
    contrast conclusion may lean on an LLM ministry's behaviour.

The question. A language model placed in the ministry seat can behave in two very different ways,
and the design is built to tell them apart:

    reasoning   ministry behaviour **tracks the payoff arm** - it changes when the payoffs change
    retrieval   ministry behaviour stays at the historical pattern **regardless** of the payoffs

The discriminating measure is therefore the *interaction between behaviour and payoff arm*, not the
level of any single arm. The framing contrast (neutral versus historical vocabulary) is explicitly
**secondary** (PLAN section 7.4).

Design (PLAN section 7.4): `FRAMINGS` (2) x `PAYOFF_ARMS` (3) x models (>= `MIN_MODELS`) x
`N_EPISODES_PER_CELL` (20) episodes.

Manipulation check (PLAN section 7.4). After each episode, **in a fresh context**, the model is
asked to name the closest real-world analogue of the game it just played. Results are stratified by
whether it named Soviet planning. The check is part of the design, not a post-hoc filter: both
strata are reported, and a cell is never dropped because its stratum is inconvenient.

Protocol (PLAN sections 2.14, 7.4). The ministry receives a text rendering of a `MinistryView` and
nothing else - no `State`, no `PlannerView`, no true quantity (CONTRACT rules 5 and 6) - and returns
strict JSON with per-enterprise forwarded values plus a free-text justification. One retry on a
parse failure, then the documented fallback action `pi = 1` (`FALLBACK_PASSTHROUGH`), i.e. fully
transparent forwarding. Model ids and versions are pinned and recorded in the manifest, and **every
prompt and every completion is logged** (CONTRACT rule 10).

Inputs
    The lead's prompt files: one per framing plus the manipulation-check prompt, loaded from the
    directory passed to `run` (their location is fixed when WO-035 is issued and recorded in the
    manifest). The Phase-2 configuration with an active ministry layer (`n_ministries > 1`,
    `ministry_passthrough` under the model's control), the adapter of WO-026, and the pinned model
    ids.

Outputs
    `runs/llm_study/report.md`       the tracking measure per (model, framing, payoff arm), the
                                     manipulation-check stratification, the fallback and parse-error
                                     rates, and the pinned model versions
    `runs/llm_study/table.parquet`   one row per (model, framing, arm, episode, period) ministry
                                     decision with the forwarded values and the parse outcome
    `runs/llm_study/transcripts/`    every prompt and completion, including the manipulation-check
                                     exchanges (CONTRACT rule 10)
    `runs/<config-hash>/`            per-run directories with `manifest.json`

    PLAN section 12.5 names no artefact paths for WO-035; these follow the `runs/<experiment>/`
    convention of the Phase-1 cards.

Cost (PLAN section 14): 2 framings x 3 arms x 2 models x 20 episodes x about 12 ministry decisions x
about 1.5k tokens - about 4M tokens, tens of dollars.

Runtime bindings. `EnvConfig` is `gosplan.config.EnvConfig` (field-for-field identical to
`spec/spec.py`, enforced by a unit test). Any model client is imported inside the function that
calls it, never at module scope, so the package imports without network dependencies.
"""

from __future__ import annotations

from pathlib import Path

from gosplan.config import EnvConfig

FRAMINGS: tuple[str, ...] = ("neutral", "historical")
"""The two framings of PLAN section 7.4: neutral vocabulary and historical (Soviet-planning)
vocabulary. The prompt text for each is written by the lead (WO-035 card) and loaded from disk; this
module holds the names only, so a prompt can never be edited by an implementer session in passing.
The framing contrast is the study's **secondary** measure."""

PAYOFF_ARMS: dict[str, dict[str, dict[str, object]]] = {
    "baseline": {},
    "padding_dominated": {"information": {"audit_rate": 1.0}},
    "overfulfilment_optimal": {"incentive": {"ratchet_lambda": 0.0}},
}
"""The three payoff arms of PLAN section 7.4, as overrides on the study's base configuration.

    baseline                the study's base payoffs, unmodified
    padding_dominated       `a = 1` with `pen` large, so that forwarding padded numbers upward is
                            strictly worse for the ministry than forwarding what it received
    overfulfilment_optimal  `lambda = 0` with `s` large, so that smoothing away overfulfilment is
                            strictly worse than passing it through

Only the two values PLAN section 7.4 states numerically appear here. The magnitudes it states
qualitatively - "pen large" (`incentive.penalty_scale`) and "s large"
(`incentive.overfulfilment_slope`) - are not invented in this module: the lead sets each when WO-035
is issued, to the smallest value inside the PLAN section 3 range for which the stated dominance is
*strict* under `bonus` and the penalty of PLAN section 2.8, verifies it analytically before any
model call, and records both the value and the verification in the manifest. A payoff arm whose
dominance has not been verified is not run - the whole design rests on the arms actually having the
payoff structure their names claim."""

DOMINANCE_CONDITIONS: dict[str, str] = {
    "padding_dominated": (
        "forwarding padded numbers upward is strictly worse for the ministry than forwarding the "
        "claims it received"
    ),
    "overfulfilment_optimal": (
        "smoothing away overfulfilment is strictly worse for the ministry than forwarding it"
    ),
}
"""The property each non-baseline arm must satisfy before it is run (PLAN section 7.4), written out
so the verification the lead performs has a stated target and the report can quote it. `baseline`
has no dominance condition by construction."""

MIN_MODELS = 2
"""At least two models (PLAN section 7.4). The ids and versions are pinned by the lead at issue time
and recorded in the manifest (CONTRACT rule 10); they are not constants here, because a model
version that changed under a study is a result-invalidating event and must be visible in the run
record rather than in source."""

N_EPISODES_PER_CELL = 20
"""Episodes per (model, framing, payoff arm) cell (PLAN sections 7.4, 14)."""

EXPECTED_DECISIONS_PER_EPISODE = 12
"""About twelve ministry decisions per episode, the figure the PLAN section 14 cost line uses (one
forwarding decision per period, with geometric termination giving roughly ten periods plus the
manipulation-check exchange). Used for cost estimation and budget checks only, never as a loop
bound: the episode length is the environment's, from the geometric termination of PLAN section
2.12."""

PRIMARY_MEASURE = (
    "whether ministry behaviour tracks the payoff arm (reasoning) or stays at the historical "
    "pattern regardless of payoffs (retrieval)"
)
"""The discriminating measure of PLAN section 7.4, stated as data so the report and the analysis
cite one sentence. Operationally it is the interaction between the forwarding behaviour and the
payoff arm, within model and framing."""

SECONDARY_MEASURE = "the neutral-versus-historical framing contrast"
"""Explicitly secondary in PLAN section 7.4. It is reported, and it is never presented as the
study's answer."""

MANIPULATION_CHECK_FRESH_CONTEXT = True
"""The manipulation check runs in a **fresh context** after each episode (PLAN section 7.4): the
model is asked to name the closest real-world analogue of the game it played, with no access to the
episode transcript that would prompt the answer."""

MANIPULATION_CHECK_STRATUM = "named Soviet planning"
"""The stratification variable of PLAN section 7.4. Both strata - the episodes where the model named
Soviet planning and those where it did not - are reported. Neither is dropped, and the check is not
used as a filter on the primary measure."""

JSON_RETRIES = 1
"""Strict-JSON responses with **one** retry on a parse failure (PLAN section 7.4). The retry is
logged like any other call."""

FALLBACK_PASSTHROUGH = 1.0
"""The documented fallback action when both attempts fail to parse (PLAN section 7.4): `pi = 1`,
i.e. `ministry_passthrough = 1.0`, fully transparent forwarding. Fallback invocations are counted
and reported per cell, because a cell with a high fallback rate is measuring the parser, not the
model."""

LOG_EVERY_PROMPT_AND_COMPLETION = True
"""PLAN section 7.4 and CONTRACT rule 10: every prompt and every completion is logged, the
manipulation-check exchanges included. Declared as data so a report can state that the transcripts
exist and where they are."""

OUT_DIR = Path("runs/llm_study")
"""Artefact directory, relative to the repository root; a WO-035 convention."""

REPORT_PATH = OUT_DIR / "report.md"
"""The tracking measure, the stratified manipulation check, the fallback rates, the pinned model
versions."""

TABLE_PATH = OUT_DIR / "table.parquet"
"""One row per ministry decision."""

TRANSCRIPT_DIR = OUT_DIR / "transcripts"
"""Every prompt and completion, per CONTRACT rule 10."""


def run(
    cfg: EnvConfig,
    prompt_dir: Path,
    model_ids: tuple[str, ...],
    out_dir: Path = OUT_DIR,
    n_episodes: int = N_EPISODES_PER_CELL,
    seed_env: int | None = None,
) -> dict[str, object]:
    """Run the LLM ministry study and write its report and transcripts.

    Takes: `cfg`, the Phase-2 configuration with an active ministry layer, already validated;
    `prompt_dir`, the directory holding the lead's prompt files - one per name in `FRAMINGS` plus
    the manipulation-check prompt (the WO-035 card assigns their authorship to the lead, so this
    module loads them and never contains prompt text); `model_ids`, the pinned model identifiers,
    at least `MIN_MODELS` of them, each including its version; `out_dir`, where the artefacts are
    written; `n_episodes`, episodes per cell; `seed_env`, the root environment seed - `None` means
    `cfg.tech.seed_env`, shared across cells so every model, framing and payoff arm meets identical
    environment draws (PLAN sections 2.15, 4.3).

    Returns: a mapping with at least

        "cells"                tuple[dict[str, object], ...], one entry per (model, framing, arm)
                               with the forwarding summary and the tracking statistic
        "tracking"             dict, the `PRIMARY_MEASURE`: behaviour by payoff arm, within model
                               and framing, with intervals
        "framing_contrast"     dict, the `SECONDARY_MEASURE`
        "manipulation_check"   dict, results stratified by `MANIPULATION_CHECK_STRATUM`, both strata
                               reported
        "parse_failures"       dict, per cell: first-attempt failures, retries, and
                               `FALLBACK_PASSTHROUGH` invocations
        "model_versions"       dict[str, str], the pinned ids and versions actually called
        "dominance_verified"   dict[str, bool], per non-baseline arm, from the lead's pre-run check
                               against `DOMINANCE_CONDITIONS`
        "artefacts"            dict[str, str], the paths written

    Procedure (PLAN sections 2.14, 7.4):

      1. Refuse to start unless every arm in `PAYOFF_ARMS` other than `baseline` carries a recorded,
         verified dominance check against `DOMINANCE_CONDITIONS`, and unless `len(model_ids) >=
         MIN_MODELS`.
      2. For each (model, framing, arm) cell, run `n_episodes` episodes in which the ministry
         decision of PLAN section 2.14 is taken by the model through the WO-026 adapter: it receives
         a text rendering of a `MinistryView` and nothing else (CONTRACT rules 5, 6), and returns
         strict JSON. On a parse failure, retry once (`JSON_RETRIES`), then fall back to
         `FALLBACK_PASSTHROUGH` and count it.
      3. After each episode, run the manipulation check in a fresh context
         (`MANIPULATION_CHECK_FRESH_CONTEXT`) and record whether the answer named Soviet planning.
      4. Log every prompt and completion to `TRANSCRIPT_DIR` and write `runs/<hash>/manifest.json`
         with the pinned model ids and versions (CONTRACT rule 10).
      5. Compute the `PRIMARY_MEASURE` - whether behaviour tracks the payoff arm - within model and
         framing, then the `SECONDARY_MEASURE`, each stratified by the manipulation check.
      6. Write `table.parquet` and `report.md`, stating the fallback rates beside every cell.

    This study is separate from the factorial (PLAN section 7.4): no number produced here enters a
    contrast, and no contrast conclusion may rest on it.

    Binds: gate G4 of PLAN section 13 - "LLM study" - and the design of PLAN section 7.4.

    Realises: PLAN sections 2.14, 7.4, 12.5 (WO-035), 13, 14. Owning WO: **WO-035**.
    """
    raise NotImplementedError("PLAN section 7.4 (WO-035) - implemented in WO-035")


def main() -> int:
    """Entry point: run every cell of the LLM ministry study and write the report.

    Takes: nothing; the configuration, the lead's prompt directory and the pinned model ids come
    from the WO-035 issue record, the design is `FRAMINGS` x `PAYOFF_ARMS` x models x
    `N_EPISODES_PER_CELL`. Any command-line surface, client construction and rate limiting is built
    inside this function.

    Returns: a process exit code - 0 when every cell ran and `runs/llm_study/report.md` was written
    with its transcripts, 1 when a cell could not run - including the case where a payoff arm's
    dominance condition was not verified, or a model version could not be pinned. The exit code says
    nothing about the finding: whether behaviour tracks the payoff arms is read off the report.

    Realises: PLAN sections 7.4, 12.5 (WO-035), 13. Owning WO: **WO-035**.
    """
    raise NotImplementedError("PLAN section 7.4 (WO-035) - implemented in WO-035")


if __name__ == "__main__":
    raise SystemExit(main())
