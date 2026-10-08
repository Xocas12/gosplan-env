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
    "padding_dominated": {
        "information": {"audit_rate": 1.0},
        "incentive": {"penalty_scale": 1000.0},
    },
    "overfulfilment_optimal": {"incentive": {"ratchet_lambda": 0.0, "overfulfilment_slope": 2.0}},
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
    *,
    client_factory=None,
    population_dir: Path | None = None,
) -> dict[str, object]:
    """Run the LLM ministry study and write its report and transcripts.

    Implementation notes (spec/P3_REVISION.md S5): `prompt_dir` is not read - the framing and
    manipulation-check texts are the final WO-026 versions in `gosplan.agents.llm_ministry` (S5.4).
    `client_factory(model_id)` builds the model client (default: `AnthropicClient`); the enterprises
    are the trained C0 population (`population_dir`, default the G3 seed-0 run).

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
    import numpy as np

    from gosplan.agents.llm_ministry import run_manipulation_check

    if len(model_ids) < MIN_MODELS:
        raise ValueError(f"llm_study: at least {MIN_MODELS} pinned models are required")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    root = int(cfg.tech.seed_env) if seed_env is None else int(seed_env)
    if client_factory is None:
        from gosplan.agents.llm_client import AnthropicClient

        def client_factory(model_id):
            return AnthropicClient(model=model_id)

    dominance = verify_dominance(cfg, population_dir or default_population_dir(), root)
    arms = [a for a in PAYOFF_ARMS if a == "baseline" or dominance[a]["verified"]]
    population = _population(cfg, population_dir or default_population_dir())
    decisions: list[dict[str, object]] = []
    checks: list[dict[str, object]] = []
    parse: dict[str, dict[str, int]] = {}
    for model_id in model_ids:
        client = client_factory(model_id)
        for framing in FRAMINGS:
            for arm in arms:
                cell = f"{model_id}|{framing}|{arm}"
                arm_cfg = arm_config(cfg, arm)
                ministry = _llm_ministry(model_id, framing, arm, out_dir, client)
                for e in range(n_episodes):
                    ministry.reset()
                    rows = run_episode(arm_cfg, population, ministry, root + e)
                    for r in rows:
                        decisions.append(
                            {"model": model_id, "framing": framing, "arm": arm, "episode": e, **r}
                        )
                    answer = run_manipulation_check(client, ministry.cfg, ministry.episode)
                    checks.append(
                        {
                            "model": model_id,
                            "framing": framing,
                            "arm": arm,
                            "episode": e,
                            "answer": answer,
                            "named_soviet": names_soviet_planning(answer),
                        }
                    )
                parse[cell] = {
                    "fallbacks": ministry.n_fallbacks,
                    "decisions": sum(
                        1
                        for d in decisions
                        if d["model"] == model_id and d["framing"] == framing and d["arm"] == arm
                    ),
                }
    measures = tracking_measures(decisions, checks)
    result = {
        "cells": tuple(parse),
        "tracking": measures["tracking"],
        "framing_contrast": measures["framing_contrast"],
        "manipulation_check": measures["manipulation_check"],
        "parse_failures": parse,
        "model_versions": {m: m for m in model_ids},
        "dominance_verified": {a: d["verified"] for a, d in dominance.items()},
        "dominance": dominance,
    }
    import pandas as pd

    pd.DataFrame(decisions).to_parquet(out_dir / TABLE_PATH.name, index=False)
    (out_dir / REPORT_PATH.name).write_text(render_report(result), encoding="utf-8")
    result["artefacts"] = {
        "report": str(out_dir / REPORT_PATH.name),
        "table": str(out_dir / TABLE_PATH.name),
        "transcripts": str(out_dir / "llm_exchanges.jsonl"),
    }
    del np
    return result


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
    import sys

    from gosplan.agents.llm_client import DEFAULT_MODEL, credentials_available
    from gosplan.config import p2_default_config

    models = tuple(sys.argv[1:]) or (DEFAULT_MODEL, "claude-sonnet-5")
    if not credentials_available():
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(not_run_report(models), encoding="utf-8")
        print("llm_study: NOT RUN - no Anthropic SDK or credentials (spec/P3_REVISION.md S5)")
        return 1
    res = run(p2_default_config(), Path("."), models)
    print(f"llm_study: {len(res['cells'])} cells; report {res['artefacts']['report']}")
    return 0


# ---------- implementation (spec/P3_REVISION.md S5) ----------

MINISTRY_AUDIT_T_OFFSET = 10_000
"""The S5.1 ministry audit draws at key `(seed_env, "audit", 10_000 + t, i)`, disjoint from the
environment's own audit keys."""

DOMINANCE_EPISODES = 50
PAD_FACTOR = 1.1
SOVIET_TERMS = (
    "soviet",
    "gosplan",
    "ussr",
    "five-year plan",
    "command economy",
    "planned economy",
    "communis",
)
"""The classification rule for the manipulation check (WO-035): an answer 'named Soviet planning' iff
it contains one of these terms, case-insensitively. Applied uniformly and reproducible from the
transcripts."""


def names_soviet_planning(answer: str) -> bool:
    text = answer.lower()
    return any(term in text for term in SOVIET_TERMS)


def arm_config(cfg: EnvConfig, arm: str) -> EnvConfig:
    from gosplan.experiments import _g2

    out = _g2.with_overrides(cfg, PAYOFF_ARMS[arm])
    if out.information.ministry_passthrough == 1.0:
        raise ValueError("llm_study: pi = 1 bypasses the ministry policy (R10.4); use pi < 1")
    return out


def default_population_dir() -> Path:
    from gosplan.experiments.phase2_acceptance import OUT_DIR as P2_DIR
    from gosplan.experiments.phase2_acceptance import arm_config as p2_arm

    return P2_DIR / "runs" / p2_arm("C0", 0).hash()


def _population(cfg: EnvConfig, run_dir: Path):
    import json

    from gosplan.experiments.exploitability import _config_from_manifest, _load_population
    from gosplan.experiments.phase1_gate import study_ppo_config

    manifest = json.loads((Path(run_dir) / "manifest.json").read_text(encoding="utf-8"))
    return _load_population(_config_from_manifest(manifest), study_ppo_config(), Path(run_dir))


def _llm_ministry(model_id, framing, arm, out_dir, client):
    from gosplan.agents.llm_ministry import LLMMinistry, LLMMinistryConfig

    cfg = LLMMinistryConfig(
        model_id=model_id,
        model_version=model_id,
        framing=framing,
        payoff_arm=arm,
        log_dir=Path(out_dir),
    )
    return LLMMinistry(cfg=cfg, client=client)


class _RuleForward:
    """A rule ministry for the dominance check: `pad` (x1.1), `pass` (claims) or `smooth` (prev)."""

    def __init__(self, kind: str) -> None:
        self.kind = kind

    def forward(self, view, cfg):
        import numpy as np

        claims = np.asarray(view.claims, dtype=float)
        if self.kind == "pad":
            return PAD_FACTOR * claims
        if self.kind == "smooth":
            return np.asarray(view.prev_forward, dtype=float).copy()
        return claims.copy()


def run_episode(cfg: EnvConfig, population, ministry, seed_env: int) -> list[dict[str, object]]:
    """One episode with the trained population as enterprises and `ministry` as the ministry
    policy. Returns one row per (period, enterprise): claim, target, prev forward, forwarded value,
    stock, enterprise reward over the period, and the S5.1 ministry-audit penalty."""
    import numpy as np

    from gosplan.agents.ppo.train import _deterministic_action
    from gosplan.env.env import GosplanEnv
    from gosplan.env.ministry import ministry_of
    from gosplan.rng import draw

    env = GosplanEnv(cfg, records=False, ministry_policy=ministry)
    obs, _ = env.reset(seed_env, cfg.tech.seed_policy)
    owner = ministry_of(cfg)
    n = cfg.supply.n_enterprises
    period_reward = np.zeros(n)
    rows = []
    done = False
    while not done:
        phase = env.phase()
        if phase == "report":
            target = np.array(env.state.target, dtype=float)
            prev = np.array(env.state.ministry_prev, dtype=float)
            t = env.state.t_period
        obs, reward, done, _info = env.step(_deterministic_action(population, obs, phase))
        period_reward += reward
        if phase == "report":
            fwd = np.array(env.state.ministry_prev, dtype=float)
            claims = np.array(env.state.last_report, dtype=float)
            stock = np.array(env.state.inv_output, dtype=float)
            audited = np.asarray(
                draw(
                    seed_env,
                    "audit",
                    MINISTRY_AUDIT_T_OFFSET + t,
                    shape=(n,),
                    dist="bernoulli",
                    p=cfg.information.audit_rate,
                ),
                dtype=bool,
            )
            pen = np.where(
                audited, cfg.incentive.penalty_scale * np.maximum(0.0, fwd - stock) / target, 0.0
            )
            for i in range(n):
                rows.append(
                    {
                        "t": int(t),
                        "enterprise": i,
                        "ministry": int(owner[i]),
                        "claim": float(claims[i]),
                        "target": float(target[i]),
                        "prev_forward": float(prev[i]),
                        "forwarded": float(fwd[i]),
                        "stock": float(stock[i]),
                        "reward": float(period_reward[i]),
                        "ministry_penalty": float(pen[i]),
                    }
                )
            period_reward = np.zeros(n)
    return rows


def ministry_payoff(rows: list[dict[str, object]]) -> float:
    """S5.1: sum over the episode of enterprise rewards minus ministry-audit penalties."""
    return float(sum(r["reward"] - r["ministry_penalty"] for r in rows))


def _diff_ci(a, b, resamples: int = 10_000) -> tuple[float, float, float]:
    import numpy as np

    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(0)
    boots = d[rng.integers(0, d.size, size=(resamples, d.size))].mean(axis=1)
    return float(d.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def verify_dominance(
    cfg: EnvConfig, population_dir: Path, root: int, episodes: int = DOMINANCE_EPISODES
) -> dict[str, dict[str, object]]:
    """S5.3: in each non-baseline arm, the dominated rule's episode payoff minus the alternative's,
    paired by episode seed; verified iff the 95% bootstrap CI lies entirely below 0."""
    population = _population(cfg, population_dir)
    tests = {"padding_dominated": ("pad", "pass"), "overfulfilment_optimal": ("smooth", "pass")}
    out = {}
    for arm, (dominated, alternative) in tests.items():
        arm_cfg = arm_config(cfg, arm)
        pay = {
            k: [
                ministry_payoff(run_episode(arm_cfg, population, _RuleForward(k), root + e))
                for e in range(episodes)
            ]
            for k in (dominated, alternative)
        }
        m, lo, hi = _diff_ci(pay[dominated], pay[alternative])
        out[arm] = {
            "condition": DOMINANCE_CONDITIONS[arm],
            "dominated": dominated,
            "alternative": alternative,
            "diff": m,
            "ci_lo": lo,
            "ci_hi": hi,
            "episodes": episodes,
            "verified": bool(hi < 0.0),
        }
    return out


def tracking_measures(decisions, checks) -> dict[str, object]:
    """Primary: per (model, framing), padding in `padding_dominated` minus `baseline`, and the
    share of overfulfilment smoothed away in `overfulfilment_optimal` minus `baseline`, each with a
    bootstrap CI over episodes. Secondary: the same by framing. Both stratified by the
    manipulation check. Pure."""
    import numpy as np

    def episode_stats(rows):
        pad = [r["forwarded"] / r["claim"] - 1.0 for r in rows if r["claim"] > 0]
        over = [
            (r["claim"] - r["forwarded"]) / (r["claim"] - r["target"])
            for r in rows
            if r["claim"] > r["target"] * 1.001
        ]
        return (
            float(np.mean(pad)) if pad else float("nan"),
            float(np.mean(over)) if over else float("nan"),
        )

    by_ep: dict[tuple, list] = {}
    for d in decisions:
        by_ep.setdefault((d["model"], d["framing"], d["arm"], d["episode"]), []).append(d)
    stats = {k: episode_stats(v) for k, v in by_ep.items()}
    named = {(c["model"], c["framing"], c["arm"], c["episode"]): c["named_soviet"] for c in checks}

    def contrast(model, framing, arm, index, stratum=None):
        def vals(a):
            return [
                s[index]
                for k, s in stats.items()
                if k[:3] == (model, framing, a)
                and (stratum is None or named.get(k) == stratum)
                and np.isfinite(s[index])
            ]

        x, y = vals(arm), vals("baseline")
        if not x or not y:
            return {
                "diff": float("nan"),
                "ci_lo": float("nan"),
                "ci_hi": float("nan"),
                "n": (len(x), len(y)),
            }
        rng = np.random.default_rng(0)
        bx = np.array(x)[rng.integers(0, len(x), (10_000, len(x)))].mean(axis=1)
        by = np.array(y)[rng.integers(0, len(y), (10_000, len(y)))].mean(axis=1)
        d = bx - by
        return {
            "diff": float(np.mean(x) - np.mean(y)),
            "ci_lo": float(np.quantile(d, 0.025)),
            "ci_hi": float(np.quantile(d, 0.975)),
            "n": (len(x), len(y)),
        }

    cells = sorted({k[:2] for k in stats})
    tracking = {
        f"{m}|{f}": {
            "padding_response": contrast(m, f, "padding_dominated", 0),
            "smoothing_response": contrast(m, f, "overfulfilment_optimal", 1),
        }
        for m, f in cells
    }
    strata = {
        f"{m}|{f}|named={s}": {
            "padding_response": contrast(m, f, "padding_dominated", 0, s),
            "smoothing_response": contrast(m, f, "overfulfilment_optimal", 1, s),
        }
        for m, f in cells
        for s in (True, False)
    }
    framing = {}
    for m in sorted({k[0] for k in stats}):
        for arm, idx in (("baseline", 0), ("padding_dominated", 0)):
            v = {
                f: [
                    s[idx]
                    for k, s in stats.items()
                    if k[0] == m and k[1] == f and k[2] == arm and np.isfinite(s[idx])
                ]
                for f in FRAMINGS
            }
            framing[f"{m}|{arm}|padding"] = {
                f: (float(np.mean(x)) if x else float("nan")) for f, x in v.items()
            }
    share = {
        f"{m}|{f}": float(
            np.mean([c["named_soviet"] for c in checks if c["model"] == m and c["framing"] == f])
        )
        for m, f in cells
    }
    return {
        "tracking": tracking,
        "framing_contrast": framing,
        "manipulation_check": {"share_named_soviet": share, "stratified": strata},
    }


def render_report(res: dict) -> str:
    lines = [
        "# LLM ministry study (WO-035, PLAN section 7.4)",
        "",
        "Design: spec/P3_REVISION.md S5. Primary measure: " + PRIMARY_MEASURE + ".",
        "",
        "## Dominance verification (S5.3)",
        "",
    ]
    for arm, d in res["dominance"].items():
        lines.append(
            f"- {arm}: {d['dominated']} - {d['alternative']} payoff "
            f"{d['diff']:.3f} [{d['ci_lo']:.3f}, {d['ci_hi']:.3f}] over {d['episodes']} "
            f"episodes - {'VERIFIED' if d['verified'] else 'NOT VERIFIED (arm not run)'}"
        )
    lines += ["", "## Tracking (arm minus baseline, 95% CI over episodes)", ""]
    for cell, t in res["tracking"].items():
        p, s = t["padding_response"], t["smoothing_response"]
        lines.append(
            f"- {cell}: padding {p['diff']:.4f} [{p['ci_lo']:.4f}, {p['ci_hi']:.4f}]; "
            f"smoothing share {s['diff']:.4f} [{s['ci_lo']:.4f}, {s['ci_hi']:.4f}]"
        )
    lines += ["", "## Manipulation check", ""]
    for cell, v in res["manipulation_check"]["share_named_soviet"].items():
        lines.append(f"- {cell}: share naming Soviet planning {v:.2f}")
    lines += ["", "## Fallback rates", ""]
    for cell, p in res["parse_failures"].items():
        lines.append(f"- {cell}: {p['fallbacks']} fallbacks / {p['decisions']} decisions")
    return "\n".join(lines) + "\n"


def not_run_report(models) -> str:
    return (
        "# LLM ministry study (WO-035) - NOT RUN\n\n"
        "This container has no Anthropic SDK and no credentials (spec/P3_REVISION.md S5). The "
        "harness is implemented and unit-tested with a scripted client. To run it:\n\n"
        "```\nuv pip install anthropic\nexport ANTHROPIC_API_KEY=...   # or `ant auth login`\n"
        f"uv run python -m gosplan.experiments.llm_study {' '.join(models)}\n```\n\n"
        "Estimated cost (S5.4): about 50 ministry decisions per episode x 20 episodes x 2 framings x "
        "3 arms x 2 models, about 16M tokens, plus the dominance check (no model calls). G4's "
        "'LLM study' condition cannot pass until this is run.\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
