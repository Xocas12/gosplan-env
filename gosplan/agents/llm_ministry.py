"""LLM ministry: the Phase-2 study of PLAN section 7.4. **Study stub - interface only.**

Realises: PLAN section 7.4 (the LLM ministry study) over the ministry layer of PLAN section 2.14,
with the manifest requirements of CONTRACT rule 10. Owning work orders: **WO-026** (the adapter -
`MinistryView` to text, strict-JSON parsing, retry, fallback, prompt/completion logging, version
pinning) and **WO-035** (the study harness; the lead writes both framing prompts and the
manipulation-check prompt).

What this replaces. `ministry_forward(view, cfg)` (PLAN section 2.14, WO-025) is the rule-based
ministry: `R_tilde_i = pi * R_i + (1 - pi) * [Rbar_i_prev + kappa_m * max(0, T_i - R_i)]`, a
smoother that pads shortfalls, with `pi = cfg.information.ministry_passthrough`. The study swaps
that function for a model call that receives *the same* `MinistryView` rendered as text and returns
per-enterprise forwarded values plus a free-text justification. Nothing else about the environment
changes: the ministry sits between enterprises and the planner, and the planner still sees only a
`PlannerView` (CONTRACT rule 5).

The design, from PLAN section 7.4: **2 framings x 3 payoff arms x >= 2 models x 20 episodes**, run
separately from the factorial of PLAN section 4.3 and never folded into it.

  framings     `neutral` and `historical` - identical decision problem, different vocabulary.
  payoff arms  `baseline`; `padding_dominated` (`audit_rate = 1` and a large `penalty_scale`, so
               forwarding padded numbers is strictly worse for the ministry); `overfulfilment_
               optimal` (`ratchet_lambda = 0` and a large `overfulfilment_slope`, so smoothing away
               overfulfilment is strictly worse). Each arm is a *configuration*, not a prompt: the
               incentives change, the instructions do not.
  models       at least two, at pinned versions (`LLMMinistryConfig.model_id`).
  episodes     20 per cell.

**The discriminating measure is whether behaviour tracks the payoff arm.** A ministry that forwards
padded numbers in `padding_dominated`, where padding is strictly worse for it, is reproducing a
historical pattern from training data (retrieval); one that stops padding there is responding to the
incentives it faces (reasoning). The framing contrast - neutral versus historical vocabulary - is
*secondary*, and no conclusion rests on it alone.

**Manipulation check.** After each episode the model is asked, in a fresh context, to name the
closest real-world analogue of the game it played (`manipulation_check_prompt`). Results are
stratified by whether it named Soviet planning, because a model that recognises the setting may be
retrieving the historical pattern rather than optimising.

Prompt discipline. Neither framing may instruct the model to pad, to smooth, to protect its
enterprises, or to fulfil the plan; they state the same decision problem in different vocabulary.
A model told what to do tells the study nothing - and a prompt that instructs padding would put the
pathology into the mechanism by hand, which is what CONTRACT rule 7 exists to prevent.

Logging and pinning (CONTRACT rule 10). Model ids and versions are pinned in the configuration and
written to the run manifest; **every prompt and every completion is logged**, including retries,
fallbacks and manipulation-check exchanges, so the study is auditable after the model provider
changes something underneath it.

Import discipline: this module imports no vendor SDK and no `json`; the client is injected as an
opaque object and the JSON handling lives in the implementation (WO-026).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from gosplan.agents.base import Array

if TYPE_CHECKING:  # runtime homes: WO-003 (config), WO-025 (ministry view); PLAN section 8
    from gosplan.config import EnvConfig
    from gosplan.env.ministry import MinistryView

Framing = Literal["neutral", "historical"]
"""Prompt framing (PLAN section 7.4): `neutral` uses ordinary business vocabulary, `historical` uses
Soviet planning vocabulary. Same decision problem, same numbers, same response schema - only the
words differ. The framing contrast is the study's secondary measure."""

PayoffArm = Literal["baseline", "padding_dominated", "overfulfilment_optimal"]
"""Payoff arm (PLAN section 7.4), realised as an `EnvConfig`, never as a prompt instruction:

    baseline                the Phase-2 configuration as it stands
    padding_dominated       `audit_rate = 1`, `penalty_scale` large: forwarding padded numbers is
                            strictly worse for the ministry
    overfulfilment_optimal  `ratchet_lambda = 0`, `overfulfilment_slope` large: smoothing away
                            overfulfilment is strictly worse

Whether behaviour tracks this label is the study's primary measure."""

FRAMINGS: tuple[Framing, ...] = ("neutral", "historical")
"""The two framings of the PLAN section 7.4 design."""

PAYOFF_ARMS: tuple[PayoffArm, ...] = (
    "baseline",
    "padding_dominated",
    "overfulfilment_optimal",
)
"""The three payoff arms of the PLAN section 7.4 design."""

EPISODES_PER_CELL = 20
"""Episodes per (framing, payoff arm, model) cell (PLAN section 7.4)."""

MIN_MODELS = 2
"""Minimum number of distinct pinned models the design requires (PLAN section 7.4: ">= 2 models").
A single-model result is not a result: the study is about model behaviour in general, and one model
cannot separate a property of language models from a property of that model."""

MAX_RETRIES = 1
"""Retries allowed after an unparseable response (PLAN section 7.4: "strict-JSON responses with one
retry"). The retry re-sends the same prompt; it does not soften the schema, add an example of a
"good" answer, or hint at a value - any of which would leak the experimenter's expectation into the
measurement."""

FALLBACK_PASSTHROUGH = 1.0
"""The documented fallback action when both attempts fail to parse (PLAN section 7.4: "a documented
fallback action (`pi = 1`)"): forward every claim unchanged, i.e. a fully transparent ministry. It
is the neutral action - it adds no distortion of the experimenter's own - and every use of it is
logged and counted, because a high fallback rate is itself a finding about the model, not a
nuisance to be hidden."""


@dataclass(frozen=True)
class LLMMinistryConfig:
    """Pinned model identity, study cell and logging destination for one LLM ministry (sec. 7.4).

    Frozen and hashable; every field is written into the run manifest (CONTRACT rule 10), which
    names "LLM model ids and versions" explicitly. Owning WO: **WO-026**.
    """

    model_id: str
    """Provider-qualified model identifier, pinned to an exact version - not a moving alias. **No
    default**: the models are chosen by the lead when WO-035 is issued and there is no defensible
    placeholder."""

    model_version: str
    """The provider's version or snapshot string for `model_id`, recorded separately so a rerun can
    be shown to have used the same weights. **No default**, same reason."""

    framing: Framing
    """Which framing this ministry is prompted with."""

    payoff_arm: PayoffArm
    """Which payoff arm the environment is configured to. Stored here only for logging: the arm is
    realised by the `EnvConfig`, and the prompt must not mention it."""

    log_dir: Path
    """Directory for the prompt/completion log (`log_exchange`). One file per run, under the run
    directory of CONTRACT rule 10."""

    temperature: float | None = None
    """Sampling temperature, or `None` to use the provider's default. **No numeric default is
    asserted here**: PLAN section 7.4 does not fix one, and a temperature is a study choice that
    goes in the manifest rather than a specification."""

    max_retries: int = MAX_RETRIES
    """Retries after an unparseable response; the design value is `MAX_RETRIES` (one)."""

    fallback_passthrough: float = FALLBACK_PASSTHROUGH
    """The `pi` used when parsing fails after the retry; the design value is `FALLBACK_PASSTHROUGH`
    (1.0, forward unchanged)."""


@dataclass
class LLMMinistry:
    """A ministry whose forwarding decision is made by a pinned model (PLAN sections 7.4, 2.14).

    It stands in for `ministry_forward` (PLAN section 2.14): one instance per ministry per episode,
    called once per plan period with that ministry's `MinistryView`, returning the values it
    forwards upward. It is *not* an enterprise policy and does not implement `act` - it never
    chooses an `EnterpriseAction`, and the `Agent` protocol's list of implementations names it as
    part of the agent layer in the broad sense only. Which protocol it satisfies is settled by the
    Phase-2 spec revision (PLAN section 0, finding F14); until then this class is interface plus
    documented behaviour and its methods raise.

    The class is deliberately thin: build a prompt from the view, call the model, parse strictly,
    retry once, fall back to passthrough, log everything. Any judgement about what a "reasonable"
    forwarded number would be belongs to the model, not to this wrapper - a wrapper that clips,
    smooths or sanity-checks the model's numbers would be measuring itself.

    Owning WO: **WO-026** (adapter), **WO-035** (study harness).
    """

    cfg: LLMMinistryConfig
    """Pinned model identity, study cell and log destination."""

    client: object
    """The model client, injected. Typed `object` on purpose: no vendor SDK may appear in this
    interface, so the study can add a model without a signature change, and the skeleton imports
    nothing beyond the standard library and numpy. The implementation calls it behind
    `_complete`-style glue that WO-026 owns."""

    def forward(self, view: MinistryView, cfg: EnvConfig) -> Array:
        """Forward this ministry's enterprises' claims upward, as the model decides (sec. 7.4).

        Takes: `view`, the `MinistryView` of PLAN section 2.14 - `ministry_id`, `enterprise_ids`,
        `claims`, `targets`, `prev_forward`, `passthrough`, `t_period`; `cfg`, the environment
        configuration (read for context that is already public, never for a true quantity). Returns:
        the forwarded values `(n_i,)`, in the same units and enterprise order as `view.claims` -
        exactly the return type of the rule-based `ministry_forward`, so the two are drop-in
        substitutable.

        Sequence: `render_ministry_view(view, self.cfg.framing)`; one model call; `parse_forward_
        response`; on a parse failure, one retry with the identical prompt (`MAX_RETRIES`); on a
        second failure, `FALLBACK_PASSTHROUGH` - forward `view.claims` unchanged - with the fallback
        recorded. Every prompt, completion, retry and fallback goes to `log_exchange` (CONTRACT rule
        10).

        Constraints on the returned values: length and order must match `view.enterprise_ids`;
        non-finite or negative values are a parse failure, not a value to be clipped silently. The
        wrapper does not otherwise bound what the model returns - a wildly padded number is a
        measurement, and the environment's own bounds (CONTRACT rule 8) do the bounding.

        Owning WO: **WO-026**.
        """
        raise NotImplementedError("PLAN section 7.4 - implemented in WO-026")

    def reset(self) -> None:
        """Clear per-episode state.

        Takes: nothing. Returns: `None`. Clears any per-episode conversation state, so each episode
        is independent and the 20 episodes of a study cell are 20 samples rather than one long
        conversation. It does not clear the log.

        Owning WO: **WO-026**.
        """
        raise NotImplementedError("PLAN section 7.4 - implemented in WO-026")


def render_ministry_view(view: MinistryView, framing: Framing) -> str:
    """Render a `MinistryView` as the text the model sees (PLAN sections 7.4, 2.14).

    Takes: `view`; `framing`, selecting the vocabulary. Returns: a deterministic plain-text
    rendering carrying every field of the record and nothing else - the ministry's index, its
    enterprises, their claims `R_i`, their targets `T_i`, what this ministry forwarded last period
    (`prev_forward`), its `passthrough` and the period index - plus the response schema the reply
    must satisfy (see `parse_forward_response`).

    Requirements. *Deterministic*: the same view and framing render byte-identically, so two runs
    differ only through the model. *Complete and no more*: the rendering may not add a quantity the
    `MinistryView` does not contain - no true output, no stock, no welfare, no other ministry's
    numbers (CONTRACT rules 5 and 6 in spirit: the ministry sees what PLAN section 2.14 gives it).
    *Instruction-free*: it states the decision and the schema; it does not say what a good answer
    looks like, does not mention the payoff arm, and does not tell the model to pad, smooth or
    fulfil.

    The two framings differ in vocabulary only - `neutral` in ordinary business terms, `historical`
    in Soviet planning terms - over an identical numeric payload and an identical schema. The lead
    writes both prompt texts (WO-035) and they are versioned with the study, so a framing effect is
    attributable to the words that were actually used.

    Owning WO: **WO-026** (renderer), **WO-035** (the two prompt texts).
    """
    raise NotImplementedError("PLAN section 7.4 - implemented in WO-026")


def parse_forward_response(text: str, n_enterprises: int) -> tuple[Array, str]:
    """Parse a strict-JSON ministry reply (PLAN section 7.4).

    Takes: `text`, the model's completion; `n_enterprises`, the number of enterprises this ministry
    covers (`len(view.enterprise_ids)`). Returns: `(forwarded, justification)` - the forwarded
    values `(n_enterprises,)` in enterprise order, and the model's free-text justification, which is
    logged and analysed but never fed back into the environment.

    Strict means strict: the completion must be a single JSON object with a numeric array of
    exactly `n_enterprises` finite, non-negative entries and a string justification. No prose
    around the JSON, no trailing commentary, no repaired quotes, no partial credit, no coercion of a
    string to a number. Any deviation raises `ValueError`, which `LLMMinistry.forward` handles as
    one retry and then the documented fallback (`FALLBACK_PASSTHROUGH`). Leniency here would
    silently
    turn a parse failure into a behavioural datum, which is precisely the confound the strict-JSON
    rule and the logged fallback rate exist to keep visible.

    Owning WO: **WO-026**.
    """
    raise NotImplementedError("PLAN section 7.4 - implemented in WO-026")


def log_exchange(log_dir: Path, record: dict[str, object]) -> None:
    """Append one prompt/completion exchange to the study log (CONTRACT rule 10).

    Takes: `log_dir`, the destination directory (`LLMMinistryConfig.log_dir`); `record`, one
    exchange. Returns: `None`. Appends the record as one JSON object per line to
    `log_dir / "llm_exchanges.jsonl"`, flushed on each call.

    Each record carries at least: `model_id` and `model_version` (pinned, CONTRACT rule 10),
    `framing`, `payoff_arm`, `episode`, `t_period`, `ministry_id`, `attempt` (0 for the first call,
    1 for the retry), the full `prompt`, the full `completion`, the parsed `forwarded` values or the
    parse error, whether the fallback fired, the temperature, and any usage counters the provider
    returns. "Every prompt and completion logged" is the design's own words; the manipulation-check
    exchanges are logged the same way, tagged as such.

    Owning WO: **WO-026**.
    """
    raise NotImplementedError("CONTRACT rule 10 - implemented in WO-026")


def manipulation_check_prompt(framing: Framing) -> str:
    """The post-episode manipulation-check prompt (PLAN section 7.4).

    Takes: `framing`, so the check is asked in the same vocabulary the episode used. Returns: the
    prompt text asking the model, **in a fresh context**, to name the closest real-world analogue of
    the game it just played.

    Fresh context is the whole point: the question is what the model recognised, not what it will
    say after being reminded. The prompt must not mention Soviet planning, Gosplan, plan
    fulfilment, targets-and-bonuses or any other candidate answer - it asks an open question and the
    analysis stratifies on whether the free-text answer names Soviet planning, which is the
    covariate that separates retrieval from reasoning in the primary measure.

    The lead writes this text (WO-035) and it is versioned with the study.

    Owning WO: **WO-035** (text), **WO-026** (the seam that calls it).
    """
    raise NotImplementedError("PLAN section 7.4 - implemented in WO-035")


def run_manipulation_check(client: object, cfg: LLMMinistryConfig, episode: int) -> str:
    """Ask the manipulation-check question after one episode and log the answer (sec. 7.4).

    Takes: `client`, the same model client, called with a **fresh** context - no episode history, no
    system prompt from the episode, nothing carried over; `cfg`, for the pinned model identity, the
    framing and the log directory; `episode`, the episode index, for the log. Returns: the model's
    free-text answer verbatim, unparsed and unjudged.

    Classification of the answer - did it name Soviet planning? - is a separate, documented step in
    the study harness (WO-035), applied uniformly and reported with its own rule, so that the
    stratification of PLAN section 7.4 is reproducible from the logs alone. The exchange is written
    through `log_exchange` tagged as a manipulation check.

    Owning WO: **WO-026** (this seam), **WO-035** (the classification rule).
    """
    raise NotImplementedError("PLAN section 7.4 - implemented in WO-026")


__all__ = [
    "EPISODES_PER_CELL",
    "FALLBACK_PASSTHROUGH",
    "FRAMINGS",
    "MAX_RETRIES",
    "MIN_MODELS",
    "PAYOFF_ARMS",
    "Framing",
    "LLMMinistry",
    "LLMMinistryConfig",
    "PayoffArm",
    "log_exchange",
    "manipulation_check_prompt",
    "parse_forward_response",
    "render_ministry_view",
    "run_manipulation_check",
]
