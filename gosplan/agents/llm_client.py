"""Model client for the LLM ministry study (WO-035; spec/P3_REVISION.md S5.4).

The one place the Anthropic SDK is imported, lazily, so `gosplan` imports without it. The adapter
exposes the `complete(prompt, temperature=...) -> (text, usage)` seam that
`gosplan.agents.llm_ministry` calls, and nothing else.

Study discipline (S5.4):
- The model is pinned by the id the owner passes (default `claude-opus-5`).
- Temperature is sent only when configured; current Opus models reject it.
- A refusal (`stop_reason == "refusal"`) returns an empty completion, which the ministry adapter's
  strict parser rejects, so the documented passthrough fallback fires and is counted.
- Server-side model fallbacks are deliberately NOT enabled: routing a declined decision to another
  model would change the model under study mid-episode.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_MODEL = "claude-opus-5"
MAX_TOKENS = 2048
"""A forwarding decision is a short JSON object; 2048 leaves room for adaptive thinking's visible
remainder without truncating the reply."""


@dataclass
class AnthropicClient:
    """`complete(prompt, temperature=None)` over the Messages API, one fresh single-turn request per
    call (no conversation state, so a manipulation check is a fresh context by construction)."""

    model: str = DEFAULT_MODEL
    max_tokens: int = MAX_TOKENS
    _client: object = field(default=None, repr=False)

    def _sdk(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def complete(self, prompt: str, temperature: float | None = None) -> tuple[str, dict]:
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        response = self._sdk().messages.create(**kwargs)
        usage = {
            "input_tokens": int(response.usage.input_tokens),
            "output_tokens": int(response.usage.output_tokens),
            "stop_reason": str(response.stop_reason),
            "model": str(response.model),
            "request_id": str(getattr(response, "_request_id", "")),
        }
        if response.stop_reason == "refusal":
            return "", usage
        text = "".join(block.text for block in response.content if block.type == "text")
        return text, usage


def credentials_available() -> bool:
    """Whether the SDK is installed and an Anthropic credential source is configured (an API key,
    an auth token or a profile). A missing SDK or credential means the study is NOT RUN (S5)."""
    import importlib.util
    import os

    if importlib.util.find_spec("anthropic") is None:
        return False
    from pathlib import Path

    env = any(
        os.environ.get(name)
        for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE")
    )
    profile = (Path.home() / ".config" / "anthropic").is_dir()  # `ant auth login` profiles
    return env or profile
