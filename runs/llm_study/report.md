# LLM ministry study (WO-035) - NOT RUN

This container has no Anthropic SDK and no credentials (spec/P3_REVISION.md S5). The harness is implemented and unit-tested with a scripted client. To run it:

```
uv pip install anthropic
export ANTHROPIC_API_KEY=...   # or `ant auth login`
uv run python -m gosplan.experiments.llm_study claude-opus-5 claude-sonnet-5
```

Estimated cost (S5.4): about 50 ministry decisions per episode x 20 episodes x 2 framings x 3 arms x 2 models, about 16M tokens, plus the dominance check (no model calls). G4's 'LLM study' condition cannot pass until this is run.
