"""Expansion: generate paraphrases + edge cases from seeds."""
from __future__ import annotations

from .seed import Example


EXPAND_PROMPT = """Given this example, produce {n} variants:
1. a paraphrase (same meaning, different words)
2. an edge case (unusual but valid input)
3. an adversarial case (tries to break the expected behavior)
Return JSON list of {{"input": ..., "output": ...}}.

Seed input: {inp}
Seed output: {out}
"""


def expand_seed(seed: Example, llm, n: int = 3) -> list[Example]:
    raw = llm(EXPAND_PROMPT.format(n=n, inp=seed.input, out=seed.output))
    # In production, parse the JSON the LLM returns.
    return [Example(v["input"], v["output"], tag="expanded") for v in raw]
