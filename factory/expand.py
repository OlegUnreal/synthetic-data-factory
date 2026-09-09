"""Expansion: generate paraphrases + edge cases from seeds."""
from __future__ import annotations

import json
import re

from .seed import Example


EXPAND_PROMPT = """Given this example, produce {n} variants:
1. a paraphrase (same meaning, different words)
2. an edge case (unusual but valid input)
3. an adversarial case (tries to break the expected behavior)
Return ONLY a JSON array of objects with keys 'input' and 'output'. No markdown, no commentary.

Seed input: {inp}
Seed output: {out}
"""


_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _parse_variants(raw: str) -> list[dict]:
    """Best-effort JSON extraction: tolerate markdown fences and prose around the array."""
    if not raw:
        return []
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    m = _JSON_ARRAY_RE.search(text)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    out: list[dict] = []
    for item in data:
        if isinstance(item, dict) and "input" in item and "output" in item:
            out.append({"input": str(item["input"]), "output": str(item["output"])})
    return out


def expand_seed(seed: Example, llm, n: int = 3) -> list[Example]:
    raw = llm(EXPAND_PROMPT.format(n=n, inp=seed.input, out=seed.output))
    return [Example(v["input"], v["output"], tag="expanded") for v in _parse_variants(raw)]
