"""Real LLM expansion + quality filter for the synthetic data factory.

`expand_seed` and `filter_candidates` accept an injected llm(prompt) -> str.
This module provides the OpenAI implementation.
"""
from __future__ import annotations

import json
import os
import re
from typing import Callable


EXPAND_SYS = (
    "You generate diverse training examples in the same style as the seed. "
    "Return a JSON array of objects with keys 'input' and 'output'. No markdown."
)
FILTER_SYS = (
    "Score this training example 0-10 on correctness and clarity. "
    "Reply with only the number."
)


def _call(prompt: str, system: str, model: str = "gpt-4o-mini") -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        temperature=0.7,
        timeout=30,
    )
    return (resp.choices[0].message.content or "").strip()


def make_llm(model: str = "gpt-4o-mini") -> Callable[[str], str]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")
    return lambda p: _call(p, EXPAND_SYS, model)


def make_filter(model: str = "gpt-4o-mini") -> Callable[[str], int]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")

    def score(text: str) -> int:
        raw = _call(text, FILTER_SYS, model)
        m = re.search(r"\d+", raw)
        return int(m.group()) if m else 0

    return score


def live_run(seeds_path, out_path, threshold: int = 6) -> int:
    from .pipeline import run
    llm = make_llm()
    fltr = make_filter()
    import factory.filter as fmod
    fmod.score = fltr  # type: ignore
    return run(seeds_path, out_path, llm, min_score=threshold)
