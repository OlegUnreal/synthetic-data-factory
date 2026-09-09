"""OpenAI-backed expansion + quality filter.

`expand_seed` and `filter_candidates` accept an injected llm(prompt) -> str.
This module provides the OpenAI implementation with retries.
"""
from __future__ import annotations

import re
from typing import Callable

from .config import SETTINGS
from .logging_config import get_logger

log = get_logger(__name__)

EXPAND_SYS = (
    "You generate diverse training examples in the same style as the seed. "
    "Return a JSON array of objects with keys 'input' and 'output'. No markdown."
)
FILTER_SYS = (
    "Score this training example 0-10 on correctness and clarity. "
    "Reply with only the number."
)


def _call(prompt: str, system: str, model: str | None = None) -> str:
    from openai import OpenAI, APIError, RateLimitError, APITimeoutError

    model = model or SETTINGS.model
    client = OpenAI(api_key=SETTINGS.openai_api_key, timeout=SETTINGS.timeout)
    last: Exception | None = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                temperature=SETTINGS.temperature,
                timeout=SETTINGS.timeout,
            )
            return (resp.choices[0].message.content or "").strip()
        except (APIError, RateLimitError, APITimeoutError) as exc:
            last = exc
            log.warning("llm_retry", extra={"attempt": attempt + 1, "error": type(exc).__name__})
            continue
    raise RuntimeError(f"LLM call failed after retries: {last}") from last


def make_llm(model: str | None = None) -> Callable[[str], str]:
    if not SETTINGS.has_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    m = model or SETTINGS.model
    return lambda p: _call(p, EXPAND_SYS, m)


def make_filter(model: str | None = None) -> Callable[[str], int]:
    if not SETTINGS.has_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    m = model or SETTINGS.model

    def score(text: str) -> int:
        raw = _call(text, FILTER_SYS, m)
        mm = re.search(r"\d+", raw)
        return int(mm.group()) if mm else 0

    return score


def live_run(seeds_path, out_path, threshold: int | None = None) -> int:
    from .pipeline import run
    import factory.filter as fmod
    llm = make_llm()
    fltr = make_filter()
    fmod.score = fltr  # type: ignore
    return run(seeds_path, out_path, llm, min_score=threshold or SETTINGS.min_score)
