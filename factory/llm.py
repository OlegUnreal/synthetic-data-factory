"""OpenAI-backed LLM callables for the two model roles in the pipeline.

Both are the same shape -- ``prompt -> str`` -- so they are interchangeable:

* :func:`make_llm`  -> the expander (feeds :func:`factory.expand.expand_seed`)
* :func:`make_filter` -> the judge (feeds :func:`factory.filter.filter_candidates`)

Parsing, tolerance for markdown fences and the keep/drop decision live in the
stage modules, not here; this module only talks to the API (with retries).
Nothing in here runs during tests.
"""
from __future__ import annotations

from typing import Callable

from .config import SETTINGS
from .logging_config import get_logger

log = get_logger(__name__)

EXPAND_SYS = (
    "You generate diverse training examples in the same style as the seed. "
    "Return a JSON array of objects with keys 'input' and 'output'. No markdown."
)
JUDGE_SYS = (
    "You are an independent data-quality judge. Reply with ONLY a JSON object "
    "of the form {\"faithfulness\": int, \"diversity\": int, \"keep\": bool}. "
    "No markdown, no commentary."
)
# Old name kept so anything importing it keeps working; the contract it described
# ("reply with only the number") is not what factory.filter parses.
FILTER_SYS = JUDGE_SYS


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
    """Expander: takes a prompt, returns raw model text (a JSON array)."""
    if not SETTINGS.has_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    m = model or SETTINGS.model
    return lambda p: _call(p, EXPAND_SYS, m)


def make_filter(model: str | None = None) -> Callable[[str], str]:
    """Judge: takes a prompt, returns raw model text (a JSON verdict object).

    This is deliberately the *same* shape as :func:`make_llm` -- ``int`` came back
    from the old version, which no caller could feed to
    :func:`factory.filter.filter_candidates` (it parses the reply as JSON). The
    extraction/retry logic lives in :mod:`factory.filter`, not here, so one model
    can serve both roles.
    """
    if not SETTINGS.has_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    m = model or SETTINGS.model
    return lambda p: _call(p, JUDGE_SYS, m)


def score_verdict(raw: str) -> int:
    """Numeric faithfulness score from a judge reply (0 when unparsable)."""
    from .filter import _parse_verdict

    try:
        return int(_parse_verdict(raw).get("faithfulness", 0))
    except (TypeError, ValueError):
        return 0


def live_run(seeds_path, out_path, threshold: int | None = None, *, judge_model: str | None = None) -> int:
    """One-shot real run: expander and (optionally separate) judge model."""
    from .pipeline import run

    llm = make_llm()
    fltr = make_filter(judge_model)
    return run(
        seeds_path, out_path, llm,
        min_score=threshold or SETTINGS.min_score,
        llm_filter=fltr,
    )
