"""Quality filter: independent judge scores each candidate."""
from __future__ import annotations

import json
import re

from .seed import Example

JUDGE_PROMPT = """Score 0-10 for (a) faithfulness to the seed's intent and (b) diversity vs existing set.
Return ONLY JSON: {{"faithfulness": int, "diversity": int, "keep": bool}}. No markdown.

Candidate: {cand}
Existing: {existing}
"""

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_verdict(raw: str) -> dict:
    if not raw:
        return {}
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    m = _JSON_OBJ_RE.search(text)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def filter_candidates(candidates: list[Example], existing: list[Example], llm, min_score: int = 6) -> list[Example]:
    kept: list[Example] = []
    seen_inputs = {e.input for e in existing}
    for c in candidates:
        if c.input in seen_inputs:
            continue
        verdict = _parse_verdict(
            llm(JUDGE_PROMPT.format(cand=c.__dict__, existing=[e.__dict__ for e in existing]))
        )
        faith = verdict.get("faithfulness", 0)
        keep = verdict.get("keep", False)
        try:
            faith = int(faith)
        except (TypeError, ValueError):
            faith = 0
        if keep and faith >= min_score:
            kept.append(c)
            seen_inputs.add(c.input)
    return kept
