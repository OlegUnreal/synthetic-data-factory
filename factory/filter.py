"""Quality filter: independent judge scores each candidate."""
from __future__ import annotations

from .seed import Example

JUDGE_PROMPT = """Score 0-10 for (a) faithfulness to the seed's intent and (b) diversity vs existing set.
Return JSON {{"faithfulness": int, "diversity": int, "keep": bool}}.

Candidate: {cand}
Existing: {existing}
"""


def filter_candidates(candidates: list[Example], existing: list[Example], llm, min_score: int = 6) -> list[Example]:
    kept: list[Example] = []
    for c in candidates:
        verdict = llm(JUDGE_PROMPT.format(cand=c.__dict__, existing=[e.__dict__ for e in existing]))
        if verdict.get("keep") and verdict.get("faithfulness", 0) >= min_score:
            kept.append(c)
    return kept
