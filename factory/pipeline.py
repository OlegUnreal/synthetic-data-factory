"""Full pipeline: seed → expand → filter → dedup → export."""
from __future__ import annotations

from pathlib import Path

from .dedup import dedup
from .expand import expand_seed
from .export import to_jsonl
from .filter import filter_candidates
from .seed import Example, load_seeds


def run(seeds_path: Path, out_path: Path, llm, min_score: int = 6) -> int:
    seeds = load_seeds(seeds_path)
    if not seeds:
        to_jsonl([], out_path)
        return 0
    pool: list[Example] = list(seeds)
    for s in seeds:
        try:
            pool.extend(expand_seed(s, llm))
        except Exception:  # noqa: BLE001
            continue
    try:
        pool = filter_candidates(pool, seeds, llm, min_score=min_score)
    except Exception:  # noqa: BLE001
        pool = list(seeds)
    pool = dedup(pool)
    to_jsonl(pool, out_path)
    return len(pool)
