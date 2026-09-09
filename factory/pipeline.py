"""Full pipeline: seed → expand → filter → dedup → export."""
from __future__ import annotations

from pathlib import Path

from .dedup import dedup
from .expand import expand_seed
from .export import to_jsonl
from .filter import filter_candidates
from .seed import Example, load_seeds


def run(seeds_path: Path, out_path: Path, llm) -> int:
    seeds = load_seeds(seeds_path)
    pool: list[Example] = list(seeds)
    for s in seeds:
        pool.extend(expand_seed(s, llm))
    pool = filter_candidates(pool, seeds, llm)
    pool = dedup(pool)
    to_jsonl(pool, out_path)
    return len(pool)
