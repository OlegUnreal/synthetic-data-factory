"""Live run with a real LLM.

Usage: OPENAI_API_KEY=... python -m factory.demo_llm seeds.json out.jsonl
"""
from __future__ import annotations

import sys
from pathlib import Path

from .llm import live_run


def main() -> None:
    seeds = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("seeds.json")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("out.jsonl")
    n = live_run(seeds, out)
    print(f"wrote {n} examples to {out}")


if __name__ == "__main__":
    main()
