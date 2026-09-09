"""CLI: python -m factory "seeds.json" [out.jsonl]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import SETTINGS
from .logging_config import setup_logging, get_logger
from .pipeline import run

log = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(prog="factory", description="Synthetic data factory")
    parser.add_argument("seeds", type=Path, help="Path to seeds JSON file")
    parser.add_argument("out", type=Path, nargs="?", default=Path("out.jsonl"), help="Output JSONL path")
    parser.add_argument("--threshold", type=int, default=None, help="Min faithfulness score (0-10)")
    args = parser.parse_args(argv)

    if not SETTINGS.has_key:
        log.error("missing_api_key", extra={"hint": "set OPENAI_API_KEY or create a .env"})
        return 2
    if not args.seeds.exists():
        log.error("seeds_not_found", extra={"path": str(args.seeds)})
        return 2

    try:
        n = run(args.seeds, args.out, None, min_score=args.threshold or SETTINGS.min_score)  # type: ignore
    except Exception as exc:  # noqa: BLE001
        log.exception("pipeline_failed", extra={"error": type(exc).__name__})
        return 1
    log.info("done", extra={"examples": n, "out": str(args.out)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
