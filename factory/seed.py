"""Seed store: load hand-written examples."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Example:
    input: str
    output: str
    tag: str = ""


def load_seeds(path: Path) -> list[Example]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Example(**row) for row in data]


def save_seeds(examples: list[Example], path: Path) -> None:
    path.write_text(
        json.dumps([e.__dict__ for e in examples], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
