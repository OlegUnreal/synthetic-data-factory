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
    if not path.exists():
        raise FileNotFoundError(f"seeds file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError(f"seeds file must be a JSON array, got {type(data).__name__}")
    out: list[Example] = []
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"seed[{i}] must be an object, got {type(row).__name__}")
        if "input" not in row or "output" not in row:
            raise ValueError(f"seed[{i}] missing 'input' or 'output' keys")
        out.append(Example(str(row["input"]), str(row["output"]), str(row.get("tag", ""))))
    return out


def save_seeds(examples: list[Example], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([e.__dict__ for e in examples], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
