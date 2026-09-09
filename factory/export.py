"""Export to JSONL in chat-completion format."""
from __future__ import annotations

import json
from pathlib import Path

from .seed import Example


def to_jsonl(examples: list[Example], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for e in examples:
            f.write(json.dumps({
                "messages": [
                    {"role": "user", "content": e.input},
                    {"role": "assistant", "content": e.output},
                ]
            }, ensure_ascii=False) + "\n")
