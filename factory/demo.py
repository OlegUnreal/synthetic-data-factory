"""Offline demo with stub LLM + stub embeddings."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from .pipeline import run
from .seed import Example


def stub_llm(prompt: str):
    if "paraphrase" in prompt or "variants" in prompt:
        return [
            {"input": "How do I reset my password?", "output": "Use the forgot-password link on the login page."},
            {"input": "Password reset steps?", "output": "Click 'forgot password', check your email, set a new one."},
        ]
    return {"faithfulness": 8, "diversity": 7, "keep": True}


def main() -> None:
    with tempfile.TemporaryDirectory() as d:
        seeds = Path(d) / "seeds.json"
        seeds.write_text(json.dumps([
            {"input": "How do I reset my password?", "output": "Use the forgot-password link."}
        ]), encoding="utf-8")
        out = Path(d) / "out.jsonl"
        n = run(seeds, out, stub_llm)
        print(f"produced {n} examples -> {out}")


if __name__ == "__main__":
    main()
