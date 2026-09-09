from factory.pipeline import run
from factory.seed import Example
import json, tempfile
from pathlib import Path


def stub(prompt):
    return {"faithfulness": 9, "diversity": 8, "keep": True}


def test_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        s = Path(d) / "s.json"
        s.write_text(json.dumps([{"input": "x", "output": "y"}]))
        o = Path(d) / "o.jsonl"
        n = run(s, o, stub)
        assert n >= 1
        assert o.exists()
