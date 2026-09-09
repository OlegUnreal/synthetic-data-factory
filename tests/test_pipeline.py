import json, tempfile
from pathlib import Path

from factory.pipeline import run
from factory.seed import Example, load_seeds
from factory.filter import filter_candidates


def stub(prompt):
    if "faithfulness" in prompt:
        return json.dumps({"faithfulness": 9, "diversity": 8, "keep": True})
    return json.dumps([{"input": "x2", "output": "y2"}, {"input": "x3", "output": "y3"}])


def test_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        s = Path(d) / "s.json"
        s.write_text(json.dumps([{"input": "x", "output": "y"}]))
        o = Path(d) / "o.jsonl"
        n = run(s, o, stub)
        assert n >= 1
        assert o.exists()
        lines = o.read_text().strip().splitlines()
        assert all(json.loads(l)["messages"][0]["role"] == "user" for l in lines)


def test_empty_seeds():
    with tempfile.TemporaryDirectory() as d:
        s = Path(d) / "s.json"
        s.write_text("[]")
        o = Path(d) / "o.jsonl"
        assert run(s, o, stub) == 0
        assert o.read_text().strip() == ""


def test_filter_drops_low_score():
    seeds = [Example("a", "b")]
    cands = [Example("a", "b2"), Example("c", "d")]
    kept = filter_candidates(cands, seeds, lambda p: json.dumps({"faithfulness": 3, "keep": False}), min_score=6)
    assert kept == []


def test_load_seeds_rejects_bad_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not-json")
    try:
        load_seeds(p)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_load_seeds_rejects_non_list(tmp_path):
    p = tmp_path / "obj.json"
    p.write_text('{"a": 1}')
    try:
        load_seeds(p)
        assert False, "expected ValueError"
    except ValueError:
        pass
