from factory.filter import _parse_verdict, filter_candidates
from factory.seed import Example
import json


def test_parse_verdict():
    assert _parse_verdict('{"faithfulness": 8, "keep": true}')["keep"] is True
    assert _parse_verdict("nope") == {}


def test_filter_skips_existing_inputs():
    seeds = [Example("a", "b")]
    cands = [Example("a", "b2")]  # duplicate input
    kept = filter_candidates(cands, seeds, lambda p: json.dumps({"faithfulness": 9, "keep": True}))
    assert kept == []
