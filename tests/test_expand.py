from factory.expand import _parse_variants, expand_seed
from factory.seed import Example


def test_parse_clean_json():
    raw = '[{"input": "a", "output": "b"}]'
    assert _parse_variants(raw) == [{"input": "a", "output": "b"}]


def test_parse_fenced():
    raw = '```json\n[{"input": "a", "output": "b"}]\n```'
    assert _parse_variants(raw) == [{"input": "a", "output": "b"}]


def test_parse_garbage():
    assert _parse_variants("not json at all") == []


def test_expand_uses_parser():
    seed = Example("in", "out")
    out = expand_seed(seed, lambda p: '[{"input": "in2", "output": "out2"}]')
    assert len(out) == 1
    assert out[0].input == "in2"
    assert out[0].tag == "expanded"
