from factory.dedup import dedup
from factory.seed import Example


def test_removes_near_dupes():
    ex = [
        Example("hello world", "a"),
        Example("hello wordl", "b"),  # near-duplicate
        Example("completely different topic about cars", "c"),
    ]
    out = dedup(ex, threshold=0.5)
    assert len(out) >= 1
