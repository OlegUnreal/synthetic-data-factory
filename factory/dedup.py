"""Near-duplicate removal: real embeddings, ANN candidates, exact verification.

The original implementation embedded text with ``hash(ch trin) % 64`` counters --
not an embedding, and not order-stable across processes (``PYTHONHASHSEED``) -- and
then compared every pair in ``O(n^2)`` Python loops. This module keeps the same
public entry point (:func:`dedup`) and the same "first occurrence wins" semantics,
and replaces the machinery underneath:

1. :mod:`factory.embeddings` turns text into L2-normalised vectors (probabilistic
   LSA over word + char TF-IDF, with a deterministic signed-hashing fallback).
2. :mod:`factory.ann` proposes *candidate* pairs with MinHash+LSH banding, an IVF
   index, or the union of both -- sub-quadratic in the corpus size.
3. Every candidate is verified with the **exact cosine** of the real embedding, and
   transitive groups are collapsed with union-find, so an exact duplicate *and* a
   reworded near-duplicate end up in one cluster with one survivor.

Because step 3 is exact, an ANN miss can only leave a duplicate in the data; it can
never delete a unique example. That asymmetry is the reason this pipeline is allowed
to use ANN at all, and :func:`factory.ann.benchmark_ann` measures how much recall is
actually lost at each setting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from .ann import DEFAULT_LSH_MARGIN, STRATEGIES, candidate_pairs
from .embeddings import Embedder, as_unit_rows, build_embedder
from .seed import Example
from .text import content_hash, shingle_set

__all__ = [
    "dedup",
    "dedup_audit",
    "find_duplicate_groups",
    "DedupAudit",
    "DEFAULT_DEDUP_THRESHOLD",
    "collapse_groups",
    "_embed",
]

DEFAULT_DEDUP_THRESHOLD = 0.92
#: what one example's "identity" is for similarity purposes
TextOf = Callable[[Example], str]


def _text_of(example: Example) -> str:
    return str(example.input)


def _embed(texts: Sequence[str], *, backend: str | None = None, dim: int = 256) -> np.ndarray:
    """Real replacement for the old char-trigram stub: fitted embeddings, unit rows.

    Kept under the original name (and with compatible defaults) because callers
    imported it; it now returns LSA/hashing embeddings rather than hash counters.
    """
    docs = [str(t) for t in texts]
    if not docs:
        return np.zeros((0, dim), dtype=np.float32)
    return build_embedder(backend, corpus=docs, dim=dim).embed(docs)


# --------------------------------------------------------------------------- #
# union-find over verified duplicate pairs
# --------------------------------------------------------------------------- #
class _DisjointSet:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, i: int) -> int:
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[i] != root:  # path compression
            self.parent[i], i = root, self.parent[i]
        return root

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri == rj:
            return
        if self.rank[ri] < self.rank[rj]:
            ri, rj = rj, ri
        self.parent[rj] = ri
        if self.rank[ri] == self.rank[rj]:
            self.rank[ri] += 1


def _pair_sim(x: np.ndarray, i: int, j: int) -> float:
    """Exact cosine of two unit rows, clamped against float drift."""
    return float(np.clip(np.dot(x[i], x[j]), -1.0, 1.0))


def collapse_groups(n: int, pairs: Sequence[tuple[int, int]]) -> list[list[int]]:
    """Transitive closure of ``pairs`` as ``[member, ...]`` groups, smallest id first.

    Union-find rather than pairwise marking, so ``A~B`` and ``B~C`` collapses even
    when ``A!~C``: three rewordings of one seed become one example, which is what a
    diversity metric should count as one.
    """
    ds = _DisjointSet(n)
    for i, j in pairs:
        ds.union(int(i), int(j))
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(ds.find(i), []).append(i)
    return [sorted(m) for m in sorted(groups.values(), key=lambda ids: ids[0])]


def find_duplicate_groups(
    examples: Sequence[Example],
    threshold: float = DEFAULT_DEDUP_THRESHOLD,
    *,
    embedder: Embedder | None = None,
    backend: str | None = None,
    dim: int = 256,
    seed: int = 13,
    strategy: str = "ivf",
    num_perm: int = 128,
    n_lists: int | None = None,
    n_probe: int = 4,
    max_exact_fallback: int = 256,
    lsh_margin: float = DEFAULT_LSH_MARGIN,
    text_of: TextOf = _text_of,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Index ``examples`` and return duplicate index groups plus run diagnostics.

    A group is a set of mutually (transitively) near-duplicate positions; groups of
    size 1 are unique examples. ``dict`` returned alongside records the embedder,
    candidate volume and how many candidates the exact check rejected.
    """
    texts = [text_of(e) for e in examples]
    n = len(texts)
    if n == 0:
        return [], {
            "groups": 0,
            "duplicate_groups": 0,
            "largest_group": 1,
            "candidate_pairs": 0,
            "verified_pairs": 0,
            "candidates_rejected": 0,
            "used_exact_fallback": True,
            "_heads": {},
            "_vectors": np.zeros((0, 1), dtype=np.float32),
            "_sim_by_pair": {},
        }
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError(f"threshold must be a cosine in [0, 1], got {threshold}")

    emb = embedder or build_embedder(backend, corpus=texts, dim=dim, seed=seed)
    if not emb.fitted:
        emb.fit(texts)
    x = as_unit_rows(emb.embed(texts))

    sets = [sorted(shingle_set(t)) for t in texts] if strategy in ("lsh", "hybrid") else None
    pairs, info = candidate_pairs(
        x,
        sets=sets,
        threshold=float(threshold),
        strategy=strategy,
        num_perm=num_perm,
        n_lists=n_lists,
        n_probe=n_probe,
        seed=seed,
        max_exact_fallback=max_exact_fallback,
        lsh_margin=lsh_margin,
    )

    # candidates are a *proposal*: the decision is the exact cosine of real vectors
    verified: list[tuple[int, int, float]] = []
    rejected = 0
    for i, j in sorted(pairs):
        sim = float(np.clip(x[i] @ x[j], -1.0, 1.0))
        if sim >= float(threshold):
            verified.append((i, j, sim))
        else:
            rejected += 1

    groups = collapse_groups(n, [(i, j) for i, j, _ in verified])
    sim_by_pair = {(i, j): s for i, j, s in verified}
    info.update(
        {
            "embedder": emb.name(),
            "embedding_backend": emb.backend,
            "dim": int(x.shape[1]),
            "threshold": round(float(threshold), 6),
            "verified_pairs": len(verified),
            "candidates_rejected": rejected,
            "strategies": list(STRATEGIES),
        }
    )
    dup_groups = [g for g in groups if len(g) > 1]
    info["groups"] = len(groups)
    info["duplicate_groups"] = len(dup_groups)
    info["largest_group"] = max((len(g) for g in dup_groups), default=1)
    # nearest surviving representative per dropped index, for the audit trail
    keepers: dict[int, tuple[int, float]] = {}
    for g in dup_groups:
        for i in g[1:]:
            earlier = [
                (j, sim_by_pair.get((j, i)) if (j, i) in sim_by_pair else _pair_sim(x, i, j))
                for j in g
                if j < i
            ]
            keepers[i] = max(earlier, key=lambda t: t[1])
    info["_vectors"] = x
    info["_heads"] = keepers
    info["_sim_by_pair"] = sim_by_pair
    return groups, info


# --------------------------------------------------------------------------- #
# audit record
# --------------------------------------------------------------------------- #
@dataclass
class DedupAudit:
    """What dedup did, in a form that belongs in the run manifest."""

    threshold: float
    strategy: str
    embedder: str
    dim: int
    n_input: int
    n_kept: int
    n_dropped: int
    candidate_pairs: int
    verified_pairs: int
    candidates_rejected: int
    duplicate_groups: int
    largest_group: int
    used_exact_fallback: bool
    dropped: list[dict[str, Any]] = field(default_factory=list)
    index_info: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "strategy": self.strategy,
            "embedder": self.embedder,
            "dim": self.dim,
            "n_input": self.n_input,
            "n_kept": self.n_kept,
            "n_dropped": self.n_dropped,
            "drop_rate": round(self.n_dropped / self.n_input, 6) if self.n_input else 0.0,
            "candidate_pairs": self.candidate_pairs,
            "verified_pairs": self.verified_pairs,
            "candidates_rejected": self.candidates_rejected,
            "candidate_precision": round(
                self.verified_pairs / self.candidate_pairs, 6
            ) if self.candidate_pairs else 1.0,
            "duplicate_groups": self.duplicate_groups,
            "largest_group": self.largest_group,
            "used_exact_fallback": self.used_exact_fallback,
            "dropped": list(self.dropped),
            "index": _clean(self.index_info),
        }

    def summary(self) -> str:
        return (
            f"dedup {self.n_input} -> {self.n_kept} "
            f"({self.n_dropped} dropped, threshold={self.threshold:.3f}, "
            f"strategy={self.strategy}, embedder={self.embedder}, "
            f"{self.duplicate_groups} duplicate groups)"
        )


def _clean(value: Any) -> Any:
    """Strip numpy/private entries so the audit is JSON-serialisable."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items() if not k.startswith("_") and k != "strategies"}
    if isinstance(value, (list, tuple, set)):
        return [_clean(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def dedup_audit(
    examples: Sequence[Example],
    threshold: float = DEFAULT_DEDUP_THRESHOLD,
    **kwargs: Any,
) -> tuple[list[Example], DedupAudit]:
    """:func:`dedup` plus the audit trail (drops, similarities, index stats)."""
    items = list(examples)
    strategy = str(kwargs.pop("strategy", "ivf"))
    groups, info = find_duplicate_groups(items, threshold, strategy=strategy, **kwargs)
    kept = [items[i] for i in (g[0] for g in groups)]
    heads: dict[int, tuple[int, float]] = info["_heads"]
    rows = []
    for i in sorted(i for g in groups for i in g[1:]):
        rep, sim = heads.get(i, (-1, 0.0))
        rows.append(
            {
                "index": int(i),
                "kept_index": int(rep),
                "similarity": round(float(sim), 6),
                "input_hash": content_hash(str(items[i].input))[:12],
                "preview": str(items[i].input)[:80],
            }
        )
    audit = DedupAudit(
        threshold=round(float(threshold), 6),
        strategy=strategy,
        embedder=str(info.get("embedder", "")),
        dim=int(info.get("dim", 0)),
        n_input=len(items),
        n_kept=len(kept),
        n_dropped=len(rows),
        candidate_pairs=int(info.get("candidate_pairs", 0)),
        verified_pairs=int(info.get("verified_pairs", 0)),
        candidates_rejected=int(info.get("candidates_rejected", 0)),
        duplicate_groups=int(info.get("duplicate_groups", 0)),
        largest_group=int(info.get("largest_group", 1)),
        used_exact_fallback=bool(info.get("used_exact_fallback", False)),
        dropped=rows,
        index_info={k: v for k, v in info.items() if k != "_vectors"},
    )
    return kept, audit


def dedup(
    examples: list[Example],
    threshold: float = DEFAULT_DEDUP_THRESHOLD,
    *,
    embedder: Embedder | None = None,
    strategy: str = "ivf",
    **kwargs: Any,
) -> list[Example]:
    """Drop near-duplicates, keeping the first example of each cluster.

    ``threshold`` is a cosine on real embeddings. Signature-compatible with the
    previous ``dedup(examples, threshold)`` call sites; ``embedder`` and
    ``strategy`` are additive so a caller can reuse one fitted embedder across
    stages (and force ``exact`` when it wants a guarantee, not a recall estimate).
    """
    kept, _ = dedup_audit(
        examples, threshold, embedder=embedder, strategy=strategy, **kwargs
    )
    return kept
