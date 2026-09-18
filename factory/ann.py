"""Sub-quadratic similarity search, plus the exact brute-force reference.

Near-duplicate detection is a range/top-k search problem. The old code did an
O(n^2) Python double loop over *fake* vectors. This module keeps brute force as the
reference implementation and adds two ANN structures over the real embeddings from
:mod:`factory.embeddings`:

``MinHashLSH``
    MinHash signatures (BLAKE2b-hashed shingles, odd multiply-add permutations with
    64-bit overflow) + LSH banding. Bands/rows are chosen so the S-curve
    ``1 - (1 - s^rows)^bands`` crosses 0.5 at the requested threshold. Build cost is
    ``O(P * S)`` (``P`` permutations, ``S`` total shingles) and bucketing is
    ``O(n * bands)`` -- linear in documents. The quadratic term only survives inside
    buckets, which stay small for a sane threshold.
``IvfIndex``
    Deterministic numpy k-means++ coarse quantiser with cosine assignment and
    inverted posting lists; a query probes its ``n_probe`` nearest centroids and
    re-scans those lists exactly. Build ``O(n * d * iters)``, query
    ``O(nprobe * avg_list * d)`` instead of ``O(n * d)``.

Both are *candidate generators*: every candidate pair is verified with the exact
cosine of the real embedding. An ANN miss can therefore only cost recall (a
duplicate that survives), never precision (a false drop) -- the only safe failure
mode for a data-generation pipeline. :func:`benchmark_ann` measures that recall
against brute force, and :func:`candidate_pairs` is what :mod:`factory.dedup` calls.
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .embeddings import as_unit_rows
from .text import hash64

__all__ = [
    "MinHashLSH",
    "IvfIndex",
    "IvfStats",
    "AnnBench",
    "PairList",
    "exact_duplicate_pairs",
    "exact_topk",
    "exact_sim_blocks",
    "candidate_pairs",
    "pairs_recall",
    "pairs_to_adjacency",
    "benchmark_ann",
    "select_band_params",
    "jaccard_from_cosine",
    "DEFAULT_LSH_MARGIN",
    "STRATEGIES",
    "UINT64_MAX",
]

PairList = list[tuple[int, int, float]]
UINT64_MAX = np.iinfo(np.uint64).max
STRATEGIES = ("exact", "lsh", "ivf", "hybrid")
MISSING_ID = -1
MISSING_SIM = -2.0


def _plain(v: Any) -> Any:
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def _ratio(num: int, den: int) -> float:
    """No ground-truth positives means nothing to find: perfect by convention."""
    return num / den if den else 1.0


def _f1(recall: float, precision: float) -> float:
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


# --------------------------------------------------------------------------- #
# exact reference
# --------------------------------------------------------------------------- #
def exact_sim_blocks(mat: np.ndarray, chunk: int = 512):
    """Yield ``(row_start, row_end, sim_block)`` over the self-similarity matrix.

    Chunked so the n x n matrix never lives in memory at once: O(chunk * n) floats
    per step rather than O(n^2).
    """
    x = as_unit_rows(mat)
    n = x.shape[0]
    size = max(1, int(chunk))
    for start in range(0, n, size):
        stop = min(start + size, n)
        yield start, stop, np.clip(x[start:stop] @ x.T, -1.0, 1.0)


def exact_duplicate_pairs(mat: np.ndarray, threshold: float, *, chunk: int = 512) -> PairList:
    """All ``i < j`` pairs with cosine >= threshold. O(n^2 d) time, O(n) extra memory."""
    x = as_unit_rows(mat)
    out: PairList = []
    for start, stop, block in exact_sim_blocks(x, chunk=chunk):
        for local_i in range(stop - start):
            i = start + local_i
            row = block[local_i]
            js = np.nonzero(row[i + 1 :] >= threshold)[0]
            if js.size:
                sims = row[i + 1 :][js]
                out.extend((i, int(i + 1 + j), float(s)) for j, s in zip(js, sims))
    out.sort()
    return out


def exact_topk(mat: np.ndarray, k: int, *, chunk: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Brute-force top-``k`` neighbours of every row, self excluded.

    Returns ``(ids, sims)`` shaped ``(n, k)``; unused slots hold ``-1`` / ``-2.0``.
    """
    x = as_unit_rows(mat)
    n = x.shape[0]
    k = max(0, int(k))
    ids = np.full((n, k), MISSING_ID, dtype=np.int64)
    sims = np.full((n, k), MISSING_SIM, dtype=np.float32)
    if n <= 1 or k == 0:
        return ids, sims
    take = min(k, n - 1)
    for start, stop, block in exact_sim_blocks(x, chunk=chunk):
        rows = np.arange(start, stop)
        block = block.copy()
        block[np.arange(stop - start), rows] = MISSING_SIM  # a point is not its own neighbour
        part = np.argpartition(-block, take - 1, axis=1)[:, :take]
        part_sims = np.take_along_axis(block, part, axis=1)
        order = np.argsort(-part_sims, axis=1, kind="stable")
        ordered = np.take_along_axis(part, order, axis=1)
        ids[rows] = ordered
        sims[rows] = np.take_along_axis(block, ordered, axis=1)
    return ids, sims


def pairs_to_adjacency(pairs: Sequence[tuple[int, int]], n: int) -> list[set[int]]:
    adj: list[set[int]] = [set() for _ in range(n)]
    for i, j in pairs:
        adj[i].add(j)
        adj[j].add(i)
    return adj


def pairs_recall(
    truth_ids: np.ndarray | Sequence[Sequence[int]],
    got_ids: np.ndarray | Sequence[Sequence[int]],
    *,
    k: int | None = None,
) -> tuple[float, float, float]:
    """Micro ``(recall@k, precision@k, f1)`` of a neighbour list against ground truth.

    Negative (missing) slots are ignored on both sides, so short result lists cost
    recall but never count as spurious hits.
    """
    hits = n_truth = n_got = 0
    for t_row, g_row in zip(truth_ids, got_ids):
        tset = {int(v) for v in t_row if int(v) >= 0}
        gset = {int(v) for v in g_row if int(v) >= 0}
        if k is not None:
            gset = set(sorted(gset)[:k])
        hits += len(tset & gset)
        n_truth += len(tset)
        n_got += len(gset)
    recall = _ratio(hits, n_truth)
    precision = hits / n_got if n_got else 0.0
    return recall, precision, _f1(recall, precision)


# --------------------------------------------------------------------------- #
# MinHash + LSH banding
# --------------------------------------------------------------------------- #
#: LSH banding is banded on **Jaccard**, but a dedup threshold is a **cosine**
#: threshold. Banding at the raw cosine value is far too strict (see
#: :func:`jaccard_from_cosine`), so candidates are banded at a deliberately lower
#: similarity and every hit is verified with exact cosine downstream.
DEFAULT_LSH_MARGIN = 0.8


def jaccard_from_cosine(cos: float, *, margin: float = DEFAULT_LSH_MARGIN) -> float:
    """Translate a cosine threshold into the Jaccard level to band MinHash at.

    For two same-length *binary* feature vectors with ``m`` ones and ``k`` shared
    ones, ``cos = k / m`` and ``jaccard = k / (2m - k)``, hence
    ``jaccard = cos / (2 - cos)``. Real text behaves similarly (word + char
    shingles, roughly equal lengths), but TF-IDF weighting and LSA projection push
    cosine above Jaccard further, so the estimate is scaled by ``margin < 1``.

    Under-banding only costs candidate volume, never correctness: :mod:`factory.dedup`
    re-checks every candidate pair with the exact cosine, so a loose LSH threshold
    wastes a little time while a tight one silently leaks duplicates.
    """
    c = float(np.clip(float(cos), 0.0, 1.0))
    j = 1.0 if c >= 1.0 else c / (2.0 - c)
    return float(min(1.0, max(1e-3, j * float(margin))))


def select_band_params(num_perm: int, threshold: float) -> tuple[int, int]:
    """Pick ``(bands, rows)`` whose S-curve crosses 0.5 nearest the threshold.

    ``P(candidate | Jaccard=s) = 1 - (1 - s^rows)^bands``. Ties break toward more
    bands (higher recall, more candidates) because candidates are verified exactly
    downstream: a false positive is free, a false negative is a leaked duplicate.
    """
    num_perm = max(2, int(num_perm))
    best: tuple[float, int, int] | None = None
    for bands in range(1, num_perm + 1):
        rows = num_perm // bands
        if rows < 1:
            break
        prob = 1.0 - (1.0 - float(threshold) ** rows) ** bands
        key = (abs(prob - 0.5), -bands, -rows)  # closest to 50%, then widest fan-out
        if best is None or key < best:
            best = key
    return -best[1], -best[2]


class MinHashLSH:
    """MinHash signatures + LSH banding over shingle sets.

    Parameters
    ----------
    num_perm:
        Signature length == number of permutations. Accuracy/cost knob.
    threshold:
        Jaccard level where the banding S-curve crosses 50%.
    bands, rows:
        Override :func:`select_band_params` (must satisfy ``bands * rows <= num_perm``).
    """

    def __init__(
        self,
        num_perm: int = 128,
        threshold: float = 0.9,
        *,
        seed: int = 13,
        bands: int | None = None,
        rows: int | None = None,
    ) -> None:
        self.num_perm = max(2, int(num_perm))
        self.threshold = float(threshold)
        self.seed = int(seed)
        self.bands, self.rows = select_band_params(self.num_perm, self.threshold)
        if bands and rows:
            if int(bands) * int(rows) > self.num_perm:
                raise ValueError(f"bands*rows ({bands}*{rows}) exceeds num_perm={self.num_perm}")
            self.bands, self.rows = int(bands), int(rows)
        self.signatures_: np.ndarray | None = None
        self.n_docs = 0
        self.build_seconds = 0.0
        self.shingle_vocab = 0
        self._coeff_a: np.ndarray | None = None
        self._coeff_c: np.ndarray | None = None
        self._base_hashes: np.ndarray = np.empty(0, dtype=np.uint64)
        self._bucket_index: dict[tuple[int, int], list[int]] | None = None
        self._band_key_cache: np.ndarray | None = None
        # odd multipliers keep the band fold position-sensitive and invertible mod 2**64
        self._band_weights = (
            np.random.default_rng(self.seed + 1)
            .integers(1, np.iinfo(np.int64).max, size=self.rows, dtype=np.uint64)
            | np.uint64(1)
        ).reshape(1, 1, self.rows)

    # -- build -------------------------------------------------------------- #
    def _ensure_coeffs(self) -> tuple[np.ndarray, np.ndarray]:
        if self._coeff_a is None:
            rng = np.random.default_rng(self.seed)
            high = np.uint64(np.iinfo(np.int64).max)
            a = rng.integers(1, high, size=self.num_perm, dtype=np.uint64) | np.uint64(1)
            c = rng.integers(0, high, size=self.num_perm, dtype=np.uint64)
            self._coeff_a, self._coeff_c = a, c
        return self._coeff_a, self._coeff_c

    def _signature(self, shingles: Sequence[str], base: dict[str, int]) -> np.ndarray:
        a, c = self._ensure_coeffs()
        sig = np.full(self.num_perm, UINT64_MAX, dtype=np.uint64)
        if not shingles:
            return sig
        h = self._base_hashes[np.fromiter((base[s] for s in shingles), np.int64, len(shingles))]
        # multiply-add with 64-bit overflow: a bijection on Z/2**64 for odd a, used as a
        # cheap stand-in for mod-prime min-wise permutations
        return np.minimum.reduce((h[:, None] * a[None, :]) + c[None, :], axis=0)

    def build(self, sets: Sequence[Sequence[str]]) -> "MinHashLSH":
        """Sketch every document's shingle set; document order is preserved."""
        t0 = time.perf_counter()
        docs = [list(dict.fromkeys(s)) for s in sets]
        self.n_docs = len(docs)
        uniq = sorted({s for d in docs for s in d})
        self.shingle_vocab = len(uniq)
        self._base_hashes = np.fromiter(
            (hash64(u, self.seed) for u in uniq), dtype=np.uint64, count=len(uniq)
        )
        base = {u: i for i, u in enumerate(uniq)}
        sig = np.empty((self.n_docs, self.num_perm), dtype=np.uint64)
        for i, d in enumerate(docs):
            sig[i] = self._signature(d, base)
        self.signatures_ = sig
        self._bucket_index = None
        self._band_key_cache = None
        self.build_seconds = time.perf_counter() - t0
        return self

    # -- banding ------------------------------------------------------------ #
    def band_keys(self) -> np.ndarray:
        """``(n_docs, bands)`` uint64 band fingerprints (cached).

        A band is ``rows`` consecutive signature values; it is folded into one
        64-bit key by a position-weighted multiply-add over ``Z/2**64`` plus a
        splitmix-style avalanche. That is a fast *universal* hash rather than a
        cryptographic one, and that is deliberate: a collision would only merge two
        candidate pools, and every candidate pair is verified with exact cosine
        downstream, so collisions cost a little time and never correctness.
        """
        if self.signatures_ is None:
            raise RuntimeError("MinHashLSH.build() must run first")
        cache = self._band_key_cache
        if cache is not None and cache.shape == (self.n_docs, self.bands):
            return cache
        # bands*rows may be < num_perm (e.g. 6*21=126 of 128): the trailing
        # permutations are simply not used by this banding scheme
        used = self.signatures_[:, : self.bands * self.rows]
        view = used.reshape(self.n_docs, self.bands, self.rows)
        z = (view ^ (view >> np.uint64(31))) * self._band_weights
        h = z.sum(axis=2, dtype=np.uint64)
        h ^= h >> np.uint64(33)
        h *= np.uint64(0xFF51AFD7ED558CCD)
        h ^= h >> np.uint64(29)
        self._band_key_cache = h
        return h

    def buckets(self) -> dict[tuple[int, int], list[int]]:
        """``(band, fingerprint) -> doc ids``. Built once and cached."""
        if self._bucket_index is None:
            keys = self.band_keys()
            out: dict[tuple[int, int], list[int]] = defaultdict(list)
            for b in range(self.bands):
                column = keys[:, b]
                for i in range(self.n_docs):
                    out[(b, int(column[i]))].append(i)
            self._bucket_index = dict(out)
        return self._bucket_index

    def candidate_pairs(self) -> set[tuple[int, int]]:
        """``i < j`` pairs sharing at least one band bucket."""
        pairs: set[tuple[int, int]] = set()
        for members in self.buckets().values():
            if len(members) < 2:
                continue
            members.sort()
            for x, i in enumerate(members):
                for j in members[x + 1 :]:
                    pairs.add((i, j))
        return pairs

    def neighbors(self, i: int) -> set[int]:
        """Candidate pool for one indexed document, excluding itself. O(bands * bucket)."""
        if self.signatures_ is None:
            raise RuntimeError("MinHashLSH.build() must run first")
        keys = self.band_keys()
        buckets = self.buckets()
        out: set[int] = set()
        for b in range(self.bands):
            out.update(buckets.get((b, keys[i, b]), ()))
        out.discard(i)
        return out

    def stats(self) -> dict[str, Any]:
        buckets = self.buckets() if self.signatures_ is not None else {}
        sizes = [len(v) for v in buckets.values()]
        return {
            "num_perm": self.num_perm,
            "bands": self.bands,
            "rows": self.rows,
            "target_jaccard": round(self.threshold, 6),
            "n_docs": self.n_docs,
            "n_buckets": len(buckets),
            "max_bucket_size": max(sizes, default=0),
            "candidate_pairs": len(self.candidate_pairs()),
            "shingle_vocab": self.shingle_vocab,
            "build_seconds": round(self.build_seconds, 6),
        }


# --------------------------------------------------------------------------- #
# IVF (inverted file) index over vectors
# --------------------------------------------------------------------------- #
@dataclass
class IvfStats:
    n_lists: int = 0
    probed_lists: int = 0
    actual_lists: int = 0
    empty_lists: int = 0
    n_docs: int = 0
    dim: int = 0
    max_iter: int = 0
    iters_run: int = 0
    avg_list_size: float = 0.0
    max_list_size: int = 0
    build_seconds: float = 0.0
    seed: int = 13

    def as_dict(self) -> dict[str, Any]:
        return {k: _plain(v) for k, v in self.__dict__.items()}


class IvfIndex:
    """Cosine IVF index: deterministic k-means++ quantiser + inverted posting lists.

    ``n_lists`` and ``n_probe`` are the accuracy/speed knobs; ``n_probe == n_lists``
    recovers exact search by construction, which makes the recall sweep in
    :func:`benchmark_ann` a clean ablation.
    """

    def __init__(
        self,
        n_lists: int = 8,
        *,
        n_probe: int = 2,
        seed: int = 13,
        max_iter: int = 15,
        tol: float = 1e-9,
    ) -> None:
        self.n_lists = max(1, int(n_lists))
        self.n_probe = max(1, int(n_probe))
        self.seed = int(seed)
        self.max_iter = max(1, int(max_iter))
        self.tol = float(tol)
        self.centroids_: np.ndarray | None = None
        self.postings_: list[list[int]] = []
        self.vectors_: np.ndarray | None = None
        self.stats_ = IvfStats(n_lists=self.n_lists, probed_lists=self.n_probe,
                               max_iter=self.max_iter, seed=self.seed)

    # -- build -------------------------------------------------------------- #
    def _init_centroids(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Deterministic cosine k-means++ seeding."""
        n = x.shape[0]
        centers = np.empty((self.n_lists, x.shape[1]), dtype=np.float64)
        centers[0] = x[int(rng.integers(n))]
        for c in range(1, self.n_lists):
            sims = np.clip(x @ centers[:c].T, -1.0, 1.0).max(axis=1)
            dist = np.clip(1.0 - sims, 0.0, None)
            total = float(dist.sum())
            idx = int(rng.integers(n)) if total <= 1e-12 else int(rng.choice(n, p=dist / total))
            centers[c] = x[idx]
        return centers

    def build(self, vectors: np.ndarray) -> "IvfIndex":
        t0 = time.perf_counter()
        x = as_unit_rows(vectors).astype(np.float64)
        n = x.shape[0]
        self.vectors_ = x.astype(np.float32)
        if n == 0:
            self.centroids_ = None
            self.postings_ = []
            return self
        self.n_lists = min(self.n_lists, n)
        self.stats_.n_lists = self.n_lists
        centers = self._init_centroids(x, np.random.default_rng(self.seed))
        assign = np.full(n, -1, dtype=np.int32)
        for it in range(self.max_iter):
            new = np.clip(x @ centers.T, -1.0, 1.0).argmax(axis=1).astype(np.int32)
            moved = int(np.count_nonzero(new != assign))
            assign = new
            for c in range(self.n_lists):
                members = x[assign == c]
                if members.size == 0:
                    # deterministic farthest-point re-seed keeps the quantiser full
                    spread = 1.0 - np.clip(x @ centers.T, -1.0, 1.0).max(axis=1)
                    centers[c] = x[int(np.argmax(spread))]
                    continue
                mu = members.mean(axis=0)
                norm = float(np.linalg.norm(mu))
                centers[c] = mu / norm if norm > 1e-12 else centers[c]
            self.stats_.iters_run = it + 1
            if moved == 0 and it > 0:
                break
        self.centroids_ = centers.astype(np.float32)
        self.postings_ = [[] for _ in range(self.n_lists)]
        for i, c in enumerate(assign.tolist()):
            self.postings_[int(c)].append(i)
        sizes = [len(p) for p in self.postings_]
        self.stats_.actual_lists = sum(1 for s in sizes if s)
        self.stats_.empty_lists = sum(1 for s in sizes if not s)
        self.stats_.n_docs = n
        self.stats_.dim = int(x.shape[1])
        self.stats_.avg_list_size = float(np.mean(sizes)) if sizes else 0.0
        self.stats_.max_list_size = int(max(sizes)) if sizes else 0
        self.stats_.build_seconds = round(time.perf_counter() - t0, 6)
        return self

    # -- search ------------------------------------------------------------- #
    def probe_lists(
        self,
        queries: np.ndarray,
        n_probe: int | None = None,
        *,
        exclude_self: bool = False,
    ) -> list[list[int]]:
        """Doc ids in the posting lists each query visits (deduped, ascending).

        ``exclude_self`` is for the case where ``queries`` *are* the indexed rows
        (top-k neighbour search over the corpus itself): a document is never its own
        neighbour.
        """
        if self.centroids_ is None:
            if self.vectors_ is None:
                raise RuntimeError("IvfIndex.build() must run first")
            n = self.vectors_.shape[0]
            return [
                [j for j in range(n) if not (exclude_self and j == i)] for i in range(len(queries))
            ]
        q = as_unit_rows(queries).astype(np.float64)
        npb = min(self.n_probe if n_probe is None else int(n_probe), self.n_lists)
        order = np.argsort(-(q @ self.centroids_.astype(np.float64).T), axis=1, kind="stable")
        per_query: list[list[int]] = []
        for i in range(q.shape[0]):
            seen: set[int] = set()
            for c in order[i, :npb].tolist():
                seen.update(self.postings_[int(c)])
            if exclude_self:
                seen.discard(i)
            per_query.append(sorted(seen))
        return per_query

    def search(
        self, queries: np.ndarray, k: int = 10, n_probe: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
        """``(ids, sims, candidate_lists)``; unprobed lists are the only recall loss.

        Similarities inside a scanned posting list are exact cosine, so returned
        scores are always true scores -- ANN can miss a neighbour, never mis-rank one.
        """
        q = as_unit_rows(queries)
        n, k = q.shape[0], max(0, int(k))
        ids = np.full((n, k), MISSING_ID, dtype=np.int64)
        sims = np.full((n, k), MISSING_SIM, dtype=np.float32)
        if k == 0:
            return ids, sims, [[] for _ in range(n)]
        pools = self.probe_lists(q, n_probe)
        for i, pool in enumerate(pools):
            if not pool:
                continue
            arr = np.asarray(pool, dtype=np.int64)
            s = np.clip(self.vectors_[arr].astype(np.float64) @ q[i].astype(np.float64), -1.0, 1.0)
            take = min(k, arr.size)
            part = np.argpartition(-s, take - 1)[:take]
            part = part[np.argsort(-s[part], kind="stable")]
            ids[i, :take] = arr[part]
            sims[i, :take] = s[part]
        return ids, sims, pools

    def candidate_pairs(self, n_probe: int | None = None) -> set[tuple[int, int]]:
        """``i < j`` pairs that share at least one jointly-probed posting list."""
        if self.vectors_ is None:
            return set()
        pairs: set[tuple[int, int]] = set()
        for i, pool in enumerate(self.probe_lists(self.vectors_, n_probe, exclude_self=True)):
            for j in pool:
                if j > i:
                    pairs.add((i, int(j)))
        return pairs

    def stats(self) -> dict[str, Any]:
        d = self.stats_.as_dict()
        d["n_probe_requested"] = self.n_probe
        return d


# --------------------------------------------------------------------------- #
# candidate generation, dispatched
# --------------------------------------------------------------------------- #
def candidate_pairs(
    vectors: np.ndarray,
    *,
    sets: Sequence[Sequence[str]] | None = None,
    threshold: float = 0.9,
    strategy: str = "lsh",
    num_perm: int = 128,
    n_lists: int | None = None,
    n_probe: int = 4,
    seed: int = 13,
    max_exact_fallback: int = 256,
    lsh_margin: float = DEFAULT_LSH_MARGIN,
    lsh_threshold: float | None = None,
) -> tuple[set[tuple[int, int]], dict[str, Any]]:
    """Candidate ``i < j`` pairs under the chosen strategy, plus index diagnostics.

    ``strategy`` is one of ``exact``, ``lsh``, ``ivf``, ``hybrid``. Corpora at or
    below ``max_exact_fallback`` rows go brute-force anyway: below that size the
    quadratic scan is cheaper than building an index, and it removes all recall risk.

    ``threshold`` is a **cosine** threshold (the dedup contract). For ``lsh`` it is
    mapped to a Jaccard level with :func:`jaccard_from_cosine`; pass
    ``lsh_threshold`` to band at an exact Jaccard instead, or ``lsh_margin`` to
    trade candidate volume against recall.
    """
    x = as_unit_rows(vectors)
    n = x.shape[0]
    strat = (strategy or "lsh").lower()
    if strat not in STRATEGIES:
        raise ValueError(f"unknown dedup strategy {strategy!r}; expected one of {STRATEGIES}")
    info: dict[str, Any] = {"strategy": strat, "n_docs": n, "max_exact_fallback": max_exact_fallback}
    if n <= 1 or strat == "exact" or n <= max_exact_fallback:
        info["used_exact_fallback"] = strat != "exact" and n <= max_exact_fallback
        pairs = {(i, j) for i, j, _ in exact_duplicate_pairs(x, threshold)}
        info["candidate_pairs"] = len(pairs)
        return pairs, info

    pairs: set[tuple[int, int]] = set()
    if strat in ("lsh", "hybrid"):
        if sets is None:
            raise ValueError(f"strategy {strat!r} needs shingle sets (see factory.text.shingle_set)")
        band_at = (
            jaccard_from_cosine(threshold, margin=lsh_margin)
            if lsh_threshold is None
            else float(lsh_threshold)
        )
        info["lsh_band_jaccard"] = round(band_at, 6)
        lsh = MinHashLSH(num_perm=num_perm, threshold=band_at, seed=seed).build(sets)
        pairs |= lsh.candidate_pairs()
        info["lsh"] = lsh.stats()
    if strat in ("ivf", "hybrid"):
        lists = int(n_lists) if n_lists else max(1, min(64, int(round(n**0.5))))
        idx = IvfIndex(lists, n_probe=n_probe, seed=seed).build(x)
        pairs |= idx.candidate_pairs()
        info["ivf"] = idx.stats()
    pairs = {p for p in pairs if p[0] != p[1]}
    info["candidate_pairs"] = len(pairs)
    info["candidate_density"] = round(len(pairs) / (n * (n - 1) / 2), 8)
    return pairs, info


# --------------------------------------------------------------------------- #
# benchmark: ANN vs exact
# --------------------------------------------------------------------------- #
@dataclass
class AnnBench:
    """Recall and latency of one ANN strategy against brute force on the same vectors."""

    strategy: str
    n: int
    dim: int
    k: int
    threshold: float
    recall_at_k: float
    precision_at_k: float
    f1_at_k: float
    pair_recall: float
    pair_precision: float
    exact_pairs: int
    ann_pairs: int
    mean_candidates: float
    max_candidates: int
    exact_ms: float
    ann_ms: float
    speedup: float
    params: dict[str, Any] = field(default_factory=dict)
    complexity_note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {k: _plain(v) for k, v in self.__dict__.items()}

    def summary(self) -> str:
        return (
            f"{self.strategy}: recall@{self.k}={self.recall_at_k:.3f} "
            f"pair-recall={self.pair_recall:.3f} f1@{self.k}={self.f1_at_k:.3f} "
            f"cands/doc={self.mean_candidates:.1f} speedup={self.speedup:.1f}x "
            f"({self.exact_ms:.1f}ms -> {self.ann_ms:.1f}ms)"
        )


def _rerank_topk(mat: np.ndarray, pools: Sequence[Sequence[int]], k: int) -> list[list[int]]:
    """Exact-cosine re-ranking inside each candidate pool."""
    x = as_unit_rows(mat)
    out: list[list[int]] = []
    for i, pool in enumerate(pools):
        cand = [j for j in pool if j != i]
        if not cand or k <= 0:
            out.append([])
            continue
        arr = np.asarray(cand, dtype=np.int64)
        s = np.clip(x[arr].astype(np.float64) @ x[i].astype(np.float64), -1.0, 1.0)
        take = min(k, arr.size)
        part = np.argpartition(-s, take - 1)[:take]
        part = part[np.argsort(-s[part], kind="stable")]
        out.append([int(v) for v in arr[part]])
    return out


def _timed(fn):
    t0 = time.perf_counter()
    res = fn()
    return res, (time.perf_counter() - t0) * 1000.0


def benchmark_ann(
    vectors: np.ndarray,
    *,
    strategies: Sequence[str] = ("ivf", "lsh", "hybrid"),
    k: int = 10,
    threshold: float = 0.9,
    n_lists: int = 16,
    n_probe: int = 6,
    num_perm: int = 128,
    sets: Sequence[Sequence[str]] | None = None,
    seed: int = 13,
    lsh_margin: float = DEFAULT_LSH_MARGIN,
    lsh_threshold: float | None = None,
) -> dict[str, AnnBench]:
    """recall@k / pair-recall / latency for each strategy, scored against brute force.

    ``sets`` (shingle sets) is required for ``lsh``/``hybrid``; those are skipped
    with a note when it is absent. ``threshold`` is a cosine threshold and is mapped
    to a Jaccard band level for LSH exactly as :func:`candidate_pairs` maps it.
    Ground truth is the exact top-k and the exact set of pairs above ``threshold`` on
    the very same vectors, so the only variable is the candidate generator.
    """
    x = as_unit_rows(vectors)
    n = x.shape[0]
    band_at = (
        jaccard_from_cosine(threshold, margin=lsh_margin)
        if lsh_threshold is None
        else float(lsh_threshold)
    )
    (exact_ids, _exact_sims), t_exact = _timed(lambda: exact_topk(x, k))
    exact_pairs, t_pairs = _timed(lambda: exact_duplicate_pairs(x, threshold))
    truth_pairs = {(i, j) for i, j, _ in exact_pairs}
    truth_topk = [sorted(int(v) for v in row if int(v) >= 0) for row in exact_ids]
    base_note = (
        f"exact reference scores all {n * max(0, n - 1) // 2} pairs (O(n^2 d)); "
        "ANN scores a candidate subset, so speedup grows with n"
    )

    results: dict[str, AnnBench] = {}
    for strategy in strategies:
        sname = str(strategy).lower()
        info: dict[str, Any] = {
            "k": k, "n_probe": n_probe, "n_lists": n_lists,
            "num_perm": num_perm, "lsh_band_jaccard": round(band_at, 6),
        }
        if sname == "exact":
            pools = [list(row) for row in truth_topk]
            t_ann = 0.0
        elif sname == "ivf":
            (idx,), t_ann = _timed(lambda: [IvfIndex(n_lists, n_probe=n_probe, seed=seed).build(x)])
            pools = idx.probe_lists(x, exclude_self=True)
            info["ivf"] = idx.stats()
        elif sname in ("lsh", "hybrid"):
            if sets is None:
                continue
            (lsh,), t_ann = _timed(lambda: [MinHashLSH(num_perm, band_at, seed=seed).build(sets)])
            pools = [sorted(lsh.neighbors(i)) for i in range(n)]
            info["lsh"] = lsh.stats()
            if sname == "hybrid":
                (idx,), t_ivf = _timed(lambda: [IvfIndex(n_lists, n_probe=n_probe, seed=seed).build(x)])
                t_ann += t_ivf
                pools = [
                    sorted(set(a) | set(b))
                    for a, b in zip(pools, idx.probe_lists(x, exclude_self=True))
                ]
                info["ivf"] = idx.stats()
        else:
            raise ValueError(f"unknown strategy {sname!r}; expected one of {STRATEGIES}")

        # candidates are always re-scored with exact cosine before being counted
        reranked = truth_topk if sname == "exact" else _rerank_topk(x, pools, k)
        rec_topk, prec_topk, f1 = pairs_recall(truth_topk, reranked, k=k)
        detected = {
            (i, j) if i < j else (j, i)
            for i, pool in enumerate(pools)
            for j in pool
            if j != i and _cos(x, i, j) >= threshold
        }
        detected = {(i, j) if i < j else (j, i) for i, j in detected}
        pair_rec = _ratio(len(truth_pairs & detected), len(truth_pairs))
        pair_prec = _ratio(len(truth_pairs & detected), len(detected))
        sizes = [len(p) for p in pools]
        results[sname] = AnnBench(
            strategy=sname,
            n=n,
            dim=int(x.shape[1]),
            k=k,
            threshold=threshold,
            recall_at_k=rec_topk,
            precision_at_k=prec_topk,
            f1_at_k=f1,
            pair_recall=pair_rec,
            pair_precision=pair_prec,
            exact_pairs=len(truth_pairs),
            ann_pairs=len(detected),
            mean_candidates=float(np.mean(sizes)) if sizes else 0.0,
            max_candidates=int(max(sizes, default=0)),
            exact_ms=round(max(t_exact, t_pairs), 3),
            ann_ms=round(t_ann, 3),
            speedup=(max(t_exact, t_pairs) / t_ann) if t_ann > 0 else float("inf"),
            params=info,
            complexity_note=base_note,
        )
    return results


def _cos(x: np.ndarray, i: int, j: int) -> float:
    return float(np.clip(x[i] @ x[j], -1.0, 1.0))
