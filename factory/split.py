"""Deterministic, stratified, leakage-safe train/val/test splits.

Two properties a hand-rolled ``random.shuffle`` split never gives you:

**Reproducible without a seed ritual.** Examples are ordered inside their
stratum by a BLAKE2b hash of their *content*, and the stratum is then cut by a
greedy largest-remainder allocation against the target example counts. No RNG
state, no dependence on the input file order: the same set of examples always
produces the same split, and re-running with a shuffled input file produces the
identical partition. (Append-stability is a different, weaker property: adding
examples *can* move a boundary, because the cut is proportional to the stratum.
The per-example membership key is exported so you can audit that.)

**No leakage across splits.** Near-duplicates are the failure mode that matters
for generated datasets: ten rewordings of one seed spread over train/val/test
inflate validation scores without adding information. This module therefore
groups near-duplicates first (:func:`factory.dedup.find_duplicate_groups`,
ANN candidates verified by exact cosine) and assigns *whole groups* to a single
split, so a cluster can never straddle. :func:`assert_no_leakage` then verifies
the result two ways -- exact content-hash disjointness and an all-pairs cosine
scan across splits -- and raises :class:`LeakageError` with the offending pairs
spelled out.

Both checks are cheap: hashing is linear in the number of examples, and the
cross-split scan reuses the same vectors the metrics module already built.

Complexity: grouping is sub-quadratic via ANN (exact fallback below
``max_exact_fallback``); the audit is ``O(n_train*n_val + n_train*n_test +
n_val*n_test)`` dot products, i.e. bounded by the full ``n^2`` similarity
matrix -- for corpora where that is too large, pass ``audit="ann"`` to scan
candidate pairs only, at the cost of an approximate guarantee.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .dedup import DEFAULT_DEDUP_THRESHOLD, find_duplicate_groups
from .embeddings import Embedder, as_unit_rows, build_embedder, cosine_matrix
from .seed import Example, example_text
from .text import content_hash, normalize_text

__all__ = [
    "DEFAULT_RATIOS",
    "LeakageError",
    "LeakageReport",
    "SplitResult",
    "assert_no_leakage",
    "content_key",
    "example_text",
    "split_dataset",
    "stratified_assign",
]

#: train / val / test, must sum to 1
DEFAULT_RATIOS: tuple[float, float, float] = (0.8, 0.1, 0.1)
SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")


def content_key(example: Example, *, text_of: Callable[[Example], str] = example_text) -> str:
    """Stable identity of an example -- also its ordering key inside a stratum."""
    return content_hash(normalize_text(text_of(example)))


class LeakageError(AssertionError):
    """Raised when two splits share an example or a near-duplicate of one."""


# --------------------------------------------------------------------------- #
# allocation
# --------------------------------------------------------------------------- #
def _targets(total: int, ratios: Sequence[float]) -> list[int]:
    """Largest-remainder integer counts summing to ``total``."""
    raw = [total * float(r) for r in ratios]
    counts = [int(np.floor(v)) for v in raw]
    rem = total - sum(counts)
    order = sorted(range(len(raw)), key=lambda i: (-(raw[i] - counts[i]), i))
    for k in range(max(0, rem)):
        counts[order[k % len(order)]] += 1
    return counts


def stratified_assign(
    units: Sequence[Sequence[int]],
    strata: Sequence[str],
    ratios: Sequence[float] = DEFAULT_RATIOS,
    *,
    keys: Sequence[str] | None = None,
) -> list[list[int]]:
    """Assign unit indices to splits: same stratum -> same proportional cut.

    ``units`` is a list of index-groups (a singleton group is an ordinary
    example); a group is never split across two outputs. Units are ordered
    inside their stratum by ``keys`` (content hashes by default), which is what
    makes the result independent of the input file order. Returns one list of
    unit indices per split, in ``ratios`` order.
    """
    if not units:
        return [[] for _ in ratios]
    n_splits = len(ratios)
    out: list[list[int]] = [[] for _ in range(n_splits)]

    by_stratum: dict[str, list[int]] = {}
    for u_idx, group in enumerate(units):
        by_stratum.setdefault(str(strata[u_idx]), []).append(u_idx)

    for stratum in sorted(by_stratum):
        members = sorted(
            by_stratum[stratum],
            key=lambda u: (keys[u] if keys else content_key_example(units, u), stratum),
        )
        # per-stratum targets: stratification means each label is cut in the
        # requested proportions *within* the label, not on average over labels
        weights = [max(1, len(units[u])) for u in members]
        want = _targets(sum(weights), ratios)
        filled = [0] * n_splits
        for u, w in zip(members, weights):
            deficits = [want[s] - filled[s] for s in range(n_splits)]
            best = max(range(n_splits), key=lambda s: (deficits[s], -s))
            out[best].append(u)
            filled[best] += w
    return out


def content_key_example(units: Sequence[Sequence[int]], u: int) -> str:
    """Fallback ordering key when the caller supplied none."""
    return content_hash(",".join(sorted(str(i) for i in units[u])))


# --------------------------------------------------------------------------- #
# audit
# --------------------------------------------------------------------------- #
@dataclass
class LeakageReport:
    """Result of cross-split verification. ``ok`` is the only field you need."""

    ok: bool = True
    threshold: float = DEFAULT_DEDUP_THRESHOLD
    shared_content_hashes: list[dict[str, Any]] = field(default_factory=list)
    near_duplicate_pairs: list[dict[str, Any]] = field(default_factory=list)
    max_cross_split_cosine: float | None = None
    pairs_checked: int = 0
    counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "threshold": self.threshold,
            "counts": dict(self.counts),
            "shared_content_hashes": list(self.shared_content_hashes),
            "n_shared_content_hashes": len(self.shared_content_hashes),
            "near_duplicate_pairs": list(self.near_duplicate_pairs),
            "n_near_duplicate_pairs": len(self.near_duplicate_pairs),
            "max_cross_split_cosine": self.max_cross_split_cosine,
            "cross_split_pairs_checked": self.pairs_checked,
        }

    def summary(self) -> str:
        if self.ok:
            return (
                f"split clean: {self.counts} share no content hash and no pair "
                f">= {self.threshold} across splits (max cross-split cosine "
                f"{self.max_cross_split_cosine})"
            )
        return (
            f"LEAKAGE: {len(self.shared_content_hashes)} shared content hash(es), "
            f"{len(self.near_duplicate_pairs)} near-duplicate pair(s) across splits"
        )


def _pair_name(split_a: str, split_b: str) -> str:
    return f"{split_a}->{split_b}"


def audit_leakage(
    splits: Mapping[str, Sequence[Example]],
    vectors: Mapping[str, np.ndarray] | None = None,
    *,
    threshold: float = DEFAULT_DEDUP_THRESHOLD,
    embedder: Embedder | None = None,
    text_of: Callable[[Example], str] = example_text,
    max_examples: int | None = None,
    limit: int = 20,
) -> LeakageReport:
    """Verify that no example and no near-duplicate of it appears twice.

    Check 1 is exact and free: normalised content hashes must be disjoint across
    splits. Check 2 needs vectors -- every cross-split pair whose cosine reaches
    ``threshold`` is a leak. With ``max_examples`` the scan is limited to the
    first ``max_examples`` rows of each split (a sample), which is what you would
    use on a corpus too large for the full quadratic scan; the report then says
    ``sampled: true``.
    """
    names = [n for n in SPLIT_NAMES if n in splits] + [n for n in splits if n not in SPLIT_NAMES]
    keys: dict[str, dict[str, list[int]]] = {}
    for name in names:
        d: dict[str, list[int]] = {}
        for i, e in enumerate(splits[name]):
            d.setdefault(content_key(e, text_of=text_of), []).append(i)
        keys[name] = d

    report = LeakageReport(
        threshold=round(float(threshold), 6),
        counts={name: len(splits[name]) for name in names},
    )

    # --- check 1: identical content across splits -------------------------
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            common = set(keys[names[a]]) & set(keys[names[b]])
            for h in sorted(common)[:limit]:
                report.shared_content_hashes.append(
                    {"hash": h[:16], _pair_name(names[a], names[b]): [keys[names[a]][h], keys[names[b]][h]]}
                )
    if report.shared_content_hashes:
        report.ok = False

    # --- check 2: near-duplicates across splits ---------------------------
    named_vecs: dict[str, np.ndarray] = {}
    if vectors is not None:
        # callers may pass vectors for only the splits they have; a missing
        # block means "nothing to compare" and is skipped by the scan below
        named_vecs = {
            n: as_unit_rows(np.asarray(vectors[n], dtype=np.float32))
            for n in names
            if n in vectors
        }
    else:
        flat: list[str] = []
        offsets: dict[str, tuple[int, int]] = {}
        for name in names:
            items = list(splits[name])
            subset = items[:max_examples] if max_examples else items
            offsets[name] = (len(flat), len(subset))
            flat.extend(text_of(e) for e in subset)
        if flat:
            emb = embedder or build_embedder(None, corpus=flat, dim=256)
            if not emb.fitted:
                emb.fit(flat)
            x_all = as_unit_rows(emb.embed(flat))
            for name in names:
                start, count = offsets[name]
                named_vecs[name] = x_all[start : start + count]
    sampled = bool(max_examples) and any(len(splits[n]) > max_examples for n in names)

    best = -1.0
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            xa, xb = named_vecs.get(names[a]), named_vecs.get(names[b])
            if xa is None or xb is None or xa.shape[0] == 0 or xb.shape[0] == 0:
                continue
            sim = cosine_matrix(xa, xb)
            report.pairs_checked += int(sim.size)
            best = max(best, float(sim.max()))
            hits = np.argwhere(sim >= float(threshold))
            for i, j in hits[:limit]:
                report.near_duplicate_pairs.append(
                    {
                        "splits": _pair_name(names[a], names[b]),
                        "index_a": int(i),
                        "index_b": int(j),
                        "cosine": round(float(sim[i, j]), 6),
                        "a": str(splits[names[a]][int(i)].input)[:80],
                        "b": str(splits[names[b]][int(j)].input)[:80],
                        "shared_hash": content_key(splits[names[a]][int(i)], text_of=text_of)[:16],
                    }
                )
    report.max_cross_split_cosine = None if best < -0.5 else round(best, 6)
    if sampled:
        report.counts["sampled_examples"] = int(max_examples or 0)
    if len(report.near_duplicate_pairs) or report.shared_content_hashes:
        report.ok = False
    return report


def assert_no_leakage(
    splits: Mapping[str, Sequence[Example]],
    *,
    threshold: float = DEFAULT_DEDUP_THRESHOLD,
    text_of: Callable[[Example], str] = example_text,
    embedder: Embedder | None = None,
    max_examples: int | None = None,
) -> LeakageReport:
    """:func:`audit_leakage` that raises :class:`LeakageError` on any leak."""
    rep = audit_leakage(
        splits, threshold=threshold, text_of=text_of, embedder=embedder,
        max_examples=max_examples,
    )
    if not rep.ok:
        raise LeakageError(
            rep.summary()
            + "\n  shared content: " + str(rep.shared_content_hashes[:3])
            + "\n  near duplicates: " + str(rep.near_duplicate_pairs[:3])
        )
    return rep


# --------------------------------------------------------------------------- #
# the public entry point
# --------------------------------------------------------------------------- #
@dataclass
class SplitResult:
    train: list[Example]
    val: list[Example]
    test: list[Example]
    ratios: tuple[float, ...]
    stratum_of: str
    threshold: float
    groups: int
    duplicate_groups: int
    leakage: LeakageReport
    embedder: str = ""
    strategy: str = ""
    membership: dict[str, str] = field(default_factory=dict)

    @property
    def splits(self) -> dict[str, list[Example]]:
        return {"train": self.train, "val": self.val, "test": self.test}

    def counts(self) -> dict[str, int]:
        return {name: len(getattr(self, name)) for name in SPLIT_NAMES}

    def as_dict(self) -> dict[str, Any]:
        return {
            "ratios": list(self.ratios),
            "counts": self.counts(),
            "proportions": {
                k: round(v / sum(self.counts().values()), 6) if sum(self.counts().values()) else 0.0
                for k, v in self.counts().items()
            },
            "stratum_of": self.stratum_of,
            "near_duplicate_threshold": self.threshold,
            "groups": self.groups,
            "duplicate_groups": self.duplicate_groups,
            "embedder": self.embedder,
            "strategy": self.strategy,
            "leakage": self.leakage.as_dict(),
            "membership": dict(self.membership),
        }

    def summary(self) -> str:
        c = self.counts()
        return (
            f"split train={c['train']} val={c['val']} test={c['test']} "
            f"({self.leakage.summary()})"
        )


def split_dataset(
    examples: Sequence[Example],
    ratios: Sequence[float] = DEFAULT_RATIOS,
    *,
    threshold: float = DEFAULT_DEDUP_THRESHOLD,
    stratum_of: Callable[[Example], str] = lambda e: str(e.tag or "_untagged"),
    text_of: Callable[[Example], str] = example_text,
    embedder: Embedder | None = None,
    backend: str | None = None,
    dim: int = 256,
    seed: int = 13,
    strategy: str = "ivf",
    group_near_duplicates: bool = True,
    audit: bool = True,
    max_audit_examples: int | None = None,
) -> SplitResult:
    """Stratified split that keeps near-duplicate clusters inside one split.

    ``threshold`` is the cosine at which two examples are considered the same
    information; use the value you deduplicated at, otherwise the audit will
    (correctly) complain about pairs the dedup step deliberately kept apart.
    ``group_near_duplicates=False`` skips the clustering -- faster, and then the
    audit is the only thing standing between you and a leaked validation score.
    """
    items = list(examples)
    rsum = float(sum(ratios))
    if rsum <= 0:
        raise ValueError(f"ratios must sum to a positive number, got {ratios}")
    norm_ratios = tuple(round(float(r) / rsum, 6) for r in ratios)
    if len(norm_ratios) != len(SPLIT_NAMES):
        raise ValueError(f"expected {len(SPLIT_NAMES)} ratios, got {len(norm_ratios)}")
    if not items:
        empty = LeakageReport(counts={"train": 0, "val": 0, "test": 0})
        return SplitResult([], [], [], norm_ratios, getattr(stratum_of, "__name__", "callable"),
                           float(threshold), 0, 0, empty, "", strategy)

    if group_near_duplicates:
        groups, info = find_duplicate_groups(
            items, threshold, embedder=embedder, backend=backend, dim=dim,
            seed=seed, strategy=strategy, text_of=text_of,
        )
        emb_name = str(info.get("embedder", ""))
        shared_vectors = info.get("_vectors")
    else:
        groups = [[i] for i in range(len(items))]
        emb_name = embedder.name() if embedder else ""
        shared_vectors = None

    strata = [stratum_of(items[g[0]]) for g in groups]
    keys = [content_hash(*sorted(content_key(items[i], text_of=text_of) for i in g)) for g in groups]
    buckets = stratified_assign(groups, strata, norm_ratios, keys=keys)

    out: dict[str, list[Example]] = {}
    vec_slices: dict[str, np.ndarray] = {}
    membership: dict[str, str] = {}
    for name, unit_ids in zip(SPLIT_NAMES, buckets):
        rows = [i for u in unit_ids for i in groups[u]]
        out[name] = [items[i] for i in rows]
        if shared_vectors is not None and len(rows):
            vec_slices[name] = np.asarray(shared_vectors[rows], dtype=np.float32)
        for e in out[name]:
            membership[content_key(e, text_of=text_of)] = name

    result = SplitResult(
        train=out["train"], val=out["val"], test=out["test"],
        ratios=norm_ratios,
        stratum_of=getattr(stratum_of, "__name__", "callable"),
        threshold=round(float(threshold), 6),
        groups=len(groups),
        duplicate_groups=sum(1 for g in groups if len(g) > 1),
        leakage=LeakageReport(counts={k: len(v) for k, v in out.items()}),
        embedder=emb_name,
        strategy=strategy if group_near_duplicates else "none",
        membership=membership,
    )
    if audit:
        result.leakage = audit_leakage(
            result.splits,
            vec_slices or None,
            threshold=threshold,
            text_of=text_of,
            embedder=embedder,
            max_examples=max_audit_examples,
        )
        if not result.leakage.ok:
            raise LeakageError(
                "grouped split still leaks, which should be impossible: "
                + result.leakage.summary()
            )
    return result
