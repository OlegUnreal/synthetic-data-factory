"""Near-duplicate removal via cosine similarity of embeddings."""
from __future__ import annotations

import numpy as np

from .seed import Example


def _embed(texts: list[str]) -> np.ndarray:
    # Stub: real version calls an embedding model. Here we use char n-grams.
    vecs = []
    for t in texts:
        v = np.zeros(64)
        for i in range(len(t) - 2):
            v[hash(t[i:i+3]) % 64] += 1
        norm = np.linalg.norm(v) or 1
        vecs.append(v / norm)
    return np.array(vecs)


def dedup(examples: list[Example], threshold: float = 0.92) -> list[Example]:
    if not examples:
        return []
    mat = _embed([e.input for e in examples])
    keep = [True] * len(examples)
    for i in range(len(examples)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(examples)):
            if keep[j] and float(mat[i] @ mat[j]) >= threshold:
                keep[j] = False
    return [e for e, k in zip(examples, keep) if k]
