"""Text normalisation and feature extraction shared by embeddings, dedup and metrics.

Everything in here is process-independent. The previous dedup stub used CPython's
builtin ``hash()`` on strings, which is salted per process (``PYTHONHASHSEED``), so
the *same* input produced a *different* vector on every run and embeddings could not
be cached, compared or reproduced. All hashing here goes through :func:`hash64`
(BLAKE2b), which is stable across runs, platforms and Python builds.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from hashlib import blake2b

__all__ = [
    "normalize_text",
    "tokenize",
    "word_shingles",
    "char_shingles",
    "features",
    "shingle_set",
    "hash64",
    "content_hash",
    "ngram_counts",
    "DIGEST_SIZE",
]

DIGEST_SIZE = 16  # bytes; 8 for the bucket index, 8 for the sign bits
_DEFAULT_SALT = b"sdf-text"  # deterministic BLAKE2b key when no seed is given

_WS_RE = re.compile(r"\s+")
# word chars plus the intra-word apostrophe/hyphen ("don't", "fine-tuning")
_WORD_TOKEN_RE = re.compile(r"[^\W_]+(?:['\-][^\W_]+)*", re.UNICODE)


def normalize_text(text: object) -> str:
    """Casefold, strip accents and punctuation, collapse whitespace.

    Kept Unicode-letter safe (category based rather than an ASCII regex) so that
    non-Latin seeds degrade to character features instead of vanishing.
    """
    if text is None:
        return ""
    t = unicodedata.normalize("NFKC", str(text)).casefold()
    t = unicodedata.normalize("NFKD", t)
    out: list[str] = []
    for ch in t:
        if unicodedata.combining(ch):
            continue  # drop the accent now that the base letter is decomposed
        cat = unicodedata.category(ch)
        if cat[0] in ("L", "N") or ch in ("'", "-"):
            out.append(ch)
        else:
            out.append(" ")
    return _WS_RE.sub(" ", "".join(out)).strip()


def tokenize(text: object) -> list[str]:
    """Whitespace tokens of the normalised text."""
    return normalize_text(text).split()


def word_shingles(text: str, ngram_range: tuple[int, int] = (1, 2)) -> list[str]:
    """Unigrams ... n-grams joined by ``_`` (mirrors sklearn's word analyzer)."""
    toks = tokenize(text)
    lo, hi = max(1, ngram_range[0]), max(ngram_range[1], 1)
    out: list[str] = []
    for n in range(lo, hi + 1):
        if len(toks) < n:
            continue
        if n == 1:
            out.extend(toks)
        else:
            out.extend("_".join(toks[i : i + n]) for i in range(len(toks) - n + 1))
    return out


def char_shingles(text: str, ngram_range: tuple[int, int] = (3, 5)) -> list[str]:
    """Character n-grams of the normalised text, padded like sklearn's ``char_wb``."""
    s = normalize_text(text)
    if not s:
        return []
    padded = " " + s + " "
    lo, hi = max(1, ngram_range[0]), max(ngram_range[1], 1)
    out: list[str] = []
    for n in range(lo, hi + 1):
        if len(padded) < n:
            continue
        out.extend(padded[i : i + n] for i in range(len(padded) - n + 1))
    return out


def features(
    text: str,
    word_ngrams: tuple[int, int] = (1, 2),
    char_ngrams: tuple[int, int] = (3, 5),
) -> list[str]:
    """The canonical feature space: prefixed word and character n-grams.

    Prefixed so a word unigram can never collide with a character n-gram, and
    identical to what the TF-IDF vectorizers in :mod:`factory.embeddings` see.
    """
    s = normalize_text(text)
    if not s:
        return []
    return (
        ["w:" + f for f in word_shingles(s, word_ngrams)]
        + ["c:" + f for f in char_shingles(s, char_ngrams)]
    )


def shingle_set(text: str, **kwargs) -> frozenset[str]:
    """Distinct features -- the universe MinHash sketches."""
    return frozenset(features(text, **kwargs))


def ngram_counts(tokens: Sequence[str] | Iterable[str], n: int) -> dict[tuple[str, ...], int]:
    """Counts of token n-grams (used for Distinct-n / TTR)."""
    toks = list(tokens)
    if n < 1 or len(toks) < n:
        return {}
    counts: dict[tuple[str, ...], int] = {}
    for i in range(len(toks) - n + 1):
        g = tuple(toks[i : i + n])
        counts[g] = counts.get(g, 0) + 1
    return counts


def hash64(token: str, seed: int = 0) -> int:
    """Stable 64-bit hash of a string (BLAKE2b, salted by ``seed``)."""
    digest = blake2b(token.encode("utf-8"), digest_size=DIGEST_SIZE, key=_seed_key(seed)).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _seed_key(seed: int) -> bytes:
    """BLAKE2b keys must be 1..16 bytes; the seed is used as a deterministic salt."""
    return _DEFAULT_SALT if int(seed) == 0 else (abs(int(seed)) % (1 << 64)).to_bytes(8, "little")


def content_hash(*parts: object) -> str:
    """Deterministic identity hash for an example (used for dedup/leakage checks)."""
    h = blake2b(digest_size=32)
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()
