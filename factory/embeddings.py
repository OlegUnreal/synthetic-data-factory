"""Real text embeddings with pluggable, offline-deterministic backends.

Backends
--------
``lsa``
    TF-IDF over word 1-2-grams (``sublinear_tf=True``) **concatenated** with TF-IDF
    over ``char_wb`` 3-5-grams via ``scipy.sparse.hstack``, projected by
    ``TruncatedSVD`` (probabilistic LSA: a seeded random-sketch truncated SVD of the
    TF-IDF operator) to a fixed dimensionality, then L2-normalised to float32.
    Shared latent topics make a paraphrase score high even with few shared surface
    tokens, which is what near-duplicate detection needs.
``hash``
    Signed BLAKE2b hashing trick over the same feature space, fixed dimensionality,
    corpus-free. Used for tiny corpora (``< min_docs`` documents) and whenever the
    TF-IDF/SVD path is rank-degenerate (see :func:`build_embedder`). Deterministic
    across processes -- unlike CPython's builtin ``hash()``, which is salted per
    process via ``PYTHONHASHSEED`` and therefore not reproducible.
``openai``
    Optional network backend, selected by ``SDF_EMBEDDING_BACKEND=openai`` or an
    explicit ``backend="openai"``. Never touched by the test suite; ``openai`` is
    imported lazily inside the method that needs it.
``st``
    Optional local ``sentence-transformers`` backend, guarded with
    ``importlib.util.find_spec`` -- the package is not installed here and must not
    be imported unconditionally.

All backends share one interface (``fit`` / ``embed`` / ``embed_one`` / ``save`` /
``load``) and return L2-normalised float32 rows, so cosine similarity == dot
product everywhere downstream (:mod:`factory.ann`, :mod:`factory.dedup`,
:mod:`factory.metrics`, :mod:`factory.split`).
"""
from __future__ import annotations

import importlib.util
import math
import os
import pickle
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Sequence
from hashlib import blake2b
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from .text import features, normalize_text

__all__ = [
    "Embedder",
    "TfidfSvdEmbedder",
    "HashingEmbedder",
    "OpenAIEmbedder",
    "SentenceTransformerEmbedder",
    "build_embedder",
    "embed_texts",
    "embedder_from_meta",
    "available_backends",
    "as_unit_rows",
    "l2_normalize",
    "cosine_matrix",
    "DEFAULT_DIM",
    "EMBEDDING_VERSION",
    "MIN_DOCS_FOR_LSA",
    "MIN_COMPONENTS_FOR_LSA",
    "BACKEND_ENV_VAR",
    "DegenerateCorpusError",
]

EMBEDDING_VERSION = "1.0.0"
DEFAULT_DIM = 256
DEFAULT_SEED = 13
MIN_DOCS_FOR_LSA = 3
#: below this many usable components a rank-limited projection inflates the cosine
#: between unrelated texts (measured: ~0.5 for disjoint sentences at 8 docs)
MIN_COMPONENTS_FOR_LSA = 16
BACKEND_ENV_VAR = "SDF_EMBEDDING_BACKEND"


class DegenerateCorpusError(ValueError):
    """Raised when a corpus cannot support the requested projection."""


# --------------------------------------------------------------------------- #
# linear-algebra helpers
# --------------------------------------------------------------------------- #
def l2_normalize(mat: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalisation to float32. Zero rows stay zero (cosine 0)."""
    a = np.asarray(mat, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    tiny = norms < eps
    out = a / np.where(tiny, 1.0, norms)
    if tiny.any():
        out = np.where(tiny, 0.0, out)
    return out.astype(np.float32, copy=False)


def as_unit_rows(mat: np.ndarray) -> np.ndarray:
    """Defensively re-normalise: callers may hand us raw or already-unit vectors."""
    arr = np.asarray(mat, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.size == 0:
        return arr.astype(np.float32)
    return l2_normalize(arr)


def cosine_matrix(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    """Cosine similarity between row matrices of unit vectors (clipped to [-1, 1])."""
    ua = as_unit_rows(a)
    ub = ua if b is None else as_unit_rows(b)
    return np.clip(ua @ ub.T, -1.0, 1.0)


def _corpus_hash(docs: Sequence[str]) -> str:
    h = blake2b(digest_size=32)
    for d in docs:
        h.update(normalize_text(d).encode("utf-8", "replace"))
        h.update(b"\x1f")
    return h.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return None
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, set, frozenset)):
        return [_jsonable(v) for v in value]
    return str(value)


# --------------------------------------------------------------------------- #
# base class
# --------------------------------------------------------------------------- #
class Embedder(ABC):
    """Common interface: fit on a corpus, embed any text as an L2 unit vector."""

    backend: ClassVar[str] = "base"
    #: True when the backend needs neither network nor an optional dependency
    offline: ClassVar[bool] = True

    def __init__(self, dim: int = DEFAULT_DIM, seed: int = DEFAULT_SEED) -> None:
        if int(dim) <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.requested_dim = int(dim)
        self.dim = int(dim)
        self.seed = int(seed)
        self.fitted = False
        self.n_fit_docs = 0
        self.fit_corpus_hash: str | None = None
        self.auto_fallback_reason: str | None = None

    # -- required API ------------------------------------------------------- #
    @abstractmethod
    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        """Backend-specific raw (un-normalised) matrix of shape (n, dim)."""

    def fit(self, texts: Sequence[str]) -> "Embedder":
        docs = [str(t) for t in texts]
        self._fit_impl(docs)
        self.fitted = True
        self.n_fit_docs = len(docs)
        self.fit_corpus_hash = _corpus_hash(docs)
        return self

    def _fit_impl(self, docs: Sequence[str]) -> None:
        """Corpus-dependent setup; stateless backends inherit the no-op."""

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        docs = [str(t) for t in texts]
        if not docs:
            return np.zeros((0, self.dim), dtype=np.float32)
        if not self.fitted:
            self.fit(docs)
        raw = np.asarray(self._encode(docs), dtype=np.float64)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        if raw.shape[1] < self.dim:
            # A rank-limited projection (LSA on a small corpus) yields fewer columns
            # than requested; zero-padding keeps every vector from this embedder the
            # same width and cannot change any cosine similarity.
            raw = np.pad(raw, ((0, 0), (0, self.dim - raw.shape[1])))
        elif raw.shape[1] > self.dim:
            self.dim = int(raw.shape[1])  # backend fixes its own width (e.g. an ST model)
        return l2_normalize(raw)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        self.fit(texts)
        return self.embed(texts)

    # -- metadata / persistence --------------------------------------------- #
    def params(self) -> dict[str, Any]:
        """Constructor kwargs -- enough to rebuild an equivalent unfitted embedder."""
        return {"dim": self.requested_dim, "seed": self.seed}

    def state(self) -> dict[str, Any]:
        """Fitted diagnostics, recorded in the run manifest for lineage."""
        return {}

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "version": EMBEDDING_VERSION,
            "name": self.name(),
            "dim": self.dim,
            "requested_dim": self.requested_dim,
            "seed": self.seed,
            "offline": self.offline,
            "fitted": self.fitted,
            "n_fit_docs": self.n_fit_docs,
            "fit_corpus_hash": self.fit_corpus_hash,
            "auto_fallback_reason": self.auto_fallback_reason,
            "params": {k: _jsonable(v) for k, v in self.params().items()},
            "state": {k: _jsonable(v) for k, v in self.state().items()},
        }

    def name(self) -> str:
        return f"{self.backend}:{self.dim}d"

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"meta": self.as_dict(), "embedder": self}
        path.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
        return path

    @classmethod
    def load(cls, path: Path | str) -> "Embedder":
        payload = pickle.loads(Path(path).read_bytes())
        meta = payload.get("meta", {})
        if str(meta.get("version", "0")).split(".")[0] != EMBEDDING_VERSION.split(".")[0]:
            raise RuntimeError(
                f"embedder saved with version {meta.get('version')} is incompatible "
                f"with {EMBEDDING_VERSION}"
            )
        obj = payload["embedder"]
        if not isinstance(obj, Embedder):  # pragma: no cover - defensive
            raise TypeError(f"saved object is not an Embedder: {type(obj)!r}")
        return obj

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(dim={self.dim}, seed={self.seed}, fitted={self.fitted})"


# --------------------------------------------------------------------------- #
# TF-IDF + probabilistic LSA
# --------------------------------------------------------------------------- #
class TfidfSvdEmbedder(Embedder):
    """TF-IDF(word 1-2gram) ++ TF-IDF(char_wb 3-5gram) --TruncatedSVD--> dim."""

    backend = "lsa"

    def __init__(
        self,
        dim: int = DEFAULT_DIM,
        seed: int = DEFAULT_SEED,
        *,
        word_ngrams: tuple[int, int] = (1, 2),
        char_ngrams: tuple[int, int] = (3, 5),
        min_df: int = 1,
        max_df: float = 1.0,
        sublinear_tf: bool = True,
        svd_n_iter: int = 7,
        whiten: bool = False,
        min_docs: int = MIN_DOCS_FOR_LSA,
        min_components: int = MIN_COMPONENTS_FOR_LSA,
    ) -> None:
        super().__init__(dim=dim, seed=seed)
        self.word_ngrams = (int(word_ngrams[0]), int(word_ngrams[1]))
        self.char_ngrams = (int(char_ngrams[0]), int(char_ngrams[1]))
        self.min_df = int(min_df)
        self.max_df = max_df
        self.sublinear_tf = bool(sublinear_tf)
        self.svd_n_iter = int(svd_n_iter)
        self.whiten = bool(whiten)
        self.min_docs = int(min_docs)
        self.min_components = int(min_components)
        self._word_vec: Any = None
        self._char_vec: Any = None
        self._svd: Any = None
        self.n_components = 0
        self.n_features = 0
        self.singular_values_: np.ndarray | None = None

    # -- internals ---------------------------------------------------------- #
    def _make_vectorizers(self) -> tuple[Any, Any]:
        """Lazy import keeps this module importable without scikit-learn installed."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        common = dict(
            lowercase=False,  # normalize_text already did it (and stays Unicode-safe)
            preprocessor=normalize_text,  # module-level, so the vectorizer stays picklable
            strip_accents=None,
            min_df=self.min_df,
            max_df=self.max_df,
            dtype=np.float32,
        )
        word = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"(?u)\b\w+(?:['\-]\w+)*\b",
            ngram_range=self.word_ngrams,
            sublinear_tf=self.sublinear_tf,
            **common,
        )
        char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.char_ngrams,
            sublinear_tf=self.sublinear_tf,
            **common,
        )
        return word, char

    def _stack(self, texts: Sequence[str], *, fit: bool):
        from scipy.sparse import hstack

        if fit:
            xw = self._word_vec.fit_transform(texts)
            xc = self._char_vec.fit_transform(texts)
        else:
            xw = self._word_vec.transform(texts)
            xc = self._char_vec.transform(texts)
        return hstack([xw, xc]).tocsr()

    def _fit_impl(self, docs: Sequence[str]) -> None:
        from sklearn.decomposition import TruncatedSVD

        if len(docs) < self.min_docs:
            raise DegenerateCorpusError(
                f"LSA needs >= {self.min_docs} documents to fit, got {len(docs)}"
            )
        self._word_vec, self._char_vec = self._make_vectorizers()
        x = self._stack(docs, fit=True)
        self.n_features = int(x.shape[1])
        if self.n_features == 0 or x.shape[0] == 0:
            raise DegenerateCorpusError("corpus produced an empty TF-IDF vocabulary")
        # TruncatedSVD requires n_components < min(n_samples, n_features) on sparse input
        k = int(min(self.requested_dim, x.shape[0] - 1, self.n_features - 1))
        if k < 1:
            raise DegenerateCorpusError(f"no usable SVD components for shape {tuple(x.shape)}")
        if k < self.min_components:
            raise DegenerateCorpusError(
                f"{x.shape[0]} documents support only {k} SVD components (< {self.min_components}): "
                "a rank-"
                f"{k} projection inflates cosine between unrelated texts, so LSA is not trustworthy here"
            )
        self.n_components = k
        self._svd = TruncatedSVD(
            n_components=k,
            algorithm="randomized",  # probabilistic low-rank sketch; seeded => reproducible
            n_iter=self.svd_n_iter,
            random_state=self.seed,
        )
        z = self._svd.fit_transform(x)
        self.singular_values_ = np.asarray(self._svd.singular_values_, dtype=np.float64)
        if not np.isfinite(z).all():  # pragma: no cover - defensive
            raise DegenerateCorpusError("TruncatedSVD produced non-finite components")

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if not self.fitted:
            self.fit(texts)
        z = np.asarray(self._svd.transform(self._stack(texts, fit=False)), dtype=np.float64)
        if self.whiten and self.singular_values_ is not None:
            sv = self.singular_values_
            z = z / np.where(sv > 1e-12, sv, 1.0)
        return z

    def params(self) -> dict[str, Any]:
        p = super().params()
        p.update(
            word_ngrams=self.word_ngrams,
            char_ngrams=self.char_ngrams,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf,
            svd_n_iter=self.svd_n_iter,
            whiten=self.whiten,
            min_docs=self.min_docs,
            min_components=self.min_components,
        )
        return p

    def state(self) -> dict[str, Any]:
        sv = self.singular_values_
        out = {
            "n_components": self.n_components,
            "n_features": self.n_features,
            "vocab_words": len(getattr(self._word_vec, "vocabulary_", {}) or {}),
            "vocab_chars": len(getattr(self._char_vec, "vocabulary_", {}) or {}),
            "projection": "randomized-truncated-svd",
        }
        if sv is not None and sv.size:
            total = float(sv @ sv) or 1.0
            out["explained_variance_ratio_top10"] = [round(float(v) ** 2 / total, 6) for v in sv[:10]]
        return out


# --------------------------------------------------------------------------- #
# signed hashing trick
# --------------------------------------------------------------------------- #
class HashingEmbedder(Embedder):
    """Corpus-free signed hashing over the shared feature space.

    Each feature is hashed with BLAKE2b to a bucket index plus a sign bit
    (Charikar/Datar-style stable random projection), weighted by ``1 + log(tf)``.
    Dot products of the projections approximate feature-overlap similarity, the
    dimensionality is fixed regardless of corpus size, and results are stable
    across processes -- which is why this is the fallback for tiny corpora.
    """

    backend = "hash"

    def __init__(
        self,
        dim: int = DEFAULT_DIM,
        seed: int = DEFAULT_SEED,
        *,
        word_ngrams: tuple[int, int] = (1, 2),
        char_ngrams: tuple[int, int] = (3, 5),
        signed: bool = True,
        sublinear_tf: bool = True,
    ) -> None:
        super().__init__(dim=dim, seed=seed)
        self.word_ngrams = (int(word_ngrams[0]), int(word_ngrams[1]))
        self.char_ngrams = (int(char_ngrams[0]), int(char_ngrams[1]))
        self.signed = bool(signed)
        self.sublinear_tf = bool(sublinear_tf)
        self._key = _hash_key(self.seed)

    def _vector(self, text: str) -> np.ndarray:
        feats = features(text, self.word_ngrams, self.char_ngrams)
        v = np.zeros(self.dim, dtype=np.float64)
        for feat, tf in Counter(feats).items():
            digest = blake2b(feat.encode("utf-8"), digest_size=16, key=self._key).digest()
            bucket = int.from_bytes(digest[:8], "little", signed=False) % self.dim
            sign = 1.0 if (not self.signed or digest[8] & 1) else -1.0
            v[bucket] += sign * (1.0 + math.log(tf) if self.sublinear_tf else float(tf))
        return v

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float64)
        return np.vstack([self._vector(t) for t in texts])

    def params(self) -> dict[str, Any]:
        p = super().params()
        p.update(
            word_ngrams=self.word_ngrams,
            char_ngrams=self.char_ngrams,
            signed=self.signed,
            sublinear_tf=self.sublinear_tf,
        )
        return p

    def state(self) -> dict[str, Any]:
        return {"hasher": "blake2b-128-signed", "buckets": self.dim}


def _hash_key(seed: int) -> bytes:
    return b"sdf-hkey" if int(seed) == 0 else (abs(int(seed)) % (1 << 64)).to_bytes(8, "little")


# --------------------------------------------------------------------------- #
# optional network / heavy backends
# --------------------------------------------------------------------------- #
class OpenAIEmbedder(Embedder):
    """Optional ``text-embedding-3-*`` backend; needs a key and the network.

    Never instantiated by the test suite -- the suite runs offline and asserts the
    backend env var is unset.
    """

    backend = "openai"
    offline = False

    def __init__(
        self,
        dim: int = DEFAULT_DIM,
        seed: int = DEFAULT_SEED,
        *,
        model: str | None = None,
        api_key: str | None = None,
        batch_size: int = 96,
        timeout: float = 30.0,
    ) -> None:
        super().__init__(dim=dim, seed=seed)
        self.model = model or os.environ.get("SDF_EMBEDDING_MODEL", "text-embedding-3-small")
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self.batch_size = int(batch_size)
        self.timeout = float(timeout)
        self.api_calls = 0
        self.tokens_used = 0
        self._client_obj: Any = None
        if not self.api_key:
            raise RuntimeError(
                "OpenAI embeddings need OPENAI_API_KEY (or api_key=...); offline backends "
                "are 'lsa' and 'hash'"
            )

    def _client(self):
        from openai import OpenAI  # lazy: keeps `import factory.embeddings` offline-safe

        if self._client_obj is None:
            self._client_obj = OpenAI(api_key=self.api_key, timeout=self.timeout, max_retries=3)
        return self._client_obj

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        client = self._client()
        rows: list[Any] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = [str(t) for t in texts[start : start + self.batch_size]]
            resp = client.embeddings.create(
                model=self.model, input=chunk, dimensions=self.dim, encoding_format="float"
            )
            self.api_calls += 1
            self.tokens_used += int(getattr(getattr(resp, "usage", None), "total_tokens", 0) or 0)
            rows.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
        return np.asarray(rows, dtype=np.float64)

    def params(self) -> dict[str, Any]:
        p = super().params()
        p.update(model=self.model, batch_size=self.batch_size, timeout=self.timeout)
        return p

    def state(self) -> dict[str, Any]:
        return {"api_calls": self.api_calls, "tokens_used": self.tokens_used}


class SentenceTransformerEmbedder(Embedder):
    """Optional local transformer backend, guarded by ``importlib.util.find_spec``."""

    backend = "st"
    offline = False  # no network at embed time, but needs a heavy optional dependency
    ST_MODEL_ENV = "SDF_ST_MODEL"

    def __init__(
        self,
        dim: int = DEFAULT_DIM,
        seed: int = DEFAULT_SEED,
        *,
        model: str | None = None,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        if importlib.util.find_spec("sentence_transformers") is None:
            raise ImportError(
                "sentence_transformers is not installed; use backend='lsa' or 'hash', "
                "or `pip install sentence-transformers`"
            )
        super().__init__(dim=dim, seed=seed)
        self.model_name = model or os.environ.get(self.ST_MODEL_ENV, "all-MiniLM-L6-v2")
        self.device = device
        self.batch_size = int(batch_size)
        self._model_obj: Any = None

    def _model(self):  # pragma: no cover - needs the optional dependency
        if self._model_obj is None:
            from sentence_transformers import SentenceTransformer

            self._model_obj = SentenceTransformer(self.model_name, device=self.device)
        return self._model_obj

    def _encode(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover
        import torch

        torch.manual_seed(self.seed)
        out = self._model().encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return np.asarray(out, dtype=np.float64)

    def params(self) -> dict[str, Any]:
        p = super().params()
        p.update(model=self.model_name, device=self.device, batch_size=self.batch_size)
        return p


# --------------------------------------------------------------------------- #
# factory
# --------------------------------------------------------------------------- #
BACKENDS: dict[str, type[Embedder]] = {
    TfidfSvdEmbedder.backend: TfidfSvdEmbedder,
    HashingEmbedder.backend: HashingEmbedder,
    OpenAIEmbedder.backend: OpenAIEmbedder,
    SentenceTransformerEmbedder.backend: SentenceTransformerEmbedder,
}


def available_backends() -> dict[str, bool]:
    """Which backends can be built right now (no network, no heavy imports)."""
    return {
        "lsa": True,
        "hash": True,
        "st": importlib.util.find_spec("sentence_transformers") is not None,
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
    }


def build_embedder(
    backend: str | None = None,
    corpus: Sequence[str] | None = None,
    dim: int = DEFAULT_DIM,
    *,
    seed: int = DEFAULT_SEED,
    min_docs: int = MIN_DOCS_FOR_LSA,
    min_components: int = MIN_COMPONENTS_FOR_LSA,
    **kwargs: Any,
) -> Embedder:
    """Build (and, given ``corpus``, fit) an embedder.

    ``backend=None`` resolves from ``SDF_EMBEDDING_BACKEND`` and defaults to
    ``"auto"``: LSA when the corpus can support the projection, otherwise the
    deterministic hashing embedder. An explicit ``"lsa"`` propagates
    :class:`DegenerateCorpusError` rather than silently switching backends.
    """
    name = (backend or os.environ.get(BACKEND_ENV_VAR) or "auto").strip().lower()
    if name in ("", "auto"):
        return _auto_embedder(corpus, dim, seed, min_docs, min_components, kwargs)
    if name not in BACKENDS:
        raise ValueError(
            f"unknown embedding backend {name!r}; choose from auto/{'/'.join(sorted(BACKENDS))}"
        )
    klass = BACKENDS[name]
    if name == "lsa":
        kwargs.setdefault("min_docs", min_docs)
        kwargs.setdefault("min_components", min_components)
    embedder = klass(dim=dim, seed=seed, **kwargs)
    if corpus is not None:
        embedder.fit(list(corpus))
    return embedder


def _auto_embedder(
    corpus: Sequence[str] | None,
    dim: int,
    seed: int,
    min_docs: int,
    min_components: int,
    kwargs: dict[str, Any],
) -> Embedder:
    docs = None if corpus is None else [str(t) for t in corpus]
    if docs is None or len(docs) < max(1, min_docs):
        return build_embedder("hash", docs, dim, seed=seed, **kwargs)
    try:
        return build_embedder("lsa", docs, dim, seed=seed, min_components=min_components, **kwargs)
    except DegenerateCorpusError as exc:
        fallback = build_embedder("hash", docs, dim, seed=seed, **kwargs)
        fallback.auto_fallback_reason = str(exc)
        return fallback


def embed_texts(
    texts: Sequence[str],
    backend: str | None = None,
    dim: int = DEFAULT_DIM,
    *,
    seed: int = DEFAULT_SEED,
    **kwargs: Any,
) -> np.ndarray:
    """One-shot: build an embedder fitted on ``texts`` and embed them."""
    return build_embedder(backend, corpus=texts, dim=dim, seed=seed, **kwargs).embed(texts)


def embedder_from_meta(meta: dict[str, Any]) -> Embedder:
    """Rebuild an unfitted embedder from manifest metadata (lineage round-trip)."""
    klass = BACKENDS.get(str(meta.get("backend", "")), HashingEmbedder)
    params = {k: v for k, v in dict(meta.get("params") or {}).items() if k not in ("dim", "seed")}
    dim = int(meta.get("requested_dim") or meta.get("dim") or DEFAULT_DIM)
    seed = int(meta.get("seed", DEFAULT_SEED))
    try:
        return klass(dim=dim, seed=seed, **params)
    except (TypeError, RuntimeError, ImportError):  # optional backend unavailable here
        return HashingEmbedder(dim=dim, seed=seed)
