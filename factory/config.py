"""Central configuration from env / .env.

Every field has an environment default, so ``Settings()`` constructs keyless and
offline -- that is what lets the test suite run without an API key. Nothing here
touches the network at import time.

The knobs fall into three groups:

* generation -- ``model``, ``temperature``, ``timeout``, ``variants_per_seed``
* judging -- ``min_score`` (the filter's keep threshold)
* data quality -- the dedup/embedding/split parameters that end up in the run
  manifest, because a dataset is only reproducible if these are recorded.

``dedup_threshold`` lives in *cosine* space (embedding space); the LSH banding
thresholds in :mod:`factory.ann` live in Jaccard space and are derived from it,
so there is a single user-facing dial.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _ratio_tuple(raw: str) -> tuple[float, ...]:
    """Parse ``"0.8,0.1,0.1"`` into floats.

    Kept permissive (a single number, trailing commas, stray spaces) but it does
    not silently normalise: a tuple that does not sum to ~1 is still returned and
    :func:`factory.split.split_dataset` raises, so the mistake surfaces at the
    call site rather than in config.
    """
    parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    return tuple(float(p) for p in parts)


def _int_env(name: str, default: str) -> int:
    raw = _env(name, default).strip()
    return int(float(raw)) if raw else 0


@dataclass(frozen=True)
class Settings:
    # -- provider ---------------------------------------------------------
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY", ""))
    model: str = field(default_factory=lambda: _env("SDF_MODEL", "gpt-4o-mini"))
    temperature: float = field(default_factory=lambda: float(_env("SDF_TEMPERATURE", "0.7")))
    timeout: int = field(default_factory=lambda: _int_env("SDF_TIMEOUT", "30"))

    # -- generation / judging --------------------------------------------
    min_score: int = field(default_factory=lambda: _int_env("SDF_MIN_SCORE", "6"))
    variants_per_seed: int = field(default_factory=lambda: _int_env("SDF_VARIANTS_PER_SEED", "3"))

    # -- dedup ------------------------------------------------------------
    dedup_threshold: float = field(default_factory=lambda: float(_env("SDF_DEDUP_THRESHOLD", "0.92")))
    # exact | lsh | ivf | hybrid -- see factory/ann.py
    dedup_strategy: str = field(default_factory=lambda: _env("SDF_DEDUP_STRATEGY", "ivf").strip().lower())
    lsh_num_perm: int = field(default_factory=lambda: _int_env("SDF_LSH_NUM_PERM", "128"))
    ivf_n_probe: int = field(default_factory=lambda: _int_env("SDF_IVF_N_PROBE", "4"))

    # -- embeddings -------------------------------------------------------
    # Empty means "auto": LSA when the corpus supports it, hashing otherwise.
    embed_backend: str = field(default_factory=lambda: _env("SDF_EMBEDDING_BACKEND", "").strip().lower())
    embed_dim: int = field(default_factory=lambda: _int_env("SDF_EMBED_DIM", "256"))
    embed_seed: int = field(default_factory=lambda: _int_env("SDF_EMBED_SEED", "13"))

    # -- splitting / lineage ----------------------------------------------
    split_ratios: tuple[float, ...] = field(default_factory=lambda: _ratio_tuple(_env("SDF_SPLIT_RATIOS", "0.8,0.1,0.1")))
    # Timestamps break byte-reproducibility of run.json, so they are opt-in.
    manifest_timestamp: bool = field(default_factory=lambda: bool(_int_env("SDF_MANIFEST_TIMESTAMP", "0")))

    @property
    def has_key(self) -> bool:
        return bool(self.openai_api_key)

    def as_dict(self) -> dict:
        """Plain dict of every setting except the secret."""
        return {k: v for k, v in self.__dict__.items() if k != "openai_api_key"}


SETTINGS = Settings()
