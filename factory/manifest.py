"""Run manifest: the lineage record that makes a dataset reproducible.

``run.json`` sits next to ``out.jsonl`` and answers the only question that
matters after a dataset has been consumed for months: *what exactly produced
this?*. It records

* a canonical-JSON SHA-256 **config hash** (so two runs that differ only in an
  unused key get different hashes, and two runs with the same hash are the same
  run),
* the **seed corpus hash** and example counts per stage, so a 412 -> 380 -> 351
  shrink is visible instead of implied,
* the **embedder** name, backend, dim, version, seed and fit-corpus hash, which
  is what makes a similarity threshold interpretable a year later,
* every **threshold** actually used (filter score, dedup cosine, split
  leakage), and
* the **model_version** and pipeline version.

Timestamps are opt-in (``include_timestamp=True``) because a wall-clock field
breaks byte-for-byte reproducibility of the manifest itself: the default
manifest of two runs of the same inputs on the same code is *identical*, which
is exactly the property you want when you diff them.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "MANIFEST_NAME",
    "PIPELINE_VERSION",
    "RunManifest",
    "config_hash",
    "corpus_hash",
    "python_version",
    "read_manifest",
    "stage_counts",
    "write_manifest",
]

MANIFEST_NAME = "run.json"
PIPELINE_VERSION = "1.1.0"


# --------------------------------------------------------------------------- #
# hashing primitives
# --------------------------------------------------------------------------- #
def _canonical(value: Any) -> Any:
    """JSON-safe, order-stable projection of nested config data."""
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple, Sequence)) and not isinstance(value, (str, bytes)):
        return [_canonical(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
            return str(value)
        return value
    if hasattr(value, "as_dict") and callable(value.as_dict):  # dataclass-like records
        return _canonical(value.as_dict())
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict

        return _canonical(asdict(value))
    return str(value)


def canonical_json(value: Any) -> str:
    """The exact bytes :func:`config_hash` digests -- sorted keys, no whitespace."""
    return json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def config_hash(value: Any) -> str:
    """SHA-256 of the canonical JSON of a config mapping (16 hex chars shown, 64 stored)."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def corpus_hash(texts: Sequence[str]) -> str:
    """Order-independent digest of a corpus.

    Per-item digests are sorted before folding, so two seed files with the same
    content in a different order hash the same -- order is not part of a
    dataset's identity, but it *is* part of a run's, which is why
    :func:`factory.pipeline.run` also records the input file's own digest.
    """
    digests = sorted(hashlib.sha256(str(t).encode("utf-8")).hexdigest() for t in texts)
    fold = hashlib.sha256()
    for d in digests:
        fold.update(d.encode("ascii"))
    return fold.hexdigest()


def python_version() -> str:
    return (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} "
        f"{platform.system().lower()}-{platform.machine().lower()}"
    )


def stage_counts(*, seeds: int, expanded: int, filtered: int, deduped: int, exported: int) -> dict[str, int]:
    """Per-stage counts, plus the drop rates each stage implies."""
    stages = {"seeds": seeds, "expanded": expanded, "filtered": filtered,
              "deduped": deduped, "exported": exported}
    out: dict[str, int] = dict(stages)
    prev = None
    for name in ("seeds", "expanded", "filtered", "deduped", "exported"):
        out[f"dropped_at_{name}"] = 0 if prev is None else max(0, prev - stages[name])
        prev = stages[name]
    return out


# --------------------------------------------------------------------------- #
# the manifest
# --------------------------------------------------------------------------- #
@dataclass
class RunManifest:
    """Serializable lineage record for one pipeline run."""

    created_utc: str | None = None
    pipeline_version: str = PIPELINE_VERSION
    model_version: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    config_hash: str = ""
    seeds_file: str = ""
    seeds_hash: str = ""
    n_seeds: int = 0
    seed_corpus_hash: str = ""
    stages: dict[str, int] = field(default_factory=dict)
    embedder: dict[str, Any] = field(default_factory=dict)
    dedup: dict[str, Any] = field(default_factory=dict)
    split: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    # -- construction ------------------------------------------------------ #
    @classmethod
    def from_run(
        cls,
        *,
        config: Mapping[str, Any] | None = None,
        seeds: Sequence[Any] = (),
        seeds_path: Path | str | None = None,
        stages: Mapping[str, Any] | None = None,
        embedder: Any = None,
        dedup: Mapping[str, Any] | Any | None = None,
        split: Mapping[str, Any] | Any | None = None,
        quality: Mapping[str, Any] | Any | None = None,
        model_version: str | None = None,
        include_timestamp: bool = False,
        extra: Mapping[str, Any] | None = None,
    ) -> "RunManifest":
        """Assemble a manifest from already-computed pieces.

        ``embedder`` may be an :class:`~factory.embeddings.Embedder` (its
        ``as_dict()`` is taken), ``dedup`` a :class:`~factory.dedup.DedupAudit`,
        ``split`` a :class:`~factory.split.SplitResult`, ``quality`` a
        :class:`~factory.metrics.DataReport` -- anything with ``as_dict()`` is
        accepted, so the manifest never needs to know their internals.
        """
        cfg = dict(config or {})
        texts = [str(getattr(s, "input", s)) for s in seeds]

        def _as_dict(v: Any) -> dict[str, Any]:
            if v is None:
                return {}
            if isinstance(v, Mapping):
                return dict(v)
            as_dict = getattr(v, "as_dict", None)
            return dict(as_dict()) if callable(as_dict) else {"value": str(v)}

        thresholds: dict[str, float] = {}
        for key in ("dedup_threshold", "split_threshold", "leak_threshold", "min_score",
                    "filter_min_score", "lsh_margin"):
            if key in cfg:
                thresholds[key] = float(cfg[key])
        for src, key in ((_as_dict(dedup), "dedup_threshold"), (_as_dict(split), "split_threshold")):
            if key not in thresholds and src.get(key) is not None:
                thresholds[key] = float(src[key])

        from .embeddings import EMBEDDING_VERSION

        dedup_d = _as_dict(dedup)
        split_d = _as_dict(split)
        quality_d = _as_dict(quality)
        emb_d = dict(embedder.as_dict()) if hasattr(embedder, "as_dict") else _as_dict(embedder)

        return cls(
            created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds") if include_timestamp else None,
            model_version=str(model_version or cfg.get("model") or ""),
            config=_canonical(cfg),  # type: ignore[arg-type]
            config_hash=config_hash(cfg),
            seeds_file=str(seeds_path) if seeds_path else "",
            seeds_hash=(
                hashlib.sha256(Path(seeds_path).read_bytes()).hexdigest()
                if seeds_path and Path(seeds_path).exists()
                else corpus_hash(texts)
            ),
            n_seeds=len(texts),
            seed_corpus_hash=corpus_hash(texts),
            stages=dict(stages or {}),
            embedder={
                "name": emb_d.get("name", ""),
                "backend": emb_d.get("backend", ""),
                "version": emb_d.get("version", EMBEDDING_VERSION),
                "dim": emb_d.get("dim"),
                "requested_dim": emb_d.get("requested_dim"),
                "seed": emb_d.get("seed"),
                "fitted": emb_d.get("fitted"),
                "n_fit_docs": emb_d.get("n_fit_docs"),
                "fit_corpus_hash": emb_d.get("fit_corpus_hash"),
                "auto_fallback_reason": emb_d.get("auto_fallback_reason"),
                "offline": emb_d.get("offline"),
            },
            dedup=dedup_d,
            split=split_d,
            quality=quality_d,
            thresholds=thresholds,
            environment={
                "python": python_version(),
                "numpy": _module_version("numpy"),
                "scikit_learn": _module_version("sklearn"),
                "scipy": _module_version("scipy"),
                "offline": not bool(os.environ.get("OPENAI_API_KEY")),
                "embedding_backend_env": os.environ.get("SDF_EMBEDDING_BACKEND", ""),
            },
            output={},
            extra=_canonical(extra) and dict(extra or {}) or {},  # type: ignore[arg-type]
        )

    # -- serialisation ----------------------------------------------------- #
    def as_dict(self) -> dict[str, Any]:
        return _canonical(self.__dict__)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunManifest":
        fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in dict(data).items() if k in fields})

    def write(self, path: Path | str, *, name: str = MANIFEST_NAME) -> Path:
        return write_manifest(self, path, name=name)

    # -- reproduction ------------------------------------------------------ #
    def reproduce_command(self, *, out: str = "out.jsonl") -> str:
        """The exact CLI invocation that regenerates this dataset."""
        cfg = self.config if isinstance(self.config, dict) else {}
        parts = [
            f"{sys.executable} -m factory",
            str(self.seeds_file or "seeds.json"),
            out,
            f"--min-score {cfg.get('min_score', 6)}",
            f"--dedup-threshold {cfg.get('dedup_threshold', 0.92)}",
            f"--dedup-strategy {cfg.get('dedup_strategy', 'ivf')}",
            f"--embed-backend {cfg.get('embed_backend', 'auto')}",
            f"--embed-dim {cfg.get('embed_dim', 256)}",
            f"--seed {cfg.get('embed_seed', 13)}",
        ]
        if cfg.get("split"):
            parts.append("--split")
        return " ".join(parts)

    def same_inputs(self, other: "RunManifest | Mapping[str, Any]") -> bool:
        """Whether another manifest describes the same reproducible run.

        Compares config hashes, not timestamps or environment: the point is to
        tell "same dataset" from "same code, different machine".
        """
        oh = other.config_hash if isinstance(other, RunManifest) else str(other.get("config_hash", ""))
        return bool(oh) and oh == self.config_hash and self.seed_corpus_hash == (
            other.seed_corpus_hash if isinstance(other, RunManifest) else other.get("seed_corpus_hash")
        )

    def summary(self) -> str:
        stage = self.stages or {}
        return (
            f"run {self.config_hash[:12]} model={self.model_version or '-'} "
            f"embedder={self.embedder.get('name') or '-'} "
            f"seeds={self.n_seeds} stages={stage}"
        )


def _module_version(name: str) -> str:
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", ""))
    except Exception:  # noqa: BLE001 - optional dependency, absence is data
        return ""


def write_manifest(manifest: RunManifest | Mapping[str, Any], path: Path | str, *, name: str = MANIFEST_NAME) -> Path:
    """Write ``run.json`` next to (or into) ``path``; returns the file written."""
    target = Path(path)
    if target.suffix and target.name != name:  # a file path was given: use its directory
        target = target.parent
    target.mkdir(parents=True, exist_ok=True)
    out = target / name
    data = manifest if isinstance(manifest, RunManifest) else RunManifest.from_dict(manifest)
    out.write_text(data.to_json() + "\n", encoding="utf-8")
    return out


def read_manifest(path: Path | str) -> RunManifest:
    """Load ``run.json`` -- pass the JSONL path or the manifest path."""
    target = Path(path)
    if target.name != MANIFEST_NAME and (target / MANIFEST_NAME).exists():
        target = target / MANIFEST_NAME
    if target.is_dir():
        target = target / MANIFEST_NAME
    return RunManifest.from_dict(json.loads(target.read_text(encoding="utf-8")))
