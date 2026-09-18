"""Full pipeline: seed -> expand -> filter -> dedup -> export, with lineage.

``run`` keeps its original signature (positional ``seeds_path, out_path, llm``)
so nothing that calls it breaks. Everything new is keyword-only with a default
that reproduces the old behaviour:

* ``llm_filter`` -- the judge model, previously smuggled in by monkeypatching a
  module attribute (which did nothing: :mod:`factory.filter` has no ``score``).
  It now flows through as an argument, and defaults to ``llm`` so single-model
  callers keep working.
* ``dedup_threshold`` / ``dedup_strategy`` / ``embed_backend`` -- the knobs that
  used to be silently ignored. ``None`` means "take it from ``SETTINGS``", which
  is how ``SDF_DEDUP_THRESHOLD`` finally reaches the dedup stage.
* ``report`` / ``split`` / ``manifest`` -- write the quality report, the
  leakage-audited splits and ``run.json`` next to the JSONL.

The return value of ``run`` is still the example count; :func:`run_detailed`
returns the whole :class:`PipelineResult` (pool, audit, report, manifest) when
you need the numbers rather than the count.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import SETTINGS, Settings
from .dedup import DedupAudit, dedup_audit
from .embeddings import Embedder, build_embedder
from .expand import expand_seed
from .export import to_jsonl
from .filter import filter_candidates
from .logging_config import get_logger
from .manifest import RunManifest, stage_counts, write_manifest
from .metrics import DataReport, quality_report
from .seed import Example, example_text, load_seeds
from .split import SplitResult, split_dataset

log = get_logger(__name__)


@dataclass
class PipelineResult:
    """Everything a run produced, so the caller can report without recomputing."""

    pool: list[Example] = field(default_factory=list)
    seeds: list[Example] = field(default_factory=list)
    generated: list[Example] = field(default_factory=list)
    stages: dict[str, int] = field(default_factory=dict)
    dedup: DedupAudit | None = None
    embedder: Embedder | None = None
    quality: DataReport | None = None
    split: SplitResult | None = None
    manifest: RunManifest | None = None
    out_path: Path | None = None
    manifest_path: Path | None = None
    split_paths: dict[str, Path] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.pool)

    def as_dict(self) -> dict[str, Any]:
        return {
            "stages": dict(self.stages),
            "dedup": self.dedup.as_dict() if self.dedup else {},
            "quality": self.quality.as_dict() if self.quality else {},
            "split": self.split.as_dict() if self.split else {},
            "out_path": str(self.out_path) if self.out_path else "",
        }


def run(
    seeds_path: Path,
    out_path: Path,
    llm,
    min_score: int = 6,
    *,
    llm_filter=None,
    dedup_threshold: float | None = None,
    dedup_strategy: str | None = None,
    embed_backend: str | None = None,
    embed_dim: int | None = None,
    variants_per_seed: int | None = None,
    settings: Settings | None = None,
    report: bool = False,
    split: bool = False,
    manifest: bool = True,
    seed_corpus: list[Example] | None = None,
) -> int:
    """Run the pipeline and return the number of exported examples.

    ``llm`` is the expander (``prompt -> str``); ``llm_filter`` defaults to it
    for the judge role. Failures in expand/filter degrade to the seeds rather
    than aborting the run, which is what makes the tool usable against a flaky
    endpoint -- but the stage counts in the manifest make that degradation
    visible instead of silent.
    """
    result = run_detailed(
        seeds_path, out_path, llm, min_score,
        llm_filter=llm_filter, dedup_threshold=dedup_threshold,
        dedup_strategy=dedup_strategy, embed_backend=embed_backend,
        embed_dim=embed_dim, variants_per_seed=variants_per_seed,
        settings=settings, report=report, split=split, manifest=manifest,
        seed_corpus=seed_corpus,
    )
    return result.n


def run_detailed(
    seeds_path: Path,
    out_path: Path,
    llm,
    min_score: int = 6,
    *,
    llm_filter=None,
    dedup_threshold: float | None = None,
    dedup_strategy: str | None = None,
    embed_backend: str | None = None,
    embed_dim: int | None = None,
    variants_per_seed: int | None = None,
    settings: Settings | None = None,
    report: bool = False,
    split: bool = False,
    manifest: bool = True,
    seed_corpus: list[Example] | None = None,
    split_ratios: tuple[float, float, float] | None = None,
) -> PipelineResult:
    """Same as :func:`run` but hand back the pool, audits and manifest."""
    s = settings or SETTINGS
    judge = llm_filter or llm
    threshold = float(s.dedup_threshold if dedup_threshold is None else dedup_threshold)
    strategy = dedup_strategy or s.dedup_strategy
    backend = embed_backend or s.embed_backend
    dim = int(embed_dim or s.embed_dim)
    n_variants = int(s.variants_per_seed if variants_per_seed is None else variants_per_seed)

    seeds = list(seed_corpus) if seed_corpus is not None else list(load_seeds(seeds_path))
    result = PipelineResult(seeds=seeds, out_path=Path(out_path))
    if not seeds:
        to_jsonl([], out_path)
        result.stages = stage_counts(seeds=0, expanded=0, filtered=0, deduped=0, exported=0)
        if manifest:
            result.manifest = _make_manifest(result, s, threshold, strategy, backend, dim,
                                             n_variants, min_score, out_path)
            result.manifest_path = write_manifest(result.manifest, out_path)
        log.info("pipeline_empty", extra={"stage": "seed", "seeds": 0})
        return result

    # -- expand -----------------------------------------------------------
    pool: list[Example] = list(seeds)
    generated: list[Example] = []
    expand_errors = 0
    for seed in seeds:
        try:
            variants = expand_seed(seed, llm, n=n_variants)
        except Exception as exc:  # noqa: BLE001 - a dead endpoint must not kill the run
            expand_errors += 1
            log.warning("expand_failed", extra={"seed": seed.input[:60], "error": type(exc).__name__})
            continue
        generated.extend(variants)
        pool.extend(variants)

    # -- filter -----------------------------------------------------------
    try:
        filtered = filter_candidates(pool, seeds, judge, min_score=min_score)
    except Exception as exc:  # noqa: BLE001
        filtered = list(seeds)
        log.warning("filter_failed", extra={"error": type(exc).__name__, "fallback": "seeds"})
    if not filtered:
        filtered = list(seeds)

    # -- dedup ------------------------------------------------------------
    embedder = build_embedder(backend, corpus=[example_text(e) for e in filtered], dim=dim,
                              seed=s.embed_seed)
    pool, audit = dedup_audit(
        filtered, threshold, embedder=embedder, strategy=strategy, dim=dim, seed=s.embed_seed
    )
    log.info("dedup_done", extra={"detail": audit.summary()})

    result.pool = pool
    result.generated = [e for e in pool if e not in seeds]
    result.embedder = embedder
    result.dedup = audit
    result.stages = stage_counts(
        seeds=len(seeds), expanded=len(pool) + len(generated) - len(seeds) + audit.n_dropped,
        filtered=len(filtered), deduped=len(pool), exported=len(pool),
    )
    result.stages["expand_errors"] = expand_errors
    result.stages["generated"] = len(generated)

    to_jsonl(pool, out_path)

    # -- optional: quality report, split, manifest --------------------------
    if report:
        result.quality = quality_report(seeds, pool, embedder=embedder, leak_threshold=threshold)
        for line in result.quality.warnings:
            log.warning("quality", extra={"warning": line})
        log.info("quality_done", extra={"summary": result.quality.summary()})
    if split:
        result.split = split_dataset(
            pool, split_ratios or s.split_ratios, threshold=threshold, embedder=embedder,
            strategy=strategy, dim=dim, seed=s.embed_seed,
        )
        result.split_paths = write_splits(result.split, Path(out_path).parent)
        log.info("split_done", extra={"summary": result.split.summary()})

    result.manifest = _make_manifest(
        result, s, threshold, strategy, backend, dim, n_variants, min_score, out_path,
        seeds_path=seeds_path,
    )
    if manifest:
        result.manifest_path = write_manifest(result.manifest, out_path)
    return result


def _make_manifest(
    result: PipelineResult,
    s: Settings,
    threshold: float,
    strategy: str,
    backend: str,
    dim: int,
    n_variants: int,
    min_score: int,
    out_path: Path,
    *,
    seeds_path: Path | None = None,
) -> RunManifest:
    config = {
        "model": s.model,
        "temperature": s.temperature,
        "min_score": min_score,
        "dedup_threshold": threshold,
        "dedup_strategy": strategy,
        "embed_backend": backend,
        "embed_dim": dim,
        "embed_seed": s.embed_seed,
        "variants_per_seed": n_variants,
        "split_ratios": list(s.split_ratios),
        "lsh_num_perm": s.lsh_num_perm,
        "ivf_n_probe": s.ivf_n_probe,
    }
    return RunManifest.from_run(
        config=config,
        seeds=result.seeds,
        seeds_path=seeds_path,
        stages=result.stages,
        embedder=result.embedder,
        dedup=result.dedup,
        split=result.split,
        quality=result.quality,
        model_version=s.model,
        include_timestamp=s.manifest_timestamp,
        extra={"out_path": str(out_path)},
    )


def write_splits(split: SplitResult, directory: Path, *, prefix: str = "split") -> dict[str, Path]:
    """Write each split to ``split-{train,val,test}.jsonl`` in ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, examples in split.splits.items():
        p = directory / f"{prefix}-{name}.jsonl"
        to_jsonl(examples, p)
        paths[name] = p
    return paths
