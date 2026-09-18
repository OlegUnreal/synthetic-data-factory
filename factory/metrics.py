"""Dataset quality and privacy metrics.

A synthetic dataset is only worth shipping if you can say *why* it is worth
shipping. This module turns a seed/generated pair into a :class:`DataReport`
of numbers that an ML reviewer can check:

Lexical diversity
    :func:`type_token_ratio` (TTR) and Distinct-1 / Distinct-2. A expansion
    model that has collapsed onto a template shows up here first: TTR falls
    while the example count grows.

Length profile
    Mean / median / percentiles overall and per tag, so a tag that is quietly
    producing one-line answers is visible instead of averaged away.

Label balance
    Tag distribution compared against the *seed* distribution with a
    chi-square goodness-of-fit statistic and the Jensen-Shannon divergence.
    Generation that over-fits one topic drifts away from the seed mix and this
    reports it with a p-value.

Embedding-space health
    Mean / median nearest-neighbour cosine distance. Too small means the pool
    is crowded (little new information per example); too large means the
    examples are unrelated to each other and the set may be noise.

Privacy / memorisation
    How many generated examples are at >= ``leak_threshold`` cosine similarity
    to *any* seed. A high rate means the "generation" step is paraphrasing the
    seeds instead of producing new data, which both leaks training data and
    inflates apparent dataset size. Exact carry-overs are reported separately
    from paraphrases because they are a different failure (a copy, not a
    memorised rewrite).

Everything is deterministic and offline: the only model used is the embedder
from :mod:`factory.embeddings`, and the default backend never touches network.

Costs: diversity/profile are O(total tokens); nearest-neighbour distance and
the memorisation check build one dense similarity block of
``n_generated x (n_generated + n_seeds)`` float32, i.e. O(n^2 * d) time and
O(n^2) memory -- fine up to a few tens of thousands of examples, beyond which
the ANN structures in :mod:`factory.ann` should replace the exact pass.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .embeddings import Embedder, build_embedder, cosine_matrix
# example_text is canonical in factory.seed: metrics judge the whole training
# string (prompt + completion) while factory.dedup keys on the prompt only -- see
# factory.seed.example_text for why that asymmetry is deliberate.
from .seed import Example, example_text
from .text import normalize_text, tokenize

try:  # optional, only used for the label-balance test
    from scipy.spatial.distance import jensenshannon as _jensen_shannon
    from scipy.stats import chisquare as _chisquare
except ImportError:  # pragma: no cover - scipy is a hard dep, this is the guard
    _jensen_shannon = None
    _chisquare = None

__all__ = [
    "DEFAULT_LEAK_THRESHOLD",
    "DEFAULT_PERCENTILES",
    "DataReport",
    "distinct_n",
    "distribution_drift",
    "example_text",
    "length_profile",
    "memorisation",
    "nn_cosine_distances",
    "quality_report",
    "tag_distribution",
    "tag_length_profile",
    "type_token_ratio",
]

#: generated example counted as a paraphrase-leak at/above this cosine
DEFAULT_LEAK_THRESHOLD = 0.92
DEFAULT_PERCENTILES: tuple[float, ...] = (50.0, 90.0, 95.0, 99.0)


def _py(value: Any) -> Any:
    """Numpy -> plain Python, recursively, so ``as_dict()`` is JSON-serialisable."""
    if isinstance(value, dict):
        return {str(k): _py(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_py(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_py(v) for v in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    return str(value)


# --------------------------------------------------------------------------
# lexical diversity
# --------------------------------------------------------------------------
def type_token_ratio(tokens: Sequence[str]) -> float:
    """Distinct tokens / total tokens. 1.0 = every token unique, ->0 = repetitive.

    Length-normalised diversity is the *hardest* single number to fake: adding
    examples of the same shape adds tokens but not types.
    """
    n = len(tokens)
    return round(len(set(tokens)) / n, 6) if n else 0.0


def distinct_n(texts: Sequence[str], n: int = 1) -> float:
    """Distinct n-grams / total n-grams over the whole corpus (Taille et al.).

    Distinct-1 unigram coverage, Distinct-2 bigram variety -- a model that only
    recombines a handful of phrases has a normal Distinct-1 and a bad Distinct-2.
    """
    counts: Counter = Counter()
    total = 0
    for text in texts:
        toks = tokenize(text)
        if len(toks) < n:
            continue
        for i in range(len(toks) - n + 1):
            counts[tuple(toks[i : i + n])] += 1
            total += 1
    return round(len(counts) / total, 6) if total else 0.0


def diversity_metrics(texts: Sequence[str]) -> dict[str, float]:
    """TTR plus Distinct-1/2 for a corpus, as one block."""
    tokens = [t for text in texts for t in tokenize(text)]
    return {
        "type_token_ratio": type_token_ratio(tokens),
        "distinct_1": distinct_n(texts, 1),
        "distinct_2": distinct_n(texts, 2),
        "n_tokens": len(tokens),
        "n_types": len(set(tokens)),
    }


# --------------------------------------------------------------------------
# lengths
# --------------------------------------------------------------------------
def _stats(values: Sequence[float], percentiles: Sequence[float]) -> dict[str, Any]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0, "mean": None, "min": None, "max": None,
            "p50": None, "p90": None, "p95": None, "p99": None,
            "percentiles": {f"p{float(p):g}": None for p in percentiles},
        }
    qs = np.percentile(arr, [float(p) for p in percentiles]) if arr.size else []
    out: dict[str, Any] = {
        "count": int(arr.size),
        "mean": round(float(arr.mean()), 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
    }
    named = {}
    for p, q in zip(percentiles, np.atleast_1d(qs)):
        named[f"p{float(p):g}"] = round(float(q), 4)
    out["percentiles"] = named
    out.update({k: v for k, v in named.items() if k in ("p50", "p90", "p95", "p99")})
    return out


def length_profile(texts: Sequence[str], *, percentiles: Sequence[float] = DEFAULT_PERCENTILES) -> dict[str, Any]:
    """Word and character length statistics for a set of texts."""
    words = [len(tokenize(t)) for t in texts]
    chars = [len(normalize_text(t)) for t in texts]
    return {
        "n_examples": len(texts),
        "words": _stats(words, percentiles),
        "chars": _stats(chars, percentiles),
        "empty_count": sum(1 for w in words if w == 0),
    }


def tag_length_profile(
    examples: Sequence[Example],
    *,
    text_of: Callable[[Example], str] = example_text,
    percentiles: Sequence[float] = DEFAULT_PERCENTILES,
) -> dict[str, dict[str, Any]]:
    """Length profile *per tag*: a tag can hide a degenerate shape inside a good mean."""
    by_tag: dict[str, list[str]] = {}
    for e in examples:
        by_tag.setdefault(str(e.tag or "_untagged"), []).append(text_of(e))
    return {tag: length_profile(txts, percentiles=percentiles) for tag, txts in sorted(by_tag.items())}


# --------------------------------------------------------------------------
# label balance
# --------------------------------------------------------------------------
def tag_distribution(examples: Sequence[Example]) -> dict[str, int]:
    """Label -> count. Empty tags are grouped under ``_untagged``."""
    counts: Counter = Counter(str(e.tag or "_untagged") for e in examples)
    return dict(sorted(counts.items()))


def _proportions(counts: Mapping[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()} if total else {}


def distribution_drift(
    observed: Mapping[str, int],
    reference: Mapping[str, int],
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Goodness-of-fit of ``observed`` against the ``reference`` label mix.

    Two complementary statistics:

    * ``chi_square`` -- :func:`scipy.stats.chisquare` with expected counts taken
      from the reference proportions rescaled to the observed total. Only the
      shared labels can enter the test (an expected count of 0 makes the
      statistic undefined), so labels that appear only in the observed set are
      counted in ``unseen_in_reference`` instead of being silently dropped.
    * ``jensen_shannon`` -- base-2 JS divergence (0 = identical, 1 = disjoint
      supports), which *is* defined over the union of labels.

    ``drifted`` is the chi-square verdict at ``alpha``; it is ``None`` (not
    ``False``) when there are too few shared labels for the test to mean
    anything. That distinction matters: "no evidence of drift" and "could not
    test" are different answers.
    """
    obs = {str(k): int(v) for k, v in observed.items()}
    ref = {str(k): int(v) for k, v in reference.items()}
    labels = sorted(set(obs) | set(ref))
    obs_total, ref_total = sum(obs.values()), sum(ref.values())
    shared = [l for l in labels if obs.get(l, 0) > 0 or ref.get(l, 0) > 0]
    fit_labels = [l for l in shared if ref.get(l, 0) > 0 and ref_total > 0]

    out: dict[str, Any] = {
        "labels": labels,
        "observed": {l: obs.get(l, 0) for l in labels},
        "reference": {l: ref.get(l, 0) for l in labels},
        "observed_proportion": {l: round(obs.get(l, 0) / obs_total, 6) for l in labels} if obs_total else {},
        "reference_proportion": {l: round(ref.get(l, 0) / ref_total, 6) for l in labels} if ref_total else {},
        "max_proportion_gap": round(
            max(
                (
                    abs(obs.get(l, 0) / obs_total - ref.get(l, 0) / ref_total)
                    for l in labels
                ),
                default=0.0,
            ),
            6,
        )
        if (obs_total and ref_total)
        else None,
        "n_new_labels": len([l for l in labels if ref.get(l, 0) == 0 and obs.get(l, 0) > 0]),
        "n_lost_labels": len([l for l in labels if obs.get(l, 0) == 0 and ref.get(l, 0) > 0]),
    }

    # chi-square goodness of fit on shared labels
    if _chisquare is None or len(fit_labels) < 2 or obs_total < 1:
        out.update(chi_square=None, chi2_p_value=None, chi2_dof=None,
                   chi2_available=False, drifted=None,
                   chi2_note="scipy missing or fewer than 2 shared non-empty labels")
    else:
        f_obs = np.asarray([obs[l] for l in fit_labels], dtype=np.float64)
        # rescale reference proportions onto the observed total of these labels
        ref_sum = sum(ref[l] for l in fit_labels)
        obs_sum = sum(obs[l] for l in fit_labels)
        f_exp = np.asarray([ref[l] / ref_sum * obs_sum for l in fit_labels], dtype=np.float64)
        res = _chisquare(f_obs, f_exp=f_exp)
        out.update(
            chi_square=round(float(res.statistic), 4),
            chi2_p_value=(None if (res.pvalue is None or not np.isfinite(res.pvalue))
                          else round(float(res.pvalue), 8)),
            chi2_dof=len(fit_labels) - 1,
            chi2_available=True,
            drifted=bool(res.pvalue < alpha),
            chi2_labels=fit_labels,
            chi2_note=None,
        )

    # Jensen-Shannon over the union of labels
    if not labels or not obs_total or not ref_total:
        out.update(js_divergence=None, js_available=False,
                   js_note="one of the distributions is empty")
    elif _jensen_shannon is None:  # pragma: no cover
        out.update(js_divergence=None, js_available=False, js_note="scipy missing")
    else:
        p = np.asarray([obs.get(l, 0) for l in labels], dtype=np.float64)
        q = np.asarray([ref.get(l, 0) for l in labels], dtype=np.float64)
        p, q = p / p.sum(), q / q.sum()
        out.update(js_divergence=round(float(_jensen_shannon(p, q, base=2.0)), 6),
                   js_available=True, js_note=None)
    return out


# --------------------------------------------------------------------------
# embedding-space health
# --------------------------------------------------------------------------
def nn_cosine_distances(x: np.ndarray) -> dict[str, Any]:
    """Nearest-neighbour cosine *distance* (1 - similarity) distribution.

    Crowded set -> distances pile up near 0 (little new information per added
    example). A good synthetic set sits in a mid band: neighbours are related
    but not copies.
    """
    x = np.asarray(x, dtype=np.float32)
    n = int(x.shape[0])
    if n < 2:
        return {"n_examples": n, "mean": None, "median": None, "p10": None, "p90": None,
                "note": "needs at least 2 examples"}
    sim = cosine_matrix(x)
    np.fill_diagonal(sim, -np.inf)
    d = 1.0 - np.max(sim, axis=1)
    return {
        "n_examples": n,
        "mean": round(float(d.mean()), 6),
        "median": round(float(np.median(d)), 6),
        "min": round(float(d.min()), 6),
        "max": round(float(d.max()), 6),
        "p10": round(float(np.percentile(d, 10)), 6),
        "p90": round(float(np.percentile(d, 90)), 6),
        "frac_below_0_10": round(float(np.mean(d < 0.10)), 6),
    }


def memorisation(
    generated: Sequence[Example],
    seeds: Sequence[Example],
    *,
    vectors: np.ndarray | None = None,
    seed_vectors: np.ndarray | None = None,
    embedder: Embedder | None = None,
    threshold: float = DEFAULT_LEAK_THRESHOLD,
    text_of: Callable[[Example], str] = example_text,
    top_offenders: int = 5,
    backend: str | None = None,
    dim: int = 256,
) -> dict[str, Any]:
    """Paraphrase-leak rate: how many generated examples sit on top of a seed.

    ``leak_rate`` is the fraction whose maximum cosine similarity to any seed
    is >= ``threshold``. ``exact_carryover_rate`` is the subset that is an
    *identical* normalised string -- a copy, which no similarity threshold is
    needed to catch. ``offenders`` lists the worst examples so the failure is
    debuggable rather than just reportable.

    A high ``leak_rate`` is the signal that expansion is rewording seeds instead
    of generating: the dataset looks bigger but carries no extra information and
    leaks whatever the seeds hold.
    """
    gen = list(generated)
    seed_list = list(seeds)
    if not gen:
        return {"n_generated": 0, "n_seeds": len(seed_list), "leak_rate": None,
                "threshold": float(threshold), "note": "no generated examples"}
    if not seed_list:
        return {"n_generated": len(gen), "n_seeds": 0, "leak_rate": None,
                "threshold": float(threshold), "note": "no seeds to compare against"}

    emb = embedder
    if vectors is None:
        if emb is None:
            emb = build_embedder(
                backend, corpus=[text_of(e) for e in (gen + seed_list)], dim=dim
            )
        vectors = np.asarray(emb.embed([text_of(e) for e in gen]), dtype=np.float32)
    if seed_vectors is None:
        if emb is None:  # pragma: no cover - defensive
            raise ValueError("memorisation(): pass embedder, vectors or seed_vectors")
        seed_vectors = np.asarray(emb.embed([text_of(e) for e in seed_list]), dtype=np.float32)

    vectors = np.asarray(vectors, dtype=np.float32)
    seed_vectors = np.asarray(seed_vectors, dtype=np.float32)

    sim = cosine_matrix(vectors, seed_vectors)  # unit rows -> dot product is cosine
    best = np.max(sim, axis=1)
    best_idx = np.argmax(sim, axis=1)
    leaks = best >= float(threshold)

    gen_norm = [normalize_text(text_of(e)) for e in gen]
    seed_norm = {normalize_text(text_of(e)) for e in seed_list}
    exact = np.asarray([t in seed_norm for t in gen_norm], dtype=bool)
    paraphrase = leaks & ~exact

    order = np.argsort(-best)[: max(0, int(top_offenders))]
    return {
        "n_generated": len(gen),
        "n_seeds": len(seed_list),
        "threshold": float(threshold),
        "leak_rate": round(float(np.mean(leaks)), 6),
        "n_leaks": int(np.count_nonzero(leaks)),
        "paraphrase_leak_rate": round(float(np.mean(paraphrase)), 6),
        "n_paraphrase_leaks": int(np.count_nonzero(paraphrase)),
        "exact_carryover_rate": round(float(np.mean(exact)), 6),
        "n_exact_carryover": int(np.count_nonzero(exact)),
        "max_sim_mean": round(float(best.mean()), 6),
        "max_sim_median": round(float(np.median(best)), 6),
        "max_sim_p90": round(float(np.percentile(best, 90)), 6),
        "offenders": [
            {
                "index": int(i),
                "input": str(gen[i].input)[:120],
                "cosine_to_seed": round(float(best[i]), 6),
                "seed_index": int(best_idx[i]),
                "identical": bool(exact[i]),
            }
            for i in order
        ],
    }


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
@dataclass
class DataReport:
    """Everything :func:`quality_report` measured, in one JSON-serialisable object."""

    n_seeds: int = 0
    n_generated: int = 0
    n_kept: int = 0
    leak_threshold: float = DEFAULT_LEAK_THRESHOLD
    embedder: str = ""
    dim: int = 0
    diversity: dict[str, Any] = field(default_factory=dict)
    seed_diversity: dict[str, Any] = field(default_factory=dict)
    length: dict[str, Any] = field(default_factory=dict)
    length_by_tag: dict[str, Any] = field(default_factory=dict)
    labels: dict[str, Any] = field(default_factory=dict)
    nearest_neighbour: dict[str, Any] = field(default_factory=dict)
    privacy: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Plain-Python dict (no numpy, no NaN/Inf) safe for ``json.dumps``."""
        return _py(asdict(self))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DataReport":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in dict(data).items() if k in known})

    def summary(self) -> str:
        """One-line human-readable digest."""
        d = self.diversity
        p = self.privacy
        return (
            f"{self.n_generated} generated / {self.n_kept} kept | "
            f"TTR={d.get('type_token_ratio')} distinct1={d.get('distinct_1')} "
            f"distinct2={d.get('distinct_2')} | "
            f"nn_dist_mean={self.nearest_neighbour.get('mean')} | "
            f"leak_rate={p.get('leak_rate')} exact={p.get('exact_carryover_rate')} | "
            f"drift={self.labels.get('drifted')} js={self.labels.get('js_divergence')}"
        )


def quality_report(
    seeds: Sequence[Example],
    generated: Sequence[Example],
    *,
    embedder: Embedder | None = None,
    backend: str | None = None,
    dim: int = 256,
    leak_threshold: float = DEFAULT_LEAK_THRESHOLD,
    text_of: Callable[[Example], str] = example_text,
    percentiles: Sequence[float] = DEFAULT_PERCENTILES,
) -> DataReport:
    """Full quality + privacy report for ``generated`` against the ``seeds`` it came from."""
    seeds_l, gen_l = list(seeds), list(generated)
    texts = [text_of(e) for e in gen_l]
    seed_texts = [text_of(e) for e in seeds_l]

    emb = embedder
    x: np.ndarray | None = None
    seed_x: np.ndarray | None = None
    pool = texts + seed_texts
    if pool:
        if emb is None:
            emb = build_embedder(backend, corpus=pool, dim=dim)
        elif not emb.fitted:
            emb.fit(pool)
        if texts:
            x = np.asarray(emb.embed(texts), dtype=np.float32)
        if seed_texts:
            seed_x = np.asarray(emb.embed(seed_texts), dtype=np.float32)

    labels = distribution_drift(tag_distribution(gen_l), tag_distribution(seeds_l))
    if gen_l:
        privacy = memorisation(
            gen_l, seeds_l, vectors=x, seed_vectors=seed_x, embedder=emb,
            threshold=leak_threshold, text_of=text_of,
        )
    else:
        privacy = {"n_generated": 0, "n_seeds": len(seeds_l), "leak_rate": None,
                   "threshold": float(leak_threshold), "note": "no generated examples"}

    report = DataReport(
        n_seeds=len(seeds_l),
        n_generated=len(gen_l),
        n_kept=len(gen_l),
        leak_threshold=float(leak_threshold),
        embedder=emb.name() if emb is not None else "",
        dim=int(x.shape[1]) if x is not None else (emb.dim if emb is not None else 0),
        diversity=diversity_metrics(texts),
        seed_diversity=diversity_metrics(seed_texts),
        length=length_profile(texts, percentiles=percentiles),
        length_by_tag=tag_length_profile(gen_l, text_of=text_of, percentiles=percentiles),
        labels=labels,
        nearest_neighbour=nn_cosine_distances(x) if x is not None else {"note": "not embedded"},
        privacy=privacy,
    )
    report.warnings = _warnings(report)
    return report


def _warnings(rep: DataReport) -> list[str]:
    """Interpret the numbers; keep the thresholds here so they are reviewable."""
    w: list[str] = []
    d2, sd2 = rep.diversity.get("distinct_2"), rep.seed_diversity.get("distinct_2")
    if isinstance(d2, float) and isinstance(sd2, float) and sd2 > 0 and d2 < 0.6 * sd2:
        w.append(
            f"distinct_2 collapsed vs seeds ({d2} vs {sd2}): expansion is recombining "
            "a small phrase set"
        )
    leak = rep.privacy.get("leak_rate")
    if isinstance(leak, float) and leak > 0.2:
        w.append(f"paraphrase-leak rate {leak} > 0.2 at threshold {rep.leak_threshold}")
    exact = rep.privacy.get("exact_carryover_rate")
    if isinstance(exact, float) and exact > 0:
        w.append(f"{exact} of generated examples are identical to a seed")
    if rep.labels.get("drifted") is True:
        w.append(
            f"tag distribution differs from seeds (chi2={rep.labels.get('chi_square')}, "
            f"p={rep.labels.get('chi2_p_value')})"
        )
    nnd = rep.nearest_neighbour.get("mean")
    if isinstance(nnd, float) and nnd < 0.05:
        w.append(f"embedding space crowded: mean nearest-neighbour cosine distance {nnd}")
    if rep.length.get("empty_count"):
        w.append(f"{rep.length['empty_count']} examples normalise to zero tokens")
    if rep.diversity.get("n_tokens") == 0 and rep.n_generated:
        w.append("no tokenised content in the generated set")
    return w
