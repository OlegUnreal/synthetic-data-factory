"""Tests for the AI layer: embeddings, ANN candidates, dedup, metrics, splits, manifest."""
from __future__ import annotations

import json

import numpy as np
import pytest

from factory.ann import candidate_pairs, select_band_params
from factory.dedup import collapse_groups, dedup_audit, find_duplicate_groups
from factory.embeddings import build_embedder
from factory.manifest import RunManifest, write_manifest
from factory.metrics import quality_report
from factory.seed import Example
from factory.split import split_dataset
from factory.text import content_hash, shingle_set


def ex(text: str, i: int = 0) -> Example:
    return Example(input=text, output=f"answer {i}")


def corpus() -> list[str]:
    topics = ["deploy", "database", "network", "caching", "auth"]
    docs = []
    for i, t in enumerate(topics):
        for k in range(4):
            docs.append(f"how to configure the {t} service in environment {k} with retries")
    return docs


# --------------------------------------------------------------------------- #
# embeddings
# --------------------------------------------------------------------------- #
def test_hashing_embeddings_are_deterministic_across_instances():
    docs = corpus()
    a = build_embedder("hash", corpus=docs, dim=128).embed(docs)
    b = build_embedder("hash", corpus=docs, dim=128).embed(docs)
    np.testing.assert_array_equal(a, b)


def test_lsa_embeddings_are_fitted_and_unit_norm():
    docs = corpus()
    emb = build_embedder("lsa", corpus=docs, dim=64)
    x = emb.embed(docs)
    assert emb.fitted
    assert x.shape == (len(docs), 64)
    norms = np.linalg.norm(x, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_the_same_text_embeds_near_itself_and_far_from_others():
    docs = corpus()
    emb = build_embedder("hash", corpus=docs, dim=128)
    x = emb.embed(docs + [docs[0]])
    sims = x[:-1] @ x[-1]
    assert sims[0] > 0.99
    assert sorted(sims[1:])[-1] < sims[0]


# --------------------------------------------------------------------------- #
# ANN candidate generation
# --------------------------------------------------------------------------- #
def test_band_params_stay_inside_the_signature_budget():
    for num_perm in (64, 128, 256):
        for threshold in (0.5, 0.8, 0.92):
            bands, rows = select_band_params(num_perm, threshold)
            assert bands >= 1 and rows >= 1
            assert bands * rows <= num_perm


def test_lsh_candidates_cover_every_true_duplicate():
    docs = corpus()
    dupes = [docs[0] + " with extra words", docs[3] + " plus detail"]
    all_docs = docs + dupes
    emb = build_embedder("hash", corpus=all_docs, dim=128)
    x = emb.embed(all_docs)
    truth = {(i, j) for i in range(len(all_docs)) for j in range(i + 1, len(all_docs))
             if float(x[i] @ x[j]) >= 0.9}
    pairs, info = candidate_pairs(
        x, sets=[set(shingle_set(d)) for d in all_docs],
        threshold=0.9, strategy="lsh", max_exact_fallback=0, seed=13,
    )
    assert not info.get("used_exact_fallback")
    assert truth <= set(pairs)


# --------------------------------------------------------------------------- #
# dedup: union-find groups + audit
# --------------------------------------------------------------------------- #
def test_collapse_groups_is_transitive():
    groups = collapse_groups(4, [(0, 1), (1, 2)])
    assert groups == [[0, 1, 2], [3]]


def test_dedup_keeps_first_occurrence_and_reports_the_drop():
    docs = corpus()
    items = [ex(d, i) for i, d in enumerate(docs)]
    items.append(ex(docs[0], 99))  # exact duplicate of the first
    kept, audit = dedup_audit(items, threshold=0.9, strategy="exact")
    assert len(kept) == len(docs)
    assert kept[0].input == docs[0]
    assert audit.n_dropped == 1
    assert audit.dropped[0]["kept_index"] == 0
    json.dumps(audit.as_dict())  # manifest-ready: must be serialisable


def test_find_duplicate_groups_diagnoses_the_run():
    docs = corpus()
    items = [ex(d) for d in docs] + [ex(docs[1])]
    groups, info = find_duplicate_groups(items, 0.9, strategy="exact")
    assert info["duplicate_groups"] == 1
    assert info["largest_group"] == 2
    dup = next(g for g in groups if len(g) > 1)
    assert 1 in dup and len(docs) in dup  # the copy of docs[1] collapsed into its original


# --------------------------------------------------------------------------- #
# metrics: diversity + memorisation
# --------------------------------------------------------------------------- #
def test_quality_report_flags_copies_and_rewards_diversity():
    docs = corpus()
    seeds = [ex(d) for d in docs]
    fresh = [ex(f"a fresh angle on topic {i}", i) for i in range(6)]
    copies = [ex(d) for d in docs]  # memorised the seed set verbatim
    clean = quality_report(seeds, fresh, backend="hash", dim=128)
    leaked = quality_report(seeds, copies, backend="hash", dim=128)
    assert clean.privacy["leak_rate"] == 0.0
    assert leaked.privacy["leak_rate"] == 1.0
    assert clean.diversity["type_token_ratio"] > 0


# --------------------------------------------------------------------------- #
# splits: grouped assignment + leakage audit
# --------------------------------------------------------------------------- #
def test_split_keeps_duplicates_in_one_split_and_audits_clean():
    docs = corpus()
    items = [ex(d) for d in docs] + [ex(docs[0]), ex(docs[2])]
    result = split_dataset(items, (0.7, 0.15, 0.15), threshold=0.9, backend="hash", dim=128)
    total = sum(len(v) for v in result.splits.values())
    assert total == len(items)
    assert result.leakage.ok, result.leakage.summary()


def test_an_empty_split_no_longer_crashes_the_leakage_audit():
    # regression: two examples with ratios (0.8, 0.1, 0.1) leave `test` empty,
    # and audit_leakage used to raise KeyError on the missing vector block
    items = [ex("only one topic here"), ex("a second distinct topic")]
    result = split_dataset(items, (0.8, 0.1, 0.1), threshold=0.9, backend="hash", dim=128)
    assert sum(len(v) for v in result.splits.values()) == 2


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #
def test_manifest_round_trips_through_json(tmp_path):
    docs = corpus()
    seeds = [ex(d) for d in docs]
    pool = seeds + [ex("one more example", 50)]
    manifest = RunManifest.from_run(
        config={"model": "stub", "dedup_threshold": 0.9},
        seeds=seeds,
        stages={"seeds": len(seeds), "exported": len(pool)},
        model_version="stub",
    )
    path = write_manifest(manifest, tmp_path / "out.jsonl")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["config"]["dedup_threshold"] == 0.9
    assert loaded["stages"]["exported"] == len(pool)
    assert loaded["n_seeds"] == len(seeds)


def test_content_hash_is_stable_and_shingles_are_deterministic():
    assert content_hash("a b c") == content_hash("a b c")
    assert shingle_set("a b c") == shingle_set("a b c")
    # shingles are order-sensitive on purpose: rewordings must not hash the same
    assert shingle_set("a b c") != shingle_set("c b a")
