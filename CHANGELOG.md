# Changelog

## 0.3.0 — 2026-09-18

The dedup stage is rebuilt on a real retrieval stack.

- **`embeddings.py`** — `Embedder` interface with four backends: deterministic signed hashing (blake2b buckets, process-stable), TF-IDF+LSA (word bigrams + `char_wb` 3–5-grams via scikit-learn), and optional OpenAI / sentence-transformers when installed; `auto` picks the best available, rows are L2-normalised.
- **`ann.py`** — sub-quadratic candidate generation: MinHash+LSH banding with S-curve band selection, an IVF index, `hybrid`, and `exact`; `benchmark_ann()` measures candidate recall against brute force (1.000 on a 325-doc corpus at threshold 0.92, 128 perms).
- **`dedup.py`** — candidates proposed by ANN, decided by the exact cosine, collapsed transitively with union-find (first occurrence wins); `DedupAudit` records every drop with its similarity and nearest survivor.
- **`metrics.py`** — quality + privacy report: memorisation leak rate against the seeds, type–token ratio, distinct-1/2, tag-distribution drift with JS divergence and chi-square.
- **`split.py`** — stratified train/val/test that keeps near-duplicate clusters inside one split, then re-audits the shipped splits for cross-split leakage (and refuses a leak); fixed a `KeyError` when a split came out empty.
- **`manifest.py`** — `run.json` lineage: config, stage counts (including LLM failures that degraded to seeds), dedup audit, library versions, content hashes.
- `SDF_DEDUP_STRATEGY`, `SDF_EMBEDDING_BACKEND`, `SDF_SPLIT_RATIOS` and friends now actually reach the stages they name; scikit-learn and scipy declared as dependencies.

## 0.2.0

CI workflow, architecture diagram, demo placeholder.

## 0.1.0

Structured logging, config, CLI, seed validation, expanded tests.
