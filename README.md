# synthetic-data-factory

Turns a handful of seed examples into a large, quality-filtered training dataset — the data half of fine-tuning, done properly.

## The idea

Real ML teams spend more time on data than on models. Most "synthetic data" scripts just call an LLM N times and dump the output. This one has **measurable quality gates**: every candidate is scored by an independent judge, near-duplicates are removed by real embeddings with an ANN index, and the result ships with a quality/privacy report, leakage-audited train/val/test splits, and a reproducible manifest.

## Pipeline

```
Seed ──► Expand ──► Filter ──► Dedup ──► Export ──► Report · Splits · Manifest
  │         │          │         │         │              │
 5-20    paraphrases  judge    cosine    JSONL      leak rate, TTR,
examples  + edge +   scores    near-dup  for FT     stratified splits,
          adversarial each one  removal   (OpenAI/HF) run.json lineage
```

1. **Seed** — your 5–20 hand-written examples (input → ideal output), validated on load.
2. **Expand** — LLM generates paraphrases, edge cases, and adversarial variants per seed.
3. **Filter** — an *independent* judge LLM scores each candidate on faithfulness + diversity; below threshold → dropped.
4. **Dedup** — real embeddings + ANN candidate search + exact-cosine verification; transitive duplicate groups collapse to one survivor.
5. **Export** — clean JSONL in chat-completion format, ready for OpenAI fine-tuning or Hugging Face.
6. **Evidence** — quality report, grouped splits with a leakage audit, and `run.json`.

## The AI stack

The dedup stage is no longer a stub. Six modules make it a small retrieval system:

- **`text.py`** — order-sensitive word/char shingles and `blake2b`-based content hashes (stable across processes, unlike Python's `hash()`).
- **`embeddings.py`** — four backends behind one `Embedder` interface: deterministic signed **hashing** (no fit step), **TF-IDF + LSA** over word bigrams and `char_wb` 3–5-grams (scikit-learn), plus optional **OpenAI** and **sentence-transformers** when installed. `auto` picks the best available; rows are L2-normalised so cosine is a dot product.
- **`ann.py`** — sub-quadratic candidate generation: **MinHash+LSH banding** (bands chosen from the S-curve so `bands × rows ≤ num_perm`), an **IVF** coarse-quantisation index, `hybrid` (their union), or `exact`. `benchmark_ann()` measures candidate recall against a brute-force ground truth — on a 325-document corpus at threshold 0.92, LSH with 128 perms recovers **1.000** of true duplicates.
- **`dedup.py`** — the safety argument that makes ANN acceptable: candidates are a *proposal*, the decision is the **exact cosine** of real vectors. An ANN miss can only leave a duplicate in the data; it can never delete a unique example. Verified pairs collapse transitively with union-find, first occurrence wins, and every drop is recorded in the audit trail.
- **`metrics.py`** — a privacy + diversity report: memorisation leak rate against the seeds (exact carryover + embedding nearest-neighbour), type–token ratio, distinct-1/2, and tag-distribution drift with JS divergence / chi-square.
- **`split.py`** — stratified train/val/test where **near-duplicate clusters are kept inside one split** (a paraphrase in train and its twin in test leaks the validation score), then re-audits the result for cross-split leakage and refuses to ship a leaky split.

Every run can end in a **manifest** (`run.json`): config, stage counts (including how many expands failed and degraded to seeds), dedup audit, library versions and content hashes — reproducibility you can grep.

## Why this is interesting

- **Quality gates, not hope** — every example that ships has a score attached and an audit trail behind it.
- **Independent judge** — the expander and the filter are different calls, so the model can't grade its own homework.
- **Dedup with a recall budget** — ANN proposes, exact cosine disposes; `benchmark_ann` tells you exactly what the approximation costs.
- **Leakage-proof splits** — duplicates grouped into one split, then audited; a leaky split raises instead of shipping.
- **Export-ready** — JSONL matches the format both OpenAI and HF trainers expect.

## Architecture

```
factory/
├── config.py          # Settings from env / .env
├── logging_config.py  # structured JSON logs (no secrets)
├── seed.py            # load/save seeds with validation
├── expand.py          # paraphrase / edge / adversarial variants
├── filter.py          # independent judge, JSON verdict parsing
├── text.py            # shingles, n-gram features, stable content hashes
├── embeddings.py      # hash / TF-IDF+LSA / OpenAI / sentence-transformers
├── ann.py             # MinHash+LSH, IVF, hybrid, exact + recall benchmark
├── dedup.py           # candidates -> exact cosine -> union-find collapse
├── metrics.py         # quality + privacy report (leak rate, TTR, drift)
├── split.py           # stratified grouped split + leakage audit
├── manifest.py        # run.json lineage: config, stages, versions, hashes
├── export.py          # chat-completion JSONL writer
├── pipeline.py        # orchestrates the whole flow
├── llm.py             # OpenAI client with retries + timeout
├── __main__.py        # CLI: python -m factory seeds.json out.jsonl
├── demo.py            # offline stub (no key)
└── demo_llm.py        # live run with real OpenAI
```

## How to run

```bash
git clone https://github.com/OlegUnreal/synthetic-data-factory.git
cd synthetic-data-factory

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

cp .env.example .env               # put OPENAI_API_KEY=sk-... in .env

# offline demo (no key)
python -m factory.demo

# live run with report, splits and manifest
python -m factory seeds.json out.jsonl --report --split

# tests
pytest -q
```

Windows notes:

- Activate with `.venv\Scripts\activate`.
- `numpy` and `scikit-learn` install cleanly on Windows via pip wheels — no compiler required.
- Output JSONL is UTF-8; open it in any editor or feed it straight to an OpenAI fine-tuning job.

### Seed file format

```json
[
  {"input": "Translate to French: hello", "output": "Bonjour"},
  {"input": "Summarize: ...", "output": "..."}
]
```

## Libraries used and why

| Library | Version | Why it is here |
|---|---|---|
| `openai` | `>=1.40` | Client for both the expander and the independent judge. Two separate calls with different system prompts — the judge never sees the expander's instructions, which is what makes the quality gate honest. |
| `numpy` | `>=1.26` | Unit-norm embeddings, cosine verification of ANN candidates, vectorised drift stats. |
| `scikit-learn` | `>=1.4` | TF-IDF (word bigrams + `char_wb` 3–5-grams) and TruncatedSVD for the LSA embedding backend. |
| `scipy` | `>=1.11` | Sparse hstack of the TF-IDF union, Jensen-Shannon divergence and chi-square for distribution drift. |
| `python-dotenv` | `>=1.0` | Loads `.env` for the API key. |
| `pytest` | `>=8.0` | (dev) Unit + integration tests, including an ANN recall benchmark and a leakage-audit regression. |

No heavyweight vector DB or torch dependency: hashing and LSA embedders run anywhere numpy runs, and the `openai` / `st` backends activate themselves only if their SDKs are installed.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | — | required for live mode |
| `SDF_MODEL` | `gpt-4o-mini` | model for expand + judge |
| `SDF_MIN_SCORE` | `6` | min faithfulness (1–10) to keep a candidate |
| `SDF_VARIANTS_PER_SEED` | `3` | paraphrases + edge + adversarial per seed |
| `SDF_TIMEOUT` | `30` | per-call timeout (seconds) |
| `SDF_DEDUP_THRESHOLD` | `0.92` | cosine cutoff for near-duplicates |
| `SDF_DEDUP_STRATEGY` | `ivf` | candidate generator: `exact` / `lsh` / `ivf` / `hybrid` |
| `SDF_LSH_NUM_PERM` | `128` | MinHash permutations for LSH banding |
| `SDF_IVF_N_PROBE` | `4` | IVF lists probed per query |
| `SDF_EMBEDDING_BACKEND` | `auto` | `hash` / `lsa` / `openai` / `st` |
| `SDF_EMBED_DIM` | `256` | embedding dimensionality |
| `SDF_EMBED_SEED` | `13` | seed for the hashing embedder and ANN |
| `SDF_SPLIT_RATIOS` | `0.8,0.1,0.1` | train/val/test proportions |
| `SDF_MANIFEST_TIMESTAMP` | `0` | put wall-clock time in run.json (off = byte-reproducible) |

## Testing

```bash
pytest -q
pytest -v tests/test_ai_stack.py  # embeddings, ANN, dedup, metrics, splits, manifest
pytest -v tests/test_dedup.py     # cosine near-dedup
pytest -v tests/test_pipeline.py  # end-to-end with stub LLM
```

Covered: seed validation, JSON parsing even with markdown fences, filter dropping low scores, dedup threshold behaviour, embedding determinism + unit norms, LSH band params staying inside the permutation budget, LSH candidate recall against brute force, union-find transitivity, first-occurrence dedup with a serialisable audit, leak-rate 0 vs 1 on clean/copied corpora, grouped-split leakage audit, empty-split regression, manifest JSON round-trip.

## Design decisions (interview notes)

1. **Why an independent judge instead of self-scoring?**
   A model grading its own output optimises for sounding good, not being correct. A separate call with a different prompt breaks that feedback loop.

2. **Why candidates + exact verification instead of all-pairs cosine?**
   All-pairs is O(n²) in Python and dies past a few thousand rows. The ANN index proposes O(n) candidates; each one is still decided by the exact cosine of real embeddings. That keeps the guarantee all-pairs had — a verified duplicate is a real duplicate — while paying sub-quadratic cost, and `benchmark_ann` quantifies the only risk left: a duplicate that survives because the index missed it.

3. **Why group near-duplicates inside one split?**
   A paraphrase in train and its rewording in test is label leakage: the eval score flatters the model. The splitter treats a duplicate cluster as one atom before assigning strata, then `audit_leakage` re-checks the shipped splits and refuses to pass a leak.

4. **Why JSONL export?**
   It's the de-facto standard for both OpenAI fine-tuning and Hugging Face `datasets`. One format, two consumers.

5. **Why validate seeds on load?**
   Garbage in, garbage out. Catching a malformed seed at the door saves a full pipeline run that produces nothing useful.

6. **Why a manifest?**
   "Which threshold, which backend, which model produced this file?" is the first question six months later. run.json answers it from the file itself — including how many LLM calls failed and degraded to seeds, so silent quality loss becomes visible.

## Project status

Working prototype with real LLM integration, embedding-backed dedup, quality gates, leakage-audited splits, manifests, and 28 tests. Not production — no distributed workers, no UI. Strong portfolio piece for data-engineering + LLM interviews.

## License

MIT.
