# synthetic-data-factory

Turns a handful of seed examples into a large, quality-filtered training dataset — the data half of fine-tuning, done properly.

## The idea

Real ML teams spend more time on data than on models. Most "synthetic data" scripts just call an LLM N times and dump the output. This one has **measurable quality gates**: every candidate is scored by an independent judge, near-duplicates are removed, and only passing examples reach the export.

## Pipeline

```
Seed ──► Expand ──► Filter ──► Dedup ──► Export
  │         │          │         │         │
 5-20    paraphrases  judge    cosine   JSONL for
examples  + edge +   scores    near-dup  fine-tuning
          adversarial each one  removal  (OpenAI/HF)
```

1. **Seed** — your 5–20 hand-written examples (input → ideal output), validated on load.
2. **Expand** — LLM generates paraphrases, edge cases, and adversarial variants per seed.
3. **Filter** — an *independent* judge LLM scores each candidate on faithfulness + diversity; below threshold → dropped.
4. **Dedup** — near-duplicate removal via embedding cosine similarity (threshold configurable).
5. **Export** — clean JSONL in chat-completion format, ready for OpenAI fine-tuning or Hugging Face.

## Why this is interesting

- **Quality gates, not hope** — every example that ships has a score attached.
- **Independent judge** — the expander and the filter are different calls, so the model can't grade its own homework.
- **Dedup that actually works** — cosine similarity catches paraphrases that exact-match misses.
- **Export-ready** — JSONL matches the format both OpenAI and HF trainers expect.

## Architecture

```
factory/
├── config.py          # Settings from env / .env
├── logging_config.py  # structured JSON logs (no secrets)
├── seed.py            # load/save seeds with validation
├── expand.py          # paraphrase / edge / adversarial variants
├── filter.py          # independent judge, JSON verdict parsing
├── dedup.py           # cosine similarity near-dedup
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

# live run
python -m factory seeds.json out.jsonl

# tests
pytest -q
```

Windows notes:

- Activate with `.venv\Scripts\activate`.
- The dedup step needs `numpy`, which installs cleanly on Windows via pip wheels — no compiler required.
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
| `numpy` | `>=1.26` | Powers the cosine-similarity dedup. Embeddings are plain float arrays; numpy's vectorised ops make similarity checks over hundreds of candidates near-instant, with no extra ML framework. |
| `python-dotenv` | `>=1.0` | Loads `.env` for the API key. |
| `pytest` | `>=8.0` | (dev) Tests for seed validation, judge scoring, dedup threshold, and end-to-end pipeline with a stub LLM. |

No embedding model dependency on purpose: the dedup step accepts any list of float vectors, so you can swap in OpenAI embeddings, a local sentence-transformer, or even random vectors for tests without touching pipeline code.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | — | required for live mode |
| `SDF_MODEL` | `gpt-4o-mini` | model for expand + judge |
| `SDF_MIN_SCORE` | `6` | min faithfulness (1–10) to keep a candidate |
| `SDF_DEDUP_THRESHOLD` | `0.92` | cosine cutoff for near-duplicates |
| `SDF_VARIANTS_PER_SEED` | `5` | paraphrases + edge + adversarial per seed |
| `SDF_TIMEOUT` | `30` | per-call timeout (seconds) |

## Testing

```bash
pytest -q
pytest -v tests/test_filter.py    # judge scoring + verdict parsing
pytest -v tests/test_dedup.py     # cosine near-dedup
pytest -v tests/test_pipeline.py  # end-to-end with stub LLM
```

Covered: seed validation, JSON parsing even with markdown fences, filter dropping low scores, dedup threshold behaviour, pipeline orchestration, log scrubbing.

## Design decisions (interview notes)

1. **Why an independent judge instead of self-scoring?**
   A model grading its own output optimises for sounding good, not being correct. A separate call with a different prompt breaks that feedback loop.

2. **Why cosine dedup over exact-match?**
   Exact-match misses paraphrases ("hello" vs "hi there"). Embeddings catch semantic duplicates; the threshold is tunable per domain.

3. **Why JSONL export?**
   It's the de-facto standard for both OpenAI fine-tuning and Hugging Face `datasets`. One format, two consumers.

4. **Why validate seeds on load?**
   Garbage in, garbage out. Catching a malformed seed at the door saves a full pipeline run that produces nothing useful.

## Project status

Working prototype with real LLM integration, quality gates, dedup, and tests. Not production — no distributed workers, no embedding-model swap-out, no UI. Strong portfolio piece for data-engineering + LLM interviews.

## License

MIT.
