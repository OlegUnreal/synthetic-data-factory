# synthetic-data-factory

Turns a handful of seed examples into a large, quality-filtered training dataset.

## Pipeline

1. **Seed** — your 5-20 hand-written examples (input → ideal output).
2. **Expand** — LLM generates paraphrases, edge cases, adversarial variants.
3. **Filter** — each candidate scored by an independent judge LLM on faithfulness + diversity; low scores dropped.
4. **Dedup** — near-duplicate removal via embedding cosine similarity.
5. **Export** — JSONL ready for fine-tuning (OpenAI / HF format).

## Why this is interesting

Real ML teams spend more time on data than on models. This project is the data half: measurable quality gates, not just "generate 1000 examples and hope".

## Architecture

```
config.py        -> Settings from env / .env
logging_config.py-> structured JSON logs (no secrets)
seed.py          -> load/save seeds with validation
expand.py        -> paraphrase / edge / adversarial variants
filter.py        -> independent judge, JSON verdict parsing
dedup.py         -> cosine similarity near-dedup
export.py        -> chat-completion JSONL
pipeline.py      -> orchestrates the whole flow
llm.py           -> OpenAI client with retries + timeout
__main__.py      -> CLI: python -m factory seeds.json out.jsonl
```

## Run

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env   # put your OPENAI_API_KEY in .env

# offline demo (no key)
python -m factory.demo

# live run
python -m factory seeds.json out.jsonl

# tests
pytest -q
```

## Config

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | — | required for live mode |
| `SDF_MODEL` | `gpt-4o-mini` | model name |
| `SDF_MIN_SCORE` | `6` | min faithfulness to keep |
| `SDF_DEDUP_THRESHOLD` | `0.92` | cosine cutoff for near-dupes |
| `SDF_TIMEOUT` | `30` | per-call timeout (seconds) |
